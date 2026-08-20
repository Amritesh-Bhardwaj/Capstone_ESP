"""Features for estimating how many people are in a room.

Kept separate from ``features.py`` on purpose. That module's feature vector
feeds the activity classifier and the ESP-Fi benchmark whose 90.0% headline is
quoted in the docs; changing it would silently invalidate that number. Counting
needs different features and a different timescale, so it gets its own module
and the activity path is left exactly as it was.

Two timescales, because they answer different questions:

* **window (~4 s)** -- instantaneous motion. Enough cycles of a walking gait to
  give the spectrum something to resolve, short enough to localise a burst.
* **aggregate (~60 s)** -- what the count is actually read from. A person
  sitting still is invisible over 4 s but not over a minute: they shift, reach,
  fidget. Counting on the aggregate is what makes sedentary occupants
  countable at all, and it is the single most important design choice here.

Counting is not a headcount. Every feature below saturates as occupants are
added, because a SISO link measures *how much independent motion* is present,
not how many bodies. Treat the output as ordinal (0/1/2/3+), never as a number
that keeps meaning anything past three.
"""

from __future__ import annotations

import numpy as np

from features import EPS, remove_common_mode, motion_score

# Bands in Hz. Unlike ``features.BANDS`` this runs past 11 Hz: walking Doppler
# at 2.4 GHz is 2v/lambda ~= 16 Hz per m/s of radial velocity, and at the
# board's ~96 Hz sample rate (Nyquist 48 Hz) that band is measurable. Measured
# on this hardware, ~30% of spectral power sits above 11 Hz.
#
# The 25-48 Hz band is above plausible human Doppler and is retained only as a
# control: if it moves between empty and occupied it is telling us about
# resampling artefacts, not people. Do not feed it to a classifier before that
# check passes.
COUNT_BANDS = (
    (0.1, 0.5),
    (0.5, 2.0),
    (2.0, 5.0),
    (5.0, 11.0),
    (11.0, 25.0),
    (25.0, 48.0),
)
COUNT_BAND_NAMES = ("breathing", "slow", "walk", "fast", "doppler", "control_hi")

# Deviation, in robust sigmas, above which a CSI cell counts as "perturbed".
PEM_SIGMA = 3.0
# Eigenvalues above this fraction of the largest count towards the rank tally.
EIG_FLOOR = 0.05


def _covariance_spectrum(window: np.ndarray) -> np.ndarray:
    """Normalised eigenvalues of the subcarrier covariance, descending.

    Common-mode removal first: an AGC step scales every subcarrier at once,
    which loads all the energy into a single eigenvector and would read as a
    rank-1 "person". What survives is differential structure across
    subcarriers, which is what a body actually produces.
    """
    centred = remove_common_mode(window)
    centred = centred - centred.mean(axis=0, keepdims=True)
    if len(centred) < 2:
        return np.zeros(window.shape[1])
    cov = np.cov(centred, rowvar=False)
    eig = np.clip(np.linalg.eigvalsh(cov), 0.0, None)[::-1]
    total = eig.sum()
    if total <= 0:
        return np.zeros_like(eig)
    return eig / total


def effective_rank(window: np.ndarray) -> float:
    """exp(entropy) of the normalised covariance spectrum.

    The hypothesis behind the whole counting attempt: N people moving
    independently perturb a higher-rank subspace of the 48 subcarriers than one
    person does, because their reflections are uncorrelated. One antenna still
    gives 48 quasi-independent looks at the channel, so rank is measurable even
    without MIMO.

    Measured live on this board it ranges 2.5-6.7 with 2.4x the dynamic range
    of ``motion_score`` and only +0.72 correlation with it, i.e. roughly half
    its variance is information ``motion_score`` does not carry. That makes it
    worth testing. It is not yet evidence that it tracks occupancy -- occupancy
    was uncontrolled in that capture.
    """
    spectrum = _covariance_spectrum(window)
    spectrum = spectrum[spectrum > 0]
    if not len(spectrum):
        return 0.0
    return float(np.exp(-(spectrum * np.log(spectrum)).sum()))


def eigen_features(window: np.ndarray) -> tuple[float, float, float]:
    """(lambda2/lambda1, lambda3/lambda1, count above EIG_FLOOR of lambda1).

    A flatter tail means the perturbation is spread over more independent
    directions. Ratios rather than absolute eigenvalues, so they do not move
    with transmitter distance or gain.
    """
    spectrum = _covariance_spectrum(window)
    if len(spectrum) < 3 or spectrum[0] <= 0:
        return 0.0, 0.0, 0.0
    top = spectrum[0]
    return (
        float(spectrum[1] / top),
        float(spectrum[2] / top),
        float(np.count_nonzero(spectrum >= EIG_FLOOR * top)),
    )


def reference_mad(windows: np.ndarray) -> np.ndarray:
    """Per-subcarrier robust scale of the empty room. The yardstick for PEM.

    Pooled over every calibration window so a single unusually quiet window
    cannot set the scale for the whole room.
    """
    if not len(windows):
        raise ValueError("reference_mad needs at least one window")
    pooled = np.concatenate([remove_common_mode(w) for w in windows], axis=0)
    median = np.median(pooled, axis=0, keepdims=True)
    return np.median(np.abs(pooled - median), axis=0) * 1.4826


def dilated_pem(
    window: np.ndarray,
    reference: np.ndarray | None = None,
    sigma: float = PEM_SIGMA,
) -> float:
    """Fraction of CSI cells deviating more than ``sigma`` empty-room sigmas.

    Adapted from the Percentage of nonzero Elements in Electronic Frog Eye
    (Xi et al., INFOCOM 2014) -- the most-validated counting feature in the CSI
    literature. It counts *how much* of the CSI matrix is disturbed rather than
    how hard, so it saturates more slowly with occupancy than variance does.

    ``reference`` is the per-subcarrier MAD of the **empty room**, from the
    calibration. Scaling by the window's own MAD instead looks natural and is
    exactly wrong: it divides out the magnitude the feature exists to measure.
    Measured, that inversion is severe enough to flip the feature's sign -- a
    smooth sinusoid never exceeds 0.95 of its own MAD, so it scores 0.000,
    while a still room full of Gaussian noise scores 0.003. The still room
    reads as *more* disturbed than the moving one. A test covers this.

    With ``reference=None`` the window-local fallback is used. That value is a
    tail-shape statistic, not a disturbance magnitude, and must not be compared
    across captures -- it is here only so features can be computed on the very
    first calibration pass, before a reference exists.
    """
    values = remove_common_mode(window)
    median = np.median(values, axis=0, keepdims=True)
    if reference is None:
        scale = np.median(np.abs(values - median), axis=0, keepdims=True) * 1.4826
    else:
        scale = np.asarray(reference, dtype=np.float64)[None, :]
    deviation = np.abs(values - median) / (scale + EPS)
    return float((deviation > sigma).mean())


def spectrum_per_subcarrier(window: np.ndarray) -> np.ndarray:
    """Mean normalised temporal power spectrum across subcarriers.

    Computed per subcarrier and then averaged -- **not** on the subcarrier
    mean. After ``remove_common_mode`` every frame is divided by its own
    across-subcarrier mean, so that mean is ~1 by construction and its spectrum
    is empty. Averaging spectra keeps the differential motion that survives
    common-mode removal.
    """
    values = remove_common_mode(window)
    values = values - values.mean(axis=0, keepdims=True)
    if len(values) < 4 or not np.any(values):
        return np.zeros(len(values) // 2 + 1)
    taper = np.hanning(len(values))[:, None]
    power = np.abs(np.fft.rfft(values * taper, axis=0)) ** 2
    total = power.sum(axis=0, keepdims=True)
    return (power / (total + EPS)).mean(axis=1)


def doppler_features(window: np.ndarray, fs: float) -> tuple[float, float]:
    """(spectral centroid, spectral spread) in Hz."""
    power = spectrum_per_subcarrier(window)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / fs)
    if len(power) != len(freqs) or not power.sum():
        return 0.0, 0.0
    centroid = float((power * freqs).sum())
    spread = float(np.sqrt((power * (freqs - centroid) ** 2).sum()))
    return centroid, spread


def band_fractions(window: np.ndarray, fs: float) -> np.ndarray:
    """Fraction of spectral power in each COUNT_BAND.

    Bands wholly above Nyquist return 0.0 rather than being dropped, so the
    feature vector keeps a fixed length across sample rates.
    """
    power = spectrum_per_subcarrier(window)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / fs)
    out = np.zeros(len(COUNT_BANDS))
    if len(power) != len(freqs):
        return out
    for i, (lo, hi) in enumerate(COUNT_BANDS):
        if lo >= fs / 2:
            continue
        out[i] = float(power[(freqs >= lo) & (freqs < hi)].sum())
    return out


def subcarrier_dispersion(window: np.ndarray) -> float:
    """Spread across subcarriers of their individual motion levels.

    People standing at different distances perturb different subcarriers by
    different amounts, so a spread-out group should disturb the band more
    unevenly than one person does.
    """
    values = remove_common_mode(window)
    cov = values.std(axis=0) / (np.abs(values.mean(axis=0)) + EPS)
    return float(cov.std() / (cov.mean() + EPS))


def window_count_features(
    window: np.ndarray, fs: float, reference: np.ndarray | None = None
) -> np.ndarray:
    """One ~4 s window -> a fixed-length count feature vector.

    ``reference`` is the empty room's per-subcarrier MAD; see ``dilated_pem``.
    """
    if window.ndim != 2:
        raise ValueError("expected a (frames, subcarriers) window")
    ratio_2, ratio_3, n_eig = eigen_features(window)
    centroid, spread = doppler_features(window, fs)
    values = [
        motion_score(window),
        effective_rank(window),
        ratio_2,
        ratio_3,
        n_eig,
        dilated_pem(window, reference),
        centroid,
        spread,
        subcarrier_dispersion(window),
    ]
    values.extend(band_fractions(window, fs))
    return np.asarray(values, dtype=np.float64)


def window_feature_names() -> list:
    return [
        "motion_score",
        "effective_rank",
        "eig_ratio_2",
        "eig_ratio_3",
        "n_eig",
        "dilated_pem",
        "doppler_centroid",
        "doppler_spread",
        "subcarrier_dispersion",
    ] + [f"band_{name}" for name in COUNT_BAND_NAMES]


def features_for_windows(
    windows: np.ndarray, fs: float, reference: np.ndarray | None = None
) -> np.ndarray:
    if not len(windows):
        return np.empty((0, len(window_feature_names())))
    return np.stack([window_count_features(w, fs, reference) for w in windows])


# ---------------------------------------------------------------------------
# 60-second aggregation -- the level counting is actually read from
# ---------------------------------------------------------------------------


def _bursts(active: np.ndarray) -> tuple[int, float]:
    """(number of contiguous active runs, mean run length in windows)."""
    if not len(active) or not active.any():
        return 0, 0.0
    padded = np.concatenate([[False], active.astype(bool), [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    lengths = ends - starts
    return int(len(starts)), float(lengths.mean())


def aggregate_features(window_feats: np.ndarray, active: np.ndarray) -> np.ndarray:
    """(n_windows, n_feats) + activity mask -> one aggregate vector.

    Three order statistics per feature rather than the mean alone. Occupancy
    shows up in the upper tail: two people who each move half the time produce
    the same mean as one person moving constantly, but a higher p90 and a very
    different burst structure.
    """
    if window_feats.ndim != 2:
        raise ValueError("expected a (windows, features) array")
    n_feats = window_feats.shape[1]
    if not len(window_feats):
        return np.zeros(3 * n_feats + 4)

    active = np.asarray(active, dtype=bool)
    n_bursts, mean_burst = _bursts(active)
    return np.concatenate([
        window_feats.mean(axis=0),
        np.percentile(window_feats, 90, axis=0),
        window_feats.max(axis=0),
        [
            float(active.mean()),
            float(n_bursts),
            mean_burst,
            float(len(window_feats)),
        ],
    ])


def aggregate_feature_names() -> list:
    base = window_feature_names()
    return (
        [f"mean_{n}" for n in base]
        + [f"p90_{n}" for n in base]
        + [f"max_{n}" for n in base]
        + ["active_fraction", "n_bursts", "mean_burst_len", "n_windows"]
    )
