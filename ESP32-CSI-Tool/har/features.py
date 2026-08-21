"""Window features for presence detection and activity classification.

Two separate jobs, deliberately kept apart:

* ``motion_score`` -- one dimensionless number per window, used for presence.
  It needs no training, only a short empty-room calibration, so it survives
  being moved to a room the system has never seen.

* ``window_features`` -- a feature vector for the activity classifier. Every
  feature is scale-invariant (a ratio or a normalised band power), because
  absolute CSI amplitude depends on transmitter distance and gain and does not
  transfer between rooms.

Both operate on UN-NORMALISED amplitude. Per-subcarrier z-scoring destroys the
variance magnitude that presence detection depends on, so the demo path runs
the pipeline with ``normalise=False``.
"""

from __future__ import annotations

import collections

import numpy as np

EPS = 1e-9

# Motion bands in Hz. At a ~22 Hz sample rate the usable range is 0-11 Hz,
# which covers breathing through walking cadence.
BANDS = ((0.0, 0.5), (0.5, 2.0), (2.0, 5.0), (5.0, 11.0))
BAND_NAMES = ("breathing", "slow", "walk", "fast")


def remove_common_mode(window: np.ndarray) -> np.ndarray:
    """Divide each frame by its own mean across subcarriers.

    The ESP32's automatic gain control changes gain in discrete steps, scaling
    every subcarrier by the same factor at once. That is indistinguishable from
    motion to any variance-based detector, and it is a major source of false
    "present" readings.

    A person moving changes the subcarriers *differentially* -- the shape of
    the frequency response changes. AGC changes them *uniformly*. Normalising
    each frame by its own across-subcarrier mean cancels the uniform component
    and keeps the differential one.
    """
    scale = window.mean(axis=1, keepdims=True)
    return window / (np.abs(scale) + EPS)


def _coefficient_of_variation(window: np.ndarray) -> np.ndarray:
    """Per-subcarrier std/mean over time. Dimensionless, so gain-invariant."""
    return window.std(axis=0) / (np.abs(window.mean(axis=0)) + EPS)


def _normalised_abs_diff(window: np.ndarray) -> np.ndarray:
    """Per-subcarrier mean |frame-to-frame change|, relative to its own level."""
    if len(window) < 2:
        return np.zeros(window.shape[1])
    return np.abs(np.diff(window, axis=0)).mean(axis=0) / (
        np.abs(window.mean(axis=0)) + EPS
    )


def motion_score(window: np.ndarray, common_mode: bool = True) -> float:
    """A single dimensionless motion figure for one window of amplitudes.

    Uses the median across subcarriers rather than the mean: a few subcarriers
    sitting in a deep fade produce huge ratios that would otherwise dominate.

    ``common_mode=True`` strips AGC gain steps first -- see remove_common_mode.
    Pass False only to reproduce the older, noisier behaviour.
    """
    if len(window) < 2:
        return 0.0
    if common_mode:
        window = remove_common_mode(window)
    return float(np.median(_coefficient_of_variation(window)))


def band_powers(window: np.ndarray, fs: float) -> np.ndarray:
    """Fraction of temporal spectral power in each motion band.

    Computed on the subcarrier-averaged signal after removing its mean, then
    normalised by total power so the result does not depend on signal strength.
    """
    signal = window.mean(axis=1)
    signal = signal - signal.mean()
    if len(signal) < 4 or not np.any(signal):
        return np.zeros(len(BANDS))

    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    total = spectrum.sum() + EPS
    return np.array(
        [spectrum[(freqs >= lo) & (freqs < hi)].sum() / total for lo, hi in BANDS]
    )


def feature_names(fs: float | None = None) -> list:
    """Names matching the vector returned by window_features, in order."""
    names = []
    for stat in ("cv", "absdiff"):
        names += [f"{stat}_{agg}" for agg in ("mean", "std", "p25", "p50", "p75", "max")]
    names += [f"band_{name}" for name in BAND_NAMES]
    names += ["subcarrier_corr", "spectral_centroid_hz", "motion_score"]
    return names


def _aggregate(values: np.ndarray) -> list:
    return [
        float(values.mean()), float(values.std()),
        float(np.percentile(values, 25)), float(np.percentile(values, 50)),
        float(np.percentile(values, 75)), float(values.max()),
    ]


def window_features(window: np.ndarray, fs: float) -> np.ndarray:
    """Turn one (frames, subcarriers) window into a 1-D feature vector."""
    if window.ndim != 2:
        raise ValueError("expected a (frames, subcarriers) window")

    values = _aggregate(_coefficient_of_variation(window))
    values += _aggregate(_normalised_abs_diff(window))
    values += list(band_powers(window, fs))

    # How much the subcarriers move together. Whole-body motion shifts many
    # subcarriers at once; electronic noise does not.
    centred = window - window.mean(axis=0, keepdims=True)
    scale = centred.std(axis=0) + EPS
    normalised = centred / scale
    corr = (normalised.T @ normalised) / len(window)
    off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
    values.append(float(np.abs(off_diagonal).mean()))

    signal = window.mean(axis=1)
    signal = signal - signal.mean()
    if len(signal) >= 4 and np.any(signal):
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
        values.append(float((spectrum * freqs).sum() / (spectrum.sum() + EPS)))
    else:
        values.append(0.0)

    values.append(motion_score(window))
    return np.asarray(values, dtype=np.float64)


def features_for_windows(windows: np.ndarray, fs: float) -> np.ndarray:
    """(n, frames, subcarriers) -> (n, n_features)."""
    if not len(windows):
        return np.empty((0, len(feature_names())))
    return np.stack([window_features(w, fs) for w in windows])


# ---------------------------------------------------------------------------
# Presence detection
# ---------------------------------------------------------------------------


class PresenceDetector:
    """Threshold presence detector with hysteresis and debouncing.

    Calibrated against an empty room: anything meaningfully above the ambient
    motion floor counts as an occupant. Hysteresis stops the readout flickering
    at the boundary, which matters more in a live demo than raw sensitivity.
    """

    def __init__(
        self,
        baseline_median: float,
        baseline_mad: float,
        enter_sigma: float = 6.0,
        exit_sigma: float = 3.0,
        debounce: int = 3,
        smoothing: int = 5,
    ):
        if exit_sigma > enter_sigma:
            raise ValueError("exit_sigma must not exceed enter_sigma")
        # A MAD floor keeps a pathologically quiet calibration from producing a
        # threshold that every later window trips.
        scale = max(baseline_mad, baseline_median * 0.05, EPS)
        self.baseline = baseline_median
        self.scale = scale
        self.enter_threshold = baseline_median + enter_sigma * scale
        self.exit_threshold = baseline_median + exit_sigma * scale
        self.debounce = max(1, int(debounce))
        # A running median over the last few windows. One anomalous window --
        # an AGC step, a packet burst after a gap -- cannot flip the state on
        # its own, which is what produced the random flapping.
        self.smoothing = max(1, int(smoothing))
        self._recent: collections.deque = collections.deque(maxlen=self.smoothing)
        self.present = False
        self._streak = 0

    @classmethod
    def calibrate(cls, windows: np.ndarray, **kwargs) -> "PresenceDetector":
        """Build a detector from empty-room windows."""
        if not len(windows):
            raise ValueError("calibration needs at least one window")
        scores = np.array([motion_score(w) for w in windows])
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)) * 1.4826)
        return cls(median, mad, **kwargs)

    def update(self, window: np.ndarray) -> tuple:
        """Feed one window. Returns (present, smoothed_score, confidence)."""
        self._recent.append(motion_score(window))
        score = float(np.median(self._recent))
        threshold = self.exit_threshold if self.present else self.enter_threshold
        wants = score > threshold

        if wants == self.present:
            self._streak = 0
        else:
            self._streak += 1
            if self._streak >= self.debounce:
                self.present = wants
                self._streak = 0

        # Confidence: how far above the entry threshold, saturating at 2x.
        span = max(self.enter_threshold - self.baseline, EPS)
        confidence = float(np.clip((score - self.baseline) / (2 * span), 0.0, 1.0))
        return self.present, score, confidence

    def describe(self) -> str:
        return (
            f"baseline {self.baseline:.4f}, enter > {self.enter_threshold:.4f}, "
            f"exit < {self.exit_threshold:.4f}"
        )
