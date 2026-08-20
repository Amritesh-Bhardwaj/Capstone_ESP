"""Tests for resampling and the counting features.

Run: csi_env/bin/python har/test_count_features.py   (or pytest)

The counting path diverges from the activity path in two ways that are easy to
get wrong and silent when wrong: it resamples onto a uniform grid, and it
computes spectra per subcarrier rather than on the subcarrier mean. Both have
tests here, because both were got wrong once already -- the second one produced
a Doppler centroid of exactly 0.0 Hz that looked plausible until it was
checked against a signal of known frequency.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import count_features as cf  # noqa: E402
import csi_pipeline as cp  # noqa: E402
import room_profile as rp  # noqa: E402

RNG = np.random.default_rng(7)


def _still(frames=400, subcarriers=48, noise=0.01):
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, subcarriers)))
    return np.tile(base, (frames, 1)) * (1 + RNG.normal(0, noise, (frames, subcarriers)))


def _tone(freq, frames=400, fs=100.0, subcarriers=48, depth=0.2):
    """A single temporal frequency, differential across subcarriers.

    The per-subcarrier phase offsets matter: a tone applied identically to
    every subcarrier is common mode and is removed by design.
    """
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, subcarriers)))
    t = np.arange(frames) / fs
    phase = np.linspace(0, 2 * np.pi, subcarriers, endpoint=False)
    swing = depth * np.sin(2 * np.pi * freq * t[:, None] + phase[None, :])
    return np.tile(base, (frames, 1)) * (1 + swing)


# --- resampling -------------------------------------------------------------


def test_resample_returns_a_uniform_grid():
    signal = _still(frames=200)
    stamps = np.cumsum(RNG.uniform(0.005, 0.02, 200))
    stamps -= stamps[0]
    _, grid, _, _ = cp.resample_uniform(signal, stamps, target_hz=100.0)
    deltas = np.diff(grid)
    assert np.allclose(deltas, deltas[0]), "grid is not uniform"
    assert abs(deltas[0] - 0.01) < 1e-9


def test_resample_preserves_a_known_tone():
    """Jittered sampling then resampling must keep the frequency intact."""
    fs, freq = 100.0, 7.0
    stamps = np.sort(RNG.uniform(0, 4.0, 800))
    t = stamps[:, None]
    phase = np.linspace(0, 2 * np.pi, 48, endpoint=False)[None, :]
    signal = 10 + 2 * np.sin(2 * np.pi * freq * t + phase)
    values, _, _, _ = cp.resample_uniform(signal, stamps, target_hz=fs)
    spectrum = cf.spectrum_per_subcarrier(values)
    freqs = np.fft.rfftfreq(len(values), d=1.0 / fs)
    peak = freqs[np.argmax(spectrum)]
    assert abs(peak - freq) < 1.0, f"peak at {peak} Hz, expected {freq} Hz"


def test_resample_marks_long_gaps_invalid():
    signal = _still(frames=100)
    stamps = np.arange(100) * 0.01
    stamps[50:] += 1.0          # a one-second dropout
    _, grid, valid, stats = cp.resample_uniform(signal, stamps, max_gap_s=0.25)
    assert stats["n_gaps"] == 1
    assert 0.0 < stats["gap_fraction"] < 1.0
    assert not valid.all(), "the dropout was silently bridged"
    assert len(valid) == len(grid)


def test_resample_short_gaps_stay_valid():
    signal = _still(frames=100)
    stamps = np.arange(100) * 0.01
    _, _, valid, stats = cp.resample_uniform(signal, stamps, max_gap_s=0.25)
    assert valid.all()
    assert stats["gap_fraction"] == 0.0


def test_resample_rejects_mismatched_lengths():
    try:
        cp.resample_uniform(_still(frames=10), np.arange(5) * 0.01)
    except ValueError:
        return
    raise AssertionError("mismatched lengths should raise")


# --- spectrum ---------------------------------------------------------------


def test_spectrum_finds_a_known_frequency():
    """Regression: computing this on the subcarrier mean gives exactly zero.

    remove_common_mode divides each frame by its own across-subcarrier mean, so
    that mean is 1 by construction and carries no spectrum at all.
    """
    window = _tone(6.0)
    spectrum = cf.spectrum_per_subcarrier(window)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / 100.0)
    assert abs(freqs[np.argmax(spectrum)] - 6.0) < 1.0


def test_doppler_centroid_is_not_pinned_to_zero():
    centroid, spread = cf.doppler_features(_tone(15.0), fs=100.0)
    assert centroid > 5.0, f"centroid collapsed to {centroid}"
    assert spread > 0.0


def test_band_above_nyquist_is_zero_not_dropped():
    """Feature length must not change with the sample rate."""
    window = _tone(3.0, fs=20.0)
    bands = cf.band_fractions(window, fs=20.0)
    assert len(bands) == len(cf.COUNT_BANDS)
    assert bands[-1] == 0.0 and bands[-2] == 0.0


def test_bands_capture_high_frequency_content():
    """The 11-25 Hz band exists precisely so this is not thrown away."""
    bands = cf.band_fractions(_tone(16.0), fs=100.0)
    assert bands[cf.COUNT_BAND_NAMES.index("doppler")] > 0.3


# --- rank features ----------------------------------------------------------


def test_effective_rank_is_low_for_one_shared_mode():
    """One motion driving every subcarrier together is rank 1 after AGC removal."""
    frames, subcarriers = 400, 48
    base = np.linspace(8, 12, subcarriers)
    swing = np.sin(np.linspace(0, 20 * np.pi, frames))[:, None]
    shape = np.linspace(-1, 1, subcarriers)[None, :]
    window = np.tile(base, (frames, 1)) * (1 + 0.2 * swing * shape)
    assert cf.effective_rank(window) < 4.0


def test_effective_rank_is_high_for_independent_subcarriers():
    window = _still(noise=0.05)
    assert cf.effective_rank(window) > 10.0


def test_effective_rank_orders_one_mode_below_many():
    frames, subcarriers = 400, 48
    base = np.linspace(8, 12, subcarriers)
    shape = np.linspace(-1, 1, subcarriers)[None, :]
    one = np.tile(base, (frames, 1)) * (
        1 + 0.2 * np.sin(np.linspace(0, 20 * np.pi, frames))[:, None] * shape
    )
    many = np.tile(base, (frames, 1)) * (1 + RNG.normal(0, 0.05, (frames, subcarriers)))
    assert cf.effective_rank(one) < cf.effective_rank(many)


def test_eigen_features_are_bounded_ratios():
    r2, r3, n_eig = cf.eigen_features(_still())
    assert 0.0 <= r3 <= r2 <= 1.0
    assert 1 <= n_eig <= 48


def test_dilated_pem_is_a_fraction():
    reference = cf.reference_mad(np.stack([_still()]))
    for window in (_still(), _tone(4.0)):
        assert 0.0 <= cf.dilated_pem(window, reference) <= 1.0


def test_dilated_pem_rises_with_disturbance():
    """Against an empty-room reference, more motion must mean a higher score."""
    reference = cf.reference_mad(np.stack([_still(noise=0.005) for _ in range(4)]))
    quiet = cf.dilated_pem(_still(noise=0.005), reference)
    busy = cf.dilated_pem(_tone(3.0, depth=0.4), reference)
    assert busy > quiet, f"quiet={quiet} busy={busy}"


def test_dilated_pem_window_local_scale_is_the_known_inversion():
    """Documents why the reference exists: self-scaling flips the feature.

    A smooth sinusoid never exceeds ~0.95 of its own MAD so it scores 0.0,
    while a still room of Gaussian noise scores ~0.003. Without an empty-room
    reference the still room looks *more* disturbed than the moving one.
    """
    quiet = cf.dilated_pem(_still(noise=0.005))
    busy = cf.dilated_pem(_tone(3.0, depth=0.4))
    assert busy == 0.0 and quiet > 0.0


def test_reference_mad_is_per_subcarrier_and_positive():
    reference = cf.reference_mad(np.stack([_still() for _ in range(3)]))
    assert reference.shape == (48,)
    assert (reference > 0).all()


# --- vectors and aggregation ------------------------------------------------


def test_window_feature_vector_matches_its_names():
    vector = cf.window_count_features(_still(), fs=100.0)
    assert len(vector) == len(cf.window_feature_names())
    assert np.isfinite(vector).all()


def test_features_survive_a_dead_subcarrier():
    window = _tone(5.0)
    window[:, 11] = 0.0
    vector = cf.window_count_features(window, fs=100.0)
    assert np.isfinite(vector).all()


def test_aggregate_matches_its_names():
    feats = cf.features_for_windows(np.stack([_still(), _tone(4.0), _still()]), 100.0)
    active = np.array([False, True, False])
    aggregate = cf.aggregate_features(feats, active)
    assert len(aggregate) == len(cf.aggregate_feature_names())
    assert np.isfinite(aggregate).all()


def test_aggregate_counts_bursts():
    feats = cf.features_for_windows(np.stack([_still()] * 6), 100.0)
    two_bursts = cf.aggregate_features(feats, np.array([1, 1, 0, 0, 1, 0], dtype=bool))
    names = cf.aggregate_feature_names()
    assert two_bursts[names.index("n_bursts")] == 2
    assert two_bursts[names.index("active_fraction")] == 0.5


def test_aggregate_of_nothing_has_the_right_length():
    empty = cf.aggregate_features(np.empty((0, len(cf.window_feature_names()))),
                                  np.array([], dtype=bool))
    assert len(empty) == len(cf.aggregate_feature_names())


# --- room profile -----------------------------------------------------------


def _fake_capture(windows):
    feats = cf.features_for_windows(windows, 100.0)
    return {
        "raw": np.concatenate(list(windows)),
        "windows": windows,
        "features": feats,
        "motion_scores": np.array([0.05 + 0.001 * i for i in range(len(windows))]),
        "mac": "AA:BB:CC:DD:EE:FF",
        "layout": "fft",
        "rssi_mean": -60.0,
        "stats": {"rate_hz": 96.0, "jitter_ratio": 2.0, "target_hz": 100.0,
                  "gap_fraction": 0.0, "mac_fraction": 0.9, "windows": len(windows),
                  "windows_dropped_to_gaps": 0},
    }


def test_profile_without_anchor_reports_zero_gain():
    """A profile built from the empty room alone must not pretend to count."""
    empty = _fake_capture(np.stack([_still() for _ in range(6)]))
    profile = rp.build_profile("t", empty, None)
    assert not profile.features["has_gain_anchor"]
    assert np.allclose(profile.gain(), 0.0)


def test_profile_gain_is_nonzero_with_an_anchor():
    empty = _fake_capture(np.stack([_still() for _ in range(6)]))
    one = _fake_capture(np.stack([_tone(4.0) for _ in range(6)]))
    profile = rp.build_profile("t", empty, one)
    assert profile.features["has_gain_anchor"]
    assert np.abs(profile.gain()).max() > 0.0


def test_profile_normalise_puts_empty_at_zero_and_one_person_at_one():
    """This mapping is the room adaptation; if it drifts, nothing transfers.

    Features must be recomputed against the profile's own reference scale --
    the same two-pass the builder does. Comparing against features computed
    before the reference existed measures two different yardsticks.
    """
    empty = _fake_capture(np.stack([_still() for _ in range(6)]))
    one = _fake_capture(np.stack([_tone(4.0) for _ in range(6)]))
    profile = rp.build_profile("t", empty, one)
    reference = profile.reference()
    live = np.abs(profile.gain()) > 1e-9

    e_feats = cf.features_for_windows(empty["windows"], 100.0, reference)
    o_feats = cf.features_for_windows(one["windows"], 100.0, reference)
    at_empty = profile.normalise(np.median(e_feats, axis=0))
    at_one = profile.normalise(np.median(o_feats, axis=0))
    assert np.allclose(at_empty[live], 0.0, atol=1e-6)
    assert np.allclose(at_one[live], 1.0, atol=1e-6)


def test_profile_stores_the_reference_scale():
    empty = _fake_capture(np.stack([_still() for _ in range(6)]))
    profile = rp.build_profile("t", empty, None)
    assert profile.reference().shape == (48,)
    assert (profile.reference() > 0).all()


def test_profile_round_trips_through_json(tmp_path=None):
    import tempfile
    empty = _fake_capture(np.stack([_still() for _ in range(6)]))
    profile = rp.build_profile("t", empty, None, geometry={"length_ft": 16.0})
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "profile.json"
        profile.save(path)
        again = rp.RoomProfile.load(path)
    assert again.room == "t"
    assert again.geometry["length_ft"] == 16.0
    assert again.presence["enter_threshold"] == profile.presence["enter_threshold"]


def test_profile_detector_uses_the_calibrated_threshold():
    empty = _fake_capture(np.stack([_still() for _ in range(6)]))
    profile = rp.build_profile("t", empty, None)
    detector = profile.detector()
    assert abs(detector.enter_threshold - profile.presence["enter_threshold"]) < 1e-9


def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
