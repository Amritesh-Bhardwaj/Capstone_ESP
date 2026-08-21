"""Tests for presence detection and the false-positive fixes.

Run: csi_env/bin/python har/test_features.py   (or pytest)

These exist because the live demo flapped between PRESENT and ABSENT. Two
causes were found and fixed; each has a test here so they cannot regress.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from features import (  # noqa: E402
    PresenceDetector,
    motion_score,
    remove_common_mode,
    window_features,
    feature_names,
)

RNG = np.random.default_rng(0)


def _still(frames=64, subcarriers=48, noise=0.01):
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, subcarriers)))
    return np.tile(base, (frames, 1)) * (1 + RNG.normal(0, noise, (frames, subcarriers)))


def _moving(frames=64, subcarriers=48, depth=0.25):
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, subcarriers)))
    swing = depth * np.sin(np.linspace(0, 8 * np.pi, frames))[:, None]
    return np.tile(base, (frames, 1)) * (1 + swing * RNG.normal(1, 0.3, (1, subcarriers)))


# --- AGC / common-mode rejection -------------------------------------------


def test_common_mode_removal_cancels_a_uniform_gain_step():
    window = _still()
    stepped = window.copy()
    stepped[32:] *= 1.6                      # AGC gain change, no motion
    a = remove_common_mode(window)
    b = remove_common_mode(stepped)
    # After normalisation the two are the same signal.
    assert np.allclose(a, b, atol=1e-6)


def test_agc_step_no_longer_looks_like_motion():
    """The bug: a pure gain step scored HIGHER than real motion."""
    still = _still()
    stepped = still.copy()
    stepped[32:] *= 1.6

    naive_still = motion_score(still, common_mode=False)
    naive_step = motion_score(stepped, common_mode=False)
    assert naive_step > 5 * naive_still, "fixture should reproduce the bug"

    fixed_still = motion_score(still)
    fixed_step = motion_score(stepped)
    assert fixed_step < 2 * fixed_still, (fixed_still, fixed_step)


def test_real_motion_still_scores_above_still():
    assert motion_score(_moving()) > 3 * motion_score(_still())


def test_common_mode_is_scale_invariant():
    window = _still()
    assert np.allclose(remove_common_mode(window), remove_common_mode(window * 7.3))


# --- smoothing and hysteresis ----------------------------------------------


def _detector(**kwargs):
    scores = np.array([motion_score(_still()) for _ in range(40)])
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)) * 1.4826)
    return PresenceDetector(median, mad, **kwargs)


def test_single_spike_cannot_flip_the_state():
    """One anomalous window must not report a person."""
    detector = _detector(enter_sigma=3, exit_sigma=2, smoothing=5, debounce=3)
    for _ in range(10):
        detector.update(_still())
    assert not detector.present
    present, _, _ = detector.update(_moving(depth=2.0))   # one huge window
    assert not present, "a single spike flipped the state"


def test_sustained_motion_does_trigger():
    detector = _detector(enter_sigma=3, exit_sigma=2, smoothing=5, debounce=3)
    for _ in range(10):
        detector.update(_still())
    for _ in range(15):
        present, _, _ = detector.update(_moving())
    assert present, "sustained motion failed to trigger"


def test_returns_to_absent_after_motion_stops():
    detector = _detector(enter_sigma=3, exit_sigma=2, smoothing=5, debounce=3)
    for _ in range(15):
        detector.update(_moving())
    assert detector.present
    for _ in range(20):
        detector.update(_still())
    assert not detector.present


def test_exit_sigma_above_enter_is_rejected():
    try:
        PresenceDetector(0.1, 0.01, enter_sigma=2, exit_sigma=5)
    except ValueError:
        return
    raise AssertionError("expected ValueError when exit_sigma > enter_sigma")


def test_smoothing_one_disables_the_median():
    detector = _detector(enter_sigma=3, exit_sigma=2, smoothing=1, debounce=1)
    for _ in range(5):
        detector.update(_still())
    present, _, _ = detector.update(_moving(depth=2.0))
    assert present, "with smoothing=1 and debounce=1 a spike should pass through"


# --- feature vector integrity ----------------------------------------------


def test_feature_vector_matches_names_and_is_finite():
    feats = window_features(_moving(), fs=22.0)
    assert feats.shape == (len(feature_names()),)
    assert np.isfinite(feats).all()


def test_features_handle_a_dead_subcarrier():
    window = _moving()
    window[:, 3] = 0.0
    assert np.isfinite(window_features(window, fs=22.0)).all()


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
