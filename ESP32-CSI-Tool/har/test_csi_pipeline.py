"""Tests for the CSI preprocessing pipeline.

Run with pytest, or standalone:  python har/test_csi_pipeline.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import (  # noqa: E402
    SYNTHETIC_MAC,
    LAYOUT_FFT,
    LAYOUT_SHIFTED,
    LayoutError,
    PipelineConfig,
    detect_layout,
    hamming_smooth,
    hampel_filter,
    normalise,
    parse_line,
    parse_lines,
    remove_null_pilot,
    run_pipeline,
    select_mac,
    sliding_windows,
    wavelet_denoise,
)

HERE = pathlib.Path(__file__).resolve().parent
FFT_FIXTURE = HERE / "testdata" / "live_lltf_fft.txt"
SHIFTED_FIXTURE = HERE.parent / "python_utils" / "example_csi.csv"

_HEADER = "CSI_DATA,PASSIVE,A8:6E:84:93:EE:60,-48,11,0,0,0,1,1,0,0,0,0,-92,0,6,0,123456,0,90,0,0,0.0,128"


def _synthetic_fft_rows(n):
    """Rows with the fft guard band nulled, so detect_layout succeeds."""
    rng = np.random.default_rng(3)
    for _ in range(n):
        vals = []
        for k in range(64):
            if k == 0 or 27 <= k <= 37:
                vals += [0, 0]
            else:
                vals += [int(rng.integers(-40, 40)), int(rng.integers(-40, 40))]
        yield vals


def _row(values, header=_HEADER):
    return f"{header},[{' '.join(str(v) for v in values)} ]"


# --- layout definitions ----------------------------------------------------


def test_both_layouts_have_48_data_bins():
    for layout in (LAYOUT_FFT, LAYOUT_SHIFTED):
        assert len(layout.data_bins) == 48, layout.name
        # 802.11 20 MHz OFDM: 48 data + 4 pilot + 12 null = 64.
        assert len(layout.guard) + len(layout.pilots) == 16, layout.name
        assert not (layout.guard & layout.pilots), layout.name


def test_pilot_bins_map_to_subcarriers_7_and_21():
    for layout in (LAYOUT_FFT, LAYOUT_SHIFTED):
        mapped = sorted(layout.to_subcarrier[i] for i in layout.pilots)
        assert mapped == [-21, -7, 7, 21], (layout.name, mapped)


def test_guard_bins_map_to_edge_subcarriers():
    for layout in (LAYOUT_FFT, LAYOUT_SHIFTED):
        mapped = sorted(abs(layout.to_subcarrier[i]) for i in layout.guard)
        # DC plus |subcarrier| >= 27 out to the band edge.
        assert mapped[0] == 0, layout.name
        assert all(v >= 27 for v in mapped[1:]), (layout.name, mapped)


# --- parsing ---------------------------------------------------------------


def test_parse_valid_row():
    values = list(range(-64, 64))
    frame = parse_line(_row(values))
    assert frame is not None
    assert frame.mac == "A8:6E:84:93:EE:60"
    assert frame.rssi == -48
    assert frame.channel == 6
    assert frame.local_timestamp == 123456
    assert frame.amplitude.shape == (64,)
    # ESP-IDF packs (imag, real): bin 0 is (-64, -63).
    assert np.isclose(frame.amplitude[0], np.hypot(-64, -63))


def test_parse_rejects_ansi_corrupted_row():
    values = [str(v) for v in range(128)]
    values[40] = "1\x1b[0;31mE"
    assert parse_line(f"{_HEADER},[{' '.join(values)} ]") is None


def test_parse_rejects_short_array():
    assert parse_line(_row(list(range(120)))) is None


def test_parse_rejects_wrong_header_field_count():
    # 3 fields but non-numeric len -> a truncated FULL row, not a MINIMAL one.
    assert parse_line(_row(list(range(128)), header="CSI_DATA,PASSIVE,-48")) is None
    # An unrecognised field count is rejected outright.
    assert parse_line(_row(list(range(128)), header="CSI_DATA,1,2,3,4")) is None


def test_parse_minimal_format():
    """Firmware seen on this hardware emits CSI_DATA,<len>,<rssi>,[...]."""
    frame = parse_line(_row(list(range(-64, 64)), header="CSI_DATA,128,-55"))
    assert frame is not None
    assert frame.rssi == -55
    assert frame.mac == SYNTHETIC_MAC
    assert frame.channel is None and frame.local_timestamp is None
    assert frame.amplitude.shape == (64,)
    assert np.isclose(frame.amplitude[0], np.hypot(-64, -63))


def test_pipeline_works_without_device_timestamps():
    """Minimal-format frames have no clock; the pipeline must still run."""
    rows = [_row(list(rng_row), header="CSI_DATA,128,-55")
            for rng_row in _synthetic_fft_rows(200)]
    frames, _ = parse_lines(rows)
    assert len(frames) == 200
    result = run_pipeline(frames, PipelineConfig(window_length=32, window_stride=8))
    assert result.stats["timing"] == "assumed"
    assert result.stats["rate_hz"] > 0
    assert result.raw.shape[1] == 48


def test_parse_ignores_non_csi_lines():
    assert parse_line("E (1288821) task_wdt: Tasks currently running:") is None
    assert parse_line("") is None


def test_parse_trusts_actual_count_not_len_field():
    """With CONFIG_SHOULD_COLLECT_ONLY_LLTF the firmware prints len=384 but
    emits 128 values, so the header field must not gate parsing."""
    header = _HEADER.rsplit(",", 1)[0] + ",384"
    frame = parse_line(_row(list(range(128)), header=header))
    assert frame is not None and frame.amplitude.shape == (64,)


def test_parse_truncates_to_lltf_when_htltf_present():
    frame = parse_line(_row(list(range(-128, 128))))
    assert frame is not None and frame.amplitude.shape == (64,)


# --- layout detection ------------------------------------------------------


def _fixture_frames(path):
    frames, report = parse_lines(path.read_text().splitlines())
    assert frames, f"no frames parsed from {path}"
    return frames, report


def test_detect_fft_layout_from_real_capture():
    frames, report = _fixture_frames(FFT_FIXTURE)
    assert report["rejected"] > 0, "fixture should contain corrupted rows"
    frames, _ = select_mac(frames)
    amplitude = np.stack([f.amplitude for f in frames])
    assert detect_layout(amplitude) is LAYOUT_FFT


def test_detect_shifted_layout_from_repo_example():
    frames, _ = _fixture_frames(SHIFTED_FIXTURE)
    amplitude = np.stack([f.amplitude for f in frames])
    assert detect_layout(amplitude) is LAYOUT_SHIFTED


def test_detect_layout_rejects_unrecognisable_data():
    rng = np.random.default_rng(0)
    try:
        detect_layout(np.abs(rng.normal(size=(200, 64))) + 1.0)
    except LayoutError:
        return
    raise AssertionError("expected LayoutError for data with no guard bands")


def test_detect_layout_needs_enough_frames():
    try:
        detect_layout(np.zeros((5, 64)))
    except LayoutError:
        return
    raise AssertionError("expected LayoutError for too few frames")


def test_remove_null_pilot_drops_guard_bands():
    frames, _ = _fixture_frames(FFT_FIXTURE)
    frames, _ = select_mac(frames)
    amplitude = np.stack([f.amplitude for f in frames])
    data = remove_null_pilot(amplitude, LAYOUT_FFT)
    assert data.shape[1] == 48
    # Every remaining bin must actually carry signal.
    assert (data.max(axis=0) > 0).all()


# --- filter stages ---------------------------------------------------------


def test_hampel_replaces_spike_with_local_median():
    signal = np.full((101, 2), 5.0)
    signal += np.linspace(0, 0.4, 101)[:, None]
    signal[50, 0] = 500.0
    filtered, mask = hampel_filter(signal, window=11, n_sigma=3.0)
    assert mask[50, 0]
    assert filtered[50, 0] < 6.0
    assert not mask[:, 1].any()


def test_hampel_leaves_clean_signal_alone():
    t = np.linspace(0, 4 * np.pi, 200)
    signal = np.stack([np.sin(t), np.cos(t)], axis=1)
    filtered, mask = hampel_filter(signal, window=11, n_sigma=3.0)
    assert mask.mean() < 0.05
    assert np.allclose(filtered[~mask.any(axis=1)], signal[~mask.any(axis=1)])


def test_hampel_handles_constant_input():
    signal = np.full((50, 3), 7.0)
    filtered, mask = hampel_filter(signal)
    assert not mask.any()
    assert np.allclose(filtered, 7.0)


def test_hampel_short_input_is_passthrough():
    signal = np.arange(6, dtype=float).reshape(3, 2)
    filtered, mask = hampel_filter(signal, window=11)
    assert np.allclose(filtered, signal) and not mask.any()


def test_hamming_smooth_preserves_length_and_level():
    rng = np.random.default_rng(1)
    clean = np.linspace(0, 1, 300)[:, None] * np.ones((1, 3)) + 10.0
    noisy = clean + rng.normal(scale=0.5, size=clean.shape)
    smoothed = hamming_smooth(noisy, window=9)
    assert smoothed.shape == noisy.shape
    # Noise is reduced, and no DC offset is introduced at the edges.
    assert np.std(smoothed - clean) < np.std(noisy - clean)
    assert abs(smoothed[0].mean() - clean[0].mean()) < 1.0
    assert abs(smoothed[-1].mean() - clean[-1].mean()) < 1.0


def test_hamming_smooth_preserves_constant():
    signal = np.full((100, 2), 3.5)
    assert np.allclose(hamming_smooth(signal, window=9), 3.5)


def test_wavelet_denoise_reduces_noise_and_keeps_length():
    rng = np.random.default_rng(2)
    t = np.linspace(0, 6 * np.pi, 512)
    clean = np.stack([np.sin(t), np.sin(t / 2)], axis=1) * 5.0
    noisy = clean + rng.normal(scale=1.0, size=clean.shape)
    denoised = wavelet_denoise(noisy)
    assert denoised.shape == noisy.shape
    assert np.std(denoised - clean) < np.std(noisy - clean)


def test_wavelet_denoise_odd_length_keeps_length():
    rng = np.random.default_rng(3)
    signal = rng.normal(size=(201, 4))
    assert wavelet_denoise(signal).shape == (201, 4)


def test_normalise_gives_zero_mean_unit_std():
    rng = np.random.default_rng(4)
    signal = rng.normal(loc=20, scale=3, size=(500, 6))
    out = normalise(signal)
    assert np.allclose(out.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(out.std(axis=0), 1, atol=1e-6)


def test_normalise_survives_dead_subcarrier():
    signal = np.zeros((50, 2))
    signal[:, 0] = np.arange(50)
    out = normalise(signal)
    assert np.isfinite(out).all()


def test_sliding_windows_shapes_and_content():
    signal = np.arange(100 * 3, dtype=float).reshape(100, 3)
    windows = sliding_windows(signal, length=20, stride=10)
    assert windows.shape == (9, 20, 3)
    assert np.allclose(windows[0], signal[0:20])
    assert np.allclose(windows[1], signal[10:30])


def test_sliding_windows_too_short_is_empty():
    assert sliding_windows(np.zeros((5, 3)), length=20, stride=5).shape == (0, 20, 3)


# --- mac selection ---------------------------------------------------------


def test_select_mac_picks_dominant_transmitter():
    frames, _ = _fixture_frames(FFT_FIXTURE)
    all_macs = {f.mac for f in frames}
    assert len(all_macs) > 1, "fixture should contain more than one transmitter"
    selected, mac = select_mac(frames)
    assert {f.mac for f in selected} == {mac}
    assert len(selected) > len(frames) / 2


# --- end to end ------------------------------------------------------------


def test_pipeline_end_to_end_on_real_capture():
    frames, _ = _fixture_frames(FFT_FIXTURE)
    config = PipelineConfig(window_length=32, window_stride=8)
    result = run_pipeline(frames, config)

    assert result.layout is LAYOUT_FFT
    assert result.raw.shape[1] == 48
    assert result.processed.shape == result.raw.shape
    assert result.windows.shape[1:] == (32, 48)
    assert np.isfinite(result.processed).all()
    assert result.stats["frames"] == len(result.raw)
    assert result.stats["rate_hz"] > 0
    assert result.timestamps[0] == 0 and (np.diff(result.timestamps) >= 0).all()
    # Denoising must not flatten the signal into nothing.
    assert result.processed.std() > 0


def test_pipeline_is_deterministic():
    frames, _ = _fixture_frames(FFT_FIXTURE)
    a = run_pipeline(frames, PipelineConfig())
    b = run_pipeline(frames, PipelineConfig())
    assert np.array_equal(a.processed, b.processed)


def test_pipeline_rejects_empty_input():
    try:
        run_pipeline([])
    except ValueError:
        return
    raise AssertionError("expected ValueError for empty frame list")


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
