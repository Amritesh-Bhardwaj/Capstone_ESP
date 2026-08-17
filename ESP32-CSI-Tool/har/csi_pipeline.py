"""CSI preprocessing pipeline for ESP32 human activity recognition.

Implements the preprocessing chain from Abuhoureyah, Wong & Mohd Isira,
"WiFi-based human activity recognition through wall using deep learning",
Engineering Applications of Artificial Intelligence 127 (2024) 107171,
Section 4.1.2, adapted to the ESP32-CSI-Tool serial format:

    null/pilot subcarrier removal -> Hampel filter -> Hamming smoothing
    -> wavelet denoising -> (normalisation) -> sliding windows

The paper works on amplitude only; phase is never used. We follow that.

Subcarrier layout is DETECTED from the data rather than hardcoded, because
different ESP32 builds order the 64 LLTF bins differently -- see LAYOUTS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pywt

# ---------------------------------------------------------------------------
# Serial format
# ---------------------------------------------------------------------------

# Field order printed by _components/csi_component.h::_wifi_csi_cb.
HEADER_FIELDS = (
    "type", "role", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding", "sgi",
    "noise_floor", "ampdu_cnt", "channel", "secondary_channel",
    "local_timestamp", "ant", "sig_len", "rx_state", "real_time_set",
    "real_timestamp", "len",
)

_INT_RE = re.compile(r"^-?\d+$")

# The ESP32 emits interleaved [1m...[0m log lines that can corrupt a CSI
# row mid-array, so every token is validated before use.
LLTF_BYTES = 128  # 64 subcarriers x (imag, real), int8 each

# ---------------------------------------------------------------------------
# Subcarrier layouts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """A mapping from buffer position to 802.11 subcarrier index."""

    name: str
    guard: frozenset            # null bins: DC + guard bands
    pilots: frozenset           # pilot bins at subcarriers +/-7, +/-21
    # buffer position -> signed subcarrier number, for reporting only
    to_subcarrier: tuple

    @property
    def data_bins(self) -> list:
        """The 48 data-carrying bins, in ascending buffer order."""
        drop = self.guard | self.pilots
        return [i for i in range(64) if i not in drop]


def _fft_order():
    # position i -> subcarrier i (i <= 31), i - 64 (i >= 32)
    return tuple(i if i <= 31 else i - 64 for i in range(64))


def _shifted_order():
    # position i -> subcarrier i - 32
    return tuple(i - 32 for i in range(64))


# Standard FFT ordering: DC at position 0, guard band split across the middle.
# Observed on the ESP32 currently attached (passive build, sig_mode=0, 20 MHz).
LAYOUT_FFT = Layout(
    name="fft",
    guard=frozenset({0}) | frozenset(range(27, 38)),
    pilots=frozenset({7, 21, 43, 57}),
    to_subcarrier=_fft_order(),
)

# Shifted ordering: subcarrier -32 at position 0, DC in the middle.
# Observed in ESP32-CSI-Tool/python_utils/example_csi.csv.
LAYOUT_SHIFTED = Layout(
    name="shifted",
    guard=frozenset(range(0, 6)) | frozenset({32}) | frozenset(range(59, 64)),
    pilots=frozenset({11, 25, 39, 53}),
    to_subcarrier=_shifted_order(),
)

LAYOUTS = (LAYOUT_FFT, LAYOUT_SHIFTED)

# Bins that should read exactly zero for each layout: the guard bands only.
# The DC bin is excluded because some builds put a non-zero constant there
# instead of a true null.
#
# Not every guard bin is reliably zero. In ESP32-CSI-Tool's own
# example_csi.csv the outermost guard bins (subcarriers -31 and +/-27) carry a
# few counts of spectral leakage, so a layout is scored by the FRACTION of its
# guard signature that is null, not by an absolute count. The two signatures
# are disjoint, which makes the comparison unambiguous.
_GUARD_SIGNATURE = {
    "fft": frozenset(range(27, 38)),
    "shifted": frozenset(range(1, 6)) | frozenset(range(59, 64)),
}


class LayoutError(RuntimeError):
    """Raised when the subcarrier layout cannot be determined from the data."""


def detect_layout(
    amplitude: np.ndarray, min_score: float = 0.6, min_margin: float = 0.3
) -> Layout:
    """Infer the subcarrier layout from the position of the guard bands.

    The 802.11 20 MHz guard bands are transmitted as nulls, so the bins that
    read zero in (almost) every frame identify the ordering.

    Args:
        amplitude: (frames, 64) subcarrier amplitudes.
        min_score: fraction of the winning layout's guard bins that must be
            null for the match to be accepted.
        min_margin: how far the winner must lead the runner-up.

    Raises:
        LayoutError: if neither known layout matches clearly.
    """
    if amplitude.ndim != 2 or amplitude.shape[1] != 64:
        raise ValueError(f"expected (frames, 64) amplitudes, got {amplitude.shape}")
    if len(amplitude) < 10:
        raise LayoutError(
            f"need at least 10 frames to detect the layout, got {len(amplitude)}"
        )

    null_frac = (amplitude == 0).mean(axis=0)
    observed = {i for i in range(64) if null_frac[i] > 0.99}

    def score(layout: Layout) -> float:
        signature = _GUARD_SIGNATURE[layout.name]
        return len(observed & signature) / len(signature)

    scored = sorted(((score(lay), lay) for lay in LAYOUTS), key=lambda p: p[0],
                    reverse=True)
    (best_score, best), (runner_score, runner) = scored[0], scored[1]

    if best_score < min_score:
        raise LayoutError(
            "could not match a known subcarrier layout; null bins were "
            f"{sorted(observed)}. Best candidate {best.name!r} matched only "
            f"{best_score:.0%} of its guard bins. Capture more frames, or "
            "check the firmware CSI configuration."
        )
    if best_score - runner_score < min_margin:
        raise LayoutError(
            f"ambiguous layout: {best.name!r} scored {best_score:.0%} and "
            f"{runner.name!r} scored {runner_score:.0%} (null bins "
            f"{sorted(observed)})"
        )
    return best


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    mac: str
    rssi: int
    channel: int | None
    sig_mode: int | None
    bandwidth: int | None
    local_timestamp: int | None   # microseconds; None if the firmware omits it
    amplitude: np.ndarray         # (64,) float


# Two serial formats occur in the wild:
#
#   FULL    25 header fields, from the stock ESP32-CSI-Tool builds:
#           CSI_DATA,role,mac,rssi,...,len,[...]
#   MINIMAL  3 header fields, from cut-down firmware seen on this hardware:
#           CSI_DATA,len,rssi,[...]
#
# The minimal form carries no MAC, channel or device timestamp, so MAC
# selection and timestamp-based rate estimation must degrade gracefully.
MINIMAL_HEADER_FIELDS = 3
SYNTHETIC_MAC = "00:00:00:00:00:00"


def _amplitude(tokens) -> np.ndarray:
    raw = np.fromiter(tokens, dtype=np.int16, count=LLTF_BYTES)
    # ESP-IDF packs each subcarrier as (imaginary, real); see the CSI_PHASE
    # branch of csi_component.h, which calls atan2(buf[2i], buf[2i+1]).
    return np.hypot(raw[1::2].astype(np.float64), raw[0::2].astype(np.float64))


def parse_line(line: str) -> Frame | None:
    """Parse one CSI_DATA row of either format. None if malformed.

    Only the first 128 bytes (the LLTF field, 64 subcarriers) are used. The
    ``len`` header field cannot be trusted for this: with
    CONFIG_SHOULD_COLLECT_ONLY_LLTF the firmware prints ``data->len`` (e.g.
    384) but emits only 128 values.
    """
    if not line.startswith("CSI_DATA,"):
        return None
    open_at = line.find("[")
    close_at = line.find("]", open_at + 1)
    if open_at < 0 or close_at < 0:
        return None

    header = line[:open_at].rstrip(",").split(",")
    tokens = line[open_at + 1:close_at].split()
    if len(tokens) < LLTF_BYTES:
        return None
    tokens = tokens[:LLTF_BYTES]
    if not all(_INT_RE.match(tok) for tok in tokens):
        return None  # ANSI log output bled into the array

    if len(header) == len(HEADER_FIELDS):
        try:
            return Frame(
                mac=header[2],
                rssi=int(header[3]),
                sig_mode=int(header[5]),
                bandwidth=int(header[7]),
                channel=int(header[16]),
                local_timestamp=int(header[18]),
                amplitude=_amplitude(tokens),
            )
        except ValueError:
            return None

    if len(header) == MINIMAL_HEADER_FIELDS:
        # Both fields must be numeric (len, rssi). Requiring this stops a
        # truncated FULL header -- "CSI_DATA,PASSIVE,-48" -- from being
        # mistaken for a valid minimal row.
        if not _INT_RE.match(header[1]):
            return None
        try:
            rssi = int(header[2])
        except ValueError:
            return None
        return Frame(
            mac=SYNTHETIC_MAC,      # one implicit transmitter
            rssi=rssi,
            channel=None,
            sig_mode=None,
            bandwidth=None,
            local_timestamp=None,   # caller must fall back to arrival time
            amplitude=_amplitude(tokens),
        )

    return None


def parse_lines(lines) -> tuple[list, dict]:
    """Parse an iterable of text lines into Frames plus a rejection summary."""
    frames, rejected = [], 0
    for line in lines:
        frame = parse_line(line.strip())
        if frame is None:
            if line.startswith("CSI_DATA,"):
                rejected += 1
            continue
        frames.append(frame)
    return frames, {"parsed": len(frames), "rejected": rejected}


def select_mac(frames: list, mac: str | None = None) -> tuple[list, str]:
    """Keep frames from a single transmitter.

    A passive-mode ESP32 sniffs every station on the channel. Interleaving
    those links produces a time series that jumps between unrelated channels,
    so a single MAC must be selected before any temporal filtering.
    """
    if not frames:
        return [], ""
    if mac is None:
        counts: dict = {}
        for frame in frames:
            counts[frame.mac] = counts.get(frame.mac, 0) + 1
        mac = max(counts, key=counts.get)
    return [f for f in frames if f.mac == mac], mac


# ---------------------------------------------------------------------------
# Preprocessing stages (paper Section 4.1.2)
# ---------------------------------------------------------------------------


def remove_null_pilot(amplitude: np.ndarray, layout: Layout) -> np.ndarray:
    """Stage 1: drop null and pilot subcarriers, leaving 48 data bins.

    "These subcarriers, utilized for channel estimation and synchronization,
    need to carry important information for subsequent analysis and are thus
    discarded." -- paper, Section 4.1.2.
    """
    return amplitude[:, layout.data_bins]


def hampel_filter(
    signal: np.ndarray, window: int = 11, n_sigma: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2: Hampel outlier removal along time, independently per subcarrier.

    Replaces any sample more than ``n_sigma`` scaled MADs from its local median
    with that median. ``window`` is the full width and is forced odd.

    Returns the filtered array and a boolean mask of replaced samples.
    """
    if signal.ndim != 2:
        raise ValueError("expected a (frames, subcarriers) array")
    window = max(3, int(window) | 1)
    half = window // 2
    if len(signal) < window:
        return signal.copy(), np.zeros_like(signal, dtype=bool)

    padded = np.pad(signal, ((half, half), (0, 0)), mode="edge")
    views = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
    medians = np.median(views, axis=-1)
    mad = np.median(np.abs(views - medians[..., None]), axis=-1)
    # 1.4826 makes the MAD a consistent estimator of sigma for Gaussian noise.
    threshold = n_sigma * 1.4826 * mad

    outliers = np.abs(signal - medians) > threshold
    # A zero threshold means the window is constant; nothing can be an outlier.
    outliers &= threshold > 0
    return np.where(outliers, medians, signal), outliers


def hamming_smooth(signal: np.ndarray, window: int = 9) -> np.ndarray:
    """Stage 3: Hamming-window FIR smoothing along time.

    "Hamming filters, belonging to the family of windowing filters, effectively
    smooth the data by attenuating abrupt variations and suppressing
    high-frequency noise." -- paper, Section 4.1.2.

    Edges are reflected so the filter does not pull the first and last samples
    toward zero.
    """
    window = max(3, int(window) | 1)
    if len(signal) < window:
        return signal.copy()
    taps = np.hamming(window)
    taps /= taps.sum()
    half = window // 2
    padded = np.pad(signal, ((half, half), (0, 0)), mode="reflect")
    out = np.empty_like(signal, dtype=np.float64)
    for col in range(signal.shape[1]):
        out[:, col] = np.convolve(padded[:, col], taps, mode="valid")
    return out


def wavelet_denoise(
    signal: np.ndarray, wavelet: str = "db4", level: int | None = None
) -> np.ndarray:
    """Stage 4: wavelet denoising by soft-thresholding the detail coefficients.

    "The noise components are attenuated by selective thresholding and
    discarding the detail coefficients at each level while the essential signal
    characteristics are preserved." -- paper, Section 4.1.2.

    Uses the universal (VisuShrink) threshold with the noise level estimated
    from the finest detail band, which is the standard choice the paper's
    description implies but does not specify.
    """
    n_frames = len(signal)
    max_level = pywt.dwt_max_level(n_frames, pywt.Wavelet(wavelet).dec_len)
    if max_level < 1:
        return signal.copy()
    level = max_level if level is None else min(int(level), max_level)

    out = np.empty_like(signal, dtype=np.float64)
    for col in range(signal.shape[1]):
        coeffs = pywt.wavedec(signal[:, col], wavelet, level=level)
        finest = coeffs[-1]
        sigma = np.median(np.abs(finest)) / 0.6745 if finest.size else 0.0
        if sigma > 0:
            threshold = sigma * np.sqrt(2.0 * np.log(n_frames))
            coeffs[1:] = [
                pywt.threshold(c, threshold, mode="soft") for c in coeffs[1:]
            ]
        # waverec can return one extra sample for odd-length inputs.
        out[:, col] = pywt.waverec(coeffs, wavelet)[:n_frames]
    return out


def normalise(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-subcarrier z-score, so no bin dominates the LSTM by scale alone."""
    mean = signal.mean(axis=0, keepdims=True)
    std = signal.std(axis=0, keepdims=True)
    return (signal - mean) / (std + eps)


def sliding_windows(signal: np.ndarray, length: int, stride: int) -> np.ndarray:
    """Cut the stream into (n_windows, length, subcarriers) LSTM sequences."""
    if length <= 0 or stride <= 0:
        raise ValueError("length and stride must be positive")
    if len(signal) < length:
        return np.empty((0, length, signal.shape[1]), dtype=signal.dtype)
    starts = range(0, len(signal) - length + 1, stride)
    return np.stack([signal[s:s + length] for s in starts])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    hampel_window: int = 11
    hampel_sigma: float = 3.0
    hamming_window: int = 9
    wavelet: str = "db4"
    wavelet_level: int | None = None
    normalise: bool = True
    window_length: int = 64
    window_stride: int = 16
    # Used only when the firmware ships no device timestamps.
    assumed_rate_hz: float = 100.0


@dataclass
class PipelineResult:
    layout: Layout
    mac: str
    raw: np.ndarray            # (frames, 48) after null/pilot removal
    processed: np.ndarray      # (frames, 48) after the full chain
    windows: np.ndarray        # (n, window_length, 48)
    timestamps: np.ndarray     # (frames,) seconds, from the ESP32 steady clock
    stats: dict = field(default_factory=dict)


def _timing_stats(timestamps: np.ndarray) -> dict:
    """Sample-rate statistics. The paper assumes a uniform rate; the ESP32
    delivers CSI only when a packet arrives, so the jitter is worth reporting.
    """
    if len(timestamps) < 2:
        return {"rate_hz": 0.0, "jitter_ratio": 0.0, "duration_s": 0.0}
    deltas = np.diff(timestamps)
    deltas = deltas[deltas > 0]
    if not len(deltas):
        return {"rate_hz": 0.0, "jitter_ratio": 0.0, "duration_s": 0.0}
    median = float(np.median(deltas))
    return {
        "rate_hz": 1.0 / median,
        "jitter_ratio": float(np.std(deltas) / median),
        "duration_s": float(timestamps[-1] - timestamps[0]),
    }


def run_pipeline(
    frames: list,
    config: PipelineConfig | None = None,
    mac: str | None = None,
    layout: Layout | None = None,
) -> PipelineResult:
    """Run the full preprocessing chain over parsed frames."""
    config = config or PipelineConfig()
    frames, chosen_mac = select_mac(frames, mac)
    if not frames:
        raise ValueError("no frames to process")

    amplitude = np.stack([f.amplitude for f in frames])
    layout = layout or detect_layout(amplitude)

    if all(f.local_timestamp is not None for f in frames):
        # The ESP32 steady clock is in microseconds and wraps; unwrap defensively.
        ticks = np.array([f.local_timestamp for f in frames], dtype=np.int64)
        wraps = np.cumsum(np.concatenate([[0], (np.diff(ticks) < 0).astype(np.int64)]))
        timestamps = (ticks + wraps * (1 << 32)) / 1e6
        timestamps -= timestamps[0]
        timing = "device-clock"
    else:
        # Minimal-format firmware ships no timestamps. Assume a uniform grid so
        # the band features have a frequency axis, and say so in the stats --
        # jitter is unmeasurable in this mode.
        timestamps = np.arange(len(frames), dtype=np.float64) / config.assumed_rate_hz
        timing = "assumed"

    raw = remove_null_pilot(amplitude, layout)

    stage = raw
    stage, outliers = hampel_filter(stage, config.hampel_window, config.hampel_sigma)
    stage = hamming_smooth(stage, config.hamming_window)
    stage = wavelet_denoise(stage, config.wavelet, config.wavelet_level)
    if config.normalise:
        stage = normalise(stage)

    windows = sliding_windows(stage, config.window_length, config.window_stride)

    stats = {
        "frames": len(frames),
        "subcarriers": raw.shape[1],
        "mac": chosen_mac,
        "layout": layout.name,
        "outlier_rate": float(outliers.mean()),
        "rssi_mean": float(np.mean([f.rssi for f in frames])),
        "n_windows": int(len(windows)),
        "timing": timing,
        **_timing_stats(timestamps),
    }
    return PipelineResult(
        layout=layout,
        mac=chosen_mac,
        raw=raw,
        processed=stage,
        windows=windows,
        timestamps=timestamps,
        stats=stats,
    )
