"""Per-room calibration: capture, load, and the RoomProfile itself.

Why a profile exists
--------------------
An activity model trained in one room does not transfer to another. A
*calibration* can, if it is small enough to redo on site. This module defines
what has to be measured in a new room and how long it takes.

Two phases, ~2 minutes total:

* **empty (60 s)** -- the zero point. Static multipath signature, per-subcarrier
  noise floor, which bins sit in a deep fade *in this room*, and the presence
  threshold. This is the part that already worked for presence detection.
* **one person walking (60 s)** -- the gain anchor. An empty room cannot tell
  you what three people look like: it fixes the offset of the count curve but
  says nothing about its scale, and the scale depends on room size, link
  geometry and how far occupants stand off the line of sight.

The saturating count curve is ``f(N) = a + b * (1 - exp(-N/k))``. ``a`` and
``b`` come from those two phases on site. ``k`` -- the saturation constant --
is the transferable part, fitted once offline across rooms. Learning the shape
offline and calibrating only offset and gain is what makes 2 minutes enough.

The filter chain, and why counting does not use it
--------------------------------------------------
``csi_pipeline.run_pipeline`` applies Hampel, then a 9-tap Hamming FIR, then a
db4 wavelet denoise. The last two are low-pass: at ~100 Hz a 9-tap Hamming is
roughly -3 dB by 11 Hz, which is the bottom of the walking-Doppler band that
counting depends on. Running counting through the activity chain would delete
the feature it needs most.

So the counting path keeps Hampel -- impulsive outliers are genuine corruption,
about 6-8% of samples on this hardware -- and stops there, resampling straight
onto a uniform grid. This is a deliberate divergence from the activity path,
not an oversight.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time

import numpy as np

import count_features as cf
import csi_pipeline as cp
import features as F

TARGET_HZ = 100.0
WINDOW_SECONDS = 4.0
STRIDE_SECONDS = 2.0
# A subcarrier whose empty-room mean sits this far below the median subcarrier
# is in a deep fade in this room and carries mostly noise.
FADE_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_to_file(port_name: str, baud: int, seconds: float, path) -> int:
    """Record raw CSI_DATA rows to a file. Returns the row count."""
    import serial  # imported here so the module is usable without hardware

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        port = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"error: cannot open {port_name}: {exc}") from exc

    port.reset_input_buffer()
    rows = 0
    started = time.time()
    with open(path, "w") as handle:
        while time.time() - started < seconds:
            line = port.readline()
            if not line:
                continue
            text = line.decode("utf-8", "replace")
            if text.startswith("CSI_DATA"):
                handle.write(text)
                rows += 1
    port.close()
    return rows


# ---------------------------------------------------------------------------
# Load -> uniform windows -> count features
# ---------------------------------------------------------------------------


def _device_timestamps(frames: list) -> tuple:
    """Seconds from the ESP32 steady clock, or wall-clock fallback."""
    if all(f.local_timestamp is not None for f in frames):
        ticks = np.array([f.local_timestamp for f in frames], dtype=np.int64)
        wraps = np.cumsum(
            np.concatenate([[0], (np.diff(ticks) < 0).astype(np.int64)])
        )
        stamps = (ticks + wraps * (1 << 32)) / 1e6
        return stamps - stamps[0], "device-clock"
    return np.arange(len(frames), dtype=np.float64) / TARGET_HZ, "assumed"


def load_capture(
    path,
    mac: str | None = None,
    target_hz: float = TARGET_HZ,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float = STRIDE_SECONDS,
    reference: np.ndarray | None = None,
) -> dict:
    """A .csi.txt file -> uniform windows plus count features and stats."""
    lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    frames, parse_stats = cp.parse_lines(lines)
    if len(frames) < 2:
        raise SystemExit(f"error: {path} yielded {len(frames)} usable frames")

    selected, chosen_mac = cp.select_mac(frames, mac)
    if len(selected) < 2:
        # The room AP leaves channel 6 for long stretches, so a pinned MAC is
        # routinely absent from a capture. Say so plainly rather than letting
        # np.stack raise "need at least one array to stack" from three frames
        # deeper, which is what it did for most of 2026-08-20.
        seen = ", ".join(f"{m} x{n}" for m, n in
                         __import__("collections").Counter(
                             f.mac for f in frames).most_common(3))
        raise SystemExit(
            f"error: {path} has {len(selected)} frames from "
            f"{mac or chosen_mac}. Transmitters present: {seen}"
        )
    amplitude = np.stack([f.amplitude for f in selected])
    layout = cp.detect_layout(amplitude)
    data = cp.remove_null_pilot(amplitude, layout)
    stamps, timing = _device_timestamps(selected)

    # Hampel only -- see the module docstring on why the rest of the chain is
    # skipped for counting.
    data, outliers = cp.hampel_filter(data)

    values, grid, valid, resample_stats = cp.resample_uniform(
        data, stamps, target_hz=target_hz
    )

    length = int(round(window_seconds * target_hz))
    stride = int(round(stride_seconds * target_hz))
    windows = cp.sliding_windows(values, length, stride)
    # Drop any window more than a tenth interpolated across a dropout.
    keep = np.array(
        [
            valid[i : i + length].mean() > 0.9
            for i in range(0, len(values) - length + 1, stride)
        ][: len(windows)],
        dtype=bool,
    )
    windows = windows[keep] if len(windows) else windows

    feats = cf.features_for_windows(windows, target_hz, reference)
    scores = np.array([F.motion_score(w) for w in windows]) if len(windows) else np.array([])

    return {
        "path": str(path),
        "mac": chosen_mac,
        "layout": layout.name,
        "raw": data,
        "resampled": values,
        "timestamps": grid,
        "windows": windows,
        "features": feats,
        "motion_scores": scores,
        "rssi_mean": float(np.mean([f.rssi for f in selected])),
        "stats": {
            **parse_stats,
            "kept_after_mac": len(selected),
            "mac_fraction": len(selected) / len(frames),
            "timing": timing,
            "outlier_rate": float(outliers.mean()),
            "windows": int(len(windows)),
            "windows_dropped_to_gaps": int((~keep).sum()) if len(keep) else 0,
            **cp._timing_stats(stamps),
            **resample_stats,
        },
    }


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RoomProfile:
    room: str
    created: str
    geometry: dict
    link: dict
    timing: dict
    subcarriers: dict
    presence: dict
    features: dict
    notes: str = ""
    # Written by gate1_check so the live view reads measured AUCs instead of
    # carrying a hardcoded copy that silently goes stale when the gate reruns.
    gate1: dict = dataclasses.field(default_factory=dict)

    def save(self, path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "RoomProfile":
        return cls(**json.loads(pathlib.Path(path).read_text()))

    def detector(self, **kwargs) -> F.PresenceDetector:
        """Rebuild the presence detector this room was calibrated for."""
        return F.PresenceDetector(
            self.presence["baseline_median"],
            self.presence["baseline_mad"],
            **kwargs,
        )

    def reference(self) -> np.ndarray:
        """The empty room's per-subcarrier scale, for dilated_pem at inference."""
        return np.asarray(self.subcarriers["reference_mad"])

    def gain(self) -> np.ndarray:
        """Per-feature ``b``: how far one walking person moves each feature.

        Zero or negative entries mean the feature did not respond to a single
        occupant in this room; it cannot contribute to a count here.
        """
        empty = np.asarray(self.features["empty"]["median"])
        one = np.asarray(self.features["one_person"]["median"])
        return one - empty

    def normalise(self, feats: np.ndarray) -> np.ndarray:
        """Map raw count features onto the room's own (empty=0, one person=1).

        This is the room adaptation. Everything downstream sees numbers already
        expressed in units of "one walking person in this room", which is what
        lets a model trained elsewhere mean anything here.
        """
        empty = np.asarray(self.features["empty"]["median"])
        gain = self.gain()
        safe = np.where(np.abs(gain) < 1e-9, 1.0, gain)
        return (np.asarray(feats) - empty) / safe


def _robust(values: np.ndarray) -> dict:
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0) * 1.4826
    return {"median": median.tolist(), "mad": mad.tolist()}


def build_profile(
    room: str,
    empty: dict,
    one_person: dict | None = None,
    geometry: dict | None = None,
    notes: str = "",
    enter_sigma: float = 6.0,
    exit_sigma: float = 3.0,
) -> RoomProfile:
    """Turn one or two loaded captures into a RoomProfile.

    ``one_person`` may be omitted, and the profile is still valid for presence
    detection. It is **not** valid for counting: without the gain anchor there
    is no scale, only an offset. ``gain()`` returns zeros and callers must
    refuse to count rather than silently reporting nonsense.
    """
    scores = empty["motion_scores"]
    if not len(scores):
        raise SystemExit("error: empty-room capture produced no windows")
    if one_person is not None and one_person["mac"] != empty["mac"]:
        raise SystemExit(
            f"error: the two calibration phases locked onto different "
            f"transmitters ({empty['mac']} vs {one_person['mac']}).\n"
            f"The gain would measure the difference between two radio links "
            f"rather than the difference one person makes. Recapture both "
            f"phases with --mac pinned."
        )
    baseline_median = float(np.median(scores))
    baseline_mad = float(np.median(np.abs(scores - baseline_median)) * 1.4826)
    scale = max(baseline_mad, baseline_median * 0.05, F.EPS)

    mean_amp = empty["raw"].mean(axis=0)
    reliable = mean_amp > FADE_FRACTION * np.median(mean_amp)

    # Two passes. The first fixes the empty room's per-subcarrier scale; the
    # second recomputes both phases against it. Without this the two phases are
    # measured with different yardsticks and the gain is meaningless -- see
    # count_features.dilated_pem on why a window-local scale inverts.
    reference = cf.reference_mad(empty["windows"])
    empty_feats = cf.features_for_windows(empty["windows"], TARGET_HZ, reference)

    n_feats = len(cf.window_feature_names())
    empty_stats = _robust(empty_feats)
    if one_person is not None and len(one_person["windows"]):
        one_feats = cf.features_for_windows(
            one_person["windows"], TARGET_HZ, reference
        )
        one_stats = _robust(one_feats)
    else:
        one_stats = {"median": empty_stats["median"], "mad": [0.0] * n_feats}

    return RoomProfile(
        room=room,
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        geometry=geometry or {},
        link={
            "mac": empty["mac"],
            "layout": empty["layout"],
            "rssi_mean_empty": empty["rssi_mean"],
        },
        timing={
            "source_rate_hz": empty["stats"].get("rate_hz", 0.0),
            "jitter_ratio": empty["stats"].get("jitter_ratio", 0.0),
            "resampled_hz": empty["stats"].get("target_hz", TARGET_HZ),
            "gap_fraction": empty["stats"].get("gap_fraction", 0.0),
            "mac_fraction": empty["stats"].get("mac_fraction", 1.0),
        },
        subcarriers={
            "mean": mean_amp.tolist(),
            "std": empty["raw"].std(axis=0).tolist(),
            "reliable": reliable.tolist(),
            "n_reliable": int(reliable.sum()),
            "reference_mad": reference.tolist(),
        },
        presence={
            "baseline_median": baseline_median,
            "baseline_mad": baseline_mad,
            "enter_threshold": baseline_median + enter_sigma * scale,
            "exit_threshold": baseline_median + exit_sigma * scale,
            "enter_sigma": enter_sigma,
            "exit_sigma": exit_sigma,
        },
        features={
            "names": cf.window_feature_names(),
            "empty": empty_stats,
            "one_person": one_stats,
            "has_gain_anchor": one_person is not None,
        },
        notes=notes,
    )
