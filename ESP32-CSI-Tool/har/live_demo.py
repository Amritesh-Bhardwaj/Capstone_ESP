#!/usr/bin/env python
"""Live human presence and activity demo.

    # Presence only -- needs no trained model, only a calibration file
    python har/live_demo.py --calibrate-only

    # Full demo with the trained classifier
    python har/live_demo.py --model har/model.joblib

    # Replay a recording instead of the radio. Use this if the venue's RF is
    # hostile, or to rehearse. The display is identical.
    python har/live_demo.py --model har/model.joblib --replay har/recordings/walking_x.csv

Press q to quit.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import threading
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import (  # noqa: E402
    detect_layout,
    hamming_smooth,
    hampel_filter,
    parse_line,
    remove_null_pilot,
    wavelet_denoise,
)
from features import PresenceDetector, motion_score, window_features  # noqa: E402

DEFAULT_PORT = "/dev/cu.usbserial-57460201261"

# Validated palette (see har/README.md).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedcd5"
SURFACE = "#fcfcfb"
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                   "#184f95", "#0d366b"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="model.joblib from train_activity.py")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="presence only: calibrate on an empty room, then run")
    parser.add_argument("--calibrate-seconds", type=float, default=15.0)
    parser.add_argument("--enter-sigma", type=float, default=6.0,
                        help="how far above the empty-room floor counts as "
                             "present; lower = more sensitive")
    parser.add_argument("--exit-sigma", type=float, default=3.0,
                        help="drop-out threshold; must be <= enter-sigma")
    parser.add_argument("--enter-threshold", type=float, default=None,
                        help="absolute motion_score to declare presence. Skips "
                             "calibration entirely -- use a value fitted from "
                             "labelled data (see har/monitor.py)")
    parser.add_argument("--exit-threshold", type=float, default=None,
                        help="absolute drop-out score; must be <= enter-threshold")
    parser.add_argument("--baseline", type=float, default=None,
                        help="observed quiet-room median, for the display only")
    parser.add_argument("--replay", default=None, help="replay a recording instead of serial")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mac", default=None)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--history", type=int, default=240, help="motion trace length")
    parser.add_argument("--max-span-factor", type=float, default=2.5,
                        help="reject a window covering more than this multiple "
                             "of its expected wall-clock duration (dropout guard)")
    parser.add_argument("--smoothing", type=int, default=5,
                        help="running median over the last N windows")
    parser.add_argument("--debounce", type=int, default=3,
                        help="consecutive windows needed to flip state")
    parser.add_argument("--snapshot", default=None,
                        help="run headless for a few seconds, save a PNG, exit "
                             "(for slides)")
    return parser.parse_args(argv)


class FrameSource:
    """Background reader that keeps a rolling buffer of frames for one MAC."""

    def __init__(self, maxlen: int):
        self.frames = collections.deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.mac = None
        self.layout = None
        self.running = True
        self.total = 0
        self.rejected = 0

    def push(self, line: str):
        frame = parse_line(line)
        if frame is None:
            if line.startswith("CSI_DATA,"):
                self.rejected += 1
            return
        if self.mac is not None and frame.mac != self.mac:
            return
        with self.lock:
            # Arrival time is the fallback clock: minimal-format firmware ships
            # no device timestamp.
            self.frames.append((frame, time.time()))
            self.total += 1

    def snapshot(self, n: int):
        with self.lock:
            if len(self.frames) < n:
                return None
            return np.stack([f.amplitude for f, _ in list(self.frames)[-n:]])

    def snapshot_span(self, n: int):
        """(amplitudes, wall-clock seconds the window covers) or None.

        The window is n *frames*, not n seconds. If the board drops out or the
        transmitter goes quiet, those n frames can straddle a long gap, and the
        slow environmental drift across it looks exactly like motion. The demo
        uses the span to reject such windows instead of reporting a person.
        """
        with self.lock:
            if len(self.frames) < n:
                return None
            recent = list(self.frames)[-n:]
        amplitude = np.stack([f.amplitude for f, _ in recent])
        if recent[0][0].local_timestamp is not None:
            ticks = np.asarray([f.local_timestamp for f, _ in recent], dtype=np.int64)
            deltas = np.diff(ticks)
            deltas = deltas[(deltas > 0) & (deltas < 5_000_000)]
            span = float(deltas.sum() / 1e6) if len(deltas) else 0.0
        else:
            span = float(recent[-1][1] - recent[0][1])
        return amplitude, span

    def rate(self) -> float:
        """Sample rate from the ESP32's own clock, not wall-clock arrival.

        This is what the band-power features need, and it stays correct when a
        recording is replayed faster than real time.
        """
        with self.lock:
            recent = list(self.frames)[-64:]
        if len(recent) < 2:
            return 0.0
        if recent[0][0].local_timestamp is not None:
            ticks = np.asarray([f.local_timestamp for f, _ in recent], dtype=np.int64)
            deltas = np.diff(ticks)
            deltas = deltas[(deltas > 0) & (deltas < 1_000_000)]  # drop wraps/gaps
            return float(1e6 / np.median(deltas)) if len(deltas) else 0.0
        span = recent[-1][1] - recent[0][1]
        return float((len(recent) - 1) / span) if span > 0 else 0.0


def serial_reader(source: FrameSource, port_name: str, baud: int):
    import serial

    try:
        port = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as exc:
        print(f"error: cannot open {port_name}: {exc}", file=sys.stderr)
        source.running = False
        return
    while source.running:
        try:
            line = port.readline().decode("utf-8", "ignore").strip()
        except Exception:  # noqa: BLE001
            continue
        if line:
            source.push(line)
    port.close()


def replay_reader(source: FrameSource, path: pathlib.Path, speed: float):
    lines = [l for l in path.read_text().splitlines() if l.startswith("CSI_DATA,")]
    if not lines:
        print(f"error: no CSI rows in {path}", file=sys.stderr)
        source.running = False
        return
    print(f"replaying {len(lines)} frames from {path.name} at {speed:g}x (loops)")
    previous = None
    while source.running:
        for line in lines:
            if not source.running:
                return
            frame = parse_line(line)
            if frame is not None and previous is not None:
                delta = (frame.local_timestamp - previous) / 1e6
                if 0 < delta < 1.0:
                    time.sleep(delta / max(speed, 0.01))
            if frame is not None:
                previous = frame.local_timestamp
            source.push(line)


def bootstrap(source: FrameSource, needed: int, mac: str | None, timeout: float = 30.0):
    """Wait for enough frames, then lock in the MAC and subcarrier layout."""
    print("waiting for CSI ...")
    deadline = time.time() + timeout
    while time.time() < deadline and source.running:
        with source.lock:
            frames = [f for f, _ in source.frames]
        if len(frames) >= max(needed, 60):
            break
        time.sleep(0.2)

    with source.lock:
        frames = [f for f, _ in source.frames]
    if len(frames) < 20:
        raise SystemExit("error: not enough CSI frames -- is the board streaming?")

    if mac is None:
        counts = collections.Counter(f.mac for f in frames)
        mac = counts.most_common(1)[0][0]
        if len(counts) > 1:
            print(f"transmitters seen: {dict(counts)}")
    source.mac = mac

    amplitude = np.stack([f.amplitude for f in frames if f.mac == mac])
    source.layout = detect_layout(amplitude)
    with source.lock:
        source.frames = collections.deque(
            [(f, t) for f, t in source.frames if f.mac == mac],
            maxlen=source.frames.maxlen,
        )
    print(f"locked to {mac}, layout '{source.layout.name}' "
          f"-> {len(source.layout.data_bins)} data bins")
    return mac


def preprocess(amplitude: np.ndarray, layout) -> np.ndarray:
    """The paper's chain, minus normalisation (features need real scale)."""
    stage = remove_null_pilot(amplitude, layout)
    stage, _ = hampel_filter(stage, window=11, n_sigma=3.0)
    stage = hamming_smooth(stage, window=9)
    return wavelet_denoise(stage, "db4")


def calibrate_live(source: FrameSource, window_length: int, seconds: float,
                   enter_sigma: float = 6.0, exit_sigma: float = 3.0,
                   smoothing: int = 5, debounce: int = 3):
    print(f"\n>>> CALIBRATING: leave the area empty for {seconds:.0f}s <<<")
    for remaining in range(int(seconds), 0, -1):
        print(f"\r    {remaining:3d}s ", end="", flush=True)
        time.sleep(1.0)
    print("\r    done.      ")

    windows = []
    with source.lock:
        frames = [f.amplitude for f, _ in source.frames]
    for start in range(0, max(1, len(frames) - window_length + 1), 8):
        chunk = frames[start:start + window_length]
        if len(chunk) == window_length:
            windows.append(preprocess(np.stack(chunk), source.layout))
    if not windows:
        raise SystemExit("error: calibration captured too few frames")
    scores = np.array([motion_score(w) for w in windows])
    detector = PresenceDetector.calibrate(
        np.stack(windows), enter_sigma=enter_sigma, exit_sigma=exit_sigma,
        smoothing=smoothing, debounce=debounce)
    print(f"calibrated on {len(windows)} empty windows: {detector.describe()}")
    print(f"  ambient motion score ranged {scores.min():.3f}-{scores.max():.3f}")
    if detector.enter_threshold > 2 * detector.baseline:
        print("  note: the ambient floor was noisy, so the threshold sits well\n"
              "        above it. If presence never triggers, re-run with a\n"
              "        lower --enter-sigma (try 3).")
    return detector


def main(argv=None) -> int:
    args = parse_args(argv)

    bundle = None
    if args.model:
        import joblib
        bundle = joblib.load(args.model)
        print(f"loaded {args.model}: labels {bundle['labels']}")

    source = FrameSource(maxlen=max(args.window_length * 6, 600))
    if args.replay:
        path = pathlib.Path(args.replay)
        if not path.exists():
            raise SystemExit(f"error: no such file: {path}")
        thread = threading.Thread(target=replay_reader,
                                  args=(source, path, args.replay_speed), daemon=True)
    else:
        thread = threading.Thread(target=serial_reader,
                                  args=(source, args.port, args.baud), daemon=True)
    thread.start()

    try:
        bootstrap(source, args.window_length, args.mac)
    except SystemExit:
        source.running = False
        raise

    detector = bundle.get("presence") if bundle else None

    if args.enter_threshold is not None:
        # Fitted thresholds beat a calibration that may itself be contaminated.
        exit_threshold = (args.exit_threshold if args.exit_threshold is not None
                          else args.enter_threshold * 0.9)
        baseline = args.baseline if args.baseline is not None else exit_threshold * 0.8
        detector = PresenceDetector(baseline, 1.0, enter_sigma=1.0, exit_sigma=1.0,
                                    smoothing=args.smoothing, debounce=args.debounce)
        detector.enter_threshold = args.enter_threshold
        detector.exit_threshold = exit_threshold
        detector.scale = max(args.enter_threshold - baseline, 1e-6)
        print(f"using FITTED thresholds: enter > {detector.enter_threshold:.4f}, "
              f"exit < {detector.exit_threshold:.4f} (baseline {baseline:.4f}) "
              "— no calibration")
    elif detector is None or args.calibrate_only:
        detector = calibrate_live(source, args.window_length, args.calibrate_seconds,
                                  args.enter_sigma, args.exit_sigma,
                                  args.smoothing, args.debounce)

    import matplotlib
    if args.snapshot:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap

    labels = bundle["labels"] if bundle else []
    history = collections.deque([np.nan] * args.history, maxlen=args.history)

    fig = plt.figure(figsize=(13, 8.5), facecolor=SURFACE)
    fig.canvas.manager.set_window_title("WiFi CSI — Human Presence & Activity")
    grid = fig.add_gridspec(3, 2, height_ratios=[0.8, 1, 1.3], width_ratios=[1, 1],
                            hspace=0.45, wspace=0.22)

    ax_status = fig.add_subplot(grid[0, :]); ax_status.axis("off")
    status_text = ax_status.text(0.01, 0.62, "—", fontsize=40, fontweight="bold",
                                 color=MUTED, va="center", transform=ax_status.transAxes)
    activity_text = ax_status.text(0.01, 0.12, "", fontsize=20, color=INK,
                                   va="center", transform=ax_status.transAxes)
    meta_text = ax_status.text(0.99, 0.62, "", fontsize=11, color=MUTED,
                               va="center", ha="right", transform=ax_status.transAxes)

    ax_motion = fig.add_subplot(grid[1, :])
    motion_line, = ax_motion.plot(range(args.history), list(history), color=BLUE, linewidth=1.8)
    ax_motion.axhline(detector.enter_threshold, color=ORANGE, linewidth=1.2,
                      linestyle="--", label="presence threshold")
    ax_motion.axhline(detector.baseline, color=GRID, linewidth=1.2,
                      linestyle=":", label="empty-room baseline")
    ax_motion.set_ylabel("motion score", color=MUTED, fontsize=10)
    ax_motion.set_title("Motion energy", color=INK, fontsize=12, loc="left")
    ax_motion.set_xlim(0, args.history)
    ax_motion.set_ylim(0, max(detector.enter_threshold * 2.5, 0.05))
    # Outside the axes: the trace eventually fills the full width, so any
    # in-axes legend position ends up sitting on top of the data.
    # Above the axes, right-aligned: the trace eventually fills the full width,
    # so any in-axes position ends up on top of the data. The title is
    # left-aligned on the same row, so the two do not collide.
    ax_motion.legend(loc="lower right", bbox_to_anchor=(1, 1.02), ncol=2,
                     fontsize=9, frameon=False, labelcolor=MUTED)

    ax_bars = fig.add_subplot(grid[2, 0])
    bars = None
    if labels:
        bars = ax_bars.barh(range(len(labels)), [0.0] * len(labels), color=BLUE, height=0.6)
        ax_bars.set_yticks(range(len(labels)), labels, color=MUTED, fontsize=10)
        ax_bars.set_xlim(0, 1)
        ax_bars.set_title("Activity confidence", color=INK, fontsize=12, loc="left")
        ax_bars.invert_yaxis()
    else:
        ax_bars.axis("off")
        ax_bars.text(0.5, 0.5, "no classifier loaded\n(presence only)", ha="center",
                     va="center", color=MUTED, fontsize=12)

    ax_water = fig.add_subplot(grid[2, 1])
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    n_bins = len(source.layout.data_bins)
    waterfall = ax_water.imshow(np.zeros((n_bins, args.window_length)), aspect="auto",
                                origin="lower", cmap=cmap)
    ax_water.set_title("Live CSI — 48 data subcarriers", color=INK, fontsize=12, loc="left")
    ax_water.set_xlabel("frames", color=MUTED, fontsize=9)

    for ax in (ax_motion, ax_bars, ax_water):
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.grid(axis="x" if ax is ax_bars else "both", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    # A window of window_length frames should span about this long. Anything
    # much longer means frames were dropped.
    nominal_span = args.window_length / max(source.rate() or 22.0, 1.0)
    stale = {"count": 0}

    def update(_):
        taken = source.snapshot_span(args.window_length)
        if taken is None:
            return
        amplitude, span = taken

        if span > args.max_span_factor * nominal_span:
            # Straddles a dropout. Hold the last state rather than invent one.
            stale["count"] += 1
            status_text.set_text("SIGNAL GAP")
            status_text.set_color(MUTED)
            meta_text.set_text(f"window spans {span:.1f}s "
                               f"(expected ~{nominal_span:.1f}s) — check the board")
            return

        window = preprocess(amplitude, source.layout)
        present, score, _ = detector.update(window)
        history.append(score)

        status_text.set_text("PERSON PRESENT" if present else "NO ONE DETECTED")
        status_text.set_color(ORANGE if present else MUTED)

        if bars is not None:
            if present:
                if bundle.get("kind") == "sequence":
                    # LSTM: consumes the window itself, not hand-made features.
                    probs = bundle["model"].predict_proba(window[None, ...])[0]
                else:
                    feats = window_features(window, max(source.rate(), 1.0))[None, :]
                    probs = bundle["model"].predict_proba(feats)[0]
                order = {c: i for i, c in enumerate(bundle["model"].classes_)}
                values = [probs[order[l]] if l in order else 0.0 for l in labels]
                best = labels[int(np.argmax(values))]
                activity_text.set_text(f"activity: {best}   ({max(values):.0%} confidence)")
            else:
                values = [0.0] * len(labels)
                activity_text.set_text("activity: —")
            for bar, value in zip(bars, values):
                bar.set_width(value)

        motion_line.set_ydata(list(history))
        seen = [v for v in history if not np.isnan(v)]
        top = max(max(seen) * 1.25, detector.enter_threshold * 1.6)
        ax_motion.set_ylim(0, top)

        waterfall.set_data(window.T)
        waterfall.set_clim(np.percentile(window, 2), np.percentile(window, 98))

        meta_text.set_text(f"{source.rate():.0f} Hz · {source.total} frames · "
                           f"{source.mac}")

    if args.snapshot:
        for _ in range(40):
            update(None)
            time.sleep(0.1)
        fig.savefig(args.snapshot, dpi=140, facecolor=SURFACE)
        source.running = False
        print(f"saved {args.snapshot}")
        return 0

    animation = FuncAnimation(fig, update, interval=150, cache_frame_data=False)
    fig.canvas.mpl_connect("key_press_event",
                           lambda e: plt.close(fig) if e.key == "q" else None)
    try:
        plt.show()
    finally:
        source.running = False
    _ = animation
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
