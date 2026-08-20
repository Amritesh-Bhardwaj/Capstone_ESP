"""Live view of the features Gate 1 validated, next to the one it rejected.

    csi_env/bin/python har/live_rank.py --room room1 --mac A8:6E:84:93:EE:60

Walk in and out of the room and watch the traces. Three are plotted:

* **effective_rank** and **n_eig** -- rank statistics of the 48x48 subcarrier
  covariance. Gate 1 scored these AUC 0.889 and 0.903 for empty-vs-occupied
  while scoring only 0.621 and 0.542 between two *different empty sessions*.
  They respond to people and ignore the session.
* **motion_score** -- plotted for contrast, not for use. It scored AUC 0.998
  for empty-vs-occupied and **0.986 between two empty rooms**. It is a near
  perfect session detector and a useless occupancy detector, which is exactly
  what the existing presence path is built on.

The shaded band on each trace is that feature's empty-room range from the room
profile (median +/- 2 MAD). A trace leaving its band is the signal; the point
of the layout is that ``motion_score`` will leave its band for reasons that have
nothing to do with you.

Why rank works where magnitude does not: ``motion_score`` and ``dilated_pem``
measure the *size* of the CSI fluctuation, which moves with link conditions and
noise floor. Rank and eigenvalue ratios measure the *shape* of the covariance,
which is scale-invariant, so a change in received power does not move them.

No resampling is needed here. The rank features come from a covariance across
subcarriers, not from a spectrum, so sample jitter does not affect them --
unlike every band feature, which is why none are plotted.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import count_features as cf  # noqa: E402
import csi_pipeline as cp  # noqa: E402
from features import motion_score  # noqa: E402

DEFAULT_PORT = "/dev/cu.usbserial-5B530174971"
ROOMS = pathlib.Path(__file__).resolve().parent / "rooms"
TRACES = ("effective_rank", "n_eig", "motion_score")
VERDICT = {
    "effective_rank": "validated  (AUC 0.889 vs 0.621 null)",
    "n_eig": "validated  (AUC 0.903 vs 0.542 null)",
    "motion_score": "REJECTED  (AUC 0.998 vs 0.986 null - tracks the session)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--room", default="room1")
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=460800)
    p.add_argument("--mac", default=None, help="pin the transmitter")
    p.add_argument("--window", type=int, default=400, help="frames per estimate (~4 s)")
    p.add_argument("--history", type=int, default=120, help="points kept on screen")
    p.add_argument("--seconds", type=float, default=0.0, help="stop after N s (0 = run until closed)")
    p.add_argument("--no-log", action="store_true", help="do not write a session log")
    return p.parse_args(argv)


LOGDIR = pathlib.Path(__file__).resolve().parent.parent.parent / "logs"


def open_log(args):
    """Every session writes a CSV. A live view that leaves no record cannot be
    checked against ground truth afterwards, which is the whole point of it."""
    if args.no_log:
        return None, None, None
    LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    path = LOGDIR / f"{stamp}-live-rank.csv"
    handle = open(path, "w", newline="", buffering=1)
    writer = csv.writer(handle)
    writer.writerow(["wall_time", "elapsed_s", *TRACES, "frames"])
    return path, handle, writer


def load_band(room: str) -> dict:
    """Empty-room median +/- 2 MAD per feature, from the room profile."""
    path = ROOMS / room / "profile.json"
    if not path.exists():
        print(f"note: no profile at {path}; running without empty-room bands")
        return {}
    data = json.loads(path.read_text())
    names = data["features"]["names"]
    med = data["features"]["empty"]["median"]
    mad = data["features"]["empty"]["mad"]
    return {n: (med[i] - 2 * mad[i], med[i] + 2 * mad[i]) for i, n in enumerate(names)}


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib
    import matplotlib.pyplot as plt
    import serial

    bands = load_band(args.room)
    names = cf.window_feature_names()
    idx = {n: names.index(n) for n in ("effective_rank", "n_eig")}

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"error: cannot open {args.port}: {exc}")
    # The board stops emitting after ~20 minutes; reset before a long view.
    port.setDTR(False)
    port.setRTS(True)
    time.sleep(0.15)
    port.setRTS(False)
    time.sleep(1.5)
    port.reset_input_buffer()

    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.canvas.manager.set_window_title("CSI occupancy — validated features vs rejected one")
    hist = {n: collections.deque([np.nan] * args.history, maxlen=args.history) for n in TRACES}
    lines, spans = {}, {}
    for ax, name in zip(axes, TRACES):
        good = name != "motion_score"
        (lines[name],) = ax.plot(range(args.history), list(hist[name]),
                                 color="#1b7f4b" if good else "#b03030", linewidth=1.8)
        if name in bands:
            lo, hi = bands[name]
            spans[name] = ax.axhspan(lo, hi, color="#888888", alpha=0.18, zorder=0)
            ax.axhline(lo, color="#888888", linewidth=0.7, linestyle=":")
            ax.axhline(hi, color="#888888", linewidth=0.7, linestyle=":")
        ax.set_ylabel(name, fontsize=10)
        ax.set_title(VERDICT[name], loc="left", fontsize=10,
                     color="#1b7f4b" if good else "#b03030")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("update (~0.5 s apart).  shaded = empty-room band from the room profile")
    fig.tight_layout()

    buf: collections.deque = collections.deque(maxlen=args.window)
    layout = None
    started = time.time()
    last = 0.0
    seen = 0
    log_path, handle, writer = open_log(args)
    print("reading... walk in and out of the room. close the window to stop.")
    if log_path:
        print(f"logging to {log_path}")
    try:
        while plt.fignum_exists(fig.number):
            raw = port.readline()
            if raw:
                frame = cp.parse_line(raw.decode("utf-8", "replace").strip())
                if frame is not None and (args.mac is None or frame.mac == args.mac):
                    buf.append(frame.amplitude)
                    seen += 1

            if len(buf) < args.window or time.time() - last < 0.5:
                plt.pause(0.001)
                if args.seconds and time.time() - started > args.seconds:
                    break
                continue
            last = time.time()

            amp = np.stack(buf)
            if layout is None:
                layout = cp.detect_layout(amp)
            win = cp.remove_null_pilot(amp, layout)
            win, _ = cp.hampel_filter(win)

            vec = cf.window_count_features(win, 100.0)
            hist["effective_rank"].append(vec[idx["effective_rank"]])
            hist["n_eig"].append(vec[idx["n_eig"]])
            hist["motion_score"].append(motion_score(win))
            if writer:
                writer.writerow([
                    dt.datetime.now().isoformat(timespec="seconds"),
                    round(time.time() - started, 2),
                    *[round(float(hist[n][-1]), 5) for n in TRACES],
                    seen,
                ])

            for name in TRACES:
                values = list(hist[name])
                lines[name].set_ydata(values)
                finite = [v for v in values if np.isfinite(v)]
                ax = axes[TRACES.index(name)]
                lo_b, hi_b = bands.get(name, (min(finite), max(finite)))
                lo = min(min(finite), lo_b)
                hi = max(max(finite), hi_b)
                pad = max((hi - lo) * 0.15, 1e-6)
                ax.set_ylim(lo - pad, hi + pad)
            fig.canvas.draw_idle()
            plt.pause(0.001)
            if args.seconds and time.time() - started > args.seconds:
                break
    finally:
        port.close()
        if handle:
            handle.close()
            summarise(log_path, hist, started, seen, bands)
    print("stopped")
    return 0


def summarise(path, hist, started, frames, bands) -> None:
    """Write a markdown summary beside the CSV, in the style of logs/."""
    if path is None:
        return
    md = path.with_suffix(".md")
    lines = [
        f"# Live rank session — {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "**Scene: NOT RECORDED.** Annotate this file with what was in the room, or the",
        "reading cannot be used as evidence later.",
        "",
        f"| | |", "|---|---|",
        f"| Duration | {time.time() - started:.0f} s |",
        f"| Frames | {frames} |",
        "",
        "## Features",
        "",
        "| Feature | median | min | max | empty-room band | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for name in TRACES:
        vals = [v for v in hist[name] if v is not None and np.isfinite(v)]
        if not vals:
            continue
        lo, hi = bands.get(name, (None, None))
        band = "—" if lo is None else f"{lo:.4f} … {hi:.4f}"
        good = "validated" if name != "motion_score" else "**rejected** (session detector)"
        lines.append(f"| `{name}` | {np.median(vals):.4f} | {min(vals):.4f} | "
                     f"{max(vals):.4f} | {band} | {good} |")
    lines += ["", "`motion_score` is logged for contrast only. Gate 1 measured it at AUC 0.986",
              "between two *empty* rooms (`RESULTS.md` §9.9); it must not be used to decide",
              "occupancy. Use `effective_rank` and `n_eig`.", ""]
    md.write_text("\n".join(lines))
    print(f"wrote {path}\nwrote {md}")


if __name__ == "__main__":
    raise SystemExit(main())
