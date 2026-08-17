#!/usr/bin/env python
"""Continuously log motion_score with wall-clock timestamps.

    csi_env/bin/python har/monitor.py --duration 300

Appends one CSV row per window to `logs/<timestamp>-monitor.csv`:

    iso_time, elapsed_s, motion_score, rate_hz, frames

Purpose: collect a trace that can be aligned afterwards against narrated
ground truth ("moving now", "stopped at 01:52"). With labels in hand the
detection threshold can be *fitted* rather than guessed from a calibration
that may itself be contaminated.

Emits no threshold decision of its own -- it only records the signal.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
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
from features import motion_score  # noqa: E402

DEFAULT_PORT = "/dev/cu.usbserial-57460201261"


def parse_args(argv=None):
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--duration", type=float, default=300.0)
    parser.add_argument("--window", type=int, default=128,
                        help="frames per window (~1.1 s at 114 Hz)")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="seconds between reported windows")
    parser.add_argument("--logdir", default=str(here.parent.parent / "logs"))
    parser.add_argument("--stall-seconds", type=float, default=3.0,
                        help="reopen the port after this long with no frames; "
                             "the board reliably hangs after ~3 min and a "
                             "fresh open resets it")
    return parser.parse_args(argv)


def _reader(port, source, stop, counter):
    """Drain the serial port continuously. Decoupled from the DSP so that
    filtering a window can never stall the read and overflow the OS buffer --
    which is what silently killed the first version at 115 Hz."""
    while not stop.is_set():
        try:
            line = port.readline().decode("utf-8", "ignore").strip()
        except Exception:  # noqa: BLE001
            continue
        if not line:
            continue
        frame = parse_line(line)
        if frame is not None:
            source.append((frame.amplitude, time.time()))
            counter[0] += 1   # monotonic; len(deque) saturates at maxlen


def main(argv=None) -> int:
    import collections
    import threading

    import serial

    args = parse_args(argv)
    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"error: cannot open {args.port}: {exc}")
    time.sleep(0.3)
    port.reset_input_buffer()

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now()
    path = logdir / f"{stamp.strftime('%Y-%m-%dT%H-%M-%S')}-monitor.csv"

    source: collections.deque = collections.deque(maxlen=args.window * 4)
    stop = threading.Event()
    counter = [0]
    holder = {"port": port}

    def start_reader():
        t = threading.Thread(target=_reader,
                             args=(holder["port"], source, stop, counter),
                             daemon=True)
        t.start()
        return t

    thread = start_reader()

    def reconnect():
        """Close and reopen. Opening asserts DTR/RTS, which resets the board."""
        nonlocal thread
        stop.set()
        time.sleep(0.3)
        try:
            holder["port"].close()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
        holder["port"] = serial.Serial(args.port, args.baud, timeout=1)
        time.sleep(0.4)
        holder["port"].reset_input_buffer()
        stop.clear()
        thread = start_reader()

    layout = None
    started = time.time()
    rows = 0
    last_seen = 0
    stalled_for = 0.0
    reconnects = 0

    with path.open("w", buffering=1) as fh:
        fh.write("iso_time,elapsed_s,motion_score,rate_hz,frames\n")
        print(f"monitoring -> {path}")
        print("narrate what you do; every window is timestamped\n")
        print(f"{'time':>12s} {'elapsed':>8s} {'score':>8s} {'Hz':>6s}")

        try:
            while time.time() - started < args.duration:
                time.sleep(args.interval)
                recent = list(source)[-args.window:]
                now = time.time()
                elapsed = now - started
                iso = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]

                if len(recent) < args.window:
                    continue
                if counter[0] == last_seen:
                    stalled_for += args.interval
                    fh.write(f"{iso},{elapsed:.2f},,,0\n")
                    if stalled_for >= args.stall_seconds:
                        print(f"{iso:>12s} {elapsed:8.1f}  RECONNECT after "
                              f"{stalled_for:.0f}s silent", flush=True)
                        try:
                            reconnect()
                            reconnects += 1
                        except Exception as exc:  # noqa: BLE001
                            print(f"    reconnect failed: {exc}", flush=True)
                        stalled_for = 0.0
                    continue
                stalled_for = 0.0
                last_seen = counter[0]

                block = np.stack([a for a, _ in recent])
                if layout is None:
                    try:
                        layout = detect_layout(block)
                    except Exception as exc:  # noqa: BLE001
                        print(f"  waiting for layout: {exc}"); continue
                    print(f"layout {layout.name} -> {len(layout.data_bins)} bins\n")

                window = remove_null_pilot(block, layout)
                window, _ = hampel_filter(window, window=11, n_sigma=3.0)
                window = hamming_smooth(window, window=9)
                window = wavelet_denoise(window, "db4")
                score = motion_score(window)

                span = recent[-1][1] - recent[0][1]
                rate = (len(recent) - 1) / span if span > 0 else 0.0
                fh.write(f"{iso},{elapsed:.2f},{score:.6f},{rate:.1f},{len(recent)}\n")
                print(f"{iso:>12s} {elapsed:8.1f} {score:8.4f} {rate:6.1f}", flush=True)
                rows += 1
        finally:
            stop.set()
            time.sleep(0.2)
            try:
                holder["port"].close()
            except Exception:  # noqa: BLE001
                pass

    print(f"\nwrote {rows} rows ({reconnects} reconnects) -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
