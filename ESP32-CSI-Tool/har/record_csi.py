#!/usr/bin/env python
"""Record raw CSI lines from the ESP32 serial port to a file.

Each recording is one labelled activity take, which is what the LSTM stage
will need later:

    python har/record_csi.py --label walking --duration 30
    python har/record_csi.py --label empty   --duration 30

Lines are written through unmodified. Cleaning happens in csi_pipeline, so a
recording stays a faithful record of what the board emitted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import time

import serial

DEFAULT_PORT = "/dev/cu.usbserial-57460201261"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=DEFAULT_PORT, help="serial device")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=30.0, help="seconds")
    parser.add_argument("--label", required=True, help="activity label, e.g. walking")
    parser.add_argument("--outdir", default=str(pathlib.Path(__file__).parent / "recordings"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = outdir / f"{args.label}_{stamp}.csv"

    try:
        port = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as exc:
        print(f"error: cannot open {args.port}: {exc}", file=sys.stderr)
        return 1

    print(f"recording '{args.label}' for {args.duration:.0f}s -> {path}")
    csi_lines = other_lines = 0
    deadline = time.time() + args.duration
    try:
        with path.open("w") as fh:
            while time.time() < deadline:
                line = port.readline().decode("utf-8", "ignore").strip()
                if not line:
                    continue
                if line.startswith("CSI_DATA,"):
                    csi_lines += 1
                    fh.write(line + "\n")
                else:
                    other_lines += 1
                if csi_lines and csi_lines % 100 == 0:
                    remaining = max(0.0, deadline - time.time())
                    print(f"\r  {csi_lines} frames, {remaining:4.1f}s left", end="")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        port.close()

    elapsed = args.duration
    print(f"\nwrote {csi_lines} CSI frames ({csi_lines / elapsed:.1f}/s), "
          f"skipped {other_lines} log lines")
    if csi_lines == 0:
        print("warning: no CSI frames captured -- check the firmware and channel",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
