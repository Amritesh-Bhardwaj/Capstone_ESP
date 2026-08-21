#!/usr/bin/env python
"""Find out what makes the presence baseline drift.

    csi_env/bin/python har/drift_study.py --hours 4 --every 8 --mac A8:6E:84:93:EE:60

Measured 2026-08-20: the empty-room baseline moved **+0.578 in 70 minutes**,
more than the whole occupancy signal, which is what makes any absolute
threshold unusable (`RESULTS.md` §9.12).

Two candidate causes were already ruled out from the session log:

* **Window duration.** The live path takes a fixed 400 *frames*, so a falling
  packet rate stretches the window from 4.2 s to 8.2 s. Plausible, but the
  partial correlation with `effective_rank` is **+0.019** once elapsed time is
  held constant, against **+0.441** for time with duration held constant. Not
  the cause. (It is still wrong, and this script always resamples to a fixed
  4.0 s window so the measurement cannot be contaminated by it.)
* **Packet rate** on its own: −0.202, and it vanishes under the same control.

So the drift tracks *time*. This script takes a short capture every few minutes
with the scene held constant and records everything that could plausibly move
with it, so the survivor can be identified rather than guessed:

* `effective_rank`, `n_eig`, `eig_ratio_3` on **fixed 4.0 s** resampled windows
* RSSI mean and spread — link strength
* per-subcarrier mean amplitude — whether the channel itself is shifting, and
  if so whether uniformly (gain, AGC) or differentially (geometry, multipath)
* reported noise floor, packet rate, jitter, AGC-proxy statistics

The board stops emitting after roughly 20 minutes, so every capture resets it
first. Only summaries are stored, plus the raw rows of the first and last
capture for anything that needs re-deriving later.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import count_features as cf  # noqa: E402
import csi_pipeline as cp  # noqa: E402
import room_profile as rp  # noqa: E402
from features import motion_score, remove_common_mode  # noqa: E402

LOGDIR = pathlib.Path(__file__).resolve().parent.parent.parent / "logs"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--port", default="/dev/cu.usbserial-5B530174971")
    p.add_argument("--baud", type=int, default=460800)
    p.add_argument("--mac", default=None)
    p.add_argument("--hours", type=float, default=4.0)
    p.add_argument("--every", type=float, default=8.0, help="minutes between captures")
    p.add_argument("--seconds", type=float, default=60.0, help="seconds per capture")
    p.add_argument("--scene", default="unrecorded",
                   help="what is in the room -- REQUIRED for the result to mean anything")
    p.add_argument("--room", default="room1")
    return p.parse_args(argv)


def reset(port_name: str, baud: int) -> None:
    import serial

    s = serial.Serial(port_name, baud, timeout=2)
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.15)
    s.setRTS(False)
    time.sleep(1.5)
    s.close()


def capture(port_name: str, baud: int, seconds: float) -> list:
    import serial

    s = serial.Serial(port_name, baud, timeout=1)
    s.reset_input_buffer()
    rows = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        line = s.readline()
        if not line:
            continue
        text = line.decode("utf-8", "replace")
        if text.startswith("CSI_DATA"):
            rows.append(text)
    s.close()
    return rows


def analyse(rows: list, mac: str | None, reference) -> dict:
    frames, stats = cp.parse_lines(rows)
    if len(frames) < 200:
        raise RuntimeError(f"only {len(frames)} frames parsed")
    sel, chosen = cp.select_mac(frames, mac)
    amp = np.stack([f.amplitude for f in sel])
    layout = cp.detect_layout(amp)
    data = cp.remove_null_pilot(amp, layout)
    rssi = np.array([f.rssi for f in sel], dtype=float)

    ticks = np.array([f.local_timestamp for f in sel], dtype=np.int64)
    wraps = np.cumsum(np.concatenate([[0], (np.diff(ticks) < 0).astype(np.int64)]))
    ts = (ticks + wraps * (1 << 32)) / 1e6
    ts -= ts[0]

    filtered, outliers = cp.hampel_filter(data)
    values, _, valid, res = cp.resample_uniform(filtered, ts, target_hz=100.0)

    # Fixed 4.0 s windows, so a changing packet rate cannot alter the estimate.
    length, stride = 400, 200
    windows = cp.sliding_windows(values, length, stride)
    feats = cf.features_for_windows(windows, 100.0, reference)
    names = cf.window_feature_names()
    got = {n: float(np.median(feats[:, names.index(n)]))
           for n in ("effective_rank", "n_eig", "eig_ratio_3", "motion_score")}

    per_sc = data.mean(axis=0)
    # AGC acts on every subcarrier at once; geometry does not. Splitting the
    # two says whether a shift is gain or channel.
    frame_mean = data.mean(axis=1)
    shape = remove_common_mode(data).mean(axis=0)

    return {
        "mac": chosen,
        "frames": len(sel),
        "parsed": stats["parsed"],
        "rejected": stats["rejected"],
        "mac_fraction": len(sel) / max(len(frames), 1),
        "span_s": float(ts[-1]),
        "rate_hz": float(1.0 / np.median(np.diff(ts)[np.diff(ts) > 0])),
        "jitter": float(np.std(np.diff(ts)) / np.median(np.diff(ts))),
        "gap_fraction": res["gap_fraction"],
        "rssi_mean": float(rssi.mean()),
        "rssi_std": float(rssi.std()),
        "outlier_rate": float(outliers.mean()),
        "amp_mean": float(frame_mean.mean()),
        "amp_std_over_time": float(frame_mean.std()),
        "windows": int(len(windows)),
        **got,
        "per_subcarrier_mean": [round(float(v), 3) for v in per_sc],
        "shape_mean": [round(float(v), 5) for v in shape],
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.scene == "unrecorded":
        print("WARNING: --scene not given. Recording it is the difference between\n"
              "         a drift measurement and an unlabelled pile of numbers.\n")
    LOGDIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    out = LOGDIR / f"{stamp}-drift-study.jsonl"

    profile_path = (pathlib.Path(__file__).resolve().parent / "rooms" / args.room
                    / "profile.json")
    reference = None
    if profile_path.exists():
        reference = rp.RoomProfile.load(profile_path).reference()

    deadline = time.time() + args.hours * 3600
    n = 0
    print(f"drift study -> {out}")
    print(f"scene: {args.scene}")
    print(f"{args.seconds:.0f}s capture every {args.every:.0f} min for {args.hours:.1f} h\n")
    with open(out, "w", buffering=1) as fh:
        while time.time() < deadline:
            started = dt.datetime.now()
            record = {"time": started.isoformat(timespec="seconds"),
                      "elapsed_min": round((time.time() - (deadline - args.hours * 3600)) / 60, 2),
                      "scene": args.scene, "index": n}
            try:
                reset(args.port, args.baud)
                rows = capture(args.port, args.baud, args.seconds)
                record.update(analyse(rows, args.mac, reference))
                if n == 0:
                    (LOGDIR / f"{stamp}-drift-first.csi.txt").write_text("".join(rows))
                record["ok"] = True
                print(f"[{n:3d}] {started.strftime('%H:%M:%S')}  "
                      f"eff_rank {record['effective_rank']:5.2f}  "
                      f"rssi {record['rssi_mean']:6.1f}  "
                      f"rate {record['rate_hz']:5.1f}  "
                      f"amp {record['amp_mean']:6.2f}", flush=True)
            except Exception as exc:  # noqa: BLE001
                record["ok"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{n:3d}] {started.strftime('%H:%M:%S')}  FAILED {record['error']}",
                      flush=True)
                traceback.print_exc(file=sys.stderr)
            fh.write(json.dumps(record) + "\n")
            n += 1
            nap = args.every * 60 - (time.time() - started.timestamp())
            if nap > 0 and time.time() + nap < deadline + args.every * 60:
                time.sleep(nap)
    try:
        rows = capture(args.port, args.baud, args.seconds)
        (LOGDIR / f"{stamp}-drift-last.csi.txt").write_text("".join(rows))
    except Exception:  # noqa: BLE001
        pass
    print(f"\n{n} captures written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
