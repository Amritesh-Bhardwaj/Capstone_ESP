#!/usr/bin/env python
"""Score a session log against ground truth you recorded by hand.

    csi_env/bin/python har/score_session.py \
        logs/2026-08-20T05-58-02-rank-session.csv \
        --truth "05:58:10=present,06:10:00=empty,06:11:30=present"

``--truth`` is a comma-separated list of ``HH:MM:SS=state`` transitions, which
is how a person actually records ground truth: "I left at 06:10, came back at
06:11:30". Everything before the first transition is unlabelled and ignored.

Reports detection latency per transition, false positive and false negative
rates, state changes per minute, and balanced accuracy.

**Balanced accuracy, not raw accuracy.** A session where you were present for
28 of 30 minutes is 93% "accurate" for a detector that always says present, so
raw accuracy hides total failure. Every number here is reported per class.

A ``--settle`` guard drops the seconds either side of each transition from the
error counts, because no detector can be right during its own response time;
latency is reported separately instead of being smeared into the error rate.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import presence_events as pe  # noqa: E402

STATES = {"present": 1, "empty": 0, "occupied": 1, "in": 1, "out": 0}


def parse_truth(spec: str) -> list:
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(f"error: bad --truth entry {chunk!r}, expected HH:MM:SS=state")
        stamp, state = chunk.split("=", 1)
        state = state.strip().lower()
        if state not in STATES:
            raise SystemExit(f"error: unknown state {state!r}; use present or empty")
        try:
            parts = [int(p) for p in stamp.strip().split(":")]
        except ValueError:
            raise SystemExit(f"error: bad time {stamp!r}") from None
        while len(parts) < 3:
            parts.append(0)
        out.append((dt.time(*parts), STATES[state]))
    if not out:
        raise SystemExit("error: --truth is empty")
    return sorted(out)


def label(times: list, truth: list) -> np.ndarray:
    """-1 before the first transition, else the state in force at that moment."""
    out = np.full(len(times), -1)
    for i, ts in enumerate(times):
        state = -1
        for when, value in truth:
            if ts >= when:
                state = value
        out[i] = state
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("log", help="a *-rank-session.csv written by the dashboard")
    p.add_argument("--truth", required=True, help="HH:MM:SS=state,HH:MM:SS=state,...")
    p.add_argument("--column", default="present",
                   help="column holding the detector's decision (default: present)")
    p.add_argument("--settle", type=float, default=10.0,
                   help="seconds either side of a transition excluded from error rates")
    p.add_argument("--replay-events", action="store_true",
                   help="ignore the logged decision and re-score with the event detector")
    p.add_argument("--timeout", type=float, default=pe.TIMEOUT_S)
    p.add_argument("--enter-z", type=float, default=pe.ENTER_Z)
    p.add_argument("--short", type=float, default=pe.SHORT_S)
    p.add_argument("--long", type=float, default=pe.LONG_S)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    path = pathlib.Path(args.log)
    if not path.exists():
        raise SystemExit(f"error: no such log: {path}")
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"error: {path} is empty")

    stamps = [dt.datetime.fromisoformat(r["wall_time"]) for r in rows]
    times = [s.time() for s in stamps]
    truth = parse_truth(args.truth)
    gt = label(times, truth)

    span = (stamps[-1] - stamps[0]).total_seconds()
    dt_s = span / max(len(stamps) - 1, 1)

    if args.replay_events:
        levels = np.array([float(r["level_raw"]) for r in rows])
        pred, zs, events = pe.run(levels, dt_s, short_s=args.short, long_s=args.long,
                                  enter_z=args.enter_z, timeout_s=args.timeout)
        pred = pred.astype(int)
        source = (f"event detector (short {args.short}s, long {args.long}s, "
                  f"z>={args.enter_z}, timeout {args.timeout}s)")
    else:
        pred = np.array([1 if r[args.column] == "1" else 0 for r in rows])
        zs, events = None, []
        source = f"logged column {args.column!r}"

    print(f"{path.name}")
    print(f"  {len(rows)} updates over {span/60:.1f} min ({dt_s:.2f}s apart)")
    print(f"  scoring: {source}")

    # --- latency per transition --------------------------------------------
    print(f"\n{'transition':>26s} {'detected':>12s} {'latency':>10s}")
    print("-" * 52)
    latencies = []
    for when, value in truth:
        idx = [i for i, ts in enumerate(times) if ts >= when]
        if not idx:
            continue
        hit = next((i for i in idx if pred[i] == value), None)
        name = f"{when.strftime('%H:%M:%S')} -> {'present' if value else 'empty'}"
        if hit is None:
            print(f"{name:>26s} {'never':>12s} {'—':>10s}")
            continue
        lag = (stamps[hit] - stamps[idx[0]]).total_seconds()
        latencies.append(lag)
        print(f"{name:>26s} {times[hit].strftime('%H:%M:%S'):>12s} {lag:9.1f}s")

    # --- error rates, excluding the settle window --------------------------
    scored = gt >= 0
    for when, _ in truth:
        for i, ts in enumerate(times):
            delta = abs((dt.datetime.combine(stamps[0].date(), ts)
                         - dt.datetime.combine(stamps[0].date(), when)).total_seconds())
            if delta <= args.settle:
                scored[i] = False

    n_present = int((scored & (gt == 1)).sum())
    n_empty = int((scored & (gt == 0)).sum())
    if not n_present or not n_empty:
        print("\nwarning: ground truth has only one class; error rates are meaningless")
        return 0

    tp = 100.0 * np.mean(pred[scored & (gt == 1)] == 1)
    tn = 100.0 * np.mean(pred[scored & (gt == 0)] == 0)
    flips = int(np.sum(pred[1:] != pred[:-1]))

    print(f"\n{'':22s} {'labelled':>9s} {'correct':>9s}")
    print("-" * 44)
    print(f"{'present (occupied)':22s} {n_present:9d} {tp:8.1f}%")
    print(f"{'empty':22s} {n_empty:9d} {tn:8.1f}%")
    print(f"\n  balanced accuracy   {(tp + tn) / 2:6.1f}%   "
          f"(50% = chance; always-present would score 50%)")
    print(f"  false positives     {100 - tn:6.1f}%   (said present in an empty room)")
    print(f"  false negatives     {100 - tp:6.1f}%   (missed an occupant)")
    print(f"  state changes       {flips:6d}     = {60 * flips / max(span, 1):.2f}/min")
    if latencies:
        print(f"  median latency      {np.median(latencies):6.1f}s")
    if events:
        print(f"  motion events       {len(events):6d}")
    if zs is not None:
        print(f"  event score |z|     median {np.median(np.abs(zs)):.2f}  "
              f"p90 {np.percentile(np.abs(zs), 90):.2f}  max {np.abs(zs).max():.2f}")
    print(f"\n  {args.settle:.0f}s either side of each transition excluded from the rates; "
          f"latency is reported instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
