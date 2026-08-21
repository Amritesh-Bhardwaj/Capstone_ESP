#!/usr/bin/env python
"""Guided interleaved recording for activity-intensity classification.

    csi_env/bin/python har/record_protocol.py --room room1 --session 1 \
        --mac A8:6E:84:93:EE:60

Prompts you through every class, twice each, inside one sitting. Writes each
segment to `har/recordings/<room>/protocol/session-NN/` with the class in the
filename and a `manifest.json` recording what was actually captured.

Why every class in every session
--------------------------------
This is the single most important rule here, and it is not a style preference.
Recording one class per session produces a classifier that identifies the
*session*, which on this hardware scores beautifully and means nothing:

* ``motion_score`` separates two **empty** rooms at AUC **0.986**
* a model handed all fifteen features drops to **49.1%** — chance —
  leave-one-session-out, because it learns the session instead of the person
* the empty-room baseline drifts **+0.578 in 70 minutes**, larger than the
  entire occupancy signal

Cycling every class inside each session balances those effects across classes
rather than confounding them with the label. Two passes per session so a class
is never tied to one moment.

What this can and cannot support
--------------------------------
**Intensity, not headcount.** Measured on this link: two people standing still
read 0.09 where an empty room reads 0.00 and one person sitting reads 0.94.
Motionless bodies do not add up. Counting needs more receivers, not more
recordings — see `DATA.md`.

Running is included but aliases: radial 3 m/s gives ~48.7 Hz Doppler against a
48 Hz Nyquist at the board's ~96 Hz rate. It still looks different from walking;
it is not measuring what its name suggests.

RSSI is logged per segment because it explains **89%** of between-session
variance (0.24 level units per dB). If it correlates with class, the experiment
is broken and `--check` will say so.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import count_features as cf  # noqa: E402
import csi_pipeline as cp  # noqa: E402
import room_profile as rp  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_PORT = "/dev/cu.usbserial-5B530174971"
USABLE_RSSI = -80.0

# (key, label, what to actually do). Ordered low to high intensity, but the
# recorder shuffles within a pass so order never encodes the label.
CLASSES = [
    ("empty",    "EMPTY",         "Leave the room. Shut the door. Nobody inside."),
    ("sitting",  "SITTING STILL", "Sit in your normal spot. Still, but breathe normally."),
    ("standing", "STANDING STILL","Stand still, roughly mid-room. No shifting about."),
    ("walk_slow","WALKING SLOWLY","Amble a loop. Comfortable, unhurried."),
    ("walk_fast","WALKING BRISKLY","Walk a loop like you are late for something."),
    ("running",  "RUNNING",       "Run on the spot, or jog the loop if there is room."),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--room", default="room1")
    p.add_argument("--session", type=int, required=True,
                   help="session number; use a NEW one each sitting")
    p.add_argument("--port", default=DEFAULT_PORT)
    p.add_argument("--baud", type=int, default=460800)
    p.add_argument("--mac", default=None, help="pin the transmitter (recommended)")
    p.add_argument("--seconds", type=float, default=120.0, help="seconds per segment")
    p.add_argument("--passes", type=int, default=2, help="times through the class list")
    p.add_argument("--only", default=None,
                   help="comma-separated class keys, to redo specific segments")
    p.add_argument("--check", action="store_true",
                   help="analyse an existing session instead of recording")
    return p.parse_args(argv)


def session_dir(room: str, n: int) -> pathlib.Path:
    return HERE / "recordings" / room / "protocol" / f"session-{n:02d}"


def reset_board(port_name: str, baud: int) -> None:
    """The board stops emitting after ~20 minutes. Reset before every segment."""
    import serial

    s = serial.Serial(port_name, baud, timeout=2)
    s.setDTR(False)
    s.setRTS(True)
    time.sleep(0.15)
    s.setRTS(False)
    time.sleep(1.6)
    s.close()


def record(port_name: str, baud: int, seconds: float, path: pathlib.Path) -> int:
    import serial

    s = serial.Serial(port_name, baud, timeout=1)
    s.reset_input_buffer()
    rows = 0
    started = time.time()
    last_tick = 0
    with open(path, "w", buffering=1) as fh:
        while time.time() - started < seconds:
            line = s.readline()
            if line:
                text = line.decode("utf-8", "replace")
                if text.startswith("CSI_DATA"):
                    fh.write(text)
                    rows += 1
            left = int(seconds - (time.time() - started))
            if left != last_tick:
                last_tick = left
                print(f"\r    {left:3d}s left   {rows:6d} rows", end="", flush=True)
    s.close()
    print()
    return rows


def summarise(path: pathlib.Path, mac: str | None) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    frames, stats = cp.parse_lines(lines)
    if not frames:
        return {"ok": False, "why": "no parsable frames"}
    counts = collections.Counter(f.mac for f in frames)
    chosen = mac or counts.most_common(1)[0][0]
    kept = [f for f in frames if f.mac == chosen]
    if len(kept) < 200:
        return {"ok": False, "why": f"only {len(kept)} frames from {chosen}",
                "transmitters": counts.most_common(3)}
    rssi = float(np.mean([f.rssi for f in kept]))
    return {
        "ok": rssi >= USABLE_RSSI,
        "why": None if rssi >= USABLE_RSSI else f"link too weak ({rssi:.1f} dBm)",
        "mac": chosen,
        "rssi_mean": round(rssi, 1),
        "rssi_std": round(float(np.std([f.rssi for f in kept])), 2),
        "frames": len(kept),
        "parsed": stats["parsed"],
        "rejected": stats["rejected"],
        "mac_fraction": round(len(kept) / len(frames), 3),
    }


# A segment whose level steps partway through was not one scene. Both of the
# mislabelled captures on 2026-08-20 look like this: the "empty" room that had
# someone sitting in it throughout, and the two-people capture that became
# three when somebody walked in at the 60 s mark. The entry was plainly visible
# in the data (level spiked to 2.27 at exactly t=60 s); nobody looked.
# Threshold set from the measured null, not guessed. Across 29 known
# single-scene captures |AUC-0.5| between head and tail runs median 0.092,
# p95 0.335, max 0.362 -- the room wanders that much on its own. So the bar
# sits just above that.
#
# **This catches gross changes only.** Measured: the 2-people capture that
# decayed to an empty-room reading scores 0.404 and is caught; the capture
# where a third person walked in scores **0.172**, inside the null, and is
# NOT caught. A threshold low enough to catch it would flag ~30% of good
# captures. Someone entering a room that already has two people in it is
# genuinely near-invisible to this sensor (`DATA.md`), so this check cannot
# recover it. Say what was in the room; the tooling cannot infer it.
STEP_AUC = 0.85          # |AUC-0.5| > 0.35 between first and last third
STEP_RATIO = 2.5         # or the median more than doubles across the capture


def label_stability(path, room: str, mac: str | None) -> dict:
    """Did the scene change partway through this segment?

    Compares the first third of the capture against the last third. A single
    steady scene should score near 0.5; a scene that changed will not.
    """
    profile_path = HERE / "rooms" / room / "profile.json"
    if not profile_path.exists():
        return {"checked": False, "why": "no room profile to score against"}
    from scipy import stats as st

    profile = rp.RoomProfile.load(profile_path)
    reference = profile.reference()
    names = cf.window_feature_names()
    idx = [names.index(n) for n in ("effective_rank", "n_eig", "eig_ratio_3")]
    try:
        cap = rp.load_capture(path, mac=mac, reference=reference)
    except Exception as exc:  # noqa: BLE001
        return {"checked": False, "why": f"{type(exc).__name__}: {exc}"}
    feats = cf.features_for_windows(cap["windows"], rp.TARGET_HZ, reference)
    if len(feats) < 12:
        return {"checked": False, "why": f"only {len(feats)} windows"}
    level = np.array([profile.normalise(v)[idx].mean() for v in feats])

    cut = len(level) // 3
    head, tail = level[:cut], level[-cut:]
    ranks = st.rankdata(np.concatenate([tail, head]))
    auc = float((ranks[:len(tail)].sum() - len(tail) * (len(tail) + 1) / 2)
                / (len(tail) * len(head)))
    ratio = float(np.median(tail) / max(abs(np.median(head)), 1e-3))
    # Where the biggest jump sits, for pointing at the moment it changed.
    step = 0.0
    at = None
    for i in range(4, len(level) - 4):
        d = abs(float(np.median(level[i:]) - np.median(level[:i])))
        if d > step:
            step, at = d, i * rp.STRIDE_SECONDS
    steady = abs(auc - 0.5) < (STEP_AUC - 0.5) and ratio < STEP_RATIO
    return {"checked": True, "steady": bool(steady), "auc_head_vs_tail": round(auc, 3),
            "median_ratio": round(ratio, 2), "largest_step": round(step, 2),
            "step_at_s": at, "windows": int(len(level)),
            "head_median": round(float(np.median(head)), 2),
            "tail_median": round(float(np.median(tail)), 2)}


def do_check(room: str, session: int, mac: str | None) -> int:
    """Has this session actually produced usable, unconfounded data?"""
    from scipy import stats as st

    folder = session_dir(room, session)
    manifest = folder / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"error: no manifest at {manifest}")
    entries = json.loads(manifest.read_text())["segments"]
    print(f"session {session:02d}: {len(entries)} segments\n")
    print(f"{'class':12s} {'pass':>4s} {'rows':>7s} {'rssi':>7s} {'mac%':>5s}  status")
    print("-" * 62)
    by_class: dict = collections.defaultdict(list)
    for e in entries:
        s = e.get("summary", {})
        flag = "ok" if s.get("ok") else f"UNUSABLE: {s.get('why')}"
        print(f"{e['class']:12s} {e['pass']:4d} {e.get('rows',0):7d} "
              f"{s.get('rssi_mean','—'):>7} {100*s.get('mac_fraction',0):4.0f}%  {flag}")
        if s.get("ok"):
            by_class[e["class"]].append(s["rssi_mean"])

    # --- did any segment contain more than one scene? ----------------------
    print(f"\n{'class':12s} {'pass':>4s} {'head':>6s} {'tail':>6s} {'AUC':>6s} "
          f"{'step':>6s}  label stability")
    print("-" * 68)
    unstable = 0
    for e in entries:
        st_ = label_stability(folder / e["file"], room, mac)
        if not st_.get("checked"):
            print(f"{e['class']:12s} {e['pass']:4d} {'':6s} {'':6s} {'':6s} {'':6s}  "
                  f"not checked: {st_.get('why')}")
            continue
        e["stability"] = st_
        if st_["steady"]:
            verdict = "steady"
        else:
            unstable += 1
            where = f" at ~{st_['step_at_s']:.0f}s" if st_["step_at_s"] else ""
            verdict = f"SCENE CHANGED{where} -- relabel or split"
        print(f"{e['class']:12s} {e['pass']:4d} {st_['head_median']:6.2f} "
              f"{st_['tail_median']:6.2f} {st_['auc_head_vs_tail']:6.3f} "
              f"{st_['largest_step']:6.2f}  {verdict}")
    if unstable:
        print(f"\n  {unstable} segment(s) look like more than one scene. A capture whose")
        print(f"  level steps partway through was not the thing on its label.")
        (folder / "manifest.json").write_text(json.dumps(
            {**json.loads((folder / 'manifest.json').read_text()), 'segments': entries},
            indent=2))

    print()
    if len(by_class) < 2:
        print("not enough usable classes to check for confounding")
        return 1
    labels, values = [], []
    for i, (k, v) in enumerate(sorted(by_class.items())):
        labels += [i] * len(v)
        values += v
    rho = st.spearmanr(labels, values).statistic if len(set(labels)) > 1 else 0.0
    spread = max(values) - min(values)
    print(f"RSSI by class: spread {spread:.1f} dB across the session")
    print(f"  rank correlation between class and RSSI: {rho:+.3f}")
    if abs(rho) > 0.5 or spread > 4.0:
        print("  WARNING: RSSI tracks the class. At 0.24 level units per dB a 4 dB")
        print("  spread is the whole occupancy signal, so a classifier may be reading")
        print("  link strength. Re-record, and do not move the board between segments.")
    else:
        print("  looks clean -- no obvious link/class confound")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.check:
        return do_check(args.room, args.session, args.mac)

    folder = session_dir(args.room, args.session)
    if folder.exists() and any(folder.glob("*.csi.txt")) and not args.only:
        raise SystemExit(
            f"error: {folder} already has recordings.\n"
            f"Use a new --session number, or --only <class> to redo segments."
        )
    folder.mkdir(parents=True, exist_ok=True)

    wanted = CLASSES
    if args.only:
        keys = {k.strip() for k in args.only.split(",")}
        wanted = [c for c in CLASSES if c[0] in keys]
        if not wanted:
            raise SystemExit(f"error: no classes match {args.only!r}")

    rng = np.random.default_rng(args.session)
    plan = []
    for p in range(1, args.passes + 1):
        order = list(wanted)
        rng.shuffle(order)
        plan += [(p, c) for c in order]

    total_min = len(plan) * args.seconds / 60
    print(f"\nsession {args.session:02d}, room {args.room}")
    print(f"{len(plan)} segments x {args.seconds:.0f}s = about {total_min:.0f} minutes")
    print("order is shuffled within each pass so position never encodes the class.")
    print("\nDo not move the ESP32 or the router at any point. Moving either\n"
          "invalidates the whole session.\n")

    manifest = {"room": args.room, "session": args.session,
                "started": dt.datetime.now().isoformat(timespec="seconds"),
                "seconds_per_segment": args.seconds, "mac": args.mac,
                "segments": []}
    path_manifest = folder / "manifest.json"

    for i, (p, (key, label, what)) in enumerate(plan, 1):
        print(f"[{i}/{len(plan)}]  pass {p}  ---  {label}")
        print(f"    {what}")
        try:
            input("    press ENTER when you are in position (Ctrl-C to stop) ... ")
        except KeyboardInterrupt:
            print("\nstopped early; keeping what has been recorded")
            break
        reset_board(args.port, args.baud)
        stamp = dt.datetime.now().strftime("%H-%M-%S")
        path = folder / f"{stamp}_p{p}_{key}.csi.txt"
        rows = record(args.port, args.baud, args.seconds, path)
        summary = summarise(path, args.mac)
        entry = {"class": key, "label": label, "pass": p, "file": path.name,
                 "time": dt.datetime.now().isoformat(timespec="seconds"),
                 "rows": rows, "summary": summary}
        manifest["segments"].append(entry)
        path_manifest.write_text(json.dumps(manifest, indent=2))
        if not summary.get("ok"):
            print(f"    !! {summary.get('why')} -- redo with "
                  f"--only {key} once the link recovers")
        else:
            print(f"    {rows} rows, {summary['rssi_mean']} dBm, "
                  f"{100*summary['mac_fraction']:.0f}% from {summary['mac']}")
        print()

    print(f"manifest -> {path_manifest}")
    print(f"check it with:\n  csi_env/bin/python har/record_protocol.py "
          f"--room {args.room} --session {args.session} --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
