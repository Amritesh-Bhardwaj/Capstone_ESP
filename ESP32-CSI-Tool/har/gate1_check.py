"""Gate 1: are the counting features alive at all?

    csi_env/bin/python har/gate1_check.py --room room1

Asks one question: **does a single walking person move the counting features
away from the empty room, by more than the empty room moves away from itself?**

What passing does and does not mean
-----------------------------------
Passing means the features respond to occupancy and it is worth recruiting
people to record 0/1/2/3. It does **not** mean counting works. Separating 0
from 1 is a presence test; counting lives or dies on 1 vs 2 vs 3, which no
amount of empty-vs-one-person data can answer. Gate 1 is a screen: failing it
kills counting, passing it only licenses the next stage.

The null control
----------------
Every feature is also scored on the empty capture's first half against its own
second half. Nobody entered the room in between, so any separation there is
drift -- thermal, AGC, changing background traffic -- and a feature that
"separates" empty from empty is measuring time, not people. A feature only
counts as real if it beats its own null by a clear margin.

No p-values are reported. Sliding windows overlap by 50% and CSI is strongly
autocorrelated, so the samples are not independent and any p-value would be
optimistic by a wide and unknown factor. Effect size and AUC are reported
instead, and the null control is what makes them interpretable.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import count_features as cf
import room_profile as rp

ROOMS = pathlib.Path(__file__).resolve().parent / "rooms"

# A feature must clear this AUC against the empty room, and beat its own
# empty-vs-empty null by this margin, to count as responding to occupancy.
AUC_FLOOR = 0.80
NULL_MARGIN = 0.15


def auc(reference: np.ndarray, test: np.ndarray) -> float:
    """P(test > reference), ties at 0.5. 0.5 means indistinguishable."""
    if not len(reference) or not len(test):
        return 0.5
    ranks = stats.rankdata(np.concatenate([test, reference]))
    n_test = len(test)
    u = ranks[:n_test].sum() - n_test * (n_test + 1) / 2
    return float(u / (n_test * len(reference)))


def null_floor(feats: np.ndarray, block: int) -> np.ndarray:
    """Worst |AUC-0.5| between equal-length blocks of the *same* empty capture.

    A single first-half/second-half split samples the null once and gets lucky
    or unlucky. Every pair of 60 s blocks samples it many times, and the worst
    case is what a real signal actually has to beat. Measured on Room 1 this
    matters a lot: effective_rank drifts to 0.269 against itself, so a
    one-person effect smaller than that is indistinguishable from the room
    sitting still and changing its mind.
    """
    blocks = [feats[i:i + block] for i in range(0, len(feats), block)]
    blocks = [b for b in blocks if len(b) >= 5]
    if len(blocks) < 2:
        half = len(feats) // 2
        blocks = [feats[:half], feats[half:]]
    out = np.zeros(feats.shape[1])
    for i in range(feats.shape[1]):
        pairs = [abs(auc(blocks[a][:, i], blocks[b][:, i]) - 0.5)
                 for a in range(len(blocks)) for b in range(a + 1, len(blocks))]
        out[i] = max(pairs) if pairs else 0.0
    return out


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                     / (len(a) + len(b) - 2))
    if pooled < 1e-12:
        return 0.0
    return float((b.mean() - a.mean()) / pooled)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--room", required=True)
    parser.add_argument("--mac", default=None)
    parser.add_argument("--empty", default=None, help="override the empty capture path")
    parser.add_argument("--one-person", default=None, help="override the one-person path")
    parser.add_argument("--null-empty", default=None,
                        help="a SECOND empty-room capture from a different session. "
                             "The within-capture null understates drift badly; this is "
                             "the null that decides whether a feature is real.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    room_dir = ROOMS / args.room
    empty_path = pathlib.Path(args.empty) if args.empty else room_dir / "empty.csi.txt"
    one_path = (pathlib.Path(args.one_person) if args.one_person
                else room_dir / "one_person.csi.txt")

    if not empty_path.exists():
        raise SystemExit(f"error: no empty capture at {empty_path}")

    empty = rp.load_capture(empty_path, mac=args.mac)
    if empty["rssi_mean"] < -80.0:
        print(f"WARNING: empty capture is on {empty['mac']} at "
              f"{empty['rssi_mean']:.1f} dBm, below the -80 dBm usable floor.\n"
              f"         Any result below is measured largely on noise.\n")
    names = cf.window_feature_names()
    # Both captures must be scored against the same empty-room yardstick, the
    # same two-pass build_profile does. dilated_pem is meaningless otherwise.
    reference = cf.reference_mad(empty["windows"])
    e_feats = cf.features_for_windows(empty["windows"], rp.TARGET_HZ, reference)
    print(f"empty      {empty_path.name}: {len(e_feats)} windows, "
          f"{empty['stats']['rate_hz']:.1f} Hz source, "
          f"{empty['stats']['gap_fraction']:.2%} gap-interpolated")

    # --- null control: empty vs itself -------------------------------------
    block = int(60 / rp.STRIDE_SECONDS)
    if len(e_feats) < 10:
        raise SystemExit("error: empty capture too short to form a null control")
    null_dev = null_floor(e_feats, block)

    if not one_path.exists():
        print(f"\nno one-person capture at {one_path}")
        print("drift check only (empty first half vs second half):\n")
        order = np.argsort(-null_dev)
        for i in order:
            flag = "  <- DRIFTING" if null_dev[i] > NULL_MARGIN else ""
            print(f"  {names[i]:24s} null |AUC-0.5| {null_dev[i]:.3f}{flag}")
        print("\n  A feature above 0.15 here must clear that plus 0.15 with a")
        print("  person present before it means anything.")
        print("\nrecord the one-person phase, then rerun:")
        print(f"  csi_env/bin/python har/calibrate_room.py --room {args.room} --phase one-person")
        return 0

    one = rp.load_capture(one_path, mac=args.mac, reference=reference)
    if one["mac"] != empty["mac"]:
        raise SystemExit(
            f"error: the two phases locked onto different transmitters\n"
            f"  empty      {empty['mac']}\n"
            f"  one person {one['mac']}\n"
            f"Any difference would be between two radio links, not between an\n"
            f"empty and an occupied room. Recapture with --mac pinned to one."
        )
    o_feats = one["features"]
    print(f"one person {one_path.name}: {len(o_feats)} windows, "
          f"{one['stats']['rate_hz']:.1f} Hz source")

    real_auc = np.array([auc(e_feats[:, i], o_feats[:, i]) for i in range(len(names))])
    d = np.array([cohens_d(e_feats[:, i], o_feats[:, i]) for i in range(len(names))])
    margin = np.abs(real_auc - 0.5) - null_dev
    passed = (np.abs(real_auc - 0.5) + 0.5 >= AUC_FLOOR) & (margin >= NULL_MARGIN)

    print(f"\n{'feature':24s} {'AUC':>6s} {'null':>6s} {'margin':>7s} {'d':>7s}")
    print("-" * 56)
    for i in np.argsort(-np.abs(real_auc - 0.5)):
        mark = "PASS" if passed[i] else "    "
        print(f"  {names[i]:22s} {real_auc[i]:6.3f} {null_dev[i]:6.3f} "
              f"{margin[i]:+7.3f} {d[i]:+7.2f}  {mark}")

    # --- the specific hypothesis Gate 1 was built to test -------------------
    er = names.index("effective_rank")
    ms = names.index("motion_score")
    print("\nverdict")
    print(f"  effective_rank  AUC {real_auc[er]:.3f}  (null {null_dev[er]:.3f})"
          f"   {'RESPONDS' if passed[er] else 'does not clear the bar'}")
    print(f"  motion_score    AUC {real_auc[ms]:.3f}  (null {null_dev[ms]:.3f})"
          f"   {'RESPONDS' if passed[ms] else 'does not clear the bar'}")

    # The 25-48 Hz band is above plausible human Doppler. If it moves as much
    # as the real Doppler band, the "Doppler" signal is a resampling artefact.
    dop = names.index("band_doppler")
    ctl = names.index("band_control_hi")
    print(f"\n  band 11-25 Hz (walk Doppler)  AUC {real_auc[dop]:.3f}")
    print(f"  band 25-48 Hz (control)       AUC {real_auc[ctl]:.3f}")
    if abs(real_auc[ctl] - 0.5) >= abs(real_auc[dop] - 0.5) - 0.02:
        print("  ^ the control band moves as much as the Doppler band.")
        print("    Treat all >11 Hz content as resampling artefact until this")
        print("    separates. Do not feed those bands to a classifier.")
    else:
        print("  ^ Doppler band outruns the control band: >11 Hz content looks real.")

    # Persist the measured AUCs into the room profile. The live dashboard reads
    # them from there, so rerunning the gate updates what the page claims.
    # The cross-session null is the one that matters. Within one capture the
    # floors came out at 0.08-0.18 and motion_score looked validated; against a
    # second empty session it scores 0.986 and is exposed as a session detector.
    cross = {}
    if args.null_empty:
        second = rp.load_capture(args.null_empty, mac=args.mac, reference=reference)
        if second["mac"] != empty["mac"]:
            raise SystemExit(
                f"error: --null-empty is on {second['mac']} but the empty phase is on "
                f"{empty['mac']}. Two different links cannot form a null control."
            )
        sf = second["features"]
        for i, name in enumerate(names):
            cross[name] = abs(auc(e_feats[:, i], sf[:, i]) - 0.5)
        print(f"\ncross-session null from {pathlib.Path(args.null_empty).name} "
              f"({len(sf)} windows)")
        print(f"{'feature':24s} {'signal':>8s} {'within':>8s} {'CROSS':>8s} {'verdict':>10s}")
        print("-" * 62)
        for i in np.argsort(-(np.abs(real_auc - 0.5) - np.array([cross[n] for n in names]))):
            real = abs(real_auc[i] - 0.5)
            ok = real - cross[names[i]] >= NULL_MARGIN and real + 0.5 >= AUC_FLOOR
            print(f"  {names[i]:22s} {real_auc[i]:8.3f} {null_dev[i]:8.3f} "
                  f"{cross[names[i]] + 0.5:8.3f} {'REAL' if ok else 'session':>10s}")

    profile_path = room_dir / "profile.json"
    if profile_path.exists():
        profile = rp.RoomProfile.load(profile_path)
        profile.gate1 = {
            "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "empty": empty_path.name,
            "occupied": one_path.name,
            "null_source": "60 s blocks within the empty capture",
            "null_cross_session_source": (pathlib.Path(args.null_empty).name
                                          if args.null_empty else None),
            "auc_floor": AUC_FLOOR,
            "null_margin": NULL_MARGIN,
            "features": {
                names[i]: {
                    "auc_vs_occupied": round(float(real_auc[i]), 4),
                    "null_within_session": round(float(null_dev[i]), 4),
                    "null_cross_session": (round(float(cross[names[i]]) + 0.5, 4)
                                           if names[i] in cross else None),
                    "margin": round(float(margin[i]), 4),
                    "cohens_d": round(float(d[i]), 3),
                    "passed": bool(passed[i]),
                    # Only a feature that beats a *cross-session* null is real.
                    "validated": (bool(
                        abs(real_auc[i] - 0.5) - cross[names[i]] >= NULL_MARGIN
                        and abs(real_auc[i] - 0.5) + 0.5 >= AUC_FLOOR)
                        if names[i] in cross else None),
                }
                for i in range(len(names))
            },
        }
        profile.save(profile_path)
        print(f"\n  wrote Gate 1 results into {profile_path}")

    n_pass = int(passed.sum())
    print(f"\n  {n_pass}/{len(names)} features respond to one occupant.")
    if passed[er] or n_pass >= 3:
        print("\n  GATE 1 PASSED -- the counting features respond to occupancy.")
        print("  This licenses recording 0/1/2/3; it does not show counting works.")
        print("  Next: 3 helpers, 4 counts x 3 takes x 2 min, then Gate 2.")
    else:
        print("\n  GATE 1 FAILED -- nothing responds beyond its own drift.")
        print("  Do not recruit people yet. Check placement (move the ESP32")
        print("  farther from the router, across the room), confirm the walker")
        print("  crossed the line of sight, and rerun.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
