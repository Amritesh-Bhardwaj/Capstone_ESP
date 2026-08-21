#!/usr/bin/env python
"""Train a presence classifier, and test it the only way that means anything.

    csi_env/bin/python har/train_presence.py --room room1

Sessions are listed in ``har/rooms/<room>/sessions.json`` as
``{"path": ..., "label": "present"|"empty"}``.

Leave-one-session-out, always
-----------------------------
A random split over sliding windows is meaningless here twice over: windows
overlap, and — far worse — a model can score near-perfectly by recognising the
*session* rather than the person. That is not hypothetical. ``motion_score``
separates empty from occupied at AUC 0.998 and separates two **empty** sessions
at 0.986, and the empty-room baseline drifted +0.578 in seventy minutes, which
is larger than the occupancy signal itself (`RESULTS.md` §9.9, §9.12).

So every fold holds out one entire recording session. If a model only works
when it has seen that session in training, it has learned the room's mood on
that day and will fail the moment it is deployed.

Balanced accuracy is reported against the majority-class rate, because these
sessions are not balanced and raw accuracy would flatter a constant predictor.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import count_features as cf  # noqa: E402
import room_profile as rp  # noqa: E402

ROOMS = pathlib.Path(__file__).resolve().parent / "rooms"
LABELS = {"present": 1, "empty": 0}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--room", default="room1")
    p.add_argument("--mac", default=None)
    p.add_argument("--sessions", default=None, help="override the sessions.json path")
    p.add_argument("--out", default=None, help="save the fitted model here (joblib)")
    p.add_argument("--normalise", action="store_true",
                   help="map features through the room profile before training")
    p.add_argument("--features", default="validated",
                   choices=["validated", "all", "rank"],
                   help="which feature set to train on")
    return p.parse_args(argv)


def load_sessions(path: pathlib.Path) -> list:
    if not path.exists():
        raise SystemExit(
            f"error: no session manifest at {path}\n"
            f"Create it as a JSON list of "
            f'{{"path": "...", "label": "present"|"empty"}}'
        )
    entries = json.loads(path.read_text())
    if len(entries) < 3:
        raise SystemExit("error: need at least three sessions to leave one out")
    return entries


def feature_columns(names: list, which: str, profile) -> list:
    if which == "all":
        return list(range(len(names)))
    if which == "rank":
        keep = ("effective_rank", "n_eig", "eig_ratio_2", "eig_ratio_3")
    else:
        gate = profile.gate1.get("features", {}) if profile else {}
        keep = tuple(n for n, i in gate.items() if i.get("validated")) or (
            "effective_rank", "n_eig", "eig_ratio_3")
    return [names.index(n) for n in keep if n in names]


def main(argv=None) -> int:
    args = parse_args(argv)
    from sklearn.ensemble import RandomForestClassifier

    room_dir = ROOMS / args.room
    profile_path = room_dir / "profile.json"
    profile = rp.RoomProfile.load(profile_path) if profile_path.exists() else None
    reference = profile.reference() if profile else None

    manifest = pathlib.Path(args.sessions) if args.sessions else room_dir / "sessions.json"
    entries = load_sessions(manifest)
    names = cf.window_feature_names()
    cols = feature_columns(names, args.features, profile)
    print(f"features ({args.features}): {[names[c] for c in cols]}")
    if args.normalise and not profile:
        raise SystemExit("error: --normalise needs a room profile")

    sessions = []
    for entry in entries:
        path = pathlib.Path(entry["path"])
        if not path.is_absolute():
            path = (manifest.parent / path).resolve()
            if not path.exists():
                path = (pathlib.Path.cwd() / entry["path"]).resolve()
        try:
            cap = rp.load_capture(path, mac=args.mac, reference=reference)
        except SystemExit as exc:
            # A session where the pinned transmitter was absent is not a
            # session; skipping beats training on a -88 dBm fallback link.
            print(f"  {path.name[:44]:44s} SKIPPED  {str(exc)[:60]}")
            continue
        feats = cf.features_for_windows(cap["windows"], rp.TARGET_HZ, reference)
        if len(feats) < 10:
            print(f"  {path.name[:44]:44s} SKIPPED  only {len(feats)} windows")
            continue
        if args.normalise:
            feats = np.stack([profile.normalise(v) for v in feats])
        sessions.append({
            "name": path.name,
            "label": LABELS[entry["label"]],
            "X": feats[:, cols],
            "rssi": cap["rssi_mean"],
        })
        print(f"  {path.name[:44]:44s} {entry['label']:8s} "
              f"{len(feats):4d} windows  {cap['rssi_mean']:6.1f} dBm")

    y_all = np.concatenate([[s["label"]] * len(s["X"]) for s in sessions])
    majority = max(np.mean(y_all == 1), np.mean(y_all == 0))
    print(f"\n{len(sessions)} sessions, {len(y_all)} windows; "
          f"majority class = {100 * majority:.1f}% raw, 50.0% balanced")

    if len({s["label"] for s in sessions}) < 2:
        raise SystemExit("error: sessions must include both labels")

    print(f"\n{'held-out session':46s} {'label':>8s} {'balanced':>9s} {'acc':>7s}")
    print("-" * 76)
    scores = []
    for i, held in enumerate(sessions):
        train = [s for j, s in enumerate(sessions) if j != i]
        if len({s["label"] for s in train}) < 2:
            print(f"{held['name'][:44]:46s} {'—':>8s} {'skipped':>9s} "
                  f"{'(one class left in training)':>7s}")
            continue
        X = np.concatenate([s["X"] for s in train])
        y = np.concatenate([[s["label"]] * len(s["X"]) for s in train])
        model = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                       class_weight="balanced", random_state=0)
        model.fit(X, y)
        pred = model.predict(held["X"])
        acc = float(np.mean(pred == held["label"]))
        # One class per held-out session, so per-fold balanced accuracy is just
        # the recall of that class; the mean over folds is the balanced figure.
        scores.append((held["label"], acc))
        print(f"{held['name'][:44]:46s} "
              f"{'present' if held['label'] else 'empty':>8s} "
              f"{'':>9s} {100 * acc:6.1f}%")

    if scores:
        pres = [a for l, a in scores if l == 1]
        emp = [a for l, a in scores if l == 0]
        bal = 100 * (np.mean(pres) + np.mean(emp)) / 2 if pres and emp else float("nan")
        print(f"\n  recall on present sessions: {100 * np.mean(pres):5.1f}%"
              if pres else "")
        print(f"  recall on empty sessions:   {100 * np.mean(emp):5.1f}%" if emp else "")
        print(f"  BALANCED ACCURACY (leave-one-session-out): {bal:.1f}%")
        print(f"  chance = 50.0%")

    # A model trained on everything, for inspection only -- never for a claim.
    X = np.concatenate([s["X"] for s in sessions])
    y = y_all
    full = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                  class_weight="balanced", random_state=0).fit(X, y)
    order = np.argsort(-full.feature_importances_)
    print("\n  feature importance (model fitted on everything, not a result):")
    for k in order[:6]:
        print(f"    {names[cols[k]]:24s} {full.feature_importances_[k]:.3f}")

    if args.out:
        import joblib
        bundle = {
            "model": full,
            "columns": cols,
            "names": [names[c] for c in cols],
            "normalised": args.normalise,
            "room": args.room,
            # Everything needed to interpret or reproduce this file.
            "trained": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "sessions": [{"name": s["name"], "label": s["label"],
                          "windows": len(s["X"]), "rssi": round(s["rssi"], 1)}
                         for s in sessions],
            "leave_one_session_out": {
                "balanced_accuracy": round(float(bal), 1) if scores else None,
                "recall_present": round(100 * float(np.mean(pres)), 1) if pres else None,
                "recall_empty": round(100 * float(np.mean(emp)), 1) if emp else None,
                "chance": 50.0,
            },
            "feature_importance": {names[cols[k]]: round(float(v), 4)
                                   for k, v in enumerate(full.feature_importances_)},
        }
        joblib.dump(bundle, args.out, compress=3)
        print(f"\n  wrote {args.out}  (fitted on all sessions -- "
              f"quote only the leave-one-session-out number above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
