#!/usr/bin/env python
"""Train the activity classifier and presence calibration from recordings.

    python har/train_activity.py --recordings har/recordings --out har/model.joblib

Recordings are named ``<label>_<timestamp>.csv`` by record_csi.py, so the
label comes from the filename. Any take labelled ``empty`` is also used to
calibrate the presence detector.

Accuracy is reported two ways on purpose. Sliding windows overlap, so
splitting them at random puts near-identical windows in both train and test
and inflates the score badly. The honest number is the grouped split, which
holds out whole recordings.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import PipelineConfig, parse_lines, run_pipeline  # noqa: E402
from features import (  # noqa: E402
    PresenceDetector,
    feature_names,
    features_for_windows,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", default=str(pathlib.Path(__file__).parent / "recordings"))
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent / "model.joblib"))
    parser.add_argument("--model", choices=("rf", "logreg"), default="rf")
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-windows", type=int, default=5,
                        help="skip takes yielding fewer windows than this")
    return parser.parse_args(argv)


def load_takes(directory: pathlib.Path, config: PipelineConfig, min_windows: int):
    """Yield one (label, take_name, windows, sample_rate) per recording."""
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise SystemExit(f"no recordings found in {directory}")

    takes = []
    for path in files:
        label = path.stem.split("_")[0]
        frames, report = parse_lines(path.read_text().splitlines())
        if not frames:
            print(f"  skip {path.name}: no usable frames")
            continue
        try:
            result = run_pipeline(frames, config)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {path.name}: {exc}")
            continue
        if len(result.windows) < min_windows:
            print(f"  skip {path.name}: only {len(result.windows)} windows")
            continue
        print(f"  {path.name:38s} label={label:10s} "
              f"{len(result.windows):3d} windows @ {result.stats['rate_hz']:.1f} Hz")
        takes.append((label, path.stem, result.windows, result.stats["rate_hz"]))
    return takes


def report_split(name, y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{name}: accuracy {accuracy:.1%}  (n={len(y_true)})")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(l) for l in labels) + 1
    print("    " + " ".join(f"{l[:6]:>6s}" for l in labels) + "   <- predicted")
    for row_label, row in zip(labels, matrix):
        total = row.sum()
        correct = row[labels.index(row_label)] / total if total else 0
        print(f"  {row_label:<{width}s}" + " ".join(f"{v:6d}" for v in row)
              + f"   {correct:5.1%} recall")
    return accuracy


def main(argv=None) -> int:
    args = parse_args(argv)
    directory = pathlib.Path(args.recordings)
    config = PipelineConfig(
        normalise=False,  # features need real amplitude scale, not z-scores
        window_length=args.window_length,
        window_stride=args.window_stride,
    )

    print(f"loading recordings from {directory}")
    takes = load_takes(directory, config, args.min_windows)
    if not takes:
        raise SystemExit("no usable recordings")

    counts = collections.Counter(label for label, _, _, _ in takes)
    print(f"\ntakes per label: {dict(counts)}")
    if len(counts) < 2:
        raise SystemExit("need at least two labels to train a classifier")
    single = [l for l, c in counts.items() if c < 2]
    if single:
        print(f"warning: {single} have only one take, so the grouped split "
              "cannot hold them out honestly. Record a second take of each.")

    X, y, groups, rates = [], [], [], []
    for label, take, windows, rate in takes:
        feats = features_for_windows(windows, rate)
        X.append(feats)
        y += [label] * len(feats)
        groups += [take] * len(feats)
        rates.append(rate)
    X = np.vstack(X)
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups, dtype=object)
    labels = sorted({str(v) for v in y})
    print(f"feature matrix {X.shape} over {len(set(groups))} takes, labels {labels}")

    def build():
        if args.model == "rf":
            return RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1
            )
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=1.0, multi_class="auto"),
        )

    # The leaky number, shown for contrast only.
    leaky_pred = np.empty_like(y)
    for train_idx, test_idx in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        model = build().fit(X[train_idx], y[train_idx])
        leaky_pred[test_idx] = model.predict(X[test_idx])
    report_split("RANDOM window split (LEAKY -- do not quote this)",
                 y, leaky_pred, labels)

    # The honest number: hold out entire recordings.
    honest_pred, honest_true, skipped = [], [], []
    splitter = LeaveOneGroupOut()
    for train_idx, test_idx in splitter.split(X, y, groups):
        held_out = str(groups[test_idx][0])
        # A fold is only meaningful if the remaining takes still cover at least
        # two classes; otherwise the classifier has nothing to choose between.
        if len({str(v) for v in y[train_idx]}) < 2:
            skipped.append(held_out)
            continue
        model = build().fit(X[train_idx], y[train_idx])
        honest_pred.append(model.predict(X[test_idx]))
        honest_true.append(y[test_idx])

    if honest_true:
        honest_true = np.concatenate(honest_true)
        honest_pred = np.concatenate(honest_pred)
        tested = {str(v) for v in honest_true}
        untested = [l for l in labels if l not in tested]
        honest = report_split("LEAVE-ONE-TAKE-OUT (quote this one)",
                              honest_true, honest_pred, labels)
        if untested:
            print(f"\n  *** NOT A VALID HEADLINE NUMBER: {untested} were never "
                  "held out, so\n      that accuracy is computed over a subset "
                  "of your classes. Record a\n      second take of each before "
                  "quoting anything.")
        else:
            baseline = DummyClassifier(strategy="most_frequent").fit(X, y)
            chance = accuracy_score(y, baseline.predict(X))
            print(f"\n  majority-class baseline: {chance:.1%}"
                  f"   ->  the model is {honest - chance:+.1%} over chance")
        if skipped:
            print(f"  folds skipped (would have left one class in training): {skipped}")
    else:
        print("\ncannot run a grouped split: every fold left fewer than two "
              "classes in training. Record at least two takes per label.")

    final = build().fit(X, y)

    presence = None
    empty_windows = [w for label, _, w, _ in takes if label == "empty"]
    if empty_windows:
        presence = PresenceDetector.calibrate(np.concatenate(empty_windows))
        print(f"\npresence calibration from {len(empty_windows)} empty take(s): "
              f"{presence.describe()}")
    else:
        print("\nwarning: no 'empty' recording, so presence detection is "
              "uncalibrated. Record one with --label empty.")

    joblib.dump(
        {
            "kind": "features",   # live_demo feeds window_features, not raw windows
            "model": final,
            "labels": labels,
            "feature_names": feature_names(),
            "sample_rate": float(np.mean(rates)),
            "window_length": args.window_length,
            "presence": presence,
            "config": config,
        },
        args.out,
    )
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
