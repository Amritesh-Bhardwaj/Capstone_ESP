#!/usr/bin/env python
"""Train the paper's LSTM activity classifier on recorded takes.

    lstm_env/bin/python har/train_lstm.py --recordings har/recordings \
        --out har/model-lstm.joblib

Same recordings, same labelling convention and the same leakage-free
validation as train_activity.py -- only the classifier differs. Run it from
``lstm_env`` (Python 3.12); PyTorch has no wheels for the 3.14 ``csi_env``.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import PipelineConfig, parse_lines, run_pipeline  # noqa: E402
from features import PresenceDetector  # noqa: E402
from lstm_model import LSTMClassifier, pick_device, train_lstm  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", default=str(pathlib.Path(__file__).parent / "recordings"))
    parser.add_argument("--out", default=str(pathlib.Path(__file__).parent / "model-lstm.joblib"))
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None, help="cpu / mps / cuda")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def load_takes(directory: pathlib.Path, config: PipelineConfig, min_windows: int):
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise SystemExit(f"no recordings found in {directory}")
    takes = []
    for path in files:
        label = path.stem.split("_")[0]
        frames, _ = parse_lines(path.read_text().splitlines())
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
        recall = row[labels.index(row_label)] / total if total else 0
        print(f"  {row_label:<{width}s}" + " ".join(f"{v:6d}" for v in row)
              + f"   {recall:5.1%} recall")
    return accuracy


def main(argv=None) -> int:
    args = parse_args(argv)
    device = args.device or pick_device()
    print(f"torch {torch.__version__} on {device}")

    directory = pathlib.Path(args.recordings)
    config = PipelineConfig(
        normalise=False,  # lstm_model does its own gain-invariant normalisation
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

    windows, y, groups, rates = [], [], [], []
    for label, take, take_windows, rate in takes:
        windows.append(take_windows)
        y += [label] * len(take_windows)
        groups += [take] * len(take_windows)
        rates.append(rate)
    X = np.concatenate(windows)
    y = np.asarray(y, dtype=object)
    groups = np.asarray(groups, dtype=object)
    labels = sorted({str(v) for v in y})
    print(f"sequences {X.shape} over {len(set(groups))} takes, labels {labels}")

    def fit(train_idx):
        return train_lstm(
            X[train_idx], y[train_idx], labels,
            epochs=args.epochs, hidden=args.hidden, dropout=args.dropout,
            lr=args.lr, device=device, verbose=args.verbose,
        )

    # Leaky, for contrast only.
    leaky_pred = np.empty(len(y), dtype=object)
    for train_idx, test_idx in StratifiedKFold(3, shuffle=True, random_state=0).split(X, y):
        leaky_pred[test_idx] = fit(train_idx).predict(X[test_idx])
    report_split("RANDOM window split (LEAKY -- do not quote this)",
                 y.astype(str), leaky_pred.astype(str), labels)

    # Honest: hold out whole recordings.
    honest_true, honest_pred, skipped = [], [], []
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held_out = str(groups[test_idx][0])
        if len({str(v) for v in y[train_idx]}) < 2:
            skipped.append(held_out)
            continue
        honest_pred.append(fit(train_idx).predict(X[test_idx]))
        honest_true.append(y[test_idx])

    if honest_true:
        honest_true = np.concatenate(honest_true).astype(str)
        honest_pred = np.concatenate(honest_pred).astype(str)
        tested = set(honest_true)
        untested = [l for l in labels if l not in tested]
        honest = report_split("LEAVE-ONE-TAKE-OUT (quote this one)",
                              honest_true, honest_pred, labels)
        if untested:
            print(f"\n  *** NOT A VALID HEADLINE NUMBER: {untested} were never "
                  "held out.\n      Record a second take of each before quoting "
                  "anything.")
        else:
            majority = max(collections.Counter(y.astype(str)).values()) / len(y)
            print(f"\n  majority-class baseline: {majority:.1%}"
                  f"   ->  the model is {honest - majority:+.1%} over chance")
        if skipped:
            print(f"  folds skipped (would have left one class in training): {skipped}")
    else:
        print("\ncannot run a grouped split: record at least two takes per label")

    print("\ntraining final model on all takes ...")
    final = fit(np.arange(len(X)))

    presence = None
    empty_windows = [w for label, _, w, _ in takes if label == "empty"]
    if empty_windows:
        presence = PresenceDetector.calibrate(np.concatenate(empty_windows))
        print(f"presence calibration from {len(empty_windows)} empty take(s): "
              f"{presence.describe()}")
    else:
        print("warning: no 'empty' recording -- presence detection uncalibrated")

    joblib.dump(
        {
            "kind": "sequence",          # live_demo feeds raw windows, not features
            "model": final,
            "labels": labels,
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
