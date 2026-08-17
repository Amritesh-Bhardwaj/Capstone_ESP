#!/usr/bin/env python
"""Benchmark our pipeline on the ESP-Fi HAR dataset.

    csi_env/bin/python har/train_espfi.py --root /path/to/espfi_data

ESP-Fi HAR (AutoSmartGroup) is CSI collected from commodity ESP32 modules --
the same hardware family as ours, amplitude only, 52 subcarriers
(48 data + 4 pilot), 7 activities, 4 indoor environments. Each .mat holds
``CSIamp`` of shape (950, 52).

That makes it the one public dataset our pipeline can consume directly, so it
gives us a real accuracy number without recording anything.

Two differences from our own captures, both handled here:
  * their 52 subcarriers keep the pilots; our recordings drop them to 48. The
    filter chain is applied to whatever width arrives.
  * their sample rate is not documented anywhere in the repo or README. Band
    features are therefore computed against a NOMINAL rate (--fs), so those
    particular features are internally consistent but not physically
    calibrated. Everything else is rate-free.

The official split is session-disjoint (test = session 3-1, train = 3-2..3-8),
which is already leakage-free at the session level. We additionally report
per-file voting, which is the setting the dataset authors classify in.
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys
import time

import numpy as np
import scipy.io as sio

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import hamming_smooth, hampel_filter, wavelet_denoise  # noqa: E402
from features import features_for_windows  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="directory containing train_amp/ and test_amp/")
    parser.add_argument("--fs", type=float, default=100.0,
                        help="NOMINAL sample rate; the dataset does not document one")
    parser.add_argument("--window-length", type=int, default=256)
    parser.add_argument("--window-stride", type=int, default=128)
    parser.add_argument("--model", choices=("rf", "logreg", "lstm"), default="rf",
                        help="lstm requires lstm_env (PyTorch)")
    parser.add_argument("--classes", default=None,
                        help="comma-separated subset, e.g. walk,turn,fall,run")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only N files per class per split (quick runs)")
    parser.add_argument("--no-filter", action="store_true",
                        help="skip the Hampel/Hamming/wavelet chain, to measure its effect")
    parser.add_argument("--epochs", type=int, default=40, help="lstm only")
    return parser.parse_args(argv)


def preprocess(amplitude: np.ndarray, apply_filter: bool) -> np.ndarray:
    if not apply_filter:
        return amplitude
    stage, _ = hampel_filter(amplitude, window=11, n_sigma=3.0)
    stage = hamming_smooth(stage, window=9)
    return wavelet_denoise(stage, "db4")


def sliding(signal: np.ndarray, length: int, stride: int) -> np.ndarray:
    if len(signal) < length:
        return np.empty((0, length, signal.shape[1]))
    starts = range(0, len(signal) - length + 1, stride)
    return np.stack([signal[s:s + length] for s in starts])


def load_split(root: pathlib.Path, split: str, args, raw: bool = False):
    """Returns (X, labels, file_ids). X is features, or raw windows if raw."""
    directory = root / split
    if not directory.is_dir():
        raise SystemExit(f"missing {directory}")

    wanted = None
    if args.classes:
        wanted = {c.strip() for c in args.classes.split(",") if c.strip()}

    X, y, files = [], [], []
    started = time.time()
    for class_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        if wanted and class_dir.name not in wanted:
            continue
        paths = sorted(class_dir.glob("*.mat"))
        if args.limit:
            paths = paths[:args.limit]
        for path in paths:
            amplitude = np.asarray(sio.loadmat(path)["CSIamp"], dtype=np.float64)
            if amplitude.shape[0] < amplitude.shape[1]:   # stored (52, 950)
                amplitude = amplitude.T
            windows = sliding(preprocess(amplitude, not args.no_filter),
                              args.window_length, args.window_stride)
            if not len(windows):
                continue
            X.append(windows.astype(np.float32) if raw
                     else features_for_windows(windows, args.fs))
            y += [class_dir.name] * len(windows)
            files += [path.stem] * len(windows)
        print(f"  {split}/{class_dir.name:10s} {len(paths):3d} files", flush=True)
    if not X:
        raise SystemExit(f"no data loaded for {split} (check --classes)")
    print(f"  ({time.time() - started:.0f}s)")
    stack = np.concatenate(X) if raw else np.vstack(X)
    return stack, np.asarray(y), np.asarray(files)


def report(name, y_true, y_pred, labels):
    from sklearn.metrics import accuracy_score, confusion_matrix

    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{name}: accuracy {accuracy:.1%}  (n={len(y_true)})")
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    width = max(len(l) for l in labels) + 1
    print("    " + " ".join(f"{l[:5]:>5s}" for l in labels) + "   <- predicted")
    for row_label, row in zip(labels, matrix):
        total = row.sum()
        recall = row[labels.index(row_label)] / total if total else 0
        print(f"  {row_label:<{width}s}" + " ".join(f"{v:5d}" for v in row)
              + f"   {recall:5.1%}")
    return accuracy


def main(argv=None) -> int:
    args = parse_args(argv)
    root = pathlib.Path(args.root)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    raw = args.model == "lstm"
    print(f"loading ESP-Fi HAR from {root}"
          f"   (model={args.model}, filter={'off' if args.no_filter else 'on'}"
          + ("" if raw else f", nominal fs={args.fs:g} Hz") + ")")
    Xtr, ytr, _ = load_split(root, "train_amp", args, raw=raw)
    Xte, yte, files_te = load_split(root, "test_amp", args, raw=raw)
    labels = sorted(set(ytr))
    print(f"\ntrain {Xtr.shape}   test {Xte.shape}   {len(labels)} classes: {labels}")

    if args.model == "lstm":
        from lstm_model import pick_device, train_lstm  # needs lstm_env

        device = pick_device()
        print(f"training LSTM on {device} for {args.epochs} epochs ...")
        model = train_lstm(Xtr, ytr, labels, epochs=args.epochs, device=device)
    elif args.model == "rf":
        model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                       random_state=0, n_jobs=-1).fit(Xtr, ytr)
    else:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=3000)).fit(Xtr, ytr)
    predictions = np.asarray(model.predict(Xte)).astype(str)
    yte = yte.astype(str)

    window_accuracy = report("PER-WINDOW, session-disjoint test set (3-1)",
                             yte, predictions, labels)

    # Per-file majority vote: the setting the dataset authors classify in.
    votes, truth = [], []
    for file_id in sorted(set(files_te)):
        mask = files_te == file_id
        votes.append(collections.Counter(predictions[mask]).most_common(1)[0][0])
        truth.append(yte[mask][0])
    file_accuracy = report("PER-FILE majority vote", np.array(truth),
                           np.array(votes), labels)

    chance = max(collections.Counter(yte).values()) / len(yte)
    print(f"\n  majority-class baseline: {chance:.1%}")
    print(f"  per-window {window_accuracy:.1%} ({window_accuracy - chance:+.1%}) | "
          f"per-file {file_accuracy:.1%} ({file_accuracy - chance:+.1%})")
    print("\n  Test files come from a session never seen in training, so this is "
          "a\n  leakage-free number and can be quoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
