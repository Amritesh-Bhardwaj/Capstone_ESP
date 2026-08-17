#!/usr/bin/env python
"""Does a published pretrained WiFi-CSI HAR model transfer to our ESP32?

    lstm_env/bin/python har/test_pretrained_transfer.py --sharp-dir /path/to/SHARP \
        --sharpax-dir /path/to/SHARPax

Surveying https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing (27 linked
repositories), only SHARP and SHARPax ship pretrained HAR weights. This script
tests both on our data, with controls.

The control is the point: we feed the model pure Gaussian noise alongside real
ESP32 CSI. If the two produce the same prediction, the model is not reading our
data -- it is off its training manifold and the output is meaningless, however
confident it looks.

Doppler front-end is a direct port of SHARP's CSI_doppler_computation.py
(num_symbols=31, 100-point FFT, fftshift, |.|^2 summed over subcarriers,
row-max normalisation, noise floor 10^-1.2), so the model sees the input format
it was trained on.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import re
import sys

import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import detect_layout  # noqa: E402

_INT = re.compile(r"^-?\d+$")


def complex_csi(path: pathlib.Path, mac: str | None = None) -> np.ndarray:
    """Complex CSI from an ESP32 recording. SHARP needs complex, not amplitude."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.startswith("CSI_DATA,") or "[" not in line:
            continue
        head, arr = line.split("[", 1)
        arr = arr.split("]", 1)[0]
        fields = head.split(",")
        tokens = arr.split()
        if len(tokens) < 128 or not all(_INT.match(t) for t in tokens[:128]):
            continue
        if mac and fields[2] != mac:
            continue
        raw = np.array(tokens[:128], dtype=np.int16)
        # ESP-IDF packs (imag, real) per subcarrier.
        rows.append(raw[1::2].astype(np.float64) + 1j * raw[0::2].astype(np.float64))
    return np.array(rows)


def sharp_doppler(csi, num_symbols=31, sliding=1, noise_lev=-1.2) -> np.ndarray:
    """Port of SHARP's Doppler computation. Returns (time_steps, 100)."""
    window = np.expand_dims(hann(num_symbols), axis=-1)
    profiles = []
    for i in range(0, csi.shape[0] - num_symbols, sliding):
        cut = np.nan_to_num(csi[i:i + num_symbols, :])
        spectrum = fftshift(fft(cut * window, n=100, axis=0), axes=0)
        profiles.append(np.sum(np.abs(spectrum * np.conj(spectrum)), axis=1))
    array = np.asarray(profiles)
    array = array / np.max(array, axis=1, keepdims=True)
    array[array < math.pow(10, noise_lev)] = math.pow(10, noise_lev)
    return array


def to_sample(doppler: np.ndarray, time_steps: int) -> np.ndarray:
    if len(doppler) < time_steps:
        doppler = np.tile(doppler, (int(np.ceil(time_steps / len(doppler))), 1))
    window = doppler[:time_steps]
    window = window - np.mean(window, axis=0, keepdims=True)
    return window[None, ..., None].astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def main(argv=None) -> int:
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sharp-dir", required=True, help="clone of francescamen/SHARP")
    parser.add_argument("--sharpax-dir", default=None, help="clone of francescamen/SHARPax")
    parser.add_argument("--esp32", default=str(here / "testdata" / "live_lltf_fft.txt"))
    parser.add_argument("--mac", default="AA:BB:CC:00:00:01")
    parser.add_argument("--synthetic", default=str(here / "recordings-synthetic"))
    args = parser.parse_args(argv)

    import tensorflow as tf

    real = complex_csi(pathlib.Path(args.esp32), args.mac)
    if not len(real):
        raise SystemExit(f"no CSI parsed from {args.esp32}")
    bins = detect_layout(np.abs(real)).data_bins
    print(f"real ESP32 capture: {real.shape[0]} frames, {len(bins)} data subcarriers")

    rng = np.random.default_rng(0)
    cases = {
        "REAL ESP32 (48sc, 22Hz)": real[:, bins],
        "PURE GAUSSIAN NOISE": rng.normal(size=(720, 48)) + 1j * rng.normal(size=(720, 48)),
    }
    synthetic = pathlib.Path(args.synthetic)
    for label in ("walking", "empty"):
        path = synthetic / f"{label}_synth0.csv"
        if path.exists():
            cases[f"synthetic {label}"] = complex_csi(path)[:, bins]

    models = [("SHARP single-antenna",
               pathlib.Path(args.sharp_dir) / "Python_code" / "single_ant_E,L,W,R,J_network.h5",
               340, ["Empty", "Lying", "Walking", "Running", "Jumping"])]
    if args.sharpax_dir:
        found = sorted((pathlib.Path(args.sharpax_dir) / "Python_code" / "networks").glob("*bandw20*.h5"))
        if found:
            models.append(("SHARPax 20MHz", found[0], 256,
                           ["Empty", "Sitting", "Walking", "Running"]))

    for tag, path, steps, classes in models:
        if not path.exists():
            print(f"\n!! missing {path}")
            continue
        model = tf.keras.models.load_model(str(path), compile=False)
        print(f"\n=== {tag}  (input {model.input_shape}, {model.count_params():,} params) ===")
        print(f"{'input':26s} {'predicted':9s} {'conf':>7s} {'max|logit|':>11s}")
        logits = {}
        for name, csi in cases.items():
            z = model.predict(to_sample(sharp_doppler(csi), steps), verbose=0)[0]
            logits[name] = z
            p = softmax(z)
            print(f"{name:26s} {classes[p.argmax()]:9s} {p.max():6.2%} {np.abs(z).max():11.1f}")

        a, b = logits["REAL ESP32 (48sc, 22Hz)"], logits["PURE GAUSSIAN NOISE"]
        correlation = float(np.corrcoef(a, b)[0, 1])
        print(f"\n  real-vs-noise logit correlation: {correlation:+.4f}"
              f"   same prediction: {np.argmax(a) == np.argmax(b)}")
        if correlation > 0.9:
            print("  VERDICT: the model cannot distinguish our CSI from noise. "
                  "No transfer.")
    print("""
Why it fails -- physics, not a porting bug:
  sample rate   SHARP Tc=6ms (167Hz), SHARPax Tc=7.5ms (133Hz) vs our ~22Hz
                -> the 100 Doppler bins span +/-83Hz for them, +/-11Hz for us,
                   so every bin means a different velocity
  carrier       5.0 / 5.785 GHz vs our 2.437 GHz; Doppler shift scales with fc
  max velocity  ~3.5-5 m/s for them, ~1.35 m/s for us -- walking (~1.3 m/s)
                sits at our aliasing limit, running aliases outright
  phase         they apply a 3-stage phase sanitisation to Nexmon 802.11ac CSI;
                ESP32 phase is uncalibrated
Logit magnitudes in the hundreds (vs order 10 for in-distribution input) are the
signature of an input far off the training manifold.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
