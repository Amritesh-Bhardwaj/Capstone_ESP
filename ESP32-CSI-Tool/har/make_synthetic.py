#!/usr/bin/env python
"""Generate synthetic CSI recordings, for testing the stack without hardware.

    python har/make_synthetic.py --outdir har/recordings-synthetic

Writes takes in the exact ESP32-CSI-Tool serial format, so they flow through
the real parser, layout detection and filter chain. Classes differ the way real
activities differ: by how strongly and how fast the subcarrier amplitudes are
modulated.

This is a self-test and a rehearsal fixture, NOT training data. Never quote an
accuracy measured on it -- the classes are separable by construction.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np

# 'fft' layout: guard bins must read exactly zero so detect_layout finds them.
GUARD = {0} | set(range(27, 38))
HEADER = ("CSI_DATA,PASSIVE,A8:6E:84:93:EE:60,-48,11,0,0,0,1,1,0,0,0,0,-92,0,6,"
          "0,{ts},0,90,0,0,0.0,128")

# label -> (noise level, [(modulation Hz, depth), ...])
PROFILES = {
    "empty":   (0.02, []),
    "sitting": (0.04, [(0.30, 0.10)]),                    # breathing-ish
    "walking": (0.09, [(1.30, 0.45), (2.60, 0.20)]),      # whole-body motion
}


def make_take(label: str, n_frames: int, fs: float, seed: int) -> list:
    rng = np.random.default_rng(seed)
    # A different static channel shape per take, as a different room position
    # would produce.
    base = 8 + 6 * np.abs(np.sin(np.linspace(0, 3.1, 64) + rng.uniform(0, 2)))
    noise, components = PROFILES[label]
    times = np.arange(n_frames) / fs
    phases = {freq: rng.uniform(0, 2 * np.pi, 64) for freq, _ in components}

    lines = []
    for i in range(n_frames):
        modulation = np.ones(64)
        for freq, depth in components:
            modulation *= 1 + depth * np.sin(2 * np.pi * freq * times[i] + phases[freq])
        amplitude = base * modulation * (1 + rng.normal(0, noise, 64))

        values = []
        for bin_index in range(64):
            if bin_index in GUARD:
                values += [0, 0]
                continue
            phase = rng.uniform(0, 2 * np.pi)
            a = amplitude[bin_index]
            values += [int(np.clip(round(a * np.sin(phase)), -127, 127)),
                       int(np.clip(round(a * np.cos(phase)), -127, 127))]
        lines.append(HEADER.format(ts=int(i * 1e6 / fs))
                     + ",[" + " ".join(map(str, values)) + " ]")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outdir", default=str(pathlib.Path(__file__).parent / "recordings-synthetic"))
    parser.add_argument("--takes", type=int, default=4, help="takes per class")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--rate", type=float, default=24.0, help="frames per second")
    args = parser.parse_args(argv)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    n_frames = int(args.seconds * args.rate)

    for label in PROFILES:
        for take in range(args.takes):
            path = outdir / f"{label}_synth{take}.csv"
            lines = make_take(label, n_frames, args.rate, seed=abs(hash((label, take))) % 99991)
            path.write_text("\n".join(lines) + "\n")
    total = len(PROFILES) * args.takes
    print(f"wrote {total} takes ({n_frames} frames each, {args.seconds:g}s "
          f"@ {args.rate:g} Hz) to {outdir}")
    print("these are separable by construction -- for testing only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
