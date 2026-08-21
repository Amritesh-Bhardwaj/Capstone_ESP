#!/usr/bin/env python
"""Capture a live CSI reading and write a timestamped log to logs/.

    csi_env/bin/python har/log_reading.py --baud 460800 --duration 45 \
        --label "seated, still, LoS between router and ESP32"

Writes `logs/<ISO-timestamp>-live-reading.md` at the repository root, holding
everything needed to interpret the reading later: link health, sample rate,
detected layout, motion-score distribution, a window-length sweep, and the
temporal energy distribution by band.

Reads are non-destructive; nothing on the board is modified.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import (  # noqa: E402
    detect_layout,
    hamming_smooth,
    hampel_filter,
    parse_line,
    remove_null_pilot,
    resample_uniform,
    select_mac,
    wavelet_denoise,
)
from count_features import spectrum_per_subcarrier  # noqa: E402
from features import motion_score  # noqa: E402

DEFAULT_PORT = "/dev/cu.usbserial-5B530174971"
BANDS = ((0.1, 0.5, "breathing 0.1-0.5 Hz"), (0.5, 2.0, "slow 0.5-2 Hz"),
         (2.0, 5.0, "walk 2-5 Hz"), (5.0, 20.0, "fast 5-20 Hz"),
         (20.0, 60.0, "noise 20-60 Hz"))
WINDOWS = (64, 128, 256, 512, 1024, 2048)


def parse_args(argv=None):
    here = pathlib.Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--duration", type=float, default=45.0)
    parser.add_argument("--label", default="", help="what the scene was during capture")
    parser.add_argument("--mac", default=None,
                        help="pin the transmitter; default is the busiest")
    parser.add_argument("--baseline", type=float, default=None,
                        help="calibrated empty-room baseline, to compare against")
    parser.add_argument("--threshold", type=float, default=None,
                        help="calibrated enter threshold, to compare against")
    parser.add_argument("--logdir", default=str(here.parent.parent / "logs"))
    parser.add_argument("--keep-raw", action="store_true",
                        help="also save the raw CSI rows beside the log")
    return parser.parse_args(argv)


def capture(port_name: str, baud: int, seconds: float):
    import serial

    try:
        port = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"error: cannot open {port_name}: {exc}")
    time.sleep(0.3)
    port.reset_input_buffer()

    rows, other, rejected = [], 0, 0
    started = time.time()
    print(f"capturing {seconds:.0f}s from {port_name} @ {baud} ...")
    try:
        while time.time() - started < seconds:
            line = port.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue
            if line.startswith("CSI_DATA,"):
                if parse_line(line) is None:
                    rejected += 1
                else:
                    rows.append(line)
            else:
                other += 1
    finally:
        port.close()
    return rows, other, rejected, time.time() - started


def main(argv=None) -> int:
    args = parse_args(argv)
    stamp = dt.datetime.now()

    rows, other, rejected, elapsed = capture(args.port, args.baud, args.duration)
    if len(rows) < 200:
        raise SystemExit(f"error: only {len(rows)} usable frames — check --baud")

    all_frames = [parse_line(r) for r in rows]
    # MAC selection is mandatory with the passive firmware: it sniffs every
    # transmitter on the channel, and pipelining a mixture produces a series
    # that jumps between unrelated radio links (CONTEXT.md 5.3). This tool
    # used to skip it, which was harmless only under the old single-MAC build.
    frames, chosen_mac = select_mac(all_frames, args.mac)
    mac_fraction = len(frames) / max(len(all_frames), 1)
    tx = collections.Counter(f.mac for f in all_frames)

    amplitude = np.stack([f.amplitude for f in frames])
    layout = detect_layout(amplitude)

    # Device clock where available; wall clock is only an approximation once
    # frames from other transmitters have been dropped.
    if all(f.local_timestamp is not None for f in frames):
        ticks = np.array([f.local_timestamp for f in frames], dtype=np.int64)
        wraps = np.cumsum(np.concatenate([[0], (np.diff(ticks) < 0).astype(np.int64)]))
        stamps = (ticks + wraps * (1 << 32)) / 1e6
        stamps -= stamps[0]
        timing = "device-clock"
    else:
        stamps = np.arange(len(frames), dtype=np.float64) * elapsed / len(frames)
        timing = "assumed"
    deltas = np.diff(stamps)
    deltas = deltas[deltas > 0]
    rate = float(1.0 / np.median(deltas)) if len(deltas) else len(amplitude) / elapsed
    jitter = float(np.std(deltas) / np.median(deltas)) if len(deltas) else 0.0

    raw = remove_null_pilot(amplitude, layout)
    raw, outliers = hampel_filter(raw, window=11, n_sigma=3.0)

    # Band analysis runs on the Hampel-only series, resampled onto a uniform
    # grid. The Hamming FIR and wavelet denoise below are low-pass -- roughly
    # -3 dB by 11 Hz at 100 Hz -- so computing band powers after them would
    # report the filter's shape rather than the room's.
    band_source, _, _, _ = resample_uniform(raw, stamps, target_hz=rate)

    stage = hamming_smooth(raw, window=9)
    stage = wavelet_denoise(stage, "db4")

    sweep = []
    for width in WINDOWS:
        if len(stage) < width + 1:
            continue
        scores = np.array([motion_score(stage[s:s + width])
                           for s in range(0, len(stage) - width + 1, max(1, width // 4))])
        sweep.append((width, width / rate, float(np.median(scores)),
                      float(np.percentile(scores, 95)),
                      float(scores.std() / scores.mean()) if scores.mean() else 0.0))

    # The previous implementation was
    #     signal = remove_common_mode(stage).mean(axis=1)
    # which is identically 1.0 for every frame -- remove_common_mode divides
    # each frame by its own across-subcarrier mean, so that mean is 1 by
    # construction. After mean removal the series was ~1e-12 and the band table
    # was the FFT of floating-point roundoff (total power ~7e-20). Spectra are
    # now computed per subcarrier and averaged.
    spectrum = spectrum_per_subcarrier(band_source)
    freqs = np.fft.rfftfreq(len(band_source), d=1.0 / rate)
    total = spectrum.sum() or 1.0
    energy = [(name, float(spectrum[(freqs >= lo) & (freqs < hi)].sum() / total))
              for lo, hi, name in BANDS]

    base = sweep[0]
    scores64 = np.array([motion_score(stage[s:s + 64])
                         for s in range(0, len(stage) - 64 + 1, 16)])

    rssi = float(np.mean([f.rssi for f in frames]))
    # PHY fields parse_line keeps, plus the ones it drops -- re-read from the
    # raw rows so rate/mcs/bandwidth homogeneity can be checked.
    # Frame keeps only a few of these, so read the full set back off the raw
    # rows. Claiming rate/mcs/ant homogeneity while checking only sig_mode,
    # bandwidth and channel would be an overclaim.
    from csi_pipeline import HEADER_FIELDS
    field_at = {n: i for i, n in enumerate(HEADER_FIELDS)}
    phy = collections.OrderedDict(
        (n, collections.Counter()) for n in
        ("sig_mode", "mcs", "bandwidth", "rate", "sgi", "stbc", "ant", "channel")
    )
    for row in rows:
        cut = row.find("[")
        if cut < 0:
            continue
        head = row[:cut].rstrip(",").split(",")
        if len(head) != len(HEADER_FIELDS) or head[field_at["mac"]] != chosen_mac:
            continue
        for name in phy:
            phy[name][head[field_at[name]]] += 1
    phy = collections.OrderedDict((n, c) for n, c in phy.items() if c)

    logdir = pathlib.Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    path = logdir / f"{stamp.strftime('%Y-%m-%dT%H-%M-%S')}-live-reading.md"

    out = []
    w = out.append
    w(f"# Live CSI reading — {stamp.isoformat(timespec='seconds')}")
    w("")
    if args.label:
        w(f"**Scene:** {args.label}")
        w("")
    w("## Link")
    w("")
    w("| | |")
    w("|---|---|")
    w(f"| Port | `{args.port}` |")
    w(f"| Baud | {args.baud} |")
    w(f"| Duration | {elapsed:.1f} s |")
    w(f"| Frames parsed | {len(rows)} |")
    w(f"| Rows rejected | {rejected} ({rejected / max(len(rows) + rejected, 1):.2%}) |")
    w(f"| Non-CSI log lines | {other} |")
    w(f"| **Sample rate** | **{rate:.1f} Hz** ({timing}, jitter {jitter:.2f}x median) |")
    w(f"| RSSI mean | {rssi:.1f} dBm |")
    w(f"| Layout | `{layout.name}` → {len(layout.data_bins)} data bins |")
    w(f"| Hampel outliers | {outliers.mean():.2%} of samples |")
    w("")
    if rssi < -75:
        w(f"> **WARNING: {rssi:.1f} dBm is below the -75 dBm usable floor.** The room "
          f"AP leaves channel 6 for long stretches and `select_mac` picks the "
          f"busiest transmitter, not the strongest. Everything below is likely "
          f"measured on noise. See `RESULTS.md` §9.5b.")
        w("")
    w("## Transmitters on channel 6")
    w("")
    w(f"Selected `{chosen_mac}` — **{mac_fraction:.0%}** of parsed frames. "
      f"MAC selection is mandatory here; the passive firmware sniffs every "
      f"transmitter and mixing them yields a series that jumps between "
      f"unrelated radio links.")
    w("")
    w("| Transmitter | Frames | RSSI mean |")
    w("|---|---|---|")
    for mac, n in tx.most_common(6):
        r = np.mean([f.rssi for f in all_frames if f.mac == mac])
        mark = " ← selected" if mac == chosen_mac else ""
        w(f"| `{mac}`{mark} | {n} | {r:.1f} dBm |")
    w("")
    w("## PHY parameters of the selected link")
    w("")
    w("| Field | Distinct values | Value |")
    w("|---|---|---|")
    for name, vals in phy.items():
        top = vals.most_common(1)[0][0]
        w(f"| `{name}` | {len(vals)} | {top} |")
    w("")
    if all(len(v) == 1 for v in phy.values()):
        w(f"All {len(phy)} single-valued: every captured frame shares the same "
          "rate, MCS, bandwidth, guard interval, STBC, antenna and mode, so the "
          "frames are directly comparable. "
          "If any of these gains a second value, client data traffic has begun "
          "reaching the CSI stream and the packet-rate confound returns "
          "(`RESULTS.md` §9.5g).")
    else:
        w("**Not single-valued.** Frames were captured at more than one PHY "
          "setting, so they are not directly comparable and the packet-rate "
          "confound is live. See `RESULTS.md` §9.5g.")
    w("")
    w("## motion_score, 64-frame windows")
    w("")
    w("```")
    w(f"min {scores64.min():.4f}  p25 {np.percentile(scores64, 25):.4f}  "
      f"median {np.median(scores64):.4f}  p75 {np.percentile(scores64, 75):.4f}  "
      f"max {scores64.max():.4f}   (n={len(scores64)})")
    w("```")
    if args.baseline is not None and args.threshold is not None:
        above = (scores64 > args.threshold).mean()
        w("")
        w(f"Against calibration baseline {args.baseline:.4f}, "
          f"threshold {args.threshold:.4f}:")
        w("")
        w(f"- windows above threshold: **{(scores64 > args.threshold).sum()}/"
          f"{len(scores64)} = {above:.1%}**")
        w(f"- median score is {np.median(scores64) / args.baseline:.2f}x the baseline")
        if np.median(scores64) < args.baseline:
            w("- **the live median is BELOW the empty-room baseline** — the "
              "calibration was noisier than the scene it is judging, so it "
              "cannot trigger. Recalibrate in the same steady state.")
    w("")
    w("## Window-length sweep")
    w("")
    w("| Window | Seconds | Median | p95 | CV |")
    w("|---|---|---|---|---|")
    for width, secs, med, p95, cv in sweep:
        w(f"| {width} | {secs:.2f} | {med:.4f} | {p95:.4f} | {cv:.3f} |")
    w("")
    w(f"Longer windows changed the median by "
      f"{(sweep[-1][2] / base[2] - 1) * 100:+.0f}% while changing its "
      f"variability by {(sweep[-1][4] / base[4] - 1) * 100:+.0f}% — "
      "stability, not sensitivity.")
    w("")
    w("## Temporal energy by band")
    w("")
    w("Computed on the Hampel-only series resampled to a uniform grid, **before** "
      "the Hamming FIR and wavelet denoise. Those are low-pass (~-3 dB by 11 Hz "
      "at 100 Hz), so band powers measured after them describe the filter, not "
      "the room. Spectra are per subcarrier then averaged.")
    w("")
    w("| Band | Share of power |")
    w("|---|---|")
    for name, share in energy:
        w(f"| {name} | {share:.1%} |")
    w("")
    w("## Frequency resolution")
    w("")
    w("| Window | Seconds | Lowest resolvable |")
    w("|---|---|---|")
    for width, secs, *_ in sweep:
        low = rate / width
        note = "too short for breathing" if low > 0.2 else "breathing resolvable"
        w(f"| {width} | {secs:.2f} | {low:.3f} Hz — {note} |")
    w("")

    path.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"\nwrote {path}")

    if args.keep_raw:
        raw = path.with_suffix(".csi.txt")
        raw.write_text("\n".join(rows) + "\n")
        print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
