#!/usr/bin/env python
"""Run the CSI preprocessing pipeline on a recording or a live serial stream.

    python har/run_pipeline.py --file har/recordings/walking_*.csv --plot
    python har/run_pipeline.py --serial --duration 20 --plot
    python har/run_pipeline.py --file take.csv --save-npz take.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from csi_pipeline import (  # noqa: E402
    LayoutError,
    PipelineConfig,
    parse_line,
    parse_lines,
    run_pipeline,
)

DEFAULT_PORT = "/dev/cu.usbserial-57460201261"

# Colours from the data-viz reference palette (validated: all checks pass).
SERIES_RAW = "#2a78d6"        # categorical slot 1
SERIES_PROCESSED = "#eb6834"  # categorical slot 2
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                   "#184f95", "#0d366b"]
# blue <-> red poles with a neutral gray midpoint, for signed (z-scored) data.
DIVERGING = ["#2a78d6", "#f0efec", "#e34948"]
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#dedcd5"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", help="recorded CSI file")
    source.add_argument("--serial", action="store_true", help="read live from the ESP32")

    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="seconds to capture in --serial mode")
    parser.add_argument("--mac", default=None,
                        help="transmitter MAC (default: the most frequent one)")

    parser.add_argument("--hampel-window", type=int, default=11)
    parser.add_argument("--hampel-sigma", type=float, default=3.0)
    parser.add_argument("--hamming-window", type=int, default=9)
    parser.add_argument("--wavelet", default="db4")
    parser.add_argument("--wavelet-level", type=int, default=None)
    parser.add_argument("--no-normalise", action="store_true")
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--window-stride", type=int, default=16)

    parser.add_argument("--subcarrier", type=int, default=24,
                        help="which of the 48 data bins to plot (0-47)")
    parser.add_argument("--plot", action="store_true", help="show the figure")
    parser.add_argument("--save-plot", default=None)
    parser.add_argument("--save-npz", default=None,
                        help="write windows/processed/timestamps for training")
    return parser.parse_args(argv)


def read_serial(port_name: str, baud: int, duration: float) -> list:
    import serial

    try:
        port = serial.Serial(port_name, baud, timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"error: cannot open {port_name}: {exc}")

    print(f"capturing {duration:.0f}s from {port_name} ...")
    frames, rejected = [], 0
    deadline = time.time() + duration
    try:
        while time.time() < deadline:
            line = port.readline().decode("utf-8", "ignore").strip()
            if not line:
                continue
            frame = parse_line(line)
            if frame is not None:
                frames.append(frame)
            elif line.startswith("CSI_DATA,"):
                rejected += 1
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        port.close()
    print(f"  parsed {len(frames)} frames, rejected {rejected} malformed rows")
    return frames


def build_figure(result, subcarrier: int):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    subcarrier = int(np.clip(subcarrier, 0, result.raw.shape[1] - 1))
    t = result.timestamps

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.4]})
    fig.patch.set_facecolor("#fcfcfb")

    # Raw and processed live on different scales (amplitude vs z-score), so
    # they are separate panels rather than two lines on one axis.
    panels = (
        (axes[0], result.raw[:, subcarrier], SERIES_RAW,
         f"Raw amplitude — data subcarrier {subcarrier}", "amplitude"),
        (axes[1], result.processed[:, subcarrier], SERIES_PROCESSED,
         "After Hampel → Hamming → wavelet denoise",
         "z-score" if result.stats.get("normalised", True) else "amplitude"),
    )
    for ax, series, colour, title, ylabel in panels:
        ax.plot(t, series, color=colour, linewidth=1.6)
        ax.set_title(title, color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=9)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)

    # Normalised output is signed around zero, which is a diverging quantity;
    # un-normalised output is a magnitude, which is sequential.
    ax = axes[2]
    if result.stats.get("normalised", True):
        cmap = LinearSegmentedColormap.from_list("div_blue_red", DIVERGING)
        limit = float(np.abs(result.processed).max())
        vmin, vmax = -limit, limit  # a diverging ramp must be centred on zero
    else:
        cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
        vmin, vmax = None, None
    image = ax.imshow(
        result.processed.T, aspect="auto", origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[t[0], t[-1], 0, result.processed.shape[1]],
    )
    ax.set_title("Processed CSI — all 48 data subcarriers", color=INK_PRIMARY,
                 fontsize=11, loc="left")
    ax.set_ylabel("data subcarrier", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("time (s)", color=INK_SECONDARY, fontsize=9)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    bar = fig.colorbar(image, ax=ax, pad=0.01)
    bar.outline.set_edgecolor(GRID)
    bar.ax.tick_params(colors=INK_SECONDARY, labelsize=8)

    stats = result.stats
    fig.suptitle(
        f"ESP32 CSI preprocessing — {stats['frames']} frames from {stats['mac']} "
        f"@ {stats['rate_hz']:.1f} Hz ({stats['layout']} layout)",
        color=INK_PRIMARY, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.serial:
        frames = read_serial(args.port, args.baud, args.duration)
    else:
        path = pathlib.Path(args.file)
        if not path.exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            return 1
        frames, report = parse_lines(path.read_text().splitlines())
        print(f"parsed {report['parsed']} frames from {path.name}, "
              f"rejected {report['rejected']} malformed rows")

    if not frames:
        print("error: no usable CSI frames", file=sys.stderr)
        return 1

    config = PipelineConfig(
        hampel_window=args.hampel_window,
        hampel_sigma=args.hampel_sigma,
        hamming_window=args.hamming_window,
        wavelet=args.wavelet,
        wavelet_level=args.wavelet_level,
        normalise=not args.no_normalise,
        window_length=args.window_length,
        window_stride=args.window_stride,
    )

    try:
        result = run_pipeline(frames, config, mac=args.mac)
    except LayoutError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    result.stats["normalised"] = config.normalise

    stats = result.stats
    print("\npipeline")
    print(f"  transmitter        {stats['mac']}  (rssi {stats['rssi_mean']:.1f} dBm)")
    print(f"  subcarrier layout  {stats['layout']}  -> {stats['subcarriers']} data bins")
    print(f"  frames             {stats['frames']} over {stats['duration_s']:.1f}s")
    print(f"  sample rate        {stats['rate_hz']:.1f} Hz "
          f"(inter-arrival jitter {stats['jitter_ratio']:.2f}x median)")
    print(f"  hampel outliers    {stats['outlier_rate']:.2%} of samples replaced")
    print(f"  LSTM windows       {stats['n_windows']} x "
          f"{config.window_length} x {stats['subcarriers']}")

    if stats["rate_hz"] < 50:
        print(f"\n  note: {stats['rate_hz']:.0f} Hz is low for activity recognition. "
              "Raise the\n        transmit rate so windows cover enough motion.")

    if args.save_npz:
        np.savez_compressed(
            args.save_npz,
            windows=result.windows,
            processed=result.processed,
            raw=result.raw,
            timestamps=result.timestamps,
            layout=stats["layout"],
            mac=stats["mac"],
        )
        print(f"\nsaved {args.save_npz}")

    if args.plot or args.save_plot:
        import matplotlib
        if not args.plot:
            matplotlib.use("Agg")
        figure = build_figure(result, args.subcarrier)
        if args.save_plot:
            figure.savefig(args.save_plot, dpi=140, facecolor=figure.get_facecolor())
            print(f"saved {args.save_plot}")
        if args.plot:
            import matplotlib.pyplot as plt
            plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
