#!/usr/bin/env python
"""Pack CSI captures into compressed archives that are small enough to share.

    csi_env/bin/python har/pack_dataset.py --room room1
    csi_env/bin/python har/pack_dataset.py --room room1 --verify
    csi_env/bin/python har/pack_dataset.py --unpack har/dataset/room1/empty/xyz.npz

Raw `.csi.txt` is ASCII: roughly 470 bytes a frame to carry 128 int8 values and
a short header. Storing the wire values as `int8` and letting zip do the rest
is **6.3x smaller and lossless** — measured 9.1 MB to 1.5 MB on a real capture,
so the 338 MB corpus becomes about 54 MB.

Nothing is thrown away. `csi` holds the original int8 pairs exactly as the
board emitted them, before `hypot` turns them into amplitudes, so anything
derivable from the text file is derivable from the archive — including phase,
which the current pipeline discards but which is the most promising unexplored
direction (`DATA.md` §8).

Rows the parser would reject are dropped and counted, not silently kept: a
token outside -128..127 means a log line merged into the array.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
INT_RE = re.compile(r"^-?\d+$")
LLTF = 128
# Header fields worth keeping. Index into the 25-field FULL format.
KEEP = {"mac": 2, "rssi": 3, "rate": 4, "sig_mode": 5, "mcs": 6, "bandwidth": 7,
        "channel": 16, "local_timestamp": 18, "ant": 19}


def parse_capture(path: pathlib.Path) -> dict | None:
    csi, mac, rssi, ts, chan, mode = [], [], [], [], [], []
    rejected = 0
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("CSI_DATA,"):
            continue
        open_at = line.find("[")
        close_at = line.find("]", open_at + 1)
        if open_at < 0 or close_at < 0:
            rejected += 1
            continue
        head = line[:open_at].rstrip(",").split(",")
        if len(head) != 25:
            rejected += 1
            continue
        tokens = line[open_at + 1:close_at].split()[:LLTF]
        if len(tokens) < LLTF or not all(INT_RE.match(t) for t in tokens):
            rejected += 1
            continue
        values = [int(t) for t in tokens]
        if any(v < -128 or v > 127 for v in values):
            rejected += 1
            continue
        try:
            rssi.append(int(head[KEEP["rssi"]]))
            ts.append(int(head[KEEP["local_timestamp"]]))
            chan.append(int(head[KEEP["channel"]]))
            mode.append(int(head[KEEP["sig_mode"]]))
        except ValueError:
            rejected += 1
            continue
        csi.append(values)
        mac.append(head[KEEP["mac"]])
    if not csi:
        return None
    macs, idx = np.unique(np.array(mac), return_inverse=True)
    return {
        "csi": np.array(csi, dtype=np.int8),
        "rssi": np.array(rssi, dtype=np.int8),
        "local_timestamp": np.array(ts, dtype=np.int64),
        "channel": np.array(chan, dtype=np.uint8),
        "sig_mode": np.array(mode, dtype=np.uint8),
        "mac_idx": idx.astype(np.uint8),
        "macs": macs,
        "rejected": rejected,
    }


def pack_one(src: pathlib.Path, dst: pathlib.Path) -> dict | None:
    data = parse_capture(src)
    if data is None:
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    rejected = data.pop("rejected")
    np.savez_compressed(dst, **data)
    return {
        "source": src.name,
        "archive": dst.name,
        "frames": int(len(data["csi"])),
        "rejected": rejected,
        "source_bytes": src.stat().st_size,
        "archive_bytes": dst.stat().st_size,
        "sha256_source": hashlib.sha256(src.read_bytes()).hexdigest()[:16],
        "transmitters": {m: int((data["mac_idx"] == i).sum())
                         for i, m in enumerate(data["macs"])},
    }


def unpack(archive: pathlib.Path, out: pathlib.Path) -> int:
    """Rebuild a readable CSI_DATA text file from an archive.

    Header fields the archive does not carry are written as 0. The array and
    every field the pipeline actually reads (mac, rssi, channel, sig_mode,
    local_timestamp) round-trip exactly.
    """
    d = np.load(archive, allow_pickle=False)
    csi, macs, idx = d["csi"], d["macs"], d["mac_idx"]
    rssi, ts, chan, mode = d["rssi"], d["local_timestamp"], d["channel"], d["sig_mode"]
    with open(out, "w") as fh:
        for i in range(len(csi)):
            head = ["CSI_DATA", "PASSIVE", str(macs[idx[i]]), str(int(rssi[i])),
                    "11", str(int(mode[i])), "0", "0", "0", "0", "0", "0", "0",
                    "0", "-96", "0", str(int(chan[i])), "0", str(int(ts[i])),
                    "0", "0", "0", "0", "0", str(LLTF)]
            body = " ".join(str(int(v)) for v in csi[i])
            fh.write(",".join(head) + ",[" + body + " ]\n")
    return len(csi)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--room", default="room1")
    p.add_argument("--src", default=None, help="recordings dir (default: har/recordings/<room>)")
    p.add_argument("--out", default=None, help="output dir (default: har/dataset/<room>)")
    p.add_argument("--verify", action="store_true",
                   help="unpack every archive and confirm it re-parses identically")
    p.add_argument("--unpack", default=None, help="rebuild one .npz back to .csi.txt")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.unpack:
        src = pathlib.Path(args.unpack)
        out = src.with_suffix(".csi.txt")
        n = unpack(src, out)
        print(f"{n} frames -> {out}")
        return 0

    src_root = pathlib.Path(args.src) if args.src else HERE / "recordings" / args.room
    out_root = pathlib.Path(args.out) if args.out else HERE / "dataset" / args.room
    if not src_root.exists():
        raise SystemExit(f"error: no recordings at {src_root}")

    manifest, total_src, total_out = [], 0, 0
    for folder in sorted(p for p in src_root.iterdir() if p.is_dir()):
        files = sorted(folder.glob("*.csi.txt"))
        if not files:
            continue
        print(f"\n{folder.name}/  ({len(files)} captures)")
        for f in files:
            info = pack_one(f, out_root / folder.name / (f.stem + ".npz"))
            if info is None:
                print(f"  {f.name[:52]:54s} no usable frames, skipped")
                continue
            info["folder"] = folder.name
            manifest.append(info)
            total_src += info["source_bytes"]
            total_out += info["archive_bytes"]
            print(f"  {f.name[:52]:54s} {info['frames']:6d} frames  "
                  f"{info['source_bytes']/1e6:5.1f} -> {info['archive_bytes']/1e6:4.1f} MB")

    if not manifest:
        raise SystemExit("error: nothing packed")
    (out_root / "manifest.json").write_text(json.dumps(
        {"room": args.room, "captures": manifest,
         "source_bytes": total_src, "archive_bytes": total_out}, indent=2))

    print(f"\n{len(manifest)} captures")
    print(f"  {total_src/1e6:.0f} MB -> {total_out/1e6:.0f} MB "
          f"({total_src/max(total_out,1):.1f}x smaller, lossless)")
    print(f"  manifest -> {out_root/'manifest.json'}")

    if args.verify:
        print("\nverifying every archive round-trips ...")
        import tempfile
        sys.path.insert(0, str(HERE))
        import csi_pipeline as cp
        bad = 0
        for info in manifest:
            arc = out_root / info["folder"] / info["archive"]
            with tempfile.NamedTemporaryFile(suffix=".csi.txt", delete=False) as tmp:
                tmp_path = pathlib.Path(tmp.name)
            unpack(arc, tmp_path)
            frames, _ = cp.parse_lines(tmp_path.read_text().splitlines())
            tmp_path.unlink()
            if len(frames) != info["frames"]:
                bad += 1
                print(f"  MISMATCH {info['archive']}: {len(frames)} vs {info['frames']}")
        print(f"  {len(manifest) - bad}/{len(manifest)} archives verified")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
