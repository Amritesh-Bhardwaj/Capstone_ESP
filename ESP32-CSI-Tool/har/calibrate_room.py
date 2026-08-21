"""Calibrate a room in about two minutes.

    # guided, both phases, writes har/rooms/room1/profile.json
    csi_env/bin/python har/calibrate_room.py --room room1

    # one phase at a time
    csi_env/bin/python har/calibrate_room.py --room room1 --phase empty
    csi_env/bin/python har/calibrate_room.py --room room1 --phase one-person

    # rebuild the profile from captures already on disk (no hardware needed)
    csi_env/bin/python har/calibrate_room.py --room room1 --fit

    # adopt an existing recording as the empty phase
    csi_env/bin/python har/calibrate_room.py --room room1 --phase empty \
        --from-file har/recordings/room1/2026-08-20T21-15-00_empty_baseline.csi.txt

The ESP32 and the router must not move between calibration and use. Any change
in placement invalidates the profile -- tape the board down and record where it
is in ``--note``.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import room_profile as rp

DEFAULT_PORT = "/dev/cu.usbserial-5B530174971"
ROOMS = pathlib.Path(__file__).resolve().parent / "rooms"
PHASE_FILES = {"empty": "empty.csi.txt", "one-person": "one_person.csi.txt"}

PROMPTS = {
    "empty": (
        "PHASE 1 of 2 -- EMPTY ROOM\n"
        "  Leave the room. Nobody inside, door shut, no pets, no fans.\n"
        "  This measures the zero point: noise floor and static multipath."
    ),
    "one-person": (
        "PHASE 2 of 2 -- ONE PERSON WALKING\n"
        "  Exactly one person, walking a normal loop around the room.\n"
        "  Not pacing one line and not standing still -- cover the floor.\n"
        "  This is the gain anchor: it sets the scale of the count curve."
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--room", required=True, help="room name, e.g. room1")
    parser.add_argument(
        "--phase",
        choices=["empty", "one-person", "both"],
        default="both",
        help="which phase to capture (default: both, guided)",
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=460800)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds per phase")
    parser.add_argument("--mac", default=None, help="pin the transmitter MAC")
    parser.add_argument("--from-file", default=None, help="use this capture instead of the radio")
    parser.add_argument("--fit", action="store_true", help="rebuild the profile, capture nothing")
    parser.add_argument("--length-ft", type=float, default=None)
    parser.add_argument("--width-ft", type=float, default=None)
    parser.add_argument("--link-ft", type=float, default=None, help="router-to-ESP32 distance")
    parser.add_argument("--note", default="", help="placement note, kept in the profile")
    return parser.parse_args(argv)


def capture_phase(args, phase: str, room_dir: pathlib.Path) -> pathlib.Path:
    target = room_dir / PHASE_FILES[phase]
    if args.from_file:
        source = pathlib.Path(args.from_file)
        if not source.exists():
            raise SystemExit(f"error: {source} does not exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(errors="replace"))
        print(f"  adopted {source} as the {phase} phase")
        return target

    print("\n" + PROMPTS[phase])
    input(f"  press ENTER to record {args.duration:.0f}s ... ")
    print(f"  recording {args.duration:.0f}s ...", flush=True)
    rows = rp.capture_to_file(args.port, args.baud, args.duration, target)
    print(f"  {rows} CSI rows -> {target}")
    if rows < 100:
        raise SystemExit(
            "error: almost no CSI captured. Check --port and --baud, and that "
            "there is Wi-Fi traffic on channel 6."
        )
    return target


USABLE_RSSI_DBM = -80.0


def check_link(label: str, capture: dict) -> None:
    """Refuse a capture taken on a transmitter too weak to carry CSI.

    ``select_mac`` picks the *chattiest* transmitter, not the strongest. On
    2026-08-20 the room's own AP (``A8:6E:84:93:EE:60``, -53 dBm) dropped off
    channel 6 for about half an hour; the only thing still talking was a distant
    AP at **-88.8 dBm**, and capture continued perfectly happily against pure
    noise. Nothing in the output looked wrong -- the rate was plausible and the
    rows parsed.

    ESP32 CSI stops being usable around -75 to -80 dBm, so anything below that
    is rejected rather than quietly analysed.
    """
    rssi = capture.get("rssi_mean", 0.0)
    if rssi >= USABLE_RSSI_DBM:
        return
    raise SystemExit(
        f"error: the {label} capture is on {capture['mac']} at {rssi:.1f} dBm, "
        f"below the {USABLE_RSSI_DBM:.0f} dBm usable floor.\n"
        f"The strong transmitter is probably off channel 6 right now. Check "
        f"with:\n"
        f"    csi_env/bin/python har/run_pipeline.py --serial --duration 15\n"
        f"and wait for a transmitter above -75 dBm before calibrating."
    )


def check_coverage(label: str, capture: dict, requested: float) -> None:
    """Fail loudly when the radio was silent for much of the capture.

    In passive mode CSI only exists while something is transmitting, and an
    empty room is often also an idle network. Measured in Room 1: a 601-second
    empty-room capture contained CSI for only 264 seconds -- 337 seconds
    produced nothing at all. Wall-clock duration therefore says nothing about
    how much data you actually have, and a calibration built on a tenth of the
    requested window would look perfectly normal.
    """
    span = capture["stats"].get("duration_s", 0.0)
    if requested <= 0 or span >= 0.7 * requested:
        return
    raise SystemExit(
        f"error: the {label} capture covers only {span:.0f}s of the "
        f"{requested:.0f}s requested.\n"
        f"The transmitter went quiet. Start a fixed-rate ping to the router "
        f"and keep it running for every capture and for the demo:\n"
        f"    sudo ping -i 0.002 <router_ip>\n"
        f"then recapture. See DEMO.md Step 0.5."
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    room_dir = ROOMS / args.room
    room_dir.mkdir(parents=True, exist_ok=True)

    if not args.fit:
        phases = ["empty", "one-person"] if args.phase == "both" else [args.phase]
        for phase in phases:
            capture_phase(args, phase, room_dir)

    empty_path = room_dir / PHASE_FILES["empty"]
    if not empty_path.exists():
        raise SystemExit(f"error: no empty-room capture at {empty_path}")

    print("\nanalysing ...")
    empty = rp.load_capture(empty_path, mac=args.mac)
    one_path = room_dir / PHASE_FILES["one-person"]
    one = rp.load_capture(one_path, mac=args.mac) if one_path.exists() else None
    for label, capture in (("empty", empty), ("one-person", one)):
        if capture is not None:
            check_link(label, capture)
            check_coverage(label, capture, args.duration)

    geometry = {
        "length_ft": args.length_ft,
        "width_ft": args.width_ft,
        "link_ft": args.link_ft,
    }
    existing = room_dir / "profile.json"
    if existing.exists():
        # Keep geometry recorded on an earlier run rather than blanking it.
        prior = rp.RoomProfile.load(existing).geometry
        geometry = {k: (v if v is not None else prior.get(k)) for k, v in geometry.items()}

    profile = rp.build_profile(
        args.room, empty, one, geometry=geometry, notes=args.note
    )
    profile.save(existing)

    stats = empty["stats"]
    print(f"\nroom profile -> {existing}")
    print(f"  transmitter        {profile.link['mac']}  ({profile.link['rssi_mean_empty']:.1f} dBm)")
    print(f"  kept after MAC     {stats['mac_fraction']:.0%} of parsed frames")
    print(f"  source rate        {stats['rate_hz']:.1f} Hz  (jitter {stats['jitter_ratio']:.2f}x)")
    print(f"  resampled to       {stats['target_hz']:.0f} Hz, {stats['gap_fraction']:.2%} interpolated over gaps")
    print(f"  reliable bins      {profile.subcarriers['n_reliable']}/{len(profile.subcarriers['mean'])}")
    print(f"  empty windows      {stats['windows']} ({stats['windows_dropped_to_gaps']} dropped to gaps)")
    print(f"  presence baseline  {profile.presence['baseline_median']:.4f}"
          f"  enter {profile.presence['enter_threshold']:.4f}")

    if one is None:
        print("\n  NO GAIN ANCHOR -- presence only. Counting needs the")
        print("  one-person phase:  --phase one-person")
        return 0

    gain = profile.gain()
    names = profile.features["names"]
    order = np.argsort(-np.abs(gain))
    print("\n  gain per feature (one walking person, empty = 0):")
    for idx in order[:8]:
        marker = "" if abs(gain[idx]) > 1e-9 else "   <- dead in this room"
        print(f"    {names[idx]:24s} {gain[idx]:+.4f}{marker}")
    dead = int(np.sum(np.abs(gain) < 1e-9))
    if dead:
        print(f"    ({dead} feature(s) did not respond to one occupant)")
    print("\n  next:  csi_env/bin/python har/gate1_check.py --room " + args.room)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
