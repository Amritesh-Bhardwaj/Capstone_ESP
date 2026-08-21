#!/usr/bin/env python
"""Browser dashboard driven by the features Gate 1 validated.

    csi_env/bin/python har/dashboard/rank_server.py --room room1 \
        --mac A8:6E:84:93:EE:60

Then open http://127.0.0.1:8001. Ctrl-C to stop. Every session writes
`logs/<timestamp>-rank-session.csv` and a `.md` summary beside it.

Why this exists alongside ``server.py``
--------------------------------------
``server.py`` renders ``PresenceDetector``, which is built on ``motion_score``.
Gate 1 measured ``motion_score`` at AUC **0.998** separating empty from occupied
and **0.986** separating one empty room from a *different empty room*. It is a
session detector. ``dilated_pem`` is worse: 1.000 and 0.999. Both look perfect
until a null control is run, and both are unusable.

This page renders ``effective_rank``, ``n_eig`` and ``eig_ratio_3``, which
scored 0.889/0.903/0.918 against occupancy and 0.621/0.542/0.561 against a
session change. Magnitude features track the link; rank features track the
room. See `RESULTS.md` 9.9.

The presence level is ``effective_rank`` mapped through the room profile so that
**0.0 = the calibrated empty room** and **1.0 = one person walking in it**. That
is the only reason the number means anything, and it is why the room must be
calibrated before this page is useful.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import pathlib
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import count_features as cf  # noqa: E402
import csi_pipeline as cp  # noqa: E402
import presence_events as pev  # noqa: E402
import room_profile as rp  # noqa: E402
from features import motion_score  # noqa: E402
from live_demo import FrameSource, bootstrap, serial_reader, replay_reader  # noqa: E402

TRACKED = ("effective_rank", "n_eig", "eig_ratio_3", "motion_score")
VALIDATED = ("effective_rank", "n_eig", "eig_ratio_3")
LOGDIR = HERE.parent.parent.parent / "logs"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--room", default="room1", help="room profile to calibrate against")
    p.add_argument("--port", default="/dev/cu.usbserial-5B530174971")
    p.add_argument("--baud", type=int, default=460800)
    p.add_argument("--mac", default=None)
    p.add_argument("--replay", default=None, help="replay a .csi.txt instead of serial")
    p.add_argument("--replay-speed", type=float, default=1.0)
    p.add_argument("--window-seconds", type=float, default=4.0,
                   help="seconds per estimate, resampled to a uniform grid")
    p.add_argument("--interval", type=float, default=0.5, help="seconds between updates")
    p.add_argument("--history", type=int, default=180)
    p.add_argument("--enter", type=float, default=0.8,
                   help="smoothed level at which an empty room becomes occupied")
    p.add_argument("--exit", dest="exit_", type=float, default=0.4,
                   help="smoothed level at which an occupied room becomes empty")
    p.add_argument("--smooth", type=int, default=11,
                   help="running median length over recent windows")
    p.add_argument("--decide", choices=["events", "level"], default="events",
                   help="events = motion contrast + timeout (drift-immune, the "
                        "measured winner); level = absolute threshold (kept for "
                        "comparison, scored BELOW CHANCE on the labelled session)")
    p.add_argument("--timeout", type=float, default=240.0,
                   help="seconds of no motion before the room is called empty")
    p.add_argument("--enter-z", type=float, default=pev.ENTER_Z)
    p.add_argument("--confirm-level", type=float, default=pev.CONFIRM_LEVEL,
                   help="level above which presence is certain regardless of motion; "
                        "defaults to empty p95 + the measured drift envelope")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8001)
    p.add_argument("--no-log", action="store_true")
    return p.parse_args(argv)


class RankEngine:
    """Inference loop. The web layer only ever reads ``snapshot()``."""

    def __init__(self, source: FrameSource, profile: rp.RoomProfile, args):
        self.source = source
        self.profile = profile
        self.args = args
        self.names = cf.window_feature_names()
        self.idx = {n: self.names.index(n) for n in TRACKED if n in self.names}
        self.reference = profile.reference() if profile else None
        self.history = collections.deque([None] * args.history, maxlen=args.history)
        self.raw_history = collections.deque([None] * args.history, maxlen=args.history)
        self.lock = threading.Lock()
        self.started = time.time()
        # Frames already buffered by bootstrap must not count against elapsed
        # time measured from here, or the rate reads nearly double.
        self.frames_at_start = source.total
        self.rssi = None
        self.updates = 0
        self.above = 0
        self.levels: list = []
        self.present = False
        # Hysteresis plus a running median, for the reason PresenceDetector has
        # them: without both, this readout flapped. Measured on Room 1, the raw
        # per-window score gives **6.58 state changes per minute in an empty
        # room** and 28.7% false positives, because the empty p95 (1.115) sits
        # above the walking median (1.008) -- single windows cannot separate
        # them. A median over 11 windows with enter 0.8 / exit 0.4 takes that to
        # 0.21 flips per minute and 1.4% false positives, with no false
        # negatives on the walking capture. See RESULTS.md 9.11.
        self.recent: collections.deque = collections.deque(maxlen=max(1, args.smooth))
        # Events, not levels, by default. The room's own multipath wanders by
        # 1.35 presence-level units over five hours with nobody moving -- more
        # than the entire occupancy signal -- so no absolute threshold survives
        # (RESULTS.md 9.14). A short-vs-long contrast cancels anything slower
        # than its long window, which is why it scored 0% false positives where
        # the level detector scored 76.3%.
        self.events = pev.EventDetector(
            args.interval, enter_z=args.enter_z, timeout_s=args.timeout,
            confirm_level=args.confirm_level)
        self.state = {"status": "waiting", "present": False, "level": None}
        self.writer = None
        self.handle = None
        if not args.no_log:
            LOGDIR.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.log_path = LOGDIR / f"{stamp}-rank-session.csv"
            self.handle = open(self.log_path, "w", newline="", buffering=1)
            self.writer = csv.writer(self.handle)
            self.writer.writerow(["wall_time", "elapsed_s", "level_smoothed",
                                  "level_raw", "present", *TRACKED,
                                  "frames", "rate_hz"])
        else:
            self.log_path = None

    def _band(self, name: str) -> tuple:
        """Empty-room median +/- 2 MAD for one feature, from the profile."""
        if not self.profile:
            return (None, None)
        i = self.profile.features["names"].index(name)
        med = self.profile.features["empty"]["median"][i]
        mad = self.profile.features["empty"]["mad"][i]
        return (med - 2 * mad, med + 2 * mad)

    def _gate1_features(self) -> dict:
        return self.profile.gate1.get("features", {}) if self.profile else {}

    def validated(self) -> list:
        """Which features the gate actually cleared, read from the profile.

        Falls back to the module constant only when the gate has not been run
        against this room, so a stale hardcoded list can never outrank a
        measurement.
        """
        feats = self._gate1_features()
        marked = [n for n, i in feats.items()
                  if i.get("validated") and n in TRACKED]
        return marked or list(VALIDATED)

    def config(self) -> dict:
        anchored = bool(self.profile and self.profile.features.get("has_gain_anchor"))
        return {
            "room": self.profile.room if self.profile else None,
            "calibrated": bool(self.profile),
            "has_gain_anchor": anchored,
            "enter": self.args.enter,
            "exit": self.args.exit_,
            "smooth": self.args.smooth,
            "decide": self.args.decide,
            "timeout_s": self.args.timeout,
            "enter_z": self.args.enter_z,
            "confirm_level": self.args.confirm_level,
            "history": self.args.history,
            "window_seconds": self.args.window_seconds,
            "interval": self.args.interval,
            "tracked": list(TRACKED),
            "validated": self.validated(),
            "bands": {n: self._band(n) for n in TRACKED},
            "source": ("replay: " + pathlib.Path(self.args.replay).name
                       if self.args.replay else f"serial {self.args.port} @ {self.args.baud}"),
            "log": str(self.log_path) if self.log_path else None,
            # Measured AUCs, read from the profile where gate1_check wrote them.
            # Empty until the gate has been run against this room.
            # Cross-session null where the gate had a second empty capture,
            # otherwise the within-session one with a flag, because the two mean
            # very different things: motion_score scores 0.114 within a session
            # and 0.986 across sessions.
            "gate1": {
                name: [info["auc_vs_occupied"],
                       info.get("null_cross_session") or info["null_within_session"]]
                for name, info in self._gate1_features().items()
                if name in TRACKED
            },
            "null_is_cross_session": bool(
                self.profile and self.profile.gate1.get("null_cross_session_source")),
            "null_source": (self.profile.gate1.get("null_cross_session_source")
                            if self.profile else None),
            "gate1_measured": (self.profile.gate1.get("measured") if self.profile else None),
        }

    def snapshot(self) -> dict:
        with self.lock:
            state = dict(self.state)
        levels = [v for v in self.levels if v is not None]
        state["history"] = list(self.history)
        state["raw_history"] = list(self.raw_history)
        # Throughput over the whole session, not FrameSource.rate(), which
        # medians only the last 64 inter-arrivals. Frames arrive in bursts, so
        # that window reads 130-240 Hz while the sustained rate is 95-110 Hz.
        # A dashboard showing the burst figure looks like the link is twice as
        # good as it is.
        elapsed = max(time.time() - self.started, 1e-9)
        state["rate_hz"] = round(
            max(self.source.total - self.frames_at_start, 0) / elapsed, 1)
        state["rate_burst_hz"] = round(self.source.rate(), 1)
        state["frames"] = self.source.total
        state["rejected"] = self.source.rejected
        state["mac"] = self.source.mac
        state["rssi"] = round(self.rssi, 1) if self.rssi is not None else None
        state["link_ok"] = (self.rssi is not None and self.rssi >= -80.0)
        state["uptime"] = round(time.time() - self.started, 1)
        state["updates"] = self.updates
        state["above_pct"] = round(100.0 * self.above / self.updates, 1) if self.updates else 0.0
        if levels:
            state["level_min"] = round(min(levels), 3)
            state["level_max"] = round(max(levels), 3)
            state["level_avg"] = round(float(np.mean(levels)), 3)
        return state

    def step(self) -> None:
        taken = self.source.snapshot_seconds(self.args.window_seconds)
        if taken is None or self.source.layout is None:
            return
        amp, stamps = taken
        with self.source.lock:
            recent = [f for f, _ in list(self.source.frames)[-len(amp):]]
        self.rssi = float(np.mean([f.rssi for f in recent])) if recent else None
        win = cp.remove_null_pilot(amp, self.source.layout)
        win, _ = cp.hampel_filter(win)
        # Fixed duration, resampled to a uniform 100 Hz grid -- the same window
        # the offline analysis uses. Taking a fixed frame count instead let the
        # span swing between 4.2 s and 8.2 s with the packet rate, so the live
        # and offline numbers were not comparable.
        try:
            win, _, _, _ = cp.resample_uniform(win, stamps, target_hz=100.0)
        except ValueError:
            return
        if len(win) < 64:
            return

        vec = cf.window_count_features(win, 100.0, self.reference)
        feats = {n: float(vec[self.idx[n]]) for n in TRACKED if n in self.idx}
        feats["motion_score"] = float(motion_score(win))

        # Presence score: the mean of the three features Gate 1 validated,
        # each mapped through the room profile so 0 = calibrated empty room and
        # 1 = one person walking in it. The mean separates better than any one
        # alone (AUC 0.930 vs 0.889/0.903/0.918) while keeping a null margin of
        # +0.332 against a second empty session. Without a gain anchor there is
        # no scale, so no level is published rather than an invented one.
        level = smoothed = None
        if self.profile and self.profile.features.get("has_gain_anchor"):
            norm = self.profile.normalise(vec)
            live = [n for n in self.validated() if n in self.idx]
            level = float(np.mean([norm[self.idx[n]] for n in live]))
            self.recent.append(level)
            smoothed = float(np.median(self.recent))
            present_ev, z, fired = self.events.update(level)
            if self.args.decide == "events":
                self.present = present_ev
            else:
                threshold = self.args.exit_ if self.present else self.args.enter
                self.present = smoothed > threshold
            self.updates += 1
            self.above += int(self.present)
            self.levels.append(smoothed)
            self.history.append(round(smoothed, 4))
            self.raw_history.append(round(level, 4))
        else:
            self.history.append(None)
            self.raw_history.append(None)

        # One subcarrier trace for the signal visualiser -- the raw shape, not
        # a derived statistic, so a stalled board is visible as a flat line.
        # The whole window rather than the last 2.4 s, decimated to keep the
        # payload small, so the plot shows the full span it is computed over.
        col = win[:, win.shape[1] // 2]
        step = max(1, len(col) // 600)
        trace = col[::step]
        csi = [round(float(v), 2) for v in trace]
        csi_span_s = len(col) / 100.0

        with self.lock:
            self.state = {
                "status": "ok",
                "present": bool(self.present) if level is not None else False,
                "level": round(smoothed, 4) if smoothed is not None else None,
                "raw_level": round(level, 4) if level is not None else None,
                "decide": self.args.decide,
                **({"event": self.events.state()} if level is not None else {}),
                "features": {k: round(v, 4) for k, v in feats.items()},
                "csi": csi,
                "csi_span_s": round(csi_span_s, 1),
                "csi_sc": int(win.shape[1] // 2),
            }
        if self.writer:
            self.writer.writerow([
                dt.datetime.now().isoformat(timespec="seconds"),
                round(time.time() - self.started, 2),
                round(smoothed, 4) if smoothed is not None else "",
                round(level, 4) if level is not None else "",
                int(self.present) if level is not None else "",
                *[round(feats.get(n, float("nan")), 4) for n in TRACKED],
                self.source.total, round(self.source.rate(), 1),
            ])

    def run(self) -> None:
        while self.source.running:
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                print(f"warning: step failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(self.args.interval)

    def close_log(self) -> None:
        if not self.handle:
            return
        self.handle.close()
        levels = [v for v in self.levels if v is not None]
        md = self.log_path.with_suffix(".md")
        lines = [
            f"# Rank session — {dt.datetime.now().isoformat(timespec='seconds')}",
            "",
            "Presence level is `effective_rank` mapped through the room profile:",
            "**0.0 = calibrated empty room, 1.0 = one person walking in it.**",
            "Features per `RESULTS.md` §9.9 — `motion_score` is logged for contrast",
            "only and must not be used; it scored AUC 0.986 between two *empty* rooms.",
            "",
            "| | |",
            "|---|---|",
            f"| Room profile | `{self.profile.room if self.profile else 'none'}` |",
            f"| Source | {self.args.source if hasattr(self.args, 'source') else self.config()['source']} |",
            f"| Transmitter | `{self.source.mac}` |",
            f"| Duration | {time.time() - self.started:.0f} s |",
            f"| Updates | {self.updates} |",
            f"| Frames | {self.source.total} (rejected {self.source.rejected}) |",
            f"| Enter / exit | {self.args.enter} / {self.args.exit_} |",
            f"| Smoothing | running median of {self.args.smooth} windows |",
            "",
        ]
        if levels:
            lines += [
                "## Presence level",
                "",
                "```",
                f"min {min(levels):.3f}  median {float(np.median(levels)):.3f}  "
                f"max {max(levels):.3f}  mean {float(np.mean(levels)):.3f}",
                "```",
                "",
                f"- above threshold: **{self.above}/{self.updates} = "
                f"{100.0 * self.above / max(self.updates, 1):.1f}%**",
                "",
            ]
        lines += [f"Raw per-update values: `{self.log_path.name}`", ""]
        md.write_text("\n".join(lines))
        print(f"\nwrote {self.log_path}\nwrote {md}")


def build_app(engine: RankEngine):
    """Wire the engine to HTTP and the WebSocket.

    ``fastapi`` is imported at **module** scope on purpose, matching
    ``server.py``. This file uses ``from __future__ import annotations``, so
    FastAPI resolves the ``socket: WebSocket`` annotation from the module
    globals; with the import inside this function the name is invisible,
    FastAPI treats the socket as an ordinary request parameter, and every
    handshake is rejected with **HTTP 403** while plain HTTP keeps working.
    The page then shows "disconnected" with a perfectly healthy server behind
    it. ``server.py`` documents this and has a test for it; this file repeated
    the mistake anyway.
    """
    import asyncio

    app = FastAPI(title="WiFi CSI - occupancy, validated features")
    page = HERE / "static" / "rank.html"

    @app.get("/")
    async def index():
        return FileResponse(page)

    @app.get("/api/config")
    async def config():
        return JSONResponse(engine.config())

    @app.get("/api/state")
    async def state():
        return JSONResponse(engine.snapshot())

    @app.websocket("/ws")
    async def stream(socket: WebSocket):
        await socket.accept()
        await socket.send_json({"type": "config", **engine.config()})
        try:
            while True:
                await socket.send_json({"type": "state", **engine.snapshot()})
                await asyncio.sleep(engine.args.interval)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app


def main(argv=None) -> int:
    args = parse_args(argv)

    profile_path = HERE.parent / "rooms" / args.room / "profile.json"
    if not profile_path.exists():
        raise SystemExit(
            f"error: no room profile at {profile_path}\n"
            f"Calibrate first:\n"
            f"  csi_env/bin/python har/calibrate_room.py --room {args.room}"
        )
    profile = rp.RoomProfile.load(profile_path)
    if not profile.features.get("has_gain_anchor"):
        print("warning: profile has no one-person anchor; no presence level will be shown")

    # Enough frames to cover the window even at a high packet rate.
    source = FrameSource(maxlen=max(int(args.window_seconds * 400), 1600))
    if args.replay:
        path = pathlib.Path(args.replay)
        if not path.exists():
            raise SystemExit(f"error: no such file: {path}")
        reader = threading.Thread(target=replay_reader,
                                  args=(source, path, args.replay_speed), daemon=True)
    else:
        reader = threading.Thread(target=serial_reader,
                                  args=(source, args.port, args.baud), daemon=True)
    reader.start()

    try:
        bootstrap(source, max(int(args.window_seconds * 60), 120), args.mac)
    except SystemExit:
        source.running = False
        raise

    engine = RankEngine(source, profile, args)
    threading.Thread(target=engine.run, daemon=True).start()

    import uvicorn
    print(f"\nrank dashboard on http://{args.host}:{args.http_port}   (Ctrl-C to stop)",
          flush=True)
    if engine.log_path:
        print(f"logging to {engine.log_path}", flush=True)
    try:
        uvicorn.run(build_app(engine), host=args.host, port=args.http_port,
                    log_level="warning")
    finally:
        source.running = False
        engine.close_log()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
