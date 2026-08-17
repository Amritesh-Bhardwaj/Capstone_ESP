#!/usr/bin/env python
"""Browser dashboard for the live presence and activity demo.

Same pipeline, same detector, same numbers as ``har/live_demo.py`` -- the
inference code is imported from it rather than reimplemented, so the two
readouts cannot drift apart. Only the rendering is different: a browser page
fed by a WebSocket instead of a matplotlib window.

    # Replay a recording (no hardware, no calibration -- rehearse with this)
    python har/dashboard/server.py --model har/model.joblib \
        --replay har/recordings-synthetic/walking_synth0.csv

    # Live off the board
    python har/dashboard/server.py --model har/model.joblib --baud 460800

    # Presence only, no trained model
    python har/dashboard/server.py --calibrate-only --baud 460800

Then open http://127.0.0.1:8000. Ctrl-C to stop.

Binds to localhost by default. ``--host 0.0.0.0`` puts CSI on the venue network
with no authentication in front of it; only do that if you need the page open
on a second machine.
"""

from __future__ import annotations

import argparse
import base64
import collections
import pathlib
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from features import PresenceDetector, window_features  # noqa: E402
from live_demo import (  # noqa: E402
    DEFAULT_PORT,
    FrameSource,
    bootstrap,
    calibrate_live,
    preprocess,
    replay_reader,
    serial_reader,
)

STATIC = pathlib.Path(__file__).resolve().parent / "static"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    # Source and model -- identical meaning to live_demo.py.
    parser.add_argument("--model", default=None, help="model.joblib from train_activity.py")
    parser.add_argument("--calibrate-only", action="store_true",
                        help="presence only: calibrate on an empty room, then run")
    parser.add_argument("--calibrate-seconds", type=float, default=15.0)
    parser.add_argument("--enter-sigma", type=float, default=6.0)
    parser.add_argument("--exit-sigma", type=float, default=3.0)
    parser.add_argument("--enter-threshold", type=float, default=None,
                        help="absolute motion_score for presence; skips calibration")
    parser.add_argument("--exit-threshold", type=float, default=None)
    parser.add_argument("--baseline", type=float, default=None,
                        help="observed quiet-room median, for the display only")
    parser.add_argument("--replay", default=None, help="replay a recording instead of serial")
    parser.add_argument("--replay-speed", type=float, default=1.0)
    parser.add_argument("--port", default=DEFAULT_PORT, help="serial port")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--mac", default=None)
    parser.add_argument("--window-length", type=int, default=64)
    parser.add_argument("--history", type=int, default=240)
    parser.add_argument("--max-span-factor", type=float, default=2.5)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--debounce", type=int, default=3)
    # Server-only.
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--interval", type=float, default=0.15,
                        help="seconds between inference steps and pushes")
    return parser.parse_args(argv)


def encode_waterfall(window: np.ndarray) -> tuple:
    """(frames, bins) window -> (base64 uint8, bins, frames), subcarrier-major.

    Quantised against the 2nd-98th percentile of this window, which is the
    same contrast stretch ``live_demo`` gives the matplotlib waterfall. Sending
    bytes rather than floats keeps a 48x64 window at 4 KB of base64 instead of
    ~60 KB of JSON numbers.
    """
    lo = float(np.percentile(window, 2))
    hi = float(np.percentile(window, 98))
    scaled = np.clip((window.T - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    data = (scaled * 255).astype(np.uint8)
    return base64.b64encode(data.tobytes()).decode("ascii"), data.shape[0], data.shape[1]


class Engine:
    """Runs the inference loop in a thread and publishes the latest state.

    The web layer never touches the pipeline; it only reads ``snapshot()``.
    """

    def __init__(self, source: FrameSource, detector: PresenceDetector, bundle, args):
        self.source = source
        self.detector = detector
        self.bundle = bundle
        self.args = args
        self.labels = list(bundle["labels"]) if bundle else []
        self.history = collections.deque([None] * args.history, maxlen=args.history)
        self.lock = threading.Lock()
        # A window of window_length frames should span about this long; much
        # longer means frames were dropped. Same guard as live_demo.
        self.nominal_span = args.window_length / max(source.rate() or 22.0, 1.0)
        self.gaps = 0
        self.started = time.time()
        self.state = {
            "status": "waiting",
            "present": False,
            "score": None,
            "confidence": 0.0,
            "activity": None,
            "activity_confidence": 0.0,
            "values": [0.0] * len(self.labels),
            "waterfall": None,
        }

    def config(self) -> dict:
        """Everything the page needs once, at connect time."""
        return {
            "labels": self.labels,
            "baseline": self.detector.baseline,
            "enter_threshold": self.detector.enter_threshold,
            "exit_threshold": self.detector.exit_threshold,
            "window_length": self.args.window_length,
            "history": self.args.history,
            "bins": len(self.source.layout.data_bins) if self.source.layout else 0,
            "source": ("replay: " + pathlib.Path(self.args.replay).name
                       if self.args.replay else f"serial {self.args.port} @ {self.args.baud}"),
            "model": pathlib.Path(self.args.model).name if self.args.model else None,
            "kind": self.bundle.get("kind", "features") if self.bundle else None,
        }

    def snapshot(self) -> dict:
        with self.lock:
            state = dict(self.state)
        state["history"] = list(self.history)
        state["rate_hz"] = round(self.source.rate(), 1)
        state["frames"] = self.source.total
        state["rejected"] = self.source.rejected
        state["gaps"] = self.gaps
        state["mac"] = self.source.mac
        state["uptime"] = round(time.time() - self.started, 1)
        return state

    def _classify(self, window: np.ndarray) -> tuple:
        if self.bundle.get("kind") == "sequence":
            # LSTM: consumes the window itself, not hand-made features.
            probs = self.bundle["model"].predict_proba(window[None, ...])[0]
        else:
            feats = window_features(window, max(self.source.rate(), 1.0))[None, :]
            probs = self.bundle["model"].predict_proba(feats)[0]
        order = {c: i for i, c in enumerate(self.bundle["model"].classes_)}
        values = [float(probs[order[l]]) if l in order else 0.0 for l in self.labels]
        best = int(np.argmax(values))
        return self.labels[best], values[best], values

    def step(self) -> None:
        taken = self.source.snapshot_span(self.args.window_length)
        if taken is None:
            return
        amplitude, span = taken

        if span > self.args.max_span_factor * self.nominal_span:
            # Straddles a dropout. Hold the last state rather than invent one.
            self.gaps += 1
            with self.lock:
                self.state["status"] = "gap"
                self.state["span"] = round(span, 2)
                self.state["nominal_span"] = round(self.nominal_span, 2)
            return

        window = preprocess(amplitude, self.source.layout)
        present, score, confidence = self.detector.update(window)
        self.history.append(round(float(score), 6))

        activity, activity_confidence, values = None, 0.0, [0.0] * len(self.labels)
        if self.bundle and present:
            activity, activity_confidence, values = self._classify(window)

        payload, bins, frames = encode_waterfall(window)
        with self.lock:
            self.state = {
                "status": "ok",
                "present": bool(present),
                "score": round(float(score), 6),
                "confidence": round(float(confidence), 4),
                "activity": activity,
                "activity_confidence": round(float(activity_confidence), 4),
                "values": [round(v, 4) for v in values],
                "waterfall": {"data": payload, "bins": bins, "frames": frames},
                "span": round(span, 2),
                "nominal_span": round(self.nominal_span, 2),
            }

    def run(self) -> None:
        while self.source.running:
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001
                # One bad window must not kill the demo mid-presentation.
                print(f"warning: inference step failed: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
            time.sleep(self.args.interval)


def build_app(engine: Engine):
    """Wire the engine to HTTP and the WebSocket.

    ``fastapi`` is imported at module scope on purpose. This file uses
    ``from __future__ import annotations``, so FastAPI resolves the
    ``socket: WebSocket`` annotation from the *module* globals; with the import
    inside this function the name is invisible, FastAPI treats the socket as a
    request parameter, and every handshake is rejected with 403. Pinned by
    test_websocket_delivers_config_then_state.
    """
    import asyncio

    app = FastAPI(title="WiFi CSI - presence and activity")

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

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


def build_detector(source: FrameSource, bundle, args) -> PresenceDetector:
    """Same three-way choice as live_demo: fitted, from the bundle, or live."""
    detector = bundle.get("presence") if bundle else None

    if args.enter_threshold is not None:
        exit_threshold = (args.exit_threshold if args.exit_threshold is not None
                          else args.enter_threshold * 0.9)
        baseline = args.baseline if args.baseline is not None else exit_threshold * 0.8
        detector = PresenceDetector(baseline, 1.0, enter_sigma=1.0, exit_sigma=1.0,
                                    smoothing=args.smoothing, debounce=args.debounce)
        detector.enter_threshold = args.enter_threshold
        detector.exit_threshold = exit_threshold
        detector.scale = max(args.enter_threshold - baseline, 1e-6)
        print(f"using FITTED thresholds: enter > {detector.enter_threshold:.4f}, "
              f"exit < {detector.exit_threshold:.4f} (baseline {baseline:.4f}) "
              "— no calibration")
    elif detector is None or args.calibrate_only:
        detector = calibrate_live(source, args.window_length, args.calibrate_seconds,
                                  args.enter_sigma, args.exit_sigma,
                                  args.smoothing, args.debounce)
    return detector


def main(argv=None) -> int:
    args = parse_args(argv)

    bundle = None
    if args.model:
        import joblib
        bundle = joblib.load(args.model)
        print(f"loaded {args.model}: labels {bundle['labels']}")

    source = FrameSource(maxlen=max(args.window_length * 6, 600))
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
        bootstrap(source, args.window_length, args.mac)
    except SystemExit:
        source.running = False
        raise

    detector = build_detector(source, bundle, args)
    engine = Engine(source, detector, bundle, args)
    threading.Thread(target=engine.run, daemon=True).start()

    import uvicorn

    print(f"\ndashboard on http://{args.host}:{args.http_port}  (Ctrl-C to stop)")
    if args.host == "0.0.0.0":  # noqa: S104
        print("warning: bound to all interfaces — the CSI stream is unauthenticated")
    try:
        uvicorn.run(build_app(engine), host=args.host, port=args.http_port,
                    log_level="warning")
    finally:
        source.running = False
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
