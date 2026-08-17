"""Tests for the browser dashboard's data path.

Run: csi_env/bin/python har/dashboard/test_dashboard.py   (or pytest)

The rendering is not tested here; what is tested is that the JSON the page
receives carries the same numbers the matplotlib demo would show, and that the
waterfall survives the quantise -> base64 -> decode trip intact.
"""

from __future__ import annotations

import base64
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from features import PresenceDetector, motion_score  # noqa: E402
from live_demo import FrameSource, bootstrap, preprocess  # noqa: E402
from server import Engine, encode_waterfall, parse_args  # noqa: E402

RECORDINGS = HERE.parent / "recordings-synthetic"
RNG = np.random.default_rng(0)


def _window(frames=64, bins=48):
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, bins)))
    return np.tile(base, (frames, 1)) * (1 + RNG.normal(0, 0.05, (frames, bins)))


def _fixture(label: str) -> pathlib.Path:
    """A synthetic take for `label`, generating one if the repo has none.

    `recordings-synthetic/` is gitignored, so a fresh checkout starts without
    it and these tests must not depend on someone having run make_synthetic.py.
    """
    path = RECORDINGS / f"{label}_synth0.csv"
    if path.exists():
        return path

    import tempfile

    from make_synthetic import make_take

    outdir = pathlib.Path(tempfile.gettempdir()) / "har-dashboard-fixtures"
    outdir.mkdir(exist_ok=True)
    generated = outdir / f"{label}_synth0.csv"
    if not generated.exists():
        generated.write_text("\n".join(make_take(label, 720, 24.0, seed=7)) + "\n")
    return generated


def _loaded_source(path: pathlib.Path, window_length: int = 64) -> FrameSource:
    source = FrameSource(maxlen=600)
    for line in path.read_text().splitlines():
        if line.startswith("CSI_DATA,"):
            source.push(line)
    bootstrap(source, window_length, None)
    return source


# --- waterfall encoding -----------------------------------------------------


def test_waterfall_is_subcarrier_major_and_the_right_size():
    window = _window(frames=64, bins=48)
    payload, bins, frames = encode_waterfall(window)
    assert (bins, frames) == (48, 64), (bins, frames)
    assert len(base64.b64decode(payload)) == 48 * 64


def test_waterfall_preserves_the_contrast_ordering():
    """A subcarrier that is brighter in the window must stay brighter."""
    window = _window()
    window[:, 10] *= 3.0
    payload, bins, frames = encode_waterfall(window)
    data = np.frombuffer(base64.b64decode(payload), dtype=np.uint8).reshape(bins, frames)
    assert data[10].mean() > data[0].mean()


def test_waterfall_survives_a_flat_window():
    """A constant window has zero percentile spread; it must not divide by 0."""
    payload, _, _ = encode_waterfall(np.full((64, 48), 7.0))
    assert np.isfinite(np.frombuffer(base64.b64decode(payload), dtype=np.uint8)).all()


# --- engine state -----------------------------------------------------------


def test_step_publishes_the_same_score_the_detector_reports():
    source = _loaded_source(_fixture("walking"))
    windows = []
    frames = [f.amplitude for f, _ in source.frames]
    for start in range(0, len(frames) - 64 + 1, 16):
        windows.append(preprocess(np.stack(frames[start:start + 64]), source.layout))
    detector = PresenceDetector.calibrate(np.stack(windows))

    args = parse_args(["--replay", str(_fixture("walking"))])
    engine = Engine(source, detector, None, args)
    engine.step()
    state = engine.snapshot()

    assert state["status"] == "ok", state
    assert state["waterfall"]["bins"] == len(source.layout.data_bins)
    # The published score is the detector's smoothed score, not a re-derived
    # one -- if these ever disagree the page is lying about the pipeline.
    assert state["score"] == round(float(np.median(detector._recent)), 6)
    assert state["history"][-1] == state["score"]


def test_a_dropout_window_is_reported_as_a_gap_not_as_presence():
    source = _loaded_source(_fixture("walking"))
    detector = PresenceDetector(0.01, 0.001)
    args = parse_args(["--replay", str(_fixture("walking"))])
    engine = Engine(source, detector, None, args)
    engine.nominal_span = 1e-6  # any real window now looks like a dropout
    engine.step()
    state = engine.snapshot()
    assert state["status"] == "gap"
    assert state["present"] is False
    assert engine.gaps == 1


def test_state_is_json_serialisable():
    import json

    source = _loaded_source(_fixture("empty"))
    detector = PresenceDetector(0.01, 0.001)
    args = parse_args(["--replay", str(_fixture("empty"))])
    engine = Engine(source, detector, None, args)
    engine.step()
    json.dumps({"type": "state", **engine.snapshot()})
    json.dumps({"type": "config", **engine.config()})


def test_presence_agrees_with_a_direct_motion_score():
    """Empty-room replay stays absent; walking replay trips the threshold."""
    for label, expect_above in (("empty", False), ("walking", True)):
        source = _loaded_source(_fixture(label))
        amplitude, _ = source.snapshot_span(64)
        window = preprocess(amplitude, source.layout)
        detector = PresenceDetector(0.0, 0.0)
        detector.enter_threshold = detector.exit_threshold = 0.02
        args = parse_args(["--replay", str(_fixture(label))])
        engine = Engine(source, detector, None, args)
        engine.step()
        assert (motion_score(window) > 0.02) is expect_above, label
        assert engine.snapshot()["score"] == round(motion_score(window), 6), label


def test_websocket_delivers_config_then_state():
    """The page's only live channel. This failed with HTTP 403 before the fix.

    ``server.py`` uses postponed annotations, so importing ``WebSocket`` inside
    ``build_app`` left FastAPI unable to resolve the socket parameter and it
    rejected every handshake -- while every plain GET still worked, so nothing
    else here noticed.
    """
    import json
    import threading
    import time

    import uvicorn
    import websockets.sync.client as client

    from server import build_app

    source = _loaded_source(_fixture("walking"))
    args = parse_args(["--replay", str(_fixture("walking"))])
    engine = Engine(source, PresenceDetector(0.01, 0.001), None, args)
    engine.step()

    server = uvicorn.Server(uvicorn.Config(build_app(engine), host="127.0.0.1",
                                           port=8799, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    assert server.started, "uvicorn did not start"

    try:
        with client.connect("ws://127.0.0.1:8799/ws", open_timeout=5) as socket:
            first = json.loads(socket.recv(timeout=5))
            second = json.loads(socket.recv(timeout=5))
    finally:
        server.should_exit = True

    assert first["type"] == "config" and first["bins"] == 48, first
    assert second["type"] == "state" and second["status"] == "ok", second
    assert len(second["waterfall"]["data"]) == 4096


def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
