"""Tests for the rank dashboard.

Run: csi_env/bin/python har/dashboard/test_rank_dashboard.py   (or pytest)

The 403 test is the reason this file exists. ``server.py`` already documented
that importing ``WebSocket`` inside ``build_app`` breaks the handshake under
postponed annotations, and already had a test pinning it -- and
``rank_server.py`` reproduced the same bug anyway, because a docstring in a
neighbouring file is not a guardrail. Every plain GET kept working, so the only
symptom was the page reading "disconnected" with a healthy server behind it.
"""

from __future__ import annotations

import json
import pathlib
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import room_profile as rp  # noqa: E402
from live_demo import FrameSource, bootstrap  # noqa: E402
from rank_server import RankEngine, build_app, parse_args  # noqa: E402

ROOMS = HERE.parent / "rooms"
RECORDINGS = HERE.parent / "recordings-synthetic"


def _fixture(label: str) -> pathlib.Path:
    path = RECORDINGS / f"{label}_synth0.csv"
    if path.exists():
        return path
    import tempfile

    from make_synthetic import make_take

    outdir = pathlib.Path(tempfile.gettempdir()) / "har-rank-fixtures"
    outdir.mkdir(exist_ok=True)
    generated = outdir / f"{label}_synth0.csv"
    if not generated.exists():
        generated.write_text("\n".join(make_take(label, 900, 24.0, seed=11)) + "\n")
    return generated


def _loaded_source(path: pathlib.Path, window_length: int) -> FrameSource:
    source = FrameSource(maxlen=max(window_length * 6, 1600))
    for line in path.read_text().splitlines():
        if line.startswith("CSI_DATA,"):
            source.push(line)
    bootstrap(source, window_length, None)
    return source


def _profile() -> rp.RoomProfile:
    """A synthetic profile, so the tests never depend on a calibrated room."""
    import count_features as cf

    rng = np.random.default_rng(3)
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, 48)))
    quiet = np.stack([np.tile(base, (400, 1)) * (1 + rng.normal(0, 0.01, (400, 48)))
                      for _ in range(5)])
    busy = np.stack([np.tile(base, (400, 1)) * (1 + rng.normal(0, 0.06, (400, 48)))
                     for _ in range(5)])

    def cap(windows):
        return {
            "raw": np.concatenate(list(windows)),
            "windows": windows,
            "features": cf.features_for_windows(windows, 100.0),
            "motion_scores": np.array([0.05 + 0.001 * i for i in range(len(windows))]),
            "mac": "AA:BB:CC:DD:EE:FF", "layout": "fft", "rssi_mean": -55.0,
            "stats": {"rate_hz": 100.0, "jitter_ratio": 1.5, "target_hz": 100.0,
                      "gap_fraction": 0.0, "mac_fraction": 1.0,
                      "windows": len(windows), "windows_dropped_to_gaps": 0},
        }

    return rp.build_profile("test", cap(quiet), cap(busy))


def _engine(window_length: int = 64) -> RankEngine:
    path = _fixture("walking")
    source = _loaded_source(path, window_length)
    args = parse_args(["--replay", str(path), "--window-seconds", "2.0", "--no-log"])
    engine = RankEngine(source, _profile(), args)
    engine.step()
    return engine


# --- the 403 regression -----------------------------------------------------


def test_websocket_delivers_config_then_state():
    """Pins the bug that made the page say "disconnected".

    Under ``from __future__ import annotations`` FastAPI resolves the
    ``socket: WebSocket`` annotation from module globals. Importing fastapi
    inside ``build_app`` hides the name, FastAPI treats the socket as a normal
    request parameter, and the handshake is rejected with HTTP 403 while every
    GET still succeeds.
    """
    import uvicorn
    import websockets.sync.client as client

    engine = _engine()
    server = uvicorn.Server(uvicorn.Config(build_app(engine), host="127.0.0.1",
                                           port=8797, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 10
    while not server.started and time.time() < deadline:
        time.sleep(0.1)
    assert server.started, "uvicorn did not start"

    try:
        with client.connect("ws://127.0.0.1:8797/ws", open_timeout=5) as socket:
            first = json.loads(socket.recv(timeout=5))
            second = json.loads(socket.recv(timeout=5))
    finally:
        server.should_exit = True

    assert first["type"] == "config", first
    assert second["type"] == "state", second
    assert "features" in second, second


def test_http_routes_answer():
    """GETs kept working while the socket was broken, so they prove nothing
    on their own -- but a 500 here would still be worth catching."""
    from fastapi.testclient import TestClient

    with TestClient(build_app(_engine())) as client:
        assert client.get("/api/config").status_code == 200
        assert client.get("/api/state").status_code == 200


# --- engine behaviour -------------------------------------------------------


def test_state_is_json_serialisable():
    """numpy scalars survive round-tripping; send_json would fail on them."""
    json.dumps(_engine().snapshot())
    json.dumps(_engine().config())


def test_level_is_anchored_to_the_profile():
    """0.0 must mean the calibrated empty room and 1.0 one person in it."""
    engine = _engine()
    empty = np.asarray(engine.profile.features["empty"]["median"])
    one = np.asarray(engine.profile.features["one_person"]["median"])
    i = engine.idx["effective_rank"]
    assert abs(float(engine.profile.normalise(empty)[i])) < 1e-6
    assert abs(float(engine.profile.normalise(one)[i]) - 1.0) < 1e-6


def test_no_level_without_a_gain_anchor():
    """An empty-only profile has no scale, so no level may be invented."""
    import count_features as cf

    rng = np.random.default_rng(5)
    base = 10 + 5 * np.abs(np.sin(np.linspace(0, 3, 48)))
    windows = np.stack([np.tile(base, (400, 1)) * (1 + rng.normal(0, 0.01, (400, 48)))
                        for _ in range(5)])
    cap = {
        "raw": np.concatenate(list(windows)), "windows": windows,
        "features": cf.features_for_windows(windows, 100.0),
        "motion_scores": np.array([0.05] * len(windows)),
        "mac": "AA", "layout": "fft", "rssi_mean": -55.0,
        "stats": {"rate_hz": 100.0, "jitter_ratio": 1.0, "target_hz": 100.0,
                  "gap_fraction": 0.0, "mac_fraction": 1.0, "windows": len(windows),
                  "windows_dropped_to_gaps": 0},
    }
    profile = rp.build_profile("noanchor", cap, None)
    path = _fixture("walking")
    args = parse_args(["--replay", str(path), "--window-seconds", "2.0", "--no-log"])
    engine = RankEngine(_loaded_source(path, 64), profile, args)
    engine.step()
    assert engine.snapshot()["level"] is None
    assert engine.config()["has_gain_anchor"] is False


def test_window_is_a_fixed_duration_not_a_frame_count():
    """A frame-count window swings with the packet rate; a duration does not.

    Measured, 400 frames spanned 4.2 s to 8.2 s as the rate fell from 94 Hz to
    56 Hz, so the live and offline numbers were computed over different spans.
    """
    engine = _engine()
    snap = engine.snapshot()
    assert engine.args.window_seconds == 2.0
    if snap.get("csi_span_s") is not None:
        assert abs(snap["csi_span_s"] - 2.0) < 0.35, snap["csi_span_s"]


def test_event_detector_is_the_default_decision():
    """Levels were measured below chance; events must be what ships."""
    engine = _engine()
    assert engine.args.decide == "events"
    assert engine.config()["decide"] == "events"


def test_reported_rate_is_sustained_not_burst():
    """The dashboard showed 230 Hz on a ~105 Hz link.

    FrameSource.rate() medians only the last 64 inter-arrivals, and frames
    arrive in bursts, so it roughly doubles the sustained figure.
    """
    engine = _engine()
    snap = engine.snapshot()
    assert "rate_burst_hz" in snap, "burst rate must stay available, just not headline"
    # Recomputed a moment later, so compare with a tolerance rather than for
    # equality -- elapsed time moves between the two reads.
    expected = (max(engine.source.total - engine.frames_at_start, 0)
                / max(time.time() - engine.started, 1e-9))
    assert abs(snap["rate_hz"] - expected) / max(expected, 1e-9) < 0.25, (
        snap["rate_hz"], expected)
    assert snap["rate_hz"] >= 0


def test_tracked_features_exist_in_the_feature_vector():
    import count_features as cf

    names = cf.window_feature_names()
    from rank_server import TRACKED, VALIDATED

    for name in TRACKED:
        assert name in names or name == "motion_score", name
    assert "motion_score" not in VALIDATED, "motion_score must never be validated"


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
