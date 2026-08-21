# Browser dashboard

The same demo as [`har/live_demo.py`](../live_demo.py), rendered in a browser
instead of a matplotlib window. Useful when the projector is easier to drive
from a browser tab, when someone wants to watch on a second machine, or when
you want the readout on screen while a terminal runs beside it.

**It does not reimplement any inference.** `server.py` imports `FrameSource`,
`bootstrap`, `preprocess`, `calibrate_live`, the serial/replay readers and
`PresenceDetector` from the existing code, so the number on the page is the
number `live_demo.py` would show. If the two ever disagree, that is a bug and
`test_dashboard.py` is where it gets pinned.

## Setup

```bash
csi_env/bin/pip install -r har/dashboard/requirements.txt
```

## Use

All commands run from `ESP32-CSI-Tool/`, like the rest of the project.

```bash
# Rehearse with no hardware (replay loops forever)
csi_env/bin/python har/dashboard/server.py --model har/model.joblib \
  --replay har/recordings-synthetic/walking_synth0.csv

# Live off the board (note the baud rate -- see docs/CONTEXT.md)
csi_env/bin/python har/dashboard/server.py --model har/model.joblib --baud 460800

# Presence only, no trained model: 15 s empty-room calibration first
csi_env/bin/python har/dashboard/server.py --calibrate-only --baud 460800

# Fitted thresholds from har/monitor.py, skipping calibration entirely
csi_env/bin/python har/dashboard/server.py --model har/model.joblib \
  --baud 460800 --enter-threshold 0.043
```

Then open <http://127.0.0.1:8000>. Every flag `live_demo.py` takes for the
source, the model and the detector means the same thing here; `--http-port`,
`--host` and `--interval` are the only additions.

Calibration happens in the terminal *before* the server starts listening, so
watch the console for the `>>> CALIBRATING: leave the area empty <<<` prompt.

## What the page shows

| Panel | Source |
|---|---|
| PERSON PRESENT / NO ONE DETECTED | `PresenceDetector.update`, with its hysteresis and debounce |
| activity + confidence bars | the loaded `model.joblib`, only while present — same as the GUI |
| Motion energy trace | the detector's smoothed score, with the entry threshold and empty-room baseline drawn in |
| Live CSI | the preprocessed window, 48 data subcarriers × 64 frames |
| SIGNAL GAP | the dropout guard: a window spanning more than `--max-span-factor` × its expected duration holds the last state instead of reporting motion that is really a gap |

## How it is wired

```
serial / replay -> FrameSource -> Engine thread (preprocess, detect, classify)
                                        |
                            state dict --+--> GET /api/state
                                         +--> WS  /ws   (push every --interval)
                                         '--> GET /api/config (labels, thresholds)
```

The page prefers the WebSocket and falls back to polling `/api/state` twice a
second if the socket cannot be established. The waterfall is quantised to
uint8 against the window's 2nd–98th percentile and sent as base64 — 4 KB per
frame instead of ~60 KB of JSON floats.

## Serving to other machines

`--host 0.0.0.0` binds all interfaces and there is **no authentication**:
anyone on the venue network can then watch the CSI stream. Prefer localhost,
or an SSH tunnel, unless the demo genuinely needs a second device.

## Tests

```bash
csi_env/bin/python har/dashboard/test_dashboard.py    # 8 tests, or use pytest
```

They run against the synthetic recordings, so no board is needed. The
WebSocket test starts a real uvicorn server on port 8799 and reads two frames
off it — that one exists because a lazily imported `WebSocket` annotation made
FastAPI reject every handshake with a 403 while every plain GET still worked.
