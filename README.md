# WiFi CSI sensing on a single ESP32

Detecting people in a room from WiFi Channel State Information, using one
commodity ESP32 rather than research NIC hardware.

**What it does today: it detects *motion*. It does not detect *presence*, and it
does not count people.** That is a measured result, not a work-in-progress
caveat — see [Findings](#findings). Everything here is scoped to what the
hardware demonstrably supports.

---

## Quick start

Three commands, assuming the board is flashed and plugged in.

```bash
cd ESP32-CSI-Tool

# 1. Is the link healthy? Want a transmitter above -75 dBm at ~95 Hz.
csi_env/bin/python har/run_pipeline.py --serial --duration 15

# 2. Calibrate the room (two prompts: leave the room, then walk a loop)
csi_env/bin/python har/calibrate_room.py --room room1 --mac <TRANSMITTER_MAC>

# 3. Start the live dashboard
csi_env/bin/python har/dashboard/rank_server.py --room room1 --mac <TRANSMITTER_MAC>
```

Then open **http://127.0.0.1:8001**.

### Starting the host in detail

```bash
cd ESP32-CSI-Tool
csi_env/bin/python har/dashboard/rank_server.py \
    --room room1 \
    --mac A8:6E:84:93:EE:60 \
    --http-port 8001
```

| Flag | Default | Notes |
|---|---|---|
| `--room` | `room1` | which calibration profile to score against |
| `--mac` | busiest | **pin this.** Otherwise it locks onto whichever AP is chattiest, which is routinely a distant one 35 dB down |
| `--port` | `/dev/cu.usbserial-5B530174971` | serial device |
| `--baud` | `460800` | must match the firmware |
| `--http-port` | `8001` | `8000` is the older activity dashboard |
| `--decide` | `events` | `events` = motion contrast + timeout. `level` exists only for comparison and scores **below chance** |
| `--host` | `127.0.0.1` | `0.0.0.0` exposes an unauthenticated CSI stream to the network |

**It needs ~90 seconds before the readout means anything** — the event
detector's long window is 45 s and it will not decide until that fills.

Every session writes `logs/<timestamp>-rank-session.csv`, one row per 0.5 s.
`Ctrl-C` stops it and writes a markdown summary alongside.

**If the page says "disconnected" or shows nothing**, the link is the usual
cause. Run step 1 above: the room's AP leaves channel 6 for long stretches, and
`select_mac` will silently fall back to a −88 dBm one. Waiting is the only fix
in passive mode.

### Without hardware

```bash
csi_env/bin/python har/dashboard/rank_server.py --room room1 \
    --replay har/recordings/room1/one_person/<capture>.csi.txt
```

---

## Hardware

| | |
|---|---|
| Board | ESP32-WROOM-32 (ESP32-D0WD-V3), single antenna, 2.4 GHz only |
| Firmware | ESP32-CSI-Tool `passive` build, channel 6, LLTF-only, **460800 baud** |
| Payload | 128 int8 values → 64 subcarriers → 48 data bins, amplitude only |
| Rate | ~95 CSI frames/s |

Flashing:

```bash
source ./esp-idf/export.sh
cd ESP32-CSI-Tool/passive && idf.py -p <PORT> flash
```

`sdkconfig.defaults` carries two settings that are not optional. Without
`CONFIG_ESP32_WIFI_CSI_ENABLED=y` the firmware aborts at `csi_component.h:96`
and boot-loops; without `CONFIG_ESP_CONSOLE_UART_CUSTOM=y` the baud silently
pins to 115200.

---

## Findings

Measured over 43 labelled captures in one room. Full evidence in
[`docs/DATA.md`](docs/DATA.md).

| Capability | Status | Evidence |
|---|---|---|
| Detecting motion | **works** | run-in detected in 2 s, event score z = +18.0 vs background p90 1.09 |
| Detecting a still occupant | **does not work** | empty vs asleep AUC **0.51** |
| Counting people | **does not work** | two standing still read **0.46**; one sitting reads 0.94 |
| Absolute level thresholds | **do not work** | 76.3% false positives; below chance |
| Event detection + timeout | **best available** | 77.3% balanced, 0% false positives |
| Presence model | **limited** | 75.5% balanced leave-one-session-out over 29 sessions |

Two results shape everything else:

**Rank features survive; magnitude features do not.** `motion_score` separates
empty from occupied at AUC 0.998 — and separates two *empty* rooms at 0.986. It
detects the session, not the person. Only `effective_rank`, `n_eig` and
`eig_ratio_3` clear a cross-session null control.

**Calibration goes stale in about 30 minutes.** The empty-room baseline drifts
+0.578 in 70 minutes, larger than the whole occupancy signal, because the room's
multipath changes on its own (subcarriers drift independently, r = −0.07).

---

## Repository layout

```
ESP32-CSI-Tool/har/          analysis, models, live dashboards
  csi_pipeline.py            parsing, layout detection, filters, resampling
  count_features.py          rank features (effective_rank, n_eig, eig_ratio_3)
  room_profile.py            calibration: capture, load, RoomProfile
  presence_events.py         drift-immune motion-event detector
  dashboard/rank_server.py   the live host
  rooms/room1/profile.json   the calibration -- tracked
  models/                    trained model -- tracked, carries its provenance
ESP32-CSI-Tool/passive/      firmware
docs/                        all documentation
esp-idf/                     vendored ESP-IDF v4.4.6
```

---

## Documentation

| Document | Read it for |
|---|---|
| [`docs/DATA.md`](docs/DATA.md) | Every recording, and what comparing them shows. **Start here.** |
| [`docs/METHODS.md`](docs/METHODS.md) | How anything here was established, and the mistake each technique answers |
| [`docs/CONTEXT.md`](docs/CONTEXT.md) | Project state: hardware, room, decisions, open items |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Every measurement with its reproduction command |
| [`docs/DATASET.md`](docs/DATASET.md) | Capture manifest, packing, and how to share 338 MB |
| [`docs/DEMO.md`](docs/DEMO.md) | Runbook, including the recording protocol |

---

## Reproducing the results

The raw captures (338 MB) are not tracked; the calibration, session manifest and
model are. Every reported number rebuilds from what is committed:

```bash
cd ESP32-CSI-Tool
csi_env/bin/python har/gate1_check.py --room room1 --mac <MAC> \
    --null-empty har/recordings/room1/empty/<second-empty>.csi.txt
csi_env/bin/python har/train_presence.py --room room1 --mac <MAC>
csi_env/bin/python har/score_session.py logs/<session>.csv \
    --truth "06:10:00=empty,06:11:30=present"
```

Inspect the shipped model without retraining:

```python
import joblib
b = joblib.load("har/models/presence_room1.joblib")
b["leave_one_session_out"]   # {'balanced_accuracy': 75.5, ...}
b["sessions"]                # every session, label, window count, RSSI
```

Captures pack losslessly, 338 MB → 54 MB:

```bash
csi_env/bin/python har/pack_dataset.py --room room1 --verify
```

---

## Tests

```bash
cd ESP32-CSI-Tool
for t in test_csi_pipeline test_features test_count_features; do
    csi_env/bin/python har/$t.py
done
csi_env/bin/python har/dashboard/test_dashboard.py
csi_env/bin/python har/dashboard/test_rank_dashboard.py
```

93 tests: 37 pipeline, 11 features, 28 counting, 8 + 9 dashboard.

---

## Environments

Two virtualenvs, because PyTorch publishes no Python 3.14 wheels.

| venv | Python | For |
|---|---|---|
| `csi_env` | 3.14 | everything except the LSTM |
| `lstm_env` | 3.12 | `train_lstm.py`, and a superset of the above |

```bash
csi_env/bin/pip install -r ESP32-CSI-Tool/har/requirements.txt
csi_env/bin/pip install -r ESP32-CSI-Tool/har/dashboard/requirements.txt
```

---

## Source papers

- Abuhoureyah, Wong & Mohd Isira, *WiFi-based human activity recognition through
  wall using deep learning*, **EAAI 127 (2024) 107171** — the method. Its 97.5%
  used a 4-antenna MIMO adapter; this is one antenna, and its through-wall and
  cross-room results do not transfer.
- Meneghello et al., *A CSI Dataset for Wireless Human Sensing on 80 MHz Wi-Fi
  Channels*, **IEEE Communications Magazine, Sept 2023** — benchmark reference.
