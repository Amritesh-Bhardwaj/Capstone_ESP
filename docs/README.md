# Documentation

WiFi CSI human activity recognition on a single ESP32.

| Document | Read it for |
|---|---|
| [`DATA.md`](DATA.md) | Every recording, and what comparing them shows. **Start here** — it is where the scope of the project is set. |
| [`METHODS.md`](METHODS.md) | How anything here was established: null controls, leave-one-session-out, and the mistake each technique answers. |
| [`CONTEXT.md`](CONTEXT.md) | Where the project stands — hardware, room, environments, decisions, open items. |
| [`RESULTS.md`](RESULTS.md) | Every measurement taken, with the command to reproduce it, and which numbers are safe to quote. |
| [`DATASET.md`](DATASET.md) | Capture manifest with checksums, lossless packing, and how to share 338 MB. |
| [`DEMO.md`](DEMO.md) | Runbook: the recording protocol, calibration, and the live dashboard. |

Repository overview and how to start the live host: [`../README.md`](../README.md).

Code and its own usage guide: [`ESP32-CSI-Tool/har/`](../ESP32-CSI-Tool/har/README.md).

All commands in these documents are run from `ESP32-CSI-Tool/`.

## Source papers

- Abuhoureyah, Wong & Mohd Isira, *WiFi-based human activity recognition
  through wall using deep learning*, Engineering Applications of Artificial
  Intelligence 127 (2024) 107171 — the method being implemented.
- Meneghello, Dal Fabbro, Garlisi, Tinnirello & Rossi, *A CSI Dataset for
  Wireless Human Sensing on 80 MHz Wi-Fi Channels*, IEEE Communications
  Magazine, September 2023 — dataset and benchmark reference.
