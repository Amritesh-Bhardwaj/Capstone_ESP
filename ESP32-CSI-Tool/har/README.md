# CSI preprocessing for human activity recognition

Implements the preprocessing chain from **Abuhoureyah, Wong & Mohd Isira,
"WiFi-based human activity recognition through wall using deep learning",
Engineering Applications of Artificial Intelligence 127 (2024) 107171**
(Section 4.1.2), adapted to the ESP32-CSI-Tool serial format.

```
CSI_DATA line -> parse -> select transmitter -> drop null/pilot subcarriers
              -> Hampel filter -> Hamming smoothing -> wavelet denoise
              -> z-score -> sliding windows  (n, window_length, 48)
```

The paper uses **amplitude only**; phase is never used, and neither is this.

## Setup

```bash
csi_env/bin/pip install -r har/requirements.txt
```

**Presenting? Follow [`docs/DEMO.md`](../../docs/DEMO.md)** — record, train,
rehearse, and the on-site recalibration step.

Documentation lives in the repository-root `docs/` folder:

| Document | What it holds |
|---|---|
| [`docs/CONTEXT.md`](../../docs/CONTEXT.md) | Project state, hardware facts, decisions, open items |
| [`docs/RESULTS.md`](../../docs/RESULTS.md) | Every measurement taken, with reproduction commands |
| [`docs/DEMO.md`](../../docs/DEMO.md) | Presentation runbook |

## Files

| File | Purpose |
|---|---|
| `csi_pipeline.py` | Parsing, layout detection, the paper's filter chain |
| `features.py` | Window features + the calibrated `PresenceDetector` |
| `record_csi.py` | Capture labelled takes from the board |
| `train_activity.py` | Train the random-forest classifier, leakage-free validation |
| `train_lstm.py` + `lstm_model.py` | The paper's LSTM (needs `lstm_env`) |
| `live_demo.py` | Live presence + activity GUI, with replay fallback |
| `dashboard/` | The same demo in a browser — FastAPI + WebSocket, imports its inference from `live_demo.py` ([README](dashboard/README.md)) |
| `run_pipeline.py` | Offline preprocessing + diagnostic figure |
| `make_synthetic.py` | Synthetic recordings for testing without hardware |

## Two environments

| venv | Python | For |
|---|---|---|
| `csi_env` | 3.14 | Everything except the LSTM |
| `lstm_env` | 3.12 | `train_lstm.py`, and `live_demo.py` with an LSTM model |

PyTorch publishes no 3.14 wheels, which is the only reason for the split.
`lstm_env` has the full dependency set, so it can run the whole stack if you
prefer one environment.

## Use

```bash
# Record a labelled take from the board
python har/record_csi.py --label walking --duration 30

# Preprocess a recording, with a before/after figure
python har/run_pipeline.py --file har/recordings/walking_*.csv --plot

# Preprocess a live 20-second capture
python har/run_pipeline.py --serial --duration 20 --plot

# Emit LSTM-ready tensors
python har/run_pipeline.py --file take.csv --save-npz take.npz
```

`--save-npz` writes `windows` `(n, window_length, 48)`, plus `processed`,
`raw`, `timestamps`, `layout` and `mac`.

From Python:

```python
from csi_pipeline import parse_lines, run_pipeline, PipelineConfig

frames, report = parse_lines(open("take.csv"))
result = run_pipeline(frames, PipelineConfig(window_length=64, window_stride=16))
result.windows          # (n, 64, 48) -> straight into an LSTM
```

## Tests

```bash
csi_env/bin/python har/test_csi_pipeline.py     # 31 tests, or use pytest
```

They run against two real captures: `testdata/live_lltf_fft.txt` (recorded from
the attached board, MACs replaced with placeholders) and the repo's own
`python_utils/example_csi.csv`.

## Three things this handles that a naive port does not

**1. The subcarrier layout is not fixed across builds.** The paper says to drop
null and pilot subcarriers, which requires knowing which buffer position is
which subcarrier. Two different orderings occur in practice:

| Layout | Buffer position 0 | Guard bins (always null) | Pilot bins |
|---|---|---|---|
| `fft` | subcarrier 0 (DC) | 27–37 | 7, 21, 43, 57 |
| `shifted` | subcarrier −32 | 1–5, 59–63 | 11, 25, 39, 53 |

The board currently attached emits `fft`; `python_utils/example_csi.csv` is
`shifted`. Hardcoding either one silently deletes 48 real subcarriers and keeps
the nulls. `detect_layout()` infers the ordering from where the guard bands sit,
scoring both hypotheses and refusing to guess when neither fits. Both layouts
resolve to exactly 48 data bins, which is the 802.11 20 MHz OFDM count
(48 data + 4 pilot + 12 null = 64) — the consistency check that confirms the
mapping.

**2. Serial rows get corrupted.** The ESP32 interleaves `task_wdt` log output
into the CSI stream, which lands mid-array. Roughly 1% of rows arrive mangled.
Every token is validated before use and bad rows are dropped and counted, rather
than raising or, worse, parsing partially.

Note also that the `len` header field cannot be trusted: with
`CONFIG_SHOULD_COLLECT_ONLY_LLTF` the firmware prints `data->len` (e.g. 384)
while emitting only 128 values. Parsing uses the actual token count and always
takes the first 128 bytes — the LLTF field, 64 subcarriers.

**3. A passive board sniffs every transmitter on the channel.** In the live
capture above, 471 frames came from two different MACs; interleaving them
produces a time series that jumps between unrelated radio channels, so every
temporal filter downstream is operating on garbage. `select_mac()` keeps one
transmitter (the most frequent, by default) before any filtering.

## Measured on the attached hardware

Live 20 s capture, ESP32 passive build, channel 6, 20 MHz, `sig_mode=0`:

| | |
|---|---|
| Frames parsed / rejected | 471 / 4 |
| Frames after MAC selection | 278 |
| Sample rate | 22.2 Hz, inter-arrival jitter 1.0× median |
| RSSI | −47.1 dBm |
| Layout detected | `fft` → 48 data bins |
| Hampel replacements | 5.2% of samples |

## Known gaps

- **Sample rate is ~22–24 Hz.** The paper's setup runs far faster, and the
  companion dataset paper (Meneghello et al., IEEE ComMag 2023) samples at
  ~173 packets/s. At 22 Hz a 64-frame window spans ~3 s, which is workable for
  coarse activities but leaves little headroom. Raising the transmit rate on the
  AP side is the highest-value change before any model training.
- **Arrival times are not uniform.** CSI is produced only when a packet arrives,
  so the series is unevenly sampled while the filters assume a uniform grid.
  `stats["jitter_ratio"]` reports the severity; resampling onto a uniform grid is
  not yet implemented.
- **No LSTM yet.** This is the preprocessing stage only. `result.windows` is
  shaped for the paper's classifier, but no model is trained or evaluated, and
  no accuracy is claimed.
- **Single ESP32, SISO, 20 MHz.** The paper's through-wall and wider-angle
  results come from a 4-antenna MIMO adapter. Those results do not transfer to
  this hardware, and the pipeline makes no attempt to claim they do.
