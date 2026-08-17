# Context ledger

State of the WiFi-CSI human activity recognition capstone. Facts here are
measured or read from source, not assumed. Last updated 2026-08-16.

See [`RESULTS.md`](RESULTS.md) for every measurement, [`DEMO.md`](DEMO.md) for
the presentation runbook, and
[`ESP32-CSI-Tool/har/README.md`](../ESP32-CSI-Tool/har/README.md) for how the
code works.

Code lives in `ESP32-CSI-Tool/har/`. Every command in these documents is run
from `ESP32-CSI-Tool/`.

---

## 1. Goal

Detect human presence and classify coarse activity from WiFi Channel State
Information, using a single low-cost ESP32 rather than research NIC hardware.

## 2. Hardware

| | |
|---|---|
| Board | **ESP32-WROOM-32**, module marking `211 161007` — dual-core D0WDxx, single antenna, 2.4 GHz only |
| Bridge | CH9102/CH343 (VID `0x1a86`, PID `0x55d4`), draws 136 mA |
| Port | `/dev/cu.usbserial-57460201261` |
| **Current state** | **WORKING.** Reflashed with cut-down firmware: **460800 baud**, 3-field header, **~100 Hz**. An earlier "hardware fault" diagnosis was wrong — see `RESULTS.md` §8b. |
| Serial | **All tools need `--baud 460800`.** No `--mac` needed (single implicit transmitter). |
| Firmware | **Cut-down build**, format `CSI_DATA,<len>,<rssi>,[...]`. Previously the ESP32-CSI-Tool `passive` build (25 fields, `role=PASSIVE`, 115200 baud). `parse_line` handles both. |
| Payload | `len=128` → LLTF only, 64 subcarriers → 48 data bins |
| Rate | **~100 CSI frames/s** (was 22–24 on the previous firmware) |
| RSSI | ≈ −47 dBm |

Handled in software:

- ~0.2% of rows arrive malformed (log output interleaving into the array).
  Every token is validated and bad rows are counted, not silently parsed.
- The **previous** firmware sniffed every transmitter on the channel, so MAC
  selection was mandatory and `--mac` had to be pinned. The current minimal
  format reports one implicit transmitter, so this no longer applies — but the
  code path remains, since the older firmware may be reflashed.

## 3. Source papers

| Paper | Role in this project |
|---|---|
| Abuhoureyah, Wong & Mohd Isira, *WiFi-based HAR through wall using deep learning*, **Engineering Applications of Artificial Intelligence 127 (2024) 107171** | The method. §4.1.2 preprocessing chain and §4.1.4 LSTM are what we implement. |
| Meneghello, Dal Fabbro, Garlisi, Tinnirello & Rossi, *A CSI Dataset for Wireless Human Sensing on 80 MHz Wi-Fi Channels*, **IEEE Communications Magazine, Sept 2023** | Benchmark/dataset reference. Its authors also wrote SHARP (see §7). |

The EAAI paper reports 97.5% on 7 activities — using a **4-antenna MIMO ALFA
AWUS1900 adapter** on a Raspberry Pi with Nexmon. Its through-wall and
wider-angle results come from that antenna array and **do not transfer to a
single SISO ESP32.** Its own limitations section states the model is
location-bound.

## 4. Environments

Two virtual environments. The split exists only because **PyTorch publishes no
Python 3.14 wheels**.

| venv | Python | Purpose |
|---|---|---|
| `csi_env` | 3.14.2 | Everything except the LSTM |
| `lstm_env` | 3.12.7 | `train_lstm.py`, `test_pretrained_transfer.py`, and `live_demo.py` with an LSTM model |

`lstm_env` is a superset and can run the whole stack.

| Package | csi_env | lstm_env |
|---|---|---|
| numpy | 2.4.2 | 2.5.2 |
| scipy | 1.18.0 | 1.18.0 |
| PyWavelets | 1.9.0 | 1.9.0 |
| scikit-learn | 1.9.0 | 1.9.0 |
| joblib | 1.5.3 | 1.5.3 |
| pyserial | 3.5 | 3.5 |
| matplotlib | 3.10.8 | 3.11.1 |
| torch | — | 2.13.0 (**MPS available**) |
| tensorflow | — | 2.21.0 |
| keras | — | 3.15.1 |
| h5py | — | 3.14.0 |

## 5. The four hardware findings that drive the code

**5.1 — Subcarrier layout is not fixed across builds.** Two orderings of the 64
LLTF bins occur in practice:

| Layout | Position 0 | Guard bins (null) | Pilot bins | Seen in |
|---|---|---|---|---|
| `fft` | subcarrier 0 (DC) | 27–37 | 7, 21, 43, 57 | the attached board |
| `shifted` | subcarrier −32 | 1–5, 59–63 | 11, 25, 39, 53 | `python_utils/example_csi.csv` |

Both resolve to exactly **48 data bins**, matching 802.11 20 MHz OFDM
(48 data + 4 pilot + 12 null = 64) — the check that confirms the mapping.
Hardcoding either one deletes 48 real subcarriers and keeps the nulls, so
`detect_layout()` infers it from where the guard bands sit. It scores both
hypotheses by the *fraction* of guard bins that are null and requires a margin,
because the outermost guard bins leak on some hardware.

**5.2 — The `len` header field cannot be trusted.** With
`CONFIG_SHOULD_COLLECT_ONLY_LLTF` the firmware prints `data->len` (e.g. 384)
while emitting only 128 values. Parsing uses the actual token count.

**5.3 — MAC filtering is mandatory when the format carries a MAC.** On the
previous firmware a 20 s capture parsed 471 frames but only 278 came from the
dominant transmitter. Interleaving links produces a series that jumps between
unrelated radio channels, so every temporal filter downstream would run on
garbage.

**5.4 — Neither the baud rate nor the header format is fixed.** The board has
run at 115200 with 25 header fields and at 460800 with 3. `parse_line`
dispatches on field count; missing device timestamps fall back to wall-clock
arrival. Assuming either format — or either baud — silently yields nothing.
This cost a wrong "hardware fault" diagnosis; see `RESULTS.md` §8b.

## 6. Pipeline

```
CSI_DATA line → parse (validate tokens) → select one MAC
              → detect layout → drop null/pilot (64→48)
              → Hampel (MAD, 3σ) → Hamming FIR → wavelet denoise (db4)
              → [normalise] → sliding windows
```

Amplitude only; phase is never used, matching the paper. ESP-IDF packs each
subcarrier as `(imag, real)` — confirmed against the `CSI_PHASE` branch of
`_components/csi_component.h`.

**Normalisation is deliberately off for the demo path.** Per-subcarrier
z-scoring destroys the variance magnitude that separates a still person from a
walking one. `features.py` and `lstm_model.py` each apply their own
gain-invariant normalisation instead.

## 7. Search for a pretrained model — closed

Three sources were investigated to avoid recording data. **None yielded a
usable pretrained model.** Full evidence in `RESULTS.md` §4–6.

| Source | Verdict |
|---|---|
| `ruvnet/wifi-densepose-pretrained` (HF) | Not an LSTM — a 9,280-param MLP taking an 8-dim feature vector. No activity head. Presence head mathematically cannot output "absent". |
| `Retsediv/WIFI_CSI_based_HAR` | No weights shipped. Needs 468 input dims (114 sc × 4 antenna pairs × amplitude+phase), Atheros hardware. |
| `NTUMARS/Awesome-WiFi-CSI-Sensing` (27 repos) | Only SHARP and SHARPax ship HAR weights. Both tested; neither can distinguish our CSI from Gaussian noise. |

**Conclusion: there is no shortcut around recording labelled data.** This was
checked exhaustively, not assumed.

## 8. What did work: ESP-Fi HAR

`AutoSmartGroup/ESP-Fi-HAR` — the one public dataset our pipeline consumes
directly. Commodity ESP32, amplitude only, `CSIamp` of shape (950, 52),
7 activities, 4 environments, 560 `.mat` files, session-disjoint official split.

- Its 52 subcarriers are 48 data + 4 pilot; our recordings drop pilots to 48.
- **The dataset documents no sample rate** anywhere. Band-power features use a
  nominal 100 Hz — internally consistent, not physically calibrated. Every
  other feature is rate-free. State this when quoting the number.
- Repo needs git-lfs for `ESP-Fi_dataset.rar`, but the `.mat` files are plain
  blobs and were extracted with `git show` (212 MB). git-lfs is **not**
  installed on this machine.

## 9. Decisions and why

| Decision | Reason |
|---|---|
| Random forest on window features is the **demo** model, not the LSTM | Head-to-head on the published ESP-Fi dataset: RF 90.0% vs LSTM 65.0% per-file (4 classes, session-disjoint). The gap holds at 3× the data and after 150 epochs. |
| LSTM kept anyway | The capstone follows an LSTM paper, and the measured RF-vs-LSTM gap is itself a finding — see `RESULTS.md` §7.3. |
| Scope live demo to 3 classes (`empty`/`sitting`/`walking`) | ESP-Fi confirms the ceiling: gross motion 80–100% recall, postural/small motion 20–40%. |
| Presence via calibrated threshold, not a model | Needs no training and recalibrates in 15 s, so it survives a room change. Activity classification does not. |
| Split by **take**, never by window | Overlapping windows leak. Demonstrated: 100% leaky vs 48.7% honest on deliberately meaningless labels. |

## 10. Files

| File | Purpose |
|---|---|
| `csi_pipeline.py` | Parsing, layout detection, filter chain |
| `features.py` | 19 scale-invariant window features + `PresenceDetector` |
| `lstm_model.py` | Paper's LSTM (needs `lstm_env`) |
| `record_csi.py` | Capture labelled takes |
| `train_activity.py` | Random-forest classifier + leakage-free validation |
| `train_lstm.py` | LSTM equivalent |
| `train_espfi.py` | Benchmark on the ESP-Fi HAR dataset |
| `test_pretrained_transfer.py` | SHARP/SHARPax transfer test with noise control |
| `test_csi_pipeline.py` | 31 tests — **31/31 pass in both venvs** |
| `run_pipeline.py` | Offline preprocessing + diagnostic figure |
| `live_demo.py` | Live presence + activity GUI, replay fallback |
| `make_synthetic.py` | Synthetic recordings for hardware-free testing |
| `testdata/live_lltf_fft.txt` | Real capture, MACs anonymised, used by tests |

## 11. Open items

1. **Board works**; remember `--baud 460800` on every tool.
2. **No real labelled recordings exist yet.** `har/recordings/` is empty;
   `DEMO.md` Step 1 is still the blocker for activity classification.
3. **Sample rate is now ~100 Hz** (was 22 Hz). No packet-rate workaround
   needed. Nothing has yet been trained on data at this rate.
4. **Arrival times are non-uniform.** Filters assume a uniform grid;
   `stats["jitter_ratio"]` reports severity. Resampling not implemented.
5. **Activity models do not transfer between rooms.** Re-record and retrain
   on site — `DEMO.md` Step 4.
6. ESP-Fi benchmark uses a nominal sample rate (§8).

## 12. Claims discipline

Quote only these:

- Leave-one-take-out or session-disjoint accuracy, always with the
  majority-class baseline beside it.
- ESP-Fi HAR gross motion (`walk`/`turn`/`fall`/`run`): **90.0% per-file /
  66.7% per-window**, session-disjoint, baseline 25.0%. ← headline
- ESP-Fi HAR all 7 activities: **64.3% per-file / 41.7% per-window**,
  baseline 14.3%.
- Presence detection: calibrated threshold, recalibrates in 15 s.

State the class subset whenever quoting 90.0%. Reporting it as "our system is
90% accurate" without saying it is four gross-motion classes is the kind of
claim this project has spent its effort avoiding.

Never quote:

- Any random-window-split accuracy.
- Any accuracy from `make_synthetic.py` — those classes are separable by
  construction; it is a self-test, not a result.
- Through-wall performance. One SISO antenna; the paper's result used 4.
- Cross-room generalisation. It does not hold.
