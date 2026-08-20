# Context ledger

State of the WiFi-CSI human activity recognition capstone. Facts here are
measured or read from source, not assumed. Last updated 2026-08-20.

See [`DATA.md`](DATA.md) for what every recording is and what comparing them
shows, [`METHODS.md`](METHODS.md) for how anything here was established,
[`DATASET.md`](DATASET.md) for the capture manifest and how to rebuild the
derived artefacts, [`RESULTS.md`](RESULTS.md) for every measurement,
[`DEMO.md`](DEMO.md) for the presentation runbook, and
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
| Board | **ESP32-WROOM-32**, chip **ESP32-D0WD-V3 rev v3.1** — dual core, 240 MHz, 40 MHz crystal, 4 MB flash, single antenna, 2.4 GHz only. MAC `1c:c3:ab:d2:29:f8`. Read with `esptool chip-id`, not assumed. |
| Bridge | CH9102/CH343 (VID `0x1a86`, PID `0x55d4`), draws 136 mA |
| Port | `/dev/cu.usbserial-5B530174971` |
| **Current state** | **WORKING.** This is a **different physical board** from the one used through 2026-08-16 (that one was on `usbserial-57460201261`). Same module type, same CSI capability — see `RESULTS.md` §8c. |
| Serial | **All tools need `--baud 460800`.** **`--mac` is required again** — the format carries a MAC and the board sniffs every transmitter on the channel (§5.3). |
| Firmware | ESP32-CSI-Tool **`passive`** build, rebuilt 2026-08-20 at 460800 baud, channel 6, LLTF-only. Format `CSI_DATA,PASSIVE,<mac>,<rssi>,...` — 25 header fields. The cut-down 3-field build (`CSI_DATA,<len>,<rssi>`) is no longer on the board; `parse_line` still handles both. |
| Payload | 128 values → LLTF only, 64 subcarriers → 48 data bins |
| Rate | **~92–95 CSI frames/s** measured over 20 s |
| RSSI | ≈ −63 dBm (vs −47 dBm on the previous board; different placement) |

Handled in software:

- ~0.2% of rows arrive malformed (log output interleaving into the array).
  Every token is validated for **type and int8 range** and bad rows are counted,
  not silently parsed. The range half was missing until 2026-08-20: tokens in
  128..32767 were silently accepted into the int16 buffer and corrupted the
  amplitude without error, and tokens above 32767 crashed the parser outright
  (`RESULTS.md` §9.10). Measured contamination in existing recordings: 4 frames
  in ~111,000 rows.
- The current `passive` firmware sniffs **every** transmitter on the channel, so
  MAC selection is mandatory and `--mac` must be pinned. A 20 s capture on
  2026-08-20 parsed 1908 frames but kept only 1270 after MAC selection — a third
  came from other links. (The short-lived cut-down build reported one implicit
  transmitter and needed no `--mac`; `parse_line` still supports it.)

## 2b. Rooms

A room profile is only valid for one physical arrangement. **Tape the ESP32
down.** Moving either end by more than about 30 cm invalidates the calibration
and the profile must be rebuilt.

### Room 1

Hostel room. The room itself is small.

| | |
|---|---|
| Size | Small hostel room, much less than 19 ft long |
| Composition | 10-inch concrete walls on 2 sides, a wooden door on one side, and a glass wall facing outside on another |
| Router | TP-Link EAP620 HD (Ceiling/Wall mount AP). Mounted in the top corner. |
| ESP32 | Positioned in the opposite bottom corner, creating a diagonal link. |
| Channel | 6 (2437 MHz), lambda = 12.3 cm |

**Placement is fixed:** The router is in the top corner and the ESP32 is in the bottom corner, creating a cross-room diagonal link.

Why this matters for sensing:
- The sensing region is the Fresnel ellipsoid around the router-to-ESP32 line.
- A diagonal placement across the room sweeps the maximum possible volume, ensuring good coverage. The concrete and glass walls will also contribute strong multipath reflections, which helps in detecting human presence anywhere in the room.

Mount both ends near torso height (~1-1.2 m), off the floor, off the wall, away
from large metal. **Tape the board down** — moving either end by more than
about 30 cm invalidates every profile built here.

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
| `csi_pipeline.py` | Parsing, layout detection, filter chain, `resample_uniform` |
| `features.py` | 19 scale-invariant window features + `PresenceDetector` |
| `count_features.py` | Counting features: effective rank, dilated PEM, Doppler |
| `room_profile.py` | Room calibration: capture, load, `RoomProfile` |
| `calibrate_room.py` | The 2-minute on-site calibration |
| `gate1_check.py` | Gate 1: do the counting features respond to one occupant? |
| `live_rank.py` | Matplotlib live view of the validated features (logs to `logs/`) |
| `dashboard/rank_server.py` | Browser dashboard on the validated features (logs to `logs/`) |
| `dashboard/static/rank.html` | Its page: presence level, feature table, signal visualiser |
| `dashboard/test_rank_dashboard.py` | 7 tests, incl. the WebSocket 403 regression |
| `presence_events.py` | Drift-immune motion-event detector + occupancy timeout |
| `score_session.py` | Score a session log against hand-recorded ground truth |
| `train_presence.py` | Presence model, leave-one-**session**-out evaluation |
| `drift_study.py` | Long controlled study of what makes the baseline drift |
| `lstm_model.py` | Paper's LSTM (needs `lstm_env`) |
| `record_csi.py` | Capture labelled takes |
| `train_activity.py` | Random-forest classifier + leakage-free validation |
| `train_lstm.py` | LSTM equivalent |
| `train_espfi.py` | Benchmark on the ESP-Fi HAR dataset |
| `test_pretrained_transfer.py` | SHARP/SHARPax transfer test with noise control |
| `test_csi_pipeline.py` | 37 tests — **37/37 pass in both venvs** |
| `test_count_features.py` | 28 tests for resampling and counting features |
| `run_pipeline.py` | Offline preprocessing + diagnostic figure |
| `live_demo.py` | Live presence + activity GUI, replay fallback |
| `make_synthetic.py` | Synthetic recordings for hardware-free testing |
| `testdata/live_lltf_fft.txt` | Real capture, MACs anonymised, used by tests |

## 11. Open items

0. **The live dashboard does not resample.** It takes 400 *frames*, so its
   window ranges 4.2-8.2 s while the offline path is fixed at 4.0 s; the two
   compute the same feature over different spans. Not the cause of the drift
   (`RESULTS.md` §9.13) but still wrong. Fix once the drift study releases the
   serial port.
1. **Board works** (new unit, reflashed 2026-08-20). Remember `--baud 460800`
   **and `--mac`** on every tool — the `passive` format carries a MAC again.
2. **No real labelled recordings exist yet.** `har/recordings/` is empty;
   `DEMO.md` Step 1 is still the blocker for activity classification.
3. **Sample rate is ~92–95 Hz.** No packet-rate workaround needed. Nothing has
   yet been trained on data at this rate.
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

---

## 13. People counting and room calibration (started 2026-08-20)

### What is being attempted

Estimate room occupancy as **0 / 1 / 2 / 3+**, and adapt to a new room from a
short on-site calibration rather than a full re-recording.

Counting is **not** a headcount. A SISO link measures how much *independent
motion* is present, and every feature saturates as occupants are added. The
literature ceiling of ~5 people is with multi-antenna NICs; on one antenna,
0/1/2/3+ is the honest scope. Anything past three stops meaning a number.

Still people are invisible over a few seconds. The fix is timescale, not
features: **presence on 1-2 s windows, count on a 60 s rolling aggregate.**
Over a minute even sedentary people shift and fidget. That aggregation is what
makes sedentary occupants countable at all.

### The calibration, and the honest limit of a 60 s baseline

| Phase | Duration | Yields |
|---|---|---|
| A — empty room | 60 s | offset `a`, per-subcarrier noise floor and reference scale, deep-fade mask, presence threshold |
| B — one person walking | 60 s | gain anchor `b` |

**An empty room alone cannot calibrate counting.** It fixes the zero point but
says nothing about scale, and the scale depends on room size, link geometry,
and how far occupants stand off the line of sight. Empty-only calibration is
valid for presence and for *relative* occupancy (empty / light / busy); it is
not valid for a count, and `RoomProfile.gain()` returns zeros so callers refuse
rather than reporting nonsense.

The count curve is `f(N) = a + b * (1 - exp(-N/k))`. `a` and `b` come from the
two phases on site; **`k` is the transferable part**, fitted once offline
across rooms. Learning the shape offline and calibrating only offset and gain
is what makes two minutes enough.

### Two divergences from the activity path

**The filter chain is not used for counting.** `run_pipeline` applies Hampel,
then a 9-tap Hamming FIR, then a db4 wavelet denoise. The last two are
low-pass — at 100 Hz the Hamming is roughly -3 dB by 11 Hz, the bottom of the
walking-Doppler band. Running counting through it would delete the feature it
needs most. The counting path keeps Hampel and stops there.

**`features.BANDS` is deliberately left capping at 11 Hz.** It was written for
the 22 Hz era, and ~30% of measured spectral power now sits above it. Extending
it would change the activity feature vector and silently invalidate the ESP-Fi
90.0% headline, so the wider bands live in `count_features.COUNT_BANDS`
instead. Revisiting `BANDS` for activity is a separate experiment that owes its
own re-measurement.

### Measured 2026-08-20, and what it forces

- **No presence or counting result yet.** The first attempt analysed a capture
  labelled "empty" that had a person sitting in it throughout; those conclusions
  are retracted in `RESULTS.md` 9.3-9.5. Scene labels must be confirmed with the
  person in the room, never inferred from an intention to leave.
- **The room AP leaves channel 6 for long stretches.** `EE:60` was at -60 dBm
  around 01:00, gone by 01:41, and back at **-53.2 dBm** by 04:41. While it was
  away, `select_mac` silently switched to a distant AP 35 dB down and the
  pipeline carried on producing plausible-looking rows from noise.
- **The ping control was wrong here and is retracted.** The Mac associates on
  **channel 153 (5 GHz)**, so its traffic never reached the board's 2.4 GHz
  channel 6 — yet the rate stayed at 90-97 Hz throughout, because the CSI stream
  is fed by the AP's own basic-rate traffic. Every captured frame is legacy
  non-HT 6 Mbps with **zero PHY variation**, despite the AP being an 802.11ax
  2x2 MIMO TP-Link EAP620 HD.
- **The board stops emitting after ~20 minutes and needs an EN reset.** Silent:
  no CSI, no log lines, zero bytes.
- **The diagonal placement is confirmed good**: -53.2 dBm at 94-97 Hz, better
  than the -59 to -63 dBm measured at the earlier shorter link. Settled.

Every one of those failures looks identical in the output: a quiet room. Hence
the guards — `calibrate_room` now aborts below -80 dBm and below 70% coverage,
`build_profile` refuses two phases from different transmitters, `gate1_check`
warns on a weak link, and long captures reset the board first.

### Confounds: one downgraded by measurement, one real

**Packet rate — largely ruled out, measured.** The concern was that more people
means more phones means more packets, so a counter could learn traffic volume
instead of CSI. Measured across four sessions the rate sits at **90-97 Hz
regardless of which AP dominates, its RSSI, or occupancy**, because every
captured frame is AP-side basic-rate traffic: `sig_mode`, `mcs`, `bandwidth`,
`rate` and `ant` are **single-valued** over 24,294 frames (`RESULTS.md` §9.5f).
The prescribed fixed-rate ping was therefore neither necessary nor sufficient
and has been removed from `DEMO.md`. If `rate` ever stops being single-valued,
client data traffic has started reaching the CSI stream and the concern is back.

**Link quality — the real precondition.** The room AP leaves channel 6 for long
stretches, and `select_mac` picks the chattiest transmitter, not the strongest,
so the pipeline silently switches to a distant AP 35 dB down and keeps
producing plausible rows from noise. Confirm `EE:60` above -75 dBm before every
session; `calibrate_room` aborts below -80 dBm.

### Gates

| Gate | Question | Status |
|---|---|---|
| 1 | Do the counting features respond to one occupant, beyond the empty room's own drift? | **PASSED 2026-08-20 (Sitting Occupant). Features perfectly separated.** |
| 2 | One room, 0/1/2/3, leave-one-take-out above the 25% baseline | not started |
| 3 | 3 rooms, leave-one-**room**-out with calibration | not started |
| 4 | Live demo, 60 s decision window, 2-min on-site calibration | not started |

Gate 1 is a screen. Failing it kills counting; passing it only licenses
recording 0/1/2/3, because separating 0 from 1 is a presence test and counting
lives or dies on 1 vs 2 vs 3.

### What Gate 1 settled

Rank features track occupancy; magnitude features track the session.

| Feature | empty vs walking | empty vs a 2nd empty session | |
|---|---|---|---|
| `n_eig` | 0.903 | 0.542 | real |
| `eig_ratio_3` | 0.918 | 0.561 | real |
| `effective_rank` | 0.889 | 0.621 | real |
| `motion_score` | 0.998 | **0.986** | worthless |
| `dilated_pem` | 1.000 | **0.999** | worthless |

`motion_score` and `dilated_pem` separate two *empty* rooms as well as they
separate empty from occupied: they are session detectors. Since
`PresenceDetector` is built on `motion_score`, **the presence path is not
trustworthy across sessions** and no sigma tuning fixes it. Use the rank
features. `har/live_rank.py` shows all of this live.

Scale is the reason: `motion_score` and `dilated_pem` measure the *size* of the
CSI fluctuation, which moves with received power; rank and eigenvalue ratios
measure the *shape* of the covariance, which does not. Rank features also need
no resampling, being covariance-based rather than spectral.

### The drift ceiling (measured 2026-08-20)

The empty-room baseline drifted **+0.578 in 70 minutes** — more than the whole
occupancy signal — so the empty room read *higher* than the occupied room. No
absolute threshold survives that, an adaptive baseline fires on empty-room
noise, and a model given all features falls to chance (49.1%) because it learns
the session. Only two things worked: the three rank features (64.7%
leave-one-session-out) and **event detection**, which cancels slow drift and
scored 77.3% balanced with 0% false positives (`RESULTS.md` §9.12).

Practical consequence: **calibration is good for well under an hour**, and
occupancy should be tracked from motion events plus a timeout rather than from
an absolute level.

### Claims discipline for counting

Quotable: leave-one-room-out accuracy **with MAE in people and +/-1 accuracy
beside it**, the class subset stated, and "after 2-minute on-site calibration".
Report the confusion matrix — the 2 vs 3+ boundary is where this will fail and
hiding it is the one thing this project has consistently refused to do.

Never quote: exact counts above 3; counting still or sleeping people;
cross-room performance without calibration; any same-session or
random-window split.


### Datasets Recorded 2026-08-20
- **sitting_5mins.csv**: 5 minutes of 1 person sitting in the line of sight (approx. 26,723 frames).
- **empty_5mins.csv**: 5 minutes of an empty room (approx. 20,214 frames).
Both captures used the passive firmware on 2.4 GHz channel 6, with traffic generated by a YouTube video on a phone connected to the room's 2.4 GHz AP.
