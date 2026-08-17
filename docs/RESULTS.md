# Results log

Every measurement taken, with how to reproduce it. Nothing here is estimated.
Last updated 2026-08-16. Project state: [`CONTEXT.md`](CONTEXT.md).
All commands are run from `ESP32-CSI-Tool/`; code lives in
[`ESP32-CSI-Tool/har/`](../ESP32-CSI-Tool/har/README.md).

**Reading guide.** Two accuracies are reported everywhere. The *leaky* one
splits overlapping sliding windows at random and is shown only for contrast.
The *honest* one holds out whole recordings or whole sessions. Only the honest
number is quotable.

---

## 1. Pipeline correctness

`csi_env/bin/python har/test_csi_pipeline.py` → **33/33 pass**
`lstm_env/bin/python har/test_csi_pipeline.py` → **33/33 pass**
`csi_env/bin/python har/test_features.py` → **11/11 pass**

Tests run against two real captures: the attached board (`fft` layout) and the
repo's own `python_utils/example_csi.csv` (`shifted` layout).

### 1.1 Subcarrier layout, measured

Mean amplitude per bin over a live capture identified the null guard bands:

| Layout | All-zero bins observed | Interpretation | Data bins |
|---|---|---|---|
| `fft` (live board) | 27–37 (11 bins) | subcarriers +27…+31, −32…−27 | 48 |
| `shifted` (example CSV) | 2–4, 60–63 (7 bins, plus leakage at 1, 5, 59) | subcarriers −30…−28, +28…+31 | 48 |

Both give **48 data + 4 pilot + 12 null = 64**, the 802.11 20 MHz OFDM
structure. This is the check confirming the mapping is correct rather than
merely plausible.

A first version of `detect_layout` used a fixed match count and **failed** on
the shifted fixture, because its outermost guard bins carry a few counts of
spectral leakage. Fixed by scoring both hypotheses fractionally with a required
margin. *The test caught this; it was not tuned away.*

## 2. Live hardware capture

### 2.0 Current firmware (460800 baud, minimal format) — use these numbers

`csi_env/bin/python har/run_pipeline.py --serial --baud 460800 --duration 20`

```
parsed 2296 frames, rejected 4 malformed rows
transmitter        00:00:00:00:00:00   (implicit; minimal format has no MAC)
subcarrier layout  fft  -> 48 data bins
frames             2296 over 22.9s
sample rate        100.0 Hz
hampel outliers    8.13% of samples replaced
LSTM windows       140 x 64 x 48
```

**~100 Hz, 4.5x the earlier rate.** See §8b. A 64-frame window spans 0.64 s
rather than 2.9 s.

### 2.1 Earlier firmware (115200 baud, full format) — historical

```
parsed 471 frames, rejected 4 malformed rows
transmitter        A8:6E:84:93:EE:60  (rssi -47.1 dBm)
subcarrier layout  fft  -> 48 data bins
frames             278 over 19.8s          <- after MAC selection
sample rate        22.2 Hz (inter-arrival jitter 1.00x median)
hampel outliers    5.23% of samples replaced
LSTM windows       14 x 64 x 48
```

Note 471 → 278: **41% of frames came from other transmitters.** MAC selection
is not optional.

### 2.2 Presence calibration on real hardware

`har/live_demo.py --calibrate-only`

| `--enter-sigma` | baseline | enter threshold | ambient range | usable? |
|---|---|---|---|---|
| 6 (default) | 0.1394 | 0.3655 | — | **No** — 2.6× the floor, never trips |
| 3 | 0.1571 | 0.1909 | 0.129–0.188 | Yes — sits just above ambient |

Real ambient CSI is far noisier than synthetic. **Use `--enter-sigma 3
--exit-sigma 2` on real hardware.**

Across two consecutive runs the demo locked onto *different* MACs
(`EE:60` then `F8:5C`) because traffic shifted. Pin `--mac` for the demo.

## 3. Validation methodology

### 3.1 Null test — labels carry no information

One continuous recording chopped into 4 takes, labelled `empty`/`walking`
arbitrarily. A correct evaluation must report chance.

| Model | Leaky random split | Honest leave-one-take-out | Baseline |
|---|---|---|---|
| Random forest | **100.0%** | 48.7% | 59.0% |
| LSTM | **100.0%** | 16.7% | — |

The leaky split reports a perfect score on meaningless labels. This is the
single most important slide in the deck.

### 3.2 Signal test — classes genuinely differ

`make_synthetic.py`: 12 takes, 3 classes differing in modulation depth and
frequency. Separable by construction — a self-test, **not a result**.

| Model | Leaky | Honest leave-one-take-out | Baseline |
|---|---|---|---|
| Random forest | 100.0% | **100.0%** | 33.3% |
| LSTM | 100.0% | **51.5%** | 33.3% |

LSTM per-class recall: `empty` 100%, `sitting` 36.9%, `walking` 17.5%.

**Why the RF wins:** 960 windows from 12 takes is far too little for a ~50k
parameter recurrent net, while the band-power features encode the
discriminating prior directly. Use the RF for the live demo; present the gap as
a finding about data volume.

## 4. `ruvnet/wifi-densepose-pretrained` — rejected

Claimed to be "a trained LSTM". It is not.

| Claim | Measured |
|---|---|
| An LSTM | **2-layer MLP.** `Linear(8,64)+BN+GELU → Linear(64,128)+BN → L2-norm`. **9,280 params, zero recurrence.** |
| CSI model | Input is **8 dims**, not CSI — pre-derived sensing scores (motion, respiration, heartbeat, anomaly, env-shift, coherence) from `rv_feature_state.h`. |
| Activity recognition | **No activity head.** Outputs = 128-dim embedding + 1 presence scalar. **0 activity classes.** |
| 82.3% accuracy | Held-out *temporal-triplet* accuracy, a self-supervised proxy. A **random untrained encoder scores 69.56%**; raw features 66.42%. Real lift ≈ 12.8 points. |
| Trained model | `training-metrics.json` logs 20 epochs flat at **0.135171–0.135174** (range 3×10⁻⁶). The reported `finalLoss: 0.065434` appears nowhere in that history. |
| Loadable | `model.safetensors` header has **3 trailing NUL bytes**; strict `safetensors` raises `JSONDecodeError`. Confirms their issue #1522. |
| Training data | One file: `overnight-1775217646.csi.jsonl`. No activity labels. |

### 4.1 The presence head cannot output "absent"

Proved from the bound, not sampled:

```
presence_head bias = 8.188      ||w||_2 = 3.668
embedding is L2-normalised  =>  z·w ∈ [-3.668, +3.668]
=> logit ∈ [4.521, 11.856]  =>  p ∈ [0.989, 0.99999]
```

Empirically, across zeros / N(0,1) / N(0,50) / uniform[−100,100] / ±1e3
(200 samples each): **0.00% of inputs ever gave p < 0.5.**

## 5. `Retsediv/WIFI_CSI_based_HAR` — rejected

- **No pretrained weights** (`.pt`/`.pth`/`.h5`/`.ckpt`/`.onnx` search: none).
- `input_dim = 468` = 114 subcarriers × 4 antenna pairs × 2 (amplitude+phase).
  Ours is **48**.
- `SEQ_DIM = 1024`; at 22 Hz that is a 46-second window.
- Atheros 2×2 MIMO on OpenWrt routers over UDP. No ESP32 path.

**Useful anyway:** their `data_calibration.py` uses `hampel(k=7, t0=3)` and
wavelet soft-thresholding (`sym5`) — the *same chain we built*, arrived at from
different hardware. Cite as independent corroboration of the preprocessing.

## 6. `Awesome-WiFi-CSI-Sensing` survey + transfer test

All 27 linked repos scanned for weight files:

| Repo | HAR weights |
|---|---|
| **SHARP** | 2 (`.h5`, single-antenna) |
| **SHARPax** | 4 (`.h5`, incl. a 20 MHz variant) |
| Other 25 (SenseFi, THAT, EfficientFi, CSI-Net, WiADG, SignFi, DeepSeg, CsiGAN, …) | **0** |

### 6.1 Transfer test — both fail

`lstm_env/bin/python har/test_pretrained_transfer.py --sharp-dir … --sharpax-dir …`

SHARP's Doppler front-end ported exactly (`num_symbols=31`, 100-point FFT,
fftshift, |·|² summed over subcarriers, row-max normalisation, 10⁻¹·² floor),
**with pure Gaussian noise as a control**:

| Model | Real ESP32 | Gaussian noise | real-vs-noise logit correlation |
|---|---|---|---|
| SHARP single-antenna (128,535 params) | Empty, 100.00% | Empty, 100.00% | **+0.9811** |
| SHARPax 20 MHz (77,834 params) | Running, 100.00% | Running, 100.00% | **+0.9510** |

Neither model distinguishes real CSI from noise. `max|logit|` was **534** and
**183**; in-distribution inputs give order ~10 — the signature of input far off
the training manifold. Both report 100% confidence on noise.

**Why — physics, not a porting bug:**

| Factor | SHARP / SHARPax | Ours |
|---|---|---|
| Sample rate | Tc = 6 ms (167 Hz) / 7.5 ms (133 Hz) | ~22 Hz |
| Doppler span of the 100 bins | ±83 Hz | ±11 Hz |
| Carrier | 5.0 / 5.785 GHz | 2.437 GHz |
| Max unambiguous velocity | ~3.5–5 m/s | ~1.35 m/s |
| Phase | 3-stage sanitisation on Nexmon 802.11ac | uncalibrated |

Walking (~1.3 m/s) sits at our aliasing limit; running aliases outright.

## 7. ESP-Fi HAR benchmark — the usable result

`csi_env/bin/python har/train_espfi.py --root <espfi_data>` — **19 seconds.**

Dataset: `AutoSmartGroup/ESP-Fi-HAR`. Commodity ESP32, amplitude only,
`CSIamp` (950, 52), 7 activities, 4 environments, 560 files. Official split is
**session-disjoint** (train 3-2…3-8, test 3-1), so leakage-free by construction.

### 7.1 All four configurations

| Model | Classes | Per-window | Per-file | Baseline | Runtime |
|---|---|---|---|---|---|
| Random forest | 7 | 41.7% | **64.3%** | 14.3% | 19 s |
| Random forest | 4 (gross motion) | 66.7% | **90.0%** | 25.0% | 12 s |
| LSTM (40 ep) | 7 | 29.3% | 37.1% | 14.3% | 105 s |
| LSTM (40 ep) | 4 (gross motion) | 47.9% | 55.0% | 25.0% | ~60 s |
| LSTM (150 ep) | 4 (gross motion) | 49.6% | 65.0% | 25.0% | ~200 s |

Gross-motion subset = `walk,turn,fall,run`, via `--classes`.

### 7.2 Per-file recall by class (random forest)

| 7-class run | | 4-class run | |
|---|---|---|---|
| walk | **100%** | walk | **100%** |
| turn | **90%** | turn | **100%** |
| fall | **90%** | fall | **100%** |
| run | **80%** | run | 60% |
| jump | 40% | | |
| squat | 30% | | |
| arm_wave | 20% | | |

**The split is exactly what the physics predicts:** whole-body gross motion
separates well; small and postural movement does not. This is the honest
ceiling for one SISO antenna, and it is the evidence for scoping the live demo
to `empty` / `sitting` / `walking`.

### 7.3 Ablation — the paper's filter chain earns its place

`--no-filter` skips Hampel → Hamming → wavelet and classifies raw amplitude.
Same data, same model, same split:

| Preprocessing | Per-window | Per-file |
|---|---|---|
| Hampel → Hamming → wavelet (paper's chain) | **66.7%** | **90.0%** |
| None (raw amplitude) | 60.4% | 77.5% |
| **Gain from the chain** | **+6.3 pts** | **+12.5 pts** |

Direct measured evidence, on published third-party ESP32 data, that the
Abuhoureyah et al. §4.1.2 preprocessing improves accuracy — not merely that we
implemented it faithfully. Reproduce with
`--classes walk,turn,fall,run --no-filter`.

### 7.4 Only one environment is available

The ESP-Fi README describes four indoor environments, but **every `.mat` file
in the git tree begins with `3-`** — a single environment. The other three are
presumably inside `ESP-Fi_dataset.rar`, which is Git LFS and was not
retrievable (git-lfs is not installed here).

Consequence: **a room-disjoint evaluation cannot be run on what we have.** The
official split is session-disjoint within one environment, so it does not
measure cross-room transfer — the very failure mode that matters most for the
live demo. Do not present 90.0% as evidence of cross-room generalisation.

### 7.5 Correction: the RF-vs-LSTM gap is not mainly about data volume

An earlier reading of §3.2 attributed the LSTM's loss to having only 12 takes.
**That explanation does not survive this dataset.** With 2,940 training windows
— roughly 3× more — the LSTM still trails by ~25 points (37.1% vs 64.3% on
7 classes, 65.0% vs 90.0% on 4). Training 150 epochs instead of 40 recovers
10 points and then plateaus, so it was partly undertrained, but longer training
does not close the gap.

The better explanation: the engineered band-power features encode the
discriminating quantity — how fast the subcarrier amplitudes are modulated —
*directly*. The LSTM has to learn that from scratch, and at this data scale it
cannot. Data volume is a contributing factor, not the main one.

This is a stronger capstone finding than the original claim, because it is
measured on a published dataset rather than on synthetic data.

*Caveat to state aloud:* the dataset documents no sample rate, so band-power
features use a nominal 100 Hz — internally consistent, not physically
calibrated. All other features are rate-free.

## 8. Summary of quotable numbers

| Result | Number | Conditions |
|---|---|---|
| **ESP-Fi HAR, gross motion, per-file** | **90.0%** | 4 classes, session-disjoint, baseline 25.0% |
| ESP-Fi HAR, gross motion, per-window | 66.7% | same, baseline 25.0% |
| ESP-Fi HAR, all activities, per-file | 64.3% | 7 classes, session-disjoint, baseline 14.3% |
| ESP-Fi HAR, all activities, per-window | 41.7% | same, baseline 14.3% |
| Leaky-vs-honest gap on null labels | **100% → 48.7%** | baseline 59.0% |
| RF vs LSTM on ESP-Fi | **90.0% vs 65.0%** | 4 classes, session-disjoint, baseline 25.0% |
| Pretrained transfer (SHARP) | **r = +0.98 vs noise** | model cannot distinguish CSI from noise |
| Live capture | 22.2 Hz, 48 bins, −47.1 dBm | 20 s, 471 parsed / 4 rejected |

Reproduce the headline number:

```bash
cd ESP32-CSI-Tool
csi_env/bin/python har/train_espfi.py --root <espfi_data> --classes walk,turn,fall,run
```

## 8b. Serial fault, misdiagnosed then resolved (2026-08-16)

### Board

**ESP32-WROOM-32**, module marking `211 161007`. Confirmed from our own
captures rather than taken on trust:

| Evidence | Reading |
|---|---|
| `task_wdt` log names `CPU 0: wifi` and `CPU 1: IDLE` | **dual core** -> ESP32-D0WDxx, i.e. WROOM-32. Rules out S2/C3/C6, which are single-core. |
| `bandwidth=0`, `len=128`, 64 subcarriers | classic ESP32 LLTF, 20 MHz |
| `ant=0`, channel 6 | single antenna, 2.4 GHz only |

The single antenna is the constraint behind every scoping decision in this
project: no MIMO, no through-wall claim, no transfer from multi-antenna
research datasets.

### What happened

Mid-session the serial stream became unreadable at 115200 baud: 31% of bytes
>= 128, zero ESP-IDF boot signatures across 98 KB, and no CSI rows.

**An intermediate writeup concluded this was a physical-layer fault -- cable,
connector or port power. That conclusion was wrong.** It rested on a baud scan
that crashed partway through and never actually tested 460800. "No clean output
at any baud tried" was true of the rates reached, but was then over-generalised
into a hardware verdict the evidence did not support.

### Actual cause

The board had been reflashed. It now runs **different firmware** at a
**different baud**:

| | Before | After |
|---|---|---|
| Baud | 115200 | **460800** |
| Header fields | 25 (`CSI_DATA,role,mac,rssi,...`) | **3** (`CSI_DATA,len,rssi`) |
| MAC / channel / device timestamp | present | **absent** |
| Sample rate | 22-24 Hz | **100-115 Hz** |

A 22-rate sweep found it unambiguously:

```
   baud   ascii  hi-bit  signatures
 115200   64.1%   34.0%  -
 230400   41.0%   23.9%  -
 460800  100.0%    0.0%  CSI_DATA   <<<
```

100% ASCII, 0% high-bit bytes, `CSI_DATA` present. Nothing was wrong with the
cable, the connector, or the board.

### Lesson worth recording

A negative result from an incomplete search is not evidence of absence. The
scan had crashed with `OSError: [Errno 6] Device not configured` after its
first rate; that error was read as further proof of a hardware fault rather
than as a truncated experiment. The fix was simply to finish the sweep.

### Consequences, all beneficial

- **Sample rate rose from ~22 Hz to ~100 Hz.** This was the single largest
  limitation in the project. A 64-frame window now spans 0.64 s rather than
  2.9 s, and a 20 s capture yields 140 windows instead of 14.
- Layout still auto-detects as `fft` -> 48 data bins, so no analysis changes.
- `parse_line` now handles both header formats, dispatching on field count.
  The minimal form requires both header fields to be numeric, so a truncated
  full row cannot masquerade as a valid minimal one.
- Missing device timestamps fall back to a uniform grid (file input) or
  wall-clock arrival (live input), reported as `stats["timing"]`.

### Verified live after the fix

```
csi_env/bin/python har/run_pipeline.py --serial --baud 460800 --duration 20

parsed 2296 frames, rejected 4 malformed rows
subcarrier layout  fft  -> 48 data bins
sample rate        100.0 Hz
frames             2296 over 22.9s
hampel outliers    8.13% of samples replaced
LSTM windows       140 x 64 x 48
```

**All tools now require `--baud 460800`.** No `--mac` is needed: the minimal
format reports a single implicit transmitter.

## 8c. Live reading: why a seated person is not detected (2026-08-16)

### Geometry

Router in one corner, ESP32 on the opposite diagonal, occupant seated between
them — i.e. **directly on the line-of-sight path**. This is the *best* possible
placement: it maximises the signal the body disturbs. The geometry is not the
problem.

### Calibration actually used

```
calibrated on 68 empty windows
baseline 0.1246   enter > 0.2032   exit < 0.1770
ambient motion score ranged 0.077 - 0.182
```

### Live reading, seated occupant, 40 s / 285 windows

```
motion_score   min 0.0169   p25 0.0321   median 0.0420   p75 0.0596   max 0.1705
windows above the 0.2032 threshold:  0 / 285  (0.0%)
```

**The entire live distribution sits below the calibration baseline.** Median
0.042 against a supposed empty-room floor of 0.125 — the "empty" calibration
was three times noisier than the occupied room.

### Cause 1 — the calibration was contaminated

The 20 s calibration ran while the operator was walking out of the area, so it
measured *walking*, not an empty room. Everything after was compared against a
motion floor, guaranteeing silence. Calibration must be taken with the scene in
the same steady state it will be judged against.

### Cause 2 — a still person is invisible to a variance detector

45 s capture, occupant seated and deliberately still (5,173 frames @ 115 Hz):

| Window | Seconds | Median score | p95 | CV of score |
|---|---|---|---|---|
| 64 | 0.56 | 0.0609 | 0.0900 | 0.305 |
| 128 | 1.11 | 0.0647 | 0.0920 | 0.267 |
| 256 | 2.23 | 0.0669 | 0.0961 | 0.236 |
| 512 | 4.45 | 0.0683 | 0.0948 | 0.223 |
| 1024 | 8.91 | 0.0712 | 0.0900 | 0.174 |
| 2048 | 17.82 | 0.0733 | 0.0809 | **0.069** |

Lengthening the window barely moves the score (0.061 → 0.073) but cuts its
variability 4.4x. So longer windows buy **stability, not sensitivity**.

`motion_score` measures how much the channel *fluctuates over time*. A seated,
still person shifts the channel to a new **static** state; they do not make it
fluctuate. Being invisible is the detector working as designed, not a bug.

### Where the energy actually is (45 s, full capture)

| Band | Share of power |
|---|---|
| breathing 0.1–0.5 Hz | **3.1%** |
| slow 0.5–2 Hz | 21.2% |
| walk 2–5 Hz | 32.0% |
| fast 5–20 Hz | 37.2% |
| noise 20–60 Hz | 0.3% |

Only 3.1% sits in the breathing band, and resolving it needs a much longer
window than the demo uses:

| Window | Seconds | Lowest resolvable frequency |
|---|---|---|
| 64 | 0.56 | 1.80 Hz — too short for breathing |
| 512 | 4.45 | 0.225 Hz — still too short |
| 2048 | 17.82 | 0.056 Hz — breathing resolvable |

### Consequence

Two different questions need two different detectors:

- **"Is someone moving?"** — `motion_score` on channel variance. Works, and is
  what the demo currently implements.
- **"Is someone there, even if still?"** — requires comparing the current
  channel *state* against a stored empty-room fingerprint, not its variance.
  Not yet implemented.

Do not describe the current demo as presence detection for a stationary
occupant. It detects motion.

## 9. Not yet measured

- Any accuracy on **our own recorded activity data** — no labelled takes exist
  yet. This is the remaining blocker.
- Whether the jump from 22 Hz to 100 Hz (§8b) improves accuracy. The data is
  now available; nothing has been trained on it.
- Cross-room generalisation on our own hardware (expected to be poor).
- Cross-environment generalisation *within* ESP-Fi. The dataset covers four
  rooms, but the official split is session-disjoint, not room-disjoint. Holding
  out a whole environment would measure the transfer problem directly — the
  single most relevant unmeasured quantity for this project.
- Whether the filter chain helps on ESP-Fi. `train_espfi.py --no-filter`
  exists to measure it; it has not been run.
