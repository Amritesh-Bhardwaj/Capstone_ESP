# Results log

Every measurement taken, with how to reproduce it. Nothing here is estimated.
Last updated 2026-08-20. Project state: [`CONTEXT.md`](CONTEXT.md).
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


## 8c. Board swap and reflash (2026-08-20)

The board was replaced with a different physical unit. **The replacement is the
same module type with the same CSI capability** — every scoping constraint in
`CONTEXT.md` still holds.

### Identification, measured not assumed

`lstm_env/bin/python -m esptool --port /dev/cu.usbserial-5B530174971 chip-id`

```
Chip type:          ESP32-D0WD-V3 (revision v3.1)
Features:           Wi-Fi, BT, Dual Core + LP Core, 240MHz, Vref calibration in eFuse
Crystal frequency:  40MHz
MAC:                1c:c3:ab:d2:29:f8
Detected flash size: 4MB   (flash-id: mfr 5e, dev 4016)
```

Dual core rules out S2/C3/C6 by the same test used in §8b. "ESP32 Wi-Fi+BT SoC"
is not a new part — every original ESP32 is Wi-Fi+BT, including the previous
board.

| | Previous board | This board |
|---|---|---|
| Chip | ESP32-D0WDxx (rev unrecorded) | ESP32-D0WD-V3 rev v3.1 |
| Module | WROOM-32, marking `211 161007` | WROOM-32 |
| Port | `usbserial-57460201261` | `usbserial-5B530174971` |
| Bridge | CH9102/CH343, 136 mA | identical |
| Cores / antenna / band | 2 / 1 / 2.4 GHz | identical |

### It arrived with factory firmware

The flash held **Espressif ESP-AT v2.4.0** (partitions `at_customize`, `ota_0`,
`ota_1`; boot banner `module_name:WROOM-32`). It emitted no `CSI_DATA` at
115200, 460800 or 921600 — 0 bytes at each until reset.

### Reflashed with the `passive` build

`active_sta` was rejected: its `sdkconfig` targets SSID `Wireless1` with an
empty password, so it would never associate and would produce no CSI.
`passive` needs no credentials.

Two build settings had to be forced; both defaulted wrong and each produced a
silent-looking failure:

| Setting | Default | Why it matters |
|---|---|---|
| `CONFIG_ESP32_WIFI_CSI_ENABLED` | **`n`** | `esp_wifi_set_csi(1)` returns `ESP_FAIL`, `csi_component.h:96` aborts, board enters a **boot loop**. The serial stream looks like garbage rather than an error. |
| `CONFIG_ESP_CONSOLE_UART_CUSTOM` | `n` | Without it `ESP_CONSOLE_UART_BAUDRATE` has no prompt and is **silently pinned to 115200** — setting the baud in `sdkconfig.defaults` alone is discarded on every build. |

Both are now recorded in `passive/sdkconfig.defaults`, so a clean rebuild
reproduces this firmware.

### Verified output

`csi_env/bin/python har/run_pipeline.py --serial --port /dev/cu.usbserial-5B530174971 --baud 460800 --duration 20`

```
parsed 1908 frames, rejected 5 malformed rows
transmitter        A8:6E:84:93:EE:60  (rssi -62.7 dBm)
subcarrier layout  fft  -> 48 data bins
frames             1270 over 19.9s
sample rate        95.5 Hz (inter-arrival jitter 1.83x median)
hampel outliers    5.79% of samples replaced
LSTM windows       76 x 64 x 48
```

**1908 -> 1270: a third of frames came from other transmitters.** MAC selection
is mandatory again — §5.3 applies to this firmware, unlike the cut-down build
it replaced. Jitter is 1.83x median, worse than the 1.00x of §2.1; open item 4
matters more now.


## 9. Counting track, Gate 1 tooling (2026-08-20)

Gate 1 asks: do the counting features respond to one occupant by more than the
empty room varies against itself? **The tooling is built and tested; the gate
has not been run**, because it needs a one-person capture. What follows is what
was measured while building it.

Reproduce:

```bash
cd ESP32-CSI-Tool
csi_env/bin/python har/test_count_features.py            # 28/28
csi_env/bin/python har/calibrate_room.py --room room1 --phase one-person
csi_env/bin/python har/gate1_check.py --room room1
```

### 9.1 Two feature bugs, both caught by tests

**Doppler on the subcarrier mean is identically zero.** `remove_common_mode`
divides each frame by its own across-subcarrier mean, so that mean is 1 by
construction and its spectrum is empty. The first implementation measured a
Doppler centroid of **0.0000 Hz** and a spread of 0.03 Hz, which looked like a
quiet room rather than a bug. Fixed by computing the spectrum per subcarrier
and averaging. A test against a tone of known frequency covers it.

**Dilated PEM scaled by its own window is inverted.** Normalising deviations by
the window's own MAD divides out the magnitude the feature exists to measure.
Measured: a smooth sinusoid never exceeds **0.95** of its own MAD so it scores
**0.000**, while a still room of Gaussian noise scores **0.003** — the still
room reads as *more* disturbed than the moving one. Fixed by scaling against
the empty room's per-subcarrier MAD, which the room profile now carries.

### 9.2 Spectral power above 11 Hz

Measured on 4 s windows, resampled to 100 Hz:

| Band | Median share |
|---|---|
| 0.1-0.5 breathing | 0.065 |
| 0.5-2 slow | 0.185 |
| 2-5 walk | 0.189 |
| 5-11 fast | 0.165 |
| **11-25 walk Doppler** | **0.178** |
| **25-48 control** | **0.113** |

**~30% of spectral power sits above 11 Hz**, where `features.BANDS` stops.
Walking Doppler at 2.4 GHz is `2v/lambda` ~= 16 Hz per m/s, so that band is
where walking lives. `features.BANDS` is deliberately unchanged (it would
invalidate the ESP-Fi 90.0% headline); the wider set lives in
`count_features.COUNT_BANDS`. The 25-48 Hz band is above plausible human
Doppler and is kept only as an artefact control.

### 9.3 RETRACTED — the "empty room" capture was occupied

**The capture analysed in the first version of 9.3-9.5 was labelled empty and
was not.** One person was sitting still in the room for its entire duration.
The label was assumed from an intention to leave and never verified against the
person or the data. Every conclusion drawn from it as an empty-room measurement
is withdrawn.

The file is retained, relabelled
`2026-08-20T01-41-34_one-person-SITTING-STILL_10ft.csi.txt`, because as a
*sedentary single occupant* recording it is genuinely useful — see 9.5. The
room profile built from it was deleted.

What still holds from that capture, because none of it depends on occupancy:

| | |
|---|---|
| Wall-clock capture | 601 s |
| **Device time containing CSI** | **264 s** |
| **Seconds producing nothing** | **337 s** |
| Rate within the active span | 91.7 Hz |
| Jitter | 2.56x median |
| Transmitter | `A8:6E:84:93:EC:0E` at **-85.1 dBm** |

Both failures here are real and silent, and neither is about people:

- More than half the capture produced no CSI. In passive mode CSI exists only
  while something transmits. Wall-clock duration says nothing about how much
  data was collected.
- The board fell back from `A8:6E:84:93:EE:60` (-59 to -63 dBm) to `EC:0E` at
  **-85.1 dBm**, roughly 10 dB below usable. Anything measured on that link is
  measured largely on noise.

The earlier reading — that the traffic collapsed *because the occupant left* —
was wrong twice over: nobody left, and the collapse is plain network idleness.

### 9.4 RETRACTED — not a presence result

The reported "0.00% false present at 6 and 3 sigma over 3 minutes" was computed
on the occupied capture. It is not a false-positive rate, because there was no
empty room to be falsely positive about.

What the numbers actually show is narrower and still worth having: **a single
still occupant produces a stable `motion_score` over four minutes** (median
0.2179 -> 0.2069, drifting 5% of itself), and a threshold calibrated on that
occupant's own first minute does not trip during the following three. That is a
stability result for a sedentary scene, not evidence that a 60 s empty-room
calibration rejects an empty room.

Whether a 60 s calibration is enough for presence is **still open**, and is what
the 2026-08-20 02:04 capture was taken to answer.

### 9.5 RETRACTED and reinterpreted — the "drift" may be the occupant

The 12-of-15 features exceeding the gate's null margin were measured between
60 s blocks of a room with a person in it. Their small movements — shifting,
reaching, breathing — are exactly what a counting feature is supposed to
respond to. **Variation across those blocks is as likely to be signal as
drift.** The numbers are kept below only so they are not rediscovered as new:

| Feature | Block-to-block |AUC-0.5| | | Feature | |AUC-0.5| |
|---|---|---|---|---|
| subcarrier_dispersion | 0.317 | | doppler_centroid | 0.191 |
| n_eig | 0.272 | | eig_ratio_3 | 0.190 |
| effective_rank | 0.269 | | band_doppler | 0.187 |
| band_breathing | 0.237 | | doppler_spread | 0.183 |
| motion_score | 0.222 | | dilated_pem | 0.150 |
| band_slow | 0.214 | | band_walk | 0.103 |
| band_control_hi | 0.193 | | band_fast | 0.102 |
| | | | eig_ratio_2 | 0.091 |

Triply confounded: an occupant was present, the link was at -85 dBm, and a
third of the capture was missing. **No claim about feature stability is
supported by this table in either direction.**

### 9.5b Root cause of every failure tonight: the AP left channel 6

The room's own access point, `A8:6E:84:93:EE:60`, **drops off channel 6 for
long stretches.** Timeline on 2026-08-20:

| Time | Dominant transmitter | RSSI | Rate |
|---|---|---|---|
| ~01:0x | `EE:60` | -59 to -63 dBm | ~95 Hz |
| 01:41-01:51 | `EC:0E` (distant) | -85.1 dBm | 91.7 Hz over 264 s of 601 s |
| 02:04-02:09 | `EC:0E` (distant) | **-88.8 dBm** | 77.8 Hz for 21 s, then nothing |
| 04:41 | **`EE:60` back** | **-53.2 dBm** | **97.3 Hz** |

`select_mac` picks the *chattiest* transmitter, not the strongest, so when the
room's AP went quiet the pipeline silently switched to a distant one 35 dB down
and carried on. Rows parsed, rates looked plausible, nothing complained.

This also explains why generating traffic did not help. The Mac is associated on
**channel 153 (5 GHz)** at -46 dBm, so every ping went out on 5 GHz and was
invisible to a 2.4 GHz-only ESP32 listening on channel 6. Pinging the gateway is
only a valid control when the pinging device is on the 2.4 GHz band the board
is watching.

### 9.5c Placement at 16 ft is confirmed good

Measured 04:41 after the move from 10 ft to 16 ft along the room's long axis,
clear line of sight:

```
transmitter  A8:6E:84:93:EE:60   -53.2 dBm
rate         97.3 Hz
```

**-53.2 dBm at 16 ft, better than the -59 to -63 dBm recorded at 10 ft** — the
clear line of sight more than pays for the 4.1 dB of extra path loss. Placement
is settled; there is nothing further to gain from moving the board.

### 9.5d The board also stops emitting and needs a hardware reset

At 02:02 the ESP32 went completely silent — no CSI, no log lines, zero bytes for
100 s — after roughly 20 minutes running. Toggling RTS to pull EN low brought it
straight back at 73.7 Hz. It stalled again ~25 s into the next capture. Long
captures must reset the board first *and* verify rows keep arriving.

### 9.5e The two captures that survive, and what they are not

| Capture | Scene | Link | Verdict |
|---|---|---|---|
| `..._one-person-SITTING-STILL_10ft` | one still occupant, 264 s usable | `EC:0E`, -85.1 dBm | mislabelled "empty"; usable only as a weak-link sedentary sample |
| `..._TRUE-empty_16ft_UNUSABLE-89dBm` | genuinely empty | `EC:0E`, **-88.8 dBm**, 21 s clean | **unusable** — below the noise floor, too short |

They differ in occupancy, placement *and* link quality, so they are not a
comparable pair and no empty-vs-occupied number can be taken from them. The
Gate 1 pair still has to be recorded, back to back, at the current 16 ft
placement, with `EE:60` above -75 dBm.

### 9.5f Every captured frame is legacy 6 Mbps — measured, and it changes the plan

The AP is a **TP-Link EAP620 HD**: 802.11ax, dual-band, 2x2 MIMO. That raised an
obvious worry — a beamforming MIMO AP steering different weights at different
clients would make consecutive frames incomparable, and MCS/rate mixing would do
the same. Checked directly against the header fields `parse_line` normally
discards:

`A8:6E:84:93:EE:60`, 2360 frames over 25 s:

| Field | Distinct values | Value |
|---|---|---|
| `sig_mode` | 1 | 0 (non-HT) |
| `mcs` | 1 | 0 |
| `bandwidth` | 1 | 0 (20 MHz) |
| `rate` | 1 | 11 = 6 Mbps OFDM |
| `ant` | 1 | 0 |
| `channel` | 1 | 6 |

**Zero variation in every PHY parameter.** The same holds for `EC:0E` over
24,294 frames. Only `sig_len` varies, which does not affect CSI comparability.

So despite the AP being 802.11ax 2x2, everything the ESP32 captures is
**legacy non-HT 6 Mbps** — basic-rate management and broadcast traffic, not
client data. No beamforming mixing, no rate mixing. The frames are directly
comparable to each other, which is a considerably better starting point than a
Wi-Fi 6 AP suggested.

### 9.5g The packet-rate confound is much weaker than claimed

Earlier notes warned that "more people means more phones means more packets", so
a counter could learn traffic volume instead of CSI, and prescribed a fixed-rate
ping as a mandatory control. **The measurement does not support that here.**

| Session | Dominant AP | RSSI | Rate |
|---|---|---|---|
| 01:41 | `EC:0E` | -85.1 dBm | 91.7 Hz |
| 02:04 | `EC:0E` | -88.8 dBm | 77.8 Hz |
| 04:41 | `EE:60` | -53.2 dBm | 97.3 Hz |
| 04:5x | `EE:60` | -53.2 dBm | 94.4 Hz |

The rate sits at **90-97 Hz regardless of which AP dominates, its signal
strength, or occupancy** — and it stayed there while the Mac was associated on
**5 GHz channel 153**, meaning none of the ping traffic reached channel 6 at
all. The CSI stream is fed by AP-side basic-rate traffic, which is why it is
both homogeneous (9.5f) and roughly constant.

Consequences:

- **The "mandatory ping" instruction was wrong for this deployment** and is
  removed from `DEMO.md`. It was neither necessary — the rate is AP-driven — nor
  sufficient, since a 5 GHz-associated client cannot feed a 2.4 GHz channel-6
  listener.
- The real precondition is not traffic but **link quality**: confirm `EE:60` is
  present above -75 dBm before each session (§9.5b).
- Measured at night in a quiet hostel across four sessions. If daytime client
  load ever pushes high-rate data frames into the CSI stream, `sig_mode`/`rate`
  would stop being single-valued — that check is the canary. The rate-only
  ablation stays in the Gate 2 plan regardless, since it costs nothing.

### 9.6 Guards added as a result

| Guard | Where | Trips when |
|---|---|---|
| Coverage check | `calibrate_room.check_coverage` | device time < 70% of the requested duration |
| MAC consistency | `room_profile.build_profile`, `gate1_check` | the two phases locked onto different transmitters, which would measure two radio links rather than one occupant |
| Pairwise-block null | `gate1_check.null_floor` | replaces a single half-split; a real effect must beat the *worst* empty-vs-empty pair by 0.15 |
| Gap-aware windows | `csi_pipeline.resample_uniform` | dropouts >0.25 s are marked invalid, and windows >10% interpolated are dropped |

### 9.7 Nothing quotable yet

No counting number exists. The only quotable result from this session is
**9.4**: a 60 s empty-room calibration gives 0.00% false presence over the
following 3 minutes. Everything in 9.5 is confounded and is recorded so it is
not rediscovered as if it were new.

### Gate 1 (2.4 GHz Passive, TP-Link, One Person Sitting)
Gate 1 PASSED! 3/15 features responded cleanly (dilated_pem, motion_score, n_eig). Doppler features were murky because the subject was sitting rather than walking, but baseline separation was achieved.


## 9.8 Live verification of the link and PHY findings (2026-08-20 05:16)

`csi_env/bin/python har/log_reading.py --duration 45` -> full log at
[`logs/2026-08-20T05-16-22-live-reading.md`](../logs/2026-08-20T05-16-22-live-reading.md).
Scene deliberately **unlabelled**: this run verifies the link and the PHY
homogeneity claim, and nothing in it may be read as a presence result.

| | |
|---|---|
| Transmitter | `A8:6E:84:93:EE:60`, **98%** of 4296 parsed frames |
| RSSI | **-53.0 dBm** (well above the -75 dBm floor) |
| Rate | **109.9 Hz** device-clock, jitter 1.81x median |
| Layout | `fft` -> 48 data bins |
| Rejected | 6 rows (0.14%) |

**PHY homogeneity confirmed on all eight fields**, 4229 frames:
`sig_mode`, `mcs`, `bandwidth`, `rate`, `sgi`, `stbc`, `ant`, `channel` are each
**single-valued**. §9.5f checked three of these; this checks the full set. The
frames are directly comparable, and any field gaining a second value is the
canary for client data traffic entering the CSI stream.

### 9.8a Three defects fixed in `log_reading.py`

**No MAC selection.** The tool pipelined every transmitter together. Harmless
under the old single-MAC firmware, wrong under `passive`, which sniffs the whole
channel — this run saw three transmitters spanning 38 dB. Now selects one and
prints the full transmitter table.

**The band table was the FFT of roundoff.** `signal =
remove_common_mode(stage).mean(axis=1)` is identically 1.0 per frame, because
`remove_common_mode` divides each frame by that very mean. Measured: the series
is constant to 12 decimal places, ~1e-12 after mean removal, total spectral
power **7e-20**. This is the same defect found in the new counting code (§9.1)
and it invalidates the band table in
`logs/2026-08-16T01-46-22-live-reading.md`, which is now annotated in place.

**Band powers were computed after the low-pass.** The Hamming FIR and wavelet
denoise run at roughly -3 dB by 11 Hz, so the high bands measured the filter.
Now computed on the Hampel-only series, resampled to a uniform grid.

Effect of the fixes on the same measurement:

| Band | 2026-08-16 (invalid) | 2026-08-20 (correct) |
|---|---|---|
| breathing 0.1-0.5 Hz | 8.1% | 8.4% |
| slow 0.5-2 Hz | 8.0% | 19.9% |
| walk 2-5 Hz | 15.6% | 19.0% |
| fast 5-20 Hz | 15.0% | 29.1% |
| **noise 20-60 Hz** | **0.1%** | **19.8%** |
| *sums to* | *46.8%* | *96.2%* |

The old table's 0.1% above 20 Hz was the low-pass, not the room. Note the
sample rate is 109.9 Hz, so Nyquist is 55 Hz and the top of that band is empty;
and with 1.81x jitter some of the 19.8% may still be resampling artefact. The
25-48 Hz control band in `count_features` exists to settle exactly that, and it
is not settled yet.


## 9.9 Gate 1 PASSED — and the null control is what makes it mean anything

Room 1 profile built from `logs/empty_5mins.csv` (300 s, empty) with the
one-person walking anchor recorded at 04:36 (60 s). Held back:
`logs/true-empty-room-v3_...csv` (600 s, **also empty**, a different session).
All four captures are on `A8:6E:84:93:EE:60`.

Two runs of the same gate. The first is the test; the second replaces the
occupied capture with a **second empty session**, so an honest feature must
score ~0.5 on it.

| Feature | empty vs WALKING | empty vs EMPTY(2) | signal − noise | |
|---|---|---|---|---|
| `n_eig` | 0.903 | 0.542 | **+0.361** | **real** |
| `eig_ratio_3` | 0.918 | 0.561 | **+0.357** | **real** |
| `effective_rank` | 0.889 | 0.621 | **+0.268** | **real** |
| `band_fast` | 0.362 | 0.430 | +0.069 | |
| **`motion_score`** | **0.998** | **0.986** | **+0.012** | **worthless** |
| `subcarrier_dispersion` | 0.048 | 0.052 | +0.003 | worthless |
| **`dilated_pem`** | **1.000** | **0.999** | **+0.001** | **worthless** |
| `doppler_centroid` | 0.526 | 0.269 | −0.206 | |
| `eig_ratio_2` | 0.480 | 0.746 | −0.226 | |

### The two features that look best are the two that are useless

`dilated_pem` scores a **perfect AUC 1.000** separating empty from occupied.
Without the null control it would have been the headline. It also scores
**0.999 separating one empty room from another empty room** — it is a
flawless session detector that knows nothing about people.

`motion_score` is the same story at 0.998 / 0.986. **The existing
`PresenceDetector` is built entirely on `motion_score`**, so the presence path
as it stands will change its mind when the session changes, not when a person
does. That is the likely root of the flapping recorded in `test_features.py`,
and it is not fixable by tuning sigmas.

### Why rank survives and magnitude does not

`motion_score` (a coefficient of variation) and `dilated_pem` (a count of cells
above a threshold) both measure the **size** of the CSI fluctuation. Size moves
with received power and noise floor, and those changed between sessions.

`effective_rank`, `n_eig` and `eig_ratio_3` measure the **shape** of the
subcarrier covariance — its eigenvalue distribution, normalised. Shape is
scale-invariant, so a change in link conditions leaves it alone while an extra
independent scatterer in the room does not. This is the hypothesis the counting
work was built on (§9.1) and it is the one that survived.

It also means the rank features need **no resampling**: they come from a
covariance across subcarriers, not a spectrum, so sample jitter does not touch
them. Every band feature does need it, and none of them cleared the null.

### Caveat that limits how far this goes

Mean RSSI and occupancy are **perfectly collinear across these four captures**
(empty −46.6, empty-v3 −48.9, sitting −53.0, walking −54.5 dBm; Spearman r
= −1.000 with median `motion_score`). With n = 4 that ordering arises by chance
1 in 24 times, so it is suggestive, not established — but it does mean this data
**cannot** separate "responds to people" from "responds to link strength" for
any magnitude feature. The rank features are less exposed, since they are
scale-invariant by construction and did not move between the two empty sessions
despite a 2.3 dB difference. Breaking the collinearity needs an empty capture at
~−53 dBm or an occupied one at ~−47 dBm.

### Consequence for the gate itself

`gate1_check` computes its null from 60 s blocks **within** one capture, which
gave floors of 0.083–0.183. The **cross-session** null is far higher for the
magnitude features (0.986 for `motion_score`) and the within-session number
badly understates it. A second empty session, recorded separately, is not
optional — it is the only thing that distinguishes a real feature from a session
detector.

### Verdict

**Gate 1 passes for `effective_rank`, `n_eig` and `eig_ratio_3` only.** That
licenses recording 0/1/2/3 for Gate 2, using those features. It does not show
counting works; separating 0 from 1 is a presence test, and counting lives or
dies on 1 vs 2 vs 3.


## 9.10 A parsing defect found by running the live view (2026-08-20)

`live_rank.py` crashed mid-session:

```
OverflowError: Python integer 291214 out of bounds for int16
  csi_pipeline._amplitude -> np.fromiter(tokens, dtype=np.int16, count=128)
```

`parse_line` validated array tokens with `_INT_RE = ^-?\d+$`, which checks that
a token **is** an integer but not that it is a valid one. ESP-IDF packs each
subcarrier as a pair of **int8**, so the legal range is -128..127. Two failure
modes followed, and the loud one was the lesser:

| Token range | Behaviour before the fix |
|---|---|
| above 32767 | `OverflowError`, kills any long-running tool |
| **128 .. 32767** | **silently accepted into the int16 buffer, corrupting the amplitude with no error** |

The silent band is the dangerous one: a merged log line running two numbers
together produces values there routinely, and nothing downstream would ever
notice. `CONTEXT.md` §2 already claimed "every token is validated and bad rows
are counted, not silently parsed" — that was true of the *type* and not of the
*range*.

Fixed by rejecting any token outside -128..127; such rows now count as
`rejected` like any other malformed row. Four tests cover it, including the
exact 291214 token that crashed.

### Did it contaminate the Gate 1 result? No

Audited every recording used in §9.9:

| Recording | Rows | Out-of-int8 | Would have crashed |
|---|---|---|---|
| `empty.csi.txt` (profile) | 20,156 | 1 | 0 |
| `one_person.csi.txt` (walking) | 5,724 | 0 | 0 |
| `sitting_5mins.csv` | 26,695 | 1 | 0 |
| `true-empty-room-v3` | 58,701 | 2 | 0 |

**4 corrupted frames in ~111,000 rows (0.004%)**, none large enough to raise.
At most one 400-frame window per recording is touched, and the Hampel filter
exists precisely to absorb a single impulsive frame. The §9.9 AUCs stand.


## 9.11 The live readout flapped, and why (2026-08-20)

First live build changed state roughly every ten seconds. Cause: the rank path
was given a bare threshold, while `PresenceDetector` — written precisely because
the old demo flapped — has hysteresis, debouncing and a running median. None of
that was carried over.

### Why a single window cannot decide

Presence level per 4 s window, mapped through the room profile
(0 = calibrated empty room, 1 = one person walking):

| Scene | median | p5 | p95 |
|---|---|---|---|
| empty | −0.002 | −0.378 | **1.115** |
| empty (2nd session) | 0.110 | −0.294 | 0.994 |
| sitting | 0.943 | −0.027 | 2.578 |
| walking | **1.008** | 0.520 | 2.207 |

**The empty room's p95 (1.115) sits above the walking median (1.008).** No
threshold on a single window can separate them; the flapping was the readout
faithfully reporting that overlap.

### Combining the three validated features helps

| Score | AUC vs occupied | AUC vs 2nd empty | margin |
|---|---|---|---|
| `effective_rank` alone | 0.889 | 0.621 | +0.268 |
| `n_eig` alone | 0.903 | 0.542 | +0.361 |
| `eig_ratio_3` alone | 0.918 | 0.561 | +0.357 |
| **mean of the three** | **0.930** | 0.598 | +0.332 |

The mean separates best while keeping a null margin twice the 0.15 bar, so it is
what the live view now uses.

### Smoothing and hysteresis, measured

State changes per minute on the labelled captures, and error rates:

| Configuration | flips/min empty | flips/min walking | false pos | false neg |
|---|---|---|---|---|
| **bare threshold 0.5 (what shipped first)** | **6.58** | 4.62 | **28.7%** | 7.7% |
| median 9 + hysteresis 0.7/0.35 | 0.21 | 0.00 | 13.5% | 0.0% |
| median 9 + hysteresis 0.8/0.4 | 0.21 | 0.00 | 7.8% | 0.0% |
| **median 11 + hysteresis 0.8/0.4** | **0.21** | **0.00** | **1.4%** | **0.0%** |

**31x fewer state changes on an empty room and false positives from 28.7% to
1.4%**, with no false negatives on the walking capture. Debouncing on top made
it worse (0.41 flips/min, 7.7% false negatives): the running median already
supplies the lag, and debounce only adds more.

A sedentary occupant is still detected only **73.6%** of the time. That is the
known hard case, not a tuning failure.

### Gate 1 results now live in the room profile

`gate1_check` writes its measured AUCs into `profile.json`, and the dashboard
reads them from there — previously they were a hardcoded copy that would have
gone stale on the next run.

`--null-empty <second empty capture>` records the **cross-session** null
alongside the within-session one. The difference decides everything:

| Feature | signal | null within session | null **across** sessions | verdict |
|---|---|---|---|---|
| `n_eig` | 0.903 | 0.083 | 0.542 | real |
| `eig_ratio_3` | 0.918 | 0.111 | 0.561 | real |
| `effective_rank` | 0.889 | 0.110 | 0.621 | real |
| `motion_score` | 0.998 | **0.114** | **0.986** | session detector |
| `dilated_pem` | 1.000 | 0.149 | 0.999 | session detector |

Judged within a session, `motion_score` looks validated. Judged across
sessions it is exposed. The dashboard shows the cross-session number whenever
one exists and says so when it does not, and the `validated` set is derived
from the profile rather than a constant, so a stale list can never outrank a
measurement.

### Sessions are logged now

`live_rank.py` and `rank_server.py` both write `logs/<timestamp>-*.csv` with
every update — smoothed level, raw level, state, and all four features — plus a
markdown summary. The earlier live runs recorded nothing at all, which is why
the 45 s reading in §9.8 could not be re-scored against the validated features
once `motion_score` was rejected.


## 9.12 Calibration drift, event detection, and a model (2026-08-20 06:10)

A live session with **ground truth recorded by hand**: the occupant left at
06:10:00 and returned at a run at 06:11:30.
`logs/2026-08-20T05-58-02-rank-session.csv`, 1781 updates.

### The drift that breaks absolute thresholds

| | raw presence level, median |
|---|---|
| calibration empty (04:59) | **−0.002** |
| **known-empty at 06:10** | **+0.576** |
| occupant back in the room | **+0.541** |

**The empty room read higher than the occupied room.** The baseline drifted
**+0.578 in 70 minutes**, which is larger than the entire occupancy signal. With
an exit threshold of 0.4 the room could never report empty except on a noise
dip — a known-empty stretch logged 70.3% "present", and the exit took 63 s to
register while the run-in took 2 s.

No absolute threshold survives this. An adaptive baseline fails the other way:
it fires on an empty room's own fluctuations (99% "present" on both empty
recordings), and a hybrid that keeps the calibrated scale then absorbs a
continuously-present occupant (walking detected 3.8% of the time).

### Events survive drift; levels do not

Slow drift cancels in a short-window-minus-long-window contrast. Scored with
`har/score_session.py` against the same ground truth:

| Detector | balanced acc | false pos | false neg | flips/min | latency |
|---|---|---|---|---|---|
| level, median-11, fixed thresholds | **42.4%** | **76.3%** | 38.8% | 1.82 | 19 s |
| event detector, 45 s timeout | 56.8% | **0.0%** | 86.3% | 0.30 | 3 s |
| event detector, 150 s timeout | 69.1% | 0.0% | 61.9% | 0.20 | 3 s |
| **event detector, 240 s timeout** | **77.3%** | **0.0%** | 45.3% | 0.20 | 3 s |
| event detector, 400 s timeout | 32.8% | 100.0% | 34.3% | 0.15 | — |

The level detector scores **below chance**. The event detector never false-alarms
in an empty room. Its event score reached **z = +18.0** for the run-in against a
background p90 of **1.09**.

Two caveats. The exit (z = +2.1) falls below the z ≥ 3 event threshold, so it is
not detected as an event at all — the room clears by timeout, not by observing
the departure. And only **70 s** of this session is labelled empty, so any
timeout near or above that scores trivially; 400 s fails for exactly that
reason. The 240 s optimum needs confirming against a session with a much longer
labelled empty stretch.

### A model, tested leave-one-session-out

`har/train_presence.py`, four labelled sessions (2 empty, 2 occupied, 616
windows), each fold holding out one entire session:

| Feature set | balanced accuracy |
|---|---|
| the three Gate 1 features | **64.7%** |
| rank features (adds `eig_ratio_2`) | 63.9% |
| **all 15 count features** | **49.1%** |

Chance is 50%. **Handing the model every feature drops it to chance.** The
magnitude features let it identify the session instead of the occupant, and
cross-session it then has nothing. That is the Gate 1 result (§9.9) reproduced
by a completely different method, and it is the strongest evidence yet that the
rank/magnitude split is real rather than an artefact of one statistic.

64.7% from a model against 77.3% from the event detector is not a fair
comparison — different data, and the model has only four sessions to learn
from, two per class. What it does establish is that a model is not a way around
the drift, only another way of measuring it.

### Tooling

| File | Purpose |
|---|---|
| `presence_events.py` | Motion events by short-vs-long contrast + occupancy timeout |
| `score_session.py` | Scores a session log against hand-recorded ground truth |
| `train_presence.py` | Presence model, leave-one-session-out only |

`score_session.py` takes ground truth the way a person records it —
`--truth "06:10:00=empty,06:11:30=present"` — reports latency per transition
separately from the error rates, excludes a settle window either side of each
transition, and always reports **balanced** accuracy, because these sessions run
93% present and raw accuracy would flatter a constant predictor.


## 9.13 Chasing the drift: two causes ruled out, study running (2026-08-20 06:32)

### Ruled out: window duration

The live path takes a fixed **400 frames**, not a fixed duration, so a falling
packet rate stretches the window. Measured across the session it ranged
**4.2 s to 8.2 s** (median 5.9 s) while the offline path is pinned at exactly
4.0 s by resampling. A longer window captures more variation and would raise
`effective_rank`, which fits the sign of the drift.

It is not the cause:

| | Spearman vs `effective_rank` |
|---|---|
| window duration | +0.202 |
| elapsed time | +0.518 |
| **duration, elapsed time held constant** | **+0.019** |
| **elapsed time, duration held constant** | **+0.441** |

Controlling for time, duration explains essentially nothing. Controlling for
duration, time still explains most of it.

**It is still a defect**, just a different one: the live dashboard and the
offline analysis compute the same feature over windows of different length, so
their numbers are not comparable. The live path should resample like
`load_capture` does. Logged as an open item; not fixed yet because the drift
study below is holding the serial port.

### Ruled out: packet rate on its own

`effective_rank` vs instantaneous rate is −0.202, and it vanishes under the same
control. The rate did fall substantially over the session (94 Hz early, 56 Hz by
06:10), which is worth explaining on its own, but it is not what moved the
baseline.

### What is left

The drift tracks **elapsed time**, which is a description rather than a cause.
Candidates that vary with time and could plausibly move a covariance-shape
statistic: board temperature, AP-side behaviour (transmit power, antenna
selection, client mix), and slow environmental change.

One gap made this harder than it should have been: **the rank session log
records no RSSI**, so link strength could not be checked against the drift
directly. `drift_study.py` records it.

### The controlled study

`har/drift_study.py`, launched 06:32 with the occupant asleep — a scene that
stays constant for hours, which is the condition this needs and which was
missing from every previous measurement.

* 60 s capture every 7 minutes for 5 hours (~43 captures)
* board reset before each capture, since it stalls after ~20 minutes
* every window fixed at **4.0 s after resampling**, so the ruled-out mechanism
  cannot contaminate the result
* records per capture: `effective_rank`, `n_eig`, `eig_ratio_3`, `motion_score`,
  **RSSI mean and spread**, per-subcarrier mean amplitude, the common-mode
  removed shape, packet rate, jitter, gap fraction, Hampel outlier rate

Per-subcarrier amplitude is recorded specifically to separate two explanations:
AGC or transmit-power change scales **every** subcarrier together, while
geometry or multipath change moves them **differentially**. If the drift is
gain, `amp_mean` moves and `shape_mean` does not.

Output: `logs/<timestamp>-drift-study.jsonl`, plus the raw rows of the first and
last capture.
