# Demo runbook

Everything below is `cd ESP32-CSI-Tool` and `csi_env/bin/python`.

**The one thing that will break the demo:** an activity model trained in your
room does not transfer to the presentation room. Different walls, different
multipath, different everything — the paper says so explicitly in its own
limitations section. Presence detection survives a room change after a 15-second
recalibration; activity classification does not. Plan to **re-record and retrain
on site**, and keep the replay fallback ready.

---

## Step 0 — see the whole thing work right now (2 min, no hardware)

Before touching the board, prove the stack runs end to end on synthetic data:

```bash
csi_env/bin/python har/make_synthetic.py --outdir har/recordings-synthetic
csi_env/bin/python har/train_activity.py --recordings har/recordings-synthetic \
  --out har/model-synth.joblib
csi_env/bin/python har/live_demo.py --model har/model-synth.joblib \
  --replay har/recordings-synthetic/walking_synth0.csv
```

The window shows PERSON PRESENT, `activity: walking`, a motion trace and the
live CSI waterfall. Swap `walking_synth0.csv` for `sitting_synth0.csv` or
`empty_synth0.csv` and the readout follows.

That synthetic data is separable by construction, so it scores 100%. **It is a
self-test, not a result** — never put its accuracy on a slide.

---

## Step 0.5 — raise the packet rate (superseded, see note)

> **Superseded 2026-08-20.** This step dates from the 22 Hz firmware. The board
> now runs at **90-97 Hz** natively and the rate is set by the AP's own
> basic-rate traffic, not by client load — pinging changed nothing measurable
> (`RESULTS.md` §9.5g). It also cannot work as written on this network: the Mac
> associates on 5 GHz channel 153 while the board listens on 2.4 GHz channel 6.
> Kept for the record; skip it.


You are at ~22 Hz because CSI only appears when a packet arrives. More traffic
on the channel means more CSI. From your Mac, on the same AP:

```bash
sudo ping -i 0.002 <your_router_ip>     # sub-0.1s intervals need root on macOS
```

Leave that running in another terminal and measure the effect:

```bash
csi_env/bin/python har/run_pipeline.py --serial --baud 460800 --duration 15
```

If `sample rate` climbs above ~50 Hz, keep the ping running for **all** recording
and for the demo itself. If it does not move, drop this step — the system works
at 22 Hz, just with less headroom.

---

## Step 1 — record (about 20 min)

Three classes. Do not try for seven — one SISO node at this rate will not
separate them, and a confident wrong answer on stage is worse than three
reliable ones.

| Label | What to do |
|---|---|
| `empty` | Nobody in the room. Leave and shut the door. |
| `sitting` | One person seated, still, breathing normally. |
| `walking` | One person walking back and forth across the link. |

**Four takes of 45 seconds each, per class.** Record them as genuinely separate
takes — stop, move the chair or your standing position slightly, start again.

> Do not record one long take and split it. Windows from a single continuous
> recording are correlated, and the honest evaluation in Step 2 will not be
> honest any more. This is the single most common way capstone accuracy numbers
> end up fictional.

```bash
csi_env/bin/python har/record_csi.py --baud 460800 --label empty   --duration 45   # x4
csi_env/bin/python har/record_csi.py --baud 460800 --label sitting --duration 45   # x4
csi_env/bin/python har/record_csi.py --baud 460800 --label walking --duration 45   # x4
```

Files land in `har/recordings/`. Twelve files total, ~9 minutes of recording.

---

## Step 2 — train (30 seconds)

```bash
csi_env/bin/python har/train_activity.py --recordings har/recordings --out har/model.joblib
```

Read the output carefully. It prints two accuracies:

- **RANDOM window split** — inflated by overlapping windows. Never quote it.
- **LEAVE-ONE-TAKE-OUT** — holds out whole recordings. This is your number.

You also get a confusion matrix and the majority-class baseline. **If
leave-one-take-out is not clearly above the baseline, the model has learned
nothing** — record more takes, or cut to two classes (`empty` vs `walking`),
which is the easiest pair to separate.

For reference, running this on a null dataset (one recording chopped up and
labelled arbitrarily) gives 100.0% on the random split and 48.7% on the honest
split against a 59.0% baseline. That gap is the check working.

### Optional: the paper's LSTM

PyTorch has no Python 3.14 wheels, so the LSTM lives in a separate venv
(`lstm_env`, Python 3.12, MPS-accelerated). Same recordings, same validation:

```bash
lstm_env/bin/python har/train_lstm.py --recordings har/recordings \
  --out har/model-lstm.joblib
```

`live_demo.py` accepts either model — run it from `lstm_env` for the LSTM one.

**Use the random forest for the live demo.** On identical synthetic data the
two scored 100% vs 51.5% leave-one-take-out. Twelve takes is far too little for
a ~50k-parameter recurrent net, while the feature model's band powers encode
the right prior directly. Present the LSTM as the paper-faithful comparison and
the gap as your finding about data volume — that is a better capstone result
than pretending the LSTM won.

---

## Step 3 — rehearse (5 min)

```bash
csi_env/bin/python har/live_demo.py --baud 460800 --model har/model.joblib
```

Walk in and out. Sit down. Watch the readout. Press `q` to quit.

Then rehearse the fallback so you know it works:

```bash
csi_env/bin/python har/live_demo.py --model har/model.joblib \
  --replay har/recordings/walking_<timestamp>.csv
```

Grab a still for your slides:

```bash
csi_env/bin/python har/live_demo.py --model har/model.joblib \
  --replay har/recordings/walking_<timestamp>.csv --snapshot slide.png
```

---

## Step 4 — at the venue (15 min before you present)

1. Plug in the ESP32, confirm it streams:
   ```bash
   csi_env/bin/python har/run_pipeline.py --serial --baud 460800 --duration 15
   ```
   You want a sensible sample rate and `layout detected`. If frames are zero,
   press EN on the board.

2. **Re-record and retrain in that room.** Two takes per class of 30 s is
   enough to rescue it:
   ```bash
   csi_env/bin/python har/record_csi.py --baud 460800 --label empty   --duration 30   # x2
   csi_env/bin/python har/record_csi.py --baud 460800 --label sitting --duration 30   # x2
   csi_env/bin/python har/record_csi.py --baud 460800 --label walking --duration 30   # x2
   csi_env/bin/python har/train_activity.py --recordings har/recordings-venue \
     --out har/model-venue.joblib
   ```

3. If there is no time to retrain, run **presence only**. It recalibrates in
   15 seconds and is far more robust:
   ```bash
   csi_env/bin/python har/live_demo.py --baud 460800 --calibrate-only
   ```
   Stand clear while it counts down, then walk in.

---

## What to claim

Say these:

- "Presence detection is calibrated on an empty room and thresholded on motion
  energy — no training data required, and it recalibrates in 15 seconds in a new
  room."
- "Activity classification reaches **X%** under leave-one-take-out validation,
  against a **Y%** majority-class baseline." (Both numbers come from Step 2.)
- "We validated our own evaluation: on deliberately mislabelled data the naive
  random split reports 100% and our grouped split reports 48.7%."

Do not say these:

- Any accuracy from the random window split.
- That it works through walls. You have one SISO node with one antenna; the
  paper's through-wall result used a 4-antenna MIMO adapter.
- That it generalises to new rooms. It does not, and you will get asked.

If someone asks why not the pretrained model on Hugging Face: it is a 9,280-
parameter MLP that takes an 8-dimensional feature vector and outputs a 128-dim
embedding plus one presence scalar. It has no activity head, and its presence
head has a +8.19 bias against a bounded ±3.67 logit range, so it returns
"present" for every possible input.

---

## Counting track — Gate 1 (built 2026-08-20, needs your one-person capture)

Separate from the activity demo above. See `CONTEXT.md` §13 for why counting is
scoped to 0/1/2/3+ and why an empty room alone cannot calibrate it.

### Before you calibrate: fix the placement

Read `CONTEXT.md` §2b. Short version for Room 1: **move the ESP32 to the corner
diagonally opposite the router (~19 ft), not closer.** Longer link, bigger
Fresnel zone, more of the floor covered. Costs ~5.5 dB and you have ~10 dB
spare. Then **tape it down** — moving it invalidates every profile built here.

Sanity check after moving:

```bash
csi_env/bin/python har/run_pipeline.py --serial --duration 15
```

Want RSSI above about -75 dBm and the rate still near ~95 Hz. If either
collapses, come back a few feet.

### Step C0 — check the link, not the traffic

An earlier version of this runbook made a fixed-rate ping mandatory here. **That
was wrong for this deployment** and is retracted — see `RESULTS.md` §9.5g. The
CSI stream is fed by the AP's own basic-rate traffic and sits at 90-97 Hz
regardless of occupancy, and the Mac associates on 5 GHz channel 153, so its
pings never reached the 2.4 GHz channel 6 the board listens on. The ping was
neither necessary nor sufficient.

What actually has to be checked is that the room's AP is on channel 6 and
audible. It is not always:

```bash
csi_env/bin/python har/run_pipeline.py --serial --duration 15
```

Want `A8:6E:84:93:EE:60` at **better than -75 dBm** and ~95 Hz. If instead you
see `EC:0E` at -85 dBm or worse, the room AP has wandered off channel 6 — wait
and retry rather than capturing. `calibrate_room` now aborts on its own below
-80 dBm, so a bad session fails loudly instead of producing a quiet, wrong
profile.

If the board returns nothing at all — no CSI *and* no log lines — it has
stalled; it does that after roughly 20 minutes. `calibrate_room` resets it
before each capture.

### Step C1 — calibrate the room (2 minutes)

```bash
cd ESP32-CSI-Tool
csi_env/bin/python har/calibrate_room.py --room room1 \
    --mac A8:6E:84:93:EE:60 \
    --length-ft 16 --width-ft 10 --link-ft 19 \
    --note "ESP32 far corner diagonally from router, torso height, taped"
```

It prompts twice:

1. **Empty room, 60 s.** Leave, shut the door. No pets, no fans.
2. **One person walking, 60 s.** Exactly one, walking a normal loop covering
   the floor — not pacing one line, not standing still.

Writes `har/rooms/room1/profile.json`.

An empty-room capture from 2026-08-20 is already saved. To reuse it instead of
recording phase 1 again:

```bash
csi_env/bin/python har/calibrate_room.py --room room1 --phase one-person \
    --mac A8:6E:84:93:EE:60
```

**Do not reuse the 2026-08-20 baseline for counting.** It was captured on the
-85 dBm transmitter with a third of its duration missing (`RESULTS.md` §9.3),
and both phases must come from the same transmitter or the gain measures two
radio links instead of one person. `build_profile` refuses outright if they
differ. Recapture both phases in one session with `EE:60` above -75 dBm.

### Step C2 — run the gate

```bash
csi_env/bin/python har/gate1_check.py --room room1 --mac A8:6E:84:93:EE:60
```

Prints an AUC per feature against the empty room, **and** a null control
(empty first half vs its own second half). A feature only counts if it beats
its own drift by a clear margin.

- **PASSED** → the features respond to occupancy. Recruit 3 helpers and record
  0/1/2/3 for Gate 2. It does *not* mean counting works: 0-vs-1 is a presence
  test, and counting lives or dies on 1 vs 2 vs 3.
- **FAILED** → do not recruit anyone. Move the ESP32 farther out, confirm the
  walker actually crossed the router-to-ESP32 line, and rerun.

Watch the two band rows at the bottom. If the 25-48 Hz control band separates
as well as the 11-25 Hz Doppler band, the high-frequency content is a
resampling artefact and must not be fed to a classifier.

### Step C3 — only after Gate 1 passes

3 helpers, 4 counts (0/1/2/3) x 3 takes x 2 min, one room ~= 25 min of
recording. Pin `--mac` throughout. No ping is needed — the rate is AP-driven
(`RESULTS.md` §9.5g) — but check `EE:60` is above -75 dBm before each take, and
re-check that `rate` is still single-valued if you record during busy hours.

---

## Activity-intensity recording protocol

The only classification this hardware supports is **intensity**, not headcount
(`DATA.md` §4-5). Recording is guided:

```bash
cd ESP32-CSI-Tool
csi_env/bin/python har/record_protocol.py --room room1 --session 1 \
    --mac A8:6E:84:93:EE:60
```

Six classes — empty, sitting, standing, slow walk, brisk walk, running — two
passes, 2 minutes each, about 24 minutes a sitting. It prompts before every
segment, resets the board (it stalls after ~20 min), and writes a manifest with
per-segment RSSI.

**Every class is recorded inside every session, in shuffled order.** This is not
tidiness. Recording one class per session yields a classifier that identifies
the session: `motion_score` separates two *empty* rooms at AUC 0.986, and a
model given all features scores 49.1% — chance — leave-one-session-out. Cycling
the classes balances session effects across labels instead of confounding them.

**Do not move the ESP32 or the router mid-session.** RSSI explains 89% of
between-session variance at 0.24 level units per dB; 4 dB is the entire
occupancy signal.

Afterwards, check the session was not confounded:

```bash
csi_env/bin/python har/record_protocol.py --room room1 --session 1 --check
```

It flags any segment on a link below −80 dBm and warns if RSSI tracks the class.
Redo individual segments with `--only walk_fast,running`.

Six or more sessions across different times and days, then train with
leave-one-session-out only.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| No CSI frames | Press EN on the board. Check the port in `--port`. |
| `LayoutError` | Capture longer; the guard bands need ~60 frames to show up. |
| Presence stuck on | You calibrated while someone was in the room. Redo it. |
| Presence never triggers | Lower the threshold: `--enter-sigma 3 --exit-sigma 2`. Measured on this board, sigma 6 put the threshold at 0.366 against a 0.139 ambient floor — too high to trip. Sigma 3 put it at 0.191, just above the ambient range of 0.129–0.188. |
| Locks onto a different MAC each run | Traffic varies, so the busiest transmitter changes. Pin it: `--mac A8:6E:84:93:EE:60`. Do this for the real demo. |
| Presence flickers on the boundary | Keep `--exit-sigma` below `--enter-sigma`; equal values disable the hysteresis. |
| Readout flickers | Raise `--calibrate-seconds`, or stand more still. |
| Activity is always one class | Model learned nothing — check Step 2's honest number. |
| Everything failed | `--replay` a recording. The display is identical. |
| Counting: "NO GAIN ANCHOR" | Only the empty phase exists. Run `--phase one-person`. Presence still works; counting refuses by design. |
| Counting: gate says every feature drifts | The empty capture is picking up something — fan, pet, someone walking past a doorway. Recapture. |
| Counting: high `gap_fraction` | Long dropouts. Usually the AP left channel 6 or the board stalled — check RSSI and that log lines are still arriving. |
