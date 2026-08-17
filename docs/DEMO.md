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

## Step 0.5 — raise the packet rate (10 min, optional but worth it)

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
