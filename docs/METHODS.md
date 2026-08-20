# Methods: how anything here got established

The techniques this project uses to decide whether a result is real, why each
one is there, and which mistake it was adopted in response to. Nearly every
entry exists because something plausible turned out to be wrong.

Companion documents: [`DATA.md`](DATA.md) for the recordings and what they show,
[`RESULTS.md`](RESULTS.md) for the measurement log,
[`CONTEXT.md`](CONTEXT.md) for project state.

---

## 1. The null control

**What.** Before believing a feature separates A from B, check what it scores
separating A from *another recording of A*.

**Why.** `motion_score` separates an empty room from an occupied one at
**AUC 0.998**. It also separates one empty room from a *different empty room* at
**0.986**. Judged only on the first number it is the best feature in the
project; judged on both it is a session detector that knows nothing about
people. `dilated_pem` is worse — 1.000 and 0.999.

**How.** `gate1_check.py --null-empty <second empty capture>`. A feature must
beat its own cross-session null by ≥0.15 to count.

**Caveat found the hard way.** The null must be *cross-session*. Computed within
one capture the floors came out at 0.08–0.18 and `motion_score` looked
validated. The same feature scores 0.986 against a second session.

## 2. Leave-one-session-out, never a random split

**What.** Every evaluation fold holds out one entire recording session.

**Why.** Sliding windows overlap, so a random split leaks. Worse, the room's own
state drifts enough between sessions that a model can score well by recognising
the session. Handed all fifteen features, the presence model scores **49.1%** —
chance — leave-one-session-out, while the same model on three rank features
scores **75.5%**. The extra features let it identify the session instead of the
person.

**How.** `train_presence.py`, which has no other mode.

## 3. Balanced accuracy, always beside the chance line

**What.** Report per-class recall and their mean, never raw accuracy alone.

**Why.** The labelled live session ran 93% "present". A detector answering
"present" forever scores 92.6% raw and 50% balanced. The level detector that
shipped first scored **42.4% balanced — below chance** — while looking
respectable on raw accuracy.

**How.** `score_session.py` reports both classes separately and prints the
chance line next to the result.

## 4. Ground truth recorded by a person, never inferred

**What.** Scene labels come from someone stating what was in the room.

**Why.** Two mislabelling failures in one day. A capture was assumed empty
because the occupant said they would leave; they sat in it throughout, and four
sections of analysis were withdrawn. A second capture labelled "two people
sitting" became three when somebody walked in at the 60 s mark.

**How.** `record_protocol.py` prompts before every segment and writes the label
into the filename and manifest. `--scene` is required by the long-running
capture tools.

**Limit, measured.** `--check` compares the first third of a capture against the
last third to catch a scene that changed midway. It catches gross changes — the
two-people capture that decayed to an empty-room reading scores 0.404 and is
flagged — but **not** the third person entering, which scores 0.172, inside the
normal null (29 single-scene captures: median 0.092, max 0.362). A threshold low
enough to catch it would flag 30% of good captures. Tooling cannot recover a
label nobody wrote down.

## 5. Physical controls inside the feature set

**What.** Carry a feature that *should not* respond, and check that it doesn't.

**Why.** The 25–48 Hz band sits above plausible human Doppler. If it separates
scenes as strongly as the 11–25 Hz walking band, the "Doppler" signal is a
resampling artefact rather than motion.

**How.** `count_features.COUNT_BANDS` keeps `control_hi` for exactly this, and
`gate1_check` prints the two side by side. As of 2026-08-20 the control band
moves as much as the Doppler band, so all >11 Hz content is still treated as
unproven.

## 6. Verifying instruments before trusting readings

**What.** Check the hardware and the link before analysing anything.

**Why.** Four separate failures produced plausible-looking output rather than
errors:

| Failure | What it looked like |
|---|---|
| Board stalled after ~20 min | a very quiet room |
| Room AP left channel 6; pipeline fell back to a −88 dBm AP | noisy data |
| Firmware boot loop (`CONFIG_ESP32_WIFI_CSI_ENABLED` unset) | an SSID that appears then vanishes |
| Serial at the wrong baud | 130 KB of bytes, one "line" |

**How.** Every capture tool resets the board first and records RSSI. Anything
below **−80 dBm** is refused. `run_pipeline.py --serial --duration 15` is the
standing pre-flight check.

## 7. Testing features on synthetic signals of known answer

**What.** Before running a feature on real data, run it on a signal whose answer
you already know.

**Why.** Two feature defects were found this way and would not have been found
otherwise:

- **Spectra computed on the subcarrier mean are identically zero.**
  `remove_common_mode` divides each frame by that mean, so it is 1.0 by
  construction. The measured Doppler centroid was 0.0000 Hz, which read as a
  quiet room. Caught by a tone of known frequency.
- **Dilated PEM scaled by its own window is inverted.** A smooth sinusoid never
  exceeds 0.95 of its own MAD and scores 0.000; a still room of Gaussian noise
  scores 0.003. Caught by a test asserting that more motion means a higher
  score.

**How.** `test_count_features.py`, 28 tests, all against constructed signals.

## 8. Checking a published number against a fresh measurement

**What.** Re-derive numbers that already appear in the docs.

**Why.** The "Temporal energy by band" table in
`logs/2026-08-16T01-46-22-live-reading.md` was the FFT of floating-point
roundoff — total spectral power **7e-20**. The visible symptom was that its
shares summed to 46.8%. It had been sitting in the repo unquestioned.

**How.** `log_reading.py` was rewritten and the old log annotated in place
rather than corrected silently.

## 9. Separating confounds by controlled comparison

**What.** When two explanations predict the same observation, find the
measurement that distinguishes them.

**Why.** The presence baseline drifts. Gain change and multipath change both
predict that.

**How.** Gain scales every subcarrier together; geometry moves them
independently. Measured over five hours with the scene constant: amplitude
changed **8.9%**, per-subcarrier shape **62.8%**, and the correlation between
subcarriers' drift was **−0.07**. Gain would give about +0.9. It is the channel.

**Still unresolved.** RSSI explains 89% of between-session variance
(**−0.240 level units per dB**, r = −0.943), but correcting for it also removes
the occupancy signal, because a body both absorbs signal and adds motion. The
two are collinear in every recording we have.

## 10. Partial correlation before believing a cause

**What.** When X and Y both correlate with Z, hold each constant and re-check.

**Why.** The live window took a fixed 400 *frames*, so a falling packet rate
stretched it from 4.2 s to 8.2 s — a plausible cause of the drift, with the
right sign.

**How.** Duration correlates +0.202 with `effective_rank`, elapsed time +0.518.
Holding time constant, duration explains **+0.019**. Holding duration constant,
time still explains **+0.441**. Not the cause. (The fixed-frame window was still
wrong and was fixed; it just was not this bug.)

## 11. Reporting the class subset with every number

**What.** No accuracy figure travels without the conditions that produced it.

**Why.** "90% accurate" for four gross-motion classes is a different claim from
"90% accurate". This predates the current work and is the one discipline the
project had from the start.

**How.** See `CONTEXT.md` §12 and `DATA.md` §9 for the quotable list.

---

## Tooling

| Tool | Purpose |
|---|---|
| `gate1_check.py` | Feature validation with cross-session null control |
| `score_session.py` | Score a live log against hand-recorded ground truth |
| `train_presence.py` | Presence model, leave-one-session-out only |
| `record_protocol.py` | Guided interleaved recording + confound checks |
| `drift_study.py` | Long controlled study of baseline drift |
| `calibrate_room.py` | Two-phase room calibration |
| `test_*.py` | 76 tests across pipeline, features, counting, dashboard |

Statistics used: rank-based AUC (Mann-Whitney), Spearman and partial
correlation, Cohen's d, robust statistics (median/MAD) throughout rather than
mean/std, because CSI outlier rates run 6–10%.

**Deliberately not used:** p-values. Sliding windows overlap by 50% and CSI is
strongly autocorrelated, so samples are not independent and any p-value would be
optimistic by an unknown factor. Effect sizes and null controls are reported
instead.
