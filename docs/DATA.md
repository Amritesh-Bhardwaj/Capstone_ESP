# The data: what we have, and what it says

Every CSI recording collected for this project, what condition it was taken
under, and what falls out of comparing them. Written 2026-08-20.

Companion documents: [`CONTEXT.md`](CONTEXT.md) for project state,
[`RESULTS.md`](RESULTS.md) for the full measurement log with reproduction
commands.

**One-line summary: on this hardware, in this room, we can detect a person
*moving*. We cannot detect a person *being there*.** Everything below is the
evidence for that sentence.

---

## 1. The recordings

All on the ESP32-WROOM-32 running the `passive` build, channel 6, 460800 baud,
48 data subcarriers, amplitude only.

### Usable — strong link, transmitter `A8:6E:84:93:EE:60`

| Scene | Capture | Time | RSSI | Windows | Rate |
|---|---|---|---|---|---|
| **0 — empty** | `empty_5mins.csv` | 04:59 | −46.6 dBm | 146 | 99.6 Hz |
| **0 — empty** | `true-empty-room-2am` | 02:21 | −51.5 dBm | 36 | — |
| **0 — empty** | `true-empty-room-v3` | 02:27 | −48.9 dBm | 296 | 98.7 Hz |
| **0 — empty** | `empty_series/*` (lunch) | 13:47→ | −50.2 dBm | 68+ | 101.5 Hz |
| **1 — sitting still** | `sitting_5mins.csv` | 04:50 | −53.0 dBm | 148 | 107.3 Hz |
| **1 — walking a loop** | `one_person.csi.txt` | 04:36 | −54.5 dBm | 26 | 108.2 Hz |
| **1 — asleep** | `drift-first` | 06:32 | −54.9 dBm | 28 | 98.8 Hz |
| **1 — asleep** | `drift-last` | 11:34 | −55.0 dBm | 27 | 99.3 Hz |
| **2 — standing in LoS** | `TWO-people-standing-LoS` | 13:41 | −52.3 dBm | 88 | 100.0 Hz |

### Unusable — weak link, transmitter `A8:6E:84:93:EC:0E`

| Capture | RSSI | Why it is kept |
|---|---|---|
| `2026-08-20T01-41-34_one-person-SITTING-STILL_10ft` | **−85.1 dBm** | Originally labelled "empty"; it was not. See §6. |
| `2026-08-20T02-04-11_TRUE-empty_16ft_UNUSABLE-89dBm` | **−88.8 dBm** | 21 usable seconds. |

These two are the reason every tool now refuses a capture below −80 dBm. Scored
against the room profile they return presence levels of **11.6 and 19.0**,
where one walking person is 1.0 — a weak link inflates every feature until the
numbers are meaningless rather than merely noisy.

### Derived logs

| Kind | Contents |
|---|---|
| `*-rank-session.csv` | Live dashboard, one row per 0.5 s: smoothed level, raw level, decision, all four features, frames, rate |
| `*-drift-study.jsonl` | 32 captures over 5 h with the scene held constant; per-capture features, RSSI, per-subcarrier amplitude |
| `*-live-reading.md` | Single-capture reports from `log_reading.py` |

---

## 2. The measurement scale

Everything is reported as a **presence level**, mapped through the room
calibration so that

* **0.0 = the calibrated empty room**
* **1.0 = one person walking a loop in it**

It is the mean of the three features Gate 1 validated — `effective_rank`,
`n_eig`, `eig_ratio_3` — each normalised the same way. That mapping is what
makes numbers from different captures comparable at all, and it is why the room
has to be calibrated before any of this means anything.

---

## 3. What the scenes actually read

| Scene | Level (median) |
|---|---|
| empty — `empty_5mins` 04:59 | **−0.00** |
| empty — `v3` 02:27 | **0.11** |
| **2 people standing in LoS** | **0.46** |
| empty — lunch series 13:47 | **0.69** |
| 1 person asleep 06:32 | **0.78** |
| 1 person asleep 11:34 | **0.79** |
| 1 person sitting still | **0.94** |
| 1 person walking | **1.01** |
| empty — `2am` 02:21 | **1.29** |

Read that ordering carefully. **An empty room at lunchtime (0.69) reads higher
than two people standing in front of the antenna (0.46), and about the same as a
sleeping person (0.78).** An empty room at 2am reads higher than anything else
in the table.

The scenes do not separate. The *sessions* do.

---

## 4. Separability, measured

AUC on the presence level between every pair of captures. 0.5 means
indistinguishable; 1.0 means perfectly separated.

| | empty 04:59 | empty 02:27 | empty lunch | 1 asleep | 1 sitting | 1 walking | 2 standing |
|---|---|---|---|---|---|---|---|
| **empty 04:59** | — | 0.60 | **0.81** | 0.82 | 0.83 | 0.93 | 0.77 |
| **empty 02:27** | 0.40 | — | **0.75** | 0.75 | 0.78 | 0.92 | 0.70 |
| **empty lunch** | 0.19 | 0.25 | — | **0.51** | 0.60 | 0.77 | **0.44** |
| **1 asleep** | 0.18 | 0.25 | **0.49** | — | 0.60 | 0.76 | 0.45 |

Three things fall out:

**Empty separates from empty.** `empty 04:59` vs `empty lunch` is **0.81**. Two
recordings of the same vacant room, on the same transmitter, score higher than
most empty-vs-occupied pairs.

**Empty does not separate from occupied.** `empty lunch` vs `1 asleep` is
**0.51** — chance. Against two people standing in the line of sight it is
**0.44**, which is not merely indistinguishable but backwards.

**Walking is the exception.** Every empty session separates from the walking
capture at **0.77–0.93**. That is the only reliable signal in the dataset, and
it is motion, not occupancy.

---

## 5. Why: the features measure motion, not bodies

Two independent measurements say the same thing.

### 5.1 Two people are quieter than one

The 2-person capture was 3 minutes, standing in the line of sight, holding
still. Within it:

| Segment | Level (median) |
|---|---|
| 0–30 s — walking into position | 0.57 |
| 30–60 s | 1.07 |
| 90–120 s | 0.46 |
| **150–180 s — holding still** | **0.09** |

The signal decayed monotonically to **0.09**, indistinguishable from an empty
room, while two people stood directly between the router and the receiver. The
live dashboard read 4.58 at the moment they walked in — that was the walking,
not the standing.

`CONTEXT.md` §13 asserted on theory that "a SISO link measures how much
independent motion is present, not how many bodies". This is the direct
measurement, and it is stronger than the theory allowed for: two motionless
people are not merely hard to count, they are **invisible**.

### 5.2 A constant scene wanders as much as an occupant does

32 captures over 4.9 hours with one person asleep and nothing else changing:

| Quantity | Range over 5 h |
|---|---|
| presence level | **1.35 units** |
| RSSI | 1.16 dB |
| packet rate | 96.6 → 103.2 Hz |

**1.35 units of wander, against an occupancy signal defined as 1.0.** The room
moves more on its own than a person moving into it does.

Where the movement comes from — the discriminating measurement:

| | change over 5 h |
|---|---|
| `amp_mean`, all subcarriers together (gain / AGC) | 8.9% |
| per-subcarrier shape, worst bin | **62.8%** |
| **correlation between subcarriers' drift** | **−0.07** |

Subcarriers move **independently**. Gain or transmit-power change would move
them together at about +0.9. The multipath structure of the room is genuinely
changing — consistent with a hostel room having a glass wall and a wooden door.

### 5.3 How fast calibration goes stale

Median difference between two captures of the *same* scene:

| Separation | Difference (level units) |
|---|---|
| 0–10 min | **0.16** |
| 10–30 min | 0.24 |
| 30–60 min | 0.24 |
| 1–2 h | 0.27 |
| 2–7 h | 0.31 |

It rises quickly then flattens. **A calibration is good for roughly 10 minutes
at full accuracy and 30 minutes degraded; waiting longer barely makes it
worse.**

---

## 6. Two labelling failures worth keeping

**A capture labelled "empty" that was not.** The 01:41 recording was assumed
empty because the occupant had said they would leave; nobody verified it. They
sat in the room throughout. Four sections of analysis were built on it and had
to be retracted. It is renamed `..._one-person-SITTING-STILL_10ft` and kept.

**A conclusion drawn from a rejected feature.** Asked whether a person had been
present during a 45 s reading, the answer given was "yes, sitting" — argued from
`motion_score` and from an RSSI match. The occupant confirmed it. But
`motion_score` had by then been measured at **AUC 0.986 between two empty
rooms**, and the reading was *higher* than the walking capture's median. The
conclusion was right; the reasoning did not support it.

Both failures share a shape: a plausible reading accepted without the control
that would have tested it. Every tool now records the scene label in the file,
and `score_session.py` exists so a claim can be scored rather than asserted.

---

## 7. What works and what does not

| | Status | Evidence |
|---|---|---|
| Detecting **motion** | **works** | Run-in detected in 2 s, event score z = +18.0 against background p90 1.09 |
| Detecting **a still occupant** | **does not work** | Empty vs asleep AUC 0.51; two standing people read 0.09 |
| **Counting** people | **does not work as designed** | 2 people read *lower* than 1 |
| Absolute level thresholds | **do not work** | 76.3% false positives; scored below chance |
| Event detection + timeout | **best available** | 77.3% balanced, 0% false positives |
| A trained model | **limited** | 64.7% leave-one-session-out; 49.1% — chance — if given all features |

The model result is worth dwelling on. Trained on the three validated features it
reaches 64.7%. Given **all fifteen** features it drops to **49.1%, chance**,
because the magnitude features let it identify the session instead of the
occupant. That is the rank-versus-magnitude split reproduced by a completely
different method.

---

## 8. What the data says to do next

1. **Re-scope from occupancy to motion.** Motion is what this hardware measures.
   A "presence" claim that fails on a sleeping person is not presence detection,
   and the honest deliverable is a motion/activity sensor.
2. **Test two people standing apart**, off the direct line. The 13:41 capture
   had them in the LoS, where bodies occlude each other rather than adding
   independent reflections. This is a 3-minute experiment and it decides whether
   counting is dead or merely restricted to moving people.
3. **Use the lunch empty series** to characterise session-to-session spread
   properly. Two empty sessions was never enough to know the null distribution;
   this gives around 19.
4. **Recalibrate every 10–30 minutes**, or use differential detection only.
5. **Try phase.** Everything here is amplitude-only, following the paper. The
   firmware already computes phase (`csi_component.h`, `CSI_PHASE` branch), and
   respiration and fine-motion sensing in the literature generally depend on it.
   It is the one change with a plausible route to detecting a still person.

---

## 9. Discipline

Quotable from this dataset: **motion detection latency (2 s), event-detector
balanced accuracy (77.3%, 0% false positives) with the labelled session named,
and the leave-one-session-out model figure (64.7%) beside its 50% chance line.**

Not quotable: any presence or occupancy accuracy — empty and occupied do not
separate. Any count. Anything from the two weak-link captures. Anything scored
within a single session, since empty-vs-empty reaches 0.84.
