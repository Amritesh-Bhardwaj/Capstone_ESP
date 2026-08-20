"""Presence from motion *events* rather than an absolute level.

Why events
----------
An absolute threshold on the presence level cannot work on this link for long.
Measured 2026-08-20: the empty-room baseline drifted **+0.578 in 70 minutes**,
while the whole occupancy signal is about 0.5. Seventy minutes after
calibration the empty room read *higher* (median 0.576) than the occupied room
(0.541), so no fixed enter/exit pair can separate them, and an adaptive
baseline just absorbs whoever is standing there.

A difference between a short and a long trailing window cancels any drift slow
compared to the short window. On the labelled session that gives a run-in event
at **z = +18.0** against a background p90 of **1.09** — a margin no level-based
detector came close to.

What this can and cannot tell you
---------------------------------
It marks the room occupied two ways: a motion contrast, and an absolute level
high enough that drift cannot explain it. The second exists because a contrast
alone never starts the timeout for someone who was already sitting still when
the detector began, so they are reported absent forever.

The motion half detects *motion*, not direction. Walking out produced z = +2.1 and running in
z = +18.0; both are positive, because both are movement. Occupancy is therefore
inferred the way a PIR sensor does it: motion marks the room occupied, and it
stays occupied until ``timeout_s`` passes with no motion.

That makes the sedentary failure **explicit and tunable** instead of accidental.
A person who sits perfectly still for longer than the timeout is reported
absent. On this hardware that is not a bug to be fixed in software: a still
occupant is genuinely near-indistinguishable from an empty room (`RESULTS.md`
§9.9, §9.12).
"""

from __future__ import annotations

import collections

import numpy as np

# Defaults chosen on the 2026-08-20 labelled session; see score_session.py to
# re-tune against your own ground truth rather than trusting these.
SHORT_S = 6.0
LONG_S = 45.0
ENTER_Z = 3.0
TIMEOUT_S = 45.0
# A level this far above the calibrated empty room cannot be drift. Derived,
# not guessed: the empty room's p95 at calibration was 1.12 and a *constant*
# scene wandered by 1.35 level units over five hours, so 1.12 + 1.35 = 2.47.
# Measured against the labelled recordings, a level above 2.5 occurs in 0.7%
# and 0.0% of windows in the two empty captures, and in occupied ones it does
# occur -- so crossing it is evidence even when nobody has moved sharply.
CONFIRM_LEVEL = 2.5


class EventDetector:
    """Motion events by short-vs-long window contrast, plus an occupancy timeout.

    ``update`` takes one presence-level sample and the seconds since the last
    one, and returns ``(present, z, event)``.
    """

    def __init__(
        self,
        dt_s: float,
        short_s: float = SHORT_S,
        long_s: float = LONG_S,
        enter_z: float = ENTER_Z,
        timeout_s: float = TIMEOUT_S,
        confirm_level: float | None = CONFIRM_LEVEL,
    ):
        if dt_s <= 0:
            raise ValueError("dt_s must be positive")
        if long_s <= short_s:
            raise ValueError("long_s must exceed short_s")
        self.dt_s = dt_s
        self.enter_z = enter_z
        self.timeout_s = timeout_s
        self.confirm_level = confirm_level
        self.n_short = max(int(round(short_s / dt_s)), 2)
        self.n_long = max(int(round(long_s / dt_s)), self.n_short + 2)
        self.short: collections.deque = collections.deque(maxlen=self.n_short)
        self.long: collections.deque = collections.deque(maxlen=self.n_long)
        self.present = False
        self.since_motion = float("inf")
        self.events = 0
        self.last_z = 0.0
        self.last_confirmed = False

    def update(self, level: float, dt_s: float | None = None) -> tuple:
        step = self.dt_s if dt_s is None else dt_s
        self.short.append(level)
        self.long.append(level)

        z = 0.0
        if len(self.long) >= self.n_long:
            base = float(np.median(self.long))
            # Interquartile range, not std: a motion burst inside the long
            # window would inflate a std and mask the very event being looked
            # for.
            spread = float(np.percentile(self.long, 75) - np.percentile(self.long, 25))
            z = (float(np.median(self.short)) - base) / max(spread, 1e-3)
        self.last_z = z

        # Two ways to mark the room occupied. The contrast catches movement;
        # the absolute level catches somebody who is simply *there* without
        # having moved sharply -- otherwise a person sitting still when the
        # detector starts is reported absent indefinitely, because the timeout
        # never gets started. The level is only trusted above CONFIRM_LEVEL,
        # which sits above the whole measured drift envelope.
        confirmed = self.confirm_level is not None and level >= self.confirm_level
        event = bool(z >= self.enter_z or confirmed)
        self.last_confirmed = confirmed
        if event:
            self.events += 1
            self.since_motion = 0.0
        else:
            self.since_motion += step

        self.present = self.since_motion < self.timeout_s
        return self.present, z, event

    def state(self) -> dict:
        return {
            "present": self.present,
            "z": round(self.last_z, 3),
            "events": self.events,
            "since_motion_s": (None if self.since_motion == float("inf")
                               else round(self.since_motion, 1)),
            "timeout_s": self.timeout_s,
            "enter_z": self.enter_z,
            "confirm_level": self.confirm_level,
            "confirmed_by_level": self.last_confirmed,
        }


def run(levels, dt_s: float, **kwargs) -> tuple:
    """Offline convenience: returns (present[], z[], event_indices)."""
    det = EventDetector(dt_s, **kwargs)
    present, zs, events = [], [], []
    for i, v in enumerate(levels):
        p, z, e = det.update(v)
        present.append(p)
        zs.append(z)
        if e:
            events.append(i)
    return np.array(present), np.array(zs), events
