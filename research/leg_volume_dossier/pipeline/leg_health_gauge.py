#!/usr/bin/env python3
"""LegHealthGauge — the composite gauge as a CAUSAL PER-TRADE DYNAMIC
INDICATOR (owner 2026-07-25: "in order for it to be causal we would need to
measure per trade as a dynamic indicator").

Streaming object: attach at trade entry, feed one bar-frame of features per
minute via update(); it maintains the leg state and leg-pure running
baselines incrementally and emits the gauge each bar:

    g = LegHealthGauge()
    for each minute:
        state = g.update(leg_age=<from live zigzag>, feats={...})
        # state = dict(vigor='ALIVE'|'FADED', sick=<int>,
        #              cell='ALIVE-0'|...|'FADED-2+', armed=bool)

No lookahead anywhere: baselines use bars strictly before the current one,
detectors latch with the same LAG=2 the studies used, and a leg change
(leg_age reset from the pivot tracker) resets detector memory exactly like
the batch walk. The replay harness (tools/replay_equivalence.py) proves
stream == batch on historical episodes before this touches anything live.

Constants mirror the dossier's tested configuration — change only via a new
version, never in place (label immutability).
"""
import statistics as st

SICK_DETECTORS = [('ldist_std', 'lo'), ('price_accel_1b', 'lo'),
                  ('vol_velocity_30', 'lo'), ('lambda_se_21', 'hi'),
                  ('price_velocity_30', 'lo'), ('swing_noise_30', 'hi')]
Z_SICK = 2.0        # dossier: leg-pure tail event
Z_FADE = 1.0        # dossier: conviction fade threshold
LAG = 2             # bars a latch must age before it counts (leading, not instant)
MIN_BASE = 3        # bars of this leg needed before z is meaningful
SICK_ARM = 2        # composite arms at 2+ (interaction study flip point)


class LegHealthGauge:
    BUFFER_BARS = 120        # raw-bar memory for retroactive leg rebuilds

    def __init__(self):
        self._leg_key = None
        self._events = {}        # (feat, tail) -> bar index latched
        self._fade_at = None
        self._bar = -1
        self._buffer = []        # (bar_idx, feats, conv) — legs are declared
                                 # RETROACTIVELY by the pivot tracker; on a leg
                                 # change we rebuild baselines from the bars
                                 # that already belong to the new leg (batch
                                 # semantics), instead of starting blank.

    def _reset_leg(self, new_key):
        self._events = {}
        self._fade_at = None

    def update(self, leg_age, feats):
        """Feed one closed 1m bar. `leg_age` from the live pivot tracker;
        `feats` must carry the detector features + body/bar_range."""
        self._bar += 1
        i = self._bar
        leg_key = i - int(leg_age)               # leg identity = its start bar
        if self._leg_key is None or abs(leg_key - self._leg_key) > 1:
            self._reset_leg(leg_key)
        self._leg_key = leg_key

        # --- baselines: SLIDING window from the buffer, anchored at the
        # CURRENT leg start every bar (leg_age wobbles; batch semantics
        # recompute the window each bar — accumulation drifts stale) --------
        lo_idx = max(0, leg_key)
        window = [(idx, bf, bc) for idx, bf, bc in self._buffer if idx >= lo_idx]

        for fname, tail in SICK_DETECTORS:
            v = feats.get(fname)
            base = [bf[fname] for _, bf, _ in window if fname in bf]
            if v is not None and len(base) >= MIN_BASE:
                sd = st.pstdev(base)
                if sd:
                    z = (v - st.mean(base)) / sd
                    fired = (z >= Z_SICK) if tail == 'hi' else (z <= -Z_SICK)
                    if fired and (fname, tail) not in self._events:
                        self._events[(fname, tail)] = i

        # --- vigor: conviction fade ---------------------------------------
        conv = None
        if feats.get('bar_range') and 'body' in feats:
            conv = feats['body'] / feats['bar_range']
        cbase = [bc for _, _, bc in window if bc is not None]
        if conv is not None and len(cbase) >= MIN_BASE:
            sd = st.pstdev(cbase)
            if sd and (conv - st.mean(cbase)) / sd <= -Z_FADE \
                    and self._fade_at is None:
                self._fade_at = i

        self._buffer.append((i, {k: v for k, v in feats.items()
                                 if v is not None}, conv))
        if len(self._buffer) > self.BUFFER_BARS:
            self._buffer.pop(0)

        # --- emit ----------------------------------------------------------
        active = sorted(f for (f, _t), t0 in self._events.items()
                        if (i - t0) >= LAG)
        sick = len(active)
        faded = self._fade_at is not None and (i - self._fade_at) >= LAG
        vigor = 'FADED' if faded else 'ALIVE'
        cell = f"{vigor}-{'2+' if sick >= SICK_ARM else sick}"
        return dict(vigor=vigor, sick=sick, cell=cell, active=active,
                    armed=(faded and sick >= SICK_ARM))
