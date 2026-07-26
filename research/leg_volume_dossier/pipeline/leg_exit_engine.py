#!/usr/bin/env python3
"""LegExitEngine — the causal mechanical exit (owner 2026-07-26: "we have a
bunch of info, let's rebuild the causal mechanical exit"). Composes the day's
validated instruments into ONE layered, streaming, causal decision. No LLM, no
lookahead — deployable in engine_v2 as-is.

Layers (doctrine: ride by default, floor always, intelligence only escalates):
  0. RIDE by default (never-bail is the proven optimum).
  1. CATASTROPHIC FLOOR — px <= running_peak - FLOOR  => EXIT (disaster stop,
     wide; trail50 was the least-bad give-back = cheap tail insurance).
  2. TERMINAL-CONFIRMED EXIT — the gauge is ARMED (vigor FADED + >=2 leg-pure
     anomalies, the −7.96pt/3bar state) AND price has confirmed off the peak
     by >= TIGHT. i.e. exit only when the instruments say the leg is dying AND
     the tape confirms it — not on a bare pullback.

Attach at trade entry; feed one closed 1m bar (features + live leg_age) per
minute; it returns {'action': 'HOLD'|'EXIT', 'reason', 'gauge'}.
Constants are the tested defaults; change only via a new version.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from leg_health_gauge import LegHealthGauge  # noqa: E402

FLOOR = 50.0        # catastrophic give-back from running peak (pts)
TIGHT = 15.0        # confirmation give-back once the gauge is armed (pts)
COST = 1.8          # RT friction charged on any active exit (reporting only)


class LegExitEngine:
    def __init__(self, floor=FLOOR, tight=TIGHT):
        self.floor = floor
        self.tight = tight
        self.gauge = LegHealthGauge()
        self._peak = None
        self._exited = False

    def update(self, px, leg_age, feats):
        """One bar. px = favorable-signed points from entry."""
        if self._exited:
            return dict(action='EXIT', reason='already-exited', gauge=None)
        g = self.gauge.update(leg_age=leg_age, feats=feats)
        self._peak = px if self._peak is None else max(self._peak, px)
        give = self._peak - px
        if give >= self.floor:
            self._exited = True
            return dict(action='EXIT', reason='catastrophic-floor', gauge=g)
        if g['armed'] and give >= self.tight:
            self._exited = True
            return dict(action='EXIT', reason='terminal-confirmed', gauge=g)
        return dict(action='HOLD', reason='ride', gauge=g)
