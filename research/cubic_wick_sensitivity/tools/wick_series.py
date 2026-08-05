#!/usr/bin/env python3
"""Wick/body-sensitive input series for the pocket-dojo cubic — RESEARCH ONLY.

Owner proposal (2026-07-30 journal): "make [the cubic] sensitive to wick and
body — this will have the hidden proportion of internal structure." The live
cubic (research/dojo_forge/tools/cubic_regression.py, called by pocket_dojo
with an 8-bar/1m window) fits CLOSES only and is blind to the wick shapes the
owner reads by eye (OWNER_PROCESS.md: "buyers struggling" = shrinking bodies +
growing upper wicks; the 2025_06_05 bar-880 fake dip = long lower wick).

This module builds DERIVED series to feed the *unchanged* cubic machinery
(imported from cubic_regression.py — no copy, no modification), so the live
tool is untouched. Two design schools for direction (a), which make OPPOSITE
sign predictions and are both tested:

  REJECTION school  p_rej = close + k*(lower_wick - upper_wick)
    A wick is where price was REJECTED: a long upper wick means sellers capped
    the bar (bearish -> push the input DOWN); a long lower wick means buyers
    defended (bullish -> push the input UP). This matches the owner's stated
    reads on both tell bars.

  EXCURSION school  p_exc = (1-w)*close + w*(high+low)/2
    A wick is where price TRADED: blend toward the bar midpoint. Note this
    moves the input TOWARD the wick side — the opposite sign. On a growing-
    upper-wick exhaustion top it makes the input MORE bullish. Included as the
    naive "obvious" candidate precisely so the data can arbitrate.

Direction (b) — additive companion metrics (price cubic untouched, read
alongside it the way volume-divergence was read on 2025_08_24):

  wick_bias = (lower_wick - upper_wick) / range   in [-1, +1], bullish positive
  body_frac = |close - open| / range              in [0, 1], conviction
"""
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_DOJO_TOOLS = os.path.join(_REPO, 'research', 'dojo_forge', 'tools')
if _DOJO_TOOLS not in sys.path:
    sys.path.insert(0, _DOJO_TOOLS)
import cubic_regression as _cub                    # noqa: E402  (reuse, don't fork)

# Window/grid identical to the live tool (pocket_dojo.py: CUBIC_W=8 on 1m bars)
# so every comparison is apples-to-apples with what the owner actually sees.
CUBIC_W = 8            # bars in the trailing cubic fit (pocket_dojo.CUBIC_W)
BAR_SECONDS = 60       # 1m bars (pocket_dojo feeds the 1m close series)

# Rejection-school gain grid: k is dimensionless (points of input shift per
# point of net wick). k=1.0 = full wick counter-weight; k=0.5 = half. Grid kept
# tiny on purpose — this is a 2-tell-bar comparison, not a tuning exercise, and
# a k tuned to two bars would be overfit by construction.
K_REJECTION_GRID = (0.5, 1.0)
W_EXCURSION = 0.5      # excursion blend: halfway between close and (H+L)/2


def _wicks(o, h, l, c):
    """Per-bar upper/lower wick lengths (points, >=0) and range."""
    o, h, l, c = (np.asarray(a, float) for a in (o, h, l, c))
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    return h - body_hi, body_lo - l, h - l


def rejection_price(o, h, l, c, k):
    """Close counter-adjusted by net wick: long lower wick lifts the input
    (buyers defended), long upper wick sinks it (sellers capped)."""
    uw, lw, _ = _wicks(o, h, l, c)
    return np.asarray(c, float) + k * (lw - uw)


def excursion_price(h, l, c, w=W_EXCURSION):
    """Blend of close toward the bar midpoint (H+L)/2 — the naive candidate."""
    h, l, c = (np.asarray(a, float) for a in (h, l, c))
    return (1.0 - w) * c + w * 0.5 * (h + l)


def wick_bias(o, h, l, c):
    """(lower_wick - upper_wick)/range in [-1,+1]; bullish positive; 0 on
    zero-range bars (a dead bar carries no shape information)."""
    uw, lw, rng = _wicks(o, h, l, c)
    return np.divide(lw - uw, rng, out=np.zeros_like(rng), where=rng > 0)


def body_frac(o, h, l, c):
    """|close-open|/range in [0,1] — bar conviction; 0 on zero-range bars."""
    o, c = np.asarray(o, float), np.asarray(c, float)
    _, _, rng = _wicks(o, h, l, c)
    return np.divide(np.abs(c - o), rng, out=np.zeros_like(rng), where=rng > 0)


def rolling_cubic(series, window=CUBIC_W, bar_seconds=BAR_SECONDS):
    """(value, slope, curvature) per bar via the UNCHANGED live machinery."""
    return _cub.rolling(series, window, bar_seconds)


def rolling_mean(series, window=CUBIC_W):
    """Trailing mean over the same window — the low-drama smoother for the
    direction-(b) companion metrics (a cubic has 4 DOF over 8 points and
    amplifies noise on bounded per-bar ratios; the mean is the honest read)."""
    s = np.asarray(series, float)
    out = np.full(len(s), np.nan)
    for i in range(window - 1, len(s)):
        out[i] = s[i - window + 1:i + 1].mean()
    return out


if __name__ == '__main__':
    # self-test: a pure hammer bar (long lower wick) must LIFT p_rej above
    # close, SINK p_exc below close, and score wick_bias near +1.
    o, h, l, c = 100.0, 100.5, 96.0, 100.25
    pr = rejection_price([o], [h], [l], [c], k=1.0)[0]
    pe = excursion_price([h], [l], [c])[0]
    wb = wick_bias([o], [h], [l], [c])[0]
    print(f'hammer: close={c} p_rej={pr:.2f} (up? {pr > c}) '
          f'p_exc={pe:.2f} (down? {pe < c}) wick_bias={wb:+.2f}')
    assert pr > c and pe < c and wb > 0.8
    print('self-test OK')
