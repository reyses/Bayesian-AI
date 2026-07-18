"""Do afternoon swing extremes land on morning-frozen levels? (long-memory claim)

The user's actual claim (Figure_3): band levels frozen in the morning stay
relevant HOURS later — later swings terminate at/near them. This is distinct
from the falsified first-touch bounce claim.

Design (causal):
- Formation: all bars from day start through 11:30 ET. Levels = extremes of
  the fast (10-min OLS +-2s on 5s) and slow (60-min OLS +-2s on 1m) band
  lines over the formation window + their values at the freeze, deduped
  within 4 ticks.
- Evaluation: 11:30 -> 16:00 ET. Swing pivots via fixed-reversal zigzag on
  1m closes (thresholds 20 and 40 ticks).
- Metric: fraction of pivots within {4, 8, 12} ticks of the nearest frozen
  level; median nearest-distance.
- Null: 400 Monte-Carlo draws shifting the WHOLE level set by a uniform
  random offset (+-10..80 ticks, sign random) — preserves the set's internal
  spacing/cluster structure, destroys placement. Empirical p-value.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
from level_hold_study import atlas, rolling_ols_bands, wilson  # noqa: E402

TICK = 0.25
REPORT_DIR = os.path.join(_REPO, 'research', 'level_hold', 'reports')
lines = []


def log(s):
    print(s)
    lines.append(s)


def zigzag_pivots(close, thr_ticks):
    """Fixed-reversal zigzag; returns interior pivot indices.
    Two-phase: establish the first confirmed direction from running max/min,
    then alternate extreme-tracking with threshold reversals."""
    thr = thr_ticks * TICK
    n = len(close)
    piv = []
    hi_i = lo_i = 0
    direction, ext_i, start = 0, 0, n
    for i in range(1, n):
        c = close[i]
        if c > close[hi_i]:
            hi_i = i
        if c < close[lo_i]:
            lo_i = i
        if close[hi_i] - c >= thr:
            direction, ext_i, start = -1, (lo_i if lo_i > hi_i else i), i + 1
            break
        if c - close[lo_i] >= thr:
            direction, ext_i, start = 1, (hi_i if hi_i > lo_i else i), i + 1
            break
    for j in range(start, n):
        c = close[j]
        if direction > 0:
            if c > close[ext_i]:
                ext_i = j
            elif close[ext_i] - c >= thr:
                piv.append(ext_i)
                direction, ext_i = -1, j
        else:
            if c < close[ext_i]:
                ext_i = j
            elif c - close[ext_i] >= thr:
                piv.append(ext_i)
                direction, ext_i = 1, j
    return np.array(sorted(set(piv)), dtype=int)


def day_levels_and_pivots(day, freeze_utc_h=16.5, end_utc_h=21.0, thr=20):
    d5 = atlas(day, '5s')
    d1 = atlas(day, '1m')
    c5, ts5 = d5['close'].to_numpy(), d5['timestamp'].to_numpy()
    c1, ts1 = d1['close'].to_numpy(), d1['timestamp'].to_numpy()
    # ATLAS session files start ~23:00 UTC the PRIOR evening; anchor the
    # freeze/end clocks on the LAST bar's calendar day (the RTH day)
    day0 = ts5[-1] - (ts5[-1] % 86400)
    t_freeze = day0 + freeze_utc_h * 3600
    t_end = day0 + end_utc_h * 3600

    up_f, lo_f, _ = rolling_ols_bands(c5, 120)
    up_s, lo_s, _ = rolling_ols_bands(c1, 60)
    m5 = ts5 < t_freeze
    m1 = ts1 < t_freeze
    if m5.sum() < 200 or m1.sum() < 70:
        return None, None
    cands = [np.nanmax(up_f[m5]), np.nanmin(lo_f[m5]),
             np.nanmax(up_s[m1]), np.nanmin(lo_s[m1]),
             up_f[m5][-1], lo_f[m5][-1], up_s[m1][-1], lo_s[m1][-1]]
    cands = sorted(c for c in cands if np.isfinite(c))
    levels = []
    for c in cands:
        if not levels or c - levels[-1] > 4 * TICK:
            levels.append(c)
    ev = (ts1 >= t_freeze) & (ts1 <= t_end)
    pv_idx = zigzag_pivots(c1[ev], thr)
    return np.array(levels), c1[ev][pv_idx]


def frac_within(pivots, levels, tol_ticks):
    d = np.abs(pivots[:, None] - levels[None, :]).min(axis=1)
    return (d <= tol_ticks * TICK).mean(), np.median(d) / TICK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=str,
                    default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26,2024_02_27")
    ap.add_argument('--thr', type=int, default=20, help='zigzag reversal (ticks)')
    ap.add_argument('--nulls', type=int, default=400)
    args = ap.parse_args()
    days = [d.strip() for d in args.days.split(',')]
    rng = np.random.default_rng(0)

    all_p, all_l = [], []
    for day in days:
        levels, pivots = day_levels_and_pivots(day, thr=args.thr)
        if levels is None or len(pivots) == 0:
            log(f"{day}: skipped (insufficient data)")
            continue
        all_p.append(pivots)
        all_l.append(levels)
        log(f"{day}: {len(levels)} frozen levels, {len(pivots)} afternoon pivots "
            f"(zigzag {args.thr} ticks)")

    log('')
    for tol in (4, 8, 12):
        real_fr, real_n = [], 0
        null_fr = np.zeros(args.nulls)
        for pv, lv in zip(all_p, all_l):
            f, _ = frac_within(pv, lv, tol)
            real_fr.append(f * len(pv))
            real_n += len(pv)
            for k in range(args.nulls):
                off = rng.uniform(10, 80) * TICK * rng.choice([-1, 1])
                fn, _ = frac_within(pv, lv + off, tol)
                null_fr[k] += fn * len(pv)
        real = sum(real_fr) / real_n
        null = null_fr / real_n
        p_emp = float((null >= real).mean())
        log(f"tol {tol:>2} ticks: real {real:.3f} of {real_n} pivots near a level | "
            f"null {null.mean():.3f} [{np.percentile(null, 2.5):.3f}, "
            f"{np.percentile(null, 97.5):.3f}] | p_emp = {p_emp:.3f}")
    meds = [frac_within(pv, lv, 8)[1] for pv, lv in zip(all_p, all_l)]
    log(f"median pivot->nearest-level distance per day (ticks): "
        f"{[f'{m:.0f}' for m in meds]}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f'pivot_level_proximity_thr{args.thr}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
