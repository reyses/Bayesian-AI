"""Does the microstructure BEFORE a trend differ from before chop? (entry filter hunt)

Moises' question: how many wall-footprints appear before a trend starts, and how
do wicks/bodies look through the buildup. Operationalized:

- Zigzag 40t on 1m -> all pivots. For each pivot, classify the FOLLOWING leg:
    TREND if next leg extent >= BIG ticks, CHOP if < BIG.
- Window = W bars BEFORE the pivot (the buildup into the turn).
- Per-bar candle/volume footprints in that window:
    body_frac = |close-open|/range        (1=marubozu, 0=doji)
    wick_frac = (upper+lower wick)/range
    vol_rel   = volume / trailing-median volume (participation, TF-consistent)
    absorption = small wick (wick_frac<0.35) AND high vol (vol_rel>1.3)
    rejection  = big wick   (wick_frac>0.65)
- Compare pre-TREND vs pre-CHOP window aggregates. Null = the two labels are
  exchangeable (bootstrap the difference). OOS 2024 vs 2025.

If pre-TREND buildups carry a distinct footprint signature, that's the entry
filter momentum-alone lacked.
"""
import argparse
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'research', 'level_hold', 'tools'))
from level_hold_study import atlas  # noqa: E402
from pivot_level_proximity import zigzag_pivots  # noqa: E402

TICK = 0.25
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s); lines.append(s)


def candle_feats(o, h, l, c, v):
    rng = np.maximum(h - l, TICK)
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    body_frac = body / rng
    wick_frac = (upper + lower) / rng
    vmed = _trailing_median(v, 30)
    vol_rel = v / np.maximum(vmed, 1e-9)
    return body_frac, wick_frac, vol_rel


def _trailing_median(a, W):
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        s = max(0, i - W + 1)
        out[i] = np.median(a[s:i + 1])
    return out


def window_agg(bf, wf, vr, s, e):
    """Aggregate footprints over bars [s,e)."""
    if e - s < 3:
        return None
    absorption = ((wf[s:e] < 0.35) & (vr[s:e] > 1.3)).mean()
    rejection = (wf[s:e] > 0.65).mean()
    return {
        'body_frac': np.nanmean(bf[s:e]),
        'wick_frac': np.nanmean(wf[s:e]),
        'vol_rel': np.nanmean(vr[s:e]),
        'absorption': absorption,
        'rejection': rejection,
    }


def collect(days, big_ticks, W):
    trend, chop = [], []
    for day in days:
        try:
            d = atlas(day, '1m')
        except Exception:
            continue
        o, h, l, c, v = (d[x].to_numpy().astype(float)
                         for x in ('open', 'high', 'low', 'close', 'volume'))
        if len(c) < W + 20:
            continue
        piv = zigzag_pivots(c, 40)
        if len(piv) < 3:
            continue
        bf, wf, vr = candle_feats(o, h, l, c, v)
        for i in range(1, len(piv) - 1):
            p = piv[i]
            nxt = piv[i + 1]
            leg_ext = abs(c[nxt] - c[p]) / TICK
            if p - W < 0:
                continue
            agg = window_agg(bf, wf, vr, p - W, p)
            if agg is None:
                continue
            (trend if leg_ext >= big_ticks else chop).append(agg)
    return trend, chop


def summarize(tag, trend, chop):
    keys = ['body_frac', 'wick_frac', 'vol_rel', 'absorption', 'rejection']
    log(f"\n[{tag}] pre-TREND n={len(trend)} | pre-CHOP n={len(chop)}")
    log(f"{'feature':<12}{'preTREND':>10}{'preCHOP':>10}{'delta':>9}{'boot p':>9}")
    rng = np.random.default_rng(0)
    for k in keys:
        t = np.array([x[k] for x in trend])
        ch = np.array([x[k] for x in chop])
        d = t.mean() - ch.mean()
        # bootstrap null: shuffle labels
        pool = np.concatenate([t, ch])
        nt = len(t)
        null = np.empty(2000)
        for b in range(2000):
            idx = rng.permutation(len(pool))
            null[b] = pool[idx[:nt]].mean() - pool[idx[nt:]].mean()
        p = (np.abs(null) >= abs(d)).mean()
        log(f"{k:<12}{t.mean():>10.3f}{ch.mean():>10.3f}{d:>+9.3f}{p:>9.3f}")


def vol_filter(tag, trend, chop):
    """Actionable: P(trend follows | buildup vol_rel quartile). Base rate vs
    top-quartile-volume lift."""
    allv = np.array([x['vol_rel'] for x in trend] + [x['vol_rel'] for x in chop])
    y = np.array([1] * len(trend) + [0] * len(chop))
    qs = np.percentile(allv, [25, 50, 75])
    base = y.mean()
    log(f"\n[{tag}] P(trend | buildup vol_rel quartile)  base rate {base:.3f}")
    edges = [-np.inf] + list(qs) + [np.inf]
    labels = ['Q1 low', 'Q2', 'Q3', 'Q4 high']
    for i in range(4):
        m = (allv >= edges[i]) & (allv < edges[i + 1])
        if m.sum() > 0:
            log(f"  {labels[i]:<8} vol_rel<{edges[i+1] if i<3 else float('inf'):.2f}  "
                f"P(trend)={y[m].mean():.3f}  n={m.sum()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--big', type=int, default=150, help='trend leg threshold (ticks)')
    ap.add_argument('--window', type=int, default=30, help='pre-onset window (1m bars)')
    args = ap.parse_args()
    alld = sorted(os.path.basename(f).replace('.parquet', '')
                  for f in glob.glob(os.path.join(_REPO, 'DATA', 'ATLAS', '1m', '*.parquet')))
    for yr in ('2024', '2025'):
        days = [d for d in alld if d.startswith(yr)]
        tr, ch = collect(days, args.big, args.window)
        summarize(f"{yr} OOS" if yr == '2025' else yr, tr, ch)
        vol_filter(f"{yr} OOS" if yr == '2025' else yr, tr, ch)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'pretrend_footprint.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
