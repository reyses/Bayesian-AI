"""Per-tier threshold tuner — finds winner vs loser separation on honest features.

For each tier, computes the distribution of key features at ENTRY for
winners vs losers. Suggests threshold changes that would improve WR.

Usage:
    python tools/tune_tier_thresholds.py [tier_name]

If tier_name omitted, runs all tiers.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.nightmare_blended import (
    _1M_OFFSET, _Z, _VR,
    _1M_VELOCITY_IDX, _1H_VELOCITY_IDX, _1H_Z_IDX,
    _1M_P_CENTER_IDX, _5M_WICK_IDX, _15M_WICK_IDX,
    _1M_WICK_IDX, _5M_VELOCITY_IDX, _5M_ACCEL_IDX,
    _1M_HURST_IDX, _1M_VOL_REL_IDX, _1M_DMI_IDX,
    _5M_BAR_RANGE_IDX, _1M_REVERSION_IDX,
)

TRADES_PKL = 'training/output/trades/blended_is.pkl'

# Features to analyze per tier
FEATURE_INDICES = {
    '1m_z_se':       _1M_OFFSET + _Z,
    '1m_vr':         _1M_OFFSET + _VR,
    '1m_velocity':   _1M_VELOCITY_IDX,
    '1m_wick':       _1M_WICK_IDX,
    '1m_vol_rel':    _1M_VOL_REL_IDX,
    '1m_hurst':      _1M_HURST_IDX,
    '1m_p_center':   _1M_P_CENTER_IDX,
    '1m_reversion':  _1M_REVERSION_IDX,
    '5m_wick':       _5M_WICK_IDX,
    '5m_velocity':   _5M_VELOCITY_IDX,
    '5m_accel':      _5M_ACCEL_IDX,
    '5m_bar_range':  _5M_BAR_RANGE_IDX,
    '15m_wick':      _15M_WICK_IDX,
    '1h_z':          _1H_Z_IDX,
    '1h_velocity':   _1H_VELOCITY_IDX,
}


def get_feature(trade, name):
    """Extract a feature value from trade's entry_features."""
    feat = trade.get('entry_features', trade.get('entry_79d', []))
    if not feat:
        return None
    idx = FEATURE_INDICES[name]
    if idx >= len(feat):
        return None
    return float(feat[idx])


def analyze_tier(tier, trades):
    """Side-by-side winner vs loser feature stats for one tier."""
    sub = [t for t in trades if t.get('entry_tier') == tier and not t.get('is_chain', False)]
    if len(sub) < 20:
        return
    winners = [t for t in sub if t['pnl'] > 0]
    losers = [t for t in sub if t['pnl'] <= 0]
    wr = len(winners) / len(sub) * 100
    total = sum(t['pnl'] for t in sub)
    avg = total / len(sub)

    print(f'\n{"="*78}')
    print(f'TIER: {tier}  ({len(sub)} trades, WR {wr:.0f}%, ${total:+,.0f}, avg ${avg:+.1f})')
    print(f'{"="*78}')
    print(f'{"Feature":<14} {"W_med":>8} {"L_med":>8} {"Diff":>7} {"|W|_q75":>8} {"|L|_q75":>8} {"Separation"}')
    print('-' * 78)

    results = []
    for name in FEATURE_INDICES:
        w_vals = [get_feature(t, name) for t in winners]
        w_vals = [v for v in w_vals if v is not None]
        l_vals = [get_feature(t, name) for t in losers]
        l_vals = [v for v in l_vals if v is not None]
        if len(w_vals) < 10 or len(l_vals) < 10:
            continue

        w_med = np.median(w_vals)
        l_med = np.median(l_vals)
        diff = w_med - l_med

        # Signed abs separation: how different are winners from losers?
        w_abs = np.median(np.abs(w_vals))
        l_abs = np.median(np.abs(l_vals))
        abs_diff = w_abs - l_abs

        # Standardized diff: Cohen's d-ish
        sd_pool = np.sqrt((np.std(w_vals)**2 + np.std(l_vals)**2) / 2)
        d = diff / max(sd_pool, 1e-9)

        flag = '***' if abs(d) > 0.3 else ' **' if abs(d) > 0.2 else '  *' if abs(d) > 0.1 else ''
        results.append((name, w_med, l_med, diff, w_abs, l_abs, d, flag))

    # Sort by |d| descending
    results.sort(key=lambda r: -abs(r[6]))
    for name, w_med, l_med, diff, w_abs, l_abs, d, flag in results:
        print(f'{name:<14} {w_med:>+8.3f} {l_med:>+8.3f} {diff:>+7.3f} '
              f'{w_abs:>8.2f} {l_abs:>8.2f}   d={d:>+5.2f} {flag}')

    # For top-separation features, show WR at different thresholds
    top3 = results[:3]
    print(f'\n  Top 3 separators — WR by threshold:')
    for name, _, _, _, _, _, d, _ in top3:
        vals = [(get_feature(t, name), t['pnl']) for t in sub]
        vals = [(v, p) for v, p in vals if v is not None]
        if not vals:
            continue
        arr = np.array(vals)
        xs = arr[:, 0]
        ps = arr[:, 1]

        print(f'  {name}:')
        # Use absolute value thresholds (most features are signed)
        abs_xs = np.abs(xs)
        for thr in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            mask = abs_xs > thr
            if mask.sum() < 20:
                continue
            n = mask.sum()
            wr_t = (ps[mask] > 0).sum() / n * 100
            pnl_t = ps[mask].sum()
            avg_t = pnl_t / n
            print(f'    |{name}| > {thr}: {n:>4} trades, WR {wr_t:>3.0f}%, avg ${avg_t:>+.1f}, total ${pnl_t:>+,.0f}')


def main():
    tier_filter = sys.argv[1] if len(sys.argv) > 1 else None

    if not os.path.exists(TRADES_PKL):
        print(f'No trades: {TRADES_PKL}')
        return

    with open(TRADES_PKL, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades):,} trades')

    tiers = ['FADE_CALM', 'RIDE_AGAINST', 'KILL_SHOT', 'FADE_AGAINST',
             'MTF_BREAKOUT', 'CASCADE', 'MTF_EXHAUSTION', 'FREIGHT_TRAIN']
    if tier_filter:
        tiers = [t for t in tiers if tier_filter.upper() in t]

    for tier in tiers:
        analyze_tier(tier, trades)


if __name__ == '__main__':
    main()
