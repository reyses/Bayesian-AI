"""
EDA for new entry strategies — run each independently, measure signal strength.

Tests REGIME_FLIP, MTF_EXHAUSTION, EXHAUSTION_BAR, ABSORPTION as standalone
entries on IS data. No CNN, pure physics. Reports WR, PnL, exit breakdown.
"""
import os
import sys
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.features_79d import FEATURE_NAMES_79D

FEAT_IDX = {name: i for i, name in enumerate(FEATURE_NAMES_79D)}
FEATURES_DIR = 'DATA/FEATURES_79D_5s_v2'
ATLAS_1M = 'DATA/ATLAS/1m'

TICK = 0.25
TV = 0.50

# Entry thresholds
STRATEGIES = {
    'REGIME_FLIP': {
        'desc': 'vr < 0.35 + hurst < 0.45 + |z| < 2.0 + vr < 1.0',
        'entry': lambda f: (
            f[FEAT_IDX['1m_variance_ratio']] < 0.35 and
            f[FEAT_IDX['1m_hurst']] < 0.45 and
            abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and
            f[FEAT_IDX['1m_variance_ratio']] < 1.0
        ),
        'direction': lambda f: 'short' if f[FEAT_IDX['1m_z_se']] > 0 else 'long',
        'exit': lambda f, entry_f: (
            f[FEAT_IDX['1m_variance_ratio']] > 0.7 or
            f[FEAT_IDX['1m_hurst']] > 0.55 or
            abs(f[FEAT_IDX['1m_z_se']]) < 0.3
        ),
    },
    'MTF_EXHAUSTION': {
        'desc': '5m decel + 5m vel > 2 + 1m vel > 1 + |z| < 2.0',
        'entry': lambda f: (
            f[FEAT_IDX['5m_acceleration']] < 0 and
            abs(f[FEAT_IDX['5m_velocity']]) > 2.0 and
            abs(f[FEAT_IDX['1m_velocity']]) > 1.0 and
            abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and
            f[FEAT_IDX['1m_variance_ratio']] < 1.0
        ),
        'direction': lambda f: 'short' if f[FEAT_IDX['5m_velocity']] > 0 else 'long',
        'exit': lambda f, entry_f: (
            f[FEAT_IDX['5m_acceleration']] > 0 or  # 5m re-accelerated
            abs(f[FEAT_IDX['1m_velocity']]) < 0.3   # 1m exhausted too
        ),
    },
    'EXHAUSTION_BAR': {
        'desc': 'bar_range > P90 + velocity decelerating + |z| < 2.0',
        'entry': lambda f: (
            f[FEAT_IDX['1m_bar_range']] > 80 and  # high bar range (climax)
            abs(f[FEAT_IDX['1m_acceleration']]) > 2.0 and  # acceleration extreme
            f[FEAT_IDX['1m_acceleration']] * f[FEAT_IDX['1m_velocity']] < 0 and  # decel (opposite signs)
            abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and
            f[FEAT_IDX['1m_variance_ratio']] < 1.0
        ),
        'direction': lambda f: 'short' if f[FEAT_IDX['1m_velocity']] > 0 else 'long',
        'exit': lambda f, entry_f: (
            f[FEAT_IDX['1m_bar_range']] < 30 or  # range compressed (climax over)
            abs(f[FEAT_IDX['1m_velocity']]) < 0.3  # velocity dead
        ),
    },
    'ABSORPTION': {
        'desc': 'vol_rel > 1.5 + bar_range < 20 + wick > 0.5 + |z| < 2.0',
        'entry': lambda f: (
            f[FEAT_IDX['1m_vol_rel']] > 1.5 and  # high volume
            f[FEAT_IDX['1m_bar_range']] < 20 and  # low range (volume absorbed)
            f[FEAT_IDX['1m_wick_ratio']] > 0.5 and  # rejection wicks
            abs(f[FEAT_IDX['1m_z_se']]) < 2.0 and
            f[FEAT_IDX['1m_variance_ratio']] < 1.0
        ),
        'direction': lambda f: 'short' if f[FEAT_IDX['1m_z_se']] > 0 else 'long',
        'exit': lambda f, entry_f: (
            f[FEAT_IDX['1m_vol_rel']] < 0.5 or  # volume died
            f[FEAT_IDX['1m_bar_range']] > 50 or  # range expanded (breakout)
            f[FEAT_IDX['1m_wick_ratio']] < 0.25  # clean bars (absorption done)
        ),
    },
}


def run_strategy(name, strat, feat_files, price_dir):
    """Run one strategy across all days, return trades."""
    trades = []

    for fpath in tqdm(feat_files, desc=name, unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        df = pd.read_parquet(fpath)

        # Load 1m prices for PnL
        price_file = os.path.join(price_dir, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            continue
        prices_df = pd.read_parquet(price_file).sort_values('timestamp')
        price_ts = prices_df['timestamp'].values
        price_close = prices_df['close'].values

        feat_cols = [c for c in df.columns if c != 'timestamp']
        timestamps = df['timestamp'].values

        in_pos = False
        entry_price = 0
        direction = ''
        entry_feat = None
        entry_bar = 0

        # Only check at 1m boundaries
        for i in range(len(df)):
            ts = timestamps[i]
            if int(ts) % 60 >= 5:  # not a 1m boundary
                continue

            feat = np.array([df.iloc[i][c] for c in feat_cols], dtype=np.float32)

            # Find price at this timestamp
            pidx = int(np.searchsorted(price_ts, ts, side='right')) - 1
            if pidx < 0 or pidx >= len(price_close):
                continue
            price = price_close[pidx]

            if in_pos:
                # Check exit
                bars_held = i - entry_bar
                if strat['exit'](feat, entry_feat) or bars_held > 500:
                    if direction == 'long':
                        pnl = (price - entry_price) / TICK * TV
                    else:
                        pnl = (entry_price - price) / TICK * TV

                    exit_reason = 'physics' if bars_held <= 500 else 'max_hold'
                    trades.append({
                        'day': day_name,
                        'dir': direction,
                        'entry_price': entry_price,
                        'pnl': pnl,
                        'held': bars_held,
                        'exit_reason': exit_reason,
                        'entry_79d': entry_feat,
                    })
                    in_pos = False

            elif not in_pos:
                # Check entry
                if strat['entry'](feat):
                    direction = strat['direction'](feat)
                    entry_price = price
                    entry_feat = feat.copy()
                    entry_bar = i
                    in_pos = True

        # Force close at EOD
        if in_pos and len(price_close) > 0:
            price = price_close[-1]
            if direction == 'long':
                pnl = (price - entry_price) / TICK * TV
            else:
                pnl = (entry_price - price) / TICK * TV
            trades.append({
                'day': day_name, 'dir': direction,
                'entry_price': entry_price, 'pnl': pnl,
                'held': len(df) - entry_bar, 'exit_reason': 'eod',
                'entry_79d': entry_feat,
            })

    return trades


def report(name, strat, trades):
    """Print strategy report."""
    n = len(trades)
    if n == 0:
        print(f'\n{name}: 0 trades')
        return

    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = sum(t['pnl'] for t in trades)
    avg = total / n

    print(f'\n{"="*60}')
    print(f'{name}: {strat["desc"]}')
    print(f'{"="*60}')
    print(f'  Trades: {n} | WR: {wins/n*100:.0f}% | Total: ${total:,.0f} | Avg: ${avg:.1f}/trade')
    print(f'  Per day: {n/277:.1f} trades/day | ${total/277:.0f}/day')

    # Hold time
    holds = [t['held'] for t in trades]
    print(f'  Hold: P25={np.percentile(holds,25):.0f} P50={np.percentile(holds,50):.0f} P75={np.percentile(holds,75):.0f} bars')

    # Exit reasons
    print(f'  Exits:')
    for reason, count in Counter(t['exit_reason'] for t in trades).most_common():
        sub = [t for t in trades if t['exit_reason'] == reason]
        w = sum(1 for t in sub if t['pnl'] > 0)
        tot = sum(t['pnl'] for t in sub)
        print(f'    {reason:<12} {count:>5} {w/count*100:>4.0f}% WR ${tot:>9,.0f}')

    # Winners vs losers
    w_pnl = [t['pnl'] for t in trades if t['pnl'] > 0]
    l_pnl = [t['pnl'] for t in trades if t['pnl'] <= 0]
    if w_pnl and l_pnl:
        print(f'  Avg winner: ${np.mean(w_pnl):.1f} | Avg loser: ${np.mean(l_pnl):.1f} | Ratio: {abs(np.mean(w_pnl)/np.mean(l_pnl)):.1f}x')

    # Direction
    long_t = [t for t in trades if t['dir'] == 'long']
    short_t = [t for t in trades if t['dir'] == 'short']
    if long_t:
        lw = sum(1 for t in long_t if t['pnl'] > 0)
        print(f'  Long:  {len(long_t)} trades, {lw/len(long_t)*100:.0f}% WR, ${sum(t["pnl"] for t in long_t):,.0f}')
    if short_t:
        sw = sum(1 for t in short_t if t['pnl'] > 0)
        print(f'  Short: {len(short_t)} trades, {sw/len(short_t)*100:.0f}% WR, ${sum(t["pnl"] for t in short_t):,.0f}')


def main():
    # IS files only (2025)
    feat_files = sorted([f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                         if '2025_' in os.path.basename(f)])
    print(f'IS days: {len(feat_files)}')

    for name, strat in STRATEGIES.items():
        trades = run_strategy(name, strat, feat_files, ATLAS_1M)
        report(name, strat, trades)

    # Save results
    os.makedirs('reports/findings', exist_ok=True)
    print(f'\nDone.')


if __name__ == '__main__':
    main()
