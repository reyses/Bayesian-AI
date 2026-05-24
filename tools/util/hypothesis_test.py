"""
Hypothesis Test — apply candidate filters and measure $/day impact.

Runs the baseline engine on IS dataset multiple times with different
filter configurations. Compares $/day, win rate, total trades.

Hypotheses tested (from feature_response_surface.txt):
1. CHOP_FILTER:    block all entries when 15m_wick_ratio > 0.62
                    OR 1h_wick_ratio > 0.60
2. KILLSHOT_CENTER: block KILL_SHOT when 1m_p_at_center > 0.5
3. NO_MTF_BREAKOUT: disable MTF_BREAKOUT entirely (43% WR, losing tier)
4. ALL: combine all three

Output: reports/findings/hypothesis_test.txt

Usage:
    python tools/hypothesis_test.py
    python tools/hypothesis_test.py --days 60   # faster
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import (
    BlendedEngine,
    _15M_WICK_IDX, _5M_WICK_IDX, _1M_P_CENTER_IDX,
)

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'reports/findings'

# Filter thresholds (from feature response surface)
CHOP_15M_WICK_MAX = 0.62
CHOP_1H_WICK_MAX = 0.60
KILLSHOT_P_CENTER_MAX = 0.5
LEVEL_MIN_HEADROOM_Z = 0.5  # min z-space to nearest S/R level

# 1h_wick_ratio is at index 86 (helper_start=72 + TF4*3 + 2)
_1H_WICK_IDX = 86

# z_high/z_low indices per TF (within the 12 core features, indices 10 and 11)
# 1h tf=4: 4*12 + 10 = 58 (z_high), 4*12 + 11 = 59 (z_low)
# 1D tf=5: 5*12 + 10 = 70 (z_high), 5*12 + 11 = 71 (z_low)
_1H_Z_HIGH = 58
_1H_Z_LOW = 59
_1D_Z_HIGH = 70
_1D_Z_LOW = 71


class FilteredEngine(BlendedEngine):
    """BlendedEngine with optional entry filters for hypothesis testing."""

    def __init__(self, chop_filter=False, killshot_center=False,
                 no_mtf_breakout=False, gravity_well=False, **kwargs):
        super().__init__(**kwargs)
        self._chop_filter = chop_filter
        self._killshot_center = killshot_center
        self._no_mtf_breakout = no_mtf_breakout
        self._gravity_well = gravity_well

    def _classify_full_tier(self, feat, z):
        # Apply chop filter — block all entries in chop regimes
        if self._chop_filter:
            wick_15m = feat[_15M_WICK_IDX]
            wick_1h = feat[_1H_WICK_IDX]
            if wick_15m > CHOP_15M_WICK_MAX or wick_1h > CHOP_1H_WICK_MAX:
                return None, None, False

        direction, tier, flipped = super()._classify_full_tier(feat, z)

        # Block KILL_SHOT when too close to center
        if self._killshot_center and tier == 'KILL_SHOT':
            if feat[_1M_P_CENTER_IDX] > KILLSHOT_P_CENTER_MAX:
                return None, None, False

        # Disable MTF_BREAKOUT
        if self._no_mtf_breakout and tier == 'MTF_BREAKOUT':
            return None, None, False

        # Gravity well: block trades against gravity (low headroom in direction)
        # Uses 1h_z_high/z_low and 1D_z_high/z_low as gravity well boundaries
        if self._gravity_well and direction is not None:
            # Current z relative to 1h regression
            # If long, headroom = 1h_z_high - current_1h_z (room until ceiling)
            # If short, headroom = current_1h_z - 1h_z_low (room until floor)
            # We use the 1h and 1D wells as the "tested gravity field"
            cur_z_1h = feat[48]  # 1h_z_se index
            cur_z_1d = feat[60]  # 1D_z_se index
            z_high_1h = feat[_1H_Z_HIGH]
            z_low_1h = feat[_1H_Z_LOW]
            z_high_1d = feat[_1D_Z_HIGH]
            z_low_1d = feat[_1D_Z_LOW]

            if direction == 'long':
                # Going up: how much room before hitting the well's ceiling
                head_1h = z_high_1h - cur_z_1h
                head_1d = z_high_1d - cur_z_1d
            else:
                # Going down: how much room before hitting the well's floor
                head_1h = cur_z_1h - z_low_1h
                head_1d = cur_z_1d - z_low_1d

            min_head = min(head_1h, head_1d)
            if min_head < LEVEL_MIN_HEADROOM_Z:
                return None, None, False

        return direction, tier, flipped


def run_config(name, engine_kwargs, feat_files):
    """Run one config across all days, return summary stats."""
    engine = FilteredEngine(use_cnn=False, **engine_kwargs)
    all_trades = []
    day_pnls = []

    for fpath in tqdm(feat_files, desc=name, unit='day', leave=False):
        day = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            t['day'] = day
        all_trades.extend(engine.trades)
        day_pnls.append(engine.daily_pnl)

    n_trades = len(all_trades)
    if n_trades == 0:
        return {'name': name, 'trades': 0, 'pnl_per_day': 0,
                'wr': 0, 'win_days': 0, 'days': len(day_pnls)}

    wins = sum(1 for t in all_trades if t['pnl'] > 0)
    total_pnl = sum(day_pnls)
    win_days = sum(1 for p in day_pnls if p > 0)

    # Per-tier breakdown
    by_tier = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0})
    for t in all_trades:
        tier = t.get('entry_tier', '?')
        by_tier[tier]['n'] += 1
        by_tier[tier]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_tier[tier]['wins'] += 1

    return {
        'name': name,
        'trades': n_trades,
        'wins': wins,
        'wr': wins / n_trades * 100,
        'pnl_per_day': total_pnl / len(day_pnls),
        'total_pnl': total_pnl,
        'win_days': win_days,
        'days': len(day_pnls),
        'avg_per_trade': total_pnl / n_trades,
        'by_tier': dict(by_tier),
    }


def write_report(results, out_path):
    lines = []
    lines.append('=' * 80)
    lines.append('HYPOTHESIS TEST — feature filter impact')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append('=' * 80)
    lines.append('')

    # Summary table
    lines.append(f'{"Config":<25} {"$/day":>10} {"Trades":>10} {"WR":>6} '
                 f'{"WinDays":>10} {"$/tr":>8}')
    lines.append('-' * 75)
    baseline = results[0]
    for r in results:
        delta = r['pnl_per_day'] - baseline['pnl_per_day']
        delta_str = f'({delta:+.0f})' if r['name'] != 'BASELINE' else ''
        lines.append(f'{r["name"]:<25} ${r["pnl_per_day"]:>+8.0f} '
                     f'{delta_str:>10} '
                     f'{r["trades"]:>10,} '
                     f'{r["wr"]:>5.0f}% '
                     f'{r["win_days"]:>4}/{r["days"]:<4} '
                     f'{r["avg_per_trade"]:>+8.1f}')
    lines.append('')

    # Per-tier comparison
    lines.append('=' * 80)
    lines.append('PER-TIER COMPARISON (trades / WR / $/trade)')
    lines.append('=' * 80)
    all_tiers = set()
    for r in results:
        all_tiers.update(r['by_tier'].keys())

    header = f'{"Tier":<18}'
    for r in results:
        header += f' {r["name"][:14]:>14}'
    lines.append(header)
    lines.append('-' * len(header))

    for tier in sorted(all_tiers):
        row = f'{tier:<18}'
        for r in results:
            t = r['by_tier'].get(tier, {'n': 0, 'wins': 0, 'pnl': 0})
            if t['n'] > 0:
                wr = t['wins'] / t['n'] * 100
                avg = t['pnl'] / t['n']
                row += f'  {t["n"]:>4}/{wr:.0f}%/${avg:+.0f}'
            else:
                row += f' {"":>14}'
        lines.append(row)
    lines.append('')

    # Verdict
    lines.append('=' * 80)
    lines.append('VERDICT')
    lines.append('=' * 80)
    best = max(results, key=lambda r: r['pnl_per_day'])
    lines.append(f'Best config: {best["name"]}  ${best["pnl_per_day"]:+,.0f}/day')
    lines.append(f'  vs baseline: {best["pnl_per_day"] - baseline["pnl_per_day"]:+.0f}/day')
    lines.append(f'  trade count: {best["trades"]:,} '
                 f'({best["trades"] - baseline["trades"]:+,} vs baseline)')
    lines.append(f'  win rate: {best["wr"]:.0f}% '
                 f'({best["wr"] - baseline["wr"]:+.1f}pp vs baseline)')
    lines.append('')

    report = '\n'.join(lines)
    print(report)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=None)
    args = parser.parse_args()

    feat_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                        if '2025_' in os.path.basename(f))
    if args.days:
        feat_files = feat_files[:args.days]
    print(f'Testing on {len(feat_files)} IS days')

    configs = [
        ('BASELINE', {}),
        ('CHOP_FILTER', {'chop_filter': True}),
        ('KILLSHOT_CENTER', {'killshot_center': True}),
        ('NO_MTF_BREAKOUT', {'no_mtf_breakout': True}),
        ('GRAVITY_WELL', {'gravity_well': True}),
        ('ALL', {'chop_filter': True, 'killshot_center': True,
                 'no_mtf_breakout': True, 'gravity_well': True}),
    ]

    results = []
    for name, kwargs in configs:
        print(f'\nRunning {name}...')
        r = run_config(name, kwargs, feat_files)
        results.append(r)
        print(f'  ${r["pnl_per_day"]:+,.0f}/day, {r["trades"]:,} trades, '
              f'{r["wr"]:.0f}% WR')

    out_path = os.path.join(OUTPUT_DIR, 'hypothesis_test.txt')
    write_report(results, out_path)


if __name__ == '__main__':
    main()
