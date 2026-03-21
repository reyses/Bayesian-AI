"""
DMI Peak -> Volume Drop -> DMI Cross sequence detection inside human seeds.

For each seed, detects:
  1. DMI peak: bar where |DMI diff aligned| is maximum (trend at max strength)
  2. Volume drop: first bar after DMI peak where volume drops below entry volume
  3. DMI cross: first bar after DMI peak where DMI diff aligned flips sign

Measures N bars between each event and correlates with optimal exit (MFE bar).

Usage:
    python tools/dmi_peak_vol_drop_sequence.py

Output:
    reports/findings/dmi_vol_sequence.txt
    reports/findings/dmi_vol_sequence_chart.png
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine


def load_seeds(seed_files):
    entries = []
    for path in seed_files:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        seeds = data.get('seeds', [])
        tf = data.get('timeframe', '1m')
        if isinstance(tf, list):
            tf = tf[0]
        tf_sec = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}.get(tf, 60)
        for s in seeds:
            lb = s.get('lookback_bars', 10)
            entries.append({
                'ts_entry': s['ts_start'] + lb * tf_sec,
                'ts_exit': s['ts_end'],
                'direction': s['direction'].upper(),
                'mfe_ticks': s.get('mfe_ticks', 0),
                'change_ticks': s.get('change_ticks', 0),
                'tf_sec': tf_sec,
            })
    return entries


def load_states(data_root):
    engine = StatisticalFieldEngine()
    result = {}
    for tf in ['15s', '1m']:
        tf_dir = os.path.join(data_root, tf)
        if not os.path.isdir(tf_dir):
            continue
        files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
        states, bars = [], []
        for fn in tqdm(files, desc=f"  {tf}"):
            df = pd.read_parquet(os.path.join(tf_dir, fn))
            if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
                df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
            raw = engine.batch_compute_states(df)
            for r in raw:
                states.append(r['state'] if isinstance(r, dict) and 'state' in r else r)
            bars.append(df)
        df_all = pd.concat(bars, ignore_index=True)
        result[tf] = (df_all['timestamp'].values.astype(np.int64), states)
    return result


def find_state(ts_arr, states, target):
    idx = int(np.searchsorted(ts_arr, target, side='right')) - 1
    return max(0, min(idx, len(states) - 1)), states[max(0, min(idx, len(states) - 1))]


def analyze_seed_sequence(seed, tf_data):
    """Detect DMI peak -> volume drop -> DMI cross sequence."""
    ts_15s, states_15s = tf_data['15s']
    ts_1m, states_1m = tf_data.get('1m', (np.array([]), []))

    entry_ts = seed['ts_entry']
    exit_ts = seed['ts_exit']
    direction = seed['direction']
    trade_dir = 1 if direction == 'LONG' else -1

    entry_idx = int(np.searchsorted(ts_15s, entry_ts))
    exit_idx = int(np.searchsorted(ts_15s, exit_ts))

    if entry_idx >= len(states_15s) or exit_idx <= entry_idx:
        return None

    n_bars = exit_idx - entry_idx

    # Collect per-bar data
    bar_data = []
    entry_vol_1m = None

    for i in range(entry_idx, min(exit_idx + 5, len(states_15s))):
        st = states_15s[i]
        bar_ts = int(ts_15s[i])
        offset = i - entry_idx

        dmi_p = getattr(st, 'dmi_plus', 0.0) or 0.0
        dmi_m = getattr(st, 'dmi_minus', 0.0) or 0.0
        dmi_diff = dmi_p - dmi_m
        dmi_aligned = dmi_diff * trade_dir
        adx = getattr(st, 'adx_strength', 0.0) or 0.0
        price = getattr(st, 'price', 0.0) or 0.0

        # 1m volume
        _, st_1m = find_state(ts_1m, states_1m, bar_ts) if len(ts_1m) > 0 else (0, None)
        vol_1m = getattr(st_1m, 'volume_delta', 0.0) or 0.0 if st_1m else 0.0
        vol_1m_aligned = vol_1m * trade_dir
        fm_1m = getattr(st_1m, 'F_momentum', 0.0) or 0.0 if st_1m else 0.0

        if offset == 0:
            entry_vol_1m = abs(vol_1m)
            entry_price = price

        # Unrealized PnL
        if entry_price > 0:
            unreal = (price - entry_price) / 0.25 * trade_dir
        else:
            unreal = 0

        bar_data.append({
            'offset': offset,
            'in_trade': offset <= n_bars,
            'dmi_aligned': dmi_aligned,
            'dmi_diff': dmi_diff,
            'adx': adx,
            'vol_1m': vol_1m,
            'vol_1m_aligned': vol_1m_aligned,
            'vol_1m_abs': abs(vol_1m),
            'fm_1m': fm_1m,
            'unreal': unreal,
            'dmi_plus': dmi_p,
            'dmi_minus': dmi_m,
        })

    if not bar_data:
        return None

    # Detect events:
    # 1. DMI peak: bar with max |dmi_aligned| during trade
    trade_bars = [b for b in bar_data if b['in_trade']]
    if not trade_bars:
        return None

    dmi_peak_bar = max(trade_bars, key=lambda b: abs(b['dmi_aligned']))
    dmi_peak_offset = dmi_peak_bar['offset']
    dmi_peak_val = dmi_peak_bar['dmi_aligned']

    # 2. Volume drop: first bar after DMI peak where vol_1m_abs < 50% of entry vol
    vol_drop_offset = None
    vol_threshold = entry_vol_1m * 0.5 if entry_vol_1m > 0 else 0
    for b in bar_data:
        if b['offset'] > dmi_peak_offset and b['vol_1m_abs'] < vol_threshold:
            vol_drop_offset = b['offset']
            break

    # Also try: first bar after DMI peak where vol_1m_aligned flips negative
    vol_flip_offset = None
    for b in bar_data:
        if b['offset'] > dmi_peak_offset and b['vol_1m_aligned'] < 0:
            vol_flip_offset = b['offset']
            break

    # 3. DMI cross: first bar after DMI peak where dmi_aligned < 0
    dmi_cross_offset = None
    for b in bar_data:
        if b['offset'] > dmi_peak_offset and b['dmi_aligned'] < 0:
            dmi_cross_offset = b['offset']
            break

    # 4. MFE bar: bar with max unrealized PnL
    mfe_bar = max(trade_bars, key=lambda b: b['unreal'])
    mfe_offset = mfe_bar['offset']

    return {
        'n_bars': n_bars,
        'mfe_ticks': seed['mfe_ticks'],
        'mfe_offset': mfe_offset,
        'mfe_pct': mfe_offset / max(n_bars, 1) * 100,
        'dmi_peak_offset': dmi_peak_offset,
        'dmi_peak_pct': dmi_peak_offset / max(n_bars, 1) * 100,
        'dmi_peak_val': dmi_peak_val,
        'vol_drop_offset': vol_drop_offset,
        'vol_drop_pct': vol_drop_offset / max(n_bars, 1) * 100 if vol_drop_offset else None,
        'vol_flip_offset': vol_flip_offset,
        'vol_flip_pct': vol_flip_offset / max(n_bars, 1) * 100 if vol_flip_offset else None,
        'dmi_cross_offset': dmi_cross_offset,
        'dmi_cross_pct': dmi_cross_offset / max(n_bars, 1) * 100 if dmi_cross_offset else None,
        # Gaps
        'dmi_peak_to_vol_drop': (vol_drop_offset - dmi_peak_offset) if vol_drop_offset else None,
        'dmi_peak_to_vol_flip': (vol_flip_offset - dmi_peak_offset) if vol_flip_offset else None,
        'vol_drop_to_dmi_cross': (dmi_cross_offset - vol_drop_offset) if (vol_drop_offset and dmi_cross_offset) else None,
        'dmi_peak_to_dmi_cross': (dmi_cross_offset - dmi_peak_offset) if dmi_cross_offset else None,
        'mfe_to_exit': n_bars - mfe_offset,
        'bar_data': bar_data,
    }


def main():
    seed_files = [
        'DATA/regime_seeds/seeds_2025-01-02_20260313_180016.json',
        'DATA/regime_seeds/seeds_2025-01-03_20260313_184535.json',
        'DATA/regime_seeds/seeds_2025-07-14_20260313_093809.json',
        'DATA/regime_seeds/seeds_2025-01-05 (+2d)_multi.json',
    ]

    print("Loading seeds...")
    seeds = load_seeds(seed_files)
    print(f"  {len(seeds)} seeds")

    print("Loading states...")
    tf_data = load_states(os.path.join('DATA', 'ATLAS'))

    print("Analyzing sequences...")
    results = []
    for seed in tqdm(seeds, desc="  Seeds"):
        r = analyze_seed_sequence(seed, tf_data)
        if r:
            results.append(r)

    print(f"  {len(results)} seeds with valid sequences")

    # Report
    lines = []
    lines.append("=" * 80)
    lines.append("DMI PEAK -> VOLUME DROP -> DMI CROSS SEQUENCE")
    lines.append("=" * 80)
    lines.append(f"\n  Seeds analyzed: {len(results)}")

    # Event occurrence
    has_vol_drop = sum(1 for r in results if r['vol_drop_offset'] is not None)
    has_vol_flip = sum(1 for r in results if r['vol_flip_offset'] is not None)
    has_dmi_cross = sum(1 for r in results if r['dmi_cross_offset'] is not None)

    lines.append(f"\n  Event occurrence:")
    lines.append(f"    DMI peak found:    {len(results)} / {len(results)} (100%)")
    lines.append(f"    Volume drop found: {has_vol_drop} / {len(results)} ({has_vol_drop/len(results)*100:.0f}%)")
    lines.append(f"    Volume flip found: {has_vol_flip} / {len(results)} ({has_vol_flip/len(results)*100:.0f}%)")
    lines.append(f"    DMI cross found:   {has_dmi_cross} / {len(results)} ({has_dmi_cross/len(results)*100:.0f}%)")

    # Timing (% through trade)
    lines.append(f"\n  Event timing (% through trade):")
    lines.append(f"    {'Event':<20} | {'Mean':>8} | {'Median':>8} | {'P25':>8} | {'P75':>8}")
    lines.append(f"    {'-'*20} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    for label, key in [('MFE (peak profit)', 'mfe_pct'),
                        ('DMI peak', 'dmi_peak_pct'),
                        ('Volume drop', 'vol_drop_pct'),
                        ('Volume flip', 'vol_flip_pct'),
                        ('DMI cross', 'dmi_cross_pct')]:
        vals = [r[key] for r in results if r[key] is not None]
        if vals:
            lines.append(f"    {label:<20} | {np.mean(vals):>7.1f}% | {np.median(vals):>7.1f}% | "
                         f"{np.percentile(vals, 25):>7.1f}% | {np.percentile(vals, 75):>7.1f}%")

    # Gaps (bars between events)
    lines.append(f"\n  Gaps between events (bars):")
    for label, key in [('DMI peak -> Vol drop', 'dmi_peak_to_vol_drop'),
                        ('DMI peak -> Vol flip', 'dmi_peak_to_vol_flip'),
                        ('Vol drop -> DMI cross', 'vol_drop_to_dmi_cross'),
                        ('DMI peak -> DMI cross', 'dmi_peak_to_dmi_cross'),
                        ('MFE -> actual exit', 'mfe_to_exit')]:
        vals = [r[key] for r in results if r[key] is not None]
        if vals:
            lines.append(f"    {label:<25}: mean={np.mean(vals):>5.1f}  median={np.median(vals):>5.0f}  "
                         f"P25={np.percentile(vals, 25):>5.0f}  P75={np.percentile(vals, 75):>5.0f}")

    # The key question: if we exited at vol drop instead of actual exit, how much regret?
    lines.append(f"\n  OPTIMAL EXIT COMPARISON:")
    lines.append(f"  If we exited at each event, how does it compare to MFE?")
    for label, offset_key in [('At DMI peak', 'dmi_peak_offset'),
                               ('At volume drop', 'vol_drop_offset'),
                               ('At volume flip', 'vol_flip_offset'),
                               ('At DMI cross', 'dmi_cross_offset')]:
        diffs = []
        for r in results:
            if r[offset_key] is not None:
                # How far from MFE (negative = before MFE = left money, positive = after MFE = gave back)
                diff = r[offset_key] - r['mfe_offset']
                diffs.append(diff)
        if diffs:
            before = sum(1 for d in diffs if d < 0)
            at = sum(1 for d in diffs if d == 0)
            after = sum(1 for d in diffs if d > 0)
            lines.append(f"    {label:<20}: before_MFE={before}  at_MFE={at}  after_MFE={after}  "
                         f"avg_gap={np.mean(diffs):>+.1f} bars")

    report = '\n'.join(lines) + '\n'
    out_dir = os.path.join('reports', 'findings')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'dmi_vol_sequence.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)

    # Chart: timeline showing event order
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('DMI Peak -> Volume Drop -> DMI Cross Sequence\n'
                 'Event timing as % of trade duration', fontsize=13, fontweight='bold')

    # Top: scatter of event timing
    ax = axes[0]
    for i, r in enumerate(results[:100]):  # first 100 for clarity
        y = i
        ax.scatter(r['mfe_pct'], y, color='gold', s=20, zorder=3, marker='*')
        ax.scatter(r['dmi_peak_pct'], y, color='blue', s=15, zorder=2)
        if r['vol_drop_pct'] is not None:
            ax.scatter(r['vol_drop_pct'], y, color='green', s=15, zorder=2, marker='v')
        if r['dmi_cross_pct'] is not None:
            ax.scatter(r['dmi_cross_pct'], y, color='red', s=15, zorder=2, marker='x')

    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('% through trade')
    ax.set_ylabel('Trade #')
    ax.set_title('Event timeline per trade (gold*=MFE, blue=DMI peak, green v=vol drop, red x=DMI cross)')
    ax.set_xlim(-5, 130)
    ax.grid(True, alpha=0.3)

    # Bottom: histogram of gaps
    ax2 = axes[1]
    gap_data = {
        'DMI peak->Vol drop': [r['dmi_peak_to_vol_drop'] for r in results if r['dmi_peak_to_vol_drop'] is not None],
        'Vol drop->DMI cross': [r['vol_drop_to_dmi_cross'] for r in results if r['vol_drop_to_dmi_cross'] is not None],
        'DMI peak->DMI cross': [r['dmi_peak_to_dmi_cross'] for r in results if r['dmi_peak_to_dmi_cross'] is not None],
    }
    colors = ['#2196F3', '#4CAF50', '#F44336']
    for (label, vals), color in zip(gap_data.items(), colors):
        if vals:
            ax2.hist(vals, bins=30, alpha=0.4, label=f'{label} (n={len(vals)}, med={np.median(vals):.0f})',
                     color=color, density=True)
            ax2.axvline(np.median(vals), color=color, linestyle='--', linewidth=2)

    ax2.set_xlabel('Gap (15s bars)')
    ax2.set_title('Bars between events')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'dmi_vol_sequence_chart.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {os.path.join(out_dir, 'dmi_vol_sequence_chart.png')}")


if __name__ == '__main__':
    main()
