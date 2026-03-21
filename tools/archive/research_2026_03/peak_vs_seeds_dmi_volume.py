"""
Peak vs Human Seeds -- DMI + Volume comparison.

Compares the DMI and volume profile at entry for:
  1. Human seeds (gold standard -- real reversals identified by human)
  2. Peak detection winners (real peaks the system caught)
  3. Peak detection losers (fake peaks the system fell for)

If real peaks look like human seeds and fake peaks don't,
we know exactly what to filter on at entry.

Usage:
    python tools/peak_vs_seeds_dmi_volume.py

Output:
    reports/findings/peak_vs_seeds_comparison.txt
    reports/findings/peak_vs_seeds_charts.png
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine


def load_seeds(seed_files):
    """Load human-placed seeds from multiple files."""
    entries = []
    for path in seed_files:
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found")
            continue
        with open(path) as f:
            data = json.load(f)

        seeds = data.get('seeds', [])
        if isinstance(data, dict) and 'days' in data:
            # Multi-day format
            for day in data['days'].values():
                seeds.extend(day.get('seeds', []))

        # Determine bar duration from timeframe
        tf = data.get('timeframe', '1m')
        if isinstance(tf, list):
            tf = tf[0]  # multi-TF: use first
        tf_seconds = {'1s': 1, '5s': 5, '15s': 15, '30s': 30,
                      '1m': 60, '3m': 180, '5m': 300, '15m': 900,
                      '30m': 1800, '1h': 3600}.get(tf, 60)

        for s in seeds:
            # ts_start = start of lookback window (10 bars BEFORE entry)
            # actual entry = ts_start + lookback_bars * bar_duration
            lookback = s.get('lookback_bars', 10)
            entry_ts = s['ts_start'] + lookback * tf_seconds

            entries.append({
                'ts': entry_ts,
                'ts_lookback_start': s['ts_start'],
                'direction': s['direction'].upper(),
                'mfe_ticks': s.get('mfe_ticks', 0),
                'mae_ticks': s.get('mae_ticks', 0),
                'duration_mins': s.get('duration_mins', 0),
                'change_ticks': s.get('change_ticks', 0),
                'timeframe': tf,
                'source': 'human_seed',
            })
        print(f"  {os.path.basename(path)}: {len(seeds)} seeds")
    return entries


def load_peak_trades(trade_log_path):
    """Load peak trades from OOS trade log."""
    df = pd.read_csv(trade_log_path)
    peak = df[df['template_id'] == -100]

    entries = []
    for _, row in peak.iterrows():
        entries.append({
            'ts': row['entry_time'],
            'direction': row['direction'].upper(),
            'mfe_ticks': row.get('trade_mfe_ticks', 0),
            'mae_ticks': 0,
            'duration_mins': row.get('hold_bars', 1) * 15 / 60,
            'actual_pnl': row['actual_pnl'],
            'result': row['result'],
            'source': f"peak_{'win' if row['result'] == 'WIN' else 'loss'}",
        })
    return entries


def get_state_at_ts(timestamps, states, target_ts):
    """Find the state at or just before target_ts."""
    idx = np.searchsorted(timestamps, target_ts, side='right') - 1
    idx = max(0, min(idx, len(states) - 1))
    return states[idx]


def enrich_with_states(entries, ts_1m, states_1m, ts_15s, states_15s):
    """Add DMI + volume + state fields to each entry."""
    for e in tqdm(entries, desc=f"Enriching {entries[0]['source'] if entries else '?'}"):
        ts = e['ts']

        # 1m state (volume + institutional DMI)
        st_1m = get_state_at_ts(ts_1m, states_1m, ts)
        e['vol_1m'] = getattr(st_1m, 'volume_delta', 0.0) or 0.0
        e['dmi_plus_1m'] = getattr(st_1m, 'dmi_plus', 0.0) or 0.0
        e['dmi_minus_1m'] = getattr(st_1m, 'dmi_minus', 0.0) or 0.0
        e['dmi_diff_1m'] = e['dmi_plus_1m'] - e['dmi_minus_1m']
        e['adx_1m'] = getattr(st_1m, 'adx_strength', 0.0) or 0.0
        e['fm_1m'] = getattr(st_1m, 'F_momentum', 0.0) or 0.0
        e['vel_1m'] = getattr(st_1m, 'velocity', 0.0) or 0.0
        e['z_1m'] = getattr(st_1m, 'z_score', 0.0) or 0.0

        # 15s state (peak detection signals)
        st_15s = get_state_at_ts(ts_15s, states_15s, ts)
        e['pac_15s'] = getattr(st_15s, 'P_at_center', 0.0) or 0.0
        e['fm_15s'] = getattr(st_15s, 'F_momentum', 0.0) or 0.0
        e['fm_abs_15s'] = abs(e['fm_15s'])
        e['coh_15s'] = getattr(st_15s, 'oscillation_entropy_normalized', 0.0) or 0.0
        e['dmi_plus_15s'] = getattr(st_15s, 'dmi_plus', 0.0) or 0.0
        e['dmi_minus_15s'] = getattr(st_15s, 'dmi_minus', 0.0) or 0.0
        e['dmi_diff_15s'] = e['dmi_plus_15s'] - e['dmi_minus_15s']
        e['adx_15s'] = getattr(st_15s, 'adx_strength', 0.0) or 0.0
        e['z_15s'] = getattr(st_15s, 'z_score', 0.0) or 0.0
        e['vol_15s'] = getattr(st_15s, 'volume_delta', 0.0) or 0.0

        # Direction-aligned metrics
        trade_dir = 1 if e['direction'] == 'LONG' else -1
        e['vol_1m_aligned'] = e['vol_1m'] * trade_dir  # positive = volume WITH trade
        e['fm_1m_aligned'] = e['fm_1m'] * trade_dir
        e['dmi_aligned_1m'] = e['dmi_diff_1m'] * trade_dir
        e['dmi_aligned_15s'] = e['dmi_diff_15s'] * trade_dir

    return entries


def analyze_and_report(seeds, peak_wins, peak_losses, output_txt, output_png):
    """Generate comparison report and charts."""
    groups = {
        'Human Seeds': pd.DataFrame(seeds),
        'Peak Winners': pd.DataFrame(peak_wins),
        'Peak Losers': pd.DataFrame(peak_losses),
    }

    lines = []
    lines.append("=" * 80)
    lines.append("PEAK vs HUMAN SEEDS -- DMI + Volume at Entry")
    lines.append("=" * 80)

    metrics = [
        ('vol_1m', 'Volume 1m (raw)'),
        ('vol_1m_aligned', 'Volume 1m (trade-aligned, + = with trade)'),
        ('fm_1m', 'F_momentum 1m (raw)'),
        ('fm_1m_aligned', 'F_momentum 1m (trade-aligned)'),
        ('dmi_diff_1m', 'DMI Diff 1m (raw)'),
        ('dmi_aligned_1m', 'DMI Diff 1m (trade-aligned, + = DMI agrees)'),
        ('adx_1m', 'ADX 1m'),
        ('vel_1m', 'Velocity 1m'),
        ('z_1m', 'Z-score 1m'),
        ('pac_15s', 'P_at_center 15s'),
        ('fm_abs_15s', '|F_momentum| 15s'),
        ('coh_15s', 'Coherence 15s'),
        ('dmi_diff_15s', 'DMI Diff 15s (raw)'),
        ('dmi_aligned_15s', 'DMI Diff 15s (trade-aligned)'),
        ('adx_15s', 'ADX 15s'),
        ('vol_15s', 'Volume 15s'),
    ]

    lines.append(f"\n  {'Metric':<40} | {'Seeds':>10} | {'Peak WIN':>10} | {'Peak LOSS':>10} | {'W-L Sep':>10}")
    lines.append(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

    for col, label in metrics:
        vals = {}
        for name, df in groups.items():
            if col in df.columns:
                vals[name] = df[col].mean()
            else:
                vals[name] = np.nan
        sep = vals.get('Peak Winners', 0) - vals.get('Peak Losers', 0)
        lines.append(f"  {label:<40} | {vals.get('Human Seeds', 0):>+10.2f} | "
                     f"{vals.get('Peak Winners', 0):>+10.2f} | "
                     f"{vals.get('Peak Losers', 0):>+10.2f} | {sep:>+10.2f}")

    # Distribution comparison
    lines.append(f"\n\n  DISTRIBUTION COMPARISON (key metrics):")
    for col, label in [('vol_1m_aligned', 'Vol 1m aligned'),
                        ('fm_1m_aligned', 'F_mom 1m aligned'),
                        ('dmi_aligned_1m', 'DMI 1m aligned'),
                        ('adx_1m', 'ADX 1m')]:
        lines.append(f"\n  {label}:")
        for name, df in groups.items():
            if col not in df.columns:
                continue
            s = df[col]
            lines.append(f"    {name:<15}: mean={s.mean():>+8.1f}  std={s.std():>8.1f}  "
                         f"P25={s.quantile(0.25):>+8.1f}  P50={s.median():>+8.1f}  "
                         f"P75={s.quantile(0.75):>+8.1f}")

    # How many peak losers would be filtered by seed-like criteria?
    lines.append(f"\n\n  FILTERING ANALYSIS:")
    peak_loss_df = groups['Peak Losers']
    seed_df = groups['Human Seeds']

    if 'vol_1m_aligned' in peak_loss_df.columns and 'vol_1m_aligned' in seed_df.columns:
        seed_vol_p25 = seed_df['vol_1m_aligned'].quantile(0.25)
        n_filtered = (peak_loss_df['vol_1m_aligned'] < seed_vol_p25).sum()
        lines.append(f"  If require vol_1m_aligned >= seed P25 ({seed_vol_p25:+.0f}):")
        lines.append(f"    Peak losers filtered: {n_filtered} / {len(peak_loss_df)} ({n_filtered/len(peak_loss_df)*100:.0f}%)")

        peak_win_df = groups['Peak Winners']
        n_win_filtered = (peak_win_df['vol_1m_aligned'] < seed_vol_p25).sum()
        lines.append(f"    Peak winners filtered: {n_win_filtered} / {len(peak_win_df)} ({n_win_filtered/len(peak_win_df)*100:.0f}%)")

    report = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)

    # Charts
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Peak Detection vs Human Seeds at Entry\n'
                 'Trade-aligned metrics (positive = signal agrees with trade direction)',
                 fontsize=13, fontweight='bold')

    chart_metrics = [
        ('vol_1m_aligned', 'Volume 1m (aligned)'),
        ('fm_1m_aligned', 'F_momentum 1m (aligned)'),
        ('dmi_aligned_1m', 'DMI Diff 1m (aligned)'),
        ('adx_1m', 'ADX 1m'),
        ('pac_15s', 'P_at_center 15s'),
        ('fm_abs_15s', '|F_momentum| 15s'),
    ]

    colors = {'Human Seeds': '#2196F3', 'Peak Winners': '#4CAF50', 'Peak Losers': '#F44336'}

    for idx, (col, title) in enumerate(chart_metrics):
        ax = axes[idx // 2][idx % 2]
        for name, df in groups.items():
            if col not in df.columns:
                continue
            data = df[col].dropna()
            # Clip extreme outliers for visualization
            lo, hi = data.quantile(0.02), data.quantile(0.98)
            data = data[(data >= lo) & (data <= hi)]
            ax.hist(data, bins=50, alpha=0.4, label=f'{name} (n={len(df)})',
                    color=colors.get(name, 'gray'), density=True)
            ax.axvline(data.mean(), color=colors.get(name, 'gray'),
                       linestyle='--', linewidth=2)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Charts saved: {output_png}")


def main():
    data_root = os.path.join('DATA', 'ATLAS')
    trade_path = 'checkpoints/oos_trade_log.csv'

    # Human-placed seed files (gold standard)
    seed_files = [
        'DATA/regime_seeds/seeds_2025-01-02_20260313_180016.json',
        'DATA/regime_seeds/seeds_2025-01-03_20260313_184535.json',
        'DATA/regime_seeds/seeds_2025-07-14_20260313_093809.json',
        'DATA/regime_seeds/seeds_2025-01-05 (+2d)_multi.json',
    ]

    # Load seeds
    print("  Loading human seeds...")
    seed_entries = load_seeds(seed_files)
    print(f"  Total: {len(seed_entries)} human seeds")

    # Load peak trades
    print("  Loading peak trades...")
    peak_entries = load_peak_trades(trade_path)
    peak_wins = [e for e in peak_entries if e['source'] == 'peak_win']
    peak_losses = [e for e in peak_entries if e['source'] == 'peak_loss']
    print(f"  {len(peak_wins)} peak wins, {len(peak_losses)} peak losses")

    # Load 1m and 15s states from ATLAS (seeds are from IS data)
    print("\n  Loading 1m states...")
    engine = StatisticalFieldEngine()

    # 1m
    tf_dir = os.path.join(data_root, '1m')
    files_1m = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    states_1m, bars_1m = [], []
    for fn in tqdm(files_1m, desc="1m"):
        df = pd.read_parquet(os.path.join(tf_dir, fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        raw = engine.batch_compute_states(df)
        for r in raw:
            states_1m.append(r['state'] if isinstance(r, dict) and 'state' in r else r)
        bars_1m.append(df)
    df_1m = pd.concat(bars_1m, ignore_index=True)
    ts_1m = df_1m['timestamp'].values.astype(np.int64)
    print(f"  1m: {len(states_1m)} states")

    # 15s
    print("  Loading 15s states...")
    tf_dir = os.path.join(data_root, '15s')
    files_15s = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    states_15s, bars_15s = [], []
    for fn in tqdm(files_15s, desc="15s"):
        df = pd.read_parquet(os.path.join(tf_dir, fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        raw = engine.batch_compute_states(df)
        for r in raw:
            states_15s.append(r['state'] if isinstance(r, dict) and 'state' in r else r)
        bars_15s.append(df)
    df_15s = pd.concat(bars_15s, ignore_index=True)
    ts_15s = df_15s['timestamp'].values.astype(np.int64)
    print(f"  15s: {len(states_15s)} states")

    # Also load OOS states for peak trades
    print("  Loading OOS 1m + 15s states for peak trades...")
    oos_root = os.path.join('DATA', 'ATLAS_OOS')
    oos_states_1m, oos_bars_1m = [], []
    for fn in sorted(os.listdir(os.path.join(oos_root, '1m'))):
        if not fn.endswith('.parquet'):
            continue
        df = pd.read_parquet(os.path.join(oos_root, '1m', fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        raw = engine.batch_compute_states(df)
        for r in raw:
            oos_states_1m.append(r['state'] if isinstance(r, dict) and 'state' in r else r)
        oos_bars_1m.append(df)
    oos_df_1m = pd.concat(oos_bars_1m, ignore_index=True)
    oos_ts_1m = oos_df_1m['timestamp'].values.astype(np.int64)

    oos_states_15s, oos_bars_15s = [], []
    for fn in sorted(os.listdir(os.path.join(oos_root, '15s'))):
        if not fn.endswith('.parquet'):
            continue
        df = pd.read_parquet(os.path.join(oos_root, '15s', fn))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        raw = engine.batch_compute_states(df)
        for r in raw:
            oos_states_15s.append(r['state'] if isinstance(r, dict) and 'state' in r else r)
        oos_bars_15s.append(df)
    oos_df_15s = pd.concat(oos_bars_15s, ignore_index=True)
    oos_ts_15s = oos_df_15s['timestamp'].values.astype(np.int64)
    print(f"  OOS 1m: {len(oos_states_1m)}, OOS 15s: {len(oos_states_15s)}")

    # Enrich seeds with IS states, peak trades with OOS states
    print("\n  Enriching entries with state data...")
    seed_entries = enrich_with_states(seed_entries, ts_1m, states_1m, ts_15s, states_15s)
    peak_wins = enrich_with_states(peak_wins, oos_ts_1m, oos_states_1m, oos_ts_15s, oos_states_15s)
    peak_losses = enrich_with_states(peak_losses, oos_ts_1m, oos_states_1m, oos_ts_15s, oos_states_15s)

    # Analyze
    out_dir = os.path.join('reports', 'findings')
    analyze_and_report(
        seed_entries, peak_wins, peak_losses,
        os.path.join(out_dir, 'peak_vs_seeds_comparison.txt'),
        os.path.join(out_dir, 'peak_vs_seeds_charts.png'),
    )


if __name__ == '__main__':
    main()
