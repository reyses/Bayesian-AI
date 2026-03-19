"""
Fake Peaks Inside Human Seeds -- ground truth for exit calibration.

Human seeds are confirmed real trades. Any peak-detection signal that fires
AGAINST the position MID-TRADE is a false peak. The signal at the end when
the move tapers is the real peak.

For each seed, replays bars and captures:
  - False peaks: peak fires against position while trade is still running
  - Real taper: the actual end of the move (last 3 bars of seed)
  - Compares volume, DMI, momentum, P_at_center between false and real

Usage:
    python tools/fake_peak_inside_seeds.py

Output:
    reports/findings/fake_vs_real_peaks.txt
    reports/findings/fake_vs_real_peaks_charts.png
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
    """Load seeds with proper entry timestamps."""
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
        tf_seconds = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}.get(tf, 60)

        for s in seeds:
            lookback = s.get('lookback_bars', 10)
            entry_ts = s['ts_start'] + lookback * tf_seconds
            exit_ts = s['ts_end']
            duration_bars = max(1, int((exit_ts - entry_ts) / tf_seconds))

            entries.append({
                'ts_entry': entry_ts,
                'ts_exit': exit_ts,
                'direction': s['direction'].upper(),
                'mfe_ticks': s.get('mfe_ticks', 0),
                'duration_bars': duration_bars,
                'tf_seconds': tf_seconds,
                'tf': tf,
            })
        print(f"  {os.path.basename(path)}: {len(seeds)} seeds (tf={tf})")
    return entries


def load_multi_tf_states(data_root):
    """Load 15s + 1m states."""
    engine = StatisticalFieldEngine()
    result = {}

    for tf in ['15s', '1m']:
        tf_dir = os.path.join(data_root, tf)
        if not os.path.isdir(tf_dir):
            continue
        files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
        states, bars = [], []
        for fn in tqdm(files, desc=f"Loading {tf}"):
            df = pd.read_parquet(os.path.join(tf_dir, fn))
            if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
                df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
            raw = engine.batch_compute_states(df)
            for r in raw:
                states.append(r['state'] if isinstance(r, dict) and 'state' in r else r)
            bars.append(df)
        df_all = pd.concat(bars, ignore_index=True)
        ts = df_all['timestamp'].values.astype(np.int64)
        result[tf] = (ts, states)
        print(f"  {tf}: {len(states)} states")

    return result


def find_state(timestamps, states, target_ts):
    """Find state at or just before target_ts."""
    idx = np.searchsorted(timestamps, target_ts, side='right') - 1
    idx = max(0, min(idx, len(states) - 1))
    return states[idx]


def analyze_seed(seed, tf_data):
    """Replay a seed, find false peaks and real taper."""
    entry_ts = seed['ts_entry']
    exit_ts = seed['ts_exit']
    direction = seed['direction']
    trade_dir = 1 if direction == 'LONG' else -1

    # Use 15s for peak detection, 1m for volume
    ts_15s, states_15s = tf_data.get('15s', (np.array([]), []))
    ts_1m, states_1m = tf_data.get('1m', (np.array([]), []))

    if len(ts_15s) == 0:
        return [], []

    # Find bar range in 15s
    start_idx = int(np.searchsorted(ts_15s, entry_ts))
    end_idx = int(np.searchsorted(ts_15s, exit_ts))

    if start_idx >= len(states_15s) or end_idx <= start_idx:
        return [], []

    n_bars = end_idx - start_idx
    taper_start = max(start_idx, end_idx - 3)  # last 3 bars = real taper

    false_peaks = []
    taper_bars = []
    prev_pac = 0.0
    prev_fm_abs = 0.0

    for i in range(max(start_idx - 1, 0), min(end_idx + 1, len(states_15s))):
        st = states_15s[i]
        bar_ts = int(ts_15s[i]) if i < len(ts_15s) else entry_ts

        pac = getattr(st, 'P_at_center', 0.0) or 0.0
        fm = getattr(st, 'F_momentum', 0.0) or 0.0
        fm_abs = abs(fm)
        coh = getattr(st, 'oscillation_entropy_normalized', 0.0) or 0.0
        z = getattr(st, 'z_score', 0.0) or 0.0
        vel = getattr(st, 'velocity', 0.0) or 0.0
        dmi_p = getattr(st, 'dmi_plus', 0.0) or 0.0
        dmi_m = getattr(st, 'dmi_minus', 0.0) or 0.0
        adx = getattr(st, 'adx_strength', 0.0) or 0.0
        vol_15s = getattr(st, 'volume_delta', 0.0) or 0.0

        # 1m state for volume
        st_1m = find_state(ts_1m, states_1m, bar_ts) if len(ts_1m) > 0 else None
        vol_1m = getattr(st_1m, 'volume_delta', 0.0) or 0.0 if st_1m else 0.0
        fm_1m = getattr(st_1m, 'F_momentum', 0.0) or 0.0 if st_1m else 0.0

        # Peak detection fires?
        pac_up = pac > prev_pac * 1.05 if prev_pac > 0.01 else False
        fm_down = fm_abs < prev_fm_abs * 0.90 if prev_fm_abs > 0.5 else False
        peak_fires = (pac_up or fm_down) and coh > 0.55

        prev_pac = pac
        prev_fm_abs = fm_abs

        if i < start_idx:
            continue  # pre-trade, just updating prev state

        bar_offset = i - start_idx
        is_taper = (i >= taper_start)

        row = {
            'bar_offset': bar_offset,
            'bar_pct': bar_offset / max(n_bars, 1),  # 0.0 = entry, 1.0 = exit
            'is_taper': is_taper,
            'peak_fires': peak_fires,
            'direction': direction,
            # 15s state
            'P_at_center': pac,
            'F_momentum_15s': fm,
            'F_momentum_abs_15s': fm_abs,
            'coherence': coh,
            'z_score': z,
            'velocity_15s': vel,
            'dmi_plus_15s': dmi_p,
            'dmi_minus_15s': dmi_m,
            'dmi_diff_15s': dmi_p - dmi_m,
            'adx_15s': adx,
            'volume_15s': vol_15s,
            # 1m state
            'volume_1m': vol_1m,
            'F_momentum_1m': fm_1m,
            # Aligned
            'vol_1m_aligned': vol_1m * trade_dir,
            'fm_1m_aligned': fm_1m * trade_dir,
            'dmi_aligned_15s': (dmi_p - dmi_m) * trade_dir,
            'vol_15s_aligned': vol_15s * trade_dir,
        }

        if peak_fires and not is_taper:
            false_peaks.append(row)
        if is_taper:
            taper_bars.append(row)

    return false_peaks, taper_bars


def main():
    seed_files = [
        'DATA/regime_seeds/seeds_2025-01-02_20260313_180016.json',
        'DATA/regime_seeds/seeds_2025-01-03_20260313_184535.json',
        'DATA/regime_seeds/seeds_2025-07-14_20260313_093809.json',
        'DATA/regime_seeds/seeds_2025-01-05 (+2d)_multi.json',
    ]

    print("  Loading human seeds...")
    seeds = load_seeds(seed_files)
    print(f"  Total: {len(seeds)} seeds")

    print("\n  Loading states...")
    tf_data = load_multi_tf_states(os.path.join('DATA', 'ATLAS'))

    all_false = []
    all_taper = []
    seeds_with_false = 0

    for seed in tqdm(seeds, desc="Analyzing seeds"):
        false_peaks, taper_bars = analyze_seed(seed, tf_data)
        if false_peaks:
            seeds_with_false += 1
        all_false.extend(false_peaks)
        all_taper.extend(taper_bars)

    df_false = pd.DataFrame(all_false) if all_false else pd.DataFrame()
    df_taper = pd.DataFrame(all_taper) if all_taper else pd.DataFrame()

    print(f"\n  Seeds with false peaks: {seeds_with_false} / {len(seeds)}")
    print(f"  Total false peak bars: {len(df_false)}")
    print(f"  Total taper bars: {len(df_taper)}")

    # Report
    lines = []
    lines.append("=" * 80)
    lines.append("FALSE PEAKS vs REAL TAPER (inside human seeds)")
    lines.append("=" * 80)
    lines.append(f"\n  Seeds analyzed: {len(seeds)}")
    lines.append(f"  Seeds with false peaks: {seeds_with_false} ({seeds_with_false/max(len(seeds),1)*100:.0f}%)")
    lines.append(f"  False peak bars: {len(df_false)}")
    lines.append(f"  Real taper bars: {len(df_taper)}")

    if len(df_false) > 0 and len(df_taper) > 0:
        metrics = [
            ('P_at_center', 'P_at_center'),
            ('F_momentum_abs_15s', '|F_momentum| 15s'),
            ('coherence', 'Coherence 15s'),
            ('volume_15s', 'Volume 15s (raw)'),
            ('vol_15s_aligned', 'Volume 15s (aligned)'),
            ('volume_1m', 'Volume 1m (raw)'),
            ('vol_1m_aligned', 'Volume 1m (aligned)'),
            ('F_momentum_1m', 'F_momentum 1m (raw)'),
            ('fm_1m_aligned', 'F_momentum 1m (aligned)'),
            ('dmi_diff_15s', 'DMI Diff 15s (raw)'),
            ('dmi_aligned_15s', 'DMI Diff 15s (aligned)'),
            ('adx_15s', 'ADX 15s'),
            ('velocity_15s', 'Velocity 15s'),
            ('z_score', 'Z-score 15s'),
        ]

        lines.append(f"\n  {'Metric':<35} | {'False Peak':>12} | {'Real Taper':>12} | {'Separation':>12}")
        lines.append(f"  {'-'*35} | {'-'*12} | {'-'*12} | {'-'*12}")

        for col, label in metrics:
            fp_mean = df_false[col].mean() if col in df_false.columns else 0
            tp_mean = df_taper[col].mean() if col in df_taper.columns else 0
            sep = tp_mean - fp_mean
            lines.append(f"  {label:<35} | {fp_mean:>+12.2f} | {tp_mean:>+12.2f} | {sep:>+12.2f}")

        # When do false peaks fire (by trade %)
        lines.append(f"\n  FALSE PEAK TIMING (% through trade):")
        for pct in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            n = ((df_false['bar_pct'] >= pct) & (df_false['bar_pct'] < pct + 0.1)).sum()
            lines.append(f"    {pct*100:.0f}-{(pct+0.1)*100:.0f}%: {n} false peaks")

    report = '\n'.join(lines) + '\n'
    out_dir = os.path.join('reports', 'findings')
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, 'fake_vs_real_peaks.txt')
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)

    # Charts
    if len(df_false) > 0 and len(df_taper) > 0:
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('False Peaks vs Real Taper (Inside Human Seeds)\n'
                     'Blue = false peak (mid-trade noise), Red = real taper (move ending)',
                     fontsize=13, fontweight='bold')

        chart_metrics = [
            ('vol_1m_aligned', 'Volume 1m (aligned)'),
            ('fm_1m_aligned', 'F_momentum 1m (aligned)'),
            ('F_momentum_abs_15s', '|F_momentum| 15s'),
            ('volume_15s', 'Volume 15s'),
            ('P_at_center', 'P_at_center'),
            ('dmi_aligned_15s', 'DMI Diff 15s (aligned)'),
        ]

        for idx, (col, title) in enumerate(chart_metrics):
            ax = axes[idx // 2][idx % 2]
            fp_data = df_false[col].dropna()
            tp_data = df_taper[col].dropna()

            # Clip outliers
            for d in [fp_data, tp_data]:
                lo, hi = d.quantile(0.02), d.quantile(0.98)
                d = d[(d >= lo) & (d <= hi)]

            ax.hist(fp_data, bins=40, alpha=0.5, label=f'False Peak (n={len(df_false)})',
                    color='#2196F3', density=True)
            ax.hist(tp_data, bins=40, alpha=0.5, label=f'Real Taper (n={len(df_taper)})',
                    color='#F44336', density=True)
            ax.axvline(fp_data.mean(), color='#2196F3', linestyle='--', linewidth=2)
            ax.axvline(tp_data.mean(), color='#F44336', linestyle='--', linewidth=2)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = os.path.join(out_dir, 'fake_vs_real_peaks_charts.png')
        fig.savefig(out_png, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Charts saved: {out_png}")


if __name__ == '__main__':
    main()
