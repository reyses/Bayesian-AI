"""
Seed I-MR with DMI + Volume overlay -- visual correlation.

For each human seed, plots the trade trajectory as an I-chart (price/PnL)
with DMI and Volume as the "range" channels instead of moving range.
This shows visually WHERE false peaks fire vs WHERE volume/DMI confirm
the real taper.

Layout per seed (or averaged):
  Top:    Price trajectory (I-chart) with false peak markers + taper zone
  Middle: 1m Volume (aligned) -- spike = real taper, quiet = false peak
  Bottom: DMI diff (aligned) -- crossing = regime change

Usage:
    python tools/seed_imr_dmi_volume.py [--n-seeds 20]

Output:
    reports/findings/seed_imr_dmi_vol_avg.png    (averaged across all seeds)
    reports/findings/seed_imr_dmi_vol_examples.png (individual seed examples)
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
    """Load seeds with entry/exit timestamps."""
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
    """Load 15s + 1m states."""
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
    return states[max(0, min(idx, len(states) - 1))]


def extract_seed_series(seed, tf_data, pre_bars=5, post_bars=10):
    """Extract per-15s-bar series for one seed with all channels."""
    ts_15s, states_15s = tf_data['15s']
    ts_1m, states_1m = tf_data.get('1m', (np.array([]), []))

    entry_ts = seed['ts_entry']
    exit_ts = seed['ts_exit']
    direction = seed['direction']
    trade_dir = 1 if direction == 'LONG' else -1
    entry_price = 0.0

    start_idx = int(np.searchsorted(ts_15s, entry_ts)) - pre_bars
    end_idx = int(np.searchsorted(ts_15s, exit_ts)) + post_bars
    entry_idx = int(np.searchsorted(ts_15s, entry_ts))
    exit_idx = int(np.searchsorted(ts_15s, exit_ts))

    start_idx = max(0, start_idx)
    end_idx = min(len(states_15s) - 1, end_idx)

    if entry_idx >= len(states_15s):
        return None

    entry_price = getattr(states_15s[entry_idx], 'price', 0.0) or 0.0
    n_trade_bars = exit_idx - entry_idx

    prev_pac = 0.0
    prev_fm_abs = 0.0

    rows = []
    for i in range(start_idx, end_idx + 1):
        if i >= len(states_15s):
            break

        st = states_15s[i]
        bar_ts = int(ts_15s[i])
        offset = i - entry_idx
        in_trade = (i >= entry_idx and i <= exit_idx)
        is_taper = (i >= exit_idx - 3 and i <= exit_idx) if n_trade_bars > 3 else (i == exit_idx)

        # Normalize offset to 0-100% of trade
        if n_trade_bars > 0:
            trade_pct = offset / n_trade_bars * 100
        else:
            trade_pct = 0

        price = getattr(st, 'price', 0.0) or 0.0
        pac = getattr(st, 'P_at_center', 0.0) or 0.0
        fm = getattr(st, 'F_momentum', 0.0) or 0.0
        fm_abs = abs(fm)
        coh = getattr(st, 'oscillation_entropy_normalized', 0.0) or 0.0
        dmi_p = getattr(st, 'dmi_plus', 0.0) or 0.0
        dmi_m = getattr(st, 'dmi_minus', 0.0) or 0.0
        vol_15s = getattr(st, 'volume_delta', 0.0) or 0.0

        # Peak detection
        pac_up = pac > prev_pac * 1.05 if prev_pac > 0.01 else False
        fm_down = fm_abs < prev_fm_abs * 0.90 if prev_fm_abs > 0.5 else False
        peak_fires = (pac_up or fm_down) and coh > 0.55
        prev_pac = pac
        prev_fm_abs = fm_abs

        # 1m
        st_1m = find_state(ts_1m, states_1m, bar_ts) if len(ts_1m) > 0 else None
        vol_1m = getattr(st_1m, 'volume_delta', 0.0) or 0.0 if st_1m else 0.0
        fm_1m = getattr(st_1m, 'F_momentum', 0.0) or 0.0 if st_1m else 0.0

        # Unrealized PnL in ticks
        if entry_price > 0:
            unreal = (price - entry_price) / 0.25 * trade_dir
        else:
            unreal = 0

        rows.append({
            'offset': offset,
            'trade_pct': trade_pct,
            'in_trade': in_trade,
            'is_taper': is_taper,
            'peak_fires': peak_fires and in_trade,
            'price': price,
            'unreal_ticks': unreal if in_trade else np.nan,
            'P_at_center': pac,
            'F_momentum_abs': fm_abs,
            'coherence': coh,
            'vol_1m_aligned': vol_1m * trade_dir,
            'vol_15s_aligned': vol_15s * trade_dir,
            'fm_1m_aligned': fm_1m * trade_dir,
            'dmi_diff_aligned': (dmi_p - dmi_m) * trade_dir,
            'dmi_plus': dmi_p,
            'dmi_minus': dmi_m,
        })

    return pd.DataFrame(rows) if rows else None


def plot_averaged(all_series, output_path):
    """Plot averaged I-chart with DMI + Volume channels."""
    # Normalize all series to trade_pct (0-100%)
    # Bin into 5% buckets
    bins = np.arange(-20, 130, 5)
    bin_labels = bins[:-1] + 2.5

    channels = ['unreal_ticks', 'vol_1m_aligned', 'fm_1m_aligned',
                'dmi_diff_aligned', 'F_momentum_abs', 'P_at_center']

    # Collect all data points binned by trade_pct
    binned = {c: {b: [] for b in bin_labels} for c in channels}
    peak_by_bin = {b: 0 for b in bin_labels}
    total_by_bin = {b: 0 for b in bin_labels}

    for series in all_series:
        if series is None:
            continue
        for _, row in series.iterrows():
            pct = row['trade_pct']
            bi = int(np.searchsorted(bins, pct)) - 1
            if 0 <= bi < len(bin_labels):
                bl = bin_labels[bi]
                for c in channels:
                    if not np.isnan(row.get(c, np.nan)):
                        binned[c][bl].append(row[c])
                total_by_bin[bl] += 1
                if row.get('peak_fires', False):
                    peak_by_bin[bl] += 1

    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(6, 1, figure=fig, hspace=0.3)
    fig.suptitle('Human Seeds: Averaged Trade Trajectory\n'
                 'I-chart (top) + Volume/DMI/Momentum channels\n'
                 'Orange dots = false peak detection fires mid-trade',
                 fontsize=14, fontweight='bold', y=0.99)

    titles = [
        'Unrealized PnL (ticks) -- the trade',
        '1m Volume (aligned) -- institutional flow',
        '1m F_momentum (aligned) -- momentum force',
        'DMI Diff (aligned) -- direction signal',
        '|F_momentum| 15s -- peak detection input',
        'P_at_center -- peak detection input',
    ]

    for idx, (channel, title) in enumerate(zip(channels, titles)):
        ax = fig.add_subplot(gs[idx])
        means = []
        stds = []
        xs = []
        for bl in bin_labels:
            vals = binned[channel][bl]
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(bl)

        xs = np.array(xs)
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(xs, means, 'b-o', markersize=3, linewidth=1.5, label='Mean')
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color='blue')

        # Mark false peak density
        if idx == 0:  # Only on PnL chart
            peak_rates = []
            peak_xs = []
            for bl in bin_labels:
                if total_by_bin[bl] > 0:
                    rate = peak_by_bin[bl] / total_by_bin[bl]
                    if rate > 0:
                        peak_rates.append(rate)
                        peak_xs.append(bl)
            if peak_rates:
                ax2 = ax.twinx()
                ax2.bar(peak_xs, peak_rates, width=4, alpha=0.3, color='orange',
                        label='False peak rate')
                ax2.set_ylabel('Peak fire rate', color='orange')
                ax2.set_ylim(0, max(peak_rates) * 2)
                ax2.legend(loc='upper left', fontsize=8)

        # Entry and exit lines
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.7, label='Entry')
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='Exit/Taper')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

        # Taper zone
        ax.axvspan(85, 100, alpha=0.1, color='red', label='Taper zone')

        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Trade progress (%)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-20, 125)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Averaged chart saved: {output_path}")


def plot_examples(all_series, seeds, output_path, n_examples=8):
    """Plot individual seed examples showing price + volume + DMI."""
    # Pick seeds with good MFE and some false peaks
    scored = []
    for i, (series, seed) in enumerate(zip(all_series, seeds)):
        if series is None:
            continue
        n_false = series['peak_fires'].sum()
        mfe = seed.get('mfe_ticks', 0)
        if mfe > 20 and n_false > 0:
            scored.append((i, mfe, n_false))

    scored.sort(key=lambda x: -x[1])  # best MFE first
    selected = scored[:n_examples]

    if not selected:
        print("  No examples with false peaks found")
        return

    fig = plt.figure(figsize=(20, 5 * len(selected)))
    gs = GridSpec(len(selected), 3, figure=fig, hspace=0.4, wspace=0.3)
    fig.suptitle('Individual Seeds: Price + Volume + DMI\n'
                 'Orange markers = false peak fires | Red zone = taper',
                 fontsize=14, fontweight='bold', y=1.01)

    for row, (si, mfe, n_false) in enumerate(selected):
        series = all_series[si]
        seed = seeds[si]
        offsets = series['offset'].values

        # Price / PnL
        ax1 = fig.add_subplot(gs[row, 0])
        ax1.plot(offsets, series['unreal_ticks'].values, 'b-', linewidth=1.5)
        # Mark false peaks
        fp = series[series['peak_fires'] == True]
        if len(fp) > 0:
            ax1.scatter(fp['offset'], fp['unreal_ticks'], color='orange',
                        s=60, zorder=5, marker='v', label='False peak')
        # Taper zone
        taper = series[series['is_taper'] == True]
        if len(taper) > 0:
            ax1.axvspan(taper['offset'].min(), taper['offset'].max(),
                        alpha=0.2, color='red')
        ax1.axvline(x=0, color='green', linestyle='--', alpha=0.5)
        ax1.set_title(f'PnL (ticks) | {seed["direction"]} | MFE={mfe:.0f}t | {n_false} false peaks',
                      fontsize=10)
        ax1.set_ylabel('Ticks')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=8)

        # Volume 1m
        ax2 = fig.add_subplot(gs[row, 1])
        ax2.bar(offsets, series['vol_1m_aligned'].values, width=0.8,
                alpha=0.6, color='steelblue')
        if len(fp) > 0:
            ax2.scatter(fp['offset'], fp['vol_1m_aligned'], color='orange',
                        s=60, zorder=5, marker='v')
        if len(taper) > 0:
            ax2.axvspan(taper['offset'].min(), taper['offset'].max(),
                        alpha=0.2, color='red')
        ax2.axvline(x=0, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_title('1m Volume (aligned)', fontsize=10)
        ax2.set_ylabel('Volume delta')
        ax2.grid(True, alpha=0.3)

        # DMI
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.plot(offsets, series['dmi_plus'].values, 'g-', linewidth=1.5, label='DI+')
        ax3.plot(offsets, series['dmi_minus'].values, 'r-', linewidth=1.5, label='DI-')
        if len(fp) > 0:
            ax3.scatter(fp['offset'], fp['dmi_diff_aligned'], color='orange',
                        s=60, zorder=5, marker='v')
        if len(taper) > 0:
            ax3.axvspan(taper['offset'].min(), taper['offset'].max(),
                        alpha=0.2, color='red')
        ax3.axvline(x=0, color='green', linestyle='--', alpha=0.5)
        ax3.set_title('DMI (DI+ green, DI- red)', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Examples chart saved: {output_path}")


def main():
    seed_files = [
        'DATA/regime_seeds/seeds_2025-01-02_20260313_180016.json',
    ]

    print("Loading seeds...")
    seeds = load_seeds(seed_files)
    print(f"  {len(seeds)} seeds")

    print("Loading states...")
    tf_data = load_states(os.path.join('DATA', 'ATLAS'))

    print("Extracting per-seed series...")
    all_series = []
    for seed in tqdm(seeds, desc="  Seeds"):
        series = extract_seed_series(seed, tf_data)
        all_series.append(series)

    valid = sum(1 for s in all_series if s is not None)
    print(f"  {valid} / {len(seeds)} seeds with valid data")

    out_dir = os.path.join('reports', 'findings')
    plot_averaged(all_series, os.path.join(out_dir, 'seed_imr_dmi_vol_avg.png'))
    plot_examples(all_series, seeds, os.path.join(out_dir, 'seed_imr_dmi_vol_examples.png'),
                  n_examples=12)


if __name__ == '__main__':
    main()
