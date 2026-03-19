"""
Inverted Entry = Exit Signal -- detect when opposite-direction entry fires during a trade.

For each human seed (confirmed real trade), replays bars and checks:
  "Would peak detection + volume + DMI fire an entry for the OPPOSITE direction?"

The bar where this fires is the natural exit point -- the taper of one trade
is the approach of the next.

Captures the full entry criteria at each bar:
  - P_at_center jump (state change)
  - |F_momentum| decay (move exhausting)
  - Coherence > 0.55 (peak gate)
  - 1m volume aligned flipping negative (institutional flow reversing)
  - DMI diff aligned flipping negative (direction reversing)

An "inverted entry" fires when ALL of these say "enter opposite."

Usage:
    python tools/inverted_entry_exit.py

Output:
    reports/findings/inverted_entry_exit.txt
    reports/findings/inverted_entry_exit_chart.png
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
                'tf_sec': tf_sec,
            })
    return entries


def load_states(data_root):
    engine = StatisticalFieldEngine()
    result = {}
    for tf in ['1s', '1m']:
        tf_dir = os.path.join(data_root, tf)
        if not os.path.isdir(tf_dir):
            print(f"  WARNING: {tf_dir} not found")
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


def analyze_seed(seed, tf_data):
    """For each 1m bar during the trade, check if inverted entry fires.

    Sensor fusion:
      - 1s: velocity + F_momentum (fast, captures actual turn)
      - 1m: volume + DMI (slow, normalizes noise, confirms institutional flow)
      - Peak detection: P_at_center + F_momentum from 1m (state change)
    """
    ts_1m, states_1m = tf_data.get('1m', (np.array([]), []))
    ts_1s, states_1s = tf_data.get('1s', (np.array([]), []))
    if len(ts_1m) == 0:
        return None

    entry_ts = seed['ts_entry']
    exit_ts = seed['ts_exit']
    direction = seed['direction']
    trade_dir = 1 if direction == 'LONG' else -1

    entry_idx = int(np.searchsorted(ts_1m, entry_ts))
    exit_idx = int(np.searchsorted(ts_1m, exit_ts))

    if entry_idx >= len(states_1m) or exit_idx <= entry_idx:
        return None

    n_bars = exit_idx - entry_idx
    entry_st = states_1m[entry_idx]
    entry_price = getattr(entry_st, 'price', 0.0) or 0.0

    # 1m entry state for peak detection baseline
    entry_pac = getattr(entry_st, 'P_at_center', 0.0) or 0.0
    entry_fm_1m = abs(getattr(entry_st, 'F_momentum', 0.0) or 0.0)

    prev_pac = entry_pac
    prev_fm_abs = entry_fm_1m

    bars = []
    first_inverted = None
    first_partial = None

    for i in range(entry_idx, min(exit_idx + 5, len(states_1m))):
        st_1m = states_1m[i]
        offset = i - entry_idx
        in_trade = offset <= n_bars
        bar_ts = int(ts_1m[i]) if i < len(ts_1m) else entry_ts

        # ── 1m: volume + DMI + peak state (noise-normalized) ──
        pac = getattr(st_1m, 'P_at_center', 0.0) or 0.0
        fm_1m = getattr(st_1m, 'F_momentum', 0.0) or 0.0
        fm_1m_abs = abs(fm_1m)
        coh_1m = getattr(st_1m, 'oscillation_entropy_normalized', 0.0) or 0.0
        price = getattr(st_1m, 'price', 0.0) or 0.0
        dmi_p = getattr(st_1m, 'dmi_plus', 0.0) or 0.0
        dmi_m = getattr(st_1m, 'dmi_minus', 0.0) or 0.0
        dmi_aligned = (dmi_p - dmi_m) * trade_dir
        vol_1m = getattr(st_1m, 'volume_delta', 0.0) or 0.0
        vol_aligned = vol_1m * trade_dir
        fm_1m_aligned = fm_1m * trade_dir

        # Peak detection on 1m (state change at institutional scale)
        pac_up = pac > prev_pac * 1.05 if prev_pac > 0.01 else False
        fm_down = fm_1m_abs < prev_fm_abs * 0.90 if prev_fm_abs > 0.5 else False
        peak_fires_1m = (pac_up or fm_down) and coh_1m > 0.55

        prev_pac = pac
        prev_fm_abs = fm_1m_abs

        # ── 1s: velocity + F_momentum (fast turn detection) ──
        vel_1s = 0.0
        fm_1s = 0.0
        if len(ts_1s) > 0:
            idx_1s = int(np.searchsorted(ts_1s, bar_ts, side='right')) - 1
            idx_1s = max(0, min(idx_1s, len(states_1s) - 1))
            st_1s = states_1s[idx_1s]
            vel_1s = getattr(st_1s, 'velocity', 0.0) or 0.0
            fm_1s = getattr(st_1s, 'F_momentum', 0.0) or 0.0

        vel_1s_aligned = vel_1s * trade_dir
        fm_1s_aligned = fm_1s * trade_dir

        # Unrealized PnL
        unreal = (price - entry_price) / 0.25 * trade_dir if entry_price > 0 else 0

        # ── Inverted entry: opposite direction entry criteria ──
        # 1s signals (fast): velocity and momentum flipped against trade
        sig_vel_1s = vel_1s_aligned < 0     # 1s velocity against trade
        sig_fm_1s = fm_1s_aligned < 0       # 1s momentum against trade

        # 1m signals (slow, noise-normalized): volume, DMI, peak
        sig_vol_1m = vol_aligned < 0        # 1m volume against trade
        sig_dmi_1m = dmi_aligned < 0        # 1m DMI against trade
        sig_peak_1m = peak_fires_1m         # 1m state change

        n_signals = sum([sig_vel_1s, sig_fm_1s, sig_vol_1m, sig_dmi_1m, sig_peak_1m])

        # Full inverted: 1s turn detected + 1m confirms (vol + dmi)
        full_inverted = (sig_vel_1s and sig_vol_1m and sig_dmi_1m)
        # Strong: 4 of 5 agree
        strong_inverted = n_signals >= 4
        # Partial: 3 of 5
        partial_inverted = n_signals >= 3

        if in_trade and full_inverted and first_inverted is None:
            first_inverted = offset
        if in_trade and strong_inverted and first_partial is None:
            first_partial = offset

        bars.append({
            'offset': offset,
            'trade_pct': offset / max(n_bars, 1) * 100,
            'in_trade': in_trade,
            'unreal': unreal,
            # 1s fast sensors
            'vel_1s_aligned': vel_1s_aligned,
            'fm_1s_aligned': fm_1s_aligned,
            'sig_vel_1s': sig_vel_1s,
            'sig_fm_1s': sig_fm_1s,
            # 1m slow sensors
            'vol_1m_aligned': vol_aligned,
            'dmi_aligned': dmi_aligned,
            'fm_1m_aligned': fm_1m_aligned,
            'sig_vol': sig_vol_1m,
            'sig_dmi': sig_dmi_1m,
            'sig_fm_1m': fm_1m_aligned < 0,
            # Peak
            'peak_fires': peak_fires_1m,
            'P_at_center': pac,
            'F_momentum_abs': fm_1m_abs,
            'coherence': coh_1m,
            # Fusion
            'n_signals': n_signals,
            'full_inverted': full_inverted,
            'strong_inverted': strong_inverted,
            'partial_inverted': partial_inverted,
        })

    # MFE bar
    trade_bars = [b for b in bars if b['in_trade']]
    if not trade_bars:
        return None
    mfe_bar = max(trade_bars, key=lambda b: b['unreal'])
    mfe_offset = mfe_bar['offset']

    return {
        'n_bars': n_bars,
        'mfe_offset': mfe_offset,
        'mfe_pct': mfe_offset / max(n_bars, 1) * 100,
        'mfe_ticks': seed['mfe_ticks'],
        'first_inverted': first_inverted,
        'first_inverted_pct': first_inverted / max(n_bars, 1) * 100 if first_inverted is not None else None,
        'first_partial': first_partial,
        'first_partial_pct': first_partial / max(n_bars, 1) * 100 if first_partial is not None else None,
        'inv_vs_mfe': (first_inverted - mfe_offset) if first_inverted is not None else None,
        'partial_vs_mfe': (first_partial - mfe_offset) if first_partial is not None else None,
        'inv_vs_exit': (n_bars - first_inverted) if first_inverted is not None else None,
        'bars': bars,
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

    print("Analyzing...")
    results = []
    for seed in tqdm(seeds, desc="  Seeds"):
        r = analyze_seed(seed, tf_data)
        if r:
            results.append(r)

    print(f"  {len(results)} valid")

    # Report
    has_full = sum(1 for r in results if r['first_inverted'] is not None)
    has_partial = sum(1 for r in results if r['first_partial'] is not None)

    lines = []
    lines.append("=" * 80)
    lines.append("INVERTED ENTRY = EXIT SIGNAL")
    lines.append("When would the system enter the OPPOSITE direction during my trade?")
    lines.append("=" * 80)

    lines.append(f"\n  Seeds: {len(results)}")
    lines.append(f"  Full inverted signal (peak + vol + dmi + fm_1m): {has_full} ({has_full/len(results)*100:.0f}%)")
    lines.append(f"  Partial (peak + 1 confirm): {has_partial} ({has_partial/len(results)*100:.0f}%)")

    # Timing
    lines.append(f"\n  TIMING (% through trade):")
    lines.append(f"  {'Event':<25} | {'Mean':>8} | {'Median':>8} | {'P25':>8} | {'P75':>8}")
    lines.append(f"  {'-'*25} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")

    for label, key in [('MFE (profit peak)', 'mfe_pct'),
                        ('Full inverted entry', 'first_inverted_pct'),
                        ('Partial inverted entry', 'first_partial_pct')]:
        vals = [r[key] for r in results if r[key] is not None]
        if vals:
            lines.append(f"  {label:<25} | {np.mean(vals):>7.1f}% | {np.median(vals):>7.1f}% | "
                         f"{np.percentile(vals, 25):>7.1f}% | {np.percentile(vals, 75):>7.1f}%")

    # Gap: inverted signal vs MFE
    lines.append(f"\n  GAP: inverted signal vs MFE (bars):")
    lines.append(f"  Negative = fired BEFORE MFE (too early), Positive = fired AFTER MFE (gave back)")
    for label, key in [('Full inverted vs MFE', 'inv_vs_mfe'),
                        ('Partial vs MFE', 'partial_vs_mfe')]:
        vals = [r[key] for r in results if r[key] is not None]
        if vals:
            before = sum(1 for v in vals if v < 0)
            at = sum(1 for v in vals if v == 0)
            after = sum(1 for v in vals if v > 0)
            lines.append(f"  {label:<25}: mean={np.mean(vals):>+6.1f}  median={np.median(vals):>+5.0f}  "
                         f"before={before}  at={at}  after={after}")

    # Gap: inverted signal vs actual exit
    lines.append(f"\n  GAP: inverted signal vs actual exit (bars saved):")
    vals = [r['inv_vs_exit'] for r in results if r['inv_vs_exit'] is not None]
    if vals:
        lines.append(f"  Full inverted -> exit: mean={np.mean(vals):>.1f} bars earlier  "
                     f"median={np.median(vals):>.0f}  P25={np.percentile(vals, 25):>.0f}  "
                     f"P75={np.percentile(vals, 75):>.0f}")

    # What % of profit captured at inverted signal?
    lines.append(f"\n  PROFIT CAPTURED AT INVERTED SIGNAL:")
    for label, inv_key in [('Full inverted', 'first_inverted'),
                            ('Partial', 'first_partial')]:
        capture_pcts = []
        for r in results:
            if r[inv_key] is not None:
                inv_offset = r[inv_key]
                trade_bars = [b for b in r['bars'] if b['in_trade']]
                if trade_bars:
                    mfe_unreal = max(b['unreal'] for b in trade_bars)
                    inv_bar = next((b for b in r['bars'] if b['offset'] == inv_offset), None)
                    if inv_bar and mfe_unreal > 0:
                        capture_pcts.append(inv_bar['unreal'] / mfe_unreal * 100)
        if capture_pcts:
            lines.append(f"  {label:<20}: mean={np.mean(capture_pcts):>.1f}%  "
                         f"median={np.median(capture_pcts):>.1f}%  "
                         f"P25={np.percentile(capture_pcts, 25):>.1f}%  "
                         f"P75={np.percentile(capture_pcts, 75):>.1f}%")

    report = '\n'.join(lines) + '\n'
    out_dir = os.path.join('reports', 'findings')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'inverted_entry_exit.txt'), 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)

    # Chart: examples showing all 4 signals + inverted marker
    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(8, 1, figure=fig, hspace=0.35)
    fig.suptitle('Inverted Entry = Exit Signal\n'
                 'Averaged across all seeds: when do opposite-direction entry criteria fire?',
                 fontsize=14, fontweight='bold', y=0.99)

    # Bin all bars by trade %
    bins = np.arange(-10, 130, 5)
    bin_labels = bins[:-1] + 2.5

    channels = ['unreal', 'vel_1s_aligned', 'fm_1s_aligned', 'vol_1m_aligned',
                'dmi_aligned', 'fm_1m_aligned', 'n_signals', 'P_at_center']
    titles = [
        'Unrealized PnL (ticks)',
        '1s Velocity (aligned) -- fast turn sensor',
        '1s F_momentum (aligned) -- fast momentum',
        '1m Volume (aligned) -- institutional flow (neg = against trade)',
        '1m DMI Diff (aligned) -- direction (neg = against trade)',
        '1m F_momentum (aligned) -- institutional momentum',
        'N sensors agreeing on opposite entry (0-5)',
        'P_at_center 1m -- state change',
    ]

    binned = {c: {b: [] for b in bin_labels} for c in channels}
    inv_by_bin = {b: 0 for b in bin_labels}
    total_by_bin = {b: 0 for b in bin_labels}

    for r in results:
        for bar in r['bars']:
            if not bar['in_trade']:
                continue
            pct = bar['trade_pct']
            bi = int(np.searchsorted(bins, pct)) - 1
            if 0 <= bi < len(bin_labels):
                bl = bin_labels[bi]
                for c in channels:
                    v = bar.get(c)
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        binned[c][bl].append(v)
                total_by_bin[bl] += 1
                if bar.get('full_inverted', False):
                    inv_by_bin[bl] += 1

    for idx, (channel, title) in enumerate(zip(channels, titles)):
        ax = fig.add_subplot(gs[idx])
        xs, means, stds = [], [], []
        for bl in bin_labels:
            vals = binned[channel][bl]
            if vals:
                xs.append(bl)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
        xs, means, stds = np.array(xs), np.array(means), np.array(stds)

        ax.plot(xs, means, 'b-o', markersize=3, linewidth=1.5)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color='blue')
        ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Entry')
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Exit')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axvspan(85, 100, alpha=0.08, color='red')

        # Overlay inverted entry rate on PnL chart
        if idx == 0:
            ax2 = ax.twinx()
            inv_xs, inv_rates = [], []
            for bl in bin_labels:
                if total_by_bin[bl] > 0:
                    rate = inv_by_bin[bl] / total_by_bin[bl]
                    inv_xs.append(bl)
                    inv_rates.append(rate)
            ax2.bar(inv_xs, inv_rates, width=4, alpha=0.3, color='orange',
                    label='Inverted entry rate')
            ax2.set_ylabel('Inverted rate', color='orange')
            ax2.legend(loc='upper left', fontsize=8)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Trade progress (%)')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-10, 125)

    fig.savefig(os.path.join(out_dir, 'inverted_entry_exit_chart.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart: {os.path.join(out_dir, 'inverted_entry_exit_chart.png')}")


if __name__ == '__main__':
    main()
