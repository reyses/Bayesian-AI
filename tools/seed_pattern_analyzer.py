#!/usr/bin/env python
"""
Seed Pattern Deep-Dive Analyzer
================================
Extracts actual price waveforms from ATLAS for human-marked seeds,
classifies shapes, analyzes cross-TF nesting, timing patterns, and
produces a comprehensive report.

Usage:
    python tools/seed_pattern_analyzer.py --seeds-dir DATA/regime_seeds
    python tools/seed_pattern_analyzer.py --seeds-dir DATA/regime_seeds --plot
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.research.data import load_atlas_tf

TICK_SIZE = 0.25
TICK_VALUE = 0.50

# Shape classification — imported from shared module
from tools.research.shape_classifier import classify_shape  # noqa: E402


def load_seeds_by_tf(seeds_dir, data_dir):
    """Load all seed files and group by TF. Also load ATLAS price data."""
    # Find multi-TF merged file first
    multi_files = sorted(Path(seeds_dir).glob('seeds_*_multi.json'))
    tf_seeds = defaultdict(list)

    if multi_files:
        for mf in multi_files:
            with open(mf) as f:
                data = json.load(f)
            for s in data.get('seeds', []):
                tf = s.get('timeframe', '1m')
                tf_seeds[tf].append(s)
    else:
        # Load individual TF files
        for sf in sorted(Path(seeds_dir).glob('seeds_*.json')):
            if 'auto_' in sf.name or 'multi' in sf.name:
                continue
            with open(sf) as f:
                data = json.load(f)
            tf = data.get('timeframe', '1m')
            for s in data.get('seeds', []):
                s['timeframe'] = tf
                tf_seeds[tf].append(s)

    # Also load 1m auto-seeds for comparison
    auto_files = sorted(Path(seeds_dir).glob('auto_seeds_*.json'))
    for af in auto_files:
        with open(af) as f:
            data = json.load(f)
        if 'days' in data:
            for day_data in data['days'].values():
                for s in day_data.get('seeds', []):
                    s['timeframe'] = '1m'
                    tf_seeds['1m_auto'].append(s)
        else:
            for s in data.get('seeds', []):
                s['timeframe'] = '1m'
                tf_seeds['1m_auto'].append(s)

    # Load ATLAS price data for each TF
    tf_prices = {}
    for tf in set(list(tf_seeds.keys()) + ['1m']):
        real_tf = tf.replace('_auto', '')
        if real_tf not in tf_prices:
            df = load_atlas_tf(data_dir, real_tf)
            if not df.empty:
                tf_prices[real_tf] = {
                    'close': df['close'].values.astype(np.float64),
                    'high': df['high'].values.astype(np.float64),
                    'low': df['low'].values.astype(np.float64),
                    'timestamp': df['timestamp'].values.astype(np.float64),
                }
            del df

    return tf_seeds, tf_prices


def extract_waveform(seed, tf_prices, tf):
    """Extract the actual price waveform for a seed from ATLAS data."""
    real_tf = tf.replace('_auto', '')
    if real_tf not in tf_prices:
        return None, 0

    prices = tf_prices[real_tf]
    close = prices['close']
    high = prices['high']
    low = prices['low']

    lb_start = seed.get('lookback_start_idx', max(0, seed['start_idx'] - 10))
    # Use regime_start_idx if available, else start_idx
    entry_bar = seed.get('regime_start_idx', seed.get('start_idx', 0))
    end_bar = seed.get('end_idx', entry_bar + 5)

    if end_bar >= len(close):
        end_bar = len(close) - 1
    if lb_start >= len(close) or entry_bar >= len(close):
        return None, 0

    waveform = close[lb_start:end_bar + 1]
    entry_within = entry_bar - lb_start

    return waveform, entry_within


def analyze_1s_microstructure(seed, data_dir):
    """Extract 1s data within the seed's time window for microstructure analysis."""
    try:
        from tools.golden_path import load_1s_index, load_1s_window
        index_1s = load_1s_index(data_dir)
        cache = {}
        df_1s = load_1s_window(index_1s, seed['ts_start'], seed['ts_end'], cache)
        if len(df_1s) < 5:
            return None

        p = df_1s['close'].values.astype(float)
        entry = p[0]
        ticks = (p - entry) / TICK_SIZE

        direction = 1 if seed['direction'] == 'LONG' else -1
        fav = ticks * direction

        # Compute velocity profile (ticks per second, smoothed)
        dt = np.diff(df_1s['timestamp'].values.astype(float))
        dp = np.diff(ticks)
        dt[dt == 0] = 1
        velocity = dp / dt

        # Smooth with 30s window
        if len(velocity) > 30:
            kernel = np.ones(30) / 30
            velocity_smooth = np.convolve(velocity, kernel, mode='same')
        else:
            velocity_smooth = velocity

        return {
            'n_bars_1s': len(p),
            'max_velocity': float(np.max(np.abs(velocity_smooth))),
            'mean_velocity': float(np.mean(np.abs(velocity_smooth))),
            'velocity_std': float(np.std(velocity_smooth)),
            'time_in_profit_pct': float(np.mean(fav > 0)) * 100,
            'max_drawdown_ticks': float(np.min(fav)),
            'max_favorable_ticks': float(np.max(fav)),
            'path_roughness': float(np.mean(np.abs(np.diff(ticks)))),
        }
    except Exception:
        return None


def format_time(ts):
    """Format timestamp to readable time."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%a %H:%M')


def run_analysis(seeds_dir, data_dir, do_plot=False, do_1s=False):
    """Main analysis pipeline."""
    print('=' * 70)
    print('  SEED PATTERN DEEP-DIVE ANALYZER')
    print('=' * 70)

    tf_seeds, tf_prices = load_seeds_by_tf(seeds_dir, data_dir)

    if not tf_seeds:
        print("  No seeds found.")
        return

    print(f"\n  Loaded TFs: {', '.join(f'{tf}={len(seeds)}' for tf, seeds in sorted(tf_seeds.items()))}")

    report_lines = []

    def rpt(line=''):
        print(line)
        report_lines.append(line)

    # ═══════════════════════════════════════════════════════════
    # Per-TF Shape Classification
    # ═══════════════════════════════════════════════════════════
    all_classified = {}  # tf -> list of (seed, shape, confidence, features)

    for tf in ['1h', '15m', '5m', '1m_auto']:
        if tf not in tf_seeds:
            continue

        seeds = tf_seeds[tf]
        rpt(f"\n{'='*70}")
        rpt(f"  {tf.upper()} — {len(seeds)} SEEDS")
        rpt(f"{'='*70}")

        classified = []
        shape_counts = defaultdict(int)
        shape_stats = defaultdict(list)  # shape -> list of feature dicts

        for s in seeds:
            waveform, entry_idx = extract_waveform(s, tf_prices, tf)
            if waveform is None:
                continue

            shape, conf, feats = classify_shape(waveform, entry_idx)
            classified.append((s, shape, conf, feats))
            shape_counts[shape] += 1
            shape_stats[shape].append(feats)

        all_classified[tf] = classified

        # Shape distribution
        rpt(f"\n  SHAPE DISTRIBUTION:")
        total = len(classified)
        for shape, count in sorted(shape_counts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar = '#' * int(pct / 2)
            rpt(f"    {shape:22s} {count:4d} ({pct:5.1f}%) {bar}")

        # Per-shape statistics
        rpt(f"\n  PER-SHAPE STATISTICS:")
        rpt(f"    {'Shape':22s} {'Count':>5s} {'Med|Net|':>8s} {'MedEff':>7s} {'MedMono':>8s} {'MedRetrace':>11s} {'MedPeakAt%':>11s}")
        rpt(f"    {'-'*22} {'-'*5} {'-'*8} {'-'*7} {'-'*8} {'-'*11} {'-'*11}")

        for shape in sorted(shape_stats.keys(), key=lambda s: -len(shape_stats[s])):
            feats_list = shape_stats[shape]
            nets = [f['abs_net'] for f in feats_list]
            effs = [f['efficiency'] for f in feats_list]
            monos = [f['monotonic_frac'] for f in feats_list]
            retrs = [f['retracement'] for f in feats_list]
            peaks = [f['peak_at_pct'] for f in feats_list]

            rpt(f"    {shape:22s} {len(feats_list):5d} {np.median(nets):7.0f}t {np.median(effs):7.2f} "
                f"{np.median(monos):8.2f} {np.median(retrs):11.2f} {np.median(peaks)*100:10.0f}%")

        # Individual seed detail (for 1h, show all; for others, top 10 by |net|)
        if tf == '1h':
            detail_seeds = classified
        else:
            detail_seeds = sorted(classified, key=lambda x: abs(x[3].get('net_ticks', 0)), reverse=True)[:15]

        rpt(f"\n  {'DETAILED SEED BREAKDOWN' if tf == '1h' else 'TOP 15 BY |NET MOVE|'}:")
        rpt(f"    {'ID':>3s} {'Dir':>5s} {'Shape':>20s} {'Conf':>5s} {'Net':>7s} {'Eff':>5s} "
            f"{'Mono':>5s} {'Retr':>5s} {'Peak%':>6s} {'LbTrend':>8s} {'Time':>10s} {'Dur':>5s}")
        rpt(f"    {'-'*3} {'-'*5} {'-'*20} {'-'*5} {'-'*7} {'-'*5} "
            f"{'-'*5} {'-'*5} {'-'*6} {'-'*8} {'-'*10} {'-'*5}")

        for s, shape, conf, feats in detail_seeds:
            tid = s.get('trade_id', '?')
            d = s['direction'][:1]
            net = feats['net_ticks']
            eff = feats['efficiency']
            mono = feats['monotonic_frac']
            retr = feats['retracement']
            peak_pct = feats['peak_at_pct'] * 100
            lb = feats['lb_trend']
            time_str = format_time(s['ts_start'])
            dur = s.get('duration_mins', 0)

            rpt(f"    {tid:3} {d:>5s} {shape:>20s} {conf:5.2f} {net:+7.0f} {eff:5.2f} "
                f"{mono:5.2f} {retr:5.2f} {peak_pct:5.0f}% {lb:+8.1f} {time_str:>10s} {dur:5.0f}m")

    # ═══════════════════════════════════════════════════════════
    # Cross-TF Nesting Analysis
    # ═══════════════════════════════════════════════════════════
    rpt(f"\n{'='*70}")
    rpt(f"  CROSS-TF NESTING: WHAT MICRO PATTERNS LIVE INSIDE MACRO MOVES?")
    rpt(f"{'='*70}")

    if '1h' in all_classified and '15m' in all_classified:
        rpt(f"\n  1h → 15m DECOMPOSITION:")
        for s_1h, shape_1h, conf_1h, feats_1h in all_classified['1h']:
            ts_start = s_1h['ts_start']
            ts_end = s_1h['ts_end']
            tid = s_1h.get('trade_id', '?')

            # Find 15m seeds within this 1h window
            nested_15m = [(s, sh, c, f) for s, sh, c, f in all_classified['15m']
                          if s['ts_start'] >= ts_start and s['ts_start'] < ts_end]

            if not nested_15m:
                rpt(f"\n    1h T{tid} ({s_1h['direction']}, {shape_1h}): NO nested 15m seeds")
                continue

            nested_shapes = [sh for _, sh, _, _ in nested_15m]
            nested_dirs = [s['direction'] for s, _, _, _ in nested_15m]
            aligned = sum(1 for d in nested_dirs if d == s_1h['direction'])

            rpt(f"\n    1h T{tid} ({s_1h['direction']}, {shape_1h}, {feats_1h['abs_net']:.0f}t, "
                f"{s_1h.get('duration_mins',0):.0f}m):")
            rpt(f"      Contains {len(nested_15m)} x 15m seeds, {aligned}/{len(nested_15m)} aligned")
            shape_summary = defaultdict(int)
            for sh in nested_shapes:
                shape_summary[sh] += 1
            rpt(f"      15m shapes: {dict(shape_summary)}")

            # Sequence
            seq = ' -> '.join(f"{s['direction'][:1]}:{sh}" for s, sh, _, _ in nested_15m)
            rpt(f"      Sequence: {seq}")

    if '1h' in all_classified and '5m' in all_classified:
        rpt(f"\n  1h → 5m DECOMPOSITION (top 5 1h by |net|):")
        top_1h = sorted(all_classified['1h'], key=lambda x: abs(x[3].get('net_ticks', 0)), reverse=True)[:5]

        for s_1h, shape_1h, conf_1h, feats_1h in top_1h:
            ts_start = s_1h['ts_start']
            ts_end = s_1h['ts_end']
            tid = s_1h.get('trade_id', '?')

            nested_5m = [(s, sh, c, f) for s, sh, c, f in all_classified['5m']
                         if s['ts_start'] >= ts_start and s['ts_start'] < ts_end]

            if not nested_5m:
                continue

            nested_dirs = [s['direction'] for s, _, _, _ in nested_5m]
            aligned = sum(1 for d in nested_dirs if d == s_1h['direction'])

            shape_summary = defaultdict(int)
            for _, sh, _, _ in nested_5m:
                shape_summary[sh] += 1

            rpt(f"\n    1h T{tid} ({s_1h['direction']}, {shape_1h}, {feats_1h['abs_net']:.0f}t):")
            rpt(f"      Contains {len(nested_5m)} x 5m, {aligned}/{len(nested_5m)} aligned, "
                f"shapes: {dict(shape_summary)}")

    # ═══════════════════════════════════════════════════════════
    # Time-of-Day Patterns
    # ═══════════════════════════════════════════════════════════
    rpt(f"\n{'='*70}")
    rpt(f"  TIME-OF-DAY PATTERN ANALYSIS")
    rpt(f"{'='*70}")

    for tf in ['1h', '15m', '5m']:
        if tf not in all_classified:
            continue

        rpt(f"\n  {tf.upper()} by session:")

        session_shapes = defaultdict(lambda: defaultdict(int))
        session_quality = defaultdict(list)

        for s, shape, conf, feats in all_classified[tf]:
            hour = datetime.fromtimestamp(s['ts_start'], tz=timezone.utc).hour

            # Map to session (UTC hours, MNQ = CME)
            if 23 <= hour or hour < 8:
                session = 'ASIA (23-08 UTC)'
            elif 8 <= hour < 13:
                session = 'EUROPE (08-13 UTC)'
            elif 13 <= hour < 21:
                session = 'US (13-21 UTC)'
            else:
                session = 'OVERLAP (21-23 UTC)'

            session_shapes[session][shape] += 1
            session_quality[session].append(feats['abs_net'])

        for session in sorted(session_shapes.keys()):
            shapes = session_shapes[session]
            nets = session_quality[session]
            total = sum(shapes.values())
            rpt(f"    {session}: {total} seeds, median |net|={np.median(nets):.0f}t")
            for sh, cnt in sorted(shapes.items(), key=lambda x: -x[1])[:5]:
                rpt(f"      {sh:20s}: {cnt} ({cnt/total*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════
    # Sequence Analysis: What follows what?
    # ═══════════════════════════════════════════════════════════
    rpt(f"\n{'='*70}")
    rpt(f"  SEQUENCE ANALYSIS: WHAT SHAPE FOLLOWS WHAT?")
    rpt(f"{'='*70}")

    for tf in ['1h', '15m', '5m']:
        if tf not in all_classified or len(all_classified[tf]) < 3:
            continue

        rpt(f"\n  {tf.upper()} transition matrix (row=current, col=next):")
        classified = all_classified[tf]
        shapes_in_order = [sh for _, sh, _, _ in classified]

        # Build transition counts
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(shapes_in_order) - 1):
            transitions[shapes_in_order[i]][shapes_in_order[i + 1]] += 1

        # Get all shapes
        all_shapes = sorted(set(shapes_in_order))

        # Print matrix
        header = f"    {'From \\ To':>20s}"
        for sh in all_shapes:
            header += f" {sh[:8]:>8s}"
        rpt(header)

        for from_sh in all_shapes:
            row = f"    {from_sh:>20s}"
            row_total = sum(transitions[from_sh].values())
            for to_sh in all_shapes:
                cnt = transitions[from_sh][to_sh]
                if cnt > 0 and row_total > 0:
                    row += f" {cnt:>4d}({cnt/row_total*100:2.0f}%)"
                else:
                    row += f" {'--':>8s}"
            rpt(row)

    # ═══════════════════════════════════════════════════════════
    # Direction Persistence Analysis
    # ═══════════════════════════════════════════════════════════
    rpt(f"\n{'='*70}")
    rpt(f"  DIRECTION PERSISTENCE: HOW LONG DO STREAKS LAST?")
    rpt(f"{'='*70}")

    for tf in ['1h', '15m', '5m']:
        if tf not in all_classified:
            continue

        classified = all_classified[tf]
        dirs = [s['direction'] for s, _, _, _ in classified]

        streaks = []
        current_dir = dirs[0]
        current_len = 1
        current_net = all_classified[tf][0][3]['net_ticks']

        for i in range(1, len(dirs)):
            if dirs[i] == current_dir:
                current_len += 1
                current_net += all_classified[tf][i][3]['net_ticks']
            else:
                streaks.append((current_dir, current_len, current_net))
                current_dir = dirs[i]
                current_len = 1
                current_net = all_classified[tf][i][3]['net_ticks']
        streaks.append((current_dir, current_len, current_net))

        streak_lens = [s[1] for s in streaks]
        rpt(f"\n  {tf.upper()}: {len(streaks)} direction streaks")
        rpt(f"    Streak lengths: min={min(streak_lens)}, median={np.median(streak_lens):.0f}, "
            f"max={max(streak_lens)}, mean={np.mean(streak_lens):.1f}")

        # Show longest streaks
        long_streaks = sorted(streaks, key=lambda x: x[1], reverse=True)[:5]
        rpt(f"    Top 5 streaks:")
        for d, length, net in long_streaks:
            rpt(f"      {d} x{length} ({net:+.0f}t net)")

    # ═══════════════════════════════════════════════════════════
    # MFE Timing Analysis
    # ═══════════════════════════════════════════════════════════
    rpt(f"\n{'='*70}")
    rpt(f"  MFE TIMING: WHEN DOES THE PEAK HAPPEN?")
    rpt(f"{'='*70}")

    for tf in ['1h', '15m', '5m']:
        if tf not in tf_seeds:
            continue

        seeds = tf_seeds[tf]
        early = 0  # MFE in first 30%
        mid = 0    # MFE in middle 40%
        late = 0   # MFE in last 30%

        mfe_pcts = []
        for s in seeds:
            dur = s.get('duration_mins', 1)
            t2mfe = s.get('time_to_mfe_mins', 0)
            if dur > 0:
                pct = t2mfe / dur
                mfe_pcts.append(pct)
                if pct < 0.3:
                    early += 1
                elif pct > 0.7:
                    late += 1
                else:
                    mid += 1

        total = early + mid + late
        if total > 0:
            rpt(f"\n  {tf.upper()}: MFE timing distribution ({total} seeds)")
            rpt(f"    Early (<30%):  {early:3d} ({early/total*100:5.1f}%)")
            rpt(f"    Middle (30-70%): {mid:3d} ({mid/total*100:5.1f}%)")
            rpt(f"    Late (>70%):   {late:3d} ({late/total*100:5.1f}%)")
            rpt(f"    Median MFE at: {np.median(mfe_pcts)*100:.0f}% of duration")

    # ═══════════════════════════════════════════════════════════
    # 1s Microstructure (optional, expensive)
    # ═══════════════════════════════════════════════════════════
    if do_1s and '1h' in all_classified:
        rpt(f"\n{'='*70}")
        rpt(f"  1s MICROSTRUCTURE (1h seeds)")
        rpt(f"{'='*70}")

        for s, shape, conf, feats in all_classified['1h']:
            micro = analyze_1s_microstructure(s, data_dir)
            if micro:
                tid = s.get('trade_id', '?')
                rpt(f"\n    T{tid} ({s['direction']}, {shape}):")
                rpt(f"      1s bars: {micro['n_bars_1s']}, max velocity: {micro['max_velocity']:.2f} t/s")
                rpt(f"      Mean velocity: {micro['mean_velocity']:.3f} t/s, "
                    f"roughness: {micro['path_roughness']:.2f}t")
                rpt(f"      Time in profit: {micro['time_in_profit_pct']:.0f}%")
                rpt(f"      Max favorable: {micro['max_favorable_ticks']:.0f}t, "
                    f"max adverse: {micro['max_drawdown_ticks']:.0f}t")

    # ═══════════════════════════════════════════════════════════
    # Plot (optional)
    # ═══════════════════════════════════════════════════════════
    if do_plot:
        save_waveform_gallery(all_classified, tf_prices, seeds_dir)

    # Save report
    report_path = os.path.join('reports', 'findings', f'seed_pattern_deep_dive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    rpt(f"\n  Report saved: {report_path}")


def save_waveform_gallery(all_classified, tf_prices, seeds_dir):
    """Save a gallery of waveforms per TF, grouped by shape."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for tf in ['1h', '15m', '5m']:
        if tf not in all_classified:
            continue

        classified = all_classified[tf]
        shapes = sorted(set(sh for _, sh, _, _ in classified))

        n_shapes = len(shapes)
        if n_shapes == 0:
            continue

        fig, axes = plt.subplots(2, min(5, n_shapes), figsize=(4 * min(5, n_shapes), 8),
                                 squeeze=False)

        for i, shape in enumerate(shapes[:10]):
            if i >= 10:
                break
            ax = axes[i // 5][i % 5]

            members = [(s, f) for s, sh, c, f in classified if sh == shape]
            for s, feats in members[:20]:
                waveform, entry_idx = extract_waveform(s, tf_prices, tf)
                if waveform is None:
                    continue

                entry = waveform[entry_idx]
                norm = (waveform - entry) / TICK_SIZE
                x = np.linspace(0, 1, len(norm))

                color = 'green' if s['direction'] == 'LONG' else 'red'
                ax.plot(x, norm, color=color, alpha=0.3, linewidth=0.5)

                # Mark entry
                ax.axvline(x=entry_idx / max(len(norm) - 1, 1), color='blue',
                           linestyle='--', alpha=0.3, linewidth=0.5)

            ax.set_title(f"{shape}\n({len(members)} seeds)", fontsize=9)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('Ticks' if i % 5 == 0 else '')

        # Hide empty axes
        for i in range(n_shapes, axes.shape[0] * axes.shape[1]):
            axes[i // 5][i % 5].set_visible(False)

        plt.suptitle(f'{tf.upper()} Waveform Gallery (green=LONG, red=SHORT)', fontsize=12)
        plt.tight_layout()
        plot_path = os.path.join('reports', 'findings', f'waveform_gallery_{tf}.png')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Gallery saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Seed Pattern Deep-Dive Analyzer')
    parser.add_argument('--seeds-dir', default='DATA/regime_seeds')
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--plot', action='store_true', help='Save waveform gallery plots')
    parser.add_argument('--1s', dest='do_1s', action='store_true',
                        help='Include 1s microstructure analysis (slow)')
    args = parser.parse_args()

    run_analysis(args.seeds_dir, args.data, do_plot=args.plot, do_1s=args.do_1s)


if __name__ == '__main__':
    main()
