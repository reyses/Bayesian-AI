"""
Nightmare EDA — Deep analysis of losing exit types.

Loads trade log + trajectory from a Nightmare ticker run and produces
a structured markdown report + CSV with raw analysis data.

Targets three problem exits:
  1. mean_reached — high WR but tiny wins, big losses
  2. trend_breakeven_protect — 0% WR, every exit is a loss
  3. trend_exhausted — low WR, big losses

Analysis per exit:
  - PnL distribution (histogram bins, median, mode, percentiles)
  - Hold time distribution
  - Entry conditions (z_se, variance_ratio, strategy)
  - Winners vs losers comparison (mean_reached)
  - Post-exit price movement (trend_breakeven_protect, trend_exhausted)
  - Time-of-day breakdown
  - Strategy breakdown

Usage:
  python tools/nightmare_eda.py
  python tools/nightmare_eda.py --trades reports/findings/nightmare_2026-01-06_to_2026-02-09_trades.csv
  python tools/nightmare_eda.py --trades TRADES.csv --trajectory TRAJ.csv --label my_run
"""

import argparse
import os
import sys
import glob
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
TV = 0.50

# ─── Default file paths ───────────────────────────────────────────────
DEFAULT_TRADES = 'reports/findings/nightmare_2026-01-06_to_2026-02-09_trades.csv'
DEFAULT_TRAJ = 'reports/findings/nightmare_2026-01-06_to_2026-02-09_trajectory.csv'

# Problem exits to analyze
PROBLEM_EXITS = ['mean_reached', 'trend_breakeven_protect', 'trend_exhausted']

# Post-exit lookahead (1m bars after exit)
POST_EXIT_BARS = [5, 10, 15, 20, 30]


def parse_args():
    p = argparse.ArgumentParser(description='Nightmare EDA — problem exit analysis')
    p.add_argument('--trades', default=DEFAULT_TRADES, help='Path to trades CSV')
    p.add_argument('--trajectory', default=DEFAULT_TRAJ, help='Path to trajectory CSV')
    p.add_argument('--label', default='clean_20260402', help='Label for output files')
    p.add_argument('--no-post-exit', action='store_true',
                   help='Skip post-exit analysis (avoids loading 1m ATLAS)')
    return p.parse_args()


def load_data(args):
    """Load trade log and trajectory."""
    trades = pd.read_csv(args.trades)
    traj = pd.read_csv(args.trajectory) if os.path.exists(args.trajectory) else None
    print(f'Loaded {len(trades)} trades from {args.trades}')
    if traj is not None:
        print(f'Loaded {len(traj)} trajectory bars from {args.trajectory}')
    return trades, traj


def load_1m_atlas(date_range):
    """Load 1m ATLAS data for post-exit analysis.

    date_range: tuple of (start_date_str, end_date_str) like ('2026-01-06', '2026-02-09')
    Returns sorted DataFrame with timestamp + close columns.
    """
    files = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
    # Filter to relevant date range
    start_ds = date_range[0].replace('-', '_')
    end_ds = date_range[1].replace('-', '_')
    relevant = [f for f in files if os.path.basename(f).replace('.parquet', '') >= start_ds
                and os.path.basename(f).replace('.parquet', '') <= end_ds]

    if not relevant:
        print(f'  WARNING: No 1m ATLAS files found for {date_range}')
        return None

    dfs = [pd.read_parquet(f)[['timestamp', 'close', 'high', 'low']] for f in relevant]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    print(f'  Loaded {len(df)} 1m bars ({len(relevant)} files) for post-exit analysis')
    return df


def pnl_distribution(series, label):
    """Compute PnL distribution stats."""
    if len(series) == 0:
        return {}
    stats = {
        f'{label}_count': len(series),
        f'{label}_mean': series.mean(),
        f'{label}_median': series.median(),
        f'{label}_std': series.std(),
        f'{label}_min': series.min(),
        f'{label}_max': series.max(),
        f'{label}_p10': np.percentile(series, 10),
        f'{label}_p25': np.percentile(series, 25),
        f'{label}_p75': np.percentile(series, 75),
        f'{label}_p90': np.percentile(series, 90),
        f'{label}_sum': series.sum(),
    }
    # Mode (rounded to $1)
    rounded = (series / 1).round() * 1
    mode_counts = Counter(rounded)
    top3 = mode_counts.most_common(3)
    stats[f'{label}_mode_top3'] = str(top3)
    return stats


def hold_time_distribution(series, label):
    """Compute hold time stats."""
    if len(series) == 0:
        return {}
    return {
        f'{label}_hold_mean': series.mean(),
        f'{label}_hold_median': series.median(),
        f'{label}_hold_min': series.min(),
        f'{label}_hold_max': series.max(),
        f'{label}_hold_p25': np.percentile(series, 25),
        f'{label}_hold_p75': np.percentile(series, 75),
    }


def entry_conditions(traj, trade_ids, label):
    """Extract entry conditions (bar=1) for given trade IDs."""
    if traj is None or len(trade_ids) == 0:
        return {}, None

    entries = traj[(traj['trade_id'].isin(trade_ids)) & (traj['bar'] == 1)].copy()
    if len(entries) == 0:
        return {}, None

    stats = {}
    for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15']:
        if col in entries.columns:
            s = entries[col].dropna()
            if len(s) > 0:
                stats[f'{label}_entry_{col}_mean'] = s.mean()
                stats[f'{label}_entry_{col}_median'] = s.median()
                stats[f'{label}_entry_{col}_std'] = s.std()
                stats[f'{label}_entry_{col}_p25'] = np.percentile(s, 25)
                stats[f'{label}_entry_{col}_p75'] = np.percentile(s, 75)
    return stats, entries


def time_of_day_analysis(trades_subset, label):
    """Break down PnL by hour of day."""
    if len(trades_subset) == 0:
        return {}, None

    df = trades_subset.copy()
    # Parse time (HH:MM format)
    df['hour'] = df['time'].apply(lambda t: int(str(t).split(':')[0]) if ':' in str(t) else -1)
    hourly = df.groupby('hour').agg(
        n_trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        mean_pnl=('pnl', 'mean'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
    ).reset_index()
    hourly.columns = ['hour', 'n_trades', 'total_pnl', 'mean_pnl', 'wr']

    # Find worst hours
    worst_hours = hourly.nsmallest(3, 'total_pnl')
    stats = {}
    for i, (_, row) in enumerate(worst_hours.iterrows()):
        stats[f'{label}_worst_hour_{i+1}'] = f"H{int(row['hour']):02d}: {int(row['n_trades'])} trades, ${row['total_pnl']:.0f}, WR={row['wr']:.0f}%"
    return stats, hourly


def strategy_breakdown(trades_subset, label):
    """Break down by strategy."""
    if len(trades_subset) == 0:
        return {}, None

    grouped = trades_subset.groupby('strategy').agg(
        n_trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        mean_pnl=('pnl', 'mean'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
        avg_held=('held', 'mean'),
    ).reset_index()
    grouped.columns = ['strategy', 'n_trades', 'total_pnl', 'mean_pnl', 'wr', 'avg_held']

    stats = {}
    for _, row in grouped.iterrows():
        stats[f'{label}_strat_{row["strategy"]}_n'] = row['n_trades']
        stats[f'{label}_strat_{row["strategy"]}_pnl'] = row['total_pnl']
        stats[f'{label}_strat_{row["strategy"]}_wr'] = row['wr']
    return stats, grouped


def winners_vs_losers(trades_subset, traj, label):
    """Compare entry conditions of winners vs losers (for mean_reached)."""
    if traj is None or len(trades_subset) == 0:
        return {}, None

    winners = trades_subset[trades_subset['pnl'] > 0]
    losers = trades_subset[trades_subset['pnl'] <= 0]

    w_stats, w_entries = entry_conditions(traj, winners['trade_id'], f'{label}_win')
    l_stats, l_entries = entry_conditions(traj, losers['trade_id'], f'{label}_loss')

    stats = {**w_stats, **l_stats}

    # Compute deltas for key features
    comparison = None
    if w_entries is not None and l_entries is not None and len(w_entries) > 0 and len(l_entries) > 0:
        rows = []
        for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15']:
            if col in w_entries.columns and col in l_entries.columns:
                w_vals = w_entries[col].dropna()
                l_vals = l_entries[col].dropna()
                if len(w_vals) > 0 and len(l_vals) > 0:
                    rows.append({
                        'feature': col,
                        'winner_mean': w_vals.mean(),
                        'winner_median': w_vals.median(),
                        'loser_mean': l_vals.mean(),
                        'loser_median': l_vals.median(),
                        'delta_mean': w_vals.mean() - l_vals.mean(),
                    })
        if rows:
            comparison = pd.DataFrame(rows)

    # Also compare hold time and peak PnL
    if len(winners) > 0 and len(losers) > 0:
        stats[f'{label}_win_avg_held'] = winners['held'].mean()
        stats[f'{label}_loss_avg_held'] = losers['held'].mean()
        stats[f'{label}_win_avg_peak'] = winners['peak'].mean()
        stats[f'{label}_loss_avg_peak'] = losers['peak'].mean()
        stats[f'{label}_win_avg_pnl'] = winners['pnl'].mean()
        stats[f'{label}_loss_avg_pnl'] = losers['pnl'].mean()

    return stats, comparison


def post_exit_analysis(trades_subset, traj, atlas_1m, label):
    """Analyze what happens to price AFTER the exit.

    For each trade, find the exit timestamp from trajectory (last bar),
    then look forward in 1m ATLAS to see if price recovered.
    """
    if atlas_1m is None or traj is None or len(trades_subset) == 0:
        return {}, None

    ts_atlas = atlas_1m['timestamp'].values
    close_atlas = atlas_1m['close'].values

    results = []
    for _, trade in trades_subset.iterrows():
        tid = trade['trade_id']
        direction = trade['dir']

        # Get exit timestamp from trajectory (last bar of this trade)
        trade_traj = traj[traj['trade_id'] == tid]
        if len(trade_traj) == 0:
            continue
        exit_ts = trade_traj['timestamp'].iloc[-1]
        exit_price = trade_traj['price'].iloc[-1]

        # Find position in atlas after exit
        idx = np.searchsorted(ts_atlas, exit_ts, side='right')
        if idx >= len(ts_atlas):
            continue

        row = {'trade_id': tid, 'exit_pnl': trade['pnl'], 'direction': direction,
               'exit_price': exit_price, 'strategy': trade['strategy']}

        # Look forward N bars
        for n_bars in POST_EXIT_BARS:
            end_idx = min(idx + n_bars, len(ts_atlas) - 1)
            if end_idx <= idx:
                row[f'post_{n_bars}bar_move'] = np.nan
                row[f'post_{n_bars}bar_would_profit'] = np.nan
                continue

            future_slice = close_atlas[idx:end_idx + 1]
            # Move in trade direction (positive = price moved in your favor after exit)
            if direction == 'LONG':
                best_future = np.max(future_slice) - exit_price
                worst_future = np.min(future_slice) - exit_price
                final_move = future_slice[-1] - exit_price
            else:
                best_future = exit_price - np.min(future_slice)
                worst_future = exit_price - np.max(future_slice)
                final_move = exit_price - future_slice[-1]

            row[f'post_{n_bars}bar_best'] = best_future
            row[f'post_{n_bars}bar_worst'] = worst_future
            row[f'post_{n_bars}bar_final'] = final_move
            # Would still holding have been profitable?
            row[f'post_{n_bars}bar_would_profit'] = 1 if final_move > 0 else 0

        results.append(row)

    if not results:
        return {}, None

    post_df = pd.DataFrame(results)

    stats = {}
    for n_bars in POST_EXIT_BARS:
        col_final = f'post_{n_bars}bar_final'
        col_profit = f'post_{n_bars}bar_would_profit'
        col_best = f'post_{n_bars}bar_best'
        if col_final in post_df.columns:
            valid = post_df[col_final].dropna()
            if len(valid) > 0:
                stats[f'{label}_post{n_bars}_mean_move'] = valid.mean()
                stats[f'{label}_post{n_bars}_median_move'] = valid.median()
                # Move in points
                stats[f'{label}_post{n_bars}_mean_pts'] = valid.mean() / TICK
        if col_profit in post_df.columns:
            valid = post_df[col_profit].dropna()
            if len(valid) > 0:
                stats[f'{label}_post{n_bars}_recovery_rate'] = valid.mean() * 100
        if col_best in post_df.columns:
            valid = post_df[col_best].dropna()
            if len(valid) > 0:
                stats[f'{label}_post{n_bars}_best_mean'] = valid.mean()

    return stats, post_df


def direction_breakdown(trades_subset, label):
    """Break down by direction (LONG vs SHORT)."""
    if len(trades_subset) == 0:
        return {}
    stats = {}
    for d in ['LONG', 'SHORT']:
        dt = trades_subset[trades_subset['dir'] == d]
        if len(dt) > 0:
            stats[f'{label}_{d}_n'] = len(dt)
            stats[f'{label}_{d}_pnl'] = dt['pnl'].sum()
            stats[f'{label}_{d}_wr'] = (dt['pnl'] > 0).mean() * 100
            stats[f'{label}_{d}_avg_pnl'] = dt['pnl'].mean()
    return stats


def day_of_week_analysis(trades_subset, label):
    """Break down by day of week."""
    if len(trades_subset) == 0 or 'day' not in trades_subset.columns:
        return {}, None

    df = trades_subset.copy()
    df['dow'] = pd.to_datetime(df['day']).dt.day_name()
    grouped = df.groupby('dow').agg(
        n_trades=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        mean_pnl=('pnl', 'mean'),
        wr=('pnl', lambda x: (x > 0).mean() * 100),
    ).reset_index()
    grouped.columns = ['dow', 'n_trades', 'total_pnl', 'mean_pnl', 'wr']

    # Order by day
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped['dow_order'] = grouped['dow'].apply(lambda x: day_order.index(x) if x in day_order else 7)
    grouped = grouped.sort_values('dow_order').drop('dow_order', axis=1)

    stats = {}
    for _, row in grouped.iterrows():
        stats[f'{label}_dow_{row["dow"][:3]}_pnl'] = row['total_pnl']
        stats[f'{label}_dow_{row["dow"][:3]}_n'] = row['n_trades']
    return stats, grouped


# ─── Report formatting ────────────────────────────────────────────────

def fmt_pnl_section(pnl_stats, hold_stats, label):
    """Format PnL + hold time stats for markdown."""
    lines = []
    prefix = label
    s = pnl_stats
    if f'{prefix}_count' in s:
        lines.append(f"| Trades | {s[f'{prefix}_count']} |")
        lines.append(f"| Total PnL | ${s[f'{prefix}_sum']:,.2f} |")
        lines.append(f"| Mean PnL | ${s[f'{prefix}_mean']:,.2f} |")
        lines.append(f"| Median PnL | ${s[f'{prefix}_median']:,.2f} |")
        lines.append(f"| Std Dev | ${s[f'{prefix}_std']:,.2f} |")
        lines.append(f"| Min | ${s[f'{prefix}_min']:,.2f} |")
        lines.append(f"| P10 | ${s[f'{prefix}_p10']:,.2f} |")
        lines.append(f"| P25 | ${s[f'{prefix}_p25']:,.2f} |")
        lines.append(f"| P75 | ${s[f'{prefix}_p75']:,.2f} |")
        lines.append(f"| P90 | ${s[f'{prefix}_p90']:,.2f} |")
        lines.append(f"| Max | ${s[f'{prefix}_max']:,.2f} |")
        lines.append(f"| Mode (top 3) | {s.get(f'{prefix}_mode_top3', 'N/A')} |")

    h = hold_stats
    if f'{prefix}_hold_mean' in h:
        lines.append(f"| Hold Mean | {h[f'{prefix}_hold_mean']:.1f} bars |")
        lines.append(f"| Hold Median | {h[f'{prefix}_hold_median']:.1f} bars |")
        lines.append(f"| Hold Range | {h[f'{prefix}_hold_min']:.0f}-{h[f'{prefix}_hold_max']:.0f} bars |")
    return lines


def fmt_post_exit_section(post_stats, label):
    """Format post-exit analysis for markdown."""
    lines = []
    for n_bars in POST_EXIT_BARS:
        key_rec = f'{label}_post{n_bars}_recovery_rate'
        key_mean = f'{label}_post{n_bars}_mean_pts'
        key_best = f'{label}_post{n_bars}_best_mean'
        if key_rec in post_stats:
            rec = post_stats[key_rec]
            mean_pts = post_stats.get(key_mean, 0)
            best = post_stats.get(key_best, 0)
            lines.append(f"| +{n_bars} bars | {rec:.1f}% would recover | avg move {mean_pts:.1f} ticks | best avg {best/TICK:.1f} ticks |")
    return lines


def generate_report(all_stats, analyses, args):
    """Generate the markdown report."""
    trades = analyses['trades']
    traj = analyses.get('traj')

    lines = []
    lines.append('# Nightmare EDA — Problem Exit Deep Dive')
    lines.append(f'')
    lines.append(f'**Run**: {args.trades}')
    lines.append(f'**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append(f'**Period**: {trades["day"].min()} to {trades["day"].max()}')
    lines.append(f'**Total trades**: {len(trades)} | **Total PnL**: ${trades["pnl"].sum():,.2f}')
    lines.append('')

    # Summary table of all exits
    lines.append('## All Exit Types — Overview')
    lines.append('')
    lines.append('| Exit | N | WR | Total PnL | Avg PnL | Avg Hold |')
    lines.append('|------|---|----|-----------|---------| ---------|')
    for ex in sorted(trades['exit'].unique()):
        et = trades[trades['exit'] == ex]
        wr = (et['pnl'] > 0).mean() * 100
        lines.append(f'| {ex} | {len(et)} | {wr:.1f}% | ${et["pnl"].sum():,.2f} | ${et["pnl"].mean():,.2f} | {et["held"].mean():.1f} |')
    lines.append('')

    # ─── EXIT 1: mean_reached ─────────────────────────────────────────
    lines.append('---')
    lines.append('## 1. mean_reached')
    lines.append('')
    lines.append('**Hypothesis**: Mean-reversion trades that "reach" the mean but had')
    lines.append('large adverse excursions before getting there, resulting in tiny wins')
    lines.append('and oversized losses when the mean is reached at a bad time.')
    lines.append('')

    label = 'mean_reached'
    pnl_s = all_stats.get(f'{label}_pnl', {})
    hold_s = all_stats.get(f'{label}_hold', {})
    lines.append('### PnL + Hold Time')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.extend(fmt_pnl_section(pnl_s, hold_s, label))
    lines.append('')

    # Entry conditions
    entry_s = all_stats.get(f'{label}_entry', {})
    if entry_s:
        lines.append('### Entry Conditions')
        lines.append('| Feature | Mean | Median | P25 | P75 |')
        lines.append('|---------|------|--------|-----|-----|')
        for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15']:
            m = entry_s.get(f'{label}_entry_{col}_mean', None)
            if m is not None:
                med = entry_s.get(f'{label}_entry_{col}_median', 0)
                p25 = entry_s.get(f'{label}_entry_{col}_p25', 0)
                p75 = entry_s.get(f'{label}_entry_{col}_p75', 0)
                lines.append(f'| {col} | {m:.3f} | {med:.3f} | {p25:.3f} | {p75:.3f} |')
        lines.append('')

    # Winners vs Losers
    wl_s = all_stats.get(f'{label}_wl', {})
    wl_comp = analyses.get(f'{label}_wl_comparison')
    if wl_s:
        lines.append('### Winners vs Losers at Entry')
        lines.append('')
        win_avg = wl_s.get(f'{label}_win_avg_pnl', 0)
        loss_avg = wl_s.get(f'{label}_loss_avg_pnl', 0)
        win_held = wl_s.get(f'{label}_win_avg_held', 0)
        loss_held = wl_s.get(f'{label}_loss_avg_held', 0)
        win_peak = wl_s.get(f'{label}_win_avg_peak', 0)
        loss_peak = wl_s.get(f'{label}_loss_avg_peak', 0)
        lines.append(f'- Winner avg PnL: ${win_avg:.2f}, avg hold: {win_held:.1f} bars, avg peak: ${win_peak:.2f}')
        lines.append(f'- Loser avg PnL: ${loss_avg:.2f}, avg hold: {loss_held:.1f} bars, avg peak: ${loss_peak:.2f}')
        lines.append('')

    if wl_comp is not None and len(wl_comp) > 0:
        lines.append('| Feature | Winner Mean | Loser Mean | Delta |')
        lines.append('|---------|------------|------------|-------|')
        for _, row in wl_comp.iterrows():
            lines.append(f'| {row["feature"]} | {row["winner_mean"]:.3f} | {row["loser_mean"]:.3f} | {row["delta_mean"]:+.3f} |')
        lines.append('')

    # Direction
    dir_s = all_stats.get(f'{label}_dir', {})
    if dir_s:
        lines.append('### Direction Breakdown')
        lines.append('| Direction | N | PnL | WR | Avg PnL |')
        lines.append('|-----------|---|-----|----|---------|')
        for d in ['LONG', 'SHORT']:
            n = dir_s.get(f'{label}_{d}_n', 0)
            pnl = dir_s.get(f'{label}_{d}_pnl', 0)
            wr = dir_s.get(f'{label}_{d}_wr', 0)
            avg = dir_s.get(f'{label}_{d}_avg_pnl', 0)
            if n > 0:
                lines.append(f'| {d} | {n} | ${pnl:,.2f} | {wr:.1f}% | ${avg:.2f} |')
        lines.append('')

    # Strategy
    strat_s = all_stats.get(f'{label}_strat', {})
    strat_df = analyses.get(f'{label}_strat_df')
    if strat_df is not None and len(strat_df) > 0:
        lines.append('### Strategy Breakdown')
        lines.append('| Strategy | N | PnL | WR | Avg Hold |')
        lines.append('|----------|---|-----|----|----------|')
        for _, row in strat_df.iterrows():
            lines.append(f'| {row["strategy"]} | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | {row["wr"]:.1f}% | {row["avg_held"]:.1f} |')
        lines.append('')

    # Time of day
    tod_s = all_stats.get(f'{label}_tod', {})
    tod_df = analyses.get(f'{label}_tod_df')
    if tod_df is not None and len(tod_df) > 0:
        lines.append('### Time of Day')
        lines.append('| Hour (UTC) | N | PnL | Avg PnL | WR |')
        lines.append('|------------|---|-----|---------|-------|')
        for _, row in tod_df.sort_values('hour').iterrows():
            h = int(row['hour'])
            lines.append(f'| {h:02d}:00 | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | ${row["mean_pnl"]:,.2f} | {row["wr"]:.1f}% |')
        lines.append('')
        if tod_s:
            lines.append('**Worst 3 hours:**')
            for i in range(1, 4):
                k = f'{label}_worst_hour_{i}'
                if k in tod_s:
                    lines.append(f'  - {tod_s[k]}')
            lines.append('')

    # Day of week
    dow_df = analyses.get(f'{label}_dow_df')
    if dow_df is not None and len(dow_df) > 0:
        lines.append('### Day of Week')
        lines.append('| Day | N | PnL | Avg PnL | WR |')
        lines.append('|-----|---|-----|---------|-------|')
        for _, row in dow_df.iterrows():
            lines.append(f'| {row["dow"]} | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | ${row["mean_pnl"]:,.2f} | {row["wr"]:.1f}% |')
        lines.append('')

    # ─── EXIT 2: trend_breakeven_protect ──────────────────────────────
    lines.append('---')
    lines.append('## 2. trend_breakeven_protect')
    lines.append('')
    lines.append('**Hypothesis**: Trend rides that were profitable, then trend weakened')
    lines.append('and PnL went negative. The exit fires when was_profitable=True but')
    lines.append('pnl<=0. These are trades that should have exited earlier (at trend_protect_profit)')
    lines.append('but missed the window.')
    lines.append('')

    label = 'trend_bp'
    pnl_s = all_stats.get(f'{label}_pnl', {})
    hold_s = all_stats.get(f'{label}_hold', {})
    lines.append('### PnL + Hold Time')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.extend(fmt_pnl_section(pnl_s, hold_s, label))
    lines.append('')

    # Entry conditions
    entry_s = all_stats.get(f'{label}_entry', {})
    if entry_s:
        lines.append('### Entry Conditions')
        lines.append('| Feature | Mean | Median | P25 | P75 |')
        lines.append('|---------|------|--------|-----|-----|')
        for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15']:
            m = entry_s.get(f'{label}_entry_{col}_mean', None)
            if m is not None:
                med = entry_s.get(f'{label}_entry_{col}_median', 0)
                p25 = entry_s.get(f'{label}_entry_{col}_p25', 0)
                p75 = entry_s.get(f'{label}_entry_{col}_p75', 0)
                lines.append(f'| {col} | {m:.3f} | {med:.3f} | {p25:.3f} | {p75:.3f} |')
        lines.append('')

    # Peak PnL before it went negative
    bp_trades = analyses.get('trend_bp_trades')
    if bp_trades is not None and len(bp_trades) > 0:
        lines.append('### Peak PnL Before Loss')
        lines.append(f'- Mean peak: ${bp_trades["peak"].mean():.2f}')
        lines.append(f'- Median peak: ${bp_trades["peak"].median():.2f}')
        lines.append(f'- P75 peak: ${np.percentile(bp_trades["peak"], 75):.2f}')
        lines.append(f'- These trades were profitable (peak > 0) but gave it ALL back plus more.')
        lines.append(f'- Average loss at exit: ${bp_trades["pnl"].mean():.2f}')
        lines.append(f'- The gap (peak - exit PnL): ${(bp_trades["peak"] - bp_trades["pnl"]).mean():.2f} average giveback')
        lines.append('')

    # Direction + Strategy + ToD + DoW (same pattern)
    for sub_label, sub_name in [('dir', 'Direction'), ('strat', 'Strategy')]:
        s = all_stats.get(f'{label}_{sub_label}', {})
        df = analyses.get(f'{label}_{sub_label}_df')
        if sub_label == 'dir' and s:
            lines.append(f'### {sub_name} Breakdown')
            lines.append('| Direction | N | PnL | WR | Avg PnL |')
            lines.append('|-----------|---|-----|----|---------|')
            for d in ['LONG', 'SHORT']:
                n = s.get(f'{label}_{d}_n', 0)
                pnl = s.get(f'{label}_{d}_pnl', 0)
                wr = s.get(f'{label}_{d}_wr', 0)
                avg = s.get(f'{label}_{d}_avg_pnl', 0)
                if n > 0:
                    lines.append(f'| {d} | {n} | ${pnl:,.2f} | {wr:.1f}% | ${avg:.2f} |')
            lines.append('')
        if sub_label == 'strat' and df is not None and len(df) > 0:
            lines.append(f'### {sub_name} Breakdown')
            lines.append('| Strategy | N | PnL | WR | Avg Hold |')
            lines.append('|----------|---|-----|----|----------|')
            for _, row in df.iterrows():
                lines.append(f'| {row["strategy"]} | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | {row["wr"]:.1f}% | {row["avg_held"]:.1f} |')
            lines.append('')

    tod_df = analyses.get(f'{label}_tod_df')
    if tod_df is not None and len(tod_df) > 0:
        lines.append('### Time of Day')
        lines.append('| Hour (UTC) | N | PnL | Avg PnL | WR |')
        lines.append('|------------|---|-----|---------|-------|')
        for _, row in tod_df.sort_values('hour').iterrows():
            h = int(row['hour'])
            lines.append(f'| {h:02d}:00 | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | ${row["mean_pnl"]:,.2f} | {row["wr"]:.1f}% |')
        lines.append('')

    # Post-exit
    post_s = all_stats.get(f'{label}_post', {})
    if post_s:
        lines.append('### Post-Exit Price Movement')
        lines.append('*Did price recover after we exited?*')
        lines.append('')
        lines.append('| Lookahead | Recovery Rate | Avg Move (ticks) | Best Avg (ticks) |')
        lines.append('|-----------|---------------|------------------|------------------|')
        lines.extend(fmt_post_exit_section(post_s, label))
        lines.append('')

    # ─── EXIT 3: trend_exhausted ──────────────────────────────────────
    lines.append('---')
    lines.append('## 3. trend_exhausted')
    lines.append('')
    lines.append('**Hypothesis**: Trend rides that got hit by a 15m trend flip.')
    lines.append('The trend flipped against the trade direction past MIN_MOVE (10 pts).')
    lines.append('Low WR suggests these are entering too late into trends that are')
    lines.append('already near exhaustion.')
    lines.append('')

    label = 'trend_ex'
    pnl_s = all_stats.get(f'{label}_pnl', {})
    hold_s = all_stats.get(f'{label}_hold', {})
    lines.append('### PnL + Hold Time')
    lines.append('| Metric | Value |')
    lines.append('|--------|-------|')
    lines.extend(fmt_pnl_section(pnl_s, hold_s, label))
    lines.append('')

    # Entry conditions
    entry_s = all_stats.get(f'{label}_entry', {})
    if entry_s:
        lines.append('### Entry Conditions')
        lines.append('| Feature | Mean | Median | P25 | P75 |')
        lines.append('|---------|------|--------|-----|-----|')
        for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15']:
            m = entry_s.get(f'{label}_entry_{col}_mean', None)
            if m is not None:
                med = entry_s.get(f'{label}_entry_{col}_median', 0)
                p25 = entry_s.get(f'{label}_entry_{col}_p25', 0)
                p75 = entry_s.get(f'{label}_entry_{col}_p75', 0)
                lines.append(f'| {col} | {m:.3f} | {med:.3f} | {p25:.3f} | {p75:.3f} |')
        lines.append('')

    # Direction + Strategy + ToD
    for sub_label, sub_name in [('dir', 'Direction'), ('strat', 'Strategy')]:
        s = all_stats.get(f'{label}_{sub_label}', {})
        df = analyses.get(f'{label}_{sub_label}_df')
        if sub_label == 'dir' and s:
            lines.append(f'### {sub_name} Breakdown')
            lines.append('| Direction | N | PnL | WR | Avg PnL |')
            lines.append('|-----------|---|-----|----|---------|')
            for d in ['LONG', 'SHORT']:
                n = s.get(f'{label}_{d}_n', 0)
                pnl = s.get(f'{label}_{d}_pnl', 0)
                wr = s.get(f'{label}_{d}_wr', 0)
                avg = s.get(f'{label}_{d}_avg_pnl', 0)
                if n > 0:
                    lines.append(f'| {d} | {n} | ${pnl:,.2f} | {wr:.1f}% | ${avg:.2f} |')
            lines.append('')
        if sub_label == 'strat' and df is not None and len(df) > 0:
            lines.append(f'### {sub_name} Breakdown')
            lines.append('| Strategy | N | PnL | WR | Avg Hold |')
            lines.append('|----------|---|-----|----|----------|')
            for _, row in df.iterrows():
                lines.append(f'| {row["strategy"]} | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | {row["wr"]:.1f}% | {row["avg_held"]:.1f} |')
            lines.append('')

    tod_df = analyses.get(f'{label}_tod_df')
    if tod_df is not None and len(tod_df) > 0:
        lines.append('### Time of Day')
        lines.append('| Hour (UTC) | N | PnL | Avg PnL | WR |')
        lines.append('|------------|---|-----|---------|-------|')
        for _, row in tod_df.sort_values('hour').iterrows():
            h = int(row['hour'])
            lines.append(f'| {h:02d}:00 | {int(row["n_trades"])} | ${row["total_pnl"]:,.2f} | ${row["mean_pnl"]:,.2f} | {row["wr"]:.1f}% |')
        lines.append('')

    # Post-exit
    post_s = all_stats.get(f'{label}_post', {})
    if post_s:
        lines.append('### Post-Exit Price Movement')
        lines.append('*Did price continue the trend after we exited?*')
        lines.append('')
        lines.append('| Lookahead | Recovery Rate | Avg Move (ticks) | Best Avg (ticks) |')
        lines.append('|-----------|---------------|------------------|------------------|')
        lines.extend(fmt_post_exit_section(post_s, label))
        lines.append('')

    # ─── CROSS-EXIT COMPARISONS ───────────────────────────────────────
    lines.append('---')
    lines.append('## Cross-Exit Comparison')
    lines.append('')

    # Aggregate the losing trades across all 3 problem exits
    problem_trades = analyses['trades'][analyses['trades']['exit'].isin(PROBLEM_EXITS)]
    other_trades = analyses['trades'][~analyses['trades']['exit'].isin(PROBLEM_EXITS)]
    lines.append(f'**Problem exits**: {len(problem_trades)} trades, ${problem_trades["pnl"].sum():,.2f}')
    lines.append(f'**Other exits**: {len(other_trades)} trades, ${other_trades["pnl"].sum():,.2f}')
    lines.append(f'**Net without problem exits**: ${other_trades["pnl"].sum():,.2f} ({other_trades["pnl"].sum()/max(len(set(other_trades["day"])),1):,.2f}/day)')
    lines.append('')

    # What if we just removed these exits entirely (no trade)?
    lines.append('### Hypothetical: Remove Problem Exits')
    lines.append(f'If these {len(problem_trades)} trades never happened:')
    lines.append(f'- Remaining PnL: ${other_trades["pnl"].sum():,.2f}')
    n_days_total = len(set(analyses['trades']['day']))
    lines.append(f'- Per day: ${other_trades["pnl"].sum()/max(n_days_total,1):,.2f}')
    lines.append('')

    return '\n'.join(lines)


def main():
    args = parse_args()
    trades, traj = load_data(args)

    # Determine date range for 1m ATLAS loading
    date_min = trades['day'].min()
    date_max = trades['day'].max()

    atlas_1m = None
    if not args.no_post_exit:
        print('Loading 1m ATLAS for post-exit analysis...')
        atlas_1m = load_1m_atlas((date_min, date_max))

    all_stats = {}
    analyses = {'trades': trades, 'traj': traj}

    # ─── Analyze each problem exit ────────────────────────────────────
    exit_configs = [
        ('mean_reached', 'mean_reached', 'mean_reached'),
        ('trend_breakeven_protect', 'trend_bp', 'trend_breakeven_protect'),
        ('trend_exhausted', 'trend_ex', 'trend_exhausted'),
    ]

    csv_rows = []  # for raw data CSV

    for exit_name, label, exit_filter in exit_configs:
        print(f'\nAnalyzing: {exit_name}...')
        subset = trades[trades['exit'] == exit_filter].copy()
        if len(subset) == 0:
            print(f'  No trades with exit={exit_filter}')
            continue

        analyses[f'{label}_trades'] = subset

        wr = (subset['pnl'] > 0).mean() * 100
        print(f'  {len(subset)} trades, WR={wr:.1f}%, PnL=${subset["pnl"].sum():,.2f}')

        # PnL distribution
        pnl_s = pnl_distribution(subset['pnl'], label)
        all_stats[f'{label}_pnl'] = pnl_s

        # Hold time
        hold_s = hold_time_distribution(subset['held'], label)
        all_stats[f'{label}_hold'] = hold_s

        # Entry conditions
        entry_s, _ = entry_conditions(traj, subset['trade_id'], label)
        all_stats[f'{label}_entry'] = entry_s

        # Direction
        dir_s = direction_breakdown(subset, label)
        all_stats[f'{label}_dir'] = dir_s

        # Strategy
        strat_s, strat_df = strategy_breakdown(subset, label)
        all_stats[f'{label}_strat'] = strat_s
        analyses[f'{label}_strat_df'] = strat_df

        # Time of day
        tod_s, tod_df = time_of_day_analysis(subset, label)
        all_stats[f'{label}_tod'] = tod_s
        analyses[f'{label}_tod_df'] = tod_df

        # Day of week
        dow_s, dow_df = day_of_week_analysis(subset, label)
        all_stats[f'{label}_dow'] = dow_s
        analyses[f'{label}_dow_df'] = dow_df

        # Exit-specific analyses
        if exit_name == 'mean_reached':
            wl_s, wl_comp = winners_vs_losers(subset, traj, label)
            all_stats[f'{label}_wl'] = wl_s
            analyses[f'{label}_wl_comparison'] = wl_comp

        if exit_name in ('trend_breakeven_protect', 'trend_exhausted'):
            post_s, post_df = post_exit_analysis(subset, traj, atlas_1m, label)
            all_stats[f'{label}_post'] = post_s
            analyses[f'{label}_post_df'] = post_df

        # Collect per-trade rows for CSV
        for _, t in subset.iterrows():
            row = t.to_dict()
            row['exit_group'] = exit_name
            # Add entry features from trajectory bar=1
            if traj is not None:
                entry_bar = traj[(traj['trade_id'] == t['trade_id']) & (traj['bar'] == 1)]
                if len(entry_bar) > 0:
                    eb = entry_bar.iloc[0]
                    for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15', 'timestamp']:
                        if col in eb.index:
                            row[f'entry_{col}'] = eb[col]
                # Add exit bar features
                exit_bar = traj[traj['trade_id'] == t['trade_id']]
                if len(exit_bar) > 0:
                    ex_bar = exit_bar.iloc[-1]
                    for col in ['z_se', 'vr', 'lam', 'dmi_1m', 'trend_15', 'price']:
                        if col in ex_bar.index:
                            row[f'exit_{col}'] = ex_bar[col]
            csv_rows.append(row)

    # ─── Generate report ──────────────────────────────────────────────
    print('\nGenerating report...')
    report = generate_report(all_stats, analyses, args)

    os.makedirs('reports/findings', exist_ok=True)
    report_path = f'reports/findings/nightmare_eda_{args.label}.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f'Report saved: {report_path}')

    # Save raw CSV
    csv_path = f'reports/findings/nightmare_eda_data_{args.label}.csv'
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_df.to_csv(csv_path, index=False)
        print(f'Raw data saved: {csv_path} ({len(csv_df)} rows)')

    print('\nDone.')


if __name__ == '__main__':
    main()
