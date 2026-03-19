"""
Peak Exit Research -- stateful trajectory analysis for peak trades.

For each peak trade in IS/OOS, replays the 15s bars from entry to exit+10
and captures the statistical state at each bar. This tells us:
  1. When does peak detection fire AGAINST the open position?
  2. How does P_center/F_momentum evolve during the trade?
  3. What separates winners from losers in stat-space?
  4. Where is the optimal exit relative to the "enter against me" signal?

Usage:
    python tools/peak_exit_research.py [--oos] [--data DATA/ATLAS]

Output:
    reports/findings/peak_exit_trajectories.csv  (per-bar-per-trade)
    reports/findings/peak_exit_summary.txt       (aggregate analysis)
"""

import os
import sys
import json
import argparse
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Project imports ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.trading_config import TradingConfig


# ── Peak detection thresholds ────────────────────────────────────────────
# regression_center moves ~1.5/bar on 15s. Sigma is ~4 ticks.
# Use absolute delta relative to sigma, not % of center level.
PEAK_PC_DELTA_TICKS = 2.0    # regression_center moved > 2 ticks (0.50 points)
PEAK_FM_FLIP = True          # F_momentum flipped sign (direction change)
PEAK_FM_MAGNITUDE = 50.0     # |F_momentum| threshold for meaningful signal


def detect_peak_direction(prev_pc, curr_pc, prev_fm, curr_fm, coherence,
                          sigma=4.0):
    """Return 'long', 'short', or None based on regression center movement.

    Uses absolute delta in the regression center (tick-scale) and
    F_momentum direction/magnitude rather than percentage thresholds.
    """
    pc_delta = curr_pc - prev_pc  # absolute change in regression center
    pc_delta_ticks = pc_delta / 0.25  # convert to MNQ ticks

    # F_momentum: check direction and magnitude
    fm_flipped = (prev_fm * curr_fm < 0)  # sign change
    fm_strong_against_long = curr_fm < -PEAK_FM_MAGNITUDE
    fm_strong_against_short = curr_fm > PEAK_FM_MAGNITUDE

    # Regression center rising = band moving up = bearish for shorts
    # Regression center falling = band moving down = bearish for longs
    pc_up = pc_delta_ticks > PEAK_PC_DELTA_TICKS
    pc_down = pc_delta_ticks < -PEAK_PC_DELTA_TICKS

    # Strong signal: center moved + momentum confirms
    if pc_up and fm_strong_against_short:
        return 'long'    # center up + positive momentum = bullish (exit shorts)
    elif pc_down and fm_strong_against_long:
        return 'short'   # center down + negative momentum = bearish (exit longs)

    # Medium signal: center moved significantly (>3 ticks)
    if pc_delta_ticks > 3.0:
        return 'long'
    elif pc_delta_ticks < -3.0:
        return 'short'

    # Weak signal: momentum flipped with magnitude
    if fm_flipped and abs(curr_fm) > PEAK_FM_MAGNITUDE * 2:
        if curr_fm > 0:
            return 'long'
        else:
            return 'short'

    return None


def load_15s_states(data_root):
    """Load 15s parquets and compute MarketState for each bar."""
    tf_dir = os.path.join(data_root, '15s')
    if not os.path.isdir(tf_dir):
        raise FileNotFoundError(f"No 15s directory in {data_root}")

    files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    if not files:
        raise FileNotFoundError(f"No parquet files in {tf_dir}")

    engine = StatisticalFieldEngine()

    all_bars = []
    all_states = []

    for fname in tqdm(files, desc="Loading 15s data"):
        fpath = os.path.join(tf_dir, fname)
        df = pd.read_parquet(fpath)

        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())

        # Compute states -- batch returns list of dicts {'bar_idx', 'state': MarketState, ...}
        raw_states = engine.batch_compute_states(df)
        # Unwrap to get MarketState objects
        for rs in raw_states:
            if isinstance(rs, dict) and 'state' in rs:
                all_states.append(rs['state'])
            else:
                all_states.append(rs)
        all_bars.append(df)

    df_all = pd.concat(all_bars, ignore_index=True)
    print(f"  Loaded {len(df_all)} bars, {len(all_states)} states")
    return df_all, all_states


def build_timestamp_index(df, states):
    """Build timestamp -> (bar_index, state) mapping."""
    ts_map = {}
    timestamps = df['timestamp'].values
    for i, ts in enumerate(timestamps):
        if i < len(states):
            ts_map[int(ts)] = (i, states[i])
    return ts_map, timestamps


def analyze_trade_trajectory(trade, ts_map, timestamps, states, lookahead=10):
    """For one trade, extract per-bar statistical trajectory."""
    entry_ts = int(trade['entry_time'])
    exit_ts = int(trade.get('exit_time', entry_ts + 300))
    direction = trade['direction'].upper()
    hold_bars = int(trade.get('hold_bars', 1))

    # Find entry bar index
    entry_idx = np.searchsorted(timestamps, entry_ts)
    if entry_idx >= len(timestamps):
        return []

    exit_idx = min(entry_idx + hold_bars, len(timestamps) - 1)
    end_idx = min(exit_idx + lookahead, len(timestamps) - 1)
    start_idx = max(entry_idx - 2, 0)  # 2 bars before entry for context

    if entry_idx >= len(states) or entry_idx < 1:
        return []

    entry_state = states[entry_idx] if entry_idx < len(states) else None
    if entry_state is None:
        return []

    entry_pc = getattr(entry_state, 'regression_center', 0.0) or 0.0
    entry_fm = getattr(entry_state, 'F_momentum', 0.0) or 0.0
    entry_price = trade['entry_price']

    rows = []
    peak_pc = entry_pc
    prev_pc = getattr(states[entry_idx - 1], 'regression_center', 0.0) if entry_idx > 0 else entry_pc
    prev_fm = getattr(states[entry_idx - 1], 'F_momentum', 0.0) if entry_idx > 0 else entry_fm

    for bar_i in range(start_idx, end_idx + 1):
        if bar_i >= len(states):
            break

        st = states[bar_i]
        curr_pc = getattr(st, 'regression_center', 0.0) or 0.0
        curr_fm = getattr(st, 'F_momentum', 0.0) or 0.0
        coherence = getattr(st, 'coherence', 0.0) or 0.0
        z_score = getattr(st, 'z_score', 0.0) or 0.0
        bar_offset = bar_i - entry_idx
        in_trade = (bar_i >= entry_idx and bar_i <= exit_idx)

        # Track statistical peak (best P_center since entry)
        if bar_i >= entry_idx:
            if direction == 'LONG':
                if curr_pc > peak_pc:
                    peak_pc = curr_pc
            else:
                if curr_pc < peak_pc:
                    peak_pc = curr_pc

        # P_center giveback from peak
        if abs(peak_pc) > 1e-8:
            if direction == 'LONG':
                pc_giveback = (peak_pc - curr_pc) / abs(peak_pc)
            else:
                pc_giveback = (curr_pc - peak_pc) / abs(peak_pc)
        else:
            pc_giveback = 0.0

        # Peak detection: would it fire against my position?
        if bar_i > 0 and bar_i < len(states):
            _prev_st = states[bar_i - 1]
            _prev_pc = getattr(_prev_st, 'regression_center', 0.0) or 0.0
            _prev_fm = getattr(_prev_st, 'F_momentum', 0.0) or 0.0
            peak_dir = detect_peak_direction(_prev_pc, curr_pc, _prev_fm, curr_fm, coherence)
        else:
            peak_dir = None

        peak_fires_against = False
        peak_fires_with = False
        if peak_dir is not None:
            if direction == 'LONG' and peak_dir == 'short':
                peak_fires_against = True
            elif direction == 'SHORT' and peak_dir == 'long':
                peak_fires_against = True
            elif direction == 'LONG' and peak_dir == 'long':
                peak_fires_with = True
            elif direction == 'SHORT' and peak_dir == 'short':
                peak_fires_with = True

        # Unrealized PnL (ticks)
        if direction == 'LONG':
            unreal_ticks = (getattr(st, 'price', entry_price) - entry_price) / 0.25
        else:
            unreal_ticks = (entry_price - getattr(st, 'price', entry_price)) / 0.25

        rows.append({
            'trade_idx': 0,  # filled by caller
            'bar_offset': bar_offset,
            'in_trade': in_trade,
            'direction': direction,
            'regression_center': curr_pc,
            'F_momentum': curr_fm,
            'coherence': coherence,
            'z_score': z_score,
            'pc_delta_from_entry': curr_pc - entry_pc,
            'fm_delta_from_entry': curr_fm - entry_fm,
            'peak_pc': peak_pc,
            'pc_giveback': pc_giveback,
            'peak_fires_against': peak_fires_against,
            'peak_fires_with': peak_fires_with,
            'peak_dir': peak_dir or '',
            'unreal_ticks': unreal_ticks if in_trade else np.nan,
            'actual_pnl': trade['actual_pnl'],
            'exit_reason': trade['exit_reason'],
            'hold_bars': hold_bars,
            'result': trade['result'],
            'template_id': trade['template_id'],
        })

    return rows


def summarize_results(traj_df, output_path):
    """Generate summary analysis from trajectory data."""
    lines = []
    lines.append("=" * 80)
    lines.append("PEAK EXIT RESEARCH -- Stateful Trajectory Analysis")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("=" * 80)

    # Only in-trade bars
    in_trade = traj_df[traj_df['in_trade'] == True]
    peak_trades = in_trade[in_trade['template_id'] == -100]
    n_trades = peak_trades['trade_idx'].nunique()

    lines.append(f"\n  Total peak trades analyzed: {n_trades}")
    lines.append(f"  Total in-trade bars: {len(peak_trades)}")

    # When does peak fire against position?
    fires_against = peak_trades[peak_trades['peak_fires_against'] == True]
    lines.append(f"\n  Bars where peak fires AGAINST position: {len(fires_against)} "
                 f"({len(fires_against)/max(len(peak_trades),1)*100:.1f}% of bars)")

    if len(fires_against) > 0:
        lines.append(f"  Avg bar_offset when fires against: {fires_against['bar_offset'].mean():.1f}")
        lines.append(f"  Median bar_offset: {fires_against['bar_offset'].median():.0f}")

        # Break down by winner/loser
        fa_win = fires_against[fires_against['result'] == 'WIN']
        fa_loss = fires_against[fires_against['result'] == 'LOSS']
        lines.append(f"\n  Fires against in WINNING trades: {len(fa_win)} bars "
                     f"(avg offset: {fa_win['bar_offset'].mean():.1f})")
        lines.append(f"  Fires against in LOSING trades:  {len(fa_loss)} bars "
                     f"(avg offset: {fa_loss['bar_offset'].mean():.1f})")

    # P_center giveback at exit bar
    exit_bars = peak_trades.groupby('trade_idx').last()
    lines.append(f"\n  P_CENTER GIVEBACK AT EXIT:")
    lines.append(f"    Winners: avg pc_giveback = {exit_bars[exit_bars['result']=='WIN']['pc_giveback'].mean():.4f}")
    lines.append(f"    Losers:  avg pc_giveback = {exit_bars[exit_bars['result']=='LOSS']['pc_giveback'].mean():.4f}")

    # By exit reason
    lines.append(f"\n  P_CENTER GIVEBACK BY EXIT REASON:")
    for reason, grp in exit_bars.groupby('exit_reason'):
        avg_gb = grp['pc_giveback'].mean()
        n = len(grp)
        avg_pnl = grp['actual_pnl'].mean()
        lines.append(f"    {reason:<24} n={n:>5}  pc_giveback={avg_gb:>+.4f}  avg_pnl=${avg_pnl:>+.2f}")

    # "Would enter against me" -- first bar where it fires, relative to MFE bar
    lines.append(f"\n  FIRST 'ENTER AGAINST ME' SIGNAL PER TRADE:")
    first_against = fires_against.groupby('trade_idx')['bar_offset'].min()
    if len(first_against) > 0:
        lines.append(f"    Trades with at least one signal: {len(first_against)} / {n_trades} "
                     f"({len(first_against)/max(n_trades,1)*100:.1f}%)")
        lines.append(f"    Avg bar_offset of first signal: {first_against.mean():.1f}")
        lines.append(f"    Median: {first_against.median():.0f}")
        lines.append(f"    P25: {first_against.quantile(0.25):.0f}  P75: {first_against.quantile(0.75):.0f}")

        # Compare to actual exit bar
        actual_hold = exit_bars['hold_bars']
        merged = pd.DataFrame({
            'first_against': first_against,
            'hold_bars': actual_hold,
            'actual_pnl': exit_bars['actual_pnl'],
            'result': exit_bars['result'],
        }).dropna()
        merged['delta'] = merged['hold_bars'] - merged['first_against']
        lines.append(f"\n    Bars between 'enter against' and actual exit:")
        lines.append(f"      avg delta: {merged['delta'].mean():.1f} bars (positive = exited AFTER signal)")
        lines.append(f"      Exited BEFORE signal: {(merged['delta'] < 0).sum()} trades")
        lines.append(f"      Exited ON signal bar: {(merged['delta'] == 0).sum()} trades")
        lines.append(f"      Exited AFTER signal: {(merged['delta'] > 0).sum()} trades")

        # PnL comparison
        before = merged[merged['delta'] < 0]
        on = merged[merged['delta'] == 0]
        after = merged[merged['delta'] > 0]
        if len(before) > 0:
            lines.append(f"\n    Exited BEFORE signal: avg_pnl=${before['actual_pnl'].mean():.2f} "
                         f"({(before['result']=='WIN').mean()*100:.0f}% WR)")
        if len(on) > 0:
            lines.append(f"    Exited ON signal:     avg_pnl=${on['actual_pnl'].mean():.2f} "
                         f"({(on['result']=='WIN').mean()*100:.0f}% WR)")
        if len(after) > 0:
            lines.append(f"    Exited AFTER signal:  avg_pnl=${after['actual_pnl'].mean():.2f} "
                         f"({(after['result']=='WIN').mean()*100:.0f}% WR)")
    else:
        lines.append(f"    No 'enter against' signals found in any trade")

    # Trades WITHOUT any "enter against" signal
    trades_with_signal = set(first_against.index) if len(first_against) > 0 else set()
    all_trade_ids = set(exit_bars.index)
    no_signal = all_trade_ids - trades_with_signal
    if no_signal:
        no_sig_trades = exit_bars.loc[list(no_signal)]
        lines.append(f"\n  TRADES WITHOUT 'ENTER AGAINST' SIGNAL: {len(no_signal)}")
        lines.append(f"    avg_pnl: ${no_sig_trades['actual_pnl'].mean():.2f}")
        lines.append(f"    WR: {(no_sig_trades['result']=='WIN').mean()*100:.1f}%")
        lines.append(f"    avg hold: {no_sig_trades['hold_bars'].mean():.1f} bars")

    # F_momentum at exit
    lines.append(f"\n  F_MOMENTUM AT EXIT:")
    lines.append(f"    Winners: avg F_mom = {exit_bars[exit_bars['result']=='WIN']['F_momentum'].mean():.4f}")
    lines.append(f"    Losers:  avg F_mom = {exit_bars[exit_bars['result']=='LOSS']['F_momentum'].mean():.4f}")

    # Bar offset distribution for peak_fires_against
    if len(fires_against) > 0:
        lines.append(f"\n  BAR OFFSET WHEN PEAK FIRES AGAINST (distribution):")
        for offset in range(0, 12):
            n = (fires_against['bar_offset'] == offset).sum()
            if n > 0:
                lines.append(f"    bar +{offset}: {n} signals")

    report = '\n'.join(lines) + '\n'

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"\n  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Peak exit trajectory research")
    parser.add_argument('--data', default=os.path.join('DATA', 'ATLAS'),
                        help="ATLAS data root")
    parser.add_argument('--oos', action='store_true',
                        help="Use OOS trade log instead of IS")
    parser.add_argument('--max-trades', type=int, default=5000,
                        help="Max trades to analyze (for speed)")
    args = parser.parse_args()

    # Load trade log
    if args.oos:
        trade_path = 'checkpoints/oos_trade_log.csv'
        data_root = os.path.join('DATA', 'ATLAS_OOS')
    else:
        # IS trade log -- check multiple locations
        for p in ['checkpoints/oracle_trade_log.csv', 'reports/is/oracle_trade_log.csv']:
            if os.path.exists(p):
                df_check = pd.read_csv(p)
                if len(df_check) > 10:
                    trade_path = p
                    break
        else:
            trade_path = 'checkpoints/oos_trade_log.csv'
            print("  WARNING: IS trade log empty/missing, falling back to OOS")
        data_root = args.data

    print(f"  Trade log: {trade_path}")
    print(f"  Data root: {data_root}")

    trades_df = pd.read_csv(trade_path)
    print(f"  Total trades: {len(trades_df)}")

    # Filter to peak trades only
    peak_trades = trades_df[trades_df['template_id'] == -100].copy()
    print(f"  Peak trades: {len(peak_trades)}")

    if len(peak_trades) > args.max_trades:
        peak_trades = peak_trades.sample(n=args.max_trades, random_state=42)
        print(f"  Sampled {args.max_trades} for speed")

    # Load 15s states
    print(f"\n  Computing 15s states...")
    df_bars, states = load_15s_states(data_root)
    timestamps = df_bars['timestamp'].values.astype(np.int64)

    # Analyze each trade
    all_rows = []
    for trade_i, (_, trade) in enumerate(tqdm(peak_trades.iterrows(),
                                               total=len(peak_trades),
                                               desc="Analyzing trades")):
        rows = analyze_trade_trajectory(trade, None, timestamps, states, lookahead=10)
        for r in rows:
            r['trade_idx'] = trade_i
        all_rows.extend(rows)

    traj_df = pd.DataFrame(all_rows)
    print(f"\n  Trajectory rows: {len(traj_df)}")

    # Save trajectories
    traj_path = os.path.join('reports', 'findings', 'peak_exit_trajectories.csv')
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    traj_df.to_csv(traj_path, index=False)
    print(f"  Saved: {traj_path}")

    # Summary
    summary_path = os.path.join('reports', 'findings', 'peak_exit_summary.txt')
    summarize_results(traj_df, summary_path)


if __name__ == '__main__':
    main()
