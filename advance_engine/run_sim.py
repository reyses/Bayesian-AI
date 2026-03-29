"""
Standalone simulation: TrajectoryAdvanceEngine on ATLAS data.

Runs the trajectory navigation system on IS data and reports results.
Uses the isolated advance_engine workspace but loads models from main checkpoints.

Usage:
  python -m advance_engine.run_sim
  python -m advance_engine.run_sim --start 2026-03-01 --end 2026-03-27
"""
import argparse
import gc
import glob
import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
ATLAS_ROOT = 'DATA/ATLAS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default=None, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date YYYY-MM-DD')
    parser.add_argument('--checkpoint-base', default='checkpoints',
                        help='Base checkpoint directory')
    args = parser.parse_args()

    from core.statistical_field_engine import StatisticalFieldEngine
    from advance_engine.core.trajectory_advance_engine import TrajectoryAdvanceEngine

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load 1m data
    print("Loading 1m data...")
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    if args.start:
        ts = pd.Timestamp(args.start).timestamp()
        df = df[df['timestamp'] >= ts].reset_index(drop=True)
    if args.end:
        ts = pd.Timestamp(args.end).timestamp()
        df = df[df['timestamp'] < ts].reset_index(drop=True)
    print(f"  Bars: {len(df):,}")

    # Compute SFE states
    print("Computing SFE states...")
    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    print(f"  States: {len(states)}")

    # Initialize engine
    print("\nInitializing TrajectoryAdvanceEngine...")
    engine = TrajectoryAdvanceEngine(
        checkpoint_base=args.checkpoint_base,
        device=device,
    )

    # Run simulation
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    timestamps = df['timestamp'].values

    trades = []
    daily_pnl = {}

    print(f"\nRunning simulation...")
    for i in tqdm(range(len(df)), desc="Sim"):
        state = states[i]['state'] if isinstance(states[i], dict) else states[i]

        result = engine.process_bar(
            '1m', prices[i], highs[i], lows[i], volumes[i], timestamps[i], state)

        if result.action == 'ENTER':
            pass  # engine tracks internally

        if result.action == 'EXIT':
            date = pd.to_datetime(timestamps[i], unit='s').strftime('%Y-%m-%d')
            trades.append({
                'bar': i,
                'date': date,
                'direction': result.direction,
                'pnl_ticks': result.pnl / TICK if result.pnl != 0 else 0,
                'pnl_dollars': result.pnl * 2,  # MNQ $2/point
                'reason': result.reason,
                'confidence': result.confidence,
                'sight': result.sight,
            })
            if date not in daily_pnl:
                daily_pnl[date] = 0
            daily_pnl[date] += result.pnl * 2

    # Results
    n = len(trades)
    if n == 0:
        print("\nNo trades generated.")
        return

    total_pnl = sum(t['pnl_dollars'] for t in trades)
    wins = len([t for t in trades if t['pnl_dollars'] > 0])
    trading_days = len(daily_pnl)
    pos_days = len([d for d in daily_pnl.values() if d > 0])

    print(f"\n{'=' * 60}")
    print(f"TRAJECTORY NAVIGATION SIMULATION")
    print(f"{'=' * 60}")
    print(f"  Trades: {n}")
    print(f"  WR: {wins / n * 100:.1f}%")
    print(f"  PnL: ${total_pnl:,.2f}")
    print(f"  $/day: ${total_pnl / trading_days:,.2f}")
    print(f"  Trading days: {trading_days} ({pos_days} green, {trading_days - pos_days} red)")
    print(f"  Trades/day: {n / trading_days:.1f}")

    if trades:
        pnls = [t['pnl_dollars'] for t in trades]
        print(f"  Avg PnL/trade: ${np.mean(pnls):,.2f}")
        print(f"  Best trade: ${max(pnls):,.2f}")
        print(f"  Worst trade: ${min(pnls):,.2f}")

    # Exit breakdown
    exits = {}
    for t in trades:
        r = t['reason']
        if r not in exits:
            exits[r] = {'n': 0, 'pnl': 0}
        exits[r]['n'] += 1
        exits[r]['pnl'] += t['pnl_dollars']
    print(f"\n  EXIT BREAKDOWN:")
    for r, v in sorted(exits.items(), key=lambda x: x[1]['pnl']):
        print(f"    {r:<30} {v['n']:>5} trades  ${v['pnl']:>10,.2f}")

    # Monthly breakdown
    df_trades = pd.DataFrame(trades)
    df_trades['month'] = df_trades['date'].str[:7]
    print(f"\n  MONTHLY:")
    for month, grp in df_trades.groupby('month'):
        m_pnl = grp['pnl_dollars'].sum()
        m_n = len(grp)
        m_days = grp['date'].nunique()
        print(f"    {month}: ${m_pnl:>10,.2f} ({m_n} trades, {m_days} days, "
              f"${m_pnl / m_days:>+.0f}/day)")

    # Save
    out_dir = 'advance_engine/results'
    os.makedirs(out_dir, exist_ok=True)
    df_trades.to_csv(os.path.join(out_dir, 'trades.csv'), index=False)
    print(f"\n  Saved: {out_dir}/trades.csv")


if __name__ == '__main__':
    main()
