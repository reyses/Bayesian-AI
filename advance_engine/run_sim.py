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
    import time as _time

    t_start = _time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load 1m data (include warmup period before start)
    WARMUP_BARS = 300  # ~5 hours of 1m bars for feature buffers + SFE
    print("Loading 1m data...")
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Filter with warmup: load bars from (start - warmup) to end
    trade_start_ts = 0
    if args.start:
        trade_start_ts = pd.Timestamp(args.start).timestamp()
        warmup_ts = trade_start_ts - WARMUP_BARS * 60  # 300 minutes before
        df = df[df['timestamp'] >= warmup_ts].reset_index(drop=True)
    if args.end:
        ts = pd.Timestamp(args.end).timestamp()
        df = df[df['timestamp'] < ts].reset_index(drop=True)

    # Find where trading window starts (after warmup)
    if trade_start_ts > 0:
        trade_start_idx = int((df['timestamp'] >= trade_start_ts).argmax())
    else:
        trade_start_idx = WARMUP_BARS
    print(f"  Bars: {len(df):,} (warmup={trade_start_idx}, trading={len(df)-trade_start_idx})")

    # Compute 1m SFE states
    print("Computing 1m SFE states...")
    sfe = StatisticalFieldEngine()
    states_1m = sfe.batch_compute_states(df)
    print(f"  1m states: {len(states_1m)}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Load 1h data + compute states
    print("Loading 1h data...")
    files_1h = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1h', '*.parquet')))
    df_1h = pd.concat([pd.read_parquet(f) for f in files_1h], ignore_index=True)
    df_1h = df_1h.sort_values('timestamp').reset_index(drop=True)
    if args.start:
        # Include warmup bars before start
        warmup_ts = pd.Timestamp(args.start).timestamp() - 3600 * 24
        df_1h = df_1h[df_1h['timestamp'] >= warmup_ts].reset_index(drop=True)
    if args.end:
        ts = pd.Timestamp(args.end).timestamp()
        df_1h = df_1h[df_1h['timestamp'] < ts].reset_index(drop=True)
    print(f"  1h bars: {len(df_1h):,}")

    print("Computing 1h SFE states...")
    sfe_1h = StatisticalFieldEngine()
    states_1h = sfe_1h.batch_compute_states(df_1h)
    del sfe_1h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    ts_1h = df_1h['timestamp'].values
    close_1h = df_1h['close'].values
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    vol_1h = df_1h['volume'].values if 'volume' in df_1h.columns else np.zeros(len(df_1h))

    # Load 1s data + extract features (no SFE — use raw extraction like training)
    print("Loading 1s data (per-month, lightweight features)...")
    from training.train_direction import extract_features_13d as _extract_13d
    from training.train_trade_cnn import extract_4_features_from_raw

    files_1s = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1s', '*.parquet')))
    # Build alignment: for each 1m bar, find the last completed 1s bar
    # Load 1s timestamps for alignment (just timestamps, not full data)
    all_1s_ts = []
    for f in files_1s:
        _df = pd.read_parquet(f, columns=['timestamp'])
        all_1s_ts.append(_df['timestamp'].values)
    ts_1s_all = np.concatenate(all_1s_ts)
    ts_1s_all.sort()
    print(f"  1s bars: {len(ts_1s_all):,}")

    # For the sim we need 1s SFE states — too expensive for 15M bars.
    # Instead: feed the LAST 1s bar before each 1m bar using the 1s trajectory model.
    # The 1s FeatureBuffer in TrajectoryAdvanceEngine builds features incrementally
    # from raw OHLCV (same as live). We just need to feed it the 1s bars.

    # Pre-load 1s data per month into a dict for streaming
    _1s_data = {}
    for f in files_1s:
        month = os.path.basename(f).replace('.parquet', '')
        _1s_data[month] = f  # lazy — load per month during sim
    print(f"  1s months: {len(_1s_data)}")

    # Build 1h alignment: for each 1m bar, find last completed 1h bar
    close_times_1h = ts_1h + 3600
    timestamps_1m = df['timestamp'].values.astype(np.int64)
    align_1h = np.searchsorted(close_times_1h, timestamps_1m, side='left') - 1
    align_1h[align_1h < 0] = -1

    # Build 1s alignment: for each 1m bar, find all 1s bars in the preceding 60s
    # We'll feed them in batch to the engine's 1s buffer
    close_times_1s = ts_1s_all + 1  # 1s bar closes 1 second after open
    align_1s_end = np.searchsorted(close_times_1s, timestamps_1m, side='left')  # first 1s bar that closes >= 1m open
    # We want the 1s bars between previous 1m bar and this 1m bar

    # Initialize engine
    print("\nInitializing TrajectoryAdvanceEngine...")
    engine = TrajectoryAdvanceEngine(
        checkpoint_base=args.checkpoint_base,
        device=device,
    )

    # Run simulation with multi-TF ticker
    prices = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    volumes = df['volume'].values if 'volume' in df.columns else np.zeros(len(df))
    timestamps = timestamps_1m

    trades = []
    daily_pnl = {}
    last_1h_idx = -1
    last_1s_pos = 0  # position in ts_1s_all for streaming

    # 1s streaming: load one day at a time to avoid memory issues
    _1s_current_date = None
    _1s_ts = None
    _1s_close = None
    _1s_high = None
    _1s_low = None
    _1s_vol = None
    _1s_idx = 0
    _1s_month_df = None
    _1s_current_month = None

    class _MockState:
        dmi_plus = 0.0
        dmi_minus = 0.0
        velocity = 0.0
    _mock_state = _MockState()

    def _load_1s_for_date(ts):
        """Load 1s bars for the date containing timestamp ts. Lazy month load."""
        nonlocal _1s_current_date, _1s_ts, _1s_close, _1s_high, _1s_low, _1s_vol, _1s_idx
        nonlocal _1s_month_df, _1s_current_month

        dt = pd.to_datetime(ts, unit='s')
        date_str = dt.strftime('%Y-%m-%d')
        if date_str == _1s_current_date:
            return

        # Load month if needed
        month = dt.strftime('%Y_%m')
        if month != _1s_current_month:
            del _1s_month_df
            gc.collect()
            path = os.path.join(ATLAS_ROOT, '1s', f'{month}.parquet')
            if os.path.exists(path):
                _1s_month_df = pd.read_parquet(path)
                _1s_month_df = _1s_month_df.sort_values('timestamp').reset_index(drop=True)
                _1s_current_month = month
            else:
                _1s_month_df = None
                _1s_current_month = month

        if _1s_month_df is None:
            _1s_ts = None
            return

        # Filter to this date only
        day_start = pd.Timestamp(date_str).timestamp()
        day_end = day_start + 86400
        mask = (_1s_month_df['timestamp'] >= day_start) & (_1s_month_df['timestamp'] < day_end)
        day_df = _1s_month_df[mask]

        if len(day_df) == 0:
            _1s_ts = None
        else:
            _1s_ts = day_df['timestamp'].values
            _1s_close = day_df['close'].values
            _1s_high = day_df['high'].values
            _1s_low = day_df['low'].values
            _1s_vol = day_df['volume'].values if 'volume' in day_df.columns else np.zeros(len(day_df))
            _1s_idx = 0

        _1s_current_date = date_str

    def _feed_1s_bars(engine, up_to_ts):
        """Feed all 1s bars up to (but not including) up_to_ts."""
        nonlocal _1s_idx
        if _1s_ts is None:
            return
        while _1s_idx < len(_1s_ts) and _1s_ts[_1s_idx] + 1 < up_to_ts:
            engine.process_bar('1s', float(_1s_close[_1s_idx]), float(_1s_high[_1s_idx]),
                               float(_1s_low[_1s_idx]), float(_1s_vol[_1s_idx]),
                               float(_1s_ts[_1s_idx]), _mock_state)
            _1s_idx += 1

    print(f"\nRunning simulation (1s + 1m + 1h ticker)...")
    for i in tqdm(range(len(df)), desc="Sim"):
        state_1m = states_1m[i]['state'] if isinstance(states_1m[i], dict) else states_1m[i]
        is_warmup = (i < trade_start_idx)

        # Feed 1h bar if a new one completed since last tick
        h_idx = align_1h[i]
        if h_idx >= 0 and h_idx != last_1h_idx and h_idx < len(states_1h):
            state_1h = states_1h[h_idx]['state'] if isinstance(states_1h[h_idx], dict) else states_1h[h_idx]
            engine.process_bar('1h', close_1h[h_idx], high_1h[h_idx], low_1h[h_idx],
                               vol_1h[h_idx], ts_1h[h_idx], state_1h)
            last_1h_idx = h_idx

        # Feed 1s bars that completed since the previous 1m bar
        _load_1s_for_date(timestamps[i])
        _feed_1s_bars(engine, timestamps[i])

        # Feed 1m bar (primary — decisions happen here)
        result = engine.process_bar(
            '1m', prices[i], highs[i], lows[i], volumes[i], timestamps[i], state_1m)

        if result.action == 'EXIT' and not is_warmup:
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
    neg_days = len([d for d in daily_pnl.values() if d < 0])
    flat_days = len([d for d in daily_pnl.values() if d == 0])

    daily_vals = sorted(daily_pnl.items())
    daily_pnls = np.array([v for _, v in daily_vals])
    daily_dates = [d for d, _ in daily_vals]

    # Consecutive losing days
    max_losing_streak = 0
    current_streak = 0
    for pnl in daily_pnls:
        if pnl < 0:
            current_streak += 1
            max_losing_streak = max(max_losing_streak, current_streak)
        else:
            current_streak = 0

    # Max drawdown
    cumulative = np.cumsum(daily_pnls)
    peak = np.maximum.accumulate(cumulative)
    drawdown = peak - cumulative
    max_dd = drawdown.max()
    max_dd_idx = drawdown.argmax()

    print(f"\n{'=' * 60}")
    print(f"TRAJECTORY NAVIGATION — DAILY VALIDATION")
    print(f"{'=' * 60}")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Trading days: {trading_days}")
    print(f"  $/day avg: ${total_pnl / trading_days:,.2f}")
    print(f"")
    print(f"  === DAILY SURVIVAL METRICS ===")
    print(f"  Green days: {pos_days}/{trading_days} ({pos_days/trading_days*100:.0f}%)")
    print(f"  Red days: {neg_days}/{trading_days} ({neg_days/trading_days*100:.0f}%)")
    print(f"  Flat days: {flat_days}")
    print(f"  Best day: ${daily_pnls.max():+,.2f} ({daily_dates[daily_pnls.argmax()]})")
    print(f"  Worst day: ${daily_pnls.min():+,.2f} ({daily_dates[daily_pnls.argmin()]})")
    print(f"  Max losing streak: {max_losing_streak} consecutive days")
    print(f"  Max drawdown: ${max_dd:,.2f} (at {daily_dates[max_dd_idx]})")
    print(f"  Daily PnL std: ${daily_pnls.std():,.2f}")
    print(f"  Sharpe (daily): {daily_pnls.mean() / (daily_pnls.std() + 1e-8):.2f}")
    print(f"")
    print(f"  === TRADE METRICS ===")
    print(f"  Trades: {n}")
    print(f"  Trade WR: {wins / n * 100:.1f}%")
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

    # Daily PnL percentiles
    print(f"\n  DAILY PnL DISTRIBUTION:")
    for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(daily_pnls, pct)
        print(f"    P{pct:>2}: ${val:>+10,.2f}")

    # Monthly breakdown
    df_trades = pd.DataFrame(trades)
    df_trades['month'] = df_trades['date'].str[:7]
    print(f"\n  MONTHLY:")
    for month, grp in df_trades.groupby('month'):
        m_pnl = grp['pnl_dollars'].sum()
        m_n = len(grp)
        m_days = grp['date'].nunique()
        m_green = sum(1 for _, g in grp.groupby('date') if g['pnl_dollars'].sum() > 0)
        print(f"    {month}: ${m_pnl:>10,.2f} ({m_n} trades, {m_days} days, "
              f"{m_green} green, ${m_pnl / m_days:>+.0f}/day)")

    # Per-day detail (first 30 + last 10)
    print(f"\n  DAILY DETAIL (worst 20 days):")
    sorted_days = sorted(daily_pnl.items(), key=lambda x: x[1])
    for date, pnl in sorted_days[:20]:
        day_trades = [t for t in trades if t['date'] == date]
        n_t = len(day_trades)
        n_w = len([t for t in day_trades if t['pnl_dollars'] > 0])
        print(f"    {date}: ${pnl:>+10,.2f} ({n_t} trades, {n_w} wins)")

    # Save
    out_dir = 'advance_engine/results'
    os.makedirs(out_dir, exist_ok=True)
    df_trades.to_csv(os.path.join(out_dir, 'trades.csv'), index=False)

    # Save daily PnL
    df_daily = pd.DataFrame({'date': daily_dates, 'pnl': daily_pnls})
    df_daily.to_csv(os.path.join(out_dir, 'daily_pnl.csv'), index=False)
    print(f"\n  Saved: {out_dir}/trades.csv, {out_dir}/daily_pnl.csv")

    t_end = _time.time()
    elapsed = t_end - t_start
    print(f"\n  Time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    if trading_days > 0:
        print(f"  Speed: {elapsed/trading_days:.1f}s per trading day")


if __name__ == '__main__':
    main()
