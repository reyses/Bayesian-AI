"""
Conviction Forward Pass — simplest possible CNN-driven trading.

Rules:
  ENTER LONG:  P(long) > entry_threshold (default 0.65)
  ENTER SHORT: P(long) < (1 - entry_threshold) (default 0.35)
  EXIT:        P(long) crosses 0.5 (direction flipped)
  BACKSTOP:    Hard SL (default 40 ticks)

One bar at a time. No templates, no gates, no brain. Just conviction.

Usage:
  python -m tools.conviction_forward_pass
  python -m tools.conviction_forward_pass --entry 0.70 --sl 30
  python -m tools.conviction_forward_pass --months 2026-02,2026-03
"""
import argparse
import gc
import glob
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from core.trade_cnn import StatePredictor
from training.train_trade_cnn import extract_features_13d

ATLAS_ROOT = 'DATA/ATLAS'
CHECKPOINT = 'checkpoints/trade_cnn/best_model.pt'
TICK = 0.25
TICK_VALUE = 0.50  # MNQ
LOOKBACK = 10
OUT_DIR = 'reports/findings'


def sigmoid_p(dmi_pred, scale=5.0):
    """Convert predicted dmi_diff to P(long) via sigmoid."""
    return 1.0 / (1.0 + np.exp(-dmi_pred / scale))


def main():
    parser = argparse.ArgumentParser(description='Conviction Forward Pass')
    parser.add_argument('--tf', default='1m', help='Execution timeframe (15s or 1m)')
    parser.add_argument('--entry', type=float, default=0.65, help='P(long) threshold for LONG entry')
    parser.add_argument('--exit', type=float, default=0.30, help='P(long) exit threshold (exit LONG when P < this, must be very sure)')
    parser.add_argument('--sl', type=float, default=40.0, help='Hard SL in ticks')
    parser.add_argument('--months', default=None, help='Comma-separated YYYY-MM (default: all)')
    parser.add_argument('--oos-start', default='2026-02', help='OOS boundary YYYY-MM')
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    parser.add_argument('--signal-tf', default='1m', help='CNN signal timeframe (must match training)')
    parser.add_argument('--bars-per-signal', type=int, default=0,
                        help='Bars per signal TF bar (auto: 4 for 15s exec + 1m signal). '
                             '0 = same TF, no aggregation.')
    parser.add_argument('--plot', action='store_true', help='Plot conviction + trade markers')
    parser.add_argument('--plot-date', default=None, help='Single date to plot (YYYY-MM-DD). Filters plot only.')
    args = parser.parse_args()

    short_threshold = 1.0 - args.entry  # mirror: 0.65 entry → 0.35 SHORT

    # Auto-detect aggregation: 15s exec with 1m signal = 4 bars per signal
    bars_per_signal = args.bars_per_signal
    if bars_per_signal == 0 and args.tf != args.signal_tf:
        tf_secs = {'1s': 1, '5s': 5, '15s': 15, '30s': 30, '1m': 60, '5m': 300}
        exec_s = tf_secs.get(args.tf, 60)
        signal_s = tf_secs.get(args.signal_tf, 60)
        bars_per_signal = signal_s // exec_s
    aggregating = bars_per_signal > 1

    print(f"\n{'='*60}")
    print(f"CONVICTION FORWARD PASS")
    print(f"{'='*60}")
    print(f"  Exec TF: {args.tf} | Signal TF: {args.signal_tf}")
    if aggregating:
        print(f"  Aggregation: {bars_per_signal} x {args.tf} bars -> 1 x {args.signal_tf} CNN prediction")
    print(f"  Entry: P>{args.entry} LONG, P<{short_threshold:.2f} SHORT")
    exit_long = args.exit           # exit LONG when P < this (default 0.40)
    exit_short = 1.0 - args.exit     # exit SHORT when P > this (default 0.60)
    print(f"  Exit: LONG when P<{exit_long}, SHORT when P>{exit_short:.2f} | SL: {args.sl} ticks")
    print(f"  OOS boundary: {args.oos_start}")

    # Load CNN
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=True)
    model_state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt))
    model = StatePredictor(n_features=13, latent_dim=64, n_labels=21).to(dev)
    model.load_state_dict(model_state)
    model.eval()
    print(f"  CNN: {args.checkpoint}")

    # Collect files
    tf_dir = os.path.join(ATLAS_ROOT, args.tf)
    all_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    if args.months:
        months = args.months.split(',')
        all_files = [f for f in all_files
                     if os.path.basename(f)[:7].replace('_', '-') in months]

    if not all_files:
        print(f"  No files found in {tf_dir}")
        return

    # Split IS/OOS
    oos_boundary = args.oos_start.replace('-', '_')
    is_files = [f for f in all_files if os.path.basename(f)[:7] < oos_boundary]
    oos_files = [f for f in all_files if os.path.basename(f)[:7] >= oos_boundary]
    print(f"  Files: {len(is_files)} IS, {len(oos_files)} OOS")

    # Run both passes
    for mode, files in [('IS', is_files), ('OOS', oos_files)]:
        if not files:
            print(f"\n  {mode}: no files, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  {mode} FORWARD PASS ({len(files)} files)")
        print(f"{'='*60}")

        # State
        feat_buffer = []       # rolling LOOKBACK of 13D features (at signal TF)
        in_position = False
        direction = None       # 'LONG' or 'SHORT'
        entry_price = 0.0
        entry_bar = 0
        entry_ts = 0.0
        entry_p = 0.0
        peak_mfe = 0.0
        cached_p_long = 0.5    # cached CNN prediction between signal bars
        agg_count = 0          # aggregation counter (for 15s->1m)
        agg_open = 0.0
        agg_high = -1e9
        agg_low = 1e9
        agg_vol = 0.0

        # Results
        trades = []
        daily_pnl = {}
        total_bars = 0

        # Per-bar data for plotting
        all_timestamps = []
        all_prices = []
        all_p_long = []
        prev_day_str = None  # track day boundary

        sfe = StatisticalFieldEngine()

        for fpath in tqdm(files, desc=f"  {mode}", unit='file'):
            fname = os.path.basename(fpath).replace('.parquet', '')

            df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
            if len(df) < LOOKBACK + 5:
                continue

            # Compute 13D features for this file
            states = sfe.batch_compute_states(df)
            feats = extract_features_13d(states, df)
            del states; gc.collect()

            prices = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            timestamps = df['timestamp'].values

            # Process one bar at a time
            for i in range(len(df)):
                total_bars += 1
                price = prices[i]
                high = highs[i]
                low = lows[i]
                ts = timestamps[i]

                # ── CNN prediction (at signal TF resolution) ──
                if aggregating:
                    # Aggregate exec TF bars into signal TF bars
                    agg_count += 1
                    if agg_count == 1:
                        agg_open = price
                    agg_high = max(agg_high, high)
                    agg_low = min(agg_low, low)

                    if agg_count >= bars_per_signal:
                        # Signal bar complete — update feature buffer with this bar's features
                        # Use the features from the last exec bar (close-of-signal-bar)
                        feat_buffer.append(feats[i])
                        if len(feat_buffer) > LOOKBACK:
                            feat_buffer = feat_buffer[-LOOKBACK:]

                        # Run CNN if we have enough lookback
                        if len(feat_buffer) >= LOOKBACK:
                            with torch.no_grad():
                                x = torch.FloatTensor(np.array(feat_buffer)).unsqueeze(0).to(dev)
                                pred = model(x)[0].cpu().numpy()
                            dmi_pred = pred[0]
                            cached_p_long = sigmoid_p(dmi_pred)

                        # Reset aggregator
                        agg_count = 0
                        agg_high = -1e9
                        agg_low = 1e9

                    p_long = cached_p_long  # use cached prediction between signal bars
                else:
                    # Same TF: predict every bar
                    feat_buffer.append(feats[i])
                    if len(feat_buffer) > LOOKBACK:
                        feat_buffer = feat_buffer[-LOOKBACK:]

                    if len(feat_buffer) < LOOKBACK:
                        continue

                    with torch.no_grad():
                        x = torch.FloatTensor(np.array(feat_buffer)).unsqueeze(0).to(dev)
                        pred = model(x)[0].cpu().numpy()
                    dmi_pred = pred[0]
                    p_long = sigmoid_p(dmi_pred)

                # Store for plotting
                all_timestamps.append(ts)
                all_prices.append(price)
                all_p_long.append(p_long)

                # Date for daily tracking
                day_str = datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d')

                # Day boundary: force close + reset
                if prev_day_str is not None and day_str != prev_day_str and in_position:
                    if direction == 'LONG':
                        pnl_ticks = (price - entry_price) / TICK
                    else:
                        pnl_ticks = (entry_price - price) / TICK
                    pnl_dollars = pnl_ticks * TICK_VALUE
                    trade = {
                        'direction': direction,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'entry_bar': entry_bar,
                        'exit_bar': total_bars,
                        'bars_held': total_bars - entry_bar,
                        'entry_ts': entry_ts,
                        'exit_ts': ts,
                        'pnl_ticks': pnl_ticks,
                        'pnl': pnl_dollars,
                        'result': 'WIN' if pnl_dollars > 0 else 'LOSS',
                        'exit_reason': 'end_of_day',
                        'entry_p': entry_p,
                        'exit_p': p_long,
                        'peak_mfe_ticks': peak_mfe,
                        'day': prev_day_str,
                    }
                    trades.append(trade)
                    if prev_day_str not in daily_pnl:
                        daily_pnl[prev_day_str] = {'pnl': 0.0, 'trades': 0, 'wins': 0}
                    daily_pnl[prev_day_str]['pnl'] += pnl_dollars
                    daily_pnl[prev_day_str]['trades'] += 1
                    if pnl_dollars > 0:
                        daily_pnl[prev_day_str]['wins'] += 1
                    in_position = False
                    direction = None
                    # Reset feature buffer for new day
                    feat_buffer = []
                prev_day_str = day_str

                # ── Position management ──
                if in_position:
                    # PnL
                    if direction == 'LONG':
                        pnl_ticks = (price - entry_price) / TICK
                        mfe_ticks = (high - entry_price) / TICK
                        mae_ticks = (entry_price - low) / TICK
                    else:
                        pnl_ticks = (entry_price - price) / TICK
                        mfe_ticks = (entry_price - low) / TICK
                        mae_ticks = (high - entry_price) / TICK

                    peak_mfe = max(peak_mfe, mfe_ticks)

                    # Exit: conviction collapse (P crossed exit threshold)
                    flip_exit = False
                    if direction == 'LONG' and p_long < exit_long:
                        flip_exit = True
                        exit_reason = 'conviction_collapse'
                    elif direction == 'SHORT' and p_long > exit_short:
                        flip_exit = True
                        exit_reason = 'conviction_collapse'

                    # Exit: conviction-weighted SL
                    # Strong conviction = wide SL (trust the CNN), weak = tight SL
                    if direction == 'LONG':
                        conviction_strength = max(0, (p_long - 0.5) * 2)
                    else:
                        conviction_strength = max(0, (0.5 - p_long) * 2)
                    SL_FLOOR = 80  # minimum SL even at zero conviction ($40)
                    dynamic_sl = SL_FLOOR + (args.sl - SL_FLOOR) * conviction_strength
                    sl_exit = mae_ticks >= dynamic_sl
                    if sl_exit:
                        exit_reason = 'dynamic_sl'

                    if flip_exit or sl_exit:
                        pnl_dollars = pnl_ticks * TICK_VALUE
                        trade = {
                            'direction': direction,
                            'entry_price': entry_price,
                            'exit_price': price,
                            'entry_bar': entry_bar,
                            'exit_bar': total_bars,
                            'bars_held': total_bars - entry_bar,
                            'entry_ts': entry_ts,
                            'exit_ts': ts,
                            'pnl_ticks': pnl_ticks,
                            'pnl': pnl_dollars,
                            'result': 'WIN' if pnl_dollars > 0 else 'LOSS',
                            'exit_reason': exit_reason,
                            'entry_p': entry_p,
                            'exit_p': p_long,
                            'peak_mfe_ticks': peak_mfe,
                            'day': day_str,
                        }
                        trades.append(trade)

                        if day_str not in daily_pnl:
                            daily_pnl[day_str] = {'pnl': 0.0, 'trades': 0, 'wins': 0}
                        daily_pnl[day_str]['pnl'] += pnl_dollars
                        daily_pnl[day_str]['trades'] += 1
                        if pnl_dollars > 0:
                            daily_pnl[day_str]['wins'] += 1

                        in_position = False
                        direction = None

                # ── Entry evaluation (only if flat) ──
                if not in_position:
                    if p_long > args.entry:
                        in_position = True
                        direction = 'LONG'
                        entry_price = price
                        entry_bar = total_bars
                        entry_ts = ts
                        entry_p = p_long
                        peak_mfe = 0.0
                    elif p_long < short_threshold:
                        in_position = True
                        direction = 'SHORT'
                        entry_price = price
                        entry_bar = total_bars
                        entry_ts = ts
                        entry_p = p_long
                        peak_mfe = 0.0

            # Force close at end of file
            if in_position:
                if direction == 'LONG':
                    pnl_ticks = (prices[-1] - entry_price) / TICK
                else:
                    pnl_ticks = (entry_price - prices[-1]) / TICK
                pnl_dollars = pnl_ticks * TICK_VALUE
                trade = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': prices[-1],
                    'entry_bar': entry_bar,
                    'exit_bar': total_bars,
                    'bars_held': total_bars - entry_bar,
                    'entry_ts': entry_ts,
                    'exit_ts': timestamps[-1],
                    'pnl_ticks': pnl_ticks,
                    'pnl': pnl_dollars,
                    'result': 'WIN' if pnl_dollars > 0 else 'LOSS',
                    'exit_reason': 'end_of_data',
                    'entry_p': entry_p,
                    'exit_p': 0.5,
                    'peak_mfe_ticks': peak_mfe,
                    'day': day_str,
                }
                trades.append(trade)
                if day_str not in daily_pnl:
                    daily_pnl[day_str] = {'pnl': 0.0, 'trades': 0, 'wins': 0}
                daily_pnl[day_str]['pnl'] += pnl_dollars
                daily_pnl[day_str]['trades'] += 1
                if pnl_dollars > 0:
                    daily_pnl[day_str]['wins'] += 1
                in_position = False
                direction = None

            del df, feats; gc.collect()

        # ── Report ─────────────────────────────────────────────
        n_trades = len(trades)
        if n_trades == 0:
            print(f"\n  {mode}: 0 trades")
            continue

        total_pnl = sum(t['pnl'] for t in trades)
        n_wins = sum(1 for t in trades if t['result'] == 'WIN')
        wr = n_wins / n_trades * 100
        n_days = len(daily_pnl)
        per_day = total_pnl / n_days if n_days else 0
        avg_trade = total_pnl / n_trades

        # Direction split
        longs = [t for t in trades if t['direction'] == 'LONG']
        shorts = [t for t in trades if t['direction'] == 'SHORT']
        long_pnl = sum(t['pnl'] for t in longs)
        short_pnl = sum(t['pnl'] for t in shorts)

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            r = t['exit_reason']
            if r not in exit_reasons:
                exit_reasons[r] = {'n': 0, 'pnl': 0.0, 'wins': 0}
            exit_reasons[r]['n'] += 1
            exit_reasons[r]['pnl'] += t['pnl']
            if t['result'] == 'WIN':
                exit_reasons[r]['wins'] += 1

        # Hold duration
        hold_bars = [t['bars_held'] for t in trades]
        avg_hold = np.mean(hold_bars)
        med_hold = np.median(hold_bars)

        # MFE capture
        mfe_ticks = [t['peak_mfe_ticks'] for t in trades]
        pnl_ticks = [t['pnl_ticks'] for t in trades]
        capture = [p / m * 100 if m > 0 else 0 for p, m in zip(pnl_ticks, mfe_ticks)]
        avg_capture = np.mean(capture)

        print(f"\n  {'='*50}")
        print(f"  {mode} RESULTS")
        print(f"  {'='*50}")
        print(f"  Trades:    {n_trades}")
        print(f"  Win Rate:  {wr:.1f}%")
        print(f"  Total PnL: ${total_pnl:,.2f}")
        print(f"  Per Day:   ${per_day:,.2f}")
        print(f"  Avg/trade: ${avg_trade:,.2f}")
        print(f"  Days:      {n_days}")
        print(f"")
        print(f"  Direction:  LONG={len(longs)} (${long_pnl:,.2f})  SHORT={len(shorts)} (${short_pnl:,.2f})")
        print(f"  Hold:       avg={avg_hold:.1f} bars  median={med_hold:.1f} bars")
        print(f"  MFE capture: {avg_capture:.1f}%")
        print(f"")
        print(f"  EXIT REASONS:")
        print(f"  {'Reason':<20} {'N':>6} {'WR%':>6} {'PnL':>12} {'Avg':>10}")
        print(f"  {'-'*58}")
        for reason, stats in sorted(exit_reasons.items(), key=lambda x: -x[1]['n']):
            wr_r = stats['wins'] / stats['n'] * 100 if stats['n'] else 0
            avg_r = stats['pnl'] / stats['n'] if stats['n'] else 0
            print(f"  {reason:<20} {stats['n']:>6} {wr_r:>5.1f}% ${stats['pnl']:>10,.2f} ${avg_r:>8,.2f}")

        # Daily ledger
        print(f"\n  DAILY LEDGER:")
        print(f"  {'Date':<12} {'Trades':>7} {'WR%':>6} {'PnL':>12} {'Cumul':>12}")
        print(f"  {'-'*55}")
        cumul = 0.0
        for day in sorted(daily_pnl.keys()):
            d = daily_pnl[day]
            cumul += d['pnl']
            wr_d = d['wins'] / d['trades'] * 100 if d['trades'] else 0
            print(f"  {day:<12} {d['trades']:>7} {wr_d:>5.1f}% ${d['pnl']:>10,.2f} ${cumul:>10,.2f}")

        # Save trade log
        out_path = os.path.join(OUT_DIR, f'conviction_{mode.lower()}_trades.csv')
        os.makedirs(OUT_DIR, exist_ok=True)
        pd.DataFrame(trades).to_csv(out_path, index=False)
        print(f"\n  Trade log: {out_path}")

        # Plot if requested
        if args.plot and all_timestamps:
            # Filter to single day if specified
            _plot_ts = all_timestamps
            _plot_pr = all_prices
            _plot_pl = all_p_long
            _plot_trades = trades
            if args.plot_date:
                _pd_ts = pd.Timestamp(args.plot_date).timestamp()
                _pd_end = _pd_ts + 86400
                _mask = [(t >= _pd_ts and t < _pd_end) for t in all_timestamps]
                _plot_ts = [t for t, m in zip(all_timestamps, _mask) if m]
                _plot_pr = [p for p, m in zip(all_prices, _mask) if m]
                _plot_pl = [p for p, m in zip(all_p_long, _mask) if m]
                _plot_trades = [t for t in trades
                                if t['entry_ts'] >= _pd_ts and t['entry_ts'] < _pd_end]
                print(f"  Plot filtered to {args.plot_date}: {len(_plot_ts)} bars, {len(_plot_trades)} trades")
            plot_conviction_trades(_plot_ts, _plot_pr, _plot_pl,
                                   _plot_trades, mode, args)

    print(f"\n{'='*60}")
    print(f"  Baseline to beat: $1,609/day OOS")
    print(f"{'='*60}")


def plot_conviction_trades(all_ts, all_prices, all_p_long, trades, mode, args):
    """Plot price colored by conviction + trade entry/exit markers."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from datetime import timezone, timedelta

    timestamps = pd.to_datetime(np.array(all_ts), unit='s', utc=True)
    prices = np.array(all_prices)
    p_long = np.array(all_p_long)
    n = len(timestamps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 9), sharex=True,
                                     gridspec_kw={'height_ratios': [2.5, 1]})
    title = (f'Conviction Forward Pass ({mode}) -- {args.tf} | '
             f'entry>{args.entry} | SL={args.sl}t | {len(trades)} trades')
    fig.suptitle(title, fontsize=12, fontweight='bold')

    # Panel 1: Price colored by P(long)
    ax1.set_facecolor('#1a1a2e')

    points = np.array([mdates.date2num(timestamps), prices]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    cmap = LinearSegmentedColormap.from_list('conv',
        ['#EF5350', '#EF5350', '#FFD700', '#26A69A', '#26A69A'])
    norm = Normalize(vmin=0.0, vmax=1.0)
    seg_colors = np.array([p_long[i+1] if i+1 < n else 0.5 for i in range(n-1)])
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=1.5)
    lc.set_array(seg_colors)
    ax1.add_collection(lc)
    ax1.set_xlim(mdates.date2num(timestamps[0]), mdates.date2num(timestamps[-1]))
    ax1.set_ylim(prices.min() - 5, prices.max() + 5)

    # Trade markers
    ts_arr = np.array(all_ts)
    for t in trades:
        # Entry marker
        eidx = np.argmin(np.abs(ts_arr - t['entry_ts']))
        marker = '^' if t['direction'] == 'LONG' else 'v'
        ax1.scatter(timestamps[eidx], t['entry_price'], marker=marker,
                    color='#00FFFF', s=50, zorder=5, edgecolors='white', linewidths=0.5)

        # Exit marker
        xidx = np.argmin(np.abs(ts_arr - t['exit_ts']))
        color = '#26A69A' if t['pnl'] > 0 else '#EF5350'
        ax1.scatter(timestamps[xidx], t['exit_price'], marker='x',
                    color=color, s=40, zorder=5, linewidths=1.5)

        # Line connecting entry to exit
        ax1.plot([timestamps[eidx], timestamps[xidx]],
                 [t['entry_price'], t['exit_price']],
                 color=color, linewidth=0.5, alpha=0.4)

    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.15)
    cb = fig.colorbar(lc, ax=ax1, pad=0.01, aspect=30)
    cb.set_label('P(long)')

    # Panel 2: P(long) with entry/exit zones
    ax2.set_facecolor('#1a1a2e')
    ax2.plot(timestamps, p_long, color='#00FFFF', linewidth=0.8)
    ax2.axhline(y=0.5, color='white', linewidth=1, alpha=0.5, linestyle='--')
    ax2.axhline(y=args.entry, color='#26A69A', linewidth=0.5, alpha=0.5, linestyle=':',
                label=f'LONG entry ({args.entry})')
    ax2.axhline(y=1-args.entry, color='#EF5350', linewidth=0.5, alpha=0.5, linestyle=':',
                label=f'SHORT entry ({1-args.entry:.2f})')
    ax2.fill_between(timestamps, 0.5, p_long,
                      where=p_long > 0.5, color='#26A69A', alpha=0.2)
    ax2.fill_between(timestamps, 0.5, p_long,
                      where=p_long < 0.5, color='#EF5350', alpha=0.2)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('P(long)')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.15)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
    plt.xticks(rotation=45)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f'conviction_{mode.lower()}_plot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot: {out_path}")


if __name__ == '__main__':
    main()
