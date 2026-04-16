"""
Strategy Ticker — zero-lookahead forward pass using 79D + NN + Brain.

The new system:
  1. Each 1m bar: compute 79D state (partial TFs, zero lookahead)
  2. NN predicts: direction + duration + expected PnL/DD + P(profit)
  3. Brain calibrates: adjusts P(profit), PnL, DD based on accumulated evidence
  4. Execution: compresses trade to fit equity (leash ratio)
  5. Exit: envelope decay with NN-predicted half-life, modulated by path divergence

Usage:
  python tools/strategy_ticker.py                          # all OOS days
  python tools/strategy_ticker.py 2026-01-15               # single day
  python tools/strategy_ticker.py 2026-01-15,2026-01-16    # specific days
  python tools/strategy_ticker.py --equity 500             # starting equity

Spec: docs/Active/NN_SPEC.md, docs/Active/EXIT_MATH_ANALYSIS.md
"""
import warnings
warnings.filterwarnings('ignore', module='numba')

import gc
import glob
import math
import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import (
    extract_features, build_all_tf_ohlcv, describe_features,
    FEATURE_NAMES, N_FEATURES, TF_ORDER
)
from core.incremental_ticker import IncrementalTicker
from core.regret_tracker import RegretTracker
from core.statistical_field_engine import StatisticalFieldEngine
from core.strategy_brain import StrategyBrain, compute_state_bin
from training.train_strategy_nn import StrategyRouterNN, DIR_CLASSES, DURATION_BUCKETS

TICK = 0.25
TV = 0.50
LN2 = 0.693  # ln(2) for half-life decay

# ─── Config ──────────────────────────────────────────────────────────
MODEL_PATH = 'checkpoints/strategy_nn/best_model.pt'
ATLAS_1M = 'DATA/ATLAS/1m'

DEFAULT_EQUITY = 100.0          # starting equity ($100)
MAX_RISK_PCT = 0.10             # max 10% of equity per trade
DAILY_RISK_PCT = 0.20           # stop trading if daily loss > 20%
MARGIN_FLOOR = 50.0             # minimum equity to trade
MIN_P_PROFIT = 0.52             # minimum P(profit) to enter
COST_PER_TRADE = 0.50           # 1 tick round trip
HISTORY_DAYS = 25               # 1m history for higher TF context

# Envelope decay
ENVELOPE_FLOOR_PCT = 0.15       # floor as fraction of peak
ENVELOPE_MIN_BARS = 2           # don't exit before this


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Strategy Ticker — 79D + NN + Brain forward pass')
    p.add_argument('target', nargs='?', default='all', help='Date(s) or "all"')
    p.add_argument('--equity', type=float, default=DEFAULT_EQUITY)
    p.add_argument('--no-brain', action='store_true', help='Disable brain calibration')
    p.add_argument('--verbose', '-v', action='store_true', help='Print every NN decision')
    return p.parse_args()


def load_model(device):
    """Load trained NN model."""
    if not os.path.exists(MODEL_PATH):
        print(f'ERROR: No model at {MODEL_PATH}. Run training first.')
        sys.exit(1)

    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = StrategyRouterNN(input_dim=N_FEATURES, dropout=0.0)  # no dropout at inference
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    print(f'  Model loaded: epoch {ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.2f}')
    print(f'  Val metrics: dir_acc={ckpt["val_metrics"]["dir_acc"]:.1f}% '
          f'dur_acc={ckpt["val_metrics"]["dur_acc"]:.1f}% '
          f'pnl_corr={ckpt["val_metrics"]["pnl_corr"]:.4f}')
    return model


def envelope_decay_exit(bars_held, half_life, peak_pnl, current_pnl, floor_pct=ENVELOPE_FLOOR_PCT):
    """Check if envelope decay triggers exit.

    Returns (should_exit, envelope_level)
    """
    if bars_held < ENVELOPE_MIN_BARS or half_life <= 0:
        return False, peak_pnl

    decay = math.exp(-LN2 * bars_held / max(1, half_life))
    floor = peak_pnl * floor_pct
    envelope = floor + (peak_pnl - floor) * decay

    # Exit when current PnL drops below envelope
    if peak_pnl > COST_PER_TRADE and current_pnl < envelope:
        return True, envelope

    return False, envelope


def run_day(day_file, model, brain, device, history_1m, equity, daily_pnl,
            prev_velocities, sfe, verbose=False):
    """Run one day: ticker feeds bars, wrapper makes decisions.

    TICKER: IncrementalTicker feeds bars one at a time (zero lookahead by construction).
    WRAPPER: This function — reads ticker state, calls NN, manages positions.

    Returns: (trades_list, updated_equity, updated_daily_pnl, updated_prev_velocities)
    """
    today_1m = pd.read_parquet(day_file).sort_values('timestamp').reset_index(drop=True)
    if len(today_1m) < 30:
        return [], equity, daily_pnl, prev_velocities

    day_name = os.path.basename(day_file).replace('.parquet', '')
    n_bars = len(today_1m)
    trades = []

    # ════════════════════════════════════════════════════════════════
    # TICKER: feeds bars one at a time. Zero lookahead by construction.
    # History is pre-loaded (all closed, safe). Ticker never sees future.
    # REGRET: forensics only — uses full day closes AFTER the fact.
    # ════════════════════════════════════════════════════════════════
    ticker = IncrementalTicker(history_1m=history_1m)
    regret = RegretTracker()
    closes = today_1m['close'].values  # full day — regret tracker uses this for what-if

    # Position state (WRAPPER concern — not the ticker's business)
    in_pos = False
    entry_price = 0.0
    direction = None
    bars_held = 0
    peak_pnl = 0.0
    half_life = 0
    strategy_id = None
    state_bin = None
    stopped_for_day = False
    consecutive_losses = 0

    import time as _time
    _t_start_day = _time.perf_counter()
    _t_ticker_total = 0.0
    _t_nn_total = 0.0

    pbar = tqdm(range(n_bars), desc=f'  {day_name}', unit='bar',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    for bar_idx in pbar:
        row = today_1m.iloc[bar_idx]
        bar = {
            'timestamp': row['timestamp'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
        }
        price = bar['close']
        ts = bar['timestamp']
        time_str = datetime.utcfromtimestamp(ts).strftime('%H:%M')

        # ── TICKER: feed one bar, get state ──
        _t0 = _time.perf_counter()
        state = ticker.feed_bar(bar)
        _t_ticker_total += _time.perf_counter() - _t0
        feat = state['features']

        # Live progress
        pos_str = f'{direction[0].upper()}' if in_pos else '-'
        elapsed = _time.perf_counter() - _t_start_day
        ms_bar = elapsed / max(bar_idx + 1, 1) * 1000
        pbar.set_postfix_str(f'pnl=${daily_pnl:+.0f} eq=${equity:.0f} tr={len(trades)} {pos_str} {ms_bar:.0f}ms/bar')

        # ── WRAPPER: all decisions happen here ──

        # === EXIT CHECK ===
        if in_pos:
            bars_held += 1

            if direction == 'long':
                pnl = (price - entry_price) / TICK * TV
            else:
                pnl = (entry_price - price) / TICK * TV

            peak_pnl = max(peak_pnl, pnl)
            exit_reason = None

            # 1. Hard stop (locked at entry — doesn't change with equity)
            if pnl < -entry_risk_budget:
                exit_reason = 'hard_stop'

            # 2. Envelope decay
            if not exit_reason:
                should_exit, envelope = envelope_decay_exit(
                    bars_held, half_life, peak_pnl, pnl
                )
                if should_exit:
                    exit_reason = 'envelope_decay'

            # 3. Max hold
            if not exit_reason and bars_held >= max(half_life * 2, 3):
                exit_reason = 'max_hold'

            # 4. End of day
            if not exit_reason and bar_idx >= n_bars - 5:
                exit_reason = 'end_of_day'

            if exit_reason:
                equity += pnl - COST_PER_TRADE
                daily_pnl += pnl - COST_PER_TRADE

                from core.features import N_CORE
                tf_1m = 1 * N_CORE
                # Entry features (saved at entry time)
                e_z = entry_feat[tf_1m + 0] if entry_feat is not None else 0
                e_dmi = entry_feat[tf_1m + 1] if entry_feat is not None else 0
                e_vr = entry_feat[tf_1m + 2] if entry_feat is not None else 0
                trades.append({
                    'trade_id': len(trades), 'day': day_name, 'time': time_str,
                    'dir': direction, 'pnl': pnl - COST_PER_TRADE,
                    'exit': exit_reason, 'held': bars_held, 'peak': peak_pnl,
                    'strategy': f'{direction}_{half_life}',
                    'strategy_id': strategy_id, 'half_life': half_life,
                    'equity_after': equity,
                    # NN output at entry
                    'nn_p_profit': entry_calibrated.get('p_profit', 0),
                    'nn_expected_pnl': entry_calibrated.get('expected_pnl', 0),
                    'nn_confidence': entry_calibrated.get('confidence', 0),
                    'nn_dir_conf': entry_calibrated.get('dir_confidence', 0),
                    'nn_dur_conf': entry_calibrated.get('dur_confidence', 0),
                    # 1m features at entry
                    'entry_z_1m': e_z, 'entry_dmi_1m': e_dmi, 'entry_vr_1m': e_vr,
                    # 1m features at exit
                    'exit_z_1m': feat[tf_1m + 0],
                    'exit_dmi_1m': feat[tf_1m + 1],
                    'exit_vr_1m': feat[tf_1m + 2],
                })

                if brain is not None:
                    brain.update({
                        'strategy_id': strategy_id, 'state_bin': state_bin,
                        'actual_pnl': pnl - COST_PER_TRADE,
                        'actual_drawdown': abs(min(0, pnl - peak_pnl)),
                        'actual_duration': bars_held,
                        'was_profitable': pnl > COST_PER_TRADE,
                    })

                if pnl <= COST_PER_TRADE:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                in_pos = False
                continue

        # === DAILY STOP ===
        if stopped_for_day:
            continue
        if daily_pnl < -(equity * DAILY_RISK_PCT):
            stopped_for_day = True
            continue

        # === ENTRY CHECK ===
        if not in_pos:
            # NN prediction (from ticker's clean state)
            _t1 = _time.perf_counter()
            feat_tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
            feat_tensor = torch.nan_to_num(feat_tensor)
            nn_output = model.predict(feat_tensor)
            _t_nn_total += _time.perf_counter() - _t1

            # Brain calibration
            if brain is not None:
                calibrated = brain.calibrate(nn_output, feat, FEATURE_NAMES)
            else:
                calibrated = nn_output

            if calibrated['direction'] == 'skip':
                continue
            if calibrated['p_profit'] < MIN_P_PROFIT:
                continue

            # Execution: leash
            max_risk = equity * MAX_RISK_PCT
            if consecutive_losses >= 3:
                max_risk *= 0.5
            if consecutive_losses >= 5:
                max_risk *= 0.25

            max_risk = min(max_risk, equity - MARGIN_FLOOR)
            if max_risk <= COST_PER_TRADE:
                continue

            expected_dd = max(calibrated['expected_drawdown'], COST_PER_TRADE)
            leash = min(1.0, max_risk / expected_dd)

            ev = (calibrated['p_profit'] * calibrated['expected_pnl'] * leash -
                  (1 - calibrated['p_profit']) * expected_dd * leash)
            if ev <= COST_PER_TRADE:
                continue

            # Enter
            in_pos = True
            entry_price = price
            direction = calibrated['direction']
            half_life = calibrated['duration'] * leash
            half_life = max(1, half_life)
            bars_held = 0
            peak_pnl = 0.0
            strategy_id = calibrated['strategy_id']
            state_bin = calibrated.get('state_bin', ())
            entry_calibrated = calibrated  # save for trade log
            entry_feat = feat.copy()  # save 79D at entry
            entry_risk_budget = max_risk  # hard stop locked at entry

            # Regret: record what-if (forensics — uses full day closes)
            regret.record_entry(
                bar_idx=bar_idx, chosen_dir=direction,
                chosen_dur=int(calibrated['duration']),
                entry_price=price, closes=closes,
                nn_p_profit=calibrated['p_profit'],
                nn_expected_pnl=calibrated['expected_pnl'],
            )


    # Force close at end of day
    if in_pos:
        pnl = ((price - entry_price) / TICK * TV if direction == 'long'
               else (entry_price - price) / TICK * TV)
        equity += pnl - COST_PER_TRADE
        daily_pnl += pnl - COST_PER_TRADE
        trades.append({
            'trade_id': len(trades), 'day': day_name, 'time': time_str,
            'dir': direction, 'pnl': pnl - COST_PER_TRADE, 'exit': 'end_of_day',
            'held': bars_held, 'peak': peak_pnl,
            'strategy': f'{direction}_{half_life:.0f}',
            'strategy_id': strategy_id, 'half_life': half_life,
            'equity_after': equity,
        })
        if brain is not None:
            brain.update({
                'strategy_id': strategy_id, 'state_bin': state_bin,
                'actual_pnl': pnl - COST_PER_TRADE,
                'actual_drawdown': abs(min(0, pnl - peak_pnl)),
                'actual_duration': bars_held,
                'was_profitable': pnl > COST_PER_TRADE,
            })

    _t_total = _time.perf_counter() - _t_start_day
    if verbose:
        print(f'\n  Timing: {_t_total:.1f}s total | '
              f'ticker={_t_ticker_total:.1f}s ({_t_ticker_total/_t_total*100:.0f}%) | '
              f'nn={_t_nn_total:.1f}s ({_t_nn_total/_t_total*100:.0f}%) | '
              f'{n_bars/_t_total:.0f} bars/sec | '
              f'{_t_ticker_total/n_bars*1000:.1f}ms/bar')

    if verbose:
        print(f'\n{ticker.perf_report()}')
        print(f'\n{regret.report()}')

    # Free GPU memory between days
    ticker.cleanup()

    return trades, equity, daily_pnl, prev_velocities


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'STRATEGY TICKER — 79D + NN + Brain')
    print(f'  Device: {device}')
    print(f'  Starting equity: ${args.equity:.2f}')
    print(f'  Max risk/trade: {MAX_RISK_PCT*100:.0f}% | Daily stop: {DAILY_RISK_PCT*100:.0f}%')
    print(f'  Min P(profit): {MIN_P_PROFIT}')

    # Load model
    model = load_model(device)

    # Brain
    brain = None if args.no_brain else StrategyBrain()

    # Find target days
    all_1m_files = sorted(glob.glob(os.path.join(ATLAS_1M, '*.parquet')))

    if args.target == 'all':
        # OOS: 2026 data where we have 1s (for validation against nightmare ticker)
        target_files = [f for f in all_1m_files if '2026_01' in os.path.basename(f)
                        or '2026_02' in os.path.basename(f)]
    elif ',' in args.target:
        dates = args.target.split(',')
        target_files = [f for f in all_1m_files
                        if os.path.basename(f).replace('.parquet', '').replace('_', '-') in dates]
    else:
        target_files = [f for f in all_1m_files
                        if os.path.basename(f).replace('.parquet', '').replace('_', '-') == args.target]

    print(f'  Target days: {len(target_files)}')

    # SFE (reused across days)
    sfe = StatisticalFieldEngine()

    # Run
    equity = args.equity
    all_trades = []
    daily_summary = []
    prev_velocities = {}

    for file_idx, day_file in enumerate(tqdm(target_files, desc='Days', unit='day')):
        day_name = os.path.basename(day_file).replace('.parquet', '')

        # Load history
        day_pos = all_1m_files.index(day_file) if day_file in all_1m_files else -1
        if day_pos > 0:
            hist_start = max(0, day_pos - HISTORY_DAYS)
            hist_files = all_1m_files[hist_start:day_pos]
            history = pd.concat([pd.read_parquet(f) for f in hist_files],
                               ignore_index=True).sort_values('timestamp')
        else:
            history = pd.DataFrame()

        daily_pnl = 0.0
        trades, equity, daily_pnl, prev_velocities = run_day(
            day_file, model, brain, device, history, equity, daily_pnl, prev_velocities, sfe,
            verbose=args.verbose
        )

        all_trades.extend(trades)
        n_trades = len(trades)
        wr = sum(1 for t in trades if t['pnl'] > 0) / max(n_trades, 1) * 100

        daily_summary.append({
            'day': day_name, 'trades': n_trades, 'wr': wr,
            'pnl': daily_pnl, 'equity': equity,
        })

        del history
        gc.collect()

    # === RESULTS ===
    print(f'\n{"="*70}')
    t = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    n_days = len(daily_summary)
    n_trades = len(t) if len(t) > 0 else 0
    total_pnl = t['pnl'].sum() if len(t) > 0 else 0
    wr = (t['pnl'] > 0).mean() * 100 if len(t) > 0 else 0

    print(f'RESULTS (zero lookahead, {n_days} days):')
    print(f'  {n_trades} trades | WR={wr:.1f}% | PnL=${total_pnl:.2f} | $/day=${total_pnl/max(n_days,1):.2f}')
    print(f'  Starting equity: ${args.equity:.2f} | Final equity: ${equity:.2f}')

    if len(t) > 0:
        # Exit breakdown
        print(f'\n  Exit breakdown:')
        for ex in sorted(t['exit'].unique()):
            et = t[t['exit'] == ex]
            ex_wr = (et['pnl'] > 0).mean() * 100
            print(f'    {ex:<20} {len(et):>4} trades  WR={ex_wr:>5.1f}%  ${et["pnl"].sum():>8.2f}  ${et["pnl"].mean():>6.2f}/tr')

        # Daily breakdown
        print(f'\n  Daily:')
        cumul = 0
        for ds in daily_summary:
            cumul += ds['pnl']
            flag = '<<<' if ds['pnl'] > 50 else '!!!' if ds['pnl'] < -50 else ''
            print(f'    {ds["day"]}  {ds["trades"]:>3} trades  {ds["wr"]:>4.0f}%  '
                  f'${ds["pnl"]:>8.2f}  cumul=${cumul:>8.2f}  eq=${ds["equity"]:>8.2f} {flag}')

        winning_days = sum(1 for ds in daily_summary if ds['pnl'] > 0)
        print(f'\n  Winning days: {winning_days}/{n_days}')

    # Brain summary
    if brain is not None:
        print(f'\n{brain.get_strategy_summary()}')

        # Detailed brain table — what the Bayesian memory looks like
        print(f'\n  Brain Context Table (top 20 by trade count):')
        print(f'  {"Key":<45} {"N":>4} {"WR":>6} {"Avg PnL":>8}')
        print(f'  {"-"*70}')
        ctx_sorted = sorted(brain.context_stats.items(),
                           key=lambda x: -x[1]['total'])
        for (sid, sbin), stats in ctx_sorted[:20]:
            if stats['total'] == 0:
                continue
            wr = stats['wins'] / stats['total'] * 100
            avg = stats['pnl_sum'] / stats['total']
            sid_label = f'{sid[0]}_{sid[1]}'
            bin_label = f'z={sbin[0]} dmi={sbin[1]} vr={sbin[2]} 1h_z={sbin[3]} 1h_dmi={sbin[4]}' if len(sbin) >= 5 else str(sbin)
            print(f'  {sid_label:<12} {bin_label:<32} {stats["total"]:>4} {wr:>5.0f}% ${avg:>7.1f}')

        # Show which contexts are profitable vs not
        profitable_ctx = sum(1 for _, s in brain.context_stats.items() if s['total'] >= 2 and s['wins'] / s['total'] > 0.5)
        losing_ctx = sum(1 for _, s in brain.context_stats.items() if s['total'] >= 2 and s['wins'] / s['total'] <= 0.5)
        print(f'\n  Contexts with 2+ trades: {profitable_ctx} profitable, {losing_ctx} losing')

    # Save trades
    if len(t) > 0:
        out_path = f'reports/findings/strategy_ticker_trades_{datetime.now():%Y%m%d}.csv'
        t.to_csv(out_path, index=False)
        print(f'\nTrades saved: {out_path}')

    del sfe
    gc.collect()


if __name__ == '__main__':
    main()
