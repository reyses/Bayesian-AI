"""
Nightmare Protocol Forward Pass — grounded features implementation.

The Master Equation:
  dXt = F_gravity + F_diffusion + F_pid + J(lambda)

Execution Logic (from the protocol):
  1. Calculate Z_fit = (X - mu) / sigma    → z_se feature
  2. Calculate lambda: is Z decaying or expanding?  → variance_ratio
  3. IF Z_fit > 2.0 (Roche Limit) AND lambda < 0 (stable) → FORCE REVERSION
     ELSE IF lambda > 0 (chaotic) → FORCE TREND

Grounded feature mapping:
  Z_fit         → z_se (standard error z-score)
  lambda        → variance_ratio (<1 = stable/reverting, >1 = chaotic/trending)
  v_micro       → velocity at 1m
  v_macro       → velocity at 1h/1D
  F_gravity     → z_se magnitude (further from mean = stronger pull back)
  F_pid Kp      → z_se (proportional error)
  F_pid Ki      → cumulative z_se drift (integral)
  F_pid Kd      → price_accel (derivative of error)
  Roche Limit   → z_se > ROCHE_K (default 2.0)
  Hurst H       → variance_ratio proxy

Multi-TF: run the protocol at 1m, but check 1h/1D for macro alignment.
  - Only trade reversion when macro agrees (1D gravity pulls same direction)
  - Only trade trend when macro confirms breakout

Exit: hold for HALF_CYCLE bars (default 4 min = 4 bars at 1m), or until
      Z returns to zero (mean reached), whichever first.

Usage:
  python -m tools.nightmare_forward_pass
  python -m tools.nightmare_forward_pass --months 2026-03 --roche 2.0
  python -m tools.nightmare_forward_pass --plot-date 2026-03-24
"""
import argparse
import gc
import glob
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
from training.train_trade_cnn import extract_features_13d, FEATURE_NAMES_13D

ATLAS_ROOT = 'DATA/ATLAS'
TICK = 0.25
TICK_VALUE = 0.50
OUT_DIR = 'reports/findings'

# Nightmare Protocol constants
ROCHE_K = 2.0           # z_se threshold for Roche Limit (event horizon)
SINGULARITY_K = 3.0     # z_se threshold for singularity (structural failure)
LAMBDA_THRESHOLD = 1.0  # variance_ratio: <1 = stable (lambda<0), >1 = chaotic (lambda>0)
HALF_CYCLE = 4           # bars to hold (half oscillation cycle at 1m = 4 minutes)
MICRO_MACRO_RATIO = 0.3  # v_micro/v_macro below this = macro dominates


def compute_nightmare_state(feats_1m, feats_1h, feats_1d, i,
                            feats_1m_arr, closes_1m):
    """Compute Nightmare Protocol state variables from grounded features.

    Returns dict with all protocol variables mapped to features.
    """
    # Feature indices (from FEATURE_NAMES_13D)
    IDX_DMI_DIFF = 0
    IDX_VELOCITY = 4
    IDX_Z_SE = 5
    IDX_PRICE_ACCEL = 6
    IDX_VARIANCE_RATIO = 8
    IDX_WICK_RATIO = 10

    # 1m state (current bar)
    z_fit = feats_1m[IDX_Z_SE]                    # Z_fit: position in gravity well
    velocity_micro = feats_1m[IDX_VELOCITY]        # v_micro: 1m velocity
    variance_ratio = feats_1m[IDX_VARIANCE_RATIO]  # lambda proxy
    price_accel = feats_1m[IDX_PRICE_ACCEL]        # Kd: derivative of error
    wick_ratio = feats_1m[IDX_WICK_RATIO]          # rejection signal
    dmi_1m = feats_1m[IDX_DMI_DIFF]

    # 1h state (macro)
    velocity_macro_1h = feats_1h[IDX_VELOCITY] if feats_1h is not None else 0.0
    z_se_1h = feats_1h[IDX_Z_SE] if feats_1h is not None else 0.0
    dmi_1h = feats_1h[IDX_DMI_DIFF] if feats_1h is not None else 0.0

    # 1D state (trend)
    velocity_macro_1d = feats_1d[IDX_VELOCITY] if feats_1d is not None else 0.0
    z_se_1d = feats_1d[IDX_Z_SE] if feats_1d is not None else 0.0
    dmi_1d = feats_1d[IDX_DMI_DIFF] if feats_1d is not None else 0.0

    # Fractal diffusion: v_micro / v_macro ratio
    v_macro = abs(velocity_macro_1h) if abs(velocity_macro_1h) > 0.01 else 0.01
    fractal_ratio = abs(velocity_micro) / v_macro

    # Lyapunov exponent proxy
    # lambda < 0: stable (variance_ratio < 1, mean-reverting)
    # lambda > 0: chaotic (variance_ratio > 1, trending)
    lam = variance_ratio - LAMBDA_THRESHOLD  # negative = stable, positive = chaotic

    # F_gravity: restoring force = z_se magnitude
    f_gravity = -z_fit  # negative z = below mean = gravity pulls up, and vice versa

    # F_pid components
    f_pid_kp = -z_fit                    # proportional: push back toward mean
    f_pid_kd = -price_accel              # derivative: dampen velocity

    # Ki (integral): cumulative z_se over last 10 bars
    if i >= 10:
        f_pid_ki = -np.mean(feats_1m_arr[i-10:i, IDX_Z_SE])
    else:
        f_pid_ki = 0.0

    # Net force vector
    v_net = f_gravity + f_pid_kp * 0.3 + f_pid_ki * 0.1 + f_pid_kd * 0.1

    # Macro alignment: does the daily/hourly trend agree?
    macro_direction = np.sign(dmi_1d) if abs(dmi_1d) > 2 else np.sign(dmi_1h)

    # 4-hour trend: price change over last 240 bars (at 1m)
    LOOKBACK_4H = 240
    if i >= LOOKBACK_4H:
        trend_4h = closes_1m[i] - closes_1m[i - LOOKBACK_4H]
    else:
        trend_4h = 0.0

    return {
        'z_fit': z_fit,
        'lam': lam,
        'variance_ratio': variance_ratio,
        'velocity_micro': velocity_micro,
        'velocity_macro_1h': velocity_macro_1h,
        'velocity_macro_1d': velocity_macro_1d,
        'fractal_ratio': fractal_ratio,
        'f_gravity': f_gravity,
        'f_pid_kp': f_pid_kp,
        'f_pid_ki': f_pid_ki,
        'f_pid_kd': f_pid_kd,
        'v_net': v_net,
        'wick_ratio': wick_ratio,
        'dmi_1m': dmi_1m,
        'dmi_1h': dmi_1h,
        'dmi_1d': dmi_1d,
        'macro_direction': macro_direction,
        'z_se_1h': z_se_1h,
        'z_se_1d': z_se_1d,
        'trend_4h': trend_4h,
    }


def nightmare_decision(state, roche_k=ROCHE_K):
    """Execute Nightmare Protocol trading logic.

    Returns: ('LONG', reason) or ('SHORT', reason) or (None, reason)
    """
    z = state['z_fit']
    lam = state['lam']
    macro_dir = state['macro_direction']
    wick = state['wick_ratio']
    fractal_r = state['fractal_ratio']

    # === THE TRIGGER (from the protocol) ===
    # RULE: Never trade against the daily trend.
    # If 1D DMI says SHORT (macro_dir < 0), only allow SHORT entries.
    # If 1D DMI says LONG (macro_dir > 0), only allow LONG entries.
    # If neutral (macro_dir == 0), allow either direction.

    # Macro bias: trend of the last 4 hours (240 bars at 1m)
    # Computed from the rolling price change over that window
    trend_4h = state.get('trend_4h', 0)  # price change over last 240 bars
    trend_4h_dir = 'LONG' if trend_4h > 0 else 'SHORT' if trend_4h < 0 else 'NEUTRAL'
    TREND_MIN_MOVE = 10  # minimum price movement (in points) to consider it a trend

    # Case 1: Roche Limit + Stable System → FORCE REVERSION
    if abs(z) > roche_k and lam < 0:
        reversion_dir = 'SHORT' if z > 0 else 'LONG'

        # HARD RULE: reversion must align with last 4h trend
        if abs(trend_4h) > TREND_MIN_MOVE:
            if reversion_dir != trend_4h_dir:
                return None, f'roche_against_4h (z={z:.1f}, revert={reversion_dir}, 4h={trend_4h_dir} {trend_4h:+.0f}pts)'

        # Wick confirmation: high wick = rejection already happening
        if wick > 0.5:
            return reversion_dir, f'roche_reversion_confirmed (z={z:.1f}, lam={lam:.2f}, wick={wick:.0%})'

        return reversion_dir, f'roche_reversion (z={z:.1f}, lam={lam:.2f})'

    # Case 2: Chaotic Expansion → FORCE TREND
    if lam > 0 and abs(z) > 1.0:
        trend_dir = 'LONG' if z > 0 else 'SHORT'

        # HARD RULE: trend must align with last 4h
        if abs(trend_4h) > TREND_MIN_MOVE:
            if trend_dir != trend_4h_dir:
                return None, f'trend_against_4h (z={z:.1f}, trend={trend_dir}, 4h={trend_4h_dir})'

        # Macro must agree for trend trades
        if macro_dir != 0 and np.sign(z) != macro_dir:
            return None, f'trend_blocked_by_macro (z={z:.1f}, macro={macro_dir})'

        # Fractal check: micro must have power relative to macro
        if fractal_r < MICRO_MACRO_RATIO:
            return None, f'trend_no_micro_power (ratio={fractal_r:.2f})'

        return trend_dir, f'chaos_trend (z={z:.1f}, lam={lam:.2f})'

    # Case 3: No signal — inside the gravity well, stable
    return None, f'no_signal (z={z:.1f}, lam={lam:.2f})'


def main():
    parser = argparse.ArgumentParser(description='Nightmare Protocol Forward Pass')
    parser.add_argument('--months', default=None, help='Comma-separated YYYY-MM')
    parser.add_argument('--oos-start', default='2026-02', help='OOS boundary')
    parser.add_argument('--roche', type=float, default=ROCHE_K, help='Roche Limit k')
    parser.add_argument('--half-cycle', type=int, default=HALF_CYCLE, help='Hold duration (bars)')
    parser.add_argument('--plot-date', default=None, help='Plot single day')
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"NIGHTMARE PROTOCOL FORWARD PASS")
    print(f"{'='*60}")
    print(f"  Roche Limit: z > {args.roche}")
    print(f"  Half cycle: {args.half_cycle} bars ({args.half_cycle} min at 1m)")
    print(f"  Logic: z > roche AND lambda < 0 -> REVERT")
    print(f"         lambda > 0 AND z > 1.0 -> TREND")

    # Load multi-TF data
    sfe = StatisticalFieldEngine()

    # 1m data
    files_1m = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
    if args.months:
        months = args.months.split(',')
        files_1m = [f for f in files_1m
                     if os.path.basename(f)[:7].replace('_', '-') in months]

    # 1h and 1D — load once, index by timestamp
    print("  Loading macro TFs...")
    df_1h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ATLAS_ROOT, '1h', '*.parquet')))],
                       ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    states_1h = sfe.batch_compute_states(df_1h)
    feats_1h_all = extract_features_13d(states_1h, df_1h)
    ts_1h = df_1h['timestamp'].values
    del states_1h; gc.collect()

    df_1d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(ATLAS_ROOT, '1D', '*.parquet')))],
                       ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    states_1d = sfe.batch_compute_states(df_1d)
    feats_1d_all = extract_features_13d(states_1d, df_1d)
    ts_1d = df_1d['timestamp'].values
    del states_1d; gc.collect()
    print(f"  1h: {len(df_1h)} bars, 1D: {len(df_1d)} bars")

    def get_macro_feats(ts_target, ts_arr, feats_arr):
        """Find the last completed bar at or before ts_target."""
        idx = np.searchsorted(ts_arr, ts_target, side='right') - 1
        if idx >= 0 and idx < len(feats_arr):
            return feats_arr[idx]
        return None

    # Split files
    oos_boundary = args.oos_start.replace('-', '_')
    is_files = [f for f in files_1m if os.path.basename(f)[:7] < oos_boundary]
    oos_files = [f for f in files_1m if os.path.basename(f)[:7] >= oos_boundary]
    print(f"  Files: {len(is_files)} IS, {len(oos_files)} OOS")

    for mode, files in [('IS', is_files), ('OOS', oos_files)]:
        if not files:
            continue

        print(f"\n{'='*60}")
        print(f"  {mode} FORWARD PASS")
        print(f"{'='*60}")

        trades = []
        daily_pnl = {}
        in_position = False
        direction = None
        entry_price = 0.0
        entry_bar = 0
        entry_ts = 0.0
        entry_reason = ''
        entry_z = 0.0
        hold_remaining = 0

        # For plotting
        all_ts = []
        all_prices = []
        all_z = []
        all_lam = []
        all_signals = []

        for fpath in tqdm(files, desc=f"  {mode}", unit='file'):
            df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
            if len(df) < 30:
                continue

            states = sfe.batch_compute_states(df)
            feats = extract_features_13d(states, df)
            del states; gc.collect()

            closes = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            timestamps = df['timestamp'].values
            n = len(df)

            for i in range(10, n):
                price = closes[i]
                ts = timestamps[i]
                day_str = datetime.utcfromtimestamp(int(ts)).strftime('%Y-%m-%d')

                # Get macro features
                feats_1h = get_macro_feats(ts, ts_1h, feats_1h_all)
                feats_1d = get_macro_feats(ts, ts_1d, feats_1d_all)

                # Compute nightmare state
                ns = compute_nightmare_state(feats[i], feats_1h, feats_1d, i, feats, closes)

                # Store for plotting
                all_ts.append(ts)
                all_prices.append(price)
                all_z.append(ns['z_fit'])
                all_lam.append(ns['lam'])

                # Day boundary: force close
                if i > 10:
                    prev_day = datetime.utcfromtimestamp(int(timestamps[i-1])).strftime('%Y-%m-%d')
                    if day_str != prev_day and in_position:
                        pnl_ticks = (price - entry_price) / TICK if direction == 'LONG' else (entry_price - price) / TICK
                        pnl = pnl_ticks * TICK_VALUE
                        trades.append({
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': price, 'pnl': pnl,
                            'pnl_ticks': pnl_ticks,
                            'result': 'WIN' if pnl > 0 else 'LOSS',
                            'exit_reason': 'end_of_day',
                            'entry_reason': entry_reason,
                            'bars_held': i - entry_bar,
                            'day': prev_day,
                        })
                        if prev_day not in daily_pnl:
                            daily_pnl[prev_day] = {'pnl': 0, 'trades': 0, 'wins': 0}
                        daily_pnl[prev_day]['pnl'] += pnl
                        daily_pnl[prev_day]['trades'] += 1
                        if pnl > 0: daily_pnl[prev_day]['wins'] += 1
                        in_position = False

                # Position management — REVERSE NIGHTMARE EXIT
                if in_position:
                    hold_remaining -= 1
                    pnl_ticks = (price - entry_price) / TICK if direction == 'LONG' else (entry_price - price) / TICK
                    mae = (entry_price - lows[i]) / TICK if direction == 'LONG' else (highs[i] - entry_price) / TICK

                    should_exit = False
                    exit_reason = ''

                    # 1. Z crossed zero — particle reached center of gravity well
                    #    Reversion complete. Take profit.
                    if abs(ns['z_fit']) < 0.5 and hold_remaining < args.half_cycle - 1:
                        should_exit = True
                        exit_reason = 'mean_reached'

                    # 2. Lambda flipped — regime changed from stable to chaotic
                    #    Reversion bet is dead. Particle is escaping.
                    elif ns['lam'] > 0.3 and hold_remaining < args.half_cycle - 1:
                        should_exit = True
                        exit_reason = 'lambda_flip'

                    # 3. (REMOVED: z_expanding — exits on normal wobble, -$28K damage)

                    # 4. Macro reversed against us — higher TF force changed
                    elif ns['macro_direction'] != 0:
                        if direction == 'LONG' and ns['macro_direction'] < 0 and ns['dmi_1d'] < -5:
                            if hold_remaining < args.half_cycle - 1:
                                should_exit = True
                                exit_reason = 'macro_against'
                        elif direction == 'SHORT' and ns['macro_direction'] > 0 and ns['dmi_1d'] > 5:
                            if hold_remaining < args.half_cycle - 1:
                                should_exit = True
                                exit_reason = 'macro_against'

                    # 5. Half cycle expired — time's up, take what you have
                    if not should_exit and hold_remaining <= 0:
                        should_exit = True
                        exit_reason = 'half_cycle'

                    # 6. Catastrophic SL — absolute backstop ($100 = 200 ticks)
                    if mae >= 200:
                        should_exit = True
                        exit_reason = 'catastrophic_sl'

                    if should_exit:
                        pnl = pnl_ticks * TICK_VALUE
                        trades.append({
                            'direction': direction, 'entry_price': entry_price,
                            'exit_price': price, 'pnl': pnl,
                            'pnl_ticks': pnl_ticks,
                            'result': 'WIN' if pnl > 0 else 'LOSS',
                            'exit_reason': exit_reason,
                            'entry_reason': entry_reason,
                            'bars_held': i - entry_bar,
                            'day': day_str,
                        })
                        if day_str not in daily_pnl:
                            daily_pnl[day_str] = {'pnl': 0, 'trades': 0, 'wins': 0}
                        daily_pnl[day_str]['pnl'] += pnl
                        daily_pnl[day_str]['trades'] += 1
                        if pnl > 0: daily_pnl[day_str]['wins'] += 1
                        in_position = False

                # Entry evaluation
                if not in_position:
                    signal, reason = nightmare_decision(ns, roche_k=args.roche)
                    all_signals.append(signal)

                    if signal is not None:
                        in_position = True
                        direction = signal
                        entry_price = price
                        entry_bar = i
                        entry_ts = ts
                        entry_reason = reason
                        entry_z = ns['z_fit']
                        hold_remaining = args.half_cycle
                else:
                    all_signals.append(None)

            del df, feats; gc.collect()

        # Report
        n_trades = len(trades)
        if n_trades == 0:
            print(f"  {mode}: 0 trades")
            continue

        total_pnl = sum(t['pnl'] for t in trades)
        n_wins = sum(1 for t in trades if t['result'] == 'WIN')
        wr = n_wins / n_trades * 100
        n_days = len(daily_pnl)
        per_day = total_pnl / n_days if n_days else 0

        longs = [t for t in trades if t['direction'] == 'LONG']
        shorts = [t for t in trades if t['direction'] == 'SHORT']

        print(f"\n  {'='*50}")
        print(f"  {mode} RESULTS")
        print(f"  {'='*50}")
        print(f"  Trades:    {n_trades}")
        print(f"  Win Rate:  {wr:.1f}%")
        print(f"  Total PnL: ${total_pnl:,.2f}")
        print(f"  Per Day:   ${per_day:,.2f}")
        print(f"  Avg/trade: ${total_pnl/n_trades:,.2f}")
        print(f"  Direction: LONG={len(longs)} SHORT={len(shorts)}")
        print(f"  Hold: avg={np.mean([t['bars_held'] for t in trades]):.1f} bars")

        # Exit reasons
        exit_reasons = {}
        for t in trades:
            r = t['exit_reason']
            if r not in exit_reasons:
                exit_reasons[r] = {'n': 0, 'pnl': 0, 'wins': 0}
            exit_reasons[r]['n'] += 1
            exit_reasons[r]['pnl'] += t['pnl']
            if t['result'] == 'WIN': exit_reasons[r]['wins'] += 1

        print(f"\n  EXIT REASONS:")
        for r, s in sorted(exit_reasons.items(), key=lambda x: -x[1]['n']):
            wr_r = s['wins']/s['n']*100 if s['n'] else 0
            print(f"  {r:<25} {s['n']:>6} {wr_r:>5.1f}% ${s['pnl']:>10,.2f} ${s['pnl']/s['n']:>8,.2f}/tr")

        # Entry reasons
        entry_reasons = {}
        for t in trades:
            r = t['entry_reason'].split(' ')[0]
            if r not in entry_reasons:
                entry_reasons[r] = {'n': 0, 'pnl': 0, 'wins': 0}
            entry_reasons[r]['n'] += 1
            entry_reasons[r]['pnl'] += t['pnl']
            if t['result'] == 'WIN': entry_reasons[r]['wins'] += 1

        print(f"\n  ENTRY REASONS:")
        for r, s in sorted(entry_reasons.items(), key=lambda x: -x[1]['n']):
            wr_r = s['wins']/s['n']*100 if s['n'] else 0
            print(f"  {r:<25} {s['n']:>6} {wr_r:>5.1f}% ${s['pnl']:>10,.2f}")

        # Daily ledger
        print(f"\n  DAILY LEDGER:")
        cumul = 0
        for day in sorted(daily_pnl.keys()):
            d = daily_pnl[day]
            cumul += d['pnl']
            wr_d = d['wins']/d['trades']*100 if d['trades'] else 0
            print(f"  {day} {d['trades']:>5} trades {wr_d:>5.1f}% ${d['pnl']:>10,.2f} cumul=${cumul:>10,.2f}")

        # Save
        os.makedirs(OUT_DIR, exist_ok=True)
        pd.DataFrame(trades).to_csv(os.path.join(OUT_DIR, f'nightmare_{mode.lower()}_trades.csv'), index=False)

    print(f"\n{'='*60}")
    print(f"  Baseline: $1,609/day OOS (TradeCNN)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
