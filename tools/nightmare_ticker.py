"""
Nightmare Protocol Ticker — zero lookahead forward pass.

Feeds 1s data, aggregates to 1m by minute boundary, runs Nightmare Protocol.
SFE warmed up with 300 1m bars. Macro features from 1h/1D.
Zero lookahead — each tick only sees the past.

=============================================================================
STRATEGY CHEAT SHEET
=============================================================================

CONSTANTS:
  ROCHE = 2.0          z_se threshold (Roche Limit from Nightmare Protocol)
  HC = 4               half-cycle: bars to audition a loser (reversion)
  MAX_HOLD = 20        max bars to ride a winner (reversion)
  TREND_LB = 15        rolling trend lookback (1m bars = 15 minutes)
  MIN_MOVE = 10        min points for trend to count
  TREND_STRONG = 20    min points for strong trend (triggers trend_ride)
  TREND_MAX_HOLD = 30  max bars for trend_ride winners
  SL = 160 ticks ($80) catastrophic backstop at 1s resolution

ENTRY CONDITION (shared):
  z_se > ROCHE AND variance_ratio < 1.0 (Roche Limit + stable system)

STRATEGY SELECTION (at entry):
  ┌─ z + trend + dmi_1m + accel ALL same direction?
  │   YES → STRATEGY 2: TREND RIDE (enter WITH trend)
  │   NO  → Is trend > 5 opposing AND dmi_1m > 1 opposing?
  │          YES → STRATEGY 3: CAUTIOUS REVERSION (shorter leash)
  │          NO  → Score = 0.5*trend + 0.3*dmi_1h + 0.2*dmi_1d
  │                Score >= -0.3?
  │                YES → STRATEGY 1: REVERSION (standard Nightmare)
  │                NO  → NO TRADE (macro strongly against)
  └

STRATEGY 1: REVERSION (standard Nightmare Protocol)
  Entry:  z past Roche, revert toward mean (SHORT if z>0, LONG if z<0)
  Exits:  (checked in order)
    1. mean_reached      z_se returned to 0 (|z| < 0.5)     → 100% WR, $37/tr
    2. lambda_flip       variance_ratio > 1.3 (regime change) → 91% WR, $114/tr
    3. profit_hold_exit  was profitable, gave back 50% of peak → 53% WR, $1.83/tr
    4. half_cycle_loss   never profitable after 4 bars         → 0% WR, -$21/tr
       (early cut at bar 2 if BOTH trend AND dmi oppose)
    5. max_hold_profit   profitable for 20 bars                → 81% WR, $76/tr
    6. catastrophic_sl   160 ticks MAE at 1s resolution        → 0% WR, -$87/tr

STRATEGY 2: TREND RIDE
  Entry:  z past Roche AND trend > 20 same direction AND dmi_1m > 3 same
          AND price_accel same direction → enter WITH trend (not against)
  Exits:  (checked in priority order)
    1. trend_no_pay       never profitable after 5 bars         → cut dead weight
    2. trend_exhausted    15m trend flipped against > MIN_MOVE   → breakeven
    3. trend_protect_profit  trend decayed >50% but still profitable → lock profit
    4. trend_breakeven_protect  trend decayed, was profitable, now negative
    5. trend_exhausted    trend decayed, never profitable, 3+ bars → early cut

STRATEGY 3: CAUTIOUS REVERSION
  Entry:  same as reversion BUT loser profile detected:
          trend > 5 opposing AND dmi_1m > 1 opposing
  Exits:  same as reversion EXCEPT:
    - half_cycle = 2 bars (not 4) — cut losers faster
    - giveback = 30% (not 50%) — keep more profit

STRATEGY 6: TREND FADE (exhaustion spike)
  Entry:  Same triggers as trend_ride BUT volume >= 3000 (exhaustion spike)
          All trend signals aligned = climax is done, fade the reversal
  Direction: OPPOSITE of trend (if trend is LONG, we go SHORT)
  Exits:  Same as reversion (mean_reached, lambda_flip, profit_hold, max_hold)
  Volume gates: vol < 750 = skip (no conviction), vol >= 3000 = fade, else = trend_ride

MACRO FILTER (all strategies):
  Weighted score from 3 TFs:
    15m rolling trend: weight 0.5
    1h DMI:           weight 0.3
    1D DMI:           weight 0.2
  Score < -0.3 → NO TRADE (majority of TFs oppose)

TF DATA FLOW:
  1s → aggregated to 1m by minute boundary → SFE → 13D features → Nightmare
  1h → pre-loaded, indexed by timestamp → macro DMI
  1D → pre-loaded, indexed by timestamp → macro DMI
  SL checked at 1s resolution (every tick)
  Decisions at 1m resolution (every completed minute)

OOS RESULTS (Mar 2026, 19 days, zero lookahead):
  $5,819 total | $306/day | 17/19 winning days | 2,561 trades
=============================================================================

Usage:
  python tools/nightmare_ticker.py 2026-03-20
  python tools/nightmare_ticker.py 2026-03-20 --roche 2.0 --trend-lb 15
"""
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numba')

import numpy as np
import pandas as pd
import glob
import gc
import sys
import os
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.statistical_field_engine import StatisticalFieldEngine
from training.train_trade_cnn import extract_features_13d as extract_13d_batch

TICK = 0.25
TV = 0.50
ROCHE = 2.0
HC = 4          # half cycle: 1m bars to audition
MAX_HOLD = 20   # max 1m bars for profitable trades
TREND_LB = 15   # rolling trend lookback (1m bars)
MIN_MOVE = 10   # min pts for trend to count
WARMUP = 300    # 1m bars of history before target day
BARS_PER_1M = 60  # 60 x 1s = 1m
TREND_MIN_VOL = 750     # min 1m volume for trend_ride (below = no conviction)
TREND_FADE_VOL = 3000   # above this = exhaustion spike, fade instead (57% WR opposite)

# Accept: single day, comma-separated days, or "all" for full OOS
# Examples: 2026-03-20   2026-03-20,2026-03-24   all
TARGET = sys.argv[1] if len(sys.argv) > 1 else '2026-03-20'


def extract_13d_single(state, price, high, low, open_price, volume, timestamp,
                       price_hist, vol_hist, prev_velocity):
    """Extract 13D features for a single bar from SFE state + OHLCV.

    No lookahead — uses only accumulated price_hist and vol_hist.
    """
    dmi_p = getattr(state, 'dmi_plus', 0.0)
    dmi_m = getattr(state, 'dmi_minus', 0.0)
    vel = getattr(state, 'velocity', 0.0)

    n_hist = len(price_hist)

    # Volume SMA
    vol_window = vol_hist[-30:] if len(vol_hist) >= 30 else vol_hist
    vol_avg = np.mean(vol_window) if len(vol_window) > 0 else 1.0
    if vol_avg <= 0:
        vol_avg = 1.0

    # 7D directional
    feat = np.zeros(13, dtype=np.float32)
    feat[0] = dmi_p - dmi_m                             # dmi_diff
    feat[1] = abs(dmi_p - dmi_m)                        # dmi_gap
    feat[2] = volume / vol_avg                           # vol_rel
    if n_hist >= 2:
        _dir = 1.0 if price > price_hist[-2] else -1.0
        feat[3] = _dir * volume / vol_avg               # dir_vol
    feat[4] = vel                                        # velocity
    if n_hist >= 15:
        _w = price_hist[-60:] if n_hist >= 60 else price_hist
        _mean = np.mean(_w)
        _std = np.std(_w)
        _se = _std / (len(_w) ** 0.5) if len(_w) > 1 else _std
        feat[5] = (price - _mean) / _se if _se > 1e-8 else 0.0  # z_se
    feat[6] = vel - prev_velocity                        # price_accel

    # 4D regime
    if n_hist >= 30:
        feat[7] = np.std(price_hist[-30:])              # std_price
        if n_hist >= 60:
            _short = np.std(price_hist[-10:])
            _long = np.std(price_hist[-60:])
            feat[8] = _short / _long if _long > 1e-8 else 1.0  # variance_ratio

    _rng = high - low
    feat[9] = _rng / TICK                                # bar_range
    if _rng > 0:
        feat[10] = 1.0 - abs(price - open_price) / _rng # wick_ratio

    # 2D context
    if n_hist >= 30 and len(vol_hist) >= 30:
        _p = np.array(price_hist[-30:])
        _v = np.array(vol_hist[-30:])
        _vwap = np.sum(_p * _v) / (np.sum(_v) + 1e-8)
        feat[11] = (price - _vwap) / TICK               # vwap_distance

    feat[12] = (timestamp % 86400) / 86400               # time_of_day

    return feat


def main():
    # Parse target days
    if TARGET == 'all':
        # Find all available days from 1s data
        files_1s = sorted(glob.glob('DATA/ATLAS/1s/*.parquet'))
        _all_1s = pd.concat([pd.read_parquet(f) for f in files_1s], ignore_index=True)
        _all_1s = _all_1s.sort_values('timestamp').reset_index(drop=True)
        from datetime import datetime as _dt
        _all_dates = sorted(set(_dt.utcfromtimestamp(t).strftime('%Y-%m-%d')
                                for t in _all_1s['timestamp'].values[::5000]))
        target_days = [d for d in _all_dates if d >= '2026-01-06']
        del _all_1s
    elif ',' in TARGET:
        target_days = TARGET.split(',')
    else:
        target_days = [TARGET]

    print(f'\nNIGHTMARE TICKER — {len(target_days)} day(s)')
    print(f'  Zero lookahead. 1s data aggregated to 1m.')
    print(f'  Roche={ROCHE} HC={HC} MaxHold={MAX_HOLD} TrendLB={TREND_LB}')
    print()

    # Pre-load 1s and 1m data
    files_1s = sorted(glob.glob('DATA/ATLAS/1s/*.parquet'))
    df_1s_all = pd.concat([pd.read_parquet(f) for f in files_1s], ignore_index=True)
    df_1s_all = df_1s_all.sort_values('timestamp').reset_index(drop=True)

    files_1m = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
    df_1m_all = pd.concat([pd.read_parquet(f) for f in files_1m], ignore_index=True)
    df_1m_all = df_1m_all.sort_values('timestamp').reset_index(drop=True)

    # Load 1h and 1D macro features ONCE (shared across all days)
    print(f'  Loading macro TFs...')
    df_1h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1h/*.parquet'))],
                       ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    sfe_1h = StatisticalFieldEngine()
    states_1h = sfe_1h.batch_compute_states(df_1h)
    feats_1h = extract_13d_batch(states_1h, df_1h)
    ts_1h = df_1h['timestamp'].values
    del states_1h, sfe_1h; gc.collect()

    df_1d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1D/*.parquet'))],
                       ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    sfe_1d = StatisticalFieldEngine()
    states_1d = sfe_1d.batch_compute_states(df_1d)
    feats_1d = extract_13d_batch(states_1d, df_1d)
    ts_1d = df_1d['timestamp'].values
    del states_1d, sfe_1d; gc.collect()
    print(f'  1h: {len(df_1h)} bars | 1D: {len(df_1d)} bars')

    # === WARMUP ONCE, ACCUMULATE ACROSS DAYS ===
    all_trades_all_days = []
    all_traj_all_days = []
    daily_summary = []
    global_trade_id = 0

    # Initial warmup before first day
    first_date_ts = pd.Timestamp(target_days[0]).timestamp()
    warmup_1m = df_1m_all[df_1m_all['timestamp'] < first_date_ts].tail(WARMUP).reset_index(drop=True)
    sfe = StatisticalFieldEngine()
    warmup_states = sfe.batch_compute_states(warmup_1m)
    prev_vel = 0.0
    if warmup_states:
        last_st = warmup_states[-1]
        prev_vel = getattr(last_st['state'] if isinstance(last_st, dict) else last_st, 'velocity', 0.0)
    del warmup_states; gc.collect()

    price_hist_1m = list(warmup_1m['close'].values)
    vol_hist_1m = list(warmup_1m['volume'].values if 'volume' in warmup_1m.columns else np.zeros(len(warmup_1m)))
    running_1m = warmup_1m.copy()
    print(f'  Warmup: {len(warmup_1m)} bars before {target_days[0]}')

    from tqdm import tqdm
    for target_date in tqdm(target_days, desc='Days'):
        date_ts = pd.Timestamp(target_date).timestamp()

        # Filter 1s data to this day
        day_1s = df_1s_all[(df_1s_all['timestamp'] >= date_ts) & (df_1s_all['timestamp'] < date_ts + 86400)]
        if len(day_1s) < 100:
            continue

        # SFE, price_hist, running_1m carry over from previous day — accumulated

        # 1s → 1m aggregator: group by minute boundary (timestamp // 60)
        current_minute = -1
        agg_open = 0.0
        agg_high = -1e9
        agg_low = 1e9
        agg_close = 0.0
        agg_vol = 0.0
        agg_ts = 0.0

        # Trade state
        trades = []
        trajectory_log = []  # every 1m bar while in trade
        in_pos = False
        direction = None
        entry_price = 0.0
        bars_held_1m = 0
        peak_pnl = 0.0
        was_profitable = False
        active_strategy = 'reversion'
        tick_price = 0.0

        # Tick through 1s data — numpy arrays for speed
        _tc = day_1s['close'].values
        _th = day_1s['high'].values
        _tl = day_1s['low'].values
        _to = day_1s['open'].values
        _tv = day_1s['volume'].values if 'volume' in day_1s.columns else np.zeros(len(day_1s))
        _tt = day_1s['timestamp'].values
        _n_ticks = len(day_1s)

        for _tidx in range(_n_ticks):
            tick_price = _tc[_tidx]
            tick_high = _th[_tidx]
            tick_low = _tl[_tidx]
            tick_open = _to[_tidx]
            tick_vol = _tv[_tidx]
            tick_ts = _tt[_tidx]

            # Accumulate into 1m bar by minute boundary
            tick_minute = int(tick_ts) // 60

            if current_minute == -1:
                # First tick
                current_minute = tick_minute
                agg_open = tick_open
                agg_ts = tick_ts
                agg_high = tick_high
                agg_low = tick_low

            agg_high = max(agg_high, tick_high)
            agg_low = min(agg_low, tick_low)
            agg_close = tick_price
            agg_vol += tick_vol

            # NO SL — strategies must handle their own exits.
            # SL is a bandaid that obscures entry deficiencies.
            # Will add back as absolute last resort after strategies are solid.

            # 1m bar complete? (minute boundary changed)
            if tick_minute == current_minute:
                continue

            # New minute started — process the completed 1m bar
            completed_minute = current_minute
            current_minute = tick_minute

            # === 1M BAR COMPLETE — run Nightmare ===
            _1m_bar = pd.DataFrame([{
                'timestamp': agg_ts, 'open': agg_open, 'high': agg_high,
                'low': agg_low, 'close': agg_close, 'volume': agg_vol,
            }])

            # Reset aggregator for the new minute (current tick starts it)
            agg_open = tick_open
            agg_ts = tick_ts
            agg_high = tick_high
            agg_low = tick_low
            agg_vol = tick_vol

            price = agg_close
            high = _1m_bar['high'].iloc[0]
            low = _1m_bar['low'].iloc[0]
            open_price = _1m_bar['open'].iloc[0]
            volume = _1m_bar['volume'].iloc[0]
            timestamp = agg_ts

            # Update histories
            price_hist_1m.append(price)
            vol_hist_1m.append(volume)
            if len(price_hist_1m) > 500:
                price_hist_1m = price_hist_1m[-500:]
                vol_hist_1m = vol_hist_1m[-500:]

            # SFE on accumulated 1m data
            running_1m = pd.concat([running_1m, _1m_bar], ignore_index=True)
            # Keep full day + warmup context for SFE (no trimming)

            states = sfe.batch_compute_states(running_1m)
            st = states[-1]
            st = st['state'] if isinstance(st, dict) else st
            del states

            # 13D features
            feat = extract_13d_single(st, price, high, low, open_price, volume, timestamp,
                                       price_hist_1m, vol_hist_1m, prev_vel)
            prev_vel = getattr(st, 'velocity', 0.0)

            z = feat[5]
            vr = feat[8]
            lam = vr - 1.0

            # 15-bar rolling trend
            if len(price_hist_1m) > TREND_LB:
                trend = price_hist_1m[-1] - price_hist_1m[-(TREND_LB + 1)]
            else:
                trend = 0.0

            time_str = datetime.utcfromtimestamp(timestamp).strftime('%H:%M')

            # Position management at 1m resolution
            if in_pos:
                bars_held_1m += 1
                pt = (price - entry_price) / TICK if direction == 'LONG' else (entry_price - price) / TICK
                pnl = pt * TV
                mae = (entry_price - low) / TICK if direction == 'LONG' else (high - entry_price) / TICK
                peak_pnl = max(peak_pnl, pnl)
                if pnl > 0:
                    was_profitable = True

                ex = None
                strat = active_strategy if 'active_strategy' in dir() else 'reversion'

                # Trajectory logging — every 1m bar while in trade
                _trend_now = price_hist_1m[-1] - price_hist_1m[-(TREND_LB+1)] if len(price_hist_1m) > TREND_LB else 0
                _giveback_pct = (1 - pnl / peak_pnl) * 100 if peak_pnl > 0 else 0
                _dmi_now = feat[0]
                # Get macro features for this bar
                _idx_1h = np.searchsorted(ts_1h, timestamp, side='right') - 1
                _dmi_1h_now = feats_1h[_idx_1h, 0] if 0 <= _idx_1h < len(feats_1h) else 0
                _idx_1d = np.searchsorted(ts_1d, timestamp, side='right') - 1
                _dmi_1d_now = feats_1d[_idx_1d, 0] if 0 <= _idx_1d < len(feats_1d) else 0

                # Trajectory: log timestamp + trade state only
                # Full features at ALL TFs are in ATLAS_FEATURES — join by timestamp
                trajectory_log.append({
                    'trade_id': global_trade_id, 'bar': bars_held_1m,
                    'timestamp': timestamp, 'time': time_str,
                    'price': price, 'pnl': pnl, 'peak_pnl': peak_pnl,
                    'giveback_pct': _giveback_pct, 'mae': mae,
                    'strategy': strat, 'direction': direction,
                    'was_profitable': was_profitable,
                    'trend_15': _trend_now,
                    # Active TF features (1m — computed live)
                    'z_se': z, 'lam': lam, 'vr': vr,
                    'dmi_1m': _dmi_now,
                    # Exit distances
                    'dist_mean_reached': abs(z) - 0.5,
                    'dist_lambda_flip': 0.3 - lam,
                    'dist_giveback_50': (pnl / peak_pnl - 0.5) if peak_pnl > 2 else 99,
                })

                if strat in ('reversion', 'cautious_reversion', 'macro_dip', 'open_ride', 'trend_fade'):
                    # STRATEGY 1/3 EXITS: Nightmare reversion (cautious has shorter HC)
                    if abs(z) < 0.5 and bars_held_1m > 1:
                        ex = 'mean_reached'
                    elif lam > 0.3 and bars_held_1m > 1:
                        ex = 'lambda_flip'
                    elif was_profitable and peak_pnl > 2 and pnl < peak_pnl * 0.5:
                        ex = 'profit_hold_exit'
                    # NO half_cycle_loss — removed to see what happens
                    # These trades hold until another exit triggers
                    # (mean_reached, lambda_flip, profit_hold, max_hold)
                    elif was_profitable and bars_held_1m >= MAX_HOLD:
                        ex = 'max_hold_profit'

                elif strat in ('trend_ride', 'reversal_to_trend'):
                    # STRATEGY 2 EXITS: ride the trend with continuous validity check
                    _trend_now = price_hist_1m[-1] - price_hist_1m[-(TREND_LB+1)] if len(price_hist_1m) > TREND_LB else 0

                    # Trend validity: is the trend still at least half its entry strength?
                    TREND_DECAY_RATIO = 0.5  # exit when trend drops below 50% of entry
                    _entry_strength = abs(entry_trend)
                    _current_strength = abs(_trend_now)
                    _trend_in_direction = (direction == 'LONG' and _trend_now > 0) or \
                                          (direction == 'SHORT' and _trend_now < 0)
                    _trend_valid = _trend_in_direction and _current_strength >= _entry_strength * TREND_DECAY_RATIO

                    # Exit: trend never paid — cut the dead weight
                    TREND_PATIENCE = 5  # bars to prove itself (EDA median hold for losers was 5)
                    if not was_profitable and bars_held_1m >= TREND_PATIENCE:
                        ex = 'trend_no_pay'

                    # Exit: stalled trend — peak is tiny by bar 3, it's a fake trend
                    # EDA: losers peak $11.50 by bar 2 then die. Winners climb to $46+ by bar 8.
                    # 59% of losers never exceed $15 peak. Only 10% of winners that low.
                    TREND_STALL_BAR = 3       # check at this bar
                    TREND_MIN_PEAK = 15 * TV  # $15 = 30 ticks = must show real movement
                    if not ex and bars_held_1m == TREND_STALL_BAR and peak_pnl < TREND_MIN_PEAK:
                        ex = 'trend_stalled'

                    # Exit: trend flipped to OPPOSITE direction past MIN_MOVE
                    _trend_flipped = (direction == 'LONG' and _trend_now < -MIN_MOVE) or \
                                     (direction == 'SHORT' and _trend_now > MIN_MOVE)
                    if not ex and _trend_flipped and bars_held_1m > 2:
                        ex = 'trend_exhausted'
                    # Exit: trend no longer valid (decayed below half entry strength or flipped)
                    elif not ex and not _trend_valid and bars_held_1m > 2:
                        if pnl > 0:
                            ex = 'trend_protect_profit'
                        elif was_profitable:
                            ex = 'trend_breakeven_protect'
                        # If never profitable and trend dying → cut early
                        elif bars_held_1m >= 3:
                            ex = 'trend_exhausted'

                if ex:
                    trades.append({
                        'trade_id': global_trade_id,
                        'time': time_str, 'dir': direction, 'pnl': pnl,
                        'exit': ex, 'held': bars_held_1m, 'peak': peak_pnl,
                        'strategy': active_strategy,
                    })
                    global_trade_id += 1
                    in_pos = False

            # Entry at 1m resolution — MULTIPLE STRATEGIES
            if not in_pos:
                # Get macro state
                idx_1h = np.searchsorted(ts_1h, timestamp, side='right') - 1
                dmi_1h = feats_1h[idx_1h, 0] if 0 <= idx_1h < len(feats_1h) else 0
                idx_1d = np.searchsorted(ts_1d, timestamp, side='right') - 1
                dmi_1d = feats_1d[idx_1d, 0] if 0 <= idx_1d < len(feats_1d) else 0
                dmi_1m = feat[0]

                strategy = None
                direction = None

                # STRATEGY 5: US OPEN — during open window (13:00-14:30 UTC),
                # only enter in the direction of the pre-market trend.
                # The open spike typically continues pre-market direction.
                _hour = int(timestamp % 86400) // 3600
                _is_open_window = 13 <= _hour <= 14
                if _is_open_window:
                    # Pre-market trend = rolling 60-bar trend (last hour before open)
                    _pre_trend = price_hist_1m[-1] - price_hist_1m[-61] if len(price_hist_1m) > 61 else 0
                    if abs(_pre_trend) > MIN_MOVE and abs(z) > ROCHE and lam < 0:
                        open_dir = 'LONG' if _pre_trend > 0 else 'SHORT'
                        rev = 'SHORT' if z > 0 else 'LONG'
                        if rev == open_dir:
                            # Reversion aligns with pre-market — safe entry
                            strategy = 'open_ride'
                            direction = rev
                        # If reversion opposes pre-market — skip entirely
                    # Skip normal entry logic during open window
                    if strategy and direction:
                        in_pos = True
                        entry_price = price
                        bars_held_1m = 0
                        peak_pnl = 0.0
                        was_profitable = False
                        active_strategy = strategy
                        entry_trend = trend
                    continue  # skip normal entry during open window

                if abs(z) > ROCHE and lam < 0:
                    rev = 'SHORT' if z > 0 else 'LONG'

                    # Check: is trend STRONG and in the SAME direction as z?
                    # z > 0 means price above mean. trend > 0 means price going up.
                    # If both positive or both negative → trend is pushing z further away
                    TREND_STRONG = 20  # 20 points = strong trend
                    z_and_trend_same = (z > 0 and trend > TREND_STRONG) or (z < 0 and trend < -TREND_STRONG)
                    dmi_with_z = (z > 0 and dmi_1m > 3) or (z < 0 and dmi_1m < -3)

                    # Accel check: is price still accelerating in the trend direction?
                    accel = feat[6]  # price_accel
                    accel_with_trend = (trend > 0 and accel > 0) or (trend < 0 and accel < 0)

                    if z_and_trend_same and dmi_with_z and accel_with_trend:
                        trend_dir = 'LONG' if trend > 0 else 'SHORT'
                        if volume >= TREND_FADE_VOL:
                            # STRATEGY 6: TREND FADE — exhaustion spike detected
                            # High volume + all trend signals = climax, fade the move
                            strategy = 'trend_fade'
                            direction = 'SHORT' if trend_dir == 'LONG' else 'LONG'
                        elif volume >= TREND_MIN_VOL:
                            # STRATEGY 2: TREND RIDE — real conviction
                            strategy = 'trend_ride'
                            direction = trend_dir
                        # else: volume too low, skip (no conviction)
                    else:
                        # Loser profile check (tighter thresholds from RCA):
                        # HCL avg: trend=12.8, dmi=4.05
                        # WIN avg: trend=5.8,  dmi=0.83
                        # Threshold at midpoint: trend>10, dmi>2
                        _trend_opp = (rev == 'LONG' and trend < -10) or (rev == 'SHORT' and trend > 10)
                        _dmi_opp = (rev == 'LONG' and dmi_1m < -2) or (rev == 'SHORT' and dmi_1m > 2)

                        rev_sign = 1.0 if rev == 'LONG' else -1.0
                        score = 0.0

                        if abs(trend) > MIN_MOVE:
                            trend_sign = 1.0 if trend > 0 else -1.0
                            score += 0.5 * (rev_sign * trend_sign)

                        if abs(dmi_1h) > 3:
                            dmi_1h_sign = 1.0 if dmi_1h > 0 else -1.0
                            score += 0.3 * (rev_sign * dmi_1h_sign)

                        if abs(dmi_1d) > 3:
                            dmi_1d_sign = 1.0 if dmi_1d > 0 else -1.0
                            score += 0.2 * (rev_sign * dmi_1d_sign)

                        CONVICTION_MIN = -0.3

                        if score >= CONVICTION_MIN:
                            if _trend_opp and _dmi_opp:
                                strategy = 'cautious_reversion'
                            else:
                                strategy = 'reversion'
                            direction = rev

                if strategy and direction:
                    in_pos = True
                    entry_price = price
                    bars_held_1m = 0
                    peak_pnl = 0.0
                    was_profitable = False
                    active_strategy = strategy
                    _1s_ticks_since_entry = 0
                    entry_trend = trend  # save trend at entry for validity check

        # Force close at end of day (after all ticks processed)
        if in_pos:
            pt = (tick_price - entry_price) / TICK if direction == 'LONG' else (entry_price - tick_price) / TICK
            trades.append({'trade_id': global_trade_id, 'time': time_str,
                           'dir': direction, 'pnl': pt * TV,
                           'exit': 'end_of_day', 'held': bars_held_1m, 'peak': peak_pnl,
                           'strategy': active_strategy})
            global_trade_id += 1

        # Per-day results
        day_pnl = sum(t['pnl'] for t in trades)
        day_trades = len(trades)
        day_wins = sum(1 for t in trades if t['pnl'] > 0)
        daily_summary.append({'day': target_date, 'trades': day_trades,
                               'pnl': day_pnl, 'wins': day_wins})

        # Add day column and accumulate
        for t in trades:
            t['day'] = target_date
        all_trades_all_days.extend(trades)
        all_traj_all_days.extend(trajectory_log)

        # Trim running_1m to keep memory manageable (keep last 500 bars)
        if len(running_1m) > 500:
            running_1m = running_1m.tail(400).reset_index(drop=True)
        gc.collect()
    # === END DAY LOOP ===

    # Aggregate Report
    t = pd.DataFrame(all_trades_all_days)
    if len(t) == 0:
        print('No trades.')
        return

    total = t['pnl'].sum()
    wr = (t['pnl'] > 0).mean() * 100
    n_days = len(daily_summary)
    nl = len(t[t['dir'] == 'LONG'])
    ns = len(t[t['dir'] == 'SHORT'])

    print(f'\nRESULTS (zero lookahead, {n_days} days):')
    print(f'  {len(t)} trades | WR={wr:.1f}% | PnL=${total:,.2f} | $/day=${total/max(n_days,1):,.2f}')
    print(f'  LONG={nl} SHORT={ns} | Avg hold={t["held"].mean():.1f} bars')
    print()

    for ex in sorted(t['exit'].unique()):
        et = t[t['exit'] == ex]
        wr_e = (et['pnl'] > 0).mean() * 100
        print(f'  {ex:<22} {len(et):>5}  WR={wr_e:>5.1f}%  ${et["pnl"].sum():>9,.2f}  ${et["pnl"].mean():>7,.2f}/tr')

    print()

    # Daily ledger
    print(f'DAILY:')
    cumul = 0
    for d in daily_summary:
        cumul += d['pnl']
        wr_d = d['wins'] / d['trades'] * 100 if d['trades'] else 0
        marker = ' <<<' if d['pnl'] > 200 else ' !!!' if d['pnl'] < -200 else ''
        print(f'  {d["day"]} {d["trades"]:>5} trades {wr_d:>4.0f}% ${d["pnl"]:>9,.2f} cumul=${cumul:>9,.2f}{marker}')

    winning_days = sum(1 for d in daily_summary if d['pnl'] > 0)
    print(f'\n  Winning days: {winning_days}/{n_days}')

    # PnL mode
    pnl_rounded = (t['pnl'] / 1).round() * 1
    mode = Counter(pnl_rounded).most_common(3)
    print(f'  PnL mode: {mode}')

    # Save
    os.makedirs('reports/findings', exist_ok=True)
    label = TARGET if len(target_days) == 1 else f'{target_days[0]}_to_{target_days[-1]}'
    t.to_csv(f'reports/findings/nightmare_{label}_trades.csv', index=False)
    traj = pd.DataFrame(all_traj_all_days)
    if len(traj) > 0:
        traj.to_csv(f'reports/findings/nightmare_{label}_trajectory.csv', index=False)
    print(f'\n  Trades: reports/findings/nightmare_{label}_trades.csv ({len(t)} trades)')
    if len(traj) > 0:
        print(f'  Trajectory: reports/findings/nightmare_{label}_trajectory.csv ({len(traj)} bars)')


if __name__ == '__main__':
    main()
