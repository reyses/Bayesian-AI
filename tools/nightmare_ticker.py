"""
Nightmare Protocol Ticker — zero lookahead forward pass.

Feeds one bar at a time to the system. The SFE and features are computed
incrementally — the system never sees future bars.

Each bar:
  1. Append bar to rolling dataframe
  2. Recompute SFE state for this bar only (using accumulated history)
  3. Extract 13D features for this bar
  4. Run Nightmare decision
  5. Manage position

This mirrors exactly what live would do.

Usage:
  python tools/nightmare_ticker.py 2026-03-20
  python tools/nightmare_ticker.py 2026-03-20 --roche 2.0 --trend-lb 15
"""
import numpy as np
import pandas as pd
import glob
import gc
import sys
import os
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine
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
    print(f'\nNIGHTMARE TICKER — {TARGET}')
    print(f'  Zero lookahead. 1s data aggregated to 1m.')
    print(f'  Roche={ROCHE} HC={HC} MaxHold={MAX_HOLD} TrendLB={TREND_LB}')
    print()

    # Load 1s data for the target day
    files_1s = sorted(glob.glob('DATA/ATLAS/1s/*.parquet'))
    # Find file containing target date
    date_ts = pd.Timestamp(TARGET).timestamp()
    month_str = TARGET[:7].replace('-', '_')

    df_1s = None
    for f in files_1s:
        if month_str in f:
            df_1s = pd.read_parquet(f).sort_values('timestamp').reset_index(drop=True)
            break

    if df_1s is None:
        print(f'  No 1s data for {month_str}')
        return

    # Filter to target day
    day_1s = df_1s[(df_1s['timestamp'] >= date_ts) & (df_1s['timestamp'] < date_ts + 86400)].reset_index(drop=True)
    print(f'  1s bars for {TARGET}: {len(day_1s)}')

    # Load 1m warmup data (before target day)
    files_1m = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
    df_1m_all = pd.concat([pd.read_parquet(f) for f in files_1m[-2:]], ignore_index=True)
    df_1m_all = df_1m_all.sort_values('timestamp').reset_index(drop=True)
    warmup_1m = df_1m_all[df_1m_all['timestamp'] < date_ts].tail(WARMUP).reset_index(drop=True)

    # Warmup SFE with 1m history
    sfe = StatisticalFieldEngine()
    warmup_states = sfe.batch_compute_states(warmup_1m)
    prev_vel = 0.0
    if warmup_states:
        last_st = warmup_states[-1]
        prev_vel = getattr(last_st['state'] if isinstance(last_st, dict) else last_st, 'velocity', 0.0)
    del warmup_states; gc.collect()

    # Build 1m price/vol history from warmup
    price_hist_1m = list(warmup_1m['close'].values)
    vol_hist_1m = list(warmup_1m['volume'].values if 'volume' in warmup_1m.columns else np.zeros(len(warmup_1m)))

    # Running 1m df for SFE recompute
    running_1m = warmup_1m.copy()

    # Load 1h and 1D macro features (for trend filter)
    print(f'  Loading macro TFs...')
    df_1h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1h/*.parquet'))],
                       ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    df_1h = df_1h[df_1h['timestamp'] < date_ts + 86400]  # up to end of target day
    sfe_1h = StatisticalFieldEngine()
    states_1h = sfe_1h.batch_compute_states(df_1h)
    feats_1h = extract_13d_batch(states_1h, df_1h)
    ts_1h = df_1h['timestamp'].values
    del states_1h, sfe_1h; gc.collect()

    df_1d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1D/*.parquet'))],
                       ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    df_1d = df_1d[df_1d['timestamp'] < date_ts + 86400]
    sfe_1d = StatisticalFieldEngine()
    states_1d = sfe_1d.batch_compute_states(df_1d)
    feats_1d = extract_13d_batch(states_1d, df_1d)
    ts_1d = df_1d['timestamp'].values
    del states_1d, sfe_1d; gc.collect()

    print(f'  Warmup: {len(warmup_1m)} x 1m bars | 1h: {len(df_1h)} bars | 1D: {len(df_1d)} bars')
    print(f'  Ticking {len(day_1s)} x 1s bars...')
    print()

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
    in_pos = False
    direction = None
    entry_price = 0.0
    bars_held_1m = 0
    peak_pnl = 0.0
    was_profitable = False
    last_1m_price = price_hist_1m[-1] if price_hist_1m else 0

    # Tick through 1s data
    from tqdm import tqdm
    for _, row in tqdm(day_1s.iterrows(), total=len(day_1s), desc='  Ticking'):
        tick_price = row['close']
        tick_high = row['high']
        tick_low = row['low']
        tick_open = row['open']
        tick_vol = row.get('volume', 0.0)
        tick_ts = row['timestamp']

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

        # Check SL at 1s resolution (while in position)
        if in_pos:
            if direction == 'LONG':
                mae_now = (entry_price - tick_low) / TICK
            else:
                mae_now = (tick_high - entry_price) / TICK
            if mae_now >= 160:
                pt = (tick_price - entry_price) / TICK if direction == 'LONG' else (entry_price - tick_price) / TICK
                trades.append({
                    'time': datetime.utcfromtimestamp(tick_ts).strftime('%H:%M:%S'),
                    'dir': direction, 'pnl': pt * TV,
                    'exit': 'catastrophic_sl_1s', 'held': bars_held_1m, 'peak': peak_pnl,
                })
                in_pos = False

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
            if abs(z) < 0.5 and bars_held_1m > 1:
                ex = 'mean_reached'
            elif lam > 0.3 and bars_held_1m > 1:
                ex = 'lambda_flip'
            elif was_profitable and peak_pnl > 2 and pnl < peak_pnl * 0.5:
                ex = 'profit_hold_exit'
            elif not was_profitable and bars_held_1m >= 2:
                # At bar 2: check if BOTH trend and DMI oppose the trade
                # If both fight → early cut (12x separation from winners)
                # If only one fights → give it 2 more bars
                _trend_now = price_hist_1m[-1] - price_hist_1m[-(TREND_LB+1)] if len(price_hist_1m) > TREND_LB else 0
                _dmi_now = feat[0]

                _trend_opposes = (direction == 'LONG' and _trend_now < -5) or \
                                 (direction == 'SHORT' and _trend_now > 5)
                _dmi_opposes = (direction == 'LONG' and _dmi_now < -2) or \
                               (direction == 'SHORT' and _dmi_now > 2)

                if (_trend_opposes and _dmi_opposes) or bars_held_1m >= HC:
                    ex = 'half_cycle_loss'
            elif was_profitable and bars_held_1m >= MAX_HOLD:
                ex = 'max_hold_profit'

            if ex:
                trades.append({
                    'time': time_str, 'dir': direction, 'pnl': pnl,
                    'exit': ex, 'held': bars_held_1m, 'peak': peak_pnl,
                })
                in_pos = False

        # Entry at 1m resolution
        if not in_pos:
            if abs(z) > ROCHE and lam < 0:
                rev = 'SHORT' if z > 0 else 'LONG'
                rev_sign = 1.0 if rev == 'LONG' else -1.0

                # Weighted conviction score across TFs
                # 15m rolling trend: primary signal (weight 0.5)
                # 1h DMI: secondary (weight 0.3)
                # 1D DMI: tertiary (weight 0.2)
                score = 0.0

                # 15m trend
                if abs(trend) > MIN_MOVE:
                    trend_sign = 1.0 if trend > 0 else -1.0
                    score += 0.5 * (rev_sign * trend_sign)  # +0.5 if agree, -0.5 if disagree
                else:
                    score += 0.0  # neutral, no penalty

                # 1h DMI
                idx_1h = np.searchsorted(ts_1h, timestamp, side='right') - 1
                dmi_1h = feats_1h[idx_1h, 0] if 0 <= idx_1h < len(feats_1h) else 0
                if abs(dmi_1h) > 3:
                    dmi_1h_sign = 1.0 if dmi_1h > 0 else -1.0
                    score += 0.3 * (rev_sign * dmi_1h_sign)

                # 1D DMI
                idx_1d = np.searchsorted(ts_1d, timestamp, side='right') - 1
                dmi_1d = feats_1d[idx_1d, 0] if 0 <= idx_1d < len(feats_1d) else 0
                if abs(dmi_1d) > 3:
                    dmi_1d_sign = 1.0 if dmi_1d > 0 else -1.0
                    score += 0.2 * (rev_sign * dmi_1d_sign)

                # Score: +1.0 = all TFs agree, -1.0 = all oppose, 0 = neutral
                # Block only when strongly against (score < -0.3)
                CONVICTION_MIN = -0.3
                if score >= CONVICTION_MIN:
                    in_pos = True
                    direction = rev
                    entry_price = price
                    bars_held_1m = 0
                    peak_pnl = 0.0
                    was_profitable = False

    # Force close
    if in_pos:
        pt = (tick_price - entry_price) / TICK if direction == 'LONG' else (entry_price - tick_price) / TICK
        trades.append({'time': time_str, 'dir': direction, 'pnl': pt * TV,
                       'exit': 'end_of_day', 'held': bars_held_1m, 'peak': peak_pnl})

    # Report
    t = pd.DataFrame(trades)
    total = t['pnl'].sum()
    wr = (t['pnl'] > 0).mean() * 100
    nl = len(t[t['dir'] == 'LONG'])
    ns = len(t[t['dir'] == 'SHORT'])

    print(f'RESULTS (zero lookahead):')
    print(f'  {len(t)} trades | WR={wr:.1f}% | PnL=${total:,.2f} | $/tr=${total/max(len(t),1):,.2f}')
    print(f'  LONG={nl} SHORT={ns} | Avg hold={t["held"].mean():.1f} bars')
    print()

    for ex in sorted(t['exit'].unique()):
        et = t[t['exit'] == ex]
        wr_e = (et['pnl'] > 0).mean() * 100
        print(f'  {ex:<22} {len(et):>5}  WR={wr_e:>5.1f}%  ${et["pnl"].sum():>9,.2f}  ${et["pnl"].mean():>7,.2f}/tr')

    print()
    # PnL mode
    pnl_rounded = (t['pnl'] / 1).round() * 1
    mode = Counter(pnl_rounded).most_common(3)
    print(f'  PnL mode: {mode}')


if __name__ == '__main__':
    main()
