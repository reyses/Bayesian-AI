"""Full OOS run with zero-lookahead 1s ticker, day by day."""
import numpy as np, pandas as pd, glob, gc, sys, os
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.statistical_field_engine import StatisticalFieldEngine
from training.train_trade_cnn import extract_features_13d as extract_13d_batch
from tools.nightmare_ticker import extract_13d_single

TICK = 0.25; TV = 0.50; ROCHE = 2.0; HC = 4; MAX_HOLD = 20
TREND_LB = 15; MIN_MOVE = 10; WARMUP = 300

print('Loading data...')
df_1s = pd.read_parquet('DATA/ATLAS/1s/2026_03.parquet').sort_values('timestamp').reset_index(drop=True)
print(f'  1s: {len(df_1s):,} bars')

files_1m = sorted(glob.glob('DATA/ATLAS/1m/*.parquet'))
df_1m_all = pd.concat([pd.read_parquet(f) for f in files_1m[-2:]], ignore_index=True).sort_values('timestamp').reset_index(drop=True)

df_1h = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1h/*.parquet'))], ignore_index=True).sort_values('timestamp')
sfe_h = StatisticalFieldEngine(); st_h = sfe_h.batch_compute_states(df_1h)
feats_1h = extract_13d_batch(st_h, df_1h); ts_1h = df_1h['timestamp'].values; del st_h, sfe_h; gc.collect()

df_1d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob('DATA/ATLAS/1D/*.parquet'))], ignore_index=True).sort_values('timestamp')
sfe_d = StatisticalFieldEngine(); st_d = sfe_d.batch_compute_states(df_1d)
feats_1d = extract_13d_batch(st_d, df_1d); ts_1d = df_1d['timestamp'].values; del st_d, sfe_d; gc.collect()
print(f'  Macro: 1h={len(df_1h)} 1D={len(df_1d)}')

dates = sorted(set(datetime.utcfromtimestamp(t).strftime('%Y-%m-%d')
    for t in df_1s['timestamp'].values[::5000]))
dates = [d for d in dates if d >= '2026-03-02']
print(f'  Days: {len(dates)}\n')

all_trades = []

for date_str in tqdm(dates, desc='Days'):
    date_ts = pd.Timestamp(date_str).timestamp()
    day_1s = df_1s[(df_1s['timestamp'] >= date_ts) & (df_1s['timestamp'] < date_ts + 86400)]
    if len(day_1s) < 100:
        continue

    warmup_1m = df_1m_all[df_1m_all['timestamp'] < date_ts].tail(WARMUP).reset_index(drop=True)
    if len(warmup_1m) < 60:
        continue

    sfe = StatisticalFieldEngine()
    ws = sfe.batch_compute_states(warmup_1m)
    prev_vel = getattr(ws[-1]['state'] if isinstance(ws[-1], dict) else ws[-1], 'velocity', 0.0) if ws else 0.0
    del ws

    price_hist = list(warmup_1m['close'].values)
    vol_hist = list(warmup_1m['volume'].values if 'volume' in warmup_1m.columns else np.zeros(len(warmup_1m)))
    running_1m = warmup_1m.copy()

    in_pos = False; direction = None; entry_price = 0; bars_held = 0
    peak_pnl = 0; was_profitable = False
    current_minute = -1
    agg_open = 0; agg_close = 0; agg_high = -1e9; agg_low = 1e9; agg_vol = 0; agg_ts = 0
    last_tick_price = 0

    for _, row in day_1s.iterrows():
        tp = row['close']; th = row['high']; tl = row['low']
        to = row['open']; tv = row.get('volume', 0.0); tts = row['timestamp']
        last_tick_price = tp

        tick_min = int(tts) // 60
        if current_minute == -1:
            current_minute = tick_min; agg_open = to; agg_ts = tts
            agg_high = th; agg_low = tl
        agg_high = max(agg_high, th); agg_low = min(agg_low, tl)
        agg_close = tp; agg_vol += tv

        # 1s SL
        if in_pos:
            mae_1s = (entry_price - tl) / TICK if direction == 'LONG' else (th - entry_price) / TICK
            if mae_1s >= 160:
                pt = (tp - entry_price) / TICK if direction == 'LONG' else (entry_price - tp) / TICK
                all_trades.append({'day': date_str, 'pnl': pt * TV, 'exit': 'catastrophic_sl', 'dir': direction})
                in_pos = False

        if tick_min == current_minute:
            continue

        # 1m bar done
        _bar = pd.DataFrame([{'timestamp': agg_ts, 'open': agg_open, 'high': agg_high,
                               'low': agg_low, 'close': agg_close, 'volume': agg_vol}])
        current_minute = tick_min
        agg_open = to; agg_ts = tts; agg_high = th; agg_low = tl; agg_vol = tv

        price = agg_close; high_1m = _bar['high'].iloc[0]; low_1m = _bar['low'].iloc[0]
        price_hist.append(price); vol_hist.append(_bar['volume'].iloc[0])
        if len(price_hist) > 500:
            price_hist = price_hist[-500:]; vol_hist = vol_hist[-500:]

        running_1m = pd.concat([running_1m, _bar], ignore_index=True)
        states = sfe.batch_compute_states(running_1m)
        st = states[-1]; st = st['state'] if isinstance(st, dict) else st; del states

        feat = extract_13d_single(st, price, high_1m, low_1m, _bar['open'].iloc[0],
                                   _bar['volume'].iloc[0], agg_ts, price_hist, vol_hist, prev_vel)
        prev_vel = getattr(st, 'velocity', 0.0)

        z = feat[5]; vr = feat[8]; lam = vr - 1.0
        trend = price_hist[-1] - price_hist[-(TREND_LB + 1)] if len(price_hist) > TREND_LB else 0

        # Position management
        if in_pos:
            bars_held += 1
            pt = (price - entry_price) / TICK if direction == 'LONG' else (entry_price - price) / TICK
            pnl = pt * TV
            peak_pnl = max(peak_pnl, pnl)
            if pnl > 0:
                was_profitable = True

            ex = None
            if abs(z) < 0.5 and bars_held > 1:
                ex = 'mean_reached'
            elif lam > 0.3 and bars_held > 1:
                ex = 'lambda_flip'
            elif was_profitable and peak_pnl > 2 and pnl < peak_pnl * 0.5:
                ex = 'profit_hold'
            elif not was_profitable and bars_held >= 2:
                _t = price_hist[-1] - price_hist[-(TREND_LB + 1)] if len(price_hist) > TREND_LB else 0
                _d = feat[0]
                _to = (direction == 'LONG' and _t < -5) or (direction == 'SHORT' and _t > 5)
                _do = (direction == 'LONG' and _d < -2) or (direction == 'SHORT' and _d > 2)
                if (_to and _do) or bars_held >= HC:
                    ex = 'half_cycle_loss'
            elif was_profitable and bars_held >= MAX_HOLD:
                ex = 'max_hold_profit'

            if ex:
                all_trades.append({'day': date_str, 'pnl': pnl, 'exit': ex, 'dir': direction})
                in_pos = False

        # Entry
        if not in_pos:
            if abs(z) > ROCHE and lam < 0:
                rev = 'SHORT' if z > 0 else 'LONG'
                rev_sign = 1.0 if rev == 'LONG' else -1.0
                score = 0.0
                if abs(trend) > MIN_MOVE:
                    score += 0.5 * (rev_sign * (1.0 if trend > 0 else -1.0))
                i1h = np.searchsorted(ts_1h, agg_ts, side='right') - 1
                d1h = feats_1h[i1h, 0] if 0 <= i1h < len(feats_1h) else 0
                if abs(d1h) > 3:
                    score += 0.3 * (rev_sign * (1.0 if d1h > 0 else -1.0))
                i1d = np.searchsorted(ts_1d, agg_ts, side='right') - 1
                d1d = feats_1d[i1d, 0] if 0 <= i1d < len(feats_1d) else 0
                if abs(d1d) > 3:
                    score += 0.2 * (rev_sign * (1.0 if d1d > 0 else -1.0))
                if score >= -0.3:
                    in_pos = True; direction = rev; entry_price = price
                    bars_held = 0; peak_pnl = 0; was_profitable = False

    # Force close
    if in_pos:
        pt = (last_tick_price - entry_price) / TICK if direction == 'LONG' else (entry_price - last_tick_price) / TICK
        all_trades.append({'day': date_str, 'pnl': pt * TV, 'exit': 'end_of_day', 'dir': direction})
    del running_1m; gc.collect()

# Report
t = pd.DataFrame(all_trades)
total = t['pnl'].sum()
n_days = t['day'].nunique()

print(f'\n{"="*60}')
print(f'FULL OOS — ZERO LOOKAHEAD — 1S TICKER')
print(f'{"="*60}')
print(f'  {len(t)} trades | WR={(t["pnl"]>0).mean()*100:.1f}% | PnL=${total:,.2f}')
print(f'  Days: {n_days} | $/day: ${total/n_days:,.2f}')
print()

for ex in sorted(t['exit'].unique()):
    et = t[t['exit'] == ex]
    print(f'  {ex:<20} {len(et):>5}  WR={(et["pnl"]>0).mean()*100:>5.1f}%  ${et["pnl"].sum():>10,.2f}  ${et["pnl"].mean():>7,.2f}/tr')

print(f'\nDAILY:')
print(f'{"Date":<12} {"N":>5} {"WR":>5} {"PnL":>10} {"$/tr":>7} {"Cumul":>10}')
print('-' * 55)
cumul = 0
for day in sorted(t['day'].unique()):
    dt = t[t['day'] == day]
    cumul += dt['pnl'].sum()
    wr = (dt['pnl'] > 0).mean() * 100
    marker = ' <<<' if dt['pnl'].sum() > 200 else ' !!!' if dt['pnl'].sum() < -200 else ''
    print(f'{day:<12} {len(dt):>5} {wr:>4.0f}% ${dt["pnl"].sum():>9,.2f} ${dt["pnl"].mean():>6,.2f} ${cumul:>9,.2f}{marker}')

# Mode
from collections import Counter
daily_pnls = [t[t['day'] == d]['pnl'].sum() for d in t['day'].unique()]
mode_bucket = Counter([int(p // 100) * 100 for p in daily_pnls]).most_common(3)
print(f'\nDaily PnL mode (buckets): {mode_bucket}')
print(f'Winning days: {sum(1 for p in daily_pnls if p > 0)}/{len(daily_pnls)}')

# Save
t.to_csv('reports/findings/nightmare_ticker_oos.csv', index=False)
print(f'\nSaved: reports/findings/nightmare_ticker_oos.csv')
