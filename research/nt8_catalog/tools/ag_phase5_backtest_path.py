"""
PATH-ACCURATE backtest of the two clean Tier-1 candidates (doc 041):
  - ROUND-05  : breach continuation. Enter close[event_idx] in breach direction.
                Target +20 / stop -20.
  - PIVOT-16 F: fade-the-touch (FLIP of article). mode bullish_bounce (article longs
                S1) -> we SHORT; bearish_bounce -> we LONG. TP +10 / stop -20.

Replaces the MFE/MAE worst-case approximation with a real bar-sequence replay on
RTH 5s bars: walk forward from entry; a bar's LOW <= stop-price triggers the stop,
HIGH >= target triggers the target (mirrored for shorts); if BOTH inside the same
5s bar, count the STOP first (conservative). Exit at EOD close if neither hits.
Day-block bootstrap CIs per year. No nulls (house directive).
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import ag_phase5_doe as D

RAW5S = os.path.abspath(os.path.join(D.BASE, '../..', 'DATA', 'ATLAS', '5s'))

CONFIGS = {
    'ROUND-05_Psych_Numbers': dict(flip=False, target=20.0, stop=20.0),
    'PIVOT-16_Floor_Levels':  dict(flip=True,  target=10.0, stop=20.0),
}

def rth(df):
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    return df[(dt.dt.time >= pd.Timestamp('08:30').time()) & (dt.dt.time <= pd.Timestamp('15:15').time())].reset_index(drop=True)

def run(dossier):
    cfg = CONFIGS[dossier]
    ev = pd.read_parquet(os.path.join(D.BASE, 'tests', dossier, 'events.parquet'))
    pnl_all, day_all, yr_all, dur_all = [], [], [], []
    for day in sorted(ev['day'].unique()):
        p = os.path.join(RAW5S, day.replace('-', '_') + '.parquet')
        if not os.path.exists(p):
            continue
        bars = rth(pd.read_parquet(p))
        hi, lo, cl = bars['high'].values, bars['low'].values, bars['close'].values
        n = len(bars)
        for _, r in ev[ev['day'] == day].iterrows():
            ei = int(r['event_idx'])
            if ei >= n - 2:
                continue
            mode = str(r['mode'])
            art_long = mode.startswith('bull')          # article-side direction
            is_long = art_long if not cfg['flip'] else (not art_long)
            entry = cl[ei]
            if is_long:
                tp, st = entry + cfg['target'], entry - cfg['stop']
            else:
                tp, st = entry - cfg['target'], entry + cfg['stop']
            pnl, exit_i = None, None
            for i in range(ei + 1, n):
                if is_long:
                    hit_st, hit_tp = lo[i] <= st, hi[i] >= tp
                else:
                    hit_st, hit_tp = hi[i] >= st, lo[i] <= tp
                if hit_st:                      # stop first if both in same bar
                    pnl = -cfg['stop']; exit_i = i; break
                if hit_tp:
                    pnl = cfg['target']; exit_i = i; break
            if pnl is None:                     # EOD close
                pnl = (cl[n - 1] - entry) if is_long else (entry - cl[n - 1])
                exit_i = n - 1
            pnl_all.append(pnl); day_all.append(day); yr_all.append(day[:4])
            dur_all.append((exit_i - ei) * 5 / 60.0)   # minutes
    pnl_all = np.array(pnl_all); day_all = np.array(day_all); yr_all = np.array(yr_all)
    dur_all = np.array(dur_all)
    print(f'\n===== {dossier}  ({"FLIP" if cfg["flip"] else "STATED"}  T{cfg["target"]:.0f}/S{cfg["stop"]:.0f}) =====')
    for y in sorted(np.unique(yr_all)):
        m = yr_all == y
        pnl, dd = pnl_all[m], day_all[m]
        ev_, lo_, hi_ = D.day_ci(pnl, dd)
        wr = (pnl > 0).mean()
        md = float(pd.Series(np.round(pnl)).mode().iloc[0])
        print(f'{y}: N={m.sum()}/{len(np.unique(dd))}d WR={wr:.2f} EV={ev_:+.2f} '
              f'CI[{lo_:+.2f},{hi_:+.2f}] mode={md:+.0f} med_dur={np.median(dur_all[m]):.0f}min '
              f'worst={sorted(np.round(pnl).astype(int))[:3]}')
    return pnl_all, day_all, yr_all

if __name__ == '__main__':
    for t in (sys.argv[1:] or list(CONFIGS)):
        run(t)
