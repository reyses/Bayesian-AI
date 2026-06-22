"""Mechanical execution validation of the 5,463 GA-Kalman trades against the 1s GROUND TRUTH.
Confirms the backtest took trades correctly (or finds the bug). Code is definitive here —
no LLM. If this fails, every kill/keep verdict on this strategy is void.

Checks per trade:
  1. entry_price == 1s close at entry_ts (to the tick)
  2. exit_price  == 1s close at exit_ts
  3. gross_usd   == (exit-entry)*dir*$2     ;  net_usd == gross - $2.5
  4. mfe_pts/mae_pts == recomputed from the actual 1s path [entry,exit]
  5. exit-rule fired: at exit, (peak-price ≥ 79.4pt trail) OR (pnl ≤ -$100 stop) OR EOD
  6. causality: exit_ts > entry_ts
Output: reports/findings/trade_execution_validation.md + _anomalies.csv (flagged trades)

USER RUNS THIS. Run: python research/validate_trade_execution.py
"""
import os
import numpy as np
import pandas as pd

CSV = 'research/kalman_tuning_eda/reports/findings/kalman_full_trades.csv'
ONE_S = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/1s'
TICK = 0.25; USD = 2.0; COST = 2.5; TRAIL = 79.4; STOP_USD = 100.0
GAP_PT = 15.0      # a single 1s step > this = a price gap (stop can't fire through it)
GAP_SEC = 300      # a >5-min gap between consecutive bars = a session halt / roll seam spanned


def main():
    tr = pd.read_csv(CSV)
    flags = []
    chk = {k: 0 for k in ['entry_px', 'exit_px', 'gross', 'net', 'mfe', 'mae', 'exit_rule', 'causal', 'no_price_gap', 'no_time_gap']}
    tot = 0
    for day, g in tr.groupby('day'):
        f = f'{ONE_S}/{day}.parquet'
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
        ts = d['timestamp'].to_numpy(np.int64); px = d['close'].to_numpy(np.float64)
        for _, t in g.iterrows():
            tot += 1
            ets, xts = int(t['entry_ts']), int(t['exit_ts'])
            ie = np.searchsorted(ts, ets); ix = np.searchsorted(ts, xts)
            sgn = 1 if t['dir'] == 'LONG' else -1
            fl = []
            # 1-2 price lookups
            c_in = px[ie] if ie < len(ts) and ts[ie] == ets else np.nan
            c_out = px[ix] if ix < len(ts) and ts[ix] == xts else np.nan
            if not (abs(c_in - t['entry_price']) <= TICK): fl.append('entry_px')
            else: chk['entry_px'] += 1
            if not (abs(c_out - t['exit_price']) <= TICK): fl.append('exit_px')
            else: chk['exit_px'] += 1
            # 3 pnl reconcile
            g_exp = (t['exit_price'] - t['entry_price']) * sgn * USD
            if abs(g_exp - t['gross_usd']) <= 0.6: chk['gross'] += 1
            else: fl.append('gross')
            if abs((t['gross_usd'] - COST) - t['net_usd']) <= 0.1: chk['net'] += 1
            else: fl.append('net')
            # 4 mfe/mae from path
            if ie < len(ts) and ix > ie:
                path = px[ie:ix + 1]
                mfe_r = (path.max() - t['entry_price']) * sgn if sgn == 1 else (t['entry_price'] - path.min())
                mae_r = (path.min() - t['entry_price']) * sgn if sgn == 1 else (t['entry_price'] - path.max())
                # signed-to-dir
                mfe_r = (path.max() - t['entry_price']) if sgn == 1 else (t['entry_price'] - path.min())
                mae_r = (path.min() - t['entry_price']) if sgn == 1 else (t['entry_price'] - path.max())
                if abs(mfe_r - t['mfe_pts']) <= TICK: chk['mfe'] += 1
                else: fl.append('mfe')
                if abs(mae_r - t['mae_pts']) <= TICK: chk['mae'] += 1
                else: fl.append('mae')
                # 5 exit rule: trail OR stop OR EOD(last bar of day)
                trail_hit = (t['mfe_pts'] - (t['exit_price'] - t['entry_price']) * sgn) >= TRAIL - TICK
                stop_hit = (t['exit_price'] - t['entry_price']) * sgn * USD <= -STOP_USD + 1
                eod = ix >= len(ts) - 2
                if trail_hit or stop_hit or eod: chk['exit_rule'] += 1
                else: fl.append('exit_rule')
                # GAP detection: a real trade has no intra-bar price gap and no session-halt span
                steps = np.abs(np.diff(path))
                if steps.size and steps.max() <= GAP_PT: chk['no_price_gap'] += 1
                else: fl.append('PRICE_GAP')        # stop defeated by a discontinuity
                tgap = np.diff(ts[ie:ix + 1])
                if tgap.size and tgap.max() <= GAP_SEC: chk['no_time_gap'] += 1
                else: fl.append('TIME_GAP')          # spans a halt / roll seam (artifact trade)
            if xts > ets: chk['causal'] += 1
            else: fl.append('causal')
            if fl:
                flags.append(dict(day=day, entry_ts=ets, dir=t['dir'], net=t['net_usd'],
                                  mfe=t['mfe_pts'], flags='|'.join(fl)))

    L = ["# Trade execution validation — 5,463 GA-Kalman trades vs 1s ground truth\n",
         f"Total trades checked: {tot}\n",
         "| check | pass | pass % |", "|---|---|---|"]
    for k, v in chk.items():
        L.append(f"| {k} | {v}/{tot} | {v/tot*100:.1f}% |")
    L += ["", f"**Flagged anomalies: {len(flags)} ({len(flags)/tot*100:.1f}%)** → _anomalies.csv",
          "If entry_px/exit_px/gross/net fail >0% → MECHANICAL BUG (verdicts void). "
          "exit_rule fails are expected only at EOD/edge; investigate if many."]
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/trade_execution_validation.md', 'w', encoding='utf-8').write("\n".join(L))
    pd.DataFrame(flags).to_csv('reports/findings/trade_execution_validation_anomalies.csv', index=False)
    print("\n".join(L))


if __name__ == '__main__':
    main()
