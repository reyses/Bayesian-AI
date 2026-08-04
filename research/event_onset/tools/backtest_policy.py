"""Standalone backtest of the PRE-REGISTERED autorun policy across many days.

One day proved nothing (N=12 trades, a 10-minute window, momentum entries in
the most volatile stretch of the session). This runs the identical rules over
every val-window day and reports a day-clustered distribution, because the
project rule is: no $/day claim without N and a CI.

Rules (identical to research/dojo_forge/tools/autorun_gbm.py):
  ENTRY  onset(fakeout_poke|leg_descent) >= 0.70 AND |60s net move| >= 2pt,
         trade WITH the move, stop -10, one at a time, max 12/day
  EXIT   ladder locks 50% of peak once MFE >= 5 (never loosens);
         entry-touch halt and 75%-retention close-through both exit;
         friction 0.89pt per round trip
"""
import glob
import os
import sys

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = '/media/moi/WindowsCode/Bayesian-AI'
sys.path.insert(0, os.path.join(REPO, 'research', 'event_onset', 'builders'))
from build_onset_dataset import _feat_matrix                  # noqa: E402

FRICTION, STOP_PT, MAX_TRADES = 0.89, 10.0, 12
ONSET_MIN, MOVE_MIN = 0.70, 2.0
LADDER_TRIG, LADDER_LOCK, RET_EXIT = 5.0, 0.50, 0.75
RTH0, RTH1 = 9 * 60 + 30, 15 * 60 + 30


def load_models():
    out = {}
    for p in sorted(glob.glob(os.path.join(REPO, 'research', 'event_onset',
                                           'models', 'gbm_*_10s.joblib'))):
        name = os.path.basename(p).replace('gbm_', '').replace('_10s.joblib', '')
        if name in ('fakeout_poke', 'leg_descent'):
            out[name] = joblib.load(p)
    return out


def day_run(day, models):
    p1 = os.path.join(REPO, 'DATA', 'ATLAS', '1s', f'{day}.parquet')
    p5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s', f'{day}.parquet')
    if not (os.path.exists(p1) and os.path.exists(p5)):
        return None
    d1, d5 = pd.read_parquet(p1), pd.read_parquet(p5)
    t5 = d5['timestamp'].to_numpy()
    et5 = pd.to_datetime(t5, unit='s', utc=True).tz_convert('America/New_York')
    mod5 = et5.hour * 60 + et5.minute
    ok = np.flatnonzero((mod5 >= RTH0) & (mod5 < RTH1))
    ok = ok[ok > 400]
    if len(ok) < 50:
        return None
    o, h, l, c = (d5[k].to_numpy() for k in ('open', 'high', 'low', 'close'))
    v = d5['volume'].to_numpy() if 'volume' in d5 else np.ones(len(d5))
    feat = _feat_matrix(t5, o, h, l, c, v, ok)
    prob = np.zeros(len(ok))
    for b in models.values():
        X = np.nan_to_num(feat[b['feats']].to_numpy(float), nan=0.,
                          posinf=0., neginf=0.)
        prob = np.maximum(prob, b['model'].predict_proba(
            b['scaler'].transform(X))[:, 1])

    t1 = d1['timestamp'].to_numpy()
    c1 = d1['close'].to_numpy()
    h1, l1 = d1['high'].to_numpy(), d1['low'].to_numpy()
    trades, i1 = [], 0
    pos = None
    for k, i5 in enumerate(ok):
        t = int(t5[i5]) + 4                     # decision at bar close
        j = int(np.searchsorted(t1, t, side='right')) - 1
        if j <= 60 or j >= len(t1) - 2:
            continue
        if pos is None:
            if len(trades) >= MAX_TRADES or prob[k] < ONSET_MIN:
                continue
            mv = float(c1[j] - c1[max(j - 60, 0)])
            if abs(mv) < MOVE_MIN:
                continue
            pos = dict(d=1 if mv > 0 else -1, e=float(c1[j]), j0=j,
                       stop=float(c1[j]) - np.sign(mv) * STOP_PT, peak=0.0)
            continue
        # walk 1s bars until an exit rule fires
        d = pos['d']
        while j < len(t1) - 1 and pos is not None:
            hi, lo, cl = float(h1[j]), float(l1[j]), float(c1[j])
            if (d > 0 and lo <= pos['stop']) or (d < 0 and hi >= pos['stop']):
                px = (min(float(d1['open'].iloc[j]), pos['stop']) if d > 0
                      else max(float(d1['open'].iloc[j]), pos['stop']))
                trades.append((px - pos['e']) * d - FRICTION)
                pos = None
                break
            fav = (hi - pos['e']) if d > 0 else (pos['e'] - lo)
            pos['peak'] = max(pos['peak'], fav)
            if pos['peak'] >= LADDER_TRIG:
                ns = pos['e'] + d * pos['peak'] * LADDER_LOCK
                if (d > 0 and ns > pos['stop']) or (d < 0 and ns < pos['stop']):
                    pos['stop'] = ns
            cur = (cl - pos['e']) * d
            if pos['peak'] > 0 and (cur / pos['peak'] if pos['peak'] else 1) < RET_EXIT \
                    and pos['peak'] >= LADDER_TRIG:
                trades.append(cur - FRICTION)
                pos = None
                break
            if pos['peak'] > 0 and cur <= 0:          # entry-touch warning
                trades.append(cur - FRICTION)
                pos = None
                break
            j += 1
        i1 = j
    return trades


if __name__ == '__main__':
    models = load_models()
    days = sorted(os.path.basename(f)[:-8] for f in
                  glob.glob(os.path.join(REPO, 'DATA', 'ATLAS', '1s',
                                         '2025_0[1-6]*.parquet')))
    days = [d for d in days if len(d) == 10]
    rows = []
    for day in tqdm(days, desc='days'):
        tr = day_run(day, models)
        if tr:
            rows.append(dict(day=day, n=len(tr), pnl=float(np.sum(tr)),
                             mean=float(np.mean(tr))))
    r = pd.DataFrame(rows)
    r.to_parquet(os.path.join(REPO, 'research', 'event_onset',
                              'policy_backtest.parquet'), index=False)
    rng = np.random.default_rng(20260804)
    dp = r['pnl'].to_numpy()
    bs = [dp[rng.integers(0, len(dp), len(dp))].mean() for _ in range(4000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'\nDAYS {len(r)} | TRADES {int(r["n"].sum()):,}')
    print(f'mean {dp.mean():+.2f} pt/day  day-bootstrap 95% CI '
          f'[{lo:+.2f}, {hi:+.2f}]')
    print(f'median {np.median(dp):+.2f} | win days {(dp>0).mean():.0%} | '
          f'mean/trade {r["pnl"].sum()/r["n"].sum():+.3f}pt')
