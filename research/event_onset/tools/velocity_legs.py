"""Legs defined the OWNER's way: a displacement in PRICE within a
displacement in TIME (2026-08-04). Velocity, not zigzag amplitude.

Why this replaces the earlier study: a zigzag leg is any >=8pt move that
eventually reverses — a 15pt drift over 40 minutes qualifies. He does not
trade that. His leg is an IMPULSE: >= D points inside <= T seconds. The two
populations are different objects, and the heat/entry numbers I reported
were computed on the wrong one.

For each (D, T) cell this measures, across 112 val days:
  - how many impulses occur, and how often
  - MAE (heat) if you enter at the trigger — the moment the impulse is
    OBSERVABLE, i.e. when the displacement has already happened
  - how much further it runs after the trigger (the tradeable part)

  python research/event_onset/tools/velocity_legs.py
"""
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = '/media/moi/WindowsCode/Bayesian-AI'
BARS = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
OUT = os.path.join(REPO, 'research', 'event_onset', 'reports')
RTH0, RTH1 = 9 * 60 + 30, 15 * 60 + 30
GRID = [(10, 30), (10, 60), (15, 30), (15, 60), (20, 60), (10, 15), (20, 30)]
FOLLOW_S = 300          # how far forward we measure the run
COOLDOWN_S = 60         # one trigger per impulse


def day_impulses(path, D, T):
    d = pd.read_parquet(path)
    ts = d['timestamp'].to_numpy()
    et = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    mod = et.hour * 60 + et.minute
    k = np.flatnonzero((mod >= RTH0) & (mod < RTH1))
    if len(k) < 600:
        return []
    ts = ts[k]
    c, h, l = (d[x].to_numpy()[k] for x in ('close', 'high', 'low'))
    n = len(c)
    out, last = [], -10 ** 9
    for i in range(T, n - 1):
        if ts[i] - last < COOLDOWN_S:
            continue
        disp = c[i] - c[i - T]                 # displacement over T seconds
        if abs(disp) < D:
            continue
        dd = 1 if disp > 0 else -1
        j1 = min(i + FOLLOW_S, n - 1)
        seg_h, seg_l = h[i:j1 + 1], l[i:j1 + 1]
        e = float(c[i])
        mae = float(((e - seg_l) if dd > 0 else (seg_h - e)).max())
        mfe = float(((seg_h - e) if dd > 0 else (e - seg_l)).max())
        out.append(dict(day=os.path.basename(path)[:-8], ts=int(ts[i]),
                        dd=dd, disp=abs(float(disp)), mae=max(mae, 0.0),
                        mfe=max(mfe, 0.0), end=float(c[j1]) ,
                        run=(float(c[j1]) - e) * dd))
        last = ts[i]
    return out


if __name__ == '__main__':
    days = [p for p in sorted(glob.glob(os.path.join(BARS, '2025_0[1-6]*.parquet')))
            if len(os.path.basename(p)) == 18]
    lines = ["# Legs as the owner defines them: price displacement in a time window",
             '', 'An impulse fires when |close(t) - close(t-T)| >= D. Entry is at '
             'the trigger — the first moment it is OBSERVABLE. MAE/MFE/run are '
             f'measured over the next {FOLLOW_S}s.', '',
             '| D (pt) | T (s) | impulses | per day | p50 heat | p95 heat | '
             'median run | mean run | P(run>0) |', '|---|---|---|---|---|---|---|---|---|']
    print(f'{"D":>4} {"T":>4} {"n":>7} {"/day":>6} {"p50 MAE":>8} {"p95 MAE":>8} '
          f'{"med run":>8} {"mean run":>9} {"P(run>0)":>9}')
    for D, T in GRID:
        rows = []
        for p in tqdm(days, desc=f'D{D}/T{T}', leave=False):
            rows += day_impulses(p, D, T)
        if not rows:
            continue
        r = pd.DataFrame(rows)
        r.to_parquet(os.path.join(OUT, f'impulses_D{D}_T{T}.parquet'), index=False)
        p50, p95 = np.percentile(r['mae'], [50, 95])
        line = (f'{D:>4} {T:>4} {len(r):>7,} {len(r)/r["day"].nunique():>6.1f} '
                f'{p50:>8.2f} {p95:>8.2f} {r["run"].median():>8.2f} '
                f'{r["run"].mean():>9.2f} {(r["run"]>0).mean():>9.1%}')
        print(line)
        lines.append(f'| {D} | {T} | {len(r):,} | {len(r)/r["day"].nunique():.1f} '
                     f'| {p50:.2f} | {p95:.2f} | {r["run"].median():+.2f} '
                     f'| {r["run"].mean():+.2f} | {(r["run"]>0).mean():.1%} |')
    open(os.path.join(OUT, 'velocity_legs.md'), 'w').write('\n'.join(lines) + '\n')
    print('\nwrote', os.path.join(OUT, 'velocity_legs.md'))
