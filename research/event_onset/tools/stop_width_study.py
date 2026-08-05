"""How wide must a stop be to HOLD a real leg? (owner objective 2026-08-04:
"the objective will be to hold thru the noise")

Entry-rule independent by design. Every prior attempt to answer this rode on
a policy whose entries have no edge, which confounds "was the stop right"
with "was the entry right". This measures a property of the TAPE:

  for every zigzag leg >= MIN_LEG pt, entering LATE_S seconds after it
  starts, what is the worst heat (MAE) before the leg completes?

That gives, per stop width, the fraction of real legs you survive and the
points you keep — the exact trade-off behind -10 vs -30.

  python research/event_onset/tools/stop_width_study.py
"""
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = '/media/moi/WindowsCode/Bayesian-AI'
BARS = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
OUT = os.path.join(REPO, 'research', 'event_onset', 'reports')
TH = 8.0                     # zigzag reversal threshold (repo canonical)
MIN_LEG = 15.0               # a "real leg" worth holding
LATE_S = (0, 30, 60, 120)    # how late you enter
STOPS = (5, 10, 15, 20, 25, 30, 40, 50)
RTH0, RTH1 = 9 * 60 + 30, 15 * 60 + 30
FRICTION = 0.89


def legs_for_day(path):
    d = pd.read_parquet(path)
    ts = d['timestamp'].to_numpy()
    et = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    mod = et.hour * 60 + et.minute
    k = np.flatnonzero((mod >= RTH0) & (mod < RTH1))
    if len(k) < 600:
        return []
    ts, c = ts[k], d['close'].to_numpy()[k]
    h, l = d['high'].to_numpy()[k], d['low'].to_numpy()[k]
    piv, direction = [0], 0
    for i in range(1, len(c)):
        if direction >= 0 and c[i] < c[piv[-1]] - TH:
            if direction > 0:
                piv.append(i)
            direction = -1
        elif direction <= 0 and c[i] > c[piv[-1]] + TH:
            if direction < 0:
                piv.append(i)
            direction = 1
        elif direction > 0 and c[i] > c[piv[-1]]:
            piv[-1] = i
        elif direction < 0 and c[i] < c[piv[-1]]:
            piv[-1] = i
    out = []
    for a, b in zip(piv[:-1], piv[1:]):
        size = float(c[b] - c[a])
        if abs(size) < MIN_LEG:
            continue
        dd = 1 if size > 0 else -1
        for late in LATE_S:
            j = int(np.searchsorted(ts, ts[a] + late))
            if j >= b:
                continue
            e = float(c[j])
            seg_h, seg_l = h[j:b + 1], l[j:b + 1]
            # ADVERSE excursion: for a LONG the pain is the LOW, for a
            # SHORT it is the HIGH. The first version had these swapped and
            # so measured favourable excursion as 'heat' — which made a -15
            # stop look like it survived 0% of legs.
            mae = float(((e - seg_l) if dd > 0 else (seg_h - e)).max())
            out.append(dict(day=os.path.basename(path)[:-8], late=late,
                            size=abs(size), dd=dd, mae=max(mae, 0.0),
                            remaining=abs(float(c[b]) - e),
                            secs=int(ts[b] - ts[j])))
    return out


if __name__ == '__main__':
    rows = []
    for p in tqdm(sorted(glob.glob(os.path.join(BARS, '2025_0[1-6]*.parquet'))),
                  desc='days'):
        if len(os.path.basename(p)) != 18:
            continue
        rows += legs_for_day(p)
    L = pd.DataFrame(rows)
    L.to_parquet(os.path.join(OUT, 'leg_mae.parquet'), index=False)
    print(f'\n{len(L):,} leg-entries across {L["day"].nunique()} days '
          f'(legs >= {MIN_LEG:g}pt)\n')
    lines = ['# How wide must a stop be to hold a real leg?', '',
             f'{len(L):,} entries into {int(len(L)/len(LATE_S)):,} legs of '
             f'>= {MIN_LEG:g}pt, {L["day"].nunique()} days. Entry-rule '
             'independent: this is the tape, not a policy.', '']
    for late in LATE_S:
        s = L[L['late'] == late]
        lines += [f'## entering {late}s after the leg starts (n={len(s):,})', '',
                  '| stop | legs survived | mean kept | median kept | E[pt] |',
                  '|---|---|---|---|---|']
        print(f'entering {late}s late (n={len(s):,}):')
        for st in STOPS:
            surv = s['mae'] < st
            kept = s.loc[surv, 'remaining']
            # expectancy: survivors keep `remaining`, casualties lose the stop
            exp = (kept.sum() - st * (~surv).sum()) / len(s) - FRICTION
            lines.append(f'| −{st} | {surv.mean():.0%} | '
                         f'{kept.mean() if len(kept) else 0:.1f} | '
                         f'{kept.median() if len(kept) else 0:.1f} | '
                         f'{exp:+.2f} |')
            print(f'   stop -{st:2d}: survive {surv.mean():5.1%} | '
                  f'mean kept {kept.mean() if len(kept) else 0:6.1f}pt | '
                  f'E {exp:+6.2f}pt')
        lines.append('')
        print()
    open(os.path.join(OUT, 'stop_width_study.md'), 'w').write('\n'.join(lines))
    print('wrote', os.path.join(OUT, 'stop_width_study.md'))
