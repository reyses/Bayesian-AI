"""Legs measured from an ANCHOR (owner 2026-08-04: "you're having trouble
stratifying the legs cuz you're not using an anchor for price displacement").

The previous study used a SLIDING window: |close(t) - close(t-T)| >= D. That
makes every second its own reference, so a "leg" could begin anywhere in the
middle of a move — which is exactly why the population would not stratify and
why it came out a coin flip. His legs are displacement FROM A STRUCTURAL
ORIGIN: a pivot, a level, the point where the move began.

Here the anchor is a confirmed swing pivot. From each anchor we ask:
  did price displace >= D points within <= T seconds of leaving it?
and we measure heat/run FROM THE ANCHOR — the position he actually takes.

Two entry regimes are reported, and the difference between them is the whole
argument:
  AT_ANCHOR   entry at the pivot price (his entry — requires calling it)
  ON_CONFIRM  entry when the displacement is first observable (the chase)

  python research/event_onset/tools/anchored_legs.py
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
PIVOT_TH = 8.0          # what confirms a swing pivot (repo canonical)
D, T = 10.0, 60         # displacement / time that defines a leg
FOLLOW_S = 300
FRICTION = 0.89


def day_rows(path):
    d = pd.read_parquet(path)
    ts = d['timestamp'].to_numpy()
    et = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    mod = et.hour * 60 + et.minute
    k = np.flatnonzero((mod >= RTH0) & (mod < RTH1))
    if len(k) < 900:
        return []
    ts = ts[k]
    c, h, l = (d[x].to_numpy()[k] for x in ('close', 'high', 'low'))
    n = len(c)
    # --- anchors: confirmed swing pivots, stamped at CONFIRMATION time ---
    piv, direction, last = [], 0, 0
    ext_i = 0
    for i in range(1, n):
        if direction >= 0 and c[i] < c[ext_i] - PIVOT_TH:
            if direction > 0:
                piv.append((ext_i, i, +1))      # (anchor bar, confirm bar, was a HIGH)
            direction, ext_i = -1, i
        elif direction <= 0 and c[i] > c[ext_i] + PIVOT_TH:
            if direction < 0:
                piv.append((ext_i, i, -1))      # anchor was a LOW
            direction, ext_i = 1, i
        elif direction > 0 and c[i] > c[ext_i]:
            ext_i = i
        elif direction < 0 and c[i] < c[ext_i]:
            ext_i = i
    rows = []
    for a_i, cf_i, was_high in piv:
        dd = -1 if was_high == +1 else 1        # leave a HIGH -> down leg
        # did it displace D within T seconds of the anchor?
        w_end = min(a_i + T, n - 1)
        disp = float((c[a_i] - c[a_i:w_end + 1].min()) if dd < 0
                     else (c[a_i:w_end + 1].max() - c[a_i]))
        is_leg = disp >= D
        for regime, e_i in (('AT_ANCHOR', a_i), ('ON_CONFIRM', cf_i)):
            j1 = min(e_i + FOLLOW_S, n - 1)
            e = float(c[e_i])
            sh, sl = h[e_i:j1 + 1], l[e_i:j1 + 1]
            mae = float(((e - sl) if dd > 0 else (sh - e)).max())
            mfe = float(((sh - e) if dd > 0 else (e - sl)).max())
            rows.append(dict(day=os.path.basename(path)[:-8], regime=regime,
                             is_leg=bool(is_leg), disp=disp, dd=dd,
                             mae=max(mae, 0.0), mfe=max(mfe, 0.0),
                             run=(float(c[j1]) - e) * dd,
                             lag_s=int(ts[cf_i] - ts[a_i])))
    return rows


if __name__ == '__main__':
    days = [p for p in sorted(glob.glob(os.path.join(BARS, '2025_0[1-6]*.parquet')))
            if len(os.path.basename(p)) == 18]
    rows = []
    for p in tqdm(days, desc='days'):
        rows += day_rows(p)
    R = pd.DataFrame(rows)
    R.to_parquet(os.path.join(OUT, 'anchored_legs.parquet'), index=False)
    rng = np.random.default_rng(20260804)

    def dayci(s, col):
        g = s.groupby('day')[col].agg(['sum', 'count'])
        sd, cd = g['sum'].to_numpy(), g['count'].to_numpy()
        pick = rng.integers(0, len(sd), size=(4000, len(sd)))
        bs = sd[pick].sum(1) / np.maximum(cd[pick].sum(1), 1)
        return np.percentile(bs, [2.5, 97.5])

    print(f'\n{len(R):,} anchor-rows, {R["day"].nunique()} days, '
          f'{int(R["is_leg"].sum()/2):,} anchors produced a leg '
          f'(>= {D:g}pt within {T}s)\n')
    print(f'{"regime":<11} {"leg?":<6} {"n":>7} {"p50 heat":>9} {"p95 heat":>9} '
          f'{"mean MFE":>9} {"mean run":>9} {"P(run>0)":>9}')
    lines = ['# Legs measured from an ANCHOR (confirmed swing pivot)', '',
             f'D={D:g}pt within T={T}s of the anchor; heat/run over '
             f'{FOLLOW_S}s; {R["day"].nunique()} days.', '',
             '| regime | is_leg | n | p50 heat | p95 heat | mean MFE | mean run | P(run>0) [day-CI] |',
             '|---|---|---|---|---|---|---|---|']
    for regime in ('AT_ANCHOR', 'ON_CONFIRM'):
        for leg in (True, False):
            s = R[(R['regime'] == regime) & (R['is_leg'] == leg)]
            if len(s) < 50:
                continue
            s = s.assign(pos=(s['run'] > 0).astype(int))
            lo, hi = dayci(s, 'pos')
            p50, p95 = np.percentile(s['mae'], [50, 95])
            print(f'{regime:<11} {str(leg):<6} {len(s):>7,} {p50:>9.2f} '
                  f'{p95:>9.2f} {s["mfe"].mean():>9.2f} {s["run"].mean():>9.2f} '
                  f'{s["pos"].mean():>8.1%}')
            lines.append(f'| {regime} | {leg} | {len(s):,} | {p50:.2f} | {p95:.2f} '
                         f'| {s["mfe"].mean():.2f} | {s["run"].mean():+.2f} '
                         f'| {s["pos"].mean():.1%} [{lo:.1%},{hi:.1%}] |')
    open(os.path.join(OUT, 'anchored_legs.md'), 'w').write('\n'.join(lines) + '\n')
    print(f'\nmedian confirmation lag: {R["lag_s"].median():.0f}s '
          f'(how long after the anchor you can KNOW)')
