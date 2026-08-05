#!/usr/bin/env python3
"""GOOD-DIP HORIZON (owner 2026-07-28, TG): the theory — good_dipped have a bounded
dip envelope (let it dip just enough to work); a trade that runs BEYOND that horizon
is a wrong/runaway to FLIP. Measure the envelope + the separation on the doc-107 pop.

  1) good_dipped dip DEPTH (MAE) + recovery TIME distributions vs wrong.
  2) P(wrong | dip reached <= -D) across D — where do good_dipped 'run out' so
     beyond = flip zone? (Note recovery_dynamics: survivorship rebound at deep D.)
  3) depth-gated flip@D pnl (flip at first dip<=-D) swept DEEP, vs never-bail.
Absolute ticks, friction 2.4t/RT. reports/wrongdir/good_dip_horizon.md
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_wrongdir as swl          # noqa: E402

T, FR = 4.0, 2.4
BAND, DIP = 4.0, 4.0
DGRID = [2, 4, 6, 8, 10, 12, 16, 20, 25, 30, 40]


def main():
    eng = swl.engagements()
    day_engs, _ = swl.scan(eng)
    rows = []
    for day, engs in day_engs.items():
        for e in engs:
            dp = np.asarray(e['per_minute_forward_drift'], float); wm = e['window_minutes']
            term = e['terminal']; md = e['mindrift']
            cls = ('wrong' if term <= -BAND else
                   ('good_dipped' if (term >= BAND and md <= -DIP) else
                    ('good_clean' if term >= BAND else 'dead_band')))
            amin = int(np.argmin(dp[:wm + 1]))
            # recovery time: dip extreme -> first breakeven after it
            rt = np.nan
            for j in range(amin + 1, wm + 1):
                if dp[j] >= 0:
                    rt = j - amin; break
            rows.append(dict(cls=cls, mae=md, amin=amin, rt=rt, dp=dp, wm=wm, term=term))
    md = {c: np.array([r['mae'] for r in rows if r['cls'] == c]) for c in ['good_dipped', 'wrong', 'good_clean']}
    gd_rt = np.array([r['rt'] for r in rows if r['cls'] == 'good_dipped' and r['rt'] == r['rt']])
    nb = np.array([r['term'] * T - FR for r in rows])
    N = len(rows)

    def q(a, p):
        return np.percentile(a, p)
    lines = ['# Good-dip horizon — envelope + flip-beyond', '',
             f'N={N:,}. Dip DEPTH = MAE (min favorable-signed drift), points.', '',
             '## 1. Dip-depth envelope (MAE, pts) by class',
             '| class | n | median | p75 | p90 | p95 | p99 | deepest |', '|---|---|---|---|---|---|---|---|']
    for c in ['good_dipped', 'wrong']:
        a = md[c]
        lines.append(f'| {c} | {len(a):,} | {q(a,50):.1f} | {q(a,25):.1f} | {q(a,10):.1f} | '
                     f'{q(a,5):.1f} | {q(a,1):.1f} | {a.min():.1f} |')
    lines += ['',
              f'good_dipped recovery time (dip→breakeven): median {np.median(gd_rt):.0f}m, '
              f'p90 {q(gd_rt,90):.0f}m, p95 {q(gd_rt,95):.0f}m.',
              '',
              '## 2. Separation — P(class | dip reached ≤ -D)',
              '| D (pts) | N reached | P(good_dipped) | P(wrong) | P(wrong)/[gd+wrong] |',
              '|---|---|---|---|---|']
    for D in DGRID:
        reached = [r for r in rows if r['mae'] <= -D]
        n = len(reached)
        if n < 30:
            continue
        pg = np.mean([r['cls'] == 'good_dipped' for r in reached])
        pw = np.mean([r['cls'] == 'wrong' for r in reached])
        purity = pw / (pw + pg) if (pw + pg) else float('nan')
        lines.append(f'| {D} | {n:,} | {pg:.1%} | {pw:.1%} | {purity:.1%} |')

    lines += ['', '## 3. Depth-gated flip@D (flip at first dip≤-D) vs never-bail (+%.2f t/ep)' % nb.mean(),
              '| D (pts) | flip mean | Δ vs never-bail | %flipped |', '|---|---|---|---|']
    best = None
    for D in DGRID:
        flip = np.empty(N); fl = 0
        for i, r in enumerate(rows):
            dp = r['dp']; wm = r['wm']
            below = np.where(dp[:wm + 1] <= -D)[0]
            if len(below):
                tf = int(below[0]); fl += 1
                flip[i] = (2 * dp[tf] - dp[wm]) * T - 2 * FR
            else:
                flip[i] = dp[wm] * T - FR
        d = flip.mean() - nb.mean()
        lines.append(f'| {D} | {flip.mean():+.2f} | {d:+.2f} | {fl/N:.0%} |')
        if best is None or flip.mean() > best[1]:
            best = (D, flip.mean())
    lines += ['',
              f'Best depth-gated flip: D={best[0]}pt (mean {best[1]:+.2f} t/ep vs never-bail '
              f'{nb.mean():+.2f}). ',
              'Read: if good_dipped MAE has a p95/p99 ceiling and P(wrong|reached -D) climbs to '
              '~100% beyond it, that D is the flip horizon. If P(wrong) plateaus (<100%) due to '
              'the survivorship rebound, flipping beyond still whipsaws the deep-dip survivors — '
              'the envelope is soft, not a clean gate.']
    out = os.path.join(HERE, '..', 'reports', 'wrongdir', 'good_dip_horizon.md')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
