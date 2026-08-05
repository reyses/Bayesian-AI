#!/usr/bin/env python3
"""FLIP-WHEN-WRONG (owner 2026-07-28, TG): instead of holding (never-bail) or
cutting a wrong-direction trade, REVERSE the position when we detect we're wrong
(first drawdown <= -X). Caveat (owner): works better the EARLIER we flip.

Mechanical, on the doc-107 population (23,378 drift paths). Absolute pnl (ticks,
$0.50/tick... MNQ $2/pt = 4t/pt so 1t=$0.50), friction 2.4t/RT:
  never-bail : dp[win]*4 - 2.4
  cut@X      : dp[t_cut]*4 - 2.4
  flip@X     : (2*dp[t_flip] - dp[win])*4 - 4.8   (2 RTs; flip captures the leg
               that keeps going AFTER the drawdown = the real direction)
Decompose by class (flip WINS on wrong runaways, LOSES on good_dipped recoveries);
sweep X early->late. Day-block CI on flip-vs-never-bail.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_wrongdir as swl          # noqa: E402

T, FR = 4.0, 2.4                        # ticks/pt, friction ticks/RT
BAND, DIP = 4.0, 4.0
XG = [2, 3, 4, 6, 8, 12]               # detection depth (pts), early -> late
BOOT, SEED = 4000, 42


def dayblock_ci(day_vals):
    rng = np.random.default_rng(SEED)
    days = list(day_vals)
    means = np.array([np.mean(day_vals[d]) for d in days])
    ns = np.array([len(day_vals[d]) for d in days])
    boot = []
    for _ in range(BOOT):
        idx = rng.integers(0, len(days), len(days))
        boot.append(np.average(means[idx], weights=ns[idx]))
    return np.percentile(boot, [2.5, 97.5])


def main():
    eng = swl.engagements()
    day_engs, _ = swl.scan(eng)
    E = []
    for day, engs in day_engs.items():
        for e in engs:
            dp = np.asarray(e['per_minute_forward_drift'], float); wm = e['window_minutes']
            terminal = e['terminal']; mindrift = e['mindrift']
            cls = ('wrong' if terminal <= -BAND else
                   ('good_dipped' if (terminal >= BAND and mindrift <= -DIP) else
                    ('good_clean' if terminal >= BAND else 'dead_band')))
            E.append((day, cls, dp, wm, terminal))
    N = len(E)
    nb = np.array([e[4] * T - FR for e in E])            # never-bail abs (ticks)
    cls = np.array([e[1] for e in E])
    days = np.array([e[0] for e in E])

    lines = ['# Flip-when-wrong — mechanical, doc-107 population', '',
             f'N={N:,}. never-bail mean abs = {nb.mean():+.2f} t/ep '
             f'(${nb.mean()*0.5:+.2f}). Flip fires at first drawdown <= -X pts.',
             '',
             '| X (pts, flip depth) | flip mean | Δ vs never-bail | 95% CI | %flipped | '
             'cut@X mean |', '|---|---|---|---|---|---|']
    per_class = {}
    for X in XG:
        flip = np.empty(N); cut = np.empty(N); flipped = 0
        dv = {}
        for i, (day, c, dp, wm, term) in enumerate(E):
            below = np.where(dp[:wm + 1] <= -X)[0]
            if len(below):
                tf = int(below[0]); flipped += 1
                flip[i] = (2 * dp[tf] - dp[wm]) * T - 2 * FR
                cut[i] = dp[tf] * T - FR
            else:
                flip[i] = dp[wm] * T - FR                 # never triggered -> never-bail
                cut[i] = dp[wm] * T - FR
            dv.setdefault(day, []).append(flip[i] - nb[i])
        lo, hi = dayblock_ci(dv)
        d = flip.mean() - nb.mean()
        sig = '*' if (lo > 0 or hi < 0) else ''
        lines.append(f'| {X} | {flip.mean():+.2f} | {d:+.2f}{sig} | [{lo:+.2f}, {hi:+.2f}] | '
                     f'{flipped/N:.0%} | {cut.mean():+.2f} |')
        per_class[X] = {c: flip[cls == c].mean() for c in ['good_clean', 'good_dipped', 'wrong', 'dead_band']}

    # class decomposition at the best (earliest) X
    bx = XG[0]
    lines += ['', f'## By class at X={bx}pt (earliest flip) — flip mean abs (ticks)',
              '| class | never-bail | flip@%d |' % bx, '|---|---|---|']
    for c in ['good_clean', 'good_dipped', 'wrong', 'dead_band']:
        lines.append(f'| {c} | {nb[cls==c].mean():+.1f} | {per_class[bx][c]:+.1f} |')
    lines += ['',
              'Read: flip should WIN on wrong (ride the real leg) and LOSE on good_dipped/'
              'good_clean (whipsawed out of recoveries/wins). Net beats never-bail only if the '
              'wrong-class capture > the good-class whipsaw. Earlier X (smaller) = flip sooner '
              '= less eaten before the ride. * = CI excludes 0. Caveat holds if Δ rises as X falls.']
    out = os.path.join(HERE, '..', 'reports', 'wrongdir', 'flip_when_wrong.md')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
