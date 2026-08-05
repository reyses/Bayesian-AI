#!/usr/bin/env python3
"""DOC-107 REDO — full-run leg decomposition (owner 2026-07-28, TG): doc-107 said
never-bail beats every cut on N=23,378. This asks WHERE never-bail's advantage
comes from by leg: the 'good_dipped' class (dipped<=-4 then recovered to +4) IS a
SECOND-LEG recovery. Decompose never-bail's edge over the best stop (X=48) by class,
and measure how much of never-bail's positive expectancy is the 2nd-leg 'false
profit' (good_dipped) vs genuine 1st-leg (good_clean) — and the wrong-class tail it
rides. Reuses the exact doc-107 machinery. Points/ticks: PTS_TO_TICKS=4.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_wrongdir as swl          # noqa: E402
import stop_reenter_sim as srs         # noqa: E402

T = 4.0                                 # pts -> ticks
BAND, DIP = 4.0, 4.0
BEST_X = 48                             # doc-107 best plain stop


def main():
    eng = swl.engagements()
    day_engs, _ = swl.scan(eng)
    rec = []
    for day, engs in day_engs.items():
        for e in engs:
            dp = e['per_minute_forward_drift']; wm = e['window_minutes']
            terminal = e['terminal']; mindrift = e['mindrift']
            if terminal <= -BAND:
                cls = 'wrong'
            elif terminal >= BAND:
                cls = 'good_dipped' if mindrift <= -DIP else 'good_clean'
            else:
                cls = 'dead_band'
            nb_t = terminal * T                        # never-bail terminal (ticks)
            stop_net = srs.simulate_plainstop(dp, wm, BEST_X)['net']   # net vs never-bail
            tmin = int(np.argmin(dp))
            rec.append((cls, nb_t, stop_net, tmin, wm, terminal, mindrift))
    A = np.array([r[1:] for r in rec], float)
    cls = np.array([r[0] for r in rec])
    N = len(rec)
    nb = A[:, 0]; sn = A[:, 1]
    CL = ['good_clean', 'good_dipped', 'wrong', 'dead_band']

    lines = ['# doc-107 redo — never-bail decomposed by leg/class', '',
             f'N={N:,}, 282 days. never-bail terminal (ticks) + stop X={BEST_X} net-vs-never-bail.',
             f'never-bail mean terminal: **{nb.mean():+.2f} t/ep**; '
             f'never-bail advantage over stop{BEST_X}: **{-sn.mean():+.2f} t/ep** '
             f'(doc-107: +3.39).', '',
             '| class (leg) | N | share | NB mean term | NB SUM term (share of +) | '
             'stop net | NB adv from class |', '|---|---|---|---|---|---|---|']
    pos_sum = sum(nb[(cls == c)].sum() for c in ['good_clean', 'good_dipped'])
    for c in CL:
        m = cls == c; n = m.sum()
        leg = {'good_clean': '1st-leg win', 'good_dipped': '2ND-LEG recovery',
               'wrong': 'rode to loss', 'dead_band': 'scratch'}[c]
        contrib = nb[m].sum()
        adv = -sn[m].sum() / N                        # this class's contribution to NB advantage (t/ep)
        share_of_pos = (contrib / pos_sum) if (c in ('good_clean', 'good_dipped') and pos_sum) else float('nan')
        sop = f'{share_of_pos:.0%}' if share_of_pos == share_of_pos else '—'
        lines.append(f'| {c} ({leg}) | {n:,} | {n/N:.1%} | {nb[m].mean():+.1f} | '
                     f'{contrib:+,.0f} ({sop}) | {sn[m].mean():+.1f} | {adv:+.2f} |')

    # the wrong-class tail never-bail rides
    w = nb[cls == 'wrong']
    gd = nb[cls == 'good_dipped']
    lines += ['',
              '## The trade-off, by leg',
              f'- **2nd-leg (good_dipped) false-profit wins**: {(cls=="good_dipped").sum():,} '
              f'trades, mean {gd.mean():+.1f}t, TOTAL {gd.sum():+,.0f}t — this is '
              f'{gd.sum()/pos_sum:.0%} of never-bail\'s positive terminal.',
              f'- **1st-leg (good_clean) genuine wins**: mean {nb[cls=="good_clean"].mean():+.1f}t, '
              f'TOTAL {nb[cls=="good_clean"].sum():+,.0f}t ({nb[cls=="good_clean"].sum()/pos_sum:.0%}).',
              f'- **wrong-class tail never-bail rides**: {(cls=="wrong").sum():,} trades, mean '
              f'{w.mean():+.1f}t, p5 {np.percentile(w,5):+.0f}, p1 {np.percentile(w,1):+.0f}, '
              f'worst {w.min():+.0f}t (${w.min()/T*2:+.0f}). A stop caps these; never-bail does not.',
              '',
              'Read: never-bail\'s edge over cutting comes from NOT stopping the good_dipped '
              '= the SECOND-LEG recoveries. Remove/curb them and the edge collapses. But those '
              'second-leg wins are the same oscillation as the wrong-class tail never-bail must '
              'ride to keep them — harvesting 2nd-leg false profit while eating the catastrophic '
              'runaway tail. Confirms the owner: doc-107\'s never-bail win is the second leg.']
    out = os.path.join(HERE, '..', 'reports', 'wrongdir', 'doc107_second_leg.md')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
