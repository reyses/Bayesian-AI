#!/usr/bin/env python3
"""TIME-GATED FLIP (owner 2026-07-28, TG): good_dipped recover FAST (median 2m,
p95 7m from the dip). A runaway is a dip that does NOT bounce in that window. So:
let a dip WORK for T minutes; if it hasn't recovered by then, FLIP (it exceeded the
good_dipped recovery horizon). This uses TIME, not depth (depth overlaps too much).

Rule per engagement: first dip = first min dp<=-DSMALL. If it returns to breakeven
within T min -> HOLD (never-bail). Else -> FLIP at first_dip+T. Sweep T x DSMALL.
Absolute ticks, friction 2.4t/RT. reports/wrongdir/time_gated_flip.md
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_wrongdir as swl          # noqa: E402

T_TICK, FR = 4.0, 2.4
BAND, DIP = 4.0, 4.0
TGRID = [3, 5, 7, 10, 15]              # let-it-work horizon (min)
DSMALL = [4, 8]                        # dip that arms the timer (pts)


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
            rows.append((cls, dp, wm))
    N = len(rows)
    nb = np.array([r[1][r[2]] * T_TICK - FR for r in rows])
    cls = np.array([r[0] for r in rows])

    lines = ['# Time-gated flip — let the dip work T min, flip if not recovered', '',
             f'N={N:,}. never-bail mean abs = {nb.mean():+.2f} t/ep. Arm at dip<=-DSMALL, '
             f'flip at +T if still underwater.', '',
             '| DSMALL | T (min) | flip mean | Δ vs never-bail | %flipped | wrong-flip% |',
             '|---|---|---|---|---|---|']
    best = None
    results = {}
    for ds in DSMALL:
        for T in TGRID:
            out = np.empty(N); flipped = 0; wrong_flipped = 0
            for i, (c, dp, wm) in enumerate(rows):
                below = np.where(dp[:wm + 1] <= -ds)[0]
                if not len(below):
                    out[i] = dp[wm] * T_TICK - FR; continue
                fd = int(below[0])
                # recovered within T after first dip?
                hi = min(fd + T, wm)
                rec = bool((dp[fd + 1:hi + 1] >= 0).any()) if hi > fd else False
                if rec:
                    out[i] = dp[wm] * T_TICK - FR                 # held (never-bail)
                else:
                    tf = hi; flipped += 1                         # flip at fd+T
                    out[i] = (2 * dp[tf] - dp[wm]) * T_TICK - 2 * FR
                    if c == 'wrong':
                        wrong_flipped += 1
            d = out.mean() - nb.mean()
            wf = wrong_flipped / flipped if flipped else float('nan')
            results[(ds, T)] = out
            lines.append(f'| {ds} | {T} | {out.mean():+.2f} | {d:+.2f} | {flipped/N:.0%} | {wf:.0%} |')
            if best is None or out.mean() > best[1]:
                best = ((ds, T), out.mean(), out)

    (bds, bt), bm, bout = best
    lines += ['', f'## Best: DSMALL={bds}, T={bt}min (mean {bm:+.2f} vs never-bail {nb.mean():+.2f})',
              '| class | never-bail | best time-flip |', '|---|---|---|']
    for c in ['good_clean', 'good_dipped', 'wrong', 'dead_band']:
        lines.append(f'| {c} | {nb[cls==c].mean():+.1f} | {bout[cls==c].mean():+.1f} |')
    lines += ['',
              'Read: if the best time-flip beats never-bail, the recovery-TIME horizon '
              'discriminates good_dipped (recover fast, held) from runaways (never bounce, '
              'flipped) where depth could not. wrong-flip% = purity of what we flip. If it '
              'still loses, good_dipped fast-recovery overlaps runaways too much even in time.']
    out = os.path.join(HERE, '..', 'reports', 'wrongdir', 'time_gated_flip.md')
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
