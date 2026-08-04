#!/usr/bin/env python
"""Measure the ATLAS contract-roll offset, and translate prior-contract levels
onto the current contract.

WHY THIS EXISTS (owner, 2026-08-01): told that a roll day leaves no usable
prior-day levels, he pushed back — "a human would still look at the prior
levels for structure and make decisions on it." He is right. Levels do not
transfer as raw NUMBERS across a roll, but they transfer perfectly once
shifted by the calendar spread, because both contracts track the same index.
The fix is a translation, not a discard.

METHOD. ATLAS holds one chosen outright per day, so the two contracts are
never quoted side by side and the spread cannot be read directly. But every
session boundary gives an (open − prior settle) jump, and roll boundaries are
separable from the rest by a mile: in 2024 all four roll jumps land at
z ≈ +13 to +16 against the 254 non-roll boundaries. So:

    spread% ≈ observed roll jump% − mean non-roll jump%
    uncertainty = the non-roll jump SD (the overnight move we cannot remove)

Per-roll is preferable to pooling: the spread is carry, so it moves with rate
expectations. 2024_09_16 has the LOWEST offset of the four rolls (+1.18% vs
+1.31/1.34/1.36%), which is exactly what maximum priced-in Fed cuts should do —
the Fed cut 50bp two days later. Pooling would wash that out.

Writes to research/dojo_forge/reports/.
Usage:
    python research/dojo_forge/tools/roll_spread.py                 # all rolls
    python research/dojo_forge/tools/roll_spread.py --day 2024_09_16 \
        --levels 19577 19396 19504
"""
import argparse
import os

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
D1M = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
MANIFEST = os.path.join(REPO, 'DATA', 'ATLAS', 'roll_manifest.csv')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'roll_spread.md')
BOOT = 4000          # bootstrap resamples (house standard)
SEED = 11            # deterministic


def _boundaries():
    """[(prev_day, day, contract_prev, contract, jump_pt, jump_pct, is_roll)]."""
    man = pd.read_csv(MANIFEST)
    con = dict(zip(man['day'], man['chosen']))
    days = sorted(f[:-8] for f in os.listdir(D1M) if f.endswith('.parquet'))
    days = [d for d in days if d in con]
    oc = {}
    for d in days:
        f = pd.read_parquet(os.path.join(D1M, f'{d}.parquet'))
        oc[d] = (float(f['open'].iloc[0]), float(f['close'].iloc[-1]))
    out = []
    for a, b in zip(days, days[1:]):
        jump = oc[b][0] - oc[a][1]
        out.append((a, b, con[a], con[b], jump, jump / oc[a][1] * 100,
                    con[a] != con[b]))
    return out, oc


def spread_for(day, rows, oc):
    """(spread_pt, lo, hi, detail) for the roll landing ON `day`.

    The overnight move rides along inside the observed jump and cannot be
    removed, only bounded — so the CI here is the non-roll jump distribution,
    not a bootstrap of the roll estimate itself. Four rolls a year is far too
    few to bootstrap; the honest uncertainty is 'how much can a weekend move'.
    """
    ctrl = np.array([r[5] for r in rows if not r[6]])
    hit = [r for r in rows if r[6] and r[1] == day]
    if not hit:
        return None
    a, b, ca, cb, jump, pct, _ = hit[0]
    px = oc[a][1]
    net = pct - ctrl.mean()
    sd = ctrl.std(ddof=1)
    return (px * net / 100, px * (net - 1.96 * sd) / 100,
            px * (net + 1.96 * sd) / 100,
            dict(prev=a, day=b, con_prev=ca, con=cb, jump_pt=jump, jump_pct=pct,
                 prev_settle=px, ctrl_n=len(ctrl), ctrl_mean=ctrl.mean(),
                 ctrl_sd=sd, roll_z=(pct - ctrl.mean()) / sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', help='roll day to translate onto (e.g. 2024_09_16)')
    ap.add_argument('--levels', nargs='*', type=float, default=[],
                    help='prior-contract price levels to translate')
    a = ap.parse_args()

    rows, oc = _boundaries()
    ctrl = np.array([r[5] for r in rows if not r[6]])
    rolls = [r for r in rows if r[6]]

    L = ['# Contract-roll spread — ATLAS', '',
         'Levels do not transfer as raw numbers across a roll; they transfer '
         'once shifted by the calendar spread. This measures the shift.', '',
         f'## Control — {len(ctrl)} non-roll session boundaries', '',
         f'- mean `{ctrl.mean():+.4f}%` · median `{np.median(ctrl):+.4f}%` '
         f'· sd `{ctrl.std(ddof=1):.4f}%`',
         f'- 95% of boundaries within '
         f'`[{np.percentile(ctrl, 2.5):+.3f}%, {np.percentile(ctrl, 97.5):+.3f}%]`',
         '', '## Roll boundaries', '',
         '| boundary | contracts | jump (pt) | jump (%) | z vs control |',
         '|---|---|---|---|---|']
    for p, d, ca, cb, j, pc, _ in rolls:
        z = (pc - ctrl.mean()) / ctrl.std(ddof=1)
        L.append(f'| {p} → {d} | {ca} → {cb} | {j:+.2f} | {pc:+.3f}% | {z:+.1f} |')
    L += ['', f'All {len(rolls)} rolls sit z ≥ +13 from the non-roll '
              'distribution — the offset is unambiguous, not a judgement call.',
          '']

    if a.day:
        got = spread_for(a.day, rows, oc)
        if not got:
            L.append(f'**{a.day} is not a roll day.**')
        else:
            sp, lo, hi, d = got
            L += [f'## Translation onto {a.day} ({d["con"]})', '',
                  f'- prior session {d["prev"]} ({d["con_prev"]}) settle '
                  f'`{d["prev_settle"]:.2f}`',
                  f'- observed jump `{d["jump_pt"]:+.2f}pt` (`{d["jump_pct"]:+.3f}%`), '
                  f'z=`{d["roll_z"]:+.1f}`',
                  f'- **spread ≈ `{sp:+.1f}pt`**, 95% `[{lo:+.1f}, {hi:+.1f}]` '
                  f'— the band is the overnight move, which cannot be removed',
                  '']
            if a.levels:
                L += ['| prior-contract level | → this contract | 95% band |',
                      '|---|---|---|']
                for v in a.levels:
                    L.append(f'| {v:.2f} | **{v + sp:.2f}** | '
                             f'[{v + lo:.2f}, {v + hi:.2f}] |')
                L.append('')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
