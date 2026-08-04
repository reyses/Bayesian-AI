#!/usr/bin/env python
"""Does the 1s acceleration-inflection stamp find real leg tops out of sample?

WHY (owner, 2026-08-01): "we need to causally measure that inflection point."
On ONE live leg, the rule "after R consecutive seconds of positive acceleration,
stamp the first non-positive one" fired exactly once — at the true top, with
zero false alarms, at R=6. That is a single observation with a hand-picked
parameter, i.e. curve-fitting until proven otherwise. This is the proof attempt.

WHAT IS AND IS NOT BEING CLAIMED. This is NOT a prediction test. The same probe
showed acceleration turns AT the extreme, not before it, and the naive
"acc<0 while vel>0" variant fired 31s early with 13pt still to come. The claim
under test is only that the stamp RECOGNISES a leg top at (or within a second or
two of) the moment it forms — earlier than the stall detector, which needs N
seconds of no-new-extreme before it can speak.

GROUND TRUTH: close-based zigzag on 1s (a close-based zigzag cannot produce the
same-bar pivot artefact that invalidated the earlier high/low tip-to-tip work;
see the 2026-08-01 journal). Self-tested on a synthetic sawtooth before use.

SCORING, per (K, R):
  precision  fires landing within TOL seconds of a true pivot / all fires
  recall     true pivots stamped / all true pivots
  timing     signed seconds from pivot to stamp (negative = early)
  fires/hr   absolute alarm rate, because a stamp nobody can act on is noise

A rule is only interesting if precision is high AND fires/hr is livable. Recall
matters less: missing some tops is survivable, crying wolf is not.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/inflection_stamp_test.py --days 60
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
D1 = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                   'inflection_stamp.md')

ZIGZAG_R = 15.0        # pt; validated threshold at 1s (median bar range 1.00pt)
TOL_S = 2              # seconds either side of the pivot that count as a hit
RTH_FROM, RTH_TO = 570, 960
K_GRID = (3, 5, 8)     # velocity/acceleration window, seconds
R_GRID = (3, 4, 5, 6, 8)
SEED = 11


def zz(c, R):
    """Close-based zigzag; returns [(index, price), ...] alternating extremes."""
    if len(c) < 3:
        return []
    piv = []
    d = 1 if c[1] >= c[0] else -1
    ext, ei = c[0], 0
    for i in range(1, len(c)):
        if d > 0:
            if c[i] > ext:
                ext, ei = c[i], i
            elif ext - c[i] >= R:
                piv.append((ei, ext)); d = -1; ext, ei = c[i], i
        else:
            if c[i] < ext:
                ext, ei = c[i], i
            elif c[i] - ext >= R:
                piv.append((ei, ext)); d = 1; ext, ei = c[i], i
    piv.append((ei, ext))
    return piv


def _selftest():
    t = np.concatenate([np.arange(0, 21, 1.), np.arange(19, -1, -1.),
                        np.arange(0, 21, 1.)])
    p = zz(t, 8.0)
    assert len(p) == 3, f'zigzag self-test FAILED: {p}'
    return True


def stamps(c, K, R):
    """Causal inflection stamps. Returns (up_idx, dn_idx).

    up   = end of a rising impulse: R consecutive seconds of acc>0, then acc<=0
    down = mirror, for lows. Strictly causal — index i uses only c[:i+1].
    """
    n = len(c)
    vel = np.full(n, np.nan)
    vel[K:] = c[K:] - c[:-K]
    acc = np.full(n, np.nan)
    acc[2 * K:] = vel[2 * K:] - vel[K:-K]
    up, dn, ru, rd = [], [], 0, 0
    for i in range(2 * K, n):
        a = acc[i]
        if not np.isfinite(a):
            continue
        if a > 0:
            ru += 1
        else:
            if ru >= R:
                up.append(i)
            ru = 0
        if a < 0:
            rd += 1
        else:
            if rd >= R:
                dn.append(i)
            rd = 0
    return up, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    a = ap.parse_args()
    _selftest()

    days = sorted(f[:-8] for f in os.listdir(D1) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())

    acc = {(K, R): dict(hit=0, fire=0, piv=0, err=[]) for K in K_GRID for R in R_GRID}
    hours = 0.0
    for d in tqdm(days, desc='inflection'):
        try:
            f = pd.read_parquet(os.path.join(D1, f'{d}.parquet'))[['timestamp', 'close']]
        except Exception:
            continue
        t = f['timestamp'].to_numpy()
        e = pd.to_datetime(t, unit='s', utc=True).tz_convert('America/New_York')
        m = (e.hour * 60 + e.minute).to_numpy()
        k = (m >= RTH_FROM) & (m < RTH_TO)
        if k.sum() < 10000:
            continue
        c = f['close'].to_numpy()[k]
        hours += k.sum() / 3600.0
        piv = zz(c, ZIGZAG_R)
        if len(piv) < 3:
            continue
        # alternate: a pivot is a TOP if the next extreme is lower
        tops = [p[0] for i, p in enumerate(piv[:-1]) if piv[i + 1][1] < p[1]]
        bots = [p[0] for i, p in enumerate(piv[:-1]) if piv[i + 1][1] > p[1]]
        for K in K_GRID:
            for R in R_GRID:
                up, dn = stamps(c, K, R)
                st = acc[(K, R)]
                st['fire'] += len(up) + len(dn)
                st['piv'] += len(tops) + len(bots)
                for fires, truth in ((up, tops), (dn, bots)):
                    if not truth:
                        continue
                    tr = np.array(truth)
                    for i in fires:
                        j = int(np.argmin(np.abs(tr - i)))
                        if abs(tr[j] - i) <= TOL_S:
                            st['hit'] += 1
                            st['err'].append(int(i - tr[j]))

    L = ['# Causal inflection stamp — out-of-sample validation', '',
         'Rule: after **R consecutive seconds of positive acceleration**, stamp '
         'the first non-positive one (mirrored for lows). Velocity and '
         'acceleration on 1s closes over K-second windows. Strictly causal.', '',
         'NOT a prediction test — the same probe showed acceleration turns AT '
         'the extreme, not before it. The claim under test is only that the '
         'stamp RECOGNISES a leg top within '
         f'±{TOL_S}s of its formation, earlier than a stall detector can.', '',
         f'Ground truth: close-based zigzag R={ZIGZAG_R:g}pt on 1s '
         f'(self-tested on a synthetic sawtooth). Sessions: **{len(days)}**, '
         f'RTH only, ~{hours:,.0f} hours of tape.',
         f'Excluded: {", ".join(a.exclude)}. The live leg that motivated this '
         'is NOT in the sample.', '',
         '| K | R | fires | fires/hr | precision | recall | median timing |',
         '|---|---|---|---|---|---|---|']
    best = None
    for K in K_GRID:
        for R in R_GRID:
            st = acc[(K, R)]
            if not st['fire']:
                continue
            prec = st['hit'] / st['fire']
            rec = st['hit'] / max(st['piv'], 1)
            med = np.median(st['err']) if st['err'] else float('nan')
            L.append(f'| {K} | {R} | {st["fire"]:,} | {st["fire"] / max(hours, 1):.1f} | '
                     f'`{prec:.3f}` | `{rec:.3f}` | '
                     f'{med:+.0f}s |' if np.isfinite(med) else
                     f'| {K} | {R} | {st["fire"]:,} | {st["fire"] / max(hours, 1):.1f} | '
                     f'`{prec:.3f}` | `{rec:.3f}` | · |')
            if best is None or prec > best[0]:
                best = (prec, K, R, st['fire'] / max(hours, 1), rec)
    if best:
        prec, K, R, fph, rec = best
        L += ['', f'**Best precision: K={K}, R={R} → `{prec:.3f}` '
                  f'({fph:.1f} fires/hr, recall {rec:.3f}).**', '']
        if prec < 0.15:
            L.append('That is close to what random stamping would achieve. '
                     '**The rule does not survive out of sample** — the live '
                     'leg was a coincidence, exactly as the single-observation '
                     'caveat warned.')
        elif prec < 0.4:
            L.append('Better than chance but far too noisy to act on: most '
                     'fires are not leg tops. Usable at best as one input among '
                     'several, never as a standalone stamp.')
        else:
            L.append('Materially better than chance. Worth wiring into the '
                     'watcher as a fifth trigger — with the fire rate stated '
                     'so the alarm burden is explicit.')
    # a random-stamp baseline, so "precision" has a scale
    L += ['', '## Baseline', '',
          'With ~1 pivot per leg and TOL of ±%ds, a stamp fired at a uniformly '
          'random second would hit at roughly (2·TOL+1)·pivots/seconds — '
          'typically well under 0.02. Compare every precision above against '
          'that, not against 0.5.' % TOL_S, '']
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
