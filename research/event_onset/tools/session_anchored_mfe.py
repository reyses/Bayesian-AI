"""Legs by RECURSIVE PARTITION from the maintenance anchor (owner 2026-08-04:
"measure from maintenance time ... then measure the displacement from that
anchor at the 24ish hours, then we start partitioning until we pick all of
them cuz oscillations are 50+ p").

Everything before this measured legs BOTTOM-UP (zigzag threshold, sliding
window) and had to guess a scale. This is TOP-DOWN and needs no threshold:

  anchor  = session open after the CME maintenance break (18:00 ET)
  level 0 = the session's MFE up and MFE down FROM that anchor
  split   = at the session extreme, which cuts the span in two
  recurse = into each side, re-anchored at its own start

Legs fall out at every scale with their own anchor, and the recursion stops
when a segment can no longer contain an oscillation (< MIN_SPAN pt) or is
too short in time. No threshold is imposed; the tape decides the scale.

  python research/event_onset/tools/session_anchored_mfe.py
"""
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = '/media/moi/WindowsCode/Bayesian-AI'
BARS = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
OUT = os.path.join(REPO, 'research', 'event_onset', 'reports')
MIN_SPAN = 50.0        # owner: "oscillations are 50+ p" — stop below this
MIN_SECS = 120         # a segment shorter than this cannot hold a leg
MAX_DEPTH = 6


def partition(ts, h, l, c, a, b, depth, anchor_px, out, day):
    """[a,b] segment, anchor_px = price at its start.

    Splits at MAXIMUM DEVIATION FROM THE CHORD (Douglas-Peucker), not at the
    first extreme. Two earlier attempts split at an extreme, but a child
    re-anchored at a split BEGINS at its own extreme, so the split point kept
    landing 2-32s into a 20,000s segment and the hierarchy never shrank.
    Chord deviation is scale-free and always finds the dominant swing.
    """
    if b - a < MIN_SECS or depth > MAX_DEPTH or b - a < 3:
        return
    seg_h, seg_l = h[a:b + 1], l[a:b + 1]
    mfe_up = float(seg_h.max() - anchor_px)
    mfe_dn = float(anchor_px - seg_l.min())
    span = float(seg_h.max() - seg_l.min())
    # chord from (a, c[a]) to (b, c[b]); deviation of every close from it
    n = b - a
    chord = c[a] + (c[b] - c[a]) * (np.arange(n + 1) / max(n, 1))
    dev = c[a:b + 1] - chord
    i_up = a + int(dev.argmax())
    i_dn = a + int(dev.argmin())
    split = i_up if abs(dev.max()) >= abs(dev.min()) else i_dn
    dom = 'up' if split == i_up else 'dn'
    out.append(dict(day=day, depth=depth, t0=int(ts[a]), t1=int(ts[b]),
                    secs=int(ts[b] - ts[a]), anchor=anchor_px,
                    mfe_up=mfe_up, mfe_dn=mfe_dn, span=span, dom=dom,
                    dev=float(abs(dev).max()), t_ext=int(ts[split]),
                    secs_to_ext=int(ts[split] - ts[a])))
    if span < MIN_SPAN or not (a < split < b):
        return
    if split - a > MIN_SECS:
        partition(ts, h, l, c, a, split, depth + 1, float(c[a]), out, day)
    if b - split > MIN_SECS:
        partition(ts, h, l, c, split, b, depth + 1, float(c[split]), out, day)


def day_tree(path):
    d = pd.read_parquet(path)
    ts = d['timestamp'].to_numpy()
    if len(ts) < 3600:
        return []
    h, l, c = (d[x].to_numpy() for x in ('high', 'low', 'close'))
    out = []
    partition(ts, h, l, c, 0, len(ts) - 1, 0, float(c[0]), out,
              os.path.basename(path)[:-8])
    return out


if __name__ == '__main__':
    days = [p for p in sorted(glob.glob(os.path.join(BARS, '2025_0[1-6]*.parquet')))
            if len(os.path.basename(p)) == 18]
    rows = []
    for p in tqdm(days, desc='sessions'):
        rows += day_tree(p)
    R = pd.DataFrame(rows)
    R.to_parquet(os.path.join(OUT, 'session_tree.parquet'), index=False)
    print(f'\n{len(R):,} segments across {R["day"].nunique()} sessions '
          f'({len(R)/R["day"].nunique():.1f} per session)\n')
    print(f'{"depth":>6} {"n":>7} {"/session":>9} {"med span":>9} '
          f'{"med secs":>9} {"med MFE up":>11} {"med MFE dn":>11}')
    for dep, g in R.groupby('depth'):
        print(f'{dep:>6} {len(g):>7,} {len(g)/R["day"].nunique():>9.1f} '
              f'{g["span"].median():>9.1f} {g["secs"].median():>9.0f} '
              f'{g["mfe_up"].median():>11.1f} {g["mfe_dn"].median():>11.1f}')
    print(f'\nsegments still >= {MIN_SPAN:g}pt at max depth: '
          f'{int((R[R["depth"]==R["depth"].max()]["span"]>=MIN_SPAN).sum())}')
    print(f'periodicity — median seconds to the first extreme, by depth:')
    for dep, g in R.groupby('depth'):
        print(f'   depth {dep}: {g["secs_to_ext"].median():6.0f}s of '
              f'{g["secs"].median():6.0f}s '
              f'({g["secs_to_ext"].median()/max(g["secs"].median(),1):.0%} in)')
