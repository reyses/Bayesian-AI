#!/usr/bin/env python
"""Does OBSERVED oscillation predict the next one? (chop-begets-chop test)

WHY (owner, 2026-08-01): "we needed to enter and wait until we hit the
oscillation point and then start osilation harnessing."

This is a different claim from the sigma-extreme entry already killed
(`sigma_fade_test.py`, null over 603 sessions). That test entered at an extreme
unconditionally. This one refuses to PREDICT the regime and waits for it to
IDENTIFY ITSELF — trade only after K oscillations have already been observed.

That matters because oscillator-vs-runaway discrimination has been stuck near
0.57 AUC for months. Waiting for confirmation sidesteps the prediction problem
entirely. The obvious counter-hypothesis, stated up front: chop is confirmable
only in hindsight, so by the time K traverses are counted the K+1'th may be no
likelier than base rate — or LESS likely, because the range is used up and a
breakout is what ends it.

DEFINITIONS (edge-triggered throughout; level-triggered counting inflated N ~10x
in a prior bug):
  z         = (close − cubic endpoint) / residual sigma, on 5s, deployed spec
  traverse  = z travels from ≥ +BAND all the way through to ≤ −BAND (or reverse)
  K         = traverses completed in the LOOKBACK_S before this one
  trade     = at a traverse completion, fade toward the opposite band
  outcome   = does the next traverse complete before STOP_PT moves against us?

STOP_PT is deliberately 20pt — outside the fakeout distribution measured in the
ground-truth work (p75 required room 24.8pt), so the fail-safe stays a
fail-safe instead of quietly becoming the exit mechanism.

Reads out P(complete) and EV vs K. Rising ⇒ chop begets chop. Flat ⇒ waiting
buys nothing. Falling ⇒ waiting is actively wrong. All three are informative.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/oscillation_harvest_test.py --exclude 2024_09_16
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import cubic_regression as _cub                                    # noqa: E402

D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                   'oscillation_harvest.md')

CUBIC_5S_WINDOW = 90     # 7.5min, deployed NT8 spec
SIGMA_MIN = 20           # residual-sigma lookback, minutes
BAND = 1.5               # sigma band defining an extreme
LOOKBACK_S = 1800        # 30min window for counting prior traverses
STOP_PT = 20.0           # outside the fakeout distribution (p75 room = 24.8pt)
STOP_MODE = 'fixed'      # 'fixed' | 'sigma'  (--stop-mode)
STOP_SIGMA = 3.0         # stop = STOP_SIGMA x residual sigma, when mode='sigma'
MAX_HOLD_S = 3600        # give the traverse an hour, then mark it timeout
RTH_FROM, RTH_TO = 570, 960
FRICTION_PT = 0.89
PT_USD = 2.0
BOOT = 4000
SEED = 11
MIN_BARS = 2000


def scan_day(path):
    d = pd.read_parquet(path)[['timestamp', 'open', 'high', 'low', 'close']]
    if len(d) < MIN_BARS:
        return []
    ts = d['timestamp'].to_numpy()
    c = d['close'].to_numpy(); hi = d['high'].to_numpy(); lo = d['low'].to_numpy()
    cub, _, _ = _cub.rolling(c, CUBIC_5S_WINDOW, 5)
    res = c - cub
    sig = pd.Series(res).rolling(SIGMA_MIN * 12, min_periods=5 * 12).std().to_numpy()
    z = np.where(sig > 0, res / sig, np.nan)
    e = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    etm = (e.hour * 60 + e.minute).to_numpy()

    # side = which band we last touched; forward-filled. A traverse completes
    # exactly when this flips.
    side = np.where(z >= BAND, 1, np.where(z <= -BAND, -1, 0))
    ff = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flip = np.flatnonzero((~np.isnan(ff[1:])) & (~np.isnan(ff[:-1]))
                          & (ff[1:] != ff[:-1])) + 1
    flip = flip[(etm[flip] >= RTH_FROM) & (etm[flip] < RTH_TO)]
    if len(flip) < 2:
        return []

    rows = []
    for n, i in enumerate(flip):
        arrived = ff[i]                 # band just reached: -1 = low, +1 = high
        sgn = 1 if arrived < 0 else -1  # fade toward the opposite band
        # K = completed traverses inside the lookback, strictly before this one
        K = int(((ts[flip[:n]] >= ts[i] - LOOKBACK_S)).sum())
        p0 = c[i]
        w = (ts > ts[i]) & (ts <= ts[i] + MAX_HOLD_S)
        if w.sum() < 12:
            continue
        zz = z[w]; hh = hi[w]; ll = lo[w]; cc = c[w]
        # first bar the opposite band is reached
        tgt = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        j_win = tgt[0] if len(tgt) else None
        # first bar the stop is breached. SIGMA MODE (owner 2026-08-01:
        # "either a careful watcher or a sigma stop") scales the stop with the
        # same sigma the target scales with — so the geometry is preserved
        # instead of the stop staying fixed while the band shrinks.
        stop_pt = (STOP_PT if STOP_MODE == 'fixed'
                   else max(2.0, STOP_SIGMA * float(sig[i])))
        adv = (ll <= p0 - stop_pt) if sgn > 0 else (hh >= p0 + stop_pt)
        stp = np.flatnonzero(adv)
        j_stop = stp[0] if len(stp) else None
        if j_win is not None and (j_stop is None or j_win < j_stop):
            out, pts = 'complete', float((cc[j_win] - p0) * sgn)
        elif j_stop is not None:
            out, pts = 'runaway', -stop_pt
        else:
            out, pts = 'timeout', float((cc[-1] - p0) * sgn)
        rows.append(dict(day=os.path.basename(path)[:-8], K=K, outcome=out,
                         pts=pts, net=pts - FRICTION_PT))
    return rows


def ci(x):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    if len(x) < 5:
        return float('nan'), float('nan')
    s = [rng.choice(x, len(x), replace=True).mean() for _ in range(BOOT)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--stop-mode', default='fixed', choices=['fixed', 'sigma'])
    ap.add_argument('--stop-sigma', type=float, default=3.0)
    a = ap.parse_args()
    global STOP_MODE, STOP_SIGMA
    STOP_MODE, STOP_SIGMA = a.stop_mode, a.stop_sigma
    days = sorted(f for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rows = []
    for f in tqdm(days, desc='oscillation'):
        try:
            rows += scan_day(os.path.join(D5, f))
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows'); return

    L = ['# Does OBSERVED oscillation predict the next one?', '',
         'Wait for the regime to identify itself, then harvest — rather than '
         'predicting it (oscillator/runaway discrimination is stuck ~0.57 AUC).',
         '',
         f'Traverse = z crosses from ±{BAND:g}σ through to the opposite band '
         f'(cubic 5s w{CUBIC_5S_WINDOW}, σ over {SIGMA_MIN}min), edge-triggered, '
         f'RTH only. At each completion: K = traverses in the prior '
         + f'{LOOKBACK_S // 60}min, fade toward the opposite band, '
         + (f'stop {STOP_PT:g}pt (outside the fakeout distribution), '
            if STOP_MODE == 'fixed'
            else f'stop {STOP_SIGMA:g}x sigma (scales with the band), ')
         + f'max hold {MAX_HOLD_S // 60}min.',
         f'Friction `{FRICTION_PT}pt` charged per attempt. '
         f'Excluded: {", ".join(a.exclude) or "none"}.',
         '', f'Sessions: **{df["day"].nunique()}** · attempts: **{len(df)}**', '',
         '## Outcome and edge vs K (prior observed traverses)', '',
         '| K | N | complete | runaway | timeout | mean net (pt) | 95% CI | $/trade |',
         '|---|---|---|---|---|---|---|---|']
    groups = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]
    for k0, k1 in groups:
        g = df[(df['K'] >= k0) & (df['K'] <= k1)]
        if len(g) < 30:
            continue
        net = g['net'].to_numpy()
        lo, hi = ci(net)
        lab = f'{k0}' if k0 == k1 else f'{k0}–{k1 if k1 < 99 else "+"}'
        L.append(f'| {lab} | {len(g)} | {(g["outcome"] == "complete").mean():.1%} | '
                 f'{(g["outcome"] == "runaway").mean():.1%} | '
                 f'{(g["outcome"] == "timeout").mean():.1%} | '
                 f'`{net.mean():+.2f}` | `[{lo:+.2f}, {hi:+.2f}]` | '
                 f'`${net.mean() * PT_USD:+.2f}` |')

    net_all = df['net'].to_numpy()
    lo, hi = ci(net_all)
    L += ['', f'**All attempts pooled:** N={len(df)}, '
              f'complete {(df["outcome"] == "complete").mean():.1%}, '
              f'mean net `{net_all.mean():+.2f}pt` 95% CI `[{lo:+.2f}, {hi:+.2f}]` '
              f'→ {"NOT significant" if lo <= 0 <= hi else "significant"}', '']

    # monotonicity: does waiting actually buy anything?
    ks = sorted(df['K'].unique())
    tr = [(k, (df[df['K'] == k]['outcome'] == 'complete').mean(),
           int((df['K'] == k).sum())) for k in ks if (df['K'] == k).sum() >= 50]
    if len(tr) >= 3:
        L += ['## P(complete) by exact K', '', '| K | N | P(complete) |',
              '|---|---|---|']
        L += [f'| {k} | {n} | {p:.1%} |' for k, p, n in tr]
        a0, a1 = tr[0][1], tr[-1][1]
        L += ['', f'K={tr[0][0]} → K={tr[-1][0]}: `{a0:.1%}` → `{a1:.1%}` '
                  f'({(a1 - a0) * 100:+.1f}pp). '
                  f'{"Chop begets chop." if a1 - a0 > 0.03 else "Waiting buys nothing." if abs(a1 - a0) <= 0.03 else "Chop gets USED UP — waiting is actively wrong."}',
              '']

    out = OUT if STOP_MODE == 'fixed' else OUT.replace('.md', '_sigmastop.md')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
