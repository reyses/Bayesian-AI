#!/usr/bin/env python
"""Is a runaway visible in the FIRST SECONDS, at 1s resolution?

WHY (owner, 2026-08-01): after seeing that the mechanical band-harvest loses
because 21.5% of attempts run away, he said — "The reason it hits is because we
need split second decition."

That is a specific, falsifiable claim: the runaway IS distinguishable early, and
the fixed stop loses only because it cannot react at that timescale. If true, a
watcher (or the student model) can cut those trades before they cost 20pt, and
only 2.56pt of saving is needed to flip the whole strategy positive.

It is also a claim that runs straight into our standing wall — oscillator/runaway
discrimination has sat near 0.57 AUC for months. But every one of those attempts
was built on 1m/5s features. This asks whether the information lives BELOW that
resolution, where we have never looked.

DESIGN. Band touches are reproduced exactly as in oscillation_harvest_test.py
(cubic 5s w90, ±1.5σ, edge-triggered). At each touch, features are computed from
ONLY the first T seconds of 1s tape, and the outcome is whatever happens AFTER
that. Trades already resolved inside T are dropped — predicting a runaway that
has already happened is not prediction. AUC is reported per feature per horizon;
0.5 is a coin flip.

Features (all computable live, all strictly causal):
  drift    net signed move over the window, adverse-positive
  mae      worst adverse excursion inside the window
  frac_adv fraction of 1s bars closing against the position
  rng      realized high-low range (volatility expansion)
  accel    second-half drift minus first-half drift (is it building?)

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/runaway_early_tell.py --days 150
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
D1 = os.path.join(REPO, 'DATA', 'ATLAS', '1s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                   'runaway_early_tell.md')

CUBIC_5S_WINDOW = 90
SIGMA_MIN = 20
BAND = 1.5
STOP_PT = 20.0
MAX_HOLD_S = 3600
RTH_FROM, RTH_TO = 570, 960
HORIZONS = (5, 10, 20, 30)     # seconds of 1s tape allowed before deciding
FRICTION_PT = 0.89
PT_USD = 2.0
SEED = 11
MIN_BARS = 2000


def auc(score, label):
    """Rank AUC; label 1 = runaway. Ties handled by average ranks."""
    score = np.asarray(score, float); label = np.asarray(label, int)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n1, n0 = int(label.sum()), int((1 - label).sum())
    if n1 < 10 or n0 < 10:
        return float('nan'), n1, n0
    r = pd.Series(score).rank().to_numpy()
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0), n1, n0


def scan_day(day):
    p5, p1 = os.path.join(D5, f'{day}.parquet'), os.path.join(D1, f'{day}.parquet')
    if not (os.path.exists(p5) and os.path.exists(p1)):
        return []
    d = pd.read_parquet(p5)[['timestamp', 'high', 'low', 'close']]
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
    side = np.where(z >= BAND, 1, np.where(z <= -BAND, -1, 0))
    ff = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flip = np.flatnonzero((~np.isnan(ff[1:])) & (~np.isnan(ff[:-1]))
                          & (ff[1:] != ff[:-1])) + 1
    flip = flip[(etm[flip] >= RTH_FROM) & (etm[flip] < RTH_TO)]
    if len(flip) < 2:
        return []

    o1 = pd.read_parquet(p1)[['timestamp', 'high', 'low', 'close']]
    t1 = o1['timestamp'].to_numpy()
    c1 = o1['close'].to_numpy(); h1 = o1['high'].to_numpy(); l1 = o1['low'].to_numpy()

    rows = []
    for i in flip:
        sgn = 1 if ff[i] < 0 else -1
        p0 = c[i]; t0 = ts[i]
        w = (ts > t0) & (ts <= t0 + MAX_HOLD_S)
        if w.sum() < 12:
            continue
        zz = z[w]; hh = hi[w]; ll = lo[w]; cc = c[w]; tt = ts[w]
        tgt = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        j_win = tgt[0] if len(tgt) else None
        adv = (ll <= p0 - STOP_PT) if sgn > 0 else (hh >= p0 + STOP_PT)
        stp = np.flatnonzero(adv)
        j_stop = stp[0] if len(stp) else None
        if j_win is not None and (j_stop is None or j_win < j_stop):
            lab, t_res, pts = 0, tt[j_win], float((cc[j_win] - p0) * sgn)
        elif j_stop is not None:
            lab, t_res, pts = 1, tt[j_stop], -STOP_PT
        else:
            continue                               # timeouts: no clean label

        for H in HORIZONS:
            if t_res <= t0 + H:
                continue                           # already resolved: not prediction
            m = (t1 > t0) & (t1 <= t0 + H)
            if m.sum() < max(3, H // 2):
                continue
            cw = c1[m]; hw = h1[m]; lw = l1[m]
            adverse = (p0 - cw) if sgn > 0 else (cw - p0)   # +ve = against us
            mae = float((p0 - lw.min()) if sgn > 0 else (hw.max() - p0))
            half = max(1, len(cw) // 2)
            rows.append(dict(day=day, H=H, label=lab, pts=pts,
                             drift=float(adverse[-1]),
                             mae=mae,
                             frac_adv=float((np.diff(cw, prepend=p0) * sgn < 0).mean()),
                             rng=float(hw.max() - lw.min()),
                             accel=float(adverse[-1] - 2 * adverse[half - 1])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=150,
                    help='random sample of sessions (1s files are large)')
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    a = ap.parse_args()
    days = sorted(f[:-8] for f in os.listdir(D1) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())

    rows = []
    for d in tqdm(days, desc='1s tell'):
        try:
            rows += scan_day(d)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows'); return

    FEATS = [('drift', 'net adverse move'), ('mae', 'worst adverse excursion'),
             ('frac_adv', 'fraction of 1s bars against'),
             ('rng', 'realized range'), ('accel', 'is it building?')]
    L = ['# Is a runaway visible in the first seconds? (1s resolution)', '',
         'Owner\'s claim: the mechanical harvest loses only because the '
         'decision needs to be split-second. This tests whether the '
         'information is actually there below 5s, where we have never looked.',
         '',
         f'Band touches reproduced exactly as in `oscillation_harvest_test.py` '
         f'(cubic 5s w{CUBIC_5S_WINDOW}, ±{BAND:g}σ, edge-triggered, RTH). '
         f'Features use ONLY the first H seconds of 1s tape; the outcome is '
         f'what happens after. **Trades already resolved inside H are dropped** '
         f'— predicting a runaway that already happened is not prediction.',
         f'AUC: 0.5 = coin flip. Label 1 = runaway.', '',
         f'Sessions sampled: **{df["day"].nunique()}**', '',
         '| H (s) | N | runaway rate | ' + ' | '.join(f for f, _ in FEATS) + ' |',
         '|---' * (3 + len(FEATS)) + '|']
    best = (0.5, None)
    for H in HORIZONS:
        g = df[df['H'] == H]
        if len(g) < 100:
            continue
        cells = []
        for f, _ in FEATS:
            v, n1, n0 = auc(g[f].to_numpy(), g['label'].to_numpy())
            cells.append('·' if not np.isfinite(v) else f'`{v:.3f}`')
            if np.isfinite(v) and abs(v - 0.5) > abs(best[0] - 0.5):
                best = (v, f'{f} @ {H}s')
        L.append(f'| {H} | {len(g)} | {g["label"].mean():.1%} | '
                 + ' | '.join(cells) + ' |')

    L += ['', f'**Strongest signal: `{best[1]}`, AUC `{best[0]:.3f}`.**', '']
    if abs(best[0] - 0.5) < 0.05:
        L.append('That is a coin flip. **The runaway is NOT visible early at 1s '
                 'resolution** — the split-second hypothesis fails on its own '
                 'terms, and the loss is not a reaction-time problem.')
    else:
        L.append('That is a real separation. A watcher CAN see it coming; the '
                 'question becomes how much of the 2.56pt it converts.')
    L.append('')

    # what an early-exit rule would actually earn, using the best single feature
    if best[1]:
        f, H = best[1].split(' @ ')
        H = int(H[:-1])
        g = df[df['H'] == H].copy()
        L += [f'## If the watcher exits on `{f}` at {H}s', '',
              '| cut at percentile | trades cut | of those, runaways | mean net all (pt) |',
              '|---|---|---|---|']
        base = g['pts'].mean() - FRICTION_PT
        for q in (60, 70, 80, 90):
            thr = np.nanpercentile(g[f], q)
            cut = g[f] >= thr
            if cut.sum() < 20:
                continue
            # cut trades exit early: assume they take the adverse move so far
            pts = g['pts'].to_numpy().copy()
            pts[cut.to_numpy()] = -g.loc[cut, 'mae'].to_numpy()
            L.append(f'| p{q} | {cut.mean():.1%} | {g.loc[cut, "label"].mean():.1%} | '
                     f'`{pts.mean() - FRICTION_PT:+.2f}` |')
        L += ['', f'Baseline (no early exit): `{base:+.2f}pt`. '
                  f'An early-exit rule only helps if it beats that — and it '
                  f'pays the adverse move already incurred on every trade it '
                  f'cuts, including the ones that would have won.', '']

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
