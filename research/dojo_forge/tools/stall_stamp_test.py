#!/usr/bin/env python
"""Is the STALL stamp a real instrument, or the next thing to die?

CONTEXT. On 2026-08-01 the owner said the watcher "should have stamped the 2
times it stalled". Defining a stall as the ABSENCE OF NEW FAVOURABLE EXTREMES
(not flatness) reproduced both of his stalls exactly, so `watch --stall N` was
shipped into the live tool. That is one leg of evidence — the same standard of
proof that had just produced the acceleration-inflection rule, which then died
out of sample at 0.037 precision against a 0.02 random baseline.

So this validates the thing already in the owner's hands. If it fails, the tool
must be labelled or pulled; shipping an unvalidated trigger and leaving it there
is how a dojo starts teaching noise.

TWO QUESTIONS, and the second is the one that matters:

  1. TIMING — when a stall stamp fires on an open position, how close is the
     position's running extreme to the leg's TRUE extreme? A stamp is useful if
     the best price is already behind us when it speaks; useless if the leg had
     much further to run.

  2. MONEY — exiting AT the stall stamp versus the two benchmarks already
     established: the ±1.5σ band exit (5.95pt average, net −0.55pt/trade) and
     simply holding for the full traverse. This is the only question that
     decides whether it belongs in the watcher.

Entries are the same band touches used throughout (cubic 5s w90, ±1.5σ,
edge-triggered, RTH) so the numbers are directly comparable to every prior test.
Friction 0.89pt charged per attempt.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/stall_stamp_test.py --days 80
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
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'stall_stamp.md')

CUBIC_5S_WINDOW = 90
SIGMA_MIN = 20
BAND = 1.5
STOP_PT = 20.0
MAX_HOLD_S = 1800
RTH_FROM, RTH_TO = 570, 960
STALL_GRID = (5, 8, 12, 20, 30)     # seconds with no new favourable extreme
FRICTION_PT = 0.89
PT_USD = 2.0
BOOT = 4000
SEED = 11


def ci(x):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    if len(x) < 5:
        return float('nan'), float('nan')
    return tuple(np.percentile(
        [rng.choice(x, len(x), replace=True).mean() for _ in range(BOOT)],
        [2.5, 97.5]))


def scan_day(day):
    p5, p1 = os.path.join(D5, f'{day}.parquet'), os.path.join(D1, f'{day}.parquet')
    if not (os.path.exists(p5) and os.path.exists(p1)):
        return []
    d = pd.read_parquet(p5)[['timestamp', 'high', 'low', 'close']]
    if len(d) < 2000:
        return []
    ts = d['timestamp'].to_numpy(); c = d['close'].to_numpy()
    hi = d['high'].to_numpy(); lo = d['low'].to_numpy()
    cub, _, _ = _cub.rolling(c, CUBIC_5S_WINDOW, 5)
    res = c - cub
    sig = pd.Series(res).rolling(SIGMA_MIN * 12, min_periods=5 * 12).std().to_numpy()
    z = np.where(sig > 0, res / sig, np.nan)
    e = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    m = (e.hour * 60 + e.minute).to_numpy()
    side = np.where(z >= BAND, 1, np.where(z <= -BAND, -1, 0))
    ff = pd.Series(np.where(side == 0, np.nan, side)).ffill().to_numpy()
    flip = np.flatnonzero((~np.isnan(ff[1:])) & (~np.isnan(ff[:-1]))
                          & (ff[1:] != ff[:-1])) + 1
    flip = flip[(m[flip] >= RTH_FROM) & (m[flip] < RTH_TO)]
    if not len(flip):
        return []

    o1 = pd.read_parquet(p1)[['timestamp', 'high', 'low', 'close']]
    t1 = o1['timestamp'].to_numpy(); c1 = o1['close'].to_numpy()
    h1 = o1['high'].to_numpy(); l1 = o1['low'].to_numpy()

    rows = []
    for i in flip:
        sgn = 1 if ff[i] < 0 else -1          # fade toward the opposite band
        t0, p0 = ts[i], c[i]
        w1 = (t1 > t0) & (t1 <= t0 + MAX_HOLD_S)
        if w1.sum() < 60:
            continue
        cw, hw, lw, tw = c1[w1], h1[w1], l1[w1], t1[w1]
        fav = (hw - p0) if sgn > 0 else (p0 - lw)      # favourable excursion
        mfe = float(np.max(fav))
        # BAND exit on the same entry, for comparison
        w5 = (ts > t0) & (ts <= t0 + MAX_HOLD_S)
        zz, hh5, ll5, cc5 = z[w5], hi[w5], lo[w5], c[w5]
        tgt = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        stp = np.flatnonzero((ll5 <= p0 - STOP_PT) if sgn > 0
                             else (hh5 >= p0 + STOP_PT))
        jt = tgt[0] if len(tgt) else None
        js = stp[0] if len(stp) else None
        if jt is not None and (js is None or jt < js):
            band_pts = float((cc5[jt] - p0) * sgn)
        elif js is not None:
            band_pts = -STOP_PT
        else:
            band_pts = float((cc5[-1] - p0) * sgn)
        rec = dict(day=day, mfe=mfe, band=band_pts)
        # STALL stamp: N seconds with no new favourable extreme
        run_ext = np.maximum.accumulate(fav)
        improved = np.concatenate(([True], run_ext[1:] > run_ext[:-1]))
        gaps = np.flatnonzero(improved)
        for N in STALL_GRID:
            fire = None
            for a_, b_ in zip(gaps, np.append(gaps[1:], len(fav))):
                if b_ - a_ >= N:
                    fire = a_ + N
                    break
            if fire is None or fire >= len(cw):
                rec[f's{N}'] = None
                rec[f'r{N}'] = None
            else:
                rec[f's{N}'] = float((cw[fire] - p0) * sgn)   # exit at the stamp
                rec[f'r{N}'] = float(run_ext[fire] / mfe) if mfe > 0 else np.nan
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=80)
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    a = ap.parse_args()
    days = sorted(f[:-8] for f in os.listdir(D1) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())
    rows = []
    for d in tqdm(days, desc='stall'):
        try:
            rows += scan_day(d)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows'); return

    band = df['band'].to_numpy() - FRICTION_PT
    blo, bhi = ci(band)
    L = ['# Does the STALL stamp earn its place in the watcher?', '',
         'A stall = N seconds with **no new favourable extreme** on the open '
         'position (not flatness). Shipped into `watch --stall N` on the '
         'strength of ONE leg — the same standard that produced the '
         'acceleration-inflection rule, which then died out of sample. This is '
         'the check.', '',
         f'Entries: the same ±{BAND:g}σ band touches used in every prior test '
         f'(cubic 5s w{CUBIC_5S_WINDOW}, edge-triggered, RTH). '
         f'Friction {FRICTION_PT}pt. Max hold {MAX_HOLD_S // 60}min.',
         f'Sessions: **{df["day"].nunique()}**, trades: **{len(df)}**. '
         f'Excluded: {", ".join(a.exclude)}.', '',
         f'Benchmark — BAND exit: mean net `{band.mean():+.2f}pt` '
         f'95% CI `[{blo:+.2f}, {bhi:+.2f}]`. MFE mean '
         f'`{df["mfe"].mean():.2f}pt`.', '',
         '| stall N | fired | exit mean net | 95% CI | vs band Δ | Δ 95% CI | '
         'sig? | % of MFE captured |',
         '|---|---|---|---|---|---|---|---|']
    best = None
    for N in STALL_GRID:
        col, rcol = f's{N}', f'r{N}'
        g = df[df[col].notna()]
        if len(g) < 50:
            continue
        x = g[col].to_numpy() - FRICTION_PT
        b = g['band'].to_numpy() - FRICTION_PT
        d_ = x - b                                   # PAIRED: same entries
        lo_, hi_ = ci(x); dlo, dhi = ci(d_)
        sig = 'YES' if (dlo > 0 or dhi < 0) else 'no'
        cap = g[rcol].mean()
        L.append(f'| {N}s | {len(g)} | `{x.mean():+.2f}` | `[{lo_:+.2f}, {hi_:+.2f}]` | '
                 f'`{d_.mean():+.2f}` | `[{dlo:+.2f}, {dhi:+.2f}]` | **{sig}** | '
                 f'{cap:.1%} |')
        if best is None or x.mean() > best[1]:
            best = (N, x.mean(), dlo, dhi, cap)
    L.append('')
    if best:
        N, mu, dlo, dhi, cap = best
        L += [f'**Best: stall {N}s → `{mu:+.2f}pt/trade` '
              f'(${mu * PT_USD:+.2f}), capturing {cap:.1%} of MFE.**', '']
        if dlo > 0:
            L.append('The stall exit **significantly beats the band exit on '
                     'identical entries**. It earns its place, and the paired '
                     'test means this is not an entry-selection effect.')
        elif dhi < 0:
            L.append('The stall exit is **significantly WORSE** than the band '
                     'exit. Pull it from the watcher or mark it clearly as '
                     'informational-only.')
        else:
            L.append('No significant difference from the band exit. The stall '
                     'stamp is **not an edge** — but as a halt-and-ask prompt '
                     'it costs nothing versus the band, so it may stay as an '
                     'attention device provided it is never sold as an exit '
                     'rule.')
        if mu < 0:
            L.append('')
            L.append(f'Note the absolute level: `{mu:+.2f}pt` is still a '
                     'LOSING trade. Beating the band is not the same as making '
                     'money, and no exit rule tested has yet cleared friction.')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
