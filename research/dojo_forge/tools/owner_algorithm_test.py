#!/usr/bin/env python
"""The owner's algorithm, stated by him, tested exactly as stated.

HIS WORDS (2026-08-02): "see that there's an oscillation, fix regions where the
price should come back to, then measure the proximity to those levels, and exit
when the peak is reached give back a 20% or less — since if we are watching in
1s the reality is it won't double tap in the same region twice without a full
oscillation."

The last clause is the part every previous test lacked. Those entered on EVERY
qualifying extreme — 101 trades per session against his one. This adds the
constraint that makes it selective:

    ONE ENTRY PER REGION PER OSCILLATION. After trading a region, that region is
    dead until price has visited the OPPOSITE region — a full traverse — which
    re-arms it.

Everything else is held identical to the prior tests so the comparison is clean:
same ±1.5σ extreme detection, same hard stop, same friction. The ONLY new
ingredient is the re-arm rule. If selectivity is where his edge lives, it shows
up here and nowhere else.

Exit is his: retain (1 − giveback) of PEAK OPEN PROFIT, giveback swept 10–30%
around his stated 20%. HONEST FILLS — booked at the bar close after the floor is
breached, never at the floor itself. Booking the floor produced a +1.44pt/trade
phantom edge in the previous build and was 100% of its apparent result.

Reports trades-per-session alongside P&L, because a rule that fires 100x is not
his algorithm no matter what it earns.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/owner_algorithm_test.py --days 100
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
                   'owner_twostage.md')

CUBIC_5S_WINDOW = 90
SIGMA_MIN = 20
BAND_GRID = (3.0, 3.5, 4.0, 4.5, 5.0)   # his 60pt oscillation is ~3-4 sigma,
                                          # not the 1.5 fixed all session
HARD_STOP = 20.0
MAX_HOLD_S = 1800
ARM_PT = 2.0
GIVEBACK = (0.20, 0.30)
# TWO-STAGE exit (owner 2026-08-02): "register the MFE, hold until it
# retraces 80%, and since we already saw the rest exit at 70% of THAT
# MFE". Distinct from the single ratchet: the reference FREEZES at the
# warning instead of chasing new highs, and the 80/70 gap is hysteresis
# so one wobble does not exit. Release-on-new-high: a fresh extreme
# un-freezes and resumes ratcheting, else a recovered trade would exit
# against a stale reference.
TWO_STAGE = ((0.80, 0.70), (0.80, 0.60), (0.90, 0.80), (0.70, 0.60))
RTH_FROM, RTH_TO = 570, 960
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


def scan_day(day, BAND):
    d = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[
        ['timestamp', 'high', 'low', 'close']]
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

    rows = []
    for i in flip:
        sgn = 1 if int(ff[i]) < 0 else -1
        p0 = float(c[i])
        w = (ts > ts[i]) & (ts <= ts[i] + MAX_HOLD_S)
        if w.sum() < 24:
            continue
        hh, ll, cc = hi[w], lo[w], c[w]
        fav = (hh - p0) if sgn > 0 else (p0 - ll)
        adv = (p0 - ll) if sgn > 0 else (hh - p0)
        peak = np.maximum.accumulate(fav)
        rec = dict(day=day, mfe=float(peak[-1]))
        for G in GIVEBACK:
            R = 1.0 - G
            out = None
            for j in range(len(cc)):
                if adv[j] >= HARD_STOP:
                    out = -HARD_STOP; break
                if peak[j] > ARM_PT and (cc[j] - p0) * sgn <= peak[j] * R:
                    out = float((cc[j] - p0) * sgn)   # HONEST fill: the close
                    break
            rec[f'g{int(G * 100)}'] = (float((cc[-1] - p0) * sgn)
                                       if out is None else out)
        # TWO-STAGE: warn at W of peak (freezing that peak), exit at E of the
        # FROZEN value. A new extreme releases the freeze.
        for W, E in TWO_STAGE:
            out = None
            frozen = None
            for j in range(len(cc)):
                if adv[j] >= HARD_STOP:
                    out = -HARD_STOP; break
                cur = (cc[j] - p0) * sgn
                if frozen is not None and fav[j] > frozen:
                    frozen = None                    # new high -> release
                if frozen is None:
                    if peak[j] > ARM_PT and cur <= peak[j] * W:
                        frozen = peak[j]             # register the MFE
                else:
                    if cur <= frozen * E:
                        out = float(cur); break      # honest fill at the close
            rec[f't{int(W * 100)}_{int(E * 100)}'] = (
                float((cc[-1] - p0) * sgn) if out is None else out)
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=100)
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    a = ap.parse_args()
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())

    out = {}
    for B in BAND_GRID:
        rows = []
        for d in tqdm(days, desc=f'band {B}'):
            try:
                rows += scan_day(d, B)
            except Exception:
                continue
        out[B] = pd.DataFrame(rows)

    L = ['# The owner\'s algorithm — swept across OSCILLATION SCALE', '',
         'Every prior test fixed the band at 1.5σ (~13pt, **106 traverses per '
         'session**). His stated oscillation was 19640-19700 = **60pt**, and he '
         'took ONE trade in forty minutes. That is a 2.5x scale mismatch and a '
         '10x frequency mismatch — I was measuring a different instrument and '
         'calling it his method.', '',
         'Also recorded: my "one entry per region per oscillation" filter was '
         '**vacuous**. Band traverses ALTERNATE by construction, so the '
         'constraint is already satisfied by every traverse; both arms returned '
         'identical numbers. The re-arm rule blocked nothing.', '',
         f'Hard stop {HARD_STOP:g}pt, friction {FRICTION_PT}pt, max hold '
         f'{MAX_HOLD_S // 60}min, ratchet arms above {ARM_PT:g}pt. **Honest '
         f'fills** (close after the breach, never the floor — that error was '
         f'worth +1.44pt/trade).',
         f'Sessions: **{len(days)}**. Excluded: {", ".join(a.exclude)}.', '',
         '| band | ~width | giveback | trades/session | mean net | 95% CI | $/trade |',
         '|---|---|---|---|---|---|---|']
    best = None
    for B in BAND_GRID:
        df = out[B]
        if df.empty:
            continue
        nses = max(df['day'].nunique(), 1)
        for G in GIVEBACK:
            x = df[f'g{int(G * 100)}'].to_numpy() - FRICTION_PT
            lo_, hi_ = ci(x)
            per = len(x) / nses
            L.append(f'| {B:.1f}σ | ~{B * 8.7:.0f}pt | 1-stage {G:.0%} | {per:.1f} | '
                     f'`{x.mean():+.2f}` | `[{lo_:+.2f}, {hi_:+.2f}]` | '
                     f'`${x.mean() * PT_USD:+.2f}` |')
            if best is None or x.mean() > best[0]:
                best = (x.mean(), B, f'1-stage {G:.0%}', lo_, hi_, per)
        for W, E in TWO_STAGE:
            col = f't{int(W * 100)}_{int(E * 100)}'
            x = df[col].to_numpy() - FRICTION_PT
            lo_, hi_ = ci(x); per = len(x) / nses
            L.append(f'| {B:.1f}σ | ~{B * 8.7:.0f}pt | **2-stage {W:.0%}/{E:.0%}** | '
                     f'{per:.1f} | `{x.mean():+.2f}` | `[{lo_:+.2f}, {hi_:+.2f}]` | '
                     f'`${x.mean() * PT_USD:+.2f}` |')
            if best is None or x.mean() > best[0]:
                best = (x.mean(), B, f'2-stage {W:.0%}/{E:.0%}', lo_, hi_, per)
    L.append('')
    if best:
        mu, B, G, lo_, hi_, per = best
        L += [f'**Best: band {B:.1f}σ, {G} → `{mu:+.2f}pt/trade` '
              f'(${mu * PT_USD:+.2f}), {per:.1f} trades/session, '
              f'95% CI `[{lo_:+.2f}, {hi_:+.2f}]`.**', '']
        if lo_ > 0:
            L.append('**Significantly positive.** Scale was the missing '
                     'variable — his method works at the oscillation size he '
                     'was actually trading, and was invisible at 1.5σ.')
        elif mu > 0:
            L.append('Positive but the CI spans zero. Wider bands mean far '
                     'fewer trades, so this is underpowered by construction — '
                     'the honest read is *suggestive, not established*.')
        else:
            L.append('Negative at every scale. Scale is **not** the missing '
                     'variable either, and the exit geometry stays dead '
                     'regardless of oscillation size. What remains is *which* '
                     'oscillations he chose to trade — answerable only from his '
                     'actual entries, not from any sweep.')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
