#!/usr/bin/env python
"""Can the owner's actual trade be replicated as a rule?

THE ASSIGNMENT (owner, 2026-08-02): "solve how to replicate it." Everything
else tested this session asked "does trigger X beat the band?" and the answer
was always no. This asks the different question: take what he ACTUALLY DID,
state it as a rule, and measure THAT.

WHAT HE DID, in his own words: "it is oscillating 1640ish to 1700ish, what we
did is identified this oscillation and wait for the test to 1700ish and then
ride down to 1640ish", exiting on "a 80% of current profit warning marker".

The distinguishing feature is NOT the entry — that is the same band/extreme
touch tested a dozen times today. It is that HE HELD. Every mechanical exit
tried so far leaves early: the band exit takes 5.95pt out of a 38pt MFE. His
exit was a RATCHET ON PEAK OPEN PROFIT — it does not arm until there is profit
to protect, and it never gives back more than a fixed fraction of the best the
trade has shown.

That specific combination — extreme entry + retain-X%-of-peak exit — has not
been tested. The earlier ratchet work used a trail from PRICE, not from PEAK
OPEN PROFIT, and it was applied to general trades rather than range-extreme
entries.

Swept because the retention fraction is the one free parameter and picking 80%
because that is what he said once would repeat the R=6 mistake.

Benchmarks, both on IDENTICAL entries so the comparison is paired:
  BAND   the ±1.5σ opposite-band exit (the standing benchmark, loses)
  HOLD   hold to the opposite extreme or a hard stop (the naive version)

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/replicate_owner_test.py --days 100
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
                   'replicate_owner.md')  # honest-fill build

CUBIC_5S_WINDOW = 90
SIGMA_MIN = 20
BAND = 1.5
HARD_STOP = 20.0        # outside the fakeout distribution, as established
MAX_HOLD_S = 1800
ARM_PT = 2.0            # peak profit must exceed this before the ratchet arms
RETAIN = (0.5, 0.6, 0.7, 0.8, 0.9)
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


def scan_day(day):
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
        sgn = 1 if ff[i] < 0 else -1
        p0 = float(c[i])
        w = (ts > ts[i]) & (ts <= ts[i] + MAX_HOLD_S)
        if w.sum() < 24:
            continue
        hh, ll, cc, zz = hi[w], lo[w], c[w], z[w]
        n = len(cc)
        fav = (hh - p0) * sgn if sgn > 0 else (p0 - ll) * (-sgn)
        fav = (hh - p0) if sgn > 0 else (p0 - ll)          # favourable excursion
        adv = (p0 - ll) if sgn > 0 else (hh - p0)          # adverse excursion
        peak = np.maximum.accumulate(fav)
        rec = dict(day=day, mfe=float(peak[-1]))

        # BAND benchmark
        tgt = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        stp = np.flatnonzero(adv >= HARD_STOP)
        jt = tgt[0] if len(tgt) else None
        js = stp[0] if len(stp) else None
        if jt is not None and (js is None or jt < js):
            rec['band'] = float((cc[jt] - p0) * sgn)
        elif js is not None:
            rec['band'] = -HARD_STOP
        else:
            rec['band'] = float((cc[-1] - p0) * sgn)

        # HOLD benchmark: opposite extreme (2x band) or hard stop
        opp = np.flatnonzero(zz >= BAND) if sgn > 0 else np.flatnonzero(zz <= -BAND)
        jo = opp[0] if len(opp) else None
        if js is not None and (jo is None or js < jo):
            rec['hold'] = -HARD_STOP
        elif jo is not None:
            rec['hold'] = float((cc[jo] - p0) * sgn)
        else:
            rec['hold'] = float((cc[-1] - p0) * sgn)

        # OWNER RULE: retain X% of peak open profit, once peak > ARM_PT
        for R in RETAIN:
            hitj = None
            for j in range(n):
                if adv[j] >= HARD_STOP:
                    hitj = ('stop', j); break
                if peak[j] > ARM_PT:
                    floor = peak[j] * R
                    cur = (cc[j] - p0) * sgn
                    if cur <= floor:
                        hitj = ('ratchet', j); break
            if hitj is None:
                rec[f'own{int(R * 100)}'] = float((cc[-1] - p0) * sgn)
            elif hitj[0] == 'stop':
                rec[f'own{int(R * 100)}'] = -HARD_STOP
            else:
                # HONEST FILL. The trigger fires once price has ALREADY passed
                # through the ratchet floor, so booking the floor itself is a
                # phantom fill -- the same class of error that made an earlier
                # T x N exit sweep look magic. Fill at the bar's CLOSE, which
                # is the first price actually reachable after the breach.
                j = hitj[1]
                rec[f'own{int(R * 100)}'] = float((cc[j] - p0) * sgn)
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
    rows = []
    for d in tqdm(days, desc='replicate'):
        try:
            rows += scan_day(d)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows'); return

    band = df['band'].to_numpy() - FRICTION_PT
    hold = df['hold'].to_numpy() - FRICTION_PT
    blo, bhi = ci(band); hlo, hhi = ci(hold)
    L = ['# Replicating the owner\'s trade as a rule', '',
         'His method: identify the oscillation, enter at an extreme, **hold**, '
         'and exit on a marker that retains X% of the PEAK open profit. The '
         'distinguishing feature is not the entry — it is that he held. Every '
         'mechanical exit tested so far leaves early.', '',
         f'Entries: the same ±{BAND:g}σ extreme touches as every prior test. '
         f'Ratchet arms only once peak profit exceeds {ARM_PT:g}pt. Hard stop '
         f'{HARD_STOP:g}pt. Friction {FRICTION_PT}pt. Max hold '
         f'{MAX_HOLD_S // 60}min.',
         f'Sessions **{df["day"].nunique()}**, trades **{len(df)}**, mean MFE '
         f'`{df["mfe"].mean():.2f}pt`. Excluded: {", ".join(a.exclude)}.', '',
         '## Benchmarks (identical entries)', '',
         f'- BAND exit: `{band.mean():+.2f}pt` 95% CI `[{blo:+.2f}, {bhi:+.2f}]`',
         f'- HOLD to opposite extreme: `{hold.mean():+.2f}pt` '
         f'95% CI `[{hlo:+.2f}, {hhi:+.2f}]`', '',
         '## Owner rule — retain X% of peak open profit', '',
         '| retain | mean net | 95% CI | vs BAND Δ | Δ 95% CI | sig? | $/trade |',
         '|---|---|---|---|---|---|---|']
    best = None
    for R in RETAIN:
        col = f'own{int(R * 100)}'
        x = df[col].to_numpy() - FRICTION_PT
        d_ = x - band
        lo_, hi_ = ci(x); dlo, dhi = ci(d_)
        sig = 'YES' if (dlo > 0 or dhi < 0) else 'no'
        L.append(f'| {R:.0%} | `{x.mean():+.2f}` | `[{lo_:+.2f}, {hi_:+.2f}]` | '
                 f'`{d_.mean():+.2f}` | `[{dlo:+.2f}, {dhi:+.2f}]` | **{sig}** | '
                 f'`${x.mean() * PT_USD:+.2f}` |')
        if best is None or x.mean() > best[1]:
            best = (R, x.mean(), dlo, dhi)
    L.append('')
    R, mu, dlo, dhi = best
    L += [f'**Best: retain {R:.0%} → `{mu:+.2f}pt/trade` (${mu * PT_USD:+.2f}).**', '']
    if mu > 0 and dlo > 0:
        L.append('**Positive AND significantly better than the band.** The '
                 'owner\'s method replicates as a rule — the edge was in '
                 'holding with a peak-profit ratchet, not in the entry.')
    elif dlo > 0:
        L.append('Significantly better than the band but still **negative**. '
                 'The ratchet captures real value the band throws away, yet '
                 'not enough to clear friction. Direction is right, magnitude '
                 'is not.')
    else:
        L.append('**Not significantly better than the band.** The rule does '
                 'not reproduce what he did — which means the replicable part '
                 'is not the exit geometry, and the search should move to what '
                 'he conditioned the ENTRY on.')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
