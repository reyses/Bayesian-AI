#!/usr/bin/env python
"""The owner's strategy tested IN ITS STATE, not generalized out of it.

HIS CRITIQUE (2026-08-02): "the strategy is for an explicit state, and sonnet
was trying to generalize it." Correct as far as it goes: every sweep so far
measured E[rule | all extremes, all days]. His claim is conditional:
E[rule | oscillation state OBSERVED first]. An unconditional null cannot refute
that. This encodes his live protocol verbatim — "we will first observe the
first 2 oscillations" — and only then trades.

STATE DEFINITION (causal, from confirmed zigzag pivots on 5s closes):
  - n_osc completed round trips between a STABLE pair of regions: the last
    (2*n_osc+1) confirmed pivots alternate T/B, all tops within TOL*range of
    each other, all bottoms within TOL*range, and range >= min_range.
  - Armed at the CONFIRMATION bar of the last pivot (a pivot confirms only
    once price has retraced R_ZZ from it — strictly causal).
  - The state DIES when price closes beyond either region by 0.5*range.

TRADE, once armed:
  - enter fading a touch of either region (within PROX*range), edge-triggered
  - ONE entry per region until the OPPOSITE region is touched. NOTE: this rule
    was vacuous for sigma-band traverses (they alternate by construction) but
    is NOT vacuous for fixed regions — price can double-tap the high without
    ever reaching the low, and this filter blocks exactly those.
  - exit: the owner's two-stage frozen-MFE ratchet (warn 80% of peak, exit at
    70% of the FROZEN peak, new extreme releases the freeze), arm above 2pt
  - hard stop max(8pt, 0.25*range) beyond entry; honest fills (close after the
    trigger, never the trigger level itself); max hold 30min

THE TEST IS THE n_osc GRADIENT: 1 vs 2 vs 3 observed oscillations before
trading. If observing the state first adds value, EV must rise with n_osc.
Flat gradient = observation buys nothing. All-negative = the explicit-state
encoding fails too, and selection must be learned from the owner's actual
corpus entries rather than any mechanical state.

Writes to research/dojo_forge/reports/.
Usage: python research/dojo_forge/tools/state_conditional_test.py --days 700
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                   'state_conditional.md')

N_OSC_GRID = (1, 2, 3)
MIN_RANGE_GRID = (20.0, 35.0, 50.0)
TOL = 0.25            # region stability: pivots within this fraction of range
PROX = 0.15           # entry proximity to a region, fraction of range
BREAK_FRAC = 0.5      # state dies when price closes this far beyond a region
WARN, EXIT_F = 0.80, 0.70   # the owner's two-stage exit
ARM_PT = 2.0
MAX_HOLD_BARS = 360   # 30min of 5s bars
RTH_FROM, RTH_TO = 570, 960
FRICTION_PT = 0.89
PT_USD = 2.0
BOOT = 4000
SEED = 11


def zz_confirm(c, R):
    """Close-based zigzag with CONFIRMATION indices.

    Returns [(pivot_idx, price, kind, confirm_idx)], kind +1=top / -1=bottom.
    A pivot exists only from its confirm_idx onward (price has retraced R),
    so any consumer indexing by confirm_idx is strictly causal."""
    if len(c) < 3:
        return []
    out = []
    d = 1 if c[1] >= c[0] else -1
    ext, ei = c[0], 0
    for i in range(1, len(c)):
        if d > 0:
            if c[i] > ext:
                ext, ei = c[i], i
            elif ext - c[i] >= R:
                out.append((ei, ext, 1, i)); d = -1; ext, ei = c[i], i
        else:
            if c[i] < ext:
                ext, ei = c[i], i
            elif c[i] - ext >= R:
                out.append((ei, ext, -1, i)); d = 1; ext, ei = c[i], i
    return out


def _selftest():
    t = np.concatenate([np.arange(0, 21, 1.), np.arange(19, -1, -1.),
                        np.arange(0, 21, 1.)])
    p = zz_confirm(t, 8.0)
    assert [k for _, _, k, _ in p] == [1, -1], f'self-test FAILED: {p}'
    assert all(ci_ > pi for pi, _, _, ci_ in p), 'confirm precedes pivot!'
    return True


def scan_day(day, n_osc, min_range):
    d = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))[
        ['timestamp', 'high', 'low', 'close']]
    if len(d) < 2000:
        return []
    e = pd.to_datetime(d['timestamp'], unit='s', utc=True).dt.tz_convert(
        'America/New_York')
    m = (e.dt.hour * 60 + e.dt.minute).to_numpy()
    k = (m >= RTH_FROM) & (m < RTH_TO)
    if k.sum() < 500:
        return []
    c = d['close'].to_numpy()[k]
    hi = d['high'].to_numpy()[k]
    lo = d['low'].to_numpy()[k]
    n = len(c)
    r_zz = max(8.0, 0.35 * min_range)
    piv = zz_confirm(c, r_zz)
    if len(piv) < 2 * n_osc + 1:
        return []

    need = 2 * n_osc + 1
    trades = []
    armed = False
    H = L = rng_ = None
    ok_h = ok_l = False
    pos = None            # dict(sgn, p0, j0, peak, frozen, stop)
    pv = 0                # next pivot (by confirm order) to consume

    for i in range(1, n):
        # consume pivots that confirm at or before this bar
        while pv < len(piv) and piv[pv][3] <= i:
            if not armed:
                lastk = piv[max(0, pv - need + 1):pv + 1]
                if len(lastk) == need and all(
                        lastk[j][2] != lastk[j + 1][2] for j in range(need - 1)):
                    tops = [p for _, p, kk, _ in lastk if kk == 1]
                    bots = [p for _, p, kk, _ in lastk if kk == -1]
                    rr = float(np.mean(tops) - np.mean(bots))
                    if (rr >= min_range
                            and max(tops) - min(tops) <= TOL * rr
                            and max(bots) - min(bots) <= TOL * rr):
                        armed = True
                        H, L, rng_ = float(np.mean(tops)), float(np.mean(bots)), rr
                        ok_h = ok_l = True
            pv += 1

        px = float(c[i])

        # manage the open position first (honest fills, close-based)
        if pos is not None:
            sgn, p0 = pos['sgn'], pos['p0']
            fav = (hi[i] - p0) if sgn > 0 else (p0 - lo[i])
            adv = (p0 - lo[i]) if sgn > 0 else (hi[i] - p0)
            cur = (px - p0) * sgn
            done = None
            if adv >= pos['stop']:
                done = -pos['stop']                      # stop order at level
            else:
                if fav > pos['peak']:
                    pos['peak'] = fav
                    pos['frozen'] = None                 # new extreme releases
                if pos['frozen'] is None:
                    if pos['peak'] > ARM_PT and cur <= pos['peak'] * WARN:
                        pos['frozen'] = pos['peak']      # register the MFE
                elif cur <= pos['frozen'] * EXIT_F:
                    done = cur                           # honest: the close
                if done is None and i - pos['j0'] >= MAX_HOLD_BARS:
                    done = cur
            if done is not None:
                trades.append(done - FRICTION_PT)
                pos = None

        if not armed:
            continue

        # state death: range broken
        if px > H + BREAK_FRAC * rng_ or px < L - BREAK_FRAC * rng_:
            armed = False
            continue

        # re-arm the opposite region on a touch (independent of trading)
        near_h = abs(px - H) <= PROX * rng_
        near_l = abs(px - L) <= PROX * rng_
        prev_h = abs(float(c[i - 1]) - H) <= PROX * rng_
        prev_l = abs(float(c[i - 1]) - L) <= PROX * rng_
        if near_h:
            ok_l = True
        if near_l:
            ok_h = True

        if pos is not None:
            continue
        stop_pt = max(8.0, 0.25 * rng_)
        if near_h and not prev_h and ok_h:               # edge-triggered
            pos = dict(sgn=-1, p0=px, j0=i, peak=0.0, frozen=None, stop=stop_pt)
            ok_h = False
        elif near_l and not prev_l and ok_l:
            pos = dict(sgn=1, p0=px, j0=i, peak=0.0, frozen=None, stop=stop_pt)
            ok_l = False

    if pos is not None:                                   # session end
        sgn, p0 = pos['sgn'], pos['p0']
        trades.append((float(c[-1]) - p0) * sgn - FRICTION_PT)
    return trades


def ci(x):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    if len(x) < 5:
        return float('nan'), float('nan')
    return tuple(np.percentile(
        [rng.choice(x, len(x), replace=True).mean() for _ in range(BOOT)],
        [2.5, 97.5]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=700)
    ap.add_argument('--exclude', nargs='*', default=['2024_09_16'])
    a = ap.parse_args()
    _selftest()
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    if len(days) > a.days:
        days = sorted(rng.choice(days, a.days, replace=False).tolist())

    res = {}
    for mr in MIN_RANGE_GRID:
        for no in N_OSC_GRID:
            rows = []
            for d in tqdm(days, desc=f'range>={mr:.0f} osc={no}'):
                try:
                    rows += scan_day(d, no, mr)
                except Exception:
                    continue
            res[(mr, no)] = np.array(rows)

    L = ['# State-conditional test — the strategy inside its explicit state', '',
         'Owner: *"the strategy is for an explicit state, and sonnet was trying '
         'to generalize it."* Correct: every prior sweep measured the '
         'unconditional EV. This encodes his live protocol — **observe the '
         'first n oscillations of a stable region pair, then harvest** — with '
         'his two-stage 80/70 frozen-MFE exit, honest fills, and the '
         'one-per-region-until-opposite-touch rule (NOT vacuous for fixed '
         'regions, unlike sigma bands).', '',
         f'Region stability tol {TOL:.0%} of range, entry proximity '
         f'{PROX:.0%}, state dies {BREAK_FRAC:.0%} beyond a region, stop '
         f'max(8pt, 25% of range), friction {FRICTION_PT}pt. '
         f'Sessions: **{len(days)}**. Excluded: {", ".join(a.exclude)}.', '',
         '**The test is the n_osc gradient.** If observing the state first adds '
         'value, EV must rise from 1 → 2 → 3 observed oscillations.', '',
         '| min range | observed osc | trades | /session | mean net | 95% CI | $/trade |',
         '|---|---|---|---|---|---|---|']
    best = None
    for mr in MIN_RANGE_GRID:
        for no in N_OSC_GRID:
            x = res[(mr, no)]
            if len(x) < 5:
                L.append(f'| {mr:.0f}pt | {no} | {len(x)} | — | too few | — | — |')
                continue
            lo_, hi_ = ci(x)
            per = len(x) / len(days)
            L.append(f'| {mr:.0f}pt | {no} | {len(x):,} | {per:.2f} | '
                     f'`{x.mean():+.2f}` | `[{lo_:+.2f}, {hi_:+.2f}]` | '
                     f'`${x.mean() * PT_USD:+.2f}` |')
            if best is None or x.mean() > best[0]:
                best = (x.mean(), mr, no, lo_, hi_, per)
        L.append('| | | | | | | |')
    L.append('')
    if best:
        mu, mr, no, lo_, hi_, per = best
        L += [f'**Best: range≥{mr:.0f}pt, {no} observed osc → '
              f'`{mu:+.2f}pt/trade` (${mu * PT_USD:+.2f}), {per:.2f} '
              f'trades/session, 95% CI `[{lo_:+.2f}, {hi_:+.2f}]`.**', '']
        if lo_ > 0:
            L.append('**Significantly positive.** The state was the missing '
                     'conditioning — the strategy works inside its state and '
                     'the generalized tests were measuring the wrong thing.')
        elif mu > 0:
            L.append('Positive point estimate, CI spans zero — suggestive and '
                     'worth powering, not established.')
        else:
            L.append('Negative even inside the explicitly observed state. The '
                     'mechanical encoding of "the state" does not rescue the '
                     'strategy; whatever the owner conditions on is not '
                     'captured by observed-oscillation stability, and the only '
                     'remaining source of truth is his actual corpus entries '
                     'versus the extremes he declined.')
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
