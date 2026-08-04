#!/usr/bin/env python
"""Does ENTERING at a cubic-sigma extreme carry an edge? (fade vs follow)

WHY (owner, 2026-08-01): "the NMP strategy is to ride the chop, by using the
sigma extremes ... same way the +-10p but in our favor."

The exit half of that is already dead — brackets lost at every width across
250k trades, and optional stopping explains it: P(hit −S before +T) = T/(S+T)
makes gross EV identically zero for ANY exit pair on a martingale, so flipping
sides changes nothing. But optional stopping constrains EXITS ONLY. It is
silent on WHERE YOU ENTER. "Enter at a sigma extreme" is the separate claim
that price is not a martingale *conditional on being stretched* — untested,
and with a standing hint: the −3σ breakout-FOLLOW variant came in at 35.7% win,
a cleanly inverted 2:1, which points at the fade.

So this measures ENTRY ONLY. No stop, no target, no trailing — just the forward
distribution from the touch, at several horizons, both directions, against a
same-session random-entry control. Anything an exit rule could add is a
separate question and deliberately out of scope: mixing them is how the earlier
T×N sweep produced phantom edge.

METHOD NOTES (both are prior bugs this avoids):
- EDGE-TRIGGERED touches. Level-triggered counting turned one sustained
  excursion into dozens of "touches" and inflated N ~10x.
- The control is drawn from the SAME sessions and the SAME clock window, so a
  time-of-day volatility effect cannot masquerade as a sigma effect.

Instrument matches the deployed spec: cubic regression endpoint on 5s bars,
window 90 (= 7.5 min), residual sigma over SIGMA_MIN minutes.

Writes to research/dojo_forge/reports/.
Usage:
  python research/dojo_forge/tools/sigma_fade_test.py --exclude 2024_09_16
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
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'sigma_fade.md')

CUBIC_5S_WINDOW = 90    # 90 x 5s = 7.5 min, the deployed NT8 spec
SIGMA_MIN = 20          # residual-sigma lookback, minutes (pocket_dojo default)
HORIZONS = (5, 10, 20)  # forward minutes
SIGMAS = (2.0, 2.5, 3.0)
RTH_FROM, RTH_TO = 570, 960     # 09:30–16:00 ET; liquid tape only
FRICTION_PT = 0.89      # MNQ round trip
PT_USD = 2.0
BOOT = 4000
SEED = 11
MIN_BARS = 2000


def _et_min(ts):
    e = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    return (e.hour * 60 + e.minute).to_numpy()


def scan_day(path, rng):
    d = pd.read_parquet(path)[['timestamp', 'open', 'high', 'low', 'close']]
    if len(d) < MIN_BARS:
        return []
    ts = d['timestamp'].to_numpy()
    c = d['close'].to_numpy(); hi = d['high'].to_numpy(); lo = d['low'].to_numpy()
    cub, _, _ = _cub.rolling(c, CUBIC_5S_WINDOW, 5)
    res = c - cub
    sig = pd.Series(res).rolling(SIGMA_MIN * 12, min_periods=5 * 12).std().to_numpy()
    z = np.where(sig > 0, res / sig, np.nan)
    etm = _et_min(ts)
    ok = (etm >= RTH_FROM) & (etm < RTH_TO) & np.isfinite(z)
    n = len(d)
    rows = []
    for S in SIGMAS:
        for side, cond in (('below', z <= -S), ('above', z >= S)):
            fire = cond & ok
            # EDGE-TRIGGER: only the bar that ENTERS the zone
            fire = fire & ~np.concatenate(([False], fire[:-1]))
            for i in np.flatnonzero(fire):
                p0 = c[i]
                # fade = bet on reversion toward the cubic
                sgn = 1 if side == 'below' else -1
                r = dict(sigma=S, side=side, i=i, ts=int(ts[i]))
                for H in HORIZONS:
                    f = (ts > ts[i]) & (ts <= ts[i] + H * 60)
                    if f.sum() < H * 12 * 0.8:
                        r = None
                        break
                    fc = c[f][-1]
                    fh, fl = hi[f].max(), lo[f].min()
                    r[f'ret{H}'] = float((fc - p0) * sgn)
                    r[f'mfe{H}'] = float((fh - p0) if sgn > 0 else (p0 - fl))
                    r[f'mae{H}'] = float((p0 - fl) if sgn > 0 else (fh - p0))
                if r:
                    rows.append(r)
    # CONTROL: same sessions, same clock window, random bars, same directions
    idx = np.flatnonzero(ok)
    if len(idx) > 40:
        for i in rng.choice(idx, 20, replace=False):
            for sgn in (1, -1):
                r = dict(sigma=0.0, side='ctrl', i=int(i), ts=int(ts[i]))
                p0 = c[i]
                for H in HORIZONS:
                    f = (ts > ts[i]) & (ts <= ts[i] + H * 60)
                    if f.sum() < H * 12 * 0.8:
                        r = None
                        break
                    r[f'ret{H}'] = float((c[f][-1] - p0) * sgn)
                    r[f'mfe{H}'] = float((hi[f].max() - p0) if sgn > 0
                                         else (p0 - lo[f].min()))
                    r[f'mae{H}'] = float((p0 - lo[f].min()) if sgn > 0
                                         else (hi[f].max() - p0))
                if r:
                    rows.append(r)
    return rows


def ci(x):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x, float)
    s = [rng.choice(x, len(x), replace=True).mean() for _ in range(BOOT)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def delta_ci(a, b):
    """95% CI on mean(b) − mean(a); populations bootstrapped independently."""
    rng = np.random.default_rng(SEED)
    a = np.asarray(a, float); b = np.asarray(b, float)
    s = [rng.choice(b, len(b), replace=True).mean()
         - rng.choice(a, len(a), replace=True).mean() for _ in range(BOOT)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', nargs='*', default=[])
    a = ap.parse_args()
    days = sorted(f for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rng = np.random.default_rng(SEED)
    rows = []
    for f in tqdm(days, desc='sigma scan'):
        try:
            rows += scan_day(os.path.join(D5, f), rng)
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        print('no rows'); return

    ctrl = df[df['side'] == 'ctrl']
    L = ['# Entering at a cubic-sigma extreme — does the FADE carry an edge?',
         '',
         'ENTRY ONLY. No stop, no target, no trail — optional stopping already '
         'settles the exit question (gross EV is zero for any exit pair on a '
         'martingale), so mixing exits in here would only manufacture phantom '
         'edge, as the earlier T×N sweep did.',
         '',
         f'Cubic endpoint on 5s, window {CUBIC_5S_WINDOW} (7.5min, deployed '
         f'spec); residual sigma over {SIGMA_MIN}min. RTH only. '
         f'**Edge-triggered** touches. Friction `{FRICTION_PT}pt` round trip '
         'shown where a trade is implied.',
         f'Excluded: {", ".join(a.exclude) or "none"}. '
         f'Control = random bars from the SAME sessions and clock window, so '
         f'time-of-day volatility cannot pose as a sigma effect.',
         '', f'Sessions scanned: **{len(days)}**. Control N = '
         f'**{len(ctrl)}**.', '']

    for H in HORIZONS:
        cr = ctrl[f'ret{H}'].to_numpy()
        clo, chi = ci(cr)
        L += [f'## Forward {H} min', '',
              f'Control: mean `{cr.mean():+.2f}pt` 95% CI '
              f'`[{clo:+.2f}, {chi:+.2f}]` (N={len(cr)})', '',
              '| entry | N | mean ret | 95% CI | vs control Δ | Δ 95% CI | sig? | median | win% |',
              '|---|---|---|---|---|---|---|---|---|']
        for S in SIGMAS:
            for side, lab in (('below', f'−{S:g}σ FADE (long)'),
                              ('above', f'+{S:g}σ FADE (short)')):
                g = df[(df['sigma'] == S) & (df['side'] == side)][f'ret{H}']
                if len(g) < 30:
                    continue
                x = g.to_numpy()
                lo_, hi_ = ci(x)
                dl, dh = delta_ci(cr, x)
                sig = 'YES' if (dl > 0 or dh < 0) else 'no'
                L.append(f'| {lab} | {len(x)} | `{x.mean():+.2f}` | '
                         f'`[{lo_:+.2f}, {hi_:+.2f}]` | `{x.mean() - cr.mean():+.2f}` | '
                         f'`[{dl:+.2f}, {dh:+.2f}]` | **{sig}** | '
                         f'`{np.median(x):+.2f}` | {(x > 0).mean():.1%} |')
        L.append('')

    L += ['## How to read this', '',
          'A row is only interesting if **sig? = YES** — the Δ-vs-control CI '
          'must exclude zero. A positive mean with a CI spanning zero is noise, '
          'no matter how large the point estimate.',
          'Any surviving edge must then clear friction '
          f'(`{FRICTION_PT}pt` = `${FRICTION_PT * PT_USD:.2f}`) before it is a '
          'trade rather than a statistic.', '']

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
