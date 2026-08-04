#!/usr/bin/env python
"""What happens in the N minutes AFTER a vertical 5s spike that immediately
fades — measured, not guessed.

WHY (owner, 2026-08-01, mid-dojo on 2024_09_16): "based on that, what's our
best guess? Of next 20 minutes?" The setup in front of him was a +26pt move
inside ONE 5s bar that gave back 13 of 34 points within 15 seconds, at 09:56
ET. Rather than answer from intuition, find every historical analog and report
the empirical forward distribution.

ANALOG DEFINITION (all causal, all measurable at the decision instant):
  1. a `spike`: max−min ≥ SPIKE_PT inside a rolling SPIKE_BARS window of 5s bars
  2. that is directional (net move ≥ 70% of the window range)
  3. followed by a giveback ≥ GIVEBACK_FRAC of the spike within FADE_S seconds
  4. starting inside an ET clock window (volatility regime must match)
The anchor t0 is the giveback point — the moment the owner would be deciding.

Reports the forward distribution from t0: return, MFE, MAE, and the share of
paths that resume the spike direction before giving back a further stop's
worth. Bootstrap 95% CI on the mean per house rules; the MEDIAN and the
quantiles carry more information than the mean for this kind of fat-tailed
forward distribution, so all are printed.

EXCLUDES the day under test — reading its own forward bars would be lookahead.

Writes to research/dojo_forge/reports/.
Usage:
  python research/dojo_forge/tools/spike_forward.py --exclude 2024_09_16 \
      --dir up --et-from 09:45 --et-to 10:15 --horizon 20
"""
import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
D5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'spike_forward.md')

SPIKE_PT = 20.0        # a "vertical" move; the live case was 26pt
SPIKE_BARS = 3         # 15s — vertical means it happened in seconds, not minutes
DIRECTIONAL = 0.70     # net/range, so a spike is one-way, not a whipsaw
GIVEBACK_FRAC = 0.30   # the live case gave back 13/34 = 38%
FADE_S = 60            # giveback must land within a minute of the spike
COOLDOWN_S = 1800      # one sample per 30min per day — overlapping windows
                       # would fake the N and shrink every CI
BOOT = 4000
SEED = 11
PT_USD = 2.0           # MNQ: 0.25 tick, $0.50/tick


def _et_min(ts):
    e = pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
    return e.hour * 60 + e.minute


def scan_day(path, direction, m_from, m_to, horizon_s):
    d = pd.read_parquet(path)[['timestamp', 'open', 'high', 'low', 'close']]
    ts = d['timestamp'].to_numpy()
    hi = d['high'].to_numpy(); lo = d['low'].to_numpy(); c = d['close'].to_numpy()
    n = len(d)
    if n < 200:
        return []
    et = np.array([_et_min(t) for t in ts])
    sgn = 1 if direction == 'up' else -1
    out, last = [], -10 ** 9
    fade_bars = FADE_S // 5
    for i in range(SPIKE_BARS, n):
        if ts[i] - last < COOLDOWN_S or not (m_from <= et[i] < m_to):
            continue
        w = slice(i - SPIKE_BARS + 1, i + 1)
        rng = hi[w].max() - lo[w].min()
        if rng < SPIKE_PT:
            continue
        net = c[i] - d['open'].to_numpy()[i - SPIKE_BARS + 1]
        if net * sgn < DIRECTIONAL * rng:
            continue
        peak = hi[w].max() if sgn > 0 else lo[w].min()
        base = lo[w].min() if sgn > 0 else hi[w].max()
        # giveback within FADE_S
        j_end = min(n - 1, i + fade_bars)
        gb = None
        for j in range(i + 1, j_end + 1):
            back = (peak - lo[j]) if sgn > 0 else (hi[j] - peak)
            if back >= GIVEBACK_FRAC * abs(peak - base):
                gb = j
                break
        if gb is None:
            continue
        t0 = ts[gb]; p0 = c[gb]
        fwd = (ts > t0) & (ts <= t0 + horizon_s)
        if fwd.sum() < horizon_s // 10:          # need most of the window
            continue
        fh, fl, fc = hi[fwd].max(), lo[fwd].min(), c[fwd][-1]
        out.append(dict(day=os.path.basename(path)[:-8], t0=int(t0), p0=float(p0),
                        spike=float(rng),
                        ret=float((fc - p0) * sgn),
                        mfe=float((fh - p0) if sgn > 0 else (p0 - fl)),
                        mae=float((p0 - fl) if sgn > 0 else (fh - p0))))
        last = t0
    return out


def boot_ci(x, f=np.mean):
    rng = np.random.default_rng(SEED)
    x = np.asarray(x)
    s = [f(rng.choice(x, len(x), replace=True)) for _ in range(BOOT)]
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--dir', default='up', choices=['up', 'down'])
    ap.add_argument('--et-from', default='09:45')
    ap.add_argument('--et-to', default='10:15')
    ap.add_argument('--horizon', type=int, default=20, help='minutes')
    a = ap.parse_args()

    m_from = int(a.et_from[:2]) * 60 + int(a.et_from[3:])
    m_to = int(a.et_to[:2]) * 60 + int(a.et_to[3:])
    days = sorted(f for f in os.listdir(D5) if f.endswith('.parquet')
                  and f[:-8] not in a.exclude)
    rows = []
    for f in tqdm(days, desc='scan'):
        try:
            rows += scan_day(os.path.join(D5, f), a.dir, m_from, m_to,
                             a.horizon * 60)
        except Exception:
            continue
    if not rows:
        print('no analogs found'); return
    df = pd.DataFrame(rows)
    r, mfe, mae = df['ret'].to_numpy(), df['mfe'].to_numpy(), df['mae'].to_numpy()
    lo, hi = boot_ci(r)
    q = np.percentile(r, [10, 25, 50, 75, 90])

    L = [f'# After a vertical spike that immediately fades — next {a.horizon} min',
         '',
         f'Analog: ≥{SPIKE_PT:.0f}pt inside {SPIKE_BARS * 5}s, directional '
         f'(net ≥ {DIRECTIONAL:.0%} of range), then ≥{GIVEBACK_FRAC:.0%} giveback '
         f'within {FADE_S}s. Direction **{a.dir}**, ET '
         f'**{a.et_from}–{a.et_to}**. Anchor = the giveback point.',
         f'Excluded: {", ".join(a.exclude) or "none"}. One sample per '
         f'{COOLDOWN_S // 60}min per day.', '',
         f'**N = {len(df)} analogs across {df["day"].nunique()} sessions.**', '',
         '## Forward return from the giveback point (spike direction = +)', '',
         f'- mean **{r.mean():+.2f}pt** (${r.mean() * PT_USD:+.2f}), '
         f'95% CI **[{lo:+.2f}, {hi:+.2f}]** '
         f'→ {"NOT significant (CI includes 0)" if lo <= 0 <= hi else "significant"}',
         f'- median **{np.median(r):+.2f}pt** · '
         f'share continuing: **{(r > 0).mean():.1%}**',
         f'- quantiles p10 `{q[0]:+.1f}` p25 `{q[1]:+.1f}` p50 `{q[2]:+.1f}` '
         f'p75 `{q[3]:+.1f}` p90 `{q[4]:+.1f}`', '',
         '## Excursions (what the path does, not just where it ends)', '',
         f'- MFE median **{np.median(mfe):.1f}pt** (p75 `{np.percentile(mfe, 75):.1f}`)',
         f'- MAE median **{np.median(mae):.1f}pt** (p75 `{np.percentile(mae, 75):.1f}`)',
         f'- both-touch: {((mfe >= 10) & (mae >= 10)).mean():.1%} of paths reach '
         f'±10pt in BOTH directions inside the window', '',
         '## Race: which is hit first from the anchor', '',
         '| ±N pt | reached favorable first | adverse first | neither |',
         '|---|---|---|---|']
    for N in (5, 10, 15, 20):
        fav = (mfe >= N) & ((mae < N) | (mfe >= N) & (mae < N))
        f_only = ((mfe >= N) & (mae < N)).mean()
        a_only = ((mae >= N) & (mfe < N)).mean()
        both = ((mfe >= N) & (mae >= N)).mean()
        L.append(f'| ±{N} | {f_only:.1%} | {a_only:.1%} | '
                 f'{1 - f_only - a_only - both:.1%} (both: {both:.1%}) |')
    L += ['', 'Note: "both" cannot be resolved into a true race without '
              'bar-order replay — treat those rows as ambiguous, not as wins.', '']

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, 'w').write('\n'.join(L) + '\n')
    print('\n'.join(L))
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
