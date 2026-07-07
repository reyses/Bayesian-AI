"""Does look-back touch history predict breakout vs rejection? (user's method)

The user's actual process: see where the band levels are NOW, look BACK at
how often price visited that zone during the session, use that to judge
where we are in the structure and the odds the CURRENT approach breaks out.

Test: for each debounced touch of a band level, count PRIOR visit episodes
of that price zone earlier in the session (causal look-back), then measure
P(break) by prior-visit bucket {0, 1, 2, 3+} with symmetric barriers at the
user's trading scale (default R=C=20 ticks; RW null = 0.5 within each
bucket, and the question is the TREND across buckets).
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
from level_hold_study import atlas, rolling_ols_bands, first_outcome, wilson  # noqa: E402

TICK = 0.25
REPORT_DIR = os.path.join(_REPO, 'research', 'level_hold', 'reports')
lines = []


def log(s):
    print(s)
    lines.append(s)


def visit_episodes_before(close, t, L, zone_ticks=8, gap_bars=60):
    """Count debounced episodes before bar t where close was within
    zone_ticks of L (episodes separated by >= gap_bars outside the zone)."""
    inz = np.abs(close[:t] - L) <= zone_ticks * TICK
    if not inz.any():
        return 0
    idx = np.nonzero(inz)[0]
    return 1 + int((np.diff(idx) >= gap_bars).sum())


def run_day(day, args):
    d5 = atlas(day, '5s')
    d1 = atlas(day, '1m')
    c5 = d5['close'].to_numpy()
    h5 = d5['high'].to_numpy()
    l5 = d5['low'].to_numpy()
    ts5 = d5['timestamp'].to_numpy()
    c1, ts1 = d1['close'].to_numpy(), d1['timestamp'].to_numpy()

    up_f, lo_f, _ = rolling_ols_bands(c5, 120)
    u1, o1, _ = rolling_ols_bands(c1, 60)
    idx1 = np.searchsorted(ts1 + 60, ts5, side='right') - 1
    ok = idx1 >= 0
    up_s = np.where(ok, u1[np.clip(idx1, 0, None)], np.nan)
    lo_s = np.where(ok, o1[np.clip(idx1, 0, None)], np.nan)

    R = C = args.r_ticks * TICK
    tol = args.tol_ticks * TICK
    events = []
    for stream in (up_f, lo_f, up_s, lo_s):
        armed = True
        last_end = 0
        t = 720  # skip first hour (bands warming, no meaningful history)
        while t < len(c5) - 12:
            L = stream[t]
            if not np.isfinite(L):
                t += 1
                continue
            near = (h5[t] >= L - tol) and (l5[t] <= L + tol)
            if near and armed and t > last_end:
                # approach side from the previous bar's close
                side = 1 if c5[t - 1] < L else -1
                out, ti, rb = first_outcome(h5, l5, t, L, side, R, C, tol,
                                            12, args.res_min * 12)
                if out in ('hold', 'break'):
                    nprior = visit_episodes_before(c5, t, L,
                                                   args.zone_ticks, args.gap_bars)
                    events.append((nprior, out == 'break'))
                    last_end = ti + (rb if rb > 0 else 0) + args.gap_bars
                armed = False
            elif not near:
                armed = True
            t += 1
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=str, required=True)
    ap.add_argument('--r-ticks', type=float, default=20)
    ap.add_argument('--tol-ticks', type=float, default=4)
    ap.add_argument('--zone-ticks', type=float, default=8)
    ap.add_argument('--gap-bars', type=int, default=60)   # 5 min debounce
    ap.add_argument('--res-min', type=int, default=30)
    args = ap.parse_args()
    days = [d.strip() for d in args.days.split(',')]

    buckets = {0: [0, 0], 1: [0, 0], 2: [0, 0], 3: [0, 0]}  # b -> [breaks, total]
    for day in days:
        try:
            ev = run_day(day, args)
        except Exception as e:
            log(f"{day}: skipped ({type(e).__name__})")
            continue
        for npr, is_break in ev:
            b = min(npr, 3)
            buckets[b][1] += 1
            buckets[b][0] += is_break

    log(f"days={len(days)}, barriers R=C={args.r_ticks}t, zone={args.zone_ticks}t, "
        f"debounce={args.gap_bars} bars")
    log(f"{'prior visits':<14}{'N':>7}{'P(break)':>10}{'95% CI':>18}")
    for b in range(4):
        k, n = buckets[b]
        p, a, c = wilson(k, n)
        tag = f"{b}" if b < 3 else "3+"
        log(f"{tag:<14}{n:>7}{p:>10.3f}   [{a:.3f}, {c:.3f}]")
    tot_k = sum(v[0] for v in buckets.values())
    tot_n = sum(v[1] for v in buckets.values())
    p, a, c = wilson(tot_k, tot_n)
    log(f"{'all':<14}{tot_n:>7}{p:>10.3f}   [{a:.3f}, {c:.3f}]")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f'touch_history_R{int(args.r_ticks)}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
