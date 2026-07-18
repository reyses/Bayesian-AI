"""Does look-back touch history predict breakout vs rejection? (user's method)

The user's actual process: see where the band levels are NOW, look BACK at
how often price visited that zone during the session, use that to judge
where we are in the structure and the odds the CURRENT approach breaks out.

CORRECTION (2026-07-07, from the user): "it's not a set amount" — the
look-back zone is NOT a fixed tick radius. The user's own NT8 config draws
the regression bands at 2-SIGMA, a width that breathes with volatility
(10-min window here). So "how near price was" must be measured relative to
the CURRENT band's own sigma, not a constant. v1 of this probe used a fixed
8-tick zone regardless of regime — likely under-counting visits on volatile
days and over-counting on quiet ones. This version defines the zone as
zone_sigma_mult * sigma(t) (same sigma the band width is built from).

Test: for each debounced touch of a band level, count PRIOR visit episodes
of that price zone earlier in the session (causal look-back, zone scaled by
the band's current sigma), then measure P(break) by prior-visit bucket
{0, 1, 2, 3+} with symmetric barriers at the user's trading scale (default
R=C=20 ticks — a separate, trade-economics choice, not the zone radius).
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


def visit_episodes_before(close, t, L, zone, gap_bars=60):
    """Count debounced episodes before bar t where close was within +-zone
    (price units, already sigma-scaled by the caller) of L."""
    inz = np.abs(close[:t] - L) <= zone
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

    up_f, lo_f, _, sig_f = rolling_ols_bands(c5, 120, return_sigma=True)
    u1, o1, _, sig_1 = rolling_ols_bands(c1, 60, return_sigma=True)
    idx1 = np.searchsorted(ts1 + 60, ts5, side='right') - 1
    ok = idx1 >= 0
    up_s = np.where(ok, u1[np.clip(idx1, 0, None)], np.nan)
    lo_s = np.where(ok, o1[np.clip(idx1, 0, None)], np.nan)
    sig_s = np.where(ok, sig_1[np.clip(idx1, 0, None)], np.nan)

    R = C = args.r_ticks * TICK
    events = []
    # (level stream, sigma stream) pairs — sigma-relative zone throughout
    for stream, sig in ((up_f, sig_f), (lo_f, sig_f), (up_s, sig_s), (lo_s, sig_s)):
        armed = True
        last_end = 0
        t = 720  # skip first hour (bands warming, no meaningful history)
        while t < len(c5) - 12:
            L, s = stream[t], sig[t]
            if not (np.isfinite(L) and np.isfinite(s)) or s <= 0:
                t += 1
                continue
            zone = args.zone_sigma * s
            near = (h5[t] >= L - zone) and (l5[t] <= L + zone)
            if near and armed and t > last_end:
                side = 1 if c5[t - 1] < L else -1
                out, ti, rb = first_outcome(h5, l5, t, L, side, R, C,
                                            zone, 12, args.res_min * 12)
                if out in ('hold', 'break'):
                    nprior = visit_episodes_before(c5, t, L, zone, args.gap_bars)
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
    ap.add_argument('--zone-sigma', type=float, default=1.0,
                    help='look-back/touch zone half-width, in units of the '
                         "band's own current sigma (replaces the old fixed-tick zone)")
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

    log(f"days={len(days)}, barriers R=C={args.r_ticks}t, "
        f"zone={args.zone_sigma}*sigma(t) [adaptive, not fixed], "
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
    out = os.path.join(REPORT_DIR, f'touch_history_sigma{args.zone_sigma}_R{int(args.r_ticks)}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
