"""Causal level-hold study: do frozen band levels predict where price turns?

See research/level_hold/README.md for the design. Symmetric barriers make the
random-walk null exactly 0.5; jittered phantom levels calibrate the
mean-reversion base rate.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
REPORT_DIR = os.path.join(_REPO, 'research', 'level_hold', 'reports')

TICK = 0.25
lines = []


def log(s):
    print(s)
    lines.append(s)


def atlas(day, tf):
    p = os.path.join(_REPO, 'DATA', 'ATLAS', tf, f'{day}.parquet')
    df = pd.read_parquet(p)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = df['timestamp'].astype('int64') // 10 ** 9
    return df.sort_values('timestamp').reset_index(drop=True)


def rolling_ols_bands(close, W, k=2.0):
    """Causal OLS endpoint +- k*sigma over trailing W bars. Returns
    (upper, lower, mid) arrays aligned to the input (nan for first W-1)."""
    n = len(close)
    x = np.linspace(-1.0, 1.0, W)
    X = np.stack([np.ones(W), x], axis=1)
    P = np.linalg.pinv(X)                       # [2, W]
    sw = np.lib.stride_tricks.sliding_window_view(close, W)  # [n-W+1, W]
    C = sw @ P.T                                 # [n-W+1, 2]
    fit = C @ X.T
    sig = np.sqrt(((sw - fit) ** 2).mean(axis=1))
    end = C[:, 0] + C[:, 1]                      # endpoint (x=1)
    pad = np.full(W - 1, np.nan)
    return (np.r_[pad, end + k * sig], np.r_[pad, end - k * sig],
            np.r_[pad, end])


def rolling_extreme(a, W, mode):
    s = pd.Series(a)
    r = s.rolling(W, min_periods=W)
    return (r.max() if mode == 'max' else r.min()).to_numpy()


def first_outcome(highs, lows, t0, L, side, R, C, tol, h_touch, h_res):
    """side=+1 resistance (approach from below), -1 support.
    Returns (outcome, touch_idx, res_bars): outcome in
    {'hold','break','unresolved','ambiguous','notouch'}."""
    n = len(highs)
    end_t = min(n, t0 + h_touch)
    if side > 0:
        touch = np.nonzero(highs[t0:end_t] >= L - tol)[0]
    else:
        touch = np.nonzero(lows[t0:end_t] <= L + tol)[0]
    if len(touch) == 0:
        return 'notouch', -1, -1
    ti = t0 + int(touch[0])
    # touch bar itself: straight-through = break; both = ambiguous
    if side > 0:
        thru = highs[ti] >= L + C
        rev = lows[ti] <= L - R
    else:
        thru = lows[ti] <= L - C
        rev = highs[ti] >= L + R
    if thru and rev:
        return 'ambiguous', ti, 0
    if thru:
        return 'break', ti, 0
    end_r = min(n, ti + 1 + h_res)
    hs, ls = highs[ti + 1:end_r], lows[ti + 1:end_r]
    if side > 0:
        b = hs >= L + C
        h = ls <= L - R
    else:
        b = ls <= L - C
        h = hs >= L + R
    bi = np.argmax(b) if b.any() else 10 ** 9
    hi = np.argmax(h) if h.any() else 10 ** 9
    if bi == 10 ** 9 and hi == 10 ** 9:
        return 'unresolved', ti, -1
    if bi == hi:
        return 'ambiguous', ti, int(bi)
    return ('break', ti, int(bi)) if bi < hi else ('hold', ti, int(hi))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - hw, c + hw


def run_day(day, args, rng):
    d5 = atlas(day, '5s')
    d1 = atlas(day, '1m')
    c5, h5, l5 = d5['close'].to_numpy(), d5['high'].to_numpy(), d5['low'].to_numpy()
    ts5 = d5['timestamp'].to_numpy()
    c1, ts1 = d1['close'].to_numpy(), d1['timestamp'].to_numpy()

    up_f, lo_f, mid_f = rolling_ols_bands(c5, args.fast_window)
    up_s1, lo_s1, mid_s1 = rolling_ols_bands(c1, args.slow_window)
    # 1m band known at bar close (start-of-bar ts + 60); map to 5s grid
    idx1 = np.searchsorted(ts1 + 60, ts5, side='right') - 1
    ok1 = idx1 >= 0
    up_s = np.where(ok1, up_s1[np.clip(idx1, 0, None)], np.nan)
    lo_s = np.where(ok1, lo_s1[np.clip(idx1, 0, None)], np.nan)

    fams = {
        'fast_band': (up_f, lo_f),
        'slow_band': (up_s, lo_s),
        'fast_extreme': (rolling_extreme(up_f, args.ext_window, 'max'),
                         rolling_extreme(lo_f, args.ext_window, 'min')),
        'slow_extreme': (rolling_extreme(up_s, args.ext_window, 'max'),
                         rolling_extreme(lo_s, args.ext_window, 'min')),
    }

    R, C, tol = args.r_ticks * TICK, args.r_ticks * TICK, args.tol_ticks * TICK
    h_touch, h_res = args.touch_min * 12, args.res_min * 12
    events = {}   # (fam, kind, level_rounded, touch_idx) -> (outcome, res_bars)
    for t0 in range(args.fast_window, len(c5) - 12, 12):   # freeze every 1 min
        price = c5[t0]
        for fam, (up, lo) in fams.items():
            for L in (up[t0], lo[t0]):
                if not np.isfinite(L):
                    continue
                dist = L - price
                side = 1 if dist > 0 else -1
                if abs(dist) < tol + TICK or abs(dist) > 100 * TICK:
                    continue
                for kind in ('real', 'phantom'):
                    if kind == 'phantom':
                        j = rng.uniform(args.jit_lo, args.jit_hi) * TICK * rng.choice([-1, 1])
                        Lx = L + j
                        if (Lx - price) * side <= tol + TICK:
                            continue
                    else:
                        Lx = L
                    out, ti, rb = first_outcome(h5, l5, t0 + 1, Lx, side, R, C,
                                                tol, h_touch, h_res)
                    if out == 'notouch':
                        continue
                    key = (fam, kind, round(Lx / TICK), ti)
                    if key not in events:
                        events[key] = (out, rb)
    return events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=str,
                    default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26,2024_02_27")
    ap.add_argument('--fast-window', type=int, default=120)   # 10 min of 5s bars
    ap.add_argument('--slow-window', type=int, default=60)    # 60 min of 1m bars
    ap.add_argument('--ext-window', type=int, default=720)    # 60 min of 5s bars
    ap.add_argument('--r-ticks', type=float, default=8)       # symmetric barrier
    ap.add_argument('--tol-ticks', type=float, default=2)
    ap.add_argument('--touch-min', type=int, default=30)
    ap.add_argument('--res-min', type=int, default=15)
    ap.add_argument('--jit-lo', type=float, default=4)
    ap.add_argument('--jit-hi', type=float, default=16)
    args = ap.parse_args()
    days = [d.strip() for d in args.days.split(',')]
    rng = np.random.default_rng(0)

    agg = {}     # (fam, kind) -> [hold, total]
    per_day = {} # (day, kind) -> [hold, total]
    res_times = {'real': [], 'phantom': []}
    dropped = {'unresolved': 0, 'ambiguous': 0}
    for day in days:
        ev = run_day(day, args, rng)
        for (fam, kind, _, _), (out, rb) in ev.items():
            if out in dropped:
                dropped[out] += 1
                continue
            a = agg.setdefault((fam, kind), [0, 0])
            a[1] += 1
            a[0] += (out == 'hold')
            d = per_day.setdefault((day, kind), [0, 0])
            d[1] += 1
            d[0] += (out == 'hold')
            res_times[kind].append(rb * 5 / 60.0)   # minutes
        log(f"{day}: {sum(v[1] for k, v in per_day.items() if k[0] == day)} resolved touches")

    log('')
    log(f"barriers R=C={args.r_ticks} ticks (RW null = 0.500), tol={args.tol_ticks} ticks; "
        f"dropped: {dropped}")
    log(f"{'family':<14}{'kind':<9}{'N':>6}{'P(hold)':>9}{'95% CI':>18}")
    for fam in ['fast_band', 'slow_band', 'fast_extreme', 'slow_extreme']:
        for kind in ['real', 'phantom']:
            k, n = agg.get((fam, kind), [0, 0])
            p, a, b = wilson(k, n)
            log(f"{fam:<14}{kind:<9}{n:>6}{p:>9.3f}   [{a:.3f}, {b:.3f}]")
    log('')
    log('per-day (all families pooled):')
    for day in days:
        kr, nr = per_day.get((day, 'real'), [0, 0])
        kp, np_ = per_day.get((day, 'phantom'), [0, 0])
        pr = kr / nr if nr else float('nan')
        pp = kp / np_ if np_ else float('nan')
        log(f"  {day}: real {pr:.3f} (n={nr}) | phantom {pp:.3f} (n={np_})")
    for kind in ['real', 'phantom']:
        rt = np.array(res_times[kind])
        if len(rt):
            log(f"time-to-resolution ({kind}): median {np.median(rt):.1f} min, "
                f"75% within {np.percentile(rt, 75):.1f} min (n={len(rt)})")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'level_hold_results.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
