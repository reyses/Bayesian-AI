"""Do running legs stretch TO prior touch-points? (Moises, 2026-07-07)

Pairs the leg-momentum finding with levels: momentum says the leg keeps going;
the question here is HOW FAR -- does a leg run to the nearest prior swing
extreme (prior touch-point) ahead of it and stop, so that distance-to-level
predicts the leg's stretch?

For each leg (zigzag on 1m), at the leg's START take the nearest PRIOR pivot
(same session, earlier) that lies AHEAD in the leg's direction = the target.
Measure:
  Q1 termination: is the leg's end within k*sigma of that target, more than a
     phantom-target null (target value jittered)?
  Q2 stretch:     does distance(start->target) predict the leg's actual extent?
     (corr, and how often the leg stops SHORT of / AT / BEYOND the target)

sigma = residual scale of a trailing 60-bar OLS on 1m (volatility-relative,
per the 2026-07-07 correction).
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'research', 'level_hold', 'tools'))
from level_hold_study import atlas, rolling_ols_bands  # noqa: E402
from pivot_level_proximity import zigzag_pivots  # noqa: E402

TICK = 0.25
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s); lines.append(s)


def run(days, thr, rng):
    near_real = near_null = n = 0
    dists, extents = [], []
    short_at_beyond = [0, 0, 0]  # stops short / near / beyond target
    for day in days:
        try:
            d1 = atlas(day, '1m')
        except Exception:
            continue
        c = d1['close'].to_numpy()
        _, _, _, sig = rolling_ols_bands(c, 60, return_sigma=True)
        piv = zigzag_pivots(c, thr)
        if len(piv) < 4:
            continue
        pv_prices = c[piv]
        for li in range(2, len(piv) - 1):
            a, b = piv[li], piv[li + 1]
            start, end = c[a], c[b]
            direction = np.sign(end - start)
            if direction == 0:
                continue
            s = sig[a]
            if not np.isfinite(s) or s <= 0:
                continue
            # prior pivots ahead in the leg direction (strictly earlier than a)
            prior = pv_prices[:li]
            ahead = prior[prior > start] if direction > 0 else prior[prior < start]
            if len(ahead) == 0:
                continue
            target = ahead.min() if direction > 0 else ahead.max()  # nearest ahead
            dist = abs(target - start)
            extent = abs(end - start)
            n += 1
            dists.append(dist / TICK)
            extents.append(extent / TICK)
            tol = 1.0 * s
            near_real += abs(end - target) <= tol
            # phantom target: jitter 4-16 ticks, same side
            jit = rng.uniform(4, 16) * TICK * rng.choice([-1, 1])
            near_null += abs(end - (target + jit)) <= tol
            # short / at / beyond
            reach = extent / max(dist, 1e-9)
            short_at_beyond[0 if reach < 0.8 else (1 if reach <= 1.2 else 2)] += 1
    return n, near_real, near_null, np.array(dists), np.array(extents), short_at_beyond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=int, default=20)
    args = ap.parse_args()
    import glob
    days = sorted(os.path.basename(f).replace('.parquet', '')
                  for f in glob.glob(os.path.join(_REPO, 'DATA', 'ATLAS', '1m', '2024_*.parquet')))
    rng = np.random.default_rng(0)
    n, nr, nn, dists, ext, sab = run(days, args.thr, rng)

    log(f"2024 | zigzag {args.thr}t | {n} legs with a prior target ahead")
    log(f"\nQ1 termination near target (within 1 sigma):")
    log(f"  real   : {nr/n:.3f}")
    log(f"  phantom: {nn/n:.3f}   (jittered target null)")
    log(f"  lift   : {(nr-nn)/n:+.3f}")
    log(f"\nQ2 stretch vs distance-to-target:")
    r = np.corrcoef(np.log(dists + 1), np.log(ext + 1))[0, 1]
    log(f"  corr(log dist-to-target, log extent) = {r:.3f}")
    log(f"  leg stops  SHORT(<0.8x): {sab[0]/n:.3f}  "
        f"AT(0.8-1.2x): {sab[1]/n:.3f}  BEYOND(>1.2x): {sab[2]/n:.3f}")
    log(f"  median dist-to-target: {np.median(dists):.0f}t | median extent: {np.median(ext):.0f}t")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f'leg_target_thr{args.thr}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
