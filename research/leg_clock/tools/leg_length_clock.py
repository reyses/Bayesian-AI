"""Is a directional leg's length MEMORYLESS or CLOCKED? (Moises, 2026-07-07)

His framing: forget bar-to-bar slope persistence. Segment sessions into legs
(the macro ebbs and flows). Learn the leg-length distribution in 2024. Then in
2025, out-of-sample, ask: given a leg has already run T minutes, how much more
should it run? If that "expected remaining" is learnable and transfers, the
leg is CLOCKED (tradeable). If remaining is constant regardless of elapsed, it
is MEMORYLESS (elapsed tells you nothing).

Core object = the mean-residual-life curve  MRL(t) = E[D - t | D > t], where D
is total leg duration:
  - flat  -> memoryless (exponential); how long it's run says nothing.
  - falling-> clocked; the longer it's run, the less it has left (predict the end).
  - rising-> fat tail / momentum; the longer it runs, the more it keeps going.
The load-bearing test: does the 2024 MRL curve PREDICT 2025's? (train/test overlay)

Legs via fixed-reversal zigzag on 1m closes. Tail (the big drives) reported
separately from the mode, because the mode is the boring legs and the tail is
the move that actually pays.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'research', 'level_hold', 'tools'))
from level_hold_study import atlas  # noqa: E402
from pivot_level_proximity import zigzag_pivots  # noqa: E402 (fixed two-phase zigzag)

TICK = 0.25
REPORT_DIR = os.path.join(_REPO, 'research', 'leg_clock', 'reports')
lines = []


def log(s):
    print(s); lines.append(s)


def legs_for_days(days, thr_ticks):
    """Return arrays of leg (duration_min, extent_ticks) across all days."""
    dur, ext = [], []
    for day in days:
        try:
            d1 = atlas(day, '1m')
        except Exception:
            continue
        c = d1['close'].to_numpy()
        ts = d1['timestamp'].to_numpy()
        piv = zigzag_pivots(c, thr_ticks)
        if len(piv) < 2:
            continue
        for a, b in zip(piv[:-1], piv[1:]):
            dur.append((ts[b] - ts[a]) / 60.0)          # minutes
            ext.append(abs(c[b] - c[a]) / TICK)          # ticks
    return np.array(dur), np.array(ext)


def mrl_curve(dur, grid):
    """Mean residual life E[D - t | D > t] on a grid of elapsed times t."""
    out = np.full(len(grid), np.nan)
    for i, t in enumerate(grid):
        sub = dur[dur > t]
        if len(sub) >= 30:
            out[i] = sub.mean() - t
    return out


def pct(a, p):
    return float(np.percentile(a, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thr', type=int, default=20, help='zigzag reversal (ticks)')
    args = ap.parse_args()

    all_days = sorted(os.path.basename(f).replace('.parquet', '')
                      for f in __import__('glob').glob(
                          os.path.join(_REPO, 'DATA', 'ATLAS', '1m', '*.parquet')))
    train = [d for d in all_days if d.startswith('2024_')]
    test = [d for d in all_days if d.startswith('2025_')]

    dur_tr, ext_tr = legs_for_days(train, args.thr)
    dur_te, ext_te = legs_for_days(test, args.thr)
    log(f"zigzag {args.thr} ticks | 2024 legs: {len(dur_tr)} ({len(train)} days) | "
        f"2025 legs: {len(dur_te)} ({len(test)} days)")

    log("\n-- leg DURATION distribution (minutes) --")
    log(f"{'set':<6}{'mode~':>7}{'median':>8}{'mean':>7}{'p90':>7}{'p99':>7}{'max':>8}")
    for tag, d in [('2024', dur_tr), ('2025', dur_te)]:
        hist, edges = np.histogram(d, bins=np.arange(0, 60, 1))
        mode = edges[hist.argmax()] + 0.5
        log(f"{tag:<6}{mode:>7.1f}{np.median(d):>8.1f}{d.mean():>7.1f}"
            f"{pct(d,90):>7.1f}{pct(d,99):>7.1f}{d.max():>8.1f}")

    log("\n-- leg EXTENT distribution (ticks) --")
    log(f"{'set':<6}{'median':>8}{'mean':>7}{'p90':>7}{'p99':>7}{'max':>8}")
    for tag, e in [('2024', ext_tr), ('2025', ext_te)]:
        log(f"{tag:<6}{np.median(e):>8.1f}{e.mean():>7.1f}{pct(e,90):>7.1f}"
            f"{pct(e,99):>7.1f}{e.max():>8.1f}")

    # The load-bearing test: MRL curve, 2024 (train) vs 2025 (OOS).
    # Adaptive elapsed grid = percentiles of train durations, so it scales
    # whether legs are 3-min micro-wiggles or hour-long macro drives.
    grid = np.unique(np.round(np.percentile(
        dur_tr, [0, 10, 25, 40, 55, 70, 82, 90]), 1))
    mrl_tr = mrl_curve(dur_tr, grid)
    mrl_te = mrl_curve(dur_te, grid)
    # exponential (memoryless) reference: MRL(t) = mean, constant.
    mem_ref = dur_tr.mean()
    log("\n-- MEAN RESIDUAL LIFE: expected MORE minutes given elapsed --")
    log("(memoryless=flat at the mean; clocked=falling; momentum=rising)")
    log(f"{'elapsed t':>10}{'2024 MRL':>10}{'2025 MRL':>10}{'memoryless':>12}")
    for i, t in enumerate(grid):
        log(f"{t:>10.0f}{mrl_tr[i]:>10.1f}{mrl_te[i]:>10.1f}{mem_ref:>12.1f}")

    # Verdict helpers: slope of MRL over 0..12 min, and train/test agreement.
    m = np.isfinite(mrl_tr) & np.isfinite(mrl_te) & (grid <= np.median(dur_tr) * 2)
    if m.sum() >= 3:
        s_tr = np.polyfit(grid[m], mrl_tr[m], 1)[0]
        agree = np.mean(np.abs(mrl_tr[m] - mrl_te[m]))
        shape = 'CLOCKED (falling)' if s_tr < -0.15 else (
            'MOMENTUM (rising)' if s_tr > 0.15 else 'MEMORYLESS (flat)')
        log(f"\nMRL slope (2024, 0-12min): {s_tr:+.2f} min per min elapsed -> {shape}")
        log(f"2024 vs 2025 MRL mean abs diff: {agree:.2f} min "
            f"(small = the clock transfers out-of-sample)")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f'leg_clock_thr{args.thr}.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
