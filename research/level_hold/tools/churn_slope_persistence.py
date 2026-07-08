"""Is directional churn rideable? (Moises' axiom, 2026-07-07)

His claim: it oscillates until something moves it; a "move" is not a break but
CHURN WITH SLOPE -- it wiggles while it drifts, and you can catch it mid-drive
and scalp the ebb-and-flow while riding the drift.

Test: bin every bar by its CURRENT slope-to-noise (drift over a short window /
residual sigma). For each bin measure the FORWARD outcome:
  - mean forward net move (ticks) over H bars
  - P(forward move continues in the current slope's direction)  [50% = no edge]
  - wiggle ratio = forward high-low range / |forward net|  (>1 = scalpable churn)

If flat bins -> ~0 forward drift & ~50% direction, and sloped bins -> forward
drift in the same direction & >50%, then slope PERSISTS = directional churn is
rideable. Null: overall P(up) base rate + a bar-shuffle of the slope label.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _HERE)
from level_hold_study import atlas  # noqa: E402

TICK = 0.25
REPORT_DIR = os.path.join(_REPO, 'research', 'level_hold', 'reports')
lines = []


def log(s):
    print(s); lines.append(s)


def rolling_slope_snr(close, W):
    """Per-bar causal OLS slope over trailing W bars, expressed as
    drift-over-window / residual-sigma (dimensionless). NaN for first W-1."""
    n = len(close)
    x = np.arange(W, dtype=np.float64)
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    sw = np.lib.stride_tricks.sliding_window_view(close, W)   # [n-W+1, W]
    ym = sw.mean(axis=1, keepdims=True)
    b = ((sw - ym) * (x - xm)).sum(axis=1) / sxx             # slope per bar
    fit = ym[:, 0][:, None] + b[:, None] * (x - xm)
    sig = np.sqrt(((sw - fit) ** 2).mean(axis=1)) + 1e-9
    drift = b * (W - 1)                                       # total rise across window
    snr = drift / sig
    return np.r_[np.full(W - 1, np.nan), snr]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=str, required=True)
    ap.add_argument('--slope-window', type=int, default=60)   # 5 min of 5s bars
    ap.add_argument('--horizon', type=int, default=120)       # 10 min forward
    ap.add_argument('--step', type=int, default=12)           # sample every 1 min (reduce overlap)
    args = ap.parse_args()
    days = [d.strip() for d in args.days.split(',')]

    edges = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
    names = ['down-drive', 'weak-down', 'FLAT', 'weak-up', 'up-drive']
    agg = {n: {'net': [], 'same': 0, 'tot': 0, 'wig': []} for n in names}
    n_up = n_all = 0

    for day in days:
        c = atlas(day, '5s')['close'].to_numpy()
        h = atlas(day, '5s')['high'].to_numpy()
        lo = atlas(day, '5s')['low'].to_numpy()
        snr = rolling_slope_snr(c, args.slope_window)
        H = args.horizon
        for t in range(args.slope_window, len(c) - H, args.step):
            s = snr[t]
            if not np.isfinite(s):
                continue
            net = (c[t + H] - c[t]) / TICK
            rng = (h[t + 1:t + H + 1].max() - lo[t + 1:t + H + 1].min()) / TICK
            n_all += 1; n_up += (net > 0)
            b = np.searchsorted(edges, s, side='right') - 1
            b = min(max(b, 0), len(names) - 1)
            nm = names[b]
            agg[nm]['net'].append(net)
            agg[nm]['tot'] += 1
            agg[nm]['same'] += (np.sign(net) == np.sign(s)) if s != 0 else 0
            if abs(net) > 1e-9:
                agg[nm]['wig'].append(rng / abs(net))

    log(f"days={len(days)}  slope_window={args.slope_window} bars ({args.slope_window*5/60:.0f} min)  "
        f"horizon={args.horizon} bars ({args.horizon*5/60:.0f} min)  step={args.step}")
    log(f"base rate P(up over horizon) = {n_up/n_all:.3f}  (N={n_all})")
    log(f"{'bin':<12}{'N':>7}{'mean net(ticks)':>16}{'P(same dir)':>13}{'wiggle/|net|':>13}")
    for nm in names:
        a = agg[nm]
        if a['tot'] == 0:
            continue
        mn = np.mean(a['net'])
        same = a['same'] / a['tot']
        wig = np.median(a['wig']) if a['wig'] else float('nan')
        log(f"{nm:<12}{a['tot']:>7}{mn:>16.1f}{same:>13.3f}{wig:>13.1f}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, 'churn_slope_persistence.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
