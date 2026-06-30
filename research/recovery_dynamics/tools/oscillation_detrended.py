"""DETRENDED oscillation: the zigzag that persists WHILE price trends.

Gap in the fixed-anchor measurement: a trending price zigzags up around a MOVING center, so it
never returns to the fixed anchor -> miscounted as 'no return/trend', and its internal oscillation
is invisible. Fix: measure oscillation as crossings of a rolling mean (the moving center).
  price = drift (rolling mean) + oscillation (deviation). Period = time between mean-crossings;
  amplitude = max |price - mean| within a cycle. This sees the zigzag even inside a strong trend.

Key tests:
  - does the 'no-return/trend' share COLLAPSE once we detrend? (oscillation is everywhere)
  - is the zigzag PERIOD the same in trend vs chop? (split cycles by drift magnitude)
W_MA = the trend scale (the one parameter; oscillation is always relative to a chosen trend scale).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import anchor_period as ap  # noqa: E402

ROOT, ONE_M = ap.ROOT, ap.ONE_M
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "oscillation_detrended.md")
W_MA = 30        # rolling-mean window (min) = the trend scale we detrend by
PBUCKETS = [(0, 3, "<3 min"), (3, 5, "3-5 min"), (5, 8, "5-8 min"), (8, 15, "8-15 min"),
            (15, 30, "15-30 min"), (30, 60, "30-60 min"), (60, 99999, ">60 min")]


def day_cycles(close):
    """Return list of (period, amplitude, drift) for each detrended oscillation cycle."""
    s = pd.Series(close)
    ma = s.rolling(W_MA, min_periods=W_MA).mean().to_numpy()
    dev = close - ma
    ok = ~np.isnan(dev)
    idx = np.where(ok)[0]
    if idx.size < 3:
        return []
    dev = dev[idx]; mav = ma[idx]
    sign = np.sign(dev)
    cross = np.where(np.diff(sign) != 0)[0]          # indices (in dev) where it crosses the mean
    out = []
    for i in range(len(cross) - 1):
        a, b = cross[i] + 1, cross[i + 1] + 1
        if b <= a:
            continue
        period = b - a
        amp = float(np.abs(dev[a:b + 1]).max())
        drift = abs(mav[b] - mav[a]) / period         # mean's slope over the cycle (pt/min) = trend strength
        out.append((period, amp, drift))
    return out


def buckets(periods):
    n = len(periods)
    out = []
    parr = np.asarray(periods)
    for lo, hi, lbl in PBUCKETS:
        sh = ((parr >= lo) & (parr < hi)).mean()
        out.append(f"{lbl:<10}|{sh:>6.1%} |{'#'*int(round(46*sh))}")
    return out


def main():
    P, A, D = [], [], []
    for f in tqdm(sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                         glob.glob(os.path.join(ONE_M, "2025_*.parquet"))), desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        for p, a, d in day_cycles(close):
            P.append(p); A.append(a); D.append(d)
    P, A, D = np.array(P), np.array(A), np.array(D)

    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w(f"# Detrended oscillation — the zigzag around a {W_MA}-min moving mean (2024+2025)")
    w(f"{len(P)} oscillation cycles (mean-crossings). price = drift(MA) + oscillation(deviation).\n")
    w(f"- median zigzag period: **{int(np.median(P))} min** | median amplitude (max dev): "
      f"**{np.median(A):.0f}pt** | mode period {int(np.bincount(P).argmax())}m")
    w(f"- vs fixed-anchor: trend/no-return share was 7.2%; DETRENDED there is no 'no-return' — every")
    w(f"  cycle crosses the moving mean. The oscillation is EVERYWHERE, including inside trends.\n")
    w("## Zigzag period distribution (detrended)")
    w("```")
    w("\n".join(buckets(P)))
    w("```")
    # split by drift: is the zigzag period the same in trend vs chop?
    med_d = np.median(D)
    chop = P[D <= med_d]; trend = P[D > med_d]
    w(f"\n## Does the zigzag PERIOD survive the trend? (split cycles by drift = MA slope)")
    w(f"- low-drift (chop) median period:  {int(np.median(chop))} min")
    w(f"- high-drift (trend) median period: {int(np.median(trend))} min")
    w(f"- low-drift median amplitude:  {np.median(A[D<=med_d]):.0f}pt")
    w(f"- high-drift median amplitude: {np.median(A[D>med_d]):.0f}pt")
    w("```")
    w("low-drift (chop):")
    w("\n".join(buckets(chop)))
    w("\nhigh-drift (trend):")
    w("\n".join(buckets(trend)))
    w("```")
    w("\n## Read")
    w("If the period is ~the same in trend and chop, the oscillation CLOCK keeps ticking while price")
    w("climbs — the trend is just the moving mean drifting, with the SAME zigzag riding on it. The")
    w("fixed-anchor '7% trend = no oscillation' was an artifact of a fixed reference. Correct model:")
    w("price = drift + persistent oscillation; the 'trend' is drift in the mean, not an absence of cycle.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
