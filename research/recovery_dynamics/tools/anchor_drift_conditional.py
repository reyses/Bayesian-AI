"""Does price keep returning to the anchor WHILE TRENDING? (Moises' point, checked directly.)

Because we anchor EVERY bar, the within-trend zigzag should already show up as returns to recent
anchors. Test: condition each anchor's outcome on the LOCAL TREND STRENGTH (trailing drift). If the
return rate stays high even in strong drift, the fixed every-bar anchor already captures oscillation
during trends, and the 'no-return' anchors are just the abandoned trailing levels (the trend's footprint).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import anchor_period as ap  # noqa: E402

ROOT, ONE_M, MAXLOOK, STRIDE = ap.ROOT, ap.ONE_M, ap.MAXLOOK, ap.STRIDE
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "anchor_drift_conditional.md")
DRIFTW = 30   # trailing window (min) to measure the local trend the anchor is born into


def day(close):
    n = len(close)
    out = []   # (abs_drift_pt_per_min, returned(0/1), period)
    for a in range(DRIFTW, n - 2, STRIDE):
        P = close[a]
        drift = abs(close[a] - close[a - DRIFTW]) / DRIFTW
        fwd = close[a + 1:a + 1 + MAXLOOK]
        if fwd.size < 2:
            break
        side = np.sign(fwd[0] - P)
        if side == 0:
            continue
        crossed = np.where((fwd - P) * side <= 0)[0]
        if crossed.size:
            out.append((drift, 1, int(crossed[0] + 1)))
        else:
            out.append((drift, 0, MAXLOOK))
    return out


def main():
    D, R, P = [], [], []
    for f in tqdm(sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                         glob.glob(os.path.join(ONE_M, "2025_*.parquet"))), desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        for d, r, p in day(close):
            D.append(d); R.append(r); P.append(p)
    D, R, P = np.array(D), np.array(R), np.array(P)

    # quintiles of trailing drift: calmest -> most-trending
    qs = np.quantile(D, [0, .2, .4, .6, .8, 1.0])
    labels = ["calmest 20%", "2nd", "middle", "4th", "MOST-TREND 20%"]
    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# Does price return to the anchor WHILE TRENDING? (every-bar, conditioned on trailing drift)")
    w(f"{len(D)} anchors, trailing drift over {DRIFTW} min. If return rate stays high in strong drift,")
    w("the fixed every-bar anchor already captures the within-trend zigzag.\n")
    w(f"{'drift quintile':<16}| drift(pt/min) | return rate | median period (returned)")
    w("-" * 74)
    for i in range(5):
        m = (D >= qs[i]) & (D <= qs[i + 1]) if i == 4 else (D >= qs[i]) & (D < qs[i + 1])
        rr = R[m].mean()
        per = P[m][R[m] == 1]
        w(f"{labels[i]:<16}| {qs[i]:.3f}-{qs[i+1]:.3f} | {rr:>9.1%}  | "
          f"{int(np.median(per)) if per.size else 0} min")
    w("")
    w(f"overall return rate: {R.mean():.1%}  (no-return = abandoned levels = trend footprint)")
    w("\n## Read")
    w("If the most-trending quintile still returns most of the time, Moises is right: while trending,")
    w("price keeps coming back to recent anchors (the zigzag), because we anchor every bar. The")
    w("no-return share rising only modestly with drift = the trend abandons only its trailing levels,")
    w("not the oscillation. The fixed every-bar measurement already contains the within-trend oscillation.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
