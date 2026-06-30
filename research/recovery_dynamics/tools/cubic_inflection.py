"""Moises' hypothesis: the 7% no-return anchors are TREND-START inflection points (convex/concave),
and a CUBIC regression lights up there — strongest in the large-period zones.

Test (descriptive; centered window can see both sides — the no-return label is itself forward):
  - at sampled anchors, fit a cubic on a centered +/-CW window; measure cubic strength |d3| and whether
    an inflection (y''=0) sits inside the window near the anchor.
  - compare NO-RETURN vs RETURN anchors.
  - relate cubic strength to the period (do large-period zones carry stronger cubics?).
If no-return anchors show stronger/closer inflections, the hypothesis holds -> next: causal (trailing)
cubic as a trend-launch detector.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import anchor_period as ap  # noqa: E402

ROOT, ONE_M, MAXLOOK = ap.ROOT, ap.ONE_M, ap.MAXLOOK
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "cubic_inflection.md")
CW = 20          # centered half-window (min) for the cubic fit
SAMPLE = 60      # anchors sampled per day (compute budget)
XN = np.linspace(-1, 1, 2 * CW + 1)
PB = [(2, 5), (5, 15), (15, 30), (30, 60), (60, MAXLOOK), (MAXLOOK, MAXLOOK + 1)]  # last = no-return


def day(close, rng):
    n = len(close)
    rows = []
    lo, hi = CW, n - CW - 2
    if hi <= lo:
        return rows
    for a in rng.integers(lo, hi, size=SAMPLE):
        a = int(a)
        P = close[a]
        fwd = close[a + 1:a + 1 + MAXLOOK]
        side = np.sign(fwd[0] - P)
        if side == 0:
            continue
        crossed = np.where((fwd - P) * side <= 0)[0]
        period = int(crossed[0] + 1) if crossed.size else MAXLOOK
        returned = crossed.size > 0
        win = close[a - CW:a + CW + 1].astype(np.float64)
        y = win - win.mean()
        d3, d2, d1, d0 = np.polyfit(XN, y, 3)            # cubic coeffs (x normalized to [-1,1])
        scale = np.abs(y).max() + 1e-9
        cub = abs(d3) / scale                            # cubic strength (normalized)
        x_infl = (-d2 / (3 * d3)) if abs(d3) > 1e-9 else 99.0
        in_win = abs(x_infl) <= 1.0
        rows.append((returned, period, cub, in_win, abs(x_infl) if in_win else np.nan))
    return rows


def main():
    rng = np.random.default_rng(0)
    ret, per, cub, inw, idist = [], [], [], [], []
    for f in tqdm(sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                         glob.glob(os.path.join(ONE_M, "2025_*.parquet"))), desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        for r, p, c, iw, idd in day(close, rng):
            ret.append(r); per.append(p); cub.append(c); inw.append(iw); idist.append(idd)
    ret = np.array(ret); per = np.array(per); cub = np.array(cub)
    inw = np.array(inw); idist = np.array(idist)
    nr = ~ret

    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w(f"# Cubic / inflection at no-return anchors — Moises' hypothesis ({len(ret)} sampled anchors)")
    w(f"centered +/-{CW}-min cubic fit. |d3| = cubic strength (normalized); inflection = y''=0 in window.\n")
    w("## NO-RETURN (trend start) vs RETURN anchors")
    w(f"- cubic strength |d3|:      no-return **{cub[nr].mean():.3f}** vs return {cub[ret].mean():.3f} "
      f"(ratio {cub[nr].mean()/max(cub[ret].mean(),1e-9):.2f}x)")
    w(f"- inflection inside window: no-return **{inw[nr].mean():.0%}** vs return {inw[ret].mean():.0%}")
    w(f"- inflection distance to anchor (if in window): no-return {np.nanmedian(idist[nr]):.2f} "
      f"vs return {np.nanmedian(idist[ret]):.2f} (0=at anchor)")
    w("\n## Cubic strength |d3| by PERIOD bucket (large-period zones carry stronger cubic?)")
    w("```")
    w(f"{'period':>14} | mean |d3| | infl-in-window | n")
    for lo, hi in PB:
        m = (per >= lo) & (per < hi) if hi <= MAXLOOK else (per >= MAXLOOK)
        if m.sum() == 0:
            continue
        lbl = "NO-RETURN" if lo >= MAXLOOK else f"{lo}-{hi}m"
        w(f"{lbl:>14} | {cub[m].mean():>8.3f} | {inw[m].mean():>13.0%} | {int(m.sum())}")
    w("```")
    w("\n## Read")
    sig = cub[nr].mean() / max(cub[ret].mean(), 1e-9)
    w(f"No-return anchors carry a {'STRONGER' if sig>1.15 else '~equal'} cubic signature "
      f"({sig:.2f}x). If the cubic strength also rises with period, the large-period/trend zones ARE")
    w("the cubic/inflection zones (hypothesis supported). Descriptive only (centered window + forward")
    w("no-return label); the payoff is a CAUSAL trailing-cubic launch detector — the next build.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
