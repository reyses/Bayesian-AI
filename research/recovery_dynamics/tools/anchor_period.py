"""MEASUREMENT (not trading): the oscillation PERIOD, sampled by anchoring (almost) every bar.

Drop an anchor at a bar's price; measure how many bars until price returns THROUGH that level
(first-return time). That is one oscillation period, sampled at that point. Do it at (almost) every
bar -> the full period field. Anchors price never returns to (within the look cap) are censored =
the trend, measured as 'no return', not traded.

This is the empirical first-return-time distribution (the mean-reversion timescale of an OU process).
No positions, no PnL — pure measurement of the period.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import opportunity_cost as oc  # noqa: E402

ROOT, ONE_M = oc.ROOT, oc.ONE_M
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "anchor_period.md")

STRIDE = 1        # anchor every bar ('almost' every bar if >1)
MAXLOOK = 360     # forward cap (min); beyond this = no-return (censored = trend)


def day_periods(close):
    n = len(close)
    periods, noreturn, total = [], 0, 0
    for a in range(0, n - 2, STRIDE):
        P = close[a]
        fwd = close[a + 1:a + 1 + MAXLOOK]
        if fwd.size < 2:
            break
        side = np.sign(fwd[0] - P)
        if side == 0:
            continue                                   # flat next bar -> no defined excursion
        total += 1
        crossed = np.where((fwd - P) * side <= 0)[0]   # first return THROUGH the anchor level
        if crossed.size:
            periods.append(int(crossed[0] + 1))        # +1: fwd starts at bar a+1
        else:
            noreturn += 1
    return periods, noreturn, total


def hist(vals, width, cap=24):
    nb = min(int(max(vals) // width) + 1, cap)
    c = [0] * nb
    for v in vals:
        c[min(int(v // width), nb - 1)] += 1
    pk = max(c) or 1
    out = [f"first-return time (oscillation period), mode {c.index(max(c))*width}-{(c.index(max(c))+1)*width}m, "
           f"median {int(np.median(vals))}m, n={len(vals)}"]
    for i in range(nb):
        tag = f">{cap*width-width}m" if (i == nb - 1 and max(vals) // width >= cap) else f"{i*width:>3}-{(i+1)*width:<3}m"
        out.append(f"{tag:>8}|{'#'*int(round(46*c[i]/pk))} {c[i]}")
    return out


def main():
    rep = {}
    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# Oscillation period — first-return time, anchored (almost) every bar")
    w(f"STRIDE={STRIDE} bar(s), forward cap {MAXLOOK} min (beyond = no-return/censored = trend). "
      f"Pure measurement — no positions.\n")
    for year in ("2024", "2025"):
        per, nr, tot = [], 0, 0
        for f in tqdm(sorted(glob.glob(os.path.join(ONE_M, f"{year}_*.parquet"))), desc=year, unit="day"):
            try:
                close = pd.read_parquet(f)["close"].to_numpy(np.float64)
            except Exception:
                continue
            p, n_, t = day_periods(close)
            per += p; nr += n_; tot += t
        rep[year] = (per, nr, tot)
        w(f"## {year}: {tot} anchors")
        w(f"- returned (finite period): {len(per)}/{tot} = {len(per)/tot:.1%}")
        w(f"- NO return within {MAXLOOK}m (censored = trend): {nr}/{tot} = {nr/tot:.1%}")
        w(f"- mode period ~{max(set(per), key=per.count)}m | median {int(np.median(per))}m | "
          f"mean {np.mean(per):.0f}m")
        w("```")
        w("\n".join(hist(per, 10)))
        w("```\n")
    if "2024" in rep and "2025" in rep:
        w("## 2024 vs 2025")
        w(f"- median period: {int(np.median(rep['2024'][0]))}m vs {int(np.median(rep['2025'][0]))}m")
        w(f"- no-return (trend) share: {rep['2024'][1]/rep['2024'][2]:.1%} vs {rep['2025'][1]/rep['2025'][2]:.1%}")
    w("\n## Read")
    w("This is the period field: how long price takes to return to an arbitrary level (the oscillation")
    w("timescale), with the no-return share = the trend (censored) fraction. It is a measurement of the")
    w("market's oscillation period, sampled everywhere — the empirical OU first-return-time distribution.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
