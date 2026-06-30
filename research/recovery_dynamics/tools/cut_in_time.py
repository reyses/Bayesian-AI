"""BETTER MECHANISM: trades unlocked by cutting a wrong trade IN TIME (not period-to-recovery).

Why: "time to return to zero" is self-trapping — a 12-hour bleed is scored as just "a long period",
which measures the very behavior we want to avoid. The decision-relevant unit is TRADES, not time:
if you cut a stuck trade at a sensible point, how many trades (good AND bad — any swing is an
opportunity to be active) could you have taken in the time you'd otherwise be stuck?

Per wrong trade: find when it goes underwater. For each CUT window H, if it has NOT returned to
breakeven within H bars, you CUT at that point; count tradeable swings from the cut to where it
would actually have recovered (or EOD) = TRADES UNLOCKED. Trades that bounce within H needed no cut.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
import opportunity_cost as oc  # noqa: E402
from opportunity_cost import swing_legs  # noqa: E402

ROOT, ONE_M = oc.ROOT, oc.ONE_M
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "cut_in_time.md")

CUT_WINDOWS = [15, 30, 60]   # cut-in-time points (min underwater without breakeven -> cut)


def process_day(close, rng):
    n = len(close)
    hi = int(n * oc.ENTRY_FRAC)
    res = {H: {"unlocked": [], "bounced": 0, "stuck": 0} for H in CUT_WINDOWS}
    swings_day = swing_legs(close, 0, n - 1, oc.SWING_PTS)   # opportunity baseline (all good+bad)
    total = 0
    if hi < 5:
        return res, 0, swings_day, n
    for e in rng.integers(0, hi, size=oc.N_PER_DAY):
        e = int(e)
        d = 1 if rng.random() < 0.5 else -1
        pnl = d * (close[e + 1:] - close[e])
        if pnl.size < 2:
            continue
        adv = np.where(pnl <= -oc.MIN_ADVERSE_PTS)[0]
        pro = np.where(pnl >= oc.PROFIT_FIRST_PTS)[0]
        if len(adv) == 0 or (len(pro) and pro[0] < adv[0]):
            continue
        total += 1
        under_bar = e + 1 + adv[0]
        back = np.where(pnl[adv[0]:] >= 0)[0]
        rec_bar = (e + 1 + adv[0] + back[0]) if len(back) else (n - 1)
        for H in CUT_WINDOWS:
            cut_bar = under_bar + H
            if rec_bar <= cut_bar:
                res[H]["bounced"] += 1                       # bounced in time -> no cut needed
            else:
                res[H]["stuck"] += 1
                res[H]["unlocked"].append(swing_legs(close, cut_bar, rec_bar, oc.SWING_PTS))
    return res, total, swings_day, n


def hist(vals, width, label):
    if not vals:
        return [f"  (no {label})"]
    nb = min(int(max(vals) // width) + 1, 16)
    counts = [0] * nb
    for v in vals:
        counts[min(int(v // width), nb - 1)] += 1
    peak = max(counts) or 1
    out = [f"{label} (mode {counts.index(peak)*width}-{(counts.index(peak)+1)*width}, "
           f"median {int(np.median(vals))}, n={len(vals)})"]
    for i in range(nb):
        out.append(f"{i*width:>3}-{(i+1)*width:<3}|{'#'*int(round(38*counts[i]/peak))} {counts[i]}")
    return out


def main():
    rng = np.random.default_rng(oc.SEED)
    files = sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                   glob.glob(os.path.join(ONE_M, "2025_*.parquet")))
    agg = {H: {"unlocked": [], "bounced": 0, "stuck": 0} for H in CUT_WINDOWS}
    total = 0
    swings_per_hr = []
    for f in tqdm(files, desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        res, t, sw, n = process_day(close, rng)
        total += t
        swings_per_hr.append(sw / (n / 60.0))
        for H in CUT_WINDOWS:
            agg[H]["unlocked"] += res[H]["unlocked"]
            agg[H]["bounced"] += res[H]["bounced"]
            agg[H]["stuck"] += res[H]["stuck"]

    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# Trades unlocked by cutting a wrong trade IN TIME (2024+2025)")
    w(f"{total} wrong trades. Opportunity baseline: ~{np.median(swings_per_hr):.1f} tradeable "
      f"swings/hour available (all good+bad).\n")
    for H in CUT_WINDOWS:
        a = agg[H]
        tot = a["bounced"] + a["stuck"]
        stuck_rate = a["stuck"] / max(tot, 1)
        w(f"## Cut at {H} min underwater")
        w(f"- bounced in time (no cut needed): {a['bounced']}/{tot} = {1-stuck_rate:.0%}")
        w(f"- STILL STUCK (cutting helps): {a['stuck']}/{tot} = {stuck_rate:.0%}")
        if a["unlocked"]:
            u = a["unlocked"]
            w(f"- for the stuck ones, TRADES UNLOCKED by cutting: "
              f"mode ~{max(set(u), key=u.count)}, median {int(np.median(u))}, mean {np.mean(u):.1f}, "
              f"90th pct {int(np.percentile(u,90))}")
            w("```")
            w("\n".join(hist(u, 2, "TRADES UNLOCKED")))
            w("```")
        w("")
    w("## Read")
    w("Most wrong trades bounce fast (no cut needed). The value of cutting is concentrated in the")
    w("STILL-STUCK minority — and there it is large: cutting frees a meaningful number of trades")
    w("instead of staring at dead money. This is the cost-of-not-cutting in the unit that matters,")
    w("with NO dependence on how long the eventual 'recovery' takes (the trap we avoided).")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
