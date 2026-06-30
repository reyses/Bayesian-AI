"""THE EXERCISE (multi-year): the cost — and the PERIOD — of holding a wrong trade to breakeven.

Per Moises: pick a RANDOM entry (bar + long/short), let it go underwater, hold until PnL returns
to ZERO. Key insight: returning to zero = price left the entry level and came back = **ONE FULL
OSCILLATION** of the local cycle. So bars-to-breakeven IS the market's oscillation-period sample,
and the swings missed in that window are the opportunity cost of not realizing we were wrong.

Run across ALL 2024 + 2025 days, random seeded entries, split by year.

Causal-honest: the grader sees the forward path; the 'trade' predicts nothing.
  - random (bar, dir). WRONG trade = draws down >= MIN_ADVERSE_PTS before it ever profits.
  - oscillation period = bars from entry to first return to >= 0 PnL. Never within day -> censored/never.
  - foregone = tradeable swings (>= SWING_PTS) inside that window.
"""
import glob
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "recovery_2024_2025.md")

# --- tunable thresholds (named, not magic) ---
MIN_ADVERSE_PTS = 5.0    # "wrong/underwater" once it draws down >= this from entry
PROFIT_FIRST_PTS = 5.0   # profits >= this BEFORE drawing down -> a 'right' trade -> skip
SWING_PTS = 8.0          # a foregone swing must travel >= this to count
N_PER_DAY = 80           # random entries sampled per day
ENTRY_FRAC = 0.85        # only sample entries in the first 85% of the day (leave recovery room)
SEED = 0                 # reproducible RNG
POINT_VALUE = 2.0        # MNQ $ per point


def swing_legs(closes, lo, hi, thresh):
    seg = closes[lo:hi + 1]
    if len(seg) < 2:
        return 0
    legs, direction, last_pivot, extreme = 0, 0, seg[0], seg[0]
    for p in seg[1:]:
        if direction == 0:
            if p - last_pivot >= thresh:
                direction = 1; extreme = p; legs += 1
            elif last_pivot - p >= thresh:
                direction = -1; extreme = p; legs += 1
        elif direction == 1:
            if p > extreme:
                extreme = p
            elif extreme - p >= thresh:
                legs += 1; direction = -1; last_pivot = extreme; extreme = p
        else:
            if p < extreme:
                extreme = p
            elif p - extreme >= thresh:
                legs += 1; direction = 1; last_pivot = extreme; extreme = p
    return legs


def process_day(close, rng):
    n = len(close)
    hi_entry = int(n * ENTRY_FRAC)
    out = {"period": [], "foregone": [], "depth": [], "never": 0}
    if hi_entry < 5:
        return out
    for e in rng.integers(0, hi_entry, size=N_PER_DAY):
        e = int(e)
        d = 1 if rng.random() < 0.5 else -1
        pnl = d * (close[e + 1:] - close[e])
        if pnl.size < 2:
            continue
        adv = np.where(pnl <= -MIN_ADVERSE_PTS)[0]
        pro = np.where(pnl >= PROFIT_FIRST_PTS)[0]
        if len(adv) == 0:
            continue                                  # never underwater
        if len(pro) and pro[0] < adv[0]:
            continue                                  # profited first -> right trade
        back = np.where(pnl[adv[0]:] >= 0)[0]
        if len(back) == 0:
            out["never"] += 1
            continue
        zero_bar = adv[0] + back[0]
        out["period"].append(int(zero_bar + 1))       # oscillation period (min)
        out["depth"].append(float(-pnl[:zero_bar + 1].min()))
        out["foregone"].append(swing_legs(close, e, e + zero_bar + 1, SWING_PTS))
    return out


def hist(vals, width, label, unit):
    if not vals:
        return [f"  (no {label})"]
    nb = int(max(vals) // width) + 1
    counts = [0] * nb
    for v in vals:
        counts[min(int(v // width), nb - 1)] += 1
    peak = max(counts) or 1
    cap = min(nb, 22)
    mode_bin = counts.index(peak)
    out = [f"{label} (mode {mode_bin*width:.0f}-{(mode_bin+1)*width:.0f}{unit}, "
           f"median {int(np.median(vals))}{unit}, n={len(vals)})"]
    for i in range(cap):
        bar = "#" * int(round(40 * counts[i] / peak))
        out.append(f"{i*width:>4.0f}-{(i+1)*width:<4.0f}{unit} |{bar} {counts[i]}")
    if nb > cap:
        out.append(f"   >{cap*width:.0f}{unit} |  (+{sum(counts[cap:])} in tail)")
    return out


def main():
    rng = np.random.default_rng(SEED)
    years = {"2024": sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet"))),
             "2025": sorted(glob.glob(os.path.join(ONE_M, "2025_*.parquet")))}
    agg = {y: {"period": [], "foregone": [], "depth": [], "never": 0, "days": 0} for y in years}
    for y, files in years.items():
        for f in tqdm(files, desc=f"{y}", unit="day"):
            try:
                close = pd.read_parquet(f)["close"].to_numpy(np.float64)
            except Exception:
                continue
            r = process_day(close, rng)
            for k in ("period", "foregone", "depth"):
                agg[y][k] += r[k]
            agg[y]["never"] += r["never"]
            agg[y]["days"] += 1

    L = []
    def w(s):
        print(s); L.append(s)
    w("# Oscillation period & opportunity cost of holding a WRONG trade to breakeven")
    w(f"2024 + 2025, 1-min, RANDOM seeded entries ({N_PER_DAY}/day). "
      f"Return-to-zero = ONE FULL OSCILLATION around the entry level.")
    w(f"thresholds: underwater>={MIN_ADVERSE_PTS}pt, swing>={SWING_PTS}pt, entry<={ENTRY_FRAC:.0%} of day\n")
    for y in years:
        a = agg[y]
        tot = len(a["period"]) + a["never"]
        w(f"## {y}  ({a['days']} days, {tot} wrong trades, "
          f"{a['never']} never-recovered = {a['never']/max(tot,1):.0%})")
        if a["period"]:
            w(f"- MODE oscillation period ~{max(set(a['period']), key=a['period'].count)} min | "
              f"median {int(np.median(a['period']))} min")
            w(f"- MODE foregone ~{max(set(a['foregone']), key=a['foregone'].count)} | "
              f"median {int(np.median(a['foregone']))} trades | median depth {np.median(a['depth']):.1f}pt "
              f"(${np.median(a['depth'])*POINT_VALUE:.0f})")
            w("```")
            w("\n".join(hist(a["period"], 15, "OSCILLATION PERIOD (entry->breakeven)", "m")))
            w("")
            w("\n".join(hist(a["foregone"], 2, "TRADES FOREGONE in that window", "")))
            w("```")
        w("")
    # year comparison
    if agg["2024"]["period"] and agg["2025"]["period"]:
        m24, m25 = np.median(agg["2024"]["period"]), np.median(agg["2025"]["period"])
        w(f"## 2024 vs 2025\n- median oscillation period: 2024 **{int(m24)} min** vs 2025 **{int(m25)} min**")
        w(f"- never-recovered rate: 2024 {agg['2024']['never']/max(len(agg['2024']['period'])+agg['2024']['never'],1):.0%}"
          f" vs 2025 {agg['2025']['never']/max(len(agg['2025']['period'])+agg['2025']['never'],1):.0%}")
        w("- => the recovery CLOCK is stable across years iff these medians agree (the cut-threshold's validity).")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
