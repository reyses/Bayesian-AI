"""THE KICKER: a full oscillation is away->back->away(other side)->back. The first return to zero
is the MIDDLE of the cycle, not the end. So a WRONG trade (adverse first) that crawls back to zero
should, if it's a true oscillation, then swing FAVORABLE (the second half). Cutting at breakeven
throws that half away.

Measure: for adverse-first trades that return to zero, the SECOND-leg favorable excursion A2 (the
kicker) vs the first adverse excursion A1. Classify the second leg:
  KICKER   : favorable swing >= MIN then returns (the symmetric other half)
  JACKPOT  : favorable swing that never returns (held the loser -> it came back -> then RAN your way)
  STALL    : never reaches MIN favorable (no real second leg)
  DEATH2   : never returned to zero at all (adverse runaway -> would have been a real loss)
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
REPORT = os.path.join(ROOT, "research", "recovery_dynamics", "reports", "oscillation_kicker.md")
MIN = 5.0   # pts to count as a real excursion (either leg)


def day(close, rng):
    n = len(close)
    hi = int(n * oc.ENTRY_FRAC)
    r = {"kicker": 0, "jackpot": 0, "stall": 0, "death2": 0, "A1": [], "A2": [], "ratio": []}
    if hi < 5:
        return r
    for e in rng.integers(0, hi, size=oc.N_PER_DAY):
        e = int(e)
        d = 1 if rng.random() < 0.5 else -1
        pnl = d * (close[e + 1:] - close[e])
        if pnl.size < 4:
            continue
        real = np.where(np.abs(pnl) >= MIN)[0]
        if len(real) == 0 or pnl[real[0]] > 0:
            continue                                   # only adverse-first ("wrong") trades
        rr = real[0]
        back = np.where(pnl[rr:] >= 0)[0]              # first return to zero
        if len(back) == 0:
            r["death2"] += 1                           # never came back -> adverse runaway
            continue
        z1 = rr + back[0]
        A1 = float(-pnl[:z1 + 1].min())
        seg = pnl[z1:]                                 # the second leg, starting at ~zero
        below = np.where(seg < 0)[0]                   # when it returns below zero again
        if len(below):
            A2 = float(seg[:below[0]].max())           # favorable peak of the second half
            kind = "kicker" if A2 >= MIN else "stall"
        else:
            A2 = float(seg.max())                      # never returned below -> favorable runaway
            kind = "jackpot" if A2 >= MIN else "stall"
        r[kind] += 1
        if A2 >= MIN:
            r["A1"].append(A1); r["A2"].append(A2); r["ratio"].append(A2 / max(A1, 1e-9))
    return r


def hist(vals, width, label, cap=14):
    if not vals:
        return [f"(no {label})"]
    nb = min(int(max(vals) // width) + 1, cap)
    c = [0] * nb
    for v in vals:
        c[min(int(v // width), nb - 1)] += 1
    pk = max(c) or 1
    out = [f"{label} (mode {c.index(max(c))*width}-{(c.index(max(c))+1)*width}pt, median {int(np.median(vals))}pt, n={len(vals)})"]
    for i in range(nb):
        out.append(f"{i*width:>3}-{(i+1)*width:<3}pt|{'#'*int(round(40*c[i]/pk))} {c[i]}")
    return out


def main():
    rng = np.random.default_rng(oc.SEED)
    files = sorted(glob.glob(os.path.join(ONE_M, "2024_*.parquet")) +
                   glob.glob(os.path.join(ONE_M, "2025_*.parquet")))
    A = {"kicker": 0, "jackpot": 0, "stall": 0, "death2": 0, "A1": [], "A2": [], "ratio": []}
    for f in tqdm(files, desc="days", unit="day"):
        try:
            close = pd.read_parquet(f)["close"].to_numpy(np.float64)
        except Exception:
            continue
        r = day(close, rng)
        for k in ("kicker", "jackpot", "stall", "death2"):
            A[k] += r[k]
        for k in ("A1", "A2", "ratio"):
            A[k] += r[k]

    tot = A["kicker"] + A["jackpot"] + A["stall"] + A["death2"]
    came_back = A["kicker"] + A["jackpot"] + A["stall"]
    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w("# The kicker — the second half of the oscillation (adverse-first trades) (2024+2025)")
    w(f"n adverse-first trades = {tot}\n")
    w(f"- came back to zero (oscillators): {came_back}/{tot} = {came_back/tot:.0%}; "
      f"never came back (adverse runaway/DEATH): {A['death2']}/{tot} = {A['death2']/tot:.0%}\n")
    w("Of the ones that came back to zero, what the SECOND leg did:")
    w(f"- **KICKER** (favorable swing >= {MIN}pt then returns): {A['kicker']/came_back:.0%}")
    w(f"- **JACKPOT** (favorable swing that RAN away, never returned): {A['jackpot']/came_back:.0%}")
    w(f"- STALL (no real second leg): {A['stall']/came_back:.0%}")
    if A["A2"]:
        w(f"\n- second-leg favorable reach A2: median {np.median(A['A2']):.0f}pt (${np.median(A['A2'])*2:.0f}), "
          f"mean {np.mean(A['A2']):.0f}pt, 90th {np.percentile(A['A2'],90):.0f}pt")
        w(f"- first adverse leg A1: median {np.median(A['A1']):.0f}pt | A2/A1 median ratio "
          f"**{np.median(A['ratio']):.2f}** (1.0 = symmetric oscillation)")
        w("```")
        w("\n".join(hist(A["A2"], 5, "KICKER amplitude (favorable 2nd-leg reach A2)")))
        w("```")
    w("\n## Read")
    w("If KICKER+JACKPOT dominate and A2/A1 ~ 1, the oscillation is symmetric: crawling back to zero is")
    w("the MIDDLE of the cycle and the favorable half follows -> cutting at breakeven discards the paying")
    w("half; holding (or flipping at the trough) captures it. The DEATH2 fraction is the adverse runaway")
    w("that ruins this -> so the hold-for-the-kicker edge REQUIRES the oscillator-vs-runaway read.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
