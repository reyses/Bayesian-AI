"""Refine the AI filler by tuning it to the 398 human picks.

The v2 labeler recovers only ~35% of human picks (it keeps the big >=7pt cubic legs; the human marks
more/smaller swings). This finds the (CUBIC_N, TREND_PTS) that best REPRODUCES the human picks:
  - for each human-pick day, run the cubic + zigzag pivots at (N, T),
  - match pivots to human picks (±TOL, same direction),
  - score recall (human found), precision (pivots that are real), F1, across all days.
Also reports the human-pick swing-size distribution (the threshold the human effectively uses).
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "tools", "viz", "core"))
sys.path.insert(0, os.path.join(ROOT, "research", "ai_auto_labeler", "pipeline"))
from cubic_utils import find_raw_turns          # noqa: E402
from ai_labeler_v2 import zigzag_turns          # noqa: E402

ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
PICKS = os.path.join(ROOT, "DATA", "cusp_picks")
REPORT = os.path.join(ROOT, "research", "ai_auto_labeler", "reports", "tune_to_human.md")
TOL = 300   # ±5 min match window


def human_days():
    out = {}
    for f in sorted(glob.glob(os.path.join(PICKS, "picks_*_multi.json"))):
        day = os.path.basename(f).replace("picks_", "").replace("_multi.json", "")
        p = [(x["timestamp"], x.get("direction", "?"))
             for x in json.load(open(f)).get("picks", []) if x.get("timeframe", "1m") == "1m"]
        dk = day.replace("-", "_")
        if p and os.path.exists(os.path.join(ONE_M, f"{dk}.parquet")):
            out[dk] = p
    return out


def pivots_at(close, ts1m, N, T):
    turns, smooth, _, _ = find_raw_turns(close, N)
    return [(float(ts1m[i]), "LONG" if ty == "bottom" else "SHORT") for i, ty, _ in zigzag_turns(smooth, turns, T)]


def score(human, piv):
    if not human:
        return 0, 0, 0
    hts = np.array([t for t, _ in human]); hd = [d for _, d in human]
    pts = np.array([t for t, _ in piv]); pd_ = [d for _, d in piv]
    rec = sum(any(abs(t - pt) <= TOL and hd[i] == pd_[j] for j, pt in enumerate(pts)) for i, t in enumerate(hts)) if len(pts) else 0
    prec = sum(any(abs(t - ht) <= TOL and pd_[j] == hd[i] for i, ht in enumerate(hts)) for j, t in enumerate(pts)) if len(hts) else 0
    return rec, prec, len(piv)


def main():
    days = human_days()
    cache = {dk: pd.read_parquet(os.path.join(ONE_M, f"{dk}.parquet")) for dk in days}
    nH = sum(len(v) for v in days.values())

    L = []
    def w(s):
        print(s.encode("ascii", "replace").decode()); L.append(s)
    w(f"# Tuning the AI filler to {nH} human picks across {len(days)} days\n")

    # human swing-size distribution: |price move| between consecutive human picks (per day)
    swings = []
    for dk, hp in days.items():
        cl = cache[dk]["close"].to_numpy(np.float64); ts = cache[dk]["timestamp"].to_numpy(np.float64)
        idx = sorted(int(np.searchsorted(ts, t)) for t, _ in hp)
        for a, b in zip(idx, idx[1:]):
            if 0 <= a < len(cl) and 0 <= b < len(cl):
                swings.append(abs(cl[min(b, len(cl)-1)] - cl[min(a, len(cl)-1)]))
    sw = np.array(swings)
    w(f"## Human swing size (|Δprice| between consecutive picks): median {np.median(sw):.1f}pt, "
      f"25th {np.percentile(sw,25):.1f}, 75th {np.percentile(sw,75):.1f}")
    w(f"- share of human swings < 7pt (below current TREND_PTS): {(sw<7).mean():.0%} "
      f"-> this is what the 7pt threshold misses.\n")

    # parameter sweep
    w("## Parameter sweep — recall (human found) / precision (pivots real) / F1")
    w("```")
    w(f"{'N':>3} {'T':>4} | {'#piv':>5} | recall | prec | F1")
    best = (0, None)
    for N in (10, 15, 20, 30):
        for T in (3, 4, 5, 6, 7, 8):
            R = P = NP = 0
            for dk, hp in days.items():
                cl = cache[dk]["close"].to_numpy(np.float64); ts = cache[dk]["timestamp"].to_numpy(np.float64)
                piv = pivots_at(cl, ts, N, T)
                r, p, npv = score(hp, piv)
                R += r; P += p; NP += npv
            rec = R / nH; prec = P / max(NP, 1); f1 = 2*rec*prec/max(rec+prec, 1e-9)
            mark = ""
            if f1 > best[0]:
                best = (f1, (N, T)); mark = " *"
            w(f"{N:>3} {T:>4} | {NP:>5} | {rec:>5.0%} | {prec:>4.0%} | {f1:.3f}{mark}")
    w("```")
    w(f"\n## Best match: CUBIC_N={best[1][0]}, TREND_PTS={best[1][1]}  (F1={best[0]:.3f})")
    w(f"- current default is N=20, TREND_PTS=7. Recommend moving toward the best-F1 cell.")
    w("- recall<100% is expected: the human marks some sub-cubic turns no pivot scale will catch;")
    w("  raising recall trades precision (more spurious pivots). Pick the knee, not max recall.")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
