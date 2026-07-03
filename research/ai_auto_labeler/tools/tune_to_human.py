"""Refine the AI filler by tuning it to the 398 human picks.

The v2 labeler recovers only ~35% of human picks (it keeps the big >=7pt cubic legs; the human marks
more/smaller swings). This finds the (CUBIC_N, TREND_PTS) that best REPRODUCES the human picks:
  - for each human-pick day, run the cubic + zigzag pivots at (N, T),
  - match pivots to human picks (±TOL, same direction),
  - score recall (human found), precision (pivots that are real), F1, across all days.
Also reports the human-pick swing-size distribution (the threshold the human effectively uses).
Now includes an Adaptive regime-scaling sweep and LODO validation.
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
from amplitude_scale import scale_for_day       # noqa: E402

ONE_M = os.path.join(ROOT, "DATA", "ATLAS", "1m")
PICKS = os.path.join(ROOT, "DATA", "cusp_picks")
REPORT = os.path.join(ROOT, "research", "ai_auto_labeler", "reports", "tune_to_human.md")
TOL = 300   # ±5 min match window

TREND_MIN_PTS = 2.5
TREND_MAX_PTS = 12.0
N_BOOT = 4000
BOOT_SEED = 20260702

K_GRID = (0.15, 0.21, 0.30, 0.42, 0.60, 0.85, 1.20, 1.70)
MODE_GRID = ("w30", "w60", "w120", "day", "day_c5", "day_c21")


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


def pivots_at(close, ts1m, N, thr_arr):
    turns, smooth, _, _ = find_raw_turns(close, N)
    return [(float(ts1m[i]), "LONG" if ty == "bottom" else "SHORT") for i, ty, _ in zigzag_turns(smooth, turns, thr_arr)]


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

    w("## Fixed Threshold Sweep (Baseline)")
    w("```")
    w(f"{'N':>3} {'T':>4} | {'#piv':>5} | recall | prec | F1")
    
    fixed_cache = {}
    best_fixed_f1 = -1
    best_fixed = None
    
    for N in (10, 15, 20, 30):
        for T in (3, 4, 5, 6, 7, 8):
            R = P = NP = 0
            day_stats = {}
            for dk, hp in days.items():
                cl = cache[dk]["close"].to_numpy(np.float64); ts = cache[dk]["timestamp"].to_numpy(np.float64)
                thr_arr = np.full(len(cl), T)
                piv = pivots_at(cl, ts, N, thr_arr)
                r, p, npv = score(hp, piv)
                day_stats[dk] = (r, p, npv, len(hp))
                R += r; P += p; NP += npv
            
            fixed_cache[(N, T)] = day_stats
            rec = R / nH; prec = P / max(NP, 1); f1 = 2*rec*prec/max(rec+prec, 1e-9)
            mark = ""
            if f1 > best_fixed_f1:
                best_fixed_f1 = f1
                best_fixed = (N, T)
                mark = " *"
            w(f"{N:>3} {T:>4} | {NP:>5} | {rec:>5.0%} | {prec:>4.0%} | {f1:.3f}{mark}")
    w("```")
    w(f"\nBest fixed match: CUBIC_N={best_fixed[0]}, TREND_PTS={best_fixed[1]}  (F1={best_fixed_f1:.3f})\n")

    w("## Adaptive Threshold Sweep (N=20)")
    w("```")
    w(f"{'K':>4} {'MODE':>8} | {'#piv':>5} | recall | prec | F1    | % clamp")
    
    adaptive_cache = {}
    best_adapt_f1 = -1
    best_adapt = None
    best_adapt_stats = None
    
    N = 20
    scale_cache = {}
    
    for K in K_GRID:
        for MODE in MODE_GRID:
            R = P = NP = 0
            day_stats = {}
            clamped_total = 0
            len_total = 0
            
            for dk, hp in days.items():
                cl = cache[dk]["close"].to_numpy(np.float64); ts = cache[dk]["timestamp"].to_numpy(np.float64)
                
                scale_arr = scale_for_day(dk, cl, MODE, ONE_M, scale_cache)
                raw_thr = K * scale_arr
                thr_arr = np.clip(raw_thr, TREND_MIN_PTS, TREND_MAX_PTS)
                
                # compute % clamped
                clamped = np.sum((raw_thr <= TREND_MIN_PTS) | (raw_thr >= TREND_MAX_PTS))
                clamped_total += clamped
                len_total += len(cl)
                
                piv = pivots_at(cl, ts, N, thr_arr)
                r, p, npv = score(hp, piv)
                day_stats[dk] = (r, p, npv, len(hp))
                R += r; P += p; NP += npv
            
            adaptive_cache[(K, MODE)] = day_stats
            rec = R / nH; prec = P / max(NP, 1); f1 = 2*rec*prec/max(rec+prec, 1e-9)
            clamp_pct = clamped_total / max(len_total, 1)
            
            mark = ""
            if f1 > best_adapt_f1:
                best_adapt_f1 = f1
                best_adapt = (K, MODE)
                best_adapt_stats = {"clamp": clamp_pct}
                mark = " *"
            w(f"{K:>4.2f} {MODE:>8} | {NP:>5} | {rec:>5.0%} | {prec:>4.0%} | {f1:.3f} | {clamp_pct:>6.1%}{mark}")
    w("```")
    w(f"\nBest adaptive match: K={best_adapt[0]}, MODE={best_adapt[1]} (F1={best_adapt_f1:.3f}, clamp={best_adapt_stats['clamp']:.1%})\n")
    if best_adapt_stats['clamp'] > 0.50:
        w("⚠️ **WARNING: Champion is clamp-saturated (>50%). This is effectively fixed-T in disguise.**\n")

    w("## LODO (Leave-One-Day-Out) Validation")
    # Fixed LODO
    fixed_lodo_r = fixed_lodo_p = fixed_lodo_np = fixed_lodo_nh = 0
    adapt_lodo_r = adapt_lodo_p = adapt_lodo_np = adapt_lodo_nh = 0
    
    for holdout_dk in days.keys():
        # Find best fixed on OTHER days
        bf_f1, bf_key = -1, None
        for k, stats in fixed_cache.items():
            R = sum(s[0] for dk, s in stats.items() if dk != holdout_dk)
            P = sum(s[1] for dk, s in stats.items() if dk != holdout_dk)
            NP = sum(s[2] for dk, s in stats.items() if dk != holdout_dk)
            NH = sum(s[3] for dk, s in stats.items() if dk != holdout_dk)
            f1 = 2*(R/NH)*(P/max(NP,1))/max((R/NH)+(P/max(NP,1)), 1e-9) if NH > 0 else 0
            if f1 > bf_f1:
                bf_f1 = f1; bf_key = k
        s_f = fixed_cache[bf_key][holdout_dk]
        fixed_lodo_r += s_f[0]; fixed_lodo_p += s_f[1]; fixed_lodo_np += s_f[2]; fixed_lodo_nh += s_f[3]
        
        # Find best adaptive on OTHER days
        ba_f1, ba_key = -1, None
        for k, stats in adaptive_cache.items():
            R = sum(s[0] for dk, s in stats.items() if dk != holdout_dk)
            P = sum(s[1] for dk, s in stats.items() if dk != holdout_dk)
            NP = sum(s[2] for dk, s in stats.items() if dk != holdout_dk)
            NH = sum(s[3] for dk, s in stats.items() if dk != holdout_dk)
            f1 = 2*(R/NH)*(P/max(NP,1))/max((R/NH)+(P/max(NP,1)), 1e-9) if NH > 0 else 0
            if f1 > ba_f1:
                ba_f1 = f1; ba_key = k
        s_a = adaptive_cache[ba_key][holdout_dk]
        adapt_lodo_r += s_a[0]; adapt_lodo_p += s_a[1]; adapt_lodo_np += s_a[2]; adapt_lodo_nh += s_a[3]
        
    f_lodo_rec = fixed_lodo_r / max(fixed_lodo_nh, 1)
    f_lodo_prec = fixed_lodo_p / max(fixed_lodo_np, 1)
    f_lodo_f1 = 2*f_lodo_rec*f_lodo_prec/max(f_lodo_rec+f_lodo_prec, 1e-9)
    
    a_lodo_rec = adapt_lodo_r / max(adapt_lodo_nh, 1)
    a_lodo_prec = adapt_lodo_p / max(adapt_lodo_np, 1)
    a_lodo_f1 = 2*a_lodo_rec*a_lodo_prec/max(a_lodo_rec+a_lodo_prec, 1e-9)
    
    w(f"- **Fixed LODO F1**: {f_lodo_f1:.3f}")
    w(f"- **Adaptive LODO F1**: {a_lodo_f1:.3f}")
    w(f"- **LODO Delta**: {a_lodo_f1 - f_lodo_f1:+.3f}\n")
    
    w("## Bootstrap Confidence Intervals")
    rng = np.random.default_rng(BOOT_SEED)
    day_keys = list(days.keys())
    
    def get_f1(stats_dict, dks):
        R = sum(stats_dict[dk][0] for dk in dks)
        P = sum(stats_dict[dk][1] for dk in dks)
        NP = sum(stats_dict[dk][2] for dk in dks)
        NH = sum(stats_dict[dk][3] for dk in dks)
        rec = R / max(NH, 1)
        prec = P / max(NP, 1)
        return 2*rec*prec/max(rec+prec, 1e-9)
        
    insamp_deltas = []
    lodo_deltas = []
    
    # Pre-calculate best models
    best_fixed_stats = fixed_cache[best_fixed]
    best_adapt_stats_cache = adaptive_cache[best_adapt]
    
    # Generate held-out results for LODO bootstrap
    # Each day has a pre-determined best model from the OTHER 8 days
    lodo_fixed_day_stats = {}
    lodo_adapt_day_stats = {}
    
    for holdout_dk in day_keys:
        bf_f1, bf_key = -1, None
        for k, stats in fixed_cache.items():
            R = sum(s[0] for dk, s in stats.items() if dk != holdout_dk)
            P = sum(s[1] for dk, s in stats.items() if dk != holdout_dk)
            NP = sum(s[2] for dk, s in stats.items() if dk != holdout_dk)
            NH = sum(s[3] for dk, s in stats.items() if dk != holdout_dk)
            f1 = 2*(R/NH)*(P/max(NP,1))/max((R/NH)+(P/max(NP,1)), 1e-9) if NH > 0 else 0
            if f1 > bf_f1: bf_f1 = f1; bf_key = k
        lodo_fixed_day_stats[holdout_dk] = fixed_cache[bf_key][holdout_dk]
        
        ba_f1, ba_key = -1, None
        for k, stats in adaptive_cache.items():
            R = sum(s[0] for dk, s in stats.items() if dk != holdout_dk)
            P = sum(s[1] for dk, s in stats.items() if dk != holdout_dk)
            NP = sum(s[2] for dk, s in stats.items() if dk != holdout_dk)
            NH = sum(s[3] for dk, s in stats.items() if dk != holdout_dk)
            f1 = 2*(R/NH)*(P/max(NP,1))/max((R/NH)+(P/max(NP,1)), 1e-9) if NH > 0 else 0
            if f1 > ba_f1: ba_f1 = f1; ba_key = k
        lodo_adapt_day_stats[holdout_dk] = adaptive_cache[ba_key][holdout_dk]
    
    for _ in range(N_BOOT):
        boot_days = rng.choice(day_keys, size=len(day_keys), replace=True)
        
        # In-sample delta
        f_in = get_f1(best_fixed_stats, boot_days)
        a_in = get_f1(best_adapt_stats_cache, boot_days)
        insamp_deltas.append(a_in - f_in)
        
        # LODO delta
        f_lodo = get_f1(lodo_fixed_day_stats, boot_days)
        a_lodo = get_f1(lodo_adapt_day_stats, boot_days)
        lodo_deltas.append(a_lodo - f_lodo)
        
    ci_in = np.percentile(insamp_deltas, [2.5, 97.5])
    ci_lodo = np.percentile(lodo_deltas, [2.5, 97.5])
    
    sig_in = ci_in[0] > 0
    sig_lodo = ci_lodo[0] > 0
    
    w(f"- **In-sample Î”F1**: {best_adapt_f1 - best_fixed_f1:+.3f} (95% CI: [{ci_in[0]:+.3f}, {ci_in[1]:+.3f}])")
    w(f"  - Significant? **{'Yes' if sig_in else 'No'}**")
    w(f"- **LODO Î”F1**: {a_lodo_f1 - f_lodo_f1:+.3f} (95% CI: [{ci_lodo[0]:+.3f}, {ci_lodo[1]:+.3f}])")
    w(f"  - Significant? **{'Yes' if sig_lodo else 'No'}**")
    
    w("\n> **Significance Note**: With N=9 days, the CI is expected to be wide. \"Not significant\" is the likely honest label even for a real win. A claim rests on directional victory + LODO consistency + Step-A mechanism evidence.")

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    open(REPORT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {REPORT}")


if __name__ == "__main__":
    main()
