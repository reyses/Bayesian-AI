"""
tier_slope_filter_within_split.py -- Proper holdout validation for LinReg
slope filter using time-based 70/30 within-tier split.

PROBLEM with naive IS/OOS validation: the existing IS file (blended_is.csv) has
all tiers EXCEPT BASE_NMP. The OOS file (blended_oos_trades.csv) is mostly
BASE_NMP-only. So you can't validate cross-period for most tiers.

SOLUTION: split each tier's trades by date — first 70% as train, last 30% as
test. Pick threshold on train, apply to test. This validates whether the
filter's edge is real or a fit to specific bars.

Outputs:
  reports/findings/tier_pnl_by_regime/2026-04-29_12_slope_filter_train_test.csv
  reports/findings/tier_pnl_by_regime/2026-04-29_12_slope_filter_train_test.md
"""
from __future__ import annotations
import argparse
import gc
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

from tools.tier_linreg_slope_filter import (
    rolling_linreg_slope, compute_slope_at_entries, evaluate_filter, load_1m_closes,
)

OUT_DIR = "reports/findings/tier_pnl_by_regime"


def find_latest_enriched():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_trades_enriched.csv")))
    return files[-1] if files else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--linreg-period", type=int, default=30)
    ap.add_argument("--thresholds", default="0.5,1.0,1.5,2.0,3.0,5.0")
    ap.add_argument("--train-pct", type=float, default=0.7)
    args = ap.parse_args()

    print("=" * 80)
    print(f"LINREG FILTER WITHIN-TIER HOLDOUT ({int(args.train_pct*100)}/{int((1-args.train_pct)*100)} time-based split)")
    print("=" * 80)

    enriched = find_latest_enriched()
    twr = pd.read_csv(enriched)
    twr["dt"] = pd.to_datetime(twr["timestamp"], unit="s", utc=True)
    print(f"Loaded {len(twr)} trades")

    bars = load_1m_closes()
    twr = compute_slope_at_entries(twr, bars, period=args.linreg_period)
    slope_col = f"linreg_slope_{args.linreg_period}"

    thresholds = [float(x) for x in args.thresholds.split(",")]
    rows = []

    for tier in twr["entry_tier"].dropna().unique():
        sub = twr[(twr["entry_tier"] == tier)].dropna(subset=[slope_col]).copy()
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        if len(sub) < 50:
            continue

        split_idx = int(len(sub) * args.train_pct)
        train = sub.iloc[:split_idx]
        test = sub.iloc[split_idx:]

        if len(test) < 10:
            continue

        train_baseline_pnl = train["pnl"].sum()
        train_baseline_n = len(train)
        test_baseline_pnl = test["pnl"].sum()
        test_baseline_n = len(test)

        # Pick best T on train
        best = None
        for T in thresholds:
            r = evaluate_filter(train, slope_col, T)
            if best is None or r["pnl_kept"] > best["pnl_kept"]:
                best = r
                best["T_picked"] = T

        T_picked = best["T_picked"]
        # Apply to test
        test_r = evaluate_filter(test, slope_col, T_picked)

        rows.append({
            "tier": tier,
            "n_total": len(sub),
            "train_n": train_baseline_n,
            "train_baseline_pnl": train_baseline_pnl,
            "T_picked": T_picked,
            "train_kept_n": best["n_kept"],
            "train_filtered_pnl": best["pnl_kept"],
            "train_improvement_total": best["pnl_kept"] - train_baseline_pnl,
            "train_improvement_per_kept_trade": (best["pnl_kept"] / best["n_kept"] - train_baseline_pnl / train_baseline_n)
                if best["n_kept"] > 0 else 0,
            "test_n": test_baseline_n,
            "test_baseline_pnl": test_baseline_pnl,
            "test_kept_n": test_r["n_kept"],
            "test_filtered_pnl": test_r["pnl_kept"],
            "test_improvement_total": test_r["pnl_kept"] - test_baseline_pnl,
            "test_improvement_per_kept_trade": (test_r["pnl_kept"] / test_r["n_kept"] - test_baseline_pnl / test_baseline_n)
                if test_r["n_kept"] > 0 else 0,
            "generalizes_total": test_r["pnl_kept"] > test_baseline_pnl,
            "generalizes_per_trade": (
                (test_r["pnl_kept"] / test_r["n_kept"]) > (test_baseline_pnl / test_baseline_n)
            ) if test_r["n_kept"] > 0 else False,
        })

    df = pd.DataFrame(rows).sort_values("train_improvement_total", ascending=False)

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(OUT_DIR, f"{today}_12_slope_filter_train_test.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== Within-tier 70/30 holdout for LinReg slope filter ===")
    print(df[["tier", "T_picked",
                "train_n", "train_baseline_pnl", "train_filtered_pnl", "train_improvement_total",
                "test_n", "test_baseline_pnl", "test_filtered_pnl", "test_improvement_total",
                "generalizes_total", "generalizes_per_trade"]].to_string(index=False))

    md_path = os.path.join(OUT_DIR, f"{today}_12_slope_filter_train_test.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# LinReg slope filter — within-tier holdout validation\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Method\n\n")
        f.write(f"For each tier:\n")
        f.write(f"1. Sort trades by timestamp.\n")
        f.write(f"2. Split into train (first {int(args.train_pct*100)}%) and test (last {int((1-args.train_pct)*100)}%).\n")
        f.write(f"3. Pick best `slope_skip_threshold` on train.\n")
        f.write(f"4. Apply that threshold to test (no peeking).\n")
        f.write(f"5. Compare PnL improvement on train vs test.\n\n")
        f.write(f"## Robustness verdict\n\n")
        n_gen_total = df["generalizes_total"].sum()
        n_gen_pt = df["generalizes_per_trade"].sum()
        f.write(f"- {n_gen_total}/{len(df)} tiers improve TOTAL PnL on test (vs baseline).\n")
        f.write(f"- {n_gen_pt}/{len(df)} tiers improve PER-TRADE PnL on test (more honest).\n\n")
        if n_gen_pt >= len(df) * 0.7:
            f.write(f"- **Filter generalizes for most tiers — safe to ship.**\n\n")
        else:
            f.write(f"- **Mixed result — only ship for tiers showing per-trade improvement on test.**\n\n")
        f.write(f"## Detailed table\n\n")
        f.write("```\n"); f.write(df.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Caveat\n\n")
        f.write(f"This is a within-tier time-split, not a true IS/OOS. The original 'IS' \n")
        f.write(f"and 'OOS' files come from different pipeline runs with different tier mixes,\n")
        f.write(f"so naive IS-train-OOS-test isn't possible.\n\n")
        f.write(f"This 70/30 split tests whether the threshold picked on early data still \n")
        f.write(f"works on later data — a basic sanity check for time-stability.\n")
    print(f"\nWrote: {csv_path}")
    print(f"Wrote: {md_path}")
    gc.collect()


if __name__ == "__main__":
    main()
