"""
tier_slope_filter_is_oos.py -- Critical robustness check: does the LinReg slope
filter improvement HOLD on OOS, or is it just IS overfit?

For each tier:
  1. Find best slope_skip_threshold using IS only
  2. Apply that threshold to OOS
  3. Compare improvement: IS gain vs OOS gain

If filter generalizes: OOS improvement should be similar magnitude (per trade)
to IS. If not, the filter is overfit.

Outputs:
  reports/findings/tier_pnl_by_regime/2026-04-29_11_slope_filter_is_oos.csv
  reports/findings/tier_pnl_by_regime/2026-04-29_11_slope_filter_is_oos.md
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
    args = ap.parse_args()

    print("=" * 80)
    print("LINREG FILTER ROBUSTNESS — IS pick threshold, apply to OOS")
    print("=" * 80)

    enriched = find_latest_enriched()
    print(f"Loading: {enriched}")
    twr = pd.read_csv(enriched)
    twr["dt"] = pd.to_datetime(twr["timestamp"], unit="s", utc=True)
    print(f"  {len(twr)} trades  (IS={(twr['split']=='IS').sum()}, OOS={(twr['split']=='OOS').sum()})")

    print(f"\nLoading 1m bars + computing LinReg slope...")
    bars = load_1m_closes()
    twr = compute_slope_at_entries(twr, bars, period=args.linreg_period)
    slope_col = f"linreg_slope_{args.linreg_period}"
    matched = (~twr[slope_col].isna()).sum()
    print(f"  Slope matched: {matched}/{len(twr)}")

    thresholds = [float(x) for x in args.thresholds.split(",")]

    rows = []
    print(f"\nProcessing per tier — pick best threshold on IS, apply to OOS...")
    for tier in twr["entry_tier"].dropna().unique():
        is_sub = twr[(twr["entry_tier"] == tier) & (twr["split"] == "IS")].dropna(subset=[slope_col])
        oos_sub = twr[(twr["entry_tier"] == tier) & (twr["split"] == "OOS")].dropna(subset=[slope_col])
        if len(is_sub) < 30 or len(oos_sub) < 10:
            continue

        is_baseline_pnl = is_sub["pnl"].sum()
        oos_baseline_pnl = oos_sub["pnl"].sum()
        is_baseline_n = len(is_sub)
        oos_baseline_n = len(oos_sub)

        # Sweep thresholds on IS
        best = None
        for T in thresholds:
            r = evaluate_filter(is_sub, slope_col, T)
            if best is None or r["pnl_kept"] > best["pnl_kept"]:
                best = r
                best["threshold"] = T

        T_picked = best["threshold"]
        is_filtered_pnl = best["pnl_kept"]
        is_kept_n = best["n_kept"]

        # Apply T_picked to OOS
        oos_r = evaluate_filter(oos_sub, slope_col, T_picked)
        oos_filtered_pnl = oos_r["pnl_kept"]
        oos_kept_n = oos_r["n_kept"]

        rows.append({
            "tier": tier,
            "is_baseline_n": is_baseline_n,
            "is_baseline_pnl": is_baseline_pnl,
            "is_baseline_per_trade": is_baseline_pnl / is_baseline_n,
            "T_picked_on_is": T_picked,
            "is_kept_n": is_kept_n,
            "is_filtered_pnl": is_filtered_pnl,
            "is_filtered_per_trade": is_filtered_pnl / is_kept_n if is_kept_n > 0 else 0,
            "is_improvement": is_filtered_pnl - is_baseline_pnl,
            "is_improvement_per_kept_trade": (is_filtered_pnl / is_kept_n - is_baseline_pnl / is_baseline_n) if is_kept_n > 0 else 0,
            "oos_baseline_n": oos_baseline_n,
            "oos_baseline_pnl": oos_baseline_pnl,
            "oos_baseline_per_trade": oos_baseline_pnl / oos_baseline_n,
            "oos_kept_n": oos_kept_n,
            "oos_filtered_pnl": oos_filtered_pnl,
            "oos_filtered_per_trade": oos_filtered_pnl / oos_kept_n if oos_kept_n > 0 else 0,
            "oos_improvement": oos_filtered_pnl - oos_baseline_pnl,
            "oos_improvement_per_kept_trade": (oos_filtered_pnl / oos_kept_n - oos_baseline_pnl / oos_baseline_n) if oos_kept_n > 0 else 0,
            "generalizes": (oos_filtered_pnl > oos_baseline_pnl) and (oos_kept_n > 5),
        })

    df = pd.DataFrame(rows).sort_values("is_improvement", ascending=False)

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(OUT_DIR, f"{today}_11_slope_filter_is_oos.csv")
    df.to_csv(csv_path, index=False)

    print("\n=== IS-trained, OOS-applied LinReg slope filter ===")
    print(df[["tier", "T_picked_on_is", "is_baseline_pnl", "is_filtered_pnl",
                "is_improvement", "oos_baseline_pnl", "oos_filtered_pnl",
                "oos_improvement", "generalizes"]].to_string(index=False))

    md_path = os.path.join(OUT_DIR, f"{today}_11_slope_filter_is_oos.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# LinReg slope filter — IS/OOS robustness check\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Method\n\n")
        f.write(f"For each tier:\n")
        f.write(f"1. Pick best slope_skip_threshold on IS-only data\n")
        f.write(f"2. Apply that EXACT threshold to OOS data\n")
        f.write(f"3. Compare per-trade improvement IS vs OOS\n\n")
        f.write(f"## Robustness verdict\n\n")
        n_generalize = df["generalizes"].sum()
        f.write(f"- {n_generalize}/{len(df)} tiers' filters GENERALIZE to OOS (filtered PnL > baseline)\n")
        if n_generalize == len(df):
            f.write(f"- **All filters generalize.** Ship safely.\n\n")
        elif n_generalize >= len(df) * 0.7:
            f.write(f"- **Most filters generalize.** Ship the ones that do; investigate non-generalizing tiers.\n\n")
        else:
            f.write(f"- **Filter is overfit on IS for some tiers.** Be cautious — only ship tiers where OOS improvement is positive.\n\n")
        f.write(f"## Detailed table\n\n")
        f.write("```\n"); f.write(df.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Interpretation\n\n")
        f.write("- `is_improvement_per_kept_trade` — per-trade gain from filter, on IS\n")
        f.write("- `oos_improvement_per_kept_trade` — per-trade gain from same threshold, on OOS\n")
        f.write("- If OOS per-trade gain is similar magnitude to IS gain → filter is real\n")
        f.write("- If OOS per-trade gain is zero or negative → filter is IS-overfit\n")
        f.write("- `generalizes` flag = filter improves OOS PnL (modest test)\n")
    print(f"\nWrote: {md_path}")
    print(f"Wrote: {csv_path}")
    gc.collect()


if __name__ == "__main__":
    main()
