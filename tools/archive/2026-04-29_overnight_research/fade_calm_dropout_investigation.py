"""
fade_calm_dropout_investigation.py -- Why did FADE_CALM stop firing after Dec 2025?

FADE_CALM had 365 trades and was the strongest tier early in 2025. It dominated
phases P05-P10. But it disappears from phases P11 onward (Nov 2025+) and
contributes ~0 trades in OOS (Jan-Feb 2026).

This script investigates:
  1. When EXACTLY did FADE_CALM last fire?
  2. What feature values did FADE_CALM trades have? (reverse-engineer entry)
  3. After last firing, did those feature values NEVER occur again?
  4. Or did they occur but with some other condition unmet?

Outputs:
  reports/findings/tier_pnl_by_regime/2026-04-29_13_fade_calm_dropout.md
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

OUT_DIR = "reports/findings/tier_pnl_by_regime"
FEATURES_DIR = "DATA/ATLAS/FEATURES_5s"


def find_latest_enriched():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_trades_enriched.csv")))
    return files[-1] if files else None


def load_features() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def main():
    print("=" * 80)
    print("FADE_CALM DROPOUT INVESTIGATION")
    print("=" * 80)

    enriched = find_latest_enriched()
    twr = pd.read_csv(enriched)
    twr["dt"] = pd.to_datetime(twr["timestamp"], unit="s", utc=True)

    fc = twr[twr["entry_tier"] == "FADE_CALM"].copy()
    fc = fc.sort_values("timestamp").reset_index(drop=True)
    print(f"\nFADE_CALM trades total: {len(fc)}")
    print(f"  First: {fc.iloc[0]['dt']}")
    print(f"  Last:  {fc.iloc[-1]['dt']}")
    print(f"  Median time-between: {fc['timestamp'].diff().median():.0f} sec")

    # Phase 1: when did it stop firing?
    last_dt = fc.iloc[-1]["dt"]
    print(f"\nFADE_CALM LAST trade: {last_dt}")
    days_since_last = (twr["dt"].max() - last_dt).days
    print(f"  Days since last FADE_CALM firing: {days_since_last}")

    # Per-month firing frequency
    fc["ym"] = fc["dt"].dt.to_period("M")
    monthly = fc.groupby("ym").size().reset_index(name="n_trades")
    print(f"\nFADE_CALM firings per month:")
    print(monthly.to_string(index=False))

    # Phase 2: feature distributions for FADE_CALM trades
    feature_cols = [c for c in fc.columns
                    if any(c.startswith(f"{tf}_") for tf in ["1m", "5m", "15m", "1h", "1D"])
                    and c not in ["entry_tier"]]
    print(f"\nAttached feature cols: {len(feature_cols)}")

    # Compare FADE_CALM trades' features vs full trade pool
    print("\n=== Feature distribution: FADE_CALM trades vs all trades ===")
    rows = []
    for c in feature_cols[:20]:  # top 20 to keep output manageable
        fc_vals = fc[c].dropna()
        all_vals = twr[c].dropna()
        if len(fc_vals) < 20 or len(all_vals) < 20:
            continue
        rows.append({
            "feature": c,
            "fc_mean": float(fc_vals.mean()),
            "all_mean": float(all_vals.mean()),
            "fc_std": float(fc_vals.std()),
            "all_std": float(all_vals.std()),
            "fc_p25": float(fc_vals.quantile(0.25)),
            "fc_p75": float(fc_vals.quantile(0.75)),
            "ratio": (fc_vals.mean() / all_vals.mean()) if all_vals.mean() != 0 else np.nan,
        })
    feat_df = pd.DataFrame(rows)
    feat_df["delta_z"] = (feat_df["fc_mean"] - feat_df["all_mean"]) / feat_df["all_std"]
    feat_df = feat_df.sort_values("delta_z", key=lambda x: -x.abs())
    print(feat_df.head(15).to_string(index=False))

    # Phase 3: features in the post-dropout period — do they look different?
    last_fc_ts = fc.iloc[-1]["timestamp"]
    other_post = twr[twr["timestamp"] > last_fc_ts].copy()
    print(f"\nTrades AFTER last FADE_CALM firing: {len(other_post)}")
    if len(other_post) >= 50:
        # For the top discriminating features, compare FADE_CALM vs post-dropout
        print(f"\n=== Top discriminating features, FADE_CALM vs post-dropout period ===")
        top_features = feat_df.head(10)["feature"].tolist()
        cmp_rows = []
        for c in top_features:
            fc_vals = fc[c].dropna()
            post_vals = other_post[c].dropna()
            if len(fc_vals) < 5 or len(post_vals) < 5:
                continue
            cmp_rows.append({
                "feature": c,
                "fc_mean": float(fc_vals.mean()),
                "post_mean": float(post_vals.mean()),
                "fc_std": float(fc_vals.std()),
                "post_std": float(post_vals.std()),
                "shift_in_z_units": (post_vals.mean() - fc_vals.mean()) / fc_vals.std() if fc_vals.std() > 0 else 0,
            })
        cmp_df = pd.DataFrame(cmp_rows)
        cmp_df = cmp_df.sort_values("shift_in_z_units", key=lambda x: -x.abs())
        print(cmp_df.to_string(index=False))

    # Phase 4: feature ranges for FADE_CALM (proxy for entry conditions)
    print(f"\n=== Inferred FADE_CALM entry envelope (5th-95th percentile range) ===")
    envelope_rows = []
    for c in top_features:
        fc_vals = fc[c].dropna()
        if len(fc_vals) < 5:
            continue
        envelope_rows.append({
            "feature": c,
            "fc_p05": float(fc_vals.quantile(0.05)),
            "fc_median": float(fc_vals.median()),
            "fc_p95": float(fc_vals.quantile(0.95)),
            "fc_min": float(fc_vals.min()),
            "fc_max": float(fc_vals.max()),
        })
    env_df = pd.DataFrame(envelope_rows)
    print(env_df.to_string(index=False))

    today = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(OUT_DIR, f"{today}_13_fade_calm_dropout.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# FADE_CALM Dropout Investigation\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Timeline\n\n")
        f.write(f"- FADE_CALM total trades: {len(fc)}\n")
        f.write(f"- First trade: {fc.iloc[0]['dt']}\n")
        f.write(f"- Last trade: {fc.iloc[-1]['dt']}\n")
        f.write(f"- Days since last firing: {days_since_last}\n\n")
        f.write(f"## Per-month firing frequency\n\n")
        f.write("```\n"); f.write(monthly.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Top discriminating features (FADE_CALM trades vs all trades)\n\n")
        f.write(f"Sorted by |delta_z| (mean difference normalized by all-trades stdev):\n\n")
        f.write("```\n"); f.write(feat_df.head(15).to_string(index=False)); f.write("\n```\n\n")
        if len(other_post) >= 50:
            f.write(f"## Feature shift, FADE_CALM period vs post-dropout\n\n")
            f.write("```\n"); f.write(cmp_df.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Inferred FADE_CALM entry envelope\n\n")
        f.write("```\n"); f.write(env_df.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Hypotheses\n\n")
        f.write("- If `shift_in_z_units` is large (>1) for some features, the post-dropout \n")
        f.write("  market conditions DON'T match FADE_CALM's typical entry envelope.\n")
        f.write("- If shift is small (<0.5), the conditions ARE there but the tier still \n")
        f.write("  doesn't fire — meaning some other condition (timing? regime?) blocks it.\n\n")
        f.write(f"## Recommended next step\n\n")
        f.write("Read `nn_v2/nightmare_blended.py` (or wherever FADE_CALM is defined) and \n")
        f.write("compare the EXACT entry conditions to the post-dropout feature values \n")
        f.write("computed here. Probably one or two specific thresholds aren't being met \n")
        f.write("in the new regime.\n")

    print(f"\nWrote: {md_path}")
    gc.collect()


if __name__ == "__main__":
    main()
