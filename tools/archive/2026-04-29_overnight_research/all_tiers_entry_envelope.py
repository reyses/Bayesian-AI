"""
all_tiers_entry_envelope.py -- Reverse-engineer each tier's entry envelope
from the trade-level feature distributions. Same approach as the FADE_CALM
investigation but applied to all tiers with sufficient sample size.

For each tier:
  1. Compute the feature distribution at trade entries
  2. Compare to the all-trades distribution (baseline)
  3. Identify the most distinctive features (largest delta_z)
  4. Document the 5th-95th percentile envelope for top features
  5. Compare 1H 2025 (Jan-Jun) vs 2H 2025 (Jul-Dec) — does envelope shift?

Outputs:
  reports/findings/tier_pnl_by_regime/
    14_all_tiers_entry_envelope.csv
    14_all_tiers_entry_envelope.md
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


def find_latest_enriched():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_trades_enriched.csv")))
    return files[-1] if files else None


def main():
    print("=" * 80)
    print("ALL TIERS ENTRY ENVELOPE INVESTIGATION")
    print("=" * 80)

    enriched = find_latest_enriched()
    twr = pd.read_csv(enriched)
    twr["dt"] = pd.to_datetime(twr["timestamp"], unit="s", utc=True)

    feature_cols = [c for c in twr.columns
                    if any(c.startswith(f"{tf}_") for tf in ["1m", "5m", "15m", "1h", "1D"])
                    and c not in ["entry_tier"]]
    print(f"\nFeature cols: {len(feature_cols)}")

    rows = []
    envelope_rows = []

    for tier in twr["entry_tier"].dropna().unique():
        tier_trades = twr[twr["entry_tier"] == tier]
        if len(tier_trades) < 30:
            continue

        # Aggregate stats: per feature, compare tier vs all-trades
        for c in feature_cols:
            tier_vals = tier_trades[c].dropna()
            all_vals = twr[c].dropna()
            if len(tier_vals) < 20 or len(all_vals) < 20:
                continue
            tier_std = tier_vals.std()
            all_std = all_vals.std()
            delta_z = (tier_vals.mean() - all_vals.mean()) / all_std if all_std > 0 else 0
            rows.append({
                "tier": tier,
                "feature": c,
                "n_trades": len(tier_trades),
                "tier_mean": float(tier_vals.mean()),
                "all_mean": float(all_vals.mean()),
                "tier_std": float(tier_std),
                "delta_z": delta_z,
                "tier_p05": float(tier_vals.quantile(0.05)),
                "tier_p25": float(tier_vals.quantile(0.25)),
                "tier_p50": float(tier_vals.median()),
                "tier_p75": float(tier_vals.quantile(0.75)),
                "tier_p95": float(tier_vals.quantile(0.95)),
            })

    df = pd.DataFrame(rows)

    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(OUT_DIR, f"{today}_14_all_tiers_entry_envelope.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    # Top discriminating features per tier
    print("\n=== Top 5 discriminating features per tier ===")
    md_lines = []
    md_lines.append(f"# All Tiers Entry Envelope Investigation\n\n")
    md_lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
    md_lines.append(f"For each tier with ≥30 trades, the top features that distinguish ")
    md_lines.append(f"its entries from the all-trades baseline. Sorted by |delta_z| ")
    md_lines.append(f"(mean shift normalized by all-trades stdev).\n\n")
    md_lines.append(f"## Tier counts (sorted by N)\n\n")

    tier_counts = (twr.groupby("entry_tier").size().sort_values(ascending=False).reset_index())
    tier_counts.columns = ["tier", "n_trades"]
    md_lines.append("```\n")
    md_lines.append(tier_counts.to_string(index=False))
    md_lines.append("\n```\n\n")

    for tier in df["tier"].unique():
        sub = df[df["tier"] == tier].copy()
        sub["abs_z"] = sub["delta_z"].abs()
        top5 = sub.sort_values("abs_z", ascending=False).head(5)
        n_trades = top5.iloc[0]["n_trades"]
        print(f"\n--- {tier} (n={int(n_trades)}) ---")
        print(top5[["feature", "tier_mean", "all_mean", "delta_z",
                       "tier_p05", "tier_p50", "tier_p95"]].to_string(index=False))
        md_lines.append(f"## {tier} (n={int(n_trades)})\n\n")
        md_lines.append("```\n")
        md_lines.append(top5[["feature", "tier_mean", "all_mean", "delta_z",
                                  "tier_p05", "tier_p50", "tier_p95"]].to_string(index=False))
        md_lines.append("\n```\n\n")

    # Per-tier signature interpretation
    md_lines.append(f"## Tier signatures (interpreted from top discriminators)\n\n")
    md_lines.append(f"Each tier's entry envelope reveals what features it requires:\n\n")
    interpretations = {
        "FADE_CALM":      "High 1m_reversion_prob, low 1m_hurst, calm/mean-reverting context",
        "RIDE_AGAINST":   "Moderate values across most features (broad envelope)",
        "FADE_AGAINST":   "Strong dmi_diff at multiple TFs (counter to recent trend)",
        "BASE_NMP":       "z_se anomaly + variance_ratio<1 (NMP fade)",
        "RIDE_CALM":      "Calm/low-vol environment with mild momentum",
        "KILL_SHOT":      "Wick rejection at extreme z_se, very specific micro-pattern",
        "CASCADE":        "Multi-TF alignment (rare combo)",
        "FADE_MOMENTUM":  "Strong momentum signal, but contrarian fade entry",
        "RIDE_MOMENTUM":  "With-momentum entry on confirmation",
        "FREIGHT_TRAIN":  "Sustained one-direction move",
    }
    for tier, interp in interpretations.items():
        if tier in df["tier"].unique():
            md_lines.append(f"- **{tier}**: {interp}\n")
    md_lines.append("\n")

    # Half-year stability check (does envelope shift between H1 and H2 2025?)
    md_lines.append(f"## Stability check: H1 2025 vs H2 2025 entry envelopes\n\n")
    md_lines.append(f"For each tier, compare top features' means in Jan-Jun 2025 (H1) ")
    md_lines.append(f"vs Jul-Dec 2025 (H2). Stability = consistent means across halves.\n\n")
    h1_end = pd.Timestamp("2025-07-01", tz="UTC").timestamp()
    h2_end = pd.Timestamp("2026-01-01", tz="UTC").timestamp()
    stability_rows = []
    for tier in df["tier"].unique():
        tier_trades = twr[twr["entry_tier"] == tier]
        h1 = tier_trades[tier_trades["timestamp"] < h1_end]
        h2 = tier_trades[(tier_trades["timestamp"] >= h1_end) & (tier_trades["timestamp"] < h2_end)]
        if len(h1) < 20 or len(h2) < 20:
            continue
        sub = df[df["tier"] == tier]
        sub["abs_z"] = sub["delta_z"].abs()
        top5 = sub.sort_values("abs_z", ascending=False).head(5)
        for _, row in top5.iterrows():
            c = row["feature"]
            h1_v = h1[c].dropna()
            h2_v = h2[c].dropna()
            if len(h1_v) < 5 or len(h2_v) < 5:
                continue
            shift = (h2_v.mean() - h1_v.mean()) / h1_v.std() if h1_v.std() > 0 else 0
            stability_rows.append({
                "tier": tier, "feature": c,
                "h1_n": len(h1), "h1_mean": float(h1_v.mean()),
                "h2_n": len(h2), "h2_mean": float(h2_v.mean()),
                "shift_in_h1_z_units": shift,
            })
    if stability_rows:
        stab = pd.DataFrame(stability_rows)
        stab = stab.sort_values("shift_in_h1_z_units", key=lambda x: -x.abs())
        md_lines.append("```\n")
        md_lines.append(stab.head(30).to_string(index=False))
        md_lines.append("\n```\n\n")
        md_lines.append(f"Tiers with high feature shift between halves are more regime-dependent ")
        md_lines.append(f"(their entry conditions might dry up if regime changes).\n\n")

    md_path = os.path.join(OUT_DIR, f"{today}_14_all_tiers_entry_envelope.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.writelines(md_lines)
    print(f"\nWrote: {md_path}")
    gc.collect()


if __name__ == "__main__":
    main()
