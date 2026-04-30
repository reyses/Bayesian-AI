"""
tier_segments_validation.py -- Validate the profitable tier-x-regime subsets
hold up across IS and OOS. Plus per-1W-segment timeline analysis.

Builds on tier_pnl_by_regime.py output:
  - For top profitable subsets: split into IS-only and OOS-only stats
  - Per 1W macro phase: tier behavior breakdown (which tiers work in which phase?)
  - Visualization: tier PnL over time, color by 1D regime direction

Outputs:
  reports/findings/tier_pnl_by_regime/
    06_is_oos_validation.csv   (IS vs OOS for top combos)
    07_tier_x_1w_phase.csv     (tier x 1W macro segment)
    08_tier_timeline.png       (cumulative tier PnL over time, segments shaded)
    09_phase_winners.md        (which tier wins which phase)

Usage:
    python tools/tier_segments_validation.py
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUT_DIR = "reports/findings/tier_pnl_by_regime"

ENRICHED_TRADES_PATH = None  # Auto-find latest

SEG_1D_PATH = "reports/findings/macro_segments/1D_segments.csv"
SEG_1W_PATH = "reports/findings/macro_segments/1W_segments.csv"


def find_latest_enriched():
    import glob
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_trades_enriched.csv")))
    return files[-1] if files else None


def attach_1w_phase(trades: pd.DataFrame, segs_1w: pd.DataFrame) -> pd.DataFrame:
    """Attach 1W segment direction + idx to each trade."""
    out = trades.copy()
    segs_1w["start_dt"] = pd.to_datetime(segs_1w["start_dt"], utc=True)
    segs_1w["end_dt"] = pd.to_datetime(segs_1w["end_dt"], utc=True)
    intervals = pd.IntervalIndex.from_arrays(segs_1w["start_dt"], segs_1w["end_dt"], closed="left")
    seg_idx = []
    for _, t in out.iterrows():
        try:
            idx = intervals.get_indexer([pd.Timestamp(t["dt"], tz="UTC")
                                            if not pd.api.types.is_datetime64_any_dtype(out["dt"])
                                            else t["dt"]])[0]
        except Exception:
            idx = -1
        seg_idx.append(idx)
    out["seg1w_idx"] = seg_idx

    # Build phase label = idx + direction + start_date
    phase_labels = []
    for i in range(len(segs_1w)):
        s = segs_1w.iloc[i]
        d = s["start_dt"].strftime("%m/%d") if hasattr(s["start_dt"], "strftime") else str(s["start_dt"])[:10]
        phase_labels.append(f"P{i:02d}_{s['direction']}_{d}")
    phase_map = {i: phase_labels[i] for i in range(len(segs_1w))}
    out["seg1w_label"] = out["seg1w_idx"].map(phase_map).fillna("UNMATCHED")
    out["seg1w_direction"] = out["seg1w_idx"].map(
        {i: segs_1w.iloc[i]["direction"] for i in range(len(segs_1w))}
    )
    return out


def crosstab(df, cols, pnl_col="pnl"):
    g = df.groupby(cols, dropna=False)
    return g.agg(
        n=(pnl_col, "count"),
        wr=(pnl_col, lambda x: 100 * (x > 0).mean()),
        total=(pnl_col, "sum"),
        mean=(pnl_col, "mean"),
        median=(pnl_col, "median"),
    ).reset_index().sort_values("total", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched-csv", default=None,
                    help="Override enriched trade CSV path")
    args = ap.parse_args()

    print("=" * 80)
    print("TIER × REGIME — IS/OOS validation + 1W macro phases")
    print("=" * 80)

    enriched_path = args.enriched_csv or find_latest_enriched()
    if not enriched_path or not os.path.exists(enriched_path):
        print(f"No enriched trades file. Run tier_pnl_by_regime.py first.")
        sys.exit(1)
    print(f"Loading enriched trades: {enriched_path}")
    twr = pd.read_csv(enriched_path)
    twr["dt"] = pd.to_datetime(twr["timestamp"], unit="s", utc=True)
    print(f"  {len(twr)} trades, {twr['split'].value_counts().to_dict()}")

    # IS/OOS validation for top combos
    print("\n=== IS/OOS validation: tier × seg_direction ===")
    is_ct = crosstab(twr[twr["split"] == "IS"][twr["seg_matched"] == True],
                     ["entry_tier", "seg_direction"])
    oos_ct = crosstab(twr[twr["split"] == "OOS"][twr["seg_matched"] == True],
                      ["entry_tier", "seg_direction"])
    is_ct["split"] = "IS"
    oos_ct["split"] = "OOS"
    combined = pd.concat([is_ct, oos_ct], ignore_index=True)
    print(combined.head(40).to_string(index=False))

    # Pivot to compare
    pivot_pnl = combined.pivot_table(
        index=["entry_tier", "seg_direction"],
        columns="split", values=["n", "wr", "total", "mean"],
        aggfunc="first",
    ).fillna(0)
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    pivot_pnl.to_csv(os.path.join(OUT_DIR, f"{today}_06_is_oos_validation.csv"))
    print(f"\nWrote: {today}_06_is_oos_validation.csv")

    # 1W macro phase attachment
    print("\nAttaching 1W phases...")
    segs_1w = pd.read_csv(SEG_1W_PATH)
    twr2 = attach_1w_phase(twr, segs_1w)
    matched_1w = (twr2["seg1w_idx"] >= 0).sum()
    print(f"  Matched: {matched_1w}/{len(twr2)}")

    # Per-tier per-1W-phase
    print("\n=== Tier × 1W phase ===")
    ct_phase = crosstab(twr2[twr2["seg1w_idx"] >= 0], ["entry_tier", "seg1w_label"])
    print(ct_phase.head(30).to_string(index=False))
    ct_phase.to_csv(os.path.join(OUT_DIR, f"{today}_07_tier_x_1w_phase.csv"), index=False)

    # Best tier per 1W phase
    print("\n=== Best tier per 1W phase ===")
    best_per_phase = ct_phase.sort_values(["seg1w_label", "total"], ascending=[True, False])
    winners = best_per_phase.groupby("seg1w_label").first().reset_index()
    print(winners.to_string(index=False))

    # ── Timeline visualization ─────────────────────────────────────────
    print("\nGenerating tier-timeline chart...")
    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor("#0a0a0a")
    twr3 = twr2.sort_values("dt").copy()
    tier_colors = {
        "FADE_CALM":      "#22c55e",
        "FADE_AGAINST":   "#0099ff",
        "RIDE_AGAINST":   "#a78bfa",
        "BASE_NMP":       "#ffaa00",
        "KILL_SHOT":      "#f59e0b",
        "RIDE_CALM":      "#94a3b8",
        "RIDE_MOMENTUM":  "#06b6d4",
        "FADE_MOMENTUM":  "#ec4899",
        "FREIGHT_TRAIN":  "#fbbf24",
        "CASCADE":        "#ef4444",
    }
    # Cumulative PnL per tier
    for tier, color in tier_colors.items():
        sub = twr3[twr3["entry_tier"] == tier].copy()
        if sub.empty:
            continue
        sub["cum"] = sub["pnl"].cumsum()
        ax.plot(sub["dt"], sub["cum"], color=color, lw=1.5, alpha=0.9,
                 label=f"{tier} (${sub['pnl'].sum():+,.0f})")

    # Shade 1W segments by direction
    for _, s in segs_1w.iterrows():
        col = "#22c55e" if s["direction"] == "UP" else "#ef4444"
        s_dt = pd.to_datetime(s["start_dt"], utc=True)
        e_dt = pd.to_datetime(s["end_dt"], utc=True)
        ax.axvspan(s_dt, e_dt, color=col, alpha=0.06, zorder=0)

    ax.axhline(0, color="white", alpha=0.3, lw=0.5)
    ax.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="white",
              fontsize=9, framealpha=0.9, ncol=2)
    ax.set_title(f"Cumulative PnL per tier (combined IS+OOS) over time\n"
                  f"Background: 1W segments (green=UP, red=DOWN)",
                  color="white", fontsize=12)
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", color="#aaa")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{today}_08_tier_timeline.png")
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)
    print(f"Wrote: {out_png}")

    # ── Markdown summary ───────────────────────────────────────────────
    md_path = os.path.join(OUT_DIR, f"{today}_09_phase_winners.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Tier × Macro phase analysis\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## IS/OOS comparison for tier × 1D direction\n\n")
        f.write("```\n"); f.write(combined.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Best tier per 1W macro phase\n\n")
        f.write("```\n"); f.write(winners.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Per-tier summary across all phases\n\n")
        all_tiers_summary = crosstab(twr2[twr2["seg1w_idx"] >= 0], ["entry_tier"])
        f.write("```\n"); f.write(all_tiers_summary.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- IS/OOS validation: `{today}_06_is_oos_validation.csv`\n")
        f.write(f"- Tier x 1W phase: `{today}_07_tier_x_1w_phase.csv`\n")
        f.write(f"- Timeline chart: `{today}_08_tier_timeline.png`\n")
    print(f"Wrote: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
