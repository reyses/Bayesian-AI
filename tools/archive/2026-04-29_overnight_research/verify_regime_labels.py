"""
verify_regime_labels.py -- Validate that atlas_regime_labeler classifications
actually correspond to the intuitive 4-class regime archetypes:

  1. Pure trend       (directional, low intraday variation = smooth ramp)
  2. Trending chop    (directional, high variation = jagged with net move)
  3. Pure chop        (non-directional, high variation = range churn)
  4. Quiet/flat       (non-directional, low variation = barely moves)

Tests:
  A. Scatter plot:  directional_strength × efficiency_ratio, colored by regime.
                    Distinct clusters = good labels; overlap = mushy labels.
  B. Sample charts: 5 random days per regime label, intraday 1m bars plotted.
                    Visually verify they match the archetype name.
  C. Distribution analysis: per-regime histograms of each metric.
                    Sanity check that ranges don't overlap too much.
  D. 2D classification check: split UP/DOWN by efficiency_ratio to see if
                    we're conflating "pure trend" with "trending chop".

Usage:
    python tools/verify_regime_labels.py
    python tools/verify_regime_labels.py --n-samples 5
"""
from __future__ import annotations
import argparse
import os
import sys
import gc
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

from tools.zigzag_genetic import load_1m_bars
from tools.atlas_regime_labeler import (
    load_regime_labels, REGIME_COLORS, REGIME_ORDER
)


# =============================================================================
# Test A: 2D scatter of metric space
# =============================================================================

def plot_metric_scatter(daily: pd.DataFrame, out_png: str):
    """Scatter directional_strength × efficiency_ratio, color = regime."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor("#0a0a0a")

    pairs = [
        ("directional_strength", "efficiency_ratio", "DirStr vs EffRatio"),
        ("directional_strength", "range_expansion", "DirStr vs RangeExp"),
        ("efficiency_ratio", "range_expansion", "EffRatio vs RangeExp"),
    ]
    for ax, (xcol, ycol, title) in zip(axes, pairs):
        for regime in REGIME_ORDER:
            sub = daily[daily["regime"] == regime]
            if sub.empty:
                continue
            ax.scatter(sub[xcol], sub[ycol],
                        color=REGIME_COLORS[regime], alpha=0.55, s=30,
                        label=f"{regime} ({len(sub)})", edgecolors="white",
                        linewidths=0.3)
        ax.set_xlabel(xcol, color="#ccc")
        ax.set_ylabel(ycol, color="#ccc")
        ax.set_title(title, color="white")
        ax.legend(fontsize=8, facecolor="#1a1a1a", labelcolor="white", loc="upper right")
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#aaa")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


# =============================================================================
# Test B: sample days per regime, intraday plot
# =============================================================================

def plot_regime_samples(bars: pd.DataFrame, daily: pd.DataFrame, n_samples: int,
                          out_dir: str, tz: str = "America/Los_Angeles"):
    """Plot N random sample days from each regime in a grid."""
    bars = bars.copy()
    bars["date"] = bars["dt_utc"].dt.tz_convert(tz).dt.date
    rng = np.random.default_rng(42)

    for regime in REGIME_ORDER:
        sub_days = daily[daily["regime"] == regime]
        if sub_days.empty:
            continue
        # Pick n_samples random days
        n_pick = min(n_samples, len(sub_days))
        chosen = sub_days.sample(n=n_pick, random_state=42)

        # Grid layout: ceil(sqrt(N)) cols
        ncols = int(np.ceil(np.sqrt(n_pick)))
        nrows = int(np.ceil(n_pick / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                   squeeze=False)
        fig.patch.set_facecolor("#0a0a0a")

        for i, (_, row) in enumerate(chosen.iterrows()):
            ax = axes[i // ncols, i % ncols]
            day_bars = bars[bars["date"] == row["date"]]
            if day_bars.empty:
                ax.axis("off")
                continue
            # Plot intraday close
            ax.plot(day_bars["dt_utc"], day_bars["close"], color="white", lw=0.8)
            ax.axhline(row["open"], color="#0099ff", alpha=0.5, lw=0.5, ls="--",
                        label=f"open {row['open']:.1f}")
            ax.axhline(row["close"], color="#ffaa00", alpha=0.5, lw=0.5, ls="--",
                        label=f"close {row['close']:.1f}")
            net = row["close"] - row["open"]
            color_net = "#22c55e" if net > 0 else "#ef4444"
            ax.set_title(
                f"{row['date']}  net{net:+.1f} pts\n"
                f"DirStr={row['directional_strength']:.2f}  "
                f"Eff={row['efficiency_ratio']:.2f}  "
                f"RngExp={row['range_expansion']:.2f}",
                fontsize=9, color="white",
            )
            ax.set_facecolor("#1a1a1a")
            ax.tick_params(colors="#aaa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_color("#444")
            ax.grid(alpha=0.2)
            # Hide x-axis labels (datetimes are noisy)
            ax.set_xticklabels([])

        # Hide empty subplots
        for j in range(n_pick, nrows * ncols):
            axes[j // ncols, j % ncols].axis("off")

        fig.suptitle(f"Regime: {regime}  ({len(sub_days)} days total in dataset)",
                       color=REGIME_COLORS[regime], fontsize=13, fontweight="bold")
        plt.tight_layout()
        out_png = f"{out_dir}/samples_{regime}.png"
        fig.savefig(out_png, dpi=100, facecolor="#0a0a0a")
        plt.close(fig)
        print(f"  Wrote: {out_png}")


# =============================================================================
# Test C: histograms per metric
# =============================================================================

def plot_metric_histograms(daily: pd.DataFrame, out_png: str):
    """Per-metric histograms colored by regime."""
    metrics = ["directional_strength", "efficiency_ratio", "range_expansion", "range"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor("#0a0a0a")
    axes = axes.flatten()

    for ax, m in zip(axes, metrics):
        for regime in REGIME_ORDER:
            sub = daily[daily["regime"] == regime]
            if sub.empty or sub[m].dropna().empty:
                continue
            ax.hist(sub[m].dropna(), bins=20, alpha=0.5,
                     color=REGIME_COLORS[regime], label=f"{regime} ({len(sub)})",
                     edgecolor="white", linewidth=0.3)
        ax.set_xlabel(m, color="#ccc")
        ax.set_ylabel("count", color="#ccc")
        ax.set_title(m, color="white")
        ax.legend(fontsize=7, facecolor="#1a1a1a", labelcolor="white")
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#aaa")

    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


# =============================================================================
# Test D: 2D refined regime check (does UP split into pure-trend vs trending-chop?)
# =============================================================================

def split_by_efficiency(daily: pd.DataFrame, eff_threshold: float = 0.5) -> pd.DataFrame:
    """Add a refined regime that splits UP/DOWN by efficiency_ratio:
       - UP_PURE_TREND   = UP with eff_ratio >= 0.5 (smooth ramp)
       - UP_TRENDING_CHOP = UP with eff_ratio < 0.5 (jagged with net up)
       Same for DOWN. CHOP/QUIET/TRANSITIONAL unchanged.
    """
    out = daily.copy()
    out["regime_v2"] = out["regime"]

    up_mask = (out["regime"] == "UP")
    out.loc[up_mask & (out["efficiency_ratio"] >= eff_threshold), "regime_v2"] = "UP_PURE"
    out.loc[up_mask & (out["efficiency_ratio"] < eff_threshold), "regime_v2"] = "UP_TRENDING_CHOP"

    dn_mask = (out["regime"] == "DOWN")
    out.loc[dn_mask & (out["efficiency_ratio"] >= eff_threshold), "regime_v2"] = "DOWN_PURE"
    out.loc[dn_mask & (out["efficiency_ratio"] < eff_threshold), "regime_v2"] = "DOWN_TRENDING_CHOP"

    return out


def report_v2_split(daily: pd.DataFrame, out_md_path: str):
    """Markdown report of v2 split with stats."""
    daily_v2 = split_by_efficiency(daily, 0.5)
    counts = daily_v2["regime_v2"].value_counts()

    by_regime = (
        daily_v2.groupby("regime_v2")
        .agg(
            n=("date", "count"),
            avg_dir=("directional_strength", "mean"),
            avg_eff=("efficiency_ratio", "mean"),
            avg_range=("range_expansion", "mean"),
            avg_net=("net_move", "mean"),
        )
        .reset_index()
    )
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# Regime label verification\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write("## Refined v2 split: UP/DOWN by efficiency_ratio (>= 0.5 = pure, < 0.5 = trending chop)\n\n")
        f.write("Tests whether the original UP/DOWN labels conflate two distinct archetypes:\n")
        f.write("  - PURE: smooth directional move (high efficiency)\n")
        f.write("  - TRENDING_CHOP: directional but jagged (low efficiency)\n\n")
        f.write("If many UP days are TRENDING_CHOP, the original label is mushy.\n\n")
        f.write("## v2 distribution\n\n")
        f.write("| Regime v2 | N | % |\n|---|---:|---:|\n")
        total = len(daily_v2)
        for r, n in counts.items():
            f.write(f"| {r} | {n} | {100*n/total:.1f}% |\n")
        f.write("\n## Per-regime metric averages (v2)\n\n")
        f.write("```\n")
        f.write(by_regime.to_string(index=False))
        f.write("\n```\n\n")
        f.write("## Interpretation guide\n\n")
        f.write("| Metric pattern | Archetype |\n|---|---|\n")
        f.write("| dir>0.6, eff>0.5, exp>1.0 | Pure UP/DOWN trend |\n")
        f.write("| dir>0.6, eff<0.5, exp>1.0 | Trending chop (directional but messy) |\n")
        f.write("| dir<0.4, exp>1.0 | Pure chop (non-directional, high vol) |\n")
        f.write("| dir<0.4, exp<0.5 | Quiet/flat (no direction, low vol) |\n")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--n-samples", type=int, default=6,
                    help="Sample days per regime to plot (default 6 = 2x3 grid)")
    ap.add_argument("--out-dir", default="reports/findings/regime_eda")
    args = ap.parse_args()

    print("=" * 80)
    print("REGIME LABEL VERIFICATION")
    print("=" * 80)

    print("Loading regime labels...")
    daily = load_regime_labels(args.atlas)
    print(f"  {len(daily)} session days")
    print()

    print("Loading 1m bars (for sample-day plots)...")
    bars = load_1m_bars(args.atlas)
    print(f"  {len(bars):,} bars")
    print()

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Test A: scatter
    print("Test A: metric-space scatter...")
    scatter_png = f"{out_dir}/{today}_metric_scatter.png"
    plot_metric_scatter(daily, scatter_png)
    print(f"  Wrote: {scatter_png}")
    print()

    # Test B: sample charts per regime
    print(f"Test B: {args.n_samples} sample days per regime...")
    samples_dir = f"{out_dir}/samples_{today}"
    os.makedirs(samples_dir, exist_ok=True)
    plot_regime_samples(bars, daily, args.n_samples, samples_dir)
    print()

    # Test C: histograms
    print("Test C: per-metric histograms...")
    hist_png = f"{out_dir}/{today}_metric_histograms.png"
    plot_metric_histograms(daily, hist_png)
    print(f"  Wrote: {hist_png}")
    print()

    # Test D: refined v2 split
    print("Test D: refined v2 split (UP/DOWN x efficiency_ratio)...")
    md_path = f"{out_dir}/{today}_label_verification.md"
    report_v2_split(daily, md_path)
    print(f"  Wrote: {md_path}")

    # Print v2 distribution to stdout
    daily_v2 = split_by_efficiency(daily, 0.5)
    print()
    print("v2 distribution (UP/DOWN split by efficiency_ratio @ 0.5):")
    counts = daily_v2["regime_v2"].value_counts()
    total = len(daily_v2)
    for r, n in counts.items():
        print(f"  {r:>22s}: {n:>4d}  ({100*n/total:.1f}%)")
    print()

    # Print metric stats per regime
    print("Per-regime metric means (v1 labels):")
    by_regime = (
        daily.groupby("regime")
        .agg(
            n=("date", "count"),
            avg_dir=("directional_strength", "mean"),
            avg_eff=("efficiency_ratio", "mean"),
            avg_range_exp=("range_expansion", "mean"),
            avg_net=("net_move", "mean"),
        )
    )
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(by_regime.to_string())

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print("Inspect:")
    print(f"  - {scatter_png}                  (do regimes form distinct clusters?)")
    print(f"  - {samples_dir}/samples_*.png    (do sample days look like the label?)")
    print(f"  - {hist_png}             (per-metric distribution overlap)")
    print(f"  - {md_path}              (v2 split into PURE vs TRENDING_CHOP)")
    gc.collect()


if __name__ == "__main__":
    main()
