"""
peak_feature_overlay.py -- Map how each 91D feature interacts with peaks
across all manually-marked and auto-detected TFs.

For each feature column in DATA/ATLAS/FEATURES_5s/*.parquet, we ask:
  - What does the feature look like AT peaks (vs at random non-peak bars)?
  - Does it separate UP peaks from DOWN peaks (i.e., HIGH vs LOW snap)?
  - Does it differ across TFs?

Outputs:
  reports/findings/peak_feature_overlay/
    01_effect_size_table.csv     (Cohen's d per feature/condition, ranked)
    02_per_feature_dists.png     (one panel per feature: peak vs baseline hist)
    03_h_vs_l_separation.png     (which features split H peaks from L peaks)
    04_per_tf_separation.png     (heatmap: feature x TF, color = effect size)
    summary.md                   (markdown overview)

Usage:
    python tools/peak_feature_overlay.py
    python tools/peak_feature_overlay.py --top-features 24 --sample-size 50000
"""
from __future__ import annotations
import argparse
import gc
import glob
import json
import os
import sys
import time
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

SEEDS_DIR = "DATA/regime_seeds"
FEATURES_DIR = "DATA/ATLAS/FEATURES_5s"
OUT_DIR = "reports/findings/peak_feature_overlay"


# =============================================================================
# Load all peaks across TFs
# =============================================================================

def find_best_peaks_file(tf: str) -> str | None:
    """Find peak file for TF: prefer manual full-range > auto > Feb 1-7."""
    pat = os.path.join(SEEDS_DIR, f"*_{tf}.json")
    files = sorted(glob.glob(pat))
    files = [f for f in files
             if "augmented" not in os.path.basename(f)
             and "merged" not in os.path.basename(f)]
    if not files:
        return None
    def _count(p):
        try:
            with open(p) as fh:
                return len(json.load(fh).get("peaks", []))
        except Exception:
            return 0
    files.sort(key=lambda p: -_count(p))
    return files[0]


def load_all_peaks() -> pd.DataFrame:
    """Load peaks from all TFs, combine into single DataFrame.
    Returns: timestamp, snap (H/L), tf, source ('manual'/'auto'), price."""
    rows = []
    for tf in ["1W", "1D", "4h", "1h", "15m", "1m"]:
        path = find_best_peaks_file(tf)
        if not path:
            continue
        with open(path) as f:
            d = json.load(f)
        source = "auto" if "auto_peaks" in os.path.basename(path) else "manual"
        for p in d.get("peaks", []):
            rows.append({
                "timestamp": float(p["timestamp"]),
                "snap": p["_snap"],
                "tf": tf,
                "source": source,
                "price": float(p.get("price", 0.0)),
                "src_file": os.path.basename(path),
            })
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# =============================================================================
# Load all features
# =============================================================================

def load_features() -> pd.DataFrame:
    """Load all FEATURES_5s parquet files, concat, sort by timestamp."""
    files = sorted(glob.glob(os.path.join(FEATURES_DIR, "*.parquet")))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            print(f"  WARN read fail {f}: {e}")
    if not parts:
        raise RuntimeError(f"no features loaded from {FEATURES_DIR}")
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


# =============================================================================
# Match peaks to nearest feature row (5s tolerance)
# =============================================================================

def match_peaks_to_features(peaks_df: pd.DataFrame, features_df: pd.DataFrame,
                              tolerance_seconds: float = 60.0) -> pd.DataFrame:
    """For each peak, find nearest feature row within tolerance.
    Returns peaks_df with feature columns merged in."""
    feat_ts = features_df["timestamp"].to_numpy(dtype=np.float64)
    peak_ts = peaks_df["timestamp"].to_numpy(dtype=np.float64)

    # Nearest-time match using searchsorted
    indices = np.searchsorted(feat_ts, peak_ts)
    indices_clipped = np.clip(indices, 1, len(feat_ts) - 1)
    left_dist = np.abs(peak_ts - feat_ts[indices_clipped - 1])
    right_dist = np.abs(peak_ts - feat_ts[np.clip(indices_clipped, 0, len(feat_ts) - 1)])
    use_left = left_dist <= right_dist
    matched_idx = np.where(use_left, indices_clipped - 1, indices_clipped)
    matched_dist = np.where(use_left, left_dist, right_dist)

    valid = matched_dist <= tolerance_seconds
    print(f"  Matched {valid.sum()}/{len(peaks_df)} peaks within "
           f"{tolerance_seconds}s tolerance")

    # Build merged DataFrame
    feat_cols = [c for c in features_df.columns if c != "timestamp"]
    merged = peaks_df.copy()
    for c in feat_cols:
        feat_array = features_df[c].to_numpy()
        merged[c] = np.where(valid, feat_array[matched_idx], np.nan)
    merged["match_dist_sec"] = matched_dist
    merged["matched"] = valid
    return merged


# =============================================================================
# Effect size analysis (Cohen's d) — peak vs baseline
# =============================================================================

def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-stdev Cohen's d between two distributions."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 5 or len(b) < 5:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def compute_effect_sizes(peaks_with_features: pd.DataFrame,
                           baseline: pd.DataFrame,
                           feature_cols: list[str]) -> pd.DataFrame:
    """For each feature: Cohen's d for (all_peaks vs baseline), (H vs L peaks),
    (peaks per-TF vs baseline)."""
    rows = []
    bl_arrays = {c: baseline[c].to_numpy(dtype=np.float64) for c in feature_cols}
    matched = peaks_with_features[peaks_with_features["matched"]]
    h_peaks = matched[matched["snap"] == "H"]
    l_peaks = matched[matched["snap"] == "L"]

    for c in feature_cols:
        peak_vals = matched[c].to_numpy(dtype=np.float64)
        h_vals = h_peaks[c].to_numpy(dtype=np.float64)
        l_vals = l_peaks[c].to_numpy(dtype=np.float64)
        bl_vals = bl_arrays[c]

        d_peak_vs_bl = cohen_d(peak_vals, bl_vals)
        d_h_vs_l = cohen_d(h_vals, l_vals)
        d_h_vs_bl = cohen_d(h_vals, bl_vals)
        d_l_vs_bl = cohen_d(l_vals, bl_vals)

        # Per-TF
        per_tf = {}
        for tf in ["1W", "1D", "4h", "1h", "15m", "1m"]:
            sub = matched[matched["tf"] == tf][c].to_numpy(dtype=np.float64)
            per_tf[f"d_{tf}_vs_bl"] = cohen_d(sub, bl_vals)

        rows.append({
            "feature": c,
            "n_peaks": len(peak_vals),
            "n_h": len(h_vals),
            "n_l": len(l_vals),
            "n_baseline": len(bl_vals),
            "peak_mean": float(np.nanmean(peak_vals)),
            "peak_std": float(np.nanstd(peak_vals)),
            "baseline_mean": float(np.nanmean(bl_vals)),
            "baseline_std": float(np.nanstd(bl_vals)),
            "d_peak_vs_bl": d_peak_vs_bl,
            "d_h_vs_l": d_h_vs_l,
            "d_h_vs_bl": d_h_vs_bl,
            "d_l_vs_bl": d_l_vs_bl,
            **per_tf,
        })

    return pd.DataFrame(rows).sort_values("d_h_vs_l", key=lambda x: -x.abs()).reset_index(drop=True)


# =============================================================================
# Plots
# =============================================================================

def plot_top_features_dist(peaks: pd.DataFrame, baseline: pd.DataFrame,
                              effect_sizes: pd.DataFrame, top_n: int,
                              out_path: str):
    """Plot histograms for top-N features (sorted by |d_h_vs_l|)."""
    top = effect_sizes.head(top_n)
    n_cols = 4
    n_rows = (len(top) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.patch.set_facecolor("#0a0a0a")
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    matched = peaks[peaks["matched"]]
    h_peaks = matched[matched["snap"] == "H"]
    l_peaks = matched[matched["snap"] == "L"]

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i // n_cols, i % n_cols]
        col = row["feature"]
        bl_vals = baseline[col].dropna().to_numpy()
        h_vals = h_peaks[col].dropna().to_numpy()
        l_vals = l_peaks[col].dropna().to_numpy()
        # Common x-axis range
        all_vals = np.concatenate([bl_vals, h_vals, l_vals])
        if len(all_vals) > 0:
            xmin, xmax = np.percentile(all_vals, [1, 99])
        else:
            xmin, xmax = 0, 1
        bins = np.linspace(xmin, xmax, 30)
        ax.hist(bl_vals, bins=bins, color="#888", alpha=0.4, density=True,
                 label=f"baseline (n={len(bl_vals)})")
        ax.hist(h_vals, bins=bins, color="#0099ff", alpha=0.6, density=True,
                 label=f"H peaks (n={len(h_vals)})")
        ax.hist(l_vals, bins=bins, color="#ffaa00", alpha=0.6, density=True,
                 label=f"L peaks (n={len(l_vals)})")
        ax.set_title(f"{col}\nd(H-L)={row['d_h_vs_l']:+.2f}  d(P-BL)={row['d_peak_vs_bl']:+.2f}",
                      fontsize=9, color="white")
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#aaa", labelsize=7)
        ax.legend(fontsize=6, facecolor="#1a1a1a", labelcolor="white",
                   framealpha=0.7)

    # Hide unused
    for j in range(len(top), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


def plot_per_tf_heatmap(effect_sizes: pd.DataFrame, top_n: int, out_path: str):
    """Heatmap: top features x TF, colored by effect size."""
    top = effect_sizes.head(top_n)
    tf_cols = [c for c in effect_sizes.columns if c.startswith("d_") and c.endswith("_vs_bl")
               and c not in ("d_peak_vs_bl", "d_h_vs_bl", "d_l_vs_bl")]
    mat = top[tf_cols].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 0.4 * top_n + 2))
    fig.patch.set_facecolor("#0a0a0a")
    cmap = plt.get_cmap("RdBu_r")
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], color="#ccc", fontsize=8)
    ax.set_xticks(range(len(tf_cols)))
    ax.set_xticklabels([c.replace("d_", "").replace("_vs_bl", "") for c in tf_cols],
                       color="#ccc")
    ax.set_title(f"Per-TF feature separation (top {top_n} features)\n"
                  "Color = Cohen's d (peak vs baseline)", color="white")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="#ccc")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


def plot_h_vs_l_separation(effect_sizes: pd.DataFrame, top_n: int, out_path: str):
    """Bar chart: top features by |d_h_vs_l|."""
    top = effect_sizes.head(top_n).iloc[::-1]  # reverse for top-down
    fig, ax = plt.subplots(figsize=(12, 0.3 * top_n + 2))
    fig.patch.set_facecolor("#0a0a0a")
    colors = ["#22c55e" if v > 0 else "#ef4444" for v in top["d_h_vs_l"]]
    ax.barh(range(len(top)), top["d_h_vs_l"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"], color="#ccc", fontsize=8)
    ax.axvline(0, color="white", alpha=0.3, lw=0.5)
    ax.set_xlabel("Cohen's d (H peaks - L peaks)", color="#ccc")
    ax.set_title(f"Top {top_n} features by H-vs-L separation\n"
                  "Positive (green) = feature higher at H peaks. Negative (red) = higher at L peaks.",
                  color="white")
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-features", type=int, default=20,
                    help="Number of top features to plot in detail")
    ap.add_argument("--sample-size", type=int, default=100000,
                    help="Baseline sample size for effect-size computation")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    print("=" * 80)
    print("PEAK x FEATURE OVERLAY ANALYSIS")
    print("=" * 80)

    # Load peaks
    print("\nLoading peaks across TFs...")
    peaks = load_all_peaks()
    print(f"  Total: {len(peaks)} peaks")
    print(f"  Per TF:")
    print(peaks.groupby(["tf", "source", "snap"]).size())
    print()

    # Load features
    print("\nLoading features...")
    t0 = time.time()
    features = load_features()
    print(f"  Loaded {len(features):,} feature rows in {time.time()-t0:.1f}s")
    feature_cols = [c for c in features.columns if c != "timestamp"]
    print(f"  Feature columns: {len(feature_cols)}")

    # Match peaks to features
    print("\nMatching peaks to nearest feature row...")
    peaks_w_features = match_peaks_to_features(peaks, features, tolerance_seconds=60.0)

    # Sample baseline (random non-peak bars from full feature set)
    print(f"\nSampling baseline (n={args.sample_size}) from feature set...")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(features), size=min(args.sample_size, len(features)), replace=False)
    baseline = features.iloc[sample_idx].reset_index(drop=True)
    print(f"  Baseline shape: {baseline.shape}")

    # Compute effect sizes
    print("\nComputing effect sizes...")
    es = compute_effect_sizes(peaks_w_features, baseline, feature_cols)
    print(f"  {len(es)} features ranked by |d_h_vs_l|")
    print(f"\nTop 15 features by H-vs-L separation:")
    print(es[["feature", "d_h_vs_l", "d_peak_vs_bl", "d_h_vs_bl", "d_l_vs_bl",
                "n_peaks", "peak_mean", "baseline_mean"]].head(15).to_string(index=False))

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join(args.out_dir, f"{today}_01_effect_size_table.csv")
    es.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    # Plots
    print("\nGenerating plots...")
    fig1 = os.path.join(args.out_dir, f"{today}_02_per_feature_dists.png")
    plot_top_features_dist(peaks_w_features, baseline, es, args.top_features, fig1)
    print(f"  Wrote: {fig1}")

    fig2 = os.path.join(args.out_dir, f"{today}_03_h_vs_l_separation.png")
    plot_h_vs_l_separation(es, args.top_features, fig2)
    print(f"  Wrote: {fig2}")

    fig3 = os.path.join(args.out_dir, f"{today}_04_per_tf_separation.png")
    plot_per_tf_heatmap(es, args.top_features, fig3)
    print(f"  Wrote: {fig3}")

    # Markdown summary
    md_path = os.path.join(args.out_dir, f"{today}_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Peak x feature overlay analysis\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Inputs\n\n")
        f.write(f"- Peaks: {len(peaks)} total across 6 TFs\n")
        f.write(f"- Features: {len(feature_cols)} columns from `{FEATURES_DIR}`\n")
        f.write(f"- Baseline: random {args.sample_size} feature rows\n")
        f.write(f"- Match tolerance: 60 seconds\n\n")
        f.write(f"## Top 20 features by |Cohen's d| (H peaks vs L peaks)\n\n")
        f.write("Positive d means feature is HIGHER at H (resistance) peaks.\n")
        f.write("Negative d means HIGHER at L (support) peaks.\n\n")
        f.write("```\n")
        f.write(es[["feature", "d_h_vs_l", "d_peak_vs_bl", "d_h_vs_bl", "d_l_vs_bl",
                       "n_peaks", "peak_mean", "baseline_mean"]].head(20).to_string(index=False))
        f.write("\n```\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- Effect size CSV: `{csv_path}`\n")
        f.write(f"- Per-feature distributions: `{fig1}`\n")
        f.write(f"- H-vs-L bar chart: `{fig2}`\n")
        f.write(f"- Per-TF heatmap: `{fig3}`\n")
    print(f"Wrote: {md_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
