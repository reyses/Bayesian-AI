"""
zigzag_regime_finder.py -- What separates zigzag-favorable days from chop days?

For each day in ATLAS, compute regime features available at the OPEN of the
trading day (= using only PRIOR-DAY data, no peek-ahead), then run the zigzag
simulator for that day to get the actual PnL. Correlate features with PnL,
extract simple decision rules.

The user's framing (2026-04-26): "find the reason -- if we do then jackpot".

Features (all FORWARD-AVAILABLE at open of day D):
  - prior_day_range                 = day D-1: high - low (in points)
  - prior_day_return                = day D-1: close - open
  - prior_day_close_minus_open      = day D-1 directional close
  - mean_range_5d                   = mean(range) over D-1..D-5
  - mean_range_20d                  = mean(range) over D-1..D-20
  - range_expansion_ratio           = prior_day_range / mean_range_20d
  - range_trajectory                = mean_range_5d / mean_range_20d
  - variance_ratio_5_20             = var(close, 5) / var(close, 20)  -- trend signal
  - dow                             = day of week (0=Mon..6=Sun)
  - prior_5d_pnl_zigzag             = optional: feedback feature (autocorrelation regime)

Output:
  reports/findings/zigzag_regime/_summary.csv      (one row per day: features + pnl)
  reports/findings/zigzag_regime/correlations.txt  (Spearman corr table)
  reports/findings/zigzag_regime/feature_*.png     (scatter/box per feature)
  reports/findings/zigzag_regime/rules.txt         (simple if-then thresholds)

Usage:
    python tools/zigzag_regime_finder.py                       # all available days
    python tools/zigzag_regime_finder.py --days 100 --seed 42  # 100 random days
    python tools/zigzag_regime_finder.py --r 30 --t1act 30 --t1dist 15 --t2pct 0.10 --sl 35
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zigzag_trail_ticker import simulate_day, ATLAS_ROOT  # type: ignore

warnings.filterwarnings("ignore")


# ── Daily OHLC loader ───────────────────────────────────────────────────────
def _load_daily_ohlc(atlas_root: str) -> pd.DataFrame:
    """Read every 1D parquet, return concat sorted by timestamp.
    Adds 'day_label' column matching the 1m/1s file naming convention."""
    daily_dir = os.path.join(atlas_root, "1D")
    rows = []
    for fname in sorted(os.listdir(daily_dir)):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(daily_dir, fname)
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty:
            continue
        # day_label e.g. "2026_03_15"
        df["day_label"] = fname[:-len(".parquet")]
        rows.append(df)
    if not rows:
        raise RuntimeError(f"No 1D parquets found in {daily_dir}")
    out = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return out


# ── Feature engineering (pre-day forward-available only) ────────────────────
def compute_features(daily: pd.DataFrame) -> pd.DataFrame:
    """For each day D, compute features using only days < D.

    Output columns appended in-place; row index aligned to `daily` row order.
    Features for day D are all derived from D-1, D-2, ... back. NEVER day D itself.
    """
    df = daily.copy()
    df["range"]       = df["high"] - df["low"]
    df["return_pts"]  = df["close"] - df["open"]
    df["return_pct"]  = (df["close"] - df["open"]) / df["open"]
    df["close_pct"]   = df["close"].pct_change()

    # All "prior_day_*" features SHIFT by 1 so day D sees day D-1's value.
    df["prior_day_range"]    = df["range"].shift(1)
    df["prior_day_return"]   = df["return_pts"].shift(1)
    df["prior_day_close_pct"] = df["close_pct"].shift(1)

    # Rolling stats — use shift(1) so day D's window ends at D-1.
    range_shift = df["range"].shift(1)
    df["mean_range_5d"]  = range_shift.rolling(5,  min_periods=3).mean()
    df["mean_range_20d"] = range_shift.rolling(20, min_periods=10).mean()
    df["std_range_20d"]  = range_shift.rolling(20, min_periods=10).std()

    df["range_expansion"] = df["prior_day_range"] / df["mean_range_20d"]
    df["range_trajectory"] = df["mean_range_5d"] / df["mean_range_20d"]

    # Variance ratio: short variance / long variance.
    # > 1: returns are trending. < 1: mean-reverting.
    close_shift = df["close"].shift(1)
    var_5  = close_shift.pct_change().rolling(5,  min_periods=3).var()
    var_20 = close_shift.pct_change().rolling(20, min_periods=10).var()
    # Annualize-equivalent ratio (multiply by 4 since we compare 5-bar to 20-bar)
    df["variance_ratio_5_20"] = (var_5 * 4.0) / var_20.replace(0, np.nan)

    # Calendar features
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["dow"] = df["timestamp_dt"].dt.dayofweek

    # Prior-day "directional drift sign"
    df["prior_dir_sign"] = np.sign(df["prior_day_return"])

    return df


# ── Run zigzag per day ──────────────────────────────────────────────────────
def evaluate_zigzag_day(day_label: str, params: dict, atlas_root: str) -> dict:
    """Run simulate_day, return key per-day stats (or zeros if no data)."""
    try:
        trades, _summary = simulate_day(
            day_label=day_label,
            r=params["r"],
            trail_activate=params["t1act"],
            trail_dist=params["t1dist"],
            trail_pct=params["t2pct"],
            atlas_root=atlas_root,
            use_filter=False,
            sl_pts=params["sl"],
        )
    except Exception:
        return {"pnl_usd": 0.0, "n_trades": 0, "error": True}
    if not trades:
        return {"pnl_usd": 0.0, "n_trades": 0, "error": False}
    pnl = sum(t.get("pnl_usd", 0.0) for t in trades)
    return {"pnl_usd": float(pnl), "n_trades": len(trades), "error": False}


# ── Correlation + rule extraction ────────────────────────────────────────────
FEATURE_COLS = [
    "prior_day_range",
    "prior_day_return",
    "prior_day_close_pct",
    "mean_range_5d",
    "mean_range_20d",
    "range_expansion",
    "range_trajectory",
    "variance_ratio_5_20",
    "prior_dir_sign",
    "dow",
]


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of each feature with day PnL."""
    rows = []
    for f in FEATURE_COLS:
        sub = df[[f, "pnl_usd"]].dropna()
        if len(sub) < 5:
            continue
        rho = sub[f].corr(sub["pnl_usd"], method="spearman")
        # Also compute "win rate when feature in top quartile vs bottom quartile"
        q75 = sub[f].quantile(0.75)
        q25 = sub[f].quantile(0.25)
        top    = sub[sub[f] >= q75]
        bottom = sub[sub[f] <= q25]
        wr_top    = (top["pnl_usd"] > 0).mean() if len(top) else float("nan")
        wr_bottom = (bottom["pnl_usd"] > 0).mean() if len(bottom) else float("nan")
        mean_top    = top["pnl_usd"].mean()    if len(top) else float("nan")
        mean_bottom = bottom["pnl_usd"].mean() if len(bottom) else float("nan")
        rows.append({
            "feature": f,
            "spearman_rho": rho,
            "abs_rho": abs(rho) if pd.notna(rho) else 0.0,
            "wr_top_quartile":    wr_top,
            "wr_bottom_quartile": wr_bottom,
            "mean_top_quartile":  mean_top,
            "mean_bottom_quartile": mean_bottom,
            "delta_mean":         (mean_top - mean_bottom) if pd.notna(mean_top) and pd.notna(mean_bottom) else float("nan"),
            "n_obs":              len(sub),
        })
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def threshold_rules(df: pd.DataFrame, top_features: list[str]) -> list[dict]:
    """For top features, find threshold (median) rules that separate winning/losing days."""
    rules = []
    for f in top_features:
        sub = df[[f, "pnl_usd"]].dropna()
        if len(sub) < 10:
            continue
        median = sub[f].median()
        above = sub[sub[f] >= median]
        below = sub[sub[f] <  median]
        rules.append({
            "feature":              f,
            "rule":                 f"{f} >= {median:.4f}",
            "n_above":              len(above),
            "mean_pnl_above":       above["pnl_usd"].mean(),
            "wr_above":             (above["pnl_usd"] > 0).mean(),
            "n_below":              len(below),
            "mean_pnl_below":       below["pnl_usd"].mean(),
            "wr_below":             (below["pnl_usd"] > 0).mean(),
            "edge_above_minus_below": above["pnl_usd"].mean() - below["pnl_usd"].mean(),
        })
    return rules


def plot_feature_vs_pnl(df: pd.DataFrame, feature: str, out_path: str):
    sub = df[[feature, "pnl_usd"]].dropna()
    if len(sub) < 5:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in sub["pnl_usd"]]
    ax.scatter(sub[feature], sub["pnl_usd"], c=colors, alpha=0.7)
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_xlabel(feature)
    ax.set_ylabel("$/day")
    rho = sub[feature].corr(sub["pnl_usd"], method="spearman")
    ax.set_title(f"{feature} vs Day PnL  (Spearman rho = {rho:+.3f}, N = {len(sub)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=ATLAS_ROOT)
    ap.add_argument("--days",  type=int, default=0,
                    help="Random sample N days; 0 = use all available days")
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--r",     type=float, default=30.0)
    ap.add_argument("--t1act", type=float, default=10.0)
    ap.add_argument("--t1dist", type=float, default=5.0)
    ap.add_argument("--t2pct", type=float, default=0.10)
    ap.add_argument("--sl",    type=float, default=25.0)
    ap.add_argument("--out",   default="reports/findings/zigzag_regime")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load all days, compute features
    daily = _load_daily_ohlc(args.atlas)
    daily = compute_features(daily)
    n_total = len(daily)
    print(f"Loaded {n_total} daily bars from {args.atlas}/1D/")

    # Pick days to evaluate
    eligible = daily.dropna(subset=FEATURE_COLS)  # need full 20-day history
    print(f"Eligible (with 20-day history): {len(eligible)}")

    if args.days > 0 and args.days < len(eligible):
        sample = eligible.sample(n=args.days, random_state=args.seed).sort_values("timestamp")
    else:
        sample = eligible

    print(f"Evaluating zigzag on {len(sample)} days...")
    print(f"Params: r={args.r}, t1act={args.t1act}, t1dist={args.t1dist}, "
          f"t2pct={args.t2pct}, sl={args.sl}")

    params = {"r": args.r, "t1act": args.t1act, "t1dist": args.t1dist,
              "t2pct": args.t2pct, "sl": args.sl}

    pnl_rows = []
    for _, row in sample.iterrows():
        day_label = row["day_label"]
        result = evaluate_zigzag_day(day_label, params, args.atlas)
        feat = {f: row[f] for f in FEATURE_COLS}
        feat["day_label"] = day_label
        feat["pnl_usd"]   = result["pnl_usd"]
        feat["n_trades"]  = result["n_trades"]
        pnl_rows.append(feat)

    df = pd.DataFrame(pnl_rows)

    # Filter out days where simulator returned 0 pnl AND 0 trades (no 1s data)
    valid = df[df["n_trades"] > 0].copy()
    print(f"Valid (>0 trades): {len(valid)}")

    # Save the per-day frame
    csv_path = os.path.join(args.out, "_summary.csv")
    valid.to_csv(csv_path, index=False)

    # Quick stats
    print(f"\nDay PnL stats (N={len(valid)}):")
    print(f"  mean   = ${valid['pnl_usd'].mean():+.2f}/day")
    print(f"  median = ${valid['pnl_usd'].median():+.2f}/day")
    print(f"  Day WR = {(valid['pnl_usd'] > 0).mean()*100:.1f}%")
    print(f"  Best   = ${valid['pnl_usd'].max():+.2f}")
    print(f"  Worst  = ${valid['pnl_usd'].min():+.2f}")

    # Correlation table
    corr = correlation_table(valid)
    corr_path = os.path.join(args.out, "correlations.txt")
    with open(corr_path, "w") as f:
        f.write(f"Day PnL stats (N={len(valid)}):\n")
        f.write(f"  mean   = ${valid['pnl_usd'].mean():+.2f}/day\n")
        f.write(f"  median = ${valid['pnl_usd'].median():+.2f}/day\n")
        f.write(f"  Day WR = {(valid['pnl_usd'] > 0).mean()*100:.1f}%\n\n")
        f.write("Spearman correlation: feature vs day PnL.\n")
        f.write("wr_top_quartile / wr_bottom_quartile: win-rate when feature is in top/bottom 25%.\n")
        f.write("delta_mean: mean PnL diff when feature is high vs low.\n\n")
        f.write(corr.to_string(index=False, float_format="%.4f"))
        f.write("\n")

    print(f"\n=== Top features by |Spearman rho| ===")
    print(corr.head(8).to_string(index=False, float_format="%.4f"))

    # Threshold rules for top 4 features
    top_features = corr.head(4)["feature"].tolist()
    rules = threshold_rules(valid, top_features)
    rules_path = os.path.join(args.out, "rules.txt")
    with open(rules_path, "w") as f:
        f.write("Median-split decision rules for top-4 |rho| features.\n")
        f.write("If 'edge_above_minus_below' is large + same-sign as rho, the\n")
        f.write("feature provides a tradeable signal.\n\n")
        for r in rules:
            f.write(f"FEATURE: {r['feature']}\n")
            f.write(f"  Rule:                 {r['rule']}\n")
            f.write(f"  ABOVE: N={r['n_above']:>3}  mean=${r['mean_pnl_above']:+8.2f}  WR={r['wr_above']*100:5.1f}%\n")
            f.write(f"  BELOW: N={r['n_below']:>3}  mean=${r['mean_pnl_below']:+8.2f}  WR={r['wr_below']*100:5.1f}%\n")
            f.write(f"  Edge (above - below): ${r['edge_above_minus_below']:+8.2f}\n\n")

    print(f"\n=== Median-split rules (top 4) ===")
    for r in rules:
        print(f"\n{r['feature']:30s}  rule: {r['rule']}")
        print(f"  ABOVE: N={r['n_above']:>3} mean=${r['mean_pnl_above']:+7.2f} WR={r['wr_above']*100:.1f}%")
        print(f"  BELOW: N={r['n_below']:>3} mean=${r['mean_pnl_below']:+7.2f} WR={r['wr_below']*100:.1f}%")
        print(f"  Edge: ${r['edge_above_minus_below']:+7.2f}/day")

    # Plot scatter for each feature
    for f in FEATURE_COLS:
        out_png = os.path.join(args.out, f"feature_{f}.png")
        plot_feature_vs_pnl(valid, f, out_png)

    print(f"\nWrote {csv_path}, {corr_path}, {rules_path}, {len(FEATURE_COLS)} feature plots")


if __name__ == "__main__":
    main()
