"""
tier_linreg_slope_filter.py -- Apply LinReg slope filter to existing tier trades.
For each tier, test whether skipping trades with adverse slope alignment saves money.

A "slope filter" rule: skip a trade if the LinReg slope at entry opposes the
trade direction by more than a threshold T. E.g., for a LONG entry: skip if
slope < -T (slope strongly negative = downtrend that we'd be fighting).

Process:
  1. Load enriched trade ledger (already has 1m_z_se, 5m_dmi_diff, etc.)
  2. Compute LinReg slope (period=30) at each entry timestamp from 1m closes
  3. For each tier, sweep slope_skip_threshold and find the best filter

Outputs:
  reports/findings/tier_pnl_by_regime/
    10_tier_linreg_slope_filter.csv
    10_tier_linreg_slope_filter.md
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
ATLAS_1M_DIR = "DATA/ATLAS/1m"


def find_latest_enriched():
    files = sorted(glob.glob(os.path.join(OUT_DIR, "*_trades_enriched.csv")))
    return files[-1] if files else None


def load_1m_closes() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, "*.parquet")))
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f)[["timestamp", "close"]])
        except Exception:
            pass
    df = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def rolling_linreg_slope(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return out
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = (x_centered ** 2).sum()
    for i in range(period - 1, n):
        y = closes[i - period + 1: i + 1]
        ymean = y.mean()
        slope = (x_centered * (y - ymean)).sum() / x_var
        out[i] = slope
    return out


def compute_slope_at_entries(trades: pd.DataFrame, bars_1m: pd.DataFrame,
                                period: int = 30) -> pd.DataFrame:
    closes = bars_1m["close"].to_numpy(dtype=np.float64)
    slope = rolling_linreg_slope(closes, period)
    feat_ts = bars_1m["timestamp"].to_numpy(dtype=np.float64)
    trade_ts = trades["timestamp"].to_numpy(dtype=np.float64)
    indices = np.searchsorted(feat_ts, trade_ts)
    indices_clipped = np.clip(indices, 1, len(feat_ts) - 1)
    left_dist = np.abs(trade_ts - feat_ts[indices_clipped - 1])
    right_dist = np.abs(trade_ts - feat_ts[np.clip(indices_clipped, 0, len(feat_ts) - 1)])
    use_left = left_dist <= right_dist
    matched_idx = np.where(use_left, indices_clipped - 1, indices_clipped)
    matched_dist = np.where(use_left, left_dist, right_dist)
    valid = matched_dist <= 60.0
    out = trades.copy()
    out[f"linreg_slope_{period}"] = np.where(valid, slope[matched_idx], np.nan)
    out["slope_match_dist_sec"] = matched_dist
    return out


def evaluate_filter(trades: pd.DataFrame, slope_col: str,
                     skip_threshold: float, dir_col: str = "dir") -> dict:
    """Skip if (LONG and slope < -T) OR (SHORT and slope > +T)."""
    df = trades.copy()
    is_long = df[dir_col].str.upper().isin(["LONG", "BUY", "L"])
    skip = ((is_long & (df[slope_col] < -skip_threshold)) |
             (~is_long & (df[slope_col] > skip_threshold)))
    keep = df[~skip]
    skip_n = skip.sum()
    keep_n = (~skip).sum()
    out = {
        "threshold": skip_threshold,
        "n_total": len(df),
        "n_kept": keep_n,
        "n_skipped": skip_n,
        "pnl_kept": float(keep["pnl"].sum()),
        "pnl_skipped": float(df[skip]["pnl"].sum()),
        "wr_kept": float((keep["pnl"] > 0).mean() * 100) if keep_n > 0 else 0.0,
        "wr_skipped": float((df[skip]["pnl"] > 0).mean() * 100) if skip_n > 0 else 0.0,
        "pnl_per_kept_trade": float(keep["pnl"].mean()) if keep_n > 0 else 0.0,
        "pnl_per_skipped_trade": float(df[skip]["pnl"].mean()) if skip_n > 0 else 0.0,
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched-csv", default=None)
    ap.add_argument("--linreg-period", type=int, default=30)
    ap.add_argument("--thresholds", default="0.5,1.0,1.5,2.0,3.0,5.0",
                    help="Comma-separated slope thresholds (pts/bar)")
    args = ap.parse_args()

    print("=" * 80)
    print("TIER × LINREG-SLOPE FILTER ANALYSIS")
    print("=" * 80)

    enriched_path = args.enriched_csv or find_latest_enriched()
    print(f"Loading: {enriched_path}")
    twr = pd.read_csv(enriched_path)
    twr["dt"] = pd.to_datetime(twr["timestamp"], unit="s", utc=True)
    print(f"  {len(twr)} trades")

    print(f"\nLoading 1m bars + computing LinReg slope ({args.linreg_period})...")
    bars = load_1m_closes()
    print(f"  {len(bars):,} bars")
    twr = compute_slope_at_entries(twr, bars, period=args.linreg_period)
    slope_col = f"linreg_slope_{args.linreg_period}"
    matched = (~twr[slope_col].isna()).sum()
    print(f"  Slope matched: {matched}/{len(twr)}")

    # Per-tier filter sweep
    thresholds = [float(x) for x in args.thresholds.split(",")]
    print(f"\nThreshold sweep: {thresholds}")
    rows = []
    for tier in twr["entry_tier"].dropna().unique():
        sub = twr[twr["entry_tier"] == tier].dropna(subset=[slope_col])
        if len(sub) < 20:
            continue
        baseline_pnl = sub["pnl"].sum()
        baseline_n = len(sub)
        baseline_wr = (sub["pnl"] > 0).mean() * 100
        rows.append({
            "tier": tier, "threshold": 0.0, "n_kept": baseline_n,
            "n_skipped": 0, "pnl_kept": baseline_pnl,
            "pnl_skipped": 0.0,
            "wr_kept": baseline_wr, "wr_skipped": 0.0,
            "pnl_delta": 0.0,
            "pnl_per_kept_trade": sub["pnl"].mean(),
        })
        for T in thresholds:
            r = evaluate_filter(sub, slope_col, T)
            r["tier"] = tier
            r["pnl_delta"] = r["pnl_kept"] - baseline_pnl
            rows.append(r)

    cols = ["tier", "threshold", "n_total", "n_kept", "n_skipped",
              "pnl_kept", "pnl_skipped", "wr_kept", "wr_skipped",
              "pnl_per_kept_trade", "pnl_per_skipped_trade", "pnl_delta"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]

    today = datetime.now().strftime("%Y-%m-%d")
    out_csv = os.path.join(OUT_DIR, f"{today}_10_tier_linreg_slope_filter.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")

    # Print best threshold per tier
    print("\n=== Best slope threshold per tier (by PnL kept) ===")
    df_no_baseline = df[df["threshold"] > 0]
    best = df_no_baseline.sort_values(["tier", "pnl_kept"], ascending=[True, False]).groupby("tier").first().reset_index()
    baseline = df[df["threshold"] == 0][["tier", "pnl_kept", "n_kept"]].rename(columns={"pnl_kept": "baseline_pnl", "n_kept": "baseline_n"})
    best = best.merge(baseline, on="tier", how="left")
    best["improvement"] = best["pnl_kept"] - best["baseline_pnl"]
    best = best.sort_values("improvement", ascending=False)
    print(best[["tier", "baseline_n", "baseline_pnl", "threshold", "n_kept",
                  "pnl_kept", "improvement", "pnl_per_kept_trade"]].to_string(index=False))

    # Markdown
    md_path = os.path.join(OUT_DIR, f"{today}_10_tier_linreg_slope_filter.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# LinReg slope filter applied per tier\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- LinReg period: {args.linreg_period} bars (1m timeframe)\n")
        f.write(f"- Filter rule: skip LONG if slope < -T, skip SHORT if slope > +T\n")
        f.write(f"- T sweep: {thresholds} pts/bar\n\n")
        f.write(f"## Best threshold per tier (sorted by improvement vs no-filter)\n\n")
        f.write("```\n")
        f.write(best[["tier", "baseline_n", "baseline_pnl", "threshold", "n_kept",
                       "pnl_kept", "improvement", "pnl_per_kept_trade"]].to_string(index=False))
        f.write("\n```\n\n")
        f.write(f"## Full sweep\n\n")
        f.write("```\n"); f.write(df.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Interpretation\n\n")
        f.write("- `improvement` > 0: skipping adverse-slope trades helped that tier\n")
        f.write("- `improvement` < 0: filter caused more harm (skipped trades that would have won)\n")
        f.write("- `pnl_per_kept_trade` going up after filter = filter is positive selection\n")
    print(f"Wrote: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
