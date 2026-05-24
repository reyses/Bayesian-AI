"""
tier_pnl_by_regime.py -- Overlay regime context onto the existing 9-tier trade
ledgers (blended_is_trades + blended_oos_trades). Find profitable SUBSETS:
which tier × regime combos actually win, which lose.

Goal: instead of running each tier as a standalone strategy, segment the
existing tier output by:
  - 1D segment direction (UP / DOWN)
  - 1D segment sub-pattern + zone behavior
  - dmi_diff at entry (from FEATURES_5s) — signed direction signal
  - normalized_slope intensity bucket

Then identify combinations where PnL/trade > $0 with sufficient sample size.

Inputs:
  - sandbox/training/output/trades/blended_is.csv (2886 trades)
  - sandbox/training/output/reports/blended_oos_trades.csv (1238 trades)
  - reports/findings/macro_segments/1D_segments.csv (77 segments)
  - DATA/ATLAS/FEATURES_5s/*.parquet (for dmi_diff lookup)

Outputs:
  reports/findings/tier_pnl_by_regime/
    01_tier_x_direction.csv
    02_tier_x_dmi_sign.csv
    03_tier_x_zone_behavior.csv
    04_profitable_subsets.csv (filtered: net>$0 + n>=20 + WR>50%)
    summary.md

Usage:
    python tools/tier_pnl_by_regime.py
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

IS_TRADES_PATH = "sandbox/training/output/trades/blended_is.csv"
OOS_TRADES_PATH = "sandbox/training/output/reports/blended_oos_trades.csv"
SEGMENTS_PATH = "reports/findings/macro_segments/1D_segments.csv"
FEATURES_DIR = "DATA/ATLAS/FEATURES_5s"


# =============================================================================
# Data loading
# =============================================================================

def load_trades(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def load_segments(path: str) -> pd.DataFrame:
    segs = pd.read_csv(path)
    segs["start_dt"] = pd.to_datetime(segs["start_dt"], utc=True)
    segs["end_dt"] = pd.to_datetime(segs["end_dt"], utc=True)
    return segs


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


# =============================================================================
# Attach regime + dmi_diff per trade
# =============================================================================

def attach_segment_regime(trades: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    out = trades.copy()
    intervals = pd.IntervalIndex.from_arrays(segments["start_dt"], segments["end_dt"], closed="left")
    seg_idx = []
    for _, t in out.iterrows():
        try:
            idx = intervals.get_indexer([t["dt"]])[0]
        except Exception:
            idx = -1
        seg_idx.append(idx)
    out["seg_idx"] = seg_idx
    seg_cols = [
        "direction", "sub_pattern", "slope_pts_per_day", "normalized_slope_abs",
        "duration_days", "n_zones_in_band",
        "start_in_zone", "end_in_zone", "escape_velocity", "captured", "between_zones",
    ]
    available = [c for c in seg_cols if c in segments.columns]
    seg_attrs = segments[available].copy()
    seg_attrs.columns = [f"seg_{c}" for c in available]
    seg_attrs["seg_idx"] = range(len(seg_attrs))
    out = out.merge(seg_attrs, on="seg_idx", how="left")
    out["seg_matched"] = out["seg_idx"] >= 0
    return out


def attach_features_at_entry(trades: pd.DataFrame, features: pd.DataFrame,
                                feature_cols: list[str]) -> pd.DataFrame:
    out = trades.copy()
    feat_ts = features["timestamp"].to_numpy(dtype=np.float64)
    trade_ts = out["timestamp"].to_numpy(dtype=np.float64)
    indices = np.searchsorted(feat_ts, trade_ts)
    indices_clipped = np.clip(indices, 1, len(feat_ts) - 1)
    left_dist = np.abs(trade_ts - feat_ts[indices_clipped - 1])
    right_dist = np.abs(trade_ts - feat_ts[np.clip(indices_clipped, 0, len(feat_ts) - 1)])
    use_left = left_dist <= right_dist
    matched_idx = np.where(use_left, indices_clipped - 1, indices_clipped)
    matched_dist = np.where(use_left, left_dist, right_dist)
    valid = matched_dist <= 60.0
    for c in feature_cols:
        if c not in features.columns:
            continue
        arr = features[c].to_numpy()
        out[c] = np.where(valid, arr[matched_idx], np.nan)
    out["feat_match_dist_sec"] = matched_dist
    out["feat_matched"] = valid
    return out


# =============================================================================
# Cross-tab helpers
# =============================================================================

def crosstab(df: pd.DataFrame, group_cols: list[str], pnl_col: str = "pnl") -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    return g.agg(
        n_trades=(pnl_col, "count"),
        wr_pct=(pnl_col, lambda x: 100 * (x > 0).mean()),
        pnl_total=(pnl_col, "sum"),
        pnl_mean=(pnl_col, "mean"),
        pnl_median=(pnl_col, "median"),
    ).reset_index().sort_values("pnl_total", ascending=False).reset_index(drop=True)


def slope_bucket(s: pd.Series) -> pd.Series:
    s2 = s.dropna()
    if s2.empty:
        return pd.Series(["unknown"] * len(s), index=s.index)
    p33, p66 = s2.quantile(0.33), s2.quantile(0.66)
    return s.apply(lambda v: "high" if v > p66 else ("medium" if v > p33 else "low")
                    if pd.notna(v) else "unknown")


def zone_label(row: pd.Series) -> str:
    if row.get("seg_escape_velocity"): return "escape_velocity"
    if row.get("seg_captured"):        return "captured"
    if row.get("seg_between_zones"):   return "between_zones"
    if row.get("seg_start_in_zone") and row.get("seg_end_in_zone"):
        return "cross_well"
    return "unknown"


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    print("=" * 80)
    print("9-TIER PnL × REGIME CROSS-TAB (with new context overlays)")
    print("=" * 80)

    # Load trades (IS + OOS, tag with split)
    print(f"\nLoading trade ledgers...")
    is_df = load_trades(IS_TRADES_PATH)
    oos_df = load_trades(OOS_TRADES_PATH)
    is_df["split"] = "IS"
    oos_df["split"] = "OOS"
    # Align columns
    common_cols = list(set(is_df.columns) & set(oos_df.columns))
    trades = pd.concat([is_df[common_cols], oos_df[common_cols]], ignore_index=True)
    trades = trades.sort_values("timestamp").reset_index(drop=True)
    print(f"  IS:  {len(is_df)} trades  (${is_df['pnl'].sum():+,.0f})")
    print(f"  OOS: {len(oos_df)} trades  (${oos_df['pnl'].sum():+,.0f})")
    print(f"  Combined: {len(trades)} trades")
    print(f"  Tiers: {sorted(trades['entry_tier'].dropna().unique())}")

    # Load segments
    print(f"\nLoading 1D segments...")
    segments = load_segments(SEGMENTS_PATH)
    print(f"  {len(segments)} 1D segments")

    # Load features (just the columns we need)
    print(f"\nLoading FEATURES_5s for dmi_diff lookup...")
    features = load_features()
    print(f"  {len(features):,} feature rows")

    # Attach regime
    print(f"\nAttaching 1D regime to trades...")
    twr = attach_segment_regime(trades, segments)
    matched_seg = twr["seg_matched"].sum()
    print(f"  Matched to segment: {matched_seg}/{len(twr)}")

    # Attach key dmi_diff features at entry
    feature_cols = ["1m_dmi_diff", "5m_dmi_diff", "15m_dmi_diff", "1h_dmi_diff", "1D_dmi_diff",
                    "1m_z_se", "5m_z_se", "15m_z_se",
                    "1m_variance_ratio", "5m_variance_ratio",
                    "1m_hurst", "5m_hurst",
                    "1m_reversion_prob"]
    print(f"\nAttaching features at entry (dmi_diff, z_se, vr, hurst)...")
    twr = attach_features_at_entry(twr, features, feature_cols)
    matched_feat = twr["feat_matched"].sum()
    print(f"  Features matched: {matched_feat}/{len(twr)}")

    # Derived buckets
    if "seg_normalized_slope_abs" in twr.columns:
        twr["seg_slope_bucket"] = slope_bucket(twr["seg_normalized_slope_abs"])
    twr["zone_behavior"] = twr.apply(zone_label, axis=1)
    twr["dmi_5m_sign"] = twr["5m_dmi_diff"].apply(
        lambda v: "pos" if v > 0 else ("neg" if v < 0 else "zero") if pd.notna(v) else "unknown"
    )
    twr["dmi_1m_sign"] = twr["1m_dmi_diff"].apply(
        lambda v: "pos" if v > 0 else ("neg" if v < 0 else "zero") if pd.notna(v) else "unknown"
    )
    twr["vr_5m_bucket"] = twr["5m_variance_ratio"].apply(
        lambda v: "high(>=1)" if v >= 1.0 else ("low(<0.7)" if v < 0.7 else "mid(0.7-1)")
        if pd.notna(v) else "unknown"
    )

    # Save full enriched trade table
    os.makedirs(args.out_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    enriched_csv = os.path.join(args.out_dir, f"{today}_trades_enriched.csv")
    twr.to_csv(enriched_csv, index=False)
    print(f"\nWrote: {enriched_csv}")

    # ── Cross-tabs ────────────────────────────────────────────────────────

    print("\n=== Tier × Direction (1D segment) ===")
    ct1 = crosstab(twr[twr["seg_matched"]],
                    ["entry_tier", "seg_direction"])
    print(ct1.to_string(index=False))
    ct1.to_csv(os.path.join(args.out_dir, f"{today}_01_tier_x_direction.csv"), index=False)

    print("\n=== Tier × dmi_5m_sign ===")
    ct2 = crosstab(twr[twr["feat_matched"]],
                    ["entry_tier", "dmi_5m_sign"])
    print(ct2.to_string(index=False))
    ct2.to_csv(os.path.join(args.out_dir, f"{today}_02_tier_x_dmi_5m.csv"), index=False)

    print("\n=== Tier × zone_behavior ===")
    ct3 = crosstab(twr[twr["seg_matched"]],
                    ["entry_tier", "zone_behavior"])
    print(ct3.to_string(index=False))
    ct3.to_csv(os.path.join(args.out_dir, f"{today}_03_tier_x_zone.csv"), index=False)

    print("\n=== Tier × Direction × dmi_5m_sign (joint) ===")
    ct4 = crosstab(twr[twr["seg_matched"] & twr["feat_matched"]],
                    ["entry_tier", "seg_direction", "dmi_5m_sign"])
    print(ct4.head(40).to_string(index=False))
    ct4.to_csv(os.path.join(args.out_dir, f"{today}_04_tier_x_direction_x_dmi.csv"), index=False)

    # ── Profitable subsets ────────────────────────────────────────────────
    # Find combos with positive PnL, decent sample, decent WR
    print("\n=== PROFITABLE SUBSETS (n>=20, pnl_total>$0, WR>50%) ===")
    profitable = []
    for ct, label in [(ct1, "tier x direction"),
                       (ct2, "tier x dmi_5m_sign"),
                       (ct3, "tier x zone"),
                       (ct4, "tier x direction x dmi_5m_sign")]:
        sub = ct[(ct["n_trades"] >= 20) & (ct["pnl_total"] > 0) & (ct["wr_pct"] > 50)].copy()
        sub["combo_label"] = label
        if not sub.empty:
            profitable.append(sub)
    if profitable:
        prof_df = pd.concat(profitable, ignore_index=True).sort_values("pnl_total", ascending=False)
        print(prof_df.head(30).to_string(index=False))
        prof_df.to_csv(os.path.join(args.out_dir, f"{today}_05_profitable_subsets.csv"), index=False)
    else:
        print("  No subsets meet criteria.")

    # ── Markdown summary ──────────────────────────────────────────────────
    md_path = os.path.join(args.out_dir, f"{today}_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# 9-Tier PnL by Regime — Cross-tabulated\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Headline\n\n")
        f.write(f"- Combined trades (IS+OOS): {len(twr)}\n")
        f.write(f"- Total PnL: ${twr['pnl'].sum():+,.0f}\n")
        f.write(f"- Trades matched to 1D segment: {matched_seg}/{len(twr)}\n")
        f.write(f"- Trades matched to features: {matched_feat}/{len(twr)}\n\n")
        f.write(f"## Tier × Direction (1D segment)\n\n")
        f.write("```\n"); f.write(ct1.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Tier × dmi_5m_sign\n\n")
        f.write("```\n"); f.write(ct2.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Tier × Zone behavior\n\n")
        f.write("```\n"); f.write(ct3.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Tier × Direction × dmi_5m_sign\n\n")
        f.write("```\n"); f.write(ct4.to_string(index=False)); f.write("\n```\n\n")
        if profitable:
            f.write(f"## Profitable subsets (n>=20, pnl>$0, WR>50%)\n\n")
            f.write("These are filter combinations where the 9-tier system DOES make money.\n\n")
            f.write("```\n"); f.write(prof_df.to_string(index=False)); f.write("\n```\n\n")
        else:
            f.write(f"## No profitable subsets met criteria.\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- Enriched trades: `{enriched_csv}`\n")
        f.write(f"- All cross-tabs: `{args.out_dir}/{today}_0*.csv`\n")
    print(f"\nWrote: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
