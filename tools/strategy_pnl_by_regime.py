"""
strategy_pnl_by_regime.py -- Cross-tab v1.0.4 strategy PnL by 1D regime label.

Goal: quantify where v1.0.4 actually wins and loses, broken down by:
  - 1D segment direction (UP / DOWN)
  - Normalized slope intensity (low / medium / high)
  - Zone behavior (free-fall / escape_velocity / captured / cross-well)
  - Sub-pattern (L_H / H_L / HH_lower / ... )

Validates the asymmetric regime hypothesis from cross-TF nesting:
  UP-legs should be the strategy's sweet spot; DOWN-legs should bleed.

Inputs:
  - DATA/ATLAS 14-month OHLC
  - reports/findings/macro_segments/1D_segments.csv (built by macro_slope_segmenter)
  - v1.0.4 strategy: R=45, counter-trend default, no SL/MFE/trail

Outputs:
  reports/findings/strategy_pnl_by_regime/v104_pnl_by_regime.csv
  reports/findings/strategy_pnl_by_regime/v104_pnl_by_regime.md
  reports/findings/strategy_pnl_by_regime/v104_trades_with_regime.csv

Usage:
    python tools/strategy_pnl_by_regime.py
    python tools/strategy_pnl_by_regime.py --r 45 --segments-csv reports/findings/macro_segments/1D_segments.csv
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

from tools.zigzag_genetic import load_1m_bars, DOLLAR_PER_POINT, COMMISSION_RT, SLIPPAGE_PTS
from tools.zigzag_linreg_eda import simulate_with_slope, rolling_linreg_slope

OUT_DIR = "reports/findings/strategy_pnl_by_regime"


# =============================================================================
# Run v1.0.4 simulation (R=45, no SL, no MFE-cut, no trail, counter-trend)
# =============================================================================

def run_v104_sim(bars: pd.DataFrame, r_points: float = 45.0) -> pd.DataFrame:
    """Run v1.0.4 strategy on bars, return DataFrame of trades."""
    closes = bars["close"].to_numpy(dtype=np.float64)
    # Compute LinReg slope (not used for direction, just for trade-time slope reference)
    slope = rolling_linreg_slope(closes, period=30)
    trades, _, _, _ = simulate_with_slope(
        bars, slope,
        r_points=r_points,
        max_loss_pts=0.0,
        mfe_cut_bars=0,
        mfe_cut_usd=0.0,
        trail_activate_pts=0.0,
        trail_giveback_pct=0.0,
        flip_direction=0,        # counter-trend default
        direction_mode="counter",
    )
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame(trades)


# =============================================================================
# Attach 1D segment regime to each trade
# =============================================================================

def attach_segment_regime(trades_df: pd.DataFrame, segments_df: pd.DataFrame) -> pd.DataFrame:
    """For each trade, find the 1D segment containing entry_dt and attach
    direction/sub_pattern/slope/zone-behavior columns."""
    if trades_df.empty or segments_df.empty:
        return trades_df

    # Ensure timestamps are tz-aware UTC
    trades = trades_df.copy()
    trades["entry_dt"] = pd.to_datetime(trades["entry_dt"], utc=True)

    segs = segments_df.copy()
    segs["start_dt"] = pd.to_datetime(segs["start_dt"], utc=True)
    segs["end_dt"]   = pd.to_datetime(segs["end_dt"], utc=True)

    # IntervalIndex for fast lookup — close on the LEFT, OPEN on the right (so
    # boundary trades go to the segment whose start matches their entry_dt).
    intervals = pd.IntervalIndex.from_arrays(segs["start_dt"], segs["end_dt"], closed="left")

    seg_idx = []
    for _, t in trades.iterrows():
        try:
            idx = intervals.get_indexer([t["entry_dt"]])[0]
        except Exception:
            idx = -1
        seg_idx.append(idx)
    trades["seg_idx"] = seg_idx

    # Pick segment-attribute columns to merge
    seg_cols = [
        "direction", "sub_pattern", "slope_pts_per_day", "normalized_slope_abs",
        "duration_days", "n_zones_in_band", "n_levels_in_band",
        "start_in_zone", "end_in_zone", "escape_velocity", "captured", "between_zones",
    ]
    available = [c for c in seg_cols if c in segs.columns]
    seg_attrs = segs[available].copy()
    seg_attrs.columns = [f"seg_{c}" for c in available]
    seg_attrs["seg_idx"] = range(len(seg_attrs))

    out = trades.merge(seg_attrs, on="seg_idx", how="left")
    out["seg_matched"] = out["seg_idx"] >= 0
    return out


# =============================================================================
# Cross-tabs
# =============================================================================

def cross_tab_by_dimension(trades_with_regime: pd.DataFrame, dim_col: str,
                              label: str) -> pd.DataFrame:
    """Aggregate trades by a single regime dimension."""
    if dim_col not in trades_with_regime.columns:
        return pd.DataFrame()
    sub = trades_with_regime[trades_with_regime["seg_matched"]].copy()
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(dim_col, dropna=False)
    out = g.agg(
        n_trades=("pnl_usd", "count"),
        wr_pct=("pnl_usd", lambda x: 100 * (x > 0).mean()),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
        pnl_median=("pnl_usd", "median"),
    ).reset_index()
    out = out.sort_values("pnl_total", ascending=False).reset_index(drop=True)
    out["dimension"] = label
    return out


def normalize_slope_bucket(seg_normalized_slope_abs: pd.Series) -> pd.Series:
    """Bucket the normalized slope into low/medium/high relative to dataset distribution."""
    s = seg_normalized_slope_abs.dropna()
    if s.empty:
        return pd.Series([], dtype="object")
    p33, p66 = s.quantile(0.33), s.quantile(0.66)
    return seg_normalized_slope_abs.apply(
        lambda v: "high" if v > p66 else ("medium" if v > p33 else "low")
        if pd.notna(v) else "unknown"
    )


def zone_behavior_label(row: pd.Series) -> str:
    """Categorical zone-behavior label."""
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
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--r", type=float, default=45.0,
                    help="Zigzag R points (default 45 = v1.0.4 baseline)")
    ap.add_argument("--segments-csv",
                    default="reports/findings/macro_segments/1D_segments.csv",
                    help="1D segments CSV from macro_slope_segmenter")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    print("=" * 80)
    print("STRATEGY PnL BY REGIME (v1.0.4 + 1D segments)")
    print("=" * 80)

    # Load bars
    print(f"\nLoading 1m bars from {args.atlas}...")
    bars = load_1m_bars(args.atlas)
    print(f"  {len(bars):,} bars  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})")

    # Run v1.0.4 sim
    print(f"\nRunning v1.0.4 sim (R={args.r}, counter-trend, no SL/MFE/trail)...")
    trades = run_v104_sim(bars, r_points=args.r)
    print(f"  {len(trades)} trades simulated")
    if trades.empty:
        print("No trades. Aborting.")
        return

    # Headline aggregate
    total_pnl = trades["pnl_usd"].sum()
    total_n = len(trades)
    total_wr = (trades["pnl_usd"] > 0).mean() * 100
    print(f"\nHeadline:")
    print(f"  Total PnL:  ${total_pnl:+,.2f}")
    print(f"  Trades:     {total_n}")
    print(f"  WR:         {total_wr:.1f}%")
    print(f"  $/trade:    ${total_pnl/total_n:+,.2f}")

    # Load 1D segments
    print(f"\nLoading 1D segments from {args.segments_csv}...")
    if not os.path.exists(args.segments_csv):
        print(f"  Missing! Re-run macro_slope_segmenter.py first.")
        sys.exit(1)
    segments = pd.read_csv(args.segments_csv)
    print(f"  {len(segments)} 1D segments")

    # Attach regime per trade
    print(f"\nAttaching 1D regime to each trade...")
    twr = attach_segment_regime(trades, segments)
    matched = twr["seg_matched"].sum()
    unmatched = (~twr["seg_matched"]).sum()
    print(f"  Matched: {matched}  Unmatched: {unmatched}")

    # Add bucket columns
    if "seg_normalized_slope_abs" in twr.columns:
        twr["seg_slope_bucket"] = normalize_slope_bucket(twr["seg_normalized_slope_abs"])
    twr["zone_behavior"] = twr.apply(zone_behavior_label, axis=1)

    # Cross-tabs
    cross_tabs = []
    for dim_col, label in [
        ("seg_direction", "Direction"),
        ("seg_sub_pattern", "Sub-pattern"),
        ("seg_slope_bucket", "Slope intensity (33/66 quantiles)"),
        ("zone_behavior", "Zone behavior"),
    ]:
        ct = cross_tab_by_dimension(twr, dim_col, label)
        if not ct.empty:
            cross_tabs.append((label, ct))

    # Print + save
    os.makedirs(args.out_dir, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(args.out_dir, f"{today}_v104_pnl_by_regime.md")
    csv_path = os.path.join(args.out_dir, f"{today}_v104_trades_with_regime.csv")
    summary_csv = os.path.join(args.out_dir, f"{today}_v104_pnl_by_regime.csv")

    twr.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}  ({len(twr)} trade rows)")

    summary_rows = []
    for label, ct in cross_tabs:
        ct["regime_dimension"] = label
        summary_rows.append(ct)
    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(summary_csv, index=False)
        print(f"Wrote: {summary_csv}")

    # Markdown report
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# v1.0.4 PnL by 1D regime - cross-tab\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Atlas: `{args.atlas}`\n")
        f.write(f"- Strategy: v1.0.4 (R={args.r}, counter-trend, no SL/MFE/trail)\n")
        f.write(f"- Segments: `{args.segments_csv}`\n\n")
        f.write(f"## Headline (full 14-month sim)\n\n")
        f.write(f"| Metric | Value |\n|---|---:|\n")
        f.write(f"| Total PnL | ${total_pnl:+,.2f} |\n")
        f.write(f"| Trades | {total_n} |\n")
        f.write(f"| Win rate | {total_wr:.1f}% |\n")
        f.write(f"| $/trade | ${total_pnl/total_n:+,.2f} |\n")
        f.write(f"| Trades with matching 1D segment | {matched}/{total_n} |\n\n")

        for label, ct in cross_tabs:
            f.write(f"## By {label}\n\n")
            f.write(ct[[c for c in ct.columns if c != "dimension"]].to_string(index=False))
            f.write("\n\n")

        # Direction asymmetry deep-dive
        if "seg_direction" in twr.columns:
            f.write(f"## Direction asymmetry (UP vs DOWN — the key hypothesis)\n\n")
            up = twr[twr["seg_direction"] == "UP"]
            dn = twr[twr["seg_direction"] == "DOWN"]
            if not up.empty and not dn.empty:
                up_pnl = up["pnl_usd"].sum()
                dn_pnl = dn["pnl_usd"].sum()
                up_per_trade = up_pnl / len(up)
                dn_per_trade = dn_pnl / len(dn)
                up_wr = 100 * (up["pnl_usd"] > 0).mean()
                dn_wr = 100 * (dn["pnl_usd"] > 0).mean()
                f.write(f"| Side | Trades | $/trade | WR | Total PnL |\n")
                f.write(f"|---|---:|---:|---:|---:|\n")
                f.write(f"| UP regimes | {len(up)} | ${up_per_trade:+.2f} | {up_wr:.1f}% | ${up_pnl:+,.2f} |\n")
                f.write(f"| DOWN regimes | {len(dn)} | ${dn_per_trade:+.2f} | {dn_wr:.1f}% | ${dn_pnl:+,.2f} |\n")
                f.write(f"| **Delta** | | **${up_per_trade - dn_per_trade:+.2f}/trade** | **{up_wr - dn_wr:+.1f}pp** | **${up_pnl - dn_pnl:+,.2f}** |\n\n")
                if up_per_trade > dn_per_trade:
                    f.write(f"**v1.0.4 makes ${up_per_trade - dn_per_trade:+.2f} more per trade in UP regimes vs DOWN.** "
                              f"Confirms the asymmetric-regime hypothesis from cross-TF nesting.\n\n")
                else:
                    f.write(f"v1.0.4 actually does BETTER in DOWN — refutes the cross-TF hypothesis. "
                              f"Investigate.\n\n")

    print(f"Wrote: {md_path}")

    # Print direction asymmetry headline
    print("\n" + "=" * 80)
    print("DIRECTION ASYMMETRY — KEY VERDICT")
    print("=" * 80)
    if "seg_direction" in twr.columns:
        up = twr[twr["seg_direction"] == "UP"]
        dn = twr[twr["seg_direction"] == "DOWN"]
        if not up.empty and not dn.empty:
            print(f"  UP regimes:   {len(up)} trades, ${up['pnl_usd'].sum():+,.2f}, "
                   f"${up['pnl_usd'].mean():+.2f}/trade, WR {(up['pnl_usd']>0).mean()*100:.1f}%")
            print(f"  DOWN regimes: {len(dn)} trades, ${dn['pnl_usd'].sum():+,.2f}, "
                   f"${dn['pnl_usd'].mean():+.2f}/trade, WR {(dn['pnl_usd']>0).mean()*100:.1f}%")
            print(f"  Delta UP-DOWN: ${up['pnl_usd'].mean() - dn['pnl_usd'].mean():+.2f}/trade, "
                   f"${up['pnl_usd'].sum() - dn['pnl_usd'].sum():+,.2f} total")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
