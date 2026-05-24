"""
zigzag_regime_eda.py -- Stage B: regime-aware cross-tab EDA.

Runs the ZigzagRunner sim under multiple direction-mode/threshold combinations,
attaches each trade to its entry-day's regime label, and produces a cross-tab
of PnL / WR / trade-count per (mode, regime).

This answers the question we couldn't before: WHICH MODE WINS IN WHICH REGIME.

Prerequisite: run `python tools/atlas_regime_labeler.py` first to produce
DATA/ATLAS/regime_labels.csv.

Usage:
    python tools/zigzag_regime_eda.py --r 45 --period 30
    python tools/zigzag_regime_eda.py --r 45 --period 30 --start 2025-01-01 --end 2026-03-21

Outputs:
    reports/findings/regime_eda/<date>_regime_crosstab_R<r>.md       (markdown)
    reports/findings/regime_eda/<date>_regime_crosstab_R<r>.csv      (CSV: long format)
    reports/findings/regime_eda/<date>_regime_crosstab_R<r>_pnl.csv  (pivot: PnL by regime × mode)

Direction modes tested by default:
  - counter (v1.0.4 baseline)
  - skip @ T = 0.5, 1.0      (skip the pivot if slope strongly opposes would-be entry)
  - adaptive @ T = 0.5, 1.0  (flip to with-slope when |slope| > T)
  - stay @ T = 0.5, 1.0      (don't exit existing position when slope still favors it)
"""
from __future__ import annotations
import argparse
import gc
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

from tools.zigzag_genetic import load_1m_bars, DOLLAR_PER_POINT, COMMISSION_RT, SLIPPAGE_PTS
from tools.zigzag_linreg_eda import rolling_linreg_slope, simulate_with_slope
from tools.atlas_regime_labeler import load_regime_labels, REGIME_ORDER


# =============================================================================
# Mode configurations
# =============================================================================

# (tag, direction_mode, continuation_threshold)
DEFAULT_MODES = [
    ("counter",         "counter",  0.0),
    ("skip_T0.5",       "skip",     0.5),
    ("skip_T1.0",       "skip",     1.0),
    ("adaptive_T0.5",   "adaptive", 0.5),
    ("adaptive_T1.0",   "adaptive", 1.0),
    ("stay_T0.5",       "stay",     0.5),
    ("stay_T1.0",       "stay",     1.0),
]


# =============================================================================
# Regime attachment
# =============================================================================

def attach_regime_to_trades(trades_df: pd.DataFrame, labels_df: pd.DataFrame,
                              tz: str = "America/Los_Angeles") -> pd.DataFrame:
    """Join each trade to the regime label of its entry day."""
    df = trades_df.copy()
    df["entry_date"] = pd.to_datetime(df["entry_dt"]).dt.tz_convert(tz).dt.date
    lbl = labels_df.copy()
    lbl["date"] = pd.to_datetime(lbl["date"]).dt.date
    df = df.merge(lbl[["date", "regime"]], left_on="entry_date", right_on="date",
                   how="left")
    df = df.drop(columns=["date"])
    df["regime"] = df["regime"].fillna("UNKNOWN")
    return df


def stats_by_regime(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Per-regime aggregates: n, WR, PnL total, PnL mean, MFE mean, bars_held mean."""
    if trades_df.empty:
        return pd.DataFrame(columns=["regime", "n", "wr", "pnl_total", "pnl_mean", "mfe_mean", "bars_held_mean"])
    g = trades_df.groupby("regime")
    out = g.agg(
        n=("pnl_usd", "count"),
        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
        mfe_mean=("mfe_usd", "mean"),
        bars_held_mean=("bars_held", "mean"),
    ).reset_index()
    return out


# =============================================================================
# Per-side cuts (Long vs Short × regime)
# =============================================================================

def stats_by_regime_side(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    g = trades_df.groupby(["regime", "side"])
    out = g.agg(
        n=("pnl_usd", "count"),
        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
    ).reset_index()
    return out


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--start", default="2025-01-01")
    ap.add_argument("--end", default="2026-03-21")
    ap.add_argument("--period", type=int, default=30, help="LinReg slope period")
    ap.add_argument("--r", type=float, default=45.0, help="Zigzag R points")
    ap.add_argument("--max-loss", type=float, default=0.0)
    ap.add_argument("--mfe-cut-bars", type=int, default=0)
    ap.add_argument("--mfe-cut-usd", type=float, default=0.0)
    ap.add_argument("--trail-activate", type=float, default=0.0)
    ap.add_argument("--trail-giveback", type=float, default=0.0)
    ap.add_argument("--out-dir", default="reports/findings/regime_eda")
    args = ap.parse_args()

    print("=" * 80)
    print("STAGE B: REGIME-AWARE EDA")
    print("=" * 80)
    print(f"Atlas:   {args.atlas}")
    print(f"Window:  {args.start} -> {args.end}")
    print(f"R:       {args.r} pts  |  LinReg period: {args.period}")
    print()

    print("Loading bars...")
    bars = load_1m_bars(args.atlas)
    bars = bars[
        (bars["dt_utc"] >= pd.Timestamp(args.start, tz="UTC"))
        & (bars["dt_utc"] <= pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(hours=23, minutes=59))
    ].reset_index(drop=True)
    print(f"  {len(bars):,} 1m bars")

    print("Loading regime labels...")
    labels = load_regime_labels(args.atlas)
    print(f"  {len(labels)} session days")
    print("  Distribution:")
    for r, n in labels["regime"].value_counts().items():
        print(f"    {r:>14s}: {n:>4d}  ({100*n/len(labels):.1f}%)")
    print()

    print(f"Computing LinReg({args.period}) slope...")
    closes = bars["close"].to_numpy(dtype=np.float64)
    slope = rolling_linreg_slope(closes, args.period)

    print(f"\nRunning sim under {len(DEFAULT_MODES)} mode/threshold combos...")
    print()

    all_stats_rows = []        # long-format CSV: tag, regime, n, wr, pnl_total, etc.
    all_side_stats_rows = []   # long-format: tag, regime, side, ...
    pivots_pnl = {}            # tag -> {regime -> pnl_total}
    pivots_n   = {}            # tag -> {regime -> n}
    pivots_wr  = {}            # tag -> {regime -> wr}

    for tag, mode, threshold in DEFAULT_MODES:
        t0 = time.time()
        trades, _, _, _ = simulate_with_slope(
            bars, slope,
            r_points=args.r,
            max_loss_pts=args.max_loss,
            mfe_cut_bars=args.mfe_cut_bars,
            mfe_cut_usd=args.mfe_cut_usd,
            trail_activate_pts=args.trail_activate,
            trail_giveback_pct=args.trail_giveback,
            direction_mode=mode,
            continuation_threshold=threshold,
        )
        elapsed = time.time() - t0
        if not trades:
            print(f"  {tag:>16s}: 0 trades")
            continue
        df = pd.DataFrame(trades)
        df = attach_regime_to_trades(df, labels)
        # Total stats
        total_pnl = df["pnl_usd"].sum()
        total_n = len(df)
        total_wr = (df["pnl_usd"] > 0).mean() * 100
        print(f"  {tag:>16s}: {total_n:>5d} trades  ${total_pnl:>+10,.2f}  WR {total_wr:.1f}%  "
               f"({elapsed:.1f}s)")

        # Per-regime stats
        st = stats_by_regime(df)
        st["mode_tag"] = tag
        all_stats_rows.append(st)
        # Per-regime side stats
        st_side = stats_by_regime_side(df)
        st_side["mode_tag"] = tag
        all_side_stats_rows.append(st_side)

        pivots_pnl[tag] = dict(zip(st["regime"], st["pnl_total"]))
        pivots_n[tag]   = dict(zip(st["regime"], st["n"]))
        pivots_wr[tag]  = dict(zip(st["regime"], st["wr"]))
        # ALL row
        pivots_pnl[tag]["ALL"] = total_pnl
        pivots_n[tag]["ALL"]   = total_n
        pivots_wr[tag]["ALL"]  = total_wr

    # Compile cross-tabs
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.out_dir, exist_ok=True)
    file_prefix = f"{today}_regime_crosstab_R{int(args.r)}_period{args.period}"

    # Pivot table: regime × mode -> pnl_total
    regimes_in_order = REGIME_ORDER + ["UNKNOWN", "ALL"]
    mode_tags = [m[0] for m in DEFAULT_MODES if m[0] in pivots_pnl]

    pnl_pivot = pd.DataFrame(index=regimes_in_order, columns=mode_tags, dtype=float)
    n_pivot   = pd.DataFrame(index=regimes_in_order, columns=mode_tags, dtype=float)
    wr_pivot  = pd.DataFrame(index=regimes_in_order, columns=mode_tags, dtype=float)
    for tag in mode_tags:
        for r in regimes_in_order:
            pnl_pivot.loc[r, tag] = pivots_pnl[tag].get(r, np.nan)
            n_pivot.loc[r, tag]   = pivots_n[tag].get(r, np.nan)
            wr_pivot.loc[r, tag]  = pivots_wr[tag].get(r, np.nan)

    # Drop empty regime rows
    pnl_pivot = pnl_pivot.dropna(how="all")
    n_pivot = n_pivot.dropna(how="all")
    wr_pivot = wr_pivot.dropna(how="all")

    # Best mode per regime
    pnl_pivot["BEST_MODE"] = pnl_pivot.idxmax(axis=1)
    pnl_pivot["BEST_PNL"]  = pnl_pivot.iloc[:, :-1].max(axis=1)

    # Print cross-tabs
    print("\n" + "=" * 80)
    print("PNL CROSS-TAB: $ per regime × mode")
    print("=" * 80)
    pd.set_option("display.float_format", lambda v: f"{v:+,.0f}")
    print(pnl_pivot.to_string())
    print()

    print("=" * 80)
    print("TRADE COUNT CROSS-TAB")
    print("=" * 80)
    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print(n_pivot.to_string())
    print()

    print("=" * 80)
    print("WIN RATE CROSS-TAB (%)")
    print("=" * 80)
    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(wr_pivot.to_string())
    print()

    # Save outputs
    pnl_pivot.to_csv(f"{args.out_dir}/{file_prefix}_pnl.csv")
    n_pivot.to_csv(f"{args.out_dir}/{file_prefix}_n.csv")
    wr_pivot.to_csv(f"{args.out_dir}/{file_prefix}_wr.csv")

    long_stats = pd.concat(all_stats_rows, ignore_index=True) if all_stats_rows else pd.DataFrame()
    long_stats.to_csv(f"{args.out_dir}/{file_prefix}_long.csv", index=False)

    long_side = pd.concat(all_side_stats_rows, ignore_index=True) if all_side_stats_rows else pd.DataFrame()
    long_side.to_csv(f"{args.out_dir}/{file_prefix}_side.csv", index=False)

    # Markdown report
    md_path = f"{args.out_dir}/{file_prefix}.md"
    pd.set_option("display.float_format", lambda v: f"{v:+,.0f}")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Regime-aware EDA cross-tab — ZigzagRunner R={args.r}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Atlas: `{args.atlas}` window {args.start} -> {args.end}\n")
        f.write(f"- Bars: {len(bars):,} 1m\n")
        f.write(f"- Regime labels from `DATA/ATLAS/regime_labels.csv`\n")
        f.write(f"- LinReg period: {args.period}\n")
        f.write(f"- R: {args.r} pts\n\n")

        f.write(f"## PnL ($) per regime × direction-mode\n\n")
        f.write("```\n")
        f.write(pnl_pivot.to_string())
        f.write("\n```\n\n")
        f.write(f"`BEST_MODE` = mode with highest PnL on that regime row.\n")
        f.write(f"Use this to decide regime-specific direction policy.\n\n")

        f.write(f"## Trade count per regime × direction-mode\n\n")
        f.write("```\n")
        pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
        f.write(n_pivot.to_string())
        f.write("\n```\n\n")

        f.write(f"## Win rate (%) per regime × direction-mode\n\n")
        f.write("```\n")
        pd.set_option("display.float_format", lambda v: f"{v:.1f}")
        f.write(wr_pivot.to_string())
        f.write("\n```\n\n")

        f.write(f"## Reading guide\n\n")
        f.write("- **counter**: v1.0.4 baseline (HighPivot->Short, LowPivot->Long).\n")
        f.write("- **skip_T0.5 / skip_T1.0**: counter-trend default but SKIP entry+exit if "
                  "slope opposes would-be direction by > T.\n")
        f.write("- **adaptive_T0.5 / adaptive_T1.0**: counter-trend default but FLIP to "
                  "with-slope direction when |slope| > T.\n")
        f.write("- **stay_T0.5 / stay_T1.0**: counter-trend default but DO NOT EXIT existing "
                  "position when slope still favors it (skip the would-be exit + new entry).\n\n")
        f.write("## Decision matrix\n\n")
        f.write("| Regime | Best mode | If best is counter | If best is skip/stay/adaptive |\n")
        f.write("|---|---|---|---|\n")
        f.write("| UP | (see table) | Counter survives in UP | Apply that mode in UP days |\n")
        f.write("| DOWN | (see table) | Counter survives in DOWN | Apply that mode in DOWN days |\n")
        f.write("| CHOP | counter (likely) | Counter is the sweet spot | Investigate why |\n")
        f.write("| QUIET | (see table) | Low-volume, low PnL all modes | Maybe skip entirely |\n\n")
        f.write("Next step: build a regime-aware NT8 strategy that picks the BEST_MODE per "
                  "regime detected in real-time.\n")
    print(f"\nWrote: {md_path}")
    print(f"Wrote: {file_prefix}_pnl.csv, _n.csv, _wr.csv, _long.csv, _side.csv")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
