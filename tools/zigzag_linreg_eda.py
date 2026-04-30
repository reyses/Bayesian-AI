"""
zigzag_linreg_eda.py -- Visual EDA tool for ZigzagRunner v1.0.4 / v1.0.6-RC /
v1.0.7-RC paired with rolling Linear Regression slope.

Reproduces the NT8-style chart layout (price + zigzag extreme + R-trigger +
trade arrows in top panel, LinReg slope in bottom panel) for arbitrary date
ranges, and answers EDA questions:

  1. Distribution of slope at entry (does direction win/lose conditional on
     prevailing slope?)
  2. Slope reversal during trade — did losers have slope flips that could
     have triggered an early exit?
  3. PnL by slope-at-entry bucket (where in slope space does the strategy
     win, where does it lose?)
  4. Hypothetical "skip if slope < -T" filter — what would PnL look like?

Output: per-day or per-window PNG charts + CSV trade ledger with slope-aware
columns + markdown summary report.

Usage:
    # Single day visualization
    python tools/zigzag_linreg_eda.py --date 2026-04-29 --period 30

    # Date range
    python tools/zigzag_linreg_eda.py --start 2026-04-28 --end 2026-04-29 --period 30

    # Whole year EDA (no PNG, just stats + CSV — way faster)
    python tools/zigzag_linreg_eda.py --start 2025-01-01 --end 2025-12-31 \\
        --no-charts --period 30

    # Custom params
    python tools/zigzag_linreg_eda.py --date 2026-04-29 --period 30 --r 45
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import time
from datetime import datetime, date, timedelta
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.dates as mdates

from tools.zigzag_genetic import (
    load_1m_bars, compute_true_range, rolling_sma_atr,
    DOLLAR_PER_POINT, COMMISSION_RT, SLIPPAGE_PTS,
)


# =============================================================================
# Linear regression slope (rolling)
# =============================================================================

def rolling_linreg_slope(closes: np.ndarray, period: int) -> np.ndarray:
    """Rolling LinReg slope over `period` bars. Slope in price units per bar.
    NaN until `period` bars filled. Replicates NT8's LinRegSlope(period)."""
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
        y_mean = y.mean()
        slope = (x_centered * (y - y_mean)).sum() / x_var
        out[i] = slope
    return out


# =============================================================================
# Trade simulator with slope capture
# =============================================================================

def simulate_with_slope(bars: pd.DataFrame, slope: np.ndarray,
                         r_points: float = 45.0,
                         max_loss_pts: float = 0.0,
                         mfe_cut_bars: int = 0,
                         mfe_cut_usd: float = 0.0,
                         trail_activate_pts: float = 0.0,
                         trail_giveback_pct: float = 0.0,
                         flip_direction: int = 0,
                         direction_mode: str = "counter",
                         continuation_threshold: float = 0.5,
                         eod_h_utc: int = 20, eod_m_utc: int = 55,
                         entry_cut_h_utc: int = 20, entry_cut_m_utc: int = 30,
                         ) -> tuple[list[dict], list[dict]]:
    """Simulate v107 (or v104 with rules disabled) and return (trades, pivots).
    Each trade dict captures slope at entry, slope at exit, slope min/max during,
    and exit reason. Each pivot dict captures the zigzag extreme and R-trigger."""

    eod_mins = eod_h_utc * 60 + eod_m_utc
    cut_mins = entry_cut_h_utc * 60 + entry_cut_m_utc

    opens = bars["open"].to_numpy(dtype=np.float64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    mins = bars["mins_of_day_utc"].to_numpy(dtype=np.int32)
    dts = bars["dt_utc"].tolist()
    n = len(closes)

    direction = 0
    extreme_price = float("nan")
    pos_dir = 0
    pos_entry_px = 0.0
    pos_entry_idx = -1
    trade_bars_held = 0
    trade_mfe_usd = 0.0
    trail_armed = False
    slope_min = float("nan")
    slope_max = float("nan")

    trades = []
    pivots = []
    extreme_history = []  # for plotting: (idx, extreme_price, direction)
    trigger_history = []  # for plotting: (idx, trigger_price)

    def close_position(exit_idx, exit_px, exit_reason):
        nonlocal pos_dir, pos_entry_px, pos_entry_idx, trade_bars_held, trade_mfe_usd, trail_armed, slope_min, slope_max
        if pos_dir == 0:
            return
        slipped_exit = exit_px - pos_dir * SLIPPAGE_PTS
        pnl_pts = pos_dir * (slipped_exit - pos_entry_px)
        pnl_g = pnl_pts * DOLLAR_PER_POINT
        pnl_n = pnl_g - COMMISSION_RT
        trades.append({
            "entry_idx":     pos_entry_idx,
            "entry_dt":      dts[pos_entry_idx],
            "entry_px":      pos_entry_px,
            "side":          "Long" if pos_dir > 0 else "Short",
            "exit_idx":      exit_idx,
            "exit_dt":       dts[exit_idx],
            "exit_px":       slipped_exit,
            "exit_reason":   exit_reason,
            "pnl_pts":       pnl_pts,
            "pnl_usd":       pnl_n,
            "mfe_usd":       trade_mfe_usd,
            "bars_held":     trade_bars_held,
            "slope_at_entry":  slope[pos_entry_idx] if pos_entry_idx < len(slope) else np.nan,
            "slope_at_exit":   slope[exit_idx] if exit_idx < len(slope) else np.nan,
            "slope_min":     slope_min,
            "slope_max":     slope_max,
        })
        pos_dir = 0
        trade_bars_held = 0
        trade_mfe_usd = 0.0
        trail_armed = False

    for i in range(n - 1):
        c = closes[i]
        next_open = opens[i + 1]
        mins_of_day = mins[i] + 1
        cur_slope = slope[i] if i < len(slope) else float("nan")

        # Track extreme + R-trigger history for chart plotting
        if not np.isnan(extreme_price):
            extreme_history.append({"idx": i, "dt": dts[i], "extreme": extreme_price, "direction": direction})
            if direction != 0:
                trigger_price = extreme_price - direction * r_points
                trigger_history.append({"idx": i, "dt": dts[i], "trigger": trigger_price})

        # EOD force-close
        if mins_of_day >= eod_mins:
            if pos_dir != 0:
                close_position(i + 1, next_open,
                                "EodExitLong" if pos_dir > 0 else "EodExitShort")
            continue

        # Update per-trade state
        if pos_dir != 0:
            trade_bars_held += 1
            unrealized_pts = pos_dir * (c - pos_entry_px)
            unrealized_usd = unrealized_pts * DOLLAR_PER_POINT
            if unrealized_usd > trade_mfe_usd:
                trade_mfe_usd = unrealized_usd
            # Track slope range during trade
            if not np.isnan(cur_slope):
                if np.isnan(slope_min) or cur_slope < slope_min:
                    slope_min = cur_slope
                if np.isnan(slope_max) or cur_slope > slope_max:
                    slope_max = cur_slope

            # Trail
            if trail_activate_pts > 0:
                activate_usd = trail_activate_pts * DOLLAR_PER_POINT
                if not trail_armed and trade_mfe_usd >= activate_usd:
                    trail_armed = True
                if trail_armed:
                    trail_threshold = trade_mfe_usd * (1.0 - trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        close_position(i + 1, next_open,
                                        "TrailExitLong" if pos_dir > 0 else "TrailExitShort")
                        continue

            # Hard SL
            if pos_dir != 0 and max_loss_pts > 0 and unrealized_pts <= -max_loss_pts:
                close_position(i + 1, next_open,
                                "HardStopLong" if pos_dir > 0 else "HardStopShort")
                continue

            # MFE-cut at bar N
            if pos_dir != 0 and mfe_cut_bars > 0 and trade_bars_held == mfe_cut_bars:
                if trade_mfe_usd <= mfe_cut_usd:
                    close_position(i + 1, next_open,
                                    "MfeCutLong" if pos_dir > 0 else "MfeCutShort")
                    continue

        if np.isnan(extreme_price):
            extreme_price = c
            continue

        # Zigzag state machine
        pivot_confirmed = False
        new_pivot_dir = 0
        if direction == 0:
            if c - extreme_price >= r_points:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = +1; extreme_price = c
            elif extreme_price - c >= r_points:
                pivot_confirmed = True; new_pivot_dir = +1
                direction = -1; extreme_price = c
        elif direction == +1:
            if c > extreme_price:
                extreme_price = c
            elif extreme_price - c >= r_points:
                pivot_confirmed = True; new_pivot_dir = +1
                direction = -1; extreme_price = c
        else:
            if c < extreme_price:
                extreme_price = c
            elif c - extreme_price >= r_points:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = +1; extreme_price = c

        if not pivot_confirmed:
            continue
        pivots.append({"idx": i, "dt": dts[i], "price": c, "pivot_dir": new_pivot_dir})

        if mins_of_day >= cut_mins:
            continue

        # Direction policy:
        #   counter  (v1.0.4 default): HighPivot->Short, LowPivot->Long
        #   trend    (always with slope): positive slope -> Long, negative -> Short
        #   adaptive (continuation FLIP): counter-trend default, but FLIP to with-slope
        #            when |slope| > continuation_threshold
        #   skip     (continuation SKIP): counter-trend, but skip if slope strongly opposes
        #            (no entry, no exit - position stays as-is or stays flat)
        #   stay     (continuation STAY): counter-trend, but DO NOT EXIT existing position
        #            if slope still favors it (skip the would-be exit + new entry; keep
        #            the existing winner running)
        slope_now = slope[i] if i < len(slope) else 0.0
        if np.isnan(slope_now):
            slope_now = 0.0

        skip_pivot = False  # set True to skip exit+entry entirely (keep current position)
        if direction_mode == "trend":
            action_side = +1 if slope_now > 0 else -1
        elif direction_mode == "adaptive":
            if abs(slope_now) > continuation_threshold:
                action_side = +1 if slope_now > 0 else -1
            else:
                action_side = -1 if new_pivot_dir == +1 else +1
        elif direction_mode == "skip":
            # Default counter-trend, but SKIP entire pivot (no exit, no entry) if
            # the would-be entry's slope opposes by > T
            if flip_direction == 0:
                tentative = -1 if new_pivot_dir == +1 else +1
            else:
                tentative = +1 if new_pivot_dir == +1 else -1
            # Would-be entry direction; if slope opposes strongly, skip
            if (tentative > 0 and slope_now < -continuation_threshold) or \
               (tentative < 0 and slope_now > +continuation_threshold):
                skip_pivot = True
                action_side = 0  # placeholder, won't be used
            else:
                action_side = tentative
        elif direction_mode == "stay":
            # Default counter-trend, but if we have a position AND slope still
            # favors that position, STAY (skip exit + new entry)
            if pos_dir > 0 and slope_now > +continuation_threshold:
                # Long with strongly positive slope: trend continues up, stay long
                skip_pivot = True
                action_side = 0
            elif pos_dir < 0 and slope_now < -continuation_threshold:
                # Short with strongly negative slope: trend continues down, stay short
                skip_pivot = True
                action_side = 0
            else:
                if flip_direction == 0:
                    action_side = -1 if new_pivot_dir == +1 else +1
                else:
                    action_side = +1 if new_pivot_dir == +1 else -1
        else:  # counter (default)
            if flip_direction == 0:
                action_side = -1 if new_pivot_dir == +1 else +1
            else:
                action_side = +1 if new_pivot_dir == +1 else -1

        if skip_pivot:
            # No exit, no entry — pivot is recorded but we ignore it for trading
            continue

        # Always exit existing first
        if pos_dir != 0:
            close_position(i + 1, next_open,
                            "PivotExitLong" if pos_dir > 0 else "PivotExitShort")

        # Open new
        slipped_entry = next_open + action_side * SLIPPAGE_PTS
        pos_dir = action_side
        pos_entry_px = slipped_entry
        pos_entry_idx = i + 1
        trade_bars_held = 0
        trade_mfe_usd = 0.0
        trail_armed = False
        slope_min = cur_slope if not np.isnan(cur_slope) else float("nan")
        slope_max = cur_slope if not np.isnan(cur_slope) else float("nan")

    # Final close
    if pos_dir != 0:
        close_position(n - 1, closes[n - 1],
                        "FinalCloseLong" if pos_dir > 0 else "FinalCloseShort")

    return trades, pivots, extreme_history, trigger_history


# =============================================================================
# Chart plotter (NT8-style overlay)
# =============================================================================

def plot_window(bars: pd.DataFrame, slope: np.ndarray,
                 trades: list[dict], pivots: list[dict],
                 extreme_history: list[dict], trigger_history: list[dict],
                 title: str, out_png: str, period: int):
    """Render a 2-panel chart: price + zigzag + trades on top, slope below."""
    fig, (ax_p, ax_s) = plt.subplots(
        2, 1, figsize=(16, 9),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}, sharex=True
    )

    # ── Top panel: price + zigzag extreme + R-trigger ──
    ax_p.plot(bars["dt_utc"], bars["close"], color="#888", lw=0.7, label="close")

    if extreme_history:
        ext_df = pd.DataFrame(extreme_history)
        ax_p.plot(ext_df["dt"], ext_df["extreme"], color="orange", lw=1.0,
                   alpha=0.9, label="zigzag extreme")
    if trigger_history:
        trg_df = pd.DataFrame(trigger_history)
        ax_p.plot(trg_df["dt"], trg_df["trigger"], color="cyan", lw=1.0,
                   alpha=0.9, label=f"R-trigger")

    # Trade markers
    for t in trades:
        edt = t["entry_dt"]; xdt = t["exit_dt"]
        ep = t["entry_px"]; xp = t["exit_px"]
        is_long = t["side"] == "Long"
        is_win = t["pnl_usd"] > 0
        # Entry arrow (up for long, down for short)
        ax_p.annotate("",
            xy=(edt, ep), xytext=(edt, ep - 8 if is_long else ep + 8),
            arrowprops=dict(arrowstyle="->", color="lime" if is_long else "magenta", lw=2),
        )
        # Exit marker
        ax_p.scatter([xdt], [xp], color="lime" if is_win else "red",
                      marker="x", s=80, zorder=5)
        # Connect entry-exit with a thin line (color = win/loss)
        ax_p.plot([edt, xdt], [ep, xp],
                   color="green" if is_win else "red",
                   alpha=0.35, lw=1.0, ls=":")
        # PnL label at exit
        ax_p.annotate(f"${t['pnl_usd']:+.0f}",
            xy=(xdt, xp), xytext=(5, 0), textcoords="offset points",
            fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.2",
                       facecolor="green" if is_win else "red",
                       edgecolor="none", alpha=0.7))

    ax_p.set_title(title, fontsize=11)
    ax_p.set_ylabel("Price", fontsize=10)
    ax_p.legend(loc="upper left", fontsize=8)
    ax_p.grid(alpha=0.3)
    ax_p.set_facecolor("#1a1a1a")

    # ── Bottom panel: LinReg slope ──
    ax_s.plot(bars["dt_utc"], slope, color="#4488ff", lw=0.9, label=f"LinReg slope({period})")
    ax_s.axhline(0, color="white", alpha=0.3, lw=0.5)
    ax_s.fill_between(bars["dt_utc"], 0, slope, where=(slope > 0),
                       color="green", alpha=0.15)
    ax_s.fill_between(bars["dt_utc"], 0, slope, where=(slope < 0),
                       color="red", alpha=0.15)
    # Mark slope at entry of each trade
    for t in trades:
        sl = t["slope_at_entry"]
        if not np.isnan(sl):
            color = "green" if t["pnl_usd"] > 0 else "red"
            ax_s.scatter([t["entry_dt"]], [sl], color=color, s=30, zorder=5,
                          edgecolors="white", linewidths=0.5)
    ax_s.set_ylabel(f"Slope({period})", fontsize=10)
    ax_s.set_xlabel("Time (UTC)", fontsize=10)
    ax_s.grid(alpha=0.3)
    ax_s.set_facecolor("#1a1a1a")
    ax_s.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))

    plt.setp(ax_p.get_xticklabels(), visible=False)
    fig.patch.set_facecolor("#0a0a0a")
    for ax in (ax_p, ax_s):
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#aaa")
        ax.yaxis.label.set_color("#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.title.set_color("#fff")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a", bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# EDA stats + report
# =============================================================================

def eda_stats(trades_df: pd.DataFrame, period: int) -> dict:
    """Conditional stats: PnL by slope-at-entry sign, magnitude buckets."""
    out = {}
    if len(trades_df) == 0:
        return out

    # Sign of slope at entry
    trades_df["slope_sign"] = np.where(trades_df["slope_at_entry"] > 0, "+",
                                        np.where(trades_df["slope_at_entry"] < 0, "-", "0"))
    by_sign = trades_df.groupby("slope_sign").agg(
        n=("pnl_usd", "count"),
        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
    )
    out["by_sign"] = by_sign

    # Sign at entry × side
    trades_df["regime"] = trades_df["slope_sign"]
    by_side_sign = trades_df.groupby(["side", "slope_sign"]).agg(
        n=("pnl_usd", "count"),
        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
    )
    out["by_side_sign"] = by_side_sign

    # Magnitude buckets
    bins = [-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf]
    labels = ["strong_neg(-1.5)", "weak_neg(-0.5)", "flat", "weak_pos(0.5)", "strong_pos(1.5)"]
    trades_df["slope_bucket"] = pd.cut(trades_df["slope_at_entry"], bins=bins, labels=labels)
    by_bucket = trades_df.groupby("slope_bucket", observed=False).agg(
        n=("pnl_usd", "count"),
        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
    )
    out["by_bucket"] = by_bucket

    # Slope reversal during trade — did slope flip sign between entry and exit?
    trades_df["slope_flipped"] = (
        np.sign(trades_df["slope_at_entry"]) != np.sign(trades_df["slope_at_exit"])
    ) & trades_df["slope_at_entry"].notna() & trades_df["slope_at_exit"].notna()
    by_flip = trades_df.groupby("slope_flipped").agg(
        n=("pnl_usd", "count"),
        wr=("pnl_usd", lambda x: (x > 0).mean() * 100),
        pnl_total=("pnl_usd", "sum"),
        pnl_mean=("pnl_usd", "mean"),
    )
    out["by_flip"] = by_flip

    # Hypothetical filters
    out["filters"] = {}
    base_pnl = trades_df["pnl_usd"].sum()
    base_n = len(trades_df)
    for thresh in [0.5, 1.0, 1.5, 2.0]:
        # Skip longs when slope < -thresh (don't long against down-trend)
        # Skip shorts when slope > +thresh (don't short against up-trend)
        keep = ~(
            ((trades_df["side"] == "Long") & (trades_df["slope_at_entry"] < -thresh))
            | ((trades_df["side"] == "Short") & (trades_df["slope_at_entry"] > thresh))
        )
        kept = trades_df[keep]
        out["filters"][f"skip_against_slope_thresh_{thresh}"] = {
            "n_kept": len(kept),
            "n_skipped": base_n - len(kept),
            "pnl_total": kept["pnl_usd"].sum(),
            "wr": (kept["pnl_usd"] > 0).mean() * 100 if len(kept) > 0 else 0.0,
            "delta_vs_base": kept["pnl_usd"].sum() - base_pnl,
        }

    # Hypothetical exit on slope reversal (early exit when slope flips against position)
    # For each trade, check if there's a slope sign flip during the trade — this proxy
    # is captured already via slope_flipped. The "early exit" PnL = unknown without
    # bar-level tracking; we report the proxy.
    out["base_pnl"] = base_pnl
    out["base_n"] = base_n
    return out


def write_report(stats: dict, args, out_md: str, n_trades: int):
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# LinReg({args.period}) EDA — ZigzagRunner R={args.r}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Window\n")
        f.write(f"- Atlas: `{args.atlas}`\n")
        if args.date:
            f.write(f"- Date: {args.date}\n")
        else:
            f.write(f"- Range: {args.start} -> {args.end}\n")
        f.write(f"- LinReg period: {args.period}\n")
        f.write(f"- R: {args.r} pts\n")
        f.write(f"- Direction: {'flip (HighPivot=Long)' if args.flip_dir else 'default (HighPivot=Short)'}\n")
        f.write(f"- Total trades: {n_trades}\n")
        f.write(f"- Total PnL: ${stats.get('base_pnl', 0):+,.2f}\n\n")

        if "by_sign" in stats:
            f.write("## PnL by slope sign at entry\n\n")
            f.write("Slope is positive (uptrend), negative (downtrend), or near zero (chop).\n\n")
            f.write("```\n")
            f.write(stats["by_sign"].to_string())
            f.write("\n```\n\n")

        if "by_side_sign" in stats:
            f.write("## PnL by side (Long/Short) × slope sign\n\n")
            f.write("Critical regime question: do longs lose more when slope is negative?\n\n")
            f.write("```\n")
            f.write(stats["by_side_sign"].to_string())
            f.write("\n```\n\n")

        if "by_bucket" in stats:
            f.write("## PnL by slope magnitude bucket\n\n")
            f.write("Where in slope space does the strategy concentrate wins/losses?\n\n")
            f.write("```\n")
            f.write(stats["by_bucket"].to_string())
            f.write("\n```\n\n")

        if "by_flip" in stats:
            f.write("## PnL by slope-flipped-during-trade\n\n")
            f.write("If slope flips sign during the trade, did the trade tend to lose?\n")
            f.write("Strong signal here = exit-on-slope-reverse rule would help.\n\n")
            f.write("```\n")
            f.write(stats["by_flip"].to_string())
            f.write("\n```\n\n")

        if "filters" in stats:
            f.write("## Hypothetical filter: skip trades against strong slope\n\n")
            f.write("'Skip long when slope < -T, skip short when slope > +T'.\n")
            f.write("Reports kept-trade PnL delta vs no-filter baseline.\n\n")
            f.write("| Threshold T | Kept | Skipped | Total PnL | WR | Delta vs base |\n")
            f.write("|---:|---:|---:|---:|---:|---:|\n")
            for k, v in stats["filters"].items():
                t = float(k.split("_")[-1])
                f.write(f"| {t} | {v['n_kept']} | {v['n_skipped']} | "
                          f"${v['pnl_total']:+,.2f} | {v['wr']:.1f}% | "
                          f"${v['delta_vs_base']:+,.2f} |\n")

        f.write("\n## Caveats\n\n")
        f.write("- LinReg slope is computed from CLOSE prices, not synchronized with intra-bar fills.\n")
        f.write("- Trade simulation uses 1m close prices; intra-bar SL fires not modeled.\n")
        f.write("- `slope_at_entry` reflects bar AT WHICH the pivot fired (one bar BEFORE entry).\n")
        f.write("- Hypothetical filter table assumes filter cleanly skips trades; ignores re-pivoting effects.\n")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--date", default=None, help="Single date YYYY-MM-DD (overrides --start/--end)")
    ap.add_argument("--start", default=None, help="Range start YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="Range end YYYY-MM-DD")
    ap.add_argument("--period", type=int, default=30, help="LinReg slope period (default 30)")
    ap.add_argument("--r", type=float, default=45.0, help="Zigzag R points (default 45 = v1.0.4 baseline)")
    ap.add_argument("--max-loss", type=float, default=0.0)
    ap.add_argument("--mfe-cut-bars", type=int, default=0)
    ap.add_argument("--mfe-cut-usd", type=float, default=0.0)
    ap.add_argument("--trail-activate", type=float, default=0.0)
    ap.add_argument("--trail-giveback", type=float, default=0.0)
    ap.add_argument("--flip-dir", action="store_true")
    ap.add_argument("--direction-mode", choices=["counter", "trend", "adaptive"], default="counter",
                    help="counter (v1.0.4 default), trend (always with slope), "
                         "adaptive (counter-trend by default but flip to with-slope "
                         "when |slope| > continuation-threshold)")
    ap.add_argument("--continuation-threshold", type=float, default=0.5,
                    help="For --direction-mode adaptive: |slope| above this = trend continues, "
                         "take with-slope trade. Default 0.5.")
    ap.add_argument("--no-charts", action="store_true",
                    help="Skip PNG generation (just stats + CSV) — faster for big windows")
    ap.add_argument("--out-dir", default="reports/findings/linreg_eda")
    args = ap.parse_args()

    # Resolve window
    if args.date:
        start_d = end_d = args.date
    else:
        start_d = args.start or "2025-01-01"
        end_d = args.end or "2026-12-31"
    print(f"Loading bars and slicing to [{start_d}, {end_d}]...")

    bars = load_1m_bars(args.atlas)
    bars = bars[
        (bars["dt_utc"] >= pd.Timestamp(start_d, tz="UTC"))
        & (bars["dt_utc"] <= pd.Timestamp(end_d, tz="UTC") + pd.Timedelta(hours=23, minutes=59))
    ].reset_index(drop=True)
    print(f"  Loaded {len(bars):,} 1m bars  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})")

    closes = bars["close"].to_numpy(dtype=np.float64)
    print(f"Computing LinReg({args.period}) slope...")
    t0 = time.time()
    slope = rolling_linreg_slope(closes, args.period)
    print(f"  Done in {time.time() - t0:.1f} sec")

    print(f"Simulating trades (R={args.r}, dir_mode={args.direction_mode}"
           f"{', T='+str(args.continuation_threshold) if args.direction_mode=='adaptive' else ''})...")
    t0 = time.time()
    trades, pivots, ext_history, trg_history = simulate_with_slope(
        bars, slope,
        r_points=args.r,
        max_loss_pts=args.max_loss,
        mfe_cut_bars=args.mfe_cut_bars,
        mfe_cut_usd=args.mfe_cut_usd,
        trail_activate_pts=args.trail_activate,
        trail_giveback_pct=args.trail_giveback,
        flip_direction=1 if args.flip_dir else 0,
        direction_mode=args.direction_mode,
        continuation_threshold=args.continuation_threshold,
    )
    print(f"  Done in {time.time() - t0:.1f} sec — {len(trades)} trades")

    if len(trades) == 0:
        print("No trades. Check params.")
        return

    trades_df = pd.DataFrame(trades)
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(args.out_dir, exist_ok=True)

    mode_tag = args.direction_mode
    if args.direction_mode == "adaptive":
        mode_tag = f"adaptive_T{args.continuation_threshold}"
    file_prefix = f"{today}_v107_linreg{args.period}_R{int(args.r)}_{mode_tag}"

    # Save trade ledger
    csv_path = f"{args.out_dir}/{file_prefix}_trades.csv"
    trades_df.to_csv(csv_path, index=False)
    print(f"Wrote: {csv_path}")

    # Compute EDA stats
    print("\nComputing EDA stats...")
    stats = eda_stats(trades_df, args.period)

    # Print headline stats
    print(f"\n=== EDA SUMMARY ===")
    print(f"Total trades: {len(trades_df)}")
    print(f"Total PnL: ${trades_df['pnl_usd'].sum():+,.2f}")
    print(f"WR: {(trades_df['pnl_usd'] > 0).mean() * 100:.1f}%")
    print()
    if "by_side_sign" in stats:
        print("PnL by Side × Slope sign at entry:")
        print(stats["by_side_sign"].to_string())
        print()
    if "filters" in stats:
        print("Hypothetical filter (skip against strong slope):")
        for k, v in stats["filters"].items():
            t = float(k.split("_")[-1])
            print(f"  T={t}: kept={v['n_kept']}, total=${v['pnl_total']:+,.0f}, "
                   f"delta=${v['delta_vs_base']:+,.0f}")

    # Markdown report
    md_path = f"{args.out_dir}/{file_prefix}_report.md"
    write_report(stats, args, md_path, len(trades_df))
    print(f"\nWrote: {md_path}")

    # Charts (one per day)
    if args.no_charts:
        print("\n--no-charts set, skipping PNG generation.")
        return

    print("\nGenerating charts...")
    bars["date"] = bars["dt_utc"].dt.date
    days = sorted(bars["date"].unique())
    print(f"  {len(days)} days to render")

    for d in days:
        day_mask = bars["date"] == d
        if day_mask.sum() < 60:  # skip thin days
            continue
        bars_day = bars[day_mask].reset_index(drop=True)
        slope_day_idx = bars[day_mask].index.to_numpy()
        slope_day = slope[slope_day_idx]
        # Filter trades/pivots/history to this day
        trades_day = [t for t in trades if t["entry_dt"].date() == d]
        pivots_day = [p for p in pivots if p["dt"].date() == d]
        ext_day = [e for e in ext_history if e["dt"].date() == d]
        trg_day = [t for t in trg_history if t["dt"].date() == d]
        title = f"{d}  R={args.r}  LinReg({args.period})  trades={len(trades_day)}"
        out_png = f"{args.out_dir}/{today}_v107_linreg{args.period}_R{int(args.r)}_{d}.png"
        plot_window(bars_day, slope_day, trades_day, pivots_day, ext_day, trg_day,
                     title, out_png, args.period)
    print(f"  Charts written to {args.out_dir}/")

    gc.collect()


if __name__ == "__main__":
    main()
