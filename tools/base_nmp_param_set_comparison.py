"""
base_nmp_param_set_comparison.py -- Test the overfit hypothesis on three
BaseNmpRunner parameter sets:

  A. SFE_CANONICAL    : matches Python sim where +$20k/14mo was validated
  B. NT8_NMP_COMB     : user's NT8 SA template "NMP Comb.xml" (28-day fit)
  C. NT8_BEST         : user's NT8 SA template "best.xml" (28-day fit)

For each set, runs the BASE_NMP simulation across the FULL 14 months and
compares: total PnL, per-trade, trade count, win rate, max drawdown.

The hypothesis: 28-day SA-fitted parameters look great on their fit window
but underperform CANONICAL on the full 14-month dataset (window-fit).

Outputs:
  reports/findings/base_nmp_param_comparison/
    2026-04-30_results.csv
    2026-04-30_results.md
    2026-04-30_chart.png  (cumulative PnL per param set over time)
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tools.zigzag_genetic import load_1m_bars, DOLLAR_PER_POINT, COMMISSION_RT, SLIPPAGE_PTS

OUT_DIR = "reports/findings/base_nmp_param_comparison"


PARAM_SETS = {
    "A_SFE_CANONICAL": {
        "ZseEntryThreshold": 2.0,
        "VrEntryThreshold": 1.0,
        "ZseExitThreshold": 0.5,
        "VrExitThreshold": 1.0,
        "RegressionPeriod": 30,
        "VrAggregationPeriod": 5,
        "VarianceSampleWindow": 30,
        "MaxLossPoints": 25.0,
        "UseEntrySlopeFilter": True,
        "SlopeFilterPeriod": 30,
        "SlopeFilterThreshold": 0.5,
    },
    "B_NT8_NMP_COMB": {
        "ZseEntryThreshold": 1.0,
        "VrEntryThreshold": 1.0,
        "ZseExitThreshold": 0.1,
        "VrExitThreshold": 1.0,
        "RegressionPeriod": 25,
        "VrAggregationPeriod": 51,
        "VarianceSampleWindow": 30,
        "MaxLossPoints": 28.0,
        "UseEntrySlopeFilter": True,
        "SlopeFilterPeriod": 30,
        "SlopeFilterThreshold": 0.6,
    },
    "C_NT8_BEST": {
        "ZseEntryThreshold": 2.0,
        "VrEntryThreshold": 0.4,
        "ZseExitThreshold": 0.1,
        "VrExitThreshold": 3.0,
        "RegressionPeriod": 46,
        "VrAggregationPeriod": 32,
        "VarianceSampleWindow": 39,
        "MaxLossPoints": 50.0,
        "UseEntrySlopeFilter": True,
        "SlopeFilterPeriod": 30,
        "SlopeFilterThreshold": 0.6,
    },
}


# =============================================================================
# z_se and variance_ratio helpers (match the NT8 strategy's formulas)
# =============================================================================

def rolling_linreg_residuals_se(closes: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """For each bar i, fit OLS regression on the prior `period` bars and return
    (linreg_value_at_i, stddev_of_residuals_over_window).
    Vectorized via numpy where possible.
    """
    n = len(closes)
    linreg_at = np.full(n, np.nan)
    se = np.full(n, np.nan)
    if n < period:
        return linreg_at, se

    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = (x_centered ** 2).sum()

    for i in range(period - 1, n):
        y = closes[i - period + 1: i + 1]
        y_mean = y.mean()
        slope = (x_centered * (y - y_mean)).sum() / x_var
        intercept = y_mean - slope * x_mean
        # Linreg value AT bar i is the line evaluated at the latest x (= period-1)
        linreg_val = intercept + slope * (period - 1)
        linreg_at[i] = linreg_val
        # Residuals over the window
        residuals = y - (intercept + slope * x)
        se[i] = residuals.std(ddof=0)
    return linreg_at, se


def compute_variance_ratio(closes: np.ndarray, q: int, sample_window: int) -> np.ndarray:
    """VR(q) = var(q-bar return) / (q * var(1-bar return)), computed over
    sample_window history at each bar.
    """
    n = len(closes)
    vr = np.full(n, np.nan)
    if n < sample_window + q:
        return vr
    ret1 = np.diff(closes, prepend=closes[0])  # 1-bar returns
    retq = np.full(n, 0.0)
    if q < n:
        retq[q:] = closes[q:] - closes[:-q]

    # Rolling stddev manually
    for i in range(sample_window + q, n):
        v1 = np.var(ret1[i - sample_window + 1: i + 1], ddof=0)
        vq = np.var(retq[i - sample_window + 1: i + 1], ddof=0)
        if v1 > 0:
            vr[i] = vq / (q * v1)
    return vr


def rolling_linreg_slope(closes: np.ndarray, period: int) -> np.ndarray:
    n = len(closes)
    slopes = np.full(n, np.nan)
    if n < period:
        return slopes
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    x_var = (x_centered ** 2).sum()
    for i in range(period - 1, n):
        y = closes[i - period + 1: i + 1]
        ymean = y.mean()
        slopes[i] = (x_centered * (y - ymean)).sum() / x_var
    return slopes


# =============================================================================
# BASE_NMP simulator (matches NT8 strategy logic)
# =============================================================================

def simulate_base_nmp(bars: pd.DataFrame, params: dict) -> dict:
    closes = bars["close"].to_numpy(dtype=np.float64)
    opens  = bars["open"].to_numpy(dtype=np.float64)
    mins   = bars["mins_of_day_utc"].to_numpy(dtype=np.int32)
    n = len(closes)

    linreg_at, se_arr = rolling_linreg_residuals_se(closes, params["RegressionPeriod"])
    vr_arr = compute_variance_ratio(closes,
                                       params["VrAggregationPeriod"],
                                       params["VarianceSampleWindow"])
    slope_arr = rolling_linreg_slope(closes, params["SlopeFilterPeriod"])

    # State
    pos_dir = 0
    pos_entry_px = 0.0
    pos_entry_idx = -1
    trades = []

    eod_mins = 20 * 60 + 55
    cut_mins = 20 * 60 + 30

    for i in range(n - 1):
        if np.isnan(se_arr[i]) or se_arr[i] <= 0 or np.isnan(vr_arr[i]):
            continue
        z_se = (closes[i] - linreg_at[i]) / se_arr[i]
        vr   = vr_arr[i]
        slope = slope_arr[i] if not np.isnan(slope_arr[i]) else 0.0
        next_open = opens[i + 1]
        mins_of_day = mins[i] + 1

        # EOD
        if mins_of_day >= eod_mins and pos_dir != 0:
            slipped_px = next_open - pos_dir * SLIPPAGE_PTS
            pnl_pts = pos_dir * (slipped_px - pos_entry_px)
            pnl_n = pnl_pts * DOLLAR_PER_POINT - COMMISSION_RT
            trades.append({"entry_idx": pos_entry_idx, "exit_idx": i+1,
                            "entry_dt": bars.iloc[pos_entry_idx]["dt_utc"],
                            "exit_dt": bars.iloc[i+1]["dt_utc"],
                            "side": "Long" if pos_dir > 0 else "Short",
                            "entry_px": pos_entry_px, "exit_px": slipped_px,
                            "pnl_usd": pnl_n, "exit_reason": "EOD"})
            pos_dir = 0
            continue

        # Position management
        if pos_dir != 0:
            unrealized_pts = pos_dir * (closes[i] - pos_entry_px)
            # Hard SL
            if params["MaxLossPoints"] > 0 and unrealized_pts <= -params["MaxLossPoints"]:
                slipped_px = next_open - pos_dir * SLIPPAGE_PTS
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DOLLAR_PER_POINT - COMMISSION_RT
                trades.append({"entry_idx": pos_entry_idx, "exit_idx": i+1,
                                "entry_dt": bars.iloc[pos_entry_idx]["dt_utc"],
                                "exit_dt": bars.iloc[i+1]["dt_utc"],
                                "side": "Long" if pos_dir > 0 else "Short",
                                "entry_px": pos_entry_px, "exit_px": slipped_px,
                                "pnl_usd": pnl_n, "exit_reason": "HardStop"})
                pos_dir = 0
                continue
            # Mean reached
            if abs(z_se) < params["ZseExitThreshold"]:
                slipped_px = next_open - pos_dir * SLIPPAGE_PTS
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DOLLAR_PER_POINT - COMMISSION_RT
                trades.append({"entry_idx": pos_entry_idx, "exit_idx": i+1,
                                "entry_dt": bars.iloc[pos_entry_idx]["dt_utc"],
                                "exit_dt": bars.iloc[i+1]["dt_utc"],
                                "side": "Long" if pos_dir > 0 else "Short",
                                "entry_px": pos_entry_px, "exit_px": slipped_px,
                                "pnl_usd": pnl_n, "exit_reason": "MeanReached"})
                pos_dir = 0
                continue
            # Regime flip
            if vr > params["VrExitThreshold"]:
                slipped_px = next_open - pos_dir * SLIPPAGE_PTS
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DOLLAR_PER_POINT - COMMISSION_RT
                trades.append({"entry_idx": pos_entry_idx, "exit_idx": i+1,
                                "entry_dt": bars.iloc[pos_entry_idx]["dt_utc"],
                                "exit_dt": bars.iloc[i+1]["dt_utc"],
                                "side": "Long" if pos_dir > 0 else "Short",
                                "entry_px": pos_entry_px, "exit_px": slipped_px,
                                "pnl_usd": pnl_n, "exit_reason": "RegimeFlip"})
                pos_dir = 0
                continue
            continue  # hold

        # Flat — entry decision
        if mins_of_day >= cut_mins:
            continue

        if abs(z_se) > params["ZseEntryThreshold"] and vr < params["VrEntryThreshold"]:
            action_side = -1 if z_se > 0 else +1  # fade
            # Slope filter
            if params["UseEntrySlopeFilter"] and params["SlopeFilterThreshold"] > 0:
                slope_opposes = (action_side > 0 and slope < -params["SlopeFilterThreshold"]) or \
                                 (action_side < 0 and slope > +params["SlopeFilterThreshold"])
                if slope_opposes:
                    continue
            slipped_entry = next_open + action_side * SLIPPAGE_PTS
            pos_dir = action_side
            pos_entry_px = slipped_entry
            pos_entry_idx = i + 1

    # Close at last bar
    if pos_dir != 0:
        slipped_px = closes[-1]
        pnl_pts = pos_dir * (slipped_px - pos_entry_px)
        pnl_n = pnl_pts * DOLLAR_PER_POINT - COMMISSION_RT
        trades.append({"entry_idx": pos_entry_idx, "exit_idx": n-1,
                        "entry_dt": bars.iloc[pos_entry_idx]["dt_utc"],
                        "exit_dt": bars.iloc[n-1]["dt_utc"],
                        "side": "Long" if pos_dir > 0 else "Short",
                        "entry_px": pos_entry_px, "exit_px": slipped_px,
                        "pnl_usd": pnl_n, "exit_reason": "FinalClose"})

    return {"trades": trades}


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 80)
    print("BASE_NMP PARAM-SET COMPARISON (CANONICAL vs NT8 SA-OPTIMIZED)")
    print("=" * 80)

    print("\nLoading bars from DATA/ATLAS...")
    bars = load_1m_bars("DATA/ATLAS")
    print(f"  {len(bars):,} bars  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})")

    summary = []
    trades_by_set = {}

    for tag, params in PARAM_SETS.items():
        print(f"\n=== {tag} ===")
        for k, v in params.items():
            print(f"  {k} = {v}")
        result = simulate_base_nmp(bars, params)
        trades = pd.DataFrame(result["trades"])
        trades_by_set[tag] = trades

        if trades.empty:
            print("  No trades — skipping summary.")
            summary.append({"set": tag, "n_trades": 0, "total_pnl": 0,
                              "wr": 0, "per_trade": 0, "max_dd": 0})
            continue

        total_pnl = trades["pnl_usd"].sum()
        n = len(trades)
        wr = (trades["pnl_usd"] > 0).mean() * 100
        per_trade = total_pnl / n
        # Max drawdown from equity curve
        trades_sorted = trades.sort_values("entry_dt").reset_index(drop=True)
        equity = trades_sorted["pnl_usd"].cumsum()
        running_max = equity.cummax()
        dd = equity - running_max
        max_dd = dd.min()

        print(f"  Trades: {n}")
        print(f"  Total PnL: ${total_pnl:+,.0f}")
        print(f"  Win rate: {wr:.1f}%")
        print(f"  $/trade: ${per_trade:+.2f}")
        print(f"  Max DD: ${max_dd:+,.0f}")
        # Time-windowed PnL
        if not trades.empty:
            trades["entry_dt"] = pd.to_datetime(trades["entry_dt"])
            mar20 = pd.Timestamp("2026-03-20", tz="UTC")
            apr29 = pd.Timestamp("2026-04-29", tz="UTC")
            in_window = trades[(trades["entry_dt"] >= mar20) & (trades["entry_dt"] <= apr29)]
            in_window_pnl = in_window["pnl_usd"].sum()
            out_window_pnl = total_pnl - in_window_pnl
            print(f"  PnL in 28-day SA window (Mar 20 - Apr 29): ${in_window_pnl:+,.0f}  ({len(in_window)} trades)")
            print(f"  PnL outside window:                          ${out_window_pnl:+,.0f}  ({n - len(in_window)} trades)")

        summary.append({
            "set": tag,
            "n_trades": n,
            "total_pnl": total_pnl,
            "wr": wr,
            "per_trade": per_trade,
            "max_dd": max_dd,
            "in_window_pnl": in_window_pnl if not trades.empty else 0,
            "in_window_n": len(in_window) if not trades.empty else 0,
            "out_window_pnl": out_window_pnl if not trades.empty else 0,
            "out_window_n": (n - len(in_window)) if not trades.empty else 0,
        })

    df = pd.DataFrame(summary)

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{today}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    # Cumulative PnL chart
    print("\nGenerating chart...")
    fig, ax = plt.subplots(figsize=(18, 8))
    fig.patch.set_facecolor("#0a0a0a")
    colors = {"A_SFE_CANONICAL": "#22c55e", "B_NT8_NMP_COMB": "#0099ff", "C_NT8_BEST": "#f59e0b"}
    for tag, df_t in trades_by_set.items():
        if df_t.empty:
            continue
        df_t = df_t.sort_values("entry_dt").reset_index(drop=True)
        df_t["cum_pnl"] = df_t["pnl_usd"].cumsum()
        ax.plot(df_t["entry_dt"], df_t["cum_pnl"], lw=1.5,
                 color=colors.get(tag, "#888"), label=f"{tag} (${df_t['cum_pnl'].iloc[-1]:+,.0f})")
    # Shade the SA fit window
    mar20 = pd.Timestamp("2026-03-20", tz="UTC")
    apr29 = pd.Timestamp("2026-04-29", tz="UTC")
    ax.axvspan(mar20, apr29, color="#888", alpha=0.15,
                label="SA fit window (Mar 20 - Apr 29)")
    ax.axhline(0, color="white", alpha=0.3, lw=0.5)
    ax.legend(loc="upper left", facecolor="#1a1a1a", labelcolor="white")
    ax.set_title("BASE_NMP cumulative PnL by parameter set — full 14 months\n"
                  "If SA-optimized sets (B, C) underperform A on the FULL data, that's overfit",
                  color="white")
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaa")
    for sp in ax.spines.values():
        sp.set_color("#444")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", color="#aaa")
    plt.tight_layout()
    chart_path = os.path.join(OUT_DIR, f"{today}_chart.png")
    fig.savefig(chart_path, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)
    print(f"Wrote: {chart_path}")

    # Markdown
    md_path = os.path.join(OUT_DIR, f"{today}_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# BaseNmpRunner param-set comparison — overfit hypothesis test\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Hypothesis\n\n")
        f.write(f"NT8 SA optimization on a 28-day window (Mar 20 – Apr 29 2026) found two\n")
        f.write(f"parameter sets ('NMP Comb' and 'best') that beat the SFE canonical\n")
        f.write(f"defaults *on that 28-day window*. **The hypothesis is that they are\n")
        f.write(f"window-fit** — they'll underperform canonical on the full 14-month dataset.\n\n")
        f.write(f"This script tests that hypothesis by running all three param sets across\n")
        f.write(f"the full 14 months and comparing.\n\n")
        f.write(f"## Results\n\n")
        f.write("```\n"); f.write(df.to_string(index=False)); f.write("\n```\n\n")
        f.write(f"## Param-set definitions\n\n")
        for tag, p in PARAM_SETS.items():
            f.write(f"### {tag}\n\n")
            f.write("| Param | Value |\n|---|---:|\n")
            for k, v in p.items():
                f.write(f"| {k} | {v} |\n")
            f.write("\n")
        f.write(f"## Verdict criteria\n\n")
        f.write(f"- If A beats B/C on full data → SA-optimized sets are window-fit; ship A.\n")
        f.write(f"- If B/C beat A on full data → SA found genuinely better params; ship best.\n")
        f.write(f"- If similar → tie; pick A as more conservative / closer to original NMP design.\n")
    print(f"Wrote: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
