"""
entry_outcome_overlay_chart.py -- Time-axis visualization of fade entries with
outcome (win/loss) overlaid on price + physics features.

Mirrors tools/peak_feature_overlay_chart.py but for trade entries instead of
peaks. The mean-regression CNN thesis says: at fade triggers (|z_se| > 2 +
vr < 1), the rest of the physics distinguishes winners from losers. This
chart lets you SEE that distinction in time order.

Each entry is drawn on the price panel with:
  - Green filled circle  = winner (pnl > 0)
  - Red   filled circle  = loser  (pnl <= 0)
  - Marker size scales with |pnl| (clipped for readability)
  - Vertical dotted line drops through all 6 feature panels at entry time

Defaults to fade tiers only (BASE_NMP, FADE_CALM, FADE_AGAINST) which are
the z-band-trigger family — the original mean-regression thesis. Use
--tiers to override.

Modes:
  per_day  -- one chart per entry-bearing trading day (default)
  sample   -- specific dates only (fast iteration)

Usage:
  python tools/entry_outcome_overlay_chart.py --mode sample \\
      --days "2025-06-09,2026-02-19"
  python tools/entry_outcome_overlay_chart.py --mode per_day
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
TRADES_CSV = "reports/findings/tier_pnl_by_regime/2026-04-29_trades_enriched.csv"
FEATURES_DIR = "DATA/ATLAS/FEATURES_5s"
PRICE_DIR = "DATA/ATLAS/5m"
OUT_DIR_DEFAULT = "reports/findings/entry_outcome_overlay"

# Default fade-family tiers (z-band trigger physics — the CNN thesis target)
FADE_TIERS_DEFAULT = ["BASE_NMP", "FADE_CALM", "FADE_AGAINST"]


# -------------------------------------------------------------------------
# Trade loading
# -------------------------------------------------------------------------
def load_trades(path: str = TRADES_CSV,
                tiers: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = df["timestamp"].astype(float)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    if tiers:
        df = df[df["entry_tier"].isin(tiers)].copy()
    df["winner"] = df["pnl"] > 0
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -------------------------------------------------------------------------
# Per-day data loading
# -------------------------------------------------------------------------
def _day_path(directory: str, day: str, ext: str = "parquet") -> str:
    return os.path.join(directory, f"{day}.{ext}")


def load_day_price(day: str) -> pd.DataFrame | None:
    p = _day_path(PRICE_DIR, day)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def load_day_features(day: str) -> pd.DataFrame | None:
    p = _day_path(FEATURES_DIR, day)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df


def discover_entry_days(trades: pd.DataFrame) -> list[str]:
    days = trades["dt"].dt.strftime("%Y_%m_%d").unique().tolist()
    days.sort()
    return days


def filter_trades_for_day(trades: pd.DataFrame, day: str) -> pd.DataFrame:
    day_str = day.replace("_", "-")
    return trades[trades["dt"].dt.strftime("%Y-%m-%d") == day_str].copy()


# -------------------------------------------------------------------------
# Chart rendering
# -------------------------------------------------------------------------
PANEL_GROUPS = [
    ("z_se", "z_se (mean-revert trigger)", [
        ("5m_z_se", "5m", "C0"),
        ("1m_z_se", "1m", "C1"),
    ], ("hlines", [(-2.0, "gray", "--", 0.5),
                    (0.0, "black", "-", 0.5),
                    (2.0, "gray", "--", 0.5)])),
    ("dmi fast", "dmi_diff 1m + 5m", [
        ("5m_dmi_diff", "5m", "C0"),
        ("1m_dmi_diff", "1m", "C1"),
    ], ("hlines", [(0.0, "black", "-", 0.5)])),
    ("dmi slow", "dmi_diff 15m + 1h", [
        ("15m_dmi_diff", "15m", "C2"),
        ("1h_dmi_diff", "1h", "C3"),
    ], ("hlines", [(0.0, "black", "-", 0.5)])),
    ("var_ratio", "variance_ratio (<1 = revert, >1 = trend)", [
        ("5m_variance_ratio", "5m", "C0"),
        ("1m_variance_ratio", "1m", "C1"),
    ], ("hlines", [(1.0, "black", "-", 0.5)])),
    ("rev/hurst", "reversion_prob + hurst (1m)", [
        ("1m_reversion_prob", "rev_prob", "C4"),
        ("1m_hurst", "hurst", "C5"),
    ], ("hlines", [(0.5, "black", "-", 0.5)])),
    ("velocity", "velocity 15m (proxied)", [
        # The enriched CSV has limited velocity columns; we'll skip if absent
    ], ("hlines", [(0.0, "black", "-", 0.5)])),
]


def _marker_size(pnl: float, scale: float = 1.5,
                  cap_pnl: float = 200.0,
                  base: float = 30.0) -> float:
    """Scale marker size by |pnl|, clipped at cap."""
    mag = min(abs(pnl), cap_pnl)
    return base + scale * mag


def render_chart(
    day: str,
    price_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    trades_day: pd.DataFrame,
    out_path: str,
) -> None:
    fig = plt.figure(figsize=(15, 14))
    gs = GridSpec(
        nrows=7, ncols=1,
        height_ratios=[3, 1, 1, 1, 1, 1, 1],
        hspace=0.18,
    )
    axes = [fig.add_subplot(gs[0])]
    for i in range(1, 7):
        axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

    # ---- Panel 0: price + entries ----
    ax0 = axes[0]
    ax0.plot(price_df["dt"], price_df["close"], color="0.25",
             linewidth=1.0, label="5m close")

    # Win/loss split for legend & rendering
    winners = trades_day[trades_day["winner"]]
    losers = trades_day[~trades_day["winner"]]

    if len(winners):
        sizes = winners["pnl"].apply(_marker_size).to_numpy()
        ax0.scatter(winners["dt"], winners["entry_price"],
                    s=sizes, color="lime", edgecolors="black",
                    linewidths=0.5, alpha=0.75, zorder=10,
                    label=f"winner (n={len(winners)})")
    if len(losers):
        sizes = losers["pnl"].apply(_marker_size).to_numpy()
        ax0.scatter(losers["dt"], losers["entry_price"],
                    s=sizes, color="crimson", edgecolors="black",
                    linewidths=0.5, alpha=0.75, zorder=10,
                    label=f"loser (n={len(losers)})")

    n_total = len(trades_day)
    n_w = len(winners)
    win_pct = (100.0 * n_w / n_total) if n_total else 0.0
    pnl_sum = trades_day["pnl"].sum()
    tier_summary = ", ".join(
        sorted(trades_day["entry_tier"].unique().tolist())
    ) if n_total else "no trades"
    ax0.set_ylabel("MNQ price", fontsize=9)
    ax0.set_title(
        f"{day.replace('_', '-')} — entries (winner=lime, loser=crimson) | "
        f"n={n_total}, WR={win_pct:.0f}%, PnL=${pnl_sum:.0f} | tiers: "
        f"{tier_summary}",
        fontsize=10,
    )
    ax0.legend(loc="upper left", fontsize=8, framealpha=0.7)
    ax0.grid(alpha=0.3)

    # ---- Panels 1-6: features ----
    for i, (ylabel, title, traces, extras) in enumerate(PANEL_GROUPS, start=1):
        ax = axes[i]
        for col, label, color in traces:
            if col in feat_df.columns:
                ax.plot(feat_df["dt"], feat_df[col],
                        color=color, linewidth=0.8, label=label, alpha=0.85)
        if extras and extras[0] == "hlines":
            for y, color, ls, lw in extras[1]:
                ax.axhline(y=y, color=color, linestyle=ls, linewidth=lw)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.text(0.005, 0.93, title, transform=ax.transAxes,
                fontsize=7, color="0.4", verticalalignment="top")
        if traces:
            ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
        ax.grid(alpha=0.3)

        # Vertical guides at entry times — color by outcome
        for _, row in trades_day.iterrows():
            color = "limegreen" if row["winner"] else "crimson"
            ax.axvline(row["dt"], color=color, linestyle=":",
                       linewidth=0.5, alpha=0.4, zorder=1)

    # vertical guides on price panel too
    for _, row in trades_day.iterrows():
        color = "limegreen" if row["winner"] else "crimson"
        ax0.axvline(row["dt"], color=color, linestyle=":",
                    linewidth=0.5, alpha=0.4, zorder=1)

    # x-axis: HH:MM only on bottom panel
    for ax in axes[:-1]:
        ax.tick_params(axis="x", labelbottom=False)
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    axes[-1].set_xlabel("Time (UTC)", fontsize=9)

    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------
# Modes
# -------------------------------------------------------------------------
def run_per_day(trades: pd.DataFrame, out_dir: str,
                 max_charts: int | None = None,
                 days_filter: list[str] | None = None) -> int:
    out_subdir = os.path.join(out_dir, "per_day")
    os.makedirs(out_subdir, exist_ok=True)

    if days_filter:
        days = sorted(days_filter)
    else:
        days = discover_entry_days(trades)

    if max_charts is not None:
        days = days[:max_charts]

    n_done = 0
    n_skipped = 0
    t0 = time.time()
    for i, day in enumerate(days):
        price_df = load_day_price(day)
        if price_df is None or len(price_df) == 0:
            n_skipped += 1
            continue
        feat_df = load_day_features(day)
        if feat_df is None or len(feat_df) == 0:
            n_skipped += 1
            continue
        trades_day = filter_trades_for_day(trades, day)
        if len(trades_day) == 0:
            n_skipped += 1
            continue
        out_path = os.path.join(out_subdir, f"{day}.png")
        try:
            render_chart(day, price_df, feat_df, trades_day, out_path)
            n_done += 1
        except Exception as e:
            print(f"  RENDER FAIL {day}: {e}", flush=True)
            n_skipped += 1
            continue
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(days) - (i + 1)) / rate if rate > 0 else 0
            print(f"  [{i + 1}/{len(days)}] done={n_done} skip={n_skipped} "
                  f"rate={rate:.1f}/s eta={eta:.0f}s", flush=True)

    print(f"  per_day complete: {n_done} charts, {n_skipped} skipped, "
          f"{time.time() - t0:.1f}s")
    return n_done


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["per_day", "sample"], default="sample")
    p.add_argument("--days", default="",
                   help="CSV of YYYY-MM-DD to render (sample mode)")
    p.add_argument("--tiers", default=",".join(FADE_TIERS_DEFAULT),
                   help=f"CSV of tier names (default: {','.join(FADE_TIERS_DEFAULT)})")
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    p.add_argument("--max-charts", type=int, default=None)
    p.add_argument("--all-tiers", action="store_true",
                   help="Override --tiers and load every tier")
    args = p.parse_args()

    print("=" * 70)
    print("ENTRY OUTCOME OVERLAY CHART")
    print("=" * 70)

    tiers_filter = (None if args.all_tiers
                    else [t.strip() for t in args.tiers.split(",") if t.strip()])
    print(f"Loading trades from {TRADES_CSV}")
    if tiers_filter:
        print(f"  filter to tiers: {tiers_filter}")
    trades = load_trades(TRADES_CSV, tiers=tiers_filter)
    print(f"  loaded: {len(trades)} trades")
    print(f"  WR (count-based): "
          f"{100.0 * trades['winner'].mean():.1f}%")
    print(f"  PnL total: ${trades['pnl'].sum():.0f}")
    print(f"  per tier:"); print(trades["entry_tier"].value_counts())

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "sample":
        if not args.days:
            args.days = ("2025-06-09,2025-03-15,2025-09-22,2025-11-04,"
                         "2026-01-08,2026-02-19")
        days = [d.strip().replace("-", "_") for d in args.days.split(",")
                if d.strip()]
        if args.max_charts is None:
            args.max_charts = len(days)
        print(f"Mode: sample ({len(days)} days)")
        n = run_per_day(trades, args.out_dir, max_charts=args.max_charts,
                         days_filter=days)
        print(f"DONE: {n} charts in {args.out_dir}/per_day/")
    elif args.mode == "per_day":
        print("Mode: per_day (all entry-bearing days)")
        n = run_per_day(trades, args.out_dir, max_charts=args.max_charts)
        print(f"DONE: {n} charts in {args.out_dir}/per_day/")


if __name__ == "__main__":
    main()
