"""
peak_feature_overlay_chart.py -- Time-axis visualization of physics features
overlaid on price with peaks marked.

Companion to the archived peak_feature_overlay.py (statistical aggregates).
This tool produces the time-aligned VISUAL the user originally asked for:
  - Top panel: 5m price + peaks marked (H peaks = red ▼, L peaks = green ▲)
    with marker size scaled by source TF (1m smallest, 1W largest)
  - 6 stacked feature panels below, all sharing the same time axis,
    showing how the physics features behave around peaks

Modes:
  per_day  -- one chart per peak-bearing trading day (default)
  per_peak -- one zoomed chart per peak (±N 5s bars window)
  sample   -- specific dates only (for fast iteration on layout)

Usage:
  python tools/peak_feature_overlay_chart.py --mode sample \\
      --days "2025-06-09,2025-03-15" --max-charts 6
  python tools/peak_feature_overlay_chart.py --mode per_day
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone
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
SEEDS_DIR = "DATA/regime_seeds"
FEATURES_DIR = "DATA/ATLAS/FEATURES_5s"
PRICE_DIR = "DATA/ATLAS/5m"
OUT_DIR_DEFAULT = "reports/findings/peak_feature_overlay"

# TFs to load peaks from, ordered low→high so larger TFs paint on top
PEAK_TFS = ["1m", "15m", "1h", "4h", "1D", "1W"]

# Marker size by TF (visually weights higher-TF peaks more)
TF_MARKER_SIZE = {
    "1m": 28, "15m": 48, "1h": 70, "4h": 100, "1D": 140, "1W": 200,
}


# -------------------------------------------------------------------------
# Peak loading (adapted from archived peak_feature_overlay.py)
# -------------------------------------------------------------------------
def find_best_peaks_file(tf: str) -> str | None:
    """Find peak file for a TF. Prefer human full-range > auto > Feb 1-7,
    by max peak count.
    """
    pat = os.path.join(SEEDS_DIR, f"*_{tf}.json")
    files = sorted(glob.glob(pat))
    files = [
        f for f in files
        if "augmented" not in os.path.basename(f)
        and "merged" not in os.path.basename(f)
    ]
    if not files:
        return None

    def _count(path: str) -> int:
        try:
            with open(path) as fh:
                return len(json.load(fh).get("peaks", []))
        except Exception:
            return 0

    files.sort(key=lambda p: -_count(p))
    return files[0]


def load_all_peaks() -> pd.DataFrame:
    """Load peaks from all TFs. Returns a DataFrame with columns:
    timestamp, dt, snap (H/L), tf, price.
    """
    rows = []
    for tf in PEAK_TFS:
        path = find_best_peaks_file(tf)
        if not path:
            continue
        with open(path) as f:
            d = json.load(f)
        for p in d.get("peaks", []):
            rows.append({
                "timestamp": float(p["timestamp"]),
                "snap": p["_snap"],
                "tf": tf,
                "price": float(p.get("price", 0.0)),
            })
    if not rows:
        return pd.DataFrame(columns=["timestamp", "dt", "snap", "tf", "price"])
    df = pd.DataFrame(rows)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# -------------------------------------------------------------------------
# Per-day data loading
# -------------------------------------------------------------------------
def _day_path(directory: str, day: str, ext: str = "parquet") -> str:
    """e.g. day='2025_06_09' → 'DATA/ATLAS/.../2025_06_09.parquet'."""
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


def discover_peak_days(peaks: pd.DataFrame) -> list[str]:
    """All distinct YYYY_MM_DD that have at least one peak."""
    days = peaks["dt"].dt.strftime("%Y_%m_%d").unique().tolist()
    days.sort()
    return days


def filter_peaks_for_day(peaks: pd.DataFrame, day: str) -> pd.DataFrame:
    """Peaks whose timestamp lands on the given day (UTC)."""
    day_str = day.replace("_", "-")
    return peaks[peaks["dt"].dt.strftime("%Y-%m-%d") == day_str].copy()


# -------------------------------------------------------------------------
# Chart rendering
# -------------------------------------------------------------------------
PANEL_GROUPS = [
    # (ylabel_short, full_title, [(feature_col, label, color), ...], extras)
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
    ("velocity", "velocity 15m + 1h", [
        ("15m_velocity", "15m", "C2"),
        ("1h_velocity", "1h", "C3"),
    ], ("hlines", [(0.0, "black", "-", 0.5)])),
]


def render_chart(
    day: str,
    price_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    peaks_day: pd.DataFrame,
    out_path: str,
    title_suffix: str = "",
) -> None:
    """Render the 7-panel time-aligned chart and save PNG."""
    fig = plt.figure(figsize=(15, 14))
    gs = GridSpec(
        nrows=7,
        ncols=1,
        height_ratios=[3, 1, 1, 1, 1, 1, 1],
        hspace=0.18,
    )
    axes = [fig.add_subplot(gs[0])]
    for i in range(1, 7):
        axes.append(fig.add_subplot(gs[i], sharex=axes[0]))

    # ---- Panel 0: price + peaks ----
    ax0 = axes[0]
    ax0.plot(price_df["dt"], price_df["close"], color="0.25",
             linewidth=1.0, label="5m close")

    # peak markers, larger TFs on top so they're visible
    for tf in PEAK_TFS:
        sub = peaks_day[peaks_day["tf"] == tf]
        size = TF_MARKER_SIZE.get(tf, 50)
        h = sub[sub["snap"] == "H"]
        l = sub[sub["snap"] == "L"]
        if len(h):
            ax0.scatter(h["dt"], h["price"], marker="v", s=size,
                        color="red", edgecolors="black",
                        linewidths=0.4, alpha=0.85,
                        label=f"H ({tf})" if size >= 70 else None,
                        zorder=10)
        if len(l):
            ax0.scatter(l["dt"], l["price"], marker="^", s=size,
                        color="lime", edgecolors="black",
                        linewidths=0.4, alpha=0.85,
                        label=f"L ({tf})" if size >= 70 else None,
                        zorder=10)

    ax0.set_ylabel("MNQ price", fontsize=9)
    ax0.set_title(f"{day.replace('_', '-')}{title_suffix} — price + peaks "
                  f"+ feature overlay", fontsize=11)
    ax0.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.7)
    ax0.grid(alpha=0.3)

    # ---- Panels 1-6: features ----
    for i, (ylabel, title, traces, extras) in enumerate(PANEL_GROUPS, start=1):
        ax = axes[i]
        for col, label, color in traces:
            if col in feat_df.columns:
                ax.plot(feat_df["dt"], feat_df[col],
                        color=color, linewidth=0.8, label=label, alpha=0.85)
        # Reference horizontal lines per panel
        if extras and extras[0] == "hlines":
            for y, color, ls, lw in extras[1]:
                ax.axhline(y=y, color=color, linestyle=ls, linewidth=lw)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.text(0.005, 0.93, title, transform=ax.transAxes,
                fontsize=7, color="0.4", verticalalignment="top")
        ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
        ax.grid(alpha=0.3)

        # Drop vertical guides at peak times across every panel
        for ts in peaks_day["dt"]:
            ax.axvline(ts, color="0.7", linestyle=":", linewidth=0.4,
                       alpha=0.5, zorder=1)

    # also drop vertical guides on the price panel
    for ts in peaks_day["dt"]:
        ax0.axvline(ts, color="0.7", linestyle=":", linewidth=0.4,
                    alpha=0.5, zorder=1)

    # Clean x-axis: HH:MM format on bottom panel only
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
def run_per_day(peaks: pd.DataFrame, out_dir: str,
                 max_charts: int | None = None,
                 days_filter: list[str] | None = None) -> int:
    """Generate one chart per peak-bearing day."""
    out_subdir = os.path.join(out_dir, "per_day")
    os.makedirs(out_subdir, exist_ok=True)

    if days_filter:
        days = sorted(days_filter)
    else:
        days = discover_peak_days(peaks)

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
        peaks_day = filter_peaks_for_day(peaks, day)
        out_path = os.path.join(out_subdir, f"{day}.png")
        try:
            render_chart(day, price_df, feat_df, peaks_day, out_path)
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


def run_sample(peaks: pd.DataFrame, out_dir: str,
                days: list[str], max_charts: int) -> int:
    """Sample mode: render only the listed days (for layout iteration)."""
    days_norm = [d.replace("-", "_") for d in days]
    return run_per_day(peaks, out_dir, max_charts=max_charts,
                        days_filter=days_norm)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["per_day", "per_peak", "sample"],
                   default="sample",
                   help="render mode (default: sample)")
    p.add_argument("--days", default="",
                   help="CSV of YYYY-MM-DD to render in sample mode")
    p.add_argument("--out-dir", default=OUT_DIR_DEFAULT,
                   help=f"output directory (default: {OUT_DIR_DEFAULT})")
    p.add_argument("--max-charts", type=int, default=None,
                   help="cap on number of charts (default: no cap)")
    p.add_argument("--zoom", type=int, default=60,
                   help="per_peak: ±N 5s bars window (default 60)")
    args = p.parse_args()

    print("=" * 70)
    print("PEAK FEATURE OVERLAY CHART")
    print("=" * 70)

    print("Loading peaks across TFs...")
    peaks = load_all_peaks()
    if len(peaks) == 0:
        print("ERROR: no peaks loaded — check DATA/regime_seeds/")
        sys.exit(1)
    counts = peaks.groupby("tf").size().to_dict()
    print(f"  total peaks: {len(peaks)}")
    print(f"  per TF: {counts}")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "sample":
        if not args.days:
            # Default sample set spanning regimes
            args.days = ("2025-06-09,2025-03-15,2025-09-22,2025-11-04,"
                         "2026-01-08,2026-02-19")
        days = [d.strip() for d in args.days.split(",") if d.strip()]
        if args.max_charts is None:
            args.max_charts = len(days)
        print(f"Mode: sample ({len(days)} days)")
        n = run_sample(peaks, args.out_dir, days, args.max_charts)
        print(f"DONE: {n} charts in {args.out_dir}/per_day/")
    elif args.mode == "per_day":
        print("Mode: per_day (all peak-bearing days)")
        n = run_per_day(peaks, args.out_dir,
                         max_charts=args.max_charts)
        print(f"DONE: {n} charts in {args.out_dir}/per_day/")
    elif args.mode == "per_peak":
        print("Mode: per_peak — not yet implemented (deferred per plan)")
        sys.exit(2)


if __name__ == "__main__":
    main()
