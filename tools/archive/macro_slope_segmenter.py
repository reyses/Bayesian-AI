"""
macro_slope_segmenter.py -- Ingest manual peak marks (1W / 1D / 4h) and produce
segments + slope statistics + human-levels cross-reference.

Reads:
  DATA/regime_seeds/human_peaks_*_<tf>.json    (user-marked peaks per TF)
  DATA/levels/levels_*.json                    (hand-marked S/R levels)
  DATA/ATLAS/{tf}/*.parquet                    (OHLC for slope normalization)

Computes per segment (between consecutive opposing peaks):
  - direction (UP if low->high, DOWN if high->low, FLAT if same-type pair after dedup)
  - duration_bars / duration_days
  - net_pts (end_price - start_price)
  - slope_pts_per_bar (= net_pts / duration_bars)
  - slope_pts_per_day
  - mean_atr_during_segment
  - normalized_slope (slope_pts_per_day / mean_daily_atr)
  - levels_within_band (count of human levels in [min_price, max_price] of segment)
  - levels_crossed (signed: did we break through resistance / support?)

Outputs:
  reports/findings/macro_segments/<tf>_segments.csv
  reports/findings/macro_segments/<tf>_chart.png       (price + peaks + segments + levels)
  reports/findings/macro_segments/<tf>_slope_dist.png  (histogram of normalized_slope)
  reports/findings/macro_segments/summary.md           (text report all TFs)

Usage:
    python tools/macro_slope_segmenter.py --tfs 1W 1D
    python tools/macro_slope_segmenter.py --tfs 1W 1D 4h --merge-runs
"""
from __future__ import annotations
import argparse
import gc
import glob
import json
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

from tools.peak_marker import (
    _load_full_1d, _resample_to_1w, load_macro_tf, load_levels_in_range,
    SEEDS_DIR, LEVELS_DIR,
)
from tools.augment_pivots_with_levels import cache_levels_by_month, lookup_active_levels_cached
from tools.research.data import load_atlas_tf

OUT_DIR = "reports/findings/macro_segments"

# Levels are placed with min spacing 100pt (per auto_levels.py); each level
# represents a touch ZONE of half-width 50pt by design.
ZONE_HALF_WIDTH_PTS = 50.0


# =============================================================================
# Peak file discovery
# =============================================================================

def find_peaks_file(tf: str) -> str | None:
    """Find the BEST human_peaks_*_<tf>.json for the given TF.

    Selection priority:
      1. File with the most peaks (richer signal wins).
      2. Tie-breaker: most recent mtime.

    This avoids the alphabetical-sort bug where a small Feb 2026 file was
    preferred over a much larger Jan 2025 -> Mar 2026 full-range file.
    """
    pat = os.path.join(SEEDS_DIR, f"human_peaks_*_{tf}.json")
    files = sorted(glob.glob(pat))
    if not files:
        return None

    def _peak_count(path: str) -> int:
        try:
            with open(path) as f:
                return len(json.load(f).get("peaks", []))
        except Exception:
            return 0

    # Sort by (n_peaks desc, mtime desc)
    scored = [(p, _peak_count(p), os.path.getmtime(p)) for p in files]
    scored.sort(key=lambda x: (-x[1], -x[2]))
    return scored[0][0]


# =============================================================================
# Peak loading + alternation enforcement
# =============================================================================

def load_peaks(tf: str) -> pd.DataFrame:
    """Load human-marked peaks for TF. Returns DataFrame sorted by timestamp."""
    path = find_peaks_file(tf)
    if not path:
        raise FileNotFoundError(f"No human_peaks_*_{tf}.json found in {SEEDS_DIR}")
    with open(path) as f:
        data = json.load(f)
    peaks = data["peaks"]
    df = pd.DataFrame(peaks)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["source_file"] = path
    return df


def merge_same_type_runs(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """If consecutive peaks have same _snap (HH or LL), keep the more extreme one.
    A double-top: keep the higher price. A double-bottom: keep the lower price.
    Returns clean strictly-alternating peak list."""
    if peaks_df.empty:
        return peaks_df
    out = []
    for _, row in peaks_df.iterrows():
        if not out:
            out.append(row.to_dict())
            continue
        last = out[-1]
        if row["_snap"] == last["_snap"]:
            # Same type — keep the more extreme
            if row["_snap"] == "H" and row["price"] > last["price"]:
                out[-1] = row.to_dict()  # replace with higher high
            elif row["_snap"] == "L" and row["price"] < last["price"]:
                out[-1] = row.to_dict()  # replace with lower low
            # else: keep the existing one (it was more extreme)
        else:
            out.append(row.to_dict())
    return pd.DataFrame(out).reset_index(drop=True)


# =============================================================================
# Segments
# =============================================================================

def compute_segments(peaks_df: pd.DataFrame, daily_atr_series: pd.Series | None = None) -> pd.DataFrame:
    """For each consecutive pair of peaks, build a segment.
    Returns DataFrame with: start_dt, end_dt, start_price, end_price, direction,
    duration_days, net_pts, slope_pts_per_day, mean_atr, normalized_slope.
    """
    if len(peaks_df) < 2:
        return pd.DataFrame()

    rows = []
    for i in range(len(peaks_df) - 1):
        a = peaks_df.iloc[i]
        b = peaks_df.iloc[i + 1]
        start_dt = a["dt"]; end_dt = b["dt"]
        start_p = a["price"]; end_p = b["price"]
        net_pts = end_p - start_p
        duration_days = max((end_dt - start_dt).total_seconds() / 86400.0, 0.5)
        slope_per_day = net_pts / duration_days

        # Direction: drives off the PRICE DELTA, not just the snap-pair type.
        # Same-type pairs (HH / LL) carry directional intent — the second peak
        # being higher/lower than the first is a continuation/reversal signal
        # the user encoded by parking markers on a particular side of the tail.
        # Sub-class records the structural pattern.
        price_delta = end_p - start_p
        if a["_snap"] == "L" and b["_snap"] == "H":
            direction = "UP"
            sub_pattern = "L_H"  # canonical low-then-high
        elif a["_snap"] == "H" and b["_snap"] == "L":
            direction = "DOWN"
            sub_pattern = "H_L"  # canonical high-then-low
        elif a["_snap"] == "H" and b["_snap"] == "H":
            # Double-top variant — second peak higher = continuation up,
            # lower = bearish reversal, equal = true double-top
            if price_delta > 0:
                direction = "UP"; sub_pattern = "HH_higher"     # continuation up
            elif price_delta < 0:
                direction = "DOWN"; sub_pattern = "HH_lower"    # bearish double-top
            else:
                direction = "FLAT"; sub_pattern = "HH_equal"    # true double-top
        elif a["_snap"] == "L" and b["_snap"] == "L":
            # Double-bottom variant — second low lower = continuation down,
            # higher = bullish reversal, equal = true double-bottom
            if price_delta < 0:
                direction = "DOWN"; sub_pattern = "LL_lower"    # continuation down
            elif price_delta > 0:
                direction = "UP"; sub_pattern = "LL_higher"     # bullish double-bottom
            else:
                direction = "FLAT"; sub_pattern = "LL_equal"    # true double-bottom
        else:
            direction = "FLAT"; sub_pattern = "UNKNOWN"

        # Mean daily ATR within segment window
        mean_atr = np.nan
        if daily_atr_series is not None and not daily_atr_series.empty:
            mask = (daily_atr_series.index >= start_dt) & (daily_atr_series.index <= end_dt)
            atr_in = daily_atr_series[mask]
            if not atr_in.empty:
                mean_atr = float(atr_in.mean())

        # Normalized slope: |slope_per_day| / mean_atr (dimensionless, sigma/day)
        normalized_slope = (abs(slope_per_day) / mean_atr) if mean_atr and mean_atr > 0 else np.nan
        # Keep sign for analysis convenience
        signed_normalized = (slope_per_day / mean_atr) if mean_atr and mean_atr > 0 else np.nan

        rows.append({
            "segment_idx": i,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "duration_days": duration_days,
            "start_price": start_p,
            "end_price": end_p,
            "min_price": min(start_p, end_p),
            "max_price": max(start_p, end_p),
            "net_pts": net_pts,
            "direction": direction,
            "sub_pattern": sub_pattern,
            "slope_pts_per_day": slope_per_day,
            "mean_atr": mean_atr,
            "normalized_slope_abs": normalized_slope,
            "normalized_slope_signed": signed_normalized,
            "start_snap": a["_snap"],
            "end_snap": b["_snap"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# Daily ATR series (for normalization)
# =============================================================================

def compute_daily_atr_series(atlas_root: str = "DATA/ATLAS", window: int = 20) -> pd.Series:
    """Return rolling N-day ATR (using daily range as proxy) indexed by daily UTC date."""
    df = _load_full_1d(atlas_root)
    if df.empty:
        return pd.Series(dtype=float)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["range"] = df["high"] - df["low"]
    df = df.set_index("dt").sort_index()
    atr = df["range"].rolling(window, min_periods=5).mean()
    return atr


# =============================================================================
# Levels -> Zones (touch zones, not point levels)
# =============================================================================
#
# 3-BODY-PROBLEM FRAMING (user's mental model): each level represents a
# gravity well / ceiling / floor that ATTRACTS the price particle. The level
# price is the well's CENTER; the well has a real width (~50pt half-width
# from the user's 100pt min-spacing rule). When two wells overlap they form
# a single combined gravity zone.
#
# A pivot inside a zone = price was caught in the well's pull.
# A pivot between zones = price was at "escape velocity" between wells.

def merge_levels_to_zones(levels: list[dict], half_width: float = ZONE_HALF_WIDTH_PTS) -> list[dict]:
    """Convert a list of point-levels into merged ZONES.

    Each level becomes a band [price - half_width, price + half_width].
    Overlapping bands merge into a single wider zone.

    Returns list of {'price_min', 'price_max', 'price_center', 'types' (set),
    'n_levels' (count), 'src_levels' (list of contributing levels)}.
    """
    if not levels:
        return []
    bands = sorted(
        [(lvl["price"] - half_width, lvl["price"] + half_width, lvl) for lvl in levels],
        key=lambda b: b[0],
    )
    zones = []
    cur_min, cur_max, cur_levels = bands[0][0], bands[0][1], [bands[0][2]]
    for lo, hi, lvl in bands[1:]:
        if lo <= cur_max:
            cur_max = max(cur_max, hi)
            cur_levels.append(lvl)
        else:
            zones.append({
                "price_min": cur_min,
                "price_max": cur_max,
                "price_center": (cur_min + cur_max) / 2.0,
                "types": tuple(sorted({l["type"] for l in cur_levels})),
                "n_levels": len(cur_levels),
                "src_levels": cur_levels,
            })
            cur_min, cur_max, cur_levels = lo, hi, [lvl]
    zones.append({
        "price_min": cur_min,
        "price_max": cur_max,
        "price_center": (cur_min + cur_max) / 2.0,
        "types": tuple(sorted({l["type"] for l in cur_levels})),
        "n_levels": len(cur_levels),
        "src_levels": cur_levels,
    })
    return zones


# =============================================================================
# Levels cross-reference (zone-based, time-varying)
# =============================================================================

def cross_reference_zones(seg_df: pd.DataFrame, levels_cache: dict[str, list[dict]],
                           half_width: float = ZONE_HALF_WIDTH_PTS) -> pd.DataFrame:
    """For each segment, look up the levels that were ACTIVE at the segment's
    start (per the closest-prior monthly levels file, no lookahead), merge them
    into zones, then compute:

      - n_zones_in_band: zones whose [min, max] overlap segment's [min_price, max_price]
      - n_levels_in_band: total point-levels inside segment band (pre-merge count)
      - start_in_zone:   bool -- did segment's START price land in any active zone?
      - end_in_zone:     bool -- did segment's END price land in any active zone?
      - escape_velocity: bool -- start in zone AND end NOT in zone (broke out)
      - captured: bool -- start NOT in zone AND end in zone (got pulled in)
      - between_zones: bool -- both start and end in voids (escape velocity throughout)
    """
    if seg_df.empty:
        return seg_df

    out = seg_df.copy()
    n_zones = []; n_levels = []
    start_in = []; end_in = []
    escape = []; captured = []; between = []
    n_zones_crossed_in_path = []
    active_month = []

    for _, s in out.iterrows():
        # Time-varying lookup: levels active at the segment's start
        active = lookup_active_levels_cached(s["start_dt"].timestamp(), levels_cache)
        active_month.append(
            pd.Timestamp(active[0]["src_date"], tz="UTC").strftime("%Y-%m")
            if active else "NONE"
        )
        zones = merge_levels_to_zones(active, half_width)

        lo = float(s["min_price"]); hi = float(s["max_price"])
        sp = float(s["start_price"]); ep = float(s["end_price"])

        # Zones overlapping segment band
        zones_in_band = [z for z in zones if (z["price_min"] <= hi and z["price_max"] >= lo)]
        n_zones.append(len(zones_in_band))
        n_levels.append(sum(z["n_levels"] for z in zones_in_band))

        # Start / end inside any zone?
        s_in = any(z["price_min"] <= sp <= z["price_max"] for z in zones)
        e_in = any(z["price_min"] <= ep <= z["price_max"] for z in zones)
        start_in.append(s_in); end_in.append(e_in)

        # Behavioral classification (3-body framing)
        escape.append(s_in and not e_in)
        captured.append(not s_in and e_in)
        between.append(not s_in and not e_in)

        # Number of zones the segment traversed (for path complexity)
        n_zones_crossed_in_path.append(sum(
            1 for z in zones
            if (z["price_min"] >= lo and z["price_max"] <= hi)
        ))

    out["n_zones_in_band"] = n_zones
    out["n_levels_in_band"] = n_levels  # raw level count, kept for reference
    out["n_zones_traversed"] = n_zones_crossed_in_path
    out["start_in_zone"] = start_in
    out["end_in_zone"] = end_in
    out["escape_velocity"] = escape       # started in well, ended outside
    out["captured"] = captured            # started outside, ended in well
    out["between_zones"] = between        # both start and end in void
    out["levels_month"] = active_month
    return out


# =============================================================================
# Plots
# =============================================================================

REGIME_COLORS = {"UP": "#22c55e", "DOWN": "#ef4444", "FLAT": "#94a3b8"}


def plot_segments(seg_df: pd.DataFrame, peaks_df: pd.DataFrame, ohlc_df: pd.DataFrame,
                    levels_cache: dict[str, list[dict]], tf: str, out_png: str):
    """Plot the price line, peaks, segment color bands, and human levels."""
    fig, ax = plt.subplots(figsize=(20, 9))
    fig.patch.set_facecolor("#0a0a0a")

    # OHLC line (close)
    if not ohlc_df.empty:
        dt_ohlc = pd.to_datetime(ohlc_df["timestamp"], unit="s", utc=True)
        ax.plot(dt_ohlc, ohlc_df["close"], color="#aaa", lw=0.8, alpha=0.7, label="close")

    # Segment color bands (vertical span between start_dt and end_dt)
    for _, s in seg_df.iterrows():
        ax.axvspan(s["start_dt"], s["end_dt"],
                    color=REGIME_COLORS.get(s["direction"], "#666"), alpha=0.12, zorder=0)

    # Levels as horizontal lines (use union across all monthly cache for chart purposes)
    all_levels_seen = []
    seen_prices = set()
    for month_levels in levels_cache.values():
        for lvl in month_levels:
            if lvl["price"] not in seen_prices:
                all_levels_seen.append(lvl); seen_prices.add(lvl["price"])
    if not ohlc_df.empty:
        y_min = ohlc_df["close"].min(); y_max = ohlc_df["close"].max()
        for lvl in all_levels_seen:
            if not (y_min <= lvl["price"] <= y_max):
                continue
            color = "#CC0000" if lvl["type"] == "resistance" else "#0066CC"
            ax.axhline(lvl["price"], color=color, lw=0.5, alpha=0.3, ls="--", zorder=1)

    # Peak markers
    for _, p in peaks_df.iterrows():
        c = "#0099ff" if p["_snap"] == "H" else "#ffaa00"
        ax.scatter(p["dt"], p["price"], color=c, s=80, marker="D",
                    edgecolors="white", linewidths=0.5, zorder=10)

    # Connecting lines between peaks
    if len(peaks_df) >= 2:
        ax.plot(peaks_df["dt"], peaks_df["price"], color="white", alpha=0.4, lw=1.0, zorder=5)

    n_up = (seg_df["direction"] == "UP").sum() if not seg_df.empty else 0
    n_dn = (seg_df["direction"] == "DOWN").sum() if not seg_df.empty else 0
    n_fl = (seg_df["direction"] == "FLAT").sum() if not seg_df.empty else 0
    ax.set_title(f"{tf}: {len(peaks_df)} peaks, {len(seg_df)} segments  "
                  f"({n_up} UP / {n_dn} DOWN / {n_fl} FLAT) | {len(all_levels_seen)} unique levels",
                  color="white", fontsize=12)
    ax.set_facecolor("#1a1a1a")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.tick_params(colors="#aaa")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", color="#aaa")
    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


def plot_slope_distribution(seg_df: pd.DataFrame, tf: str, out_png: str):
    """Histogram of slope_pts_per_day and normalized_slope_signed."""
    if seg_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0a0a0a")

    for ax, col, title, xlabel in [
        (axes[0], "slope_pts_per_day", "Raw slope (pts/day)", "pts/day"),
        (axes[1], "normalized_slope_signed", "Normalized slope (slope/daily_ATR)", "sigma/day"),
    ]:
        data = seg_df[col].dropna()
        if data.empty:
            continue
        ax.hist(data, bins=20, color="#4488ff", edgecolor="white", linewidth=0.5, alpha=0.8)
        ax.axvline(0, color="white", alpha=0.4, lw=0.5)
        ax.set_title(f"{tf}: {title}  (n={len(data)})", color="white")
        ax.set_xlabel(xlabel, color="#ccc")
        ax.set_ylabel("count", color="#ccc")
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ax.tick_params(colors="#aaa")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def process_tf(tf: str, daily_atr: pd.Series, levels_cache: dict,
                merge_runs: bool, atlas_root: str, half_width: float) -> dict:
    print(f"\n=== Processing {tf} ===")
    peaks_df = load_peaks(tf)
    print(f"  Loaded {len(peaks_df)} peaks from {peaks_df.attrs['source_file']}")

    if merge_runs:
        before = len(peaks_df)
        peaks_df = merge_same_type_runs(peaks_df)
        if len(peaks_df) != before:
            print(f"  Merged same-type runs: {before} -> {len(peaks_df)} peaks")

    # OHLC for plotting
    if tf == "1W":
        ohlc_df = load_macro_tf(atlas_root, "1W")
    elif tf == "1D":
        ohlc_df = load_macro_tf(atlas_root, "1D")
    else:
        ohlc_df = load_atlas_tf(atlas_root, tf, months=None)

    seg_df = compute_segments(peaks_df, daily_atr_series=daily_atr)
    print(f"  Built {len(seg_df)} segments")

    # Zone-based + time-varying cross-reference (replaces flat all-levels lookup)
    seg_df = cross_reference_zones(seg_df, levels_cache, half_width=half_width)

    # Stats
    if not seg_df.empty:
        print(f"  Direction breakdown: "
               f"UP={int((seg_df['direction']=='UP').sum())}, "
               f"DOWN={int((seg_df['direction']=='DOWN').sum())}, "
               f"FLAT={int((seg_df['direction']=='FLAT').sum())}")
        print(f"  Mean duration_days: {seg_df['duration_days'].mean():.1f}")
        print(f"  Median |slope_pts_per_day|: "
               f"{seg_df['slope_pts_per_day'].abs().median():.2f}")
        norm_med = seg_df['normalized_slope_abs'].dropna().median()
        if not pd.isna(norm_med):
            print(f"  Median |normalized_slope|: {norm_med:.2f} sigma/day")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{tf}_segments.csv")
    seg_df.to_csv(csv_path, index=False)
    print(f"  Wrote: {csv_path}")

    chart_path = os.path.join(OUT_DIR, f"{tf}_chart.png")
    plot_segments(seg_df, peaks_df, ohlc_df, levels_cache, tf, chart_path)
    print(f"  Wrote: {chart_path}")

    slope_path = os.path.join(OUT_DIR, f"{tf}_slope_dist.png")
    plot_slope_distribution(seg_df, tf, slope_path)
    print(f"  Wrote: {slope_path}")

    return {
        "tf": tf,
        "n_peaks": len(peaks_df),
        "n_segments": len(seg_df),
        "n_up": int((seg_df["direction"] == "UP").sum()) if not seg_df.empty else 0,
        "n_down": int((seg_df["direction"] == "DOWN").sum()) if not seg_df.empty else 0,
        "n_flat": int((seg_df["direction"] == "FLAT").sum()) if not seg_df.empty else 0,
        "mean_duration_days": float(seg_df["duration_days"].mean()) if not seg_df.empty else 0.0,
        "median_abs_slope_per_day": float(seg_df["slope_pts_per_day"].abs().median()) if not seg_df.empty else 0.0,
        "median_normalized_slope": float(seg_df["normalized_slope_abs"].dropna().median())
            if not seg_df.empty and not seg_df["normalized_slope_abs"].dropna().empty else 0.0,
        "csv_path": csv_path,
        "chart_path": chart_path,
        "slope_path": slope_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--tfs", nargs="+", default=["1W", "1D"],
                    help="TFs to process (1W, 1D, 4h available; 1h/15m/etc if marked)")
    ap.add_argument("--merge-runs", action="store_true",
                    help="Merge consecutive same-type peaks (HH/LL) into one extreme")
    ap.add_argument("--zone-half-width", type=float, default=ZONE_HALF_WIDTH_PTS,
                    help=f"Zone half-width in points (default {ZONE_HALF_WIDTH_PTS} = 100pt min spacing rule)")
    args = ap.parse_args()

    print("=" * 80)
    print("MACRO SLOPE SEGMENTER")
    print("=" * 80)

    print("Computing daily ATR series for normalization...")
    daily_atr = compute_daily_atr_series(args.atlas, window=20)
    print(f"  Daily ATR: {len(daily_atr)} bars, "
           f"mean={daily_atr.dropna().mean():.1f}pts")

    print(f"Pre-caching monthly level files (zones with half-width={args.zone_half_width}pt)...")
    levels_cache = cache_levels_by_month(LEVELS_DIR)
    print(f"  {len(levels_cache)} monthly level files cached")
    total_levels = sum(len(v) for v in levels_cache.values())
    n_res_total = sum(1 for v in levels_cache.values() for l in v if l["type"] == "resistance")
    n_sup_total = sum(1 for v in levels_cache.values() for l in v if l["type"] == "support")
    print(f"  Total point-levels across all months: {total_levels} "
           f"({n_res_total} resistance, {n_sup_total} support)")

    summaries = []
    for tf in args.tfs:
        try:
            res = process_tf(tf, daily_atr, levels_cache, args.merge_runs,
                              args.atlas, args.zone_half_width)
            summaries.append(res)
        except FileNotFoundError as e:
            print(f"  SKIP {tf}: {e}")
        except Exception as e:
            print(f"  ERROR {tf}: {e}")
            import traceback; traceback.print_exc()

    # Summary markdown
    today = datetime.now().strftime("%Y-%m-%d")
    md_path = os.path.join(OUT_DIR, f"{today}_summary.md")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Macro slope segmenter — multi-TF\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Atlas: `{args.atlas}`\n")
        f.write(f"- TFs processed: {', '.join(s['tf'] for s in summaries)}\n")
        f.write(f"- Merge same-type runs: {args.merge_runs}\n")
        f.write(f"- Zone half-width: {args.zone_half_width} pts (per 100pt min-spacing rule)\n")
        f.write(f"- Total point-levels (across {len(levels_cache)} monthly files): "
                f"{total_levels} ({n_res_total} resistance, {n_sup_total} support)\n\n")
        f.write(f"## Per-TF summary\n\n")
        f.write("| TF | Peaks | Segments | UP | DOWN | FLAT | Mean Dur (days) | Median |slope| pts/day | Median normalized sigma/day |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            f.write(f"| {s['tf']} | {s['n_peaks']} | {s['n_segments']} | "
                      f"{s['n_up']} | {s['n_down']} | {s['n_flat']} | "
                      f"{s['mean_duration_days']:.1f} | "
                      f"{s['median_abs_slope_per_day']:.2f} | "
                      f"{s['median_normalized_slope']:.2f} |\n")
        f.write(f"\n## Outputs per TF\n\n")
        for s in summaries:
            f.write(f"### {s['tf']}\n\n")
            f.write(f"- Segments CSV: `{s['csv_path']}`\n")
            f.write(f"- Chart: `{s['chart_path']}`\n")
            f.write(f"- Slope distribution: `{s['slope_path']}`\n\n")
        f.write(f"## How to interpret\n\n")
        f.write("- Each segment = the leg between two consecutive opposing peaks (L→H = UP, H→L = DOWN).\n")
        f.write("- `slope_pts_per_day` = raw rate of change. Sign indicates direction.\n")
        f.write("- `normalized_slope` = `|slope_pts_per_day| / daily_ATR`. Dimensionless intensity in sigma/day.\n")
        f.write("  Values > 0.5 = strong drift relative to vol. < 0.1 = essentially flat.\n")
        f.write("- `n_levels_in_band` = how many human S/R levels lie within the segment's price band.\n")
        f.write("- `breaks_resistance_up` (UP segments) / `breaks_support_down` (DOWN segments) = "
                  "did the segment's end price exceed key levels = breakout signal.\n\n")
        f.write("## Next steps\n\n")
        f.write("1. Eyeball each `<tf>_chart.png` to verify segments match intuition.\n")
        f.write("2. Look at `<tf>_slope_dist.png` histograms — natural threshold candidates?\n")
        f.write("3. Decide TREND-vs-FLAT cut on normalized slope (probably ~0.3 sigma/day).\n")
        f.write("4. Cross-TF analysis: do 1D and 4h segments nest cleanly inside 1W segments?\n")
    print(f"\nWrote summary: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
