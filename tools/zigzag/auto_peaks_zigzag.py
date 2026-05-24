"""
auto_peaks_zigzag.py -- Multi-TF auto-peak detection using the canonical
ZigZag detect_swings logic from auto_swing_marker.py.

This is the RIGHT-algorithm replacement for the deleted auto_peak_detector.py.
Instead of find_pivots (window-based local max/min), uses detect_swings
(R-threshold reversal detection) which matches our trading strategy's logic.

Workflow:
  1. Load full-range OHLC for the target TF (1W, 1D, 4h, 1h, 15m).
  2. If manual peaks exist for that TF (or for a subset window like Feb 1-7),
     calibrate min_reversal + min_bars against the manual marks.
  3. Apply best params to full data.
  4. Save peaks in peak_marker JSON format (compatible with macro_slope_segmenter).

Usage:
    python tools/auto_peaks_zigzag.py --tf 1h
    python tools/auto_peaks_zigzag.py --tf 15m --tolerance 3
    python tools/auto_peaks_zigzag.py --tf 1h --no-calibrate \\
        --min-reversal 50 --min-bars 5
"""
from __future__ import annotations
import argparse
import gc
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

# Import the canonical ZigZag swing detector
from tools.auto_swing_marker import detect_swings, TICK_SIZE
from tools.peak_marker import (
    _load_full_1d, _resample_to_1w, load_macro_tf, SEEDS_DIR,
)
from tools.research.data import load_atlas_tf

OUT_DIR = "reports/findings/regime_eda"


# =============================================================================
# Manual peak file lookup
# =============================================================================

def find_manual_peaks_file(tf: str) -> str | None:
    """Find the manual peak file with the most peaks for this TF."""
    pat = os.path.join(SEEDS_DIR, f"human_peaks_*_{tf}.json")
    files = sorted(glob.glob(pat))
    if not files:
        return None

    def _peak_count(path):
        try:
            with open(path) as f:
                return len(json.load(f).get("peaks", []))
        except Exception:
            return 0

    files.sort(key=lambda p: -_peak_count(p))
    return files[0]


def load_tf_bars(atlas_root: str, tf: str) -> pd.DataFrame:
    """Load full-range OHLC for the requested TF."""
    if tf == "1W":
        return load_macro_tf(atlas_root, "1W")
    if tf == "1D":
        return load_macro_tf(atlas_root, "1D")
    return load_atlas_tf(atlas_root, tf, months=None)


# =============================================================================
# detect_swings -> peak format
# =============================================================================

def swings_to_peaks(close: np.ndarray, bars: pd.DataFrame, pivot_indices: list[int],
                     tf: str) -> list[dict]:
    """Convert detect_swings output (list of bar indices) into peak_marker format.

    Each pivot index becomes a peak with snap (H or L) determined by whether
    the pivot is a local max or min relative to its neighbors.
    """
    peaks = []
    if len(pivot_indices) < 2:
        return peaks

    # Determine type for each pivot from the swing direction
    # If pivot[i+1] > pivot[i] price-wise, then pivot[i] is a LOW (start of up-leg)
    for i, idx in enumerate(pivot_indices):
        if idx >= len(close):
            continue
        ts = float(bars.iloc[idx]["timestamp"])
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        # Determine snap: compare to neighbors
        if i == 0:
            # First pivot: use direction to next
            if i + 1 < len(pivot_indices):
                snap = "L" if close[pivot_indices[i + 1]] > close[idx] else "H"
            else:
                snap = "?"
        elif i == len(pivot_indices) - 1:
            # Last pivot: use direction from previous
            snap = "H" if close[pivot_indices[i - 1]] < close[idx] else "L"
        else:
            # Middle: pivot is a LOW if both neighbors are higher
            prev_p = close[pivot_indices[i - 1]]
            next_p = close[pivot_indices[i + 1]]
            if close[idx] < prev_p and close[idx] < next_p:
                snap = "L"
            elif close[idx] > prev_p and close[idx] > next_p:
                snap = "H"
            else:
                # Inflection — use stronger side
                snap = "L" if close[idx] < (prev_p + next_p) / 2 else "H"

        peaks.append({
            "bar_index": int(idx),
            "timestamp": ts,
            "time_utc": dt.strftime("%H:%M:%S"),
            "price": float(bars.iloc[idx]["high"] if snap == "H" else bars.iloc[idx]["low"]),
            "close": float(bars.iloc[idx]["close"]),
            "high": float(bars.iloc[idx]["high"]),
            "low": float(bars.iloc[idx]["low"]),
            "_snap": snap,
            "_direction_hint": "AUTO_ZIGZAG",
            "tf": tf,
        })
    return peaks


# =============================================================================
# Calibrate against manual marks
# =============================================================================

def match_pivots(auto_indices: list[int], manual_peaks: list[dict],
                   tolerance_bars: int = 2) -> dict:
    """Match auto pivots to manual marks (within tolerance bars). Each manual
    peak can match AT MOST ONE auto pivot (otherwise recall > 1).
    Greedy nearest-neighbor matching: process autos in order, claim closest
    available manual within tolerance.
    """
    manual_idx_list = sorted(m["bar_index"] for m in manual_peaks)
    used_manuals = set()
    matched = 0
    for ai in auto_indices:
        # Find closest unused manual within tolerance
        best_dist = tolerance_bars + 1
        best_m = None
        for mi in manual_idx_list:
            if mi in used_manuals:
                continue
            d = abs(ai - mi)
            if d <= tolerance_bars and d < best_dist:
                best_dist = d
                best_m = mi
        if best_m is not None:
            matched += 1
            used_manuals.add(best_m)
    n_manual = len(manual_peaks)
    n_auto = len(auto_indices)
    precision = matched / n_auto if n_auto > 0 else 0.0
    recall = matched / n_manual if n_manual > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "n_matched": matched,
        "n_auto": n_auto,
        "n_manual": n_manual,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calibrate(close: np.ndarray, manual_peaks: list[dict],
                 win_start: int, win_end: int,
                 min_reversal_grid: list[int],
                 min_bars_grid: list[int],
                 max_bars: int,
                 tolerance: int) -> pd.DataFrame:
    """Sweep min_reversal × min_bars, measure F1 against manual peaks."""
    rows = []
    manual_in_win = [m for m in manual_peaks
                     if win_start <= m["bar_index"] <= win_end]
    for mr in min_reversal_grid:
        for mb in min_bars_grid:
            try:
                pivots = detect_swings(close, min_reversal=mr, min_bars=mb,
                                          max_bars=max_bars)
            except Exception as e:
                rows.append({"min_reversal": mr, "min_bars": mb, "error": str(e),
                              "f1": 0.0})
                continue
            pivots_in_win = [p for p in pivots if win_start <= p <= win_end]
            m = match_pivots(pivots_in_win, manual_in_win, tolerance)
            rows.append({
                "min_reversal": mr,
                "min_bars": mb,
                "n_pivots": len(pivots_in_win),
                "n_manual": m["n_manual"],
                "n_matched": m["n_matched"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--tf", default="1h",
                    help="TF to auto-detect (1W, 1D, 4h, 1h, 15m, etc.)")
    ap.add_argument("--tolerance", type=int, default=2,
                    help="Match tolerance in bars (default 2)")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip calibration; use --min-reversal and --min-bars directly")
    ap.add_argument("--min-reversal", type=int, default=30,
                    help="ZigZag min reversal in TICKS (default 30 = 7.5pt MNQ)")
    ap.add_argument("--min-bars", type=int, default=3,
                    help="Min bars between pivots (default 3)")
    ap.add_argument("--max-bars", type=int, default=0,
                    help="Max bars per swing (0 = no cap)")
    ap.add_argument("--reversal-grid", default="20,30,50,80,120,200",
                    help="Comma-separated min_reversal candidates (in ticks)")
    ap.add_argument("--bars-grid", default="2,3,5,8",
                    help="Comma-separated min_bars candidates")
    args = ap.parse_args()

    print("=" * 80)
    print(f"AUTO PEAKS (ZigZag) - TF={args.tf}")
    print("=" * 80)

    print(f"\nLoading {args.tf} OHLC bars...")
    bars = load_tf_bars(args.atlas, args.tf)
    if bars is None or bars.empty:
        print(f"No {args.tf} data found in {args.atlas}")
        sys.exit(1)
    print(f"  Loaded {len(bars):,} {args.tf} bars")

    close = bars["close"].to_numpy(dtype=np.float64)

    # Calibration phase
    best_mr = args.min_reversal
    best_mb = args.min_bars
    cal_df = pd.DataFrame()
    manual_path = None

    if not args.no_calibrate:
        manual_path = find_manual_peaks_file(args.tf)
        if not manual_path:
            print(f"\nNo manual marks for TF={args.tf}; using defaults "
                   f"(min_reversal={args.min_reversal}, min_bars={args.min_bars})")
        else:
            with open(manual_path) as f:
                manual_data = json.load(f)
            manual_peaks = manual_data["peaks"]
            print(f"\nManual file: {manual_path}")
            print(f"Manual peaks: {len(manual_peaks)}")

            manual_idxs = [m["bar_index"] for m in manual_peaks]
            win_start = min(manual_idxs)
            win_end = max(manual_idxs)
            win_start_dt = pd.Timestamp(bars.iloc[win_start]["timestamp"], unit="s", tz="UTC")
            win_end_dt = pd.Timestamp(bars.iloc[win_end]["timestamp"], unit="s", tz="UTC")
            print(f"Manual window: bars {win_start}-{win_end}  "
                   f"({win_start_dt.date()} -> {win_end_dt.date()})")

            mr_grid = [int(x) for x in args.reversal_grid.split(",")]
            mb_grid = [int(x) for x in args.bars_grid.split(",")]
            print(f"\nCalibration sweep: "
                   f"min_reversal={mr_grid} ticks, min_bars={mb_grid}")
            cal_df = calibrate(close, manual_peaks,
                                win_start, win_end,
                                mr_grid, mb_grid,
                                args.max_bars, args.tolerance)
            print("\n=== Calibration results (sorted by F1) ===")
            cal_sorted = cal_df.sort_values("f1", ascending=False)
            print(cal_sorted.to_string(index=False))

            best_row = cal_sorted.iloc[0]
            best_mr = int(best_row["min_reversal"])
            best_mb = int(best_row["min_bars"])
            print(f"\nBest: min_reversal={best_mr} ticks, min_bars={best_mb}, "
                   f"F1={best_row['f1']:.3f}, precision={best_row['precision']:.3f}, "
                   f"recall={best_row['recall']:.3f}")

    # Apply best params to full data
    print(f"\nApplying min_reversal={best_mr} ticks ({best_mr*TICK_SIZE:.2f}pts), "
           f"min_bars={best_mb} to full {args.tf} dataset...")
    pivots = detect_swings(close, min_reversal=best_mr, min_bars=best_mb,
                              max_bars=args.max_bars)
    print(f"  Detected: {len(pivots)} pivots")

    # Convert to peak_marker format
    auto_peaks = swings_to_peaks(close, bars, pivots, args.tf)
    print(f"  Auto peaks (after H/L labeling): {len(auto_peaks)}")

    # Save
    full_start_dt = pd.Timestamp(bars.iloc[0]["timestamp"], unit="s", tz="UTC")
    full_end_dt = pd.Timestamp(bars.iloc[-1]["timestamp"], unit="s", tz="UTC")
    range_label = f"{full_start_dt.strftime('%Y-%m-%d')}_to_{full_end_dt.strftime('%Y-%m-%d')}"
    out_path = os.path.join(SEEDS_DIR, f"auto_peaks_{range_label}_{args.tf}.json")
    with open(out_path, "w") as f:
        json.dump({
            "date": range_label,
            "tf": args.tf,
            "n_peaks": len(auto_peaks),
            "method": "zigzag_detect_swings",
            "min_reversal_ticks": best_mr,
            "min_reversal_pts": best_mr * TICK_SIZE,
            "min_bars": best_mb,
            "max_bars": args.max_bars,
            "calibrated_against": manual_path,
            "calibration_f1": float(cal_df["f1"].max()) if not cal_df.empty else None,
            "peaks": auto_peaks,
        }, f, indent=2)
    print(f"\nWrote: {out_path}")

    # Markdown report
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    md_path = os.path.join(OUT_DIR, f"{today}_auto_peaks_zigzag_{args.tf}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Auto peaks (ZigZag) - TF={args.tf}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Atlas: `{args.atlas}`\n")
        f.write(f"- TF: {args.tf}\n")
        f.write(f"- Total bars: {len(bars):,}\n")
        f.write(f"- Algorithm: detect_swings from auto_swing_marker.py (ZigZag with R-threshold)\n")
        if manual_path:
            f.write(f"- Calibrated against: `{manual_path}`\n")
            f.write(f"- Tolerance: +/- {args.tolerance} bars\n")
            best_row = cal_df.sort_values("f1", ascending=False).iloc[0]
            f.write(f"- Best F1: {best_row['f1']:.3f} "
                      f"(precision={best_row['precision']:.3f}, recall={best_row['recall']:.3f})\n\n")
            f.write(f"## Calibration sweep\n\n")
            f.write("```\n")
            f.write(cal_df.sort_values("f1", ascending=False).to_string(index=False))
            f.write("\n```\n\n")
        f.write(f"## Final\n\n")
        f.write(f"- min_reversal: {best_mr} ticks ({best_mr*TICK_SIZE:.2f} pts)\n")
        f.write(f"- min_bars: {best_mb}\n")
        f.write(f"- max_bars: {args.max_bars}\n")
        f.write(f"- Auto peaks (full range): {len(auto_peaks)}\n")
        f.write(f"- Output: `{out_path}`\n\n")
        f.write(f"## Next\n\n")
        f.write(f"```\n# Re-run macro segmenter to use these auto peaks:\n")
        f.write(f"python tools/macro_slope_segmenter.py --tfs {args.tf}\n```\n")
    print(f"Wrote report: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
