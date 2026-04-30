"""
augment_pivots_with_levels.py -- Snap auto-detected pivots to nearest human-marked
S/R level (within tolerance). Adds 'snapped_to_level', 'snap_distance_pts',
'level_type' fields to each pivot. Pivots far from any level are kept but flagged.

Why: human-marked levels are validated S/R prices. An auto-detected pivot near a
level is more likely real (the swing made by retesting that level). A pivot
nowhere near a level may be a false positive from the algorithm.

This is a POST-HOC enhancer: takes any peaks JSON (manual or auto) + monthly
levels JSON files, produces an augmented peaks JSON.

Usage:
    python tools/augment_pivots_with_levels.py \\
        --peaks DATA/regime_seeds/auto_peaks_2025-01-02_to_2026-03-20_4h.json \\
        --tolerance 15

Output:
    DATA/regime_seeds/<peaks_basename>_augmented.json   (same format + new fields)
    reports/findings/regime_eda/<date>_pivot_augmentation_summary.md
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

LEVELS_DIR = "DATA/levels"
OUT_DIR = "reports/findings/regime_eda"


# =============================================================================
# Time-varying levels loader
# =============================================================================

def load_active_levels_at(timestamp: float, levels_dir: str = LEVELS_DIR) -> list[dict]:
    """Return the levels active at this timestamp = the most recent
    levels_YYYY-MM-DD.json whose date is <= timestamp's month start.

    This avoids lookahead: we only use levels that were known at the time.
    """
    if not os.path.isdir(levels_dir):
        return []
    target_dt = pd.Timestamp(timestamp, unit="s", tz="UTC")
    files = sorted(glob.glob(os.path.join(levels_dir, "levels_*.json")))
    # Find file with date closest to but not after target
    best_file = None
    best_dt = None
    for f in files:
        try:
            with open(f) as fh:
                data = json.load(fh)
            file_dt = pd.Timestamp(data.get("date", ""), tz="UTC")
        except Exception:
            continue
        if file_dt > target_dt:
            continue
        if best_dt is None or file_dt > best_dt:
            best_dt = file_dt
            best_file = f
    if best_file is None:
        return []
    with open(best_file) as fh:
        data = json.load(fh)
    out = []
    for lvl in data.get("levels", []):
        out.append({
            "price": float(lvl["price"]),
            "type": lvl.get("type", "unknown"),
            "src_date": data.get("date", ""),
        })
    return out


def cache_levels_by_month(levels_dir: str = LEVELS_DIR) -> dict[str, list[dict]]:
    """Pre-load all monthly level files into a dict: 'YYYY-MM' -> levels list."""
    cache = {}
    if not os.path.isdir(levels_dir):
        return cache
    for f in sorted(glob.glob(os.path.join(levels_dir, "levels_*.json"))):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue
        date_str = data.get("date", "")
        try:
            mkey = pd.Timestamp(date_str).strftime("%Y-%m")
        except Exception:
            continue
        cache[mkey] = [{
            "price": float(lvl["price"]),
            "type": lvl.get("type", "unknown"),
            "src_date": date_str,
        } for lvl in data.get("levels", [])]
    return cache


def lookup_active_levels_cached(timestamp: float, cache: dict[str, list[dict]]) -> list[dict]:
    """Cached version of load_active_levels_at — finds the most recent month
    in the cache that's <= the timestamp's month."""
    target_month = pd.Timestamp(timestamp, unit="s", tz="UTC").strftime("%Y-%m")
    sorted_months = sorted(cache.keys())
    best = None
    for m in sorted_months:
        if m <= target_month:
            best = m
        else:
            break
    return cache.get(best, []) if best else []


# =============================================================================
# Snap a pivot to nearest level
# =============================================================================

def snap_pivot(pivot: dict, levels: list[dict], tolerance_pts: float) -> dict:
    """Snap pivot's price to the nearest level if within tolerance.

    Returns a copy of pivot with these new fields:
      - snapped_to_level: bool
      - snap_distance_pts: float (None if no snap)
      - snapped_level_type: str (None if no snap)
      - snapped_level_price: float (None if no snap)
      - original_price: float (the un-snapped price)

    Match priority: nearest by absolute distance.
    Type-match preference: H pivot prefers resistance, L prefers support — but
    will fall back to opposite type if it's clearly closer.
    """
    out = dict(pivot)
    out["original_price"] = float(pivot["price"])

    if not levels:
        out.update({
            "snapped_to_level": False,
            "snap_distance_pts": None,
            "snapped_level_type": None,
            "snapped_level_price": None,
        })
        return out

    p_price = float(pivot["price"])
    snap_type = pivot.get("_snap", "")
    preferred_type = "resistance" if snap_type == "H" else (
        "support" if snap_type == "L" else None
    )

    # Find candidates within tolerance
    candidates = [(lvl, abs(p_price - lvl["price"])) for lvl in levels
                   if abs(p_price - lvl["price"]) <= tolerance_pts]
    if not candidates:
        out.update({
            "snapped_to_level": False,
            "snap_distance_pts": None,
            "snapped_level_type": None,
            "snapped_level_price": None,
        })
        return out

    # Prefer matching type within tolerance; otherwise nearest of any type
    pref = [(l, d) for l, d in candidates if l["type"] == preferred_type]
    pool = pref if pref else candidates
    pool.sort(key=lambda x: x[1])
    best_lvl, best_dist = pool[0]

    out.update({
        "snapped_to_level": True,
        "snap_distance_pts": float(best_dist),
        "snapped_level_type": best_lvl["type"],
        "snapped_level_price": float(best_lvl["price"]),
        # Also snap the actual price to the level
        "price": float(best_lvl["price"]),
    })
    return out


# =============================================================================
# Main augmentation
# =============================================================================

def augment_peaks_file(peaks_path: str, levels_dir: str, tolerance_pts: float,
                         out_path: str | None = None) -> tuple[str, dict]:
    """Load peaks JSON, snap each pivot to nearest active level, write augmented."""
    with open(peaks_path) as f:
        data = json.load(f)
    peaks = data.get("peaks", [])
    print(f"Loaded {len(peaks)} peaks from {peaks_path}")

    # Pre-cache monthly levels
    print(f"Pre-caching monthly levels from {levels_dir}...")
    levels_cache = cache_levels_by_month(levels_dir)
    print(f"  {len(levels_cache)} monthly level files cached")

    augmented = []
    n_snapped = 0
    n_no_levels = 0
    snap_distances = []
    by_month = {}  # month -> n_snapped
    type_match_count = {"correct": 0, "fallback": 0}

    for p in peaks:
        ts = float(p["timestamp"])
        active_levels = lookup_active_levels_cached(ts, levels_cache)
        if not active_levels:
            n_no_levels += 1
        aug = snap_pivot(p, active_levels, tolerance_pts)
        if aug["snapped_to_level"]:
            n_snapped += 1
            snap_distances.append(aug["snap_distance_pts"])
            mkey = pd.Timestamp(ts, unit="s", tz="UTC").strftime("%Y-%m")
            by_month[mkey] = by_month.get(mkey, 0) + 1
            # Type-match check
            preferred = "resistance" if p.get("_snap") == "H" else "support"
            if aug["snapped_level_type"] == preferred:
                type_match_count["correct"] += 1
            else:
                type_match_count["fallback"] += 1
        augmented.append(aug)

    # Output path
    if not out_path:
        base = os.path.basename(peaks_path).replace(".json", "_augmented.json")
        out_path = os.path.join(os.path.dirname(peaks_path), base)

    out_data = dict(data)
    out_data["peaks"] = augmented
    out_data["augmented"] = {
        "tolerance_pts": tolerance_pts,
        "n_total": len(peaks),
        "n_snapped": n_snapped,
        "n_no_levels_active": n_no_levels,
        "snap_rate": n_snapped / len(peaks) if peaks else 0.0,
        "median_snap_distance_pts": float(np.median(snap_distances)) if snap_distances else None,
        "p90_snap_distance_pts": float(np.percentile(snap_distances, 90)) if snap_distances else None,
        "type_match_correct": type_match_count["correct"],
        "type_match_fallback": type_match_count["fallback"],
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nWrote: {out_path}")

    summary = {
        "input_path": peaks_path,
        "output_path": out_path,
        "tolerance_pts": tolerance_pts,
        "n_peaks": len(peaks),
        "n_snapped": n_snapped,
        "n_no_levels": n_no_levels,
        "snap_rate": n_snapped / len(peaks) if peaks else 0.0,
        "median_snap_distance": float(np.median(snap_distances)) if snap_distances else 0.0,
        "p90_snap_distance": float(np.percentile(snap_distances, 90)) if snap_distances else 0.0,
        "type_match_correct": type_match_count["correct"],
        "type_match_fallback": type_match_count["fallback"],
        "by_month": by_month,
    }
    return out_path, summary


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peaks", required=True, help="Input peaks JSON path")
    ap.add_argument("--tolerance", type=float, default=15.0,
                    help="Snap tolerance in points (default 15 = ~MNQ proximity)")
    ap.add_argument("--levels-dir", default=LEVELS_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 80)
    print("AUGMENT PIVOTS WITH HUMAN LEVELS")
    print("=" * 80)
    print(f"Tolerance: +/-{args.tolerance} pts")
    print()

    out_path, summary = augment_peaks_file(args.peaks, args.levels_dir,
                                              args.tolerance, args.out)

    print(f"\n=== Summary ===")
    print(f"Total peaks: {summary['n_peaks']}")
    print(f"Snapped to a level: {summary['n_snapped']} ({100*summary['snap_rate']:.1f}%)")
    print(f"Type-match (snap_type==level_type): "
           f"{summary['type_match_correct']} correct, "
           f"{summary['type_match_fallback']} fallback (opposite type)")
    print(f"Median snap distance: {summary['median_snap_distance']:.2f} pts")
    print(f"P90 snap distance:    {summary['p90_snap_distance']:.2f} pts")
    print(f"No levels active for: {summary['n_no_levels']} peaks")
    print()

    # Markdown report
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.basename(args.peaks).replace(".json", "")
    md_path = os.path.join(OUT_DIR, f"{today}_augment_{base}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Pivot augmentation with human levels\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Inputs\n\n")
        f.write(f"- Peaks file: `{summary['input_path']}`\n")
        f.write(f"- Levels dir: `{args.levels_dir}`\n")
        f.write(f"- Tolerance: +/- {summary['tolerance_pts']} pts\n\n")
        f.write(f"## Results\n\n")
        f.write(f"- Total peaks: {summary['n_peaks']}\n")
        f.write(f"- Snapped to a level: {summary['n_snapped']} ({100*summary['snap_rate']:.1f}%)\n")
        f.write(f"- Type match (H->resistance / L->support): "
                f"{summary['type_match_correct']} correct, "
                f"{summary['type_match_fallback']} fallback to opposite type\n")
        f.write(f"- Median snap distance: {summary['median_snap_distance']:.2f} pts\n")
        f.write(f"- P90 snap distance:    {summary['p90_snap_distance']:.2f} pts\n")
        f.write(f"- Peaks with no active levels: {summary['n_no_levels']}\n")
        f.write(f"- Output file: `{summary['output_path']}`\n\n")
        f.write(f"## Snaps by month\n\n")
        f.write("| Month | N snapped |\n|---|---:|\n")
        for m in sorted(summary['by_month'].keys()):
            f.write(f"| {m} | {summary['by_month'][m]} |\n")
        f.write(f"\n## Interpretation\n\n")
        f.write("- High snap rate (>=70%) = auto-detector is finding pivots near "
                  "human-validated S/R = high quality.\n")
        f.write("- Type match correct = H pivots near resistance, L pivots near support "
                  "— behaves like S/R retests.\n")
        f.write("- Fallback type matches = a H pivot landing near support (or vice versa) "
                  "= breakthrough setup.\n")
        f.write("- High median snap distance close to tolerance = pivots are 'near' "
                  "but not exactly on levels — might want to tighten tolerance.\n\n")
        f.write(f"## Next step\n\n")
        f.write(f"Re-run macro_slope_segmenter.py to use the augmented pivots:\n")
        f.write(f"```\n")
        f.write(f"python tools/macro_slope_segmenter.py --tfs 4h\n")
        f.write(f"```\n")
        f.write(f"(Will pick up the augmented file via find_peaks_file logic if it has more peaks.)\n")
    print(f"Report: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    gc.collect()


if __name__ == "__main__":
    main()
