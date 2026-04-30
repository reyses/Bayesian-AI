"""
cross_tf_nesting.py -- Validate that lower-TF peaks NEST inside higher-TF
segments. Two modes:

  --mode feb_1_7   (default): use the high-resolution Feb 1-7 OOS week
                              with 4h/1h/15m/1m manual peaks.

  --mode full_14mo: use full 14-month coverage across 5 TFs:
                    1W (manual) > 1D (manual) > 4h (auto) > 1h (auto) > 15m (auto)
                    where auto peaks were calibrated against partial-week manual
                    ground truth via tools/auto_peaks_zigzag.py.

Tests:
  - Direction agreement: parent direction confirmed by child net move?
  - Child peaks per parent segment (mean / median / max)
  - UP/DOWN asymmetry: are UP-legs more sub-structured than DOWN-legs?
  - Empty parents: parent segments with zero child peaks

Usage:
    python tools/cross_tf_nesting.py --mode full_14mo
    python tools/cross_tf_nesting.py --mode feb_1_7
"""
from __future__ import annotations
import argparse
import gc
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

SEEDS_DIR = "DATA/regime_seeds"
OUT_DIR = "reports/findings/regime_eda"

# Feb 1-7 OOS week files (multi-resolution ground truth)
FEB_1_7_FILES = {
    "4h":  "human_peaks_2026-02-01_to_2026-02-07_4h.json",
    "1h":  "human_peaks_2026-02-01_to_2026-02-07_1h.json",
    "15m": "human_peaks_2026-02-01_to_2026-02-07_15m.json",
    "1m":  "human_peaks_2026-02-01_to_2026-02-07_1m.json",
}


def find_full_range_file(tf: str) -> str | None:
    """Find the file with the most peaks for a TF. Prefers files spanning
    the broadest date range (auto-detected via filename '_to_' substring).
    Used for full-14-month mode."""
    pat = os.path.join(SEEDS_DIR, f"*_{tf}.json")
    files = sorted(glob.glob(pat))
    # Skip Feb 1-7 partial files
    files = [f for f in files
             if "2026-02-01_to_2026-02-07" not in os.path.basename(f)
             and "2026-02-05" not in os.path.basename(f)
             and "augmented" not in os.path.basename(f)
             and "merged" not in os.path.basename(f)]
    if not files:
        return None
    def _count(p):
        try:
            with open(p) as fh:
                return len(json.load(fh).get("peaks", []))
        except Exception:
            return 0
    files.sort(key=lambda p: -_count(p))
    return files[0]


def get_files_for_mode(mode: str) -> dict:
    """Return {tf: filepath} for the requested mode."""
    if mode == "feb_1_7":
        return {tf: os.path.join(SEEDS_DIR, fname)
                for tf, fname in FEB_1_7_FILES.items()}
    elif mode == "full_14mo":
        out = {}
        for tf in ["1W", "1D", "4h", "1h", "15m"]:
            f = find_full_range_file(tf)
            if f:
                out[tf] = f
        return out
    else:
        raise ValueError(f"Unknown mode: {mode}")


import glob  # imported here to keep top-level imports tidy


def load_peaks(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    with open(path) as f:
        d = json.load(f)
    df = pd.DataFrame(d.get("peaks", []))
    if df.empty:
        return df
    df["dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("dt").reset_index(drop=True)
    return df


def build_segments(peaks_df: pd.DataFrame) -> pd.DataFrame:
    """Each consecutive pair of peaks = a segment."""
    if len(peaks_df) < 2:
        return pd.DataFrame()
    rows = []
    for i in range(len(peaks_df) - 1):
        a = peaks_df.iloc[i]
        b = peaks_df.iloc[i + 1]
        dur = (b["dt"] - a["dt"]).total_seconds() / 3600.0
        net = b["price"] - a["price"]
        if a["_snap"] == "L" and b["_snap"] == "H":
            direction = "UP"
        elif a["_snap"] == "H" and b["_snap"] == "L":
            direction = "DOWN"
        elif b["price"] > a["price"]:
            direction = "UP"
        elif b["price"] < a["price"]:
            direction = "DOWN"
        else:
            direction = "FLAT"
        rows.append({
            "seg_idx": i,
            "start_dt": a["dt"],
            "end_dt": b["dt"],
            "start_price": a["price"],
            "end_price": b["price"],
            "duration_hr": dur,
            "net_pts": net,
            "direction": direction,
        })
    return pd.DataFrame(rows)


def count_nested_peaks(parent_seg: pd.Series, child_peaks: pd.DataFrame) -> dict:
    """Count child peaks inside parent segment's time window."""
    s = parent_seg["start_dt"]; e = parent_seg["end_dt"]
    in_win = child_peaks[(child_peaks["dt"] >= s) & (child_peaks["dt"] <= e)]
    return {
        "n_total": len(in_win),
        "n_high": (in_win["_snap"] == "H").sum(),
        "n_low":  (in_win["_snap"] == "L").sum(),
    }


def run_nesting_analysis(mode: str = "feb_1_7"):
    title = "Feb 1-7 OOS week" if mode == "feb_1_7" else "Full 14 months"
    print("=" * 80)
    print(f"CROSS-TF NESTING ANALYSIS - {title}")
    print("=" * 80)

    files_by_tf = get_files_for_mode(mode)
    if not files_by_tf:
        print(f"No peak files found for mode={mode}")
        return

    # Load all TFs
    peaks_by_tf = {}
    for tf, path in files_by_tf.items():
        df = load_peaks(path)
        peaks_by_tf[tf] = df
        print(f"  {tf}: {len(df)} peaks  ({path})")

    # Build segments for each
    segs_by_tf = {tf: build_segments(df) for tf, df in peaks_by_tf.items()}
    for tf, sdf in segs_by_tf.items():
        print(f"  {tf}: {len(sdf)} segments")
    print()

    # Nesting parent-child pairs depend on mode
    if mode == "feb_1_7":
        parent_child_pairs = [("4h", "1h"), ("4h", "15m"), ("4h", "1m"),
                                ("1h", "15m"), ("1h", "1m"), ("15m", "1m")]
    else:  # full_14mo
        parent_child_pairs = [
            ("1W", "1D"), ("1W", "4h"), ("1W", "1h"), ("1W", "15m"),
            ("1D", "4h"), ("1D", "1h"), ("1D", "15m"),
            ("4h", "1h"), ("4h", "15m"),
            ("1h", "15m"),
        ]

    results_summary = []
    for parent_tf, child_tf in parent_child_pairs:
        parent_segs = segs_by_tf[parent_tf]
        child_peaks = peaks_by_tf[child_tf]
        if parent_segs.empty or child_peaks.empty:
            continue
        print(f"\n=== {parent_tf} segments contain {child_tf} peaks ===")
        rows = []
        for _, ps in parent_segs.iterrows():
            counts = count_nested_peaks(ps, child_peaks)
            rows.append({
                "parent_dir": ps["direction"],
                "parent_dur_hr": ps["duration_hr"],
                "parent_net_pts": ps["net_pts"],
                **counts,
            })
        d = pd.DataFrame(rows)
        agg = {
            "parent_segments": len(d),
            "child_peaks_total": d["n_total"].sum(),
            "child_per_parent_mean": d["n_total"].mean(),
            "child_per_parent_median": d["n_total"].median(),
            "child_per_parent_max": d["n_total"].max(),
            "n_empty_parents": (d["n_total"] == 0).sum(),
        }
        print(f"  Parent {parent_tf}: {agg['parent_segments']} segments")
        print(f"  Child {child_tf}: {len(child_peaks)} peaks total, "
               f"{agg['child_peaks_total']} land within parent windows")
        print(f"  Per parent: mean={agg['child_per_parent_mean']:.1f}, "
               f"median={agg['child_per_parent_median']:.0f}, "
               f"max={agg['child_per_parent_max']}, "
               f"empty={agg['n_empty_parents']}")
        # Direction-conditional nesting
        for dirn in ["UP", "DOWN"]:
            sub = d[d["parent_dir"] == dirn]
            if not sub.empty:
                print(f"    {dirn}: n={len(sub)}, mean child peaks={sub['n_total'].mean():.1f}")
        results_summary.append({
            "parent_tf": parent_tf,
            "child_tf": child_tf,
            **agg,
        })

    # Direction agreement: do child peaks alternate within parent segment?
    # E.g., a 4h UP segment should mostly contain L->H sequences in 1h
    print("\n=== Direction agreement (parent direction vs child net) ===")
    direction_agreements = []
    for parent_tf, child_tf in parent_child_pairs:
        parent_segs = segs_by_tf[parent_tf]
        child_peaks = peaks_by_tf[child_tf]
        if parent_segs.empty or child_peaks.empty:
            continue
        agree = 0; total = 0
        for _, ps in parent_segs.iterrows():
            in_win = child_peaks[(child_peaks["dt"] >= ps["start_dt"]) &
                                    (child_peaks["dt"] <= ps["end_dt"])]
            if len(in_win) < 2:
                continue
            child_net = in_win["price"].iloc[-1] - in_win["price"].iloc[0]
            child_dir = "UP" if child_net > 0 else ("DOWN" if child_net < 0 else "FLAT")
            total += 1
            if child_dir == ps["direction"]:
                agree += 1
        if total > 0:
            pct = 100 * agree / total
            print(f"  {parent_tf}->{child_tf}: {agree}/{total} = {pct:.0f}%")
            direction_agreements.append({
                "parent_tf": parent_tf, "child_tf": child_tf,
                "n_segments": total, "n_agree": agree, "pct_agree": pct,
            })

    # Save results CSV
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(OUT_DIR, exist_ok=True)
    mode_tag = mode  # 'feb_1_7' or 'full_14mo'
    out_csv = os.path.join(OUT_DIR, f"{today}_cross_tf_nesting_{mode_tag}.csv")
    pd.DataFrame(results_summary).to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")
    if direction_agreements:
        dir_csv = os.path.join(OUT_DIR, f"{today}_cross_tf_direction_agreement_{mode_tag}.csv")
        pd.DataFrame(direction_agreements).to_csv(dir_csv, index=False)
        print(f"Wrote: {dir_csv}")

    # Visualization: timeline with TFs of peaks stacked
    print("\nGenerating visualization...")
    fig, ax = plt.subplots(figsize=(22, 9))
    fig.patch.set_facecolor("#0a0a0a")

    # Use highest-density TF as price-line proxy
    densest_tf = max(peaks_by_tf.keys(), key=lambda k: len(peaks_by_tf[k]))
    if not peaks_by_tf[densest_tf].empty:
        ax.plot(peaks_by_tf[densest_tf]["dt"], peaks_by_tf[densest_tf]["price"],
                 color="#666", lw=0.4, alpha=0.5, label=f"{densest_tf} path")

    # Different colors/sizes per TF
    tf_styles = {
        "1W":  {"color": "#ffaa00", "size": 220, "zorder": 14},
        "1D":  {"color": "#22c55e", "size": 140, "zorder": 13},
        "4h":  {"color": "#0099ff", "size": 80, "zorder": 12},
        "1h":  {"color": "#f59e0b", "size": 40, "zorder": 11},
        "15m": {"color": "#a78bfa", "size": 18, "zorder": 10},
        "1m":  {"color": "#ef4444", "size": 10, "zorder": 9},
    }

    for tf in ["1W", "1D", "4h", "1h", "15m", "1m"]:
        if tf not in peaks_by_tf or peaks_by_tf[tf].empty:
            continue
        df = peaks_by_tf[tf]
        sty = tf_styles[tf]
        h_mask = df["_snap"] == "H"
        l_mask = df["_snap"] == "L"
        if h_mask.any():
            ax.scatter(df.loc[h_mask, "dt"], df.loc[h_mask, "price"],
                        color=sty["color"], marker="^", s=sty["size"],
                        edgecolors="white", linewidths=0.4,
                        zorder=sty["zorder"], alpha=0.7,
                        label=f"{tf} ({len(df)})")
        if l_mask.any():
            ax.scatter(df.loc[l_mask, "dt"], df.loc[l_mask, "price"],
                        color=sty["color"], marker="v", s=sty["size"],
                        edgecolors="white", linewidths=0.4,
                        zorder=sty["zorder"], alpha=0.7)

    # Color bands for largest-TF segments (1W if available, else 1D)
    band_tf = "1W" if "1W" in segs_by_tf else "1D"
    if band_tf in segs_by_tf:
        for _, s in segs_by_tf[band_tf].iterrows():
            col = "#22c55e" if s["direction"] == "UP" else "#ef4444"
            ax.axvspan(s["start_dt"], s["end_dt"], color=col, alpha=0.07, zorder=0)

    ax.legend(loc="upper right", facecolor="#1a1a1a", labelcolor="white",
              fontsize=10, framealpha=0.9, ncol=2)
    title = (f"Cross-TF nesting - {title}  "
             f"({' | '.join(f'{tf}: {len(peaks_by_tf.get(tf, []))}' for tf in tf_styles if tf in peaks_by_tf)})")
    ax.set_title(title, color="white", fontsize=11)
    ax.set_facecolor("#1a1a1a")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.grid(alpha=0.2)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", color="#aaa")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, f"{today}_cross_tf_nesting_{mode_tag}.png")
    fig.savefig(out_png, dpi=110, facecolor="#0a0a0a")
    plt.close(fig)
    print(f"Wrote: {out_png}")

    # Markdown summary
    md_path = os.path.join(OUT_DIR, f"{today}_cross_tf_nesting_{mode_tag}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Cross-TF nesting analysis - {title}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Inputs\n\n")
        f.write("| TF | Peaks | Segments | Source |\n|---|---:|---:|---|\n")
        for tf in ["1W", "1D", "4h", "1h", "15m", "1m"]:
            if tf not in peaks_by_tf:
                continue
            src = files_by_tf.get(tf, "")
            src_short = os.path.basename(src) if src else ""
            f.write(f"| {tf} | {len(peaks_by_tf[tf])} | {len(segs_by_tf[tf])} | "
                      f"{src_short} |\n")
        f.write(f"\n## Nesting summary\n\n")
        f.write("Mean number of child-TF peaks inside each parent-TF segment:\n\n")
        f.write("| Parent | Child | Parent Segs | Child Peaks Total | Mean / parent | Median / parent | Max / parent | Empty parents |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in results_summary:
            f.write(f"| {r['parent_tf']} | {r['child_tf']} | {r['parent_segments']} | "
                      f"{r['child_peaks_total']} | {r['child_per_parent_mean']:.1f} | "
                      f"{r['child_per_parent_median']:.0f} | {r['child_per_parent_max']} | "
                      f"{r['n_empty_parents']} |\n")
        f.write(f"\n## Interpretation\n\n")
        f.write("- If child peaks/parent stays roughly constant per parent direction, "
                  "lower TFs nest cleanly inside higher TFs.\n")
        f.write("- High variance in peaks/parent means low TFs encode independent structure, "
                  "not just sub-noise of high TFs.\n")
        f.write("- Empty parents = parent segments that have NO child peaks "
                  "(very rare; would indicate a marker miss).\n\n")
        f.write(f"## Files\n\n")
        f.write(f"- Chart: `{out_png}`\n")
        f.write(f"- CSV: `{out_csv}`\n")
    print(f"Wrote: {md_path}")
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["feb_1_7", "full_14mo"], default="full_14mo",
                    help="feb_1_7: high-res Feb 1-7 (4h/1h/15m/1m). "
                         "full_14mo: full coverage (1W/1D/4h/1h/15m).")
    args = ap.parse_args()
    run_nesting_analysis(mode=args.mode)


if __name__ == "__main__":
    main()
