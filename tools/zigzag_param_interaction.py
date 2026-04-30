"""
zigzag_param_interaction.py -- Pairwise parameter interaction matrix for
v1.3-RC. PNL is the response variable (Y).

For each pair of parameters, sweep a 2D grid (other params held at defaults),
run simulate_day across N sample days, aggregate net $/day, and plot a heatmap.

Output:
    reports/findings/zigzag_v13_interaction/<param_a>_x_<param_b>.png
    reports/findings/zigzag_v13_interaction/_summary.csv
    reports/findings/zigzag_v13_interaction/_top10.txt

Limitations vs NT8 v1.3-RC:
  - Single-TF approximation (1m). No multi-TF Pivot/SL/Trail split.
  - No StagnationMonitor (use --stagnation-bars=0 to mirror disabled state).
  - No Tier2 transition: simulator uses max(t1Trail, pct*peak) directly,
    matching v1.3-RC's Tier1->Tier2 ratchet behavior at the crossover but
    without the explicit Tier2ActivatePoints threshold.
  - Useful as a DIRECTIONAL pre-filter before NT8 Strategy Analyzer GA;
    DO NOT trust absolute $/day numbers for live deployment.

Usage:
    python tools/zigzag_param_interaction.py                 # default 5-param matrix
    python tools/zigzag_param_interaction.py --days 60       # use 60 random sample days
    python tools/zigzag_param_interaction.py --grid 5        # coarser 5x5 grid (faster)
    python tools/zigzag_param_interaction.py --pairs r,t1act # restrict to one pair
"""
from __future__ import annotations

import argparse
import os
import sys
import random
import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Make zigzag_trail_ticker importable from tools/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zigzag_trail_ticker import simulate_day, ATLAS_ROOT  # type: ignore

warnings.filterwarnings("ignore")


# ── Param search space ─────────────────────────────────────────────────────
# Each param: (display_name, defaults, sweep_values, simulate_kwarg_name)
@dataclass(frozen=True)
class Param:
    key: str            # short name for CLI / filename
    label: str          # human-readable label for plot axis
    default: float
    sweep: tuple        # values to sweep
    sim_kwarg: str      # kwarg name for simulate_day(...)


PARAMS = {
    "r":      Param("r",      "RPoints",                   30.0, (15, 20, 25, 30, 35, 40, 50),         "r"),
    "t1act":  Param("t1act",  "Tier1 Activate (pts)",      10.0, (4, 6, 8, 10, 14, 20, 30),            "trail_activate"),
    "t1dist": Param("t1dist", "Tier1 Trail Distance (pts)", 5.0, (2, 3, 5, 7, 10, 15),                 "trail_dist"),
    "t2pct":  Param("t2pct",  "Tier2 Trail Percent",        0.10, (0.05, 0.08, 0.10, 0.15, 0.20, 0.30), "trail_pct"),
    "sl":     Param("sl",     "Hard Stop Loss (pts)",      25.0, (0, 10, 15, 20, 25, 35, 50),          "sl_pts"),
}


# ── Helpers ─────────────────────────────────────────────────────────────────
def _list_days(atlas_root: str) -> list[str]:
    """Return sorted day labels (YYYY_MM_DD) present in ATLAS/1m/.
    Format matches the filenames so simulate_day can construct the path
    via f'{day_label}.parquet'."""
    files = os.listdir(os.path.join(atlas_root, "1m"))
    days = []
    for f in files:
        if not f.endswith(".parquet"):
            continue
        days.append(f[:-len(".parquet")])
    return sorted(days)


def _evaluate(days: list[str], r: float, t1act: float, t1dist: float,
              t2pct: float, sl: float, atlas_root: str) -> dict:
    """Run simulate_day across `days`, aggregate net $/day."""
    daily_pnl = []
    n_trades = 0
    for d in days:
        try:
            trades, _summary = simulate_day(
                day_label=d,
                r=r,
                trail_activate=t1act,
                trail_dist=t1dist,
                trail_pct=t2pct,
                atlas_root=atlas_root,
                use_filter=False,
                sl_pts=sl,
            )
        except Exception:
            continue
        if not trades:
            daily_pnl.append(0.0)
            continue
        day_pnl = sum(t.get("pnl_usd", 0.0) for t in trades)
        daily_pnl.append(day_pnl)
        n_trades += len(trades)

    arr = np.array(daily_pnl) if daily_pnl else np.array([0.0])
    return {
        "mean":      float(arr.mean()),
        "median":    float(np.median(arr)),
        "n_trades":  n_trades,
        "n_days":    len(daily_pnl),
        "winners":   int((arr > 0).sum()),
        "losers":    int((arr <= 0).sum()),
        "best":      float(arr.max()) if len(arr) else 0.0,
        "worst":     float(arr.min()) if len(arr) else 0.0,
    }


def _heatmap(grid: np.ndarray, x_vals, y_vals, x_label: str, y_label: str,
             out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    vmax = max(abs(grid.min()), abs(grid.max())) if grid.size else 1.0
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([f"{v:g}" for v in x_vals], rotation=30)
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([f"{v:g}" for v in y_vals])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Annotate each cell with $/day value.
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            color = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                    color=color, fontsize=8)
    fig.colorbar(im, ax=ax, label="Mean $/day")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ── Main sweep ──────────────────────────────────────────────────────────────
def run_pair_sweep(p_a: Param, p_b: Param, days: list[str], atlas_root: str,
                   defaults: dict) -> tuple[np.ndarray, list[dict]]:
    """Sweep grid of (p_a, p_b), return ($/day grid, list of cell records)."""
    rows = []
    grid = np.zeros((len(p_b.sweep), len(p_a.sweep)))   # rows = y, cols = x
    for i, vb in enumerate(p_b.sweep):
        for j, va in enumerate(p_a.sweep):
            cfg = dict(defaults)
            cfg[p_a.sim_kwarg] = va
            cfg[p_b.sim_kwarg] = vb
            stats = _evaluate(days,
                              r=cfg["r"],
                              t1act=cfg["trail_activate"],
                              t1dist=cfg["trail_dist"],
                              t2pct=cfg["trail_pct"],
                              sl=cfg["sl_pts"],
                              atlas_root=atlas_root)
            grid[i, j] = stats["mean"]
            rows.append({
                "pair": f"{p_a.key}_x_{p_b.key}",
                p_a.key: va,
                p_b.key: vb,
                "mean_pnl_usd_day": stats["mean"],
                "median_pnl_usd_day": stats["median"],
                "n_days": stats["n_days"],
                "n_trades": stats["n_trades"],
                "winning_days": stats["winners"],
                "losing_days":  stats["losers"],
            })
    return grid, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas",  default=ATLAS_ROOT)
    ap.add_argument("--days",   type=int, default=40,
                    help="Number of random sample days from ATLAS/1m/ (default 40)")
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--out",    default="reports/findings/zigzag_v13_interaction")
    ap.add_argument("--pairs",  default="all",
                    help='Comma-list of pair keys ("r,t1act"); "all" runs every C(5,2)=10 pair.')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Pick N random days from ATLAS for sampling.
    all_days = _list_days(args.atlas)
    rng = random.Random(args.seed)
    sample_days = rng.sample(all_days, min(args.days, len(all_days)))
    sample_days.sort()

    print(f"Atlas:        {args.atlas}")
    print(f"Sample days:  {len(sample_days)}  (seed={args.seed})")
    print(f"Range:        {sample_days[0]} -> {sample_days[-1]}")
    print(f"Output dir:   {args.out}")
    print()

    defaults = {
        "r":               PARAMS["r"].default,
        "trail_activate":  PARAMS["t1act"].default,
        "trail_dist":      PARAMS["t1dist"].default,
        "trail_pct":       PARAMS["t2pct"].default,
        "sl_pts":          PARAMS["sl"].default,
    }

    # All C(5,2) = 10 pairs
    keys = list(PARAMS.keys())
    pairs = list(itertools.combinations(keys, 2))
    if args.pairs != "all":
        wanted = set(args.pairs.split(","))
        pairs = [(a, b) for a, b in pairs if a in wanted or b in wanted]

    summary_rows = []
    n_total_cells = sum(len(PARAMS[a].sweep) * len(PARAMS[b].sweep) for a, b in pairs)
    print(f"Pairs: {len(pairs)}, total cells: {n_total_cells}, ~{len(sample_days)} backtests/cell")
    print(f"Estimated total backtests: {n_total_cells * len(sample_days)}")
    print()

    pbar = tqdm(total=n_total_cells, desc="cells")
    for (a_key, b_key) in pairs:
        p_a = PARAMS[a_key]
        p_b = PARAMS[b_key]

        rows = []
        grid = np.zeros((len(p_b.sweep), len(p_a.sweep)))
        for i, vb in enumerate(p_b.sweep):
            for j, va in enumerate(p_a.sweep):
                cfg = dict(defaults)
                cfg[p_a.sim_kwarg] = va
                cfg[p_b.sim_kwarg] = vb
                stats = _evaluate(sample_days,
                                  r=cfg["r"],
                                  t1act=cfg["trail_activate"],
                                  t1dist=cfg["trail_dist"],
                                  t2pct=cfg["trail_pct"],
                                  sl=cfg["sl_pts"],
                                  atlas_root=args.atlas)
                grid[i, j] = stats["mean"]
                rows.append({
                    "pair": f"{p_a.key}_x_{p_b.key}",
                    p_a.key: va,
                    p_b.key: vb,
                    "mean_pnl_usd_day":   stats["mean"],
                    "median_pnl_usd_day": stats["median"],
                    "n_days":             stats["n_days"],
                    "n_trades":           stats["n_trades"],
                    "winning_days":       stats["winners"],
                    "losing_days":        stats["losers"],
                })
                pbar.update(1)

        summary_rows.extend(rows)

        # Plot heatmap
        out_png = os.path.join(args.out, f"{p_a.key}_x_{p_b.key}.png")
        title = (f"{p_a.label} (X) vs {p_b.label} (Y) -- mean $/day\n"
                 f"defaults locked: " +
                 ", ".join(f"{k}={v:g}" for k, v in defaults.items()
                           if k != p_a.sim_kwarg and k != p_b.sim_kwarg))
        _heatmap(grid, p_a.sweep, p_b.sweep, p_a.label, p_b.label, out_png, title)

    pbar.close()

    # Summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(args.out, "_summary.csv")
    df.to_csv(csv_path, index=False)

    # Top 10
    top = df.sort_values("mean_pnl_usd_day", ascending=False).head(10)
    top_path = os.path.join(args.out, "_top10.txt")
    with open(top_path, "w") as f:
        f.write("Top 10 cells by mean $/day across all pair sweeps.\n")
        f.write("Note: Each row reports ONE cell of ONE pair sweep. Overlap likely.\n\n")
        f.write(top.to_string(index=False))
        f.write("\n")
    print()
    print(f"Saved {len(pairs)} heatmaps + summary CSV + top10 to: {args.out}")
    print()
    print("Top 10 cells:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
