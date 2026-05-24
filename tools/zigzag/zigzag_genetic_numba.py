"""
zigzag_genetic_numba.py -- numba @njit-accelerated genetic optimizer for
ZigzagRunner v1.0.6-RC and v1.0.7-RC. Same CLI/output schema as the pure-CPU
and CUDA versions. Use this when:
  - CUDA fails parity test or you don't have GPU
  - You want a CPU verification path independent of numba.cuda

Performance: ~30-50x speedup over pure CPython simulate() via LLVM-compiled
native code, then x6 cores via multiprocessing. v107 GA in ~30-90 sec on
Ryzen 5 5600X.

Usage:
    python tools/zigzag_genetic_numba.py --version v106 --maxiter 30 --popsize 30
    python tools/zigzag_genetic_numba.py --version v107 --maxiter 50 --popsize 40

Caveats: same as CPU/CUDA — 1m-resolution sim, SMA ATR vs Wilder, ~2x trade
count vs NT8 SA. Use rank-order to pick combos.
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

try:
    from numba import njit
except ImportError:
    print("FATAL: numba not installed. Run: pip install numba", file=sys.stderr)
    sys.exit(1)

from scipy.optimize import differential_evolution

from tools.zigzag_genetic import (
    load_1m_bars,
    split_train_holdout,
    compute_true_range,
    rolling_sma_atr,
    V106_BOUNDS, V106_NAMES, V107_BOUNDS, V107_NAMES,
    vector_to_params_v106, vector_to_params_v107,
    DOLLAR_PER_POINT, COMMISSION_RT, SLIPPAGE_PTS,
    evaluate_combos,
)


# =============================================================================
# JIT'd simulation kernels
# =============================================================================
# Returns (pnl_net, n_trades, n_wins). Daily metrics computed CPU-side after.
# Uses primitive types only — no dicts, no Python objects.

@njit(cache=True, fastmath=True)
def simulate_jit_v106(
    opens, closes, mins,
    r_eff,
    max_loss_pts, mfe_cut_bars, mfe_cut_usd,
    trail_activate_pts, trail_giveback_pct,
):
    DPP = 2.0
    COMM = 1.90
    SLIP = 0.25
    EOD_MINS = 20 * 60 + 55
    CUT_MINS = 20 * 60 + 30

    direction = 0
    extreme_price = np.nan
    pos_dir = 0
    pos_entry_px = 0.0
    trade_bars_held = 0
    trade_mfe_usd = 0.0
    trail_armed = False
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    T = len(closes)

    for i in range(T - 1):
        c = closes[i]
        next_open = opens[i + 1]
        mins_of_day = mins[i] + 1

        if mins_of_day >= EOD_MINS:
            if pos_dir != 0:
                slipped_px = next_open - pos_dir * SLIP
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DPP - COMM
                pnl_total += pnl_n
                n_trades += 1
                if pnl_n > 0.0:
                    n_wins += 1
                pos_dir = 0
                trade_bars_held = 0
                trade_mfe_usd = 0.0
                trail_armed = False
            continue

        if pos_dir != 0:
            trade_bars_held += 1
            unrealized_pts = pos_dir * (c - pos_entry_px)
            unrealized_usd = unrealized_pts * DPP
            if unrealized_usd > trade_mfe_usd:
                trade_mfe_usd = unrealized_usd

            # Trail
            if trail_activate_pts > 0.0:
                activate_usd = trail_activate_pts * DPP
                if (not trail_armed) and (trade_mfe_usd >= activate_usd):
                    trail_armed = True
                if trail_armed:
                    trail_threshold = trade_mfe_usd * (1.0 - trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        slipped_px = next_open - pos_dir * SLIP
                        pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total += pnl_n
                        n_trades += 1
                        if pnl_n > 0.0:
                            n_wins += 1
                        pos_dir = 0
                        trade_bars_held = 0
                        trade_mfe_usd = 0.0
                        trail_armed = False

            # SL
            if pos_dir != 0 and max_loss_pts > 0.0:
                unrealized_pts = pos_dir * (c - pos_entry_px)
                if unrealized_pts <= -max_loss_pts:
                    slipped_px = next_open - pos_dir * SLIP
                    pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                    pnl_n = pnl_pts * DPP - COMM
                    pnl_total += pnl_n
                    n_trades += 1
                    if pnl_n > 0.0:
                        n_wins += 1
                    pos_dir = 0
                    trade_bars_held = 0
                    trade_mfe_usd = 0.0
                    trail_armed = False

            # MFE-cut
            if pos_dir != 0 and mfe_cut_bars > 0:
                if trade_bars_held == mfe_cut_bars:
                    if trade_mfe_usd <= mfe_cut_usd:
                        slipped_px = next_open - pos_dir * SLIP
                        pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total += pnl_n
                        n_trades += 1
                        if pnl_n > 0.0:
                            n_wins += 1
                        pos_dir = 0
                        trade_bars_held = 0
                        trade_mfe_usd = 0.0
                        trail_armed = False

        if np.isnan(extreme_price):
            extreme_price = c
            continue

        pivot_confirmed = False
        new_pivot_dir = 0
        if direction == 0:
            if c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = 1; extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = 1
                direction = -1; extreme_price = c
        elif direction == 1:
            if c > extreme_price:
                extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = 1
                direction = -1; extreme_price = c
        else:
            if c < extreme_price:
                extreme_price = c
            elif c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = 1; extreme_price = c

        if not pivot_confirmed:
            continue
        if mins_of_day >= CUT_MINS:
            continue

        action_side = -1 if new_pivot_dir == 1 else 1

        if pos_dir != 0:
            slipped_px = next_open - pos_dir * SLIP
            pnl_pts = pos_dir * (slipped_px - pos_entry_px)
            pnl_n = pnl_pts * DPP - COMM
            pnl_total += pnl_n
            n_trades += 1
            if pnl_n > 0.0:
                n_wins += 1

        slipped_entry = next_open + action_side * SLIP
        pos_dir = action_side
        pos_entry_px = slipped_entry
        trade_bars_held = 0
        trade_mfe_usd = 0.0
        trail_armed = False

    if pos_dir != 0:
        last_close = closes[T - 1]
        pnl_pts = pos_dir * (last_close - pos_entry_px)
        pnl_n = pnl_pts * DPP - COMM
        pnl_total += pnl_n
        n_trades += 1
        if pnl_n > 0.0:
            n_wins += 1

    return pnl_total, n_trades, n_wins


@njit(cache=True, fastmath=True)
def simulate_jit_v107(
    opens, closes, mins, atr_values,
    r_warmup,
    atr_multiplier, min_r_points, max_r_points,
    max_loss_pts, mfe_cut_bars, mfe_cut_usd,
    trail_activate_pts, trail_giveback_pct,
):
    DPP = 2.0
    COMM = 1.90
    SLIP = 0.25
    EOD_MINS = 20 * 60 + 55
    CUT_MINS = 20 * 60 + 30

    direction = 0
    extreme_price = np.nan
    pos_dir = 0
    pos_entry_px = 0.0
    trade_bars_held = 0
    trade_mfe_usd = 0.0
    trail_armed = False
    pnl_total = 0.0
    n_trades = 0
    n_wins = 0
    T = len(closes)

    for i in range(T - 1):
        c = closes[i]
        next_open = opens[i + 1]
        mins_of_day = mins[i] + 1

        # Effective R
        atr_val = atr_values[i]
        if np.isnan(atr_val) or atr_val <= 0.0:
            r_eff = r_warmup
        else:
            r_eff = atr_val * atr_multiplier
            if r_eff < min_r_points:
                r_eff = min_r_points
            if r_eff > max_r_points:
                r_eff = max_r_points

        if mins_of_day >= EOD_MINS:
            if pos_dir != 0:
                slipped_px = next_open - pos_dir * SLIP
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DPP - COMM
                pnl_total += pnl_n
                n_trades += 1
                if pnl_n > 0.0:
                    n_wins += 1
                pos_dir = 0
                trade_bars_held = 0
                trade_mfe_usd = 0.0
                trail_armed = False
            continue

        if pos_dir != 0:
            trade_bars_held += 1
            unrealized_pts = pos_dir * (c - pos_entry_px)
            unrealized_usd = unrealized_pts * DPP
            if unrealized_usd > trade_mfe_usd:
                trade_mfe_usd = unrealized_usd

            if trail_activate_pts > 0.0:
                activate_usd = trail_activate_pts * DPP
                if (not trail_armed) and (trade_mfe_usd >= activate_usd):
                    trail_armed = True
                if trail_armed:
                    trail_threshold = trade_mfe_usd * (1.0 - trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        slipped_px = next_open - pos_dir * SLIP
                        pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total += pnl_n
                        n_trades += 1
                        if pnl_n > 0.0:
                            n_wins += 1
                        pos_dir = 0
                        trade_bars_held = 0
                        trade_mfe_usd = 0.0
                        trail_armed = False

            if pos_dir != 0 and max_loss_pts > 0.0:
                unrealized_pts = pos_dir * (c - pos_entry_px)
                if unrealized_pts <= -max_loss_pts:
                    slipped_px = next_open - pos_dir * SLIP
                    pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                    pnl_n = pnl_pts * DPP - COMM
                    pnl_total += pnl_n
                    n_trades += 1
                    if pnl_n > 0.0:
                        n_wins += 1
                    pos_dir = 0
                    trade_bars_held = 0
                    trade_mfe_usd = 0.0
                    trail_armed = False

            if pos_dir != 0 and mfe_cut_bars > 0:
                if trade_bars_held == mfe_cut_bars:
                    if trade_mfe_usd <= mfe_cut_usd:
                        slipped_px = next_open - pos_dir * SLIP
                        pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total += pnl_n
                        n_trades += 1
                        if pnl_n > 0.0:
                            n_wins += 1
                        pos_dir = 0
                        trade_bars_held = 0
                        trade_mfe_usd = 0.0
                        trail_armed = False

        if np.isnan(extreme_price):
            extreme_price = c
            continue

        pivot_confirmed = False
        new_pivot_dir = 0
        if direction == 0:
            if c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = 1; extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = 1
                direction = -1; extreme_price = c
        elif direction == 1:
            if c > extreme_price:
                extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = 1
                direction = -1; extreme_price = c
        else:
            if c < extreme_price:
                extreme_price = c
            elif c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = 1; extreme_price = c

        if not pivot_confirmed:
            continue
        if mins_of_day >= CUT_MINS:
            continue

        action_side = -1 if new_pivot_dir == 1 else 1
        if pos_dir != 0:
            slipped_px = next_open - pos_dir * SLIP
            pnl_pts = pos_dir * (slipped_px - pos_entry_px)
            pnl_n = pnl_pts * DPP - COMM
            pnl_total += pnl_n
            n_trades += 1
            if pnl_n > 0.0:
                n_wins += 1

        slipped_entry = next_open + action_side * SLIP
        pos_dir = action_side
        pos_entry_px = slipped_entry
        trade_bars_held = 0
        trade_mfe_usd = 0.0
        trail_armed = False

    if pos_dir != 0:
        last_close = closes[T - 1]
        pnl_pts = pos_dir * (last_close - pos_entry_px)
        pnl_n = pnl_pts * DPP - COMM
        pnl_total += pnl_n
        n_trades += 1
        if pnl_n > 0.0:
            n_wins += 1

    return pnl_total, n_trades, n_wins


# =============================================================================
# Fitness wrappers (for scipy DE)
# =============================================================================

def fitness_v106_numba(x, opens, closes, mins, tr):
    p = vector_to_params_v106(x)
    pnl, _, _ = simulate_jit_v106(
        opens, closes, mins,
        p["r_points"],
        p["max_loss_pts"], p["mfe_cut_bars"], p["mfe_cut_usd"],
        p["trail_activate_pts"], p["trail_giveback_pct"],
    )
    return -pnl


def fitness_v107_numba(x, opens, closes, mins, tr):
    p = vector_to_params_v107(x)
    atr_values = rolling_sma_atr(tr, p["atr_lookback"])
    pnl, _, _ = simulate_jit_v107(
        opens, closes, mins, atr_values,
        30.0,  # r_warmup fallback
        p["atr_multiplier"], p["min_r_points"], p["max_r_points"],
        p["max_loss_pts"], p["mfe_cut_bars"], p["mfe_cut_usd"],
        p["trail_activate_pts"], p["trail_giveback_pct"],
    )
    return -pnl


# =============================================================================
# Main GA
# =============================================================================

def run_ga(version: str, args):
    gc.collect()
    pd.reset_option("display.float_format", silent=True) if hasattr(pd, "reset_option") else None

    print("=" * 80)
    print(f"PYTHON GENETIC OPTIMIZATION (numba JIT) — ZigzagRunner_{version}")
    print("=" * 80)
    print(f"Atlas:        {args.atlas}")
    print(f"Train end:    {args.train_end}")
    print(f"Holdout from: {args.holdout_start}")
    print(f"Maxiter:      {args.maxiter}")
    print(f"Popsize mult: {args.popsize}")
    print(f"Workers:      {args.workers if args.workers else 'all cores'}")
    print()

    bars = load_1m_bars(args.atlas)
    bars_train, bars_holdout = split_train_holdout(bars, args.train_end, args.holdout_start)
    print(f"Train: {len(bars_train):,} bars")
    print(f"Holdout: {len(bars_holdout):,} bars\n")

    opens_t = bars_train["open"].to_numpy(dtype=np.float64)
    closes_t = bars_train["close"].to_numpy(dtype=np.float64)
    highs_t = bars_train["high"].to_numpy(dtype=np.float64)
    lows_t = bars_train["low"].to_numpy(dtype=np.float64)
    mins_t = bars_train["mins_of_day_utc"].to_numpy(dtype=np.int32)
    tr_t = compute_true_range(highs_t, lows_t, closes_t)

    bounds = V107_BOUNDS if version == "v107" else V106_BOUNDS
    names = V107_NAMES if version == "v107" else V106_NAMES
    fitness = fitness_v107_numba if version == "v107" else fitness_v106_numba

    # JIT warmup
    print("JIT warming up...")
    t_warmup = time.time()
    if version == "v107":
        _ = fitness_v107_numba(np.array([60.0, 5.0, 5.0, 200.0, 90.0, 17, 2.0, 21.0, 0.05]),
                                opens_t, closes_t, mins_t, tr_t)
    else:
        _ = fitness_v106_numba(np.array([45.0, 90.0, 17, 2.0, 21.0, 0.05]),
                                opens_t, closes_t, mins_t, tr_t)
    print(f"  Warmup took {time.time() - t_warmup:.1f} sec\n")

    workers = args.workers if args.workers else -1
    print(f"Starting DE: {len(bounds)}-D, popsize={args.popsize}, maxiter={args.maxiter}")
    t0 = time.time()
    result = differential_evolution(
        fitness,
        bounds=bounds,
        args=(opens_t, closes_t, mins_t, tr_t),
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=False,
        mutation=(0.5, 1.5),
        recombination=0.7,
        tol=1e-6,
        workers=workers,
        updating="deferred",
        init="sobol",
    )
    elapsed = time.time() - t0
    print(f"DE finished in {elapsed:.1f} sec  (best fitness = ${-result.fun:+,.2f})\n")

    # Top-K via re-sampling around the best
    rng = np.random.default_rng(args.seed + 1)
    best_x = result.x.copy()
    candidates = [best_x.copy()]
    for _ in range(args.top_k * 4):
        pert = best_x + rng.normal(0, 0.05, len(best_x)) * np.array([b[1] - b[0] for b in bounds])
        for i, (lo, hi) in enumerate(bounds):
            pert[i] = max(lo, min(hi, pert[i]))
        candidates.append(pert)

    print(f"Scoring {len(candidates)} candidates on train (CPU sim)...")
    if version == "v107":
        combos = [vector_to_params_v107(x) for x in candidates]
    else:
        combos = [vector_to_params_v106(x) for x in candidates]
    df_train = evaluate_combos(bars_train, combos, version)
    df_train = df_train.sort_values("pnl_net", ascending=False).drop_duplicates(
        subset=names, keep="first"
    ).reset_index(drop=True)
    df_top = df_train.head(args.top_k).copy()
    df_top["rank_train"] = range(1, len(df_top) + 1)

    if len(bars_holdout) > 0:
        print(f"Scoring top-{args.top_k} on holdout...")
        df_holdout = evaluate_combos(bars_holdout, df_top[names].to_dict("records"), version)
        df_holdout.columns = [c if c in names else f"holdout_{c}" for c in df_holdout.columns]
        df_top = pd.concat([df_top.reset_index(drop=True),
                            df_holdout.drop(columns=names).reset_index(drop=True)], axis=1)
    else:
        for c in ["holdout_pnl_net", "holdout_n_trades", "holdout_pnl_per_day", "holdout_win_rate"]:
            df_top[c] = np.nan

    df_top["pnl_drop_pct"] = (
        100.0 * (df_top["pnl_net"] - df_top["holdout_pnl_net"]) / df_top["pnl_net"].replace(0, np.nan)
    )
    df_top["robust_flag"] = (df_top.get("holdout_pnl_per_day", pd.Series(np.nan)) >= 30.0) & (df_top["pnl_drop_pct"] < 30.0)

    show_cols = names + [
        "pnl_net", "pnl_per_day", "n_trades", "win_rate",
        "holdout_pnl_net", "holdout_pnl_per_day", "holdout_n_trades", "holdout_win_rate",
        "pnl_drop_pct", "robust_flag",
    ]
    show_cols = [c for c in show_cols if c in df_top.columns]
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(f"\n=== TOP-{len(df_top)} WITH HOLDOUT ===")
    print(df_top[show_cols].to_string(index=False))

    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("reports/findings", exist_ok=True)
    csv_path = f"reports/findings/{today}_genetic_numba_{version}_topk.csv"
    df_top.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    md_path = f"reports/findings/{today}_genetic_numba_{version}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# numba JIT Genetic optimization report — ZigzagRunner_{version}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n- Atlas: `{args.atlas}`\n- Train: through `{args.train_end}` ({len(bars_train):,} 1m bars)\n")
        f.write(f"- Holdout: from `{args.holdout_start}` ({len(bars_holdout):,} 1m bars)\n")
        f.write(f"- Optimizer: scipy DE + numba @njit kernel + multiprocessing\n")
        f.write(f"- DE wall time: {elapsed:.1f} sec\n\n")
        f.write(f"## Top-{len(df_top)} candidates (CPU float64 verification + holdout)\n\n")
        try:
            f.write(df_top[show_cols].to_markdown(index=False, floatfmt=".2f"))
        except ImportError:
            f.write("```\n")
            f.write(df_top[show_cols].to_string(index=False))
            f.write("\n```\n")
        f.write("\n\n## Decision matrix\n\n")
        f.write("| Outcome | Action |\n|---|---|\n")
        f.write("| robust_flag=True | Pick that combo. Holdout >= $30/day AND drop < 30%. |\n")
        f.write("| All fail | Strategy overfits — stay on v1.0.4. |\n")
    print(f"Wrote: {md_path}")

    del bars, bars_train, bars_holdout
    del opens_t, closes_t, highs_t, lows_t, mins_t, tr_t
    pd.reset_option("display.float_format", silent=True) if hasattr(pd, "reset_option") else None
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v106", "v107"], default="v106")
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--holdout-start", default="2026-01-01")
    ap.add_argument("--maxiter", type=int, default=30)
    ap.add_argument("--popsize", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if args.workers == 0:
        args.workers = -1
    run_ga(args.version, args)


if __name__ == "__main__":
    main()
