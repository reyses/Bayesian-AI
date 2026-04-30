"""
zigzag_genetic_cuda.py -- GPU-accelerated genetic optimizer for ZigzagRunner
v1.0.6-RC and v1.0.7-RC. Uses numba.cuda for the per-bar simulation kernel
(1 thread = 1 sim, population parallelism across threads) and PyTorch for
manual differential evolution. Reusable framework for future strategies.

Reuse pattern: factor out `simulate_kernel_v107` + `de_evolve` for any
strategy that fits the (per-bar state machine, per-population fitness) mold.

Performance target: v107 GA in 5-15 seconds end-to-end on RTX 3060 12 GB
(vs 25-100 min on CPU multiprocessing).

Usage:
    # Parity test FIRST — confirms CUDA sim matches CPU sim before running GA
    python tools/zigzag_genetic_cuda.py --parity-test --version v107

    # GA-A: v1.0.6-RC, 6-D static-R search
    python tools/zigzag_genetic_cuda.py --version v106 --maxiter 30 --popsize 30

    # GA-B: v1.0.7-RC, 9-D dynamic-R search
    python tools/zigzag_genetic_cuda.py --version v107 --maxiter 50 --popsize 40

Outputs (same schema as CPU version):
    reports/findings/2026-04-28_genetic_cuda_<version>_topk.csv
    reports/findings/2026-04-28_genetic_cuda_<version>_report.md

Caveats:
    - Float32 internally for speed. Final top-K re-validated on CPU sim
      (float64) for accuracy reporting.
    - 1m-resolution sim. Hard SL fires at 1m close, not 1s like real NT8.
    - Python/CPU ATR uses simple moving average; NT8 uses Wilder smoothing.
      This tool keeps SMA for parity with the CPU GA tool. Difference is
      small for moderate N.
"""
from __future__ import annotations
import argparse
import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow `python tools/foo.py` invocation by ensuring repo root is on sys.path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

# CUDA stack
try:
    import torch
    from numba import cuda, float32, int32
except ImportError as e:
    print(f"FATAL: missing CUDA dependency: {e}", file=sys.stderr)
    print("       This tool requires torch + numba (CUDA-enabled).", file=sys.stderr)
    sys.exit(1)

# Reuse CPU helpers (data loading, parity-target sim)
from tools.zigzag_genetic import (
    load_1m_bars,
    split_train_holdout,
    compute_true_range,
    rolling_sma_atr,
    simulate as simulate_cpu,
    V106_BOUNDS, V106_NAMES, V107_BOUNDS, V107_NAMES,
    vector_to_params_v106, vector_to_params_v107,
    DOLLAR_PER_POINT, COMMISSION_RT, SLIPPAGE_PTS,
    evaluate_combos,
)


# =============================================================================
# CUDA setup
# =============================================================================

def cuda_setup() -> torch.device:
    if not torch.cuda.is_available():
        print("FATAL: torch.cuda.is_available() == False.", file=sys.stderr)
        sys.exit(1)
    if not cuda.is_available():
        print("FATAL: numba.cuda.is_available() == False.", file=sys.stderr)
        sys.exit(1)
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"CUDA device: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    return device


# =============================================================================
# True range + ATR table (GPU)
# =============================================================================

def compute_true_range_torch(highs: torch.Tensor, lows: torch.Tensor, closes: torch.Tensor) -> torch.Tensor:
    """Vectorized TR on GPU. (T,) -> (T,)."""
    n = highs.shape[0]
    tr = torch.empty(n, dtype=highs.dtype, device=highs.device)
    tr[0] = highs[0] - lows[0]
    hl = highs[1:] - lows[1:]
    hp = (highs[1:] - closes[:-1]).abs()
    lp = (lows[1:] - closes[:-1]).abs()
    tr[1:] = torch.maximum(torch.maximum(hl, hp), lp)
    return tr


def precompute_atr_table(tr: torch.Tensor, lookback_min: int, lookback_max: int) -> torch.Tensor:
    """Pre-compute SMA ATR for every integer lookback in [lookback_min, lookback_max].
    Returns (L, T) where L = lookback_max - lookback_min + 1.
    NaN for warmup positions (idx < lookback - 1)."""
    n = tr.shape[0]
    L = lookback_max - lookback_min + 1
    out = torch.full((L, n), float("nan"), dtype=tr.dtype, device=tr.device)
    cumsum = torch.cumsum(tr, dim=0)
    for i, lb in enumerate(range(lookback_min, lookback_max + 1)):
        if n >= lb:
            out[i, lb - 1] = cumsum[lb - 1] / lb
            out[i, lb:] = (cumsum[lb:] - cumsum[:-lb]) / lb
    return out


# =============================================================================
# numba.cuda simulation kernel — v107 (dynamic R)
# =============================================================================
# 1 thread = 1 simulation. Each thread runs the full bar loop sequentially
# (state lives in registers — fast). Population evaluated in parallel across
# threads. Output: per-sim total PnL, n_trades, n_wins.
#
# Hard-coded constants for kernel speed (no global memory reads for them):
#   DOLLAR_PER_POINT = 2.0
#   COMMISSION_RT    = 1.90
#   SLIPPAGE_PTS     = 0.25
#   eod_mins         = 20*60 + 55 = 1255
#   cut_mins         = 20*60 + 30 = 1230
#   r_warmup         = 30.0  (fallback during ATR warmup)

@cuda.jit
def simulate_kernel_v107(opens, closes, mins, atr_table, atr_lookback_min,
                          params, flip_direction, out_pnl, out_n_trades, out_n_wins):
    """v1.0.7-RC simulation. params shape: (P, 9):
       [atr_lookback, atr_mult, min_r, max_r, max_loss_pts, mfe_cut_bars,
        mfe_cut_usd, trail_activate_pts, trail_giveback_pct]"""
    p = cuda.grid(1)
    P = params.shape[0]
    if p >= P:
        return

    T = opens.shape[0]

    # Constants (in registers — kernel-only)
    DPP = float32(2.0)
    COMM = float32(1.90)
    SLIP = float32(0.25)
    EOD_MINS = int32(1255)
    CUT_MINS = int32(1230)
    R_WARMUP = float32(30.0)

    # Unpack params for this sim
    atr_idx = int32(int(params[p, 0]) - atr_lookback_min)
    atr_mult = float32(params[p, 1])
    min_r = float32(params[p, 2])
    max_r = float32(params[p, 3])
    max_loss_pts = float32(params[p, 4])
    mfe_cut_bars = int32(int(params[p, 5]))
    mfe_cut_usd = float32(params[p, 6])
    trail_activate_pts = float32(params[p, 7])
    trail_giveback_pct = float32(params[p, 8])

    # State (registers — no global memory traffic)
    direction = int32(0)
    extreme_price = float32(math.nan)
    pos_dir = int32(0)
    pos_entry_px = float32(0.0)
    trade_bars_held = int32(0)
    trade_mfe_usd = float32(0.0)
    trail_armed = False

    # Aggregates
    pnl_total = float32(0.0)
    n_trades = int32(0)
    n_wins = int32(0)

    for i in range(T - 1):
        c = float32(closes[i])
        next_open = float32(opens[i + 1])
        mins_of_day = int32(mins[i] + 1)

        # Effective R
        atr_val = float32(atr_table[atr_idx, i])
        if math.isnan(atr_val) or atr_val <= float32(0.0):
            r_eff = R_WARMUP
        else:
            r_eff = atr_val * atr_mult
            if r_eff < min_r:
                r_eff = min_r
            if r_eff > max_r:
                r_eff = max_r

        # EOD force-close
        if mins_of_day >= EOD_MINS:
            if pos_dir != int32(0):
                slipped_px = next_open - float32(pos_dir) * SLIP
                pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DPP - COMM
                pnl_total = pnl_total + pnl_n
                n_trades = n_trades + int32(1)
                if pnl_n > float32(0.0):
                    n_wins = n_wins + int32(1)
                pos_dir = int32(0)
                trade_bars_held = int32(0)
                trade_mfe_usd = float32(0.0)
                trail_armed = False
            continue

        # Update per-trade state
        if pos_dir != int32(0):
            trade_bars_held = trade_bars_held + int32(1)
            unrealized_pts = float32(pos_dir) * (c - pos_entry_px)
            unrealized_usd = unrealized_pts * DPP
            if unrealized_usd > trade_mfe_usd:
                trade_mfe_usd = unrealized_usd

            # Rule 4: Trail
            if trail_activate_pts > float32(0.0):
                activate_usd = trail_activate_pts * DPP
                if (not trail_armed) and (trade_mfe_usd >= activate_usd):
                    trail_armed = True
                if trail_armed:
                    trail_threshold = trade_mfe_usd * (float32(1.0) - trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        slipped_px = next_open - float32(pos_dir) * SLIP
                        pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total = pnl_total + pnl_n
                        n_trades = n_trades + int32(1)
                        if pnl_n > float32(0.0):
                            n_wins = n_wins + int32(1)
                        pos_dir = int32(0)
                        trade_bars_held = int32(0)
                        trade_mfe_usd = float32(0.0)
                        trail_armed = False

            # Rule 1: Hard SL
            if pos_dir != int32(0) and max_loss_pts > float32(0.0):
                # recompute unrealized in case position changed
                unrealized_pts = float32(pos_dir) * (c - pos_entry_px)
                if unrealized_pts <= -max_loss_pts:
                    slipped_px = next_open - float32(pos_dir) * SLIP
                    pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                    pnl_n = pnl_pts * DPP - COMM
                    pnl_total = pnl_total + pnl_n
                    n_trades = n_trades + int32(1)
                    if pnl_n > float32(0.0):
                        n_wins = n_wins + int32(1)
                    pos_dir = int32(0)
                    trade_bars_held = int32(0)
                    trade_mfe_usd = float32(0.0)
                    trail_armed = False

            # Rule 2: MFE-cut at bar N
            if pos_dir != int32(0) and mfe_cut_bars > int32(0):
                if trade_bars_held == mfe_cut_bars:
                    if trade_mfe_usd <= mfe_cut_usd:
                        slipped_px = next_open - float32(pos_dir) * SLIP
                        pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total = pnl_total + pnl_n
                        n_trades = n_trades + int32(1)
                        if pnl_n > float32(0.0):
                            n_wins = n_wins + int32(1)
                        pos_dir = int32(0)
                        trade_bars_held = int32(0)
                        trade_mfe_usd = float32(0.0)
                        trail_armed = False

        # Init extreme
        if math.isnan(extreme_price):
            extreme_price = c
            continue

        # Zigzag state machine
        pivot_confirmed = False
        new_pivot_dir = int32(0)
        if direction == int32(0):
            if c - extreme_price >= r_eff:
                pivot_confirmed = True
                new_pivot_dir = int32(-1)
                direction = int32(1)
                extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True
                new_pivot_dir = int32(1)
                direction = int32(-1)
                extreme_price = c
        elif direction == int32(1):
            if c > extreme_price:
                extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True
                new_pivot_dir = int32(1)
                direction = int32(-1)
                extreme_price = c
        else:  # direction == -1
            if c < extreme_price:
                extreme_price = c
            elif c - extreme_price >= r_eff:
                pivot_confirmed = True
                new_pivot_dir = int32(-1)
                direction = int32(1)
                extreme_price = c

        if not pivot_confirmed:
            continue
        if mins_of_day >= CUT_MINS:
            continue

        # Direction policy:
        #   flip_direction == 0 (default, "counter-trend on completed move"):
        #     HighPivot -> Short, LowPivot -> Long
        #   flip_direction == 1 ("trend-follow new direction"):
        #     HighPivot -> Long,  LowPivot -> Short
        if flip_direction == int32(0):
            action_side = int32(-1) if new_pivot_dir == int32(1) else int32(1)
        else:
            action_side = int32(1) if new_pivot_dir == int32(1) else int32(-1)

        # ALWAYS exit existing first
        if pos_dir != int32(0):
            slipped_px = next_open - float32(pos_dir) * SLIP
            pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
            pnl_n = pnl_pts * DPP - COMM
            pnl_total = pnl_total + pnl_n
            n_trades = n_trades + int32(1)
            if pnl_n > float32(0.0):
                n_wins = n_wins + int32(1)

        # Open new
        slipped_entry = next_open + float32(action_side) * SLIP
        pos_dir = action_side
        pos_entry_px = slipped_entry
        trade_bars_held = int32(0)
        trade_mfe_usd = float32(0.0)
        trail_armed = False

    # Final close at last bar (avoid leaving position open)
    if pos_dir != int32(0):
        last_close = float32(closes[T - 1])
        pnl_pts = float32(pos_dir) * (last_close - pos_entry_px)
        pnl_n = pnl_pts * DPP - COMM
        pnl_total = pnl_total + pnl_n
        n_trades = n_trades + int32(1)
        if pnl_n > float32(0.0):
            n_wins = n_wins + int32(1)

    out_pnl[p] = pnl_total
    out_n_trades[p] = n_trades
    out_n_wins[p] = n_wins


# =============================================================================
# numba.cuda simulation kernel — v106 (static R)
# =============================================================================

@cuda.jit
def simulate_kernel_v106(opens, closes, mins, params, flip_direction, out_pnl, out_n_trades, out_n_wins):
    """v1.0.6-RC simulation. params shape: (P, 6):
       [r_points, max_loss_pts, mfe_cut_bars, mfe_cut_usd, trail_activate_pts,
        trail_giveback_pct]"""
    p = cuda.grid(1)
    P = params.shape[0]
    if p >= P:
        return

    T = opens.shape[0]
    DPP = float32(2.0)
    COMM = float32(1.90)
    SLIP = float32(0.25)
    EOD_MINS = int32(1255)
    CUT_MINS = int32(1230)

    r_eff = float32(params[p, 0])
    max_loss_pts = float32(params[p, 1])
    mfe_cut_bars = int32(int(params[p, 2]))
    mfe_cut_usd = float32(params[p, 3])
    trail_activate_pts = float32(params[p, 4])
    trail_giveback_pct = float32(params[p, 5])

    direction = int32(0)
    extreme_price = float32(math.nan)
    pos_dir = int32(0)
    pos_entry_px = float32(0.0)
    trade_bars_held = int32(0)
    trade_mfe_usd = float32(0.0)
    trail_armed = False
    pnl_total = float32(0.0)
    n_trades = int32(0)
    n_wins = int32(0)

    for i in range(T - 1):
        c = float32(closes[i])
        next_open = float32(opens[i + 1])
        mins_of_day = int32(mins[i] + 1)

        if mins_of_day >= EOD_MINS:
            if pos_dir != int32(0):
                slipped_px = next_open - float32(pos_dir) * SLIP
                pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                pnl_n = pnl_pts * DPP - COMM
                pnl_total = pnl_total + pnl_n
                n_trades = n_trades + int32(1)
                if pnl_n > float32(0.0):
                    n_wins = n_wins + int32(1)
                pos_dir = int32(0)
                trade_bars_held = int32(0)
                trade_mfe_usd = float32(0.0)
                trail_armed = False
            continue

        if pos_dir != int32(0):
            trade_bars_held = trade_bars_held + int32(1)
            unrealized_pts = float32(pos_dir) * (c - pos_entry_px)
            unrealized_usd = unrealized_pts * DPP
            if unrealized_usd > trade_mfe_usd:
                trade_mfe_usd = unrealized_usd

            # Trail
            if trail_activate_pts > float32(0.0):
                activate_usd = trail_activate_pts * DPP
                if (not trail_armed) and (trade_mfe_usd >= activate_usd):
                    trail_armed = True
                if trail_armed:
                    trail_threshold = trade_mfe_usd * (float32(1.0) - trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        slipped_px = next_open - float32(pos_dir) * SLIP
                        pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total = pnl_total + pnl_n
                        n_trades = n_trades + int32(1)
                        if pnl_n > float32(0.0):
                            n_wins = n_wins + int32(1)
                        pos_dir = int32(0)
                        trade_bars_held = int32(0)
                        trade_mfe_usd = float32(0.0)
                        trail_armed = False

            # Hard SL
            if pos_dir != int32(0) and max_loss_pts > float32(0.0):
                unrealized_pts = float32(pos_dir) * (c - pos_entry_px)
                if unrealized_pts <= -max_loss_pts:
                    slipped_px = next_open - float32(pos_dir) * SLIP
                    pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                    pnl_n = pnl_pts * DPP - COMM
                    pnl_total = pnl_total + pnl_n
                    n_trades = n_trades + int32(1)
                    if pnl_n > float32(0.0):
                        n_wins = n_wins + int32(1)
                    pos_dir = int32(0)
                    trade_bars_held = int32(0)
                    trade_mfe_usd = float32(0.0)
                    trail_armed = False

            # MFE-cut
            if pos_dir != int32(0) and mfe_cut_bars > int32(0):
                if trade_bars_held == mfe_cut_bars:
                    if trade_mfe_usd <= mfe_cut_usd:
                        slipped_px = next_open - float32(pos_dir) * SLIP
                        pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
                        pnl_n = pnl_pts * DPP - COMM
                        pnl_total = pnl_total + pnl_n
                        n_trades = n_trades + int32(1)
                        if pnl_n > float32(0.0):
                            n_wins = n_wins + int32(1)
                        pos_dir = int32(0)
                        trade_bars_held = int32(0)
                        trade_mfe_usd = float32(0.0)
                        trail_armed = False

        if math.isnan(extreme_price):
            extreme_price = c
            continue

        pivot_confirmed = False
        new_pivot_dir = int32(0)
        if direction == int32(0):
            if c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = int32(-1)
                direction = int32(1); extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = int32(1)
                direction = int32(-1); extreme_price = c
        elif direction == int32(1):
            if c > extreme_price:
                extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = int32(1)
                direction = int32(-1); extreme_price = c
        else:
            if c < extreme_price:
                extreme_price = c
            elif c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = int32(-1)
                direction = int32(1); extreme_price = c

        if not pivot_confirmed:
            continue
        if mins_of_day >= CUT_MINS:
            continue

        if flip_direction == int32(0):
            action_side = int32(-1) if new_pivot_dir == int32(1) else int32(1)
        else:
            action_side = int32(1) if new_pivot_dir == int32(1) else int32(-1)
        if pos_dir != int32(0):
            slipped_px = next_open - float32(pos_dir) * SLIP
            pnl_pts = float32(pos_dir) * (slipped_px - pos_entry_px)
            pnl_n = pnl_pts * DPP - COMM
            pnl_total = pnl_total + pnl_n
            n_trades = n_trades + int32(1)
            if pnl_n > float32(0.0):
                n_wins = n_wins + int32(1)

        slipped_entry = next_open + float32(action_side) * SLIP
        pos_dir = action_side
        pos_entry_px = slipped_entry
        trade_bars_held = int32(0)
        trade_mfe_usd = float32(0.0)
        trail_armed = False

    if pos_dir != int32(0):
        last_close = float32(closes[T - 1])
        pnl_pts = float32(pos_dir) * (last_close - pos_entry_px)
        pnl_n = pnl_pts * DPP - COMM
        pnl_total = pnl_total + pnl_n
        n_trades = n_trades + int32(1)
        if pnl_n > float32(0.0):
            n_wins = n_wins + int32(1)

    out_pnl[p] = pnl_total
    out_n_trades[p] = n_trades
    out_n_wins[p] = n_wins


# =============================================================================
# Python wrappers around the kernels
# =============================================================================

# Stash this lookback range as a constant — must align with V107_BOUNDS[0]
ATR_LOOKBACK_MIN = 20
ATR_LOOKBACK_MAX = 240


def batched_simulate_v107(
    opens_d: torch.Tensor,
    closes_d: torch.Tensor,
    mins_d: torch.Tensor,
    atr_table: torch.Tensor,
    params: torch.Tensor,
    flip_direction: int = 0,
):
    """Launch the v107 kernel. Returns (pnl, n_trades, n_wins) torch tensors."""
    P = params.shape[0]
    out_pnl = torch.zeros(P, dtype=torch.float32, device=params.device)
    out_n_trades = torch.zeros(P, dtype=torch.int32, device=params.device)
    out_n_wins = torch.zeros(P, dtype=torch.int32, device=params.device)

    threads = 64
    blocks = (P + threads - 1) // threads
    simulate_kernel_v107[blocks, threads](
        cuda.as_cuda_array(opens_d),
        cuda.as_cuda_array(closes_d),
        cuda.as_cuda_array(mins_d),
        cuda.as_cuda_array(atr_table),
        ATR_LOOKBACK_MIN,
        cuda.as_cuda_array(params),
        np.int32(flip_direction),
        cuda.as_cuda_array(out_pnl),
        cuda.as_cuda_array(out_n_trades),
        cuda.as_cuda_array(out_n_wins),
    )
    cuda.synchronize()
    return out_pnl, out_n_trades, out_n_wins


def batched_simulate_v106(
    opens_d: torch.Tensor,
    closes_d: torch.Tensor,
    mins_d: torch.Tensor,
    params: torch.Tensor,
    flip_direction: int = 0,
):
    P = params.shape[0]
    out_pnl = torch.zeros(P, dtype=torch.float32, device=params.device)
    out_n_trades = torch.zeros(P, dtype=torch.int32, device=params.device)
    out_n_wins = torch.zeros(P, dtype=torch.int32, device=params.device)
    threads = 64
    blocks = (P + threads - 1) // threads
    simulate_kernel_v106[blocks, threads](
        cuda.as_cuda_array(opens_d),
        cuda.as_cuda_array(closes_d),
        cuda.as_cuda_array(mins_d),
        cuda.as_cuda_array(params),
        np.int32(flip_direction),
        cuda.as_cuda_array(out_pnl),
        cuda.as_cuda_array(out_n_trades),
        cuda.as_cuda_array(out_n_wins),
    )
    cuda.synchronize()
    return out_pnl, out_n_trades, out_n_wins


# =============================================================================
# Manual differential evolution on GPU (PyTorch tensors)
# =============================================================================

def de_evolve(
    fitness_fn,                # callable(params: (P,D) tensor) -> (P,) negative-pnl tensor
    bounds: list[tuple[float, float]],
    popsize_mult: int,
    maxiter: int,
    seed: int,
    device: torch.device,
    F_lo: float = 0.5,
    F_hi: float = 1.5,
    CR: float = 0.7,
    verbose: bool = True,
):
    """Manual rand/1/bin DE on GPU. Returns (best_pop_x, all_final_pop, all_final_fitness)."""
    rng = torch.Generator(device=device).manual_seed(seed)
    D = len(bounds)
    P = popsize_mult * D
    lo = torch.tensor([b[0] for b in bounds], device=device, dtype=torch.float32)
    hi = torch.tensor([b[1] for b in bounds], device=device, dtype=torch.float32)

    pop = lo + (hi - lo) * torch.rand((P, D), generator=rng, device=device)
    fitness = fitness_fn(pop)

    best_idx = torch.argmin(fitness)
    if verbose:
        print(f"  Gen   0/{maxiter}: best fitness = {-fitness[best_idx].item():+,.2f}")

    idx_arange = torch.arange(P, device=device)
    for gen in range(1, maxiter + 1):
        # Pick 3 distinct random indices for each i (rejection sampling — GA-grade is fine)
        a = torch.randint(0, P, (P,), generator=rng, device=device)
        b = torch.randint(0, P, (P,), generator=rng, device=device)
        c = torch.randint(0, P, (P,), generator=rng, device=device)

        F = F_lo + (F_hi - F_lo) * torch.rand((P, 1), generator=rng, device=device)
        trial = pop[a] + F * (pop[b] - pop[c])

        cr_mask = torch.rand((P, D), generator=rng, device=device) < CR
        forced = torch.randint(0, D, (P,), generator=rng, device=device)
        cr_mask[idx_arange, forced] = True

        candidates = torch.where(cr_mask, trial, pop)
        candidates = torch.maximum(candidates, lo)
        candidates = torch.minimum(candidates, hi)

        new_fitness = fitness_fn(candidates)
        better = new_fitness < fitness
        pop = torch.where(better.unsqueeze(1), candidates, pop)
        fitness = torch.where(better, new_fitness, fitness)

        best_idx = torch.argmin(fitness)
        if verbose and (gen % 5 == 0 or gen == maxiter):
            print(f"  Gen {gen:3d}/{maxiter}: best fitness = {-fitness[best_idx].item():+,.2f}")

    return pop[best_idx].clone(), pop, fitness


# =============================================================================
# Parity test
# =============================================================================

def parity_test(version: str, atlas_root: str = "DATA/ATLAS", n_combos: int = 5, seed: int = 7):
    """Sanity check: run N random param combos through both CPU and CUDA sims,
    compare net PnL. Should match within ~$10 tolerance (float32 vs float64
    drift over 314k bars)."""
    print("=" * 80)
    print(f"PARITY TEST: CPU vs CUDA simulation (v{version})")
    print("=" * 80)
    device = cuda_setup()

    bars = load_1m_bars(atlas_root)
    # Use last 60 days for parity (fast)
    bars = bars.tail(60 * 1440).reset_index(drop=True)
    print(f"Test window: {len(bars):,} 1m bars  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})")

    bounds = V107_BOUNDS if version == "v107" else V106_BOUNDS
    rng = np.random.default_rng(seed)
    test_vectors = []
    for _ in range(n_combos):
        v = np.array([rng.uniform(b[0], b[1]) for b in bounds], dtype=np.float64)
        test_vectors.append(v)

    # CPU run
    print("\n[CPU baseline]")
    if version == "v107":
        cpu_combos = [vector_to_params_v107(x) for x in test_vectors]
    else:
        cpu_combos = [vector_to_params_v106(x) for x in test_vectors]
    df_cpu = evaluate_combos(bars, cpu_combos, version)
    print(df_cpu[["pnl_net", "n_trades", "n_wins"]].to_string(index=False))

    # CUDA run
    print("\n[CUDA]")
    opens_d = torch.from_numpy(bars["open"].to_numpy(dtype=np.float32)).to(device)
    closes_d = torch.from_numpy(bars["close"].to_numpy(dtype=np.float32)).to(device)
    mins_d = torch.from_numpy(bars["mins_of_day_utc"].to_numpy(dtype=np.int32)).to(device)
    if version == "v107":
        highs_d = torch.from_numpy(bars["high"].to_numpy(dtype=np.float32)).to(device)
        lows_d = torch.from_numpy(bars["low"].to_numpy(dtype=np.float32)).to(device)
        tr_d = compute_true_range_torch(highs_d, lows_d, closes_d)
        atr_table = precompute_atr_table(tr_d, ATR_LOOKBACK_MIN, ATR_LOOKBACK_MAX)
    params_np = np.stack(test_vectors).astype(np.float32)
    # For v107, swap order: params layout for kernel uses int(round(atr_lookback))
    # Round int-valued dims to integers
    if version == "v107":
        params_np[:, 0] = np.round(params_np[:, 0])  # atr_lookback
        params_np[:, 5] = np.round(params_np[:, 5])  # mfe_cut_bars
    else:
        params_np[:, 2] = np.round(params_np[:, 2])  # mfe_cut_bars
    params_d = torch.from_numpy(params_np).to(device)

    if version == "v107":
        pnl_cuda, ntr_cuda, nw_cuda = batched_simulate_v107(
            opens_d, closes_d, mins_d, atr_table, params_d, flip_direction=0
        )
    else:
        pnl_cuda, ntr_cuda, nw_cuda = batched_simulate_v106(
            opens_d, closes_d, mins_d, params_d, flip_direction=0
        )
    pnl_cuda = pnl_cuda.cpu().numpy()
    ntr_cuda = ntr_cuda.cpu().numpy()
    nw_cuda = nw_cuda.cpu().numpy()

    # Compare
    print("\n[COMPARISON]")
    print(f"{'#':>3}  {'CPU $':>12}  {'CUDA $':>12}  {'diff $':>10}  {'CPU n':>6}  {'CUDA n':>6}  {'diff n':>6}")
    max_diff_pnl = 0.0
    max_diff_ntr = 0
    for i in range(n_combos):
        cpu_p = df_cpu["pnl_net"].iloc[i]
        cuda_p = pnl_cuda[i]
        cpu_n = int(df_cpu["n_trades"].iloc[i])
        cuda_n = int(ntr_cuda[i])
        diff_p = cuda_p - cpu_p
        diff_n = cuda_n - cpu_n
        max_diff_pnl = max(max_diff_pnl, abs(diff_p))
        max_diff_ntr = max(max_diff_ntr, abs(diff_n))
        print(f"{i:>3}  {cpu_p:>+12.2f}  {cuda_p:>+12.2f}  {diff_p:>+10.2f}  {cpu_n:>6}  {cuda_n:>6}  {diff_n:>+6}")

    print(f"\nMax |diff PnL|:   ${max_diff_pnl:.2f}")
    print(f"Max |diff trades|: {max_diff_ntr}")
    pnl_tol = 50.0  # float32 drift over 314k bars
    ntr_tol = 5     # at most a handful of edge-case trades may differ
    if max_diff_pnl <= pnl_tol and max_diff_ntr <= ntr_tol:
        print(f"PARITY OK (within tol: ${pnl_tol} PnL, {ntr_tol} trades)")
        return True
    else:
        print(f"PARITY FAIL (tol exceeded: ${pnl_tol} PnL, {ntr_tol} trades)")
        return False


# =============================================================================
# Main GA driver
# =============================================================================

def run_ga(version: str, args):
    gc.collect()
    pd.reset_option("display.float_format", silent=True) if hasattr(pd, "reset_option") else None

    print("=" * 80)
    print(f"PYTHON GENETIC OPTIMIZATION (CUDA) — ZigzagRunner_{version}")
    print("=" * 80)
    print(f"Atlas:        {args.atlas}")
    print(f"Train end:    {args.train_end}")
    print(f"Holdout from: {args.holdout_start}")
    print(f"Maxiter:      {args.maxiter}")
    print(f"Popsize mult: {args.popsize}")
    print(f"Seed:         {args.seed}")
    print()

    device = cuda_setup()

    bars = load_1m_bars(args.atlas)
    print(f"Loaded {len(bars):,} 1m bars")
    bars_train, bars_holdout = split_train_holdout(bars, args.train_end, args.holdout_start)
    print(f"  Train:    {len(bars_train):,} bars  ({bars_train['dt_utc'].iloc[0]} -> {bars_train['dt_utc'].iloc[-1]})")
    if len(bars_holdout) > 0:
        print(f"  Holdout:  {len(bars_holdout):,} bars  ({bars_holdout['dt_utc'].iloc[0]} -> {bars_holdout['dt_utc'].iloc[-1]})")
    else:
        print(f"  Holdout:  EMPTY")
    print()

    # Move TRAIN data to GPU
    opens_d = torch.from_numpy(bars_train["open"].to_numpy(dtype=np.float32)).to(device)
    closes_d = torch.from_numpy(bars_train["close"].to_numpy(dtype=np.float32)).to(device)
    mins_d = torch.from_numpy(bars_train["mins_of_day_utc"].to_numpy(dtype=np.int32)).to(device)
    if version == "v107":
        highs_d = torch.from_numpy(bars_train["high"].to_numpy(dtype=np.float32)).to(device)
        lows_d = torch.from_numpy(bars_train["low"].to_numpy(dtype=np.float32)).to(device)
        tr_d = compute_true_range_torch(highs_d, lows_d, closes_d)
        print(f"Pre-computing ATR table for lookbacks [{ATR_LOOKBACK_MIN}, {ATR_LOOKBACK_MAX}]...")
        atr_table = precompute_atr_table(tr_d, ATR_LOOKBACK_MIN, ATR_LOOKBACK_MAX)
        atr_mb = atr_table.numel() * 4 / 1024 / 1024
        print(f"  ATR table shape: {tuple(atr_table.shape)}  ({atr_mb:.1f} MB)")
    print()

    bounds = V107_BOUNDS if version == "v107" else V106_BOUNDS
    names = V107_NAMES if version == "v107" else V106_NAMES
    D = len(bounds)
    P = args.popsize * D
    print(f"DE: {D}-D space, popsize={args.popsize} (actual pop={P}), maxiter={args.maxiter}")

    # Build fitness closure (handles int rounding for kernel)
    flip_dir = 1 if args.flip_dir else 0
    if version == "v107":
        def fitness_fn(params):
            # Round int-valued params before sending to kernel
            params_rounded = params.clone()
            params_rounded[:, 0] = torch.round(params_rounded[:, 0])  # atr_lookback
            params_rounded[:, 5] = torch.round(params_rounded[:, 5])  # mfe_cut_bars
            pnl, _, _ = batched_simulate_v107(opens_d, closes_d, mins_d, atr_table, params_rounded, flip_dir)
            return -pnl
    else:
        def fitness_fn(params):
            params_rounded = params.clone()
            params_rounded[:, 2] = torch.round(params_rounded[:, 2])  # mfe_cut_bars
            pnl, _, _ = batched_simulate_v106(opens_d, closes_d, mins_d, params_rounded, flip_dir)
            return -pnl

    # JIT warmup (first kernel launch compiles)
    print("JIT warming up CUDA kernel...")
    t_warmup = time.time()
    dummy_params = torch.tensor([[(b[0]+b[1])/2 for b in bounds]], device=device, dtype=torch.float32)
    if version == "v107":
        dummy_params[:, 0] = torch.round(dummy_params[:, 0])
        dummy_params[:, 5] = torch.round(dummy_params[:, 5])
        _ = batched_simulate_v107(opens_d, closes_d, mins_d, atr_table, dummy_params, flip_dir)
    else:
        dummy_params[:, 2] = torch.round(dummy_params[:, 2])
        _ = batched_simulate_v106(opens_d, closes_d, mins_d, dummy_params, flip_dir)
    print(f"  Warmup took {time.time() - t_warmup:.1f} sec\n")

    # GA
    print(f"Starting DE...")
    t0 = time.time()
    best_x, final_pop, final_fitness = de_evolve(
        fitness_fn, bounds, args.popsize, args.maxiter, args.seed, device,
    )
    elapsed = time.time() - t0
    print(f"\nDE finished in {elapsed:.1f} sec")

    best_pnl = -final_fitness.min().item()
    print(f"  Best train PnL: ${best_pnl:+,.2f}")
    print(f"  Best vector:    {best_x.cpu().numpy()}")
    print()

    # Top-K extraction (sort the final population by fitness)
    sorted_idx = torch.argsort(final_fitness)
    K = min(args.top_k, P)
    top_pop = final_pop[sorted_idx[:K]]
    top_fitness = final_fitness[sorted_idx[:K]]
    top_x_np = top_pop.cpu().numpy()

    # Build top-K combos in CPU dict format
    top_combos = []
    for x in top_x_np:
        if version == "v107":
            top_combos.append(vector_to_params_v107(x))
        else:
            top_combos.append(vector_to_params_v106(x))

    # Re-evaluate top-K on TRAIN with CPU sim (float64) for verification + report
    print(f"Re-evaluating top-{K} on train (CPU float64 sim for accuracy)...")
    df_train = evaluate_combos(bars_train, top_combos, version, flip_direction=flip_dir).copy()
    df_train["rank_train"] = range(1, len(df_train) + 1)

    # Re-evaluate top-K on HOLDOUT
    if len(bars_holdout) > 0:
        print(f"Re-evaluating top-{K} on holdout...")
        df_holdout = evaluate_combos(bars_holdout, top_combos, version, flip_direction=flip_dir)
        for c in df_holdout.columns:
            if c not in names:
                df_train[f"holdout_{c}"] = df_holdout[c].values
    else:
        for c in ["holdout_pnl_net", "holdout_n_trades", "holdout_pnl_per_day", "holdout_win_rate"]:
            df_train[c] = np.nan

    df_train["pnl_drop_pct"] = (
        100.0 * (df_train["pnl_net"] - df_train["holdout_pnl_net"]) / df_train["pnl_net"].replace(0, np.nan)
    )
    df_train["robust_flag"] = (df_train.get("holdout_pnl_per_day", pd.Series(np.nan)) >= 30.0) & (df_train["pnl_drop_pct"] < 30.0)

    show_cols = names + [
        "pnl_net", "pnl_per_day", "n_trades", "win_rate",
        "holdout_pnl_net", "holdout_pnl_per_day", "holdout_n_trades", "holdout_win_rate",
        "pnl_drop_pct", "robust_flag",
    ]
    show_cols = [c for c in show_cols if c in df_train.columns]
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(f"\n=== TOP-{K} WITH HOLDOUT ===")
    print(df_train[show_cols].to_string(index=False))

    # Save
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("reports/findings", exist_ok=True)
    dir_tag = "_flipdir" if flip_dir == 1 else ""
    csv_path = f"reports/findings/{today}_genetic_cuda_{version}{dir_tag}_topk.csv"
    df_train.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    md_path = f"reports/findings/{today}_genetic_cuda_{version}{dir_tag}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# CUDA Genetic optimization report — ZigzagRunner_{version}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Atlas: `{args.atlas}`\n")
        f.write(f"- Train: through `{args.train_end}` ({len(bars_train):,} 1m bars)\n")
        f.write(f"- Holdout: from `{args.holdout_start}` ({len(bars_holdout):,} 1m bars)\n")
        f.write(f"- Optimizer: numba.cuda kernel + manual rand/1/bin DE on GPU\n")
        f.write(f"- Search dim: {D}, popsize_mult: {args.popsize} (actual P={P}), maxiter: {args.maxiter}, seed: {args.seed}\n")
        f.write(f"- DE wall time: {elapsed:.1f} sec\n")
        f.write(f"- Best train PnL (float32 sim): ${best_pnl:+,.2f}\n\n")
        f.write(f"## Top-{K} candidates (CPU float64 verification + holdout)\n\n")
        try:
            f.write(df_train[show_cols].to_markdown(index=False, floatfmt=".2f"))
        except ImportError:
            # tabulate not installed — fall back to plain string
            f.write("```\n")
            f.write(df_train[show_cols].to_string(index=False))
            f.write("\n```\n")
        f.write("\n\n## Decision matrix\n\n")
        f.write("| Outcome | Action |\n|---|---|\n")
        f.write("| ≥1 row with `robust_flag=True` | Pick that combo. Holdout PnL ≥ $30/day AND drop < 30% from train. |\n")
        f.write("| All rows fail `robust_flag` | Strategy overfits the train window. **Stay on v1.0.4 baseline.** |\n")
        f.write("| Best holdout drop > 50% | Severe overfit. Re-run with stricter `--maxiter` or different `--seed`. |\n\n")
        f.write("## Caveats\n\n")
        f.write("- CUDA sim uses **float32** for speed; CPU verification uses **float64**. PnL drift typically $<10$ over 314k bars.\n")
        f.write("- 1m-resolution sim — hard SL fires at 1m close, not 1s like real NT8.\n")
        f.write("- Python ATR uses SMA, NT8 uses Wilder smoothing. Small but non-zero discrepancy.\n")
        f.write("- Prior parity work showed Python sim has ~2× trade count vs NT8 SA. Use rank-order to pick combos.\n")
    print(f"Wrote: {md_path}")

    # Cleanup
    del bars, bars_train, bars_holdout
    del opens_d, closes_d, mins_d
    if version == "v107":
        del highs_d, lows_d, tr_d, atr_table
    del final_pop, final_fitness, top_pop
    pd.reset_option("display.float_format", silent=True) if hasattr(pd, "reset_option") else None
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v106", "v107"], default="v107")
    ap.add_argument("--atlas", default="DATA/ATLAS")
    ap.add_argument("--train-end", default="2025-12-31")
    ap.add_argument("--holdout-start", default="2026-01-01")
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--popsize", type=int, default=30,
                    help="Population multiplier — actual pop = popsize * D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--flip-dir", action="store_true",
                    help="Flip direction policy: HighPivot->Long, LowPivot->Short "
                         "(trend-follow new direction). Default = counter-trend "
                         "(HighPivot->Short, LowPivot->Long, matches v1.0.4 baseline).")
    ap.add_argument("--parity-test", action="store_true",
                    help="Run CPU vs CUDA parity test instead of GA. Use BEFORE first real run.")
    args = ap.parse_args()

    if args.parity_test:
        ok = parity_test(args.version, args.atlas)
        sys.exit(0 if ok else 1)

    run_ga(args.version, args)


if __name__ == "__main__":
    main()
