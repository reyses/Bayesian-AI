"""
zigzag_genetic.py -- Python-side genetic optimizer for ZigzagRunner v1.0.6-RC
and v1.0.7-RC, with train/holdout cross-validation. Substitutes for NT8's
Strategy Analyzer GA when the NT8 instrument import is blocked.

Search algorithm: scipy.optimize.differential_evolution (genetic-style — uses
crossover/mutation/selection on a population). With workers=-1 it parallelizes
across all CPU cores.

Usage:
    # GA-A: v1.0.6-RC, 7-D static-R search
    python tools/zigzag_genetic.py --version v106 --maxiter 30 --popsize 30

    # GA-B: v1.0.7-RC, 12-D dynamic-R search
    python tools/zigzag_genetic.py --version v107 --maxiter 50 --popsize 40

    # Custom train/holdout split (default: train through 2025-12-31, holdout 2026-01-01+)
    python tools/zigzag_genetic.py --version v106 --train-end 2025-10-31 --holdout-start 2025-11-01

    # Keep top-10 (default 5) and re-run each on holdout
    python tools/zigzag_genetic.py --version v107 --top-k 10

Outputs:
    reports/findings/2026-04-28_genetic_<version>_train.csv     (every population eval, last gen)
    reports/findings/2026-04-28_genetic_<version>_topk.csv      (top-K candidates with holdout PnL)
    reports/findings/2026-04-28_genetic_<version>_report.md     (markdown summary)

Caveats:
    - 1m-resolution sim. Hard SL fires at 1m close, not 1s like real NT8. May
      OVERSTATE SL effectiveness (real fires earlier inside the bar). Numbers
      are directional, not authoritative for live PnL.
    - Python ATR uses simple moving average (SMA), not NT8's Wilder smoothing.
      Difference is small but non-zero. Documented in report.
    - Prior parity work showed Python sim has ~2x trade count vs NT8 SA. So
      $/day numbers may be inflated. Use rank-order to pick combos, not
      absolute $.
"""
from __future__ import annotations
import argparse
import gc
import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

DOLLAR_PER_POINT = 2.0   # MNQ
COMMISSION_RT = 1.90     # $/round-trip per contract
SLIPPAGE_PTS = 0.25      # 1 MNQ tick


# =============================================================================
# Data loading
# =============================================================================

def load_1m_bars(atlas_root: str = "DATA/ATLAS") -> pd.DataFrame:
    """Load all 1m parquet files from ATLAS_root/1m/ -> sorted DataFrame.
    Adds dt_utc, dt_local (PT), date, mins_of_day_utc helpers."""
    folder = Path(atlas_root) / "1m"
    if not folder.exists():
        raise FileNotFoundError(f"missing {folder}")
    parts = []
    for f in sorted(folder.glob("*.parquet")):
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"  WARN {f.name}: {e}", file=sys.stderr)
            continue
        if df.empty:
            continue
        parts.append(df)
    if not parts:
        raise RuntimeError(f"no rows loaded from {folder}")
    bars = pd.concat(parts, ignore_index=True)
    bars = bars.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    bars["dt_utc"] = pd.to_datetime(bars["timestamp"], unit="s", utc=True)
    bars["mins_of_day_utc"] = bars["dt_utc"].dt.hour * 60 + bars["dt_utc"].dt.minute
    bars["date"] = bars["dt_utc"].dt.date
    return bars


def split_train_holdout(bars: pd.DataFrame, train_end: str, holdout_start: str):
    """Split bars by ISO date strings."""
    train_end_ts = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    holdout_start_ts = pd.Timestamp(holdout_start, tz="UTC")
    bars_train = bars[bars["dt_utc"] <= train_end_ts].reset_index(drop=True)
    bars_holdout = bars[bars["dt_utc"] >= holdout_start_ts].reset_index(drop=True)
    return bars_train, bars_holdout


# =============================================================================
# True-range pre-computation
# =============================================================================

def compute_true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Standard TR = max(high-low, |high - prev_close|, |low - prev_close|)."""
    n = len(highs)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = highs[0] - lows[0]
    hl = highs[1:] - lows[1:]
    hp = np.abs(highs[1:] - closes[:-1])
    lp = np.abs(lows[1:] - closes[:-1])
    tr[1:] = np.maximum(np.maximum(hl, hp), lp)
    return tr


def rolling_sma_atr(tr: np.ndarray, lookback: int) -> np.ndarray:
    """Simple moving average ATR. NaN until lookback bars filled."""
    n = len(tr)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < lookback:
        return out
    cumsum = np.cumsum(np.insert(tr, 0, 0.0))
    out[lookback - 1:] = (cumsum[lookback:] - cumsum[:-lookback]) / lookback
    return out


# =============================================================================
# Simulation core (numpy-vectorized data, Python state machine)
# =============================================================================

def simulate(
    opens: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    mins_of_day_utc: np.ndarray,
    dates_int: np.ndarray,        # bar's date as int (e.g. 20260101) for day boundary detection
    atr_values: np.ndarray | None,  # None for v106, ATR-by-bar for v107
    *,
    # Common params
    r_points: float,
    max_loss_pts: float,
    mfe_cut_bars: int,
    mfe_cut_usd: float,
    trail_activate_pts: float,
    trail_giveback_pct: float,
    # v107 only (ignored if atr_values is None)
    use_dynamic_r: bool = False,
    atr_multiplier: float = 5.0,
    min_r_points: float = 5.0,
    max_r_points: float = 200.0,
    # Direction policy: 0 = counter-trend (HighPivot->Short, LowPivot->Long, default v1.0.4),
    #                   1 = trend-follow new direction (HighPivot->Long, LowPivot->Short)
    flip_direction: int = 0,
    # Schedule
    eod_mins: int = 20 * 60 + 55,
    cut_mins: int = 20 * 60 + 30,
    contracts: int = 1,
    slippage_pts: float = SLIPPAGE_PTS,
    commission_rt_usd: float = COMMISSION_RT,
) -> dict:
    """Run one simulation. Returns dict of metrics (no per-trade ledger to keep
    fitness eval cheap). For ledger output, call after best-combo identified."""
    n = len(closes)
    # Zigzag state
    direction = 0
    extreme_price = float("nan")
    # Position state
    pos_dir = 0
    pos_entry_px = 0.0
    # Per-trade
    trade_bars_held = 0
    trade_mfe_usd = 0.0
    trail_armed = False
    # Aggregates
    n_trades = 0
    n_wins = 0
    pnl_total = 0.0
    pnl_gross = 0.0
    daily_pnl = {}  # date_int -> cumulative pnl
    n_pivots = 0
    last_date = -1

    for i in range(n - 1):
        c = closes[i]
        next_open = opens[i + 1]
        mins_of_day = mins_of_day_utc[i] + 1  # close is 1 min after bar timestamp's start
        cur_date = dates_int[i]

        # Effective R for this bar
        if use_dynamic_r and atr_values is not None:
            atr_val = atr_values[i]
            if not np.isnan(atr_val) and atr_val > 0:
                r_eff = atr_val * atr_multiplier
                if r_eff < min_r_points: r_eff = min_r_points
                if r_eff > max_r_points: r_eff = max_r_points
            else:
                r_eff = r_points
        else:
            r_eff = r_points

        # EOD force-close
        if mins_of_day >= eod_mins:
            if pos_dir != 0:
                slipped_px = next_open - pos_dir * slippage_pts
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_g = pnl_pts * DOLLAR_PER_POINT * contracts
                pnl_n = pnl_g - commission_rt_usd
                pnl_total += pnl_n
                pnl_gross += pnl_g
                if pnl_n > 0: n_wins += 1
                n_trades += 1
                daily_pnl[cur_date] = daily_pnl.get(cur_date, 0.0) + pnl_n
                pos_dir = 0
                trade_bars_held = 0
                trade_mfe_usd = 0.0
                trail_armed = False
            continue

        # Update per-trade state
        if pos_dir != 0:
            trade_bars_held += 1
            unrealized_pts = pos_dir * (c - pos_entry_px)
            unrealized_usd = unrealized_pts * DOLLAR_PER_POINT * contracts
            if unrealized_usd > trade_mfe_usd:
                trade_mfe_usd = unrealized_usd

            # Rule 4: Trail
            if trail_activate_pts > 0:
                activate_usd = trail_activate_pts * DOLLAR_PER_POINT
                if not trail_armed and trade_mfe_usd >= activate_usd:
                    trail_armed = True
                if trail_armed:
                    trail_threshold = trade_mfe_usd * (1.0 - trail_giveback_pct)
                    if unrealized_usd <= trail_threshold:
                        slipped_px = next_open - pos_dir * slippage_pts
                        pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                        pnl_g = pnl_pts * DOLLAR_PER_POINT * contracts
                        pnl_n = pnl_g - commission_rt_usd
                        pnl_total += pnl_n
                        pnl_gross += pnl_g
                        if pnl_n > 0: n_wins += 1
                        n_trades += 1
                        daily_pnl[cur_date] = daily_pnl.get(cur_date, 0.0) + pnl_n
                        pos_dir = 0
                        trade_bars_held = 0
                        trade_mfe_usd = 0.0
                        trail_armed = False

            # Rule 1: Hard SL
            if pos_dir != 0 and max_loss_pts > 0 and unrealized_pts <= -max_loss_pts:
                slipped_px = next_open - pos_dir * slippage_pts
                pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                pnl_g = pnl_pts * DOLLAR_PER_POINT * contracts
                pnl_n = pnl_g - commission_rt_usd
                pnl_total += pnl_n
                pnl_gross += pnl_g
                if pnl_n > 0: n_wins += 1
                n_trades += 1
                daily_pnl[cur_date] = daily_pnl.get(cur_date, 0.0) + pnl_n
                pos_dir = 0
                trade_bars_held = 0
                trade_mfe_usd = 0.0
                trail_armed = False

            # Rule 2: MFE-cut at bar N
            if pos_dir != 0 and mfe_cut_bars > 0 and trade_bars_held == mfe_cut_bars:
                if trade_mfe_usd <= mfe_cut_usd:
                    slipped_px = next_open - pos_dir * slippage_pts
                    pnl_pts = pos_dir * (slipped_px - pos_entry_px)
                    pnl_g = pnl_pts * DOLLAR_PER_POINT * contracts
                    pnl_n = pnl_g - commission_rt_usd
                    pnl_total += pnl_n
                    pnl_gross += pnl_g
                    if pnl_n > 0: n_wins += 1
                    n_trades += 1
                    daily_pnl[cur_date] = daily_pnl.get(cur_date, 0.0) + pnl_n
                    pos_dir = 0
                    trade_bars_held = 0
                    trade_mfe_usd = 0.0
                    trail_armed = False

        # Init extreme on first bar
        if np.isnan(extreme_price):
            extreme_price = c
            continue

        # Zigzag state machine
        pivot_confirmed = False
        new_pivot_dir = 0
        if direction == 0:
            if c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = +1; extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = +1
                direction = -1; extreme_price = c
        elif direction == +1:
            if c > extreme_price:
                extreme_price = c
            elif extreme_price - c >= r_eff:
                pivot_confirmed = True; new_pivot_dir = +1
                direction = -1; extreme_price = c
        else:  # direction == -1
            if c < extreme_price:
                extreme_price = c
            elif c - extreme_price >= r_eff:
                pivot_confirmed = True; new_pivot_dir = -1
                direction = +1; extreme_price = c

        if not pivot_confirmed:
            continue
        n_pivots += 1

        # Past entry cutoff: skip entry (but no exit either — pivot just doesn't enter)
        if mins_of_day >= cut_mins:
            continue

        # Direction policy:
        #   flip_direction == 0 (default): HIGH pivot -> Short, LOW pivot -> Long  (v1.0.4 baseline)
        #   flip_direction == 1: HIGH pivot -> Long, LOW pivot -> Short            (trend-follow new direction)
        if flip_direction == 0:
            action_side = -1 if new_pivot_dir == +1 else +1
        else:
            action_side = +1 if new_pivot_dir == +1 else -1

        # ALWAYS exit existing first
        if pos_dir != 0:
            slipped_px = next_open - pos_dir * slippage_pts
            pnl_pts = pos_dir * (slipped_px - pos_entry_px)
            pnl_g = pnl_pts * DOLLAR_PER_POINT * contracts
            pnl_n = pnl_g - commission_rt_usd
            pnl_total += pnl_n
            pnl_gross += pnl_g
            if pnl_n > 0: n_wins += 1
            n_trades += 1
            daily_pnl[cur_date] = daily_pnl.get(cur_date, 0.0) + pnl_n

        # Open new
        slipped_entry = next_open + action_side * slippage_pts
        pos_dir = action_side
        pos_entry_px = slipped_entry
        trade_bars_held = 0
        trade_mfe_usd = 0.0
        trail_armed = False

    # Final close at last bar (avoid leaving a position open)
    if pos_dir != 0:
        last_close = closes[-1]
        pnl_pts = pos_dir * (last_close - pos_entry_px)
        pnl_g = pnl_pts * DOLLAR_PER_POINT * contracts
        pnl_n = pnl_g - commission_rt_usd
        pnl_total += pnl_n
        pnl_gross += pnl_g
        if pnl_n > 0: n_wins += 1
        n_trades += 1
        last_date_int = dates_int[-1]
        daily_pnl[last_date_int] = daily_pnl.get(last_date_int, 0.0) + pnl_n

    n_days = max(len(daily_pnl), 1)
    pos_days = sum(1 for v in daily_pnl.values() if v > 0)

    return {
        "n_trades": n_trades,
        "n_wins": n_wins,
        "n_pivots": n_pivots,
        "pnl_net": pnl_total,
        "pnl_gross": pnl_gross,
        "n_days": n_days,
        "pos_days": pos_days,
        "pnl_per_day": pnl_total / n_days,
        "win_rate": n_wins / max(n_trades, 1),
        "day_win_rate": pos_days / n_days,
    }


# =============================================================================
# GA wrappers
# =============================================================================

# Parameter bounds — exposed at module level so DE workers can pickle
V106_BOUNDS = [
    (15.0, 75.0),      # 0: r_points
    (30.0, 150.0),     # 1: max_loss_pts
    (0.0, 30.0),       # 2: mfe_cut_bars (rounded to int)
    (0.0, 50.0),       # 3: mfe_cut_usd
    (0.0, 50.0),       # 4: trail_activate_pts
    (0.05, 0.50),      # 5: trail_giveback_pct
]
V106_NAMES = ["r_points", "max_loss_pts", "mfe_cut_bars", "mfe_cut_usd",
              "trail_activate_pts", "trail_giveback_pct"]

V107_BOUNDS = [
    (20.0, 240.0),     # 0: atr_lookback (int)
    (0.5, 15.0),       # 1: atr_multiplier
    (5.0, 30.0),       # 2: min_r_points
    (50.0, 300.0),     # 3: max_r_points
    (30.0, 150.0),     # 4: max_loss_pts
    (0.0, 30.0),       # 5: mfe_cut_bars (int)
    (0.0, 50.0),       # 6: mfe_cut_usd
    (0.0, 50.0),       # 7: trail_activate_pts
    (0.05, 0.50),      # 8: trail_giveback_pct
]
V107_NAMES = ["atr_lookback", "atr_multiplier", "min_r_points", "max_r_points",
              "max_loss_pts", "mfe_cut_bars", "mfe_cut_usd",
              "trail_activate_pts", "trail_giveback_pct"]


def vector_to_params_v106(x: np.ndarray) -> dict:
    return {
        "r_points":           float(x[0]),
        "max_loss_pts":       float(x[1]),
        "mfe_cut_bars":       int(round(x[2])),
        "mfe_cut_usd":        float(x[3]),
        "trail_activate_pts": float(x[4]),
        "trail_giveback_pct": float(x[5]),
    }


def vector_to_params_v107(x: np.ndarray) -> dict:
    return {
        "atr_lookback":       int(round(x[0])),
        "atr_multiplier":     float(x[1]),
        "min_r_points":       float(x[2]),
        "max_r_points":       float(x[3]),
        "max_loss_pts":       float(x[4]),
        "mfe_cut_bars":       int(round(x[5])),
        "mfe_cut_usd":        float(x[6]),
        "trail_activate_pts": float(x[7]),
        "trail_giveback_pct": float(x[8]),
    }


# Fitness functions — receive arrays via scipy DE's `args=()` parameter.
# This pickles arrays ONCE per worker at pool startup (not per fitness call),
# and works correctly with Windows spawn-based multiprocessing where module
# globals do NOT propagate to child processes.

def fitness_v106(x, opens, closes, highs, lows, mins, dates, tr):
    """Negative net PnL — DE minimizes. tr is unused for v106 but kept in
    signature for symmetry with v107 caller."""
    p = vector_to_params_v106(x)
    res = simulate(
        opens, closes, highs, lows, mins, dates, None,
        r_points=p["r_points"],
        max_loss_pts=p["max_loss_pts"],
        mfe_cut_bars=p["mfe_cut_bars"],
        mfe_cut_usd=p["mfe_cut_usd"],
        trail_activate_pts=p["trail_activate_pts"],
        trail_giveback_pct=p["trail_giveback_pct"],
        use_dynamic_r=False,
    )
    return -res["pnl_net"]


def fitness_v107(x, opens, closes, highs, lows, mins, dates, tr):
    """Negative net PnL — DE minimizes. ATR recomputed each call from `tr`
    (a few ms per call; cheaper than maintaining a worker-local cache that
    can't be shared across processes anyway)."""
    p = vector_to_params_v107(x)
    atr_values = rolling_sma_atr(tr, p["atr_lookback"])
    res = simulate(
        opens, closes, highs, lows, mins, dates, atr_values,
        r_points=30.0,  # fallback during ATR warmup
        max_loss_pts=p["max_loss_pts"],
        mfe_cut_bars=p["mfe_cut_bars"],
        mfe_cut_usd=p["mfe_cut_usd"],
        trail_activate_pts=p["trail_activate_pts"],
        trail_giveback_pct=p["trail_giveback_pct"],
        use_dynamic_r=True,
        atr_multiplier=p["atr_multiplier"],
        min_r_points=p["min_r_points"],
        max_r_points=p["max_r_points"],
    )
    return -res["pnl_net"]


# =============================================================================
# Top-K extraction by re-sampling around the best
# =============================================================================

def evaluate_combos(
    bars: pd.DataFrame,
    combos: list[dict],
    version: str,
    flip_direction: int = 0,
) -> pd.DataFrame:
    """Run each combo on the given bars, return DataFrame with metrics per combo."""
    opens = bars["open"].to_numpy(dtype=np.float64)
    closes = bars["close"].to_numpy(dtype=np.float64)
    highs = bars["high"].to_numpy(dtype=np.float64)
    lows = bars["low"].to_numpy(dtype=np.float64)
    mins = bars["mins_of_day_utc"].to_numpy(dtype=np.int32)
    dates = (
        bars["dt_utc"].dt.year * 10000
        + bars["dt_utc"].dt.month * 100
        + bars["dt_utc"].dt.day
    ).to_numpy(dtype=np.int32)
    tr = compute_true_range(highs, lows, closes)

    rows = []
    for c in combos:
        if version == "v106":
            res = simulate(
                opens, closes, highs, lows, mins, dates, None,
                r_points=c["r_points"],
                max_loss_pts=c["max_loss_pts"],
                mfe_cut_bars=int(c["mfe_cut_bars"]),
                mfe_cut_usd=c["mfe_cut_usd"],
                trail_activate_pts=c["trail_activate_pts"],
                trail_giveback_pct=c["trail_giveback_pct"],
                use_dynamic_r=False,
                flip_direction=flip_direction,
            )
        else:  # v107
            atr_values = rolling_sma_atr(tr, int(c["atr_lookback"]))
            res = simulate(
                opens, closes, highs, lows, mins, dates, atr_values,
                r_points=30.0,
                max_loss_pts=c["max_loss_pts"],
                mfe_cut_bars=int(c["mfe_cut_bars"]),
                mfe_cut_usd=c["mfe_cut_usd"],
                trail_activate_pts=c["trail_activate_pts"],
                trail_giveback_pct=c["trail_giveback_pct"],
                use_dynamic_r=True,
                atr_multiplier=c["atr_multiplier"],
                min_r_points=c["min_r_points"],
                max_r_points=c["max_r_points"],
                flip_direction=flip_direction,
            )
        row = dict(c)
        row.update(res)
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def run_ga(version: str, args):
    # ── Cache/state cleanup at START ───────────────────────────────────
    # Defensive: clear any leftover objects from prior imports/sessions
    # before allocating large numpy arrays for the GA. No-op on fresh
    # process invocations but guarantees a clean baseline if this
    # function is ever called from a notebook or REPL session.
    gc.collect()
    pd.reset_option("display.float_format", silent=True) if hasattr(pd, "reset_option") else None

    print("=" * 80)
    print(f"PYTHON GENETIC OPTIMIZATION — ZigzagRunner_{version}")
    print("=" * 80)
    print(f"Atlas:        {args.atlas}")
    print(f"Train end:    {args.train_end}")
    print(f"Holdout from: {args.holdout_start}")
    print(f"Maxiter:      {args.maxiter}")
    print(f"Popsize:      {args.popsize}")
    print(f"Seed:         {args.seed}")
    print(f"Workers:      {args.workers if args.workers else 'all cores'}")
    print()

    bars = load_1m_bars(args.atlas)
    print(f"Loaded {len(bars):,} 1m bars  ({bars['dt_utc'].iloc[0]} -> {bars['dt_utc'].iloc[-1]})")
    bars_train, bars_holdout = split_train_holdout(bars, args.train_end, args.holdout_start)
    print(f"  Train:    {len(bars_train):,} bars  ({bars_train['dt_utc'].iloc[0]} -> {bars_train['dt_utc'].iloc[-1]})")
    if len(bars_holdout) > 0:
        print(f"  Holdout:  {len(bars_holdout):,} bars  ({bars_holdout['dt_utc'].iloc[0]} -> {bars_holdout['dt_utc'].iloc[-1]})")
    else:
        print(f"  Holdout:  EMPTY — adjust --holdout-start")
    print()

    # Pre-compute arrays for the train sim — these are passed to DE via args=()
    # so they're pickled once per worker process at pool startup (Windows spawn
    # safe — does NOT rely on module globals which don't propagate to workers).
    opens_t = bars_train["open"].to_numpy(dtype=np.float64)
    closes_t = bars_train["close"].to_numpy(dtype=np.float64)
    highs_t = bars_train["high"].to_numpy(dtype=np.float64)
    lows_t = bars_train["low"].to_numpy(dtype=np.float64)
    mins_t = bars_train["mins_of_day_utc"].to_numpy(dtype=np.int32)
    dates_t = (
        bars_train["dt_utc"].dt.year * 10000
        + bars_train["dt_utc"].dt.month * 100
        + bars_train["dt_utc"].dt.day
    ).to_numpy(dtype=np.int32)
    tr_t = compute_true_range(highs_t, lows_t, closes_t)

    bounds = V106_BOUNDS if version == "v106" else V107_BOUNDS
    fitness = fitness_v106 if version == "v106" else fitness_v107
    names = V106_NAMES if version == "v106" else V107_NAMES

    workers = args.workers if args.workers else -1

    print(f"Starting DE: {len(bounds)}-D space, popsize={args.popsize}, maxiter={args.maxiter}")
    t0 = time.time()
    result = differential_evolution(
        fitness,
        bounds=bounds,
        args=(opens_t, closes_t, highs_t, lows_t, mins_t, dates_t, tr_t),
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
    print(f"DE finished in {elapsed/60:.1f} min")
    print(f"  Best fitness (negative net PnL): {result.fun:.2f}")
    print(f"  Best vector: {result.x}")
    print()

    # The DE result only gives us the single best. To get top-K we need to
    # re-sample around the best vector. Strategy: take the final population
    # (DE doesn't expose it directly via scipy's API — but we can sample N
    # nearby points + the best itself).
    rng = np.random.default_rng(args.seed + 1)
    best_x = result.x.copy()
    candidates = [best_x.copy()]
    # Add small perturbations within bounds
    for _ in range(args.top_k * 4):
        pert = best_x + rng.normal(0, 0.05, len(best_x)) * np.array([b[1] - b[0] for b in bounds])
        for i, (lo, hi) in enumerate(bounds):
            pert[i] = max(lo, min(hi, pert[i]))
        candidates.append(pert)

    # Score all candidates on TRAIN
    print(f"Scoring {len(candidates)} candidates on train...")
    combos_train = []
    for x in candidates:
        if version == "v106":
            p = vector_to_params_v106(x)
        else:
            p = vector_to_params_v107(x)
        combos_train.append(p)

    df_train = evaluate_combos(bars_train, combos_train, version)
    df_train = df_train.sort_values("pnl_net", ascending=False).drop_duplicates(
        subset=names, keep="first"
    ).reset_index(drop=True)
    df_top = df_train.head(args.top_k).copy()
    df_top["rank_train"] = range(1, len(df_top) + 1)

    print(f"\nTop {args.top_k} on train:")
    print(df_top[names + ["pnl_net", "n_trades", "win_rate", "pnl_per_day"]].to_string(index=False))

    # Re-evaluate top-K on HOLDOUT
    if len(bars_holdout) > 0:
        print(f"\nRe-scoring top {args.top_k} on holdout...")
        df_holdout = evaluate_combos(
            bars_holdout,
            df_top[names].to_dict("records"),
            version,
        )
        df_holdout.columns = [c if c in names else f"holdout_{c}" for c in df_holdout.columns]
        # Merge by index (top-K is ordered, so just concat)
        df_top = pd.concat([df_top.reset_index(drop=True),
                            df_holdout.drop(columns=names).reset_index(drop=True)], axis=1)
    else:
        for c in ["holdout_pnl_net", "holdout_n_trades", "holdout_pnl_per_day", "holdout_win_rate", "holdout_day_win_rate"]:
            df_top[c] = np.nan

    # Compute drop and decision flags
    df_top["pnl_drop_pct"] = (
        100.0 * (df_top["pnl_net"] - df_top["holdout_pnl_net"]) / df_top["pnl_net"].replace(0, np.nan)
    )
    df_top["holdout_pnl_per_day"] = df_top.get("holdout_pnl_per_day", np.nan)
    df_top["robust_flag"] = (df_top["holdout_pnl_per_day"] >= 30.0) & (df_top["pnl_drop_pct"] < 30.0)

    print(f"\n=== TOP-K WITH HOLDOUT ===")
    show_cols = names + [
        "pnl_net", "pnl_per_day", "n_trades", "win_rate",
        "holdout_pnl_net", "holdout_pnl_per_day", "holdout_n_trades", "holdout_win_rate",
        "pnl_drop_pct", "robust_flag",
    ]
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(df_top[show_cols].to_string(index=False))

    # Save outputs
    today = datetime.now().strftime("%Y-%m-%d")
    os.makedirs("reports/findings", exist_ok=True)
    csv_path = f"reports/findings/{today}_genetic_{version}_topk.csv"
    df_top.to_csv(csv_path, index=False)
    print(f"\nWrote: {csv_path}")

    md_path = f"reports/findings/{today}_genetic_{version}_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Genetic optimization report — ZigzagRunner_{version}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M}\n\n")
        f.write(f"## Setup\n\n")
        f.write(f"- Atlas: `{args.atlas}`\n")
        f.write(f"- Train: through `{args.train_end}` ({len(bars_train):,} 1m bars)\n")
        f.write(f"- Holdout: from `{args.holdout_start}` ({len(bars_holdout):,} 1m bars)\n")
        f.write(f"- Optimizer: `scipy.optimize.differential_evolution`, popsize={args.popsize}, maxiter={args.maxiter}, seed={args.seed}\n")
        f.write(f"- Search space: {len(bounds)}-D\n")
        f.write(f"- Run time: {elapsed/60:.1f} min\n\n")
        f.write(f"## Best on train\n\n")
        f.write(f"- Net PnL: ${-result.fun:+,.2f}\n")
        f.write(f"- Vector: `{result.x}`\n\n")
        f.write(f"## Top-{args.top_k} candidates with holdout validation\n\n")
        f.write(df_top[show_cols].to_markdown(index=False, floatfmt=".2f"))
        f.write("\n\n## Decision matrix\n\n")
        f.write("| Outcome | Action |\n|---|---|\n")
        f.write("| ≥1 row with `robust_flag=True` | Pick that combo. Holdout PnL ≥ $30/day AND drop < 30% from train. |\n")
        f.write("| All rows fail `robust_flag` | Strategy overfits the train window. **Stay on v1.0.4 baseline.** |\n")
        f.write("| Best holdout drop > 50% | Severe overfit. Re-run with stricter `--maxiter` or different `--seed`. |\n\n")
        f.write("## Caveats\n\n")
        f.write("- 1m-resolution sim. Hard SL fires at 1m close, not 1s like real NT8. Numbers are directional.\n")
        f.write("- Python ATR uses simple moving average (SMA). NT8 uses Wilder smoothing. Difference is small for moderate N.\n")
        f.write("- Prior parity work showed Python sim has ~2× trade count vs NT8 SA. Use rank-order to pick combos, not absolute $.\n")
    print(f"Wrote: {md_path}")

    # ── Cache/state cleanup at END ─────────────────────────────────────
    # Release the large numpy arrays + DataFrames so a subsequent run
    # of run_ga() (e.g. driving v106 then v107 from the same Python
    # process) starts from a clean baseline. scipy's DE pool closes
    # itself on completion; this releases the data side.
    del bars, bars_train, bars_holdout
    del opens_t, closes_t, highs_t, lows_t, mins_t, dates_t, tr_t
    del result, df_train, df_top
    if "df_holdout" in locals():
        del df_holdout
    pd.reset_option("display.float_format", silent=True) if hasattr(pd, "reset_option") else None
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["v106", "v107"], default="v106",
                    help="Strategy version to optimize (default v106)")
    ap.add_argument("--atlas", default="DATA/ATLAS",
                    help="ATLAS root (default DATA/ATLAS — Databento 14 months)")
    ap.add_argument("--train-end", default="2025-12-31",
                    help="ISO date — last day INCLUDED in train (default 2025-12-31)")
    ap.add_argument("--holdout-start", default="2026-01-01",
                    help="ISO date — first day INCLUDED in holdout (default 2026-01-01)")
    ap.add_argument("--maxiter", type=int, default=30,
                    help="DE max generations (default 30)")
    ap.add_argument("--popsize", type=int, default=30,
                    help="DE population size multiplier (default 30 — actual N = popsize * D)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=5,
                    help="Top-K candidates to re-evaluate on holdout (default 5)")
    ap.add_argument("--workers", type=int, default=0,
                    help="DE parallel workers (0 = all cores, 1 = serial)")
    args = ap.parse_args()

    if args.workers == 0:
        args.workers = -1

    run_ga(args.version, args)


if __name__ == "__main__":
    main()
