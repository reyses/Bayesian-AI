"""Per-cell exit-threshold optimizer.

For each (regime, tier) cell, takes the cell's pnl_paths from regret labels
and grid-searches (tp_pts, sl_pts, gb_min, gb_keep, time_stop_bars) to find
the combination that maximizes summed simulated PnL.

Hierarchy fallback:
    cell n >= MIN_N_CELL          → use cell's own optimum
    cell n <  MIN_N_CELL          → use tier's optimum (pooled across regimes)
    tier n < MIN_N_TIER           → use universal optimum

Output: JSON config consumed by `exits.py` via `position.extras['thresholds']`.

Numba-accelerated. ~80,000 (combo × path) evals/sec without; ~5M/sec with.
"""
from __future__ import annotations

import os
import json
import pickle
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

from training.regret.regret import RegretLabel
from training.utils.state import REGIME_VOCAB


# ── Coarse grid (tuned for runtime; expand if optimizer is fast enough) ─

TP_GRID = [15.0, 25.0, 40.0, 60.0, 100.0, 200.0, 999.0]   # POINTS (1pt=$2)
SL_GRID = [3.0,  5.0,  8.0,  12.5, 20.0, 999.0]
GB_MIN_GRID = [15.0, 30.0, 60.0, 999.0]                       # $; 999 = disabled
GB_KEEP_GRID = [0.3, 0.5, 0.7]
TIME_STOP_GRID = [60, 180, 360, 540, 720]                       # 5s bars

# Hierarchical fallback thresholds
MIN_N_CELL = 60      # below this, use tier-only optimum
MIN_N_TIER = 30      # below this, use universal optimum

# Default fallback when even the universal pool is empty
DEFAULT_THRESHOLDS = {
    'tp_pts': 60.0, 'sl_pts': 12.5,
    'gb_min': 30.0, 'gb_keep': 0.5,
    'time_stop_bars': 360,
}


# ── Numba-accelerated simulator ──────────────────────────────────────────

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        def deco(f): return f
        return deco


@njit(cache=True)
def _simulate_one(path: np.ndarray, tp_usd: float, sl_usd: float,
                       gb_min: float, gb_keep: float,
                       time_stop_bars: int) -> float:
    """Mirror of regret.simulate_exit — numba-friendly inner loop.

    Returns realized PnL. tp_usd/sl_usd are in $ already (caller converts pts→$).
    """
    n = path.shape[0]
    if n == 0:
        return 0.0
    armed = False
    peak = -1e18
    cap = time_stop_bars if time_stop_bars < n else n - 1

    for i in range(cap + 1):
        v = path[i]
        if v > peak:
            peak = v
        if v >= tp_usd:
            return v
        if v <= sl_usd:
            return v
        if peak >= gb_min:
            armed = True
        if armed and v < peak * gb_keep:
            return v
    return path[cap]


@njit(cache=True)
def _grid_total(paths: np.ndarray, lengths: np.ndarray,
                     tp_usd: float, sl_usd: float, gb_min: float,
                     gb_keep: float, time_stop_bars: int) -> float:
    """Sum simulated PnL across a stack of paths for one threshold combo.

    `paths` is (M, H_max) zero-padded; `lengths[m]` is the real length of paths[m].
    """
    total = 0.0
    M = paths.shape[0]
    for m in range(M):
        sub = paths[m, :lengths[m]]
        total += _simulate_one(sub, tp_usd, sl_usd, gb_min, gb_keep,
                                      time_stop_bars)
    return total


@njit(cache=True)
def _grid_per_trade(paths: np.ndarray, lengths: np.ndarray,
                          tp_usd: float, sl_usd: float, gb_min: float,
                          gb_keep: float, time_stop_bars: int) -> np.ndarray:
    """Per-trade simulated PnL array for one threshold combo (numba-fast)."""
    M = paths.shape[0]
    out = np.empty(M, dtype=np.float64)
    for m in range(M):
        sub = paths[m, :lengths[m]]
        out[m] = _simulate_one(sub, tp_usd, sl_usd, gb_min, gb_keep,
                                      time_stop_bars)
    return out


# ── Path stacking helpers ────────────────────────────────────────────────

def _stack_paths(labels: List[RegretLabel]
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of variable-length pnl_paths to (M, H_max) padded array
    + lengths vector + day_idx vector (int factor codes for grouping).
    Drops labels without a stored path.
    """
    valid = [l for l in labels if l.pnl_path is not None]
    if not valid:
        return (np.zeros((0, 0), dtype=np.float32),
                  np.zeros(0, dtype=np.int64),
                  np.zeros(0, dtype=np.int64))
    H = max(len(l.pnl_path) for l in valid)
    M = len(valid)
    arr = np.zeros((M, H), dtype=np.float32)
    lens = np.zeros(M, dtype=np.int64)
    days = []
    for i, l in enumerate(valid):
        p = l.pnl_path
        arr[i, :len(p)] = p
        lens[i] = len(p)
        days.append(l.entry_day)
    # Factorize days to int codes for fast group-by
    unique_days = sorted(set(days))
    day_to_idx = {d: i for i, d in enumerate(unique_days)}
    day_idx = np.array([day_to_idx[d] for d in days], dtype=np.int64)
    return arr, lens, day_idx


# ── Optimizer ────────────────────────────────────────────────────────────

def _aggregate_objective(per_trade_pnl: np.ndarray, day_idx: np.ndarray,
                                    n_days: int, objective: str,
                                    trim: float = 0.10) -> float:
    """Reduce a per-trade PnL vector + day_idx → scalar objective.

    Objectives:
        total       : sum over all trades. Tail-event-dominated (legacy).
        median_day  : median of per-day PnL totals (robust to outlier days).
        trimmed_day : trimmed mean of per-day PnL totals (drop top/bottom `trim`).
    """
    if per_trade_pnl.size == 0:
        return 0.0
    if objective == 'total':
        return float(per_trade_pnl.sum())
    daily = np.bincount(day_idx, weights=per_trade_pnl, minlength=n_days)
    if objective == 'median_day':
        return float(np.median(daily))
    if objective == 'trimmed_day':
        if len(daily) < 4:
            return float(daily.mean()) if len(daily) else 0.0
        sorted_d = np.sort(daily)
        k = int(np.floor(len(sorted_d) * trim))
        return float(sorted_d[k:len(sorted_d)-k].mean()) if k > 0 else float(sorted_d.mean())
    raise ValueError(f'Unknown objective: {objective}')


def optimize_cell(labels_subset: List[RegretLabel],
                       tp_grid: List[float] = TP_GRID,
                       sl_grid: List[float] = SL_GRID,
                       gb_min_grid: List[float] = GB_MIN_GRID,
                       gb_keep_grid: List[float] = GB_KEEP_GRID,
                       ts_grid: List[int] = TIME_STOP_GRID,
                       progress: bool = False,
                       objective: str = 'median_day',
                       trim: float = 0.10,
                       ) -> Tuple[Dict, float]:
    """Find the threshold combo maximizing the chosen objective on the cell's paths."""
    paths_arr, lens, day_idx = _stack_paths(labels_subset)
    if len(paths_arr) == 0:
        return DEFAULT_THRESHOLDS.copy(), 0.0
    n_days = int(day_idx.max()) + 1 if len(day_idx) else 0

    best_score = -1e18
    best_combo = None
    iter_combos = []
    for tp in tp_grid:
        for sl in sl_grid:
            for gb_min in gb_min_grid:
                for gb_keep in gb_keep_grid:
                    for ts in ts_grid:
                        iter_combos.append((tp, sl, gb_min, gb_keep, ts))

    it = tqdm(iter_combos, desc='grid', leave=False) if progress else iter_combos
    for tp, sl, gb_min, gb_keep, ts in it:
        if objective == 'total':
            score = _grid_total(paths_arr, lens,
                                       tp * 2.0, -abs(sl) * 2.0,
                                       gb_min, gb_keep, int(ts))
        else:
            per_trade = _grid_per_trade(paths_arr, lens,
                                                  tp * 2.0, -abs(sl) * 2.0,
                                                  gb_min, gb_keep, int(ts))
            score = _aggregate_objective(per_trade, day_idx, n_days,
                                                  objective, trim=trim)
        if score > best_score:
            best_score = score
            best_combo = (tp, sl, gb_min, gb_keep, ts)

    tp, sl, gb_min, gb_keep, ts = best_combo
    return ({'tp_pts': tp, 'sl_pts': sl,
                'gb_min': gb_min, 'gb_keep': gb_keep,
                'time_stop_bars': int(ts)},
              float(best_score))


def optimize_all_cells(labels: Iterable[RegretLabel],
                              min_n_cell: int = MIN_N_CELL,
                              min_n_tier: int = MIN_N_TIER,
                              objective: str = 'median_day',
                              trim: float = 0.10,
                              ) -> Dict:
    """Build a hierarchical threshold map.

    Returns:
        {
            'cells':       {f'{regime}|{tier}': thresholds_dict},
            'tier_pools':  {tier_name: thresholds_dict},
            'universal':   thresholds_dict,
            'meta':        {n_cells, n_paths, ...},
        }
    """
    labels = list(labels)
    if not labels:
        return {
            'cells': {}, 'tier_pools': {},
            'universal': DEFAULT_THRESHOLDS.copy(), 'meta': {'n_paths': 0},
        }

    # Group
    by_cell = defaultdict(list)
    by_tier = defaultdict(list)
    for l in labels:
        by_cell[(int(l.entry_regime_idx), str(l.entry_tier))].append(l)
        by_tier[str(l.entry_tier)].append(l)

    # Universal first
    print(f'Optimizing universal pool ({len(labels)} paths) [objective={objective}]...')
    universal_thr, universal_pnl = optimize_cell(
        labels, progress=True, objective=objective, trim=trim)

    # Tier pools
    tier_thr = {}
    for tier, sub in by_tier.items():
        if len(sub) < min_n_tier:
            tier_thr[tier] = universal_thr.copy()
            print(f'  Tier {tier}: n={len(sub)} < {min_n_tier} -> universal')
            continue
        print(f'Optimizing tier pool {tier} ({len(sub)} paths)...')
        thr, _ = optimize_cell(sub, objective=objective, trim=trim)
        tier_thr[tier] = thr

    # Cells with hierarchical fallback
    cell_thr = {}
    for (regime, tier), sub in by_cell.items():
        key = f'{regime}|{tier}'
        if len(sub) < min_n_cell:
            cell_thr[key] = tier_thr.get(tier, universal_thr).copy()
            continue
        thr, _ = optimize_cell(sub, objective=objective, trim=trim)
        cell_thr[key] = thr

    return {
        'cells': cell_thr,
        'tier_pools': tier_thr,
        'universal': universal_thr,
        'meta': {
            'n_paths': len(labels),
            'n_cells': len(by_cell),
            'min_n_cell': min_n_cell,
            'min_n_tier': min_n_tier,
            'objective': objective,
            'trim': trim if objective == 'trimmed_day' else None,
            'tp_grid': list(TP_GRID),
            'sl_grid': list(SL_GRID),
            'gb_min_grid': list(GB_MIN_GRID),
            'gb_keep_grid': list(GB_KEEP_GRID),
            'time_stop_grid': list(TIME_STOP_GRID),
        },
    }


# ── Lookup helper for engine integration ─────────────────────────────────

def lookup_thresholds(threshold_map: Dict, regime: int, tier: str) -> Dict:
    """Resolve thresholds via the (cell -> tier -> universal) fallback chain."""
    key = f'{regime}|{tier}'
    cells = threshold_map.get('cells', {})
    if key in cells:
        return cells[key]
    pools = threshold_map.get('tier_pools', {})
    if tier in pools:
        return pools[tier]
    return threshold_map.get('universal', DEFAULT_THRESHOLDS).copy()


# ── CLI ────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Per-cell threshold optimizer')
    p.add_argument('--regret', type=str,
                       default='training_iso_v2/output/regret_labels.pkl')
    p.add_argument('--out', type=str,
                       default='training_iso_v2/output/thresholds.json')
    p.add_argument('--min-n-cell', type=int, default=MIN_N_CELL)
    p.add_argument('--min-n-tier', type=int, default=MIN_N_TIER)
    p.add_argument('--objective', type=str, default='median_day',
                       choices=['total', 'median_day', 'trimmed_day'],
                       help='Optimization criterion. median_day is robust to '
                              'outlier days; total maximizes summed PnL (legacy).')
    p.add_argument('--trim', type=float, default=0.10,
                       help='Trim fraction for trimmed_day objective (default 0.10)')
    return p.parse_args()


def main():
    args = _parse_args()
    print(f'Loading regret labels from {args.regret}...')
    with open(args.regret, 'rb') as f:
        labels = pickle.load(f)
    print(f'  {len(labels)} labels')
    if HAVE_NUMBA:
        print('  numba acceleration: ON')
    else:
        print('  numba acceleration: OFF (install numba for 50x speedup)')

    thr_map = optimize_all_cells(labels,
                                              min_n_cell=args.min_n_cell,
                                              min_n_tier=args.min_n_tier,
                                              objective=args.objective,
                                              trim=args.trim)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(thr_map, f, indent=2)
    print(f'Saved -> {args.out}')

    # Summary table
    print('\nUniversal:', thr_map['universal'])
    print('\nTier pools:')
    for t, thr in thr_map['tier_pools'].items():
        print(f'  {t:<18}: {thr}')
    print('\nCells (fitted, not pooled):')
    for k, thr in thr_map['cells'].items():
        # Count tells us if it was fitted vs fell back to a pool
        try:
            r, t = k.split('|', 1)
            r_name = REGIME_VOCAB[int(r)]
        except Exception:
            r_name = k
        print(f'  {k} ({r_name:<12} x {t:<14}): {thr}')


if __name__ == '__main__':
    main()
