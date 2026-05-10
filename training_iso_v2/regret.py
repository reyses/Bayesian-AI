"""Exit-focused regret analysis.

For each closed trade, replays the actual price path from entry over a fixed
horizon (regardless of when our engine actually exited) and produces a label
set that's the FOUNDATION for adaptive exit-threshold optimization:

    Label                       Meaning
    ──────────────────────────  ──────────────────────────────────────────────
    peak_pnl                    Max PnL achieved in [entry, entry+H]
    peak_bar                    5s-bar offset from entry where peak was hit
    time_to_peak_s              Same in seconds
    mae_pnl                     Max ADVERSE excursion (most-negative PnL)
    mae_bar                     5s-bar offset where MAE was hit
    capture_ratio               actual_pnl / peak_pnl  (0..1; 1 = perfect exit)
    optimal_pnl                 same as peak_pnl (single-exit-bar oracle)
    regret_pnl                  optimal_pnl - actual_pnl  (always >= 0)
    pnl_path                    np.ndarray (H_bars,) — full PnL trajectory
                                  for downstream threshold simulation
    final_pnl_at_horizon        PnL at entry+H (anchor for time-stop sim)

Exit-method counterfactuals (`simulate_exits`) accept the pnl_path and
return what each (tp_pts, sl_pts, giveback_min_peak, giveback_keep,
time_stop_bars) tuple would have realized — used by threshold_optimizer.py.

Horizon default = 60 minutes (720 5s bars). Long enough to capture
mean-reversion fade-outs and short trend rides, short enough to keep
labels well-defined within the trading day.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from training_iso_v2.ledger import ClosedTrade, TICK, TICK_VALUE


ATLAS_ROOT = 'DATA/ATLAS'
HORIZON_BARS = 720         # 60 min × 12 bars/min (5s anchor)
TICKS_PER_DOLLAR = 1.0 / TICK_VALUE


@dataclass
class RegretLabel:
    """Per-trade regret label set. Pickle-friendly."""
    # Identity (back-references the source trade)
    entry_day: str
    entry_ts: float
    entry_price: float
    direction: str
    entry_tier: str
    entry_regime_idx: int

    # Actuals (from the original closed trade)
    actual_pnl: float
    actual_bars_held: int
    actual_exit_reason: str

    # Peak / MAE
    peak_pnl: float
    peak_bar: int
    time_to_peak_s: float
    mae_pnl: float
    mae_bar: int

    # Derived
    capture_ratio: float       # actual_pnl / peak_pnl, undefined when peak<=0
    optimal_pnl: float
    regret_pnl: float          # optimal_pnl - actual_pnl
    final_pnl_at_horizon: float

    # Path for downstream simulation (npz-pickled, optionally None)
    pnl_path: Optional[np.ndarray] = None  # (H,) float32

    # Misc
    extras: Dict = field(default_factory=dict)


def _load_5s_ohlcv(day: str, atlas_root: str = ATLAS_ROOT) -> Optional[pd.DataFrame]:
    path = os.path.join(atlas_root, '5s', f'{day}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _compute_pnl_path(closes: np.ndarray, entry_price: float,
                            direction: str) -> np.ndarray:
    """Per-bar PnL in $ (one contract). Long = (close-entry); short = inverse."""
    if direction == 'long':
        delta = closes - entry_price
    else:
        delta = entry_price - closes
    return (delta / TICK * TICK_VALUE).astype(np.float32)


def label_trade(trade: ClosedTrade, ohlcv_5s_by_day: Dict[str, pd.DataFrame],
                  horizon_bars: int = HORIZON_BARS,
                  store_path: bool = True) -> Optional[RegretLabel]:
    """Compute regret label for one trade. Returns None if data missing."""
    df = ohlcv_5s_by_day.get(trade.entry_day)
    if df is None:
        return None

    ts_arr = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(np.float64)

    entry_idx = int(np.searchsorted(ts_arr, int(trade.entry_ts), side='left'))
    if entry_idx >= len(ts_arr):
        return None

    end_idx = min(entry_idx + horizon_bars, len(ts_arr))
    if end_idx - entry_idx < 2:
        return None

    closes = close[entry_idx:end_idx]
    pnl = _compute_pnl_path(closes, trade.entry_price, trade.direction)
    if len(pnl) == 0:
        return None

    peak_bar = int(pnl.argmax())
    peak_pnl = float(pnl[peak_bar])
    mae_bar = int(pnl.argmin())
    mae_pnl = float(pnl[mae_bar])
    final_pnl = float(pnl[-1])
    capture = (trade.pnl / peak_pnl) if peak_pnl > 0 else float('nan')
    optimal_pnl = max(peak_pnl, 0.0)
    regret = optimal_pnl - trade.pnl

    return RegretLabel(
        entry_day=trade.entry_day,
        entry_ts=trade.entry_ts,
        entry_price=trade.entry_price,
        direction=trade.direction,
        entry_tier=trade.entry_tier,
        entry_regime_idx=trade.entry_regime_idx,
        actual_pnl=trade.pnl,
        actual_bars_held=trade.bars_held,
        actual_exit_reason=trade.exit_reason,
        peak_pnl=peak_pnl,
        peak_bar=peak_bar,
        time_to_peak_s=float(peak_bar * 5.0),
        mae_pnl=mae_pnl,
        mae_bar=mae_bar,
        capture_ratio=float(capture),
        optimal_pnl=float(optimal_pnl),
        regret_pnl=float(regret),
        final_pnl_at_horizon=final_pnl,
        pnl_path=(pnl.astype(np.float32) if store_path else None),
    )


def label_trades(trades: Iterable[ClosedTrade],
                       atlas_root: str = ATLAS_ROOT,
                       horizon_bars: int = HORIZON_BARS,
                       store_path: bool = True) -> List[RegretLabel]:
    """Batch-label a list of closed trades. Loads each day's 5s OHLCV once."""
    trades_list = list(trades)
    days = sorted({t.entry_day for t in trades_list})
    ohlcv_by_day: Dict[str, pd.DataFrame] = {}
    for day in tqdm(days, desc='regret: load 5s'):
        df = _load_5s_ohlcv(day, atlas_root)
        if df is not None:
            ohlcv_by_day[day] = df

    labels: List[RegretLabel] = []
    for t in tqdm(trades_list, desc='regret: label trades'):
        lbl = label_trade(t, ohlcv_by_day, horizon_bars=horizon_bars,
                                store_path=store_path)
        if lbl is not None:
            labels.append(lbl)
    return labels


# ─────────────────────────────────────────────────────────────────────────
# Counterfactual exit simulation — used by the threshold optimizer
# ─────────────────────────────────────────────────────────────────────────

def simulate_exit(pnl_path: np.ndarray,
                       tp_pts: float = float('inf'),
                       sl_pts: float = float('inf'),
                       giveback_min_peak: float = float('inf'),
                       giveback_keep: float = 0.5,
                       time_stop_bars: int = 720,
                       ) -> Tuple[float, int, str]:
    """Simulate a (tp, sl, giveback, time_stop) exit policy on a pnl_path.

    Args (all in $ per contract; tp_pts/sl_pts internally convert from $/pt):
        pnl_path           : np.ndarray of per-bar PnL in $.
        tp_pts             : take-profit in POINTS (MNQ: 1pt = $2). Inf = disabled.
        sl_pts             : stop-loss   in POINTS. Inf = disabled.
        giveback_min_peak  : arm giveback only after peak_pnl >= this $ amount.
        giveback_keep      : exit if pnl < peak * keep (after armed).
        time_stop_bars     : force-close at this bar offset.

    Returns:
        (exit_pnl, exit_bar, reason)
    """
    n = len(pnl_path)
    if n == 0:
        return 0.0, 0, 'empty'
    tp_usd = tp_pts * 2.0
    sl_usd = -abs(sl_pts) * 2.0
    armed = False
    peak = -float('inf')
    cap = min(time_stop_bars, n - 1)

    for i in range(cap + 1):
        v = float(pnl_path[i])
        if v > peak:
            peak = v
        if v >= tp_usd:
            return v, i, 'tp'
        if v <= sl_usd:
            return v, i, 'sl'
        if peak >= giveback_min_peak:
            armed = True
        if armed and v < peak * giveback_keep:
            return v, i, 'giveback'
    # Time stop or end of path
    last = min(cap, n - 1)
    return float(pnl_path[last]), last, 'time_stop'


def simulate_exits_for_grid(pnl_paths: List[np.ndarray],
                                       tp_grid: List[float],
                                       sl_grid: List[float],
                                       gb_min_grid: List[float],
                                       gb_keep_grid: List[float],
                                       time_stop_grid: List[int],
                                       ) -> pd.DataFrame:
    """Evaluate every (tp, sl, gb_min, gb_keep, time_stop) combo on a list
    of pnl_paths. Returns DataFrame with one row per combo + total/mean PnL.

    For threshold_optimizer use. O(|combos| × |paths| × H) — keep grids coarse.
    """
    rows = []
    n_paths = len(pnl_paths)
    for tp in tp_grid:
        for sl in sl_grid:
            for gb_min in gb_min_grid:
                for gb_keep in gb_keep_grid:
                    for ts in time_stop_grid:
                        total = 0.0
                        n_ok = 0
                        for path in pnl_paths:
                            pnl, _, _ = simulate_exit(
                                path, tp_pts=tp, sl_pts=sl,
                                giveback_min_peak=gb_min,
                                giveback_keep=gb_keep,
                                time_stop_bars=ts,
                            )
                            total += pnl
                            n_ok += 1
                        rows.append({
                            'tp_pts': tp, 'sl_pts': sl,
                            'gb_min': gb_min, 'gb_keep': gb_keep,
                            'time_stop_bars': ts,
                            'total_pnl': total,
                            'mean_pnl': total / max(n_ok, 1),
                            'n_paths': n_ok,
                        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description='Exit-focused regret labeller')
    p.add_argument('--trades', type=str,
                       default='training_iso_v2/output/is.pkl',
                       help='Path to closed-trades pickle')
    p.add_argument('--out', type=str,
                       default='training_iso_v2/output/regret_labels.pkl',
                       help='Output regret-labels pickle')
    p.add_argument('--horizon-bars', type=int, default=HORIZON_BARS)
    p.add_argument('--no-path', action='store_true',
                       help='Drop pnl_path arrays (saves disk; loses '
                              'optimization grid input)')
    return p.parse_args()


def main():
    args = _parse_args()
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    if not trades:
        print(f'No trades in {args.trades}')
        return
    print(f'Loaded {len(trades)} trades from {args.trades}')

    labels = label_trades(trades, horizon_bars=args.horizon_bars,
                                store_path=not args.no_path)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(labels, f)
    print(f'Saved {len(labels)} regret labels -> {args.out}')

    # Quick summary
    if labels:
        df = pd.DataFrame([{
            'tier': l.entry_tier,
            'regime': l.entry_regime_idx,
            'actual_pnl': l.actual_pnl,
            'peak_pnl': l.peak_pnl,
            'capture_ratio': l.capture_ratio,
            'regret_pnl': l.regret_pnl,
            'time_to_peak_s': l.time_to_peak_s,
        } for l in labels])
        print('\nRegret summary by tier:')
        agg = df.groupby('tier').agg(
            n=('actual_pnl', 'size'),
            mean_actual=('actual_pnl', 'mean'),
            mean_peak=('peak_pnl', 'mean'),
            mean_regret=('regret_pnl', 'mean'),
            median_capture=('capture_ratio', 'median'),
            median_ttp_s=('time_to_peak_s', 'median'),
        ).round(2)
        print(agg.to_string())


if __name__ == '__main__':
    main()
