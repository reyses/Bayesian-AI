"""Full counterfactual regret analysis — V2-native.

For every trade, computes what WOULD have happened under each of 6 alternative
actions, plus the original actual outcome. Picks the BEST_ACTION and
quantifies the REGRET (gap between actual and best).

The 6 alternative actions:

    SAME_EARLY       exit at the optimal bar BEFORE actual exit (capture peak)
    SAME_AT_EXIT     actual exit (baseline)
    SAME_EXTENDED    hold past actual exit, exit at peak in the extended window
    COUNTER_EARLY    flip direction at bar 1-3, exit at flip-peak
    COUNTER_AT_EXIT  flip direction at the actual exit bar
    COUNTER_EXTENDED flip at entry and hold to flip-peak

Plus EARLY_ENTRY: for each bar in [entry-LOOKBACK, entry], what if we'd
entered there in same/counter direction and held to its respective peak?

This is the legacy `nn_v2/regret.py` ported to V2-native at 5s anchor
resolution. Output is the LABEL set used to train CNN flip/hold/risk
classifiers and to identify systematic exit-timing problems.

Usage:
    python -m training_v2.regret_full --trades training_v2/output/nmp_only.pkl \
        --out training_v2/output/regret_full.pkl

Output: pickle of FullRegretLabel objects, one per trade.
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from training_v2.ledger import ClosedTrade, TICK, TICK_VALUE
from training_v2.regret import _load_5s_ohlcv, _compute_pnl_path


ATLAS_ROOT = 'DATA/ATLAS'
LOOKBACK_BARS = 360       # 30 min at 5s — early-entry window
LOOKAHEAD_BARS = 720      # 60 min at 5s — same/counter extended window


# ─── Action options ──────────────────────────────────────────────────────

ACTIONS = ('same_early', 'same_at_exit', 'same_extended',
                'counter_early', 'counter_at_exit', 'counter_extended')


# ─── Data container ──────────────────────────────────────────────────────

@dataclass
class FullRegretLabel:
    """Full multi-axis regret label per trade."""
    # Identity
    entry_day: str
    entry_ts: float
    entry_price: float
    direction: str
    entry_tier: str
    entry_regime_idx: int

    # Actuals
    actual_pnl: float
    actual_bars_held: int
    actual_exit_reason: str

    # SAME-direction action options
    same_early_best: float       # max PnL strictly before actual exit
    same_early_bar: int          # bar offset where it occurred
    same_at_exit: float          # PnL at actual exit bar
    same_extended_best: float    # max PnL strictly after actual exit
    same_extended_bar: int       # bar offset (from entry)
    same_best_overall: float     # max over the full path
    same_best_bar: int

    # COUNTER-direction action options
    counter_early_best: float    # max counter PnL in first 3 bars (flip-and-go)
    counter_early_bar: int
    counter_at_exit: float       # counter PnL at the actual exit bar
    counter_extended_best: float # max counter PnL after the actual exit
    counter_extended_bar: int
    counter_best_overall: float
    counter_best_bar: int

    # Decision
    best_action: str             # one of ACTIONS
    best_pnl: float
    regret: float                # best_pnl - actual_pnl  (>= 0)

    # Early-entry counterfactual (best of LOOKBACK_BARS pre-entry bars)
    early_entry_gain: float      # extra $ if we'd entered at the best earlier bar
    early_best_bars_before: int
    early_best_same_peak: float
    early_best_counter_peak: float

    # Path-level summaries (already in legacy RegretLabel; preserved here)
    peak_pnl: float
    peak_bar: int
    mae_pnl: float
    mae_bar: int
    capture_ratio: float
    final_pnl_at_horizon: float

    # Optional: full SAME and COUNTER curves (for downstream training)
    same_curve: Optional[np.ndarray] = None    # (LOOKAHEAD_BARS,) float32
    counter_curve: Optional[np.ndarray] = None
    pnl_path: Optional[np.ndarray] = None      # alias for same_curve, legacy compat

    # Misc
    extras: Dict = field(default_factory=dict)


# ─── Core compute ────────────────────────────────────────────────────────

def _empty_label(trade: ClosedTrade) -> FullRegretLabel:
    return FullRegretLabel(
        entry_day=trade.entry_day, entry_ts=trade.entry_ts,
        entry_price=trade.entry_price, direction=trade.direction,
        entry_tier=trade.entry_tier, entry_regime_idx=trade.entry_regime_idx,
        actual_pnl=trade.pnl, actual_bars_held=trade.bars_held,
        actual_exit_reason=trade.exit_reason,
        same_early_best=0.0, same_early_bar=0,
        same_at_exit=trade.pnl, same_extended_best=0.0, same_extended_bar=0,
        same_best_overall=trade.pnl, same_best_bar=0,
        counter_early_best=0.0, counter_early_bar=0,
        counter_at_exit=0.0, counter_extended_best=0.0, counter_extended_bar=0,
        counter_best_overall=0.0, counter_best_bar=0,
        best_action='same_at_exit', best_pnl=trade.pnl, regret=0.0,
        early_entry_gain=0.0, early_best_bars_before=0,
        early_best_same_peak=0.0, early_best_counter_peak=0.0,
        peak_pnl=0.0, peak_bar=0, mae_pnl=0.0, mae_bar=0,
        capture_ratio=float('nan'), final_pnl_at_horizon=trade.pnl,
    )


def label_trade_full(trade: ClosedTrade,
                          ohlcv_5s_by_day: Dict[str, pd.DataFrame],
                          lookback_bars: int = LOOKBACK_BARS,
                          lookahead_bars: int = LOOKAHEAD_BARS,
                          store_curves: bool = True
                          ) -> Optional[FullRegretLabel]:
    """Compute multi-axis regret for one trade. Returns None if data missing."""
    df = ohlcv_5s_by_day.get(trade.entry_day)
    if df is None:
        return None
    ts_arr = df['timestamp'].values.astype(np.int64)
    close = df['close'].values.astype(np.float64)
    n = len(ts_arr)

    entry_idx = int(np.searchsorted(ts_arr, int(trade.entry_ts), side='left'))
    if entry_idx >= n:
        return None

    held = int(trade.bars_held)
    exit_idx = min(entry_idx + held, n - 1)

    # === SAME and COUNTER curves from entry to entry + held + LOOKAHEAD ===
    end_idx = min(entry_idx + held + lookahead_bars, n)
    if end_idx - entry_idx < 2:
        return None
    same_path = _compute_pnl_path(close[entry_idx:end_idx], trade.entry_price,
                                              trade.direction)
    counter_path = -same_path  # exact: flip direction = invert pnl path

    # SAME EARLY: bar 0 .. held-1
    if held > 0 and held <= len(same_path):
        same_early_seg = same_path[:held]
        same_early_bar = int(np.argmax(same_early_seg))
        same_early_best = float(same_early_seg[same_early_bar])
    else:
        same_early_bar, same_early_best = 0, 0.0

    # SAME AT EXIT
    same_at_exit = float(same_path[held]) if held < len(same_path) else float(trade.pnl)

    # SAME EXTENDED: bars held .. end
    if held < len(same_path):
        ext_seg = same_path[held:]
        ext_bar_rel = int(np.argmax(ext_seg))
        same_extended_best = float(ext_seg[ext_bar_rel])
        same_extended_bar = held + ext_bar_rel
    else:
        same_extended_best, same_extended_bar = float(trade.pnl), held

    # SAME OVERALL
    same_best_bar = int(np.argmax(same_path))
    same_best_overall = float(same_path[same_best_bar])

    # COUNTER EARLY: flip at bar 1-3 (and check up to bar held-1, capped)
    counter_early_window = min(4, len(counter_path))
    if counter_early_window > 0:
        counter_early_seg = counter_path[:counter_early_window]
        counter_early_bar = int(np.argmax(counter_early_seg))
        counter_early_best = float(counter_early_seg[counter_early_bar])
    else:
        counter_early_bar, counter_early_best = 0, 0.0

    # COUNTER AT EXIT: flip at the actual exit bar
    counter_at_exit = float(counter_path[held]) if held < len(counter_path) else 0.0

    # COUNTER EXTENDED: flip at entry, hold to its peak in the extended window
    if held < len(counter_path):
        cext_seg = counter_path[held:]
        cext_rel = int(np.argmax(cext_seg))
        counter_extended_best = float(cext_seg[cext_rel])
        counter_extended_bar = held + cext_rel
    else:
        counter_extended_best, counter_extended_bar = 0.0, held

    # COUNTER OVERALL
    counter_best_bar = int(np.argmax(counter_path))
    counter_best_overall = float(counter_path[counter_best_bar])

    # === EARLY-ENTRY counterfactual ===
    lb_start = max(0, entry_idx - lookback_bars)
    early_entry_gain = 0.0
    early_best_bars_before = 0
    early_best_same_peak = 0.0
    early_best_counter_peak = 0.0
    if lb_start < entry_idx:
        for lb_idx in range(lb_start, entry_idx):
            lb_price = close[lb_idx]
            lb_end = min(lb_idx + held + lookahead_bars, n)
            lb_path_same = _compute_pnl_path(close[lb_idx:lb_end], lb_price,
                                                          trade.direction)
            lb_path_counter = -lb_path_same
            lb_same_peak = float(lb_path_same.max()) if len(lb_path_same) else 0.0
            lb_counter_peak = float(lb_path_counter.max()) if len(lb_path_counter) else 0.0
            best_lb = max(lb_same_peak, lb_counter_peak)
            ref_actual = max(float(trade.pnl), 0.0)
            gain = best_lb - ref_actual
            if gain > early_entry_gain:
                early_entry_gain = gain
                early_best_bars_before = entry_idx - lb_idx
                early_best_same_peak = lb_same_peak
                early_best_counter_peak = lb_counter_peak

    # === BEST ACTION ===
    options = {
        'same_early': same_early_best,
        'same_at_exit': same_at_exit,
        'same_extended': same_extended_best,
        'counter_early': counter_early_best,
        'counter_at_exit': counter_at_exit,
        'counter_extended': counter_extended_best,
    }
    best_action = max(options, key=options.get)
    best_pnl = options[best_action]
    regret = best_pnl - float(trade.pnl)

    # === Path summaries ===
    peak_bar = int(same_path.argmax())
    peak_pnl = float(same_path[peak_bar])
    mae_bar = int(same_path.argmin())
    mae_pnl = float(same_path[mae_bar])
    final_pnl_at_horizon = float(same_path[-1])
    capture_ratio = (trade.pnl / peak_pnl) if peak_pnl > 0 else float('nan')

    return FullRegretLabel(
        entry_day=trade.entry_day, entry_ts=trade.entry_ts,
        entry_price=trade.entry_price, direction=trade.direction,
        entry_tier=trade.entry_tier, entry_regime_idx=trade.entry_regime_idx,
        actual_pnl=float(trade.pnl), actual_bars_held=held,
        actual_exit_reason=trade.exit_reason,
        same_early_best=same_early_best, same_early_bar=same_early_bar,
        same_at_exit=same_at_exit,
        same_extended_best=same_extended_best, same_extended_bar=same_extended_bar,
        same_best_overall=same_best_overall, same_best_bar=same_best_bar,
        counter_early_best=counter_early_best, counter_early_bar=counter_early_bar,
        counter_at_exit=counter_at_exit,
        counter_extended_best=counter_extended_best, counter_extended_bar=counter_extended_bar,
        counter_best_overall=counter_best_overall, counter_best_bar=counter_best_bar,
        best_action=best_action, best_pnl=float(best_pnl), regret=float(regret),
        early_entry_gain=float(early_entry_gain),
        early_best_bars_before=int(early_best_bars_before),
        early_best_same_peak=float(early_best_same_peak),
        early_best_counter_peak=float(early_best_counter_peak),
        peak_pnl=peak_pnl, peak_bar=peak_bar,
        mae_pnl=mae_pnl, mae_bar=mae_bar,
        capture_ratio=float(capture_ratio),
        final_pnl_at_horizon=final_pnl_at_horizon,
        same_curve=same_path.astype(np.float32) if store_curves else None,
        counter_curve=counter_path.astype(np.float32) if store_curves else None,
        pnl_path=same_path.astype(np.float32) if store_curves else None,
    )


def label_trades_full(trades: Iterable[ClosedTrade],
                            atlas_root: str = ATLAS_ROOT,
                            lookback_bars: int = LOOKBACK_BARS,
                            lookahead_bars: int = LOOKAHEAD_BARS,
                            store_curves: bool = False
                            ) -> List[FullRegretLabel]:
    """Batch label. Loads each day's 5s OHLCV once."""
    trades_list = list(trades)
    days = sorted({t.entry_day for t in trades_list})
    ohlcv_by_day: Dict[str, pd.DataFrame] = {}
    for day in tqdm(days, desc='regret_full: load 5s'):
        df = _load_5s_ohlcv(day, atlas_root)
        if df is not None:
            ohlcv_by_day[day] = df

    out: List[FullRegretLabel] = []
    for t in tqdm(trades_list, desc='regret_full: label'):
        lbl = label_trade_full(t, ohlcv_by_day, lookback_bars=lookback_bars,
                                       lookahead_bars=lookahead_bars,
                                       store_curves=store_curves)
        if lbl is not None:
            out.append(lbl)
    return out


# ─── CLI + summary ────────────────────────────────────────────────────────

def summarize(labels: List[FullRegretLabel]) -> pd.DataFrame:
    """Per-trade DataFrame for analysis (best_action distribution etc.)."""
    rows = []
    for l in labels:
        rows.append({
            'tier': l.entry_tier, 'direction': l.direction,
            'regime': l.entry_regime_idx,
            'actual_pnl': l.actual_pnl, 'best_pnl': l.best_pnl,
            'regret': l.regret, 'best_action': l.best_action,
            'same_early_best': l.same_early_best,
            'same_extended_best': l.same_extended_best,
            'counter_early_best': l.counter_early_best,
            'counter_at_exit': l.counter_at_exit,
            'counter_extended_best': l.counter_extended_best,
            'early_entry_gain': l.early_entry_gain,
            'peak_pnl': l.peak_pnl, 'mae_pnl': l.mae_pnl,
            'capture_ratio': l.capture_ratio,
        })
    return pd.DataFrame(rows)


def _parse_args():
    p = argparse.ArgumentParser(description='V2-native full counterfactual regret')
    p.add_argument('--trades', default='training_v2/output/nmp_only.pkl')
    p.add_argument('--out', default='training_v2/output/regret_full.pkl')
    p.add_argument('--lookback-bars', type=int, default=LOOKBACK_BARS)
    p.add_argument('--lookahead-bars', type=int, default=LOOKAHEAD_BARS)
    p.add_argument('--store-curves', action='store_true',
                       help='Store full SAME/COUNTER curves (large pickle)')
    return p.parse_args()


def main():
    args = _parse_args()
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades)} trades from {args.trades}')

    labels = label_trades_full(trades, lookback_bars=args.lookback_bars,
                                          lookahead_bars=args.lookahead_bars,
                                          store_curves=args.store_curves)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(labels, f)
    print(f'Saved {len(labels)} full regret labels -> {args.out}')

    if not labels:
        return
    df = summarize(labels)
    print(f'\nBest-action distribution:')
    ba = df['best_action'].value_counts(normalize=False)
    bap = df['best_action'].value_counts(normalize=True)
    for action in ACTIONS:
        n = int(ba.get(action, 0))
        pct = float(bap.get(action, 0.0))
        sub = df[df['best_action'] == action]
        if len(sub) == 0:
            print(f'  {action:<18} {n:>6} ({pct:>5.1%})')
            continue
        mean_actual = sub['actual_pnl'].mean()
        mean_best = sub['best_pnl'].mean()
        mean_regret = sub['regret'].mean()
        print(f'  {action:<18} {n:>6} ({pct:>5.1%})  '
                  f'actual ${mean_actual:>+6.2f}  best ${mean_best:>+6.2f}  '
                  f'regret ${mean_regret:>+6.2f}')

    print(f'\nMean values overall:')
    print(f'  actual_pnl     : ${df["actual_pnl"].mean():>+7.2f}')
    print(f'  best_pnl       : ${df["best_pnl"].mean():>+7.2f}')
    print(f'  regret         : ${df["regret"].mean():>+7.2f}')
    print(f'  early_entry_gain: ${df["early_entry_gain"].mean():>+7.2f}')
    print(f'  peak (full)    : ${df["peak_pnl"].mean():>+7.2f}')

    # Per-tier best-action breakdown
    print(f'\nBest-action mix per tier:')
    for tier in df['tier'].unique():
        sub = df[df['tier'] == tier]
        print(f'\n  {tier}: n={len(sub)}, mean actual ${sub["actual_pnl"].mean():>+5.2f}, '
                  f'mean regret ${sub["regret"].mean():>+5.2f}')
        for action in ACTIONS:
            n = int((sub['best_action'] == action).sum())
            pct = n / max(len(sub), 1)
            print(f'    {action:<18} {n:>5} ({pct:>5.1%})')


if __name__ == '__main__':
    main()
