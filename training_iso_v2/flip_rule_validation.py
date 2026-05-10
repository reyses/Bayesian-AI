"""Validate the regime-direction flip rule with walk-forward + OOS.

The rule (from per-cell EDA on full IS):
    UP_SMOOTH × short  -> flip to long   (NMP fades extreme high z, but uptrend continues)
    UP_CHOPPY × short  -> flip to long
    DOWN_SMOOTH × long -> flip to short  (NMP fades extreme low z, but downtrend continues)

(DOWN_CHOPPY borderline; FLAT regimes wash — left as-is.)

Validation methodology:
  1. Re-simulate exits on regret pnl_paths for both original direction and
     flipped direction using the SAME threshold policy. This makes the
     comparison apples-to-apples (no exit-policy difference confounding).
  2. Walk-forward: learn flip cells from first 70% of IS days,
     apply on last 30% of IS days.
  3. OOS: apply IS-learned flip cells to OOS, bootstrap CI on delta.

For each experiment the pipeline is:
  baseline = sim_exit(path, threshold_policy)            # always keep direction
  flipped  = sim_exit(path or -path based on flip cells, threshold_policy)
  delta    = flipped - baseline
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from training_iso_v2.regret import RegretLabel, simulate_exit
from training_iso_v2.ledger import ClosedTrade
from training_iso_v2.tier_discovery import load_joined
from training_iso_v2.threshold_optimizer import lookup_thresholds, DEFAULT_THRESHOLDS
from training_iso_v2.exits import SL_PTS_FLOOR
from training_iso_v2.state import REGIME_VOCAB


# ─── Flip cell discovery ──────────────────────────────────────────────────

def learn_flip_cells(df: pd.DataFrame,
                          min_n: int = 200,
                          loss_threshold: float = -0.4,
                          flip_peak_advantage: float = 20.0,
                          ) -> Set[Tuple[int, str]]:
    """Identify (regime_idx, direction) cells where:
        - sample size n >= min_n (statistical reliability)
        - actual_pnl/trade <= loss_threshold (current direction loses)
        - flip_peak > fade_peak + flip_peak_advantage (the flip would have
          had a clearly higher peak)
    """
    grp = df.groupby(['regime_idx', 'direction']).agg(
        n=('actual_pnl', 'size'),
        actual_per_trade=('actual_pnl', 'mean'),
        fade_peak=('fade_peak', 'mean'),
        flip_peak=('flip_peak', 'mean'),
    )
    cells = set()
    for (regime, direction), row in grp.iterrows():
        if row['n'] < min_n:
            continue
        if row['actual_per_trade'] > loss_threshold:
            continue
        if row['flip_peak'] - row['fade_peak'] < flip_peak_advantage:
            continue
        cells.add((int(regime), str(direction)))
    return cells


# ─── Re-simulation (apples-to-apples comparison) ──────────────────────────

def simulate_with_thresholds(label: RegretLabel, threshold_map: Dict,
                                       flip: bool = False) -> float:
    """Re-simulate exits on the regret pnl_path using a threshold policy.

    If flip=True, simulate on -path (the opposite direction's PnL trajectory).
    """
    if label.pnl_path is None or len(label.pnl_path) == 0:
        return 0.0
    thr = lookup_thresholds(threshold_map, label.entry_regime_idx, label.entry_tier)
    sl_pts_eff = min(thr['sl_pts'], SL_PTS_FLOOR)
    path = label.pnl_path if not flip else (-label.pnl_path)
    pnl, _, _ = simulate_exit(
        path,
        tp_pts=thr['tp_pts'],
        sl_pts=sl_pts_eff,
        giveback_min_peak=thr['gb_min'],
        giveback_keep=thr['gb_keep'],
        time_stop_bars=int(thr['time_stop_bars']),
    )
    return float(pnl)


def evaluate_flip_rule(df: pd.DataFrame, threshold_map: Dict,
                              flip_cells: Set[Tuple[int, str]]
                              ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run baseline + flipped simulation; return per-trade arrays + flip count."""
    base_arr = np.zeros(len(df))
    flip_arr = np.zeros(len(df))
    n_flips = 0
    for i, (_, row) in enumerate(df.iterrows()):
        # Reconstruct a regret-label-like object for the simulator
        class L:
            pass
        l = L()
        l.pnl_path = row['_pnl_path']
        l.entry_regime_idx = int(row['regime_idx'])
        l.entry_tier = 'REVERSION'
        base_arr[i] = simulate_with_thresholds(l, threshold_map, flip=False)
        key = (int(row['regime_idx']), str(row['direction']))
        if key in flip_cells:
            flip_arr[i] = simulate_with_thresholds(l, threshold_map, flip=True)
            n_flips += 1
        else:
            flip_arr[i] = base_arr[i]
    return base_arr, flip_arr, n_flips


def daily_pnl(df: pd.DataFrame, per_trade: np.ndarray) -> pd.Series:
    out = pd.DataFrame({'day': df['day'].values, 'pnl': per_trade})
    return out.groupby('day')['pnl'].sum()


def bootstrap_ci(deltas: np.ndarray, B: int = 4000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.zeros(B)
    for i in range(B):
        idx = rng.integers(0, len(deltas), len(deltas))
        boots[i] = deltas[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ─── Main experiments ─────────────────────────────────────────────────────

def attach_path(df: pd.DataFrame, regret_labels: List[RegretLabel]) -> pd.DataFrame:
    """Attach pnl_path to df by (entry_day, entry_ts) key."""
    label_map = {(l.entry_day, l.entry_ts): l.pnl_path for l in regret_labels}
    df = df.copy()
    df['_pnl_path'] = df.apply(
        lambda r: label_map.get((r['day'], r['ts'])), axis=1)
    df = df[df['_pnl_path'].notna()].reset_index(drop=True)
    return df


def main():
    p = argparse.ArgumentParser(description='Validate regime-direction flip rule')
    p.add_argument('--is-trades', default='training_iso_v2/output/nmp_only.pkl')
    p.add_argument('--is-regret', default='training_iso_v2/output/regret_nmp.pkl')
    p.add_argument('--oos-trades', default='training_iso_v2/output/nmp_only_oos.pkl')
    p.add_argument('--oos-regret-out', default='training_iso_v2/output/regret_nmp_oos.pkl')
    p.add_argument('--thresholds', default='training_iso_v2/output/thresholds_prod.json')
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--min-n', type=int, default=200)
    p.add_argument('--loss-thr', type=float, default=-0.4,
                       help='cell qualifies for flip if $/trade <= this')
    p.add_argument('--flip-adv', type=float, default=20.0,
                       help='cell qualifies if flip_peak - fade_peak >= this')
    args = p.parse_args()

    # Load thresholds
    with open(args.thresholds, 'r') as f:
        threshold_map = json.load(f)

    # Build OOS regret labels if not yet present
    if not os.path.exists(args.oos_regret_out):
        from training_iso_v2.regret import label_trades
        with open(args.oos_trades, 'rb') as f:
            oos_trades = pickle.load(f)
        print(f'Building OOS regret labels for {len(oos_trades)} trades...')
        oos_labels = label_trades(oos_trades)
        with open(args.oos_regret_out, 'wb') as f:
            pickle.dump(oos_labels, f)
    else:
        with open(args.oos_regret_out, 'rb') as f:
            oos_labels = pickle.load(f)
        print(f'Loaded {len(oos_labels)} OOS regret labels')

    # Load joined IS + regret
    print(f'Loading IS trades + regret...')
    is_df = load_joined(args.is_trades, args.is_regret)
    with open(args.is_regret, 'rb') as f:
        is_labels = pickle.load(f)
    is_df = attach_path(is_df, is_labels)
    print(f'  IS joined: {len(is_df)} trades')

    # Load joined OOS + regret
    print(f'Loading OOS trades + regret...')
    oos_df = load_joined(args.oos_trades, args.oos_regret_out)
    oos_df = attach_path(oos_df, oos_labels)
    print(f'  OOS joined: {len(oos_df)} trades')

    # ── Experiment 1: Flip rule learned from FULL IS, applied to OOS ─────
    print(f'\n=== Experiment 1: Full IS flip cells -> OOS ===')
    full_cells = learn_flip_cells(is_df, args.min_n, args.loss_thr, args.flip_adv)
    print(f'Flip cells (full IS, n>={args.min_n}, loss<={args.loss_thr}, '
              f'flip_adv>={args.flip_adv}):')
    for r, d in sorted(full_cells):
        print(f'  ({REGIME_VOCAB[r]}, {d}) -> flip')

    # IS apples-to-apples (baseline simulated with prod thresholds, flipped applied)
    is_base, is_flip, n_is_flips = evaluate_flip_rule(is_df, threshold_map, full_cells)
    is_base_daily = daily_pnl(is_df, is_base)
    is_flip_daily = daily_pnl(is_df, is_flip)
    is_delta = is_flip_daily.values - is_base_daily.values
    is_lo, is_hi = bootstrap_ci(is_delta)
    print(f'\nIS (apples-to-apples re-sim, prod thresholds):')
    print(f'  Baseline:  ${is_base.sum():>+10.2f} total, ${is_base_daily.mean():>+7.2f}/day')
    print(f'  Flipped:   ${is_flip.sum():>+10.2f} total, ${is_flip_daily.mean():>+7.2f}/day')
    print(f'  Delta/day: ${is_delta.mean():>+7.2f}    95% CI [{is_lo:>+7.2f}, {is_hi:>+7.2f}]')
    print(f'  N flips applied: {n_is_flips}/{len(is_df)}')

    # OOS apply
    oos_base, oos_flip, n_oos_flips = evaluate_flip_rule(oos_df, threshold_map, full_cells)
    oos_base_daily = daily_pnl(oos_df, oos_base)
    oos_flip_daily = daily_pnl(oos_df, oos_flip)
    oos_delta = oos_flip_daily.values - oos_base_daily.values
    oos_lo, oos_hi = bootstrap_ci(oos_delta)
    print(f'\nOOS (apply IS-learned flip cells):')
    print(f'  Baseline:  ${oos_base.sum():>+10.2f} total, ${oos_base_daily.mean():>+7.2f}/day')
    print(f'  Flipped:   ${oos_flip.sum():>+10.2f} total, ${oos_flip_daily.mean():>+7.2f}/day')
    print(f'  Delta/day: ${oos_delta.mean():>+7.2f}    95% CI [{oos_lo:>+7.2f}, {oos_hi:>+7.2f}]')
    print(f'  N flips applied: {n_oos_flips}/{len(oos_df)}')
    sig = 'YES' if oos_lo > 0 else 'NO'
    print(f'  Significant: {sig}')

    # ── Experiment 2: Walk-forward inside IS ─────────────────────────────
    print(f'\n=== Experiment 2: Walk-forward inside IS (train {args.train_frac:.0%}) ===')
    is_sorted = is_df.sort_values('ts').reset_index(drop=True)
    cut = int(len(is_sorted) * args.train_frac)
    train_df = is_sorted.iloc[:cut]
    val_df = is_sorted.iloc[cut:].reset_index(drop=True)

    train_cells = learn_flip_cells(train_df, args.min_n, args.loss_thr, args.flip_adv)
    print(f'Flip cells learned from {len(train_df)} train trades:')
    for r, d in sorted(train_cells):
        print(f'  ({REGIME_VOCAB[r]}, {d})')

    val_base, val_flip, n_val_flips = evaluate_flip_rule(val_df, threshold_map, train_cells)
    val_base_daily = daily_pnl(val_df, val_base)
    val_flip_daily = daily_pnl(val_df, val_flip)
    val_delta = val_flip_daily.values - val_base_daily.values
    val_lo, val_hi = bootstrap_ci(val_delta)
    print(f'\nIS-VAL ({len(val_df)} trades on last {1-args.train_frac:.0%}):')
    print(f'  Baseline:  ${val_base.sum():>+10.2f} total, ${val_base_daily.mean():>+7.2f}/day')
    print(f'  Flipped:   ${val_flip.sum():>+10.2f} total, ${val_flip_daily.mean():>+7.2f}/day')
    print(f'  Delta/day: ${val_delta.mean():>+7.2f}    95% CI [{val_lo:>+7.2f}, {val_hi:>+7.2f}]')
    print(f'  N flips applied: {n_val_flips}/{len(val_df)}')
    sig = 'YES' if val_lo > 0 else 'NO'
    print(f'  Significant: {sig}')

    # ── Per-cell breakdown on OOS ────────────────────────────────────────
    print(f'\n=== Per-cell breakdown on OOS ===')
    print(f'Including ALL cells (not just flipped) so you can see how each behaves OOS.')
    print(f'  {"regime":<13} {"dir":>5} {"n":>5} {"base$/t":>9} {"flip$/t":>9} '
              f'{"delta":>8} {"flipped?":>9}')
    oos_df_idx = oos_df.copy()
    oos_df_idx['base_sim_pnl'] = oos_base
    oos_df_idx['flip_sim_pnl'] = oos_flip
    grp = oos_df_idx.groupby(['regime_idx', 'direction']).agg(
        n=('base_sim_pnl', 'size'),
        base_per_trade=('base_sim_pnl', 'mean'),
        flip_per_trade=('flip_sim_pnl', 'mean'),
    )
    for (regime, direction), row in grp.iterrows():
        flipped = '*FLIP*' if (int(regime), str(direction)) in full_cells else ''
        rname = REGIME_VOCAB[int(regime)]
        delta = row['flip_per_trade'] - row['base_per_trade']
        print(f'  {rname:<13} {direction:>5} {int(row["n"]):>5} '
                  f'${row["base_per_trade"]:>+8.2f} ${row["flip_per_trade"]:>+8.2f} '
                  f'${delta:>+7.2f}  {flipped:>9}')


if __name__ == '__main__':
    main()
