"""Test vol-adaptive exit thresholds via re-simulation.

For each trade, look up the entry vol_mean, find its IS-derived vol-quintile bin,
re-simulate the exit using that bin's threshold set instead of the production one.
Compare per-day PnL: production thresholds vs vol-binned thresholds.

If signal is real → integrate into engine. If not → reject.

Methodology:
  1. Build IS-only vol-quintile bin edges on L2_1m_vol_mean_15
  2. Per bin, derive (tp, sl, gb_min, gb_keep, time_stop) from that bin's
     trade peak/MAE/TtP/capture distribution via the same Bayesian formulas
     as threshold_bayesian.py — NO grid search.
  3. Re-simulate ALL IS trades with both threshold policies, compare
  4. Re-simulate ALL OOS trades, bootstrap CI on delta
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training_iso_v2.regret import RegretLabel, simulate_exit
from training_iso_v2.ledger import ClosedTrade
from training_iso_v2.tier_discovery import load_joined
from training_iso_v2.threshold_optimizer import lookup_thresholds, DEFAULT_THRESHOLDS
from training_iso_v2.threshold_bayesian import (
    _compute_stats, _stats_to_thresholds, _shrink, RegimeStats,
    Q_TP_DEFAULT, Q_SL_DEFAULT, TTP_FACTOR_DEFAULT,
)
from training_iso_v2.exits import SL_PTS_FLOOR


VOL_FEATURE = 'L2_1m_vol_mean_15'
N_BINS = 5


def attach_vol(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the vol feature out of entry_v2 into a column."""
    j = FEATURE_NAMES.index(VOL_FEATURE)
    df = df.copy()
    df['vol'] = np.stack(df['entry_v2'].values)[:, j]
    return df


def attach_path(df: pd.DataFrame, regret_path: str) -> pd.DataFrame:
    with open(regret_path, 'rb') as f:
        labels: List[RegretLabel] = pickle.load(f)
    label_map = {(l.entry_day, l.entry_ts): l.pnl_path for l in labels}
    df = df.copy()
    df['_pnl_path'] = df.apply(
        lambda r: label_map.get((r['day'], r['ts'])), axis=1)
    return df[df['_pnl_path'].notna()].reset_index(drop=True)


def derive_vol_thresholds(is_df: pd.DataFrame,
                                  bin_edges: np.ndarray,
                                  q_tp: float = Q_TP_DEFAULT,
                                  q_sl: float = Q_SL_DEFAULT,
                                  ttp_factor: float = TTP_FACTOR_DEFAULT,
                                  ) -> Dict[int, Dict]:
    """Per-bin threshold map. Returns {bin_idx: thresholds_dict}.

    Thresholds derived via the same Bayesian formulas as threshold_bayesian.py.
    """
    # Build per-trade RegretLabel-shape stats from is_df
    out = {}

    # Universal pool
    universal_labels = []
    for _, r in is_df.iterrows():
        if r['_pnl_path'] is None: continue
        l = type('L', (), {})()
        l.peak_pnl = r['fade_peak']
        l.mae_pnl = -r['flip_peak']  # mae = -flip_peak by construction
        l.time_to_peak_s = r['time_to_peak_s']
        l.capture_ratio = r['capture_ratio']
        l.entry_regime_idx = r['regime_idx']
        l.entry_tier = 'NMP'
        universal_labels.append(l)
    universal_stats = _compute_stats(universal_labels, q_tp, q_sl)
    universal_thr = _stats_to_thresholds(universal_stats, ttp_factor)

    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        if b == 0:
            mask = (is_df['vol'] >= lo) & (is_df['vol'] <= hi)
        else:
            mask = (is_df['vol'] > lo) & (is_df['vol'] <= hi)
        sub = is_df[mask]
        if len(sub) < 100:
            out[b] = dict(universal_thr)
            continue
        labels = [l for l, k in zip(universal_labels, mask) if k]
        raw = _compute_stats(labels, q_tp, q_sl)
        shrunk = _shrink(raw, universal_stats, shrinkage_n=200)
        out[b] = _stats_to_thresholds(shrunk, ttp_factor)

    return out


def assign_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin assignment via np.digitize, clipped to [0, n_bins-1]."""
    bins = np.digitize(values, edges[1:-1])
    return np.clip(bins, 0, len(edges) - 2)


def simulate_with_map(df: pd.DataFrame, threshold_lookup) -> np.ndarray:
    """Simulate per-trade PnL using a threshold-lookup callable.

    threshold_lookup(row) -> dict with tp_pts, sl_pts, gb_min, gb_keep, time_stop_bars.
    """
    out = np.zeros(len(df))
    for i, (_, r) in enumerate(df.iterrows()):
        thr = threshold_lookup(r)
        sl_pts_eff = min(thr['sl_pts'], SL_PTS_FLOOR)
        path = r['_pnl_path']
        pnl, _, _ = simulate_exit(
            path, tp_pts=thr['tp_pts'],
            sl_pts=sl_pts_eff,
            giveback_min_peak=thr['gb_min'],
            giveback_keep=thr['gb_keep'],
            time_stop_bars=int(thr['time_stop_bars']),
        )
        out[i] = float(pnl)
    return out


def bootstrap_ci(deltas: np.ndarray, B: int = 4000, seed: int = 42):
    rng = np.random.default_rng(seed)
    boots = np.zeros(B)
    for i in range(B):
        idx = rng.integers(0, len(deltas), len(deltas))
        boots[i] = deltas[idx].mean()
    return np.percentile(boots, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--is-trades', default='training_iso_v2/output/nmp_only.pkl')
    p.add_argument('--is-regret', default='training_iso_v2/output/regret_nmp.pkl')
    p.add_argument('--oos-trades', default='training_iso_v2/output/nmp_only_oos.pkl')
    p.add_argument('--oos-regret', default='training_iso_v2/output/regret_nmp_oos.pkl')
    p.add_argument('--prod-thresholds', default='training_iso_v2/output/thresholds_prod.json')
    p.add_argument('--out', default='training_iso_v2/output/vol_adaptive_thresholds.json')
    p.add_argument('--n-bins', type=int, default=N_BINS)
    args = p.parse_args()

    print(f'Loading IS...')
    is_df = load_joined(args.is_trades, args.is_regret)
    is_df = attach_vol(is_df)
    is_df = attach_path(is_df, args.is_regret)
    print(f'  IS: {len(is_df)} trades')

    print(f'Loading OOS...')
    oos_df = load_joined(args.oos_trades, args.oos_regret)
    oos_df = attach_vol(oos_df)
    oos_df = attach_path(oos_df, args.oos_regret)
    print(f'  OOS: {len(oos_df)} trades')

    # Bin edges from IS only
    edges = np.quantile(is_df['vol'].dropna(), np.linspace(0, 1, args.n_bins + 1))
    print(f'\nVol-quintile edges (IS): {edges.round(1)}')

    # Derive thresholds per bin
    print(f'\nDeriving Bayesian thresholds per vol bin...')
    bin_thr = derive_vol_thresholds(is_df, edges)
    print(f'  {"bin":<5} {"vol_lo":>10} {"vol_hi":>10} {"tp_pts":>7} {"sl_pts":>7} '
              f'{"gb_min":>8} {"gb_keep":>7} {"ts_bars":>7}')
    for b, thr in bin_thr.items():
        print(f'  Q{b+1:<4} {edges[b]:>+10.1f} {edges[b+1]:>+10.1f} '
                  f'{thr["tp_pts"]:>+7.1f} {thr["sl_pts"]:>+7.1f} '
                  f'{thr["gb_min"]:>+8.1f} {thr["gb_keep"]:>+7.2f} '
                  f'{thr["time_stop_bars"]:>+7d}')

    # Save
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({
            'bin_edges': edges.tolist(),
            'bin_thresholds': {str(k): v for k, v in bin_thr.items()},
            'vol_feature': VOL_FEATURE,
        }, f, indent=2)
    print(f'\nSaved -> {args.out}')

    # Load production thresholds for comparison
    with open(args.prod_thresholds, 'r') as f:
        prod_map = json.load(f)

    def prod_lookup(r):
        return lookup_thresholds(prod_map, int(r['regime_idx']), 'REVERSION')

    is_df['vol_bin'] = assign_bin(is_df['vol'].values, edges)
    oos_df['vol_bin'] = assign_bin(oos_df['vol'].values, edges)

    def vol_lookup_factory():
        def lookup(r):
            return bin_thr.get(int(r['vol_bin']), bin_thr[0])
        return lookup
    vol_lookup = vol_lookup_factory()

    # ── IS comparison ────────────────────────────────────────────────────
    print(f'\n=== IS re-simulation: production thr vs vol-adaptive thr ===')
    is_prod = simulate_with_map(is_df, prod_lookup)
    is_vol = simulate_with_map(is_df, vol_lookup)
    is_prod_daily = pd.DataFrame({'day': is_df['day'], 'pnl': is_prod}).groupby('day')['pnl'].sum()
    is_vol_daily = pd.DataFrame({'day': is_df['day'], 'pnl': is_vol}).groupby('day')['pnl'].sum()
    is_delta = is_vol_daily.values - is_prod_daily.values
    is_lo, is_hi = bootstrap_ci(is_delta)
    print(f'  Production: ${is_prod.sum():>+10.2f} total, ${is_prod_daily.mean():>+7.2f}/day')
    print(f'  Vol-adapt : ${is_vol.sum():>+10.2f} total, ${is_vol_daily.mean():>+7.2f}/day')
    print(f'  Delta/day : ${is_delta.mean():>+7.2f}    95% CI [${is_lo:>+7.2f}, ${is_hi:>+7.2f}]')

    # ── OOS comparison ───────────────────────────────────────────────────
    print(f'\n=== OOS re-simulation: production thr vs vol-adaptive thr ===')
    oos_prod = simulate_with_map(oos_df, prod_lookup)
    oos_vol = simulate_with_map(oos_df, vol_lookup)
    oos_prod_daily = pd.DataFrame({'day': oos_df['day'], 'pnl': oos_prod}).groupby('day')['pnl'].sum()
    oos_vol_daily = pd.DataFrame({'day': oos_df['day'], 'pnl': oos_vol}).groupby('day')['pnl'].sum()
    oos_delta = oos_vol_daily.values - oos_prod_daily.values
    oos_lo, oos_hi = bootstrap_ci(oos_delta)
    sig = 'YES' if oos_lo > 0 else 'no'
    print(f'  Production: ${oos_prod.sum():>+10.2f} total, ${oos_prod_daily.mean():>+7.2f}/day')
    print(f'  Vol-adapt : ${oos_vol.sum():>+10.2f} total, ${oos_vol_daily.mean():>+7.2f}/day')
    print(f'  Delta/day : ${oos_delta.mean():>+7.2f}    95% CI [${oos_lo:>+7.2f}, ${oos_hi:>+7.2f}]   sig: {sig}')

    # ── Per-bin OOS breakdown ────────────────────────────────────────────
    print(f'\n=== Per-bin OOS PnL breakdown ===')
    print(f'  {"bin":<4} {"n":>5} {"prod_$/t":>10} {"vol_$/t":>10} {"delta":>8}')
    oos_df_aug = oos_df.copy()
    oos_df_aug['prod_pnl'] = oos_prod
    oos_df_aug['vol_pnl'] = oos_vol
    for b in range(args.n_bins):
        sub = oos_df_aug[oos_df_aug['vol_bin'] == b]
        if len(sub) == 0:
            continue
        delta = (sub['vol_pnl'] - sub['prod_pnl']).mean()
        print(f'  Q{b+1:<3} {len(sub):>5} ${sub["prod_pnl"].mean():>+9.2f} '
                  f'${sub["vol_pnl"].mean():>+9.2f}  ${delta:>+7.2f}')


if __name__ == '__main__':
    main()
