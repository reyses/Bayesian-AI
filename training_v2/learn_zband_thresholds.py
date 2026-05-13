"""Learn per-(regime, direction) z-band anchor thresholds.

For each cell, find the threshold on the cell's TOP z-band discriminator
(`L3_1m_z_se_15` etc.) that maximizes same_extended vs counter_extended
separation accuracy on IS train, then validate on IS holdout (70/30).

Output: JSON with per-cell rules:
    {
        "regime_idx|direction": {
            "feature": "L3_1m_z_se_15",
            "threshold": -1.95,
            "flip_when_above": True,      # True = flip if value > threshold
            "d_train": 0.34,
            "d_val": 0.31,
            "split_acc_val": 0.58,
            "n_train_same": 1067,
            "n_train_counter": 1033,
        }
    }

Used by FilteredZBandReversion strategy as a per-cell flip-or-keep gate
on top of (or replacing) the simple regime-direction flip rule.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training_v2.regret_full import FullRegretLabel
from training_v2.regret import RegretLabel
from training_v2.ledger import ClosedTrade
from training_v2.tier_discovery import load_joined, cohens_d
from training_v2.state import REGIME_VOCAB


# Top discriminator per cell (from 2026-05-05 anchor_search).
# These are the features with the largest |Cohen's d| between same_extended
# and counter_extended cohorts within each cell. Hardcoded so we don't
# re-fit feature selection on every run (which would itself overfit).
TOP_DISCRIMINATOR = {
    'UP_SMOOTH': 'L3_1m_z_se_15',
    'UP_CHOPPY': 'L3_1m_z_low_15',
    'DOWN_SMOOTH': 'L3_1m_z_high_15',
    'DOWN_CHOPPY': 'L2_1m_price_accel_15',
    'FLAT_SMOOTH': 'L3_1D_SE_low_5',     # weak (d=0.077) — included for completeness
    'FLAT_CHOPPY': 'L2_5m_vol_accel_9',  # weak (d=0.078) — included for completeness
}


def find_best_threshold(values: np.ndarray, labels: np.ndarray
                              ) -> Tuple[Optional[float], bool, float]:
    """Find threshold that best separates labels (1=counter_ext, 0=same_ext).

    Returns (threshold, flip_when_above, accuracy):
      - flip_when_above = True  : "value > threshold" tags counter_ext (flip)
      - flip_when_above = False : "value <= threshold" tags counter_ext (flip)
    """
    valid = ~np.isnan(values)
    v = values[valid]
    l = labels[valid]
    if len(v) < 50:
        return None, False, 0.5

    sorted_idx = np.argsort(v)
    sv = v[sorted_idx]
    sl = l[sorted_idx]
    n = len(sv)
    cum = np.cumsum(sl)

    base_acc = max(sl.mean(), 1 - sl.mean())
    best_acc, best_thr, best_above = base_acc, None, False

    for k in range(20, n - 20, max(1, n // 200)):
        thr = sv[k]
        below_pos = cum[k - 1]
        above_pos = cum[-1] - below_pos
        n_below, n_above = k, n - k

        # Case A: above = counter (flip when above), below = same
        acc_a = (above_pos + (n_below - below_pos)) / n
        # Case B: below = counter (flip when below), above = same
        acc_b = (below_pos + (n_above - above_pos)) / n

        if acc_a > best_acc:
            best_acc = acc_a
            best_thr = float(thr)
            best_above = True
        if acc_b > best_acc:
            best_acc = acc_b
            best_thr = float(thr)
            best_above = False

    return best_thr, best_above, best_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trades', default='training_v2/output/nmp_only.pkl')
    p.add_argument('--regret-simple', default='training_v2/output/regret_nmp.pkl')
    p.add_argument('--regret-full', default='training_v2/output/regret_full_nmp.pkl')
    p.add_argument('--out', default='training_v2/output/zband_anchors.json')
    p.add_argument('--train-frac', type=float, default=0.7)
    args = p.parse_args()

    print(f'Loading...')
    df = load_joined(args.trades, args.regret_simple)
    df = df.sort_values('ts').reset_index(drop=True)

    # Attach best_action from full regret
    with open(args.regret_full, 'rb') as f:
        fr_labels = pickle.load(f)
    fr_map = {(l.entry_day, l.entry_ts): l.best_action for l in fr_labels}
    df['best_action'] = df.apply(
        lambda r: fr_map.get((r['day'], r['ts'])), axis=1)
    df = df[df['best_action'].notna()].reset_index(drop=True)

    # Filter to same_extended OR counter_extended (the only cohorts with mass)
    df = df[df['best_action'].isin(['same_extended', 'counter_extended'])].copy()
    df['is_counter'] = (df['best_action'] == 'counter_extended').astype(np.int32)
    df['regime'] = df['regime_idx'].apply(
        lambda r: REGIME_VOCAB[int(r)] if int(r) < len(REGIME_VOCAB) else f'R{r}')
    print(f'  Trades after filter: {len(df)}')

    cut = int(len(df) * args.train_frac)
    train = df.iloc[:cut]
    val = df.iloc[cut:]
    print(f'  Train: {len(train)}, Val: {len(val)}')

    out: Dict[str, Dict] = {}

    print()
    print(f'  {"cell":<28} {"feature":<28} {"thr":>8} {"side":>6} '
              f'{"acc_T":>6} {"acc_V":>6} {"d_T":>6} {"d_V":>6} {"surv?":>5}')
    print('-' * 110)

    for r in sorted(df['regime_idx'].unique()):
        rname = REGIME_VOCAB[int(r)]
        feature = TOP_DISCRIMINATOR.get(rname)
        if feature is None or feature not in FEATURE_NAMES:
            continue
        feat_idx = FEATURE_NAMES.index(feature)
        for direction in ('long', 'short'):
            cell = f'{rname}|{direction}'
            train_sub = train[(train['regime'] == rname) & (train['direction'] == direction)]
            val_sub = val[(val['regime'] == rname) & (val['direction'] == direction)]
            if len(train_sub) < 100 or len(val_sub) < 30:
                print(f'  {cell:<28} (insufficient data)')
                continue

            train_feats = np.stack(train_sub['entry_v2'].values)
            val_feats = np.stack(val_sub['entry_v2'].values)
            train_vals = train_feats[:, feat_idx]
            val_vals = val_feats[:, feat_idx]
            train_lbl = train_sub['is_counter'].values
            val_lbl = val_sub['is_counter'].values

            thr, above, acc_t = find_best_threshold(train_vals, train_lbl)
            if thr is None:
                print(f'  {cell:<28} (no threshold)')
                continue

            # Validation accuracy
            valid_v = ~np.isnan(val_vals)
            vv = val_vals[valid_v]
            vl = val_lbl[valid_v]
            if above:
                pred_v = (vv > thr).astype(np.int32)
            else:
                pred_v = (vv <= thr).astype(np.int32)
            acc_v = (pred_v == vl).mean()

            # d on train and val
            same_train = train_vals[(train_lbl == 0) & ~np.isnan(train_vals)]
            counter_train = train_vals[(train_lbl == 1) & ~np.isnan(train_vals)]
            d_t = cohens_d(same_train, counter_train)

            same_val = val_vals[(val_lbl == 0) & ~np.isnan(val_vals)]
            counter_val = val_vals[(val_lbl == 1) & ~np.isnan(val_vals)]
            d_v = cohens_d(same_val, counter_val)

            same_sign = np.sign(d_t) == np.sign(d_v) and d_t != 0 and d_v != 0
            surv = bool(same_sign and abs(d_v) >= 0.10 and acc_v > 0.52)

            print(f'  {cell:<28} {feature:<28} {thr:>+8.2f} '
                      f'{"above" if above else "below":>6} '
                      f'{acc_t:>5.1%} {acc_v:>5.1%} '
                      f'{d_t:>+6.3f} {d_v:>+6.3f} {"YES" if surv else "no":>5}')

            if surv:
                out[cell] = {
                    'feature': feature,
                    'threshold': float(thr),
                    'flip_when_above': bool(above),
                    'acc_train': float(acc_t),
                    'acc_val': float(acc_v),
                    'd_train': float(d_t),
                    'd_val': float(d_v),
                    'n_train_same': int((train_lbl == 0).sum()),
                    'n_train_counter': int((train_lbl == 1).sum()),
                }

    print(f'\nSurviving cells: {len(out)}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()
