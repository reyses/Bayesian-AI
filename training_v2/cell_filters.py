"""Learn per-(regime × direction) entry quality filters.

For each cell where the top discriminator survives walk-forward validation,
identify a single-feature threshold that skips the "loser tail" — the value
range where loser density >> winner density.

Output: JSON config mapping {regime_idx|direction: filter_spec} consumed by
FilteredRegimeAwareReversion at entry time.

Filter format per cell:
    {
        "feature": "L1_1m_bar_range",
        "threshold": 25.7,
        "skip_above": true,        # skip if value > threshold
        "n_train": 1090,
        "win_kept": 0.85,          # fraction of original winners that survive the gate
        "loss_kept": 0.55,         # fraction of original losers that survive
    }

The threshold is chosen by maximizing winner-vs-loser split accuracy on the
training portion of IS, then validated on a 30% hold-out (rejected if it
doesn't survive).
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
from training_v2.regret import RegretLabel
from training_v2.ledger import ClosedTrade
from training_v2.tier_discovery import load_joined, cohens_d
from training_v2.full_feature_eda import spearman_corr
from training_v2.state import REGIME_VOCAB


def find_best_threshold(values: np.ndarray, labels: np.ndarray) -> Tuple[Optional[float], bool, float]:
    """Find the threshold + direction that best separates labels (1=win, 0=loss).

    Returns (threshold, skip_above, accuracy):
      - threshold = float
      - skip_above = True if "skip when value > threshold" gates losers
      - accuracy = best classification accuracy
    None if no usable split found.
    """
    valid = ~np.isnan(values)
    v = values[valid]
    l = labels[valid]
    if len(v) < 100:
        return None, False, 0.5

    sorted_idx = np.argsort(v)
    sv = v[sorted_idx]
    sl = l[sorted_idx]

    n = len(sv)
    cum = np.cumsum(sl)               # cum sum of winners up to index k

    best_acc = max(sl.mean(), 1 - sl.mean())
    best_thr = None
    best_skip_above = False

    # Vectorize: for each split k in [50, n-50], compute accuracy assuming
    # "below threshold = winners" or "above threshold = winners"
    for k in range(50, n - 50, max(1, n // 200)):
        thr = sv[k]
        winners_below = cum[k - 1]
        winners_above = cum[-1] - winners_below
        n_below = k
        n_above = n - k

        # Case A: below = winners, above = losers (skip_above = True)
        acc_a = (winners_below + (n_above - winners_above)) / n
        # Case B: above = winners, below = losers (skip_above = False, i.e., skip-below)
        acc_b = (winners_above + (n_below - winners_below)) / n

        if acc_a > best_acc:
            best_acc = acc_a
            best_thr = float(thr)
            best_skip_above = True
        if acc_b > best_acc:
            best_acc = acc_b
            best_thr = float(thr)
            best_skip_above = False

    return best_thr, best_skip_above, best_acc


def evaluate_filter(values: np.ndarray, pnl: np.ndarray,
                          threshold: float, skip_above: bool) -> Dict:
    """Apply a filter to a (values, pnl) cohort; report stats."""
    if skip_above:
        kept = values <= threshold
    else:
        kept = values >= threshold
    n_total = len(pnl)
    n_kept = int(kept.sum())
    n_skip = int((~kept).sum())
    return {
        'n_total': n_total,
        'n_kept': n_kept,
        'n_skip': n_skip,
        'kept_pnl': float(pnl[kept].sum()) if n_kept else 0.0,
        'skip_pnl': float(pnl[~kept].sum()) if n_skip else 0.0,
        'kept_per_trade': float(pnl[kept].mean()) if n_kept else 0.0,
        'skip_per_trade': float(pnl[~kept].mean()) if n_skip else 0.0,
        'kept_wr': float((pnl[kept] > 0).mean()) if n_kept else 0.0,
    }


def learn_cell_filter(sub: pd.DataFrame, margin: float = 5.0,
                            train_frac: float = 0.7, top_k: int = 5
                            ) -> Optional[Dict]:
    """For one (regime, direction) cohort, find best single-feature filter."""
    if len(sub) < 200:
        return None
    sub = sub.sort_values('ts').reset_index(drop=True)
    cut = int(len(sub) * train_frac)
    train = sub.iloc[:cut]
    val = sub.iloc[cut:]
    if len(train) < 100 or len(val) < 50:
        return None

    train_pnl = train['actual_pnl'].values.astype(np.float64)
    train_feats = np.stack(train['entry_v2'].values)
    train_win_label = (train_pnl > margin).astype(np.int32)
    train_los_mask = train_pnl < -margin

    # Rank features on train by Cohen's d (winner vs loser)
    win_mask = train_pnl > margin
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col = train_feats[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 100:
            continue
        d = cohens_d(col[win_mask], col[train_los_mask])
        rho = spearman_corr(col, train_pnl)
        rows.append({'feature': name, 'd': d, 'rho': rho,
                          'score': abs(d) + abs(rho), 'j': j})
    if not rows:
        return None
    rdf = pd.DataFrame(rows).sort_values('score', ascending=False).head(top_k)

    # Try the top-K and pick the one with the best validation accuracy
    val_pnl = val['actual_pnl'].values.astype(np.float64)
    val_feats = np.stack(val['entry_v2'].values)
    val_lbl = (val_pnl > margin).astype(np.int32)

    best = None
    for _, r in rdf.iterrows():
        j = int(r['j'])
        # Find threshold on TRAIN only
        valid_train = ~np.isnan(train_feats[:, j])
        bin_lbl = (train_pnl > margin).astype(np.int32)
        thr, skip_above, train_acc = find_best_threshold(
            train_feats[valid_train, j], bin_lbl[valid_train])
        if thr is None:
            continue
        # Validate on VAL
        valid_val = ~np.isnan(val_feats[:, j])
        if valid_val.sum() < 30:
            continue
        if skip_above:
            kept_val = val_feats[valid_val, j] <= thr
        else:
            kept_val = val_feats[valid_val, j] >= thr
        if kept_val.sum() < 30:
            continue
        # Validation: did the filter help PnL?
        val_kept_pnl = val_pnl[valid_val][kept_val].mean()
        val_skip_pnl = val_pnl[valid_val][~kept_val].mean()
        # Accuracy = ratio of kept being winners
        kept_wr_val = (val_pnl[valid_val][kept_val] > 0).mean()
        all_wr_val = (val_pnl > 0).mean()
        # Survival criterion: kept WR > all WR by at least 2pp AND kept_per_trade > all_per_trade
        survives = (kept_wr_val > all_wr_val + 0.02 and
                       val_kept_pnl > val_pnl.mean())
        candidate = {
            'feature': r['feature'],
            'threshold': float(thr),
            'skip_above': bool(skip_above),
            'cohens_d': float(r['d']),
            'train_acc': float(train_acc),
            'train_n': int(len(train)),
            'val_n_kept': int(kept_val.sum()),
            'val_n_skip': int((~kept_val).sum()),
            'val_kept_per_trade': float(val_kept_pnl),
            'val_skip_per_trade': float(val_skip_pnl),
            'val_kept_wr': float(kept_wr_val),
            'val_all_wr': float(all_wr_val),
            'survives': bool(survives),
        }
        if best is None or (candidate['survives'] and not best['survives']) or \
              (candidate['survives'] == best['survives'] and
               candidate['val_kept_per_trade'] > best['val_kept_per_trade']):
            best = candidate

    return best


def main():
    p = argparse.ArgumentParser(description='Learn per-cell entry quality filters')
    p.add_argument('--trades', default='training_v2/output/nmp_only.pkl')
    p.add_argument('--regret', default='training_v2/output/regret_nmp.pkl')
    p.add_argument('--out', default='training_v2/output/cell_filters.json')
    p.add_argument('--margin', type=float, default=5.0)
    p.add_argument('--train-frac', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=5)
    args = p.parse_args()

    df = load_joined(args.trades, args.regret)
    print(f'Loaded {len(df)} trades')

    filters = {}
    print()
    print(f'  {"cell":<25} {"n":>5} {"feature":<32} {"thr":>9} {"side":>6} '
              f'{"kept_WR":>8} {"all_WR":>7} {"surv?":>6}')
    print('-' * 110)
    for r in sorted(df['regime_idx'].unique()):
        rname = REGIME_VOCAB[int(r)] if int(r) < len(REGIME_VOCAB) else f'R{r}'
        for direction in ('long', 'short'):
            sub = df[(df['regime_idx'] == r) & (df['direction'] == direction)]
            cell_label = f'{rname}|{direction}'
            spec = learn_cell_filter(sub, margin=args.margin,
                                              train_frac=args.train_frac,
                                              top_k=args.top_k)
            if spec is None:
                print(f'  {cell_label:<25} {len(sub):>5}  (insufficient data)')
                continue
            side = 'above' if spec['skip_above'] else 'below'
            ok = 'YES' if spec['survives'] else 'no'
            print(f'  {cell_label:<25} {len(sub):>5} {spec["feature"]:<32} '
                      f'{spec["threshold"]:>+8.2f} {side:>6} '
                      f'{spec["val_kept_wr"]:>7.1%} {spec["val_all_wr"]:>6.1%} '
                      f'{ok:>6}')
            if spec['survives']:
                filters[f'{int(r)}|{direction}'] = spec

    print(f'\nSurviving filters: {len(filters)}')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(filters, f, indent=2)
    print(f'Saved -> {args.out}')


if __name__ == '__main__':
    main()
