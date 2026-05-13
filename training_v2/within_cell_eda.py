"""Within-cell V2 feature EDA — per (regime × direction) cell.

For each of the 12 (regime, direction) cells, run the same WINNER vs LOSER
Cohen's d analysis as full_feature_eda.py but RESTRICTED to that cell's trades.
This surfaces cell-specific patterns that the global EDA averages out.

Output: per-cell ranking of top discriminating features. Each is also
walk-forward validated INSIDE the cell (70/30 split by date).

Usage:
    python -m training_v2.within_cell_eda
    python -m training_v2.within_cell_eda --top-k 10 --margin 5
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training_v2.regret import RegretLabel
from training_v2.ledger import ClosedTrade
from training_v2.tier_discovery import load_joined, cohens_d
from training_v2.full_feature_eda import spearman_corr
from training_v2.state import REGIME_VOCAB


def analyze_cell(sub_df: pd.DataFrame, margin: float, top_k: int,
                       train_frac: float = 0.7) -> Tuple[pd.DataFrame, int, int]:
    """Run feature ranking + walk-forward inside one cell."""
    pnl = sub_df['actual_pnl'].values.astype(np.float64)
    feats = np.stack(sub_df['entry_v2'].values)
    win = pnl > margin
    los = pnl < -margin
    if win.sum() < 30 or los.sum() < 30:
        return pd.DataFrame(), int(win.sum()), int(los.sum())

    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col = feats[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 100:
            continue
        d = cohens_d(col[win], col[los])
        rho = spearman_corr(col, pnl)
        rows.append({
            'feature': name, 'd': d, 'rho': rho,
            'abs_d': abs(d), 'abs_rho': abs(rho),
            'mean_winner': float(np.nanmean(col[win])) if win.sum() else 0,
            'mean_loser': float(np.nanmean(col[los])) if los.sum() else 0,
        })
    if not rows:
        return pd.DataFrame(), int(win.sum()), int(los.sum())

    out = pd.DataFrame(rows)
    out['score'] = out['abs_d'] + out['abs_rho']
    out = out.sort_values('score', ascending=False).reset_index(drop=True)

    # Walk-forward survival within cell
    sub_sorted = sub_df.sort_values('ts').reset_index(drop=True)
    cut = int(len(sub_sorted) * train_frac)
    if cut < 50 or (len(sub_sorted) - cut) < 50:
        out['survives'] = False
        return out.head(top_k), int(win.sum()), int(los.sum())

    train_pnl = sub_sorted.iloc[:cut]['actual_pnl'].values.astype(np.float64)
    val_pnl = sub_sorted.iloc[cut:]['actual_pnl'].values.astype(np.float64)
    train_feats = np.stack(sub_sorted.iloc[:cut]['entry_v2'].values)
    val_feats = np.stack(sub_sorted.iloc[cut:]['entry_v2'].values)
    name_to_j = {n: i for i, n in enumerate(FEATURE_NAMES)}

    surv_flags = []
    for name in out.head(top_k)['feature']:
        j = name_to_j[name]
        rho_t = spearman_corr(train_feats[:, j], train_pnl)
        rho_v = spearman_corr(val_feats[:, j], val_pnl)
        same_sign = np.sign(rho_t) == np.sign(rho_v) and rho_t != 0 and rho_v != 0
        surv_flags.append(bool(same_sign and abs(rho_v) >= 0.05))
    head = out.head(top_k).copy()
    head['survives'] = surv_flags
    return head, int(win.sum()), int(los.sum())


def main():
    p = argparse.ArgumentParser(description='Within-cell V2 feature EDA')
    p.add_argument('--trades', default='training_v2/output/nmp_only.pkl')
    p.add_argument('--regret', default='training_v2/output/regret_nmp.pkl')
    p.add_argument('--margin', type=float, default=5.0)
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--out', default='reports/findings/v2_within_cell_eda.md')
    args = p.parse_args()

    df = load_joined(args.trades, args.regret)
    print(f'Loaded {len(df)} trades')

    md_lines = ['# Within-cell V2 feature EDA on Base NMP\n',
                    f'Trades: {len(df)}; margin = ${args.margin}\n']

    print()
    print('=' * 80)
    print(f'{"cell":<25} {"n":>5} {"WIN":>5} {"LOSE":>5} {"top_d":>7} '
              f'{"top_feature":<30} {"surv?":>5}')
    print('=' * 80)

    summary_rows = []
    for r in sorted(df['regime_idx'].unique()):
        rname = REGIME_VOCAB[int(r)] if int(r) < len(REGIME_VOCAB) else f'R{r}'
        for direction in ('long', 'short'):
            sub = df[(df['regime_idx'] == r) & (df['direction'] == direction)]
            if len(sub) < 100:
                continue
            head, n_win, n_los = analyze_cell(sub, args.margin, args.top_k)
            if head.empty:
                continue
            top = head.iloc[0]
            cell_label = f'{rname}|{direction}'
            print(f'{cell_label:<25} {len(sub):>5} {n_win:>5} {n_los:>5} '
                      f'{top["d"]:>+7.3f} {top["feature"]:<30} '
                      f'{"YES" if top["survives"] else "no":>5}')

            md_lines.append(f'\n## {cell_label}  (n={len(sub)}, '
                                  f'WIN={n_win}, LOSE={n_los})\n')
            md_lines.append(f'\n| feature | d | rho | mean WIN | mean LOSE | survives |\n'
                                  f'|---|---:|---:|---:|---:|---|')
            for _, row in head.iterrows():
                ok = 'YES' if row['survives'] else 'no'
                md_lines.append(f'\n| {row["feature"]} | {row["d"]:+.3f} | '
                                      f'{row["rho"]:+.3f} | {row["mean_winner"]:+.2f} | '
                                      f'{row["mean_loser"]:+.2f} | {ok} |')
            summary_rows.append({
                'cell': cell_label, 'n': len(sub),
                'top_feature': top['feature'], 'top_d': top['d'],
                'top_rho': top['rho'], 'survives': top['survives'],
            })

    print('=' * 80)
    print(f'\nSummary across {len(summary_rows)} cells:')
    surv_count = sum(1 for r in summary_rows if r['survives'])
    print(f'  Top-feature walk-forward survivors: {surv_count}/{len(summary_rows)}')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f'\nFull report -> {args.out}')


if __name__ == '__main__':
    main()
