"""Regret analysis on iso pipeline output — per-tier multi-axis counterfactuals.

For each tier pickle produced by run_iso.py, computes FullRegretLabel
(early/late entry × early/late exit × same/counter direction) and reports
per-tier:
    - best_action distribution (which counterfactual was best on average)
    - mode + mean of actual_pnl, best_pnl, regret  (mode bin = $2 / $10)
    - PF-based Trade WR (CLAUDE.md metric)

Usage:
    # Default: glob every IS tier pickle from run_iso.py
    python -m training_iso_v2.regret_iso

    # OOS
    python -m training_iso_v2.regret_iso --prefix training_iso_v2/output/oos
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_iso_v2.regret_full import (FullRegretLabel, label_trades_full,
                                                          ACTIONS)
from training_iso_v2.ledger import ClosedTrade
from training_iso_v2.run_iso import _histogram_mode, _pf_trade_wr


def per_tier_regret(trades: List[ClosedTrade]) -> List[FullRegretLabel]:
    """Build full regret labels for one tier's trades."""
    if not trades:
        return []
    return label_trades_full(trades, store_curves=False)


def summary(name: str, labels: List[FullRegretLabel]) -> dict:
    if not labels:
        return {'tier': name, 'n': 0}
    actual = np.array([l.actual_pnl for l in labels], dtype=np.float64)
    best = np.array([l.best_pnl for l in labels], dtype=np.float64)
    regret = np.array([l.regret for l in labels], dtype=np.float64)
    actions = pd.Series([l.best_action for l in labels])
    action_pcts = actions.value_counts(normalize=True)

    out = {
        'tier': name, 'n': len(labels),
        'mode_actual': _histogram_mode(actual, bin_width=2.0),
        'mean_actual': float(actual.mean()),
        'mode_best': _histogram_mode(best, bin_width=10.0),
        'mean_best': float(best.mean()),
        'mode_regret': _histogram_mode(regret, bin_width=10.0),
        'mean_regret': float(regret.mean()),
        'pf_wr': _pf_trade_wr(actual),
    }
    for action in ACTIONS:
        out[f'pct_{action}'] = float(action_pcts.get(action, 0.0))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefix', default='training_iso_v2/output/is',
                       help='glob prefix; finds {prefix}_<TIER>.pkl')
    p.add_argument('--out', default=None,
                       help='save per-tier regret pickles to <out>_<TIER>.pkl')
    args = p.parse_args()

    paths = sorted(glob.glob(f'{args.prefix}_*.pkl'))
    # Skip files like `*_summary.csv`, `*_<TIER>_regret.pkl`, etc. Keep tier files only.
    paths = [p for p in paths
                  if not (p.endswith('_summary.csv') or '_regret.pkl' in p)]
    if not paths:
        print(f'No tier pickles found at {args.prefix}_*.pkl')
        return

    print(f'Found {len(paths)} tier pickles:')
    for p in paths:
        print(f'  {p}')

    all_summaries = []
    for path in paths:
        # Extract tier name from path
        base = os.path.basename(path).replace('.pkl', '')
        tier = base.split('_', 1)[1] if '_' in base else base

        with open(path, 'rb') as f:
            trades = pickle.load(f)
        print(f'\n[{tier}]  loading {len(trades)} trades...')
        if not trades:
            all_summaries.append({'tier': tier, 'n': 0})
            continue

        labels = per_tier_regret(trades)
        s = summary(tier, labels)
        all_summaries.append(s)

        if args.out:
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            out_path = f'{args.out}_{tier}_regret.pkl'
            with open(out_path, 'wb') as f:
                pickle.dump(labels, f)
            print(f'  saved -> {out_path}')

    # ── Per-tier summary table ──────────────────────────────────────────
    print(f'\n{"="*100}')
    print(f'PER-TIER REGRET SUMMARY')
    print(f'  $/trade: mode (bin=$2) and mean   |   $regret: mode (bin=$10) and mean')
    print(f'  PF-WR  : (sum_profits / |sum_losses|) - 1   (0 = break-even)')
    print(f'{"="*100}')
    print(f'{"tier":<16} {"n":>6} {"mode$/t":>8} {"mean$/t":>8} {"PF-WR":>7} '
              f'{"mode_reg":>9} {"mean_reg":>9}  '
              f'{"%sm_x":>5} {"%ct_x":>5} {"%sm_e":>5} {"%ct_e":>5}')
    print('-' * 100)
    for s in all_summaries:
        if s['n'] == 0:
            print(f'{s["tier"]:<16} {0:>6}')
            continue
        pf = s['pf_wr']
        pf_str = f'{pf:>+6.2f}' if pf != float('inf') else '   inf'
        print(f'{s["tier"]:<16} {s["n"]:>6} '
                  f'${s["mode_actual"]:>+7.2f} ${s["mean_actual"]:>+7.2f} {pf_str} '
                  f'${s["mode_regret"]:>+8.2f} ${s["mean_regret"]:>+8.2f}  '
                  f'{s.get("pct_same_extended",0)*100:>4.1f}% '
                  f'{s.get("pct_counter_extended",0)*100:>4.1f}% '
                  f'{s.get("pct_same_early",0)*100:>4.1f}% '
                  f'{s.get("pct_counter_early",0)*100:>4.1f}%')

    df = pd.DataFrame(all_summaries)
    csv_path = f'{args.prefix}_regret_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nCSV summary saved: {csv_path}')


if __name__ == '__main__':
    main()
