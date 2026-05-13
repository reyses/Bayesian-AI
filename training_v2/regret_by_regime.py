"""Regret stratified by regime — per-(regime, direction) best-action distribution.

We already know the global picture:
    50.1% of trades wanted same_extended, 49.3% wanted counter_extended.

But regime-aware tiers were the splitter that actually survived OOS validation
(2026-05-04 flip rule). The question: does the regret distribution lean
asymmetrically by regime? If UP_SMOOTH long has 80% same_extended and only
20% counter_extended, we have strong directional bias for that cell.
If FLAT_CHOPPY is 50/50, no extra signal.

Per (regime, direction), reports:
  - Best-action distribution (6 actions)
  - Mean actual_pnl, best_pnl, regret
  - Mean fade_peak (peak in original direction)
  - Mean flip_peak (peak if direction had been opposite)
  - Mean early_entry_gain
  - Capture ratio (actual/peak)

Usage:
    python -m training_v2.regret_by_regime
    python -m training_v2.regret_by_regime --regret training_v2/output/regret_full_nmp.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
from typing import List

import numpy as np
import pandas as pd

from training_v2.regret_full import FullRegretLabel, ACTIONS
from training_v2.state import REGIME_VOCAB


def labels_to_df(labels: List[FullRegretLabel]) -> pd.DataFrame:
    """Flatten labels for groupby analysis."""
    rows = []
    for l in labels:
        rname = (REGIME_VOCAB[int(l.entry_regime_idx)]
                     if int(l.entry_regime_idx) < len(REGIME_VOCAB)
                     else f'R{l.entry_regime_idx}')
        rows.append({
            'tier': l.entry_tier,
            'regime_idx': int(l.entry_regime_idx),
            'regime': rname,
            'direction': l.direction,
            'actual_pnl': l.actual_pnl,
            'best_pnl': l.best_pnl,
            'regret': l.regret,
            'best_action': l.best_action,
            'fade_peak': l.peak_pnl,                       # peak in original direction
            'flip_peak': -l.mae_pnl,                       # peak if direction had been opposite
            'early_entry_gain': l.early_entry_gain,
            'capture_ratio': l.capture_ratio,
            'time_to_peak_s': (l.peak_bar * 5.0),
            'mae_pnl': l.mae_pnl,
        })
    return pd.DataFrame(rows)


def best_action_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows = (regime, direction), cols = best_action %, plus mean regret."""
    rows = []
    for (regime, direction), sub in df.groupby(['regime', 'direction']):
        n = len(sub)
        actions = sub['best_action'].value_counts(normalize=True)
        row = {
            'regime': regime, 'direction': direction, 'n': n,
            'mean_actual': sub['actual_pnl'].mean(),
            'mean_best': sub['best_pnl'].mean(),
            'mean_regret': sub['regret'].mean(),
            'mean_fade_peak': sub['fade_peak'].mean(),
            'mean_flip_peak': sub['flip_peak'].mean(),
            'mean_early_entry_gain': sub['early_entry_gain'].mean(),
        }
        for action in ACTIONS:
            row[f'pct_{action}'] = actions.get(action, 0.0) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description='Regret stratified by regime')
    p.add_argument('--regret', default='training_v2/output/regret_full_nmp.pkl')
    p.add_argument('--out', default='reports/findings/v2_regret_by_regime.md')
    args = p.parse_args()

    with open(args.regret, 'rb') as f:
        labels = pickle.load(f)
    df = labels_to_df(labels)
    print(f'Loaded {len(df)} labeled trades')

    # ── Per-regime best-action distribution ───────────────────────────────
    print(f'\n=== Best-action distribution by (regime, direction) ===')
    by_cell = best_action_distribution(df)
    by_cell = by_cell.sort_values(['regime', 'direction']).reset_index(drop=True)

    # Print compact view
    print(f'\n{"regime":<13} {"dir":>5} {"n":>5} {"actual":>8} {"best":>8} '
              f'{"regret":>8} {"fadeP":>7} {"flipP":>7} | '
              f'{"sm_e":>5} {"sm_at":>5} {"sm_x":>5} {"ct_e":>5} {"ct_at":>5} {"ct_x":>5}')
    print('-' * 130)
    for _, r in by_cell.iterrows():
        print(f'{r["regime"]:<13} {r["direction"]:>5} {int(r["n"]):>5} '
                  f'${r["mean_actual"]:>+6.2f} ${r["mean_best"]:>+6.2f} '
                  f'${r["mean_regret"]:>+6.2f} ${r["mean_fade_peak"]:>+5.0f} '
                  f'${r["mean_flip_peak"]:>+5.0f} | '
                  f'{r["pct_same_early"]:>4.1f}% {r["pct_same_at_exit"]:>4.1f}% '
                  f'{r["pct_same_extended"]:>4.1f}% {r["pct_counter_early"]:>4.1f}% '
                  f'{r["pct_counter_at_exit"]:>4.1f}% {r["pct_counter_extended"]:>4.1f}%')

    # ── Direction asymmetry: which cells lean strongly to same vs counter? ──
    print(f'\n=== Direction asymmetry per (regime, direction) cell ===')
    print(f'(strong asymmetry = clear trend signal at this cell — flip is unambiguous)')
    print()
    print(f'{"regime":<13} {"dir":>5} {"sm_extended":>13} {"ct_extended":>13} {"asymmetry":>11} {"verdict":>12}')
    print('-' * 80)
    for _, r in by_cell.iterrows():
        sm_x = r['pct_same_extended']
        ct_x = r['pct_counter_extended']
        asym = sm_x - ct_x
        if asym > 15:
            verdict = 'KEEP (fade)'
        elif asym < -15:
            verdict = 'FLIP (ride)'
        else:
            verdict = 'mixed'
        print(f'{r["regime"]:<13} {r["direction"]:>5} {sm_x:>11.1f}% '
                  f'{ct_x:>11.1f}% {asym:>+10.1f}% {verdict:>12}')

    # ── Regret per regime — where is the most opportunity? ───────────────
    print(f'\n=== Regret distribution by regime (which regime leaks most?) ===')
    by_regime = df.groupby('regime').agg(
        n=('actual_pnl', 'size'),
        mean_actual=('actual_pnl', 'mean'),
        mean_best=('best_pnl', 'mean'),
        mean_regret=('regret', 'mean'),
        mean_fade_peak=('fade_peak', 'mean'),
        mean_flip_peak=('flip_peak', 'mean'),
        mean_early_gain=('early_entry_gain', 'mean'),
        median_capture=('capture_ratio', 'median'),
    ).reset_index().sort_values('mean_regret', ascending=False)
    print(f'{"regime":<14} {"n":>5} {"actual":>8} {"best":>8} {"regret":>8} '
              f'{"fadeP":>7} {"flipP":>7} {"early$":>8} {"capture":>8}')
    for _, r in by_regime.iterrows():
        cap = r['median_capture'] if r['median_capture'] == r['median_capture'] else 0
        print(f'{r["regime"]:<14} {int(r["n"]):>5} ${r["mean_actual"]:>+6.2f} '
                  f'${r["mean_best"]:>+6.2f} ${r["mean_regret"]:>+6.2f} '
                  f'${r["mean_fade_peak"]:>+5.0f} ${r["mean_flip_peak"]:>+5.0f} '
                  f'${r["mean_early_gain"]:>+6.2f} {cap:>7.2f}')

    # ── Save markdown ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('# V2 Regret Stratified by Regime\n\n')
        f.write(f'Source: {args.regret}\n\n')
        f.write(f'Trades: {len(df)}\n\n')
        f.write('## Best-action distribution per (regime, direction) cell\n\n')
        f.write(by_cell.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
        f.write(f'\n\n## Per-regime regret summary\n\n')
        f.write(by_regime.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print(f'\nSaved -> {args.out}')


if __name__ == '__main__':
    main()
