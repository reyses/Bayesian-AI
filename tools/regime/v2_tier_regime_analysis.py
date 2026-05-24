"""
v2_tier_regime_analysis.py — analyze each of the 9 ExNMP tiers stratified
by regime_2d, to identify which tier benefits most from a regime gate.

For each tier:
  1. Baseline: total $/day, $/trade, WR, count
  2. Per-regime: same metrics + bootstrap CI on $/day
  3. Per-(regime, split): IS/VAL/OOS breakdown
  4. Regime-gated $/day: $/day if the tier only fires in its top-N regimes
  5. Lift vs baseline + 95% bootstrap CI on the lift

Output ranks tiers by:
  - Magnitude of expected lift from gating (delta vs baseline)
  - Statistical confidence (CI on delta excludes 0)
  - Trade-count preservation (don't gate to noise)

Outputs:
  reports/findings/v2_tier_regime/
    tier_regime_summary.csv     per (tier, regime): n, pnl, $/day, $/trade, WR
    tier_regime_split.csv       per (tier, regime, split)
    gate_recommendations.csv    tier-by-tier gate candidates with lift CI
    summary.md
"""

from __future__ import annotations
import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.atlas_regime_labeler_2d import load_regime_labels, REGIME_2D_ORDER


TIERS = [
    'CASCADE', 'KILL_SHOT', 'FREIGHT_TRAIN', 'FADE_AGAINST',
    'RIDE_AGAINST', 'FADE_CALM', 'MTF_BREAKOUT', 'MTF_EXHAUSTION',
    'BASE_NMP',
]


def bootstrap_ci_per_day(daily_pnl: np.ndarray, n_resamples: int = 4000,
                          ci_level: float = 0.95) -> tuple[float, float]:
    if len(daily_pnl) < 5:
        return float('nan'), float('nan')
    rng = np.random.default_rng(42)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(daily_pnl), len(daily_pnl))
        boots[i] = daily_pnl[idx].mean()
    lo = float(np.quantile(boots, (1 - ci_level) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci_level) / 2))
    return lo, hi


def bootstrap_ci_delta_paired(a: np.ndarray, b: np.ndarray,
                                  n_resamples: int = 4000) -> tuple[float, float]:
    """CI on mean(b) - mean(a) via PAIRED bootstrap (same indices both arms).

    Use when a and b are aligned per-day measurements (e.g. baseline per-day
    PnL and gated per-day PnL on the same calendar). Treats correlation
    between paired samples correctly; tighter CI than independent bootstrap.
    """
    if len(a) < 5 or len(b) < 5 or len(a) != len(b):
        return float('nan'), float('nan')
    rng = np.random.default_rng(42)
    n = len(a)
    boots = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        boots[i] = b[idx].mean() - a[idx].mean()
    lo = float(np.quantile(boots, 0.025))
    hi = float(np.quantile(boots, 0.975))
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isolated-dir', default='training/output/isolated')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_tier_regime')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load labels
    labels = load_regime_labels(args.labels_csv).copy()
    labels['date'] = labels['date'].astype(str).str[:10]

    summary_rows = []
    split_rows = []
    gate_rows = []

    for tier in TIERS:
        path = os.path.join(args.isolated_dir, f'{tier}.pkl')
        if not os.path.exists(path):
            print(f'  {tier}: missing isolated file, skipping')
            continue
        with open(path, 'rb') as f:
            trades = pickle.load(f)
        df = pd.DataFrame(trades)
        if len(df) == 0:
            continue

        # Convert day "2025_01_02" -> "2025-01-02"
        df['date'] = df['day'].astype(str).str.replace('_', '-')
        df = df.merge(labels[['date', 'regime_2d', 'split']],
                       on='date', how='left')
        df = df.dropna(subset=['regime_2d'])

        # ---- Baseline ----
        daily_pnl_baseline = df.groupby('date')['pnl'].sum().values
        baseline_per_day = float(daily_pnl_baseline.mean())
        bl_lo, bl_hi = bootstrap_ci_per_day(daily_pnl_baseline)
        baseline_n_days = len(daily_pnl_baseline)
        baseline_n_trades = len(df)
        baseline_wr = float((df['pnl'] > 0).mean())
        baseline_per_trade = float(df['pnl'].mean())

        # Per regime
        for regime in REGIME_2D_ORDER:
            sub = df[df['regime_2d'] == regime]
            if len(sub) == 0:
                continue
            daily_pnl_r = sub.groupby('date')['pnl'].sum().values
            n_days_r = len(daily_pnl_r)
            n_trades_r = len(sub)
            mean_per_day_r = float(daily_pnl_r.mean()) if n_days_r > 0 else 0
            wr_r = float((sub['pnl'] > 0).mean())
            per_trade_r = float(sub['pnl'].mean())
            lo_r, hi_r = bootstrap_ci_per_day(daily_pnl_r)
            summary_rows.append({
                'tier': tier,
                'regime_2d': regime,
                'n_days': n_days_r,
                'n_trades': n_trades_r,
                'sum_pnl': float(sub['pnl'].sum()),
                'mean_per_day': mean_per_day_r,
                'mean_per_trade': per_trade_r,
                'win_rate': wr_r,
                'ci_lo_per_day': lo_r,
                'ci_hi_per_day': hi_r,
                'baseline_per_day': baseline_per_day,
            })

            # split breakdown
            for split in ['IS', 'VAL', 'OOS']:
                ssub = sub[sub['split'] == split]
                if len(ssub) == 0:
                    continue
                ddaily = ssub.groupby('date')['pnl'].sum().values
                split_rows.append({
                    'tier': tier,
                    'regime_2d': regime,
                    'split': split,
                    'n_days': len(ddaily),
                    'n_trades': len(ssub),
                    'mean_per_day': float(ddaily.mean()) if len(ddaily) else 0,
                    'mean_per_trade': float(ssub['pnl'].mean()),
                    'win_rate': float((ssub['pnl'] > 0).mean()),
                })

        # ---- Gate recommendations ----
        # For each candidate gate (single regime, pair of regimes, etc.),
        # compute filtered $/day vs baseline. Report ones with CI on delta
        # excluding 0.

        # Single-regime gates
        for regime in REGIME_2D_ORDER:
            keep_mask = df['regime_2d'] == regime
            if keep_mask.sum() < 30:
                continue
            kept = df[keep_mask]
            daily_kept = kept.groupby('date')['pnl'].sum().reindex(
                df['date'].unique(), fill_value=0).values
            # use UNION of dates so gates that fire less often look at the
            # same denominator
            mean_kept = float(daily_kept.mean())
            # delta vs baseline
            delta = mean_kept - baseline_per_day
            lo_d, hi_d = bootstrap_ci_delta_paired(daily_pnl_baseline, daily_kept)
            gate_rows.append({
                'tier': tier,
                'gate': f'KEEP_{regime}',
                'kept_n_trades': int(keep_mask.sum()),
                'kept_pct_trades': 100.0 * keep_mask.sum() / max(len(df), 1),
                'kept_mean_per_day': mean_kept,
                'baseline_mean_per_day': baseline_per_day,
                'delta_per_day': delta,
                'delta_ci_lo': lo_d,
                'delta_ci_hi': hi_d,
                'delta_significant': lo_d > 0 or hi_d < 0,
            })

        # Best 2-regime gate (positive-$/day pair)
        regime_means = {}
        for regime in REGIME_2D_ORDER:
            sub = df[df['regime_2d'] == regime]
            if len(sub) < 30:
                continue
            ddaily = sub.groupby('date')['pnl'].sum().values
            if len(ddaily) > 0:
                regime_means[regime] = float(ddaily.sum() / max(baseline_n_days, 1))

        # Sort regimes by per-day contribution
        sorted_regimes = sorted(regime_means.items(), key=lambda kv: -kv[1])
        for k in [1, 2, 3, 4]:
            top_k_regimes = [r for r, _ in sorted_regimes[:k]]
            keep_mask = df['regime_2d'].isin(top_k_regimes)
            if keep_mask.sum() < 30:
                continue
            kept = df[keep_mask]
            daily_kept = kept.groupby('date')['pnl'].sum().reindex(
                df['date'].unique(), fill_value=0).values
            mean_kept = float(daily_kept.mean())
            delta = mean_kept - baseline_per_day
            lo_d, hi_d = bootstrap_ci_delta_paired(daily_pnl_baseline, daily_kept)
            gate_rows.append({
                'tier': tier,
                'gate': f'KEEP_TOP_{k}: ' + '+'.join(top_k_regimes),
                'kept_n_trades': int(keep_mask.sum()),
                'kept_pct_trades': 100.0 * keep_mask.sum() / max(len(df), 1),
                'kept_mean_per_day': mean_kept,
                'baseline_mean_per_day': baseline_per_day,
                'delta_per_day': delta,
                'delta_ci_lo': lo_d,
                'delta_ci_hi': hi_d,
                'delta_significant': lo_d > 0 or hi_d < 0,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(args.output_dir, 'tier_regime_summary.csv'),
                        index=False)
    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(os.path.join(args.output_dir, 'tier_regime_split.csv'),
                      index=False)
    gate_df = pd.DataFrame(gate_rows).sort_values(
        ['tier', 'delta_per_day'], ascending=[True, False])
    gate_df.to_csv(os.path.join(args.output_dir, 'gate_recommendations.csv'),
                     index=False)

    # Print: per-tier baseline + per-regime mean $/day
    print(f"{'='*100}")
    print(f"  PER-TIER BASELINE + PER-REGIME $/DAY (348 days IS+VAL+OOS combined)")
    print(f"{'='*100}")
    print(f"  {'tier':>16}  {'days':>4}  {'tr':>5}  {'baseline':>9}  "
          f"{'UP_SM':>7}  {'UP_CH':>7}  {'DN_SM':>7}  {'DN_CH':>7}  {'FL_SM':>7}  {'FL_CH':>7}")
    for tier in TIERS:
        if tier not in summary_df['tier'].values:
            continue
        base_per_day = summary_df[summary_df['tier'] == tier].iloc[0]['baseline_per_day']
        n_total_days = int(summary_df[summary_df['tier'] == tier]['n_days'].sum())
        n_trades = int(summary_df[summary_df['tier'] == tier]['n_trades'].sum())
        regime_to_per_day = dict(zip(
            summary_df[summary_df['tier'] == tier]['regime_2d'],
            summary_df[summary_df['tier'] == tier]['mean_per_day']))
        cols = []
        for rr in REGIME_2D_ORDER:
            v = regime_to_per_day.get(rr, float('nan'))
            cols.append(f'{v:>+7.2f}')
        print(f"  {tier:>16}  {n_total_days:>4}  {n_trades:>5}  "
              f"{base_per_day:>+9.2f}  {'  '.join(cols)}")

    # Highlight: top gate recommendations by delta with significant CI
    print(f"\n{'='*100}")
    print(f"  TOP REGIME-GATE RECOMMENDATIONS (delta_per_day with 95% CI)")
    print(f"{'='*100}")
    sig = gate_df[gate_df['delta_significant']].copy()
    sig = sig.sort_values('delta_per_day', ascending=False)
    print(f"  {'tier':>16}  {'gate':>50}  {'kept_pct':>8}  "
          f"{'delta':>7}  {'CI_low':>7}  {'CI_high':>7}")
    for _, r in sig.head(40).iterrows():
        print(f"  {r['tier']:>16}  {r['gate']:>50}  "
              f"{r['kept_pct_trades']:>7.1f}%  "
              f"{r['delta_per_day']:>+7.2f}  "
              f"{r['delta_ci_lo']:>+7.2f}  {r['delta_ci_hi']:>+7.2f}")

    # Best gate per tier
    print(f"\n{'='*100}")
    print(f"  BEST GATE PER TIER (by delta_per_day, with CI)")
    print(f"{'='*100}")
    best_per_tier = []
    for tier, g in gate_df.groupby('tier'):
        # filter to single-regime gates and top-K with significance
        g_sig = g[g['delta_significant'] & (g['delta_per_day'] > 0)]
        if len(g_sig) == 0:
            best_per_tier.append({
                'tier': tier, 'gate': 'NO_SIGNIFICANT_GATE',
                'delta_per_day': 0, 'kept_pct': 100,
                'delta_ci_lo': float('nan'), 'delta_ci_hi': float('nan'),
            })
            continue
        best = g_sig.sort_values('delta_per_day', ascending=False).iloc[0]
        best_per_tier.append({
            'tier': tier, 'gate': best['gate'],
            'delta_per_day': best['delta_per_day'],
            'kept_pct': best['kept_pct_trades'],
            'delta_ci_lo': best['delta_ci_lo'],
            'delta_ci_hi': best['delta_ci_hi'],
        })

    bp_df = pd.DataFrame(best_per_tier).sort_values('delta_per_day',
                                                              ascending=False)
    bp_df.to_csv(os.path.join(args.output_dir, 'best_gate_per_tier.csv'),
                   index=False)

    print(f"  {'tier':>16}  {'best_gate':>50}  {'kept':>5}  {'delta':>7}  CI")
    for _, r in bp_df.iterrows():
        ci = (f'[{r["delta_ci_lo"]:>+5.1f},{r["delta_ci_hi"]:>+5.1f}]'
                if not pd.isna(r['delta_ci_lo']) else 'n/a')
        print(f"  {r['tier']:>16}  {r['gate']:>50}  {r['kept_pct']:>4.1f}%  "
              f"{r['delta_per_day']:>+7.2f}  {ci}")

    # Markdown summary
    md = os.path.join(args.output_dir, 'summary.md')
    with open(md, 'w', encoding='utf-8') as f:
        f.write(f"# 9-tier regime analysis - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write("## Per-tier baseline + per-regime $/day\n\n")
        # build pivot table
        pv = summary_df.pivot_table(index='tier', columns='regime_2d',
                                          values='mean_per_day').fillna(0)
        pv = pv.reindex(columns=[r for r in REGIME_2D_ORDER if r in pv.columns],
                          index=[t for t in TIERS if t in pv.index])
        f.write(pv.round(2).to_string())
        f.write("\n\n## Best regime gate per tier\n\n")
        f.write(bp_df.to_string(index=False))
        f.write("\n\n## All significant gate recommendations\n\n")
        f.write(sig.to_string(index=False))
        f.write("\n")
    print(f"\n  [saved] {md}")


if __name__ == '__main__':
    main()
