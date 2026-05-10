"""Day-of-week conditioning on the catastrophic-risk table.

Reuses the bar_substrate.parquet from bayes_table_at_bar_cat_risk.py
(no recomputation needed) and adds DOW as a conditioning axis.

Tests user hypothesis: are catastrophic events clustered around specific
days of the week?

USAGE
    python tools/bayes_table_cat_risk_dow.py
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist, chi2_contingency

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def beta_p(k, n):
    a, b = k + 1, n - k + 1
    return (float(a / (a + b)),
             float(beta_dist.ppf(0.025, a, b)),
             float(beta_dist.ppf(0.975, a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--substrate',
                     default='reports/findings/segments/bayes_table_at_bar_cat_risk/bar_substrate.parquet')
    ap.add_argument('--out-dir',
                     default='reports/findings/segments/bayes_table_cat_risk_dow')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.substrate)
    print(f'Loaded {len(df):,} sampled bars from {df["day"].nunique()} days')
    print(f'Unconditional P(cat in next 60m): {100*df["cat_in_next_60m"].mean():.2f}%')

    DOW_ORDER = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    df['dow'] = pd.Categorical(df['dow'], categories=DOW_ORDER, ordered=True)
    df = df[df['dow'].notna()]

    # ===== Marginal: P(cat | DOW) =====
    print(f'\n=== P(cat in next 60m | DOW) ===')
    rows = []
    for dow, g in df.groupby('dow', observed=True):
        n = len(g)
        if n < 100:
            continue
        k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        rows.append({'dow': str(dow), 'n_bars': n,
                      'P_cat_60m': round(p, 4),
                      'cilo': round(lo, 4),
                      'cihi': round(hi, 4)})
    dow_df = pd.DataFrame(rows)
    dow_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_dow.csv'), index=False)
    print(dow_df.to_string(index=False))

    # Chi-squared on DOW
    cont = df.pivot_table(index='dow', columns='cat_in_next_60m',
                           values='ts', aggfunc='count', observed=True).fillna(0)
    if cont.shape[1] >= 2:
        chi2, p_val, dof, _ = chi2_contingency(cont.values)
        print(f'\nChi-squared DOW dependence:  chi2={chi2:.2f}  dof={dof}  p={p_val:.2e}')

    # ===== Joint: DOW x TOD =====
    print(f'\n=== JOINT P(cat in next 60m | DOW, TOD) — top 20 highest cells (n>=200) ===')
    joint = []
    for (dow, hr), g in df.groupby(['dow', 'tod_hour'], observed=True):
        n = len(g)
        if n < 200:
            continue
        k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        joint.append({'dow': str(dow), 'tod_hour': int(hr), 'n': n,
                       'P_cat_60m': round(p, 4),
                       'cilo': round(lo, 4),
                       'cihi': round(hi, 4)})
    joint_df = pd.DataFrame(joint)
    joint_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_dow_x_tod.csv'),
                     index=False)
    top = joint_df.sort_values('P_cat_60m', ascending=False).head(20)
    print(top.to_string(index=False))

    print(f'\n=== BOTTOM 10 SAFEST DOW x TOD cells ===')
    print(joint_df.sort_values('P_cat_60m').head(10).to_string(index=False))

    # ===== Joint: DOW x sigma_q =====
    print(f'\n=== JOINT P(cat in next 60m | DOW, sigma_q) ===')
    joint_s = []
    for (dow, sq), g in df.groupby(['dow', 'sigma_rank_q'], observed=True):
        if sq < 0:
            continue
        n = len(g)
        if n < 200:
            continue
        k = int(g['cat_in_next_60m'].sum())
        p, lo, hi = beta_p(k, n)
        joint_s.append({'dow': str(dow), 'sigma_rank_q': int(sq), 'n': n,
                          'P_cat_60m': round(p, 4),
                          'cilo': round(lo, 4),
                          'cihi': round(hi, 4)})
    joint_s_df = pd.DataFrame(joint_s)
    joint_s_df.to_csv(os.path.join(args.out_dir, 'p_cat_by_dow_x_sigma.csv'),
                       index=False)
    pivot_s = joint_s_df.pivot(index='dow', columns='sigma_rank_q',
                                 values='P_cat_60m')
    pivot_s = pivot_s.reindex(DOW_ORDER)
    print(pivot_s.round(4).to_string())

    # ===== CHARTS =====
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    # 1. DOW marginal
    ax = axes[0][0]
    dow_ord = [d for d in DOW_ORDER if d in dow_df['dow'].values]
    sub = dow_df.set_index('dow').reindex(dow_ord)
    bars = ax.bar(dow_ord, sub['P_cat_60m'].values * 100,
                    color='#1976D2', alpha=0.85)
    base = df['cat_in_next_60m'].mean() * 100
    ax.axhline(base, color='black', ls='--', lw=0.8, label=f'baseline {base:.2f}%')
    ax.set_ylabel('P(cat in next 60m) %')
    ax.set_title('In-flight cat risk by day-of-week')
    ax.legend(); ax.grid(True, alpha=0.3)
    for bar, p in zip(bars, sub['P_cat_60m'].values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                  f'{100*p:.2f}%', ha='center', fontsize=9)

    # 2. DOW × TOD heatmap
    ax = axes[0][1]
    pivot_t = joint_df.pivot(index='dow', columns='tod_hour',
                                values='P_cat_60m').reindex(DOW_ORDER)
    if not pivot_t.empty:
        im = ax.imshow(pivot_t.values * 100, cmap='RdYlGn_r', aspect='auto',
                        vmin=0, vmax=30)
        ax.set_xticks(range(len(pivot_t.columns)))
        ax.set_xticklabels(pivot_t.columns, fontsize=8)
        ax.set_yticks(range(len(pivot_t.index)))
        ax.set_yticklabels(pivot_t.index)
        ax.set_xlabel('UTC hour'); ax.set_ylabel('DOW')
        ax.set_title('P(cat in next 60m): DOW x TOD heatmap')
        plt.colorbar(im, ax=ax, label='%')
        for i in range(pivot_t.shape[0]):
            for j in range(pivot_t.shape[1]):
                v = pivot_t.values[i, j]
                if not np.isnan(v):
                    color = 'white' if v > 0.10 else 'black'
                    ax.text(j, i, f'{100*v:.0f}', ha='center', va='center',
                              fontsize=6, color=color)

    # 3. DOW × sigma heatmap
    ax = axes[1][0]
    if not pivot_s.empty:
        im = ax.imshow(pivot_s.values * 100, cmap='RdYlGn_r', aspect='auto')
        ax.set_xticks(range(len(pivot_s.columns)))
        ax.set_xticklabels([f'sig_q{c}' for c in pivot_s.columns])
        ax.set_yticks(range(len(pivot_s.index)))
        ax.set_yticklabels(pivot_s.index)
        ax.set_title('P(cat in next 60m): DOW x sigma_rank heatmap')
        plt.colorbar(im, ax=ax, label='%')
        for i in range(pivot_s.shape[0]):
            for j in range(pivot_s.shape[1]):
                v = pivot_s.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{100*v:.2f}', ha='center', va='center',
                              fontsize=8)

    # 4. DOW marginal sorted by risk
    ax = axes[1][1]
    sorted_dow = dow_df.sort_values('P_cat_60m', ascending=False)
    colors = ['#C62828', '#E53935', '#FB8C00', '#FFB300', '#43A047', '#1976D2', '#5E35B1']
    cols = colors[:len(sorted_dow)]
    bars = ax.bar(sorted_dow['dow'], sorted_dow['P_cat_60m'].values * 100,
                    color=cols, alpha=0.85)
    ax.errorbar(sorted_dow['dow'], sorted_dow['P_cat_60m'].values * 100,
                  yerr=[(sorted_dow['P_cat_60m']-sorted_dow['cilo']).values * 100,
                         (sorted_dow['cihi']-sorted_dow['P_cat_60m']).values * 100],
                  fmt='none', ecolor='black', capsize=4)
    ax.set_ylabel('P(cat in next 60m) %')
    ax.set_title('DOW risk ranking with 95% CI')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(args.out_dir, 'p_cat_dow_analysis.png')
    plt.savefig(out_png, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'\nChart -> {out_png}')


if __name__ == '__main__':
    main()
