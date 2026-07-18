"""KT1-lite: does lambda separate NMP winners from losers WITHIN strata,
even though it's FLAT globally?  (2026-06-12)

User thesis: global lambda-hat came back flat because it averages over many feature
clusters operating at different intervals (snap-back legs with lambda<0 + runaway legs with
lambda>0 cancel). Stratify ("lambda as a function of the regressions") and the signal returns.

Uses ONLY the existing Stage-0 NMP forward-pass trades (reports/findings/strategy_runs/
nmp_fade_raw_{is,oos}_atr4.csv) -- which already log entry lambda + z_se + reversion_prob +
pnl. No segment-index join needed: z_se (the regression's standardized residual) and
reversion_prob (its OU prob) ARE functions of the regression, and they're causal entry
features (firewall-safe).

Sign of the thesis: entries with lambda_hat < 0 (|z_se| already contracting = snap-back
underway) should out-earn lambda_hat > 0 (still extending = runaway). Metric = mean pnl_usd
delta (lambda<0 minus lambda>0), day-block bootstrap 95% CI. We compare GLOBAL (expect ~flat)
vs WITHIN strata of |z_se| and reversion_prob.

CAVEAT: nmp_fade_raw uses the OLD z-band [1.5,1.8], not the recalibrated Z*=1.8481/0.4752
(nmp_v2, not built). And lambda logged is the k=12 variant. This is a first KT1 read; if
stratification de-flattens lambda here, build nmp_v2 for the proper numbers.
"""
import os
import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)
NB = 4000

def dayblock_delta_ci(df):
    """mean pnl(lambda<0) - mean pnl(lambda>=0), day-block bootstrap 95% CI."""
    neg = df['lam'] < 0
    per_day = []
    for d, g in df.groupby('day'):
        gn = g[g['lam'] < 0]['pnl_usd'].values
        gp = g[g['lam'] >= 0]['pnl_usd'].values
        per_day.append((gn.sum(), len(gn), gp.sum(), len(gp)))
    pd_arr = np.array(per_day, float)
    def delta(rows):
        sn, nn, sp, npp = rows[:, 0].sum(), rows[:, 1].sum(), rows[:, 2].sum(), rows[:, 3].sum()
        if nn == 0 or npp == 0:
            return np.nan
        return sn / nn - sp / npp
    pt = delta(pd_arr)
    nd = len(pd_arr)
    boots = [delta(pd_arr[RNG.randint(0, nd, nd)]) for _ in range(NB)]
    boots = [b for b in boots if not np.isnan(b)]
    ci = np.percentile(boots, [2.5, 97.5]) if boots else [np.nan, np.nan]
    return pt, ci, int(neg.sum()), int((~neg).sum())

def row(label, df):
    if len(df) < 50 or (df['lam'] < 0).sum() < 10 or (df['lam'] >= 0).sum() < 10:
        print(f"  {label:32s}  n={len(df):5d}  (too few in a cohort, skip)")
        return
    pt, ci, nn, npp = dayblock_delta_ci(df)
    mneg = df[df['lam'] < 0]['pnl_usd'].mean()
    mpos = df[df['lam'] >= 0]['pnl_usd'].mean()
    sig = 'SIG +' if ci[0] > 0 else ('SIG -' if ci[1] < 0 else 'n.s.')
    print(f"  {label:32s}  lam<0 ${mneg:7.2f} (n={nn:5d}) | lam>=0 ${mpos:7.2f} (n={npp:5d}) | "
          f"delta ${pt:7.2f} CI[{ci[0]:7.2f},{ci[1]:7.2f}] {sig}")

def analyze(path, tag):
    if not os.path.exists(path):
        print(f"[{tag}] missing {path}"); return
    df = pd.read_csv(path)
    df = df.rename(columns={'extra_entry_lambda_hat': 'lam', 'extra_z_se': 'z', 'extra_reversion_prob': 'rp'})
    df = df[df['lam'].notna()].copy()
    # drop exact-zero lambda (missing/default at warmup bars)
    n0 = (df['lam'] == 0.0).sum()
    df = df[df['lam'] != 0.0].copy()
    print(f"\n{'='*78}\n[{tag}] {len(df):,} trades with non-zero entry lambda "
          f"({n0:,} dropped lam==0)  |  base mean pnl ${df['pnl_usd'].mean():.2f}/trade, "
          f"{len(df['day'].unique())} days")
    print(f"  thesis: lambda<0 (snap-back underway) should out-earn lambda>=0 (runaway)")

    print("\n-- GLOBAL (expect ~flat: this is the washed-out average) --")
    row("ALL trades", df)

    print("\n-- stratified by |z_se| quartile (the regression's standardized residual) --")
    df['azq'] = pd.qcut(df['z'].abs(), 4, labels=['Q1_low|z|', 'Q2', 'Q3', 'Q4_high|z|'], duplicates='drop')
    for q in df['azq'].cat.categories:
        row(f"|z_se| {q}", df[df['azq'] == q])

    print("\n-- stratified by reversion_prob quartile (the regression's OU prob) --")
    df['rpq'] = pd.qcut(df['rp'], 4, labels=['Q1_low', 'Q2', 'Q3', 'Q4_high'], duplicates='drop')
    for q in df['rpq'].cat.categories:
        row(f"reversion_prob {q}", df[df['rpq'] == q])

    print("\n-- stratified by leg direction --")
    for d in ('LONG', 'SHORT'):
        row(d, df[df['leg_dir'] == d])

    # lambda quintile mean-pnl curve (monotone in lambda? thesis: rises as lambda falls)
    print("\n-- mean pnl by lambda_hat quintile (monotone? thesis: low lambda = high pnl) --")
    df['lq'] = pd.qcut(df['lam'], 5, labels=['L1_most_neg', 'L2', 'L3', 'L4', 'L5_most_pos'], duplicates='drop')
    g = df.groupby('lq')['pnl_usd'].agg(['mean', 'count'])
    for idx, r in g.iterrows():
        print(f"  {idx:14s}  mean pnl ${r['mean']:7.2f}  n={int(r['count']):,}")

if __name__ == '__main__':
    analyze('reports/findings/strategy_runs/nmp_fade_raw_is_atr4.csv', 'IS (Databento)')
    analyze('reports/findings/strategy_runs/nmp_fade_raw_oos_atr4.csv', 'OOS (NT8 — honest)')
