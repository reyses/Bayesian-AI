"""Pre-committed: does vr (the de-facto V1 NMP gate) separate NMP winners from losers? (2026-06-13)

λ-completion named TWO missing pieces: λ (hardcoded 0) and vr (std10/std60 < 1, the gate V1
actually fired on, dropped from V2). λ tested -> DEAD (well-powered). vr never tested; vr_exact
is materialized in L4. This is the other shot.

Join nmp_fade_raw trades -> L4_1m_vr_exact at entry. Original NMP GO condition = vr < 1
(compression: short-term vol < long-term vol). THESIS: compression entries (vr<1) out-earn
expansion entries (vr>=1). delta = mean pnl(vr<1) - mean pnl(vr>=1), day-block 95% CI.

PRE-COMMITTED (before seeing OOS):
  - vr = L4_1m_vr_exact at the entry bar; cohort = vr < 1.0 vs vr >= 1.0.
  - PRIMARY GATE: OOS delta CI lower bound > 0 -> vr separates (the missing gate). Else -> vr
    also dead -> NMP entry is a coin-flip in V2; pivot off "complete the equation".
Secondary context (NOT the gate): vr quintile mean-pnl curve; vr x macro-velocity regime;
combined V1 gate (vr<1 AND lambda<0).

Same join harness as test_lambda_macrovel_oneshot.py. nmp_fade_raw uses the OLD z-band (first
read); if vr CONFIRMS, rebuild on nmp_v2.
"""
import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '.')
from core_v2.features import load_features

RNG = np.random.RandomState(42); NB = 4000
VR, VEL, LAM = 'L4_1m_vr_exact', 'L1_4h_price_velocity_1b', 'L4_1m_lambda_hat_21'
NEED = [VR, VEL, LAM]

def join_features(trades, feat_root):
    out = {c: np.full(len(trades), np.nan) for c in NEED}
    for day, g in tqdm(trades.groupby('day'), desc=f'join {os.path.basename(feat_root)}', ncols=70):
        try:
            f = load_features([day], tfs=['4h', '1h', '1m'], layers=['L1', 'L2', 'L4'],
                              root=feat_root, require_all=False)
        except Exception:
            continue
        if f.empty or 'timestamp' not in f.columns:
            continue
        f = f.sort_values('timestamp').reset_index(drop=True)
        fts = f['timestamp'].values
        have = [c for c in NEED if c in f.columns]
        for ridx, ets in zip(g.index, g['entry_ts'].values):
            pos = np.searchsorted(fts, ets, side='right') - 1
            if pos < 0:
                continue
            for c in have:
                out[c][ridx] = f[c].values[pos]
    for c in NEED:
        trades[c] = out[c]
    return trades

def dayblock_delta(df, cohort):
    """mean pnl(cohort) - mean pnl(~cohort), day-block bootstrap 95% CI."""
    df = df.assign(_c=cohort.values if hasattr(cohort, 'values') else cohort)
    per_day = []
    for d, g in df.groupby('day'):
        a = g[g['_c']]['pnl_usd'].values
        b = g[~g['_c']]['pnl_usd'].values
        per_day.append((a.sum(), len(a), b.sum(), len(b)))
    arr = np.array(per_day, float)
    def dl(r):
        sa, na, sb, nb = r[:,0].sum(), r[:,1].sum(), r[:,2].sum(), r[:,3].sum()
        return (sa/na - sb/nb) if (na and nb) else np.nan
    pt = dl(arr); nd = len(arr)
    bs = [dl(arr[RNG.randint(0,nd,nd)]) for _ in range(NB)]
    bs = [x for x in bs if not np.isnan(x)]
    ci = np.percentile(bs, [2.5,97.5]) if bs else [np.nan, np.nan]
    return pt, ci, int(df['_c'].sum()), int((~df['_c']).sum())

def load_trades(path, feat_root, tag):
    df = pd.read_csv(path)
    df = join_features(df, feat_root)
    before = len(df)
    df = df[df[VR].notna()].reset_index(drop=True)
    print(f"[{tag}] {len(df):,}/{before:,} joined, {len(df['day'].unique())} days, "
          f"base ${df['pnl_usd'].mean():.2f}/trade | vr<1: {100*(df[VR]<1).mean():.0f}% of trades")
    return df

def gateline(df, cohort, label, gate=False):
    pt, ci, na, nb = dayblock_delta(df, cohort)
    ma = df[cohort]['pnl_usd'].mean(); mb = df[~cohort]['pnl_usd'].mean()
    sig = 'SIG +' if ci[0]>0 else ('SIG -' if ci[1]<0 else 'n.s.')
    star = '  <<< GATE' if gate else ''
    print(f"  {label:34s} A ${ma:7.2f} (n={na:5d}) vs B ${mb:7.2f} (n={nb:5d}) | "
          f"delta ${pt:7.2f} CI[{ci[0]:7.2f},{ci[1]:7.2f}] {sig}{star}")

def report(df, tag, gate=False):
    print(f"\n--- {tag} ---  (A = condition true)")
    gateline(df, df[VR] < 1.0, "vr<1 (compression) vs vr>=1", gate=gate)
    # secondary context
    gateline(df, (df[VR] < 1.0) & (df[LAM] < 0), "vr<1 AND lam<0 (full V1 GO) vs rest")
    flat = df[VEL].abs() < df[VEL].abs().median()
    gateline(df[flat], df[flat][VR] < 1.0, "vr<1 vs >=1 | FLAT-vel regime")
    gateline(df[~flat], df[~flat][VR] < 1.0, "vr<1 vs >=1 | TREND-vel regime")
    df = df.copy(); df['vq'] = pd.qcut(df[VR], 5, labels=['Q1_low','Q2','Q3','Q4','Q5_high'], duplicates='drop')
    means = df.groupby('vq', observed=True)['pnl_usd'].agg(['mean','count'])
    print("    (ctx) mean pnl by vr quintile (thesis: low vr = high pnl):")
    for idx, r in means.iterrows():
        print(f"        {idx:9s} ${r['mean']:7.2f}  n={int(r['count']):,}")

def main():
    IS = load_trades('reports/findings/strategy_runs/nmp_fade_raw_is_atr4.csv',
                     os.path.abspath('DATA/ATLAS/FEATURES_5s_v2'), 'IS')
    report(IS, 'IS (in-sample reference)')
    OOS = load_trades('reports/findings/strategy_runs/nmp_fade_raw_oos_atr4.csv',
                      os.path.abspath('DATA/ATLAS_NT8/FEATURES_5s_v2'), 'OOS')
    report(OOS, 'OOS (SEALED — the gate)', gate=True)
    print("\nGATE: OOS vr<1-vs->=1 delta CI lower > 0 -> vr is the missing gate; else vr DEAD too -> pivot.")

if __name__ == '__main__':
    main()
