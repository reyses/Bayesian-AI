"""ONE pre-committed shot: does lambda separate NMP winners from losers within the
FLAT macro-velocity regime?  (2026-06-13)

Prior KT1-lite (analyze_nmp_lambda_stratified.py) stratified by LOCAL z_se/reversion_prob and
got a well-powered NEGATIVE. This tests the stratifier the DOE actually ranked + the one the
user meant ("lambda as a function of the regressions"): the MACRO mean-velocity regime.

Thesis: in a FLAT-mean regime, price stretched far out snaps back -> NMP's snap-back bet pays
when lambda<0 (|z| already contracting). In a TREND regime, the snap-back bets into a runaway
and burns. So lambda should separate winners from losers WITHIN the flat regime (where the
global average was diluted by the trend cluster).

PRE-COMMITTED (chosen before seeing OOS, written here so it can't drift):
  - regime stratifier = |L1_4h_price_velocity_1b|; FLAT = below the IS median (threshold set on
    IS, applied unchanged to OOS).
  - lambda = L4_1m_lambda_hat_21 (k=21 PRIMARY). k=30, lambda_t_21, 1h-velocity = secondary
    context only, NOT the gate.
  - cohort: lambda<0 (snap-back underway) vs lambda>=0 (runaway).
  - metric: mean pnl_usd delta (lambda<0 minus lambda>=0) within FLAT regime, day-block 95% CI.
  - GATE: OOS FLAT-regime delta CI lower bound > 0 -> CONFIRMED. Else -> trade-level
    lambda-separation is DEAD; pivot.
Secondary (descriptive, not the gate): the full 2x2 (flat/trend x lambda sign), k=30, lambda_t.

Joins existing Stage-0 nmp_fade_raw trades to V2 features at the entry bar (timestamp match).
nmp_fade_raw uses the OLD z-band (not nmp_v2) -- a first read; if CONFIRMED, rebuild on nmp_v2.
"""
import os, sys, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, '.')
from core_v2.features import load_features

RNG = np.random.RandomState(42); NB = 4000
VEL = 'L1_4h_price_velocity_1b'
LAM21, LAM30, LAMT21 = 'L4_1m_lambda_hat_21', 'L4_1m_lambda_hat_30', 'L4_1m_lambda_t_21'
NEED = [VEL, LAM21, LAM30, LAMT21]

def join_features(trades, feat_root):
    out = {c: np.full(len(trades), np.nan) for c in NEED}
    for day, g in tqdm(trades.groupby('day'), desc=f'join {os.path.basename(feat_root)}', ncols=80):
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
            pos = np.searchsorted(fts, ets, side='right') - 1   # last closed bar at/<= entry
            if pos < 0:
                continue
            for c in have:
                out[c][ridx] = f[c].values[pos]
    for c in NEED:
        trades[c] = out[c]
    return trades

def dayblock_delta(df, lamcol):
    """mean pnl(lambda<0) - mean pnl(lambda>=0), day-block bootstrap 95% CI."""
    per_day = []
    for d, g in df.groupby('day'):
        gn = g[g[lamcol] < 0]['pnl_usd'].values
        gp = g[g[lamcol] >= 0]['pnl_usd'].values
        per_day.append((gn.sum(), len(gn), gp.sum(), len(gp)))
    a = np.array(per_day, float)
    def dl(r):
        sn, nn, sp, npp = r[:,0].sum(), r[:,1].sum(), r[:,2].sum(), r[:,3].sum()
        return (sn/nn - sp/npp) if (nn and npp) else np.nan
    pt = dl(a); nd = len(a)
    bs = [dl(a[RNG.randint(0,nd,nd)]) for _ in range(NB)]
    bs = [b for b in bs if not np.isnan(b)]
    ci = np.percentile(bs, [2.5,97.5]) if bs else [np.nan, np.nan]
    nn = int((df[lamcol] < 0).sum()); npp = int((df[lamcol] >= 0).sum())
    return pt, ci, nn, npp

def load_trades(path, feat_root, tag):
    df = pd.read_csv(path).rename(columns={'extra_z_se':'z'})
    df = join_features(df, feat_root)
    before = len(df)
    df = df[df[VEL].notna() & df[LAM21].notna() & (df[LAM21] != 0.0)].reset_index(drop=True)
    print(f"[{tag}] {len(df):,}/{before:,} trades joined (have vel+lambda), "
          f"{len(df['day'].unique())} days, base ${df['pnl_usd'].mean():.2f}/trade")
    return df

def main():
    IS = load_trades('reports/findings/strategy_runs/nmp_fade_raw_is_atr4.csv',
                     os.path.abspath('DATA/ATLAS/FEATURES_5s_v2'), 'IS')
    thr = IS[VEL].abs().median()      # FLAT threshold set on IS, frozen
    print(f"\n[PRE-COMMIT] FLAT regime = |{VEL}| < {thr:.4f} (IS median)\n")

    def report(df, tag, gate=False):
        df = df.copy()
        df['flat'] = df[VEL].abs() < thr
        print(f"--- {tag} ---")
        for reg, sub in (('FLAT ', df[df['flat']]), ('TREND', df[~df['flat']])):
            pt, ci, nn, npp = dayblock_delta(sub, LAM21)
            mneg = sub[sub[LAM21]<0]['pnl_usd'].mean(); mpos = sub[sub[LAM21]>=0]['pnl_usd'].mean()
            sig = 'SIG +' if ci[0]>0 else ('SIG -' if ci[1]<0 else 'n.s.')
            star = '  <<< PRE-COMMITTED GATE' if (gate and reg.strip()=='FLAT') else ''
            print(f"  {reg} regime | lam<0 ${mneg:7.2f} (n={nn:5d}) vs lam>=0 ${mpos:7.2f} "
                  f"(n={npp:5d}) | delta ${pt:7.2f} CI[{ci[0]:7.2f},{ci[1]:7.2f}] {sig}{star}")
        # secondary context: k=30, lambda_t, 2x2 means
        for lamcol, nm in ((LAM30,'k=30'), (LAMT21,'lambda_t k=21')):
            sub = df[df['flat']]
            pt, ci, _, _ = dayblock_delta(sub.rename(columns={lamcol:'_x'}).assign(**{lamcol:sub[lamcol]}), lamcol) \
                if lamcol in sub else (np.nan,[np.nan,np.nan],0,0)
            print(f"    (ctx) FLAT delta via {nm:14s}: ${pt:7.2f} CI[{ci[0]:7.2f},{ci[1]:7.2f}]")
        print(f"    (ctx) 2x2 mean pnl: "
              f"FLAT/lam<0 ${df[df['flat']&(df[LAM21]<0)]['pnl_usd'].mean():.2f} | "
              f"FLAT/lam>=0 ${df[df['flat']&(df[LAM21]>=0)]['pnl_usd'].mean():.2f} | "
              f"TREND/lam<0 ${df[~df['flat']&(df[LAM21]<0)]['pnl_usd'].mean():.2f} | "
              f"TREND/lam>=0 ${df[~df['flat']&(df[LAM21]>=0)]['pnl_usd'].mean():.2f}")
        print()

    report(IS, 'IS (in-sample reference)')
    OOS = load_trades('reports/findings/strategy_runs/nmp_fade_raw_oos_atr4.csv',
                      os.path.abspath('DATA/ATLAS_NT8/FEATURES_5s_v2'), 'OOS')
    report(OOS, 'OOS (SEALED — the gate)', gate=True)
    print("GATE: OOS FLAT-regime delta CI lower bound > 0 -> CONFIRMED; else trade-level "
          "lambda-separation DEAD -> pivot.")

if __name__ == '__main__':
    main()
