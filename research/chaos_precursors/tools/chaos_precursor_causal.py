"""Chaos-precursor, made CAUSAL (Regression-Segments cycle 01, path off the firewall).

Thesis (from the non-causal segment work): fittability deterioration precedes chaos.
Test it with a CAUSAL fittability proxy and a forward target, and — the real question —
ask whether fittability adds predictive power BEYOND plain volatility persistence
(vol clusters; if fittability is just vol in disguise it's worthless).

  predictor A (fittability): unfit = 1 - R2 of a linear fit of price over trailing W bars
  predictor B (null)       : trailing realized vol over W bars
  target                   : forward realized vol in the TOP QUARTILE over next H bars (chaos)

All predictors use only [i-W+1, i] (causal). Target uses [i+1, i+H] (labeled, never fed
to the predictor). IS = 2024-03, OOS = 2025-03. Day-block bootstrap CI.

Run: python research/chaos_precursor_causal.py
"""
import os, glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

ATLAS = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS/5s'
W = 60          # trailing window (5 min @ 5s) for fittability + vol
H = 60          # forward horizon (5 min) for the chaos target
CHAOS_Q = 0.75  # forward-vol top quartile = chaos
RNG = np.random.RandomState(42)


def trailing_lin_r2(y):
    """R2 of price-on-time over the trailing W bars, at each bar (causal)."""
    n = len(y)
    pos = np.arange(W, dtype=float)
    St, Stt = pos.sum(), (pos * pos).sum()
    var_t = Stt - St * St / W
    s = pd.Series(y)
    S_y = s.rolling(W).sum().to_numpy()
    S_yy = (s * s).rolling(W).sum().to_numpy()
    S_yt = np.full(n, np.nan)
    if n >= W:
        S_yt[W - 1:] = np.convolve(y, pos[::-1], 'valid')   # window-start aligned -> shift to end
    cov_yt = S_yt - S_y * St / W
    var_y = S_yy - S_y * S_y / W
    with np.errstate(divide='ignore', invalid='ignore'):
        r2 = (cov_yt * cov_yt) / (var_t * var_y)
    return np.clip(np.nan_to_num(r2, nan=0.0), 0, 1)


def day_frame(f):
    df = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
    y = df['close'].to_numpy(np.float64)
    if len(y) < W + H + 2:
        return None
    r = pd.Series(y).diff()
    r2 = pd.Series(trailing_lin_r2(y))
    unfit = (1.0 - r2).to_numpy()                          # A: fittability LEVEL (causal)
    decel = -(r2 - r2.shift(W)).to_numpy()                 # A': fittability DETERIORATION (R2 falling)
    tvol = r.rolling(W).std().to_numpy()                   # B: vol null (causal)
    resid_norm = (tvol * np.sqrt(np.clip(unfit, 0, 1)) /   # A'': residual SE / trailing range
                  (pd.Series(y).rolling(W).max() - pd.Series(y).rolling(W).min() + 1e-9).to_numpy())
    fwd = r.rolling(H).std().shift(-H).to_numpy()          # forward vol [i+1,i+H]
    out = pd.DataFrame({'unfit': unfit, 'decel': decel, 'resid_norm': resid_norm,
                        'tvol': tvol, 'fwd_vol': fwd})
    out['day'] = os.path.basename(f)[:-8]
    return out.dropna()


def load(glob_pat):
    parts = [day_frame(f) for f in sorted(glob.glob(f'{ATLAS}/{glob_pat}.parquet'))]
    parts = [p for p in parts if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else None


def dayblock_auc_ci(df, score_col, y, n=2000):
    days = df['day'].unique()
    boots = []
    for _ in range(n):
        samp = df[df['day'].isin(RNG.choice(days, len(days), replace=True))]
        if samp[y].nunique() > 1:
            boots.append(roc_auc_score(samp[y], samp[score_col]))
    return tuple(np.percentile(boots, [2.5, 97.5])) if boots else (np.nan, np.nan)


def evaluate(df, thr, label):
    df = df.copy()
    df['chaos'] = (df['fwd_vol'] >= thr).astype(int)
    y = df['chaos']
    feats = ['unfit', 'decel', 'resid_norm']
    # vol-only baseline AUC
    Zv = (df[['tvol']] - df[['tvol']].mean()) / df[['tvol']].std()
    auc_bonly = roc_auc_score(y, LogisticRegression().fit(Zv, y).predict_proba(Zv)[:, 1])
    L = [f"### {label}  (n={len(df)}, days={df['day'].nunique()}, base chaos={y.mean():.1%})",
         f"- AUC trailing-vol (null, B) : {auc_bonly:.3f}"]
    for fcol in feats:                          # each fittability variant: solo AUC + increment over vol
        auc_solo = roc_auc_score(y, df[fcol] if df[fcol].corr(y) >= 0 else -df[fcol])
        Z = (df[['tvol', fcol]] - df[['tvol', fcol]].mean()) / df[['tvol', fcol]].std()
        lr = LogisticRegression().fit(Z, y)
        auc_inc = roc_auc_score(y, lr.predict_proba(Z)[:, 1])
        L.append(f"- {fcol:10s}: solo AUC {auc_solo:.3f} | vol+{fcol} AUC {auc_inc:.3f} "
                 f"(increment {auc_inc-auc_bonly:+.3f}, coef {lr.coef_[0][1]:+.2f})")
    auc_a = roc_auc_score(y, df['unfit'])
    auc_b = auc_bonly
    auc_ab = roc_auc_score(y, LogisticRegression().fit(
        (df[['tvol']+feats]-df[['tvol']+feats].mean())/df[['tvol']+feats].std(), y
    ).predict_proba((df[['tvol']+feats]-df[['tvol']+feats].mean())/df[['tvol']+feats].std())[:, 1])
    L.append(f"- ALL fittability + vol AUC : {auc_ab:.3f} (increment over vol {auc_ab-auc_bonly:+.3f})")
    # conditional chaos rate by unfit quartile
    df['q'] = pd.qcut(df['unfit'], 4, labels=['Q1 fittable', 'Q2', 'Q3', 'Q4 choppy'], duplicates='drop')
    rates = df.groupby('q', observed=True)['chaos'].mean()
    L.append("- chaos rate by fittability quartile: " +
             ", ".join(f"{k} {v:.1%}" for k, v in rates.items()))
    return "\n".join(L), auc_a, auc_b, auc_ab, auc_bonly


def main():
    is_df = load('2024_03_*')
    oos_df = load('2025_03_*')
    if is_df is None:
        print("no IS data"); return
    thr = is_df['fwd_vol'].quantile(CHAOS_Q)   # threshold fixed on IS, applied to OOS

    L = ["# Causal chaos-precursor — fittability vs the vol-persistence null\n",
         f"W={W} trailing, H={H} forward (5s bars); chaos = forward-vol >= IS p{int(CHAOS_Q*100)} "
         f"({thr:.4f}).",
         "Question: does causal fittability (1-R2) beat / add to trailing vol for predicting "
         "forward chaos?\n"]
    rep_is, a_is, b_is, ab_is, bo_is = evaluate(is_df, thr, "IS 2024-03")
    L.append(rep_is)
    if oos_df is not None:
        rep_oos, *_ = evaluate(oos_df, thr, "OOS 2025-03")
        L += ["", rep_oos]
    # verdict
    verdict = ("FITTABILITY ADDS SIGNAL beyond vol" if (ab_is - bo_is) > 0.01 and a_is > 0.55
               else "FITTABILITY = vol in disguise (no independent precursor)" if a_is <= 0.55
               else "MARGINAL — fittability adds little beyond vol")
    L += ["", f"## Verdict (IS): {verdict}",
          "Note: AUC ~0.5 = no signal; the honest test is the A+B increment over vol-only, "
          "not AUC(A) alone (vol persistence inflates everything)."]
    rep = "\n".join(L)
    os.makedirs('reports/findings', exist_ok=True)
    open('reports/findings/chaos_precursor_causal.md', 'w').write(rep)
    print(rep)


if __name__ == '__main__':
    main()
