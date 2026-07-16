"""
COMBINER PREVIEW — pool all dossier signal streams into ONE calibrated P(right).

Stage 2 of the funnel (Moises 2026-07-15): "we mix all the signals to boost as much as
possible" -> the completed signal is later handed to the Mamba RL engine for trade
management. This is the minimal mixer: pooled logistic over every stream's fires with
per-stream one-hot identity + the shared causal features, so the model learns both
each stream's base reliability AND how the shared context (pivot age, leg agreement,
time of day) modulates it.

Also reports per-timestamp DENSITY fusion: for each fire, how many OTHER streams fired
within +-3 min in the SAME direction (consensus), as an extra feature — the first step
toward the full state-vector fusion the Mamba will consume.

Reads:  reports/signal_rows_<det>.parquet (from dossier_signal_pipeline.py)
Writes: reports/combiner_preview.md
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dossier_signal_pipeline import day_block_ci  # vectorized day-block bootstrap
REP = os.path.abspath(os.path.join(HERE, '..', 'reports'))
CONSENSUS_S = 180   # +-3 min window for cross-stream direction consensus
BASE = ['pivot_age_min', 'sig_with_leg', 'tod', 'inter']


def load_pool():
    frames = []
    for f in sorted(glob.glob(os.path.join(REP, 'signal_rows_*.parquet'))):
        det = os.path.basename(f)[12:-8]
        df = pd.read_parquet(f)
        df['det'] = det
        frames.append(df)
    P = pd.concat(frames, ignore_index=True).sort_values('ts').reset_index(drop=True)
    # consensus: same-direction fires from OTHER streams within +-3 min.
    # Vectorized: windowed count via prefix sums = same-direction total minus the
    # row's own (det, direction) group — self cancels, det!=self enforced exactly.
    ts = P['ts'].values.astype(np.int64)
    lng = P['is_long'].values.astype(bool)
    lo = np.searchsorted(ts, ts - CONSENSUS_S, 'left')
    hi = np.searchsorted(ts, ts + CONSENSUS_S, 'right')
    def wcount(flags):
        c = np.concatenate([[0], np.cumsum(flags)])
        return c[hi] - c[lo]
    same_dir = np.where(lng, wcount(lng), wcount(~lng))
    own = np.zeros(len(P), dtype=np.int64)
    for (d, is_l), g in P.groupby(['det', 'is_long'], sort=False):
        flags = np.zeros(len(P), dtype=np.int64)
        flags[g.index.values] = 1
        own[g.index.values] = wcount(flags)[g.index.values]
    P['consensus'] = (same_dir - own).astype(np.int16)
    return P


def fit_report(P):
    P = P.dropna(subset=['y']).copy()
    P['year'] = P['day'].str[:4]
    dets = sorted(P['det'].unique())
    for d in dets: P[f'is_{d}'] = (P['det'] == d).astype(int)
    cols = BASE + ['consensus'] + [f'is_{d}' for d in dets]
    trm, tem = P['year'] == '2024', P['year'] != '2024'
    Xtr, ytr = P.loc[trm, cols].values.astype(float), P.loc[trm, 'y'].astype(int).values
    Xte, yte = P.loc[tem, cols].values.astype(float), P.loc[tem, 'y'].astype(int).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    pte = clf.predict_proba((Xte - mu) / sd)[:, 1]
    auc = roc_auc_score(yte, pte)
    lines = ['# Combiner preview — pooled P(right) across all dossier streams',
             f'- pooled N = {len(P)} fires ({trm.sum()} train 2024 / {tem.sum()} test 2025+26) '
             f'across {len(dets)} streams',
             f'- pooled OOS AUC **{auc:.3f}** (test base agreement {yte.mean():.3f})',
             f'- coefs (standardized): ' + ', '.join(
                 f'{c}={v:+.3f}' for c, v in zip(cols, clf.coef_[0])), '']
    # decile calibration with day-block CIs on the extremes
    q = pd.qcut(pte, 10, labels=False, duplicates='drop')
    days_te = P.loc[tem, 'day'].values
    lines.append('| P-decile | N | mean P | observed agreement | day-block 95% CI |')
    lines.append('|---|---|---|---|---|')
    for b in range(10):
        m = q == b
        if m.sum() < 10: continue
        lo, hi = day_block_ci(yte[m], days_te[m])
        lines.append(f'| {b} | {int(m.sum())} | {pte[m].mean():.2f} | {yte[m].mean():.2f} '
                     f'| [{lo:.2f},{hi:.2f}] |')
    # consensus effect, raw
    lines.append('\n## Consensus effect (test set, raw)')
    lines.append('| same-direction co-fires (+-3min) | N | agreement |')
    lines.append('|---|---|---|')
    cte = P.loc[tem, 'consensus'].values
    for c0, c1, lab in [(0, 0, '0'), (1, 2, '1-2'), (3, 5, '3-5'), (6, 999, '6+')]:
        m = (cte >= c0) & (cte <= c1)
        if m.sum() >= 10:
            lines.append(f'| {lab} | {int(m.sum())} | {yte[m].mean():.2f} |')
    out = os.path.join(REP, 'combiner_preview.md')
    with open(out, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines[:6]))
    print(f'\nwrote {out}')


if __name__ == '__main__':
    fit_report(load_pool())
