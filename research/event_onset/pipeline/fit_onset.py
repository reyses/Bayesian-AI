"""Fit the EVENT-ONSET probe: is a named event predictable H seconds before
its causal confirmation?

Two models per (event, horizon): logistic (linear baseline) and gradient
boosting (nonlinear). GroupKFold(5) by DAY — no day straddles folds, so a
model cannot memorise a session. Standardisation is fit on train folds only.

Verdict rule, pre-registered before looking at any number
(the program's discrimination wall is ~0.57 AUC):
    AUC_lo (mean - 1.96*se over folds) > 0.60  -> CLEARS  (train the student)
    AUC_lo > 0.57                              -> MARGINAL (probe deeper)
    else                                       -> WALL     (do not train)

Writes research/event_onset/reports/onset_probe.md
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(REPO, 'research', 'event_onset')
REPORT = os.path.join(DATA_DIR, 'reports', 'onset_probe.md')
CLEAR_AT, MARGINAL_AT = 0.60, 0.57
NFOLD = 5


def cv_auc(df, model_fn):
    feats = [c for c in df.columns if c not in ('y', 'day', 'ts')]
    X = df[feats].to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df['y'].to_numpy()
    g = df['day'].to_numpy()
    aucs = []
    for tr, te in GroupKFold(NFOLD).split(X, y, g):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))
    a = np.array(aucs)
    se = a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0, a.mean() - 1.96 * se, a


def verdict(lo):
    return ('CLEARS' if lo > CLEAR_AT
            else 'MARGINAL' if lo > MARGINAL_AT else 'WALL')


if __name__ == '__main__':
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    rows, lines = [], []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, 'onset_*.parquet'))):
        m = re.match(r'onset_(.+)_(\d+)s\.parquet', os.path.basename(path))
        event, hz = m.group(1), int(m.group(2))
        df = pd.read_parquet(path)
        if len(df) < 200 or df['day'].nunique() < NFOLD:
            print(f'{event} H={hz}s SKIPPED (n={len(df)})')
            continue
        lin = cv_auc(df, lambda: LogisticRegression(C=1.0, max_iter=1000))
        gbm = cv_auc(df, lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=4,
            random_state=20260804))
        best_lo = max(lin[2], gbm[2])
        rows.append(dict(event=event, horizon_s=hz, n=len(df),
                         days=int(df['day'].nunique()),
                         auc_lin=round(lin[0], 4), auc_lin_lo=round(lin[2], 4),
                         auc_gbm=round(gbm[0], 4), auc_gbm_lo=round(gbm[2], 4),
                         verdict=verdict(best_lo)))
        print(f'{event:22s} H={hz:2d}s  n={len(df):7d}  '
              f'lin {lin[0]:.4f}+-{lin[1]:.4f}  gbm {gbm[0]:.4f}+-{gbm[1]:.4f}'
              f'  -> {verdict(best_lo)}')
    res = pd.DataFrame(rows).sort_values(['event', 'horizon_s'])
    lines += ['# EVENT-ONSET PROBE — can a named event be seen coming?', '',
              'Pre-registered verdict rule (set before any number was read; '
              f'the program wall is ~0.57): AUC_lo > {CLEAR_AT} CLEARS, '
              f'> {MARGINAL_AT} MARGINAL, else WALL.',
              'AUC_lo = fold mean - 1.96*SE. GroupKFold(5) by day, '
              'standardisation fit on train folds only, balanced 1:1 design '
              '(negatives >=5min from any same-type event). Live sim day '
              '2024_09_16 excluded.', '',
              res.to_string(index=False), '']
    if len(res):
        best = res.loc[res[['auc_lin_lo', 'auc_gbm_lo']].max(axis=1).idxmax()]
        lines += ['## Headline', '',
                  f"Best cell: **{best['event']} at H={best['horizon_s']}s** — "
                  f"lin {best['auc_lin']:.4f}, gbm {best['auc_gbm']:.4f} "
                  f"(n={best['n']}, {best['days']} days) -> "
                  f"**{best['verdict']}**", '',
                  'Counts: ' + str(dict(res['verdict'].value_counts())), '']
    open(REPORT, 'w').write('\n'.join(lines) + '\n')
    res.to_json(os.path.join(DATA_DIR, 'onset_probe.json'),
                orient='records', indent=1)
    print('\nwrote', REPORT)
