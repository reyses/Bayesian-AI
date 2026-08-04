"""Fit the GBM baseline on the MATCHED-negative probe datasets.

This script existed only as an inline heredoc when it produced
`matched_probe.json`; the audit flagged it missing from the repo (numbers
were reproducible, the code was not). Committed so the baseline is
regenerable.

NOTE — the numbers this produces are the OLD, INFLATED baseline. They carry
two defects the audit proved (`reports/audit_pipeline.md`):
  1. ATLAS `timestamp` marks the bar OPEN, so a feature row stamped t holds
     tape through t+4.99 (+5s of future).
  2. GroupKFold over all days puts the SEALED TEST days in 4/5 of the
     training folds.
Pass --causal --temporal to get the honest bar that ONSET_MAMBA_SPEC.md §7
is now registered against (0.6435 / 0.7580 / 0.8201).

  python research/event_onset/pipeline/fit_matched.py --causal --temporal
"""
import argparse
import glob
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
DATA = os.path.join(REPO, 'research', 'event_onset')
SEED = 20260804


def day_split(day):
    import datetime as dt
    y, m, d = day.split('_')[:3]
    x = dt.date(int(y), int(m), int(d))
    if x < dt.date(2025, 1, 1):
        return 'train'
    return 'val' if x <= dt.date(2025, 6, 30) else 'test'


def cv_auc(df, feats, model_fn):
    X = np.nan_to_num(df[feats].to_numpy(float), nan=0., posinf=0., neginf=0.)
    y, g = df['y'].to_numpy(), df['day'].to_numpy()
    a = []
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        m = model_fn().fit(sc.transform(X[tr]), y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:, 1]))
    return float(np.mean(a)), float(np.std(a, ddof=1))


def temporal_auc(df, feats, model_fn):
    """Fit on train days only, score val — the protocol the Mamba faces."""
    sp = df['day'].map(day_split)
    tr, va = df[sp == 'train'], df[sp == 'val']
    if len(tr) < 500 or len(va) < 500:
        return None
    X = lambda d: np.nan_to_num(d[feats].to_numpy(float), nan=0.,
                                posinf=0., neginf=0.)
    sc = StandardScaler().fit(X(tr))
    m = model_fn().fit(sc.transform(X(tr)), tr['y'].to_numpy())
    p = m.predict_proba(sc.transform(X(va)))[:, 1]
    auc = roc_auc_score(va['y'].to_numpy(), p)
    # day-clustered CI
    rng = np.random.default_rng(SEED)
    days = va['day'].to_numpy()
    uq = np.unique(days)
    bs = []
    for _ in range(400):
        pick = rng.choice(uq, size=len(uq), replace=True)
        idx = np.concatenate([np.flatnonzero(days == d) for d in pick])
        yy = va['y'].to_numpy()[idx]
        if len(np.unique(yy)) < 2:
            continue
        bs.append(roc_auc_score(yy, p[idx]))
    return auc, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--causal', action='store_true',
                    help='shift features one 5s bar back to remove the '
                         'bar-OPEN +5s lookahead')
    ap.add_argument('--temporal', action='store_true',
                    help='fit train-only / score val instead of GroupKFold')
    a = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(DATA, 'matched_*.parquet'))):
        m = re.match(r'matched_(.+)_(\d+)s\.parquet', os.path.basename(path))
        event, hz = m.group(1), int(m.group(2))
        df = pd.read_parquet(path)
        feats = [c for c in df.columns if c not in ('y', 'day', 'ts')]
        if a.causal:
            df = df.sort_values(['day', 'ts'])
            df[feats] = df.groupby('day')[feats].shift(1)
            df = df.dropna(subset=feats)
        gbm = lambda: HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, max_depth=4, random_state=SEED)
        if a.temporal:
            r = temporal_auc(df, feats, gbm)
            if r:
                rows.append(dict(event=event, H=hz, n=len(df),
                                 auc=round(r[0], 4), lo=round(r[1], 4),
                                 hi=round(r[2], 4)))
                print(f'{event:20s} H={hz:2d}s n={len(df):7d} '
                      f'AUC {r[0]:.4f} [{r[1]:.4f},{r[2]:.4f}]')
        else:
            mu, sd = cv_auc(df, feats, gbm)
            lin = cv_auc(df, feats,
                         lambda: LogisticRegression(C=1.0, max_iter=1000))
            rows.append(dict(event=event, H=hz, n=len(df), gbm=round(mu, 4),
                             gbm_sd=round(sd, 4), lin=round(lin[0], 4)))
            print(f'{event:20s} H={hz:2d}s n={len(df):7d} '
                  f'gbm {mu:.4f}+-{sd:.4f} lin {lin[0]:.4f}')
    tag = ('causal_' if a.causal else '') + ('temporal' if a.temporal else 'cv')
    pd.DataFrame(rows).to_json(os.path.join(DATA, f'matched_probe_{tag}.json'),
                               orient='records', indent=1)
    print('\nwrote', f'matched_probe_{tag}.json')
