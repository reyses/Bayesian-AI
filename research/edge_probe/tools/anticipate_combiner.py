#!/usr/bin/env python3
"""ANTICIPATE-THE-COMBINER go/no-go (owner 2026-07-27, via TG): can the combiner's
FIRE + DIRECTION be predicted N bars EARLY from the 22 streams + curve regression
(z_se)? If not, qwen can't anticipate it either — this gates rebuilding the packets.

Per RTH 1m bar over full ATLAS (atlas_backtest vectors):
  features = 22 combiner streams f_* (+ n_fires, gov_dir sign) + z_se(L3_1m).
  Two sets: WITH the combiner's own P (reads how close it is) vs WITHOUT P
  (pure anticipation from the ingredients only).
  target A (fire):  will a combiner ENTRY fire within the next H bars?
  target B (dir):   on pre-fire bars, the upcoming fire's direction (long/short).
Walk-forward by day, OOS AUC. Reports reports/anticipate_combiner.md.
"""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
ZF = os.path.join(REPO, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m')
OUT = os.path.join(PROJ, 'reports', 'anticipate_combiner.md')
HORIZONS = [2, 3, 5, 8]
SEED = 42


def main():
    files = sorted(glob.glob(os.path.join(VEC, '*.parquet')))
    fcols = [c for c in pd.read_parquet(files[0]).columns if c.startswith('f_')]
    rows = []
    for f in files:
        day = os.path.basename(f)[:10]
        v = pd.read_parquet(f).sort_values('bar_ts').reset_index(drop=True)
        zp = os.path.join(ZF, f'{day}.parquet')
        if not os.path.exists(zp):
            continue
        z = pd.read_parquet(zp, columns=['timestamp', 'L3_1m_z_se_15'])
        zd = dict(zip(z['timestamp'].astype('int64'), z['L3_1m_z_se_15'].astype(float)))
        v['z_se'] = v['bar_ts'].astype('int64').map(zd).astype(float)
        ent = v['entry'].to_numpy(); gdir = v['gov_dir'].to_numpy(); n = len(v)
        # next fire index >= t+1 (O(n) reverse scan), and its direction
        nxt = np.full(n, n)             # n = "none ahead"
        nx = n
        for t in range(n - 2, -1, -1):
            if ent[t + 1] == 1:
                nx = t + 1
            nxt[t] = nx
        lead = nxt - np.arange(n)       # bars until the next fire
        for H in HORIZONS:
            fire = ((nxt < n) & (lead <= H)).astype(int)
            fdir = np.where(fire == 1, gdir[np.clip(nxt, 0, n - 1)], 0)
            v[f'fire_{H}'] = fire; v[f'fdir_{H}'] = fdir
        v['day'] = day
        rows.append(v)
    df = pd.concat(rows, ignore_index=True)
    df = df.dropna(subset=['z_se']).reset_index(drop=True)
    feat_all = fcols + ['n_fires_topk', 'gov_dir', 'z_se', 'P_topk', 'P_any']
    feat_noP = fcols + ['n_fires_topk', 'gov_dir', 'z_se']
    days = sorted(df['day'].unique())
    # coarse expanding folds (K test blocks) — go/no-go, not per-day
    K = 4
    bounds = [int(len(days) * i / (K + 1)) for i in range(1, K + 2)]

    def wf_auc(feats, ycol, mask=None):
        preds, ys = [], []
        for i in range(K):
            trd = set(days[:bounds[i]]); ted = set(days[bounds[i]:bounds[i + 1]])
            tr = df[df['day'].isin(trd)]; te = df[df['day'].isin(ted)]
            if mask is not None:
                tr = tr[mask(tr)]; te = te[mask(te)]
            if len(te) < 20 or tr[ycol].nunique() < 2 or te[ycol].nunique() < 2:
                continue
            clf = HistGradientBoostingClassifier(max_depth=4, max_iter=120,
                                                 learning_rate=0.06, random_state=SEED)
            clf.fit(tr[feats].to_numpy(), tr[ycol].to_numpy())
            preds.append(clf.predict_proba(te[feats].to_numpy())[:, 1])
            ys.append(te[ycol].to_numpy())
        if not ys:
            return float('nan'), 0
        y = np.concatenate(ys); p = np.concatenate(preds)
        return roc_auc_score(y, p), len(y)

    lines = ['# Anticipate-the-combiner probe (full ATLAS, walk-forward OOS)',
             f'{len(df):,} bars, {len(days)} days, {len(fcols)} streams + z_se. '
             'Can the combiner fire/direction be called N bars early?', '',
             '## A. FIRE anticipation — AUC(fire within next H bars)',
             '| H (bars early) | AUC with P | AUC streams+z_se ONLY (no P) | base rate |',
             '|---|---|---|---|']
    for H in HORIZONS:
        a_all, _ = wf_auc(feat_all, f'fire_{H}')
        a_noP, _ = wf_auc(feat_noP, f'fire_{H}')
        br = df[f'fire_{H}'].mean()
        lines.append(f'| {H} | {a_all:.3f} | {a_noP:.3f} | {br:.1%} |')

    lines += ['', '## B. DIRECTION anticipation — AUC(upcoming fire is LONG) on pre-fire bars',
              '| H | AUC with P | AUC no P | n pre-fire bars |', '|---|---|---|---|']
    for H in HORIZONS:
        df[f'islong_{H}'] = (df[f'fdir_{H}'] > 0).astype(int)
        msk = (lambda d, H=H: d[f'fire_{H}'] == 1)
        a_all, nA = wf_auc(feat_all, f'islong_{H}', mask=msk)
        a_noP, _ = wf_auc(feat_noP, f'islong_{H}', mask=msk)
        lines.append(f'| {H} | {a_all:.3f} | {a_noP:.3f} | {nA:,} |')

    lines += ['', 'Read: AUC>0.6 (no-P) at H>=3 => the streams+regression carry '
              'genuine EARLY anticipation of the combiner (not just reading a near-'
              'threshold P) => qwen anticipation is worth building. ~0.5 no-P => '
              'the combiner is not anticipatable from its ingredients; the fire is '
              'the information, and anticipating it is a mirage.']
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
