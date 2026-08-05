#!/usr/bin/env python3
"""Fair ceiling for LEG-direction anticipation (autonomous, 2026-07-27): qwen got
58% (below gov_dir 61.6%). But qwen's target was leg_dir_true (the actual leg),
which is HARDER than 'anticipate the combiner's gov_dir' (what the earlier 0.88-AUC
probe measured). This asks: at the SAME early anticipation frame, can a GBM on the
numeric ingredients (22 streams + P + gov_dir + cubic slope/curv + z_se band) beat
the gov_dir baseline on leg_dir_true? If GBM<=gov_dir, leg-direction isn't
anticipatable early and qwen was never going to help. If GBM>gov_dir, the signal
exists numerically and qwen is just the wrong tool (distill the GBM, not the LLM).
Same 30-day window/episodes as the qwen run. Walk-forward OOS.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
import cubic_regression as cub                        # noqa: E402
VEC = os.path.join(REPO, 'research', 'nt8_port', 'atlas_backtest')
A1 = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
A5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s')
ZF = os.path.join(REPO, 'DATA', 'ATLAS', 'FEATURES_1s_v2', 'L3_1m')
REGC = [f'L3_1m_{a}_15' for a in ['z_se', 'z_high', 'z_low', 'SE_high', 'SE_low', 'hurst', 'reversion_prob']]
MAXBACK, LAST = 10, 30


def main():
    files = sorted(glob.glob(os.path.join(VEC, '*.parquet')))[-LAST:]
    fcols = [c for c in pd.read_parquet(files[0]).columns if c.startswith('f_')]
    rows = []
    for f in files:
        day = os.path.basename(f)[:10]
        v = pd.read_parquet(f).sort_values('bar_ts').reset_index(drop=True)
        p1 = os.path.join(A1, f'{day}.parquet'); p5 = os.path.join(A5, f'{day}.parquet')
        zp = os.path.join(ZF, f'{day}.parquet')
        if not all(os.path.exists(x) for x in (p1, p5, zp)):
            continue
        cl = dict(zip(pd.read_parquet(p1)['timestamp'].astype('int64'),
                      pd.read_parquet(p1)['close'].astype(float)))
        d5 = pd.read_parquet(p5, columns=['timestamp', 'close']).sort_values('timestamp')
        t5 = d5['timestamp'].astype('int64').to_numpy()
        cval, cslp, ccur = cub.rolling(d5['close'].astype(float).to_numpy(), 90, 5)
        z = pd.read_parquet(zp)
        zt = z['timestamp'].astype('int64').to_numpy()
        zmap = {c: dict(zip(zt, z[c].astype(float))) for c in REGC if c in z}
        bts = v['bar_ts'].astype('int64').to_numpy()
        ent = v['entry'].to_numpy(); gd = v['gov_dir'].to_numpy(); age = v['zz_pivot_age_min'].to_numpy()
        zc = v['zz_confirm'].to_numpy(); fmat = v[fcols].to_numpy()
        Pk = v['P_topk'].to_numpy(); Pa = v['P_any'].to_numpy(); nf = v['n_fires_topk'].to_numpy()
        for e in range(len(bts)):
            if ent[e] != 1 or np.isnan(age[e]):
                continue
            n = int(age[e]); piv = e - n
            if piv < 0 or int(bts[piv]) not in cl:
                continue
            p0 = cl[int(bts[piv])]
            end = len(bts) - 1
            for j in range(e + 1, len(bts)):
                if int(zc[j]) == -int(gd[e]):
                    end = j; break
            seg = [cl.get(int(bts[t])) for t in range(piv, end + 1)]
            seg = [x for x in seg if x is not None]
            if len(seg) < 3:
                continue
            leg = 1 if (max(seg) - p0) >= (p0 - min(seg)) else -1
            t0 = max(piv, e - MAXBACK)              # the early anticipation bar
            ts0 = int(bts[t0]); px0 = cl.get(ts0)
            if px0 is None:
                continue
            j5 = int(np.searchsorted(t5, ts0, side='right')) - 1
            cv = cval[j5] if j5 >= 0 else np.nan
            feat = list(fmat[t0]) + [Pk[t0], Pa[t0], gd[t0], nf[t0],
                                     (px0 - cv) if np.isfinite(cv) else 0.0,
                                     cslp[j5] if j5 >= 0 else 0.0,
                                     ccur[j5] if j5 >= 0 else 0.0]
            feat += [zmap.get(c, {}).get(ts0, 0.0) for c in REGC]
            rows.append(dict(day=day, feat=feat, leg=leg, gov=int(gd[t0]),
                             cub=(1 if (cslp[j5] if j5 >= 0 else 0) > 0 else -1)))
    df = pd.DataFrame(rows)
    df['islong'] = (df['leg'] > 0).astype(int)
    days = sorted(df['day'].unique())
    K = 4; b = [int(len(days) * i / (K + 1)) for i in range(1, K + 2)]
    X = np.array(df['feat'].tolist())
    preds, ys = [], []
    for i in range(K):
        tr = df['day'].isin(days[:b[i]]).to_numpy(); te = df['day'].isin(days[b[i]:b[i + 1]]).to_numpy()
        if te.sum() < 20 or df['islong'][tr].nunique() < 2:
            continue
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=42)
        clf.fit(X[tr], df['islong'].to_numpy()[tr])
        preds.append(clf.predict_proba(X[te])[:, 1]); ys.append(df['islong'].to_numpy()[te])
    y = np.concatenate(ys); p = np.concatenate(preds)
    gbm_acc = ((p > 0.5).astype(int) == y).mean(); gbm_auc = roc_auc_score(y, p)
    gov_acc = (df['gov'] == df['leg']).mean(); cub_acc = (df['cub'] == df['leg']).mean()
    out = (f"# Leg-direction anticipation — GBM ceiling (early frame, {len(df)} episodes)\n"
           f"- **GBM: acc {gbm_acc:.1%}, AUC {gbm_auc:.3f}**\n"
           f"- gov_dir baseline: {gov_acc:.1%}\n- cubic-slope sign: {cub_acc:.1%}\n"
           f"- qwen (same task): 58.0% (below gov_dir 61.6%)\n\n"
           f"Verdict: {'GBM BEATS gov_dir -> signal exists numerically; qwen is the wrong tool' if gbm_acc > gov_acc + 0.02 else 'GBM ~= gov_dir -> leg direction NOT anticipatable early beyond the combiner lean itself'}\n")
    open(os.path.join(REPO, 'research', 'edge_probe', 'reports', 'anticipate_leg_gbm.md'), 'w').write(out)
    print(out)


if __name__ == '__main__':
    main()
