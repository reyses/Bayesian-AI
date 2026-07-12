"""
Phase-5 TELESCOPING LADDER inspection (doc-017), multi-timeframe.

Prior extractions used only L*_5s -- the 5s timeframe. FEATURES_5s_v2 actually holds
L1-L5 for EVERY tf (5s..1D) on the 5s grid. This builds the true telescoping ladder:
each tier looks back N bars AT ITS OWN RESOLUTION (a tf feature only updates every
tf/5s bars, so we step by that stride):

  5s  tier: lags [0,1,2]         (3 x 5s   -> completes 15s)
  15s tier: lags [0,3,6,9]       (4 x 15s  -> completes 1m)
  1m  tier: lags [0,12,24,36]    (4 x 1m   -> completes 5m)
  5m  tier: lags [0,60,120]      (3 x 5m   -> completes 15m)
  15m tier: lags [0,180,360,540] (4 x 15m  -> surrounding ~45min)

Layers L1-L5 per tf (velocity, sigma, z/hurst/reversion, lambda/vr, ldist). Goal:
can the multi-tf entry state DIFFERENTIATE response-completing from response-failing
events -- and within the ride branch, flag the catastrophic runs from the +13 cluster.

Judge: time-ordered forward split (train early days, test later days). No null.
"""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import ag_phase5_doe as D
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

LADDER = {'5s': [0, 1, 2], '15s': [0, 3, 6, 9], '1m': [0, 12, 24, 36],
          '5m': [0, 60, 120], '15m': [0, 180, 360, 540]}
LAYERS = ['L1', 'L2', 'L3', 'L4', 'L5']
MAXLAG = 540
CACHE = os.path.join(D.BASE, 'tools', '_ladder_cache')
os.makedirs(CACHE, exist_ok=True)

def extract(dossier):
    cp = os.path.join(CACHE, dossier + '.npz')
    if os.path.exists(cp):
        z = np.load(cp, allow_pickle=True)
        return z['X'], z['y'], z['mag'], z['dy'], list(z['names'])
    ev = pd.read_parquet(os.path.join(D.BASE, 'tests', dossier, 'events.parquet'))
    cc, names = {}, None
    X, y, mag, dy = [], [], [], []
    for day in sorted(ev['day'].unique()):
        dfmt = day.replace('-', '_')
        # load every needed tier/layer parquet, RTH-slice
        pj = {}
        ok = True
        for tf in LADDER:
            for L in LAYERS:
                key = f'{L}_{tf}'
                p = os.path.join(D.FEAT, key, f'{dfmt}.parquet')
                if not os.path.exists(p):
                    ok = False; break
                pj[key] = D._rth(pd.read_parquet(p))
            if not ok:
                break
        if not ok:
            continue
        for _, r in ev[ev['day'] == day].iterrows():
            ei = int(r['event_idx'])
            if ei - MAXLAG < 0 or any(ei >= len(pj[k]) for k in pj):
                continue
            vec, nm = [], []
            for tf, lags in LADDER.items():
                for lg in lags:
                    for L in LAYERS:
                        d = pj[f'{L}_{tf}']
                        key = f'{L}_{tf}'
                        if key not in cc:
                            cc[key] = D._cols(d)
                        vals = np.nan_to_num(d.iloc[ei - lg][cc[key]].values.astype(float), nan=0.0)
                        vec.extend(vals)
                        if names is None:
                            nm += [f'{c}@{tf}-{lg}' for c in cc[key]]
            if names is None:
                names = nm
            X.append(vec); y.append(int(r['hit'])); mag.append(float(r['magnitude'])); dy.append(day)
    X, y, mag, dy = np.array(X), np.array(y), np.array(mag), np.array(dy)
    o = np.argsort(dy); X, y, mag, dy = X[o], y[o], mag[o], dy[o]
    np.savez(cp, X=X, y=y, mag=mag, dy=dy, names=np.array(names))
    return X, y, mag, dy, names

def forward(dossier, frac=0.6):
    X, y, mag, dy, names = extract(dossier)
    names = np.array(names)
    print(f'{dossier}: ladder X={X.shape} hit={y.mean():.3f} days={len(np.unique(dy))}')
    ud = np.unique(dy); cut = ud[int(len(ud) * frac)]
    mtr, mte = dy < cut, dy >= cut
    mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-9
    Xtr, Xte = (X[mtr] - mu) / sd, (X[mte] - mu) / sd
    cv = LogisticRegressionCV(Cs=[0.02, 0.05, 0.1, 0.3], penalty='l1', solver='liblinear',
                              cv=5, scoring='neg_log_loss', max_iter=800).fit(Xtr, y[mtr])
    sel = np.where(np.abs(cv.coef_[0]) > 1e-6)[0]
    if len(sel) == 0:
        corr = np.nan_to_num([abs(np.corrcoef(Xtr[:, i], y[mtr])[0, 1]) for i in range(Xtr.shape[1])])
        sel = np.argsort(corr)[-8:]
    clf = LogisticRegression(penalty=None, max_iter=1000).fit(Xtr[:, sel], y[mtr])
    ptr = clf.predict_proba(Xtr[:, sel])[:, 1]
    plo, phi = np.percentile(ptr, 15), np.percentile(ptr, 85)
    pte = clf.predict_proba(Xte[:, sel])[:, 1]
    magte, dyte, yte = mag[mte], dy[mte], y[mte]
    print(f'train {mtr.sum()} (<{cut}) / test {mte.sum()} fwd; {len(sel)} features from {X.shape[1]}')
    top = sel[np.argsort(-np.abs(clf.coef_[0]))[:12]]
    print('top features:', [f'{names[i]}({clf.coef_[0][list(sel).index(i)]:+.2f})' for i in top])
    for nm, msk, sg, pos in [('ACT   ', pte >= phi, magte, True), ('INVERT', pte <= plo, -magte, False)]:
        mm, dd, yy = sg[msk], dyte[msk], yte[msk]
        if msk.sum() == 0:
            print(f'{nm}: N=0'); continue
        ev, lo, hi = D.day_ci(mm, dd)
        md = float(pd.Series(np.round(mm)).mode().iloc[0])
        wr = (yy == 1).mean() if pos else (yy == 0).mean()
        neg = int((mm < 0).sum())
        print(f'{nm}: N={msk.sum()}/{len(np.unique(dd))}d win%={wr:.2f} EV={ev:+.1f} CI[{lo:+.1f},{hi:+.1f}] '
              f'med={np.median(mm):+.0f} mode={md:+.0f} losers={neg} worst={sorted(np.round(mm).astype(int))[:4]}')

if __name__ == '__main__':
    for t in (sys.argv[1:] or ['ATR-09_Statistical_Fade']):
        forward(t)
