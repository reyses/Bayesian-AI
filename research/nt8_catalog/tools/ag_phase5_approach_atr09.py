"""
ATR-09 approach-ladder discriminator with a TIME-ORDERED forward split (null-free).

Design (Moises): characterize how F-space APPROACHES the entry t(e) -- entry bar plus
lead-in lags -- pool both years for power, and judge ONLY by a chronological forward
split (train earliest days, test later days never seen). No surrogate null (rejected:
a near-copy surrogate leaks the response and falsely reads as failure).

Result 2026-07-11: INVERT (ride) wins 94% forward, mode +12, but mean EV CI crosses 0
(fat left tail). High-hit-rate structure with tail risk. Next: simulate the real ride
trade with a stop instead of the -magnitude mirror approximation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import ag_phase5_doe as D   # reuse RTH-aligned tiers + day-block bootstrap

DOSSIER = 'ATR-09_Statistical_Fade'
LAGS = [0, 3, 6, 12, 24, 48]      # 0s..4min before entry, in 5s bars
BASE = D.BASE

def extract_approach():
    ev = pd.read_parquet(os.path.join(BASE, 'tests', DOSSIER, 'events.parquet'))
    cc = {}
    X, y, mag, dy = [], [], [], []
    for day in sorted(ev['day'].unique()):
        dfmt = day.replace('-', '_')
        tdf, ok = {}, True
        for t, sub in D.TIERS.items():
            p = os.path.join(D.FEAT, sub, f'{dfmt}.parquet')
            if not os.path.exists(p):
                ok = False; break
            tdf[t] = D._rth(pd.read_parquet(p))
        if not ok:
            continue
        for _, r in ev[ev['day'] == day].iterrows():
            ei = int(r['event_idx'])
            if ei - max(LAGS) < 0 or any(ei >= len(tdf[t]) for t in D.TIERS):
                continue
            vec = []
            for t in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']:
                d = tdf[t]
                if t not in cc:
                    cc[t] = D._cols(d)
                for lg in LAGS:
                    vec.extend(np.nan_to_num(d.iloc[ei - lg][cc[t]].values.astype(float), nan=0.0))
            X.append(vec); y.append(int(r['hit'])); mag.append(float(r['magnitude'])); dy.append(day)
    X, y, mag, dy = np.array(X), np.array(y), np.array(mag), np.array(dy)
    o = np.argsort(dy)
    return X[o], y[o], mag[o], dy[o]

def main():
    X, y, mag, dy = extract_approach()
    print(f'approach-ladder X={X.shape} hit={y.mean():.3f} days={len(np.unique(dy))}')
    udays = np.unique(dy); cut = udays[int(len(udays) * 0.6)]
    mtr, mte = dy < cut, dy >= cut
    mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-9
    Xtr, Xte = (X[mtr] - mu) / sd, (X[mte] - mu) / sd
    cv = LogisticRegressionCV(Cs=[0.02, 0.05, 0.1, 0.3, 1.0], penalty='l1', solver='liblinear',
                              cv=5, scoring='neg_log_loss', max_iter=600).fit(Xtr, y[mtr])
    sel = np.where(np.abs(cv.coef_[0]) > 1e-6)[0]
    if len(sel) == 0:
        corr = np.nan_to_num([abs(np.corrcoef(Xtr[:, i], y[mtr])[0, 1]) for i in range(Xtr.shape[1])])
        sel = np.argsort(corr)[-5:]
    clf = LogisticRegression(penalty=None, max_iter=1000).fit(Xtr[:, sel], y[mtr])
    ptr = clf.predict_proba(Xtr[:, sel])[:, 1]
    plo, phi = np.percentile(ptr, 15), np.percentile(ptr, 85)
    pte = clf.predict_proba(Xte[:, sel])[:, 1]
    magte, dyte, yte = mag[mte], dy[mte], y[mte]
    print(f'train {mtr.sum()} (<{cut}) / test {mte.sum()} forward; {len(sel)} features')
    for nm, msk, sg, wr_pos in [('ACT   ', pte >= phi, magte, True), ('INVERT', pte <= plo, -magte, False)]:
        mm, dd, yy = sg[msk], dyte[msk], yte[msk]
        if msk.sum() == 0:
            print(f'{nm}: N=0'); continue
        ev, lo, hi = D.day_ci(mm, dd)
        md = float(pd.Series(np.round(mm)).mode().iloc[0])
        wr = (yy == 1).mean() if wr_pos else (yy == 0).mean()
        print(f'{nm}: N={msk.sum()}/{len(np.unique(dd))}d win%={wr:.2f} EV={ev:+.1f} CI[{lo:+.1f},{hi:+.1f}] mode={md:+.0f}')

if __name__ == '__main__':
    main()
