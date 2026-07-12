"""
Phase-5 VISUAL GALLERY: one distribution plot per dossier of the F-space
discriminator's branches, for visual inspection.

For each dossier (from the DOE cache), fit the entry discriminator on 2024, freeze
thresholds, and plot the 2025 forward magnitude distributions of the three branches
(ACT / SKIP / INVERT) with MODE, median and mean marked (mode-first). This is the
visual companion to reports/AG_cat_00_PHASE5_DOE.md.

Reading the plots:
  - a TAKEABLE branch = a tight cluster clearly on one side of 0, mode away from 0,
    few opposite-tail outliers.
  - a LOTTERY branch = mass near 0 / opposite side with a few huge outliers carrying
    the mean (mean far from mode).
  - NOISE = ACT and INVERT look the same as SKIP.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import ag_phase5_doe as D
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

OUT = os.path.join(D.BASE, 'reports', 'assets', 'phase5_gallery')
os.makedirs(OUT, exist_ok=True)

def branches(dossier):
    got = D.extract(dossier)
    if got is None:
        return None
    X, y, mag, yr, dy, mode = got
    if mode == 'exclude' or '2024' not in yr or '2025' not in yr:
        return None
    mtr, mte = (yr == '2024'), (yr == '2025')
    if mtr.sum() < 30 or mte.sum() < 20 or len(np.unique(y[mtr])) < 2:
        return None
    mu, sd = X[mtr].mean(0), X[mtr].std(0) + 1e-9
    Xtr, Xte = (X[mtr] - mu) / sd, (X[mte] - mu) / sd
    try:
        cv = LogisticRegressionCV(Cs=[0.02, 0.05, 0.1, 0.3, 1.0], penalty='l1', solver='liblinear',
                                  cv=5, scoring='neg_log_loss', max_iter=500).fit(Xtr, y[mtr])
        sel = np.where(np.abs(cv.coef_[0]) > 1e-6)[0]
    except Exception:
        sel = np.array([], int)
    if len(sel) == 0:
        corr = np.nan_to_num([abs(np.corrcoef(Xtr[:, i], y[mtr])[0, 1]) for i in range(Xtr.shape[1])])
        sel = np.argsort(corr)[-5:]
    clf = LogisticRegression(penalty=None, max_iter=1000).fit(Xtr[:, sel], y[mtr])
    ptr = clf.predict_proba(Xtr[:, sel])[:, 1]
    plo, phi = np.percentile(ptr, 15), np.percentile(ptr, 85)
    pte = clf.predict_proba(Xte[:, sel])[:, 1]
    m = mag[mte]
    return {'ACT (take)': m[pte >= phi], 'SKIP': m[(pte < phi) & (pte > plo)],
            'INVERT (ride)': -m[pte <= plo]}

def plot(dossier, br):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{dossier} — F-space discriminator branches (2025 forward)', fontsize=13)
    for a, (name, d) in zip(ax, br.items()):
        if len(d) == 0:
            a.set_title(f'{name} N=0'); continue
        a.hist(d, bins=40, color='steelblue', edgecolor='k', alpha=.7)
        a.axvline(0, color='k', lw=.8)
        md = float(pd.Series(np.round(d)).mode().iloc[0])
        a.axvline(md, color='purple', ls='-', lw=1.5, label=f'mode {md:+.0f}')
        a.axvline(np.median(d), color='g', ls='--', label=f'med {np.median(d):+.0f}')
        a.axvline(np.mean(d), color='r', ls='--', label=f'mean {np.mean(d):+.0f}')
        a.set_title(f'{name} N={len(d)}'); a.legend(fontsize=8); a.set_xlabel('raw pts')
    plt.tight_layout()
    fp = os.path.join(OUT, f'{dossier}.png')
    plt.savefig(fp, dpi=85); plt.close()
    return fp

if __name__ == '__main__':
    targets = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p))
                                     for p in glob.glob(os.path.join(D.BASE, 'tests', '*', 'events.parquet')))
    done = []
    for t in targets:
        try:
            br = branches(t)
        except Exception as e:
            print(f'{t}: ERR {e}'); continue
        if br is None:
            print(f'{t}: skip (thin/foreign/1class)'); continue
        fp = plot(t, br); done.append(t)
        print(f'{t}: plotted -> {os.path.basename(fp)}')
    print(f'\n{len(done)} gallery plots in reports/assets/phase5_gallery/')
