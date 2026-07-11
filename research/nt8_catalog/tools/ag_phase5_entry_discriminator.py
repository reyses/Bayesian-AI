"""
Phase-5 ENTRY-ONLY discriminator (leakage-free rewrite).

Fixes vs ag_phase5_final.py:
  1. Uses ONLY the entry anchor (PhE). The old model concatenated PhE+PhXit+PhPost;
     PhXit is at the resolution bar and PhPost is AFTER it, so predicting `hit`
     (determined at resolution) from those features was lookahead contamination.
  2. Real DAY-BLOCK bootstrap (resample days, not events) per the effective-N rule.
  3. Thresholds p_hi/p_lo frozen on the train year, evaluated once on the test year.
  4. Sub-friction gate: a branch whose magnitude MODE < 2.0 pts is flagged.

Honesty flags emitted in the report:
  - INVERT EV is a MIRROR APPROXIMATION (EV = -magnitude of the article-side trade),
    NOT a simulated opposite trade with its own exits. Suggestive only.
  - Features are the 5s-timeframe V2 snapshot at entry (L0-L5_5s), a single bar
    per tier -- NOT yet the full multi-timeframe telescoping ladder (doc 017).
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
ATLAS = os.path.abspath(os.path.join(BASE, '../..', 'DATA', 'ATLAS'))
SUBFRICTION_PTS = 2.0          # doc 027 rule 3
THRESH_PCTILE = (15, 85)       # p_lo, p_hi frozen on train
BOOT = 4000

def tier_dirs(is_1s):
    root = os.path.join(ATLAS, 'FEATURES_1s_v2' if is_1s else 'FEATURES_5s_v2')
    return {'L0': os.path.join(root, 'L0'),
            'L1': os.path.join(root, 'L1_5s'), 'L2': os.path.join(root, 'L2_5s'),
            'L3': os.path.join(root, 'L3_5s'), 'L4': os.path.join(root, 'L4_5s'),
            'L5': os.path.join(root, 'L5_5s')}

def extract_entry(dossier):
    """Return X (entry F-space), y (hit), mag (signed pts), year, day -- PhE only."""
    ev = pd.read_parquet(os.path.join(BASE, 'tests', dossier, 'events.parquet'))
    tiers = tier_dirs('ORDERFLOW' in dossier)
    colcache = {}
    X, y, mag, yr, dy = [], [], [], [], []
    for day in sorted(ev['day'].unique()):
        dfmt = day.replace('-', '_')
        dfs = {}
        ok = True
        for t, r in tiers.items():
            p = os.path.join(r, f"{dfmt}.parquet")
            if not os.path.exists(p):
                ok = False; break
            dfs[t] = pd.read_parquet(p)
        if not ok:
            continue
        for _, row in ev[ev['day'] == day].iterrows():
            ei = int(row['event_idx'])
            vec, good = [], True
            for t in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']:
                d = dfs[t]
                if ei >= len(d):
                    good = False; break
                if t not in colcache:
                    colcache[t] = [c for c in d.columns
                                   if c not in ('timestamp', 'open', 'high', 'low', 'close', 'volume')]
                vec.extend(np.nan_to_num(d.iloc[ei][colcache[t]].values.astype(float), nan=0.0))
            if not good:
                continue
            X.append(vec); y.append(int(row['hit'])); mag.append(float(row['magnitude']))
            yr.append(day[:4]); dy.append(day)
    return (np.array(X), np.array(y), np.array(mag), np.array(yr), np.array(dy))

def day_block_ci(mags, days, n_boot=BOOT):
    """Bootstrap the mean by resampling unique DAYS with replacement."""
    if len(mags) == 0:
        return (np.nan, np.nan, np.nan)
    uniq = np.unique(days)
    by_day = {d: mags[days == d] for d in uniq}
    means = []
    for _ in range(n_boot):
        samp = np.random.choice(uniq, size=len(uniq), replace=True)
        pool = np.concatenate([by_day[d] for d in samp])
        means.append(pool.mean())
    return (float(np.mean(mags)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

def run(dossier):
    X, y, mag, yr, dy = extract_entry(dossier)
    years = sorted(np.unique(yr))
    if len(years) < 2:
        print(f"[{dossier}] <2 years ({years}); skip"); return None
    tr_y, te_y = years[0], years[1]
    m_tr, m_te = (yr == tr_y), (yr == te_y)
    if m_tr.sum() < 30 or m_te.sum() < 10:
        print(f"[{dossier}] insufficient N (train {m_tr.sum()}, test {m_te.sum()})"); return None

    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-9
    Xtr, Xte = (X[m_tr] - mu) / sd, (X[m_te] - mu) / sd

    # L1 selection with CV-chosen C on TRAIN ONLY
    try:
        cv = LogisticRegressionCV(Cs=[0.02, 0.05, 0.1, 0.3, 1.0], penalty='l1',
                                  solver='liblinear', cv=5, scoring='neg_log_loss', max_iter=500)
        cv.fit(Xtr, y[m_tr])
        sel = np.where(np.abs(cv.coef_[0]) > 1e-6)[0]
    except Exception as e:
        print(f"[{dossier}] CV failed: {e}"); sel = np.array([], int)
    if len(sel) == 0:
        corr = np.nan_to_num([abs(np.corrcoef(Xtr[:, i], y[m_tr])[0, 1]) for i in range(Xtr.shape[1])])
        sel = np.argsort(corr)[-5:]

    clf = LogisticRegression(penalty=None, max_iter=1000).fit(Xtr[:, sel], y[m_tr])
    p_tr = clf.predict_proba(Xtr[:, sel])[:, 1]
    p_lo, p_hi = np.percentile(p_tr, THRESH_PCTILE[0]), np.percentile(p_tr, THRESH_PCTILE[1])
    p_te = clf.predict_proba(Xte[:, sel])[:, 1]

    mag_te, day_te, y_te = mag[m_te], dy[m_te], y[m_te]
    res = {'dossier': dossier, 'n_feat': len(sel), 'train_y': tr_y, 'test_y': te_y,
           'n_train': int(m_tr.sum()), 'n_test': int(m_te.sum()), 'base_rate': float(y[m_tr].mean())}

    for name, mask, signed in [('ACT', p_te >= p_hi, mag_te), ('INVERT', p_te <= p_lo, -mag_te)]:
        mm, dd, yy = signed[mask], day_te[mask], y_te[mask]
        n = int(mask.sum())
        if n == 0:
            res[name] = dict(n=0, ev=np.nan, ci=(np.nan, np.nan), mode=np.nan, ndays=0, valid=False, wr=np.nan)
            continue
        ev, lo, hi = day_block_ci(mm, dd)
        mode = float(pd.Series(np.round(mm)).mode().iloc[0])
        wr = float((yy == 1).mean()) if name == 'ACT' else float((yy == 0).mean())
        ndays = int(len(np.unique(dd)))
        # A finding needs power: >=30 events across >=20 days, CI excludes 0, above friction.
        valid = (n >= 30) and (ndays >= 20) and (lo > 0) and (abs(mode) >= SUBFRICTION_PTS)
        res[name] = dict(n=n, ev=ev, ci=(lo, hi), mode=mode, ndays=ndays,
                         valid=bool(valid), wr=wr)
    return res

def fmt(r):
    def branch(b):
        if b['n'] == 0:
            return "N=0"
        sub = "" if abs(b['mode']) >= SUBFRICTION_PTS else " [SUB-FRICTION]"
        power = "" if b['n'] >= 30 and b['ndays'] >= 20 else " [UNDERPOWERED]"
        return (f"N={b['n']} ({b['ndays']}d) WR={b['wr']:.2f} EV={b['ev']:+.2f}pts "
                f"CI[{b['ci'][0]:+.2f},{b['ci'][1]:+.2f}] mode={b['mode']:+.0f}{sub}{power} "
                f"{'VALID' if b['valid'] else 'not sig'}")
    return (f"### {r['dossier']}\n"
            f"- train {r['train_y']} N={r['n_train']} (base {r['base_rate']:.3f}) -> test {r['test_y']} N={r['n_test']}; "
            f"{r['n_feat']} features selected\n"
            f"- **ACT**   {branch(r['ACT'])}\n"
            f"- **INVERT**{branch(r['INVERT'])}\n")

if __name__ == '__main__':
    targets = sys.argv[1:] or ['ATR-09_Statistical_Fade', 'FIB-17_Confluence', 'VA-13_Rotation']
    out = ["# Phase-5 Entry Discriminator (leakage-free, PhE only)\n",
           "Entry-anchor V2 F-space (5s snapshot) -> P(registered response). Thresholds frozen on "
           "train year, evaluated once on test year. Day-block bootstrap (4000). "
           "Sub-friction gate: |mode| < 2 pts.\n",
           "> INVERT EV is a MIRROR APPROXIMATION (-magnitude), not a simulated opposite trade.\n",
           "> Features = 5s-TF entry snapshot, NOT yet the full multi-TF telescoping ladder.\n"]
    for t in targets:
        r = run(t)
        if r:
            block = fmt(r); print(block); out.append(block)
    with open(os.path.join(BASE, 'reports', 'AG_cat_00_PHASE5_ENTRY.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(out))
    print("\nwrote reports/AG_cat_00_PHASE5_ENTRY.md")
