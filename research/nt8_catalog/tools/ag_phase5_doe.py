"""
Phase-5 DOE: leakage-free ENTRY discriminator across ALL dossiers.

CRITICAL FIX vs ag_phase5_final.py / ag_phase5_entry_discriminator.py:
  The prior extractions indexed the FULL-session feature parquet (row 0 = 17:00 CT)
  with event_idx that is RTH-RELATIVE (row 0 = 08:30 CT) for most dossiers. That read
  OVERNIGHT features for DAYTIME trades (verified: ATR-09 event #0 -> 23:15 prior night,
  z_se -0.38, vs the correct 14:11 bar z_se +1.46). All prior Phase-5 numbers are void.

  This script detects each dossier's index convention and aligns correctly:
    - 'rth'  : event_idx indexes the RTH slice (08:30-15:15 CT). Slice, then index.
    - 'full' : event_idx indexes the full-session parquet (SEASON-12). Index directly.
    - 'exclude': event_idx is in a foreign space (RENKO brick-space). Skipped.

Model: entry anchor (PhE) ONLY -> P(registered response). Thresholds frozen on the
train year, evaluated once on test year. Day-block bootstrap. Power gate N>=30/20d.
"""
import os, sys, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FEAT = os.path.abspath(os.path.join(BASE, '../..', 'DATA', 'ATLAS', 'FEATURES_5s_v2'))
CACHE = os.path.join(BASE, 'tools', '_doe_cache')
os.makedirs(CACHE, exist_ok=True)
TIERS = {'L0': 'L0', 'L1': 'L1_5s', 'L2': 'L2_5s', 'L3': 'L3_5s', 'L4': 'L4_5s', 'L5': 'L5_5s'}
RTH0, RTH1 = pd.Timestamp('08:30').time(), pd.Timestamp('15:15').time()
SUBFRICTION, BOOT = 2.0, 4000

def _rth(df):
    dt = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.tz_convert('America/Chicago')
    return df[(dt.dt.time >= RTH0) & (dt.dt.time <= RTH1)].reset_index(drop=True)

def _cols(df):
    return [c for c in df.columns if c not in ('timestamp', 'open', 'high', 'low', 'close', 'volume')]

def extract(dossier):
    cpath = os.path.join(CACHE, dossier + '.npz')
    if os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        return z['X'], z['y'], z['mag'], z['yr'], z['dy'], str(z['mode'])
    ev = pd.read_parquet(os.path.join(BASE, 'tests', dossier, 'events.parquet'))
    eidx_max = int(ev['event_idx'].max())
    mode = 'rth' if eidx_max < 5000 else ('full' if eidx_max < 16000 else 'exclude')
    if mode == 'exclude':
        np.savez(cpath, X=np.zeros((0, 1)), y=np.zeros(0), mag=np.zeros(0),
                 yr=np.array([]), dy=np.array([]), mode=mode)
        return np.zeros((0, 1)), np.zeros(0), np.zeros(0), np.array([]), np.array([]), mode
    colcache = {}
    Xs, ys, ms, yrs, dys = [], [], [], [], []
    for day in sorted(ev['day'].unique()):
        dfmt = day.replace('-', '_')
        tdf, ok = {}, True
        for t, sub in TIERS.items():
            p = os.path.join(FEAT, sub, f"{dfmt}.parquet")
            if not os.path.exists(p):
                ok = False; break
            d = pd.read_parquet(p)
            tdf[t] = _rth(d) if mode == 'rth' else d
        if not ok:
            continue
        de = ev[ev['day'] == day]
        idxs = de['event_idx'].astype(int).values
        maxlen = min(len(tdf[t]) for t in TIERS)
        keep = idxs < maxlen
        if not keep.any():
            continue
        idxs_k = idxs[keep]
        blocks = []
        for t in ['L0', 'L1', 'L2', 'L3', 'L4', 'L5']:
            d = tdf[t]
            if t not in colcache:
                colcache[t] = _cols(d)
            blocks.append(np.nan_to_num(d.iloc[idxs_k][colcache[t]].values.astype(float), nan=0.0))
        Xs.append(np.hstack(blocks))
        ys.append(de['hit'].values[keep].astype(int))
        ms.append(de['magnitude'].values[keep].astype(float))
        yrs.append(np.array([day[:4]] * keep.sum()))
        dys.append(np.array([day] * keep.sum()))
    if not Xs:
        return None
    X, y, mag = np.vstack(Xs), np.concatenate(ys), np.concatenate(ms)
    yr, dy = np.concatenate(yrs), np.concatenate(dys)
    np.savez(cpath, X=X, y=y, mag=mag, yr=yr, dy=dy, mode=mode)
    return X, y, mag, yr, dy, mode

def day_ci(mags, days, nb=BOOT):
    if len(mags) == 0:
        return np.nan, np.nan, np.nan
    uq = np.unique(days); by = {d: mags[days == d] for d in uq}
    mu = [np.concatenate([by[d] for d in np.random.choice(uq, len(uq), True)]).mean() for _ in range(nb)]
    return float(mags.mean()), float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))

def run(dossier, train_y='2024', test_y='2025'):
    got = extract(dossier)
    if got is None:
        return {'dossier': dossier, 'skip': 'no valid events'}
    X, y, mag, yr, dy, mode = got
    if mode == 'exclude':
        return {'dossier': dossier, 'skip': 'foreign index space (brick)'}
    years = sorted(np.unique(yr))
    if train_y not in years or test_y not in years:
        return {'dossier': dossier, 'skip': f'not both years ({years})'}
    m_tr, m_te = (yr == train_y), (yr == test_y)
    if m_tr.sum() < 30 or m_te.sum() < 20:
        return {'dossier': dossier, 'skip': f'thin (tr {m_tr.sum()}, te {m_te.sum()})'}
    mu, sd = X[m_tr].mean(0), X[m_tr].std(0) + 1e-9
    Xtr, Xte = (X[m_tr] - mu) / sd, (X[m_te] - mu) / sd
    ytr = y[m_tr]
    if len(np.unique(ytr)) < 2:
        return {'dossier': dossier, 'skip': 'one class in train'}
    try:
        cv = LogisticRegressionCV(Cs=[0.02, 0.05, 0.1, 0.3, 1.0], penalty='l1', solver='liblinear',
                                  cv=5, scoring='neg_log_loss', max_iter=500).fit(Xtr, ytr)
        sel = np.where(np.abs(cv.coef_[0]) > 1e-6)[0]
    except Exception:
        sel = np.array([], int)
    if len(sel) == 0:
        corr = np.nan_to_num([abs(np.corrcoef(Xtr[:, i], ytr)[0, 1]) for i in range(Xtr.shape[1])])
        sel = np.argsort(corr)[-5:]
    clf = LogisticRegression(penalty=None, max_iter=1000).fit(Xtr[:, sel], ytr)
    ptr = clf.predict_proba(Xtr[:, sel])[:, 1]
    p_lo, p_hi = np.percentile(ptr, 15), np.percentile(ptr, 85)
    pte = clf.predict_proba(Xte[:, sel])[:, 1]
    magte, dyte, yte = mag[m_te], dy[m_te], y[m_te]
    out = {'dossier': dossier, 'nfeat': len(sel), 'ntr': int(m_tr.sum()), 'nte': int(m_te.sum()),
           'base': float(ytr.mean())}
    for nm, msk, sg in [('ACT', pte >= p_hi, magte), ('INV', pte <= p_lo, -magte)]:
        mm, dd, yy = sg[msk], dyte[msk], yte[msk]
        n = int(msk.sum())
        if n == 0:
            out[nm] = dict(n=0, valid=False, ev=np.nan, lo=np.nan, hi=np.nan, mode=np.nan, nd=0, wr=np.nan)
            continue
        ev, lo, hi = day_ci(mm, dd)
        md = float(pd.Series(np.round(mm)).mode().iloc[0]); nd = int(len(np.unique(dd)))
        wr = float((yy == 1).mean()) if nm == 'ACT' else float((yy == 0).mean())
        valid = (n >= 30) and (nd >= 20) and (lo > 0) and (abs(md) >= SUBFRICTION)
        out[nm] = dict(n=n, valid=bool(valid), ev=ev, lo=lo, hi=hi, mode=md, nd=nd, wr=wr)
    return out

def line(r):
    if 'skip' in r:
        return f"| {r['dossier'][:22]} | SKIP: {r['skip']} |"
    def b(x):
        if x['n'] == 0:
            return "N=0"
        flag = "VALID" if x['valid'] else ("under" if x['n'] < 30 or x['nd'] < 20 else "ns")
        return f"N={x['n']}/{x['nd']}d WR{x['wr']:.2f} EV{x['ev']:+.1f} CI[{x['lo']:+.1f},{x['hi']:+.1f}] m{x['mode']:+.0f} {flag}"
    return (f"| {r['dossier'][:22]} | tr{r['ntr']}/te{r['nte']} b{r['base']:.2f} f{r['nfeat']} | "
            f"{b(r['ACT'])} | {b(r['INV'])} |")

if __name__ == '__main__':
    targets = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p))
                                     for p in glob.glob(os.path.join(BASE, 'tests', '*', 'events.parquet')))
    rows = []
    for t in targets:
        r = run(t)
        rows.append(r)
        print(line(r))
    md = ["# Phase-5 DOE — Entry Discriminator across all proposals (ALIGNMENT-FIXED)\n",
          "Leakage-free: entry anchor (PhE) only, RTH-aligned V2 5s features, thresholds frozen on "
          "2024, evaluated on 2025, day-block bootstrap (4000). VALID = branch N>=30 & >=20 days & "
          "day-block CI excludes 0 & |mode|>=2pts. INV EV = mirror approx (-magnitude).\n",
          "| Dossier | train/test base feats | ACT branch | INVERT branch |",
          "|---|---|---|---|"]
    md += [line(r) for r in rows]
    valids = [r for r in rows if 'skip' not in r and (r['ACT']['valid'] or r['INV']['valid'])]
    md.append(f"\n**VALID branches: {len(valids)}** "
              + (", ".join(r['dossier'] for r in valids) if valids else "— none survived."))
    with open(os.path.join(BASE, 'reports', 'AG_cat_00_PHASE5_DOE.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
    print("\nwrote reports/AG_cat_00_PHASE5_DOE.md")
