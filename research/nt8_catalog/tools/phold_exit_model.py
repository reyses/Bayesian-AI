"""
P_hold(tau) — the during-trade confidence model (doc: user theory 2026-07-16).

Same binary-logistic machinery as the ENTRY combiner (combiner_preview.md), but run
DURING the open trade on the FULL V2 F-space, with NO fixed horizon. The confidence
that "we are still in the entry-direction move" should DECAY as the move turns over;
P_hold crossing 0.5 = the inverse signal (the label has flipped direction).

Pipeline
--------
POPULATION (engagements): econ_drift_rows.parquet fires with P >= p90(train P);
    same frozen threshold on test; de-dup co-fires within 60s / same day / same dir.
ROWS (during-trade panel): per engagement, per 1-min anchor tau=1..min(60, label_end+10):
    - FULL V2 vector at the last closed 5s anchor <= entry_ts+60*tau (ALL 41 families)
    - context: elapsed_min, drift_so_far (pts, entry dir), entry_P, trail_vol (pts)
    - nan_count (per-row, over V2 features)
    - y = 1 if ACTIVE AI label at that minute agrees with entry dir; 0 if opposite
      label active; row DROPPED if no label active.
MODELS (train 2024 rows; standardize by train mean/std; NaN->train median):
    FULL      = LogisticRegression on [V2 + context + nan_count]
    BASELINE  = LogisticRegression on [context only]  (the pre-registered bar)
READOUTS (test 2025-26): OOS AUC FULL vs BASELINE (overall + tau buckets), calibration,
    decay curves (flipped vs not), flip lead-time, exit-policy capture (displacement).

Usage:  python3.11 phold_exit_model.py
Outputs: reports/phold_rows.parquet, reports/phold_exit_model.md, reports/phold_run.log
Reusable / self-contained. Only external import = day_block_ci (dossier_signal_pipeline).
"""
import os, sys, json, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
from dossier_signal_pipeline import day_block_ci  # day-block bootstrap CI on mean(y)

ECON = os.path.join(ROOT, 'research', 'nt8_catalog', 'reports', 'econ_drift_rows.parquet')
LBLDIR = os.path.join(ROOT, 'DATA', 'ai_cusp_picks')
FEATROOT = os.path.join(ROOT, 'DATA', 'ATLAS', 'FEATURES_5s_v2')
D5DIR = os.path.join(ROOT, 'DATA', 'ATLAS', '5s')
REP = os.path.join(ROOT, 'research', 'nt8_catalog', 'reports')

# ---- constants (no magic numbers) ----------------------------------------------------
BAR_S = 5              # 5s base bar; row B closes at B+BAR_S (build_dataset.py:96)
P_PCTL = 90            # entry-P percentile that defines an engagement (top decile)
DEDUP_S = 60           # co-fires within this many seconds/same dir/day = one engagement
TAU_MAX = 60           # hard cap on during-trade minutes
TAU_PAD = 10           # minutes past the active label's end to keep watching
TRAIL_N = 60           # trail_vol = rolling std of last N 5s closes (points)
TRAIL_MINP = 10        # min periods for trail_vol
ROW_CAP = 1_500_000    # subsample train rows above this (fixed seed)
SEED = 12345
BOOTS = 4000           # headline day-block bootstrap resamples
FAMILIES = sorted(os.listdir(FEATROOT))            # L0 + L1..L5 x 8 TFs = 41 folders
CTX_COLS = ['elapsed_min', 'drift_so_far', 'entry_P', 'trail_vol']
TAU_BUCKETS = [('1-5', 1, 5), ('6-10', 6, 10), ('11-20', 11, 20),
               ('21-40', 21, 40), ('41-60', 41, 60)]
CAP_BIN = 2.0          # points-per-bin for capture/drift mode histograms (MNQ 1pt=$2)

LOG = []
def out(*a):
    s = ' '.join(str(x) for x in a)
    print(s); LOG.append(s)

# ---- feature schema (reference day) --------------------------------------------------
def build_featcols():
    """Stable ordered list of all V2 feature columns across the 41 families."""
    cols = []
    ref = None
    for fam in FAMILIES:
        fs = sorted(glob.glob(os.path.join(FEATROOT, fam, '*.parquet')))
        if not fs:
            continue
        c = [x for x in pd.read_parquet(fs[0]).columns if x != 'timestamp']
        cols.extend(c)
    return cols

# ---- per-day feature panel -----------------------------------------------------------
def load_day(day, featcols):
    """Return (ts_grid int64, feat float32 [n,len(featcols)], close f64, trail_vol f64)
    for one day, all 41 families merged on timestamp + the raw 5s close/vol store."""
    d5 = pd.read_parquet(os.path.join(D5DIR, f'{day}.parquet'))
    master = d5['timestamp'].values.astype(np.int64)
    close = d5['close'].values.astype(np.float64)
    mi = pd.Index(master)
    mats = []
    have = set()
    for fam in FAMILIES:
        f = os.path.join(FEATROOT, fam, f'{day}.parquet')
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f).set_index('timestamp')
        df = df.reindex(mi)
        mats.append(df)
        have.update(df.columns)
    merged = pd.concat(mats, axis=1)
    # reindex to the stable featcol order (missing families/cols -> NaN)
    merged = merged.reindex(columns=featcols)
    feat = merged.values.astype(np.float32)
    tv = pd.Series(close).rolling(TRAIL_N, min_periods=TRAIL_MINP).std(ddof=1).values
    return master, feat, close, tv

# ---- labels --------------------------------------------------------------------------
def load_labels(day):
    """Return list of (entry_ts, exit_ts, is_long, entry_price, exit_price) for a day,
    or None if the label file is missing."""
    f = os.path.join(LBLDIR, f"ai_picks_{day.replace('_', '-')}_multi.json")
    if not os.path.exists(f):
        return None
    tr = json.load(open(f)).get('trades', [])
    out_ = []
    for t in tr:
        if t.get('exit_ts') is None:
            continue
        out_.append((float(t['entry_ts']), float(t['exit_ts']),
                     t.get('direction') == 'LONG',
                     float(t.get('entry_price', np.nan)),
                     float(t.get('exit_price', np.nan))))
    return out_

def active_label(labs, t):
    """First label active at time t, else None."""
    for a, b, lg, ep, xp in labs:
        if a <= t <= b:
            return (a, b, lg, ep, xp)
    return None

# ---- population ----------------------------------------------------------------------
def engagements():
    econ = pd.read_parquet(ECON, columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, P_PCTL))
    out('# P_hold(tau) — during-trade confidence on the full V2 F-space')
    out(f'entry-P p{P_PCTL} threshold (frozen on train) = {thr:.5f}')
    res = {}
    counts = {}
    for split in ['train', 'test']:
        sub = econ[(econ.split == split) & (econ.P >= thr)].copy()
        sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
        # greedy de-dup: keep a fire only if >DEDUP_S past the last kept (same day+dir)
        last = {}
        keep = []
        for r in sub.itertuples():
            k = (r.day, r.is_long)
            if k in last and r.ts - last[k] <= DEDUP_S:
                continue
            last[k] = r.ts
            keep.append(r.Index)
        dd = sub.loc[keep].reset_index(drop=True)
        dd['eid'] = (dd['day'] + '_' + dd['ts'].astype(str) + '_'
                     + np.where(dd['is_long'], 'L', 'S'))
        res[split] = dd
        counts[split] = (len(sub), len(dd))
        out(f'{split}: fires>=thr = {len(sub)}  ->  engagements = {len(dd)} '
            f'(days {dd["day"].nunique()})')
    return res, counts, thr

# ---- panel build for one split -------------------------------------------------------
def build_panel(engs, featcols, want_features):
    """Iterate the split's days, emit during-trade rows.
    want_features=True  -> return (X float32 [N,F+5], meta DataFrame) for training.
    want_features=False -> return only (meta DataFrame) with p_hold slots + summaries;
        (features predicted inline by caller via callback in the streaming path)."""
    # meta accumulators (light columns kept for every row)
    m_eid, m_day, m_tau, m_y = [], [], [], []
    m_drift, m_elapsed, m_entryP, m_trailvol, m_nan = [], [], [], [], []
    blocks = [] if want_features else None
    # per-engagement summary for exit-policy readouts (entry/oracle geometry)
    summ = {}
    skipped_nolabel_entry = 0
    for day, g in tqdm(engs.groupby('day', sort=False), total=engs['day'].nunique(),
                       desc='days'):
        labs = load_labels(day)
        if labs is None or len(labs) == 0:
            continue
        ts_grid, feat, close, tv = load_day(day, featcols)
        n = len(ts_grid)
        for e in g.itertuples():
            al = active_label(labs, e.ts)
            if al is None:
                skipped_nolabel_entry += 1
                continue
            _, lbl_exit_ts, lbl_dir, lbl_ep, lbl_xp = al
            label_end_min = (lbl_exit_ts - e.ts) / 60.0
            taumax = int(min(TAU_MAX, np.floor(label_end_min) + TAU_PAD))
            if taumax < 1:
                continue
            ei = int(np.searchsorted(ts_grid, e.ts, side='right') - 1)
            if ei < 0:
                continue
            entry_close = close[ei]
            # oracle geometry
            oi = int(np.searchsorted(ts_grid, lbl_exit_ts, side='right') - 1)
            oi = min(max(oi, 0), n - 1)
            oclose = close[oi]
            oracle_cap = (oclose - entry_close) if e.is_long else (entry_close - oclose)
            label_disp = (lbl_xp - lbl_ep) if lbl_dir else (lbl_ep - lbl_xp)
            summ[e.eid] = dict(day=day, is_long=bool(e.is_long), entry_P=float(e.P),
                               entry_close=float(entry_close), taumax=taumax,
                               oracle_cap=float(oracle_cap),
                               label_disp=float(label_disp))
            for tau in range(1, taumax + 1):
                t = e.ts + 60 * tau
                if t > ts_grid[-1]:
                    break
                ai = int(np.searchsorted(ts_grid, t - BAR_S, side='right') - 1)
                if ai < 0:
                    break
                lab_t = active_label(labs, t)
                if lab_t is None:
                    continue                       # no label active -> drop row
                y = int(lab_t[2] == e.is_long)
                c = close[ai]
                drift = (c - entry_close) if e.is_long else (entry_close - c)
                trailv = tv[ai]
                if want_features:
                    frow = feat[ai]
                    nanc = int(np.isnan(frow).sum())
                    ctx = np.array([tau, drift, e.P,
                                    trailv if np.isfinite(trailv) else np.nan, nanc],
                                   dtype=np.float32)
                    blocks.append(np.concatenate([frow, ctx]).astype(np.float32))
                else:
                    frow = feat[ai]
                    nanc = int(np.isnan(frow).sum())
                m_eid.append(e.eid); m_day.append(day); m_tau.append(tau); m_y.append(y)
                m_drift.append(float(drift)); m_elapsed.append(tau)
                m_entryP.append(float(e.P))
                m_trailvol.append(float(trailv) if np.isfinite(trailv) else np.nan)
                m_nan.append(nanc)
    meta = pd.DataFrame(dict(engagement_id=m_eid, day=m_day, tau=m_tau, y=m_y,
                             drift_so_far=m_drift, elapsed_min=m_elapsed,
                             entry_P=m_entryP, trail_vol=m_trailvol, nan_count=m_nan))
    X = np.vstack(blocks).astype(np.float32) if want_features and blocks else None
    return X, meta, summ, skipped_nolabel_entry

# ---- exit-policy helpers -------------------------------------------------------------
def _first_sustained(p, thr, k=2):
    """First index i (0-based within the tau series) where p[i-k+1..i] all < thr;
    returns the tau at that sustained breach, else None."""
    below = p < thr
    run = 0
    for i in range(len(below)):
        run = run + 1 if below[i] else 0
        if run >= k:
            return i
    return None

def policy_captures(rows, summ):
    """rows: light test DataFrame w/ engagement_id, tau, drift_so_far, p_hold_full.
    Returns dict policy -> (captures array, ratios array) over engagements."""
    pol = {'fixed_5m': ([], []), 'phold_lt060_2m': ([], []),
           'phold_lt050_2m': ([], []), 'oracle': ([], [])}
    pol_days = {k: [] for k in pol}
    for eid, g in rows.groupby('engagement_id', sort=False):
        g = g.sort_values('tau')
        taus = g['tau'].values
        drift = g['drift_so_far'].values
        p = g['p_hold_full'].values
        s = summ.get(eid)
        if s is None or len(taus) == 0:
            continue
        day = s['day']; ld = s['label_disp']
        def cap_at_tau(tt):
            # captured drift at the exit tau (nearest available <= tt, else last)
            idx = np.searchsorted(taus, tt, side='right') - 1
            idx = min(max(idx, 0), len(taus) - 1)
            return drift[idx]
        # fixed 5m
        c5 = cap_at_tau(5)
        # p_hold policies (exit at 2nd consecutive breach minute; else hold to last)
        i06 = _first_sustained(p, 0.60)
        c06 = drift[i06] if i06 is not None else drift[-1]
        i05 = _first_sustained(p, 0.50)
        c05 = drift[i05] if i05 is not None else drift[-1]
        oc = s['oracle_cap']
        for k, cap in [('fixed_5m', c5), ('phold_lt060_2m', c06),
                       ('phold_lt050_2m', c05), ('oracle', oc)]:
            pol[k][0].append(float(cap))
            pol[k][1].append(float(cap / ld) if ld not in (0.0,) and np.isfinite(ld)
                             else np.nan)
            pol_days[k].append(day)
    return pol, pol_days

# ---- stats helpers -------------------------------------------------------------------
def hist_mode(x, bw=CAP_BIN):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    lo, hi = np.floor(x.min() / bw) * bw, np.ceil(x.max() / bw) * bw + bw
    edges = np.arange(lo, hi + bw, bw)
    h, e = np.histogram(x, bins=edges)
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)

def auc(y, p):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, p))

def auc_delta_ci(y, pf, pb, days, boots=BOOTS, seed=SEED):
    """Day-block bootstrap 95% CI on AUC(full)-AUC(baseline)."""
    uq, inv = np.unique(days, return_inverse=True)
    idx_by_day = [np.flatnonzero(inv == k) for k in range(len(uq))]
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(boots):
        pick = rng.integers(0, len(uq), size=len(uq))
        ridx = np.concatenate([idx_by_day[j] for j in pick])
        yy = y[ridx]
        if len(np.unique(yy)) < 2:
            continue
        ds.append(auc(yy, pf[ridx]) - auc(yy, pb[ridx]))
    if not ds:
        return (float('nan'), float('nan'))
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))

# ---- main ----------------------------------------------------------------------------
def main():
    from sklearn.linear_model import LogisticRegression
    os.makedirs(REP, exist_ok=True)
    featcols = build_featcols()
    out(f'V2 feature columns across {len(FAMILIES)} families = {len(featcols)}')
    engs, counts, thr = engagements()

    # ---- TRAIN panel (with features) ----
    out('\n[build] train panel ...')
    Xtr, meta_tr, summ_tr, skip_tr = build_panel(engs['train'], featcols, True)
    out(f'train rows = {len(meta_tr)}  (skipped no-label-at-entry engagements = {skip_tr})')
    # subsample guard
    if len(meta_tr) > ROW_CAP:
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(meta_tr), ROW_CAP, replace=False)
        sel.sort()
        Xtr = Xtr[sel]; meta_tr = meta_tr.iloc[sel].reset_index(drop=True)
        out(f'  subsampled train to {ROW_CAP} rows (seed {SEED})')

    modelcols = list(featcols) + CTX_COLS + ['nan_count']
    base_idx = [modelcols.index(c) for c in CTX_COLS]
    # impute NaN with TRAIN column median, then standardize by train mean/std
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(np.isnan(Xtr))
    Xtr[inds] = np.take(med, inds[1])
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    Xtr_s = (Xtr - mu) / sd
    ytr = meta_tr['y'].values.astype(int)
    out(f'train class balance y=1: {ytr.mean():.3f}  (N={len(ytr)})')

    out('[fit] FULL logistic (V2 + context + nan_count) ...')
    clf_full = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
    out('[fit] BASELINE logistic (context only) ...')
    clf_base = LogisticRegression(max_iter=2000).fit(Xtr_s[:, base_idx], ytr)
    # train p_hold for saving
    meta_tr['p_hold_full'] = clf_full.predict_proba(Xtr_s)[:, 1]
    meta_tr['p_hold_baseline'] = clf_base.predict_proba(Xtr_s[:, base_idx])[:, 1]
    meta_tr['split'] = 'train'
    coefs = dict(zip(modelcols, clf_full.coef_[0]))
    del Xtr, Xtr_s  # free ~1.3GB

    # ---- TEST panel (streamed day-by-day; predict inline, keep light cols only) ----
    out('\n[build+predict] test panel (streamed) ...')
    eng_test = engs['test']
    lt_eid, lt_day, lt_tau, lt_y = [], [], [], []
    lt_drift, lt_elapsed, lt_entryP, lt_tv, lt_nan = [], [], [], [], []
    lt_pf, lt_pb = [], []
    summ_te = {}
    skip_te = 0
    for day, g in tqdm(eng_test.groupby('day', sort=False),
                       total=eng_test['day'].nunique(), desc='test-days'):
        labs = load_labels(day)
        if labs is None or len(labs) == 0:
            continue
        ts_grid, feat, close, tv = load_day(day, featcols)
        n = len(ts_grid)
        rows_feat = []; rows_meta = []
        for e in g.itertuples():
            al = active_label(labs, e.ts)
            if al is None:
                skip_te += 1
                continue
            _, lbl_exit_ts, lbl_dir, lbl_ep, lbl_xp = al
            label_end_min = (lbl_exit_ts - e.ts) / 60.0
            taumax = int(min(TAU_MAX, np.floor(label_end_min) + TAU_PAD))
            if taumax < 1:
                continue
            ei = int(np.searchsorted(ts_grid, e.ts, side='right') - 1)
            if ei < 0:
                continue
            entry_close = close[ei]
            oi = min(max(int(np.searchsorted(ts_grid, lbl_exit_ts, 'right') - 1), 0), n - 1)
            oracle_cap = (close[oi] - entry_close) if e.is_long else (entry_close - close[oi])
            label_disp = (lbl_xp - lbl_ep) if lbl_dir else (lbl_ep - lbl_xp)
            summ_te[e.eid] = dict(day=day, is_long=bool(e.is_long), entry_P=float(e.P),
                                  entry_close=float(entry_close), taumax=taumax,
                                  oracle_cap=float(oracle_cap), label_disp=float(label_disp))
            for tau in range(1, taumax + 1):
                t = e.ts + 60 * tau
                if t > ts_grid[-1]:
                    break
                ai = int(np.searchsorted(ts_grid, t - BAR_S, side='right') - 1)
                if ai < 0:
                    break
                lab_t = active_label(labs, t)
                if lab_t is None:
                    continue
                y = int(lab_t[2] == e.is_long)
                c = close[ai]
                drift = (c - entry_close) if e.is_long else (entry_close - c)
                trailv = tv[ai]
                frow = feat[ai]
                nanc = int(np.isnan(frow).sum())
                ctx = np.array([tau, drift, e.P,
                                trailv if np.isfinite(trailv) else np.nan, nanc],
                               dtype=np.float32)
                rows_feat.append(np.concatenate([frow, ctx]).astype(np.float32))
                rows_meta.append((e.eid, day, tau, y, float(drift), tau, float(e.P),
                                  float(trailv) if np.isfinite(trailv) else np.nan, nanc))
        if not rows_feat:
            continue
        Xte = np.vstack(rows_feat)
        indn = np.where(np.isnan(Xte))
        Xte[indn] = np.take(med, indn[1])
        Xte_s = (Xte - mu) / sd
        pf = clf_full.predict_proba(Xte_s)[:, 1]
        pb = clf_base.predict_proba(Xte_s[:, base_idx])[:, 1]
        for j, rm in enumerate(rows_meta):
            lt_eid.append(rm[0]); lt_day.append(rm[1]); lt_tau.append(rm[2])
            lt_y.append(rm[3]); lt_drift.append(rm[4]); lt_elapsed.append(rm[5])
            lt_entryP.append(rm[6]); lt_tv.append(rm[7]); lt_nan.append(rm[8])
            lt_pf.append(float(pf[j])); lt_pb.append(float(pb[j]))
    test = pd.DataFrame(dict(engagement_id=lt_eid, day=lt_day, tau=lt_tau, y=lt_y,
                             drift_so_far=lt_drift, elapsed_min=lt_elapsed,
                             entry_P=lt_entryP, trail_vol=lt_tv, nan_count=lt_nan,
                             p_hold_full=lt_pf, p_hold_baseline=lt_pb))
    test['split'] = 'test'
    out(f'test rows = {len(test)}  (skipped no-label-at-entry = {skip_te})')
    out(f'test engagements with rows = {test["engagement_id"].nunique()}')

    # ---- save rows parquet (light columns only) ----
    keep = ['engagement_id', 'day', 'tau', 'y', 'p_hold_full', 'p_hold_baseline',
            'drift_so_far', 'elapsed_min', 'split']
    rows_out = pd.concat([meta_tr[keep], test[keep]], ignore_index=True)
    rows_out.to_parquet(os.path.join(REP, 'phold_rows.parquet'))
    out(f'wrote phold_rows.parquet ({len(rows_out)} rows)')

    # ================= READOUTS (test only) =================
    yv = test['y'].values.astype(int)
    pfv = test['p_hold_full'].values
    pbv = test['p_hold_baseline'].values
    dv = test['day'].values
    tv_arr = test['tau'].values

    out('\n## READOUT 1 — OOS AUC: FULL vs BASELINE')
    a_full = auc(yv, pfv); a_base = auc(yv, pbv)
    out(f'OVERALL  FULL {a_full:.4f}  BASELINE {a_base:.4f}  '
        f'delta {a_full - a_base:+.4f}  (test base y=1 {yv.mean():.3f}, N={len(yv)})')
    dlo, dhi = auc_delta_ci(yv, pfv, pbv, dv)
    out(f'         day-block 95% CI on AUC delta [{dlo:+.4f}, {dhi:+.4f}] (boots={BOOTS})')
    bucket_rows = []
    for name, a, b in TAU_BUCKETS:
        m = (tv_arr >= a) & (tv_arr <= b)
        if m.sum() < 50:
            out(f'  tau {name:5}: N={int(m.sum())} too few'); continue
        af, ab = auc(yv[m], pfv[m]), auc(yv[m], pbv[m])
        out(f'  tau {name:5}: FULL {af:.4f}  BASE {ab:.4f}  delta {af-ab:+.4f}  N={int(m.sum())}')
        bucket_rows.append((name, int(m.sum()), af, ab, af - ab))
    killA = (a_full - a_base) < 0.05
    out(f'KILL-POINT A: FULL-BASELINE overall = {a_full-a_base:+.4f}  '
        + ('=> BELOW 0.05 house bar: the F-space adds ~nothing over trivial state.'
           if killA else '=> clears the 0.05 house bar: F-space adds signal over trivial state.'))

    out('\n## READOUT 2 — Calibration of FULL P_hold (deciles)')
    dec = pd.qcut(pfv, 10, labels=False, duplicates='drop')
    cal_rows = []
    for d in range(int(dec.max()) + 1):
        m = dec == d
        lo, hi = day_block_ci(yv[m].astype(float), dv[m], boots=BOOTS)
        cal_rows.append((d, int(m.sum()), float(pfv[m].mean()), float(yv[m].mean()), lo, hi))
        out(f'  dec {d}: N={int(m.sum()):6}  pred {pfv[m].mean():.3f}  '
            f'obs {yv[m].mean():.3f}  CI[{lo:.3f},{hi:.3f}]')

    out('\n## READOUT 3 — Decay curves: mean P_hold vs tau (flipped-so-far vs not)')
    # flipped_so_far per row = cumulative-any opposite-label (y==0) up to this tau
    test_sorted = test.sort_values(['engagement_id', 'tau'])
    flip_cum = test_sorted.groupby('engagement_id')['y'].transform(
        lambda s: (s == 0).cummax().shift(1, fill_value=False))
    test_sorted = test_sorted.assign(flipped_so_far=flip_cum.values)
    decay_rows = []
    for tau in range(1, TAU_MAX + 1):
        sub = test_sorted[test_sorted.tau == tau]
        if len(sub) < 20:
            continue
        a = sub[~sub.flipped_so_far]['p_hold_full']
        b = sub[sub.flipped_so_far]['p_hold_full']
        decay_rows.append((tau, len(a), a.mean() if len(a) else np.nan,
                           len(b), b.mean() if len(b) else np.nan))
    for tau, na, ma, nb, mb in decay_rows:
        if tau in (1, 5, 10, 20, 30, 40, 50, 60):
            out(f'  tau {tau:2}: not-flipped mean P {ma:.3f} (N={na}) | '
                f'flipped mean P {mb:.3f} (N={nb})')

    out('\n## READOUT 4 — Flip lead-time (P_hold<0.5 sustained 2m minus label-flip minute)')
    leads = []
    for eid, g in test.groupby('engagement_id', sort=False):
        g = g.sort_values('tau')
        yg = g['y'].values; taus = g['tau'].values; p = g['p_hold_full'].values
        zero = np.flatnonzero(yg == 0)
        if len(zero) == 0:
            continue                                  # no flip in window
        flip_min = taus[zero[0]]
        icross = _first_sustained(p, 0.50)
        if icross is None:
            continue
        leads.append(taus[icross] - flip_min)
    leads = np.array(leads, float)
    if len(leads):
        out(f'  flipped engagements w/ a sustained P<0.5 cross: N={len(leads)}')
        out(f'  lead-time (min; negative=early warning): mode {hist_mode(leads,1.0):+.1f}  '
            f'median {np.median(leads):+.1f}  p25 {np.percentile(leads,25):+.1f}  '
            f'p75 {np.percentile(leads,75):+.1f}  mean {leads.mean():+.2f}')
        out(f'  share with EARLY warning (lead<=0): {(leads<=0).mean():.3f}')
    else:
        out('  no flipped engagements with a sustained cross')

    out('\n## READOUT 5 — Exit-policy captured displacement (points; NO dollar claims)')
    pol, pol_days = policy_captures(test, summ_te)
    def cap_ci(vals, days):
        v = np.asarray(vals, float); d = np.asarray(days)
        fm = np.isfinite(v)
        return day_block_ci(v[fm], d[fm], boots=BOOTS)
    pol_summary = {}
    for k in ['fixed_5m', 'phold_lt060_2m', 'phold_lt050_2m', 'oracle']:
        caps = np.array(pol[k][0], float); rats = np.array(pol[k][1], float)
        lo, hi = cap_ci(caps, pol_days[k])
        md = hist_mode(caps); med = np.nanmedian(caps); mn = np.nanmean(caps)
        rmed = np.nanmedian(rats)
        pol_summary[k] = (len(caps), md, med, mn, lo, hi, rmed)
        out(f'  {k:16}: N={len(caps):6}  mode {md:+.1f}  median {med:+.2f}  '
            f'mean {mn:+.2f} CI[{lo:+.2f},{hi:+.2f}]  ratio-median {rmed:+.3f}')
    fixed_med = pol_summary['fixed_5m'][2]
    p06_med = pol_summary['phold_lt060_2m'][2]
    p05_med = pol_summary['phold_lt050_2m'][2]
    killB = not (p06_med > fixed_med or p05_med > fixed_med)
    out(f'KILL-POINT B: fixed-5m median {fixed_med:+.2f} vs P<0.6 {p06_med:+.2f} / '
        f'P<0.5 {p05_med:+.2f}  '
        + ('=> neither P_hold policy beats fixed-5m median: open-ended exit NOT yet earned.'
           if killB else '=> a P_hold policy beats fixed-5m median: open-ended exit earned.'))

    # ---- top-30 |coef| ----
    top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:30]
    out('\n## Top-30 |coef| FULL features (which F-space dims carry the exit signal)')
    for nm, cv in top:
        out(f'  {cv:+.4f}  {nm}')

    # ================= write report =================
    write_report(counts, thr, len(featcols), a_full, a_base, dlo, dhi, killA, killB,
                 bucket_rows, cal_rows, decay_rows, leads, pol_summary, top,
                 len(rows_out), len(test), test['engagement_id'].nunique())
    with open(os.path.join(REP, 'phold_run.log'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG))
    out('\nwrote phold_exit_model.md + phold_run.log')


def write_report(counts, thr, nfeat, a_full, a_base, dlo, dhi, killA, killB,
                 bucket_rows, cal_rows, decay_rows, leads, pol_summary, top,
                 nrows_out, ntest, nteng):
    L = []
    L.append('# P_hold(tau) — during-trade confidence on the full V2 F-space\n')
    L.append('Binary logistic run DURING the open trade on the full V2 vector, no fixed '
             'horizon. P_hold = P(active AI label still agrees with entry direction). '
             'The confidence should decay as the entry-direction move turns over.\n')
    L.append('## Population')
    L.append(f'- entry-P p90 threshold (frozen on train 2024) = **{thr:.5f}**')
    L.append(f'- train: {counts["train"][0]} fires >= thr -> **{counts["train"][1]} '
             f'engagements** after 60s/day/dir de-dup')
    L.append(f'- test : {counts["test"][0]} fires >= thr -> **{counts["test"][1]} '
             f'engagements** after de-dup')
    L.append(f'- during-trade rows saved: {nrows_out} (train+test); test rows {ntest} '
             f'across {nteng} engagements')
    L.append(f'- full V2 F-space = **{nfeat}** feature columns (41 families) + 4 context '
             f'+ nan_count\n')
    L.append('## Readout 1 — OOS AUC (test): FULL vs BASELINE (context-only bar)')
    L.append(f'- **overall FULL {a_full:.4f}  vs  BASELINE {a_base:.4f}  '
             f'(delta {a_full-a_base:+.4f})**; day-block 95% CI on delta '
             f'[{dlo:+.4f}, {dhi:+.4f}]')
    L.append('')
    L.append('| tau bucket | N | FULL AUC | BASE AUC | delta |')
    L.append('|---|---|---|---|---|')
    for name, N, af, ab, dl in bucket_rows:
        L.append(f'| {name} | {N} | {af:.4f} | {ab:.4f} | {dl:+.4f} |')
    L.append(f'\n**KILL-POINT A**: overall delta {a_full-a_base:+.4f} — '
             + ('**BELOW** the 0.05 house bar; the F-space adds ~nothing over trivial '
                'during-trade state (elapsed/drift/entry_P/trail_vol).'
                if killA else '**clears** the 0.05 house bar; the F-space adds signal '
                'over trivial during-trade state.') + '\n')
    L.append('## Readout 2 — Calibration of FULL P_hold (deciles)')
    L.append('| decile | N | pred mean | obs mean | day-block 95% CI |')
    L.append('|---|---|---|---|---|')
    for d, N, pm, om, lo, hi in cal_rows:
        L.append(f'| {d} | {N} | {pm:.3f} | {om:.3f} | [{lo:.3f},{hi:.3f}] |')
    L.append('')
    L.append('## Readout 3 — Decay curves: mean P_hold vs tau')
    L.append('(a) engagements not-yet-flipped at tau vs (b) already-flipped-so-far\n')
    L.append('| tau | N not-flipped | mean P (a) | N flipped | mean P (b) |')
    L.append('|---|---|---|---|---|')
    for tau, na, ma, nb, mb in decay_rows:
        if tau in (1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 60):
            L.append(f'| {tau} | {na} | {ma:.3f} | {nb} | {mb:.3f} |')
    L.append('')
    L.append('## Readout 4 — Flip lead-time (P_hold<0.5 sustained 2m − label-flip minute)')
    if len(leads):
        L.append(f'- N flipped engagements w/ sustained P<0.5 cross = {len(leads)}')
        L.append(f'- **mode {hist_mode(leads,1.0):+.1f} min | median {np.median(leads):+.1f} '
                 f'| p25 {np.percentile(leads,25):+.1f} | p75 {np.percentile(leads,75):+.1f} '
                 f'| mean {leads.mean():+.2f}** (negative = early warning)')
        L.append(f'- share with early warning (lead<=0): {(leads<=0).mean():.3f}\n')
    else:
        L.append('- no flipped engagements with a sustained cross\n')
    L.append('## Readout 5 — Exit-policy captured displacement (points; mode-first)')
    L.append('| policy | N eng | mode | median | mean | day-block 95% CI | capture-ratio median |')
    L.append('|---|---|---|---|---|---|---|')
    names = {'fixed_5m': 'fixed 5-min hold (ref)', 'phold_lt060_2m': 'P_hold<0.6 sust 2m',
             'phold_lt050_2m': 'P_hold<0.5 sust 2m', 'oracle': 'ORACLE (label end)'}
    for k in ['fixed_5m', 'phold_lt060_2m', 'phold_lt050_2m', 'oracle']:
        N, md, med, mn, lo, hi, rmed = pol_summary[k]
        L.append(f'| {names[k]} | {N} | {md:+.1f} | {med:+.2f} | {mn:+.2f} | '
                 f'[{lo:+.2f},{hi:+.2f}] | {rmed:+.3f} |')
    fixed_med = pol_summary['fixed_5m'][2]
    L.append(f'\n**KILL-POINT B**: fixed-5m median {fixed_med:+.2f} pts vs '
             f'P<0.6 {pol_summary["phold_lt060_2m"][2]:+.2f} / '
             f'P<0.5 {pol_summary["phold_lt050_2m"][2]:+.2f} — '
             + ('neither P_hold policy beats the fixed-5m median capture: the open-ended '
                'exit is **NOT yet earned**.'
                if killB else 'a P_hold policy beats the fixed-5m median capture: the '
                'open-ended exit is **earned**.') + '\n')
    L.append('## Top-30 |coef| FULL features (F-space dims carrying the exit signal)')
    L.append('| coef | feature |')
    L.append('|---|---|')
    for nm, cv in top:
        L.append(f'| {cv:+.4f} | {nm} |')
    with open(os.path.join(REP, 'phold_exit_model.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


if __name__ == '__main__':
    main()
