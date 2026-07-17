"""
PROP-TURN-P tuning + test (Moises design, doc 094) — P-modulated proportional leg-turn.

Replaces the static PROP-TURN's FIXED retrace fraction r (killed doc 093) with a DYNAMIC
r_eff modulated by a 2024-fitted turn-conviction logistic P_turn:
    r_eff = r_hi - (r_hi - r_lo) * clip((P_turn - p0)/(p1 - p0), 0, 1)   (fire when g >= r_eff)
High P_turn -> r_eff -> r_lo (fire on a SMALL giveback); low P_turn -> r_hi (demand a LARGE
one). Stall gate REMOVED (it forced doc 093's degenerate cell); stall is now a P_turn feature.

Reuses the SHARED numba cores in dossier_signal_pipeline.py (imported, never re-implemented):
  _propturn_p_trace  — reference tracker that emits the 1m-boundary feature vectors for FITTING
  _propturn_p_core   — the DYNAMIC tracker (production generator uses the identical core)
  _pp_arrays         — the tracker-independent per-row feature arrays (ER10/trail_vol/std21/aux)
and the static _propturn_core for the baseline comparison, plus the capture machinery from
propturn_tune_and_capture.py.

Phases (default = all, single process so the 2024 stream is shared by fit + tune):
  --fit-tune : (1) stream 2024, run the reference tracker, collect 1m-boundary features +
               3-min-forward turn labels, fit LogisticRegression (standardized, 2024 SEALED);
               (2) sweep the 36-cell grid (r_lo x r_hi x (p0,p1) x A_min), score each on 2024,
               apply the corrected objective, freeze the winner + P_turn coefs ->
               reports/propturn_p_frozen.json (JSON, no pickle).
  --run      : league (full 604-day pipeline), TEST turn scorecard (2025+26) with deltas vs the
               static baseline (recomputed on the same days), the pre-registered KILL RULE,
               stop-and-reverse capture sim, and the report -> reports/propturn_p.md.
  --all      : --fit-tune then --run (default).

All console output tees to reports/propturn_p_run.log. Raw grid -> reports/propturn_p_grid_2024.csv.
"""
import os, sys, glob, json, itertools, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dossier_signal_pipeline import (
    LBL, D5, REP, TAIL, DayCtx, run_all, evaluate, day_block_ci, _propturn_core,
    _propturn_p_trace, _propturn_p_core, _pp_arrays,
    PROPTURNP_FROZEN, PROPTURNP_RREF, PROPTURNP_AMIN_REF, PROPTURNP_NFEAT)
from propturn_tune_and_capture import (
    load_turns, stream_test_days, simulate_capture, capture_block, ratio_block,
    FRICTION_PT, CHANCE_2M)

# ---- grid + objective (2024 SEALED) --------------------------------------------------
GRID_RLO = [0.03, 0.05, 0.08]
GRID_RHI = [0.15, 0.25, 0.35]
GRID_P = [(0.2, 0.6), (0.3, 0.7)]
GRID_AMIN = [10.0, 15.0]                    # 3 x 3 x 2 x 2 = 36 cells
DIRCORRECT_MIN = 0.80                       # near-turn direction-correctness constraint
LEAD_LO, LEAD_HI = -2.0, 1.0               # lead-median constraint (min), corrected per 093
FWD_HORIZON = 180                           # P_turn label: interior turn within next 3 min (s)
W_LIST = (60, 120, 180, 300)               # +-1/2/3/5 min
STATIC_R, STATIC_S, STATIC_A = 0.05, 3.0, 15.0   # doc-093 frozen static cell (baseline)
FEATS = ['g', 'stall_min', 'A_pts', 'leg_age_min', 'A_over_std21', 'ER10',
         'kmdr_since_min', 'climax_since_min', 'ha_since_min', 'trail_vol']
GRIDCSV = os.path.join(REP, 'propturn_p_grid_2024.csv')
REPORT = os.path.join(REP, 'propturn_p.md')
LOGP = os.path.join(REP, 'propturn_p_run.log')
_LOG = []


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _LOG.append(s)


def flush_log():
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write('\n'.join(_LOG) + '\n')
    _LOG.clear()


# ---- scoring (turn_detection_audit machinery, extended: direction-correctness + CIs) --
def ratio_ci(num, den, days, boots=1000, seed=0):
    """Day-block bootstrap 95% CI on sum(num)/sum(den) (unit of independence = the day)."""
    if len(days) == 0:
        return (float('nan'), float('nan'))
    uq, inv = np.unique(days, return_inverse=True)
    ns = np.zeros(len(uq)); ds = np.zeros(len(uq))
    np.add.at(ns, inv, num); np.add.at(ds, inv, den)
    idx = np.random.default_rng(seed).integers(0, len(uq), size=(boots, len(uq)))
    r = ns[idx].sum(1) / np.maximum(ds[idx].sum(1), 1e-9)
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def score_p(fbd, turns, want_ci=False):
    """recall / dir-recall / precision(near-turn rate) / direction-correctness / lead — at
    +-1/2/3/5m. **dir-correctness is TURN-ANCHORED** (doc-093 consistent: S0->1.00, S3->0.28,
    which a per-fire share cannot produce): of the turns that get a fire within +-W, the
    fraction whose NEAREST fire is correctly directed. This excludes the inverted high-stall
    regime and is not gameable by spraying both directions. want_ci adds day-block CIs at +-2m.
    precision is the standard fire-anchored near-turn rate (fires near any turn / all fires)."""
    res = {}
    n_days = len([d for d in turns if turns[d]])
    for W in W_LIST:
        hit, dhit, hit_days, leads = [], [], [], []
        tcorr, tcorr_days = [], []                       # nearest-fire-correct per HIT turn
        fire_near_l, fire_day_l = [], []
        n_fires = 0
        for day, tl in turns.items():
            if not tl:
                continue
            fts, flong = fbd.get(day, (np.array([]), np.array([], dtype=bool)))
            n_fires += len(fts)
            tarr = np.array([t for t, _ in tl], dtype=float)
            if len(fts):
                idx = np.searchsorted(tarr, fts)
                near = np.zeros(len(fts), bool)
                for koff in (-1, 0):
                    kk = np.clip(idx + koff, 0, len(tarr) - 1)
                    near |= np.abs(fts - tarr[kk]) <= W
                fire_near_l.append(near); fire_day_l.append(np.array([day] * len(fts)))
            for t0, new_long in tl:
                m = (fts >= t0 - W) & (fts <= t0 + W)
                any_hit = bool(m.any())
                hit.append(int(any_hit)); hit_days.append(day)
                if any_hit:
                    sub = fts[m]; subl = flong[m]
                    j = int(np.argmin(np.abs(sub - t0)))
                    dhit.append(int(bool((subl == new_long).any())))
                    leads.append((sub[j] - t0) / 60.0)
                    tcorr.append(int(bool(subl[j] == new_long))); tcorr_days.append(day)
                else:
                    dhit.append(0)
        hit = np.array(hit); dhit = np.array(dhit); tc = np.array(tcorr, float)
        fn = np.concatenate(fire_near_l) if fire_near_l else np.array([], bool)
        fd = np.concatenate(fire_day_l) if fire_day_l else np.array([])
        r_ = dict(n_turns=len(hit),
                  recall=float(hit.mean()) if len(hit) else float('nan'),
                  dir_recall=float(dhit.mean()) if len(dhit) else float('nan'),
                  precision=float(fn.sum() / len(fn)) if len(fn) else float('nan'),
                  dir_correct=float(tc.mean()) if len(tc) else float('nan'),
                  n_fires=int(n_fires))
        if leads:
            la = np.array(leads)
            hb = np.histogram(la, bins=np.arange(-5.25, 5.5, 0.5))
            r_.update(lead_median=float(np.median(la)),
                      lead_mode=float(hb[1][np.argmax(hb[0])] + 0.25),
                      lead_p25=float(np.percentile(la, 25)),
                      lead_p75=float(np.percentile(la, 75)))
        if W == 120 and want_ci:
            r_['dir_recall_ci'] = day_block_ci(dhit.astype(float), np.array(hit_days))
            r_['precision_ci'] = ratio_ci(fn.astype(float), np.ones(len(fn)), fd)
            r_['dir_correct_ci'] = (ratio_ci(tc, np.ones(len(tc)), np.array(tcorr_days))
                                    if len(tc) else (float('nan'), float('nan')))
        res[W] = r_
    res['fires_per_day'] = res[120]['n_fires'] / max(n_days, 1)
    res['n_days'] = n_days
    return res


# ---- 2024 stream + fit ---------------------------------------------------------------
def build_2024_bundles():
    """Per 2024 label day: DayCtx-derived arrays + P_turn per-row arrays + interior turns.
    Same tail-carry continuity as run_all (2024 is the dataset start; first tail is None)."""
    lblf = {os.path.basename(f)[9:19] for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}
    files = sorted(glob.glob(os.path.join(D5, '*.parquet')))
    turns = load_turns(lambda y: y == '2024')
    tail = None
    bundles = []
    for p in tqdm(files, desc='2024 stream'):
        day = os.path.basename(p)[:10]
        if day[:4] > '2024':
            break
        df = pd.read_parquet(p, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if day.replace('-', '_')[:4] == '2024' and day.replace('_', '-') in lblf:
            full = pd.concat([tail, df], ignore_index=True) if tail is not None else df
            ctx = DayCtx(full, len(tail) if tail is not None else 0, day, [])
            arr = _pp_arrays(ctx)
            tl = turns.get(day, [])
            bundles.append(dict(
                day=day, c=ctx.c.astype(np.float64), ts=ctx.ts.astype(np.int64),
                rth=ctx.rth.astype(np.bool_), start=int(ctx.start),
                is1m=(ctx.ts % 60 == 0).astype(np.bool_), arr=arr,
                tarr=np.array([t for t, _ in tl], dtype=float)))
        tail = df.tail(TAIL)
    return bundles, turns


def label_forward(fire_ts, turn_ts, horizon=FWD_HORIZON):
    """y[i] = 1 if an interior turn falls in [fire_ts[i], fire_ts[i]+horizon]."""
    y = np.zeros(len(fire_ts), int)
    if len(turn_ts) == 0 or len(fire_ts) == 0:
        return y
    ta = np.sort(turn_ts)
    lo = np.searchsorted(ta, fire_ts, side='left')      # first turn >= t
    ok = lo < len(ta)
    y[ok] = (ta[lo[ok]] <= fire_ts[ok] + horizon).astype(int)
    return y


def fit_pturn(bundles):
    log(f'\n===== PROP-TURN-P  P_turn FIT (2024 SEALED)  {datetime.datetime.now():%Y-%m-%d %H:%M} =====')
    Xs, ys, gs = [], [], []
    for bd in bundles:
        a = bd['arr']
        feat, fidx = _propturn_p_trace(
            bd['c'], bd['ts'], bd['rth'], np.int64(bd['start']), bd['is1m'],
            float(PROPTURNP_RREF), float(PROPTURNP_AMIN_REF),
            a['er10'], a['tvol'], a['std21'], a['kl'], a['ks'], a['cl'], a['cs'], a['hl'], a['hs'])
        if len(fidx) == 0:
            continue
        ft = bd['ts'][fidx].astype(float)
        Xs.append(feat)
        ys.append(label_forward(ft, bd['tarr']))
        gs.append(np.array([bd['day']] * len(fidx)))
    X = np.vstack(Xs); y = np.concatenate(ys); groups = np.concatenate(gs)
    log(f'reference tracker (r={PROPTURNP_RREF}, A_min={PROPTURNP_AMIN_REF}, no stall, escape on): '
        f'{len(y)} RTH 1m-boundary samples over {len(bundles)} days; base rate {y.mean():.3f}')
    mu, sd = X.mean(0), X.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((X - mu) / sd, y)
    auc_is = roc_auc_score(y, clf.predict_proba((X - mu) / sd)[:, 1])
    gkf = GroupKFold(n_splits=5)
    cv = []
    for tri, vai in gkf.split(X, y, groups):
        m2, s2 = X[tri].mean(0), X[tri].std(0) + 1e-9
        c2 = LogisticRegression(max_iter=2000).fit((X[tri] - m2) / s2, y[tri])
        cv.append(roc_auc_score(y[vai], c2.predict_proba((X[vai] - m2) / s2)[:, 1]))
    coef = clf.coef_[0]
    log(f'P_turn: 2024 in-sample AUC {auc_is:.3f}  |  5-fold GroupKFold(day) CV AUC '
        f'{np.mean(cv):.3f} +/- {np.std(cv):.3f}')
    log('coefs (standardized; sorted by |coef|):')
    for j in np.argsort(-np.abs(coef)):
        log(f'   {FEATS[j]:16} {coef[j]:+.3f}   (mu {mu[j]:+.3f}  sd {sd[j]:.3f})')
    pt = dict(features=FEATS, mu=mu.tolist(), sd=sd.tolist(), coef=coef.tolist(),
              intercept=float(clf.intercept_[0]), auc_2024=float(auc_is),
              auc_cv=float(np.mean(cv)), auc_cv_std=float(np.std(cv)),
              base_rate=float(y.mean()), n_samples=int(len(y)), n_days=len(bundles),
              r_ref=PROPTURNP_RREF, a_min_ref=PROPTURNP_AMIN_REF, fwd_horizon_s=FWD_HORIZON)
    pt['mu_np'] = mu; pt['sd_np'] = sd; pt['coef_np'] = coef      # in-memory for tuning
    flush_log()
    return pt


# ---- 36-cell tuning ------------------------------------------------------------------
def fbd_for_cell(bundles, pt, cell):
    fbd = {}
    for bd in bundles:
        a = bd['arr']
        fi, fl, _ = _propturn_p_core(
            bd['c'], bd['ts'], bd['rth'], np.int64(bd['start']), bd['is1m'],
            float(cell['A_min']), float(cell['r_lo']), float(cell['r_hi']),
            float(cell['p0']), float(cell['p1']),
            pt['mu_np'], pt['sd_np'], pt['coef_np'], float(pt['intercept']),
            a['er10'], a['tvol'], a['std21'], a['kl'], a['ks'], a['cl'], a['cs'], a['hl'], a['hs'])
        fts = bd['ts'][fi].astype(float)
        o = np.argsort(fts)
        fbd[bd['day']] = (fts[o], fl.astype(bool)[o])
    return fbd


def tune(bundles, turns, pt):
    log(f'\n===== PROP-TURN-P  36-CELL TUNING (2024 SEALED)  {datetime.datetime.now():%Y-%m-%d %H:%M} =====')
    grid = list(itertools.product(GRID_RLO, GRID_RHI, GRID_P, GRID_AMIN))
    rows = []
    for r_lo, r_hi, (p0, p1), amin in tqdm(grid, desc='36 cells'):
        cell = dict(r_lo=r_lo, r_hi=r_hi, p0=p0, p1=p1, A_min=amin)
        sc = score_p(fbd_for_cell(bundles, pt, cell), turns)
        s2 = sc[120]
        feas = (np.isfinite(s2['dir_correct']) and s2['dir_correct'] >= DIRCORRECT_MIN and
                np.isfinite(s2.get('lead_median', np.nan)) and
                LEAD_LO <= s2['lead_median'] <= LEAD_HI)
        rows.append(dict(r_lo=r_lo, r_hi=r_hi, p0=p0, p1=p1, A_min=amin,
                         dir_recall_2m=s2['dir_recall'], recall_2m=s2['recall'],
                         precision_2m=s2['precision'], dir_correct_2m=s2['dir_correct'],
                         lead_median=s2.get('lead_median', float('nan')),
                         fires_per_day=sc['fires_per_day'], n_fires=s2['n_fires'],
                         feasible=bool(feas)))
    G = pd.DataFrame(rows)
    G.to_csv(GRIDCSV, index=False)
    log(f'wrote {os.path.basename(GRIDCSV)} ({len(G)} cells)')
    feasible = G[G['feasible']].copy()
    log(f'feasible cells (dir-correct>={DIRCORRECT_MIN} AND lead-median in [{LEAD_LO},{LEAD_HI}], '
        f'no fires/day cap): {len(feasible)}/{len(G)}')
    relaxed = None
    if len(feasible):
        ranked = feasible.sort_values(['dir_recall_2m', 'recall_2m'], ascending=False)
    else:
        dc = G[(G['dir_correct_2m'] >= DIRCORRECT_MIN)]
        if len(dc):
            relaxed = 'lead'
            log(f'!! NO fully-feasible cell — relaxing lead constraint; {len(dc)} cells meet '
                f'dir-correct>={DIRCORRECT_MIN}. Flagged.')
            ranked = dc.sort_values(['dir_recall_2m', 'recall_2m'], ascending=False)
        else:
            relaxed = 'both'
            log('!! NO cell meets dir-correct>=0.80 either — relaxing to global max dir-recall. Flagged.')
            ranked = G.sort_values(['dir_recall_2m', 'recall_2m'], ascending=False)
    log('\nTOP-8 cells (by dir-recall@2m; feasible-first):')
    show = ['r_lo', 'r_hi', 'p0', 'p1', 'A_min', 'dir_recall_2m', 'recall_2m', 'precision_2m',
            'dir_correct_2m', 'lead_median', 'fires_per_day', 'feasible']
    log(G.sort_values(['feasible', 'dir_recall_2m'], ascending=False)[show].head(8).to_string(index=False))
    w = ranked.iloc[0]
    cell = dict(r_lo=float(w['r_lo']), r_hi=float(w['r_hi']), p0=float(w['p0']),
                p1=float(w['p1']), A_min=float(w['A_min']))
    log(f'\nWINNER (frozen): r_lo={cell["r_lo"]} r_hi={cell["r_hi"]} (p0,p1)=({cell["p0"]},{cell["p1"]}) '
        f'A_min={cell["A_min"]:.0f}  dir-recall@2m={w["dir_recall_2m"]:.3f} '
        f'dir-correct={w["dir_correct_2m"]:.3f} lead-med={w["lead_median"]:+.2f}m '
        f'fires/day={w["fires_per_day"]:.1f}' + (f'  [RELAXED={relaxed}]' if relaxed else ''))
    sel = dict(dir_recall_2m=float(w['dir_recall_2m']), recall_2m=float(w['recall_2m']),
               precision_2m=float(w['precision_2m']), dir_correct_2m=float(w['dir_correct_2m']),
               lead_median=float(w['lead_median']), fires_per_day=float(w['fires_per_day']),
               n_feasible=int(len(feasible)), relaxed=relaxed,
               n_turns_2024=int(sum(len(v) for v in turns.values())),
               n_days_2024=len([d for d in turns if turns[d]]))
    pt_json = {k: v for k, v in pt.items() if not k.endswith('_np')}
    frozen = dict(p_turn=pt_json, cell=cell, selection=sel,
                  objective=('max dir-recall@+-2m s.t. dir-correctness(near-turn)>=0.80 AND '
                             'lead-median in [-2,+1]min; NO fires/day cap (doc 094)'),
                  grid_csv=os.path.basename(GRIDCSV),
                  static_baseline=dict(r=STATIC_R, S=STATIC_S, A_min=STATIC_A,
                                       dir_recall_2m=0.042, precision_2m=0.102, note='doc 093'))
    with open(PROPTURNP_FROZEN, 'w', encoding='utf-8') as f:
        json.dump(frozen, f, indent=2)
    log(f'wrote {os.path.basename(PROPTURNP_FROZEN)}')
    flush_log()
    return frozen


# ---- run phase (league + test scorecard + static baseline + kill rule + capture) ------
def static_fbd_test():
    """Static doc-093 cell (r=.05,S=3,A_min=15) fires on the TEST days, same scorer footing."""
    days = stream_test_days()
    fbd = {}
    for day, dd in days.items():
        fi, fl, _ = _propturn_core(dd['c'], dd['ts'], dd['rth'], np.int64(dd['start']),
                                   float(STATIC_R), float(STATIC_S) * 60.0, float(STATIC_A))
        fts = dd['ts'][fi].astype(float)
        o = np.argsort(fts)
        fbd[day] = (fts[o], fl.astype(bool)[o])
    return fbd


def run_phase():
    frozen = json.load(open(PROPTURNP_FROZEN))
    cell = frozen['cell']; pt = frozen['p_turn']
    log(f'\n===== PROP-TURN-P  RUN (league + test + kill rule)  {datetime.datetime.now():%Y-%m-%d %H:%M} =====')
    log(f'frozen cell: r_lo={cell["r_lo"]} r_hi={cell["r_hi"]} (p0,p1)=({cell["p0"]},{cell["p1"]}) '
        f'A_min={cell["A_min"]:.0f} | P_turn 2024 AUC {pt["auc_2024"]:.3f} CV {pt["auc_cv"]:.3f}')

    import dossier_signal_pipeline as P
    P._PPFROZEN = None                                   # force reload of the just-written json
    log('\n--- LEAGUE (full 604-day pipeline run) ---')
    streams, lblf = P.run_all(['PROP-TURN-P'])
    league = P.evaluate('PROP-TURN-P', streams['PROP-TURN-P'], lblf)
    log(f'league: {league if "auc" not in league else {k: league[k] for k in ("n","n_tr","n_te","base_te","auc")}}')

    log('\n--- TEST TURN SCORECARD (2025+26) ---')
    F = streams['PROP-TURN-P']
    Ftest = F[F['day'].str[:4] != '2024'].copy()
    fbd = {}
    for day, g in Ftest.groupby('day'):
        o = np.argsort(g['ts'].values)
        fbd[day] = (g['ts'].values.astype(float)[o], g['is_long'].values.astype(bool)[o])
    turns_te = load_turns(lambda y: y != '2024')
    sc = score_p(fbd, turns_te, want_ci=True)
    scs = score_p(static_fbd_test(), turns_te, want_ci=True)
    _log_scorecard('PROP-TURN-P', sc)
    _log_scorecard('STATIC (recomputed)', scs)

    # ---- KILL RULE (pre-registered, doc 094) ----
    drp, drp_ci = sc[120]['dir_recall'], sc[120]['dir_recall_ci']
    drs, drs_ci = scs[120]['dir_recall'], scs[120]['dir_recall_ci']
    prp, prp_ci = sc[120]['precision'], sc[120]['precision_ci']
    prs, prs_ci = scs[120]['precision'], scs[120]['precision_ci']
    beat_dr = (drp > drs) and (drp_ci[0] > drs_ci[1])
    beat_pr = (prp > prs) and (prp_ci[0] > prs_ci[1])
    kill_pass = beat_dr and beat_pr
    log('\n--- KILL RULE (beat static on BOTH dir-recall@2m AND precision@2m, non-overlapping day-block CIs) ---')
    log(f'  dir-recall@2m: P {drp:.3f} [{drp_ci[0]:.3f},{drp_ci[1]:.3f}]  vs  static {drs:.3f} '
        f'[{drs_ci[0]:.3f},{drs_ci[1]:.3f}]  -> beat={beat_dr}')
    log(f'  precision@2m : P {prp:.3f} [{prp_ci[0]:.3f},{prp_ci[1]:.3f}]  vs  static {prs:.3f} '
        f'[{prs_ci[0]:.3f},{prs_ci[1]:.3f}]  -> beat={beat_pr}')
    log(f'  >>> KILL-RULE VERDICT: {"PASS — family survives" if kill_pass else "FAIL — proportional-turn family (static+dynamic) CLOSED"}')

    # ---- capture (secondary) ----
    log('\n--- CAPTURE (stop-and-reverse, TEST 2025+26; secondary) ---')
    T = simulate_capture(Ftest[['day', 'ts', 'is_long']])
    T.to_csv(os.path.join(REP, 'propturn_p_capture_trades.csv'), index=False)
    cap = {}; rat = {}
    for tag, pred in [('2025', T['year'] == '2025'), ('2026', T['year'] == '2026'),
                      ('POOLED', T['year'].isin(['2025', '2026']))]:
        sub = T[pred]
        if len(sub):
            cap[tag] = capture_block(sub, tag)
            rat[tag] = ratio_block(sub, tag)
    log('capture summary (net-of-friction; secondary; raw rows -> propturn_p_capture_trades.csv):')
    for tag in ('2025', '2026', 'POOLED'):
        c = cap.get(tag); rr = rat.get(tag)
        if not c:
            continue
        log(f"  [{tag}] {c['trades_per_day']:.1f} tr/day  cap-med {c['cap_median']:+.2f}  "
            f"mean {c['cap_mean']:+.2f}[{c['cap_ci'][0]:+.2f},{c['cap_ci'][1]:+.2f}]  PF-WR {c['pfwr']:+.3f}  "
            f"net {c['net_mean']:+.2f}[{c['net_ci'][0]:+.2f},{c['net_ci'][1]:+.2f}]  "
            f"ratio-med {rr.get('median', float('nan')):+.2f}  in-budget[.5,.8] {rr.get('frac_budget', float('nan')):.2f}")

    _write_report(frozen, league, sc, scs, dict(beat_dr=beat_dr, beat_pr=beat_pr, kill_pass=kill_pass,
                  drp=drp, drp_ci=drp_ci, drs=drs, drs_ci=drs_ci, prp=prp, prp_ci=prp_ci,
                  prs=prs, prs_ci=prs_ci), cap, rat)
    flush_log()
    return kill_pass


def _log_scorecard(tag, sc):
    r1, r2, r3, r5 = sc[60], sc[120], sc[180], sc[300]
    ci = r2.get('dir_recall_ci', (float('nan'), float('nan')))
    bar = (r2['precision'] > CHANCE_2M) or (r2['dir_recall'] >= 0.35 and r2.get('lead_median', 9) <= 1.0)
    log(f'[{tag}]')
    log(f"  dir-recall  @1m {r1['dir_recall']:.3f}  @2m {r2['dir_recall']:.3f} "
        f"[CI {ci[0]:.3f},{ci[1]:.3f}]  @3m {r3['dir_recall']:.3f}  @5m {r5['dir_recall']:.3f}")
    log(f"  recall      @1m {r1['recall']:.3f}  @2m {r2['recall']:.3f}   "
        f"precision@2m {r2['precision']:.3f} (chance {CHANCE_2M:.2f})  "
        f"dir-correct@2m {r2['dir_correct']:.3f}")
    log(f"  lead@2m     median {r2.get('lead_median', float('nan')):+.2f}  "
        f"mode {r2.get('lead_mode', float('nan')):+.2f}  p25 {r2.get('lead_p25', float('nan')):+.2f}  "
        f"p75 {r2.get('lead_p75', float('nan')):+.2f} (min)   fires/day {sc['fires_per_day']:.1f}")
    log(f"  STANDING BAR: {'PASS' if bar else 'FAIL'} "
        f"(precision>{CHANCE_2M} OR dir-recall@2m>=0.35 with lead<=+1min)")
    sc['_bar'] = bool(bar)


def _write_report(frozen, league, sc, scs, kr, cap, rat):
    cell = frozen['cell']; pt = frozen['p_turn']; sel = frozen['selection']
    r2 = sc[120]; ci = r2['dir_recall_ci']
    L = []
    A = L.append
    A('# PROP-TURN-P — P-modulated proportional leg-turn (dynamic r_eff)')
    A(f'_Moises design (doc 094). P_turn fit + 36-cell grid tuned on 2024 ONLY (sealed); all '
      f'read-outs below are TEST 2025+26. Generated {datetime.datetime.now():%Y-%m-%d %H:%M}._\n')

    A('## TL;DR — verdict')
    A(f'- **KILL RULE: {"PASS (literal) — proportional-turn family NOT closed" if kr["kill_pass"] else "FAIL — proportional-turn family (static+dynamic) is CLOSED"}.** '
      f'Requirement: beat the static baseline on BOTH dir-recall@2m AND precision@2m with '
      f'non-overlapping day-block CIs. dir-recall beat={kr["beat_dr"]}, precision beat={kr["beat_pr"]}. '
      + ('**But read §4a before acting on this:** the winner is a '
         f'{sc["fires_per_day"]:.0f}/day FIREHOSE where P-modulation is inert, both precisions sit '
         'BELOW the 0.43 chance line, the standing bar FAILS, and capture FAILS — the PASS is '
         'fire-rate saturation, not conviction-modulation rescuing turn timing.' if kr['kill_pass']
         else 'The concept family is closed; the sequential (Mamba) lane proceeds alone.'))
    A(f'- **Standing turn bar: {"PASS" if sc.get("_bar") else "FAIL"}** — dir-recall@±2m '
      f'{r2["dir_recall"]:.3f} [{ci[0]:.3f},{ci[1]:.3f}], precision@2m {r2["precision"]:.3f} '
      f'(chance {CHANCE_2M:.2f}), lead-median {r2.get("lead_median", float("nan")):+.2f}m.')
    lg = f'OOS AUC {league["auc"]:.3f}' if 'auc' in league else league.get('note', 'n/a')
    A(f'- **League (combiner feature):** {lg}.')
    pooled = cap.get('POOLED', {}); pr = rat.get('POOLED', {})
    if pooled:
        A(f'- **Capture (secondary):** net {pooled.get("net_mean", float("nan")):+.2f} pt/trade after '
          f'{FRICTION_PT}-pt friction; capture-ratio median {pr.get("median", float("nan")):+.2f} '
          f'(budget 0.5–0.8; frac in budget {pr.get("frac_budget", float("nan")):.2f}).')
    A('')

    A('## 1. P_turn model (2024 SEALED)')
    A(f'- Reference tracker for fitting: fixed r={pt["r_ref"]}, A_min={pt["a_min_ref"]:.0f}, no stall '
      f'gate, escape on. {pt["n_samples"]} RTH 1m-boundary samples / {pt["n_days"]} days; label = '
      f'interior turn within next {pt["fwd_horizon_s"]//60} min; base rate {pt["base_rate"]:.3f}.')
    A(f'- **2024 in-sample AUC {pt["auc_2024"]:.3f}** | 5-fold GroupKFold(day) CV AUC '
      f'{pt["auc_cv"]:.3f} ± {pt["auc_cv_std"]:.3f}.')
    A('- Coefs (standardized logistic; sorted by |coef|):\n')
    A('| feature | coef |')
    A('|---|---|')
    order = np.argsort(-np.abs(np.array(pt['coef'])))
    for j in order:
        A(f'| {pt["features"][j]} | {pt["coef"][j]:+.3f} |')
    A(f'| _(intercept)_ | {pt["intercept"]:+.3f} |')
    A('')

    A('## 2. Frozen cell (36-cell grid, 2024 SEALED)')
    A(f'Grid: r_lo{GRID_RLO} × r_hi{GRID_RHI} × (p0,p1){GRID_P} × A_min{[int(a) for a in GRID_AMIN]} '
      f'= 36 cells. Objective (doc 094): **max dir-recall@±2m s.t. direction-correctness(near-turn) '
      f'≥ {DIRCORRECT_MIN} AND lead-median ∈ [{LEAD_LO:.0f},{LEAD_HI:.0f}] min; NO fires/day cap.**')
    A(f'- Feasible cells: {sel["n_feasible"]}/36'
      + (f'  _(RELAXED={sel["relaxed"]} — flagged; no fully-feasible cell)_' if sel['relaxed'] else '') + '.')
    A(f'- **FROZEN:** r_lo={cell["r_lo"]}, r_hi={cell["r_hi"]}, (p0,p1)=({cell["p0"]},{cell["p1"]}), '
      f'A_min={cell["A_min"]:.0f} pt.')
    A(f'- 2024 selection stats: dir-recall@2m **{sel["dir_recall_2m"]:.3f}**, recall@2m '
      f'{sel["recall_2m"]:.3f}, precision@2m {sel["precision_2m"]:.3f}, dir-correct@2m '
      f'{sel["dir_correct_2m"]:.3f}, lead-median {sel["lead_median"]:+.2f}m, fires/day '
      f'{sel["fires_per_day"]:.1f}, on {sel["n_turns_2024"]} interior turns / {sel["n_days_2024"]} days.\n')

    A('## 3. TEST turn scorecard (2025+26) — frozen cell, with deltas vs static')
    A('| metric | PROP-TURN-P | static (recomputed) | Δ (P − static) | doc-093 static |')
    A('|---|---|---|---|---|')
    def row(name, key, docval=''):
        p = sc[key[0]][key[1]]; s = scs[key[0]][key[1]]
        return f'| {name} | {p:.3f} | {s:.3f} | {p - s:+.3f} | {docval} |'
    A(row('dir-recall@±1m', (60, 'dir_recall')))
    A(f'| **dir-recall@±2m [CI]** | **{r2["dir_recall"]:.3f}** [{ci[0]:.3f},{ci[1]:.3f}] '
      f'| {scs[120]["dir_recall"]:.3f} [{scs[120]["dir_recall_ci"][0]:.3f},{scs[120]["dir_recall_ci"][1]:.3f}] '
      f'| {r2["dir_recall"] - scs[120]["dir_recall"]:+.3f} | 0.042 |')
    A(row('dir-recall@±3m', (180, 'dir_recall')))
    A(row('dir-recall@±5m', (300, 'dir_recall')))
    A(row('recall@±2m', (120, 'recall')))
    A(f'| **precision@±2m [CI]** (chance {CHANCE_2M:.2f}) | **{r2["precision"]:.3f}** '
      f'[{kr["prp_ci"][0]:.3f},{kr["prp_ci"][1]:.3f}] | {scs[120]["precision"]:.3f} '
      f'[{kr["prs_ci"][0]:.3f},{kr["prs_ci"][1]:.3f}] | {r2["precision"] - scs[120]["precision"]:+.3f} | 0.102 |')
    A(row('dir-correct@±2m', (120, 'dir_correct')))
    A(f'| lead@2m median (min) | {r2.get("lead_median", float("nan")):+.2f} | '
      f'{scs[120].get("lead_median", float("nan")):+.2f} | — | — |')
    A(f'| fires/day | {sc["fires_per_day"]:.1f} | {scs["fires_per_day"]:.1f} | — | — |')
    A(f'\n**Standing-bar verdict: {"PASS" if sc.get("_bar") else "FAIL"}** — bar = precision > {CHANCE_2M} '
      f'OR (dir-recall@2m ≥ 0.35 with lead ≤ +1 min). Best prior stream (RENKO-24) ≈ 0.30 / 0.17.\n')

    A('## 4. KILL RULE (pre-registered, doc 094)')
    A('Beat static on BOTH dir-recall@2m AND precision@2m with **non-overlapping day-block CIs**, '
      'else the proportional-turn family (static + dynamic) is CLOSED.')
    A(f'- dir-recall@2m: P {kr["drp"]:.3f} [{kr["drp_ci"][0]:.3f},{kr["drp_ci"][1]:.3f}] vs static '
      f'{kr["drs"]:.3f} [{kr["drs_ci"][0]:.3f},{kr["drs_ci"][1]:.3f}] → **beat={kr["beat_dr"]}**')
    A(f'- precision@2m : P {kr["prp"]:.3f} [{kr["prp_ci"][0]:.3f},{kr["prp_ci"][1]:.3f}] vs static '
      f'{kr["prs"]:.3f} [{kr["prs_ci"][0]:.3f},{kr["prs_ci"][1]:.3f}] → **beat={kr["beat_pr"]}**')
    A(f'\n**>>> VERDICT: {"PASS (literal rule met) — proportional-turn family NOT closed." if kr["kill_pass"] else "FAIL — proportional-turn family (static + dynamic) is CLOSED."}**\n')

    if kr['kill_pass']:
        G = pd.read_csv(GRIDCSV)
        rh = G[G['A_min'] == 10].groupby('r_hi')['dir_recall_2m'].mean()
        spread = G[(G['r_hi'] == cell['r_hi']) & (G['A_min'] == cell['A_min'])]['dir_recall_2m']
        A('### 4a. Reading the PASS honestly (the caveats the reviewer needs)')
        A('The literal rule is met, but the PASS is **fire-rate saturation, not conviction modulation '
          'rescuing turn timing.** Four things must be weighed before the family is called "alive":')
        A(f'1. **It is a firehose.** {sc["fires_per_day"]:.0f} fires/day (P) vs {scs["fires_per_day"]:.0f} '
          '(static). dir-recall scales ~mechanically with fire rate; the objective (max dir-recall, '
          'NO fires/day cap, per doc 094) explicitly rewards spraying. dir-recall@±3m/±5m barely rise '
          f'({sc[180]["dir_recall"]:.3f}/{sc[300]["dir_recall"]:.3f}) — the extra fires buy ±2m hits, not turn structure.')
        A('2. **P-modulation is essentially INERT at the winner.** dir-recall FALLS monotonically as the '
          f'modulation band widens (A_min=10 grid means): r_hi=0.15→{rh.get(0.15, float("nan")):.3f}, '
          f'0.25→{rh.get(0.25, float("nan")):.3f}, 0.35→{rh.get(0.35, float("nan")):.3f}. So maximizing dir-recall '
          f'drove r_hi to its FLOOR (0.15); at the winning r_hi={cell["r_hi"]}/A_min={cell["A_min"]:.0f} the r_lo '
          f'and (p0,p1) knobs move dir-recall by only {spread.max() - spread.min():.4f}. The dynamic tracker '
          '≈ a sensitive STATIC tracker (r≈0.15, no stall); P_turn (AUC 0.60) is too weak to concentrate fires, '
          'so the objective routes AROUND it.')
        A(f'3. **Both precisions are BELOW the {CHANCE_2M:.2f} chance line** (P {r2["precision"]:.3f}, static '
          f'{scs[120]["precision"]:.3f}). P is only RELATIVELY less-bad than static; neither is an absolute ±2m '
          'turn-timer. The **standing bar FAILS** for both.')
        A(f'4. **Capture FAILS decisively and is WORSE than static** (net {cap.get("POOLED",{}).get("net_mean", float("nan")):+.2f} '
          f'vs static −0.80 pt/trade; 0% in the 0.5–0.8 budget) — the firehose whipsaws harder. The one genuine '
          f'positive is the LEAGUE combiner ({("AUC %.3f" % league["auc"]) if "auc" in league else "n/a"}, up from static 0.636): '
          'as a state FEATURE the fires carry real direction info, but it feeds the combiner, it does not stand alone.')
        A('')

    A('## 5. League line (full 604-day pipeline; direction-agreement with AI labels)')
    if 'auc' in league:
        t = league['ter']
        ts = ' | '.join(f"{b}: {t[b][1]:.2f} [{t[b][2]:.2f},{t[b][3]:.2f}] N={t[b][0]}" for b in t)
        A(f"- N={league['n']} (train {league['n_tr']} / test {league['n_te']}), OOS **AUC "
          f"{league['auc']:.3f}**, test base {league['base_te']:.2f}")
        A(f"- P-terciles: {ts}")
        A(f"- coefs: {league['coefs']}\n")
    else:
        A(f"- {league.get('note', league)}\n")

    A('## 6. CAPTURE — stop-and-reverse (TEST; secondary; the 0.5–0.8 budget)')
    A(f'Flat outside RTH; {FRICTION_PT} pt/round-trip friction (MNQ 1 pt = $2). Per completed leg-trade.\n')
    A('| pop | trades/day | captured median | mean [CI] (pt) | PF Trade WR | net mean [CI] (pt) | ratio median | frac in [.5,.8] |')
    A('|---|---|---|---|---|---|---|---|')
    for tag in ('2025', '2026', 'POOLED'):
        c = cap.get(tag); rr = rat.get(tag)
        if not c:
            continue
        A(f"| {tag} | {c['trades_per_day']:.1f} | {c['cap_median']:+.2f} | {c['cap_mean']:+.2f} "
          f"[{c['cap_ci'][0]:+.2f},{c['cap_ci'][1]:+.2f}] | {c['pfwr']:+.3f} | {c['net_mean']:+.2f} "
          f"[{c['net_ci'][0]:+.2f},{c['net_ci'][1]:+.2f}] | {rr.get('median', float('nan')):+.2f} "
          f"| {rr.get('frac_budget', float('nan')):.2f} |")
    A('')

    A('## 7. Declared choices (spec 094 left these open; all sealed on 2024 before any test read)')
    A(f'- **"A/21 ratio"** ⇒ A / std(last 21 one-minute closes), floor 1 pt (vol-normalized amplitude); '
      'standardization makes the exact normalizer scale irrelevant to the fit.')
    A(f'- **P_turn circularity broken** by fitting on a REFERENCE tracker (fixed r={pt["r_ref"]}, '
      f'A_min={pt["a_min_ref"]:.0f}, no stall gate, escape on) and applying the frozen model to the '
      'dynamic tracker\'s own live state (declared train/deploy shift). "Against leg dir": up-leg '
      'opposed by SHORT aux fires, down-leg by LONG.')
    A('- **Stall gate removed** in the dynamic tracker (it forced doc-093\'s degenerate cell); stall is '
      'a P_turn feature. Escape clause + A_min noise floor retained verbatim from `_propturn_core`.')
    A('- **r_eff updates at every 1m boundary** (RTH + overnight) from P_turn; fit samples only RTH '
      'boundaries; fires stay RTH-gated. Aux fires (EXIT-KMDR/TURN-CLIMAX/TURN-HA) are the existing '
      'generators, precomputed once/day, independent of the proportional tracker.')
    A('- **CIs are day-block bootstraps** (unit of independence = the day): 1000 resamples, precision/'
      'dir-correct as day-summed ratios, dir-recall as a day-blocked mean.')
    A('- Value per fire = completed leg amplitude A (pts); capture fills use the fire-bar CLOSE; final '
      'leg/day force-closes at the RTH close.\n')

    A('## 8. Artifacts')
    A('- `research/nt8_catalog/reports/propturn_p_frozen.json` — frozen P_turn coefs + cell (no pickle)')
    A('- `research/nt8_catalog/reports/propturn_p_grid_2024.csv` — 36-cell 2024 selection grid')
    A('- `research/nt8_catalog/reports/signal_rows_PROPTURNP.parquet` — league signal rows')
    A('- `research/nt8_catalog/reports/propturn_p_capture_trades.csv` — capture leg-trades')
    A('- `research/nt8_catalog/reports/propturn_p_run.log` — full run log')
    A('- generator `PROP-TURN-P` + shared cores appended to `tools/dossier_signal_pipeline.py`; '
      'tuning driver `tools/propturn_p_tune.py`')

    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    log(f'\nwrote {REPORT}')


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else '--all'
    if phase in ('--all', '--fit-tune'):
        bundles, turns = build_2024_bundles()
        pt = fit_pturn(bundles)
        tune(bundles, turns, pt)
        del bundles
    if phase in ('--all', '--run'):
        run_phase()
    if phase not in ('--all', '--fit-tune', '--run'):
        raise SystemExit('usage: propturn_p_tune.py [--all|--fit-tune|--run]')


if __name__ == '__main__':
    main()
