"""
NMP9 league runner (doc 101, Opus drone). Runs the dossier league for EXACTLY the 9
NMP9-* streams (the original 2026-04-08 nine-tier ExNMP waterfall, ported plain) WITHOUT
touching the shared 46-stream dossier_signal_league.md. Writes each stream's
signal_rows_NMP9<TIER>.parquet (via the pipeline's own evaluate, so no drift), then emits
a machine-readable nmp9_results.json (league table + NMPT counterparts) that the report
consumes. Reviewer reproduces one stream from its parquet.

Run:  python3.11 research/nt8_catalog/tools/nmp9_league.py
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dossier_signal_pipeline as P
from dossier_signal_pipeline import (run_all, evaluate, day_block_ci, COLS, REP)

NMP9 = [f'NMP9-{t}' for t in P.NMP9_TIERS]
# NMPT counterpart for the 6 tiers that have one (doc 101 §comparison); the 3 recovered
# tiers (FADEMOM/RIDEMOM/RIDECALM) have NO NMPT counterpart.
COUNTERPART = {'CASCADE': 'NMPT-CASCADE', 'KILLSHOT': 'NMPT-KILLSHOT',
               'FREIGHT': 'NMPT-FREIGHT', 'FADEAGAINST': 'NMPT-FADEAGN',
               'RIDEAGAINST': 'NMPT-RIDEAGN', 'FADECALM': 'NMPT-FADECALM'}


def eval_from_parquet(path):
    """Reproduce OOS-AUC / base / terciles from a saved signal_rows parquet (identical
    train-2024/test logistic to pipeline.evaluate; the reviewer's reproduction path)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    F = pd.read_parquet(path)
    if 'inter' not in F:
        F['inter'] = F['sig_with_leg'] * F['pivot_age_min']
    F = F.dropna(subset=['y'])
    F['year'] = F['day'].str[:4]
    trm, tem = F['year'] == '2024', F['year'] != '2024'
    if trm.sum() < 100 or tem.sum() < 100:
        return dict(n=len(F), note='thin split')
    Xtr, ytr = F.loc[trm, COLS].values, F.loc[trm, 'y'].astype(int).values
    Xte, yte = F.loc[tem, COLS].values, F.loc[tem, 'y'].astype(int).values
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return dict(n=len(F), note='one-class')
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, ytr)
    pte = clf.predict_proba((Xte - mu) / sd)[:, 1]
    auc = float(roc_auc_score(yte, pte))
    days_te = F.loc[tem, 'day'].values
    lo, hi = day_block_ci(yte, days_te)
    try:
        q = pd.qcut(pte, 3, labels=['low', 'mid', 'high'])
    except ValueError:
        q = pd.Series(['mid'] * len(pte))
    ter = {}
    for b in ['low', 'mid', 'high']:
        m = np.asarray(q == b)
        if m.sum() < 10:
            continue
        tlo, thi = day_block_ci(yte[m], days_te[m])
        ter[b] = [int(m.sum()), float(yte[m].mean()), tlo, thi]
    n_te_days = len(np.unique(days_te))
    return dict(n=int(len(F)), n_tr=int(trm.sum()), n_te=int(tem.sum()),
                base_te=float(yte.mean()), base_ci=[lo, hi], auc=auc, ter=ter,
                n_te_days=n_te_days, fires_day_te=float(tem.sum() / max(n_te_days, 1)))


def main():
    print('running NMP9 league (9 streams)...')
    streams, lblf = run_all(NMP9)
    out = {'baseline_combiner_documented': 0.689, 'streams': {}, 'counterparts': {}}
    for det in NMP9:
        F = streams[det]
        total_fires = int(len(F))
        n_days = int(F['day'].nunique()) if total_fires else 0
        r = evaluate(det, F, lblf)          # writes signal_rows_NMP9<TIER>.parquet
        r['total_fires'] = total_fires
        r['n_all_days'] = n_days
        r['fires_day_all'] = float(total_fires / n_days) if n_days else 0.0
        # overall test day-block CI (evaluate only gives per-tercile CIs)
        pq = os.path.join(REP, f'signal_rows_{det.replace("-", "")}.parquet')
        if os.path.exists(pq) and 'auc' in r:
            er = eval_from_parquet(pq)
            r['base_ci'] = er.get('base_ci')
            r['fires_day_te'] = er.get('fires_day_te')
            r['n_te_days'] = er.get('n_te_days')
        out['streams'][det] = r
        tag = f"AUC {r.get('auc'):.3f}" if 'auc' in r else r.get('note', '?')
        print(f"  {det:16} N={r.get('n', 0):6} {tag}  fires/day(all)={r['fires_day_all']:.2f}")
    # NMPT counterparts from existing parquets
    for tier, nmpt in COUNTERPART.items():
        pq = os.path.join(REP, f'signal_rows_{nmpt.replace("-", "")}.parquet')
        if os.path.exists(pq):
            out['counterparts'][tier] = dict(nmpt=nmpt, **eval_from_parquet(pq))
        else:
            out['counterparts'][tier] = dict(nmpt=nmpt, note='parquet missing')
    dest = os.path.join(HERE, 'nmp9_results.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
