"""
NMP9 retuned league + combiner delta (doc 102, Opus drone). Runs the 9-stream NMP9 league
with the QUANTILE-RETUNED constants (dossier_signal_pipeline.NMP9_USE_RETUNED=True), captures
the AFTER per-tier stats, computes the pooled-combiner AUC on the SAME 55-stream pool with
the NMP9 parquets swapped verbatim<->retuned, and RESTORES the verbatim parquets so on-disk
state stays byte-reproducible for the verbatim run. Commits nothing.

  BEFORE per-tier stats come from the verbatim run's tools/nmp9_results.json.
  AFTER  per-tier stats are computed here (flag on).
  Combiner: pooled logistic (combiner_preview logic, inlined so combiner_preview.md is not
  clobbered) over 46 non-NMP9 + 9 NMP9 streams, evaluated with verbatim NMP9 (BEFORE) and
  retuned NMP9 (AFTER) -- the honest same-pool delta vs the 0.676/55-stream anchor.

Run:  python3.11 research/nt8_catalog/tools/nmp9_retune_run.py
Out:  reports/nmp9_retune_results.json  (BEFORE/AFTER per-tier + combiner before/after)
"""
import os, sys, json, glob, shutil
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dossier_signal_pipeline as P
from dossier_signal_pipeline import run_all, evaluate, REP
from nmp9_league import NMP9, eval_from_parquet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CONSENSUS_S = 180
CBASE = ['pivot_age_min', 'sig_with_leg', 'tod', 'inter']


def combiner_auc():
    """Inlined combiner_preview.load_pool + fit -> pooled OOS AUC on all on-disk parquets."""
    frames = []
    dets = []
    for f in sorted(glob.glob(os.path.join(REP, 'signal_rows_*.parquet'))):
        det = os.path.basename(f)[12:-8]
        df = pd.read_parquet(f); df['det'] = det
        frames.append(df); dets.append(det)
    Pp = pd.concat(frames, ignore_index=True).sort_values('ts').reset_index(drop=True)
    ts = Pp['ts'].values.astype(np.int64)
    lng = Pp['is_long'].values.astype(bool)
    lo = np.searchsorted(ts, ts - CONSENSUS_S, 'left')
    hi = np.searchsorted(ts, ts + CONSENSUS_S, 'right')
    def wcount(flags):
        c = np.concatenate([[0], np.cumsum(flags)])
        return c[hi] - c[lo]
    same_dir = np.where(lng, wcount(lng), wcount(~lng))
    own = np.zeros(len(Pp), dtype=np.int64)
    for (d, is_l), g in Pp.groupby(['det', 'is_long'], sort=False):
        flags = np.zeros(len(Pp), dtype=np.int64); flags[g.index.values] = 1
        own[g.index.values] = wcount(flags)[g.index.values]
    Pp['consensus'] = (same_dir - own).astype(np.int16)
    Pp = Pp.dropna(subset=['y']).copy()
    Pp['year'] = Pp['day'].str[:4]
    ud = sorted(Pp['det'].unique())
    for d in ud: Pp[f'is_{d}'] = (Pp['det'] == d).astype(int)
    cols = CBASE + ['consensus'] + [f'is_{d}' for d in ud]
    trm, tem = Pp['year'] == '2024', Pp['year'] != '2024'
    Xtr, ytr = Pp.loc[trm, cols].values.astype(float), Pp.loc[trm, 'y'].astype(int).values
    Xte, yte = Pp.loc[tem, cols].values.astype(float), Pp.loc[tem, 'y'].astype(int).values
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sd, ytr)
    pte = clf.predict_proba((Xte - mu) / sd)[:, 1]
    return float(roc_auc_score(yte, pte)), len(ud), int(len(Pp))


def per_tier_after():
    """Run retuned NMP9 league; return per-tier AFTER stats. Overwrites signal_rows_NMP9*."""
    P.NMP9_USE_RETUNED = True
    streams, lblf = run_all(NMP9)
    res = {}
    for det in NMP9:
        F = streams[det]
        total = int(len(F)); nd = int(F['day'].nunique()) if total else 0
        r = evaluate(det, F, lblf)      # writes retuned signal_rows_NMP9<TIER>.parquet
        r['total_fires'] = total; r['n_all_days'] = nd
        r['fires_day_all'] = float(total / nd) if nd else 0.0
        pq = os.path.join(REP, f'signal_rows_{det.replace("-", "")}.parquet')
        if os.path.exists(pq) and 'auc' in r:
            er = eval_from_parquet(pq)
            r['base_ci'] = er.get('base_ci'); r['fires_day_te'] = er.get('fires_day_te')
            r['n_te_days'] = er.get('n_te_days'); r['n_te'] = er.get('n_te')
        res[det] = r
        tag = f"AUC {r.get('auc'):.3f}" if 'auc' in r else r.get('note', '?')
        print(f'  {det:16} N={r.get("n",0):6} {tag} fires/day(all)={r["fires_day_all"]:.2f}')
    return res


def main():
    nmp9_pq = sorted(glob.glob(os.path.join(REP, 'signal_rows_NMP9*.parquet')))
    assert len(nmp9_pq) == 9, f'expected 9 verbatim NMP9 parquets, found {len(nmp9_pq)}'
    bkdir = os.path.join(REP, 'nmp9_verbatim_backup')
    os.makedirs(bkdir, exist_ok=True)
    for f in nmp9_pq:
        shutil.copy2(f, os.path.join(bkdir, os.path.basename(f)))
    print(f'backed up 9 verbatim NMP9 parquets -> {bkdir}')

    print('running RETUNED NMP9 league (9 streams)...')
    after = per_tier_after()

    print('combiner AFTER (46 non-NMP9 + 9 retuned NMP9)...')
    auc_after, n_after, npool_after = combiner_auc()
    print(f'  pooled OOS AUC {auc_after:.4f} over {n_after} streams, {npool_after} fires')

    # restore verbatim parquets, recompute combiner BEFORE (reproduce the 0.676 anchor)
    for f in nmp9_pq:
        shutil.copy2(os.path.join(bkdir, os.path.basename(f)), f)
    print('restored verbatim NMP9 parquets; combiner BEFORE...')
    auc_before, n_before, npool_before = combiner_auc()
    print(f'  pooled OOS AUC {auc_before:.4f} over {n_before} streams, {npool_before} fires')

    before = json.load(open(os.path.join(HERE, 'nmp9_results.json')))['streams']
    out = dict(
        before=before, after=after,
        combiner=dict(before_auc=round(auc_before, 4), after_auc=round(auc_after, 4),
                      delta=round(auc_after - auc_before, 4),
                      n_streams=n_before, anchor_doc101='0.676 (55-stream same-pool)'),
    )
    dest = os.path.join(REP, 'nmp9_retune_results.json')
    with open(dest, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1, default=float)
    print(f'\nwrote {dest}')
    print(f'combiner: {auc_before:.4f} (verbatim) -> {auc_after:.4f} (retuned)  '
          f'delta {auc_after-auc_before:+.4f}')


if __name__ == '__main__':
    main()
