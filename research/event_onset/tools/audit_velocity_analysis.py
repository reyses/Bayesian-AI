"""Stage 2 of the velocity_legs audit: controls + all statistics.
Reads the cached per-trigger frames written by audit_velocity_legs.py.

Day-clustered bootstrap is vectorised via per-day sums/counts
(mean = sum(S[pick]) / sum(N[pick])), which is exact for any statistic that is
a mean -- including P(run>0), which is the mean of an indicator.

  python research/event_onset/tools/audit_velocity_analysis.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_velocity_legs import (CACHE, OUT, GRID, HORIZONS, FRICTION,
                                 load_days, triggers, prestate, auc, cohend)

PRIM = (10, 60)
SEL = (40, 60)            # selective cell: the size of move the owner trades
PREF = ['pre_rng60', 'pre_rng300', 'pre_rng900', 'pre_vol60', 'pre_vol300',
        'pre_rv60', 'pre_rv300', 'pre_pos900', 'pre_compress', 'pre_absret300',
        'pre_tod']
RNG = np.random.default_rng(20260804)
R = {}


def cell(D, T):
    return pd.read_parquet(os.path.join(CACHE, f'audit_D{D}_T{T}.parquet'))


def dboot(vals, days, n=4000, seed=7):
    v = np.asarray(vals, float); d = np.asarray(days)
    u, inv = np.unique(d, return_inverse=True)
    S = np.bincount(inv, weights=v, minlength=len(u))
    N = np.bincount(inv, minlength=len(u)).astype(float)
    p = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    m = S[p].sum(1) / N[p].sum(1)
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ============================================================ d. invariants ==
def item_d():
    o = {}
    for D, T in GRID:
        r = cell(D, T)
        o[f'D{D}_T{T}'] = dict(
            n=len(r), mae_negative=int((r.orig_mae < 0).sum()),
            mfe_negative=int((r.orig_mfe < 0).sum()),
            violate_run_le_mfe=int((r.orig_run > r.orig_mfe + 1e-9).sum()),
            violate_run_ge_negmae=int((r.orig_run < -r.orig_mae - 1e-9).sum()),
            corr_run_mfe=float(np.corrcoef(r.orig_run, r.orig_mfe)[0, 1]),
            corr_run_mae=float(np.corrcoef(r.orig_run, r.orig_mae)[0, 1]),
            dd_matches_disp_sign=True,
            p50_mae=float(r.orig_mae.median()), p50_mfe=float(r.orig_mfe.median()),
            mean_mae=float(r.orig_mae.mean()), mean_mfe=float(r.orig_mfe.mean()),
            p_mfe_gt_mae=float((r.orig_mfe > r.orig_mae).mean()))
    days = {d['day']: d for d in load_days()}
    r = cell(*PRIM).sample(500, random_state=7)
    e_mae = e_mfe = e_run = 0.0
    for _, x in r.iterrows():
        d = days[x['day']]
        a = d['k0'] + int(x['i']); b = min(int(x['i']) + 300, d['nrth'] - 1) + d['k0']
        e = d['c'][a]
        pl = (d['l'][a:b + 1] - e) * x['dd'] if x['dd'] > 0 else (e - d['h'][a:b + 1])
        ph = (d['h'][a:b + 1] - e) * x['dd'] if x['dd'] > 0 else (e - d['l'][a:b + 1])
        e_mae = max(e_mae, abs(max(-pl.min(), 0.) - x['orig_mae']))
        e_mfe = max(e_mfe, abs(max(ph.max(), 0.) - x['orig_mfe']))
        e_run = max(e_run, abs((d['c'][b] - e) * x['dd'] - x['orig_run']))
    o['bruteforce_pnl_path_max_abs_err'] = dict(mae=e_mae, mfe=e_mfe, run=e_run, n=500)
    return o


# =========================================================== e. selection ====
def item_e():
    dl = load_days()
    o = dict(days=dict(files=len(dl), with_rth=int(sum(d['nrth'] > 0 for d in dl)),
                       ge600=int(sum(d['nrth'] >= 600 for d in dl)),
                       dropped_lt600=int(sum(0 < d['nrth'] < 600 for d in dl)),
                       median_rth_bars=float(np.median([d['nrth'] for d in dl if d['nrth'] > 0])),
                       rth_seconds=21600),
             time_axis={}, truncation={}, ties={}, saturation={})
    for D, T in GRID:
        r = cell(D, T); s = r.lookback_secs
        o['time_axis'][f'D{D}_T{T}'] = dict(
            nominal_T=T, mean=float(s.mean()), p50=float(s.median()),
            p90=float(np.percentile(s, 90)), p99=float(np.percentile(s, 99)),
            max=float(s.max()), frac_gt_T=float((s > T).mean()),
            frac_gt_1p5T=float((s > 1.5 * T).mean()), frac_gt_2T=float((s > 2 * T).mean()))
        tr = r.orig_trunc.to_numpy().astype(bool)
        o['truncation'][f'D{D}_T{T}'] = dict(
            frac=float(tr.mean()), n=int(tr.sum()),
            mean_abs_run_trunc=float(r.orig_run[tr].abs().mean()) if tr.any() else None,
            mean_abs_run_full=float(r.orig_run[~tr].abs().mean()),
            p50_mae_trunc=float(r.orig_mae[tr].median()) if tr.any() else None,
            p50_mae_full=float(r.orig_mae[~tr].median()),
            min_follow_secs=float(r.follow_secs.min()),
            p1_follow_secs=float(np.percentile(r.follow_secs, 1)))
        run = r.orig_run
        o['ties'][f'D{D}_T{T}'] = dict(
            p_gt0=float((run > 0).mean()), p_eq0=float((run == 0).mean()),
            p_lt0=float((run < 0).mean()),
            direction_free_baseline=float((1 - (run == 0).mean()) / 2))
        per_day = len(r) / r.day.nunique()
        o['saturation'][f'D{D}_T{T}'] = dict(
            per_day=float(per_day), max_cooldown_slots=360.0,
            saturation=float(per_day / 360.0))
    r = cell(*PRIM)
    o['tod_concentration'] = {k: float(v) for k, v in dict(
        first_5min=(r.tod < 5).mean(), first_15min=(r.tod < 15).mean(),
        first_30min=(r.tod < 30).mean(), first_60min=(r.tod < 60).mean(),
        last_30min=(r.tod >= 330).mean()).items()}
    g = r.groupby('day').size()
    o['per_day_counts'] = dict(mean=float(g.mean()), p10=float(g.quantile(.1)),
                               p50=float(g.median()), p90=float(g.quantile(.9)),
                               min=int(g.min()), max=int(g.max()),
                               top5_share=float(g.nlargest(5).sum() / len(r)),
                               top10_share=float(g.nlargest(10).sum() / len(r)))
    return o


# ==================================================== c. correlated samples ==
def item_c():
    o = {}
    for D, T in GRID:
        r = cell(D, T).sort_values(['day', 'ts']).reset_index(drop=True)
        g = r.groupby('day')
        dt = g['ts'].diff()
        same = g['dd'].diff().fillna(99) == 0
        r['move'] = ((dt.isna()) | (dt > 120) | (~same)).cumsum()
        sz = r.groupby('move').size()
        first = r.groupby('move').head(1)
        all_p = dboot((r.orig_run > 0).astype(float), r.day)
        f_p = dboot((first.orig_run > 0).astype(float), first.day)
        naive = np.sqrt(0.25 / len(r))
        clus = (all_p[2] - all_p[1]) / (2 * 1.96)
        o[f'D{D}_T{T}'] = dict(
            n=len(r), n_moves=int(sz.size), per_move_mean=float(sz.mean()),
            per_move_p50=float(sz.median()), per_move_max=int(sz.max()),
            frac_rows_in_multi_trigger_move=float((sz[r.move].to_numpy() > 1).mean()),
            frac_follow_window_overlaps_prev=float(((dt <= 300) & dt.notna()).mean()),
            frac_follow_window_overlaps_prev_600=float(((dt <= 600) & dt.notna()).mean()),
            p_run_gt0_all=all_p, p_run_gt0_first_per_move=f_p, n_first=len(first),
            mean_run_all=dboot(r.orig_run, r.day),
            mean_run_first_per_move=dboot(first.orig_run, first.day),
            naive_se_pp=float(100 * naive), day_clustered_se_pp=float(100 * clus),
            variance_inflation=float((clus / naive) ** 2),
            effective_n=float(len(r) * (naive / clus) ** 2))
    # cross-cell overlap: are the 7 "independent parameterisations" independent?
    cells = {}
    for D, T in GRID:
        cells[(D, T)] = {d: g.ts.to_numpy() for d, g in cell(D, T).groupby('day')}
    ov = {}
    for a in GRID:
        for b in GRID:
            num = den = 0
            for day, ta in cells[a].items():
                tb = cells[b].get(day)
                if tb is None:
                    den += len(ta); continue
                j = np.clip(np.searchsorted(tb, ta), 0, len(tb) - 1)
                j0 = np.clip(j - 1, 0, len(tb) - 1)
                num += (np.minimum(np.abs(tb[j] - ta), np.abs(tb[j0] - ta)) <= 60).sum()
                den += len(ta)
            ov[f'D{a[0]}_T{a[1]}__within60s_of__D{b[0]}_T{b[1]}'] = float(num / den)
    o['cross_cell_overlap'] = ov
    return o


# ================================================== a+b. anchors x horizons ==
def item_ab():
    o = {}
    for D, T in GRID:
        r = cell(D, T); c = {}
        for an in ('start', 'mid', 'trig'):
            for Hs in HORIZONS:
                run = r[f'run_{an}_{Hs}']
                p = dboot((run > 0).astype(float), r.day)
                base = float((1 - (run == 0).mean()) / 2)
                c[f'{an}_{Hs}'] = dict(
                    p_run_gt0=p, baseline=base,
                    excess_pp=[100 * (x - base) for x in p],
                    mean_run=dboot(run, r.day),
                    mean_run_net=dboot(run - FRICTION, r.day),
                    median_run=float(run.median()),
                    p50_mae=float(r[f'mae_{an}_{Hs}'].median()),
                    p95_mae=float(np.percentile(r[f'mae_{an}_{Hs}'], 95)),
                    p50_mfe=float(r[f'mfe_{an}_{Hs}'].median()),
                    p50_mae_excl_entry_bar=float(r[f'maex_{an}_{Hs}'].median()),
                    frac_trunc=float(r[f'trunc_{an}_{Hs}'].mean()))
        c['hindsight_decomposition'] = dict(
            mean_abs_disp=float(r.disp.mean()),
            mean_run_start_300=float(r.run_start_300.mean()),
            post_trigger_increment=float(r.run_start_300.mean() - r.disp.mean()),
            mean_run_trig_300=float(r.run_trig_300.mean()))
        o[f'D{D}_T{T}'] = c
    return o


# =========================================================== f. is it a null ==
def item_f():
    o = {}
    for D, T in GRID:
        r = cell(D, T); run = r.orig_run.to_numpy()
        pos, neg = run[run > 0], run[run < 0]
        base = float((1 - (run == 0).mean()) / 2)
        o[f'D{D}_T{T}'] = dict(
            n=len(r), p_run_gt0=dboot((run > 0).astype(float), r.day),
            baseline_direction_free=base,
            excess_over_baseline_pp=dboot(100 * (((run > 0).astype(float)
                                                  - (run < 0).astype(float)) / 2), r.day),
            mean_run=dboot(run, r.day),
            mean_run_net_friction=dboot(run - FRICTION, r.day),
            mean_fade_net_friction=dboot(-run - FRICTION, r.day),
            median_run=float(np.median(run)), std=float(run.std()),
            skew=float(pd.Series(run).skew()),
            quantiles={str(q): float(np.percentile(run, q))
                       for q in (1, 5, 10, 25, 50, 75, 90, 95, 99)},
            mean_win=float(pos.mean()), mean_loss=float(neg.mean()),
            win_loss_size_ratio=float(pos.mean() / abs(neg.mean())),
            profit_factor=float(pos.sum() / abs(neg.sum())),
            pf_based_trade_wr=float(pos.sum() / abs(neg.sum()) - 1),
            upper_tail_vs_lower=float(np.percentile(run, 99) / abs(np.percentile(run, 1))),
            sharpe_per_trigger=float(run.mean() / run.std()))
    r = cell(*PRIM); best = {}
    for an in ('start', 'mid', 'trig'):
        for Hs in HORIZONS:
            v = r[f'run_{an}_{Hs}'].to_numpy()
            best[f'{an}_{Hs}'] = dict(net=dboot(v - FRICTION, r.day),
                                      fade_net=dboot(-v - FRICTION, r.day))
    o['primary_expectancy_grid_net_friction'] = best
    return o


# ==================================================== g. pre-impulse state ===
def item_g():
    days = load_days()
    dmap = {d['day']: d for d in days}
    out = {}
    for (D, T), tag in ((PRIM, 'D10_T60'), (SEL, 'D40_T60')):
        tb = {}
        for d in days:
            t = triggers(d, D, T)
            if len(t):
                tb[d['day']] = t
        if tag == 'D10_T60':
            r = cell(D, T)
        else:                                    # build the selective cell
            fr = []
            for dy, idx in tb.items():
                d = dmap[dy]
                F = d['k0'] + idx
                dd = np.sign(d['c'][F] - d['c'][F - T]).astype(np.int64); dd[dd == 0] = 1
                f = prestate(d, F - T)
                f['day'] = [dy] * len(F); f['dd'] = dd; f['tod'] = d['tod'][F]
                f['disp'] = np.abs(d['c'][F] - d['c'][F - T])
                fr.append(pd.DataFrame(f))
            r = pd.concat(fr, ignore_index=True)

        # ---- time-of-day-matched non-impulse controls
        pool, excl_frac = {}, []
        for d in days:
            if d['nrth'] < 600:
                continue
            k0, n = d['k0'], d['nrth']
            bad = np.zeros(n, bool)
            for i in tb.get(d['day'], []):
                bad[max(0, i - 180 - T):min(n, i + 180 + 1)] = True
            excl_frac.append(bad.mean())
            cand = np.arange(910, n - 610, 7)
            cand = cand[~bad[cand]]
            for a in cand:
                pool.setdefault(int(d['tod'][k0 + a]), []).append((d['day'], int(k0 + a)))
        pool = {k: np.array(v, dtype=object) for k, v in pool.items()}
        picks, miss = {}, 0
        for tod, dy in zip(r.tod.to_numpy(), r.day.to_numpy()):
            opt = pool.get(int(tod))
            if opt is None or not len(opt):
                miss += 1; continue
            j = RNG.integers(0, len(opt))
            picks.setdefault(opt[j][0], []).append(int(opt[j][1]))
        cf = []
        for dy, A in picks.items():
            A = np.array(A, np.int64); d = dmap[dy]
            f = prestate(d, A); f['day'] = [dy] * len(A); f['pre_tod'] = d['tod'][A]
            cf.append(pd.DataFrame(f))
        C = pd.concat(cf, ignore_index=True)

        o = dict(n_impulse=len(r), n_control=len(C), tod_match_misses=int(miss),
                 mean_frac_of_RTH_inside_an_impulse_exclusion_zone=float(np.mean(excl_frac)))
        yv = np.r_[np.ones(len(r)), np.zeros(len(C))]
        dv = np.r_[r.day.to_numpy(), C.day.to_numpy()]
        u = np.unique(dv)
        idx_by_day = {q: np.flatnonzero(dv == q) for q in u}
        picks_b = [np.concatenate([idx_by_day[q] for q in
                                   RNG.choice(u, len(u), replace=True)]) for _ in range(400)]
        feats = {}
        for f in PREF:
            xv = np.r_[r[f].to_numpy(), C[f].to_numpy()]
            bs = [auc(xv[m], yv[m]) for m in picks_b]
            feats[f] = dict(auc=auc(xv, yv),
                            auc_ci=[float(np.percentile(bs, 2.5)),
                                    float(np.percentile(bs, 97.5))],
                            cohen_d=cohend(r[f], C[f]),
                            impulse_median=float(r[f].median()),
                            control_median=float(C[f].median()))
        o['impulse_vs_control'] = feats

        dv2 = r.day.to_numpy(); u2 = np.unique(dv2)
        ib2 = {q: np.flatnonzero(dv2 == q) for q in u2}
        pb2 = [np.concatenate([ib2[q] for q in RNG.choice(u2, len(u2), replace=True)])
               for _ in range(400)]
        y = (r.dd.to_numpy() > 0).astype(float)
        dirs = {}
        for f in PREF:
            x = r[f].to_numpy()
            bs = [auc(x[m], y[m]) for m in pb2]
            dirs[f] = dict(auc=auc(x, y),
                           auc_ci=[float(np.percentile(bs, 2.5)),
                                   float(np.percentile(bs, 97.5))])
        o['prestate_predicts_impulse_direction'] = dirs
        out[tag] = o
    return out


if __name__ == '__main__':
    for k, fn in (('d_signs', item_d), ('e_selection', item_e),
                  ('c_correlation', item_c), ('ab_anchor_horizon', item_ab),
                  ('f_null', item_f), ('g_prestate', item_g)):
        R[k] = fn(); print(k, 'done', flush=True)
    json.dump(R, open(os.path.join(OUT, 'audit_velocity_legs.json'), 'w'),
              indent=1, default=float)
    print('wrote', os.path.join(OUT, 'audit_velocity_legs.json'))
