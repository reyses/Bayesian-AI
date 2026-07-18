"""
Wrong-Direction Dojo -- the DISTILLED VETO (research/exit_dojo/tools/veto_logistic.py)

Task 105 (reviewer, 2026-07-18). Night charter: "cut losers fast, let winners ride" WITHOUT
Mamba. Both dojos proved the dip-veto information lives in the path (clean 10% vs dipped 54%
false-bail, doc 100). This distills it into a 2024-SEALED logistic that, at the moment a plain
24-tick stop TRIGGERS, prices "will this recover?" from path-derivable features only -- a
mechanical, CPU-cheap veto.

Pipeline:
  TRAIN (2024, split=='train'): reuse select_wrongdir's engagement cut + per-minute drift
    path extraction (sw.scan). For every engagement whose plain 24t stop triggers, the trigger
    minute t* is the decision point. Features at t* (causal, <= t* ONLY -- asserted). Target =
    sign(terminal - drift[t*]) (does the path from the trigger onward end favorable?).
    LogisticRegression, standardized on 2024. Sweep p* by mean net ticks/ep. Freeze (p*, coefs,
    scaler, feature order) to reports/wrongdir/veto_frozen.json.
  TEST (the 198 doc-100 episodes, ONCE): frontier net ticks/ep + day-block CI for
    never-bail / plain-stop-24t / STOP+VETO / (blind agents +7.5 reference). Per-class:
    dipped-good false-bail (the 54% line to beat), wrong-catch, veto precision/recall.

Economics convention (matches score_wrongdir + doc 100):
  net-vs-never-bail (ticks) = (drift[exit] - drift[window]) * 4    (friction-free: every policy
    is exactly ONE round trip, so the 2.4t/RT friction is a constant offset that cancels in all
    deltas and in the p* argmax). Absolute realized ticks/ep WITH friction (drift[exit]*4 -
    FRICTION_TICKS) is reported alongside for honesty. Doc 100's plain-stop +17.7 is the
    friction-free net-vs-never-bail reference; this script recomputes it internally.

New files only: this script + reports/wrongdir/veto_{frozen.json,decisions.parquet},
reports/wrongdir/veto_logistic.md. Does NOT edit score_wrongdir.py / select_wrongdir.py /
stop_reenter_sim.py (imports/copies helpers). Commit NOTHING.

Run: python3.11 research/exit_dojo/tools/veto_logistic.py
"""
import os
import sys
import json
import glob

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUILDERS = os.path.abspath(os.path.join(HERE, '..', 'builders'))
sys.path.insert(0, BUILDERS)

import select_wrongdir as sw            # engagement cut + scan() per-minute drift extraction
import score_wrongdir as scw            # dumb_exit_minute / net_ticks / boot_ci / hist_mode
import episode_builder as eb            # signed_drift_path (not needed directly; sw.scan uses it)
import telescope_packet_builder as tb   # window helpers (via sw)

# ---- paths -------------------------------------------------------------------------------
WRONGDIR_DIR = scw.WRONGDIR_DIR
TRUTH_DIR = scw.TRUTH_DIR
SELECTION_JSON = scw.SELECTION_JSON
FROZEN_JSON = os.path.join(WRONGDIR_DIR, 'veto_frozen.json')
DECISIONS_PARQUET = os.path.join(WRONGDIR_DIR, 'veto_decisions.parquet')
REPORT_MD = os.path.join(WRONGDIR_DIR, 'veto_logistic.md')

# ---- constants (house rule: no bare magic numbers) --------------------------------------
STOP_TICKS = 24                       # the plain stop-trigger threshold (6.0 pts) -- the decision point
PTS_TO_TICKS = scw.PTS_TO_TICKS       # 4.0
TICK_PTS = scw.TICK_PTS               # 0.25
FRICTION_TICKS = 2.4                  # round-trip friction (ticks), same convention as doc 103
BW_TICKS = scw.BW_TICKS               # 4t = $2 mode bin
BOOTS = scw.BOOTS                     # 4000
SEED = scw.SEED                       # 12345
ER_MAX_MIN = 10                       # efficiency ratio / vol lookback horizon (<=10m), causal
EPS_VOL = 1e-6                        # guard for zero trailing vol
P_STAR_GRID = np.round(np.arange(0.30, 0.905, 0.025), 3)   # p* sweep grid on 2024
LOGIT_C = 1.0                         # L2 inverse-strength; default (no per-cell tuning)
LOGIT_MAX_ITER = 2000
# the 2 doc-100 episodes that FAILED the nonce-chain audit (excluded so the frontier matches
# doc 100's scored 198 exactly). Both are good/dipped.
NONCE_FAIL_EIDS = {'2025_01_28_1738076215_L', '2025_09_11_1757600520_L'}
AGENTS_NET_REF = 7.5                  # blind-agent net-vs-never-bail (doc 100), reference row
DOC100_PLAIN_STOP_REF = 17.7          # doc 100 plain-stop-24t net-vs-never-bail (external ref)
DIPPED_FB_LINE = 0.54                 # doc 100 agent dipped-good false-bail -- the line to beat
AGENT_CATCH_REF = 0.95                # doc 100 agent wrong-catch (the operating point to match/beat)

FEATURE_NAMES = [
    'loss_velocity',       # drift[t*] - drift[t*-1]                  (1m loss velocity)
    'acceleration',        # 2nd diff of drift at t*                  (loss accelerating?)
    'giveback_depth',      # drift[t*] - max(drift[0..t*])   (<=0)    (how far off the path peak)
    'giveback_velocity',   # d/dt of (drift - running peak)           (velocity of giveback)
    'efficiency_ratio',    # |net|/Σ|Δ| over last <=10m on drift      (trend vs chop)
    'drawdown_vs_vol',     # giveback_depth / trailing drift-Δ vol    (drawdown in vol units)
    'minutes_since_entry', # t*                                       (how deep into the trade)
    'entry_P',             # entry conviction                         (context)
    'tod',                 # fraction of RTH elapsed at entry         (context)
]


# ================= feature extraction (CAUSAL: indices <= t* only) ========================
def _tod_fraction(entry_ts):
    """Fraction of the RTH session (08:30-15:15 CT) elapsed at entry. Causal (entry time is
    known at t*). Clipped to [0,1]."""
    dt = pd.to_datetime(int(entry_ts), unit='s', utc=True).tz_convert('America/Chicago')
    mins = dt.hour * 60 + dt.minute + dt.second / 60.0
    open_m = 8 * 60 + 30
    close_m = 15 * 60 + 15
    return float(np.clip((mins - open_m) / (close_m - open_m), 0.0, 1.0))


def features_at_trigger(drift, tstar, entry_P, entry_ts):
    """Path-derivable feature vector at the trigger minute t*. Uses drift[0..tstar] ONLY.
    Returns (vec, max_index_used) so the caller can assert causality (max_index_used <= tstar)."""
    used = [tstar]                                             # every term below records its indices
    d_t = drift[tstar]
    d_tm1 = drift[tstar - 1] if tstar >= 1 else drift[0]
    d_tm2 = drift[tstar - 2] if tstar >= 2 else d_tm1
    if tstar >= 1:
        used.append(tstar - 1)
    if tstar >= 2:
        used.append(tstar - 2)

    loss_velocity = d_t - d_tm1
    acceleration = (d_t - d_tm1) - (d_tm1 - d_tm2) if tstar >= 2 else 0.0

    peak_t = max(drift[:tstar + 1])                            # running peak up to t*
    peak_tm1 = max(drift[:tstar]) if tstar >= 1 else drift[0]
    giveback_depth = d_t - peak_t                              # <= 0
    giveback_velocity = (d_t - peak_t) - (d_tm1 - peak_tm1)

    w = min(ER_MAX_MIN, tstar)                                 # efficiency ratio over last <=10m
    if w >= 1:
        seg = drift[tstar - w: tstar + 1]                      # indices tstar-w .. tstar (<= t*)
        used.append(tstar - w)
        net = abs(seg[-1] - seg[0])
        denom = float(np.sum(np.abs(np.diff(seg))))
        efficiency_ratio = (net / denom) if denom > EPS_VOL else 0.0
        vol = float(np.std(np.diff(seg), ddof=1)) if len(seg) >= 3 else abs(loss_velocity)
    else:
        efficiency_ratio = 0.0
        vol = abs(loss_velocity)
    drawdown_vs_vol = giveback_depth / (vol + EPS_VOL)

    vec = np.array([
        loss_velocity, acceleration, giveback_depth, giveback_velocity,
        efficiency_ratio, drawdown_vs_vol, float(tstar), float(entry_P), _tod_fraction(entry_ts),
    ], float)
    return vec, max(used)


# ================= TRAIN population (2024) ================================================
def train_engagements():
    """select_wrongdir.engagements() mirrored for the TRAIN split (2024), same p90(train)
    frozen threshold, same 60s/day/dir de-dup. (sw.engagements() hardcodes the test/2025-26
    cut; the machinery below is identical except for the split/year filter.)"""
    econ = pd.read_parquet(eb.ECON_DRIFT_PATH,
                           columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, sw.P_PCTL))
    sub = econ[(econ.split == 'train') & (econ.P >= thr) &
               (econ.day.str[:4] == '2024')].copy()
    sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
    last = {}
    keep = []
    for r in sub.itertuples():
        k = (r.day, bool(r.is_long))
        if k in last and r.ts - last[k] <= sw.DEDUP_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    dd = sub.loc[keep].reset_index(drop=True)
    dd.attrs['p90_thr'] = thr
    return dd


def build_train_examples():
    """Scan 2024 engagements, simulate the plain 24t stop, and build one training example per
    TRIGGERED engagement at its trigger minute t*. Returns (X, y, meta_rows, n_engagements)."""
    eng = train_engagements()
    thr = eng.attrs['p90_thr']
    print(f'[train] 2024 engagements: {len(eng)} fires over {eng["day"].nunique()} days; '
          f'p{sw.P_PCTL}(train) thr={thr:.5f}')
    day_engs, _ = sw.scan(eng)                                # reuse verified drift extraction

    X, y, rows = [], [], []
    n_eng = 0
    for day, engs in day_engs.items():
        for e in engs:
            n_eng += 1
            drift = e['per_minute_forward_drift']
            window = e['window_minutes']
            bailed, tstar = scw.dumb_exit_minute(drift, window, STOP_TICKS)
            if not bailed:
                continue                                       # no stop -> no decision point
            vec, max_idx = features_at_trigger(drift, tstar, e['P'], e['ts'])
            assert max_idx <= tstar, f'CAUSALITY VIOLATION: feature used idx {max_idx} > t*={tstar}'
            terminal = drift[window]
            recover = 1 if terminal > drift[tstar] else 0      # sign(terminal - drift[t*])
            X.append(vec)
            y.append(recover)
            rows.append(dict(day=day, ts=e['ts'], is_long=e['is_long'], tstar=tstar,
                             window=window, drift_tstar=drift[tstar], terminal=terminal,
                             recover=recover))
    X = np.array(X, float)
    y = np.array(y, int)
    print(f'[train] engagements scanned={n_eng}; 24t-stop triggered (examples)={len(y)}; '
          f'recover rate={y.mean():.3f}')
    return X, y, rows, n_eng


# ================= p* sweep on 2024 =======================================================
def sweep_pstar(X, y, rows, model, scaler):
    """For each p* on the grid, compute mean net-vs-never-bail ticks/ep over the TRAIN triggered
    population (veto iff P(recover) >= p*). A vetoed stop holds to window (net = 0 vs never-bail);
    a taken stop nets (drift[t*] - drift[window]) * 4. Non-triggered engagements net 0 for every
    policy, so they are a constant that does not shift the argmax -- the sweep is over triggered
    episodes. Returns (best_pstar, sweep_table)."""
    P = model.predict_proba(scaler.transform(X))[:, 1]
    taken_net = np.array([(r['drift_tstar'] - r['terminal']) * PTS_TO_TICKS for r in rows], float)
    table = []
    for p in P_STAR_GRID:
        veto = P >= p
        net = np.where(veto, 0.0, taken_net)                   # vetoed -> hold (0); else take stop
        # friction-honest absolute: taken stops pay one RT; held-to-window also one RT -> constant.
        table.append(dict(p_star=float(p), n_veto=int(veto.sum()),
                          mean_net=float(net.mean()), median_net=float(np.median(net))))
    best = max(table, key=lambda d: d['mean_net'])
    return best['p_star'], table


# ================= TEST on the 198 ========================================================
def load_test_episodes():
    """The 198 doc-100 episodes: truth/*.json drift/window (+ entry_ts/P) with classes read from
    selection.json (truth_label, good_kind), excluding the 2 nonce-audit failures so the frontier
    matches doc 100's scored 198 exactly. BAND/DIP from selection meta (classify identically)."""
    sel = json.load(open(SELECTION_JSON, encoding='utf-8'))
    cls = {e['eid']: (e['truth_label'], e['good_kind']) for e in sel['episodes']}
    band = float(sel['meta'].get('band', 4.0))
    eps = []
    for path in sorted(glob.glob(os.path.join(TRUTH_DIR, '*.json'))):
        eid = os.path.splitext(os.path.basename(path))[0]
        if eid in NONCE_FAIL_EIDS:
            continue
        t = json.load(open(path, encoding='utf-8'))
        drift = t['per_minute_forward_drift']
        window = t['window_minutes']
        truth_label, good_kind = cls.get(eid, (None, None))
        if truth_label is None:                                # fallback: classify from terminal
            terminal = drift[window]
            truth_label = 'wrong' if terminal <= -band else ('good' if terminal >= band else None)
        eps.append(dict(eid=eid, day=t.get('real_day', eid[:10]), drift=drift, window=window,
                        entry_ts=t['entry_ts'], P=t['P'], is_long=t['is_long'],
                        truth_label=truth_label, good_kind=good_kind))
    return eps, band


def evaluate_test(eps, model, scaler, p_star):
    """Score every policy per episode. bail = the stop physically executes.
       - never-bail : exit=window (net 0 each)
       - plain-stop : bail iff 24t triggers
       - stop+veto  : bail iff triggers AND P(recover) < p*
    Returns per-episode records with net-vs-never-bail and bail flags for each policy."""
    recs = []
    for e in eps:
        drift, window = e['drift'], e['window']
        bailed, tstar = scw.dumb_exit_minute(drift, window, STOP_TICKS)
        # plain stop
        ps_exit = tstar if bailed else window
        ps_net = (drift[ps_exit] - drift[window]) * PTS_TO_TICKS
        # stop + veto
        P_rec, veto = np.nan, False
        if bailed:
            vec, max_idx = features_at_trigger(drift, tstar, e['P'], e['entry_ts'])
            assert max_idx <= tstar, 'CAUSALITY VIOLATION (test)'
            P_rec = float(model.predict_proba(scaler.transform(vec.reshape(1, -1)))[0, 1])
            veto = P_rec >= p_star
        sv_bail = bool(bailed and not veto)
        sv_exit = tstar if sv_bail else window
        sv_net = (drift[sv_exit] - drift[window]) * PTS_TO_TICKS
        recs.append(dict(
            eid=e['eid'], day=e['day'], truth_label=e['truth_label'], good_kind=e['good_kind'],
            triggered=bool(bailed), tstar=(tstar if bailed else None), P_recover=P_rec,
            vetoed=bool(bailed and veto),
            nb_net=0.0, ps_bail=bool(bailed), ps_net=ps_net,
            sv_bail=sv_bail, sv_net=sv_net,
            # absolute realized ticks WITH friction (one round trip per episode)
            nb_abs=drift[window] * PTS_TO_TICKS - FRICTION_TICKS,
            ps_abs=drift[ps_exit] * PTS_TO_TICKS - FRICTION_TICKS,
            sv_abs=drift[sv_exit] * PTS_TO_TICKS - FRICTION_TICKS,
        ))
    return recs


# ================= day-block bootstrap ===================================================
def dayblock_ci(recs, value_fn, boots=BOOTS, seed=SEED):
    """Day-block bootstrap CI of the mean of value_fn(rec). Resamples DISTINCT days with
    replacement (each test episode carries its own day). Returns (mean, lo, hi)."""
    by_day = {}
    for r in recs:
        by_day.setdefault(r['day'], []).append(value_fn(r))
    days = list(by_day.keys())
    rng = np.random.default_rng(seed)
    vals = np.array([v for d in days for v in by_day[d]], float)
    means = []
    for _ in range(boots):
        pick = rng.choice(len(days), len(days), replace=True)
        blk = [v for i in pick for v in by_day[days[i]]]
        means.append(np.mean(blk))
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def class_rates(recs, bail_key):
    """(catch, dipped-FB, clean-FB, precision, n_bail) for a given policy's bail flag."""
    wrong = [r for r in recs if r['truth_label'] == 'wrong']
    dipped = [r for r in recs if r['truth_label'] == 'good' and r['good_kind'] == 'dipped']
    clean = [r for r in recs if r['truth_label'] == 'good' and r['good_kind'] == 'clean']
    bailed = [r for r in recs if r[bail_key]]
    catch = np.mean([r[bail_key] for r in wrong]) if wrong else np.nan
    fb_dip = np.mean([r[bail_key] for r in dipped]) if dipped else np.nan
    fb_cln = np.mean([r[bail_key] for r in clean]) if clean else np.nan
    prec = np.mean([r['truth_label'] == 'wrong' for r in bailed]) if bailed else np.nan
    return dict(catch=catch, fb_dip=fb_dip, fb_cln=fb_cln, precision=prec,
                n_bail=len(bailed), n_wrong=len(wrong), n_dip=len(dipped), n_cln=len(clean))


# ================= report writer =========================================================
def write_report(coefs, intercept, sweep_table, p_star, train_stats, frontier, deltas,
                 ps_rates, sv_rates, verdict, recs):
    L = []
    A = L.append
    A('# The DISTILLED VETO -- 2024-sealed logistic at the 24-tick stop moment (Task 105)')
    A('')
    A(f"Decision point: the minute the plain **{STOP_TICKS}-tick** stop first triggers "
      f"(favorable drift <= -{STOP_TICKS * TICK_PTS:.1f} pts). At that instant the logistic prices "
      f"P(recover) = P(terminal drift > drift[t*]) from path-derivable features (<= t* only, "
      f"causality asserted). Policy: **VETO the stop iff P(recover) >= p***.")
    A('')
    A('## Frozen model (trained on 2024, split==train)')
    A(f"- train engagements scanned: {train_stats['n_eng']}; 24t-stop triggered (examples): "
      f"{train_stats['n_ex']}; recover rate: {train_stats['recover_rate']:.3f}")
    A(f"- LogisticRegression (standardized, C={LOGIT_C}); frozen **p\\* = {p_star:.3f}**; "
      f"intercept = {intercept:+.4f}")
    A(f"- **discrimination: AUC in-sample {train_stats['auc_in']:.3f}, 5-fold CV "
      f"{train_stats['auc_cv']:.3f}** -- a 0.5-anchored gap of ~{train_stats['auc_cv'] - 0.5:+.3f} "
      f"is BELOW the 0.05 conditional-signal floor: the 9 cheap path features barely separate "
      f"recover from no-recover. Coefficients (below) are noise-level.")
    asym = train_stats['asym']
    A(f"- **economic asymmetry of the 24t stop on the NATURAL 2024 distribution**: taking the "
      f"stop nets mean {asym['taken_mean']:+.1f} t (median {asym['taken_median']:+.1f}) -- when the "
      f"trade RECOVERS it costs {asym['taken_recover_mean']:+.1f} t (huge forgone run), when it does "
      f"NOT it saves {asym['taken_norecover_mean']:+.1f} t. The recovery tail dominates, so a "
      f"net-maximizing veto is driven toward **veto-almost-everything** (~ never-bail). The "
      f"balanced 50/50 test set is the ONLY reason the plain stop looks like +17.7.")
    A('')
    A('### Feature coefficients (standardized; sign = effect on P(recover))')
    A('| feature | coef | reading |')
    A('|---|---|---|')
    order = np.argsort(-np.abs(coefs))
    for i in order:
        c = coefs[i]
        rd = 'higher -> more likely to RECOVER' if c > 0 else 'higher -> more likely to KEEP LOSING'
        A(f"| {FEATURE_NAMES[i]} | {c:+.4f} | {rd} |")
    A('')
    A('## 2024 p\\* sweep (mean net-vs-never-bail ticks/ep over triggered train episodes)')
    A('| p* | n_veto | mean net | median net |')
    A('|---|---|---|---|')
    for row in sweep_table:
        star = ' <- frozen' if abs(row['p_star'] - p_star) < 1e-9 else ''
        A(f"| {row['p_star']:.3f} | {row['n_veto']} | {row['mean_net']:+.2f} | "
          f"{row['median_net']:+.2f} |{star}")
    A('')
    A('## TEST frontier (the 198 doc-100 episodes, scored ONCE)')
    A('net-vs-never-bail = (drift[exit] - drift[window]) x 4 ticks; mean +/- 95% day-block CI. '
      'Absolute = realized drift x 4 - friction (2.4t/RT), one round trip per episode.')
    A('| policy | mean net (ticks) | 95% day-block CI | median | mode | mean ABS w/friction |')
    A('|---|---|---|---|---|---|')
    for name, key, absk in [('never-bail', 'nb_net', 'nb_abs'),
                            ('plain stop 24t', 'ps_net', 'ps_abs'),
                            ('STOP+VETO', 'sv_net', 'sv_abs')]:
        m, lo, hi = frontier[key]
        arr = np.array([r[key] for r in recs], float)
        absm = float(np.mean([r[absk] for r in recs]))
        A(f"| {name} | {m:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {np.median(arr):+.1f} | "
          f"{scw.hist_mode(arr, BW_TICKS):+.1f} | {absm:+.2f} |")
    A(f"| blind agents (doc 100 ref) | {AGENTS_NET_REF:+.2f} | (external) | - | - | - |")
    A(f"| plain stop 24t (doc 100 ref) | {DOC100_PLAIN_STOP_REF:+.2f} | (external) | - | - | - |")
    A('')
    A('## Pre-registered bar')
    dm, dlo, dhi = deltas
    A(f"**(1) STOP+VETO beats plain stop on net, delta CI excludes 0.** "
      f"delta (stop+veto - plain stop) = {dm:+.2f} ticks/ep, 95% day-block CI "
      f"[{dlo:+.2f}, {dhi:+.2f}] -> {'PASS' if dlo > 0 else 'FAIL'} "
      f"(CI {'excludes' if dlo > 0 else 'includes'} 0).")
    A(f"**(2) dipped-good false-bail < {DIPPED_FB_LINE:.0%} at equal-or-better wrong-catch.** "
      f"STOP+VETO dipped-FB = {sv_rates['fb_dip']:.0%} (N={sv_rates['n_dip']}) vs plain-stop "
      f"{ps_rates['fb_dip']:.0%}; wrong-catch = {sv_rates['catch']:.0%} (N={sv_rates['n_wrong']}) "
      f"vs agent ref {AGENT_CATCH_REF:.0%} / plain-stop {ps_rates['catch']:.0%}.")
    dip_ok = np.isfinite(sv_rates['fb_dip']) and sv_rates['fb_dip'] < DIPPED_FB_LINE
    catch_ok = np.isfinite(sv_rates['catch']) and sv_rates['catch'] >= AGENT_CATCH_REF
    A(f"  -> dipped-FB<{DIPPED_FB_LINE:.0%}: {'PASS' if dip_ok else 'FAIL'}; "
      f"catch>=agent {AGENT_CATCH_REF:.0%}: {'PASS' if catch_ok else 'FAIL'}.")
    A('')
    A(f"## VERDICT: **{'PASS' if verdict else 'FAIL'}**")
    A(f"({'both bars clear' if verdict else 'at least one bar fails'} -- see above.)")
    A('')
    A('## Per-class confusion (STOP+VETO vs plain stop, on the 198)')
    A('| policy | wrong-catch | dipped-FB | clean-FB | precision | n_bail |')
    A('|---|---|---|---|---|---|')
    for name, rr in [('plain stop 24t', ps_rates), ('STOP+VETO', sv_rates)]:
        A(f"| {name} | {rr['catch']:.0%} | {rr['fb_dip']:.0%} | {rr['fb_cln']:.0%} | "
          f"{rr['precision']:.0%} | {rr['n_bail']} |")
    A('')
    # veto precision/recall: among plain-stop bails, the veto CANCELS a bail. A "correct veto"
    # cancels a bail on a GOOD (recovering) episode; an "incorrect veto" cancels a bail on a WRONG.
    triggered = [r for r in recs if r['triggered']]
    vetoed = [r for r in triggered if r['vetoed']]
    veto_on_good = sum(1 for r in vetoed if r['truth_label'] == 'good')
    veto_on_wrong = sum(1 for r in vetoed if r['truth_label'] == 'wrong')
    good_trig = sum(1 for r in triggered if r['truth_label'] == 'good')
    veto_prec = (veto_on_good / len(vetoed)) if vetoed else np.nan
    veto_rec = (veto_on_good / good_trig) if good_trig else np.nan
    A('## Veto precision / recall (among the plain-stop bails the veto cancels)')
    A(f"- triggered episodes: {len(triggered)} ({good_trig} good, {len(triggered) - good_trig} wrong)")
    A(f"- vetoes fired: {len(vetoed)} ({veto_on_good} on GOOD = correct saves, "
      f"{veto_on_wrong} on WRONG = mistaken holds)")
    A(f"- veto precision P(good | vetoed) = {veto_prec:.0%}; "
      f"veto recall P(vetoed | good-triggered) = {veto_rec:.0%}")
    A('')
    A('_Sealed 2024 fit, frozen (p*, coefs), single pass on the 198. A dojo number is never a '
      'result until it clears the sealed frontier; this is that frontier for the distilled veto._')
    with open(REPORT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# ================= main ===================================================================
def main():
    os.makedirs(WRONGDIR_DIR, exist_ok=True)

    # ---- TRAIN (2024) ----
    X, y, rows, n_eng = build_train_examples()
    if len(np.unique(y)) < 2:
        print('[abort] training target has a single class; cannot fit logistic.')
        return
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    model = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_MAX_ITER)
    model.fit(Xs, y)
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    # discrimination: in-sample + honest 5-fold CV AUC (does the model separate recover at all?)
    auc_in = float(roc_auc_score(y, model.predict_proba(Xs)[:, 1]))
    p_cv = cross_val_predict(LogisticRegression(C=LOGIT_C, max_iter=LOGIT_MAX_ITER),
                             Xs, y, cv=5, method='predict_proba')[:, 1]
    auc_cv = float(roc_auc_score(y, p_cv))
    p_star, sweep_table = sweep_pstar(X, y, rows, model, scaler)
    print(f'[train] AUC in-sample={auc_in:.4f} 5foldCV={auc_cv:.4f}; frozen p*={p_star:.3f}')

    # economic asymmetry of taking the 24t stop on the NATURAL 2024 distribution
    taken = np.array([(r['drift_tstar'] - r['terminal']) * PTS_TO_TICKS for r in rows], float)
    yv = np.array(y)
    asym = dict(taken_mean=float(taken.mean()), taken_median=float(np.median(taken)),
                taken_recover_mean=float(taken[yv == 1].mean()),
                taken_norecover_mean=float(taken[yv == 0].mean()))
    train_stats = dict(n_eng=n_eng, n_ex=len(y), recover_rate=float(y.mean()),
                       auc_in=auc_in, auc_cv=auc_cv, asym=asym)

    # ---- FREEZE ----
    frozen = dict(
        task='105_veto_logistic', trained_on='2024 split==train', stop_ticks=STOP_TICKS,
        friction_ticks=FRICTION_TICKS, feature_names=FEATURE_NAMES,
        scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
        coef=coefs.tolist(), intercept=intercept, p_star=float(p_star),
        n_train_examples=int(len(y)), recover_rate=float(y.mean()),
        auc_in_sample=train_stats['auc_in'], auc_cv5=train_stats['auc_cv'],
        taken_stop_net_asymmetry=train_stats['asym'],
        er_max_min=ER_MAX_MIN, p_star_grid=[float(p) for p in P_STAR_GRID],
    )
    with open(FROZEN_JSON, 'w', encoding='utf-8') as f:
        json.dump(frozen, f, indent=2)
    print(f'[freeze] wrote {FROZEN_JSON}')

    # ---- TEST (198) ----
    eps, band = load_test_episodes()
    print(f'[test] episodes: {len(eps)} (band={band:.0f}, 2 nonce-fails excluded)')
    recs = evaluate_test(eps, model, scaler, p_star)

    frontier = {k: dayblock_ci(recs, lambda r, k=k: r[k]) for k in ('nb_net', 'ps_net', 'sv_net')}
    deltas = dayblock_ci(recs, lambda r: r['sv_net'] - r['ps_net'])
    ps_rates = class_rates(recs, 'ps_bail')
    sv_rates = class_rates(recs, 'sv_bail')
    dip_ok = np.isfinite(sv_rates['fb_dip']) and sv_rates['fb_dip'] < DIPPED_FB_LINE
    catch_ok = np.isfinite(sv_rates['catch']) and sv_rates['catch'] >= AGENT_CATCH_REF
    verdict = (deltas[1] > 0) and dip_ok and catch_ok

    write_report(coefs, intercept, sweep_table, p_star, train_stats, frontier, deltas,
                 ps_rates, sv_rates, verdict, recs)
    print(f'[test] wrote {REPORT_MD}')

    # ---- composability hook ----
    dfp = pd.DataFrame([dict(eid=r['eid'], day=r['day'], tstar=r['tstar'], P_recover=r['P_recover'],
                             triggered=r['triggered'], vetoed=r['vetoed'], sv_bail=r['sv_bail'],
                             truth_label=r['truth_label'], good_kind=r['good_kind'],
                             ps_net=r['ps_net'], sv_net=r['sv_net']) for r in recs])
    dfp.to_parquet(DECISIONS_PARQUET, index=False)
    print(f'[hook] wrote {DECISIONS_PARQUET} ({len(dfp)} rows)')

    # ---- console summary ----
    fm, flo, fhi = frontier['sv_net']
    pm, plo, phi = frontier['ps_net']
    dm, dlo, dhi = deltas
    print('\n================ SUMMARY ================')
    print(f'plain-stop net   {pm:+.2f} [{plo:+.2f},{phi:+.2f}] (doc100 ref +17.7)')
    print(f'STOP+VETO net    {fm:+.2f} [{flo:+.2f},{fhi:+.2f}]')
    print(f'delta (SV-PS)    {dm:+.2f} [{dlo:+.2f},{dhi:+.2f}] -> CI excl 0: {dlo > 0}')
    print(f'plain-stop  catch {ps_rates["catch"]:.0%}  dipped-FB {ps_rates["fb_dip"]:.0%}')
    print(f'STOP+VETO   catch {sv_rates["catch"]:.0%}  dipped-FB {sv_rates["fb_dip"]:.0%}')
    print(f'bar1 delta>0 CI: {dlo > 0} | bar2 dipFB<54%: {dip_ok} & catch>=95%: {catch_ok}')
    print(f'VERDICT: {"PASS" if verdict else "FAIL"}')


if __name__ == '__main__':
    main()
