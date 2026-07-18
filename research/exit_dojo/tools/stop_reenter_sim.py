"""
Wrong-Direction Dojo -- STOP + RE-ENTRY simulator (research/exit_dojo/tools/stop_reenter_sim.py)

Task 103 (reviewer, 2026-07-18). Moises' counter-proposal (verbatim): "the ups I exited
early mechanism you reenter but with a slightly worst position." Doc 100 showed a dumb
adverse-drawdown stop at 24t nets +17.7 t/ep, but its cost is concentrated in KNIFED
dipped-goods -- and that knife is catastrophic only because it is IRREVERSIBLE. This tool
tests whether making the knife REVERSIBLE (bail, then re-enter the same direction after a
confirmation margin) beats the plain stop.

MECHANISM (per-episode path math, on the 1m favorable-signed drift series):
  - Base stop: bail at the first minute where favorable-signed drift <= -X ticks
    (GLOBAL drift, measured from the ORIGINAL entry -- the doc-100 rule verbatim).
  - RE-ENTRY: after a bail at drift level `bail_level`, re-enter the SAME direction at the
    first later minute where drift RECOVERS to >= bail_level + M ticks (confirmation
    margin). The new leg then stops on the SAME global rule (drift <= -X). Cap re-entries
    per episode at B.
  - FRICTION: 2.4 ticks (0.6 pt MNQ round trip) charged on EVERY leg (original + each
    re-entry). Named constant FRICTION_TICKS.

ECONOMICS (net vs NEVER-BAIL, extends score_wrongdir.net_ticks_vs_neverbail):
  Let d[m] = drift[m] * 4  (favorable ticks from original entry; drift is in points).
  Sum over realized legs of (d[exit] - d[entry]) = total captured ticks.
  net = (sum_legs (d[exit]-d[entry]) - d[window]) - (n_legs - 1) * FRICTION_TICKS
  The original leg's round-trip friction CANCELS against never-bail's single round trip
  (both trade once); only the EXTRA re-entry legs pay incremental friction. Hence the
  plain-stop bar is UNCHANGED from its no-friction value (doc-100 +17.7) -- friction bites
  only re-entry. never-bail net = 0 by construction (floor).

SEALED PROTOCOL:
  1. TUNE on a 2024 population built with select_wrongdir machinery pointed at split='train'
     (engagements_train replica + the module's scan/allocate, same economic truth BAND=4,
     balanced one-per-day like the test set). Grid X in {8,16,24,32,48}, M in {4,8,16},
     B in {1,2}. Winner = max mean net ticks/ep. Plain-stop best-X ALSO frozen on 2024.
  2. FREEZE. Evaluate ONCE on the 198 doc-100 test episodes (reports/wrongdir/truth via
     score_wrongdir.score_episode -- same scored set, same BAND classification).
  3. Day-block bootstrap CI on the delta vs plain-stop best-X (pre-registered PASS/FAIL).

CAVEAT: 1m path resolution understates intrabar stop/trigger crossings; all numbers are
1m-granularity estimates (a real intrabar stop would fire earlier/deeper).

Run:  python3.11 research/exit_dojo/tools/stop_reenter_sim.py
Out:  research/exit_dojo/reports/wrongdir/stop_reenter.md
"""
import os
import sys
import glob
import json
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score_wrongdir as sw            # net helper, boot_ci, hist_mode, score_episode, dirs
import select_wrongdir as swl          # scan / allocate / per_day_candidates machinery
BUILDERS = os.path.abspath(os.path.join(HERE, '..', 'builders'))
sys.path.insert(0, BUILDERS)
import episode_builder as eb           # ECON_DRIFT_PATH, load_day_data, signed_drift_path

# ---- constants (house rule: no bare magic numbers) -----------------------------------
TICK_PTS = sw.TICK_PTS                 # 0.25 (MNQ tick, points)
PTS_TO_TICKS = sw.PTS_TO_TICKS         # 4.0
BW_TICKS = sw.BW_TICKS                 # 4 ticks = $2 mode bin
BOOTS = sw.BOOTS                       # 4000
SEED = sw.SEED                         # 12345
FRICTION_TICKS = 2.4                   # 0.6 pt MNQ round trip; charged per leg (task 103)
# sealed grid (task 103)
X_GRID = [8, 16, 24, 32, 48]           # adverse-drawdown stop, ticks
M_GRID = [4, 8, 16]                    # re-entry confirmation margin, ticks
B_GRID = [1, 2]                        # max re-entries per episode
TRAIN_SEED = swl.SELECTION_SEED        # 20260717 -- same seed the test selection used
N_PER_CLASS = swl.N_PER_CLASS          # 100 (balanced one-per-day, mirrors the test set)

OUT_MD = os.path.join(sw.WRONGDIR_DIR, 'stop_reenter.md')


# ================= core path simulators ==============================================
def simulate_plainstop(drift, window, x_ticks):
    """Plain adverse-drawdown stop (doc-100). Bail at first minute drift <= -X; net vs
    never-bail = (d[eff_exit]-d[window]). One round trip == never-bail's one -> friction
    cancels, so this equals score_wrongdir.net_ticks_vs_neverbail exactly."""
    bailed, eff_exit = sw.dumb_exit_minute(drift, window, x_ticks)
    net = sw.net_ticks_vs_neverbail(drift, window, eff_exit)
    return dict(net=net, bailed=bailed, eff_exit=eff_exit)


def simulate_reenter(drift, window, x_ticks, m_ticks, b_cap):
    """Stop + re-entry on the 1m drift path. Returns net ticks vs never-bail (with per-leg
    friction), leg/bail/re-entry counts, and per-re-entry give-up (margin actually crossed).

    legs: list of (entry_min, entry_drift, exit_min, exit_drift). n_bails = legs that ended
    on the stop. give-ups = drift[reenter]-drift[bail] in TICKS for each re-entry event."""
    x_pts = x_ticks * TICK_PTS
    m_pts = m_ticks * TICK_PTS
    legs = []
    giveups = []            # ticks the trade recovered before we re-entered (>= M by design)
    reenter_mins = []
    m = 0
    entry_level = drift[0]
    reentries = 0
    n_bails = 0
    while True:
        # find the stop from the current position: first later minute with drift <= -X
        bail_m = None
        for k in range(m + 1, window + 1):
            if drift[k] <= -x_pts:
                bail_m = k
                break
        if bail_m is None:                         # held to the window end
            legs.append((m, entry_level, window, drift[window]))
            break
        bail_level = drift[bail_m]
        legs.append((m, entry_level, bail_m, bail_level))
        n_bails += 1
        if reentries >= b_cap:                     # re-entry budget exhausted -> stay flat
            break
        # re-entry: first later minute recovering to >= bail_level + M
        reenter_m = None
        for r in range(bail_m + 1, window + 1):
            if drift[r] >= bail_level + m_pts:
                reenter_m = r
                break
        if reenter_m is None:                      # never recovered -> stay flat to end
            break
        giveups.append((drift[reenter_m] - bail_level) * PTS_TO_TICKS)
        reenter_mins.append(reenter_m)
        reentries += 1
        m = reenter_m
        entry_level = drift[reenter_m]

    n_legs = len(legs)
    captured = sum(ex - en for (_, en, _, ex) in legs) * PTS_TO_TICKS
    net = captured - drift[window] * PTS_TO_TICKS - (n_legs - 1) * FRICTION_TICKS
    return dict(net=net, n_legs=n_legs, n_bails=n_bails, n_reentries=reentries,
                giveups=giveups, reenter_mins=reenter_mins,
                friction_paid=(n_legs - 1) * FRICTION_TICKS)


# ================= 2024 tuning population =============================================
def engagements_train():
    """phold engagements() replica pointed at split=='train' (== 2024). Same p90(train-P)
    threshold, same 60s/day/dir de-dup. (select_wrongdir.engagements hardcodes test/2025-26;
    this is the train twin -- identical logic, disjoint years, no leakage.)"""
    econ = pd.read_parquet(eb.ECON_DRIFT_PATH,
                           columns=['ts', 'day', 'det', 'is_long', 'P', 'split'])
    thr = float(np.percentile(econ.loc[econ.split == 'train', 'P'].values, swl.P_PCTL))
    sub = econ[(econ.split == 'train') & (econ.P >= thr) &
               (econ.day.str[:4] == '2024')].copy()
    sub = sub.sort_values(['day', 'is_long', 'ts', 'det']).reset_index(drop=True)
    last = {}
    keep = []
    for r in sub.itertuples():
        k = (r.day, bool(r.is_long))
        if k in last and r.ts - last[k] <= swl.DEDUP_S:
            continue
        last[k] = r.ts
        keep.append(r.Index)
    dd = sub.loc[keep].reset_index(drop=True)
    dd.attrs['p90_thr'] = thr
    return dd


def build_train_population(band):
    """Scan 2024 train engagements -> balanced one-per-day WRONG/GOOD selection (same
    scan/allocate machinery as the test cut). Returns list of dicts with drift + class."""
    eng = engagements_train()
    print(f'[train] engagements: {len(eng)} fires over {eng["day"].nunique()} 2024 days; '
          f'p{swl.P_PCTL} thr={eng.attrs["p90_thr"]:.5f}')
    day_engs, _terminals = swl.scan(eng)
    days = sorted(day_engs.keys())
    selected, wrong, good = swl.allocate(day_engs, days, band, swl.DIP_PTS,
                                         TRAIN_SEED, N_PER_CLASS)
    pop = []
    for s in selected:
        drift = s['per_minute_forward_drift']
        window = s['window_minutes']
        terminal = s['terminal_drift']
        is_wrong = (terminal <= -band)
        good_kind = None if is_wrong else s['good_kind']
        pop.append(dict(day=s['day'], drift=drift, window=window, terminal=terminal,
                        is_wrong=is_wrong, good_kind=good_kind))
    n_w = sum(1 for p in pop if p['is_wrong'])
    print(f'[train] population: {len(pop)} eps ({n_w} wrong / {len(pop)-n_w} good, '
          f'dipped {sum(1 for p in pop if p["good_kind"]=="dipped")}/'
          f'clean {sum(1 for p in pop if p["good_kind"]=="clean")}); '
          f'distinct days={len({p["day"] for p in pop})}')
    return pop


# ================= test population (the doc-100 198) =================================
def load_test_population(band, dip):
    truth_files = sorted(glob.glob(os.path.join(sw.TRUTH_DIR, '*.json')))
    eids = [os.path.splitext(os.path.basename(p))[0] for p in truth_files]
    played = [e for e in eids
              if os.path.exists(os.path.join(sw.GATE_STATE_DIR, f'{e}.transcript.jsonl'))]
    rows = [r for r in (sw.score_episode(e, band, dip) for e in played) if r is not None]
    scored = [r for r in rows if r.get('scored')]
    pop = []
    for r in scored:
        pop.append(dict(day=r['day'], drift=r['drift'], window=r['window'],
                        terminal=r['terminal'], is_wrong=r['is_wrong'],
                        good_kind=r['good_kind']))
    return pop


# ================= grid search =======================================================
def grid_search(pop):
    """Full X x M x B grid on the tuning population. Returns rows sorted by mean net desc."""
    rows = []
    for x, m, b in itertools.product(X_GRID, M_GRID, B_GRID):
        nets, reentered, churn = [], 0, 0
        for p in pop:
            res = simulate_reenter(p['drift'], p['window'], x, m, b)
            nets.append(res['net'])
            if res['n_reentries'] > 0:
                reentered += 1
            if res['n_bails'] >= 2:
                churn += 1
        nets = np.array(nets, float)
        rows.append(dict(x=x, m=m, b=b, mean=float(nets.mean()),
                         median=float(np.median(nets)), mode=float(sw.hist_mode(nets, BW_TICKS)),
                         reentered=reentered, churn=churn, n=len(nets)))
    rows.sort(key=lambda d: -d['mean'])
    return rows


def plainstop_grid(pop):
    rows = []
    for x in X_GRID:
        nets = np.array([simulate_plainstop(p['drift'], p['window'], x)['net'] for p in pop], float)
        rows.append(dict(x=x, mean=float(nets.mean()), median=float(np.median(nets))))
    rows.sort(key=lambda d: -d['mean'])
    return rows


# ================= day-block bootstrap on a paired delta =============================
def dayblock_delta_ci(pairs, boots=BOOTS, seed=SEED):
    """pairs: list of (day, delta). Resample DAYS with replacement (day-block), average the
    deltas of the drawn days. Returns (mean_delta, lo, hi)."""
    by_day = defaultdict(list)
    for day, d in pairs:
        by_day[day].append(d)
    days = list(by_day.keys())
    day_arr = [np.array(by_day[dd], float) for dd in days]
    rng = np.random.default_rng(seed)
    means = np.empty(boots)
    ndays = len(days)
    for i in range(boots):
        idx = rng.integers(0, ndays, ndays)
        vals = np.concatenate([day_arr[j] for j in idx])
        means[i] = vals.mean()
    all_d = np.concatenate(day_arr)
    return float(all_d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ================= report ============================================================
def fmt_ci(m, lo, hi):
    sig = '' if (lo <= 0 <= hi) else ' *'
    return f'{m:+.1f} [95% CI {lo:+.1f},{hi:+.1f}]{sig}'


def main():
    band, dip = sw.load_band_dip()
    print(f'[cfg] BAND={band:.0f} DIP={dip:.0f} friction={FRICTION_TICKS}t/leg')

    # ---- 1. TUNE on 2024 ----
    train = build_train_population(band)
    grid = grid_search(train)
    ps_train = plainstop_grid(train)
    winner = grid[0]
    ps_best = ps_train[0]
    print(f'[freeze] re-entry winner X={winner["x"]} M={winner["m"]} B={winner["b"]} '
          f'-> train mean {winner["mean"]:+.1f} t/ep')
    print(f'[freeze] plain-stop best-X={ps_best["x"]} -> train mean {ps_best["mean"]:+.1f} t/ep')

    # ---- 2. FREEZE + evaluate ONCE on the 198 ----
    test = load_test_population(band, dip)
    Xw, Mw, Bw = winner['x'], winner['m'], winner['b']
    Xps = ps_best['x']

    re_rows, ps_frozen_net, ps_sameX_net, nb_net = [], [], [], []
    for p in test:
        r = simulate_reenter(p['drift'], p['window'], Xw, Mw, Bw)
        r.update(day=p['day'], is_wrong=p['is_wrong'], good_kind=p['good_kind'])
        re_rows.append(r)
        ps_frozen_net.append(simulate_plainstop(p['drift'], p['window'], Xps)['net'])
        ps_sameX_net.append(simulate_plainstop(p['drift'], p['window'], Xw)['net'])
        nb_net.append(0.0)
    re_net = np.array([r['net'] for r in re_rows], float)
    ps_frozen_net = np.array(ps_frozen_net, float)
    ps_sameX_net = np.array(ps_sameX_net, float)

    # also: plain-stop best-X chosen ON the 198 (report-only sensitivity)
    ps_test_grid = plainstop_grid(test)
    ps_test_best = ps_test_grid[0]

    # ---- 3. day-block bootstrap deltas ----
    days = [r['day'] for r in re_rows]
    d_vs_frozen = list(zip(days, re_net - ps_frozen_net))       # PRE-REGISTERED bar
    d_vs_sameX = list(zip(days, re_net - ps_sameX_net))         # mechanism isolation
    d_vs_nb = list(zip(days, re_net - np.array(nb_net)))
    m_fr, lo_fr, hi_fr = dayblock_delta_ci(d_vs_frozen)
    m_sx, lo_sx, hi_sx = dayblock_delta_ci(d_vs_sameX)
    m_nb, lo_nb, hi_nb = dayblock_delta_ci(d_vs_nb)
    passed = (m_fr > 0) and (lo_fr > 0)

    # ---- per-class breakdown + knife ----
    def cls_net(rows, arr, pred):
        idx = [i for i, r in enumerate(rows) if pred(r)]
        return np.array([arr[i] for i in idx], float), idx
    classes = [('WRONG', lambda r: r['is_wrong']),
               ('GOOD-dipped', lambda r: (not r['is_wrong']) and r['good_kind'] == 'dipped'),
               ('GOOD-clean', lambda r: (not r['is_wrong']) and r['good_kind'] == 'clean')]

    # chop-churn under B=2 (recompute frozen X/M at B=2 to expose churn regardless of Bw)
    churn_rows = [simulate_reenter(p['drift'], p['window'], Xw, Mw, 2) for p in test]
    churn_eps = [(test[i], churn_rows[i]) for i in range(len(test)) if churn_rows[i]['n_bails'] >= 2]

    # ---- write report ----
    L = []
    A = L.append
    A('# Stop + Re-entry sim (task 103) -- Moises\' "oops, re-enter" mechanism')
    A('')
    A('Mechanism: adverse-drawdown stop at X ticks (bail when favorable-signed drift <= -X, '
      'measured from the ORIGINAL entry -- the doc-100 rule), then RE-ENTER the same '
      'direction when drift recovers to >= bail_level + M ticks; cap re-entries at B. '
      f'Friction = **{FRICTION_TICKS}t/leg** (0.6pt MNQ round trip) on the original AND every '
      're-entry leg. net vs never-bail extends score_wrongdir.net_ticks_vs_neverbail (drift '
      'in points x4 = ticks); the original round trip cancels against never-bail\'s one, so '
      'only re-entry legs pay incremental friction.')
    A('')
    A('> **1m-granularity caveat**: the drift series is per-minute. Intrabar stop/trigger '
      'crossings are invisible -- a real stop would fire earlier and often deeper, and a real '
      're-entry trigger could fire and reverse within a minute. ALL numbers below are '
      '1m-resolution estimates and are OPTIMISTIC about clean fills.')
    A('')
    A(f'BAND={band:.0f}pts (WRONG=terminal<=-{band:.0f}, GOOD>=+{band:.0f}); DIP={dip:.0f}pts. '
      f'Tuning = 2024 train split (disjoint from the 2025-26 test); test = the 198 doc-100 '
      f'scored episodes.')
    A('')

    # -- grid --
    A('## 1. Full 2024 tuning grid (X x M x B), sorted by mean net ticks/ep')
    A(f'Population: {len(train)} balanced one-per-day episodes '
      f'({sum(1 for p in train if p["is_wrong"])} wrong / '
      f'{sum(1 for p in train if not p["is_wrong"])} good), same select_wrongdir machinery.')
    A('')
    A('| X (t) | M (t) | B | mean net | median | mode | #re-entered | #churn(>=2 bails) |')
    A('|---|---|---|---|---|---|---|---|')
    for r in grid:
        star = '  <-- WINNER' if (r['x'], r['m'], r['b']) == (Xw, Mw, Bw) else ''
        A(f"| {r['x']} | {r['m']} | {r['b']} | {r['mean']:+.1f} | {r['median']:+.1f} | "
          f"{r['mode']:+.1f} | {r['reentered']} | {r['churn']} |{star}")
    A('')
    A('### Plain-stop grid on the SAME 2024 population (friction cancels -> = doc-100 convention)')
    A('| X (t) | mean net | median |')
    A('|---|---|---|')
    for r in ps_train:
        star = '  <-- best-X' if r['x'] == Xps else ''
        A(f"| {r['x']} | {r['mean']:+.1f} | {r['median']:+.1f} |{star}")
    A('')
    A(f'**FROZEN**: re-entry **X={Xw}, M={Mw}, B={Bw}** (2024 mean {winner["mean"]:+.1f} t/ep); '
      f'plain-stop best-X **X={Xps}** (2024 mean {ps_best["mean"]:+.1f} t/ep).')
    A('')

    # -- test verdict --
    A('## 2. FROZEN evaluation on the 198 test episodes (single shot)')
    A('')
    A('| policy | mean net (ticks/ep) | median | mode |')
    A('|---|---|---|---|')
    A(f'| never-bail (reference) | +0.0 | +0.0 | +0.0 |')
    A(f'| blind agents (doc-100) | +7.5 | +0.0 | +2.0 |')
    A(f'| plain-stop best-X (X={Xps}, frozen 2024) | {ps_frozen_net.mean():+.1f} | '
      f'{np.median(ps_frozen_net):+.1f} | {sw.hist_mode(ps_frozen_net, BW_TICKS):+.1f} |')
    A(f'| plain-stop same-X (X={Xw}) | {ps_sameX_net.mean():+.1f} | '
      f'{np.median(ps_sameX_net):+.1f} | {sw.hist_mode(ps_sameX_net, BW_TICKS):+.1f} |')
    A(f'| **stop+re-entry (X={Xw},M={Mw},B={Bw})** | **{re_net.mean():+.1f}** | '
      f'{np.median(re_net):+.1f} | {sw.hist_mode(re_net, BW_TICKS):+.1f} |')
    A('')
    A(f'Reference: doc-100 plain-stop best-X on the 198 = +17.7 @ X=24 (wider grid); on this '
      f'grid the 198 plain-stop best-X = {ps_test_best["mean"]:+.1f} @ X={ps_test_best["x"]} '
      f'(report-only; NOT the frozen bar).')
    A('')
    A('### Day-block bootstrap deltas (198 distinct days; 4000 resamples; * = CI excludes 0)')
    A(f'- **re-entry - plain-stop best-X (X={Xps}, frozen)** = {fmt_ci(m_fr, lo_fr, hi_fr)}  '
      f'<- PRE-REGISTERED BAR')
    A(f'- re-entry - plain-stop same-X (X={Xw})           = {fmt_ci(m_sx, lo_sx, hi_sx)}  '
      f'(isolates the re-entry add-on at fixed X)')
    A(f'- re-entry - never-bail                            = {fmt_ci(m_nb, lo_nb, hi_nb)}')
    A('')
    A(f'### PRE-REGISTERED VERDICT: **{"PASS" if passed else "FAIL"}**')
    A(f'Bar: stop+re-entry retained ONLY if test net > plain-stop best-X AND the delta CI '
      f'excludes 0. Delta = {m_fr:+.1f} t/ep, CI [{lo_fr:+.1f},{hi_fr:+.1f}]. '
      f'{"CI excludes 0 and delta positive -> PASS." if passed else "CI includes 0 (or delta <= 0) -> FAIL: re-entry does NOT beat the plain stop."}')
    A('')

    # -- per-class knife --
    A('## 3. Per-class breakdown -- the dipped-good knife, before vs after re-entry')
    A('| class | N | plain-stop (X={0}) mean | re-entry mean | delta | re-entry mode |'.format(Xps))
    A('|---|---|---|---|---|---|')
    for name, pred in classes:
        r_arr, idx = cls_net(re_rows, re_net, pred)
        ps_arr = np.array([ps_frozen_net[i] for i in idx], float)
        if len(r_arr) == 0:
            continue
        A(f"| {name} | {len(r_arr)} | {ps_arr.mean():+.1f} | {r_arr.mean():+.1f} | "
          f"{r_arr.mean()-ps_arr.mean():+.1f} | {sw.hist_mode(r_arr, BW_TICKS):+.1f} |")
    A('')
    dip_r, dip_idx = cls_net(re_rows, re_net, lambda r: (not r['is_wrong']) and r['good_kind'] == 'dipped')
    dip_ps = np.array([ps_frozen_net[i] for i in dip_idx], float)
    dip_reentered = sum(1 for i in dip_idx if re_rows[i]['n_reentries'] > 0)
    A(f'**The knife**: dipped-goods are the trades a plain stop bails at the dip then watches '
      f'run without them. Plain-stop (X={Xps}) nets {dip_ps.mean():+.1f} t/ep on the '
      f'{len(dip_idx)} dipped-goods; re-entry nets {dip_r.mean():+.1f} t/ep '
      f'({dip_reentered}/{len(dip_idx)} re-entered). '
      f'{"Re-entry recovers the knifed run (turns the irreversible cut into an M-tick+friction give-up)." if dip_r.mean() > dip_ps.mean() else "Re-entry does NOT repair the knife here."}')
    A('')

    # -- chop churn --
    A('## 4. Chop-churn cost (episodes with >= 2 bails under B=2)')
    n_churn = len(churn_eps)
    if n_churn:
        churn_net = np.array([simulate_reenter(p['drift'], p['window'], Xw, Mw, 2)['net']
                              for p, _ in churn_eps], float)
        w_ch = sum(1 for p, _ in churn_eps if p['is_wrong'])
        tot_giveup = np.array([sum(rr['giveups']) for _, rr in churn_eps], float)
        A(f'{n_churn}/{len(test)} episodes bail >= 2x at (X={Xw},M={Mw},B=2) '
          f'({w_ch} wrong / {n_churn-w_ch} good). These are the whipsaw payers: each extra '
          f'bail-and-re-enter cycle gives up its confirmation margin + friction.')
        A(f'- churn-episode net (B=2): mean {churn_net.mean():+.1f} | median '
          f'{np.median(churn_net):+.1f} | mode {sw.hist_mode(churn_net, BW_TICKS):+.1f} t/ep')
        A(f'- total give-up (sum of re-entry margins crossed) on churn eps: mean '
          f'{tot_giveup.mean():+.1f} t/ep')
    else:
        A(f'No episodes bail >= 2x at (X={Xw},M={Mw},B=2) -- no chop-churn on this test set.')
    A('')
    # give-up quantification across all re-entered test eps
    all_giveups = [g for r in re_rows for g in r['giveups']]
    n_reentered = sum(1 for r in re_rows if r['n_reentries'] > 0)
    if all_giveups:
        gu = np.array(all_giveups, float)
        A(f'**Give-up quantified** (all re-entries, frozen B={Bw}): {n_reentered}/{len(test)} '
          f'test eps re-entered, {len(gu)} re-entry events. Margin crossed on re-entry: mean '
          f'{gu.mean():+.1f} | median {np.median(gu):+.1f} | mode {sw.hist_mode(gu, BW_TICKS):+.1f} '
          f'ticks (>= M={Mw} by design); each also pays {FRICTION_TICKS}t friction. This is the '
          f'"slightly worse position" toll Moises described.')
    A('')

    # -- distribution mode-first --
    A('## 5. Distribution (mode-first) -- stop+re-entry net ticks/ep on the 198')
    lo_all, hi_all = sw.boot_ci(re_net)
    A(f'- mode **{sw.hist_mode(re_net, BW_TICKS):+.1f}** | median {np.median(re_net):+.1f} | '
      f'mean {re_net.mean():+.1f} [95% CI {lo_all:+.1f},{hi_all:+.1f}] ticks/ep (N=198).')
    # coarse histogram
    edges = np.arange(np.floor(re_net.min() / BW_TICKS) * BW_TICKS,
                      np.ceil(re_net.max() / BW_TICKS) * BW_TICKS + BW_TICKS, BW_TICKS)
    h, e = np.histogram(re_net, bins=edges)
    peak = max(h.max(), 1)
    A('```')
    for i in range(len(h)):
        if h[i] == 0:
            continue
        A(f"  [{e[i]:+6.0f},{e[i+1]:+6.0f}) {h[i]:4d} {'#' * max(1, int(round(40*h[i]/peak)))}")
    A('```')
    A('')
    A('_1m-granularity path sim on the sealed doc-100 test set; a dojo/path number is a '
      'hypothesis, not a live result -- any retained rule still graduates through the sealed '
      'harness (graduation firewall)._')

    os.makedirs(sw.WRONGDIR_DIR, exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'\nwrote {OUT_MD}')
    print(f'  WINNER X={Xw} M={Mw} B={Bw}: test net {re_net.mean():+.1f} vs plain-stop '
          f'best-X(X={Xps}) {ps_frozen_net.mean():+.1f}')
    print(f'  delta {m_fr:+.1f} CI[{lo_fr:+.1f},{hi_fr:+.1f}] -> {"PASS" if passed else "FAIL"}')


if __name__ == '__main__':
    main()
