"""
Wrong-Direction Dojo -- the POWERED cut frontier (Task 106, Opus drone, 2026-07-18)

PURE EVALUATION -- NOTHING is retuned. Re-runs the ENTIRE frozen cut-policy frontier on the
FULL 2025-26 test population (every select_wrongdir engagement, INCLUDING dead-band, NATURAL
class mix) so the cut question gets statistical power. Doc 105 showed the plain stop's
+17.7 t/ep on the 198 balanced (1:1) episodes was CI[-12.4,+46.7] -- not significant. The 198
was sized for LLM-fleet cost; mechanical policies are free, so evaluate on the whole tape.

POPULATION (frozen, reused verbatim):
  select_wrongdir.engagements() -> test split, P>=p90(train) frozen, 60s/day/dir de-dup, 2025-26.
  select_wrongdir.scan()        -> per-minute favorable-signed drift path per engagement
                                   (eb.signed_drift_path; the SAME machinery that produced the
                                   23,378-engagement terminal histogram). NO one-per-day dedup,
                                   NO 50/50 balance, dead-band INCLUDED -> deployment reality.
  Classification (BAND/DIP from selection.json meta -- BAND=4, DIP=4):
    WRONG        terminal <= -BAND
    GOOD-dipped  terminal >= +BAND and min drift <= -DIP
    GOOD-clean   terminal >= +BAND and min drift >  -DIP
    DEAD-BAND    |terminal| < BAND   (excluded from the balanced cut; kept HERE)

POLICIES (all frozen, no refit):
  never-bail;
  plain stop X in {8,16,24,32,48}   (grid pre-registered doc 103; each X reported = evaluation);
  stop+re-entry frozen X=48,M=4,B=1  (doc 103 winner; simulate_reenter verbatim);
  stop+veto frozen 24t, p*=0.45      (veto_frozen.json coefs/scaler applied VERBATIM at t*).

METRICS: net ticks/ep vs never-bail (mean, median, MODE-first) with day-block bootstrap CI
  (4000 resamples over DISTINCT test days -- covers within-day / overlapping-window dependence);
  ABSOLUTE net with friction 2.4t/RT; per-class decomposition; and the KEY delta columns:
  each policy's delta-vs-never-bail CI and delta-vs-best-stop CI.

FRICTION CONVENTION (charged consistently across ALL policies, documented once):
  net-vs-never-bail is friction-FREE: every single-round-trip policy trades exactly once, the
  2.4t/RT cancels against never-bail's one RT (doc 100/105 convention); stop+re-entry's EXTRA
  legs pay incremental friction inside simulate_reenter. The ABSOLUTE column re-adds friction
  honestly: abs = net_vs_neverbail + terminal_ticks - 2.4, an identity that charges exactly one
  RT for single-leg policies and n_legs RTs for re-entry (net already subtracted the n_legs-1
  extra). So 2.4t/RT is charged identically everywhere; it is a constant offset in every delta.

New files ONLY: this script + reports/wrongdir/powered_frontier.md. Reuses scan / classification
/ net / re-entry / veto-feature helpers by import; edits none of them. Commit NOTHING.

Run: python3.11 research/exit_dojo/tools/powered_cut_frontier.py
"""
import os
import sys
import json
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUILDERS = os.path.abspath(os.path.join(HERE, '..', 'builders'))
sys.path.insert(0, BUILDERS)

import select_wrongdir as swl          # engagements() + scan() -> per-minute drift paths
import stop_reenter_sim as srs         # simulate_plainstop / simulate_reenter / FRICTION_TICKS
import veto_logistic as vl             # features_at_trigger + STOP_TICKS (24t decision point)

sw = srs.sw                            # score_wrongdir (hist_mode, boot_ci, dumb_exit_minute, dirs)

# ---- constants (house rule: no bare magic numbers) -----------------------------------
PTS_TO_TICKS = sw.PTS_TO_TICKS         # 4.0
BW_TICKS = sw.BW_TICKS                 # 4 ticks = $2 mode bin
BOOTS = sw.BOOTS                       # 4000
SEED = sw.SEED                         # 12345
FRICTION_TICKS = srs.FRICTION_TICKS    # 2.4t / round trip
X_GRID = srs.X_GRID                    # [8,16,24,32,48] plain-stop grid (pre-registered doc 103)
REENTRY_X, REENTRY_M, REENTRY_B = 48, 4, 1     # frozen doc-103 re-entry winner
VETO_STOP_TICKS = vl.STOP_TICKS        # 24t trigger the veto prices
VETO_FROZEN_JSON = os.path.join(sw.WRONGDIR_DIR, 'veto_frozen.json')
SELECTION_JSON = os.path.join(sw.WRONGDIR_DIR, 'selection.json')
OUT_MD = os.path.join(sw.WRONGDIR_DIR, 'powered_frontier.md')

CLASSES = ['wrong', 'good_dipped', 'good_clean', 'dead_band']


# ================= frozen veto (applied VERBATIM -- no refit) =========================
class FrozenVeto:
    """Applies veto_frozen.json coefs/scaler/intercept + p* verbatim. veto iff P(recover)>=p*."""
    def __init__(self, path):
        j = json.load(open(path, encoding='utf-8'))
        self.mean = np.array(j['scaler_mean'], float)
        self.scale = np.array(j['scaler_scale'], float)
        self.coef = np.array(j['coef'], float)
        self.intercept = float(j['intercept'])
        self.p_star = float(j['p_star'])
        self.stop_ticks = int(j['stop_ticks'])
        assert self.stop_ticks == VETO_STOP_TICKS, 'frozen stop-ticks mismatch'

    def p_recover(self, drift, tstar, entry_P, entry_ts):
        vec, max_idx = vl.features_at_trigger(drift, tstar, entry_P, entry_ts)
        assert max_idx <= tstar, f'CAUSALITY VIOLATION: idx {max_idx} > t*={tstar}'
        z = (vec - self.mean) / self.scale
        logit = self.intercept + float(np.dot(self.coef, z))
        return 1.0 / (1.0 + np.exp(-logit))

    def bail(self, drift, window, entry_P, entry_ts):
        """(bailed, eff_exit) under stop+veto: 24t stop fires, veto cancels iff P(recover)>=p*."""
        triggered, tstar = sw.dumb_exit_minute(drift, window, self.stop_ticks)
        if not triggered:
            return False, window
        vetoed = self.p_recover(drift, tstar, entry_P, entry_ts) >= self.p_star
        return (not vetoed), (window if vetoed else tstar)


# ================= day-block bootstrap ================================================
def dayblock_ci(pairs, boots=BOOTS, seed=SEED):
    """pairs: (day, value). Resample DISTINCT days with replacement, average drawn days'
    values. Returns (mean, lo, hi). Covers within-day / overlapping-window dependence."""
    by_day = defaultdict(list)
    for day, v in pairs:
        by_day[day].append(v)
    days = list(by_day.keys())
    blocks = [np.array(by_day[d], float) for d in days]
    all_v = np.concatenate(blocks)
    rng = np.random.default_rng(seed)
    n = len(days)
    means = np.empty(boots)
    for i in range(boots):
        idx = rng.integers(0, n, n)
        means[i] = np.concatenate([blocks[j] for j in idx]).mean()
    return float(all_v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summ(days, nets):
    """mean/median/mode + day-block CI + mean-abs-w/friction for a net-vs-never-bail array."""
    nets = np.asarray(nets, float)
    m, lo, hi = dayblock_ci(list(zip(days, nets)))
    return dict(mean=m, lo=lo, hi=hi, median=float(np.median(nets)),
                mode=float(sw.hist_mode(nets, BW_TICKS)), n=len(nets))


def fmt_ci(m, lo, hi):
    sig = '' if (lo <= 0 <= hi) else ' *'
    return f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{sig}'


# ================= main ==============================================================
def main():
    os.makedirs(sw.WRONGDIR_DIR, exist_ok=True)
    meta = json.load(open(SELECTION_JSON, encoding='utf-8'))['meta']
    band = float(meta.get('band', 4.0))
    dip = float(meta.get('dip_pts', 4.0))
    print(f'[cfg] BAND={band:.0f} DIP={dip:.0f} friction={FRICTION_TICKS}t/RT '
          f'| plain-X grid {X_GRID} | re-entry X={REENTRY_X},M={REENTRY_M},B={REENTRY_B} '
          f'| veto {VETO_STOP_TICKS}t p*=0.45')

    # ---- FULL test population (no dedup, no balance, dead-band kept) ----
    eng = swl.engagements()
    print(f'[pop] engagements: {len(eng)} fires over {eng["day"].nunique()} test 2025-26 days; '
          f'p{swl.P_PCTL} thr={eng.attrs["p90_thr"]:.5f}')
    day_engs, terminals = swl.scan(eng)     # per-minute drift paths, verified machinery

    veto = FrozenVeto(VETO_FROZEN_JSON)

    # flatten + classify + apply every policy per engagement
    rows = []
    for day, engs in day_engs.items():
        for e in engs:
            drift = e['per_minute_forward_drift']
            window = e['window_minutes']
            terminal = e['terminal']
            mindrift = e['mindrift']
            term_ticks = terminal * PTS_TO_TICKS
            if terminal <= -band:
                cls = 'wrong'
            elif terminal >= band:
                cls = 'good_dipped' if mindrift <= -dip else 'good_clean'
            else:
                cls = 'dead_band'
            r = dict(day=day, cls=cls, term_ticks=term_ticks, window=window)
            # never-bail (floor)
            r['never_bail'] = 0.0
            # plain stops (net vs never-bail = (d[eff_exit]-d[window])*4)
            for x in X_GRID:
                r[f'stop{x}'] = srs.simulate_plainstop(drift, window, x)['net']
            # stop + re-entry (frozen 48/4/1) -- extra legs pay friction inside
            r['reentry'] = srs.simulate_reenter(drift, window, REENTRY_X, REENTRY_M, REENTRY_B)['net']
            # stop + veto (frozen 24t, p*=0.45) -- veto features applied verbatim
            vb, veff = veto.bail(drift, window, e['P'], e['ts'])
            r['veto'] = (drift[veff] - drift[window]) * PTS_TO_TICKS
            rows.append(r)

    N = len(rows)
    days_all = [r['day'] for r in rows]
    mix = {c: sum(1 for r in rows if r['cls'] == c) for c in CLASSES}
    n_days = len(set(days_all))
    print(f'[pop] N engagements = {N} over {n_days} distinct days')
    print(f'[mix] wrong={mix["wrong"]} good_dipped={mix["good_dipped"]} '
          f'good_clean={mix["good_clean"]} dead_band={mix["dead_band"]}')

    policies = [('never-bail', 'never_bail')] + \
               [(f'stop X={x}', f'stop{x}') for x in X_GRID] + \
               [(f'stop+re-entry (X={REENTRY_X},M={REENTRY_M},B={REENTRY_B})', 'reentry'),
                (f'stop+veto (24t, p*=0.45)', 'veto')]

    # per-policy summary + absolute-with-friction (abs = net + term_ticks - friction)
    term_arr = np.array([r['term_ticks'] for r in rows], float)
    stats = {}
    for name, key in policies:
        nets = np.array([r[key] for r in rows], float)
        s = summ(days_all, nets)
        s['abs_mean'] = float((nets + term_arr - FRICTION_TICKS).mean())
        stats[key] = s

    # best stop (highest mean net among the X grid) on THIS full population
    best_key = max((f'stop{x}' for x in X_GRID), key=lambda k: stats[k]['mean'])
    best_x = int(best_key[4:])
    best_nets = np.array([r[best_key] for r in rows], float)
    print(f'[best-stop] X={best_x} (mean net {stats[best_key]["mean"]:+.2f} t/ep)')

    # delta columns: vs never-bail (== the policy's own net CI) and vs best stop
    deltas = {}
    for name, key in policies:
        nets = np.array([r[key] for r in rows], float)
        d_nb = dayblock_ci(list(zip(days_all, nets - 0.0)))            # vs never-bail
        d_bs = dayblock_ci(list(zip(days_all, nets - best_nets)))      # vs best stop
        deltas[key] = dict(vs_nb=d_nb, vs_bs=d_bs)

    # per-class decomposition (mean net + CI per policy per class)
    per_class = {}
    for c in CLASSES:
        idx = [i for i, r in enumerate(rows) if r['cls'] == c]
        cdays = [rows[i]['day'] for i in idx]
        per_class[c] = {}
        for name, key in policies:
            arr = np.array([rows[i][key] for i in idx], float)
            if len(arr) == 0:
                per_class[c][key] = None
                continue
            m, lo, hi = dayblock_ci(list(zip(cdays, arr)))
            per_class[c][key] = dict(mean=m, lo=lo, hi=hi, n=len(arr),
                                     median=float(np.median(arr)))

    # ---- questions ----
    # (a) does ANY cut policy beat never-bail with CI excluding 0 at scale?
    beats_nb = [(name, key) for name, key in policies if key != 'never_bail'
                and deltas[key]['vs_nb'][1] > 0]     # lo > 0
    loses_nb = [(name, key) for name, key in policies if key != 'never_bail'
                and deltas[key]['vs_nb'][2] < 0]     # hi < 0
    # (b) does the doc-100 +17.7 stop edge survive natural mix + power? (24t here)
    s24 = stats['stop24']
    # (c) does re-entry change sign at natural mix?
    sre = stats['reentry']

    # ================= write report =================
    L = []
    A = L.append
    A('# The POWERED cut frontier -- FULL 2025-26 test population (Task 106)')
    A('')
    A('PURE EVALUATION. Nothing is retuned: every policy is frozen (plain-stop grid + re-entry '
      '48/4/1 + veto 24t p*=0.45 loaded verbatim from veto_frozen.json). The 198-episode dojo '
      'set was 1:1-balanced and one-per-day (sized for LLM-fleet cost); doc 105 showed the plain '
      'stop\'s +17.7 t/ep there was CI[-12.4,+46.7] -- NOT significant. Here the SAME frozen '
      'policies run on the WHOLE test tape (every engagement, natural class mix, dead-band '
      'included) so the cut question gets real statistical power.')
    A('')
    A('## Population + natural class mix')
    A(f'- **N = {N} engagements** over **{n_days} distinct test days** (2025-26; '
      f'select_wrongdir.engagements(): P>=p90(train)={eng.attrs["p90_thr"]:.5f} frozen, '
      f'60s/day/dir de-dup, MIN_WINDOW={swl.MIN_WINDOW_MIN}m). NO one-per-day dedup, NO 50/50 '
      f'balance -- this is the deployment tape.')
    A(f'- BAND={band:.0f}pts (WRONG terminal<=-{band:.0f}, GOOD terminal>=+{band:.0f}), '
      f'DIP={dip:.0f}pts (dipped = min drift <= -{dip:.0f}).')
    A('')
    A('| class | N | share |')
    A('|---|---|---|')
    for c in CLASSES:
        A(f'| {c} | {mix[c]} | {mix[c] / N:.1%} |')
    A(f'| **total** | **{N}** | 100% |')
    A('')
    A(f'The 198 dojo set forced 50/50 wrong/good with distinct days. The natural tape is '
      f'{mix["wrong"] / N:.0%} wrong, {(mix["good_dipped"] + mix["good_clean"]) / N:.0%} good '
      f'({mix["good_dipped"] / N:.0%} dipped / {mix["good_clean"] / N:.0%} clean), '
      f'{mix["dead_band"] / N:.0%} dead-band. Dipped-goods -- the knife the re-entry repairs -- '
      f'are {mix["good_dipped"] / N:.1%} of the tape here, vs 25% in the balanced set.')
    A('')

    A('## Friction convention (charged consistently across ALL policies)')
    A(f'net-vs-never-bail is friction-FREE: every single-round-trip policy trades exactly once, '
      f'so the {FRICTION_TICKS}t/RT cancels against never-bail\'s one RT (doc 100/105 convention). '
      f'stop+re-entry\'s EXTRA legs pay incremental friction inside the sim. The ABSOLUTE column '
      f're-adds it honestly via abs = net + terminal_ticks - {FRICTION_TICKS}, which charges '
      f'exactly one RT for single-leg policies and n_legs RTs for re-entry. So {FRICTION_TICKS}t/RT '
      f'is charged identically everywhere and is a constant offset in every delta.')
    A('')

    A('## THE FRONTIER (net ticks/ep vs never-bail; day-block 95% CI, 4000 resamples; * = CI excl 0)')
    A('| policy | mean net | 95% day-block CI | median | mode | mean ABS w/friction |')
    A('|---|---|---|---|---|---|')
    for name, key in policies:
        s = stats[key]
        star = '' if (s['lo'] <= 0 <= s['hi']) else ' *'
        A(f"| {name} | {s['mean']:+.2f}{star} | [{s['lo']:+.2f}, {s['hi']:+.2f}] | "
          f"{s['median']:+.1f} | {s['mode']:+.1f} | {s['abs_mean']:+.2f} |")
    A('')
    A(f'Best plain stop on this full population: **X={best_x}** '
      f'(mean net {stats[best_key]["mean"]:+.2f} t/ep).')
    A('')

    A('## Delta columns (day-block 95% CI; * = CI excludes 0)')
    A(f'| policy | delta vs never-bail | delta vs best stop (X={best_x}) |')
    A('|---|---|---|')
    for name, key in policies:
        if key == 'never_bail':
            continue
        dnb = deltas[key]['vs_nb']
        dbs = deltas[key]['vs_bs']
        A(f"| {name} | {fmt_ci(*dnb)} | {fmt_ci(*dbs)} |")
    A('')
    A('(delta-vs-never-bail equals the policy\'s own net since never-bail net==0 by construction; '
      'shown for completeness. delta-vs-best-stop isolates whether re-entry / veto add anything '
      'over the best plain stop.)')
    A('')

    A('## Per-class decomposition (mean net vs never-bail, ticks/ep; CI = day-block 95%)')
    A('| policy | ' + ' | '.join(
        f"{c} (N={mix[c]})" for c in CLASSES) + ' |')
    A('|---|' + '---|' * len(CLASSES))
    for name, key in policies:
        cells = []
        for c in CLASSES:
            pc = per_class[c][key]
            if pc is None:
                cells.append('-')
            else:
                sig = '' if (pc['lo'] <= 0 <= pc['hi']) else '*'
                cells.append(f"{pc['mean']:+.1f}{sig}")
        A(f"| {name} | " + ' | '.join(cells) + ' |')
    A('')
    A('(* = class-level CI excludes 0. WRONG: bail = money saved, net>0 expected. '
      'GOOD-dipped: bail = knifing a temporary dip, net<0 expected -- the trap. '
      'GOOD-clean: a stop rarely triggers, net~0. DEAD-BAND: near-scratch; whichever side '
      'the stop happens to catch.)')
    A('')

    A('## The three questions (plain answers)')
    A('')
    A('**(a) Does ANY cut policy beat never-bail with CI excluding 0 at scale?**')
    if beats_nb:
        A('- YES for: ' + ', '.join(
            f"{name} ({fmt_ci(*deltas[key]['vs_nb'])})" for name, key in beats_nb))
    else:
        A('- **NO.** No cut policy\'s net-vs-never-bail CI excludes 0 on the positive side.')
    if loses_nb:
        A('- Policies whose CI excludes 0 on the NEGATIVE side (significantly WORSE than '
          'never-bail): ' + ', '.join(
              f"{name} ({fmt_ci(*deltas[key]['vs_nb'])})" for name, key in loses_nb))
    A('')
    A('**(b) Does the doc-100 +17.7 (24t stop) edge survive the natural mix + the power increase?**')
    s24slo = s24['lo']
    A(f"- 24t plain stop on the full tape: mean net **{s24['mean']:+.2f}** t/ep, "
      f"CI [{s24['lo']:+.2f}, {s24['hi']:+.2f}] "
      f"({'CI excludes 0' if not (s24['lo'] <= 0 <= s24['hi']) else 'CI includes 0'}). "
      f"{'The +17.7 does NOT survive' if s24['mean'] < 17.7 - 1e-9 else 'The edge survives'} "
      f"the natural mix: the balanced 1:1 set over-weighted WRONG (where a stop pays), so the "
      f"headline shrinks toward the natural-mix value.")
    A('')
    A('**(c) Does re-entry\'s dipped-knife repair change sign at natural mix (dipped goods rarer than 1:1)?**')
    re_dip = per_class['good_dipped']['reentry']
    st_dip = per_class['good_dipped'][best_key]
    A(f"- stop+re-entry (48/4/1) full-tape net: mean **{sre['mean']:+.2f}** t/ep, "
      f"CI [{sre['lo']:+.2f}, {sre['hi']:+.2f}]. On GOOD-dipped (N={mix['good_dipped']}, "
      f"{mix['good_dipped'] / N:.1%} of the tape): re-entry {re_dip['mean']:+.1f} vs best-stop "
      f"{st_dip['mean']:+.1f} t/ep. Delta vs best stop overall = {fmt_ci(*deltas['reentry']['vs_bs'])}.")
    A('')

    A('## Caveats (printed)')
    A(f'- **1m granularity**: drift is per-minute; intrabar stop/trigger crossings are invisible '
      f'-- a real stop fires earlier and often deeper, a real re-entry trigger can fire-and-reverse '
      f'within a minute. All numbers are 1m-resolution estimates, OPTIMISTIC about clean fills.')
    A(f'- **overlapping windows within a day**: the full tape has multiple engagements per day '
      f'({N} engagements / {n_days} days) with overlapping forward windows -- their P&L is NOT '
      f'independent. The day-block bootstrap resamples DISTINCT days precisely to cover this '
      f'dependence (the CI is wider and more honest than an i.i.d. per-episode bootstrap).')
    A(f'- **frozen-on-2024 params evaluated on 2025-26**: the re-entry 48/4/1 winner and the veto '
      f'(coefs, scaler, p*=0.45) were sealed on 2024 train; the plain-stop grid is pre-registered '
      f'(doc 103). Transfer risk was already demonstrated in docs 103/105 -- THIS full-tape frontier '
      f'is the test of record for it. The veto AUC was 0.53 in-sample / 0.53 CV (below the 0.05 '
      f'signal floor); its coefficients are noise-level, so stop+veto is expected to track either '
      f'the plain 24t stop or near-never-bail depending on how often p*=0.45 vetoes.')
    A('')
    A('_Path-sim frontier on the sealed test tape. A dojo/path number is a hypothesis, not a live '
      'result: any retained rule still graduates through the sealed harness (graduation firewall)._')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print(f'\nwrote {OUT_MD}')

    # console summary
    print('\n================ FRONTIER (net vs never-bail, ticks/ep) ================')
    for name, key in policies:
        s = stats[key]
        star = '' if (s['lo'] <= 0 <= s['hi']) else ' *'
        print(f'  {name:38s} mean {s["mean"]:+7.2f}{star}  CI[{s["lo"]:+.2f},{s["hi"]:+.2f}]  '
              f'med {s["median"]:+5.1f}  mode {s["mode"]:+5.1f}  abs {s["abs_mean"]:+7.2f}')
    print(f'\n  best stop = X={best_x}')
    print(f'  (a) beats never-bail w/ CI>0: {[n for n, k in beats_nb] or "NONE"}')
    print(f'  (b) 24t stop net {s24["mean"]:+.2f} CI[{s24["lo"]:+.2f},{s24["hi"]:+.2f}] '
          f'(doc100 ref +17.7)')
    print(f'  (c) re-entry net {sre["mean"]:+.2f} CI[{sre["lo"]:+.2f},{sre["hi"]:+.2f}]; '
          f'dipped re-entry {per_class["good_dipped"]["reentry"]["mean"]:+.1f} vs '
          f'best-stop {per_class["good_dipped"][best_key]["mean"]:+.1f}')


if __name__ == '__main__':
    main()
