"""
Wrong-Direction Dojo -- scorer, ECONOMIC-TRUTH (research/exit_dojo/tools/score_wrongdir.py)

Task 099 (reviewer re-cut 2026-07-17). Reads wrongdir gate transcripts, VERIFIES the nonce
chain per episode (reusing score_full_run.audit_nonce_chain -- the stepwise-blind audit),
then scores the BINARY bail-detector against ECONOMIC truth (terminal drift, NOT the label
taxonomy). The BAND + DIP are read from selection.json meta so classification matches the
selector EXACTLY.

    truth  := WRONG if terminal drift <= -BAND ; GOOD if terminal drift >= +BAND
              good_kind := 'dipped' if min drift <= -DIP else 'clean'
    bailed := episode has a binding EXIT.  pred = WRONG if bailed else GOOD.

Metrics (doc 099 re-cut):
  - CATCH      = P(bail | WRONG)                   (cut the loser)
  - FALSE-BAIL = P(bail | GOOD), split dipped-vs-clean (knifing a temporary dip = the trap)
  - precision  = P(WRONG | bail)
  - speed      : on CAUGHT wrong, bail minute + %ile-of-window (mode-first)
  - DAMAGE-AVOIDED IN TIME (the "cut before damage" metric): on each caught WRONG,
        ticks_saved = (|terminal_adverse| - |drift_at_bail|) * 4   [pts->ticks; MNQ tick=0.25]
      mode-first + median + bootstrap CI.
  - economics (TICKS, mode-first): per episode net vs NEVER-BAIL =
        (drift[eff_exit] - drift[window]) * 4   (the realized $ diff of the bail policy)
  - THE HONEST BAR (Moises' "dumb selector"): a naive "bail if adverse drift <= -X ticks"
      stop swept over an X grid on the SAME episodes -> its (catch, false-bail, net-ticks)
      ROC. The agent PASSES only if its operating point sits ABOVE that ROC (not dominated)
      AND its mean net-ticks beats the best-X dumb stop. never-bail (0,0,0) is the floor.

Run:  python3.11 research/exit_dojo/tools/score_wrongdir.py
Out:  research/exit_dojo/reports/wrongdir/scorecard.md + synthesis.md
"""
import os
import re
import sys
import json
import glob

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from score_full_run import read_transcript, audit_nonce_chain   # reuse the nonce audit

DOJO_ROOT = os.path.abspath(os.path.join(HERE, '..'))
WRONGDIR_DIR = os.path.join(DOJO_ROOT, 'reports', 'wrongdir')
GATE_STATE_DIR = os.path.join(WRONGDIR_DIR, 'gate_state')
TRUTH_DIR = os.path.join(WRONGDIR_DIR, 'truth')
SELECTION_JSON = os.path.join(WRONGDIR_DIR, 'selection.json')
SCORECARD = os.path.join(WRONGDIR_DIR, 'scorecard.md')
SYNTHESIS = os.path.join(WRONGDIR_DIR, 'synthesis.md')

# ---- constants (house rule: no bare magic numbers) -----------------------------------
TICK_PTS = 0.25                 # MNQ tick size (points); ticks = points / TICK_PTS
PTS_TO_TICKS = 1.0 / TICK_PTS   # = 4.0
BW_TICKS = 4.0                  # 4 ticks = $2 mode bin (metric mandate: $2 bins for $/trade)
BOOTS = 4000
SEED = 12345
# dumb-stop adverse-drawdown grid, TICKS (1..25 pts): the naive comparator to beat
DUMB_X_TICKS = [4, 8, 12, 16, 20, 24, 32, 40, 48, 60, 80, 100]
DEFAULT_BAND = 4.0              # pts; fallback if selection.json meta has none
DEFAULT_DIP = 4.0

# bail-reason vocabulary (grammar citation audit over binding-EXIT reasons)
VOCAB = {
    'adverse / against': r'advers|against|opposite|wrong[- ]?(way|direction)',
    'giveback / retrace': r'giveback|gave back|retrace|pull ?back',
    'no recovery / stall': r'no recover|not recover|stall|fail(ing|ed)? to|flat',
    'accelerating loss': r'accelerat|deepen|compound|worse|drop|plung|sink|new .*extreme',
    'against-fires (aux)': r'kmdr|climax|\bha\b|propp|counter[- ]?fire',
    'ER10 / efficiency': r'\ber10\b|efficien',
    'vol / volatility': r'\bvol\b|volatil',
}


# ================= helpers ============================================================
def boot_ci(x, boots=BOOTS, seed=SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(boots)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hist_mode(x, bw):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    lo, hi = np.floor(x.min() / bw) * bw, np.ceil(x.max() / bw) * bw + bw
    edges = np.arange(lo, hi + bw, bw)
    h, e = np.histogram(x, bins=edges)
    k = int(np.argmax(h))
    return float((e[k] + e[k + 1]) / 2)


def dumb_exit_minute(drift, window, x_ticks):
    """First minute m in 1..window where favorable-signed drift <= -x_ticks (in ticks).
    Returns (bailed: bool, eff_exit: int). Never-triggered -> (False, window)."""
    thr_pts = -x_ticks * TICK_PTS
    for m in range(1, window + 1):
        if drift[m] <= thr_pts:
            return True, m
    return False, window


def net_ticks_vs_neverbail(drift, window, eff_exit):
    """(drift[eff_exit] - drift[window]) in TICKS. 0 if held to window end."""
    return (drift[eff_exit] - drift[window]) * PTS_TO_TICKS


# ================= per-episode scoring ================================================
def score_episode(eid, band, dip):
    tr_path = os.path.join(GATE_STATE_DIR, f'{eid}.transcript.jsonl')
    truth_path = os.path.join(TRUTH_DIR, f'{eid}.json')
    if not os.path.exists(truth_path) or not os.path.exists(tr_path):
        return None
    serves, commits, finish = read_transcript(tr_path)
    ok, reason, exit_frame = audit_nonce_chain(serves, commits)
    with open(truth_path, encoding='utf-8') as f:
        truth = json.load(f)
    drift = truth['per_minute_forward_drift']
    window = truth['window_minutes']
    terminal = drift[window]
    mindrift = min(drift)
    # ECONOMIC truth classification (identical rule to the selector)
    if terminal <= -band:
        is_wrong, truth_label, good_kind = True, 'wrong', None
    elif terminal >= band:
        is_wrong, truth_label, good_kind = False, 'good', ('dipped' if mindrift <= -dip else 'clean')
    else:
        return dict(eid=eid, audit='DEAD-BAND (|terminal|<BAND)', scored=False,
                    is_wrong=None, good_kind=None, day=truth.get('real_day'))
    row = dict(eid=eid, truth_label=truth_label, is_wrong=is_wrong, good_kind=good_kind,
               terminal=terminal, min_drift=mindrift, day=truth.get('real_day'),
               audit=('PASS' if ok else f'FAIL: {reason}'), scored=ok, n_commits=len(commits))
    if not ok:
        return row
    bailed = exit_frame is not None
    eff_exit = min(exit_frame, window) if bailed else window
    row.update(
        bailed=bailed, exit_min=(exit_frame if bailed else None), eff_exit=eff_exit,
        window=window, exit_pctile=(eff_exit / window if (bailed and window > 0) else np.nan),
        net_ticks=net_ticks_vs_neverbail(drift, window, eff_exit),
        # damage-avoided (caught wrong only): |terminal loss| - |loss at the bail point|
        ticks_saved=((abs(terminal) - abs(drift[eff_exit])) * PTS_TO_TICKS
                     if (is_wrong and bailed) else np.nan),
        drift=drift)
    return row


# ================= aggregate confusion + economics ===================================
def confusion(scored):
    wrong = [r for r in scored if r['is_wrong']]
    good = [r for r in scored if not r['is_wrong']]
    dipped = [r for r in good if r['good_kind'] == 'dipped']
    clean = [r for r in good if r['good_kind'] == 'clean']
    bailed = [r for r in scored if r['bailed']]

    def rate(sub):
        return (np.mean([r['bailed'] for r in sub]) if sub else np.nan, len(sub))

    catch, n_wrong = rate(wrong)
    fb_all, n_good = rate(good)
    fb_dip, n_dip = rate(dipped)
    fb_cln, n_cln = rate(clean)
    prec = (np.mean([r['is_wrong'] for r in bailed]) if bailed else np.nan)
    return dict(catch=catch, n_wrong=n_wrong, fb_all=fb_all, n_good=n_good,
                fb_dip=fb_dip, n_dip=n_dip, fb_cln=fb_cln, n_cln=n_cln,
                precision=prec, n_bailed=len(bailed), n_total=len(scored),
                wrong=wrong, good=good, dipped=dipped, clean=clean)


def dumb_roc(scored, x_grid):
    wrong = [r for r in scored if r['is_wrong']]
    good = [r for r in scored if not r['is_wrong']]
    roc = []
    for x in x_grid:
        nets, cw, cg = [], 0, 0
        for r in scored:
            b, ee = dumb_exit_minute(r['drift'], r['window'], x)
            nets.append(net_ticks_vs_neverbail(r['drift'], r['window'], ee))
            if b and r['is_wrong']:
                cw += 1
            if b and not r['is_wrong']:
                cg += 1
        roc.append(dict(x=x, catch=(cw / len(wrong) if wrong else np.nan),
                        false_bail=(cg / len(good) if good else np.nan),
                        net_mean=float(np.mean(nets)), net_median=float(np.median(nets))))
    return roc


def verdict(cf, agent_net_mean, roc):
    """Agent PASSES iff (a) not dominated by any dumb point on the ROC AND (b) its mean
    net-ticks beats the best-X dumb net."""
    ca, fa = cf['catch'], cf['fb_all']
    dominators = [d for d in roc
                  if np.isfinite(d['catch']) and np.isfinite(d['false_bail'])
                  and np.isfinite(ca) and np.isfinite(fa)
                  and d['catch'] >= ca and d['false_bail'] <= fa
                  and (d['catch'] > ca or d['false_bail'] < fa)]
    above_roc = (len(dominators) == 0)
    best_x = max(roc, key=lambda d: d['net_mean']) if roc else None
    beat_net = (best_x is not None and agent_net_mean > best_x['net_mean'])
    return dict(above_roc=above_roc, dominators=dominators, best_x=best_x,
                beat_net=beat_net, overall=(above_roc and beat_net),
                agent_point=(ca, fa, agent_net_mean))


# ================= reason audit =======================================================
def bail_reasons():
    reasons = []
    for tr in glob.glob(os.path.join(GATE_STATE_DIR, '*.transcript.jsonl')):
        try:
            with open(tr, encoding='utf-8') as f:
                for line in f:
                    if '"event": "commit"' in line or '"event":"commit"' in line:
                        d = json.loads(line)
                        if d.get('decision') == 'EXIT':
                            reasons.append((d.get('reason') or '').lower())
                            break
        except (OSError, json.JSONDecodeError):
            pass
    return reasons


# ================= writers ============================================================
def write_scorecard(rows, scored, cf, roc, vd, agent_net, band, dip):
    L = []
    A = L.append
    A('# Wrong-Direction Dojo -- scorecard (ECONOMIC truth; cut BAD trades before damage)')
    A('')
    npass = len(scored); nfail = len(rows) - npass
    A(f'Played episodes: {len(rows)} | nonce-chain audit PASS + in-band: {npass} | '
      f'FAIL/dead-band: {nfail}. Truth: WRONG=terminal<=-{band:.0f}pts, GOOD=terminal>=+{band:.0f}pts; '
      f'good_kind dipped=min<=-{dip:.0f}pts.')
    A('')
    A('## Per-episode')
    A('| eid | truth | good_kind | terminal(pts) | audit | bailed | exit(min) | %ile | '
      'net vs never-bail (ticks) | damage-avoided (ticks) |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        if not r.get('scored'):
            A(f"| {r['eid']} | {r.get('truth_label','?')} | {r.get('good_kind') or '-'} | - | "
              f"{r['audit']} | - | - | - | - | - |")
            continue
        pct = f"{r['exit_pctile']:.2f}" if np.isfinite(r['exit_pctile']) else '-'
        ts = f"{r['ticks_saved']:+.1f}" if np.isfinite(r['ticks_saved']) else '-'
        A(f"| {r['eid']} | {r['truth_label'].upper() if r['is_wrong'] else 'good'} | "
          f"{r['good_kind'] or '-'} | {r['terminal']:+.1f} | PASS | "
          f"{'yes' if r['bailed'] else 'no'} | {r['exit_min'] if r['bailed'] else '-'} | {pct} | "
          f"{r['net_ticks']:+.1f} | {ts} |")
    A('')
    A('## Binary confusion')
    A(f"- **CATCH** P(bail|WRONG) = **{cf['catch']:.0%}** (N_wrong={cf['n_wrong']})")
    A(f"- **FALSE-BAIL** P(bail|GOOD) = **{cf['fb_all']:.0%}** (N_good={cf['n_good']}) | "
      f"**dipped {cf['fb_dip']:.0%}** (N={cf['n_dip']}, the hard case) | "
      f"clean {cf['fb_cln']:.0%} (N={cf['n_cln']})")
    A(f"- **precision** P(WRONG|bail) = **{cf['precision']:.0%}** (N_bailed={cf['n_bailed']})")
    A('')
    A('## Damage-avoided in time (caught WRONG; |terminal loss| - |loss at bail|, ticks)')
    caught = [r for r in cf['wrong'] if r['bailed']]
    if caught:
        ts = np.array([r['ticks_saved'] for r in caught], float)
        em = np.array([r['eff_exit'] for r in caught], float)
        pc = np.array([r['exit_pctile'] for r in caught], float)
        lo, hi = boot_ci(ts)
        A(f"- N caught = {len(caught)}; damage-avoided mode {hist_mode(ts, BW_TICKS):+.1f} | "
          f"median {np.median(ts):+.1f} | mean {np.mean(ts):+.1f} ticks [95% CI {lo:+.1f},{hi:+.1f}]")
        A(f"- speed: bail-minute mode {hist_mode(em, 1.0):.0f} (median {np.median(em):.1f}); "
          f"%ile-of-window mode {hist_mode(pc, 0.1):.2f} (median {np.median(pc):.2f}); lower=faster.")
    else:
        A('- no caught wrong-siders.')
    A('')
    A('## Economics (ticks, mode-first; net vs NEVER-BAIL)')
    allnet = np.array([r['net_ticks'] for r in scored], float)
    wnet = np.array([r['net_ticks'] for r in cf['wrong']], float)
    gnet = np.array([r['net_ticks'] for r in cf['good']], float)
    for name, arr in [('ALL', allnet), ('WRONG (bail=saved)', wnet), ('GOOD (bail=forgone)', gnet)]:
        if len(arr) == 0:
            continue
        lo, hi = boot_ci(arr)
        sig = '' if (lo <= 0 <= hi) else ' *'
        A(f"- **{name}** N={len(arr)}: mode {hist_mode(arr, BW_TICKS):+.1f} | median "
          f"{np.median(arr):+.1f} | mean {np.mean(arr):+.1f} ticks [95% CI {lo:+.1f},{hi:+.1f}]{sig}")
    A('- never-bail floor = 0.0 ticks (by construction).')
    A('')
    A('## THE HONEST BAR -- dumb adverse-drawdown stop ROC (Moises\' dumb selector)')
    A('naive rule: bail at first minute favorable drift <= -X ticks.')
    A('| X (ticks) | X (pts) | catch | false-bail | net-ticks mean | net-ticks median |')
    A('|---|---|---|---|---|---|')
    for d in roc:
        fb = f"{d['false_bail']:.0%}" if np.isfinite(d['false_bail']) else 'n/a'
        cat = f"{d['catch']:.0%}" if np.isfinite(d['catch']) else 'n/a'
        A(f"| {d['x']} | {d['x'] * TICK_PTS:.1f} | {cat} | {fb} | "
          f"{d['net_mean']:+.1f} | {d['net_median']:+.1f} |")
    bx = vd['best_x']
    ca, fa, an = vd['agent_point']
    fa_s = f"{fa:.0%}" if np.isfinite(fa) else 'n/a'
    ca_s = f"{ca:.0%}" if np.isfinite(ca) else 'n/a'
    A('')
    A(f"- best-X dumb net = {bx['net_mean']:+.1f} ticks @ X={bx['x']} "
      f"(catch {bx['catch']:.0%}, false-bail "
      f"{bx['false_bail']:.0%})" if np.isfinite(bx['false_bail']) else
      f"- best-X dumb net = {bx['net_mean']:+.1f} ticks @ X={bx['x']}")
    A(f"- **agent operating point**: catch {ca_s}, false-bail {fa_s}, net {an:+.1f} ticks")
    A(f"- above dumb ROC (not dominated)? **{'YES' if vd['above_roc'] else 'NO'}**"
      + ('' if vd['above_roc'] else f" (dominated by X={[d['x'] for d in vd['dominators']]})"))
    A(f"- beats best-X net-ticks? **{'YES' if vd['beat_net'] else 'NO'}** "
      f"({an:+.1f} vs {bx['net_mean']:+.1f})")
    A(f"- ## VERDICT: **{'PASS' if vd['overall'] else 'FAIL'}** "
      f"(agent {'beats' if vd['overall'] else 'does NOT beat'} the dumb stop on BOTH axes)")
    A('')
    A('_Gate-audited stepwise-blind play. A dojo number is never itself a result: any rule '
      'must still pass the sealed harness (graduation firewall)._')
    os.makedirs(WRONGDIR_DIR, exist_ok=True)
    with open(SCORECARD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def write_synthesis(scored, cf, roc, vd, band):
    L = []
    A = L.append
    A('# Wrong-Direction Dojo -- synthesis (ECONOMIC truth, mode-first)\n')
    A(f"N scored = {len(scored)} (nonce-chain audited). Distinct days = "
      f"{len({r['day'] for r in scored})}. Job: cut a BAD trade (terminal <= -{band:.0f}pts) "
      f"in time, WITHOUT knifing a good trade that merely dips.\n")
    A('## Headline')
    fa = cf['fb_all']; ca = cf['catch']
    A(f"- CATCH {('%.0f%%' % (ca*100)) if np.isfinite(ca) else 'n/a'} | "
      f"FALSE-BAIL {('%.0f%%' % (fa*100)) if np.isfinite(fa) else 'n/a'} "
      f"(dipped {('%.0f%%' % (cf['fb_dip']*100)) if np.isfinite(cf['fb_dip']) else 'n/a'}, "
      f"clean {('%.0f%%' % (cf['fb_cln']*100)) if np.isfinite(cf['fb_cln']) else 'n/a'}) | "
      f"precision {('%.0f%%' % (cf['precision']*100)) if np.isfinite(cf['precision']) else 'n/a'}")
    an = vd['agent_point'][2]; bx = vd['best_x']
    A(f"- economics: agent net {an:+.1f} ticks/ep vs best-X dumb {bx['net_mean']:+.1f} @X={bx['x']}; "
      f"never-bail floor 0.")
    A(f"- **dumb-stop comparison: {'PASS' if vd['overall'] else 'FAIL'}** "
      f"(above ROC={'Y' if vd['above_roc'] else 'N'}, beats best-X net="
      f"{'Y' if vd['beat_net'] else 'N'}).\n")
    A('## The dipped-vs-clean trap (why a dumb stop bleeds)')
    A(f"- false-bail on DIPPED goods = "
      f"{('%.0f%%' % (cf['fb_dip']*100)) if np.isfinite(cf['fb_dip']) else 'n/a'} (N={cf['n_dip']}); "
      f"on CLEAN goods {('%.0f%%' % (cf['fb_cln']*100)) if np.isfinite(cf['fb_cln']) else 'n/a'} "
      f"(N={cf['n_cln']}). A dumb adverse-drawdown stop bails EVERY dipped good; the agent's "
      f"edge is holding those while still cutting the true losers.\n")
    A('## Bail-reason grammar (binding-EXIT reasons)')
    reasons = bail_reasons()
    A(f"Binding-EXIT reasons collected: {len(reasons)}.")
    A('| signal cited | episodes | share of bails |')
    A('|---|---|---|')
    counts = [(name, sum(1 for s in reasons if re.search(rx, s))) for name, rx in VOCAB.items()]
    for name, c in sorted(counts, key=lambda kv: -kv[1]):
        A(f"| {name} | {c} | {(c / max(len(reasons), 1)):.0%} |")
    A('\n_The conditions blind agents cited to cut a bad trade -- the candidate rule set to '
      'seed the dumb selector + the mamba (must still graduate through the sealed harness)._')
    with open(SYNTHESIS, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def load_band_dip():
    if os.path.exists(SELECTION_JSON):
        try:
            meta = json.load(open(SELECTION_JSON, encoding='utf-8')).get('meta', {})
            return float(meta.get('band', DEFAULT_BAND)), float(meta.get('dip_pts', DEFAULT_DIP))
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return DEFAULT_BAND, DEFAULT_DIP


def main():
    band, dip = load_band_dip()
    truth_files = sorted(glob.glob(os.path.join(TRUTH_DIR, '*.json')))
    eids = [os.path.splitext(os.path.basename(p))[0] for p in truth_files]
    played = [e for e in eids
              if os.path.exists(os.path.join(GATE_STATE_DIR, f'{e}.transcript.jsonl'))]
    if not played:
        print(f'no played episodes (no transcripts) in {GATE_STATE_DIR}')
        return
    rows = [r for r in (score_episode(e, band, dip) for e in played) if r is not None]
    scored = [r for r in rows if r.get('scored')]
    if not scored:
        print('no episodes passed the nonce-chain audit / all dead-band; nothing to score.')
        return
    cf = confusion(scored)
    roc = dumb_roc(scored, DUMB_X_TICKS)
    agent_net = float(np.mean([r['net_ticks'] for r in scored]))
    vd = verdict(cf, agent_net, roc)

    write_scorecard(rows, scored, cf, roc, vd, agent_net, band, dip)
    write_synthesis(scored, cf, roc, vd, band)
    print(f'wrote {SCORECARD}')
    print(f'wrote {SYNTHESIS}')
    print(f"  BAND={band:.0f} | catch {cf['catch'] if np.isfinite(cf['catch']) else float('nan'):.2f} "
          f"| false-bail {cf['fb_all'] if np.isfinite(cf['fb_all']) else float('nan'):.2f} "
          f"(dip {cf['fb_dip'] if np.isfinite(cf['fb_dip']) else float('nan'):.2f}, "
          f"cln {cf['fb_cln'] if np.isfinite(cf['fb_cln']) else float('nan'):.2f}) "
          f"| precision {cf['precision'] if np.isfinite(cf['precision']) else float('nan'):.2f}")
    print(f"  agent net {agent_net:+.1f} ticks vs best-X dumb {vd['best_x']['net_mean']:+.1f} "
          f"@X={vd['best_x']['x']} | VERDICT {'PASS' if vd['overall'] else 'FAIL'}")
    for r in rows:
        if not r.get('scored'):
            print(f"  {r['eid']:34s} {r['audit']}")
            continue
        print(f"  {r['eid']:34s} {r['truth_label']:5s} {r['good_kind'] or '-':6s} "
              f"term{r['terminal']:+.0f} bailed={r['bailed']} exit={r['exit_min']} "
              f"net={r['net_ticks']:+.1f}t saved={r['ticks_saved'] if np.isfinite(r['ticks_saved']) else float('nan'):+.1f}t")


if __name__ == '__main__':
    main()
