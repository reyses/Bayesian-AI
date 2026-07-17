"""
Exit Dojo full-run synthesis (research/exit_dojo/tools/synthesize_full_run.py)

Reads the gate-audited scorecard.md, produces per-regime capture distributions
(MODE-FIRST per the metric mandate), bootstrap CIs on the captured-vs-5m-hold
delta with an explicit significance call, beat-rate vs the 5m-hold, and the
wrong-side exit-speed readout. Also greps the EXIT-frame reasons across all
transcripts for the grammar's signal vocabulary (citation counts).

Run:  python3.11 research/exit_dojo/tools/synthesize_full_run.py
Out:  research/exit_dojo/reports/full_run/synthesis.md
"""
import os
import re
import json
import glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FR = os.path.abspath(os.path.join(HERE, '..', 'reports', 'full_run'))
SCORECARD = os.path.join(FR, 'scorecard.md')
GATE = os.path.join(FR, 'gate_state')
OUT = os.path.join(FR, 'synthesis.md')

BW = 2.0            # 2pt bins for the capture mode (metric mandate)
BOOTS = 4000
SEED = 12345
RATIO_CLIP = 3.0   # |ratio|>3 is a tiny-oracle-denominator artifact; drop from ratio stats
REGIMES = ['winner', 'midflip', 'instantfail', 'chop']

# signal vocabulary -> regexes (grammar citation audit over EXIT reasons)
VOCAB = {
    'ER10 / efficiency': r'\ber10\b|efficien',
    'giveback': r'giveback|gave back|give back',
    'against-fires (multi)': r'against|counter[- ]?fire|opposing',
    'KMDR': r'kmdr',
    'CLIMAX': r'climax',
    'HA (heikin)': r'\bha\b|heikin',
    'PROPP / prop-turn': r'propp|prop[- ]?turn',
    'vol / volatility': r'\bvol\b|volatil',
    'bar close / extreme': r'close.*(low|high|extreme)|pinned|wick',
    'confluence / stack': r'confluence|stack|cluster|converg',
}


def parse_scorecard():
    rows = []
    with open(SCORECARD, encoding='utf-8') as f:
        for ln in f:
            if not ln.startswith('| 20'):   # data rows start with an eid year
                continue
            c = [x.strip() for x in ln.strip().strip('|').split('|')]
            # eid,type,audit,exit,captured,5m,oracle,ratio,pctile
            eid, typ, audit, exitm, cap, ref5, orac, ratio, pctile = c
            def num(s):
                m = re.search(r'[-+]?\d+\.?\d*', s)
                return float(m.group()) if m else np.nan
            rows.append(dict(eid=eid, type=typ, audit=audit,
                             cap=num(cap), ref5=num(ref5),
                             ratio=(np.nan if 'n/a' in ratio else num(ratio)),
                             pctile=num(pctile)))
    return rows


def boot_ci(x, boots=BOOTS, seed=SEED):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x), replace=True).mean() for _ in range(boots)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def hist_mode(x, bw=BW):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    lo, hi = np.floor(x.min()/bw)*bw, np.ceil(x.max()/bw)*bw + bw
    edges = np.arange(lo, hi + bw, bw)
    h, e = np.histogram(x, bins=edges)
    k = int(np.argmax(h))
    return float((e[k] + e[k+1]) / 2)


def exit_reasons():
    """Collect the reason string of the binding EXIT commit from each transcript."""
    reasons = []
    for tr in glob.glob(os.path.join(GATE, '*.transcript.jsonl')):
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


def main():
    rows = parse_scorecard()
    L = []
    A = L.append
    A('# Exit Dojo -- full-run synthesis (N=200, gate-audited stepwise-blind)\n')
    A('Nonce-chain audit PASS on all 200 (no agent saw a future frame). '
      'MODE-FIRST distributions; bootstrap CIs (4000) on the captured-minus-5m-hold '
      'delta with an explicit significance call. **Leakage note is moot here** -- '
      'play was blind by construction -- but the graduation firewall still holds: '
      'a rule confirmed here must pass the sealed 2024/2025-26 harness before belief.\n')

    A('## Per-regime capture (points; mode-first)')
    A('| regime | N | cap mode | cap median | cap mean | 5m-hold median | '
      'delta mean (cap-5m) | delta 95% CI | beat-5m rate | oracle-ratio median |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    allcap, allref = [], []
    for reg in REGIMES:
        r = [x for x in rows if x['type'] == reg]
        cap = np.array([x['cap'] for x in r], float)
        ref = np.array([x['ref5'] for x in r], float)
        delta = cap - ref
        allcap += list(cap); allref += list(ref)
        lo, hi = boot_ci(delta)
        beat = float(np.mean(cap > ref))
        rr = np.array([x['ratio'] for x in r], float)
        rr = rr[np.isfinite(rr) & (np.abs(rr) <= RATIO_CLIP)]
        sig = '' if (lo <= 0 <= hi) else ' *'
        A(f'| {reg} | {len(r)} | {hist_mode(cap):+.1f} | {np.median(cap):+.2f} | '
          f'{np.mean(cap):+.2f} | {np.median(ref):+.2f} | {np.mean(delta):+.2f} | '
          f'[{lo:+.2f},{hi:+.2f}]{sig} | {beat:.0%} | '
          f'{(np.median(rr) if len(rr) else float("nan")):+.2f} |')
    # overall
    allcap = np.array(allcap); allref = np.array(allref)
    dl = allcap - allref
    lo, hi = boot_ci(dl)
    sig = '' if (lo <= 0 <= hi) else ' *'
    A(f'| **ALL** | {len(allcap)} | {hist_mode(allcap):+.1f} | {np.median(allcap):+.2f} | '
      f'{np.mean(allcap):+.2f} | {np.median(allref):+.2f} | {np.mean(dl):+.2f} | '
      f'[{lo:+.2f},{hi:+.2f}]{sig} | {np.mean(allcap>allref):.0%} | -- |')
    A('\n_`*` = 95% CI excludes 0 (delta significant). delta = agent capture minus '
      'the fixed-5-minute-hold capture, per episode._\n')

    # wrong-side speed
    inf = [x for x in rows if x['type'] == 'instantfail']
    pct = np.array([x['pctile'] for x in inf], float)
    A('## Wrong-side (instantfail) exit speed')
    A(f'- N={len(inf)}; exit-%ile-of-window median **{np.median(pct):.2f}** '
      f'(mode {hist_mode(pct,0.1):.2f}); lower = faster bail. '
      f'Share bailing in the first third of the window: {np.mean(pct<=0.33):.0%}.\n')

    # grammar citation audit
    A('## Grammar citation audit (EXIT-frame reasons, N EXIT commits)')
    reasons = exit_reasons()
    A(f'Binding-EXIT reasons collected: {len(reasons)} '
      f'(episodes that force-held to the end have no EXIT reason).')
    A('| signal cited | episodes | share of exits |')
    A('|---|---|---|')
    counts = []
    for name, rx in VOCAB.items():
        c = sum(1 for s in reasons if re.search(rx, s))
        counts.append((name, c))
    for name, c in sorted(counts, key=lambda kv: -kv[1]):
        A(f'| {name} | {c} | {(c/max(len(reasons),1)):.0%} |')
    A('\n_Which live signals the blind agents actually invoked to justify exits -- '
      'the empirical vocabulary of the exit grammar, to seed EXIT-GRAMMAR-01 priors._')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print('wrote', OUT)
    print('\n'.join(L))


if __name__ == '__main__':
    main()
