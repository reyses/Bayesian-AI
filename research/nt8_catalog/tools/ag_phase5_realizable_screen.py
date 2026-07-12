"""
REALIZABLE-TRADE SCREEN across ALL dossiers (the MAE-vs-target geometry test).

Motivation (docs 039/040): distributions can look profitable while the trade is
unaffordable (SEASON-12), and articles can be backwards while the FLIP is tradable
(PIVOT-16). This screens every dossier BOTH ways as a real trade built from stored
excursions, worst-case ordering:

  AS-STATED : target hit if MFE >= T ; stopped if MAE >= S (stop dominates) ; else scratch 0
  FLIPPED   : target hit if MAE >= T ; stopped if MFE >= S (stop dominates) ; else scratch 0

Grid: (T,S) in {(10,10),(10,20),(15,15),(20,20)}. Day-block bootstrap CIs (2000).
FINDING = same direction+config significant (CI_lo > 0) in BOTH years.
Multiple-comparisons note: ~24 x 2 x 4 = 192 tests; both-year joint significance
keeps expected false findings ~1-2 — treat single-config hits as candidates, and
config-robust hits (like PIVOT/ROUND) as real.
"""
import os, sys, glob
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import ag_phase5_doe as D

GRID = [(10, 10), (10, 20), (15, 15), (20, 20)]
NB = 2000
MAXN = 20000   # subsample cap per dossier (RENKO)

def day_ci(pnl, days):
    uq = np.unique(days); by = {d: pnl[days == d] for d in uq}
    mu = [np.concatenate([by[d] for d in np.random.choice(uq, len(uq), True)]).mean() for _ in range(NB)]
    return float(pnl.mean()), float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))

def screen(dossier):
    ev = pd.read_parquet(os.path.join(D.BASE, 'tests', dossier, 'events.parquet'))
    if not {'mfe', 'mae', 'day'}.issubset(ev.columns):
        return None
    ev = ev.dropna(subset=['mfe', 'mae'])
    years = sorted({str(d)[:4] for d in ev['day']})
    pair = [y for y in years if (ev['day'].str.startswith(y)).sum() >= 100][:2]
    if len(pair) < 2:
        pair = years[:2]
    if len(pair) < 2:
        return None
    if len(ev) > MAXN:
        ev = ev.sample(MAXN, random_state=0)
    out = []
    for direction in ['stated', 'flip']:
        fav, adv = ('mfe', 'mae') if direction == 'stated' else ('mae', 'mfe')
        for (T, S) in GRID:
            row = {'dir': direction, 'T': T, 'S': S, 'years': {}}
            both_sig = True
            for y in pair:
                s = ev[ev['day'].str.startswith(y)]
                if len(s) < 50:
                    both_sig = False; continue
                stopped = s[adv].values >= S
                hit_t = s[fav].values >= T
                pnl = np.where(stopped, -S, np.where(hit_t, T, 0.0)).astype(float)
                m, lo, hi = day_ci(pnl, s['day'].values)
                row['years'][y] = (len(s), m, lo, hi)
                if not (lo > 0):
                    both_sig = False
            row['both_sig'] = both_sig and len(row['years']) == 2
            out.append(row)
    return {'dossier': dossier, 'pair': pair, 'rows': out}

if __name__ == '__main__':
    targets = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p))
                                     for p in glob.glob(os.path.join(D.BASE, 'tests', '*', 'events.parquet')))
    findings, lines = [], []
    lines.append("# Realizable-Trade Screen — ALL dossiers, both directions (worst-case ordering)\n")
    lines.append("PnL: +T if favorable excursion>=T, -S if adverse>=S (stop dominates), else 0. "
                 "Day-block CIs (2000). FINDING = CI_lo>0 in BOTH years, same config.\n")
    for t in targets:
        try:
            r = screen(t)
        except Exception as e:
            print(f'{t}: ERR {e}'); continue
        if r is None:
            print(f'{t}: skip (no mfe/mae or years)'); continue
        best = None
        for row in r['rows']:
            if row['both_sig']:
                mean2 = np.mean([v[1] for v in row['years'].values()])
                if best is None or mean2 > best[0]:
                    best = (mean2, row)
        tag = ''
        if best:
            _, row = best
            ys = r['pair']
            desc = " | ".join(f"{y}: {row['years'][y][1]:+.2f} [{row['years'][y][2]:+.2f},{row['years'][y][3]:+.2f}]" for y in ys)
            nsig = sum(1 for rr in r['rows'] if rr['both_sig'] and rr['dir'] == row['dir'])
            tag = f"** {row['dir'].upper()} T{row['T']}/S{row['S']} -> {desc} (configs sig: {nsig}/4)"
            findings.append((t, row['dir'], row['T'], row['S'], desc, nsig))
        print(f"{t[:26]:28} {tag if tag else '-- nothing both-year sig'}")
        lines.append(f"\n## {t} (years {r['pair']})")
        for row in r['rows']:
            ys = " | ".join(f"{y}: {v[1]:+.2f} [{v[2]:+.2f},{v[3]:+.2f}] N={v[0]}" for y, v in row['years'].items())
            lines.append(f"- {row['dir']:6} T{row['T']}/S{row['S']}: {ys} {'  <-- BOTH-YEAR SIG' if row['both_sig'] else ''}")
    lines.append("\n\n# FINDINGS (both-year significant)\n")
    for f in sorted(findings, key=lambda x: -x[5]):
        lines.append(f"- **{f[0]}** {f[1].upper()} T{f[2]}/S{f[3]}: {f[4]} — configs sig {f[5]}/4")
    lines.append(f"\nMultiple-comparisons: ~192 tests run; expect ~1-2 both-year false findings. "
                 f"Trust config-robust (3-4/4) findings; treat 1/4 as candidates.")
    with open(os.path.join(D.BASE, 'reports', 'AG_cat_00_REALIZABLE.md'), 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))
    print(f"\n{len(findings)} both-year findings -> reports/AG_cat_00_REALIZABLE.md")
