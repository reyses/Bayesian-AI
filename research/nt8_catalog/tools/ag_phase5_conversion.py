"""
Phase-5 THREE-ANCHOR conversion view for ALL dossiers (entry -> exit -> post-exit).

For each dossier, pull the interpretable NMP features (z_se, reversion_prob, lambda_hat)
at ENTRY (event_idx), EXIT (resolution_idx), and POST-EXIT (+1min) -- RTH-aligned per
the dossier's index convention -- and characterize how the state CONVERTS, split by
outcome (response occurred vs not). Answers doc-017's descriptive half:
how does F-space evolve leading into the exit, and how does it convert after.

A clean reversion signature (like ATR-09) = winners enter extended (z_se away from 0)
and z_se crosses toward 0/opposite by exit, while non-responders stay flat.

Output: reports/AG_cat_00_CONVERSION.md (table) + per-dossier plots in
reports/assets/phase5_gallery/conversion/.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
import ag_phase5_doe as D

ZSE, REV, LAM = 'L3_5s_z_se_30', 'L3_5s_reversion_prob_30', 'L4_5s_lambda_hat_12'
POST = 12          # +1min after exit
MAX_EV = 3000      # subsample big dossiers (means are stable)
OUT = os.path.join(D.BASE, 'reports', 'assets', 'phase5_gallery', 'conversion')
os.makedirs(OUT, exist_ok=True)

def collect(dossier):
    ev = pd.read_parquet(os.path.join(D.BASE, 'tests', dossier, 'events.parquet'))
    if 'resolution_idx' not in ev.columns:
        return None
    eidx_max = int(ev['event_idx'].max())
    mode = 'rth' if eidx_max < 5000 else ('full' if eidx_max < 16000 else 'exclude')
    if mode == 'exclude':
        return None
    if len(ev) > MAX_EV:
        ev = ev.sample(MAX_EV, random_state=0)
    rows = []
    for day in sorted(ev['day'].unique()):
        dfmt = day.replace('-', '_')
        try:
            L3 = pd.read_parquet(f'{D.FEAT}/L3_5s/{dfmt}.parquet')
            L4 = pd.read_parquet(f'{D.FEAT}/L4_5s/{dfmt}.parquet')
        except Exception:
            continue
        if mode == 'rth':
            L3, L4 = D._rth(L3), D._rth(L4)
        for _, r in ev[ev['day'] == day].iterrows():
            ei, xi = int(r['event_idx']), int(r['resolution_idx'])
            if xi <= ei:            # need a valid forward exit anchor
                continue
            pi = xi + POST
            if pi >= len(L3) or pi >= len(L4):
                continue
            rows.append((int(r['hit']), float(r['magnitude']),
                         L3[ZSE].iloc[ei], L3[ZSE].iloc[xi], L3[ZSE].iloc[pi],
                         L4[LAM].iloc[ei], L4[LAM].iloc[xi], L4[LAM].iloc[pi]))
    if len(rows) < 30:
        return None
    return pd.DataFrame(rows, columns=['hit', 'mag', 'ze', 'zx', 'zp', 'le', 'lx', 'lp'])

def plot(dossier, df):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f'{dossier} — state conversion (entry->exit->post)', fontsize=12)
    xs = [0, 1, 2]
    for grp, m, col in [('response (hit=1)', df['hit'] == 1, 'green'), ('no-response (hit=0)', df['hit'] == 0, 'red')]:
        s = df[m]
        if len(s) == 0:
            continue
        ax[0].plot(xs, [s['ze'].mean(), s['zx'].mean(), s['zp'].mean()], '-o', color=col, label=f'{grp} N={len(s)}')
        ax[1].plot(xs, [s['le'].mean(), s['lx'].mean(), s['lp'].mean()], '-o', color=col, label=grp)
    for a, t in zip(ax, ['z_se (extension)', 'lambda_hat']):
        a.axhline(0, color='k', lw=.6); a.set_xticks(xs); a.set_xticklabels(['entry', 'exit', 'post'])
        a.set_title(t); a.legend(fontsize=8)
    plt.tight_layout(); fp = os.path.join(OUT, f'{dossier}.png'); plt.savefig(fp, dpi=85); plt.close()

if __name__ == '__main__':
    targets = sys.argv[1:] or sorted(os.path.basename(os.path.dirname(p))
                                     for p in glob.glob(os.path.join(D.BASE, 'tests', '*', 'events.parquet')))
    lines = ["# Phase-5 State Conversion — all dossiers (entry -> exit -> post-exit)\n",
             "z_se and lambda_hat means at the 3 anchors, split by whether the article's response "
             "occurred. `conv` = z_se(exit)-z_se(entry) for responders; a large |conv| that "
             "non-responders lack = a clean reversion/continuation signature.\n",
             "| Dossier | N | resp | z_se resp e->x->p | z_se noresp | conv(resp) | lambda resp e->x->p |",
             "|---|---|---|---|---|---|---|"]
    for t in targets:
        try:
            df = collect(t)
        except Exception as e:
            print(f'{t}: ERR {e}'); continue
        if df is None:
            print(f'{t}: skip'); lines.append(f"| {t[:20]} | skip | | | | | |"); continue
        w, l = df[df['hit'] == 1], df[df['hit'] == 0]
        conv = (w['zx'].mean() - w['ze'].mean()) if len(w) else float('nan')
        zr = f"{w['ze'].mean():+.2f}->{w['zx'].mean():+.2f}->{w['zp'].mean():+.2f}" if len(w) else "-"
        zn = f"{l['ze'].mean():+.2f}->{l['zx'].mean():+.2f}->{l['zp'].mean():+.2f}" if len(l) else "-"
        lr = f"{w['le'].mean():+.3f}->{w['lx'].mean():+.3f}->{w['lp'].mean():+.3f}" if len(w) else "-"
        plot(t, df)
        lines.append(f"| {t[:20]} | {len(df)} | {len(w)} | {zr} | {zn} | {conv:+.2f} | {lr} |")
        print(f"{t}: N={len(df)} resp={len(w)} conv(z_se)={conv:+.2f}")
    with open(os.path.join(D.BASE, 'reports', 'AG_cat_00_CONVERSION.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print("\nwrote reports/AG_cat_00_CONVERSION.md + conversion plots")
