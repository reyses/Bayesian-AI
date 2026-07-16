"""
ZIGZAG confirmation PHASE-IN-LABEL (actionability of the 0.96 agreement).

The causal zigzag pivot confirmation agrees with the active AI label ~0.96 — partly
by construction (the labeler's turns are zigzag-like turns). What decides whether the
signal is USABLE is timing: phase = (fire_ts - label_entry) / (label_exit - entry).
Early phase (<0.25) = most of the label's move still ahead. Also reports remaining
minutes and remaining displacement fraction stand-ins via phase buckets, per year.

Reads reports/signal_rows_ZIGZAG.parquet + DATA/ai_cusp_picks. Writes
reports/zigzag_phase_in_label.md.
"""
import os, glob, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
LBL = os.path.join(ROOT, 'DATA', 'ai_cusp_picks')
REP = os.path.abspath(os.path.join(HERE, '..', 'reports'))

F = pd.read_parquet(os.path.join(REP, 'signal_rows_ZIGZAG.parquet'))
lblf = {os.path.basename(f)[9:19]: f for f in glob.glob(os.path.join(LBL, 'ai_picks_*_multi.json'))}

rows = []
for day, g in F.groupby('day', sort=False):
    labs = [(t['entry_ts'], t['exit_ts'], t.get('direction') == 'LONG')
            for t in json.load(open(lblf[day.replace('_', '-')])).get('trades', [])
            if t.get('exit_ts')]
    for _, r in g.iterrows():
        hit = [(a, b, lg) for a, b, lg in labs if a <= r['ts'] <= b]
        if not hit: continue
        a, b, lg = hit[0]
        rows.append(dict(year=day[:4], phase=(r['ts'] - a) / max(1, b - a),
                         mins_in=(r['ts'] - a) / 60.0, mins_left=(b - r['ts']) / 60.0,
                         agree=int(lg == r['is_long'])))
D = pd.DataFrame(rows)
lines = ['# ZIGZAG confirmation — phase in label (N=%d fires inside a label)' % len(D)]
lines.append('\nphase distribution (all years): mode-first buckets')
buck = pd.cut(D['phase'], [0, .1, .25, .5, .75, 1.0])
tab = D.groupby(buck, observed=True).agg(n=('agree', 'size'), agree=('agree', 'mean'),
                                          med_mins_left=('mins_left', 'median'))
lines.append(tab.to_string())
lines.append('\nmedian minutes INTO label at fire: %.1f | median minutes LEFT: %.1f'
             % (D['mins_in'].median(), D['mins_left'].median()))
lines.append('mode of phase (0.05 bins): %.2f'
             % ((np.histogram(D['phase'], bins=20, range=(0, 1))[0].argmax() + 0.5) / 20))
for yr, g in D.groupby('year'):
    lines.append('%s: N=%d, median phase %.2f, median mins-left %.1f, agree %.2f'
                 % (yr, len(g), g['phase'].median(), g['mins_left'].median(), g['agree'].mean()))
out = os.path.join(REP, 'zigzag_phase_in_label.md')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('\n'.join(lines))
