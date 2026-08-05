"""Score the owner's directional theses against what the tape actually did.

The one measurement the teacher-student program rests on and that nobody has
taken. Discipline, fixed before any outcome was computed:

  1. Theses are EXTRACTED and CLASSIFIED first; outcomes are only joined
     afterwards, in a second pass.
  2. Horizons are fixed in advance: 5, 15, 60 minutes of SIM time.
  3. A statement that cannot be scored is COUNTED as unscoreable, never
     silently dropped.
  4. Only messages that carry a DIRECTION are scored. Instructions ("go
     short") are excluded unless they also assert what price will do —
     an order is not a prediction.

Sim-clock mapping: each message inherits the sim bar of the most recent
PRECEDING dojo event (both logs carry wall-clock), so a thesis is scored from
the moment the owner could actually see, not from wall time.

  python research/dojo_forge/tools/score_corpus.py --extract   # pass 1
  python research/dojo_forge/tools/score_corpus.py --score     # pass 2
"""
import argparse
import json
import os
import re
import sqlite3

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CORPUS = os.path.join(REPO, 'research', 'dojo_forge', 'reports',
                      'corpus_tagged.parquet')
DB = os.path.join(REPO, 'research', 'dojo_forge', 'gate_state',
                  'pocket_dojo.db')
OUT = os.path.join(REPO, 'research', 'dojo_forge', 'reports')
HORIZONS_MIN = (5, 15, 60)
MOVE_MIN_PT = 3.0          # smaller than this = no call resolved either way

DOWN = r'\b(down|short|fall|drop|crash|decline|lower|bear|descen|decent|sell off|selloff|dump)\b'
UP = r'\b(up|long|rally|rise|higher|bounce|recover|bull|climb|pump)\b'
# a thesis asserts what price WILL do; an order just says what to trade
THESIS = r'\b(thesis|predict|prediction|think|believe|expect|will|going to|gonna|likely|mostlikely|most likely|feels?|i see|my take|almost certain)\b'


def extract():
    c = pd.read_parquet(CORPUS)
    m = c[(c['direction'] == 'in') & c['day'].notna()].copy()
    rows = []
    for _, r in m.iterrows():
        t = (r['text'] or '').lower()
        if len(t) < 15:
            continue
        has_thesis = bool(re.search(THESIS, t))
        d_hit, u_hit = bool(re.search(DOWN, t)), bool(re.search(UP, t))
        if not has_thesis or (d_hit == u_hit):     # need exactly one direction
            continue
        rows.append(dict(day=r['day'], wall=r['wall'], bar=r['bar'],
                         call='down' if d_hit else 'up',
                         text=(r['text'] or '').replace('\n', ' ')[:200]))
    e = pd.DataFrame(rows)
    e.to_parquet(os.path.join(OUT, 'corpus_theses.parquet'), index=False)
    print(f'{len(e)} directional theses extracted from {len(m)} tagged owner '
          f'messages')
    print(e.groupby(['day', 'call']).size().to_string())
    return e


def score():
    e = pd.read_parquet(os.path.join(OUT, 'corpus_theses.parquet'))
    con = sqlite3.connect(DB)
    ev = pd.read_sql('select day, bar, wall from events order by wall', con)
    ev['w'] = pd.to_datetime(ev['wall'].str.slice(0, 19),
                             format='%Y-%m-%dT%H:%M:%S')
    e['w'] = pd.to_datetime(e['wall'].str.slice(0, 19),
                            format='%Y-%m-%dT%H:%M:%S')
    e = e.sort_values('w')
    out = []
    for day, g in e.groupby('day'):
        p = os.path.join(REPO, 'DATA', 'ATLAS', '1m', f'{day}.parquet')
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        ts, c = d['timestamp'].to_numpy(), d['close'].to_numpy()
        evd = ev[ev['day'] == day].sort_values('w')
        for _, r in g.iterrows():
            prior = evd[evd['w'] <= r['w']]
            if not len(prior) or pd.isna(prior.iloc[-1]['bar']):
                out.append(dict(day=day, call=r['call'], scoreable=False,
                                why='no sim clock', text=r['text']))
                continue
            b = int(prior.iloc[-1]['bar'])
            if b >= len(c) - max(HORIZONS_MIN) - 1:
                out.append(dict(day=day, call=r['call'], scoreable=False,
                                why='too near day end', text=r['text']))
                continue
            rec = dict(day=day, call=r['call'], scoreable=True, bar=b,
                       text=r['text'])
            for hz in HORIZONS_MIN:
                mv = float(c[min(b + hz, len(c) - 1)] - c[b])
                rec[f'move_{hz}'] = mv
                if abs(mv) < MOVE_MIN_PT:
                    rec[f'hit_{hz}'] = None
                else:
                    rec[f'hit_{hz}'] = bool((mv < 0) == (r['call'] == 'down'))
            out.append(rec)
    s = pd.DataFrame(out)
    s.to_parquet(os.path.join(OUT, 'corpus_scored.parquet'), index=False)
    ok = s[s['scoreable']]
    print(f'\n{len(s)} theses | scoreable {len(ok)} | '
          f'unscoreable {len(s)-len(ok)} '
          f'({dict(s[~s["scoreable"]]["why"].value_counts()) if len(s)>len(ok) else ""})')
    rng = np.random.default_rng(20260804)
    print(f'\n{"horizon":>8} {"n":>5} {"hit rate":>9} {"day-clustered 95% CI":>22}')
    for hz in HORIZONS_MIN:
        col = f'hit_{hz}'
        v = ok[ok[col].notna()]
        if not len(v):
            continue
        h = v[col].astype(int).to_numpy()
        days = v['day'].to_numpy()
        uq = np.unique(days)
        bs = []
        for _ in range(4000):
            pick = rng.choice(uq, size=len(uq), replace=True)
            idx = np.concatenate([np.flatnonzero(days == d) for d in pick])
            bs.append(h[idx].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f'{hz:>7}m {len(v):>5} {h.mean():>8.1%} '
              f'[{lo:.1%}, {hi:.1%}]'.rjust(24)
              + ('  <- excludes 50%' if lo > 0.5 or hi < 0.5 else ''))
    print(f'\nno-move (|move| < {MOVE_MIN_PT:g}pt, unresolved): ' +
          ', '.join(f'{hz}m {int(ok[f"hit_{hz}"].isna().sum())}'
                    for hz in HORIZONS_MIN))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--score', action='store_true')
    a = ap.parse_args()
    if a.extract or not a.score:
        extract()
    if a.score:
        score()
