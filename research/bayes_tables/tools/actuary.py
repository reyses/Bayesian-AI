"""ACTUARY LOOKUP — the read side of the Bayesian tables.

Given an event name, the question, and the live context, return what the
corpus says: posterior mean, day-clustered 95% interval, N, and whether the
cell is ACTIONABLE (survived FDR + clustered bootstrap) or merely the base
rate wearing a costume.

Design rule: NEVER invent precision. If the exact cell is missing or thin,
back off one context dimension at a time (least-informative first) and say
which dimensions were dropped. If nothing survives, return the global rate
labelled BASE.

    from actuary import lookup
    r = lookup('fakeout_poke', 'exceed_ref_first',
               kind='RETURN', dir_s='dn', depth_b='<=0.5',
               age_b='30m+', clock_b='0930')
    print(r.line())
"""
import glob
import os
from dataclasses import dataclass, field

import pandas as pd

TABLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'tables')
# dropped in this order when a cell is missing: the ones that carry the least
# separation in the v0 report go first
BACKOFF_ORDER = ['age_b', 'clock_b', 'depth_b', 'dir_s', 'ratio_b',
                 'chain_b', 'day_class', 'kind']
_cache = {}


@dataclass
class Answer:
    event: str
    question: str
    p: float
    lo: float
    hi: float
    n: int
    days: int
    actionable: bool
    dropped: list = field(default_factory=list)
    basis: str = 'CELL'

    def line(self):
        tag = ('ACTIONABLE' if self.actionable
               else 'base rate' if self.basis == 'BASE' else 'not separated')
        ci = (f' [{self.lo:.0%},{self.hi:.0%}]'
              if self.lo == self.lo else '')
        drop = f' (dropped {",".join(self.dropped)})' if self.dropped else ''
        return (f'{self.event}/{self.question}: {self.p:.0%}{ci} '
                f'n={self.n} — {tag}{drop}')


def _load(event, question):
    key = (event, question)
    if key not in _cache:
        hits = glob.glob(os.path.join(TABLES, f'{event}__{question}__*.parquet'))
        _cache[key] = pd.read_parquet(hits[0]) if hits else None
    return _cache[key]


def lookup(event, question, **ctx):
    t = _load(event, question)
    if t is None or not len(t):
        return None
    dims = [c for c in t.columns
            if c not in ('n', 'hits', 'days', 'raw', 'post', 'lo', 'hi',
                         'day_lo', 'day_hi', 'p', 'lift', 'actionable',
                         'glob')]
    use = {k: str(v) for k, v in ctx.items() if k in dims and v is not None}
    dropped = []
    while True:
        sel = t
        for k, v in use.items():
            sel = sel[sel[k] == v]
        if len(sel) >= 1:
            if len(sel) == 1:
                r = sel.iloc[0]
                lo = r['day_lo'] if r['day_lo'] == r['day_lo'] else r['lo']
                hi = r['day_hi'] if r['day_hi'] == r['day_hi'] else r['hi']
                return Answer(event, question, float(r['post']), float(lo),
                              float(hi), int(r['n']), int(r.get('days', 0)),
                              bool(r['actionable']), dropped)
            # POOLED cell (a dimension was dropped): recompute the SAME
            # Beta posterior from summed counts, rather than unioning the
            # constituent intervals — the union ran [41%,100%], technically
            # true and operationally useless. Actionable only if every
            # constituent survived on its own; a pooled cell was never the
            # unit that FDR tested.
            from scipy import stats
            n = int(sel['n'].sum())
            hits = int(sel['hits'].sum()) if 'hits' in sel else None
            glob = float(sel['glob'].iloc[0]) if 'glob' in sel else 0.5
            if hits is None:
                p = float((sel['post'] * sel['n']).sum() / max(n, 1))
                lo = hi = float('nan')
            else:
                a0, b0 = glob * 20.0, (1 - glob) * 20.0
                p = (a0 + hits) / (a0 + b0 + n)
                lo, hi = stats.beta.ppf([0.025, 0.975], a0 + hits,
                                        b0 + n - hits)
            return Answer(event, question, p, lo, hi, n,
                          int(sel['days'].sum()) if 'days' in sel else 0,
                          bool(sel['actionable'].all()), dropped,
                          basis='POOLED')
        # back off the least informative dimension still in play
        nxt = next((d for d in BACKOFF_ORDER if d in use), None)
        if nxt is None:
            break
        use.pop(nxt)
        dropped.append(nxt)
    # nothing matched — the corpus-wide rate, honestly labelled
    tot = int(t['n'].sum())
    p = float((t['post'] * t['n']).sum() / max(tot, 1))
    return Answer(event, question, p, float('nan'), float('nan'), tot,
                  int(t['days'].max()) if 'days' in t else 0, False,
                  dropped, basis='BASE')


def available():
    out = []
    for p in sorted(glob.glob(os.path.join(TABLES, '*.parquet'))):
        ev, q, _ = os.path.basename(p)[:-8].split('__')
        out.append((ev, q))
    return out


if __name__ == '__main__':
    print('tables:', available(), '\n')
    for ctx in (dict(kind='RETURN', dir_s='dn', depth_b='<=0.5',
                     age_b='30m+', clock_b='0930'),
                dict(kind='BREAKOUT', dir_s='up', depth_b='1-2',
                     age_b='<5m', clock_b='1200'),
                dict(kind='RETURN', dir_s='dn', depth_b='<=0.5',
                     age_b='30m+', clock_b='2359')):     # nonexistent bucket
        print(lookup('fakeout_poke', 'exceed_ref_first', **ctx).line())
    print(lookup('stall', 'race', dir_s='up', clock_b='1200').line())
    print(lookup('leg_descent', 'race', chain_b='3+', clock_b='1030').line())
