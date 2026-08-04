"""BAYESIAN TABLE ACTUARY (owner 2026-08-04, overnight order: "run the full
2024-2025, no epoch, Bayesian table").

No gradient descent, no epochs. For every named event, split the corpus by
CAUSAL context and COUNT what happened, with a proper posterior on each cell:

    cell posterior = Beta(a0 + hits, b0 + misses)

where the prior (a0, b0) is the EVENT'S GLOBAL RATE scaled by PRIOR_STRENGTH
— i.e. hierarchical shrinkage: a thin cell is pulled toward the event's own
base rate instead of screaming from 3 samples. Reported per cell:
  n, raw rate, posterior mean, 95% credible interval, and LIFT vs the event's
  global rate with the CI on that lift.

A cell is only ACTIONABLE if its 95% CI excludes the global rate. Everything
else is explicitly labelled NOT DISTINGUISHABLE — the table is honest about
what it does not know, which is the whole point of an actuary.

Context dimensions are chosen per event from fields the detector already
stamps CAUSALLY (nothing forward-looking). Day-shape stays welded on, per the
2026-08-04 finding that abstracting it away destroyed the shelf edge.

Run from repo root:
  python research/bayes_tables/builders/build_tables.py
Writes research/bayes_tables/tables/<event>.parquet + reports/tables_v0.md
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
EVENTS_DIR = os.path.join(REPO, 'research', 'event_library', 'events')
OUT_DIR = os.path.join(REPO, 'research', 'bayes_tables')
EXCLUDE_DAYS = {'2024_09_16'}
PRIOR_STRENGTH = 20.0     # pseudo-counts of the global rate on every cell
MIN_CELL = 15             # below this a cell is reported but never actionable
MIN_DAYS = 25             # audit: n=15/9-day cells produced ZERO-WIDTH
                          # bootstrap CIs and floored p-values that walked
                          # straight through BH as 'actionable'
BOOT = 4000               # day-clustered bootstrap draws (repo standard)
FDR_Q = 0.05              # Benjamini-Hochberg false-discovery rate
CI = 0.95

# (event, outcome column, positive value, context dimensions)
SPECS = [
    # v1 dims were re-derivations of the volatility clock (audit 2026-08-04).
    # These are the fields the audit proved dominant, all emitted AT the
    # event bar (no lookahead): position-in-box, giveback fraction, defense
    # size, bounce size.
    # exceed_ref_first is the BOUNDED question — "does the poke clear the
    # level before a 10pt adverse move" (global 0.78, BREAKOUT 0.905 vs
    # RETURN 0.666). The unbounded exceed_ref sits at 0.95 for everything and
    # asks nothing worth asking.
    ('fakeout_poke', 'exceed_ref_first', True,
     ['kind', 'dir_s', 'depth_b', 'age_b', 'clock_b']),
    ('fakeout_poke', 'sym_race', 'CONT',
     ['kind', 'dir_s', 'depth_b', 'clock_b']),
    ('leg_descent', 'race', 'NEW_LOW', ['defense_b', 'chain_b']),
    ('stall', 'race', 'NEW_EXTREME', ['give_b', 'dir_s']),
    ('ultra_chop', 'escape_dir', 1, ['midbox_b', 'ratio_b']),
    ('defended_poke_shelf', 'outcome', 'CRACK',
     ['bounce_b', 'day_class']),
]


def bucket_clock(ts):
    et = pd.to_datetime(ts, unit='s', utc=True).dt.tz_convert('America/New_York')
    m = et.dt.hour * 60 + et.dt.minute
    return pd.cut(m, [-1, 600, 630, 720, 840, 1e4],
                  labels=['0930', '1000', '1030', '1200', '1400'])


def add_context(df, event):
    out = df.copy()
    out['clock_b'] = bucket_clock(out['ts'])
    if 'dir' in out:
        out['dir_s'] = np.where(out['dir'] > 0, 'up', 'dn')
    if 'poke_depth' in out:
        # BREAKOUT pokes run to 80+pt past the reference — the original
        # <=2pt buckets silently dropped EVERY breakout row (51% coverage,
        # and every 'strongest cell' was a RETURN by construction).
        out['depth_b'] = pd.cut(out['poke_depth'],
                                [-0.01, 0.5, 1.0, 2.01, 5, 15, 1e6],
                                labels=['<=0.5', '0.5-1', '1-2', '2-5',
                                        '5-15', '15+'])
    if 'ref_age_s' in out:
        out['age_b'] = pd.cut(out['ref_age_s'], [-1, 300, 1800, 1e9],
                              labels=['<5m', '5-30m', '30m+'])
    if 'box_ambient_ratio' in out:
        out['ratio_b'] = pd.cut(out['box_ambient_ratio'],
                                [-0.01, 0.4, 0.5, 0.61],
                                labels=['tight', 'mid', 'loose'])
    if {'mid_px', 'box_lo', 'box_hi'} <= set(out.columns):
        # WHERE IN THE BOX the tape sits when chop is stamped — the audit
        # showed this runs P(up) 0.36 -> 0.66 (AUC 0.62) while the clock
        # dims I shipped were flat. Causal: all three fields are stamped at
        # the event bar.
        rel = ((out['mid_px'] - out['box_lo'])
               / (out['box_hi'] - out['box_lo']).replace(0, np.nan))
        out['midbox_b'] = pd.qcut(rel, 5,
                                  labels=['q1_low', 'q2', 'q3', 'q4',
                                          'q5_high'], duplicates='drop')
    if 'give_frac' in out:
        out['give_b'] = pd.qcut(out['give_frac'], 5,
                                labels=['g1', 'g2', 'g3', 'g4', 'g5'],
                                duplicates='drop')
    if 'defense_pt' in out:
        out['defense_b'] = pd.qcut(out['defense_pt'], 5,
                                   labels=['d1', 'd2', 'd3', 'd4', 'd5'],
                                   duplicates='drop')
    if 'bounce_pt' in out:
        out['bounce_b'] = pd.qcut(out['bounce_pt'], 4,
                                  labels=['b1', 'b2', 'b3', 'b4'],
                                  duplicates='drop')
    if 'chain_n' in out:
        out['chain_b'] = pd.cut(out['chain_n'], [-1, 1, 2, 99],
                                labels=['1', '2', '3+'])
    return out


def posterior(hits, n, glob):
    """Beta posterior with the global rate as a PRIOR_STRENGTH-weight prior."""
    a0, b0 = glob * PRIOR_STRENGTH, (1 - glob) * PRIOR_STRENGTH
    a, b = a0 + hits, b0 + (n - hits)
    lo, hi = stats.beta.ppf([(1 - CI) / 2, 1 - (1 - CI) / 2], a, b)
    return a / (a + b), lo, hi


def build(event, col, positive, dims):
    path = os.path.join(EVENTS_DIR, f'{event}.parquet')
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df = df[~df['day'].isin(EXCLUDE_DAYS)]
    if col not in df.columns:
        return None
    df = df[df[col].notna()].copy()
    if not len(df):
        return None
    df = add_context(df, event)
    dims = [d for d in dims if d in df.columns]
    if not dims:
        return None
    df['_hit'] = (df[col] == positive).astype(int)
    glob = float(df['_hit'].mean())
    rng = np.random.default_rng(20260804)
    rows = []
    for key, g in df.groupby(dims, observed=True, dropna=True):
        key = key if isinstance(key, tuple) else (key,)
        n, hits = len(g), int(g['_hit'].sum())
        pm, lo, hi = posterior(hits, n, glob)
        # DAY-CLUSTERED bootstrap: events inside one session are correlated,
        # so the iid Beta interval is too narrow. Resample DAYS, not events.
        dlo, dhi, p_emp = np.nan, np.nan, 1.0
        if n >= MIN_CELL:
            by_day = g.groupby('day')['_hit'].agg(['sum', 'count'])
            sd, cd = by_day['sum'].to_numpy(), by_day['count'].to_numpy()
            nd = len(sd)
            if nd >= 5:
                pick = rng.integers(0, nd, size=(BOOT, nd))
                bs = sd[pick].sum(1) / np.maximum(cd[pick].sum(1), 1)
                dlo, dhi = np.percentile(bs, [2.5, 97.5])
                # two-sided empirical p vs the global rate
                p_emp = 2 * min((bs <= glob).mean(), (bs >= glob).mean())
                p_emp = float(min(1.0, max(p_emp, 1.0 / BOOT)))
        rows.append({**dict(zip(dims, [str(k) for k in key])),
                     'n': n, 'hits': hits, 'days': int(g['day'].nunique()),
                     'raw': round(hits / n, 4),
                     'post': round(pm, 4), 'lo': round(lo, 4),
                     'hi': round(hi, 4),
                     'day_lo': round(float(dlo), 4) if dlo == dlo else None,
                     'day_hi': round(float(dhi), 4) if dhi == dhi else None,
                     'p': p_emp,
                     'lift': round(pm - glob, 4)})
    t = pd.DataFrame(rows).sort_values('n', ascending=False)
    # BENJAMINI-HOCHBERG across every cell in this table: 108 cells at 95%
    # would hand back ~5 false 'actionable' by luck alone.
    m = len(t)
    order = np.argsort(t['p'].to_numpy())
    thresh = np.zeros(m, bool)
    pv = t['p'].to_numpy()[order]
    crit = FDR_Q * (np.arange(1, m + 1) / m)
    passed = np.flatnonzero(pv <= crit)
    if len(passed):
        thresh[order[:passed[-1] + 1]] = True
    t['actionable'] = (thresh & (t['n'] >= MIN_CELL)
                       & (t['days'] >= MIN_DAYS))
    t.attrs['global'] = glob
    t['glob'] = round(glob, 6)      # pooling needs the prior that built it
    os.makedirs(os.path.join(OUT_DIR, 'tables'), exist_ok=True)
    t.to_parquet(os.path.join(OUT_DIR, 'tables',
                              f'{event}__{col}__{positive}.parquet'),
                 index=False)
    return t, glob, len(df)


if __name__ == '__main__':
    os.makedirs(os.path.join(OUT_DIR, 'reports'), exist_ok=True)
    lines = ['# BAYESIAN TABLE ACTUARY v0', '',
             'Counted, not trained. Every cell is a Beta posterior whose prior '
             f'is the event\'s own global rate with {PRIOR_STRENGTH:.0f} '
             'pseudo-counts (hierarchical shrinkage), so a thin cell is pulled '
             'toward the base rate rather than shouting from 3 samples. A cell '
             'is ACTIONABLE only when it survives BOTH a day-clustered '
             'bootstrap (4,000 draws resampling DAYS, because events inside '
             'one session are correlated and the iid Beta interval is too '
             'narrow) AND Benjamini-Hochberg FDR control at q=0.05 across '
             f'every cell in the table, with n >= {MIN_CELL}. '
             'Live sim day excluded.', '']
    for event, col, positive, dims in SPECS:
        r = build(event, col, positive, dims)
        if r is None:
            lines += [f'## {event} / {col}=={positive} — NO DATA', '']
            continue
        t, glob, n = r
        act = t[t['actionable']]
        lines += [f'## {event} — P({col} == {positive})', '',
                  f'N = {n:,} events, global rate {glob:.3f}, '
                  f'{len(t)} cells, {len(act)} ACTIONABLE', '']
        # raw sits next to its OWN interval (the bootstrap resamples raw
        # counts); post is the shrunk estimate and is NOT what day_lo/day_hi
        # bracket — the audit caught 7 cells where post fell outside them.
        show = t.head(12)[[*dims, 'n', 'days', 'raw', 'day_lo', 'day_hi',
                           'post', 'lift', 'actionable']]
        lines += [show.to_string(index=False), '']
        if len(act):
            top = act.reindex(act['lift'].abs().sort_values(
                ascending=False).index).head(6)
            lines += ['**Strongest actionable cells (|lift| vs global):**', '',
                      top[[*dims, 'n', 'days', 'raw', 'day_lo', 'day_hi',
                           'post', 'lift', 'p']].to_string(index=False), '']
        else:
            lines += ['**No cell separates from the global rate** — this '
                      'question is answered by the base rate alone.', '']
        print(f'{event}/{col}: N={n} global={glob:.3f} cells={len(t)} '
              f'actionable={len(act)}')
    open(os.path.join(OUT_DIR, 'reports', 'tables_v0.md'),
         'w').write('\n'.join(lines) + '\n')
    print('\nwrote', os.path.join(OUT_DIR, 'reports', 'tables_v0.md'))
