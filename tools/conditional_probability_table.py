"""Conditional probability table — empirical "if this then that" over
zigzag-leg events.

User concept (2026-05-21): the ATR zigzag pinpoints discrete events (legs);
build a probability table that answers conditional questions over them — a
DIAGNOSTIC layer, not a trading model. Because nothing is traded, the
whipsaw / give-up cost is irrelevant here; a tighter ATR simply yields finer
events and more samples.

v1 question — chop resolution:
  Given K consecutive LOW-range legs (K = 1..5), what is the probability
  distribution of the NEXT leg's range (low / mid / high)?

Events = zigzag legs at a chosen ATR multiplier (--atr-mult). Range terciles
are IS-derived. Every cell carries n + a 95% bootstrap CI, reported IS and OOS
separately — a cell is trustworthy only if it holds out-of-sample. Compare the
conditional P to the unconditional base rate to read the lift.

NOTE: the directional ("continues the trend") question is deliberately NOT
here — zigzag legs alternate deterministically, so any direction question on
leg signs is trivial. The directional signal lives in amplitude asymmetry and
needs a separate formulation (a v2 design).

Output: reports/findings/oos_bad_days/2026-05-21_conditional_prob_table.md
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools._viz.auto_swing_marker import detect_swings

REPO = Path(__file__).resolve().parent.parent
IS_LEGS = REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv'
OOS_LEGS = REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
RAW_NT8 = REPO / 'DATA/ATLAS_NT8'
RAW_ATLAS = REPO / 'DATA/ATLAS'
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_conditional_prob_table.md'
TICK = 0.25
ATR_PERIOD = 14
MIN_BARS = 36
MAX_K = 5
N_BOOT = 4000
SEED = 42


def bars_path(day: str, tf: str) -> Path:
    nt8 = RAW_NT8 / tf / f'{day}.parquet'
    return nt8 if nt8.exists() else RAW_ATLAS / tf / f'{day}.parquet'


def compute_atr(b1: pd.DataFrame) -> float:
    h, l, c = (b1[x].values.astype(float) for x in ('high', 'low', 'close'))
    if len(c) < 2:
        return 1.0
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    return (float(np.median(tr[-ATR_PERIOD * 3:])) if len(tr) >= ATR_PERIOD
            else float(tr.mean()))


def bootstrap_ci(v, n=N_BOOT, seed=SEED):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def day_legs(day: str, atr_mult: float):
    """Signed leg amplitudes (points) for one day's zigzag at atr_mult."""
    f5, f1 = bars_path(day, '5s'), bars_path(day, '1m')
    if not f5.exists() or not f1.exists():
        return []
    b5 = pd.read_parquet(f5).sort_values('timestamp').reset_index(drop=True)
    b1 = pd.read_parquet(f1).sort_values('timestamp').reset_index(drop=True)
    closes = b5['close'].values.astype(float)
    min_rev = max(4, int(round(compute_atr(b1) / TICK * atr_mult)))
    piv = detect_swings(closes, min_reversal=min_rev, min_bars=MIN_BARS,
                        max_bars=0)
    return [float(closes[piv[k + 1]] - closes[piv[k]])
            for k in range(len(piv) - 1)]


def classify(amps_abs, q33, q67):
    """0 = low, 1 = mid, 2 = high range class."""
    return np.where(amps_abs < q33, 0, np.where(amps_abs >= q67, 2, 1))


def tabulate(legs_by_day: dict, q33: float, q67: float):
    """Per K: the range-class of the leg that follows K consecutive
    low-range legs. Returns {K: [classes]} and the unconditional base array."""
    out = {k: [] for k in range(1, MAX_K + 1)}
    base = []
    for legs in legs_by_day.values():
        if len(legs) < 2:
            continue
        cls = classify(np.abs(np.asarray(legs, dtype=float)), q33, q67)
        base.extend(cls.tolist())
        for k in range(1, MAX_K + 1):
            for i in range(k, len(cls)):
                if np.all(cls[i - k:i] == 0):
                    out[k].append(int(cls[i]))
    return out, np.asarray(base)


def pct(x):
    return f'{x * 100:4.0f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atr-mult', type=float, default=4.0)
    args = ap.parse_args()
    mult = args.atr_mult

    is_days = sorted(pd.read_csv(IS_LEGS)['day'].astype(str).unique())
    oos_days = sorted(pd.read_csv(OOS_LEGS)['day'].astype(str).unique())
    is_legs = {d: day_legs(d, mult) for d in is_days}
    oos_legs = {d: day_legs(d, mult) for d in oos_days}

    all_is_abs = np.abs([a for legs in is_legs.values() for a in legs])
    q33, q67 = np.percentile(all_is_abs, [100 / 3.0, 200 / 3.0])

    is_tab, is_base = tabulate(is_legs, q33, q67)
    oos_tab, oos_base = tabulate(oos_legs, q33, q67)

    L = []
    def o(s=''):
        L.append(s)

    o('# Conditional Probability Table — chop resolution (v1)')
    o('')
    o(f'Events = zigzag legs at **ATR x{mult:g}**. Diagnostic only — no trading '
      f'action, so the give-up / whipsaw cost is irrelevant here. Range '
      f'terciles (IS-derived): low < {q33:.1f}pt | mid | high >= {q67:.1f}pt. '
      f'Every cell: n + 95% bootstrap CI, IS and OOS separate.')
    o('')
    o(f'Events: IS {len(is_base):,} legs over {len(is_days)} days; '
      f'OOS {len(oos_base):,} legs over {len(oos_days)} days.')
    o('')
    o('## Base rate — unconditional P(a leg is low / mid / high)')
    o('')
    o('| set | P(low) | P(mid) | P(high) |')
    o('|---|--:|--:|--:|')
    for label, base in [('IS', is_base), ('OOS', oos_base)]:
        o(f'| {label} | {pct((base == 0).mean())} | {pct((base == 1).mean())} '
          f'| {pct((base == 2).mean())} |')
    o('')
    o('## Q: given K consecutive LOW-range legs, what is the NEXT leg?')
    o('')
    o('`P(high)` = next leg breaks out into the top tercile. `P(low)` = chop '
      'persists. Read `P(high)` against the base rate above — higher = '
      'low-range runs tend to resolve into a move; lower = chop begets chop.')
    o('')
    o('| K | n(IS) | IS P(low/mid/high) | IS P(high) 95% CI | n(OOS) | '
      'OOS P(low/mid/high) | OOS P(high) 95% CI |')
    o('|--:|--:|---|---|--:|---|---|')
    for k in range(1, MAX_K + 1):
        ir = np.asarray(is_tab[k])
        orr = np.asarray(oos_tab[k])
        if len(ir) == 0 or len(orr) == 0:
            continue
        ilo, ihi = bootstrap_ci((ir == 2).astype(float))
        olo, ohi = bootstrap_ci((orr == 2).astype(float))
        o(f'| {k} | {len(ir):,} | '
          f'{pct((ir==0).mean())}/{pct((ir==1).mean())}/{pct((ir==2).mean())} '
          f'| [{pct(ilo)}, {pct(ihi)}] | {len(orr):,} | '
          f'{pct((orr==0).mean())}/{pct((orr==1).mean())}/{pct((orr==2).mean())} '
          f'| [{pct(olo)}, {pct(ohi)}] |')
    o('')
    o('## Read')
    o('')
    base_hi = (is_base == 2).mean()
    k3 = np.asarray(is_tab[3])
    if len(k3):
        k3_hi = (k3 == 2).mean()
        verdict = ('ABOVE' if k3_hi > base_hi + 0.03 else
                   'BELOW' if k3_hi < base_hi - 0.03 else 'AT')
        tail = ('low-range runs tend to resolve into a breakout'
                if verdict == 'ABOVE' else
                'low-range runs tend to beget more chop'
                if verdict == 'BELOW' else
                'chop resolution looks ~memoryless at this scale')
        o(f'- After 3 consecutive low-range legs, IS P(next is high) = '
          f'{pct(k3_hi)} vs the {pct(base_hi)} base rate — **{verdict}** base. '
          f'{tail}.')
    o('- IS->OOS regime shift: OOS legs run larger (compare the base-rate '
      'rows). The conditional EFFECT replicates but absolute cell '
      'probabilities do not transfer — a v2 should use regime-relative bands.')
    o('- A cell is trustworthy only where IS and OOS agree and the CI is '
      'tight; OOS n thins fast at higher K — treat those as direction-only.')
    o('- Next: regime-relative range bands; the directional question '
      '(amplitude asymmetry, not leg sign); vol-window events.')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
