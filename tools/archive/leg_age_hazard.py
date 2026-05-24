"""Conditional probability table -- entry 4: leg-age exhaustion hazard.

Given a leg has already run M minutes, what is the probability it continues
another X minutes before the pivot?  The "is this move expended?" question
posed as a clean conditional-survival curve.

Object = hardened FLAT-forward-pass legs (the tradeable leg: R-trigger entry
-> next-pivot exit).  Leg age = SECONDS SINCE THE R-TRIGGER ENTRY -- the clock
a live trade actually has.  A leg "survives" to age m if its trade duration
D >= m; it "continues another X" if D >= m + X.

  S(X | m) = P(D >= m + X | D >= m)            conditional survival
  h(m)     = P(m <= D < m + bin | D >= m)      hazard over the next bin

SHAPE TEST -- watch S(X | m) for a FIXED X as the age m grows:
  - S(X|m) FALLS with m      -> legs EXHAUST (a time-stop has a job)
  - S(X|m) FLAT              -> MEMORYLESS (leg age carries no information)
  - S(X|m) RISES with m      -> momentum (old legs are the survivors)
  - S(X|m) FALLS then RISES  -> HUMP: a danger window at the trough, with
                                survivors past it persisting. Test the
                                fresh-vs-trough and aged-vs-trough ARMS --
                                an endpoint-only fresh-vs-aged test misses
                                the trough and mislabels a hump "exhaustion".

This is leg-INTERNAL -- one leg's own duration -- so there is NO zigzag-
alternation parity artifact (cf. entry 3, which had to be leg-decoupled).
TWO real caveats: (1) the zigzag uses min_bars=36 (a 3-min minimum zigzag
leg), so a hardened leg rarely dies below ~3 min -- the steep early hazard
ramp is partly this construction floor, not market momentum; (2) legs still
alive at a high age skew toward low-volatility overnight legs (a genuine
selection). Both are noted in the read.

IS + OOS, n + 95% bootstrap CI (resampled at the leg level, 4000 resamples).

Output: reports/findings/oos_bad_days/2026-05-21_leg_age_hazard.md
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
IS_LEGS = REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv'
OOS_LEGS = REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_leg_age_hazard.md'

N_BOOT = 4000
SEED = 42
AGE_GRID_MIN = [0.5, 1, 2, 3, 5, 8, 12, 18]   # leg-age rows
X_MIN = [2, 5]                                 # continuation horizons (min)
HAZ_BIN_MIN = 1.0                              # hazard bin width (min)
MIN_N_IS = 100                                 # at-risk-n floor for the verdict
MIN_N_OOS = 30
FLOOR_MIN = 3.0          # zigzag min_bars=36 -> 36 5s-bars -> 180s -> 3-min leg


def load_durations(path: Path) -> np.ndarray:
    """Trade durations (seconds) of every hardened leg with D > 0."""
    df = pd.read_csv(path)
    d = (df['exit_ts'].astype(float) - df['entry_ts'].astype(float)).to_numpy()
    return d[d > 0]


def survival(dur: np.ndarray, m: float, x: float):
    """Point estimate of S(x|m) = P(D >= m+x | D >= m), plus n at risk."""
    denom = int((dur >= m).sum())
    if denom == 0:
        return float('nan'), 0
    return float((dur >= m + x).sum()) / denom, denom


def boot_curve(dur, ages_s, xs_s, n=N_BOOT, seed=SEED):
    """Leg-level bootstrap of the whole survival surface.
    Returns acc[(m,x)] = array(n) of resampled S(x|m) values."""
    rng = np.random.default_rng(seed)
    nn = len(dur)
    acc = {(m, x): np.empty(n) for m in ages_s for x in xs_s}
    thr = sorted({m for m in ages_s} | {m + x for m in ages_s for x in xs_s})
    for b in range(n):
        bs = np.sort(dur[rng.integers(0, nn, nn)])
        cnt = {t: nn - int(np.searchsorted(bs, t, 'left')) for t in thr}
        for m in ages_s:
            denom = cnt[m]
            for x in xs_s:
                acc[(m, x)][b] = cnt[m + x] / denom if denom else np.nan
    return acc


def ci(v):
    return (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))


def pct(x):
    return f'{x * 100:.0f}%' if np.isfinite(x) else '—'


def main():
    try:                                  # Windows console is cp1252
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    is_dur = load_durations(IS_LEGS)
    oos_dur = load_durations(OOS_LEGS)
    ages_s = [a * 60.0 for a in AGE_GRID_MIN]
    xs_s = [x * 60.0 for x in X_MIN]

    is_acc = boot_curve(is_dur, ages_s, xs_s)
    oos_acc = boot_curve(oos_dur, ages_s, xs_s)

    L = []
    o = L.append

    o('# Leg-age exhaustion hazard — is the move expended? (table entry 4)')
    o('')
    o('Given a leg has already run **M minutes**, P(it continues another X '
      'min before the pivot). Object = hardened FLAT legs (R-trigger entry '
      '→ next-pivot exit); age = seconds since entry — the clock a '
      'live trade has. `S(X|m) = P(D >= m+X | D >= m)`. Leg-internal, so no '
      'zigzag-alternation parity artifact.')
    o('')

    for tag, d in (('IS', is_dur), ('OOS', oos_dur)):
        q = np.percentile(d, [25, 50, 75, 90])
        o(f'**{tag} duration** (n={len(d):,}): median {q[1] / 60:.1f} min, '
          f'mean {d.mean() / 60:.1f}, p25 {q[0] / 60:.1f}, p75 {q[2] / 60:.1f}'
          f', p90 {q[3] / 60:.1f}, max {d.max() / 60:.0f}.')
    o('')

    for x_min, x_s in zip(X_MIN, xs_s):
        o(f'## P(leg continues >= +{x_min:g} min) by current age M')
        o('')
        o('| age M (min) | n(IS) | IS P(continues) [CI] | n(OOS) '
          '| OOS P(continues) [CI] |')
        o('|--:|--:|---|--:|---|')
        for m_min, m_s in zip(AGE_GRID_MIN, ages_s):
            is_s, is_n = survival(is_dur, m_s, x_s)
            oos_s, oos_n = survival(oos_dur, m_s, x_s)
            il, ih = ci(is_acc[(m_s, x_s)])
            ol, oh = ci(oos_acc[(m_s, x_s)])
            o(f'| {m_min:g} | {is_n:,} | {pct(is_s)} [{pct(il)}, {pct(ih)}] '
              f'| {oos_n:,} | {pct(oos_s)} [{pct(ol)}, {pct(oh)}] |')
        o('')

    o('## Hazard and median remaining life by age')
    o('')
    o(f'Hazard h(M) = P(leg ends within the next {HAZ_BIN_MIN:g} min | alive '
      f'at M). Median remaining = median(D − M) over legs alive at M.')
    o('')
    o('| age M (min) | IS h(M) | IS med. remaining (min) | OOS h(M) '
      '| OOS med. remaining (min) |')
    o('|--:|--:|--:|--:|--:|')
    haz_s = HAZ_BIN_MIN * 60.0
    for m_min, m_s in zip(AGE_GRID_MIN, ages_s):
        row = [f'{m_min:g}']
        for d in (is_dur, oos_dur):
            ar = d[d >= m_s]
            if len(ar):
                h = (ar < m_s + haz_s).sum() / len(ar)
                medr = np.median(ar - m_s) / 60.0
                row += [pct(h), f'{medr:.1f}']
            else:
                row += ['—', '—']
        o('| ' + ' | '.join(row) + ' |')
    o('')

    healthy = [(mm, ms) for mm, ms in zip(AGE_GRID_MIN, ages_s)
               if (is_dur >= ms).sum() >= MIN_N_IS
               and (oos_dur >= ms).sum() >= MIN_N_OOS]
    o('## Exhaustion verdict')
    o('')
    if len(healthy) >= 3:
        ref_x, ref_min = xs_s[0], X_MIN[0]
        is_pt = {ms: survival(is_dur, ms, ref_x)[0] for _, ms in healthy}
        tr_min, tr_s = min(healthy, key=lambda p: is_pt[p[1]])
        hz = {}
        for mm, ms in healthy:
            ar = is_dur[is_dur >= ms]
            hz[mm] = (ar < ms + haz_s).sum() / len(ar) if len(ar) else 0.0
        pk_min = max(hz, key=hz.get)
        o(f'Reference horizon +{ref_min:g} min. IS hazard peaks at '
          f'**{pk_min:g} min** ({hz[pk_min] * 100:.0f}%/min); IS continuation '
          f'troughs at **{tr_min:g} min** — call ~{tr_min:g} min the danger '
          f'window. The shape is read with two ARMS off that trough, not an '
          f'endpoint fresh-vs-aged test (which would skip the trough).')
        o('')
        (fr_min, fr_s), (la_min, la_s) = healthy[0], healthy[-1]
        arms = {}
        for tag, acc in (('IS', is_acc), ('OOS', oos_acc)):
            arms[tag] = {}
            for nm, am, asec in (('early', fr_min, fr_s),
                                 ('late', la_min, la_s)):
                if asec == tr_s:
                    arms[tag][nm] = None
                    continue
                diff = acc[(asec, ref_x)] - acc[(tr_s, ref_x)]
                d0 = float(np.nanmean(diff))
                dl, dh = ci(diff)
                sig = dl > 0 or dh < 0
                arms[tag][nm] = (am, d0, dl, dh, sig)
                o(f'- **{tag} {nm} arm** — S(+{ref_min:g}m | {am:g}m) − '
                  f'S(+{ref_min:g}m | {tr_min:g}m) = {d0 * 100:+.0f}pp '
                  f'[{dl * 100:+.0f}, {dh * 100:+.0f}]'
                  f'{"  SIGNIFICANT" if sig else "  not sig"}.')
        o('')
        above = [(mm, ms) for mm, ms in healthy if mm >= FLOOR_MIN]
        if len(above) >= 2:
            (ab_min, ab_s), (al_min, al_s) = above[0], above[-1]
            for tag, acc in (('IS', is_acc), ('OOS', oos_acc)):
                diff = acc[(ab_s, ref_x)] - acc[(al_s, ref_x)]
                d0 = float(np.nanmean(diff))
                dl, dh = ci(diff)
                sig = dl > 0 or dh < 0
                o(f'- **{tag} above-floor** — S(+{ref_min:g}m | {ab_min:g}m) '
                  f'− S(+{ref_min:g}m | {al_min:g}m) = {d0 * 100:+.0f}pp '
                  f'[{dl * 100:+.0f}, {dh * 100:+.0f}]'
                  f'{"  SIGNIFICANT" if sig else "  not sig"} '
                  f'(positive = exhaustion persists above the 3-min floor).')
            o('')

        def both(nm):
            a, b = arms['IS'].get(nm), arms['OOS'].get(nm)
            return bool(a and a[4] and a[1] > 0 and b and b[4] and b[1] > 0)

        if both('early') and both('late'):
            o(f'**HUMP-SHAPED HAZARD — NOT monotone exhaustion.** '
              f'Continuation is high for a fresh leg, troughs at the '
              f'~{tr_min:g}-min danger window, then RECOVERS for aged legs — '
              f'both arms significant in IS and OOS. An endpoint-only '
              f'"fresh vs aged" test would skip the trough and mislabel '
              f'this "exhaustion". **min_bars caveat**: the zigzag enforces '
              f'a 3-min minimum leg, so a hardened leg rarely dies below '
              f'~{FLOOR_MIN:g} min — the steep early arm is partly that '
              f'construction floor, not market momentum. The clean, '
              f'non-structural signal is the danger window itself and the '
              f'recovery past it: a leg that clears ~{tr_min:g} min becomes '
              f'MORE persistent, not less. A pure "old leg = cut it" '
              f'time-stop is unsupported; only a danger-window-aware check '
              f'is backed by the data.')
        elif both('early'):
            o('**MONOTONE EXHAUSTION** — continuation falls with age and '
              'does not significantly recover. The early arm is partly '
              'contaminated by the min_bars 3-min floor — read the '
              'above-floor test for the non-structural picture.')
        elif both('late'):
            o('**MONOTONE MOMENTUM** — continuation rises with age; older '
              'legs are the survivors. Cutting on age alone would be wrong.')
        else:
            o('**NO RELIABLE AGE EFFECT** — neither arm is significant in '
              'both IS and OOS; leg duration is ~memoryless over this range '
              'and a pure time-based exit is not supported.')
    else:
        o('- Too few ages clear the at-risk-n floor to run the verdict.')
    o('')

    o('## Read')
    o('')
    o('- Compare S(X|m) DOWN each table column. Falling = exhaustion; flat = '
      'memoryless; rising = momentum; falls-then-rises = a HUMP (a danger '
      'window). The hazard h(M) is the same story per-bin.')
    o('- Trust a row only where IS and OOS agree.')
    o('- min_bars caveat: the zigzag enforces a 3-min minimum leg '
      '(min_bars=36 5s-bars), so a hardened leg rarely dies below ~3 min — '
      'the low early hazard / steep early ramp is partly this construction '
      'floor, NOT pure market momentum. Read the above-floor test for the '
      'non-structural picture.')
    o('- Selection caveat: legs still alive at a high age skew toward '
      'low-volatility overnight legs — a real part of the conditional '
      'law, but it means a "still alive at 18 min" leg is not the same '
      'animal as a fresh RTH leg.')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
