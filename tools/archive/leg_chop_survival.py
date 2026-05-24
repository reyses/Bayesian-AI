"""Conditional probability table — entry 2: does EARLY underlying chop forecast
a long wide leg?

User question (2026-05-21): in a long (wide-ATR) leg with underlying chop at a
lower ATR, how much does the chop determine the leg's length / continuation?
Diagnostic, not a trading model.

v1 (confounded, superseded) measured chop over the WHOLE leg — but the chop
ratio co-grows with the leg's tight-leg count, which co-grows with its length,
so "choppy legs are longer" was largely tautological. This v2 fixes that:

  CHOP is measured over a FIXED EARLY WINDOW — the first K tight-ATR legs of
  the wide leg only. early_chop_ratio(K) = (path length of the first K tight
  legs) / (net displacement over those K legs). This is decoupled from the
  wide leg's eventual length, and it is causal (the first K legs are early).

  OUTCOME = the wide leg's eventual total tight-leg count C (its length), and
  P(C >= K + 5) — i.e. P(it runs at least 5 more tight legs past the window).

If early chop genuinely forecasts a longer leg, P(C >= K+5) rises across the
early-chop bands. If it is flat, early chop carries no length information.

Wide legs at --wide-mult (default 8), tight legs at --tight-mult (default 1).
Windows K swept (3, 5). IS + OOS, n + 95% bootstrap CI per cell.

Output: reports/findings/oos_bad_days/2026-05-21_leg_chop_survival.md
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.viz.auto_swing_marker import detect_swings

REPO = Path(__file__).resolve().parent.parent
IS_LEGS = REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv'
OOS_LEGS = REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv'
RAW_NT8 = REPO / 'DATA/ATLAS_NT8'
RAW_ATLAS = REPO / 'DATA/ATLAS'
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_leg_chop_survival.md'
TICK = 0.25
ATR_PERIOD = 14
MIN_BARS = 36
EPS = 1.0          # points — floor for the net-move denominator
WINDOWS = [3, 5]   # fixed early-window sizes (tight legs)
RUN_ON = 5         # "long" margin: P(C >= K + RUN_ON)
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


def day_records(day: str, wide_mult: float, tight_mult: float):
    """Per wide leg: a list, one entry per nested tight leg in order, of
    (cumulative tight path, net displacement from wide-leg start)."""
    f5, f1 = bars_path(day, '5s'), bars_path(day, '1m')
    if not f5.exists() or not f1.exists():
        return []
    b5 = pd.read_parquet(f5).sort_values('timestamp').reset_index(drop=True)
    b1 = pd.read_parquet(f1).sort_values('timestamp').reset_index(drop=True)
    closes = b5['close'].values.astype(float)
    atr = compute_atr(b1)
    wmin = max(4, int(round(atr / TICK * wide_mult)))
    tmin = max(4, int(round(atr / TICK * tight_mult)))
    wpiv = detect_swings(closes, min_reversal=wmin, min_bars=MIN_BARS, max_bars=0)
    tpiv = detect_swings(closes, min_reversal=tmin, min_bars=MIN_BARS, max_bars=0)
    tlegs = [(tpiv[k], tpiv[k + 1]) for k in range(len(tpiv) - 1)]
    recs = []
    for k in range(len(wpiv) - 1):
        ws, we = wpiv[k], wpiv[k + 1]
        nested = [(a, b) for (a, b) in tlegs if ws <= a < we]
        if not nested:
            continue
        cum, seq = 0.0, []
        for a, b in nested:
            cum += abs(closes[b] - closes[a])
            seq.append((cum, abs(closes[b] - closes[ws])))
        recs.append(seq)
    return recs


def collect(days, wide_mult, tight_mult):
    out = []
    for d in days:
        out.extend(day_records(str(d), wide_mult, tight_mult))
    return out


def window_rows(recs, K):
    """For wide legs with >= K tight legs: (early_chop_ratio over first K,
    total tight-leg count C)."""
    rows = []
    for seq in recs:
        if len(seq) < K:
            continue
        cum_k, net_k = seq[K - 1]
        rows.append((cum_k / max(net_k, EPS), len(seq)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--wide-mult', type=float, default=8.0)
    ap.add_argument('--tight-mult', type=float, default=1.0)
    args = ap.parse_args()
    W, T = args.wide_mult, args.tight_mult

    is_days = sorted(pd.read_csv(IS_LEGS)['day'].astype(str).unique())
    oos_days = sorted(pd.read_csv(OOS_LEGS)['day'].astype(str).unique())
    is_rec = collect(is_days, W, T)
    oos_rec = collect(oos_days, W, T)

    L = []
    def o(s=''):
        L.append(s)

    o('# Leg-Chop Survival v2 — does EARLY chop forecast a long wide leg?')
    o('')
    o(f'Wide legs = zigzag at ATR x{W:g}; underlying chop = nested zigzag legs '
      f'at ATR x{T:g}. **Fixed early window**: chop is measured over only the '
      f'first K tight legs of the wide leg — `early_chop_ratio` = path of '
      f'those K legs / their net displacement (~1 clean, >>1 choppy). This is '
      f'decoupled from the wide leg\'s eventual length (the v1 confound). '
      f'Outcome = total tight-leg count C; "runs long" = C >= K+{RUN_ON}.')
    o(f'IS {len(is_rec):,} wide legs / {len(is_days)} days; '
      f'OOS {len(oos_rec):,} / {len(oos_days)} days.')
    o('')

    for K in WINDOWS:
        is_rows = window_rows(is_rec, K)
        oos_rows = window_rows(oos_rec, K)
        if not is_rows or not oos_rows:
            continue
        is_ratio = np.array([r[0] for r in is_rows])
        q33, q67 = np.percentile(is_ratio, [100 / 3.0, 200 / 3.0])

        def band(x):
            return 'clean' if x < q33 else ('choppy' if x >= q67 else 'medium')

        o(f'## Window K = {K} tight legs')
        o('')
        for label, rows in [('IS', is_rows), ('OOS', oos_rows)]:
            ratio = np.array([r[0] for r in rows])
            C = np.array([r[1] for r in rows], dtype=float)
            corr = float(np.corrcoef(ratio, C)[0, 1]) if len(rows) > 2 else float('nan')
            o(f'**{label}** — {len(rows):,} wide legs reached >={K} tight legs.  '
              f'corr(early chop ratio, total length C) = {corr:+.3f}')
        o('')
        o(f'| set | early-chop band | n | median C | P(C>={K + RUN_ON}) | 95% CI |')
        o('|---|---|--:|--:|--:|---|')
        for label, rows in [('IS', is_rows), ('OOS', oos_rows)]:
            for bnd in ('clean', 'medium', 'choppy'):
                sub = [r for r in rows if band(r[0]) == bnd]
                if not sub:
                    continue
                C = np.array([r[1] for r in sub], dtype=float)
                runs = (C >= K + RUN_ON).astype(float)
                lo, hi = bootstrap_ci(runs)
                o(f'| {label} | {bnd} | {len(sub):,} | {np.median(C):.0f} | '
                  f'{runs.mean() * 100:.0f}% | '
                  f'[{lo * 100:.0f}%, {hi * 100:.0f}%] |')
        o('')

    o('## Read')
    o('')
    o('- `early_chop_ratio` is measured over a FIXED count of tight legs, so '
      'it is NOT mechanically tied to the wide leg\'s eventual length — '
      'unlike the confounded v1.')
    o('- If `P(C>=K+N)` and the corr are flat / ~0 across the early-chop '
      'bands, **early chop does not forecast leg length** — the chop content '
      'carries no continuation information.')
    o('- If `choppy` early shows a higher P than `clean`, an early-choppy '
      'wide leg genuinely tends to run longer (and vice versa). Trust it only '
      'where IS and OOS agree.')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
