"""Zigzag cold-start divergence — does anchoring the streaming zigzag at a
mid-day point (vs the day's first bar) change the legs significantly?

The production zigzag (build_zigzag_pivot_dataset.py) runs detect_swings on
each day's FULL 5s closes — anchored at the day's first bar ("beginning of
time"). A live engine launched mid-session cold-starts its zigzag mid-day, at
a different anchor. Question (user, 2026-05-21): they WILL diverge — but
enough to matter?

Per day, per cold-start offset K (fraction of the day):
  reference  = detect_swings(closes[0:])    -- beginning-of-time
  coldstart  = detect_swings(closes[K:])    -- anchored at bar K
Compare the two pivot sequences over bars >= K:
  - resync lag : bars from K until the two sequences become an identical
                 tail (detect_swings is deterministic -> once they share a
                 pivot + matching suffix, they are identical forever after).
  - transient  : ref pivots in [K, resync) the cold-start mis-tiled.
  - $ proxy    : leg-amplitude captured (ref vs coldstart) over the transient
                 window, x $2/pt -- the value "at stake" while desynced.

Output: reports/findings/regret_oracle/2026-05-21_zigzag_coldstart_divergence.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.viz.auto_swing_marker import detect_swings

TICK_SIZE = 0.25
DOLLAR_PER_POINT = 2.0
ATR_PERIOD = 14
ATR_TR_WINDOW = ATR_PERIOD * 3      # 42 — median window, matches builder
ATR_MULT = 4.0
MIN_BARS = 36
MIN_REV_FLOOR = 4
K_FRACTIONS = [0.10, 0.25, 0.50, 0.75]
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 42
ATLAS_5S = Path('DATA/ATLAS_NT8/5s')
ATLAS_1M = Path('DATA/ATLAS_NT8/1m')


def bootstrap_ci(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def compute_min_rev(bars1m: pd.DataFrame) -> int:
    """ATR(14) median-of-42-TRs * 4 -> min_reversal in ticks. Matches
    build_zigzag_pivot_dataset.compute_atr."""
    h = bars1m['high'].values.astype(np.float64)
    l = bars1m['low'].values.astype(np.float64)
    c = bars1m['close'].values.astype(np.float64)
    if len(c) < 2:
        return MIN_REV_FLOOR
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    atr_pts = (float(np.median(tr[-ATR_TR_WINDOW:])) if len(tr) >= ATR_PERIOD
               else float(tr.mean()))
    return max(MIN_REV_FLOOR, int(round(atr_pts / TICK_SIZE * ATR_MULT)))


def leg_amplitude_usd(closes, pivots) -> float:
    """Sum of |price move| across the legs spanned by `pivots` (a flat zigzag
    captures each leg's amplitude). x $2/pt."""
    if len(pivots) < 2:
        return 0.0
    amp = sum(abs(closes[pivots[i + 1]] - closes[pivots[i]])
              for i in range(len(pivots) - 1))
    return amp * DOLLAR_PER_POINT


def resync_point(ref_after, cs):
    """First pivot index P where ref_after and cs share P and have an
    identical suffix from P on. Returns (P, idx_in_ref) or (None, None)."""
    cs_set = set(cs)
    for ri, p in enumerate(ref_after):
        if p not in cs_set:
            continue
        ci = cs.index(p)
        if ref_after[ri:] == cs[ci:]:
            return p, ri
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/2026-05-21_zigzag_coldstart_divergence.txt')
    args = ap.parse_args()

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    days = sorted(p.stem for p in ATLAS_5S.glob('2026_*.parquet'))
    out('=' * 80)
    out('ZIGZAG COLD-START DIVERGENCE  --  mid-day anchor vs beginning-of-time')
    out('=' * 80)
    out(f'Days available: {len(days)}   (DATA/ATLAS_NT8/5s/)')
    out(f'Cold-start offsets: {[f"{int(f*100)}%" for f in K_FRACTIONS]} into the day')
    out('detect_swings, ATR(14)x4, min_bars=36 -- production params.')
    out('')

    rows = []          # (day, K_frac, resync_lag_bars, n_transient, amp_diff_usd, never)
    n_days_used = 0
    for day in days:
        f5 = ATLAS_5S / f'{day}.parquet'
        f1 = ATLAS_1M / f'{day}.parquet'
        if not f1.exists():
            continue
        b5 = pd.read_parquet(f5).sort_values('timestamp').reset_index(drop=True)
        b1 = pd.read_parquet(f1).sort_values('timestamp').reset_index(drop=True)
        closes = b5['close'].values.astype(np.float64)
        if len(closes) < 2000:
            continue
        min_rev = compute_min_rev(b1)
        ref = list(detect_swings(closes, min_reversal=min_rev,
                                 min_bars=MIN_BARS, max_bars=0))
        n_days_used += 1
        n = len(closes)
        for frac in K_FRACTIONS:
            K = int(n * frac)
            cs_rel = detect_swings(closes[K:], min_reversal=min_rev,
                                   min_bars=MIN_BARS, max_bars=0)
            cs = [int(j) + K for j in cs_rel]
            ref_after = [p for p in ref if p >= K]
            P, ri = resync_point(ref_after, cs)
            if P is None:
                # never re-syncs for the rest of the day
                lag = n - K
                transient_ref = ref_after
                transient_cs = cs
                never = True
            else:
                lag = P - K
                transient_ref = ref_after[:ri + 1]
                transient_cs = cs[:cs.index(P) + 1]
                never = False
            amp_diff = abs(leg_amplitude_usd(closes, transient_ref)
                           - leg_amplitude_usd(closes, transient_cs))
            rows.append({'day': day, 'K_frac': frac, 'resync_bars': lag,
                         'resync_min': lag * 5 / 60.0,
                         'n_transient': len(transient_ref) - 1 if not never else len(ref_after),
                         'amp_diff_usd': amp_diff, 'never': never})

    if not rows:
        out('No usable days.')
        Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
        return

    df = pd.DataFrame(rows)
    out(f'Days used: {n_days_used}   cold-start samples: {len(df)}')
    out('')
    out('-' * 80)
    out('RE-SYNC LAG  (bars from the mid-day cold-start until the two zigzags')
    out('become identical -- i.e. the duration of the divergent transient)')
    out('-' * 80)
    out(f'{"K offset":>9}  {"n":>4}  {"resync median":>14}  {"resync mean":>12}  '
        f'{"p90":>8}  {"never-resync":>12}')
    for frac in K_FRACTIONS:
        s = df[df['K_frac'] == frac]
        nev = int(s['never'].sum())
        out(f'{int(frac*100):>7}%   {len(s):>4}  '
            f'{s["resync_min"].median():>10.0f} min  '
            f'{s["resync_min"].mean():>9.0f} min  '
            f'{s["resync_min"].quantile(0.9):>5.0f}min  '
            f'{nev:>6}/{len(s):<5}')
    out('')
    allr = df['resync_min'].values
    lo, hi = bootstrap_ci(allr)
    out(f'  ALL: resync lag mean {allr.mean():.0f} min  CI [{lo:.0f}, {hi:.0f}]  '
        f'median {np.median(allr):.0f} min  max {allr.max():.0f} min')
    out(f'  never-resyncs: {int(df["never"].sum())}/{len(df)} '
        f'({df["never"].mean()*100:.0f}%)')
    out('')
    out('-' * 80)
    out('TRANSIENT COST  (leg-amplitude mis-tiled while desynced, $ @ $2/pt)')
    out('-' * 80)
    amp = df['amp_diff_usd'].values
    alo, ahi = bootstrap_ci(amp)
    out(f'  mean ${amp.mean():+.0f}   median ${np.median(amp):+.0f}   '
        f'CI [${alo:.0f}, ${ahi:.0f}]   max ${amp.max():.0f}')
    out(f'  transient mis-tiled legs: mean {df["n_transient"].mean():.1f}  '
        f'median {df["n_transient"].median():.0f}  max {int(df["n_transient"].max())}')
    out('')
    out('=' * 80)
    out('VERDICT')
    out('=' * 80)
    med = np.median(allr)
    nev_frac = df['never'].mean()
    if nev_frac > 0.15 or med > 120:
        out('SIGNIFICANT: a mid-day cold-start frequently fails to re-sync, or')
        out('the transient runs hours. The live engine MUST anchor at the')
        out("day's first bar (catch the zigzag up over today's bars at startup).")
    else:
        out(f'BOUNDED TRANSIENT: a mid-day cold-start re-syncs in ~{med:.0f} min')
        out('median; after that the zigzag is identical to the beginning-of-time')
        out('run. The divergence is a bounded warmup cost, not a structural')
        out('break. Whether it is worth a code fix depends on the $ cost above')
        out('vs how often the engine is actually launched mid-session.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
