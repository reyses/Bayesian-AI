"""Conditional probability table — entry 3 (v2, leg-decoupled): after a chop
or a fakeout, does the preceding trend continue?

v1 was confounded — every directional figure was a zigzag-alternation parity
artifact (the chop table alternated by K parity). Its windows were anchored at
leg boundaries and short enough that a single parity-locked adjacent leg
dominated them.

v2 decouples direction from the leg structure:
  - "Trend" / "continuation" = the SIGN OF THE REGRESSION SLOPE of raw 5s
    closes over a LONG window (default 90 min) — no single leg can dominate it.
  - The zigzag detects the chop / fakeout EVENT only.

VALIDATION GATE: the chop table must NOT alternate by K parity. If it does, the
decoupling failed and the run prints FAIL.

CHOP    = a run of K consecutive low-range x4 legs (K = 1..5).
FAKEOUT = a single mid/high-range x4 leg whose direction opposes the trend.
Preceding trend D = slope-sign of the window before the event; `continues` =
slope-sign of the window after the event == D. Compared to the
direction-persistence base rate. IS + OOS, n + 95% bootstrap CI.

Output: reports/findings/oos_bad_days/2026-05-21_trend_continuation.md
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
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_trend_continuation.md'
TICK = 0.25
ATR_PERIOD = 14
ATR_MULT = 4.0
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


def slope_sign(closes: np.ndarray, a: int, b: int) -> float:
    """Sign of the least-squares regression slope of closes[a:b] vs bar index.
    A leg-decoupled trend direction — every bar in the window contributes."""
    if a < 0 or b > len(closes) or b - a < 3:
        return 0.0
    seg = closes[a:b]
    x = np.arange(len(seg), dtype=np.float64)
    num = float(((x - x.mean()) * (seg - seg.mean())).sum())
    return float(np.sign(num))


def day_legs(day: str):
    f5, f1 = bars_path(day, '5s'), bars_path(day, '1m')
    if not f5.exists() or not f1.exists():
        return None, []
    b5 = pd.read_parquet(f5).sort_values('timestamp').reset_index(drop=True)
    b1 = pd.read_parquet(f1).sort_values('timestamp').reset_index(drop=True)
    closes = b5['close'].values.astype(float)
    min_rev = max(4, int(round(compute_atr(b1) / TICK * ATR_MULT)))
    piv = detect_swings(closes, min_reversal=min_rev, min_bars=MIN_BARS,
                        max_bars=0)
    legs = [(piv[k], piv[k + 1], closes[piv[k + 1]] - closes[piv[k]])
            for k in range(len(piv) - 1)]
    return closes, legs


def scan(days, q33, q67, w):
    base = []
    chop = {k: [] for k in range(1, MAX_K + 1)}
    fake = []
    for day in days:
        closes, legs = day_legs(str(day))
        if closes is None or len(legs) < 2:
            continue
        n = len(closes)
        amp = np.array([abs(l[2]) for l in legs])
        cls = np.where(amp < q33, 0, np.where(amp >= q67, 2, 1))
        for i, (s, e, a) in enumerate(legs):
            if s - w >= 0 and s + w < n:
                d0 = slope_sign(closes, s - w, s)
                if d0 != 0:
                    base.append(int(slope_sign(closes, s, s + w) == d0))
            if cls[i] != 0 and s - w >= 0 and e + w < n:
                D = slope_sign(closes, s - w, s)
                if D != 0 and np.sign(a) == -D:
                    fake.append(int(slope_sign(closes, e, e + w) == D))
        for K in range(1, MAX_K + 1):
            for i in range(K - 1, len(legs)):
                if not np.all(cls[i - K + 1:i + 1] == 0):
                    continue
                cs, ce = legs[i - K + 1][0], legs[i][1]
                if cs - w < 0 or ce + w >= n:
                    continue
                D = slope_sign(closes, cs - w, cs)
                if D != 0:
                    chop[K].append(int(slope_sign(closes, ce, ce + w) == D))
    return base, chop, fake


def pct(x):
    return f'{x * 100:.0f}%'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--window-min', type=float, default=90.0,
                    help='trend / outcome window, minutes')
    args = ap.parse_args()
    w = int(round(args.window_min * 60 / 5))

    is_days = sorted(pd.read_csv(IS_LEGS)['day'].astype(str).unique())
    oos_days = sorted(pd.read_csv(OOS_LEGS)['day'].astype(str).unique())

    is_amps = []
    for d in is_days:
        _, legs = day_legs(str(d))
        is_amps.extend(abs(l[2]) for l in legs)
    q33, q67 = np.percentile(np.array(is_amps), [100 / 3.0, 200 / 3.0])

    is_base, is_chop, is_fake = scan(is_days, q33, q67, w)
    oos_base, oos_chop, oos_fake = scan(oos_days, q33, q67, w)

    # ---- validation gate: does the IS chop K-series alternate by parity? ----
    ks = [k for k in range(1, MAX_K + 1) if is_chop[k]]
    vals = [float(np.mean(is_chop[k])) for k in ks]
    diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    alternating = (len(diffs) >= 3 and
                   all(diffs[i] * diffs[i + 1] < 0 for i in range(len(diffs) - 1)))
    odd = np.mean([v for k, v in zip(ks, vals) if k % 2 == 1])
    even = np.mean([v for k, v in zip(ks, vals) if k % 2 == 0]) if any(
        k % 2 == 0 for k in ks) else float('nan')
    parity_gap = abs(odd - even) if np.isfinite(even) else 0.0
    gate_fail = alternating and parity_gap > 0.12

    L = []
    def o(s=''):
        L.append(s)

    o('# After a chop or a fakeout — does the preceding trend continue? (v2)')
    o('')
    o(f'Leg-DECOUPLED rebuild. Trend = sign of the regression slope of 5s '
      f'closes over a {args.window_min:g}-min window; the zigzag marks the '
      f'event only. `continues` = the slope after the event matches the slope '
      f'before it.')
    o('')
    gate = 'FAIL' if gate_fail else 'PASS'
    o(f'**VALIDATION GATE: {gate}** — IS chop K-series parity gap '
      f'{parity_gap * 100:.0f}pp'
      f'{"; still confounded — do NOT use" if gate_fail else "; no parity artifact"}.')
    o('')
    bl = bootstrap_ci(is_base)
    ol = bootstrap_ci(oos_base)
    o(f'**Base rate** — P(90-min trend direction persists across a leg '
      f'boundary): IS {pct(np.mean(is_base))} [{pct(bl[0])}, {pct(bl[1])}] '
      f'(n={len(is_base):,}); OOS {pct(np.mean(oos_base))} '
      f'[{pct(ol[0])}, {pct(ol[1])}] (n={len(oos_base):,}).')
    o('')
    o('## After a CHOP (K consecutive low-range legs) — P(preceding trend continues)')
    o('')
    o('| K | n(IS) | IS P(continues) [CI] | n(OOS) | OOS P(continues) [CI] |')
    o('|--:|--:|---|--:|---|')
    for K in range(1, MAX_K + 1):
        ic, oc = is_chop[K], oos_chop[K]
        if not ic or not oc:
            continue
        il, ih = bootstrap_ci(ic)
        oll, oh = bootstrap_ci(oc)
        o(f'| {K} | {len(ic):,} | {pct(np.mean(ic))} [{pct(il)}, {pct(ih)}] '
          f'| {len(oc):,} | {pct(np.mean(oc))} [{pct(oll)}, {pct(oh)}] |')
    o('')
    o('## After a FAKEOUT (a notable counter-trend leg) — P(preceding trend continues)')
    o('')
    fl = bootstrap_ci(is_fake)
    fo = bootstrap_ci(oos_fake)
    o(f'- IS: {pct(np.mean(is_fake))} [{pct(fl[0])}, {pct(fl[1])}] '
      f'(n={len(is_fake):,})')
    o(f'- OOS: {pct(np.mean(oos_fake))} [{pct(fo[0])}, {pct(fo[1])}] '
      f'(n={len(oos_fake):,})')
    o('')
    o('## Read')
    o('')
    if gate_fail:
        o('- GATE FAILED — the chop table still alternates by K parity. The '
          f'{args.window_min:g}-min window did not fully decouple from the '
          'leg structure. Re-run with a longer --window-min before trusting '
          'any number here.')
    else:
        o('- Gate passed — no parity artifact. Compare each P(continues) to '
          'the base rate: ABOVE = the event is followed by trend continuation '
          'more than usual (the mechanical counter-trend entry is wrong, a '
          'direction gate has something to catch); AT base = no directional '
          'information; BELOW = the event predicts a reversal.')
    o('- Trust a cell only where IS and OOS agree.')

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(f'\nWrote {OUT_MD}')
    print(f'GATE: {"FAIL" if gate_fail else "PASS"}  parity_gap={parity_gap*100:.0f}pp')


if __name__ == '__main__':
    main()
