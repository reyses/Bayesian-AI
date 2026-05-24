"""ATR-consensus measure — does cross-ATR-level agreement discriminate good
legs from chop?  OFFLINE vs CAUSAL.

User observation (2026-05-21): in a large/real leg the zigzag agrees across
ATR multipliers; in chop the scales decohere. For each production ATR x4 leg,
  consensus = fraction of {x2,x3,x6,x8} zigzags whose leg direction at the
              x4 leg's entry bar matches the x4 leg's direction.

Computed TWO ways:
  OFFLINE — detect_swings sees the whole day; a wide scale "agreeing" leaks
            hindsight ("this turned out to be a real leg"). Inflated.
  CAUSAL  — causal_direction: a single forward pass, each scale's direction
            confirmed using only PAST bars. The honest, live-available signal
            — and it IS the user's lag idea (a wide scale that has not yet
            confirmed at the x4 entry = decoherence).

If CAUSAL consensus still discriminates leg outcomes, it is a real trainable
feature for a chop / leg-quality candidate. If it collapses, it was hindsight.

Output: reports/findings/oos_bad_days/2026-05-21_atr_consensus.md
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from tools.viz.auto_swing_marker import detect_swings

REPO = Path(__file__).resolve().parent.parent
TARGETS = [
    ('IS',  REPO / 'reports/findings/regret_oracle/is_hardened_legs.csv',
     REPO / 'DATA/ATLAS'),
    ('OOS', REPO / 'reports/findings/regret_oracle/oos_hardened_legs_full.csv',
     REPO / 'DATA/ATLAS_NT8'),
]
OUT_MD = REPO / 'reports/findings/oos_bad_days/2026-05-21_atr_consensus.md'
MULTS = [2, 3, 4, 6, 8]
REF = 4
TICK = 0.25
ATR_PERIOD = 14
MIN_BARS = 36
N_BOOT = 4000
SEED = 42


def compute_atr(b1: pd.DataFrame) -> float:
    h, l, c = (b1[x].values.astype(float) for x in ('high', 'low', 'close'))
    if len(c) < 2:
        return 1.0
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    return (float(np.median(tr[-ATR_PERIOD * 3:])) if len(tr) >= ATR_PERIOD
            else float(tr.mean()))


def offline_direction(closes: np.ndarray, min_rev: int) -> np.ndarray:
    """Hindsight per-bar leg direction: each bar gets the direction of the
    detect_swings leg that CONTAINS it (known only after the leg completes)."""
    piv = detect_swings(closes, min_reversal=min_rev, min_bars=MIN_BARS,
                        max_bars=0)
    d = np.zeros(len(closes), dtype=np.int8)
    for k in range(len(piv) - 1):
        a, b = piv[k], piv[k + 1]
        d[a:b] = 1 if closes[b] > closes[a] else -1
    return d


def causal_direction(closes: np.ndarray, min_rev: int) -> np.ndarray:
    """Forward pass per-bar leg direction — a single forward pass mirroring
    auto_swing_marker.detect_swings' core loop (no hindsight merge pass).
    d[i] is confirmed using only closes[0..i]."""
    n = len(closes)
    d = np.zeros(n, dtype=np.int8)
    if n < 3:
        return d
    ct = closes / TICK
    direction = 0
    ex_idx, ex_val = 0, ct[0]
    for i in range(1, n):
        p = ct[i]
        if direction == 0:
            if p > ex_val:
                ex_val, ex_idx = p, i
            if p < ct[0] and ct[0] - p >= min_rev:
                direction, ex_val, ex_idx = -1, p, i
            elif p > ct[0] and p - ct[0] >= min_rev:
                direction, ex_val, ex_idx = 1, p, i
        elif direction == 1:
            if p >= ex_val:
                ex_val, ex_idx = p, i
            elif ex_val - p >= min_rev and i - ex_idx >= MIN_BARS:
                direction, ex_val, ex_idx = -1, p, i
        else:  # direction == -1
            if p <= ex_val:
                ex_val, ex_idx = p, i
            elif p - ex_val >= min_rev and i - ex_idx >= MIN_BARS:
                direction, ex_val, ex_idx = 1, p, i
        d[i] = direction
    return d


def bootstrap_ci(v, n=N_BOOT, seed=SEED):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def trade_wr(pnl):
    pnl = np.asarray(pnl, dtype=float)
    w, l = pnl[pnl > 0].sum(), pnl[pnl < 0].sum()
    return float(w / abs(l) - 1.0) if l != 0 else float('nan')


def measure(legs_csv: Path, root: Path) -> pd.DataFrame:
    legs_all = pd.read_csv(legs_csv)
    others = [m for m in MULTS if m != REF]
    rows = []
    for day, legs in legs_all.groupby('day'):
        f5, f1 = root / '5s' / f'{day}.parquet', root / '1m' / f'{day}.parquet'
        if not f5.exists() or not f1.exists():
            continue
        b5 = pd.read_parquet(f5).sort_values('timestamp').reset_index(drop=True)
        b1 = pd.read_parquet(f1).sort_values('timestamp').reset_index(drop=True)
        closes = b5['close'].values.astype(float)
        ts = b5['timestamp'].values.astype(np.int64)
        atr = compute_atr(b1)
        off, cau = {}, {}
        for m in MULTS:
            mr = max(4, int(round(atr / TICK * m)))
            off[m] = offline_direction(closes, mr)
            cau[m] = causal_direction(closes, mr)
        for _, lg in legs.iterrows():
            idx = min(int(np.searchsorted(ts, lg['entry_ts'])), len(closes) - 1)
            d4 = 1 if lg['leg_dir'] == 'LONG' else -1
            c_off = sum(off[m][idx] == d4 for m in others) / len(others)
            c_cau = sum(cau[m][idx] == d4 for m in others) / len(others)
            rows.append(dict(cons_off=c_off, cons_cau=c_cau,
                             pnl=float(lg['pnl_usd']),
                             amp=abs(float(lg['pnl_pts']))))
    return pd.DataFrame(rows)


def bucket_table(df: pd.DataFrame, col: str, out):
    out(f'{"consensus":>10}  {"n":>6}  {"mean $/leg":>11}  {"95% CI":>20}  '
        f'{"trade WR":>9}  {"win%":>6}  {"mean amp":>9}')
    for cv in [0.0, 0.25, 0.5, 0.75, 1.0]:
        s = df[np.isclose(df[col], cv)]
        if len(s) == 0:
            continue
        lo, hi = bootstrap_ci(s['pnl'].values)
        out(f'{cv:>10.2f}  {len(s):>6}  {s["pnl"].mean():>+10.2f}  '
            f'[{lo:>+8.1f},{hi:>+8.1f}]  {trade_wr(s["pnl"].values):>+9.2f}  '
            f'{(s["pnl"] > 0).mean() * 100:>5.0f}%  {s["amp"].mean():>9.1f}')
    hi_c = df[df[col] >= 0.75]['pnl']
    lo_c = df[df[col] <= 0.50]['pnl']
    corr = float(np.corrcoef(df[col], df['pnl'])[0, 1])
    out('')
    out(f'  correlation {corr:+.3f}   |   high(>=.75) ${hi_c.mean():+.1f}/leg '
        f'(n={len(hi_c)})   low(<=.5) ${lo_c.mean():+.1f}/leg (n={len(lo_c)})  '
        f'  spread ${hi_c.mean() - lo_c.mean():+.1f}/leg')


def main():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    L = []
    def out(s=''):
        L.append(s)
    out('# Cross-ATR consensus vs leg outcome — OFFLINE vs CAUSAL (MEASURE)')
    out('')
    out('For each production ATR x4 leg, consensus = fraction of {x2,x3,x6,x8} '
        'zigzags agreeing with the x4 leg direction at the x4 entry bar. '
        'OFFLINE uses detect_swings (hindsight — inflated). CAUSAL uses a '
        'forward-pass streaming detector (live-available — the honest signal). '
        'If CAUSAL still discriminates, consensus is a trainable feature.')
    out('')
    for label, csv, root in TARGETS:
        df = measure(csv, root)
        if len(df) == 0:
            out(f'## {label}: no data')
            out('')
            continue
        out(f'## {label} — {len(df)} legs')
        out('')
        out('### OFFLINE consensus (hindsight — inflated)')
        bucket_table(df, 'cons_off', out)
        out('')
        out('### CAUSAL consensus (streaming — the honest, live signal)')
        bucket_table(df, 'cons_cau', out)
        out('')
    OUT_MD.write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))
    print(f'\nWrote {OUT_MD}')


if __name__ == '__main__':
    main()
