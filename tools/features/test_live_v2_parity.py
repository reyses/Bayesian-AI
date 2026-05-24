"""V2 streaming-vs-batch parity test.

Replays one ATLAS day through LiveFeatureEngineV2 and compares its
get_v2_vector() output column-by-column to the batch
DATA/ATLAS/FEATURES_5s_v2/{family}/{day}.parquet files for the same day.

Pass criterion (per anchor, per column):
    - both NaN, OR
    - max(|streaming - batch|) < TOL

TOL is 1e-6 in absolute terms for non-NaN cells. Anything else is a bug.

Usage:
    python tools/test_live_v2_parity.py
    python tools/test_live_v2_parity.py --day 2025_06_03
    python tools/test_live_v2_parity.py --day 2025_06_03 --max-anchors 1000
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.live_feature_engine_v2 import LiveFeatureEngineV2
from core_v2.features import FEATURE_NAMES as V2_FEATURE_NAMES, LAYER_FAMILIES

ATLAS_ROOT = 'DATA/ATLAS_NT8'
V2_ROOT = 'DATA/ATLAS_NT8/FEATURES_5s_v2'

# Absolute tolerance for float comparison; column-relative cells with
# magnitudes >> 1 get up to TOL_REL * magnitude
TOL_ABS = 1e-6
TOL_REL = 1e-9


def load_batch_row(day: str, ts: float) -> pd.Series | None:
    """Load the canonical V2 row for (day, ts) by joining all family parquets."""
    row = pd.Series(np.nan, index=V2_FEATURE_NAMES, dtype=np.float64)
    for family, meta in LAYER_FAMILIES.items():
        path = os.path.join(V2_ROOT, family, f'{day}.parquet')
        if not os.path.exists(path):
            return None
        df = pd.read_parquet(path)
        match = df[df['timestamp'] == ts]
        if match.empty:
            return None
        m = match.iloc[0]
        for col in meta['features']:
            if col in m.index:
                row[col] = float(m[col]) if not pd.isna(m[col]) else np.nan
    return row


def diff_rows(streaming: pd.Series, batch: pd.Series) -> dict:
    """Return per-column diff stats. Empty dict means perfect match."""
    diffs = {}
    for col in V2_FEATURE_NAMES:
        s = streaming.get(col, np.nan)
        b = batch.get(col, np.nan)
        s_nan = pd.isna(s)
        b_nan = pd.isna(b)
        if s_nan and b_nan:
            continue
        if s_nan != b_nan:
            diffs[col] = ('NAN_MISMATCH', float(s) if not s_nan else None,
                            float(b) if not b_nan else None)
            continue
        d = abs(float(s) - float(b))
        tol = max(TOL_ABS, TOL_REL * max(abs(float(s)), abs(float(b))))
        if d > tol:
            diffs[col] = ('DIFF', float(s), float(b), d)
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', type=str, default=None,
                    help='Day to test (YYYY_MM_DD). Default: pick middle of ATLAS.')
    ap.add_argument('--max-anchors', type=int, default=300,
                    help='Max anchor timestamps to test (default 300 for speed).')
    ap.add_argument('--first-skip', type=int, default=2000,
                    help='Skip the first N anchors (warmup, NaN-heavy).')
    args = ap.parse_args()

    # Pick test day
    days = sorted(p.stem for p in Path(os.path.join(ATLAS_ROOT, '5s')).glob('*.parquet'))
    if not days:
        print(f'No ATLAS 5s parquets found under {ATLAS_ROOT}/5s/')
        return 1
    if args.day:
        if args.day not in days:
            print(f'Day {args.day} not in ATLAS. Available: {days[0]} to {days[-1]}.')
            return 1
        day = args.day
    else:
        day = days[len(days) // 2]
    print(f'Testing V2 parity on day {day}')

    # Verify batch V2 exists for this day
    l0_path = os.path.join(V2_ROOT, 'L0', f'{day}.parquet')
    if not os.path.exists(l0_path):
        print(f'No batch V2 parquet for {day}: {l0_path}')
        return 1

    # Init engine, load history excluding the test day
    print(f'Loading engine history (exclude_day={day})...')
    lfe = LiveFeatureEngineV2(atlas_root=ATLAS_ROOT)
    lfe.load_history(exclude_day=day)

    # Replay the test day's 5s bars
    bars_5s_path = os.path.join(ATLAS_ROOT, '5s', f'{day}.parquet')
    bars = pd.read_parquet(bars_5s_path)
    print(f'  Day has {len(bars)} 5s bars')

    n_tested = 0
    n_pass = 0
    n_warmup = 0
    n_batch_missing = 0
    all_diffs: dict[float, dict] = {}

    for i, bar in bars.iterrows():
        # Feed the bar into the engine
        bar_dict = {
            'timestamp': float(bar['timestamp']),
            'open': float(bar['open']),
            'high': float(bar['high']),
            'low': float(bar['low']),
            'close': float(bar['close']),
            'volume': int(bar['volume']),
        }
        # on_bar returns 91D V1 vector but as a side effect populates V2 caches
        lfe.on_bar(bar_dict)

        if i < args.first_skip:
            continue
        if n_tested >= args.max_anchors:
            break

        ts = bar_dict['timestamp']
        streaming = lfe.get_v2_row(ts)
        if streaming is None:
            n_warmup += 1
            continue

        batch = load_batch_row(day, ts)
        if batch is None:
            n_batch_missing += 1
            continue

        n_tested += 1
        diffs = diff_rows(streaming, batch)
        if not diffs:
            n_pass += 1
        else:
            all_diffs[ts] = diffs

    print()
    print(f'Anchors tested:        {n_tested}')
    print(f'  Perfect match:       {n_pass}')
    print(f'  With diffs:          {len(all_diffs)}')
    print(f'  Skipped (warmup):    {n_warmup}')
    print(f'  Skipped (no batch):  {n_batch_missing}')

    if all_diffs:
        # Top 10 worst per column
        col_max_diff = {}
        for ts, diffs in all_diffs.items():
            for col, info in diffs.items():
                if info[0] == 'DIFF':
                    d = info[3]
                    if col not in col_max_diff or d > col_max_diff[col][0]:
                        col_max_diff[col] = (d, ts, info[1], info[2])
                else:
                    # NAN_MISMATCH
                    col_max_diff.setdefault(col, ('NAN', ts, info[1], info[2]))
        print()
        print(f'Columns with diffs (top 15 by max abs):')
        sorted_cols = sorted(col_max_diff.items(),
                              key=lambda kv: kv[1][0] if isinstance(kv[1][0], (int, float)) else 1e18,
                              reverse=True)[:15]
        for col, info in sorted_cols:
            if info[0] == 'NAN':
                print(f'  {col:<45}  NAN_MISMATCH  s={info[2]}  b={info[3]}')
            else:
                d, ts, s, b = info
                print(f'  {col:<45}  max_diff={d:.3e}  s={s:.6g}  b={b:.6g}  ts={ts}')
        return 1

    print()
    print('PASS: streaming V2 == batch V2 within tolerance.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
