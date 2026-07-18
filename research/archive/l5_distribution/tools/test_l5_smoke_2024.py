"""Real-data smoke test of the L5 build path on ONE 2024 session day.
Replicates build_dataset's L5 pass (compute_L5_ldist -> drop bar_ts -> _align_to_anchor
-> insert timestamp) on actual ATLAS 1s + 5s, and sanity-checks the output. Light
(one day's 1s ~50-80k rows) — does NOT run the heavy full build_dataset.
Run: python research/test_l5_smoke_2024.py [YYYY_MM_DD]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine, TF_SECONDS  # noqa: E402
from core_v2.build_dataset import _align_to_anchor, _last_closed_idx  # noqa: E402
from core_v2.features import LAYER_FAMILIES  # noqa: E402

DAY = sys.argv[1] if len(sys.argv) > 1 else '2024_01_08'
ATLAS = 'DATA/ATLAS'


def main():
    p1s = f'{ATLAS}/1s/{DAY}.parquet'
    p5s = f'{ATLAS}/5s/{DAY}.parquet'
    for p in (p1s, p5s):
        if not os.path.exists(p):
            print(f"MISSING {p}"); sys.exit(1)

    df_1s = pd.read_parquet(p1s)
    anchor = pd.read_parquet(p5s)
    anchor_ts = anchor['timestamp'].to_numpy(np.int64)
    print(f"DAY={DAY}  1s rows={len(df_1s):,}  5s anchor rows={len(anchor):,}")
    print(f"1s span: {df_1s['timestamp'].min()} -> {df_1s['timestamp'].max()} "
          f"(close {df_1s['close'].min():.2f}..{df_1s['close'].max():.2f})")
    print()

    sfe = StatisticalFieldEngine()
    fails = []

    for tf in ('5s', '1m', '1h'):
        period = TF_SECONDS[tf]
        l5 = sfe.compute_L5_ldist(df_1s, tf)
        if len(l5) == 0:
            fails.append(f"{tf}: compute_L5_ldist returned 0 rows on real data"); continue

        l5_ts = l5['bar_ts'].to_numpy(np.int64)
        l5_feat = l5.drop(columns=['bar_ts']).reset_index(drop=True)
        aligned = _align_to_anchor(l5_ts, l5_feat, anchor_ts, period)
        aligned.insert(0, 'timestamp', anchor_ts)

        # cols match the registered family (minus 'timestamp')
        fam = LAYER_FAMILIES[f'L5_{tf}']['features']
        out_cols = [c for c in aligned.columns if c != 'timestamp']
        if out_cols != fam:
            fails.append(f"{tf}: aligned cols {out_cols} != family {fam}")

        # causality re-check on real data
        idx = _last_closed_idx(l5_ts, anchor_ts, period)
        bad = [(anchor_ts[i], l5_ts[idx[i]]) for i in range(len(anchor_ts))
               if idx[i] >= 0 and l5_ts[idx[i]] + period > anchor_ts[i]]
        if bad:
            fails.append(f"{tf}: {len(bad)} anchors used a forming bar (LOOKAHEAD), e.g. {bad[0]}")

        ncol = f'L5_{tf}_ldist_n'
        n_arr = l5_feat[ncol].to_numpy()
        nonwarm = int(np.sum(idx >= 0))
        nan_rows = int(aligned[ncol].isna().sum())
        med = l5_feat[f'L5_{tf}_ldist_median'].dropna()
        std = l5_feat[f'L5_{tf}_ldist_std'].dropna()
        skew = l5_feat[f'L5_{tf}_ldist_skew'].dropna()
        print(f"[{tf}] period={period}s  l5_bars={len(l5)}  per-bar n: "
              f"min={int(n_arr.min())} med={int(np.median(n_arr))} max={int(n_arr.max())}")
        print(f"      anchor rows={len(aligned)}  non-warmup={nonwarm}  NaN(warmup)={nan_rows}  "
              f"finite skew bars={len(skew)}")
        if len(med):
            print(f"      sample: median~[{med.min():.2f},{med.max():.2f}]  "
                  f"std med={std.median():.4f}  skew med={skew.median():+.4f}")
        # basic sanity: median within the day's price range, std >= 0
        if len(med) and not (df_1s['close'].min() - 1 <= med.min() and med.max() <= df_1s['close'].max() + 1):
            fails.append(f"{tf}: median outside day price range")
        if len(std) and (std < -1e-9).any():
            fails.append(f"{tf}: negative std")
        print()

    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS: L5 build path works on real 2024 data (cols match family, causal, sane stats).")


if __name__ == '__main__':
    main()
