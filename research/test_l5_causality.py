"""Causality test for the L5 build integration: the step-fill onto the 5s anchor
must use period=TF_SECONDS[tf] so that NO L5 value at anchor ts ever draws on a
tf-bar that is still forming (bar_ts + period > ts). Uses the REAL build helpers.
Run: python research/test_l5_causality.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import StatisticalFieldEngine, TF_SECONDS  # noqa: E402
from core_v2.build_dataset import _align_to_anchor, _last_closed_idx  # noqa: E402


def main():
    np.random.seed(11)
    base = 1_700_000_040                      # minute-aligned
    n1s = 3600                                # one hour of 1s
    ts1s = base + np.arange(n1s, dtype=np.int64)
    close = 20000.0 + np.cumsum(np.random.randn(n1s) * 0.25)
    df1s = pd.DataFrame({'timestamp': ts1s, 'open': close, 'high': close,
                         'low': close, 'close': close, 'volume': 1})
    anchor_ts = np.arange(base, base + n1s, 5, dtype=np.int64)   # 5s anchors over the hour

    sfe = StatisticalFieldEngine()
    fails = []

    for tf in ('1m', '5m', '15m'):
        period = TF_SECONDS[tf]
        l5 = sfe.compute_L5_ldist(df1s, tf)
        l5_ts = l5['bar_ts'].to_numpy(np.int64)
        l5_feat = l5.drop(columns=['bar_ts']).reset_index(drop=True)
        aligned = _align_to_anchor(l5_ts, l5_feat, anchor_ts, period)

        if len(aligned) != len(anchor_ts):
            fails.append(f"{tf}: aligned len {len(aligned)} != anchors {len(anchor_ts)}")

        idx = _last_closed_idx(l5_ts, anchor_ts, period)
        ncol = f'L5_{tf}_ldist_n'
        warmup_n = 0
        for i, ai in enumerate(anchor_ts):
            if idx[i] >= 0:
                # CAUSALITY: the used bar must have CLOSED at or before ai
                if l5_ts[idx[i]] + period > ai:
                    fails.append(f"{tf} anchor {ai}: used forming bar "
                                 f"(bar_ts={l5_ts[idx[i]]}, +{period} > {ai}) = LOOKAHEAD")
                    break
            else:
                warmup_n += 1
                # warmup rows must be NaN (no bar closed yet)
                if not pd.isna(aligned[ncol].iloc[i]):
                    fails.append(f"{tf} anchor {ai}: warmup row not NaN")
                    break

        # warmup = anchors before the FIRST bar closes (first bar may open before
        # `base` when base isn't period-aligned -> partial first bar, still causal)
        expect_warmup = int(np.sum(anchor_ts < l5_ts[0] + period))
        got_warmup = int(np.sum(idx < 0))
        if got_warmup != expect_warmup:
            fails.append(f"{tf}: warmup count {got_warmup} != expected {expect_warmup}")

        # spot check: a mid-hour anchor uses the immediately-prior closed bar
        probe = base + period + period // 2          # inside the 2nd bar
        j = int(np.searchsorted(anchor_ts, probe))
        used = l5_ts[idx[j]]
        if used + period > anchor_ts[j] or anchor_ts[j] - used > 2 * period:
            fails.append(f"{tf}: mid probe used bar {used} not the last-closed for anchor {anchor_ts[j]}")

        print(f"[{tf}] period={period}s  l5_bars={len(l5)}  warmup_anchors={got_warmup}  "
              f"(expected {expect_warmup})  aligned_rows={len(aligned)}")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS: L5 step-fill is causal (period=TF_SECONDS[tf]; no forming bar ever used; warmup NaN).")


if __name__ == '__main__':
    main()
