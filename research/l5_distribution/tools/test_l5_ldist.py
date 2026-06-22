"""Unit test for SFE.compute_L5_ldist — validates the within-bar 1s distribution
battery against an INDEPENDENT numpy reference under the pinned conventions, plus
the return-shape (one row per closed tf-bar), grouping, and the 1D special case.
Run: python research/test_l5_ldist.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.statistical_field_engine import (  # noqa: E402
    StatisticalFieldEngine, TF_SECONDS, L5_QUANTILE_METHOD, L5_OUTLIER_K,
    L5_MIN_SAMPLES_MOMENTS, L5_EPS)


def ref_group(c, x):
    """Independent reference for one group's battery."""
    n = c.size
    out = {'n': n}
    q = np.quantile(c, [0.0, 0.25, 0.5, 0.75, 1.0], method=L5_QUANTILE_METHOD)
    out['min'], out['q1'], out['median'], out['q3'], out['max'] = q
    mean = c.mean()
    out['mean'] = mean
    out['std'] = c.std(ddof=1) if n >= 2 else 0.0
    if n >= L5_MIN_SAMPLES_MOMENTS:
        dev = c - mean
        m2 = np.mean(dev * dev)
        if m2 > L5_EPS:
            out['skew'] = np.mean(dev ** 3) / m2 ** 1.5
            out['kurtosis'] = np.mean(dev ** 4) / m2 ** 2 - 3.0
            out['outlier_pct'] = np.mean(np.abs(dev) > L5_OUTLIER_K * np.sqrt(m2)) * 100.0
        else:
            out['skew'] = out['kurtosis'] = out['outlier_pct'] = np.nan
    else:
        out['skew'] = out['kurtosis'] = out['outlier_pct'] = np.nan
    xm = x.mean()
    xv = np.sum((x - xm) ** 2)
    out['level'] = (mean - (np.sum((x - xm) * (c - mean)) / xv) * xm) if (n >= 2 and xv > L5_EPS) else c[-1]
    return out


def main():
    np.random.seed(7)
    base = 1_700_000_040            # minute-aligned (base % 60 == 0) -> clean 60-pt bars
    n1s = 180                       # 3 one-minute bars
    ts = base + np.arange(n1s, dtype=np.int64)
    close = 20000.0 + np.cumsum(np.random.randn(n1s) * 0.25)
    df = pd.DataFrame({'timestamp': ts, 'open': close, 'high': close,
                       'low': close, 'close': close, 'volume': 1})

    sfe = StatisticalFieldEngine()
    fails = []

    # --- 1m: expect 3 rows, n=60 each, all stats == reference ---
    out = sfe.compute_L5_ldist(df, '1m').reset_index(drop=True)
    if len(out) != 3:
        fails.append(f"1m: got {len(out)} rows, expected 3")
    P = 'L5_1m_ldist_'
    period = TF_SECONDS['1m']
    for i in range(min(len(out), 3)):
        b = int(out['bar_ts'].iloc[i])
        m = (ts // period) * period == b
        c = close[m]
        x = ts[m].astype(np.float64) - (b + period - 1)
        ref = ref_group(c, x)
        if c.size != 60:
            fails.append(f"1m bar {i}: n={c.size} != 60")
        for k, v in ref.items():
            got = out[P + k].iloc[i]
            if np.isnan(v):
                if not np.isnan(got):
                    fails.append(f"1m bar {i} {k}: got {got}, ref NaN")
            elif not np.isclose(got, v, rtol=1e-9, atol=1e-9):
                fails.append(f"1m bar {i} {k}: got {got}, ref {v}")

    # --- column completeness ---
    expect_cols = {P + k for k in ('min', 'q1', 'median', 'q3', 'max', 'mean', 'std',
                                   'skew', 'kurtosis', 'n', 'level', 'outlier_pct')}
    missing = expect_cols - set(out.columns)
    if missing:
        fails.append(f"1m: missing cols {missing}")

    # --- 1D special case: one row, n = all 180 ---
    outd = sfe.compute_L5_ldist(df, '1D').reset_index(drop=True)
    if len(outd) != 1:
        fails.append(f"1D: got {len(outd)} rows, expected 1 (one-bar-per-session)")
    elif int(outd['L5_1D_ldist_n'].iloc[0]) != 180:
        fails.append(f"1D: n={outd['L5_1D_ldist_n'].iloc[0]} != 180")

    # --- small-n moment guard: a 5s bar (5 pts) < L5_MIN_SAMPLES_MOMENTS(=4)? 5>=4 so moments present;
    #     make a 3-pt bar by trimming and confirm skew/kurt NaN ---
    df3 = df.iloc[:3].copy()
    out3 = sfe.compute_L5_ldist(df3, '1m').reset_index(drop=True)
    if not np.isnan(out3['L5_1m_ldist_skew'].iloc[0]):
        fails.append("n=3 bar should have NaN skew (below L5_MIN_SAMPLES_MOMENTS)")
    if int(out3['L5_1m_ldist_n'].iloc[0]) != 3:
        fails.append("n=3 bar: n column wrong")

    print(f"1m rows={len(out)} (n per bar = {[int(out[P+'n'].iloc[i]) for i in range(len(out))]})")
    print(f"1D rows={len(outd)} (n={int(outd['L5_1D_ldist_n'].iloc[0]) if len(outd) else 'NA'})")
    print(f"sample 1m bar0: median={out[P+'median'].iloc[0]:.4f} std={out[P+'std'].iloc[0]:.4f} "
          f"skew={out[P+'skew'].iloc[0]:.4f} level={out[P+'level'].iloc[0]:.4f}")
    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("PASS: compute_L5_ldist matches numpy reference (battery + shape + 1D + small-n guard).")


if __name__ == '__main__':
    main()
