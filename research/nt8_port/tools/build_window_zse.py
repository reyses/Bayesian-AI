"""
Build L3_1m z_se_15 (the SANCTIONED N=15 schema) for a date window, for the P3 diff.

Reviewer ruling (2026-07-18): the frozen combiner + C# port consume L3_1m_z_se_15;
current code N_BASE['1m']=30 is config drift. We do NOT change code defaults -- we
produce a z_se_15 store for the diff only, written to a DEDICATED dir so the standard
FEATURES_5s_v2 store (z_se_30) is left intact.

Derivation = the canonical SFE OLS-endpoint kernel at N=15 (core_v2.compute_L3), aligned
to the 5s anchor via build_dataset._align_to_anchor (_last_closed_idx). Spot-check: the
z_se per 1m bar is re-derived with the research/nmp_state verified method (explicit
window-15 OLS endpoint, ddof=2) and asserted bit-close BEFORE any window day is written.

Usage:
  python3.11 research/nt8_port/tools/build_window_zse.py \
      --atlas-root DATA/ATLAS_NT8 --start 2026_06_22 --end 2026_07_08 \
      --out research/nt8_port/golden_backtest/_zse15
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '../../..'))
sys.path.insert(0, ROOT)

import core_v2.build_dataset as bd
from core_v2.statistical_field_engine import StatisticalFieldEngine

TF = '1m'
N_1M = 15                       # SANCTIONED N_BASE['1m'] per MEMORY (code default 30 = drift)
ZCOL = f'L3_1m_z_se_{N_1M}'


def nmp_state_zse(closes, window):
    """research/nmp_state verified kernel: OLS endpoint z, residual std ddof=2."""
    n = len(closes)
    z = np.full(n, np.nan)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for k in range(window - 1, n):
        y = closes[k - window + 1:k + 1]
        y_mean = y.mean()
        cov = np.sum((x - x_mean) * (y - y_mean))
        slope = cov / x_var
        intercept = y_mean - slope * x_mean
        endpoint = slope * (window - 1) + intercept
        resids = y - (slope * x + intercept)
        var_resid = np.sum(resids ** 2) / (window - 2)
        z[k] = (y[-1] - endpoint) / np.sqrt(var_resid) if var_resid > 0 else 0.0
    return z


def compute_l3_1m(atlas_root, end_day):
    files = [p for p in bd._list_day_files(atlas_root, TF) if bd._day_from_path(p) <= end_day]
    if not files:
        raise FileNotFoundError(f"no {TF} files through {end_day} under {atlas_root}")
    tf = (pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
            .drop_duplicates('timestamp', keep='last').sort_values('timestamp').reset_index(drop=True))
    tf_ts = tf['timestamp'].values.astype(np.int64)
    sfe = StatisticalFieldEngine()
    l3 = sfe.compute_L3(tf, tf=TF, N=N_1M)
    print(f"  loaded {len(tf):,} {TF} bars ({len(files)} files); computed L3 z_se (N={N_1M})")
    return tf, tf_ts, l3


def spot_check(tf, l3):
    """Assert compute_L3 z_se == nmp_state verified kernel on the 1m bar series."""
    a = l3[ZCOL].values
    b = nmp_state_zse(tf['close'].values.astype(float), N_1M)
    both = ~(np.isnan(a) | np.isnan(b))
    mad = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
    nanmis = int((np.isnan(a) != np.isnan(b)).sum())
    print(f"SPOT-CHECK vs nmp_state kernel: n={int(both.sum())} max|dz_se|={mad:.3e} nan_mismatch={nanmis}")
    return mad, nanmis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas-root', default='DATA/ATLAS_NT8')
    ap.add_argument('--start', required=True)
    ap.add_argument('--end', required=True)
    ap.add_argument('--out', default='research/nt8_port/golden_backtest/_zse15')
    args = ap.parse_args()

    bd.set_anchor_globals('5s')
    tf, tf_ts, l3 = compute_l3_1m(args.atlas_root, args.end)

    mad, nanmis = spot_check(tf, l3)
    if mad > 1e-6 or nanmis > 0:
        print("  SPOT-CHECK FAIL -> aborting"); sys.exit(2)
    print("  SPOT-CHECK OK (compute_L3 N=15 == nmp_state verified kernel)")

    outdir = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(outdir, exist_ok=True)
    anchor_files = bd._list_day_files(args.atlas_root, '5s')
    days = [bd._day_from_path(p) for p in anchor_files
            if args.start <= bd._day_from_path(p) <= args.end]
    for day in days:
        anchor = bd._load_anchor_day(args.atlas_root, day)
        anchor_ts = anchor['timestamp'].values.astype(np.int64)
        aligned = bd._align_to_anchor(tf_ts, l3[[ZCOL]], anchor_ts, bd.TF_SECONDS[TF])
        aligned.insert(0, 'timestamp', anchor_ts)
        aligned.to_parquet(os.path.join(outdir, f'{day}.parquet'), index=False)
        print(f"  wrote {day}: rows={len(aligned)} finite={int(np.isfinite(aligned[ZCOL]).sum())}")
    print(f"\ndone: {len(days)} days -> {outdir}")


if __name__ == '__main__':
    main()
