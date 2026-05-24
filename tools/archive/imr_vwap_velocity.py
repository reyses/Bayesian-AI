"""I-MR chart of 4h vwap with SIGNED moving range = velocity.

Statistical-Process-Control I-MR chart of `L2_4h_vwap_18` across the
full dataset. The SIGNED moving range (vwap[t] - vwap[t-1]) IS the
per-bar velocity of the volume-weighted regression mean — no separate
calculation needed. Visualizing it as an I-MR chart reveals:

    Individual (I) panel:
        - The vwap track over the full series
        - 3σ control limits (mean ± 3·σ_short)
        - Out-of-control points = days where vwap drifted abnormally

    Moving Range (MR) panel:
        - SIGNED MR = velocity (positive = vwap rising, negative = falling)
        - UCL = D4·MR̄ (Western Electric SPC constant; D4=3.267 for n=2)
        - Spikes beyond UCL = momentum bursts (regime transitions)

The SPC framing identifies WHEN the trend is structurally changing
(out-of-control runs) vs. WHEN it's noise within stable bounds.

Usage:
    python tools/imr_vwap_velocity.py
    python tools/imr_vwap_velocity.py --tf 1h
    python tools/imr_vwap_velocity.py --col L2_1h_vwap_12 --days 2025_06_15,2025_06_16
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import load_features


# Western Electric SPC constants for n=2
D4 = 3.267       # MR upper control multiplier
D3 = 0.0          # MR lower (negative not produced for unsigned)
d2 = 1.128       # mean-MR-to-σ conversion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--col', default='L2_4h_vwap_18',
                          help='V2 feature column for the vwap series')
    ap.add_argument('--tf', default=None,
                          help='Shortcut: --tf 4h sets col=L2_4h_vwap_18, --tf 1h → L2_1h_vwap_12')
    ap.add_argument('--features-root', default='DATA/ATLAS/FEATURES_5s_v2')
    ap.add_argument('--days', default=None,
                          help='Comma-separated YYYY_MM_DD list. Default = all IS (2025_*)')
    ap.add_argument('--out',
                          default='reports/findings/imr_vwap_velocity')
    ap.add_argument('--sample-stride', type=int, default=12,
                          help='Subsample every Nth bar to keep plot tractable (default 12 = 1m)')
    args = ap.parse_args()

    if args.tf == '4h':
        args.col = 'L2_4h_vwap_18'
    elif args.tf == '1h':
        args.col = 'L2_1h_vwap_12'
    elif args.tf == '1D':
        args.col = 'L2_1D_vwap_5'

    # Resolve days
    if args.days:
        days = [d.strip() for d in args.days.split(',') if d.strip()]
    else:
        l0 = os.path.join(args.features_root, 'L0')
        files = sorted(glob.glob(os.path.join(l0, '*.parquet')))
        days = [os.path.basename(f).replace('.parquet', '') for f in files
                    if os.path.basename(f).startswith('2025_')]
        if not days:
            print(f'No IS days under {l0}'); sys.exit(1)

    print(f'Loading {args.col} across {len(days)} days...')

    # Load the series in chunks
    series_ts, series_val = [], []
    for day in tqdm(days, desc='load'):
        f = load_features(days=[day], root=args.features_root)
        if f.empty or args.col not in f.columns:
            continue
        f = f.sort_values('timestamp').reset_index(drop=True)
        ts = f['timestamp'].astype('int64').values[::args.sample_stride]
        val = f[args.col].values[::args.sample_stride]
        series_ts.extend(ts.tolist())
        series_val.extend(val.tolist())

    if not series_val:
        print('No data; aborting'); sys.exit(1)

    arr = np.asarray(series_val, dtype=np.float64)
    ts_arr = np.asarray(series_ts, dtype=np.int64)
    finite = np.isfinite(arr)
    arr = arr[finite]
    ts_arr = ts_arr[finite]
    if len(arr) < 100:
        print(f'Too few points ({len(arr)}); aborting'); sys.exit(1)

    # Signed Moving Range = velocity
    mr_signed = np.diff(arr)            # length n-1
    mr_unsigned = np.abs(mr_signed)
    ts_mr = ts_arr[1:]

    # SPC limits — based on the unsigned MR bar
    mr_bar = float(mr_unsigned.mean())
    sigma_est = mr_bar / d2          # I-chart short-term σ estimate
    cl_i = float(arr.mean())
    ucl_i = cl_i + 3 * sigma_est
    lcl_i = cl_i - 3 * sigma_est
    ucl_mr = D4 * mr_bar
    # For SIGNED MR, control limits are ± UCL_MR (symmetric)

    # Out-of-control flags
    i_oc = (arr > ucl_i) | (arr < lcl_i)
    mr_oc = np.abs(mr_signed) > ucl_mr

    print(f'\n{"=" * 80}')
    print(f'I-MR CHART  col={args.col}')
    print(f'  n_bars         : {len(arr):>8}    sample_stride: {args.sample_stride}')
    print(f'  CL (mean)      : {cl_i:>10.3f}')
    print(f'  σ̂ (from MR̄/d2): {sigma_est:>10.4f}')
    print(f'  I-chart UCL/LCL: {ucl_i:>10.3f} / {lcl_i:>10.3f}')
    print(f'  MR̄ (avg |MR|) : {mr_bar:>10.4f}    UCL_MR = D4·MR̄ = {ucl_mr:>.4f}')
    print(f'  Out-of-control I points : {int(i_oc.sum()):>5} ({i_oc.mean()*100:.2f}%)')
    print(f'  Out-of-control MR points: {int(mr_oc.sum()):>5} ({mr_oc.mean()*100:.2f}%)')
    print(f'\n  SIGNED MR distribution (the velocity):')
    print(f'    mean  : {mr_signed.mean():+10.4f}')
    print(f'    std   : {mr_signed.std():>10.4f}')
    print(f'    q05   : {np.quantile(mr_signed, 0.05):+10.4f}')
    print(f'    q50   : {np.quantile(mr_signed, 0.50):+10.4f}')
    print(f'    q95   : {np.quantile(mr_signed, 0.95):+10.4f}')
    print(f'    %positive : {(mr_signed > 0).mean()*100:>5.1f}%')
    print(f'    %negative : {(mr_signed < 0).mean()*100:>5.1f}%')

    # ── Plot ────────────────────────────────────────────────────────────
    os.makedirs(args.out, exist_ok=True)
    dt_arr = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_arr]
    dt_mr = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ts_mr]

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.4, 1.0], hspace=0.18)

    # Panel 1: Individual chart (the vwap track)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(dt_arr, arr, color='black', lw=0.5, alpha=0.7)
    ax1.scatter(np.array(dt_arr)[i_oc], arr[i_oc],
                     color='red', s=8, zorder=5, alpha=0.5,
                     label=f'I out-of-control ({int(i_oc.sum())})')
    ax1.axhline(cl_i, color='blue', lw=0.6, linestyle='--', alpha=0.6,
                     label=f'CL={cl_i:.1f}')
    ax1.axhline(ucl_i, color='red', lw=0.5, linestyle=':', alpha=0.6,
                     label=f'±3σ={ucl_i:.1f}/{lcl_i:.1f}')
    ax1.axhline(lcl_i, color='red', lw=0.5, linestyle=':', alpha=0.6)
    ax1.set_ylabel(f'{args.col}\n(individual)')
    ax1.set_title(f'I-MR chart — {args.col} ({len(days)} days, stride {args.sample_stride})')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: SIGNED Moving Range = velocity
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    pos_mask = mr_signed > 0
    ax2.scatter(np.array(dt_mr)[pos_mask], mr_signed[pos_mask],
                     color='green', s=2, alpha=0.4, label='positive (rising)')
    ax2.scatter(np.array(dt_mr)[~pos_mask], mr_signed[~pos_mask],
                     color='red', s=2, alpha=0.4, label='negative (falling)')
    ax2.scatter(np.array(dt_mr)[mr_oc], mr_signed[mr_oc],
                     color='black', s=10, zorder=5, alpha=0.7,
                     label=f'|MR| > UCL ({int(mr_oc.sum())})')
    ax2.axhline(0, color='gray', lw=0.6)
    ax2.axhline(ucl_mr, color='red', lw=0.5, linestyle=':', alpha=0.6,
                     label=f'±UCL={ucl_mr:.4f}')
    ax2.axhline(-ucl_mr, color='red', lw=0.5, linestyle=':', alpha=0.6)
    ax2.set_ylabel(f'SIGNED MR\n(= velocity)')
    ax2.set_xlabel('time (UTC)')
    ax2.legend(loc='upper left', fontsize=8, ncol=4)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    out_png = os.path.join(args.out,
                                       f'imr_{args.col.replace(".","_")}.png')
    plt.savefig(out_png, dpi=110, bbox_inches='tight')
    plt.close()
    print(f'\nPlot saved -> {out_png}')

    # Save the signed MR as parquet for downstream sign-of-MR analysis
    mr_df = pd.DataFrame({
        'timestamp': ts_mr,
        'value': arr[1:],
        'signed_mr': mr_signed,
        'is_oc': mr_oc.astype(bool),
    })
    mr_path = os.path.join(args.out,
                                       f'mr_{args.col.replace(".","_")}.parquet')
    mr_df.to_parquet(mr_path, index=False)
    print(f'MR parquet -> {mr_path}')


if __name__ == '__main__':
    main()
