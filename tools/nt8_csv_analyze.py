"""
nt8_csv_analyze.py -- Comprehensive analysis of NT8 strategy CSV ledger output.

Handles both v1.0/v1.1 CSV format (13 cols) and v1.2.1+ format (16 cols, w/ MFE).
For each CSV:
  - Per-trade economics: PF Trade WR, mean/median/mode + 95% CI
  - Exit reason breakdown (trail vs pivot vs SL vs EOD)
  - Per-day P&L summary, day WR
  - Capture analysis (only for CSVs with MFE column)
  - Top 10 best / worst trades
  - Day-of-week performance

Usage:
    python tools/nt8_csv_analyze.py reports/findings/nt8_zigzag_v1.2_trades.csv
    python tools/nt8_csv_analyze.py path/to/csv1 path/to/csv2 path/to/csv3
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def boot_ci(arr, n_boot=2000):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0: return 0.0, 0.0, 0.0
    rng = np.random.default_rng(42)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = a[rng.integers(0, len(a), len(a))].mean()
    return float(a.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def hist_mode(arr, bin_width=2.0):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0: return 0.0
    lo = np.floor(a.min() / bin_width) * bin_width
    hi = np.ceil(a.max() / bin_width) * bin_width
    if hi <= lo: return float(a.mean())
    bins = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(a, bins=bins)
    if counts.sum() == 0: return float(a.mean())
    k = int(counts.argmax())
    return float((edges[k] + edges[k + 1]) / 2.0)


def trade_wr_pf(p):
    a = np.asarray(p, dtype=float)
    a = a[np.isfinite(a)]
    pos = a[a > 0].sum()
    neg = abs(a[a < 0].sum())
    return float(pos / neg - 1) if neg > 0 else 0.0


SCHEMA_13 = ['close_time_utc','day','entry_time_utc','exit_time_utc','direction',
             'entry_price','exit_price','qty','pnl_points','pnl_usd','held_minutes',
             'entry_reason','exit_reason']
SCHEMA_16 = SCHEMA_13 + ['mfe_pts','mae_pts','capture_pct']


def load_csv(path: str) -> pd.DataFrame:
    """Read CSV with or without header row. Detects schema by column count."""
    # Peek at the first line
    with open(path, 'r', encoding='utf-8') as f:
        first = f.readline()
    looks_like_header = first.startswith('close_time_utc')
    if looks_like_header:
        df = pd.read_csv(path)
    else:
        # Read without header; assign schema by column count
        df_peek = pd.read_csv(path, header=None, nrows=1)
        n_cols = df_peek.shape[1]
        if n_cols == 13:
            df = pd.read_csv(path, header=None, names=SCHEMA_13)
        elif n_cols == 16:
            df = pd.read_csv(path, header=None, names=SCHEMA_16)
        else:
            raise ValueError(f'{path}: unknown column count {n_cols}')
    return df


def analyze(csv_path: str):
    if not os.path.exists(csv_path):
        print(f'NOT FOUND: {csv_path}')
        return

    df = load_csv(csv_path)
    df = df.dropna(subset=['pnl_usd']).reset_index(drop=True)
    if len(df) == 0:
        print(f'{csv_path}: empty after NA filter')
        return

    has_mfe = 'mfe_pts' in df.columns

    print('=' * 100)
    print(f'  {os.path.basename(csv_path)}')
    print(f'  N trades = {len(df):,}  |  N days = {df["day"].nunique()}  |  '
          f'MFE column = {"yes" if has_mfe else "no"}')
    print('=' * 100)

    # ── Per-trade economics ─────────────────────────────────────────────
    pnl = df['pnl_usd'].values
    twr = trade_wr_pf(pnl)
    m, lo, hi = boot_ci(pnl)
    mode = hist_mode(pnl, 2.0)
    print(f'\nPER-TRADE')
    print(f'  Trade WR (PF-based)   : {twr:>+8.4f}  ({twr*100:>+5.1f}%)')
    print(f'  Mean $/trade  (95%CI) : ${m:>+7.2f}   [${lo:>+6.2f}, ${hi:>+6.2f}]')
    print(f'  Mode $/trade  ($2 bin): ${mode:>+7.2f}')
    print(f'  Median $/trade        : ${np.median(pnl):>+7.2f}')
    print(f'  Total $               : ${pnl.sum():>+10.0f}')

    # ── Exit reason breakdown ────────────────────────────────────────────
    print(f'\nEXIT REASONS')
    print(f'  {"reason":>22} {"N":>6} {"share":>7} {"WR%":>7} {"PF Trade WR":>13} {"mean $":>10} {"mean MFE $":>11}')
    for reason, sub in df.groupby('exit_reason'):
        share = 100 * len(sub) / len(df)
        wr = (sub['pnl_usd'] > 0).mean() * 100
        twr_r = trade_wr_pf(sub['pnl_usd'].values)
        mean = sub['pnl_usd'].mean()
        mfe = sub['mfe_pts'].mean() * 2.0 if has_mfe else float('nan')
        mfe_str = f'${mfe:>+8.2f}' if not np.isnan(mfe) else '   n/a   '
        print(f'  {reason:>22} {len(sub):>6,} {share:>6.1f}% {wr:>6.1f}% {twr_r:>+12.4f} ${mean:>+8.2f} {mfe_str}')

    # ── Per-day stats ────────────────────────────────────────────────────
    daily = df.groupby('day')['pnl_usd'].sum()
    arr = daily.values
    d_m, d_lo, d_hi = boot_ci(arr)
    print(f'\nPER-DAY  (N days = {len(arr)})')
    print(f'  Mean $/day  (95%CI)   : ${d_m:>+7.2f}   [${d_lo:>+6.2f}, ${d_hi:>+6.2f}]')
    print(f'  Median $/day          : ${np.median(arr):>+7.2f}')
    print(f'  Mode $/day  ($25 bin) : ${hist_mode(arr, 25.0):>+7.2f}')
    print(f'  Day WR  (count-based) : {(arr>0).mean()*100:>5.1f}%  ({(arr>0).sum()}W / {(arr<=0).sum()}L)')
    print(f'  Best day              : ${arr.max():>+7.0f}')
    print(f'  Worst day             : ${arr.min():>+7.0f}')
    print(f'  Std                   : ${arr.std():>+7.0f}')

    # ── Per-day breakdown (top 10 best + worst days) ─────────────────────
    print(f'\nTOP 5 BEST / WORST DAYS')
    by_day = df.groupby('day').agg(
        pnl=('pnl_usd', 'sum'),
        n=('pnl_usd', 'size'),
        n_trail=('exit_reason', lambda x: int((x.str.startswith('TrailStop')).sum())),
        n_pivot=('exit_reason', lambda x: int((x.str.startswith('FlipExit')).sum())),
        n_eod=('exit_reason', lambda x: int(x.str.contains('Eod|session', case=False, na=False).sum())),
    ).reset_index().sort_values('pnl', ascending=False)
    print(f'  {"day":<12} {"$/day":>9} {"n":>4} {"trail":>6} {"pivot":>6} {"eod":>5}')
    for _, r in by_day.head(5).iterrows():
        print(f'  {r["day"]:<12} ${r["pnl"]:>+7.0f} {r["n"]:>4} {r["n_trail"]:>6} {r["n_pivot"]:>6} {r["n_eod"]:>5}')
    print('  --- worst ---')
    for _, r in by_day.tail(5).iterrows():
        print(f'  {r["day"]:<12} ${r["pnl"]:>+7.0f} {r["n"]:>4} {r["n_trail"]:>6} {r["n_pivot"]:>6} {r["n_eod"]:>5}')

    # ── Capture analysis (only when MFE column present) ──────────────────
    if has_mfe and 'capture_pct' in df.columns:
        print(f'\nCAPTURE ANALYSIS  (mfe-based)')
        # Trail-only: capture should be high (peak − eff_dist) / peak
        trail_df = df[df['exit_reason'].str.startswith('TrailStop')]
        if len(trail_df) > 0:
            tcap = trail_df['capture_pct'].values
            print(f'  Trail-exit trades        : {len(trail_df):>5,}')
            print(f'  Trail capture % (mean)   : {tcap.mean():>+6.1f}%')
            print(f'  Trail capture % (median) : {np.median(tcap):>+6.1f}%')
            print(f'  Trail capture % p25/p75  : {np.quantile(tcap, 0.25):>+6.1f}% / {np.quantile(tcap, 0.75):>+6.1f}%')
            # Bucket by MFE size
            print(f'\n  Capture % by MFE bucket (only trail exits):')
            trail_df = trail_df.copy()
            trail_df['mfe_bucket'] = pd.cut(trail_df['mfe_pts'],
                bins=[-0.001, 5, 10, 20, 50, 100, 1000],
                labels=['<5pt','5-10pt','10-20pt','20-50pt','50-100pt','>100pt'])
            for b, sub in trail_df.groupby('mfe_bucket', observed=True):
                if len(sub) == 0: continue
                cap_mean = sub['capture_pct'].mean()
                pnl_mean = sub['pnl_usd'].mean()
                print(f'    MFE {str(b):<10}  N={len(sub):>5,}  '
                      f'mean cap={cap_mean:>+6.1f}%  mean ${pnl_mean:>+7.2f}')

    # ── Best / worst trades ──────────────────────────────────────────────
    print(f'\nTOP 5 BEST / WORST TRADES')
    cols = ['day','direction','entry_price','exit_price','pnl_usd','exit_reason']
    if has_mfe: cols += ['mfe_pts','capture_pct']
    print(df.nlargest(5, 'pnl_usd')[cols].to_string(index=False))
    print('  --- worst ---')
    print(df.nsmallest(5, 'pnl_usd')[cols].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csvs', nargs='+', help='CSV path(s) to analyze')
    args = ap.parse_args()

    for path in args.csvs:
        analyze(path)
        print()


if __name__ == '__main__':
    main()
