"""
Validate all ATLAS TF data against 1s ground truth.

Checks every bar at every TF: does the high/low/close fall within
the range of the underlying 1s bars? Fixes any that don't.

Usage:
  python tools/validate_data.py

Processes month by month to keep memory bounded.
Progress bars on everything.
"""
import gc
import glob
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

ATLAS = 'DATA/ATLAS'
TICK = 0.25
ALL_TF_SECONDS = {
    '5s': 5, '15s': 15, '1m': 60, '5m': 300, '15m': 900,
    '1h': 3600, '4h': 14400, '1D': 86400, '1W': 604800,
}

# CLI filter: pass --tf 1D,1W to only validate those
import sys
_tf_filter = None
for arg in sys.argv[1:]:
    if arg.startswith('--tf='):
        _tf_filter = arg.split('=')[1].split(',')
    elif arg == '--tf' and sys.argv.index(arg) + 1 < len(sys.argv):
        _tf_filter = sys.argv[sys.argv.index(arg) + 1].split(',')

if _tf_filter:
    TF_SECONDS = {k: v for k, v in ALL_TF_SECONDS.items() if k in _tf_filter}
else:
    TF_SECONDS = ALL_TF_SECONDS


def main():
    total_fixed = 0
    total_checked = 0

    months_1s = sorted(glob.glob(os.path.join(ATLAS, '1s', '*.parquet')))
    print(f"Validating {len(months_1s)} months of data against 1s ground truth")
    print(f"TFs to check: {list(TF_SECONDS.keys())}")
    print()

    for f_1s in tqdm(months_1s, desc="Months"):
        month = os.path.basename(f_1s).replace('.parquet', '')

        df_1s = pd.read_parquet(f_1s).sort_values('timestamp').reset_index(drop=True)
        ts_1s = df_1s['timestamp'].values
        high_1s = df_1s['high'].values
        low_1s = df_1s['low'].values
        close_1s = df_1s['close'].values
        open_1s = df_1s['open'].values
        vol_1s = df_1s['volume'].values

        for tf, tf_sec in TF_SECONDS.items():
            path = os.path.join(ATLAS, tf, f'{month}.parquet')
            if not os.path.exists(path):
                continue

            df = pd.read_parquet(path).sort_values('timestamp').reset_index(drop=True)
            fixes = 0
            modified = False

            for i in tqdm(range(len(df)), desc=f"  {tf} {month}", leave=False):
                bar_ts = df.iloc[i]['timestamp']
                si = int(np.searchsorted(ts_1s, bar_ts, side='left'))
                ei = int(np.searchsorted(ts_1s, bar_ts + tf_sec, side='left'))

                if ei <= si:
                    continue

                total_checked += 1
                true_high = float(high_1s[si:ei].max())
                true_low = float(low_1s[si:ei].min())

                bar_high = df.iloc[i]['high']
                bar_low = df.iloc[i]['low']
                bar_close = df.iloc[i]['close']

                bad = (bar_high > true_high + TICK or
                       bar_low < true_low - TICK or
                       bar_close > true_high + TICK or
                       bar_close < true_low - TICK)

                if bad:
                    df.iloc[i, df.columns.get_loc('high')] = true_high
                    df.iloc[i, df.columns.get_loc('low')] = true_low
                    df.iloc[i, df.columns.get_loc('close')] = float(close_1s[ei - 1])
                    df.iloc[i, df.columns.get_loc('open')] = float(open_1s[si])
                    df.iloc[i, df.columns.get_loc('volume')] = float(vol_1s[si:ei].sum())
                    fixes += 1
                    modified = True

            if modified:
                df.to_parquet(path, index=False)
            if fixes > 0:
                tqdm.write(f"  {tf} {month}: fixed {fixes} bars")
            total_fixed += fixes

        del df_1s
        gc.collect()

    print(f"\n{'='*50}")
    print(f"VALIDATION COMPLETE")
    print(f"  Checked: {total_checked:,} bars")
    print(f"  Fixed:   {total_fixed} bad bars")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
