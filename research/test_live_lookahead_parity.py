"""Live<->offline parity test for the forming-bar lookahead fix.

For mid-day 2024 anchors, simulate the live buffer (5s bars up to the latest CLOSED
bar = a-5) and call get_v2_vector(a); compare its higher-TF features to the OFFLINE
parquet row at timestamp==a. Post-fix they must MATCH (live no longer uses the
forming bar). Tests 1m/1h/4h (short windows -> a ~35-day buffer is ample warmup).
Run: python research/test_live_lookahead_parity.py
"""
import os, glob, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.live_features import LiveFeatureEngine
from core_v2.features import FEATURE_NAMES, N_BASE

ATLAS = 'DATA/ATLAS'
FEAT = f'{ATLAS}/FEATURES_5s_v2'
DAY = '2024_06_20'
BUF_DAYS = 35


def offline_val(tf, metric, a):
    fam = f"L{metric[0]}_{tf}"
    col = metric[1]
    df = pd.read_parquet(f"{FEAT}/{fam}/{DAY}.parquet", columns=['timestamp', col])
    row = df[df['timestamp'] == a]
    return float(row[col].iloc[0]) if len(row) else np.nan


def main():
    files = sorted(glob.glob(f'{ATLAS}/5s/2024_*.parquet'))
    names = [os.path.basename(f) for f in files]
    i = names.index(f'{DAY}.parquet')
    buf_files = files[max(0, i - BUF_DAYS + 1):i + 1]
    df = pd.concat([pd.read_parquet(f) for f in buf_files], ignore_index=True)
    df = df.drop_duplicates('timestamp', keep='last').sort_values('timestamp').reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('dt')

    day_ts = pd.read_parquet(f'{ATLAS}/5s/{DAY}.parquet')['timestamp'].to_numpy(np.int64)
    anchors = [int(day_ts[len(day_ts) // 3]), int(day_ts[len(day_ts) // 2]), int(day_ts[2 * len(day_ts) // 3])]

    # (layer, column-suffix) per TF
    def metrics(tf):
        N = N_BASE[tf]
        return [('1', f'L1_{tf}_price_velocity_1b'),
                ('2', f'L2_{tf}_price_sigma_{N}'),
                ('3', f'L3_{tf}_z_se_{N}')]

    eng = LiveFeatureEngine(atlas_root=ATLAS, max_days=BUF_DAYS)
    fails, checks = [], 0
    for a in anchors:
        eng._df_5s = df[df['timestamp'] <= a - 5].copy()
        eng._bars['5s'] = eng._df_5s
        eng._new_bars = []
        vec = eng.get_v2_vector(a)
        if vec is None:
            fails.append(f'anchor {a}: live vec is None'); continue
        live = dict(zip(FEATURE_NAMES, vec))
        print(f"anchor {a} (buffer last {int(eng._df_5s['timestamp'].iloc[-1])} = a-5):")
        for tf in ('1m', '1h', '4h'):
            for lyr, col in metrics(tf):
                off = offline_val(tf, (lyr, col), a)
                lv = float(live.get(col, np.nan))
                checks += 1
                both_nan = np.isnan(off) and np.isnan(lv)
                ok = both_nan or np.isclose(off, lv, rtol=1e-4, atol=1e-4)
                if not ok:
                    fails.append(f"  {col} @ {a}: live={lv:.6g} offline={off:.6g}")
                mark = 'OK ' if ok else 'MISMATCH'
                print(f"    [{mark}] {col}: live={lv:.5g}  offline={off:.5g}")
    print()
    if fails:
        print(f"FAIL ({len(fails)}/{checks}):")
        for f in fails: print("  -", f)
        sys.exit(1)
    print(f"PASS: live == offline on all {checks} higher-TF feature checks "
          "(forming-bar lookahead fixed; live now mirrors _last_closed_idx).")


if __name__ == '__main__':
    main()
