"""Build RUN_C (mirrored, bar-close-sampled) as a transform of B2C.

Reconstruction of Gemini's Run C (builder lost): 1s base, window = TF wall-clock horizon
(W = n_base*tf_sec, SAME as B2C), but features UPDATE at bar-close cadence and are HELD
between closes (vs B2C which recomputes every 1s). So Run C = B2C downsampled to each
window's bar-close cadence (every tf_sec), forward-filled.

This isolates ONE axis vs B2C: update cadence (every-1s continuous  vs  bar-close-sampled+held).
Causal (only past info; ffill holds the last closed value). Same 5 windows / layers as B2C.

OUTPUT: DATA/ATLAS/FEATURES_RUN_C2024/<day>.parquet  (contains "RUN_C" -> stage-1 reads directly).
"""
import os, sys, argparse
import numpy as np
import pandas as pd

# label -> base TF seconds (emission cadence); window content already baked into B2C columns
TF_SEC = {'5s': 5, '15s': 15, '1m': 60, '3m': 180, '5m': 300}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--atlas_root', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")))
    args = ap.parse_args()

    src = os.path.join(args.atlas_root, 'FEATURES_RUN_B2C', f'{args.day}.parquet')
    if not os.path.exists(src):
        print(f"[FATAL] need B2C features first: {src}")
        return
    df = pd.read_parquet(src).reset_index(drop=True)
    ts = df['timestamp'].to_numpy(np.int64)

    out = df.copy()
    for col in df.columns:
        if col in ('timestamp', 'L0_time_of_day'):
            continue
        parts = col.split('_')
        label = parts[1] if len(parts) > 1 else None
        tf_sec = TF_SEC.get(label)
        if tf_sec is None:
            continue
        # keep value only at bar-close rows (ts % tf_sec == 0), hold (ffill) between closes
        s = df[col].copy()
        mask = (ts % tf_sec) != 0
        s[mask] = np.nan
        out[col] = s.ffill()

    out_dir = os.path.join(args.atlas_root, 'FEATURES_RUN_C2024')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.day}.parquet')
    out.to_parquet(out_path)
    nan = float(out.drop(columns=['timestamp']).isnull().any(axis=1).mean())
    print(f"[RUNC] wrote {out_path}  shape={out.shape}  rows-with-any-NaN={nan:.1%}")


if __name__ == '__main__':
    main()
