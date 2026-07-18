"""Build single-resolution base features at a chosen base TF (5s/15s/1m) for the base-resolution sweep.

Tests the hunch: coarser base -> longer regimes. Uses the NATIVE TF bars directly (no step-fill),
computes L1/L2/L3 on them; stage-1 then segments on that bar grid (--tf <base>).
OUTPUT: DATA/ATLAS/FEATURES_RUN_B<tf>/<day>.parquet  (contains "RUN_B" -> stage-1 reads directly).
"""
import os, sys, argparse
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core_v2.statistical_field_engine import StatisticalFieldEngine

NBASE = {'5s': 9, '15s': 12, '1m': 15}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--tf', required=True, choices=list(NBASE))
    ap.add_argument('--atlas_root', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")))
    args = ap.parse_args()
    src = os.path.join(args.atlas_root, args.tf, f'{args.day}.parquet')
    if not os.path.exists(src):
        print(f"[FATAL] {src} missing"); return
    df = pd.read_parquet(src).reset_index(drop=True)
    N = NBASE[args.tf]
    sfe = StatisticalFieldEngine()
    feats = pd.concat([df[['timestamp']].copy(), sfe.compute_L0(df),
                       sfe.compute_L1(df, args.tf),
                       sfe.compute_L2(df, args.tf, N=N),
                       sfe.compute_L3(df, args.tf, N=N)], axis=1)
    out_dir = os.path.join(args.atlas_root, f'FEATURES_RUN_B{args.tf}')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{args.day}.parquet')
    feats.to_parquet(out)
    nan = float(feats.drop(columns=['timestamp']).isnull().any(axis=1).mean())
    print(f"[BASE {args.tf}] {len(df)} bars -> {out}  shape={feats.shape}  NaN={nan:.1%}")

if __name__ == '__main__':
    main()
