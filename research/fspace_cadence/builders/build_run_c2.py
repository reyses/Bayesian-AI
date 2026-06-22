"""Build RUN C-2 — continuous sliding window, FREE window length (decoupled from TF labels).

C-2 is the purest cardinal-scale variant: like B2C (continuous, recomputed every 1s) but the
window length W is a FREE hyperparameter swept on a clean log ladder, NOT pinned to TF horizons.
ONE window per F-space (one-window-at-a-time) so the R-curve maps cleanly as a function of W
(the cardinal-scale response surface) with no cross-window collinearity.

Reuses the VALIDATED B2C math: custom rolling-L1 + production SFE compute_L2/L3 at N=W. Same
corrected-rule / enriched stage-1 downstream. Run on 2024_02_20 + Brownian + Fourier.

OUTPUT: DATA/ATLAS/FEATURES_RUN_C2_w{W}/<day>.parquet  (contains "RUN_C" -> stage-1 reads directly).
"""
import os, sys, argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core_v2.statistical_field_engine import StatisticalFieldEngine
from build_run_b2c import rolling_L1   # validated rolling-L1 (range/body/wick/velocity over trailing W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--window', type=int, required=True, help='window length W in seconds (= rows on 1s grid)')
    ap.add_argument('--atlas_root', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")))
    args = ap.parse_args()

    src = os.path.join(args.atlas_root, '1s', f'{args.day}.parquet')
    if not os.path.exists(src):
        print(f"[FATAL] 1s OHLCV not found: {src}")
        return
    df = pd.read_parquet(src).reset_index(drop=True)
    W = args.window
    n = len(df)
    if W >= n:
        print(f"[C2] window {W} >= day length {n}; skip.")
        return
    label = f"w{W}"   # free-window label (not a TF)
    sfe = StatisticalFieldEngine()
    parts = [df[['timestamp']].copy(), sfe.compute_L0(df),
             rolling_L1(df, label, W),
             sfe.compute_L2(df, label, N=W),
             sfe.compute_L3(df, label, N=W)]
    out = pd.concat(parts, axis=1)

    out_dir = os.path.join(args.atlas_root, f'FEATURES_RUN_C2_w{W}')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.day}.parquet')
    out.to_parquet(out_path)
    nan = float(out.drop(columns=['timestamp']).isnull().any(axis=1).mean())
    print(f"[C2] W={W}s -> {out_path}  shape={out.shape}  rows-with-any-NaN={nan:.1%}")


if __name__ == '__main__':
    main()
