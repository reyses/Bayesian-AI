"""Build RUN_B2C — the CONTINUOUS sliding-window F-space from the 1s base.

THESIS (see research/THESIS_continuous_fspace.md):
    Timeframes are a human attention crutch, not a market property. There is ONE
    continuous price signal; the right machine representation reads it with SLIDING
    WINDOWS from the 1s base, where "higher TF" == "bigger sample window" (a CARDINAL
    quantity in seconds), not a segregated, clock-tiled, ordinal bar.

WHAT THIS IS (vs Gemini's RUN_B2):
    Gemini's RUN_B2 phase-interleaved proper-TF bars onto the 1s grid (still tiled
    bars, just refreshed every second -> the ZOH step-fill that shattered regimes).
    RUN_B2C instead computes every feature as a TRUE TRAILING WINDOW of W seconds on
    the raw 1s stream, evaluated at every second. No clock phase, no bar-close reset,
    no step-function. The window length W *is* the (former) timeframe.

KEY REUSE: the production SFE compute_L2/compute_L3 are vectorized over ROWS and treat
    row-spacing as "1 bar". Fed the raw 1s stream with N = W (seconds), they compute
    EXACTLY the trailing-W-second OLS z-family / means / sigma at every second. So we
    reuse the validated formulas verbatim; only L1 (range/body/wick/velocity over the
    trailing-W bar, not a single 1s row) needs a custom rolling implementation.

WINDOW SET: W = n_base_bars * tf_seconds, i.e. the *actual lookback horizon* the old
    TF used (e.g. the 1m TF with N_BASE=15 looked back 15*60=900s -> W=900). This makes
    RUN_B2C an apples-to-apples horizon-matched continuous analog of the tiled F-space.

SCOPE (v1, finishable): L0 + custom-L1 + L2 + L3 over 5 windows (5s..5m).
    DEFERRED to v2 (noted, not silently dropped):
      - L4-NMP (vr/z21 are W-independent; lambda needs per-W z_se wiring) — TODO.
      - L5-dist (rolling trailing-W quantiles per second — expensive) — TODO.
      - 10m/15m windows — their Hurst warmup (W*8) exceeds a single day; add with
        multi-day history.

NON-CAUSAL? No — every window here is strictly trailing (sliding), so the FEATURES are
    causal. (The downstream stage-1 SEGMENTATION remains non-causal/diagnostic as ever.)

OUTPUT: DATA/ATLAS/FEATURES_RUN_B2C/<day>.parquet  (dir contains "RUN_B" so the stage-1
    loader reads it directly via the RUN_A/RUN_B/RUN_C branch).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from core_v2.statistical_field_engine import StatisticalFieldEngine

# (label, tf_seconds, n_base_bars) -> sliding window W = tf_seconds * n_base_bars (seconds == rows on 1s grid).
# n_base mirrors N_BASE in the SFE so the horizon matches the old tiled TF exactly.
WINDOWS = [
    ('5s',   5,  9),    # W = 45s
    ('15s',  15, 12),   # W = 180s
    ('1m',   60, 15),   # W = 900s
    ('3m',   180, 12),  # W = 2160s
    ('5m',   300, 9),   # W = 2700s
    # ('10m', 600, 12), # W = 7200s   -> v2 (Hurst warmup 57600s > 1 day)
    # ('15m', 900, 12), # W = 10800s  -> v2
]


def rolling_L1(df: pd.DataFrame, label: str, W: int) -> pd.DataFrame:
    """L1 primitives over the trailing-W-second bar (the sliding-window analog of a TF bar).

    Mirrors SFE.compute_L1 semantics but with "1 bar" == the trailing W seconds:
      - bar open  = close W seconds ago (price at the window's left edge)
      - bar high/low/vol = rolling max/min/sum over the trailing W seconds
      - bar close = current 1s close
    velocity_1b = change over one W-bar; accel_1b = change of that over one W-bar.
    """
    close = df['close'].astype(np.float64)
    high = df['high'].astype(np.float64)
    low = df['low'].astype(np.float64)
    vol = df['volume'].astype(np.float64)

    open_W = close.shift(W)                       # window left-edge price
    high_W = high.rolling(W, min_periods=W).max()
    low_W = low.rolling(W, min_periods=W).min()
    vol_W = vol.rolling(W, min_periods=W).sum()

    price_v = close - open_W                      # one W-bar price change
    price_a = price_v - price_v.shift(W)
    vol_v = vol_W - vol_W.shift(W)
    vol_a = vol_v - vol_v.shift(W)

    bar_range = high_W - low_W
    body = close - open_W
    upper_wick = high_W - np.maximum(open_W, close)
    lower_wick = np.minimum(open_W, close) - low_W

    return pd.DataFrame({
        f'L1_{label}_price_velocity_1b': price_v,
        f'L1_{label}_price_accel_1b':    price_a,
        f'L1_{label}_vol_velocity_1b':   vol_v,
        f'L1_{label}_vol_accel_1b':      vol_a,
        f'L1_{label}_bar_range':         bar_range,
        f'L1_{label}_body':              body,
        f'L1_{label}_upper_wick':        upper_wick,
        f'L1_{label}_lower_wick':        lower_wick,
    }, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True, help='YYYY_MM_DD (must be a 2024 Databento IS day)')
    ap.add_argument('--atlas_root', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")))
    args = ap.parse_args()

    src = os.path.join(args.atlas_root, '1s', f'{args.day}.parquet')
    if not os.path.exists(src):
        print(f"[FATAL] 1s OHLCV not found: {src}")
        return
    df = pd.read_parquet(src).reset_index(drop=True)
    for c in ('open', 'high', 'low', 'close', 'volume', 'timestamp'):
        if c not in df.columns:
            print(f"[FATAL] 1s source missing column '{c}'. Have: {list(df.columns)}")
            return
    n = len(df)
    print(f"[B2C] {args.day}: {n} 1s rows. Windows: {[(l, s*b) for l, s, b in WINDOWS]} (seconds)")

    sfe = StatisticalFieldEngine()
    parts = [df[['timestamp']].copy(), sfe.compute_L0(df)]

    for label, tf_sec, n_base in WINDOWS:
        W = tf_sec * n_base
        if W >= n:
            print(f"[B2C]   {label} (W={W}s): window >= day length ({n}), skipping.")
            continue
        print(f"[B2C]   {label}: W={W}s  -> L1(rolling) + L2(N={W}) + L3(N={W})")
        parts.append(rolling_L1(df, label, W))
        parts.append(sfe.compute_L2(df, label, N=W))
        parts.append(sfe.compute_L3(df, label, N=W))

    out_df = pd.concat(parts, axis=1)

    out_dir = os.path.join(args.atlas_root, 'FEATURES_RUN_B2C')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.day}.parquet')
    out_df.to_parquet(out_path)
    nan_frac = float(out_df.drop(columns=['timestamp']).isnull().any(axis=1).mean())
    print(f"[B2C] wrote {out_path}  shape={out_df.shape}  rows-with-any-NaN={nan_frac:.1%}")


if __name__ == '__main__':
    main()
