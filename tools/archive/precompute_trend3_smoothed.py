"""DMI-like smoothing layer on top of trend3 raw predictions.

Inputs:  raw per-bar P(LONG), P(SHORT), P(NEUTRAL) from trend3_cache_*.parquet
Outputs: same rows + smoothed columns + state-machine direction

Columns added:
  p_long_ema    : EMA(N) of P(LONG)
  p_short_ema   : EMA(N) of P(SHORT)
  dx            : 100 × |+DI − -DI| / (+DI + -DI)   ∈ [0, 100]
  adx           : EMA(N) of dx                       ∈ [0, 100]
  regime_dir    : 'LONG' / 'SHORT' / 'NEUTRAL'  — state-machine output
  regime_change : True on bars where regime_dir flipped from prior bar

Rules for state machine:
  - LONG  if p_long_ema  > p_short_ema by MARGIN and ADX > ADX_FLOOR
  - SHORT if p_short_ema > p_long_ema  by MARGIN and ADX > ADX_FLOOR
  - else: hold previous regime (sticky — that's the "blip filter")

Per user 2026-05-17: filters single-bar blips by requiring sustained
prior agreement before flipping direction.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple EMA with alpha = 2/(N+1). Recursive — infinite-history."""
    if period <= 1:
        return arr.copy()
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


def windowed_ema(arr: np.ndarray, span: int, window_bars: int) -> np.ndarray:
    """EMA bounded to a sliding `window_bars` lookback. Older data is DROPPED
    entirely (hard cutoff), within the window weights decay exponentially.

    Per user 2026-05-17: avoids stale state from hours ago dominating the
    smoothed signal. 2-3 hours of recent context is what matters.

    Implementation: for each bar t, compute weighted mean over [t-window+1, t]
    with weights = alpha × (1-alpha)^k for k=0 (most recent) to window-1
    (oldest). Normalized by sum of weights so values stay in [0, 1] range.
    """
    if span <= 1 or window_bars <= 1:
        return arr.copy()
    alpha = 2.0 / (span + 1)
    n = len(arr)
    # Precompute decay weights (most-recent-first ordering)
    w = np.array([alpha * (1.0 - alpha) ** k for k in range(window_bars)],
                 dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window_bars + 1)
        slc = arr[lo:i+1][::-1]   # most-recent first
        ww = w[:len(slc)]
        out[i] = float(np.dot(ww, slc) / ww.sum())
    return out


def compute_dx(p_long_ema: np.ndarray, p_short_ema: np.ndarray) -> np.ndarray:
    """DX = 100 × |+DI − -DI| / (+DI + -DI). Bounded [0, 100]."""
    denom = p_long_ema + p_short_ema
    denom = np.where(denom < 1e-9, 1e-9, denom)
    return 100.0 * np.abs(p_long_ema - p_short_ema) / denom


def run_state_machine(p_long_ema, p_short_ema, adx,
                      margin: float, adx_floor: float) -> tuple:
    """Returns (regime_dir, regime_change) arrays of length N.

    State-machine rules:
      - Flip to LONG when (+DI − -DI) > margin AND ADX > adx_floor
      - Flip to SHORT when (-DI − +DI) > margin AND ADX > adx_floor
      - Else: hold current state (sticky)
    """
    n = len(p_long_ema)
    regime = np.empty(n, dtype=object)
    change = np.zeros(n, dtype=bool)
    current = 'NEUTRAL'
    for i in range(n):
        pl, ps, ad = p_long_ema[i], p_short_ema[i], adx[i]
        diff = pl - ps
        new = current
        if diff > margin and ad > adx_floor:
            new = 'LONG'
        elif diff < -margin and ad > adx_floor:
            new = 'SHORT'
        if new != current:
            change[i] = True
            current = new
        regime[i] = current
    return regime, change


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True,
                    help='raw trend3 cache parquet (with p_long, p_short, p_neutral)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ema-period', type=int, default=5,
                    help='EMA period (span) for smoothing P_long/P_short (default 5 bars)')
    ap.add_argument('--adx-period', type=int, default=10,
                    help='EMA period for ADX (default 10 bars)')
    ap.add_argument('--window-bars', type=int, default=180,
                    help='Hard cutoff lookback for EMA — bars older than this '
                         'are dropped (default 180 = 3 hours at 1m cadence). '
                         'Pass 0 to use unbounded recursive EMA.')
    ap.add_argument('--margin', type=float, default=0.05,
                    help='Required +DI - -DI gap to confirm a regime (default 0.05)')
    ap.add_argument('--adx-floor', type=float, default=15.0,
                    help='Required ADX strength to confirm a regime (default 15)')
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    print(f'Loaded {len(df)} rows. Days: {df["day"].nunique()}')

    # Smooth per-day
    rows = []
    for day, gdf in df.groupby('day'):
        g = gdf.sort_values('timestamp').reset_index(drop=True)
        pl = g['p_long'].values.astype(np.float64)
        ps = g['p_short'].values.astype(np.float64)
        if args.window_bars > 0:
            pl_ema = windowed_ema(pl, args.ema_period, args.window_bars)
            ps_ema = windowed_ema(ps, args.ema_period, args.window_bars)
        else:
            pl_ema = ema(pl, args.ema_period)
            ps_ema = ema(ps, args.ema_period)
        dx = compute_dx(pl_ema, ps_ema)
        if args.window_bars > 0:
            adx = windowed_ema(dx, args.adx_period, args.window_bars)
        else:
            adx = ema(dx, args.adx_period)
        regime, change = run_state_machine(pl_ema, ps_ema, adx,
                                            args.margin, args.adx_floor)
        g['p_long_ema']    = pl_ema.astype(np.float32)
        g['p_short_ema']   = ps_ema.astype(np.float32)
        g['dx']            = dx.astype(np.float32)
        g['adx']           = adx.astype(np.float32)
        g['regime_dir']    = regime
        g['regime_change'] = change
        rows.append(g)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(args.out, index=False)
    print(f'Wrote: {args.out}')

    # Summary
    vc = out['regime_dir'].value_counts()
    print(f'\nRegime distribution: {dict(vc)}')
    n_change = int(out['regime_change'].sum())
    n_days   = out['day'].nunique()
    print(f'Regime flips: {n_change}  ({n_change / max(n_days, 1):.1f} per day)')
    print(f'\nADX stats:  median={out["adx"].median():.1f}  '
          f'p90={out["adx"].quantile(0.9):.1f}')


if __name__ == '__main__':
    main()
