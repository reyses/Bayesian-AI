"""Generate a Brownian-motion NULL 1s series matched to a real day.

WHY: the continuous (B2C) F-space scored 73% pristine vs tiled (B2T) 5% on 2024_02_20.
That gap may be SIGNAL — or pure autocorrelation: sliding windows smooth ANY series, so a
random walk would also be locally linear. Brownian motion has ZERO predictability by
construction, so it calibrates the NULL: run the identical B2C/B2T pristine A/B on it.
  - If Brownian also gives ~73/5  -> the gap is representation artifact, NOT signal.
  - If real >> Brownian            -> real structure clears the null (first signal evidence).

MATCHING (apples-to-apples so the stage-1 error band behaves comparably):
  - same row count & timestamps as the real day
  - close = random walk with per-second return sigma = real day's 1s close-change std
  - iid Gaussian returns (destroys all autocorrelation/structure = the random-walk null)
  - tick-rounded (0.25); OHLC built from the walk; volume = shuffled real volume

OUTPUT: DATA/ATLAS/1s/<day>_BROWN.parquet  (so the existing builders + stage-1 read it via --day <day>_BROWN).
"""
import os, sys, argparse
import numpy as np
import pandas as pd

TICK = 0.25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True, help='real source day, e.g. 2024_02_20')
    ap.add_argument('--null', choices=['brownian', 'fourier'], default='brownian',
                    help='brownian = iid random walk (no autocorr); fourier = phase-randomized '
                         '(preserves power spectrum / autocorrelation, kills phase coupling)')
    ap.add_argument('--seed', type=int, default=20240220)
    ap.add_argument('--atlas_root', default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS")))
    args = ap.parse_args()

    src = os.path.join(args.atlas_root, '1s', f'{args.day}.parquet')
    real = pd.read_parquet(src).reset_index(drop=True)
    n = len(real)
    close = real['close'].to_numpy(np.float64)
    rets = np.diff(close)
    sigma = float(np.nanstd(rets))          # match per-second volatility
    p0 = float(close[0])
    rng = np.random.default_rng(args.seed)

    if args.null == 'brownian':
        walk_rets = rng.normal(0.0, sigma, size=n - 1)    # iid -> pure random walk (no signal)
        bm = np.empty(n); bm[0] = p0; bm[1:] = p0 + np.cumsum(walk_rets)
        suffix = 'BROWN'
    else:  # fourier phase-randomized surrogate of the PRICE level (preserves power spectrum exactly)
        x = close - close.mean()
        X = np.fft.rfft(x)
        phases = rng.uniform(0, 2 * np.pi, size=X.shape[0])
        phases[0] = 0.0                                   # keep DC real
        if n % 2 == 0:
            phases[-1] = 0.0                              # keep Nyquist real
        Xs = np.abs(X) * np.exp(1j * phases)
        bm = np.fft.irfft(Xs, n=n) + close.mean()         # same |spectrum| => same autocorrelation
        suffix = 'FOUR'
    bm = np.round(bm / TICK) * TICK                        # tick grid

    opn = np.empty(n); opn[0] = bm[0]; opn[1:] = bm[:-1]   # open = prior close
    # small symmetric intrabar wick so high/low are not degenerate (scaled to per-sec sigma)
    wick = np.abs(rng.normal(0.0, sigma * 0.5, size=n))
    wick = np.round(wick / TICK) * TICK
    high = np.maximum(opn, bm) + wick
    low = np.minimum(opn, bm) - wick
    vol = real['volume'].to_numpy(np.float64).copy()
    rng.shuffle(vol)                                       # break any vol-price coupling

    out = pd.DataFrame({
        'timestamp': real['timestamp'].to_numpy(),
        'open': opn, 'high': high, 'low': low, 'close': bm, 'volume': vol,
    })
    out_path = os.path.join(args.atlas_root, '1s', f'{args.day}_{suffix}.parquet')
    out.to_parquet(out_path)
    print(f"[{suffix}] wrote {out_path}  n={n}  sigma_1s={sigma:.4f}pts  p0={p0:.2f}  null={args.null}")

if __name__ == '__main__':
    main()
