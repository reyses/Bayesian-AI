"""Forward-increment predictive test — the signal discriminator autocorrelation can't fake.

LOGIC (per the 2026-06-18 discussion): the segment pristine% is in-sample regress-ability of the
cumulative PATH (close - anchor), which is highly autocorrelated -> a smooth random walk fits it too.
The honest test is predicting the INCREMENT close[t+h]-close[t] FORWARD, out-of-sample: increments are
near-independent for price, so a positive OOS R^2 cannot come from autocorrelation freebies.

NULL CALIBRATION: run identical test on Brownian (no structure) and Fourier (matched power spectrum,
phase-randomized). Real must beat BOTH — especially Fourier — to claim structure beyond linear autocorr.

CAUSAL: features are trailing-window (causal); target is strictly forward; train/test are time-disjoint
with an h-row embargo so no target straddles the split. No segment/tier labels used (those are non-causal).

METRIC: out-of-sample R^2 (1 - SSres/SStot vs the TRAIN mean baseline; negative => worse than predicting
the mean), plus pred-vs-actual correlation and directional accuracy. Signal iff real OOS R^2 > 0 AND > null.

Writes reports/findings/forward_increment_2024_02_20.md
"""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ATLAS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "DATA", "ATLAS"))
HORIZONS = [1, 5, 30, 60]      # forward seconds (rows on the 1s grid)
TRAIN_FRAC = 0.70
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]

CELLS = [
    ('B2C_continuous', 'FEATURES_RUN_B2C', '2024_02_20',       'REAL'),
    ('B2C_continuous', 'FEATURES_RUN_B2C', '2024_02_20_BROWN', 'BROWN'),
    ('B2C_continuous', 'FEATURES_RUN_B2C', '2024_02_20_FOUR',  'FOUR'),
    ('B2T_tiled',      'FEATURES_RUN_B2T', '2024_02_20',       'REAL'),
    ('B2T_tiled',      'FEATURES_RUN_B2T', '2024_02_20_BROWN', 'BROWN'),
    ('B2T_tiled',      'FEATURES_RUN_B2T', '2024_02_20_FOUR',  'FOUR'),
]


def run_cell(froot, day):
    feat = pd.read_parquet(os.path.join(ATLAS, froot, f'{day}.parquet'))
    close = pd.read_parquet(os.path.join(ATLAS, '1s', f'{day}.parquet'))['close'].to_numpy(np.float64)
    n = min(len(feat), len(close))
    feat = feat.iloc[:n].reset_index(drop=True); close = close[:n]
    Xcols = [c for c in feat.columns if c != 'timestamp' and 'price_mean' not in c and 'vwap' not in c]
    X = feat[Xcols].to_numpy(np.float64)

    out = {}
    for h in HORIZONS:
        y = np.full(n, np.nan); y[:n - h] = close[h:] - close[:n - h]   # forward increment
        valid = (~np.isnan(X).any(axis=1)) & (~np.isnan(y))
        idx = np.where(valid)[0]
        if len(idx) < 2000:
            out[h] = None; continue
        cut = int(len(idx) * TRAIN_FRAC)
        tr = idx[:max(1, cut - h)]    # embargo h rows so no train target overlaps test window
        te = idx[cut:]
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=ALPHAS).fit(sc.transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        ytr_mean = y[tr].mean()
        ss_res = float(np.sum((y[te] - pred) ** 2))
        ss_tot = float(np.sum((y[te] - ytr_mean) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        corr = float(np.corrcoef(pred, y[te])[0, 1]) if np.std(pred) > 1e-12 else 0.0
        diracc = float(np.mean(np.sign(pred) == np.sign(y[te])))
        out[h] = dict(r2=r2, corr=corr, diracc=diracc, n_test=len(te))
    return out


def main():
    rows = []
    print(f"Forward-increment OOS test | horizons(s)={HORIZONS} | train_frac={TRAIN_FRAC}\n")
    hdr = f"{'cell':22s} | " + " | ".join(f"h={h:<3d} R2(corr/dir)" for h in HORIZONS)
    print(hdr); print("-" * len(hdr))
    results = {}
    for rep, froot, day, lab in CELLS:
        res = run_cell(froot, day)
        results[(rep, lab)] = res
        cells = []
        for h in HORIZONS:
            r = res.get(h)
            cells.append(f"{r['r2']:+.4f}({r['corr']:+.2f}/{r['diracc']:.0%})" if r else "  n/a       ")
        line = f"{rep+'/'+lab:22s} | " + " | ".join(cells)
        print(line); rows.append(line)

    # real-minus-null OOS R^2 deltas (the signal test)
    print("\n=== real - null  OOS R^2 (per horizon) ===")
    deltas = []
    for rep in ('B2C_continuous', 'B2T_tiled'):
        for null in ('BROWN', 'FOUR'):
            real = results.get((rep, 'REAL')); nul = results.get((rep, null))
            if not real or not nul: continue
            d = " ".join(f"h{h}:{(real[h]['r2']-nul[h]['r2']):+.4f}" for h in HORIZONS if real.get(h) and nul.get(h))
            line = f"{rep:16s} REAL-{null:5s}: {d}"
            print(line); deltas.append(line)

    md = "# Forward-increment OOS predictive test — 2024_02_20\n\n"
    md += f"Predict close[t+h]-close[t] from causal F-space at t. Train/test 70/30 time-disjoint, embargo h, "
    md += f"RidgeCV. Metric = OOS R^2 (vs train-mean baseline; <0 = worse than mean), (corr/dir-acc).\n\n```\n"
    md += hdr + "\n" + "\n".join(rows) + "\n\n=== real - null OOS R^2 ===\n" + "\n".join(deltas) + "\n```\n"
    md += "\nSIGNAL iff real OOS R^2 > 0 AND > both nulls. Nulls expected ~0 (no forecastable structure).\n"
    outp = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports",
                                        "forward_increment_2024_02_20.md"))
    open(outp, 'w').write(md)
    print(f"\nwrote {outp}")


if __name__ == '__main__':
    main()
