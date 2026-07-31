#!/usr/bin/env python
"""CUBIC-AS-FAKEOUT-FILTER + CHURN AMPLITUDE (owner 2026-07-31: "add the cubic
a fakeout filter on it? Also the churn we saw -- measure the amplitude and see
if we can harness it with NMP and/or filter it out from the cubic").

PART A -- cubic fakeout filter, on ground truth:
  For every intra-leg dip TOUCH (edge-triggered crossing of the 80%-of-running-
  MFE line, peak >= MINPK) inside hindsight-zigzag legs, compute CAUSAL cubic
  features at the touch bar (5s closes, window 90 = 7.5min, matching the
  deployed spec): endpoint slope (pts/min) and residual z. Label the touch
  FAKEOUT if a new leg-high occurs after it (before the leg ends), TERMINAL if
  it is the slide into the true cusp. Question: does cubic state at the touch
  separate the two better than chance / than depth alone?

PART B -- churn:
  Churn windows = causal rolling Kaufman ER (5min) below a threshold for a
  minimum duration. Measure swing amplitude inside windows (small-R zigzag),
  window durations, then two verdicts:
   - NMP-harness: does the median intra-churn swing clear round-trip friction
     with enough margin to be worth a real NMP-ported test?
   - cubic filter: what fraction of cubic slope sign-flips happen inside churn
     windows (flips removable by gating) vs the fraction of time blanked?

Read-only; 5s data; nothing live touched.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cusp_ground_truth import hindsight_zigzag, D5, FRICTION_PT, PT_USD
import cubic_regression as _cub

MINPK = 10.0
WARN = 0.80
CUBIC_W = 90          # 5s bars = 7.5 min (deployed NT8 spec basis)
ER_W = 60             # 5s bars = 5 min
ER_CHOP = 0.10        # deepest-chop bin boundary from the adaptive-stop study
CHURN_MIN = 36        # bars = 3 min minimum window


def efficiency_ratio(c, W):
    d = np.abs(np.diff(c, prepend=c[0]))
    den = pd.Series(d).rolling(W, min_periods=W).sum().to_numpy()
    num = np.abs(c - pd.Series(c).shift(W).to_numpy())
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def leg_touches(hi, lo, piv, min_leg=20.0):
    """Yield [touch_bar, leg_dir, depth_pt, peak_pt, label] for every 80%-line
    touch. label: 1=fakeout (new leg-high followed), 0=terminal.

    FIX 2026-07-31: walk (i0, i2] -- through the NEXT pivot -- because a
    ground-truth leg ENDS at its cusp, so the terminal decline lives in the
    following leg's span. Walking only to i1 makes terminal touches
    unobservable by construction (first run: 36,333 touches, zero terminal)."""
    out = []
    for j in range(len(piv) - 2):
        (i0, p0, k0), (i1, p1, k1), (i2, p2, k2) = piv[j], piv[j + 1], piv[j + 2]
        if i1 - i0 < 3: continue
        d = 1 if k0 == 'B' else -1
        if (p1 - p0) * d < min_leg: continue
        peak, below = 0.0, False
        pend = []
        for i in range(i0 + 1, i2 + 1):
            fav = (hi[i] - p0) if d > 0 else (p0 - lo[i])
            adv = (lo[i] - p0) if d > 0 else (p0 - hi[i])
            if i <= i1 and fav > peak:
                for t_ in pend: t_[4] = 1              # exceeded -> fakeouts
                out += pend; pend = []
                peak, below = fav, False
                continue
            if peak >= MINPK:
                nb = adv <= peak * WARN
                if nb and not below:
                    pend.append([i, d, peak - adv, peak, 0])
                below = nb
        out += pend                                    # unresolved = terminal
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=60)
    ap.add_argument('--R', type=float, default=30.0)
    a = ap.parse_args()
    days = sorted(f[:-8] for f in os.listdir(D5) if f.endswith('.parquet'))[-a.days:]

    # ---------- PART A ----------
    rows = []
    churn_stats = dict(flips_in=0, flips_out=0, bars_in=0, bars_out=0, amps=[], durs=[])
    for day in tqdm(days, desc='scan'):
        df = pd.read_parquet(os.path.join(D5, f'{day}.parquet'))
        if len(df) < 1000: continue
        hi, lo, cl = df['high'].to_numpy(), df['low'].to_numpy(), df['close'].to_numpy()
        cub, slp, _ = _cub.rolling(cl, CUBIC_W, 5)
        res = cl - cub
        sig = pd.Series(res).rolling(240, min_periods=60).std().to_numpy()
        piv = hindsight_zigzag(hi, lo, a.R)
        for i, d, depth, peak, lab in leg_touches(hi, lo, piv):
            if not np.isfinite(slp[i]): continue
            z = res[i] / sig[i] if (np.isfinite(sig[i]) and sig[i] > 0) else np.nan
            rows.append(dict(day=day, i=i, dir=d, depth=depth, peak=peak, lab=lab,
                             slope=slp[i] * d,          # >0 = cubic WITH the leg
                             z=z * d if np.isfinite(z) else np.nan))
        # ---------- PART B ----------
        er = efficiency_ratio(cl, ER_W)
        in_churn = er < ER_CHOP
        in_churn = np.where(np.isfinite(er), in_churn, False)
        # min-duration windows
        w = np.zeros(len(cl), bool); run = 0
        for j in range(len(cl)):
            run = run + 1 if in_churn[j] else 0
            if run >= CHURN_MIN: w[j - run + 1:j + 1] = True
        sflip = np.diff(np.sign(slp), prepend=np.nan) != 0
        sflip &= np.isfinite(slp)
        churn_stats['flips_in'] += int(np.nansum(sflip & w))
        churn_stats['flips_out'] += int(np.nansum(sflip & ~w))
        churn_stats['bars_in'] += int(w.sum()); churn_stats['bars_out'] += int((~w).sum())
        # amplitude inside each maximal window: small-R zigzag swings
        j = 0
        while j < len(cl):
            if not w[j]: j += 1; continue
            k = j
            while k < len(cl) and w[k]: k += 1
            seg_h, seg_l = hi[j:k], lo[j:k]
            if k - j >= CHURN_MIN:
                pv = hindsight_zigzag(seg_h, seg_l, 5.0)
                swings = [abs(p2 - p1) for (_, p1, _), (_, p2, _) in zip(pv, pv[1:])]
                churn_stats['amps'] += swings
                churn_stats['durs'].append((k - j) * 5 / 60.0)
            j = k

    r = pd.DataFrame(rows)
    print(f"\n===== PART A -- CUBIC AT THE DIP TOUCH (R={a.R:g}, {len(r)} touches: "
          f"{int(r.lab.sum())} fakeout / {int((1-r.lab).sum())} terminal) =====")
    r['aligned'] = r['slope'] > 0
    for cond, name in [(r.aligned, 'cubic WITH leg'), (~r.aligned, 'cubic AGAINST leg')]:
        g = r[cond]
        print(f"  {name:>18}: n={len(g):>6}  P(fakeout)={100*g.lab.mean():5.1f}%")
    # slope magnitude quartiles
    print("  by slope (signed, pts/min) quartile:")
    r['q'] = pd.qcut(r['slope'], 4, labels=False, duplicates='drop')
    for q in sorted(r['q'].dropna().unique()):
        g = r[r['q'] == q]
        print(f"    q{int(q)} [{g.slope.min():+6.1f},{g.slope.max():+6.1f}]  n={len(g):>6}  "
              f"P(fakeout)={100*g.lab.mean():5.1f}%")
    # crude AUC (rank-based) of slope for fakeout-vs-terminal
    x1 = r[r.lab == 1]['slope'].to_numpy(); x0 = r[r.lab == 0]['slope'].to_numpy()
    ranks = pd.Series(np.concatenate([x1, x0])).rank().to_numpy()
    auc = (ranks[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0))
    print(f"  AUC(slope -> fakeout) = {auc:.3f}   (0.5 = useless)")
    zv = r.dropna(subset=['z'])
    x1 = zv[zv.lab == 1]['z'].to_numpy(); x0 = zv[zv.lab == 0]['z'].to_numpy()
    ranks = pd.Series(np.concatenate([x1, x0])).rank().to_numpy()
    auc_z = (ranks[:len(x1)].sum() - len(x1) * (len(x1) + 1) / 2) / (len(x1) * len(x0))
    print(f"  AUC(resid z -> fakeout) = {auc_z:.3f}")

    amps = np.array(churn_stats['amps']); durs = np.array(churn_stats['durs'])
    fi, fo = churn_stats['flips_in'], churn_stats['flips_out']
    bi, bo = churn_stats['bars_in'], churn_stats['bars_out']
    print(f"\n===== PART B -- CHURN (ER<{ER_CHOP:g} for >=3min, {a.days} days) =====")
    print(f"  time in churn: {100*bi/(bi+bo):.1f}%   windows: {len(durs)}  "
          f"median duration {np.median(durs) if len(durs) else float('nan'):.1f} min")
    if len(amps):
        print(f"  intra-churn swing amplitude (R=5 zigzag): "
              f"p25={np.percentile(amps,25):.1f}  p50={np.percentile(amps,50):.1f}  "
              f"p75={np.percentile(amps,75):.1f} pt   (n={len(amps)} swings)")
        print(f"  NMP-harness arithmetic: median swing {np.median(amps):.1f}pt vs "
              f"friction {FRICTION_PT}pt round-trip -> gross:net ratio "
              f"{np.median(amps)/FRICTION_PT:.1f}x IF both sides captured; realistically "
              f"capture <= half a swing per fade.")
    print(f"  cubic slope sign-flips: {fi} inside churn / {fo} outside")
    print(f"  -> gating the cubic by this churn flag removes {100*fi/(fi+fo):.1f}% of "
          f"all sign-flips while blanking {100*bi/(bi+bo):.1f}% of the session")
    flips_per_hr_in = fi / (bi * 5 / 3600) if bi else float('nan')
    flips_per_hr_out = fo / (bo * 5 / 3600) if bo else float('nan')
    print(f"  flip rate: {flips_per_hr_in:.1f}/hr inside vs {flips_per_hr_out:.1f}/hr outside")


if __name__ == '__main__':
    main()
