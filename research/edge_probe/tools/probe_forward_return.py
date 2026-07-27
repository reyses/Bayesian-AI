#!/usr/bin/env python3
"""FORWARD-RETURN PROBE (blackboard prereq #1, owner 2026-07-26): does ANY
causal edge exist in the 1m sequence? A LINEAR probe (Ridge) predicts next-K-bar
return from PAST-ONLY OHLCV features, walk-forward OOS, on ATLAS 1m history.

Discipline (leakage gates from RIDE_EDGE_GATE_SPEC):
- Features causal (<= t only); target = fwd K-bar log return (label may use
  future; features may not). No cross-DAY target (within-day only).
- Standardization fit on TRAIN fold only (no fold bleed).
- Strictly chronological walk-forward (train past -> test next block, roll).
- Verdict on OOS INFORMATION COEFFICIENT (per-test-day IC), day-block bootstrap.
  IC significantly > 0 => causal predictability exists (blackboard has a target).
  IC ~ 0 => efficient at this scale; the loop has nothing to interpret.
CPU-only. Writes reports/probe_forward_return.md + assets/probe_ic.png.
"""
import glob
import os
import random
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(PROJ))
ATLAS = os.path.join(REPO, 'DATA', 'ATLAS', '1m')
OUT_MD = os.path.join(PROJ, 'reports', 'probe_forward_return.md')
OUT_PNG = os.path.join(PROJ, 'reports', 'assets', 'probe_ic.png')

K = 5                 # forward horizon (bars)
TRAIN_DAYS = 60
TEST_DAYS = 20
STEP = 20
N_BOOT = 4000
SEED = 42
FEATS = ['r1', 'r5', 'r15', 'r30', 'rvol30', 'volz30', 'rng', 'body', 'accel']


def day_features(df):
    c = df['close'].to_numpy(float)
    lc = np.log(c)
    out = pd.DataFrame(index=df.index)
    out['r1'] = np.concatenate([[0], np.diff(lc)])
    for k in (5, 15, 30):
        out[f'r{k}'] = lc - np.concatenate([np.full(k, lc[0]), lc[:-k]])
    out['rvol30'] = pd.Series(out['r1']).rolling(30).std().to_numpy()
    v = df['volume'].to_numpy(float)
    vm = pd.Series(v).rolling(30).mean().to_numpy()
    vs = pd.Series(v).rolling(30).std().to_numpy()
    out['volz30'] = np.where(vs > 0, (v - vm) / np.where(vs == 0, 1, vs), 0)
    out['rng'] = (df['high'].to_numpy(float) - df['low'].to_numpy(float)) / c
    out['body'] = (c - df['open'].to_numpy(float)) / c
    out['accel'] = out['r5'] - out['r15']
    # target: fwd K-bar log return, within-day (last K rows NaN)
    fwd = np.concatenate([lc[K:], np.full(K, np.nan)]) - lc
    out['y'] = fwd
    return out


def main():
    files = sorted(glob.glob(os.path.join(ATLAS, '*.parquet')))
    frames = []
    for f in files:
        day = os.path.basename(f).replace('.parquet', '')
        df = pd.read_parquet(f)
        if len(df) < 100:
            continue
        fe = day_features(df)
        fe['day'] = day
        frames.append(fe.dropna())
    data = pd.concat(frames, ignore_index=True)
    days = sorted(data['day'].unique())

    rng = random.Random(SEED)
    per_day_ic, per_day_dir, per_day_gic = {}, {}, {}
    fold_ics = []
    start = TRAIN_DAYS
    while start + TEST_DAYS <= len(days):
        tr_days = set(days[start - TRAIN_DAYS:start])
        te_days = days[start:start + TEST_DAYS]
        tr = data[data['day'].isin(tr_days)]
        Xtr = tr[FEATS].to_numpy()
        ytr = tr['y'].to_numpy()
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1
        model = Ridge(alpha=10.0)
        model.fit((Xtr - mu) / sd, ytr)
        gbm = HistGradientBoostingRegressor(max_depth=3, max_iter=120,
                                            learning_rate=0.05,
                                            random_state=SEED)
        gbm.fit(Xtr, ytr)
        fold_pred, fold_real = [], []
        for d in te_days:
            te = data[data['day'] == d]
            if len(te) < 20:
                continue
            Xte = (te[FEATS].to_numpy() - mu) / sd
            pred = model.predict(Xte)
            gpred = gbm.predict(te[FEATS].to_numpy())
            real = te['y'].to_numpy()
            if np.std(pred) > 0 and np.std(real) > 0:
                per_day_ic[d] = float(np.corrcoef(pred, real)[0, 1])
                per_day_dir[d] = float(np.mean(np.sign(pred) == np.sign(real)))
                fold_pred.append(pred)
                fold_real.append(real)
            if np.std(gpred) > 0 and np.std(real) > 0:
                per_day_gic[d] = float(np.corrcoef(gpred, real)[0, 1])
        if fold_pred:
            p = np.concatenate(fold_pred)
            r = np.concatenate(fold_real)
            fold_ics.append(float(np.corrcoef(p, r)[0, 1]))
        start += STEP

    ics = list(per_day_ic.values())
    dirs = list(per_day_dir.values())
    test_days = sorted(per_day_ic)
    mean_ic = st.mean(ics)
    # day-block bootstrap on per-day IC
    bs = sorted(st.mean([per_day_ic[d] for d in rng.choices(test_days, k=len(test_days))])
                for _ in range(N_BOOT))
    lo, hi = bs[int(0.025 * N_BOOT)], bs[int(0.975 * N_BOOT)]
    mean_dir = st.mean(dirs)
    # drift: is OOS IC decaying across folds?
    slope = np.polyfit(range(len(fold_ics)), fold_ics, 1)[0] if len(fold_ics) > 2 else 0

    sig = lo > 0
    lines = [
        '# Forward-return probe — does causal edge exist? (linear, walk-forward)',
        f'ATLAS 1m: {len(days)} days, {len(data):,} bars. K={K}-bar target. '
        f'{TRAIN_DAYS}d train / {TEST_DAYS}d test, step {STEP} = {len(fold_ics)} folds.',
        '',
        f'- **OOS Information Coefficient (per-test-day mean): {mean_ic:+.4f}**, '
        f'95% day-block CI [{lo:+.4f}, {hi:+.4f}] '
        + ('— **SIGNIFICANT: causal edge EXISTS**' if sig else
           '— not distinguishable from 0 (efficient at this scale)'),
        f'- directional accuracy: {mean_dir:.1%} (50% = chance)',
        (lambda g: f'- **NONLINEAR (gradient boosting) OOS IC: {st.mean(g):+.4f}** '
                   f'over {len(g)} days'
         )(list(per_day_gic.values())),
        f'- IC drift slope across folds: {slope:+.5f}/fold '
        + ('(decaying)' if slope < -1e-4 else '(stable)'),
        f'- test days: {len(test_days)}',
        '',
        '## Verdict for the blackboard',
        ('Causal predictability is REAL (small but significant OOS). The '
         'blackboard has a target: a flexible model (mamba) may capture more '
         'than this linear floor. Next: does a nonlinear/sequence model beat '
         'this IC OOS? If yes, dig; the IC magnitude vs costs decides '
         'tradeability.' if sig else
         'No linear causal edge at this horizon/featureset. Before shelving: '
         'a nonlinear/sequence model MIGHT find structure a linear probe '
         'cannot (interactions, temporal) — but the linear null is a strong '
         'prior that the sequence is near-efficient. The blackboard would be '
         'chasing a faint or absent signal.'),
        '',
        f'Honest scale note: an IC of ~{mean_ic:.3f} is '
        + ('tiny — even if real, tradeability depends on turnover vs the '
           '~3.6-tick round-trip cost. Predictability != profitability.'
           if abs(mean_ic) < 0.05 else 'sizeable for 1m data — verify costs.'),
    ]
    with open(OUT_MD, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(range(len(fold_ics)), fold_ics, 'o-', color='tab:blue')
    ax.axhline(0, color='black', lw=1)
    ax.axhline(mean_ic, color='tab:red', ls='--', label=f'mean IC {mean_ic:+.4f}')
    ax.fill_between(range(len(fold_ics)), lo, hi, alpha=0.15, color='tab:red',
                    label='95% CI')
    ax.set_xlabel('walk-forward fold (chronological)')
    ax.set_ylabel('OOS information coefficient')
    ax.set_title('Is there causal forward-return edge? (linear probe, OOS)')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PNG)
    print('\n'.join(lines))
    print('chart:', OUT_PNG)


if __name__ == '__main__':
    main()
