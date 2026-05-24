"""Mid-leg entry / missed-signal late-join forward pass  --  FORK 1: B9-gated.

QUESTION (user, 2026-05-20): the live L5 engine misses R-trigger entries
(cold start; or busy -- 1 contract / 1 position). When it later goes flat
while a missed leg is STILL running, can it late-join and capture +EV?

A late-joiner enters at bar K (K bars after the missed trigger), rides to the
leg's hardened zigzag-pivot exit, capturing
    remaining_pnl_usd = exit_pnl_usd - pnl_usd_so_far
which is EXACTLY B9's training target. The K-row feature vector is identical
whether or not you entered at the original trigger -> B9 is in-distribution
for this decision. No retraining, no distribution shift.

Strategies per K horizon:
  A unconditional : late-join every leg still running at K
  B principled    : join iff B9 predicted remaining > friction (Bayes rule,
                    no tuning)
  B' IS-calib     : join iff B9 pred > threshold tuned on IS, frozen for OOS

SCOPE: unconstrained opportunity test -- every OOS leg is a candidate. Live
you can only late-join while FLAT, so joined-leg counts are an UPPER BOUND.
The position-constrained sim is a separate script (midleg_constrained_sim.py).

Data : reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet (51d sealed)
       reports/findings/regret_oracle/trade_trajectory_IS.parquet (threshold calib)
Models: reports/findings/regret_oracle/b9_remaining_amplitude_K{K}.pkl
Output: reports/findings/regret_oracle/2026-05-20_midleg_fork1_b9.txt
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


K_HORIZONS = [5, 10, 30, 60, 120]          # 5s units: 25s .. 10min late
FRICTION_USD_PER_LEG = 6.0                 # round-trip commission+slippage, MNQ project convention
FRICTION_GRID = [0.0, 6.0, 12.0]           # sensitivity; also covers exit_pnl gross/net ambiguity
N_BOOTSTRAP = 4000                         # CLAUDE.md metric defs
BOOTSTRAP_SEED = 42
MODE_BIN_USD = 25.0                        # $/day mode histogram bin width (CLAUDE.md)
THRESHOLD_GRID = list(range(-20, 65, 5))   # B9 predicted-remaining-$ gate thresholds


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mode_usd(values, bin_width=MODE_BIN_USD):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan')
    lo = np.floor(values.min() / bin_width) * bin_width
    hi = np.ceil(values.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + 2 * bin_width, bin_width)
    if len(edges) < 2:
        return float(values.mean())
    counts, _ = np.histogram(values, bins=edges)
    return float(edges[int(np.argmax(counts))] + bin_width / 2.0)


def load_b9(models_dir, K):
    with open(Path(models_dir) / f'b9_remaining_amplitude_K{K}.pkl', 'rb') as f:
        return pickle.load(f)


def b9_predict(df, b9):
    feat_cols = b9['feat_cols']
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise SystemExit(f'B9 feat_cols missing from parquet ({len(missing)}): {missing[:6]}')
    return b9['model'].predict(df[feat_cols].fillna(0.0).values)


def per_day_net(df_K, mask, all_days, friction):
    """Per-day summed net P&L from late-joining masked legs, reindexed to ALL
    days so no-join days correctly contribute $0 (else $/day inflates)."""
    sub = df_K[mask]
    net = sub['remaining_pnl_usd'].values - friction
    s = pd.Series(net, index=np.asarray(sub['day'].values)).groupby(level=0).sum()
    return s.reindex(all_days, fill_value=0.0).values.astype(np.float64)


def fmt(label, per_day, n_join, n_total):
    mean = float(per_day.mean())
    lo, hi = bootstrap_ci(per_day)
    md = mode_usd(per_day)
    tag = 'SIG +' if lo > 0 else ('SIG -' if hi < 0 else 'not sig')
    return (f'  {label:<31} join {n_join:>5}/{n_total:<5} '
            f'${mean:>+7.0f}/day  mode ${md:>+6.0f}  '
            f'CI [${lo:>+7.0f},${hi:>+7.0f}]  {tag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos',
                    default='reports/findings/regret_oracle/trade_trajectory_OOS_full.parquet')
    ap.add_argument('--is-traj', dest='is_traj',
                    default='reports/findings/regret_oracle/trade_trajectory_IS.parquet')
    ap.add_argument('--models-dir', default='reports/findings/regret_oracle')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/2026-05-20_midleg_fork1_b9.txt')
    args = ap.parse_args()

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    oos = pd.read_parquet(args.oos)
    is_ = pd.read_parquet(args.is_traj)
    for d in (oos, is_):
        d['remaining_pnl_usd'] = d['exit_pnl_usd'] - d['pnl_usd_so_far']
    oos_days = sorted(oos['day'].unique())
    is_days = sorted(is_['day'].unique())
    n_days = len(oos_days)

    out('=' * 80)
    out('MID-LEG ENTRY  forward pass  --  FORK 1: B9-gated late join')
    out('=' * 80)
    out(f'OOS : {Path(args.oos).name}  {oos["leg_id"].nunique():,} legs  {n_days} days (sealed)')
    out(f'IS  : {Path(args.is_traj).name}  {is_["leg_id"].nunique():,} legs (threshold calib only)')
    out('B9  : b9_remaining_amplitude_K{K}.pkl -- trained on IS, OOS here is sealed/honest')
    out(f'Friction: ${FRICTION_USD_PER_LEG:.0f}/leg primary; sensitivity {FRICTION_GRID}')
    out('')
    out('Late-joiner enters at bar K, rides to leg hardened exit, captures')
    out('remaining_pnl_usd = exit_pnl_usd - pnl_usd_so_far  (= B9 target).')
    out('')
    out('SCOPE: unconstrained -- every OOS leg treated as a late-join candidate.')
    out('Live you can only late-join while FLAT -> joined counts are UPPER BOUND.')
    out('exit_pnl_usd gross/net ambiguity is covered by the friction grid.')
    out('')
    leg5 = oos[oos['K'] == 5]
    out('Scale ref: sum of ALL OOS leg exit P&L (overlapping, NOT 1-contract')
    out(f'  constrained) = ${leg5["exit_pnl_usd"].sum()/n_days:+.0f}/day, {len(leg5):,} legs,')
    out(f'  mean exit ${leg5["exit_pnl_usd"].mean():+.2f}  '
        f'median ${leg5["exit_pnl_usd"].median():+.2f}')
    out('')

    results = {}
    for K in K_HORIZONS:
        oos_K = oos[oos['K'] == K].reset_index(drop=True)
        is_K = is_[is_['K'] == K].reset_index(drop=True)
        b9 = load_b9(args.models_dir, K)
        results[K] = dict(
            oos_K=oos_K, is_K=is_K,
            oos_pred=b9_predict(oos_K, b9),
            is_pred=b9_predict(is_K, b9),
        )

    verdict = []
    for K in K_HORIZONS:
        r = results[K]
        oos_K, is_K = r['oos_K'], r['is_K']
        oos_pred, is_pred = r['oos_pred'], r['is_pred']
        n_tot = len(oos_K)
        rem = oos_K['remaining_pnl_usd'].values

        out('-' * 80)
        out(f'K = {K}  ({K*5}s late)   {n_tot:,} OOS legs still running at K')
        try:
            pear, _ = pearsonr(oos_pred, rem)
        except Exception:
            pear = float('nan')
        out(f'  gross remaining: mean ${rem.mean():+.2f}  median ${np.median(rem):+.2f}  '
            f'frac>0 {(rem > 0).mean()*100:.0f}%')
        out(f'  B9 pred: mean ${oos_pred.mean():+.2f}  std ${oos_pred.std():.2f}  '
            f'OOS Pearson(pred,actual) {pear:+.3f}')

        pdA = per_day_net(oos_K, np.ones(n_tot, bool), oos_days, FRICTION_USD_PER_LEG)
        out(fmt('A unconditional join', pdA, n_tot, n_tot))

        mB = oos_pred > FRICTION_USD_PER_LEG
        pdB = per_day_net(oos_K, mB, oos_days, FRICTION_USD_PER_LEG)
        out(fmt(f'B principled (pred>${FRICTION_USD_PER_LEG:.0f})', pdB, int(mB.sum()), n_tot))
        if mB.any() and (~mB).any():
            out(f'      discrimination: joined mean rem ${rem[mB].mean():+.0f}  '
                f'vs skipped ${rem[~mB].mean():+.0f}')

        best_theta, best_m = THRESHOLD_GRID[0], -1e18
        for theta in THRESHOLD_GRID:
            m = per_day_net(is_K, is_pred > theta, is_days, FRICTION_USD_PER_LEG).mean()
            if m > best_m:
                best_m, best_theta = m, theta
        mC = oos_pred > best_theta
        pdC = per_day_net(oos_K, mC, oos_days, FRICTION_USD_PER_LEG)
        out(fmt(f"B' IS-calib (pred>${best_theta})", pdC, int(mC.sum()), n_tot))
        out('      NOTE theta tuned on in-sample B9 preds -> optimistic; '
            'B principled is the honest deploy rule')

        bits = []
        for fr in FRICTION_GRID:
            bits.append(f'${fr:.0f}:${per_day_net(oos_K, mB, oos_days, fr).mean():+.0f}')
        out('      friction sensitivity (B principled): ' + '  '.join(bits))

        verdict.append((K, pdB, int(mB.sum()), n_tot))

    out('-' * 80)
    out('DIAGNOSTIC: OOS threshold sweep ($/day) -- IN-SAMPLE selection, NOT deployable')
    out('  K\\theta ' + ' '.join(f'{t:>+5}' for t in THRESHOLD_GRID))
    for K in K_HORIZONS:
        r = results[K]
        row = []
        for theta in THRESHOLD_GRID:
            v = per_day_net(r['oos_K'], r['oos_pred'] > theta,
                            oos_days, FRICTION_USD_PER_LEG).mean()
            row.append(f'{v:>+5.0f}')
        out(f'  {K:>3}    ' + ' '.join(row))

    out('=' * 80)
    out('VERDICT  --  Fork 1: B9 principled gate (pred > $6), $6 friction')
    out('=' * 80)
    any_sig = False
    for K, pdv, nj, nt in verdict:
        lo, hi = bootstrap_ci(pdv)
        mean = pdv.mean()
        if lo > 0:
            any_sig = True
            tag = f'SIGNIFICANT  +${mean:.0f}/day'
        elif hi < 0:
            tag = f'SIGNIFICANTLY NEGATIVE  ${mean:.0f}/day'
        else:
            tag = f'not significant  (${mean:+.0f}/day)'
        out(f'  K={K:>3}  join {nj:>4}/{nt:<4}  CI [${lo:>+6.0f},${hi:>+6.0f}]  {tag}')
    out('')
    if any_sig:
        out('>> At least one horizon shows a significant positive add. Mid-leg')
        out('   entry passes the unconstrained filter -> proceed to Fork 2')
        out('   (B1-B6 augmentation) and the position-constrained sim.')
    else:
        out('>> No horizon clears significance on the MOST GENEROUS test.')
        out('   B9-gated late join is not a demonstrable positive add.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
