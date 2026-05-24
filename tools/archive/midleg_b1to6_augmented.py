"""Mid-leg entry  --  FORK 2: B1-B6 pivot-structure augmentation + other options.

E1 (forward_pass_midleg_entry.py) showed B9-gated late join is significantly
+EV at K=5/K=10 (within ~50s of the missed trigger) and decays to noise by
K>=30. This script tests the user's Fork-2 hypothesis -- that the B1-B6
pivot-structure models "carry leg state" and should help a mid-leg decision.

B1-B6 were trained on ALL 1m bars (B1/B3/B4/B5/B6) or all pivot rows (B2) of
the zigzag pivot dataset -> they ARE in-distribution at a mid-leg bar (a
mid-leg bar is just one more 1m bar). B2 is the exception (pivot-row trained).

EXPERIMENTS
  E2  augmentation : model A = GBM on B9's feature set (reproduces B9);
                     model B = A's features + 18 B1-B6 prediction features.
                     Both trained on full IS trajectory, tested on sealed OOS.
                     Headline = delta(B-A) in late-join $/day, 95% CI (paired).
  E4a structure gates : standalone B1/B3/B5 gates; B9 + structure filter.
  E4b pullback filter : B9 gate conditioned on intra-leg adverse excursion.

Honest note: model B is trained on IS-quality B1-B6 predictions; on OOS the
B1-B6 predictions are noisier (honest degradation) -> a mild bias AGAINST B.
If B still wins, the result is robust.

Output: reports/findings/regret_oracle/2026-05-20_midleg_fork2_b1to6.txt
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor


K_HORIZONS = [5, 10, 30, 60, 120]
FRICTION_USD_PER_LEG = 6.0                 # round-trip commission+slippage, MNQ project convention
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 42
MODE_BIN_USD = 25.0
GBM_KW = dict(max_iter=200, max_depth=6, learning_rate=0.05,
              random_state=42, l2_regularization=1.0)   # identical to B9 trainer
RD = 'reports/findings/regret_oracle'


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


def per_day_net(df_K, mask, all_days, friction=FRICTION_USD_PER_LEG):
    sub = df_K[mask]
    net = sub['remaining_pnl_usd'].values - friction
    s = pd.Series(net, index=np.asarray(sub['day'].values)).groupby(level=0).sum()
    return s.reindex(all_days, fill_value=0.0).values.astype(np.float64)


def load_pkl(name):
    with open(Path(RD) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


def build_b_features(df):
    """Run B1-B6 on df's V2 columns -> DataFrame of 18 prediction features."""
    feats = {}

    def Xof(v2_cols):
        miss = [c for c in v2_cols if c not in df.columns]
        if miss:
            raise SystemExit(f'B-model v2_cols missing from trajectory parquet '
                             f'({len(miss)}): {miss[:5]}')
        return df[v2_cols].fillna(0.0).values

    b1 = load_pkl('b1_pivot_imminent')
    for K in (1, 3, 5, 10):
        s = b1[K]
        feats[f'b1_pivot_{K}m'] = s['model'].predict_proba(Xof(s['v2_cols']))[:, 1]

    b2 = load_pkl('b2_fakeout')
    for K in (3, 5, 10):
        s = b2[K]
        feats[f'b2_fakeout_{K}m'] = s['model'].predict_proba(Xof(s['v2_cols']))[:, 1]

    b3 = load_pkl('b3_ttp_regressor')
    feats['b3_ttp_s'] = b3['model'].predict(Xof(b3['v2_cols']))

    b4 = load_pkl('b4_pivot_region')
    for W in (30, 60, 120, 300):
        s = b4[W]
        feats[f'b4_region_{W}s'] = s['model'].predict_proba(Xof(s['v2_cols']))[:, 1]

    b5 = load_pkl('b5_leg_phase')
    proba = b5['model'].predict_proba(Xof(b5['v2_cols']))
    names = b5['label_encoder'].inverse_transform(list(b5['model'].classes_))
    for i, nm in enumerate(names):
        feats[f'b5_phase_{nm}'] = proba[:, i]

    b6 = load_pkl('b6_directional_pivot')
    s = b6['models'][5]
    proba = s['model'].predict_proba(Xof(s['v2_cols']))
    names = b6['label_encoder'].inverse_transform(list(s['model'].classes_))
    for i, nm in enumerate(names):
        feats[f'b6_dir5_{nm}'] = proba[:, i]

    return pd.DataFrame(feats, index=df.index)


def b9_feat_cols():
    """B9's feature set = trajectory cols minus skip set (matches B9 trainer)."""
    return load_pkl('b9_remaining_amplitude_K5')['feat_cols']


def gate_line(label, df_K, mask, all_days):
    pdv = per_day_net(df_K, mask, all_days)
    mean = float(pdv.mean())
    lo, hi = bootstrap_ci(pdv)
    tag = 'SIG +' if lo > 0 else ('SIG -' if hi < 0 else 'not sig')
    return (f'    {label:<26} join {int(mask.sum()):>5}/{len(df_K):<5} '
            f'${mean:>+7.0f}/day  mode ${mode_usd(pdv):>+6.0f}  '
            f'CI [${lo:>+7.0f},${hi:>+7.0f}]  {tag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', default=f'{RD}/trade_trajectory_OOS_full.parquet')
    ap.add_argument('--is-traj', dest='is_traj', default=f'{RD}/trade_trajectory_IS.parquet')
    ap.add_argument('--out', default=f'{RD}/2026-05-20_midleg_fork2_b1to6.txt')
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

    out('=' * 84)
    out('MID-LEG ENTRY  --  FORK 2: B1-B6 pivot-structure augmentation + options')
    out('=' * 84)
    out(f'OOS {oos["leg_id"].nunique():,} legs / {len(oos_days)} sealed days   '
        f'IS {is_["leg_id"].nunique():,} legs')
    out('B1-B6 trained on all 1m bars (B2 on pivot rows) -> in-distribution mid-leg.')
    out(f'Friction ${FRICTION_USD_PER_LEG:.0f}/leg. Gate = principled (pred > friction).')
    out('')

    out('Building B1-B6 prediction features (18) on IS + OOS trajectory rows...')
    oos_bf = build_b_features(oos)
    is_bf = build_b_features(is_)
    aug_cols = list(oos_bf.columns)
    oos = pd.concat([oos, oos_bf], axis=1)
    is_ = pd.concat([is_, is_bf], axis=1)
    feat_cols = b9_feat_cols()
    out(f'  base features (B9 set): {len(feat_cols)}   augmentation features: {len(aug_cols)}')
    out(f'  aug: {", ".join(aug_cols)}')
    out('')

    # ---------------------------------------------------------------- E2
    out('=' * 84)
    out('E2  AUGMENTATION  --  model A (B9 features) vs model B (A + B1-B6 preds)')
    out('=' * 84)
    out('Both GBMs trained on full IS trajectory, evaluated on sealed OOS.')
    out('delta = per-day(B) - per-day(A); paired bootstrap CI over 51 days.')
    out('')
    e2_rows = []
    for K in K_HORIZONS:
        is_K = is_[is_['K'] == K].reset_index(drop=True)
        oos_K = oos[oos['K'] == K].reset_index(drop=True)
        y_is = is_K['remaining_pnl_usd'].values
        y_oos = oos_K['remaining_pnl_usd'].values

        modelA = HistGradientBoostingRegressor(**GBM_KW)
        modelA.fit(is_K[feat_cols].fillna(0.0).values, y_is)
        predA = modelA.predict(oos_K[feat_cols].fillna(0.0).values)

        modelB = HistGradientBoostingRegressor(**GBM_KW)
        modelB.fit(is_K[feat_cols + aug_cols].fillna(0.0).values, y_is)
        predB = modelB.predict(oos_K[feat_cols + aug_cols].fillna(0.0).values)

        pearA = pearsonr(predA, y_oos)[0]
        pearB = pearsonr(predB, y_oos)[0]
        pdA = per_day_net(oos_K, predA > FRICTION_USD_PER_LEG, oos_days)
        pdB = per_day_net(oos_K, predB > FRICTION_USD_PER_LEG, oos_days)
        dPD = pdB - pdA
        dlo, dhi = bootstrap_ci(dPD)
        dtag = 'SIG +' if dlo > 0 else ('SIG -' if dhi < 0 else 'not sig')

        out(f'K={K:>3}  OOS Pearson  A {pearA:+.3f} -> B {pearB:+.3f}  ({pearB-pearA:+.3f})')
        out(f'       A (B9 repro)  ${pdA.mean():>+7.0f}/day   '
            f'B (augmented)  ${pdB.mean():>+7.0f}/day')
        out(f'       delta(B-A)    ${dPD.mean():>+7.0f}/day  CI [${dlo:+.0f}, ${dhi:+.0f}]  {dtag}')
        e2_rows.append((K, pdA.mean(), pdB.mean(), dPD.mean(), dlo, dhi))
        out('')

    # ---------------------------------------------------------------- corr
    out('=' * 84)
    out('B1-B6 FEATURE CORRELATION with remaining_pnl_usd  (OOS, Pearson)')
    out('=' * 84)
    for K in (5, 60):
        oos_K = oos[oos['K'] == K]
        cors = [(c, oos_K[c].corr(oos_K['remaining_pnl_usd'])) for c in aug_cols]
        cors.sort(key=lambda t: -abs(t[1]))
        out(f'K={K:>3}  ' + '   '.join(f'{c}:{v:+.3f}' for c, v in cors[:6]))
    out('')

    # ---------------------------------------------------------------- E4a
    out('=' * 84)
    out('E4a  STANDALONE STRUCTURE GATES  (no OOS tuning -- principled thresholds)')
    out('=' * 84)
    for K in (5, 30, 60):
        is_K = is_[is_['K'] == K].reset_index(drop=True)
        oos_K = oos[oos['K'] == K].reset_index(drop=True)
        # B9 baseline gate
        modelA = HistGradientBoostingRegressor(**GBM_KW)
        modelA.fit(is_K[feat_cols].fillna(0.0).values, is_K['remaining_pnl_usd'].values)
        predA = modelA.predict(oos_K[feat_cols].fillna(0.0).values)
        g_b9 = predA > FRICTION_USD_PER_LEG
        # structure signals
        b5_early = oos_K['b5_phase_EARLY'].values > oos_K['b5_phase_LATE'].values
        b1_low = oos_K['b1_pivot_5m'].values < 0.5
        ttp_thr = float(is_K['b3_ttp_s'].median())   # frozen on IS
        b3_hi = oos_K['b3_ttp_s'].values > ttp_thr
        g_struct = b5_early & b1_low

        out(f'K={K:>3}  (b3_ttp median threshold from IS = {ttp_thr:.0f}s)')
        out(gate_line('B9 only (pred>$6)', oos_K, g_b9, oos_days))
        out(gate_line('b5 EARLY>LATE', oos_K, b5_early, oos_days))
        out(gate_line('b1 pivot_5m<0.5', oos_K, b1_low, oos_days))
        out(gate_line('b3 ttp>IS-median', oos_K, b3_hi, oos_days))
        out(gate_line('struct (b5early&b1low)', oos_K, g_struct, oos_days))
        out(gate_line('B9 & b5early', oos_K, g_b9 & b5_early, oos_days))
        out(gate_line('B9 & struct', oos_K, g_b9 & g_struct, oos_days))
        out('')

    # ---------------------------------------------------------------- E4b
    out('=' * 84)
    out('E4b  PULLBACK FILTER  --  B9 gate conditioned on intra-leg adverse excursion')
    out('=' * 84)
    out('Hypothesis: late-joining after a pullback (higher mae_pts_so_far) is a')
    out('better entry than chasing. Among B9 gated joins, require mae >= threshold.')
    out('')
    for K in (10, 30, 60):
        is_K = is_[is_['K'] == K].reset_index(drop=True)
        oos_K = oos[oos['K'] == K].reset_index(drop=True)
        modelA = HistGradientBoostingRegressor(**GBM_KW)
        modelA.fit(is_K[feat_cols].fillna(0.0).values, is_K['remaining_pnl_usd'].values)
        predA = modelA.predict(oos_K[feat_cols].fillna(0.0).values)
        g_b9 = predA > FRICTION_USD_PER_LEG
        mae = oos_K['mae_pts_so_far'].values
        out(f'K={K:>3}')
        for mae_thr in (0.0, 2.0, 4.0, 6.0):
            out(gate_line(f'B9 & mae>={mae_thr:.0f}pt', oos_K, g_b9 & (mae >= mae_thr), oos_days))
        out('')

    # ---------------------------------------------------------------- verdict
    out('=' * 84)
    out('VERDICT  --  Fork 2 (B1-B6 augmentation)')
    out('=' * 84)
    any_aug_sig = False
    for K, a, b, d, dlo, dhi in e2_rows:
        if dlo > 0:
            any_aug_sig = True
            v = f'augmentation SIGNIFICANTLY HELPS (+${d:.0f}/day)'
        elif dhi < 0:
            v = f'augmentation SIGNIFICANTLY HURTS (${d:.0f}/day)'
        else:
            v = f'no significant effect (${d:+.0f}/day, CI crosses 0)'
        out(f'  K={K:>3}  A ${a:+.0f} -> B ${b:+.0f}   {v}')
    out('')
    if any_aug_sig:
        out('>> B1-B6 augmentation adds significant lift at >=1 horizon. Fork 2')
        out('   is worth pursuing -- consider B1-B6 during-trade analogs.')
    else:
        out('>> B1-B6 augmentation does NOT add significant lift. The B9 GBM')
        out('   already extracts the pivot-structure signal from the raw V2')
        out('   features; stacked B1-B6 predictions add only noise. Fork 1')
        out('   (B9 alone) stands. See E4a for whether structure FILTERS help.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
