"""Drawdown-conditional stop analysis.

User proposal (2026-05-20): cut a trade at -$100 UNLESS there is statistical
evidence it will recoup. Refinement of a dumb hard stop -- the cut is the
DEFAULT, holding requires evidence (a cleaner action surface than the failed
C11 binary cut, where cutting required confidence).

Measures, on the 51-day sealed OOS trajectory dataset:
  1. Of legs that reach -$100 drawdown (MAE >= 50 pts, MNQ $2/pt), what
     fraction recover above -$100 / finish green?
  2. Dumb stop -- cut every leg at the drawdown level. $/day delta + worst
     trade. Swept over levels {-60..-150}.
  3. B9-conditional stop -- at the -$100 checkpoint, HOLD iff B9 predicts the
     leg exits above the stop level (predicted_exit = pnl_so_far + B9 pred
     remaining); else CUT. Cut is the default; holding needs B9 evidence.
  4. Fresh model -- HistGB classifier trained ONLY on the IS -$100-drawdown
     subpopulation (checkpoint features -> recovered?), evaluated on OOS.
     Does specialising beat off-the-shelf B9?

Friction cancels in every comparison (same entry, same single exit) so it is
omitted. Stop slippage is modelled explicitly ($5 primary, {$0,$10} sens).
Checkpoint = first K-horizon row where cumulative MAE crosses the level; the
cut/hold $ outcomes are exact, only the B9-eval timing is snapped to the
K-grid (verdict robust to +/- a few bars).

Output: reports/findings/regret_oracle/2026-05-20_drawdown_stop_analysis.txt
"""
from __future__ import annotations
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RD = 'reports/findings/regret_oracle'
K_HORIZONS = [5, 10, 30, 60, 120]
DOLLAR_PER_POINT = 2.0                       # MNQ
DRAWDOWN_LEVELS = [-60.0, -80.0, -100.0, -120.0, -150.0]
PRIMARY_LEVEL = -100.0
STOP_SLIP_PRIMARY = 5.0                      # adverse fill when the stop triggers
STOP_SLIP_GRID = [0.0, 5.0, 10.0]
N_BOOTSTRAP = 4000
BOOTSTRAP_SEED = 42
HOLD_MARGIN_GRID = [-100.0, -50.0, 0.0]      # hold iff B9 predicted_exit > this


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_b9(K):
    with open(Path(RD) / f'b9_remaining_amplitude_K{K}.pkl', 'rb') as f:
        return pickle.load(f)


def add_b9_pred(traj):
    """Add b9_pred_remaining to every K-row (per-K B9 model)."""
    traj = traj.copy()
    traj['b9_pred_remaining'] = np.nan
    feat_cols = None
    for K in K_HORIZONS:
        b9 = load_b9(K)
        feat_cols = b9['feat_cols']
        m = traj['K'] == K
        traj.loc[m, 'b9_pred_remaining'] = b9['model'].predict(
            traj.loc[m, feat_cols].fillna(0.0).values)
    return traj, feat_cols


def checkpoints(traj, level):
    """First K-row per leg where cumulative MAE crosses the drawdown level."""
    thr_pts = abs(level) / DOLLAR_PER_POINT
    hit = traj[traj['mae_pts_so_far'] >= thr_pts]
    return (hit.sort_values(['leg_id', 'K'])
               .groupby('leg_id', sort=False).head(1).reset_index(drop=True))


def per_day_delta(cp, cut_mask, level, slip, all_days):
    """Per-day summed P&L delta vs no-stop. Cut legs realise level-slip;
    held legs realise their natural exit (delta 0)."""
    exit_pnl = cp['exit_pnl_usd'].values
    outcome = np.where(cut_mask, level - slip, exit_pnl)
    delta = outcome - exit_pnl
    s = pd.Series(delta, index=np.asarray(cp['day'].values)).groupby(level=0).sum()
    return s.reindex(all_days, fill_value=0.0).values.astype(np.float64)


def worst_trade(all_legs, cp, cut_mask, level, slip):
    """Worst single-trade P&L when the rule cuts the masked checkpoint legs."""
    pnl = dict(zip(all_legs['leg_id'].values, all_legs['exit_pnl_usd'].values))
    for lid, cut in zip(cp['leg_id'].values, cut_mask):
        if cut:
            pnl[lid] = level - slip
    return min(pnl.values())


def summarise(label, pdv, n_cut, n_hit, out):
    mean = float(pdv.mean())
    lo, hi = bootstrap_ci(pdv)
    tag = 'SIG +' if lo > 0 else ('SIG -' if hi < 0 else 'not sig')
    out(f'  {label:<34} cut {n_cut:>4}/{n_hit:<4}  '
        f'${mean:>+7.0f}/day vs no-stop  CI [${lo:>+6.0f},${hi:>+6.0f}]  {tag}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', default=f'{RD}/trade_trajectory_OOS_full.parquet')
    ap.add_argument('--is-traj', dest='is_traj', default=f'{RD}/trade_trajectory_IS.parquet')
    ap.add_argument('--out', default=f'{RD}/2026-05-20_drawdown_stop_analysis.txt')
    args = ap.parse_args()

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    oos = pd.read_parquet(args.oos)
    is_ = pd.read_parquet(args.is_traj)
    for d in (oos, is_):
        d['remaining_pnl_usd'] = d['exit_pnl_usd'] - d['pnl_usd_so_far']
    oos, feat_cols = add_b9_pred(oos)
    is_, _ = add_b9_pred(is_)
    oos_days = sorted(oos['day'].unique())
    all_legs = oos.drop_duplicates('leg_id')[['leg_id', 'day', 'exit_pnl_usd']]
    n_legs = len(all_legs)

    out('=' * 84)
    out('DRAWDOWN-CONDITIONAL STOP ANALYSIS  --  "cut at -$100 unless evidence of recovery"')
    out('=' * 84)
    out(f'OOS {n_legs:,} legs / {len(oos_days)} sealed days. MNQ $2/pt.')
    out(f'Stop slippage ${STOP_SLIP_PRIMARY:.0f} primary; friction omitted (cancels).')
    out('')

    # ---- 1. recovery base rate at each level --------------------------------
    out('-' * 84)
    out('1. DRAWDOWN-HIT POPULATION & RECOVERY BASE RATE')
    out('-' * 84)
    for L in DRAWDOWN_LEVELS:
        cp = checkpoints(oos, L)
        if len(cp) == 0:
            out(f'  {L:>+6.0f}: no legs hit this drawdown')
            continue
        ex = cp['exit_pnl_usd'].values
        out(f'  hit {L:>+6.0f} ({abs(L)/DOLLAR_PER_POINT:.0f}pt MAE): '
            f'{len(cp):>4} legs ({len(cp)/n_legs*100:4.1f}% of all)  |  '
            f'exit > {L:+.0f}: {(ex>L).mean()*100:4.1f}%   '
            f'exit > $0 (green): {(ex>0).mean()*100:4.1f}%   '
            f'mean exit ${ex.mean():+.0f}')
    out('')
    out('  Reading: "exit > level" = the leg recovered above where the stop would')
    out('  have cut it -> a dumb stop COSTS money on those. Higher = stop hurts more.')
    out('')

    # ---- 2. dumb stop, level + slippage sweep -------------------------------
    out('-' * 84)
    out('2. DUMB STOP  (cut every drawdown-hit leg, no recovery check)')
    out('-' * 84)
    for L in DRAWDOWN_LEVELS:
        cp = checkpoints(oos, L)
        if len(cp) == 0:
            continue
        bits = []
        for slip in STOP_SLIP_GRID:
            pdv = per_day_delta(cp, np.ones(len(cp), bool), L, slip, oos_days)
            bits.append(f'slip${slip:.0f}:${pdv.mean():+.0f}')
        wt = worst_trade(all_legs, cp, np.ones(len(cp), bool), L, STOP_SLIP_PRIMARY)
        out(f'  level {L:>+6.0f}:  ' + '  '.join(bits) +
            f'   worst trade ${wt:+.0f}')
    out(f'  (no-stop worst trade: ${all_legs["exit_pnl_usd"].min():+.0f})')
    out('')

    # ---- 3. B9-conditional stop at -$100 ------------------------------------
    L = PRIMARY_LEVEL
    slip = STOP_SLIP_PRIMARY
    cp = checkpoints(oos, L)
    n_hit = len(cp)
    out('-' * 84)
    out(f'3. B9-CONDITIONAL STOP at {L:+.0f}  ({n_hit} OOS legs hit this drawdown)')
    out('-' * 84)
    pred_exit = cp['pnl_usd_so_far'].values + cp['b9_pred_remaining'].values
    recovered = cp['exit_pnl_usd'].values > L
    # B9 discrimination
    try:
        auc_b9 = roc_auc_score(recovered, pred_exit)
    except Exception:
        auc_b9 = float('nan')
    out(f'  B9 separates recoverers from goners:  OOS AUC {auc_b9:.3f}  '
        f'(0.5 = no signal)')
    out(f'  base recovery rate (exit > {L:+.0f}): {recovered.mean()*100:.1f}%')
    out('')
    out('  Rule: HOLD iff B9 predicted_exit > margin, else CUT. Cut is default.')
    pdv_dumb = per_day_delta(cp, np.ones(n_hit, bool), L, slip, oos_days)
    summarise('dumb stop (cut all)', pdv_dumb, n_hit, n_hit, out)
    for margin in HOLD_MARGIN_GRID:
        hold = pred_exit > margin
        cut = ~hold
        pdv = per_day_delta(cp, cut, L, slip, oos_days)
        # decision quality
        good_hold = int((hold & recovered).sum())
        bad_hold = int((hold & ~recovered).sum())
        summarise(f'B9-cond (hold if pred>{margin:+.0f})', pdv, int(cut.sum()), n_hit, out)
        wt = worst_trade(all_legs, cp, cut, L, slip)
        out(f'      held {int(hold.sum())} ({good_hold} recovered / {bad_hold} went worse)'
            f'   worst trade ${wt:+.0f}')
    out('')

    # ---- 4. fresh model trained on the drawdown subpopulation ---------------
    out('-' * 84)
    out(f'4. FRESH MODEL  (HistGB classifier trained on IS {L:+.0f}-drawdown legs)')
    out('-' * 84)
    cp_is = checkpoints(is_, L)
    y_is = (cp_is['exit_pnl_usd'].values > L).astype(int)
    out(f'  IS training legs: {len(cp_is)}   (recovery rate {y_is.mean()*100:.1f}%)')
    if len(cp_is) < 200 or y_is.sum() < 30 or (len(y_is) - y_is.sum()) < 30:
        out('  -> too few IS drawdown legs to train a reliable model; skipping.')
    else:
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, learning_rate=0.05,
            random_state=42, l2_regularization=1.0)
        clf.fit(cp_is[feat_cols].fillna(0.0).values, y_is)
        p_rec = clf.predict_proba(cp[feat_cols].fillna(0.0).values)[:, 1]
        try:
            auc_fresh = roc_auc_score(recovered, p_rec)
        except Exception:
            auc_fresh = float('nan')
        out(f'  fresh-model OOS AUC {auc_fresh:.3f}   (vs B9 AUC {auc_b9:.3f})')
        for thr in (0.5, 0.6, 0.7):
            hold = p_rec > thr
            cut = ~hold
            pdv = per_day_delta(cp, cut, L, slip, oos_days)
            good_hold = int((hold & recovered).sum())
            bad_hold = int((hold & ~recovered).sum())
            summarise(f'fresh-cond (hold if P>{thr:.1f})', pdv, int(cut.sum()), n_hit, out)
            wt = worst_trade(all_legs, cp, cut, L, slip)
            out(f'      held {int(hold.sum())} ({good_hold} recovered / {bad_hold} went worse)'
                f'   worst trade ${wt:+.0f}')
    out('')

    # ---- verdict ------------------------------------------------------------
    out('=' * 84)
    out('VERDICT')
    out('=' * 84)
    out(f'no-stop worst trade ${all_legs["exit_pnl_usd"].min():+.0f}; '
        f'dumb -$100 stop caps it near ${L - slip:.0f}.')
    out('The dumb-stop $/day delta (section 2) is the cost of the tail cap.')
    out('A conditional stop only beats the dumb stop if B9 / the fresh model')
    out(f'has real AUC at {L:+.0f} (section 3-4). AUC ~0.5 => no evidence => the')
    out('"unless it might recoup" clause cannot be honoured -- it is a dumb stop.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
