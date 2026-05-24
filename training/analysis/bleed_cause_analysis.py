"""Forward pass analysis of OOS bleed zones — what STRUCTURAL conditions
distinguish bleed zones from profit zones?

Loser autopsy showed three bleed zones in production OOS NMP_REGIME:
  A. Hours 14-17 UTC (NY mid-session)        net -$2,168, 62% of all loser$
  B. FLAT_CHOPPY regime                       net -$3,395 of which losers $-29K
  C. Round-trip losers (peak >$20 then lost)  64% of losers, $-35K

The question isn't WHERE these bleed; it's WHY. For each zone, this script
compares the V2 feature distribution at entry between:
    bleed_population vs non_bleed_population
    bleed_zone WINNERS vs bleed_zone LOSERS

Cohen's d on each V2 column reveals which features SHIFT between populations.
Large positive d on volatility-class features confirms a "vol expansion"
mechanism; large d on regime-state features (z_se, hurst, reversion_prob)
suggests a regime-misclassification mechanism.
"""
from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from core_v2.features import FEATURE_NAMES
from training.regret.regret import RegretLabel
from training.ledger import ClosedTrade
from training.calibration.tier_discovery import cohens_d
from training.analysis.feature_eda import spearman_corr
from training.utils.state import REGIME_VOCAB


def load_full(trades_path: str, regret_path: str) -> pd.DataFrame:
    with open(trades_path, 'rb') as f:
        trades: List[ClosedTrade] = pickle.load(f)
    with open(regret_path, 'rb') as f:
        labels: List[RegretLabel] = pickle.load(f)
    label_by_key = {(l.entry_day, l.entry_ts): l for l in labels}
    rows = []
    for t in trades:
        l = label_by_key.get((t.entry_day, t.entry_ts))
        if l is None or t.entry_v2 is None or len(t.entry_v2) != len(FEATURE_NAMES):
            continue
        rows.append({
            'day': t.entry_day, 'ts': t.entry_ts,
            'tier': t.entry_tier, 'direction': t.direction,
            'regime_idx': t.entry_regime_idx, 'pnl': t.pnl,
            'bars_held': t.bars_held, 'exit_reason': t.exit_reason,
            'peak_pnl': l.peak_pnl, 'mae_pnl': l.mae_pnl,
            'time_to_peak_s': l.time_to_peak_s,
            'entry_v2': np.asarray(t.entry_v2, dtype=np.float32),
        })
    df = pd.DataFrame(rows)
    df['hour_utc'] = df['ts'].apply(lambda ts: datetime.utcfromtimestamp(int(ts)).hour)
    return df


def compare(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str,
                top_k: int = 15) -> pd.DataFrame:
    """Compare V2 feature distributions between two populations.

    Returns DataFrame ranked by |Cohen's d| (positive d means df_b has higher mean).
    """
    if len(df_a) < 30 or len(df_b) < 30:
        return pd.DataFrame()
    fa = np.stack(df_a['entry_v2'].values)
    fb = np.stack(df_b['entry_v2'].values)
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col_a = fa[:, j]
        col_b = fb[:, j]
        col_a = col_a[~np.isnan(col_a)]
        col_b = col_b[~np.isnan(col_b)]
        if len(col_a) < 30 or len(col_b) < 30:
            continue
        d = cohens_d(col_a, col_b)
        rows.append({
            'feature': name, 'd': d, 'abs_d': abs(d),
            f'mean_{label_a}': float(col_a.mean()),
            f'mean_{label_b}': float(col_b.mean()),
        })
    out = pd.DataFrame(rows).sort_values('abs_d', ascending=False).reset_index(drop=True)
    return out.head(top_k)


def feature_class(name: str) -> str:
    """Heuristic classification for grouping the top discriminators."""
    n = name.lower()
    if 'vol_mean' in n or 'vol_sigma' in n or 'vol_velocity' in n or 'vol_accel' in n:
        return 'volume'
    if 'price_sigma' in n or 'price_mean' in n:
        return 'price-stat'
    if 'price_velocity' in n or 'price_accel' in n:
        return 'price-momentum'
    if 'bar_range' in n or 'body' in n:
        return 'bar-shape'
    if 'swing_noise' in n:
        return 'chop'
    if 'z_se' in n or 'z_high' in n or 'z_low' in n or 'SE_high' in n or 'SE_low' in n:
        return 'band-pos'
    if 'hurst' in n or 'reversion_prob' in n:
        return 'regime-meta'
    if 'time_of_day' in n:
        return 'time'
    if 'vwap' in n:
        return 'vwap'
    return 'other'


def summarize_classes(top: pd.DataFrame, label_a: str, label_b: str) -> str:
    """Group top features by class to surface the dominant mechanism."""
    if top.empty:
        return '(no data)'
    top = top.copy()
    top['class'] = top['feature'].apply(feature_class)
    by_class = top.groupby('class').agg(
        n=('feature', 'size'),
        mean_d=('d', 'mean'),
    ).sort_values('n', ascending=False)
    out = []
    for cls, r in by_class.iterrows():
        sign = '+' if r['mean_d'] > 0 else '-'
        direction = f'higher in {label_b}' if r['mean_d'] > 0 else f'higher in {label_a}'
        out.append(f'{cls}: n={int(r["n"])}/{len(top)}, mean d={sign}{abs(r["mean_d"]):.2f}  ({direction})')
    return '; '.join(out)


def main():
    p = argparse.ArgumentParser(description='Forward pass analysis of OOS bleed zones')
    p.add_argument('--trades', default='training_iso_v2/output/nmp_regime_oos.pkl')
    p.add_argument('--regret', default='training_iso_v2/output/regret_nmp_regime_oos.pkl')
    p.add_argument('--out', default='reports/findings/v2_bleed_cause.md')
    p.add_argument('--top-k', type=int, default=15)
    args = p.parse_args()

    df = load_full(args.trades, args.regret)
    print(f'Loaded {len(df)} trades')
    print()

    # ── Bleed zone A: NY mid-session vs profitable hours ──────────────────
    print('=' * 80)
    print('A. WHY do hours 14-17 UTC bleed? Compare to profitable hours 12-13.')
    print('=' * 80)
    ny = df[df['hour_utc'].between(14, 17)]
    prof = df[df['hour_utc'].between(12, 13)]
    print(f'   NY mid-session (14-17): n={len(ny):>5}, $/trade ${ny["pnl"].mean():>+.2f}')
    print(f'   Profit zone   (12-13): n={len(prof):>5}, $/trade ${prof["pnl"].mean():>+.2f}')

    cmp_a = compare(ny, prof, 'NY', 'PROF', top_k=args.top_k)
    print(f'\n   Top {args.top_k} V2 features distinguishing NY mid-session FROM 12-13:')
    print(f'   (positive d = higher in 12-13 PROF; negative = higher in 14-17 NY)')
    print()
    print(f'   {"feature":<32} {"d":>7} {"mean_NY":>11} {"mean_PROF":>12} {"class":<14}')
    for _, r in cmp_a.iterrows():
        cls = feature_class(r['feature'])
        print(f'   {r["feature"]:<32} {r["d"]:>+7.3f} {r["mean_NY"]:>+10.2f} '
                  f'{r["mean_PROF"]:>+11.2f}  {cls}')
    print(f'\n   Class summary: {summarize_classes(cmp_a, "NY", "PROF")}')

    # ── Within-NY: winners vs losers — does the zone affect the trade equally? ──
    ny_win = ny[ny['pnl'] > 0]
    ny_los = ny[ny['pnl'] < 0]
    print(f'\n   Within hours 14-17: winners {len(ny_win)} (${ny_win["pnl"].mean():+.2f}/t), '
              f'losers {len(ny_los)} (${ny_los["pnl"].mean():+.2f}/t)')
    cmp_a2 = compare(ny_win, ny_los, 'WIN', 'LOSE', top_k=args.top_k)
    print(f'\n   Top {args.top_k} features that separate WIN vs LOSE WITHIN hours 14-17:')
    print(f'   (positive d = higher in losers)')
    print()
    print(f'   {"feature":<32} {"d":>7} {"mean_WIN":>11} {"mean_LOSE":>12} {"class":<14}')
    for _, r in cmp_a2.iterrows():
        cls = feature_class(r['feature'])
        print(f'   {r["feature"]:<32} {r["d"]:>+7.3f} {r["mean_WIN"]:>+10.2f} '
                  f'{r["mean_LOSE"]:>+11.2f}  {cls}')
    print(f'\n   Class summary: {summarize_classes(cmp_a2, "WIN", "LOSE")}')

    # ── Bleed zone B: FLAT_CHOPPY vs every other regime ──────────────────
    print('\n' + '=' * 80)
    print('B. WHY does FLAT_CHOPPY bleed? Compare to all other regimes.')
    print('=' * 80)
    fc_idx = REGIME_VOCAB.index('FLAT_CHOPPY')
    fc = df[df['regime_idx'] == fc_idx]
    other = df[df['regime_idx'] != fc_idx]
    print(f'   FLAT_CHOPPY     : n={len(fc):>5}, $/trade ${fc["pnl"].mean():>+.2f}')
    print(f'   Other regimes   : n={len(other):>5}, $/trade ${other["pnl"].mean():>+.2f}')

    # Within FLAT_CHOPPY: winners vs losers
    fc_win = fc[fc['pnl'] > 0]
    fc_los = fc[fc['pnl'] < 0]
    print(f'\n   Within FLAT_CHOPPY: winners {len(fc_win)} (${fc_win["pnl"].mean():+.2f}/t), '
              f'losers {len(fc_los)} (${fc_los["pnl"].mean():+.2f}/t)')
    cmp_b = compare(fc_win, fc_los, 'WIN', 'LOSE', top_k=args.top_k)
    print(f'\n   Top {args.top_k} features that separate WIN vs LOSE WITHIN FLAT_CHOPPY:')
    print()
    print(f'   {"feature":<32} {"d":>7} {"mean_WIN":>11} {"mean_LOSE":>12} {"class":<14}')
    for _, r in cmp_b.iterrows():
        cls = feature_class(r['feature'])
        print(f'   {r["feature"]:<32} {r["d"]:>+7.3f} {r["mean_WIN"]:>+10.2f} '
                  f'{r["mean_LOSE"]:>+11.2f}  {cls}')
    print(f'\n   Class summary: {summarize_classes(cmp_b, "WIN", "LOSE")}')

    # ── Bleed zone C: round-trip losers vs winners ───────────────────────
    print('\n' + '=' * 80)
    print('C. WHY do round-trips happen? Compare round-trip-losers vs straight winners.')
    print('=' * 80)
    losers = df[df['pnl'] < 0]
    winners = df[df['pnl'] > 0]
    rt_los = losers[losers['peak_pnl'] > 20]      # had real profit, gave it back
    straight_win = winners[winners['mae_pnl'] >= -10]  # never went deep red
    print(f'   Round-trip losers   : n={len(rt_los):>5}, '
              f'mean peak ${rt_los["peak_pnl"].mean():>+.1f}, realized ${rt_los["pnl"].mean():>+.2f}')
    print(f'   Straight winners    : n={len(straight_win):>5}, '
              f'mean peak ${straight_win["peak_pnl"].mean():>+.1f}, realized ${straight_win["pnl"].mean():>+.2f}')

    cmp_c = compare(straight_win, rt_los, 'STRAIGHT_WIN', 'ROUND_TRIP_LOSE', top_k=args.top_k)
    print(f'\n   Top {args.top_k} features distinguishing ROUND-TRIP LOSERS from STRAIGHT WINNERS:')
    print(f'   (positive d = higher in round-trip losers)')
    print()
    print(f'   {"feature":<32} {"d":>7} {"mean_WIN":>11} {"mean_RT":>11} {"class":<14}')
    for _, r in cmp_c.iterrows():
        cls = feature_class(r['feature'])
        print(f'   {r["feature"]:<32} {r["d"]:>+7.3f} '
                  f'{r["mean_STRAIGHT_WIN"]:>+10.2f} {r["mean_ROUND_TRIP_LOSE"]:>+10.2f}  {cls}')
    print(f'\n   Class summary: {summarize_classes(cmp_c, "STRAIGHT_WIN", "ROUND_TRIP_LOSE")}')

    # ── Quick volatility-grade lookup ────────────────────────────────────
    print('\n' + '=' * 80)
    print('VOL GRADE: compare key volatility features across hours and regimes')
    print('=' * 80)
    key_vol_features = [
        'L2_1m_vol_mean_15', 'L2_5m_vol_mean_9', 'L1_5m_bar_range',
        'L1_1m_bar_range', 'L3_1m_swing_noise_15', 'L2_5m_price_sigma_9',
    ]
    feat_idx = {n: FEATURE_NAMES.index(n) for n in key_vol_features
                     if n in FEATURE_NAMES}
    feats_arr = np.stack(df['entry_v2'].values)

    print(f'\n   By hour UTC:')
    print(f'   {"hr":>3} {"n":>5} {"mean_$":>8} ', end='')
    for n in feat_idx:
        print(f'{n[-12:]:>13} ', end='')
    print()
    for h in sorted(df['hour_utc'].unique()):
        m = (df['hour_utc'] == h).values
        if m.sum() < 30:
            continue
        print(f'   {int(h):>3} {int(m.sum()):>5} ${df.loc[m, "pnl"].mean():>+7.2f} ', end='')
        for j in feat_idx.values():
            v = np.nanmean(feats_arr[m, j])
            print(f'{v:>+13.2f} ', end='')
        print()

    print(f'\n   By regime:')
    print(f'   {"regime":<14} {"n":>5} {"mean_$":>8} ', end='')
    for n in feat_idx:
        print(f'{n[-12:]:>13} ', end='')
    print()
    for r in sorted(df['regime_idx'].unique()):
        m = (df['regime_idx'] == r).values
        if m.sum() < 30:
            continue
        rname = REGIME_VOCAB[int(r)]
        print(f'   {rname:<14} {int(m.sum()):>5} ${df.loc[m, "pnl"].mean():>+7.2f} ', end='')
        for j in feat_idx.values():
            v = np.nanmean(feats_arr[m, j])
            print(f'{v:>+13.2f} ', end='')
        print()


if __name__ == '__main__':
    main()
