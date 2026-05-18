"""Directional signal accuracy — does the predicted direction match the zigzag?

This is NOT a trading evaluation. It measures, bar-by-bar:
  - How often the predicted direction matches the zigzag leg_direction (truth)
  - Per-class precision/recall (LONG / SHORT)
  - Flip-lag in BARS at zigzag pivots — how many bars after a pivot before the
    predicted regime catches up
  - Accuracy as a function of confidence (P_dir - P_neutral)

We compare two signals against the same truth:
  RAW       = argmax(p_long, p_short, p_neutral) from trend3 raw cache
  SMOOTHED  = regime_dir from the DMI windowed-EMA state machine

Truth source: zigzag leg_direction per 1m bar (ATRx4 calibration).

Per user 2026-05-17: directional signal validation. NOT P&L. NOT trading.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def derive_leg_direction(truth_df: pd.DataFrame) -> pd.DataFrame:
    """Older OOS truth parquet only has is_pivot/pivot_dir on bars near each
    pivot (±60s zone). Reconstruct per-bar leg_direction by finding pivot
    centroids and assigning each bar the direction of the leg it sits in.

    Convention from the builder: pivot_dir is the direction of the LEG that
    STARTS at that pivot (so a UP pivot starts an up-leg → bars after = LONG).

    Adds columns: 'leg_direction' (LONG/SHORT/''), 'trend_class' (with
    NEUTRAL within ±120s of any pivot).
    """
    out_parts = []
    for day, g in truth_df.groupby('day'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        # Find pivot centroids: collapse consecutive is_pivot==1 runs to one row
        piv_rows = g[g['is_pivot'] == 1].copy()
        if len(piv_rows) == 0:
            g['leg_direction'] = ''
            g['trend_class'] = 'NEUTRAL'
            out_parts.append(g); continue
        # Group consecutive pivot bars (run-length) by checking ts gap > 90s
        piv_rows = piv_rows.sort_values('timestamp').reset_index(drop=True)
        groups = []
        cur = [0]
        for i in range(1, len(piv_rows)):
            if (piv_rows['timestamp'].iloc[i] - piv_rows['timestamp'].iloc[i-1]) > 90:
                groups.append(cur); cur = [i]
            else:
                cur.append(i)
        groups.append(cur)
        # Centroid = middle timestamp of each group, direction = majority pivot_dir
        pivots = []
        for grp in groups:
            sub = piv_rows.iloc[grp]
            ts_c = int(sub['timestamp'].median())
            dirs = sub['pivot_dir'].mode()
            d = dirs.iloc[0] if len(dirs) else ''
            pivots.append((ts_c, d))
        pivots.sort(key=lambda x: x[0])

        # Assign leg_direction: for each bar, find latest pivot at or before
        leg = ['' for _ in range(len(g))]
        if len(pivots) >= 1:
            p_ts = np.array([p[0] for p in pivots])
            p_d  = [p[1] for p in pivots]
            ts_arr = g['timestamp'].values
            idx = np.searchsorted(p_ts, ts_arr, side='right') - 1
            for i, k in enumerate(idx):
                if k >= 0 and k < len(pivots):
                    leg[i] = p_d[k]
        g['leg_direction'] = leg
        # trend_class — same but NEUTRAL within ±120s of any pivot
        trend = list(leg)
        for pts, _ in pivots:
            near = (g['timestamp'] >= pts - 120) & (g['timestamp'] <= pts + 120)
            for i in g.index[near]:
                trend[i] = 'NEUTRAL'
        g['trend_class'] = trend
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True)


def argmax3(pl, ps, pn) -> str:
    """Return 'LONG' / 'SHORT' / 'NEUTRAL' from 3 probabilities."""
    if pl >= ps and pl >= pn:
        return 'LONG'
    if ps >= pl and ps >= pn:
        return 'SHORT'
    return 'NEUTRAL'


def evaluate_signal(df, pred_col, truth_col):
    """df has aligned pred_col and truth_col (both string LONG/SHORT/NEUTRAL).
    Truth_col is the zigzag leg_direction (always LONG/SHORT, no NEUTRAL).
    Pred_col may be NEUTRAL.

    Returns dict of metrics.
    """
    n_total = len(df)
    pred = df[pred_col].values
    truth = df[truth_col].values
    valid = (truth == 'LONG') | (truth == 'SHORT')   # drop unlabeled bars
    pred = pred[valid]
    truth = truth[valid]
    n_eval = len(pred)
    if n_eval == 0:
        return None

    # Overall accuracy ignoring NEUTRAL predictions (signal abstains)
    n_signal = int(((pred == 'LONG') | (pred == 'SHORT')).sum())
    n_neutral_pred = int((pred == 'NEUTRAL').sum())
    n_correct = int(((pred == truth) & (pred != 'NEUTRAL')).sum())
    n_wrong = int(n_signal - n_correct)
    acc_on_signal = n_correct / max(n_signal, 1)

    # Per-class precision/recall
    tp_long  = int(((pred == 'LONG')  & (truth == 'LONG')).sum())
    fp_long  = int(((pred == 'LONG')  & (truth == 'SHORT')).sum())
    fn_long  = int(((pred != 'LONG')  & (truth == 'LONG')).sum())
    tp_short = int(((pred == 'SHORT') & (truth == 'SHORT')).sum())
    fp_short = int(((pred == 'SHORT') & (truth == 'LONG')).sum())
    fn_short = int(((pred != 'SHORT') & (truth == 'SHORT')).sum())

    prec_long  = tp_long  / max(tp_long + fp_long, 1)
    rec_long   = tp_long  / max(tp_long + fn_long, 1)
    prec_short = tp_short / max(tp_short + fp_short, 1)
    rec_short  = tp_short / max(tp_short + fn_short, 1)

    # Direction-correctness on bars where signal fires (excludes NEUTRAL)
    pl_fires = (pred == 'LONG')
    ps_fires = (pred == 'SHORT')

    return {
        'n_truth_bars': n_eval,
        'n_signal_bars': n_signal,
        'n_neutral_pred': n_neutral_pred,
        'coverage': n_signal / max(n_eval, 1),
        'acc_on_signal': acc_on_signal,
        'n_correct': n_correct,
        'n_wrong': n_wrong,
        'prec_long': prec_long, 'rec_long': rec_long,
        'prec_short': prec_short, 'rec_short': rec_short,
        'tp_long': tp_long, 'fp_long': fp_long, 'fn_long': fn_long,
        'tp_short': tp_short, 'fp_short': fp_short, 'fn_short': fn_short,
    }


def flip_lag_at_pivots(df, pred_col, day_col='day'):
    """For each zigzag pivot (leg direction changes), measure how many bars
    until the predicted direction matches the new leg direction.

    Truth flips are detected as leg_direction transitions. Lag is counted in
    1m bars post-flip. NEUTRAL predictions are skipped (we wait for the next
    directional prediction).

    Returns a Series of lag values (one per truth flip).
    """
    lags = []
    for day, g in df.groupby(day_col):
        g = g.sort_values('timestamp').reset_index(drop=True)
        truth = g['leg_direction'].values
        pred = g[pred_col].values
        # Find truth flips
        for i in range(1, len(truth)):
            if truth[i] != truth[i-1] and truth[i] in ('LONG', 'SHORT'):
                new_dir = truth[i]
                # Walk forward, count bars until pred == new_dir
                lag = None
                for j in range(i, len(truth)):
                    if pred[j] == new_dir:
                        lag = j - i
                        break
                    if truth[j] != new_dir:   # truth flipped again before we caught up
                        break
                if lag is not None:
                    lags.append(lag)
    return np.array(lags)


def accuracy_by_strength(df, pred_col, p_long_col, p_short_col, p_neut_col,
                          truth_col):
    """Bucket bars by directional strength (P_dir - P_neutral) and report
    accuracy on bars where pred is directional (not NEUTRAL)."""
    strength = np.where(
        df[pred_col] == 'LONG',
        df[p_long_col].values - df[p_neut_col].values,
        np.where(df[pred_col] == 'SHORT',
                  df[p_short_col].values - df[p_neut_col].values,
                  np.nan),
    )
    df = df.assign(_strength=strength)
    df = df[df[pred_col].isin(['LONG', 'SHORT'])]
    df = df[df[truth_col].isin(['LONG', 'SHORT'])]
    if len(df) == 0:
        return pd.DataFrame()
    df = df.assign(_correct=(df[pred_col] == df[truth_col]).astype(int))
    bins = [-1.0, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.01]
    labels = [f'{bins[i]:.2f}-{bins[i+1]:.2f}' for i in range(len(bins)-1)]
    df['_bin'] = pd.cut(df['_strength'], bins=bins, labels=labels,
                        include_lowest=True)
    agg = df.groupby('_bin', observed=True).agg(
        n=('_correct', 'size'),
        acc=('_correct', 'mean'),
    ).reset_index()
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-cache', default='reports/findings/regret_oracle/trend3_cache_OOS_NT8.parquet')
    ap.add_argument('--smoothed-cache', default='reports/findings/regret_oracle/trend3_smoothed_OOS_NT8.parquet')
    ap.add_argument('--truth', default='reports/findings/regret_oracle/zigzag_pivot_dataset_OOS_atr4.parquet')
    ap.add_argument('--out', default='reports/findings/regret_oracle/direction_signal_accuracy.txt')
    args = ap.parse_args()

    print(f'Loading truth: {args.truth}')
    truth_df = pd.read_parquet(args.truth)
    print(f'  {len(truth_df)} 1m bars / {truth_df["day"].nunique()} days')
    if 'leg_direction' not in truth_df.columns:
        print('  truth file predates leg_direction column — deriving from is_pivot/pivot_dir')
        truth_df = derive_leg_direction(truth_df)
        n_long = int((truth_df['leg_direction'] == 'LONG').sum())
        n_short = int((truth_df['leg_direction'] == 'SHORT').sum())
        n_blank = int((truth_df['leg_direction'] == '').sum())
        print(f'  derived: LONG={n_long:,}  SHORT={n_short:,}  unlabeled={n_blank:,}')

    print(f'\nLoading RAW cache: {args.raw_cache}')
    raw_df = pd.read_parquet(args.raw_cache)
    print(f'  {len(raw_df)} rows / cols: {list(raw_df.columns)}')

    print(f'\nLoading SMOOTHED cache: {args.smoothed_cache}')
    sm_df = pd.read_parquet(args.smoothed_cache)
    print(f'  {len(sm_df)} rows / cols: {list(sm_df.columns)}')

    # Derive RAW predicted direction (argmax)
    raw_df['pred_raw'] = [argmax3(l, s, n) for l, s, n
                         in zip(raw_df['p_long'], raw_df['p_short'],
                                raw_df['p_neutral'])]

    # Merge on (day, timestamp)
    merged = truth_df[['day', 'timestamp', 'leg_direction', 'trend_class']].merge(
        raw_df[['day', 'timestamp', 'p_long', 'p_short', 'p_neutral', 'pred_raw']],
        on=['day', 'timestamp'], how='inner'
    ).merge(
        sm_df[['day', 'timestamp', 'p_long_ema', 'p_short_ema',
                'adx', 'regime_dir', 'regime_change']].rename(
            columns={'regime_dir': 'pred_smoothed'}
        ),
        on=['day', 'timestamp'], how='inner'
    )
    print(f'\nMerged: {len(merged)} bars over {merged["day"].nunique()} days')

    # Restrict to bars where truth is LONG or SHORT (in a leg, not gaps)
    merged = merged[merged['leg_direction'].isin(['LONG', 'SHORT'])].copy()
    print(f'  with valid truth leg_direction: {len(merged)} bars')

    # === Headline accuracy ===
    raw_metrics = evaluate_signal(merged, 'pred_raw', 'leg_direction')
    sm_metrics  = evaluate_signal(merged, 'pred_smoothed', 'leg_direction')

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('\n' + '='*72)
    out('DIRECTIONAL SIGNAL ACCURACY (truth = zigzag leg_direction @ ATRx4)')
    out('='*72)
    out(f'Bars evaluated:         {raw_metrics["n_truth_bars"]:,}')
    out(f'Days:                   {merged["day"].nunique()}')

    def block(name, m):
        out(f'\n--- {name} ---')
        out(f'  Coverage (non-NEUTRAL):   {m["coverage"]*100:.1f}%   '
            f'({m["n_signal_bars"]:,} / {m["n_truth_bars"]:,})')
        out(f'  Accuracy ON SIGNAL bars:  {m["acc_on_signal"]*100:.2f}%   '
            f'({m["n_correct"]:,} correct / {m["n_wrong"]:,} wrong)')
        out(f'  LONG :  prec={m["prec_long"]*100:.1f}%  rec={m["rec_long"]*100:.1f}%   '
            f'tp={m["tp_long"]} fp={m["fp_long"]} fn={m["fn_long"]}')
        out(f'  SHORT:  prec={m["prec_short"]*100:.1f}%  rec={m["rec_short"]*100:.1f}%   '
            f'tp={m["tp_short"]} fp={m["fp_short"]} fn={m["fn_short"]}')

    block('RAW (argmax of p_long/p_short/p_neutral)', raw_metrics)
    block('SMOOTHED (DMI windowed-EMA + state machine, regime_dir)', sm_metrics)

    # === Flip lag at zigzag pivots ===
    out('\n' + '='*72)
    out('FLIP LAG AT ZIGZAG PIVOTS (bars from truth-flip -> predicted catch-up)')
    out('='*72)
    raw_lags = flip_lag_at_pivots(merged, 'pred_raw')
    sm_lags  = flip_lag_at_pivots(merged, 'pred_smoothed')

    def lag_block(name, lags):
        if len(lags) == 0:
            out(f'  {name}: NO catch-up events'); return
        out(f'  {name}: n={len(lags):,}  '
            f'median={np.median(lags):.1f}  '
            f'mean={lags.mean():.1f}  '
            f'p25={np.percentile(lags,25):.1f}  '
            f'p75={np.percentile(lags,75):.1f}  '
            f'p90={np.percentile(lags,90):.1f}')

    lag_block('RAW     ', raw_lags)
    lag_block('SMOOTHED', sm_lags)

    # === Accuracy by directional strength ===
    out('\n' + '='*72)
    out('ACCURACY vs DIRECTIONAL STRENGTH (P_pred - P_neutral)')
    out('='*72)
    out('RAW:')
    raw_strength = accuracy_by_strength(
        merged, 'pred_raw', 'p_long', 'p_short', 'p_neutral', 'leg_direction'
    )
    for _, r in raw_strength.iterrows():
        out(f'  strength {r["_bin"]}:  n={int(r["n"]):>6,}  acc={r["acc"]*100:.2f}%')

    out('SMOOTHED:')
    sm_strength = accuracy_by_strength(
        merged, 'pred_smoothed', 'p_long_ema', 'p_short_ema',
        # No p_neutral_ema column; use raw p_neutral as proxy for "abstain mass"
        'p_neutral', 'leg_direction'
    )
    for _, r in sm_strength.iterrows():
        out(f'  strength {r["_bin"]}:  n={int(r["n"]):>6,}  acc={r["acc"]*100:.2f}%')

    # === Per-day accuracy ===
    out('\n' + '='*72)
    out('PER-DAY ACCURACY (signal bars only)')
    out('='*72)
    per_day = []
    for day, g in merged.groupby('day'):
        g_sig_raw = g[g['pred_raw'].isin(['LONG', 'SHORT'])]
        g_sig_sm  = g[g['pred_smoothed'].isin(['LONG', 'SHORT'])]
        per_day.append({
            'day': day,
            'n_bars': len(g),
            'raw_cov': len(g_sig_raw) / max(len(g), 1),
            'raw_acc': (g_sig_raw['pred_raw'] == g_sig_raw['leg_direction']).mean()
                if len(g_sig_raw) else np.nan,
            'sm_cov':  len(g_sig_sm) / max(len(g), 1),
            'sm_acc':  (g_sig_sm['pred_smoothed'] == g_sig_sm['leg_direction']).mean()
                if len(g_sig_sm) else np.nan,
        })
    per_day_df = pd.DataFrame(per_day)
    out(f'  Days: {len(per_day_df)}')
    out(f'  RAW       acc mean={per_day_df["raw_acc"].mean()*100:.2f}%  '
        f'median={per_day_df["raw_acc"].median()*100:.2f}%')
    out(f'  SMOOTHED  acc mean={per_day_df["sm_acc"].mean()*100:.2f}%  '
        f'median={per_day_df["sm_acc"].median()*100:.2f}%')
    out(f'  RAW       cov mean={per_day_df["raw_cov"].mean()*100:.1f}%')
    out(f'  SMOOTHED  cov mean={per_day_df["sm_cov"].mean()*100:.1f}%')
    n_sm_better = int((per_day_df['sm_acc'] > per_day_df['raw_acc']).sum())
    n_eq        = int((per_day_df['sm_acc'] == per_day_df['raw_acc']).sum())
    out(f'  Smoothed beats raw on {n_sm_better}/{len(per_day_df)} days '
        f'({n_eq} ties)')

    # === Save ===
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    csv_path = out_path.with_suffix('.per_day.csv')
    per_day_df.to_csv(csv_path, index=False)
    print(f'\nWrote: {out_path}')
    print(f'Wrote: {csv_path}')


if __name__ == '__main__':
    main()
