"""Forward-pass simulator on 2026 OOS using direction-confidence selector.

For each OOS daisy trade:
  - Predict P(LONG) from V2 entry features via the LR direction classifier
    (this is the cleanest signal we have — AUC 0.864 on IS test).
  - Selector dial: take trade if |P - 0.5| > (T - 0.5), i.e., P > T or P < 1-T
  - Trade outcome depends on framing:

Framing A — ORACLE-EXIT (upper bound, perfect exit at MFE):
    correct_dir:   P/L = +mfe_dollars
    wrong_dir:     P/L = -mfe_dollars   (mirror approximation — wrong-side
                                          MFE-magnitude movement against us)
    skip:          P/L = 0

Framing B — FIXED R/R (realistic, TP=SL=X ticks via MFE/MAE simulation):
    correct_dir, mfe_ticks >= X:   P/L = +X * $0.50  (TP hit)
    correct_dir, mfe_ticks < X:    P/L = +mfe_dollars (no SL hit per oracle
                                                       def, exit at MFE)
    wrong_dir, mfe_ticks >= X:     P/L = -X * $0.50  (SL hit — opposite move)
    wrong_dir, mfe_ticks < X:      P/L = -mfe_dollars (mirror, MFE was less
                                                       than X — small loss)
    skip:                          P/L = 0

Both framings report: $/day mode, mean + 95% bootstrap CI, Day WR (count),
Trade WR (PF-based), n_trades, n_days at each threshold.

Bootstrap: 4000 percentile resamples per CLAUDE.md.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


TICK_VALUE = 0.50  # MNQ


def bootstrap_ci(arr: np.ndarray, stat=np.mean, n_boot: int = 4000,
                 ci: float = 0.95, rng: np.random.Generator = None) -> tuple:
    if rng is None:
        rng = np.random.default_rng(42)
    arr = np.asarray(arr)
    if len(arr) < 2:
        return float('nan'), float('nan'), float('nan')
    boots = np.empty(n_boot)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, n)]
        boots[i] = stat(sample)
    lo = np.percentile(boots, 2.5)
    hi = np.percentile(boots, 97.5)
    return float(stat(arr)), float(lo), float(hi)


def mode_dollars(arr: np.ndarray, bin_width: float = 25.0) -> float:
    """Mode via histogram (bin width default $25 per CLAUDE.md)."""
    if len(arr) == 0:
        return float('nan')
    edges = np.arange(arr.min() - bin_width, arr.max() + 2*bin_width, bin_width)
    if len(edges) < 2:
        return float(arr.mean())
    hist, _ = np.histogram(arr, bins=edges)
    idx = np.argmax(hist)
    return float((edges[idx] + edges[idx+1]) / 2)


def pf_wr(pnl: np.ndarray) -> float:
    """PF-based Trade WR per CLAUDE.md: (sum_winners / |sum_losers|) - 1.
    0 = break-even, +1 = winners 2x loser size."""
    wins = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses < 1e-6:
        return float('inf') if wins > 0 else 0.0
    return float(wins / losses - 1)


def simulate(p_long: np.ndarray, y_dir: np.ndarray, mfe_dollars: np.ndarray,
             mfe_ticks: np.ndarray, dates: np.ndarray, threshold: float,
             framing: str = 'oracle', rr_ticks: int = 8) -> dict:
    """Run forward pass at a given confidence threshold.

    p_long: P(LONG) from classifier
    y_dir: actual oracle direction (1=LONG, 0=SHORT)
    mfe_dollars: signed-positive MFE (always positive)
    mfe_ticks: MFE in ticks
    dates: session_date per trade (for daily aggregation)
    threshold: confidence threshold; take trade if max(P, 1-P) > threshold
    framing: 'oracle' or 'rr'
    rr_ticks: fixed TP=SL distance in ticks (for framing='rr')
    """
    pred_long = p_long > 0.5
    confidence = np.maximum(p_long, 1 - p_long)
    fire_mask = confidence > threshold

    # Direction correctness for fired trades
    pred_dir = pred_long.astype(np.int8)
    correct = (pred_dir == y_dir)

    pnl = np.zeros_like(mfe_dollars)
    if framing == 'oracle':
        pnl[fire_mask & correct] = mfe_dollars[fire_mask & correct]
        pnl[fire_mask & ~correct] = -mfe_dollars[fire_mask & ~correct]
    elif framing == 'rr':
        rr_dollars = rr_ticks * TICK_VALUE
        # Correct dir:
        cd = fire_mask & correct
        # TP hit if mfe_ticks >= rr_ticks
        cd_tp = cd & (mfe_ticks >= rr_ticks)
        cd_notp = cd & ~(mfe_ticks >= rr_ticks)
        pnl[cd_tp] = rr_dollars
        pnl[cd_notp] = mfe_dollars[cd_notp]  # smaller win, exit at MFE
        # Wrong dir:
        wd = fire_mask & ~correct
        wd_sl = wd & (mfe_ticks >= rr_ticks)
        wd_nosl = wd & ~(mfe_ticks >= rr_ticks)
        pnl[wd_sl] = -rr_dollars
        pnl[wd_nosl] = -mfe_dollars[wd_nosl]  # smaller loss
    else:
        raise ValueError(f'Unknown framing {framing}')

    # Per-day aggregation
    df = pd.DataFrame({'date': dates, 'pnl': pnl, 'fired': fire_mask})
    daily = df.groupby('date').agg(
        pnl_total=('pnl', 'sum'),
        n_fired=('fired', 'sum'),
        n_correct_fires=('pnl', lambda x: (x > 0).sum()),
        n_wrong_fires=('pnl', lambda x: (x < 0).sum()),
    ).reset_index()
    # Days with at least one fire
    active = daily[daily['n_fired'] > 0]

    if len(active) == 0:
        return {
            'threshold': threshold, 'framing': framing,
            'rr_ticks': rr_ticks if framing == 'rr' else None,
            'coverage': 0.0,
            'n_fired': 0, 'n_active_days': 0, 'n_total_days': int(daily.shape[0]),
            'mean_pnl_day': 0.0, 'pnl_day_ci_lo': 0.0, 'pnl_day_ci_hi': 0.0,
            'mode_pnl_day': 0.0, 'median_pnl_day': 0.0,
            'day_wr_active': 0.0, 'day_wr_total': 0.0,
            'mean_trade_pnl': 0.0, 'trade_wr_pf': 0.0,
            'dir_acc_fired': 0.0,
        }

    mean_d, lo_d, hi_d = bootstrap_ci(active['pnl_total'].values)
    mode_d = mode_dollars(active['pnl_total'].values, bin_width=25.0)
    med_d = float(np.median(active['pnl_total'].values))
    day_wr_active = float((active['pnl_total'] > 0).mean())
    day_wr_total = float((daily['pnl_total'] > 0).sum() / daily.shape[0])

    fired_pnl = pnl[fire_mask]
    return {
        'threshold': float(threshold), 'framing': framing,
        'rr_ticks': rr_ticks if framing == 'rr' else None,
        'coverage': float(fire_mask.mean()),
        'n_fired': int(fire_mask.sum()),
        'n_active_days': int(len(active)),
        'n_total_days': int(daily.shape[0]),
        'mean_pnl_day': mean_d,
        'pnl_day_ci_lo': lo_d, 'pnl_day_ci_hi': hi_d,
        'mode_pnl_day': mode_d, 'median_pnl_day': med_d,
        'day_wr_active': day_wr_active,
        'day_wr_total': day_wr_total,
        'mean_trade_pnl': float(fired_pnl.mean()) if len(fired_pnl) else 0.0,
        'mode_trade_pnl': mode_dollars(fired_pnl, bin_width=2.0),
        'trade_wr_pf': pf_wr(fired_pnl),
        'dir_acc_fired': float(correct[fire_mask].mean()) if fire_mask.any() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--is-input', required=True)
    ap.add_argument('--oos-input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--p-long-npz', default=None,
                    help='If provided, use this p_long from npz instead of '
                         'fitting LR (must have oracle_idx + p_long keys)')
    ap.add_argument('--label', default='LR',
                    help='Model label for output (LR, LSTM, etc.)')
    ap.add_argument('--thresholds', nargs='+', type=float,
                    default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    ap.add_argument('--rr-ticks', nargs='+', type=int, default=[4, 8, 16])
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.parent.mkdir(parents=True, exist_ok=True)

    # Train direction classifier on IS V2 entry features
    print(f'Loading IS: {args.is_input}')
    is_df = pd.read_parquet(args.is_input)
    v2_cols = [c for c in is_df.columns if c.startswith(('L1_','L2_','L3_'))]
    X_is = is_df[v2_cols].fillna(0).values.astype(np.float32)
    y_is = (is_df['direction'].values == 'LONG').astype(np.int8)
    print(f'  IS: {len(is_df)} trades x {len(v2_cols)} features, LONG rate {y_is.mean():.3f}')

    scaler = StandardScaler()
    X_is_s = scaler.fit_transform(X_is)
    clf = LogisticRegression(max_iter=400, C=1.0, solver='lbfgs')
    clf.fit(X_is_s, y_is)
    print(f'  Train acc: {clf.score(X_is_s, y_is):.4f}')

    # Predict on OOS
    print(f'\nLoading OOS: {args.oos_input}')
    oos_df = pd.read_parquet(args.oos_input)
    # Align column order
    X_oos = oos_df[v2_cols].fillna(0).values.astype(np.float32)
    y_oos = (oos_df['direction'].values == 'LONG').astype(np.int8)
    X_oos_s = scaler.transform(X_oos)

    if args.p_long_npz:
        print(f'  Loading P(LONG) from {args.p_long_npz} ({args.label})')
        pz = np.load(args.p_long_npz)
        # Align by oracle_idx
        pz_idx = pz['oracle_idx']
        pz_pl = pz['p_long']
        oos_oid = oos_df['oracle_idx'].values
        oidx_lookup = {int(o): i for i, o in enumerate(pz_idx)}
        p_long = np.array([pz_pl[oidx_lookup[int(o)]] for o in oos_oid],
                          dtype=np.float64)
    else:
        p_long = clf.predict_proba(X_oos_s)[:, 1]
    mfe_dollars = oos_df['mfe_dollars'].values.astype(np.float64)
    mfe_ticks = oos_df['mfe_ticks'].values.astype(np.float64)
    dates = oos_df['session_date'].values
    print(f'  OOS: {len(oos_df)} trades across {len(np.unique(dates))} days')
    print(f'  OOS dir acc at threshold 0.5: '
          f'{((p_long > 0.5).astype(np.int8) == y_oos).mean():.4f}')

    # Threshold sweep across framings
    results = []
    print(f'\n=== Framing A: oracle-exit ===')
    for T in args.thresholds:
        r = simulate(p_long, y_oos, mfe_dollars, mfe_ticks, dates,
                     threshold=T, framing='oracle')
        results.append(r)
        print(f'  T={T:.2f}  fires={r["n_fired"]:4d} cov={r["coverage"]:.2%}  '
              f'dir_acc={r["dir_acc_fired"]:.3f}  '
              f'$/day mean=${r["mean_pnl_day"]:7.1f} [{r["pnl_day_ci_lo"]:.0f}, {r["pnl_day_ci_hi"]:.0f}]  '
              f'mode=${r["mode_pnl_day"]:6.0f}  '
              f'DayWR(active)={r["day_wr_active"]:.3f}  '
              f'TradeWR_PF={r["trade_wr_pf"]:+.3f}')

    for rr_x in args.rr_ticks:
        print(f'\n=== Framing B: fixed R/R = {rr_x} ticks (${rr_x*TICK_VALUE:.0f}) ===')
        for T in args.thresholds:
            r = simulate(p_long, y_oos, mfe_dollars, mfe_ticks, dates,
                         threshold=T, framing='rr', rr_ticks=rr_x)
            results.append(r)
            print(f'  T={T:.2f}  fires={r["n_fired"]:4d} cov={r["coverage"]:.2%}  '
                  f'dir_acc={r["dir_acc_fired"]:.3f}  '
                  f'$/day mean=${r["mean_pnl_day"]:7.1f} [{r["pnl_day_ci_lo"]:.0f}, {r["pnl_day_ci_hi"]:.0f}]  '
                  f'mode=${r["mode_pnl_day"]:6.0f}  '
                  f'DayWR(active)={r["day_wr_active"]:.3f}  '
                  f'TradeWR_PF={r["trade_wr_pf"]:+.3f}')

    res_df = pd.DataFrame(results)
    res_df.to_csv(args.out, index=False)
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
