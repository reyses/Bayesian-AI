"""
v2_features_persistence_eda.py — Within-day persistence (Step #2).

Once a feature reaches Q4 (top quintile within its regime), how long does
the signal persist? Two views:

(A) Feature-state autocorrelation
    P(Q4 at t+k | Q4 at t) for k = 1..20 base bars.
    Distinguishes "event-like" features (Q4 at one bar, gone next bar) from
    "persistent state" features (Q4 sustained for many bars).

(B) Forward-return persistence by run length
    For each Q4-run of length L (consecutive Q4 bars), bucket the bars
    by their position in the run (bar_1, bar_2, ..., bar_L). Compare
    mean_fwd at bar_1 vs bar_>=5. Tells us whether the edge is FRONT-
    LOADED (entry-bar advantage that decays) or DISTRIBUTED (held
    throughout).

Both views are stratified by regime — a feature can be event-like in
SMOOTH but persistent in CHOPPY.

Outputs:
  reports/findings/v2_features_persistence/
    autocorr.csv          (concept, tf, regime, lag_k, p_q4_given_q4)
    runlen_dist.csv       (concept, tf, regime, run_len, count)
    runpos_summary.csv    (concept, tf, regime, run_pos_bucket, n, mean_fwd)
    half_life.csv         per (concept, tf, regime): k where autocorr drops to 0.5
    summary.md
    plot_autocorr_<concept>.png  one panel per regime
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tools.research.data import load_atlas_tf
from tools.research.features_v2 import (
    load_v2_features, align_v2_to_base_tf,
)
from tools.atlas_regime_labeler_2d import (
    load_regime_labels, REGIME_2D_ORDER,
)
from tools.v2_features_tf_sweep_eda import feature_column_for


DEFAULT_CONCEPTS = [
    'price_sigma_w', 'bar_range', 'vol_mean_w', 'vol_velocity_w',
    'price_velocity_w', 'body', 'price_velocity_1b', 'swing_noise_w',
]
DEFAULT_TFS = ['5s', '1m', '5m', '15m', '1h']

RUN_POS_BUCKETS = [
    ('bar_1',     1, 1),
    ('bar_2',     2, 2),
    ('bar_3_4',   3, 4),
    ('bar_5_8',   5, 8),
    ('bar_9_16',  9, 16),
    ('bar_17p',   17, 9999),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--cache', default='DATA/ATLAS/FEATURES_5s_v2')
    parser.add_argument('--base-tf', default='5m')
    parser.add_argument('--labels-csv', default='DATA/ATLAS/regime_labels_2d.csv')
    parser.add_argument('--concepts', nargs='+', default=DEFAULT_CONCEPTS)
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--forward-n', type=int, default=12)
    parser.add_argument('--split', default='IS')
    parser.add_argument('--quantile', type=int, default=4)
    parser.add_argument('--quantiles', type=int, default=5)
    parser.add_argument('--max-lag', type=int, default=20)
    parser.add_argument('--min-cell-n', type=int, default=30)
    parser.add_argument('--top-plots', type=int, default=10)
    parser.add_argument('--output-dir',
                        default='reports/findings/v2_features_persistence')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 features within-day persistence (Step #2)")
    print(f"  Concepts: {args.concepts}")
    print(f"  TFs: {args.tfs}")
    print(f"  Tracking quantile Q{args.quantile} of {args.quantiles}")
    print(f"  Max lag: {args.max_lag} bars")
    print(f"{'='*70}")

    # ---- load + merge ----
    print(f"\n--- Loading data ---")
    base_df = load_atlas_tf(args.data, args.base_tf)
    if pd.api.types.is_datetime64_any_dtype(base_df['timestamp']):
        ts_int = base_df['timestamp'].astype('int64') // 10**9
    else:
        ts_int = base_df['timestamp'].astype(np.int64)
    base_df = base_df.copy()
    base_df['ts_int'] = ts_int
    dt_la = pd.to_datetime(ts_int, unit='s', utc=True).dt.tz_convert('America/Los_Angeles')
    base_df['date'] = dt_la.dt.date.astype(str)

    labels_df = load_regime_labels(args.labels_csv).copy()
    labels_df['date'] = labels_df['date'].astype(str).str[:10]
    merged = base_df.merge(
        labels_df[['date', 'regime_2d', 'split']], on='date', how='inner')
    if args.split.upper() != 'ALL':
        merged = merged[merged['split'] == args.split.upper()].reset_index(drop=True)
    print(f"  After split={args.split}: {len(merged):,} bars")

    ts_int = merged['ts_int'].values.astype(np.int64)
    features_5s = load_v2_features(
        v2_dir=args.cache, atlas_root=args.data, day_strs=None,
        ts_range=(int(ts_int.min()), int(ts_int.max())), verbose=False,
    )
    aligned = align_v2_to_base_tf(features_5s, ts_int)
    full = pd.concat([merged.reset_index(drop=True),
                       aligned.reset_index(drop=True)], axis=1)

    close = full['close'].values.astype(np.float64)
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > args.forward_n:
        fwd[:-args.forward_n] = close[args.forward_n:] - close[:-args.forward_n]

    regimes = full['regime_2d'].values.astype(str)
    dates = full['date'].values.astype(str)

    print(f"\n--- Sweeping {len(args.concepts)} concepts x {len(args.tfs)} TFs "
          f"x {len(REGIME_2D_ORDER)} regimes ---")

    autocorr_rows = []
    runpos_rows = []
    runlen_rows = []
    half_life_rows = []
    autocorr_cache = {}  # (concept, tf, regime) -> [p_k for k in 1..max_lag]

    for concept in args.concepts:
        for tf in args.tfs:
            col = feature_column_for(concept, tf)
            if col not in full.columns:
                continue
            v = full[col].values.astype(np.float64)
            for regime in REGIME_2D_ORDER:
                regime_mask = (regimes == regime)
                if regime_mask.sum() < 200:
                    continue
                v_r = v[regime_mask]
                valid = ~np.isnan(v_r)
                if valid.sum() < args.quantiles * 5:
                    continue
                qs = np.quantile(v_r[valid], np.linspace(0, 1, args.quantiles + 1))
                qs[0] -= 1e-9
                qs[-1] += 1e-9
                bin_idx_full = np.digitize(v, qs[1:-1])

                # bool flag: bar is in regime AND in target Q
                is_target = regime_mask & (bin_idx_full == args.quantile) & ~np.isnan(v)
                base_count = int(is_target.sum())
                if base_count < args.min_cell_n * 5:
                    continue

                # ---------- (A) autocorrelation ----------
                # P(Q4 at t+k | Q4 at t), but only counting (t, t+k) pairs
                # that share the same date (no cross-day contamination)
                p_at_lag = []
                for k in range(1, args.max_lag + 1):
                    # shift by k
                    if k >= n:
                        p_at_lag.append(float('nan'))
                        continue
                    a = is_target[:-k]
                    b = is_target[k:]
                    same_date = (dates[:-k] == dates[k:])
                    cond = a & same_date
                    if cond.sum() < args.min_cell_n:
                        p_at_lag.append(float('nan'))
                        continue
                    p = float(b[cond].mean())
                    p_at_lag.append(p)
                    autocorr_rows.append({
                        'concept': concept,
                        'tf': tf,
                        'regime_2d': regime,
                        'lag_k': k,
                        'n_pairs': int(cond.sum()),
                        'p_q4_given_q4': p,
                    })
                autocorr_cache[(concept, tf, regime)] = p_at_lag

                # half-life: smallest k where p drops to <= 0.5 (or, if it
                # never gets that low, take the value at max_lag and report
                # asymptote)
                p_arr = np.array(p_at_lag)
                if np.all(np.isnan(p_arr)):
                    continue
                # find first k <= 0.5
                hk = None
                for k_idx, p_val in enumerate(p_at_lag):
                    if not np.isnan(p_val) and p_val <= 0.5:
                        hk = k_idx + 1
                        break
                if hk is None:
                    last_valid = [p for p in p_at_lag if not np.isnan(p)]
                    asymptote = last_valid[-1] if last_valid else float('nan')
                    hk_str = f'>{args.max_lag}'
                else:
                    asymptote = p_at_lag[-1]
                half_life_rows.append({
                    'concept': concept,
                    'tf': tf,
                    'regime_2d': regime,
                    'p_lag_1': p_at_lag[0],
                    'p_lag_5': p_at_lag[4] if len(p_at_lag) >= 5 else float('nan'),
                    'p_lag_10': p_at_lag[9] if len(p_at_lag) >= 10 else float('nan'),
                    'half_life_k': hk if hk is not None else args.max_lag + 1,
                    'half_life_str': str(hk) if hk is not None else f'>{args.max_lag}',
                    'asymptote': asymptote,
                })

                # ---------- (B) run-position analysis ----------
                # for each consecutive run of is_target=True within the same
                # date, mark each bar's position in the run (1-indexed)
                run_pos = np.zeros(n, dtype=np.int32)
                cur = 0
                for i in range(n):
                    if not is_target[i]:
                        cur = 0
                        continue
                    if i > 0 and dates[i] != dates[i - 1]:
                        cur = 0
                    cur += 1
                    run_pos[i] = cur

                # run-length distribution (count of runs of each length)
                # a run ends when run_pos[i+1] is not run_pos[i]+1
                if base_count > 0:
                    run_lengths = []
                    in_run = 0
                    for i in range(n):
                        if run_pos[i] > 0:
                            in_run = run_pos[i]
                        else:
                            if in_run > 0:
                                run_lengths.append(in_run)
                                in_run = 0
                    if in_run > 0:
                        run_lengths.append(in_run)
                    rl = pd.Series(run_lengths).value_counts().sort_index()
                    for length, cnt in rl.items():
                        runlen_rows.append({
                            'concept': concept,
                            'tf': tf,
                            'regime_2d': regime,
                            'run_len': int(length),
                            'count': int(cnt),
                        })

                # bucket each Q4 bar by its run position; compute mean_fwd
                for name, lo, hi in RUN_POS_BUCKETS:
                    mask = (run_pos >= lo) & (run_pos <= hi) & ~np.isnan(fwd)
                    n_cell = int(mask.sum())
                    if n_cell < args.min_cell_n:
                        continue
                    f = fwd[mask]
                    runpos_rows.append({
                        'concept': concept,
                        'tf': tf,
                        'regime_2d': regime,
                        'run_pos': name,
                        'lo': lo,
                        'hi': hi,
                        'n': n_cell,
                        'mean_fwd': float(f.mean()),
                        'win_rate': float((f > 0).mean()),
                    })

    # ---- save ----
    pd.DataFrame(autocorr_rows).to_csv(
        os.path.join(args.output_dir, 'autocorr.csv'), index=False)
    print(f"  [saved] autocorr.csv ({len(autocorr_rows)} cells)")

    runpos_df = pd.DataFrame(runpos_rows)
    runpos_df.to_csv(os.path.join(args.output_dir, 'runpos_summary.csv'),
                       index=False)
    print(f"  [saved] runpos_summary.csv ({len(runpos_df)} cells)")

    pd.DataFrame(runlen_rows).to_csv(
        os.path.join(args.output_dir, 'runlen_dist.csv'), index=False)
    print(f"  [saved] runlen_dist.csv ({len(runlen_rows)} entries)")

    hl_df = pd.DataFrame(half_life_rows).sort_values(
        ['half_life_k', 'p_lag_1'], ascending=[False, False])
    hl_df.to_csv(os.path.join(args.output_dir, 'half_life.csv'), index=False)
    print(f"  [saved] half_life.csv ({len(hl_df)} entries)")

    # ---- print summary ----
    print(f"\n  Top 20 most-PERSISTENT (concept, tf, regime) by half_life_k:")
    print(f"    {'concept':>22}  {'tf':>4}  {'regime':>14}  "
          f"{'p_lag_1':>8} {'p_lag_5':>8} {'p_lag_10':>8}  {'half':>5}  "
          f"{'asymp':>6}")
    for _, r in hl_df.head(20).iterrows():
        print(f"    {r['concept']:>22}  {r['tf']:>4}  {r['regime_2d']:>14}  "
              f"{r['p_lag_1']:>8.3f} {r['p_lag_5']:>8.3f} {r['p_lag_10']:>8.3f}"
              f"  {r['half_life_str']:>5}  {r['asymptote']:>6.3f}")

    # Front-loaded vs distributed: compare mean_fwd at bar_1 vs bar_5_8 / bar_9_16
    print(f"\n  Front-loaded vs distributed signals "
          f"(bar_1 mean_fwd vs later bars):")
    if len(runpos_df) > 0:
        pivot = runpos_df.pivot_table(
            index=['concept', 'tf', 'regime_2d'],
            columns='run_pos', values='mean_fwd')
        if 'bar_1' in pivot.columns:
            pivot['delta_b1_to_b5_8'] = (pivot.get('bar_5_8', np.nan)
                                              - pivot['bar_1'])
            pivot_sorted = pivot.dropna(subset=['bar_1']).copy()
            pivot_sorted['abs_b1'] = pivot_sorted['bar_1'].abs()
            top = pivot_sorted.sort_values('abs_b1',
                                              ascending=False).head(20)
            print(f"    {'concept':>22}  {'tf':>4}  {'regime':>14}  "
                  f"{'b1':>8} {'b2':>8} {'b3-4':>8} {'b5-8':>8} {'b9-16':>8} "
                  f"{'b17+':>8}")
            for idx, r in top.iterrows():
                concept, tf, regime = idx
                def g(k):
                    v = r.get(k, np.nan)
                    return f'{v:>8.2f}' if not pd.isna(v) else f'{"nan":>8}'
                print(f"    {concept:>22}  {tf:>4}  {regime:>14}  "
                      f"{g('bar_1')} {g('bar_2')} {g('bar_3_4')} "
                      f"{g('bar_5_8')} {g('bar_9_16')} {g('bar_17p')}")

    # ---- plot ----
    plotted = 0
    plot_concepts = (hl_df.groupby('concept')['half_life_k']
                          .max().sort_values(ascending=False)
                          .head(args.top_plots).index.tolist())
    for concept in plot_concepts:
        for tf in args.tfs:
            keys = [k for k in autocorr_cache.keys()
                     if k[0] == concept and k[1] == tf]
            if not keys:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for (_, _, regime) in keys:
                p_arr = autocorr_cache[(concept, tf, regime)]
                ks = list(range(1, len(p_arr) + 1))
                ax.plot(ks, p_arr, marker='o', label=regime, alpha=0.85)
            ax.axhline(0.5, color='gray', linestyle=':', alpha=0.6)
            ax.axhline(1.0 / args.quantiles, color='red', linestyle=':',
                        alpha=0.6, label=f'random = 1/{args.quantiles}')
            ax.set_xlabel('lag (5m bars)')
            ax.set_ylabel(f'P(Q{args.quantile} at t+k | Q{args.quantile} at t)')
            ax.set_title(f'{concept} {tf} — Q{args.quantile} state autocorrelation')
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
            fig.tight_layout()
            png_path = os.path.join(args.output_dir,
                                       f'plot_autocorr_{concept}_{tf}.png')
            fig.savefig(png_path, dpi=120, bbox_inches='tight',
                          facecolor='white')
            plt.close(fig)
            plotted += 1
            if plotted >= args.top_plots:
                break
        if plotted >= args.top_plots:
            break
    print(f"\n  [saved] {plotted} autocorr plots")

    # ---- markdown ----
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 features within-day persistence (Step #2) - "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Concepts:** {args.concepts}\n\n")
        f.write(f"**TFs:** {args.tfs}\n\n")
        f.write(f"**Tracking Q{args.quantile} of {args.quantiles} | max lag {args.max_lag}**\n\n")
        f.write("## Top 30 most-persistent (concept, tf, regime)\n\n")
        f.write(hl_df.head(30).to_string(index=False))
        f.write("\n\n## Run-position forward returns (front-loaded vs distributed)\n\n")
        if len(runpos_df) > 0 and 'bar_1' in pivot.columns:
            f.write(top.to_string())
        f.write("\n")
    print(f"  [saved] {md_path}")


if __name__ == '__main__':
    main()
