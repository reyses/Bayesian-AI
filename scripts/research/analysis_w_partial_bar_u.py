#!/usr/bin/env python3
"""
Analysis W: Analysis U Under Live Conditions (Partial Bar Robustness)
=====================================================================
Standalone research script. No core/ or live/ modifications.

Tests whether k-NN direction prediction (Analysis U) survives when slow TFs
(4h, 1D, 1W) use stale bars — simulating live conditions where these bars
are incomplete mid-formation.

Scenarios:
  1. Complete bars (ground truth baseline)
  2. Partial ALL slow TFs (4h, 1D, 1W use N-2 instead of N-1)
  3. Degrade 4h only
  4. Degrade 1D only
  5. Degrade 1W only

Gate:
  PROMOTE if partial-bar accuracy within 5pp of complete
  KILL if degradation > 15pp
  DEFER if one specific TF causes >10pp drop (can zero it out)

Usage:
    python scripts/research/analysis_w_partial_bar_u.py
    python scripts/research/analysis_w_partial_bar_u.py --data DATA/ATLAS_1MONTH
    python scripts/research/analysis_w_partial_bar_u.py --data DATA/ATLAS --analysis-days 270
"""
import argparse
import os
import sys
import time
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.research.data import (
    load_atlas_tf, compute_tf_physics, extract_16d,
    TF_HIERARCHY, TF_SECONDS, FEATURE_NAMES,
)
from config.oracle_config import ORACLE_LOOKAHEAD_BARS

ANALYSIS_ID = 'W_partial_bar_u'
OUT_DIR = os.path.join('reports', 'research', ANALYSIS_ID)
os.makedirs(OUT_DIR, exist_ok=True)

TICK = 0.25

# TFs considered "slow" — their bars take hours/days to complete
SLOW_TFS = {'4h', '1D', '1W'}


# ---------------------------------------------------------------------------
# Build 192D matrices with configurable degradation
# ---------------------------------------------------------------------------
def build_matrices_with_degradation(all_tf_states, base_tf, base_df,
                                     context_days, analysis_days,
                                     degrade_tfs=None):
    """Build (12, 16) matrices like build_stacked_matrices but with
    configurable bar staleness for specified TFs.

    degrade_tfs: set of TF names to use N-2 (extra stale) instead of N-1.
                 None = use standard alignment (complete bars).
    """
    if degrade_tfs is None:
        degrade_tfs = set()

    base_states = all_tf_states.get(base_tf, {})
    if not base_states:
        return [], np.array([]), np.array([]), []

    base_timestamps = sorted(base_states.keys())
    t_min, t_max = base_timestamps[0], base_timestamps[-1]

    from datetime import datetime, timezone
    data_span_days = (t_max - t_min) / 86400

    ctx = context_days
    if ctx > 0 and data_span_days < ctx + 1:
        ctx = max(0, int(data_span_days * 0.3))

    t_warmup_end = t_min + ctx * 86400
    t_analysis_end = (t_warmup_end + analysis_days * 86400) if analysis_days > 0 else t_max + 1

    analysis_ts = [t for t in base_timestamps if t_warmup_end <= t < t_analysis_end]

    if not analysis_ts:
        return [], np.array([]), np.array([]), []

    # Pre-sort timestamps per TF
    tf_sorted_ts = {}
    for tf in TF_HIERARCHY:
        if tf in all_tf_states and all_tf_states[tf]:
            tf_sorted_ts[tf] = sorted(all_tf_states[tf].keys())

    # Base DF index mapping
    ts_col = base_df['timestamp'].values if 'timestamp' in base_df.columns else np.arange(len(base_df))
    ts_to_idx = {int(ts_col[i]): i for i in range(len(ts_col))}

    lookahead = ORACLE_LOOKAHEAD_BARS.get(base_tf, 16)
    base_secs = TF_SECONDS.get(base_tf, 900)

    matrices, signed_mfes, meta = [], [], []

    degrade_label = ','.join(sorted(degrade_tfs)) if degrade_tfs else 'none'
    desc = f"Matrices (degrade={degrade_label})"

    for t in tqdm(analysis_ts, desc=desc, ascii=True, dynamic_ncols=True, mininterval=0.3):
        mat = np.zeros((12, 16))
        has_data = 0

        for depth_idx, tf in enumerate(TF_HIERARCHY):
            if tf not in tf_sorted_ts:
                continue

            tf_ts_list = tf_sorted_ts[tf]
            tf_secs = TF_SECONDS.get(tf, 60)

            # Standard alignment: slower TFs use N-1 (last completed bar)
            if tf_secs > base_secs:
                offset = -2  # N-1 in searchsorted terms
            else:
                offset = -1

            # Extra degradation: use N-2 for specified TFs
            if tf in degrade_tfs:
                offset -= 1  # one bar staler

            idx = np.searchsorted(tf_ts_list, t, side='right') + offset
            if idx < 0:
                continue

            nearest_ts = tf_ts_list[idx]
            state = all_tf_states[tf][nearest_ts]
            mat[depth_idx, :] = extract_16d(state, tf)
            has_data += 1

        if has_data < 3:
            continue

        if t not in ts_to_idx:
            continue
        bar_idx = ts_to_idx[t]
        if bar_idx + lookahead >= len(base_df):
            continue

        entry_price = float(base_df.iloc[bar_idx]['close'])
        future = base_df.iloc[bar_idx + 1: bar_idx + 1 + lookahead]
        if future.empty:
            continue

        max_up = float(future['high'].max() - entry_price)
        max_down = float(entry_price - future['low'].min())
        signed_mfe = max_up - max_down  # positive = LONG better

        matrices.append(mat)
        signed_mfes.append(signed_mfe)
        meta.append({'ts': t, 'idx': len(matrices) - 1})

    return matrices, np.array(signed_mfes), meta, analysis_ts


# ---------------------------------------------------------------------------
# k-NN experiment (reused from Analysis V)
# ---------------------------------------------------------------------------
def run_knn_experiment(X, y, label, k=50):
    """Run k-NN direction + CI prediction. Returns metrics dict."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    split = int(n * 0.75)
    if split < k + 10 or n - split < 10:
        return None

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    scaler = StandardScaler().fit(X_train)
    X_train_sc = np.nan_to_num(scaler.transform(X_train), nan=0.0, posinf=0.0, neginf=0.0)
    X_test_sc = np.nan_to_num(scaler.transform(X_test), nan=0.0, posinf=0.0, neginf=0.0)

    k_actual = min(k, len(X_train) // 10, len(X_train) - 1)
    if k_actual < 5:
        return None

    knn = NearestNeighbors(n_neighbors=k_actual, metric='euclidean', n_jobs=-1)
    knn.fit(X_train_sc)
    dists, indices = knn.kneighbors(X_test_sc)

    results = []
    for i in range(len(X_test)):
        nbr_mfe = y_train[indices[i]]
        p10, p25, p50, p75, p90 = np.percentile(nbr_mfe, [10, 25, 50, 75, 90])
        predicted_dir = 'LONG' if p50 > 0 else 'SHORT'
        actual_dir = 'LONG' if y_test[i] > 0 else 'SHORT'
        nbr_long_pct = np.mean(nbr_mfe > 0)
        consensus = max(nbr_long_pct, 1 - nbr_long_pct)

        results.append({
            'p10': p10, 'p25': p25, 'p50': p50, 'p75': p75, 'p90': p90,
            'predicted_dir': predicted_dir, 'actual_dir': actual_dir,
            'actual_signed_mfe': y_test[i],
            'consensus': consensus,
            'ci_width': p75 - p25,
        })

    rdf = pd.DataFrame(results)
    dir_correct = (rdf['predicted_dir'] == rdf['actual_dir']).sum()
    dir_acc = dir_correct / len(rdf)
    baseline = max((rdf['actual_dir'] == 'LONG').sum(),
                   (rdf['actual_dir'] == 'SHORT').sum()) / len(rdf)

    ci50 = ((rdf['p25'] <= rdf['actual_signed_mfe']) &
            (rdf['actual_signed_mfe'] <= rdf['p75'])).mean()
    ci80 = ((rdf['p10'] <= rdf['actual_signed_mfe']) &
            (rdf['actual_signed_mfe'] <= rdf['p90'])).mean()

    hi_cons = rdf[rdf['consensus'] >= 0.90]
    hi_cons_acc = (hi_cons['predicted_dir'] == hi_cons['actual_dir']).mean() \
        if len(hi_cons) > 0 else 0.0

    return {
        'label': label,
        'n_test': len(rdf),
        'dir_accuracy': dir_acc,
        'baseline': baseline,
        'lift': dir_acc - baseline,
        'ci50_coverage': ci50,
        'ci80_coverage': ci80,
        'ci_width_median': rdf['ci_width'].median(),
        'hi_cons_n': len(hi_cons),
        'hi_cons_acc': hi_cons_acc,
        'results_df': rdf,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plots(scenario_results):
    """Generate the 4 required plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    valid = {k: v for k, v in scenario_results.items() if v is not None}
    if len(valid) < 2:
        print("  Not enough results for plots.")
        return

    complete = valid.get('Complete bars')

    # ---- Plot 1: Direction accuracy bar chart (all scenarios) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(valid.keys())
    accs = [valid[l]['dir_accuracy'] * 100 for l in labels]
    colors = ['#228833' if l == 'Complete bars' else
              '#EE6677' if 'ALL' in l else '#4477AA' for l in labels]
    bars = ax.bar(range(len(labels)), accs, color=colors, alpha=0.85)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Direction Accuracy %')
    ax.set_title('W: Direction Accuracy — Complete vs Partial Bars')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.4)
    if complete:
        ax.axhline(complete['dir_accuracy'] * 100, color='#228833',
                    linestyle=':', alpha=0.6, label='Complete baseline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot1_accuracy_scenarios.png'), dpi=150)
    plt.close()

    # ---- Plot 2: CI calibration overlay ----
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, color in [('Complete bars', '#228833'), ('Partial ALL slow', '#EE6677')]:
        r = valid.get(label)
        if r is None:
            continue
        rdf = r['results_df']
        nominal = [0.5, 0.8]
        actual_cov = [r['ci50_coverage'], r['ci80_coverage']]
        ax.bar([f'{label}\nCI50', f'{label}\nCI80'], [c * 100 for c in actual_cov],
               color=color, alpha=0.7)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.4, label='50% target')
    ax.axhline(80, color='gray', linestyle=':', alpha=0.4, label='80% target')
    ax.set_ylabel('Actual Coverage %')
    ax.set_title('CI Coverage: Complete vs Partial')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot2_ci_coverage.png'), dpi=150)
    plt.close()

    # ---- Plot 3: Per-TF degradation waterfall ----
    fig, ax = plt.subplots(figsize=(8, 6))
    if complete:
        base_acc = complete['dir_accuracy'] * 100
        degrade_labels = []
        deltas = []
        for label in ['Degrade 4h', 'Degrade 1D', 'Degrade 1W', 'Partial ALL slow']:
            r = valid.get(label)
            if r:
                degrade_labels.append(label)
                deltas.append(r['dir_accuracy'] * 100 - base_acc)

        colors = ['#EE6677' if d < -2 else '#CCBB44' if d < 0 else '#228833' for d in deltas]
        bars = ax.bar(range(len(degrade_labels)), deltas, color=colors, alpha=0.85)
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.2 if d >= 0 else -0.5),
                    f'{d:+.1f}pp', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(range(len(degrade_labels)))
        ax.set_xticklabels(degrade_labels, rotation=15, ha='right')
        ax.set_ylabel('Direction Accuracy Delta (pp)')
        ax.set_title(f'Per-TF Degradation vs Complete ({base_acc:.1f}%)')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axhline(-5, color='#EE6677', linestyle='--', alpha=0.5, label='-5pp threshold')
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot3_degradation_waterfall.png'), dpi=150)
    plt.close()

    # ---- Plot 4: Accuracy by consensus bin (complete vs partial) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ['50-60%', '60-70%', '70-80%', '80-90%', '90-100%']

    for label, color, offset in [('Complete bars', '#228833', -0.15),
                                  ('Partial ALL slow', '#EE6677', 0.15)]:
        r = valid.get(label)
        if r is None:
            continue
        rdf = r['results_df']
        accs, counts = [], []
        for j in range(len(bins) - 1):
            mask = (rdf['consensus'] >= bins[j]) & (rdf['consensus'] < bins[j + 1])
            subset = rdf[mask]
            if len(subset) > 0:
                accs.append((subset['predicted_dir'] == subset['actual_dir']).mean() * 100)
                counts.append(len(subset))
            else:
                accs.append(0)
                counts.append(0)
        x = np.arange(len(bin_labels))
        bars = ax.bar(x + offset, accs, width=0.28, label=label, color=color, alpha=0.85)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'n={cnt}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(np.arange(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel('Neighbor Consensus')
    ax.set_ylabel('Direction Accuracy %')
    ax.set_title('W: Accuracy by Consensus — Complete vs Partial')
    ax.legend()
    ax.axhline(50, color='gray', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'plot4_consensus_bins.png'), dpi=150)
    plt.close()

    print(f"  Plots saved to {OUT_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Analysis W: Partial Bar Robustness')
    parser.add_argument('--data', default='DATA/ATLAS_1MONTH',
                        help='ATLAS data directory')
    parser.add_argument('--base-tf', default='15m',
                        help='Base timeframe (default: 15m)')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup/context days')
    parser.add_argument('--analysis-days', type=int, default=0,
                        help='Analysis window days (0=all remaining)')
    parser.add_argument('--k', type=int, default=50,
                        help='k neighbors (default: 50)')
    args = parser.parse_args()

    t0 = time.perf_counter()
    print(f'Analysis W: Partial Bar Robustness Test')
    print(f'Data: {args.data}, Base TF: {args.base_tf}, k={args.k}')
    print('=' * 70)

    # ---- 1. Load data & compute physics ----
    print("\n[1/4] Loading ATLAS data and computing physics...")
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="TF physics", ascii=True):
        df = load_atlas_tf(args.data, tf)
        if df.empty:
            continue
        states = compute_tf_physics(tf, df)
        if states:
            all_tf_states[tf] = states

    base_df = load_atlas_tf(args.data, args.base_tf)
    if base_df.empty:
        print(f"ERROR: No data for base TF {args.base_tf}")
        return

    available_slow = [tf for tf in SLOW_TFS if tf in all_tf_states]
    print(f"  Loaded {len(all_tf_states)} TFs, base has {len(base_df)} bars")
    print(f"  Slow TFs available for degradation: {available_slow}")

    # ---- 2. Build matrices for each scenario ----
    print("\n[2/4] Building matrices for each scenario...")

    scenarios = {}

    # Scenario 1: Complete bars (baseline)
    print("\n  --- Complete bars (baseline) ---")
    mats, y, meta, _ = build_matrices_with_degradation(
        all_tf_states, args.base_tf, base_df,
        args.context_days, args.analysis_days, degrade_tfs=None)
    if mats:
        X = np.array([m.flatten() for m in mats])
        scenarios['Complete bars'] = (X, y)
        print(f"    {len(X)} samples")

    # Scenario 2: Partial ALL slow TFs
    print("\n  --- Partial ALL slow TFs ---")
    mats, y, meta, _ = build_matrices_with_degradation(
        all_tf_states, args.base_tf, base_df,
        args.context_days, args.analysis_days,
        degrade_tfs=set(available_slow))
    if mats:
        X = np.array([m.flatten() for m in mats])
        scenarios['Partial ALL slow'] = (X, y)
        print(f"    {len(X)} samples")

    # Scenarios 3-5: Degrade one TF at a time
    for tf in available_slow:
        label = f"Degrade {tf}"
        print(f"\n  --- {label} ---")
        mats, y, meta, _ = build_matrices_with_degradation(
            all_tf_states, args.base_tf, base_df,
            args.context_days, args.analysis_days,
            degrade_tfs={tf})
        if mats:
            X = np.array([m.flatten() for m in mats])
            scenarios[label] = (X, y)
            print(f"    {len(X)} samples")

    # ---- 3. Run k-NN for each scenario ----
    print("\n[3/4] Running k-NN experiments...")
    scenario_results = {}
    for label, (X, y) in scenarios.items():
        print(f"\n  --- {label} ---")
        result = run_knn_experiment(X, y, label, k=args.k)
        scenario_results[label] = result
        if result:
            print(f"    Dir accuracy: {result['dir_accuracy']*100:.1f}%, "
                  f"CI50: {result['ci50_coverage']*100:.1f}%, "
                  f"CI80: {result['ci80_coverage']*100:.1f}%")

    # ---- 4. Results ----
    print("\n[4/4] Compiling results...")
    lines = []
    lines.append("=" * 80)
    lines.append("ANALYSIS W: PARTIAL BAR ROBUSTNESS TEST")
    lines.append(f"Data: {args.data}, Base TF: {args.base_tf}, k={args.k}")
    lines.append(f"Slow TFs tested: {available_slow}")
    lines.append("=" * 80)
    lines.append("")

    complete = scenario_results.get('Complete bars')
    base_acc = complete['dir_accuracy'] if complete else 0

    header = f"{'Scenario':<25} {'N_test':>7} {'DirAcc':>8} {'Baseline':>8} {'Lift':>7} {'CI50':>7} {'CI80':>7} {'vs Complete':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for label, r in scenario_results.items():
        if r is None:
            lines.append(f"{label:<25} {'(failed)':>7}")
            continue
        delta = (r['dir_accuracy'] - base_acc) * 100 if complete else 0
        delta_str = f"{delta:+.1f}pp" if label != 'Complete bars' else 'baseline'
        lines.append(f"{label:<25} {r['n_test']:>7} {r['dir_accuracy']*100:>7.1f}% "
                     f"{r['baseline']*100:>7.1f}% {r['lift']*100:>+6.1f}% "
                     f"{r['ci50_coverage']*100:>6.1f}% {r['ci80_coverage']*100:>6.1f}% "
                     f"{delta_str:>12}")

    # Gate evaluation
    lines.append("")
    lines.append("=" * 70)
    lines.append("GATE EVALUATION")
    lines.append("=" * 70)

    if complete:
        partial = scenario_results.get('Partial ALL slow')
        if partial:
            delta_acc = (partial['dir_accuracy'] - complete['dir_accuracy']) * 100
            delta_ci50 = (partial['ci50_coverage'] - complete['ci50_coverage']) * 100
            delta_ci80 = (partial['ci80_coverage'] - complete['ci80_coverage']) * 100

            lines.append(f"  Direction accuracy delta (partial vs complete): {delta_acc:+.1f}pp")
            lines.append(f"  CI50 coverage delta:  {delta_ci50:+.1f}pp")
            lines.append(f"  CI80 coverage delta:  {delta_ci80:+.1f}pp")
            lines.append("")

            # Per-TF breakdown
            lines.append("  Per-TF impact:")
            worst_tf = None
            worst_delta = 0
            for tf in available_slow:
                r = scenario_results.get(f'Degrade {tf}')
                if r:
                    d = (r['dir_accuracy'] - complete['dir_accuracy']) * 100
                    lines.append(f"    {tf:>4}: {d:+.1f}pp")
                    if d < worst_delta:
                        worst_delta = d
                        worst_tf = tf

            lines.append("")

            if abs(delta_acc) <= 5.0 and abs(delta_ci50) <= 5.0:
                verdict = ">>> PROMOTE: k-NN is robust to partial bars — safe for live deployment"
            elif abs(delta_acc) > 15.0:
                verdict = ">>> KILL: Severe degradation — k-NN needs partial bar interpolation before live"
            elif worst_tf and worst_delta < -10.0:
                verdict = (f">>> DEFER: {worst_tf} causes {worst_delta:+.1f}pp drop — "
                           f"deploy WITHOUT {worst_tf} features (zero them out)")
            else:
                verdict = f">>> MODERATE: {delta_acc:+.1f}pp degradation — review per-TF contributions"

            lines.append(verdict)
        else:
            lines.append("  No partial bar results — cannot evaluate gate.")

    results_text = "\n".join(lines)
    print("\n" + results_text)

    with open(os.path.join(OUT_DIR, 'results.txt'), 'w') as f:
        f.write(results_text)

    # Save detail CSVs
    for label, r in scenario_results.items():
        if r and 'results_df' in r:
            safe_label = label.replace(' ', '_').replace('/', '_')
            r['results_df'].to_csv(
                os.path.join(OUT_DIR, f'detail_{safe_label}.csv'), index=False)

    # Plots
    print("\n  Generating plots...")
    make_plots(scenario_results)

    # Journal
    _append_journal(ANALYSIS_ID, results_text)

    elapsed = time.perf_counter() - t0
    print(f'\nCompleted in {elapsed:.1f}s')
    print(f'Results: {OUT_DIR}/results.txt')


def _append_journal(analysis_id, text):
    journal = 'docs/reference/RESEARCH_JOURNAL.txt'
    if not os.path.exists(journal):
        print(f"  Journal not found at {journal}, skipping append.")
        return
    with open(journal, 'a', encoding='utf-8') as f:
        f.write(f'\n\n{"=" * 77}\n')
        f.write(f'ANALYSIS {analysis_id.upper()} (auto-generated {time.strftime("%Y-%m-%d")})\n')
        f.write(f'{"=" * 77}\n\n')
        f.write(text)
    print(f"  Appended to {journal}")


if __name__ == '__main__':
    main()
