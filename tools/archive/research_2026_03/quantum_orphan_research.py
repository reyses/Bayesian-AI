"""
Quantum Orphan Variable Research
================================
Validates whether the 9 orphaned quantum state variables (computed on GPU
every bar but never read) correlate with ACTUAL trade outcomes.

Uses oracle_trade_log entries as ground truth. For each trade entry,
finds the nearest 15s bar, computes the full MarketState, and extracts
the orphaned quantum fields. Then correlates with WIN/LOSS.

Usage:
    python tools/quantum_orphan_research.py --data DATA/ATLAS_1WEEK
    python tools/quantum_orphan_research.py --data DATA/ATLAS --month 2025_06
    python tools/quantum_orphan_research.py --data DATA/ATLAS_1DAY  (quick test)
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ORPHAN_FIELDS = ['P_at_center', 'P_near_upper', 'P_near_lower', 'entropy',
                 'entropy_normalized', 'breakout_probability', 'reversion_probability',
                 'reversion_potential']

INACTIVE_FIELDS = ['lyapunov_exponent', 'pattern_maturity']

ACTIVE_FIELDS = ['F_momentum', 'mean_reversion_force', 'net_force', 'velocity',
                 'hurst_exponent', 'adx_strength', 'regression_sigma', 'term_pid',
                 'z_score']

ALL_QUANTUM_FIELDS = ORPHAN_FIELDS + INACTIVE_FIELDS + ACTIVE_FIELDS


def load_15s_data(data_dir: str, month: str = None):
    """Load 15s bars from ATLAS."""
    tf_dir = os.path.join(data_dir, '15s')
    if not os.path.isdir(tf_dir):
        print(f"  ERROR: {tf_dir} not found")
        sys.exit(1)
    files = sorted(Path(tf_dir).glob('*.parquet'))
    if month:
        files = [f for f in files if month in f.stem]
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df


def run_quantum_orphan_analysis(data_dir: str, month: str = None):
    from core.statistical_field_engine import StatisticalFieldEngine

    engine = StatisticalFieldEngine(regression_period=21, use_gpu=True)

    print("=" * 70)
    print("  QUANTUM ORPHAN VARIABLE RESEARCH (Trade-Anchored)")
    print(f"  Data: {data_dir}" + (f" (month={month})" if month else ""))
    print("=" * 70)

    # --- Load oracle trade log ---
    print("\n[1] Loading oracle trade log...")
    trade_log_path = 'checkpoints/oracle_trade_log_old.csv'
    if not os.path.exists(trade_log_path):
        # Try current
        trade_log_path = 'reports/is/oracle_trade_log.csv'
    if not os.path.exists(trade_log_path):
        # Try shards
        shard_dir = 'reports/is/shards/'
        if os.path.isdir(shard_dir):
            shard_files = sorted(Path(shard_dir).glob('oracle_trade_log_*.csv'))
            if shard_files:
                dfs = [pd.read_csv(f) for f in shard_files]
                df_trades = pd.concat(dfs, ignore_index=True)
                print(f"  Loaded {len(df_trades):,} trades from {len(shard_files)} shards")
            else:
                print("  ERROR: No oracle trade log found")
                sys.exit(1)
        else:
            print("  ERROR: No oracle trade log found")
            sys.exit(1)
    else:
        df_trades = pd.read_csv(trade_log_path)
        print(f"  Loaded {len(df_trades):,} trades from {trade_log_path}")

    # --- Load 15s data ---
    print("\n[2] Loading 15s data...")
    df_bars = load_15s_data(data_dir, month)
    print(f"  {len(df_bars):,} bars loaded")
    bar_ts = df_bars['timestamp'].values

    # --- Filter trades to data range ---
    ts_min, ts_max = bar_ts[0], bar_ts[-1]
    df_trades = df_trades[(df_trades['entry_time'] >= ts_min) &
                          (df_trades['entry_time'] <= ts_max)].copy()
    print(f"  {len(df_trades):,} trades within data range")

    if len(df_trades) < 20:
        print("  ERROR: Too few trades. Use more data (try --data DATA/ATLAS --month 2025_06)")
        sys.exit(1)

    # --- Compute all market states ---
    print("\n[3] Computing market states (GPU)...")
    state_results = engine.batch_compute_states(df_bars, use_cuda=True)
    print(f"  {len(state_results):,} states computed")

    # Build timestamp → state index lookup
    state_timestamps = np.array([s['state'].timestamp for s in state_results])

    # --- For each trade, find nearest bar and extract quantum state ---
    print(f"\n[4] Extracting quantum state at {len(df_trades):,} trade entries...")
    records = []
    for _, trade in tqdm(df_trades.iterrows(), total=len(df_trades), desc="Matching"):
        entry_ts = trade['entry_time']
        # Find nearest 15s bar (within 15 seconds)
        idx = np.searchsorted(state_timestamps, entry_ts)
        if idx >= len(state_timestamps):
            idx = len(state_timestamps) - 1
        # Check if previous bar is closer
        if idx > 0 and abs(state_timestamps[idx-1] - entry_ts) < abs(state_timestamps[idx] - entry_ts):
            idx = idx - 1
        # Skip if too far (> 30 seconds)
        if abs(state_timestamps[idx] - entry_ts) > 30:
            continue

        state = state_results[idx]['state']
        rec = {
            'template_id': trade.get('template_id', -1),
            'direction': trade.get('direction', ''),
            'actual_pnl': trade.get('actual_pnl', 0),
            'result': trade.get('result', ''),
            'exit_reason': trade.get('exit_reason', ''),
            'hold_bars': trade.get('hold_bars', 0),
            'entry_depth': trade.get('entry_depth', 0),
            'root_tf': trade.get('root_tf', ''),
            'belief_conviction': trade.get('belief_conviction', 0),
            'oracle_mfe': trade.get('oracle_mfe', 0),
            'oracle_mae': trade.get('oracle_mae', 0),
        }
        # Extract ALL quantum fields
        for field in ALL_QUANTUM_FIELDS:
            rec[field] = getattr(state, field, 0.0)
        # Coherence is renamed
        rec['coherence'] = getattr(state, 'oscillation_entropy_normalized', 0.0)

        records.append(rec)

    df = pd.DataFrame(records)
    print(f"  Matched {len(df):,} trades to quantum states")

    # --- Classify outcomes ---
    df['is_win'] = (df['result'] == 'WIN').astype(int)
    wins = df[df['is_win'] == 1]
    losses = df[df['is_win'] == 0]
    total_wr = wins.shape[0] / len(df) * 100
    print(f"  Overall: {len(df):,} trades, {wins.shape[0]:,} wins, {losses.shape[0]:,} losses, WR={total_wr:.1f}%")

    # --- Orphan field statistics ---
    print(f"\n[5] Orphaned field statistics at trade entries:")
    for col in ORPHAN_FIELDS + INACTIVE_FIELDS:
        vals = df[col]
        n_zero = (vals == 0.0).sum()
        pct_zero = n_zero / len(vals) * 100
        status = "LIVE" if pct_zero < 95 else "DEAD"
        print(f"  {col:30s}  min={vals.min():8.4f}  max={vals.max():8.4f}  "
              f"mean={vals.mean():8.4f}  zeros={pct_zero:.1f}%  [{status}]")

    # --- Correlation analysis ---
    print(f"\n[6] CORRELATION: quantum fields vs WIN/LOSS")
    print(f"  {'Field':30s}  {'Corr':>8s}  {'p-value':>10s}  {'Win Med':>8s}  {'Loss Med':>8s}  {'Gap':>8s}")
    print("  " + "-" * 85)

    from scipy import stats

    correlations = {}
    for col in ORPHAN_FIELDS + ACTIVE_FIELDS + ['coherence']:
        vals = df[col]
        if vals.std() == 0:
            print(f"  {col:30s}  {'N/A':>8s}  {'constant':>10s}")
            continue

        corr, pval = stats.pointbiserialr(df['is_win'], vals)
        w_med = wins[col].median() if len(wins) > 0 else 0
        l_med = losses[col].median() if len(losses) > 0 else 0
        gap = w_med - l_med
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        correlations[col] = (corr, pval)

        print(f"  {col:30s}  {corr:>+8.4f}  {pval:>10.6f}{sig:3s}  "
              f"{w_med:>8.4f}  {l_med:>8.4f}  {gap:>+8.4f}")

    # --- Original scoring formula validation ---
    print(f"\n[7] ORIGINAL SCORING FORMULA: P_i > 0.75 AND tunnel > 0.60 AND low entropy")
    p_dominant = df[['P_at_center', 'P_near_upper', 'P_near_lower']].max(axis=1)
    df['p_dominant'] = p_dominant

    entropy_p25 = df['entropy'].quantile(0.25)
    entropy_p50 = df['entropy'].quantile(0.50)

    thresholds = [
        ("Original strict: P>0.75, tun>0.60, ent<P25", 0.75, 0.60, entropy_p25),
        ("Relaxed: P>0.65, tun>0.50, ent<P50",         0.65, 0.50, entropy_p50),
        ("Loose: P>0.55, tun>0.40, any entropy",        0.55, 0.40, 999),
        ("All trades (baseline)",                        0.00, 0.00, 999),
    ]

    for label, p_thresh, tun_thresh, ent_thresh in thresholds:
        mask = (p_dominant >= p_thresh) & (df['reversion_probability'] >= tun_thresh)
        if ent_thresh < 999:
            mask = mask & (df['entropy'] < ent_thresh)
        sub = df[mask]
        if len(sub) == 0:
            print(f"\n  {label}: 0 trades")
            continue
        n = len(sub)
        wr = sub['is_win'].mean() * 100
        avg_pnl = sub['actual_pnl'].mean()
        total_pnl = sub['actual_pnl'].sum()
        print(f"\n  {label}:")
        print(f"    Trades: {n:,}  WR: {wr:.1f}%  Avg PnL: ${avg_pnl:.2f}  Total: ${total_pnl:,.2f}")

    # --- Bucket analysis ---
    print(f"\n[8] BUCKET ANALYSIS")

    for col, label in [('p_dominant', 'P_dominant (wave function)'),
                        ('entropy', 'Entropy (chaos)'),
                        ('reversion_probability', 'Reversion probability (tunnel)'),
                        ('coherence', 'Coherence (TF alignment)')]:
        vals = df[col]
        if vals.nunique() < 4:
            continue
        print(f"\n  {label}:")
        try:
            df['_q'] = pd.qcut(vals, 4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                                duplicates='drop')
        except ValueError:
            continue
        for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
            sub = df[df['_q'] == q]
            if len(sub) == 0:
                continue
            n = len(sub)
            wr = sub['is_win'].mean() * 100
            avg_pnl = sub['actual_pnl'].mean()
            print(f"    {q:12s}  n={n:>5,}  WR={wr:5.1f}%  avg_pnl=${avg_pnl:>7.2f}")
        df.drop(columns=['_q'], inplace=True)

    # --- Mann-Whitney U test for key orphans ---
    print(f"\n[9] MANN-WHITNEY U TEST (non-parametric): WIN vs LOSS distributions")
    for col in ORPHAN_FIELDS + ['coherence']:
        if df[col].std() == 0:
            continue
        w_vals = wins[col].dropna()
        l_vals = losses[col].dropna()
        if len(w_vals) < 10 or len(l_vals) < 10:
            continue
        u_stat, u_pval = stats.mannwhitneyu(w_vals, l_vals, alternative='two-sided')
        sig = "***" if u_pval < 0.001 else "**" if u_pval < 0.01 else "*" if u_pval < 0.05 else ""
        print(f"  {col:30s}  U={u_stat:>10.0f}  p={u_pval:.6f}{sig}")

    # --- Decision tree on quantum fields only ---
    print(f"\n[10] DECISION TREE: quantum orphans predicting WIN/LOSS")
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score

    feature_cols = ORPHAN_FIELDS + ['coherence', 'hurst_exponent', 'adx_strength',
                                     'term_pid', 'F_momentum', 'mean_reversion_force']

    X = df[feature_cols].fillna(0).values
    y = df['is_win'].values

    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(20, len(df) // 50))
    scores = cross_val_score(tree, X, y, cv=5, scoring='accuracy')
    f1 = cross_val_score(tree, X, y, cv=5, scoring='f1')
    print(f"  5-fold CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
    print(f"  5-fold CV F1:       {f1.mean():.3f} +/- {f1.std():.3f}")

    tree.fit(X, y)
    importances = sorted(zip(feature_cols, tree.feature_importances_),
                          key=lambda x: -x[1])
    print(f"\n  Feature importances:")
    for feat, imp in importances:
        if imp > 0.01:
            bar = "#" * int(imp * 40)
            print(f"    {feat:30s}  {imp:.4f}  {bar}")

    print(f"\n  Tree rules (depth 3):")
    print(export_text(tree, feature_names=feature_cols, max_depth=3))

    # --- Orphans-only tree (no active fields) ---
    print(f"\n[11] ORPHANS-ONLY TREE (no active fields — pure quantum scoring)")
    X_orphan = df[ORPHAN_FIELDS].fillna(0).values
    tree_orphan = DecisionTreeClassifier(max_depth=3, min_samples_leaf=max(20, len(df) // 50))
    scores_orphan = cross_val_score(tree_orphan, X_orphan, y, cv=5, scoring='accuracy')
    f1_orphan = cross_val_score(tree_orphan, X_orphan, y, cv=5, scoring='f1')
    print(f"  5-fold CV accuracy: {scores_orphan.mean():.3f} +/- {scores_orphan.std():.3f}")
    print(f"  5-fold CV F1:       {f1_orphan.mean():.3f} +/- {f1_orphan.std():.3f}")

    tree_orphan.fit(X_orphan, y)
    importances_orphan = sorted(zip(ORPHAN_FIELDS, tree_orphan.feature_importances_),
                                 key=lambda x: -x[1])
    print(f"\n  Orphan feature importances:")
    for feat, imp in importances_orphan:
        if imp > 0.01:
            bar = "#" * int(imp * 40)
            print(f"    {feat:30s}  {imp:.4f}  {bar}")

    print(f"\n  Orphan tree rules:")
    print(export_text(tree_orphan, feature_names=ORPHAN_FIELDS, max_depth=3))

    # --- Save report ---
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"reports/findings/quantum_orphan_research_{ts}.txt"
    lines = []
    lines.append(f"QUANTUM ORPHAN RESEARCH (Trade-Anchored) — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Data: {data_dir}" + (f" (month={month})" if month else ""))
    lines.append(f"Trades: {len(df):,}  WR: {total_wr:.1f}%")
    lines.append("")
    lines.append("CORRELATIONS (point-biserial r):")
    for col, (corr, pval) in sorted(correlations.items(), key=lambda x: -abs(x[1][0])):
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        lines.append(f"  {col:30s}  r={corr:+.4f}  p={pval:.6f}{sig}")
    lines.append("")
    lines.append(f"ALL FIELDS TREE: {scores.mean():.3f} accuracy, {f1.mean():.3f} F1")
    lines.append(f"ORPHAN-ONLY TREE: {scores_orphan.mean():.3f} accuracy, {f1_orphan.mean():.3f} F1")
    for feat, imp in importances[:5]:
        lines.append(f"  {feat:30s}  {imp:.4f}")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum Orphan Variable Research')
    parser.add_argument('--data', default='DATA/ATLAS_1WEEK', help='ATLAS data directory')
    parser.add_argument('--month', default=None, help='Filter to specific month (e.g. 2025_06)')
    args = parser.parse_args()

    run_quantum_orphan_analysis(args.data, args.month)
