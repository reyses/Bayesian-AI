"""
v2 Composite Directional — L-Aggregator
========================================
Reads per-TF Analysis L predictions exported by standalone_research.py
(`analysis_l_predictions.csv` in each `tools/plots/standalone/1y_<tf>/` dir),
projects them onto a common decision cadence, and evaluates several
combinator strategies.

This sidesteps the target-mismatch problem: standalone L's signed-MFE
target uses regime-determined direction (smooth segmentation), not raw
per-bar excursion. By reading the standalone outputs directly, we get
exact parity with the per-TF reports — no refit needed.

Combinators evaluated:
  - Majority vote (>=N TFs agree, stratified)
  - Strict-all (every active TF agrees, no dissent)
  - Magnitude-weighted average (sign of weighted sum of pred_signed_mfe)
  - Confidence-gated majority (only count voters whose |pred| > threshold)

Outputs:
  reports/findings/v2_composite_l/
    per_tf_summary.csv
    composite_majority.csv
    composite_strict.csv
    composite_weighted.csv
    composite_confgated.csv
    summary.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from config.oracle_config import ORACLE_LOOKAHEAD_BARS

# Default voter set (1D excluded — overfits with 97 test samples × 185 features)
DEFAULT_TFS = ['1m', '5m', '15m', '1h', '4h']
DEFAULT_COMMON_TF = '5m'
PLOTS_BASE = 'tools/plots/standalone'


def load_per_tf_predictions(tf: str, plots_base: str = PLOTS_BASE) -> pd.DataFrame | None:
    """Load analysis_l_predictions.csv for one base TF run.

    Expects columns: timestamp, actual_signed_mfe, pred_signed_mfe, pred_dir.
    Returns None if file missing.
    """
    path = os.path.join(plots_base, f'1y_{tf}', 'analysis_l_predictions.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def project_onto_common(per_tf_dfs: dict[str, pd.DataFrame],
                        common_ts: np.ndarray) -> dict[str, np.ndarray]:
    """For each TF, project (pred_signed_mfe, pred_dir) onto common cadence.

    Returns dict {tf: (pred_arr, dir_arr)} — each array of len(common_ts).
    Bars before any TF prediction get 0 (FLAT).
    """
    out = {}
    for tf, df in per_tf_dfs.items():
        ts = df['timestamp'].values.astype(np.int64)
        pred = df['pred_signed_mfe'].values.astype(np.float64)
        direction = df['pred_dir'].values.astype(np.float64)
        idx = np.searchsorted(ts, common_ts, side='right') - 1
        valid = idx >= 0
        proj_pred = np.zeros(len(common_ts), dtype=np.float64)
        proj_dir = np.zeros(len(common_ts), dtype=np.float64)
        proj_pred[valid] = pred[idx[valid]]
        proj_dir[valid] = direction[idx[valid]]
        out[tf] = (proj_pred, proj_dir)
    return out


def compute_actual_direction(common_df: pd.DataFrame, common_test_mask: np.ndarray,
                              lookahead: int) -> np.ndarray:
    """Compute actual signed-MFE direction at each common-TF bar.

    Returns array of -1, 0, +1 for SHORT, FLAT, LONG (FLAT when both legs
    equal, near-zero, or near end-of-data).
    """
    high = common_df['high'].values[common_test_mask]
    low = common_df['low'].values[common_test_mask]
    close = common_df['close'].values[common_test_mask]
    n = len(close)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n - lookahead):
        entry = close[i]
        max_up = float(high[i + 1: i + 1 + lookahead].max() - entry)
        max_dn = float(entry - low[i + 1: i + 1 + lookahead].min())
        if max_up > max_dn:
            out[i] = 1.0
        elif max_dn > max_up:
            out[i] = -1.0
    return out


def evaluate_combinator(consensus: np.ndarray, actual_dir: np.ndarray,
                         valid: np.ndarray, baseline_acc: float,
                         label: str) -> dict:
    """Score a per-bar consensus vector against actual direction.

    Returns dict with overall accuracy, lift, and per-side stats."""
    m = (consensus != 0) & valid
    if m.sum() == 0:
        return {'label': label, 'n_active': 0, 'accuracy': float('nan'),
                'lift': float('nan'), 'pct_data': 0.0}
    sub_acc = float((consensus[m] == actual_dir[m]).mean())
    n_long = int((consensus[m] == 1).sum())
    n_short = int((consensus[m] == -1).sum())
    long_acc = float((actual_dir[m & (consensus == 1)] == 1).mean()) \
        if (consensus[m] == 1).sum() > 0 else float('nan')
    short_acc = float((actual_dir[m & (consensus == -1)] == -1).mean()) \
        if (consensus[m] == -1).sum() > 0 else float('nan')
    return {
        'label': label,
        'n_active': int(m.sum()),
        'pct_data': float(m.sum() / valid.sum() * 100),
        'accuracy': sub_acc,
        'lift': sub_acc - baseline_acc,
        'n_long': n_long,
        'n_short': n_short,
        'long_acc': long_acc,
        'short_acc': short_acc,
    }


# ── Combinator strategies ────────────────────────────────────────────────

def combinator_majority(dir_matrix: np.ndarray) -> np.ndarray:
    """Per bar: +1 if more TFs LONG than SHORT, -1 if reverse, 0 if tied."""
    n_long = (dir_matrix == 1).sum(axis=1)
    n_short = (dir_matrix == -1).sum(axis=1)
    out = np.zeros(dir_matrix.shape[0], dtype=np.int8)
    out[n_long > n_short] = 1
    out[n_short > n_long] = -1
    return out


def combinator_strict(dir_matrix: np.ndarray) -> np.ndarray:
    """Per bar: +1 if ALL non-zero voters agree LONG; -1 if ALL agree SHORT; 0 else."""
    n_voters = dir_matrix.shape[1]
    out = np.zeros(dir_matrix.shape[0], dtype=np.int8)
    for i in range(dir_matrix.shape[0]):
        votes = dir_matrix[i]
        long_count = int((votes == 1).sum())
        short_count = int((votes == -1).sum())
        if long_count >= 1 and short_count == 0:
            out[i] = 1
        elif short_count >= 1 and long_count == 0:
            out[i] = -1
    return out


def combinator_weighted(pred_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sign of sum of normalized predictions across TFs.

    Each TF's predictions get standardized (zero mean, unit std on its row's
    distribution) before summing, so a TF with naturally larger predictions
    doesn't dominate.

    Returns (consensus, weighted_sum) — consensus = sign of weighted_sum.
    """
    # Normalize each TF column to unit std
    stds = pred_matrix.std(axis=0, keepdims=True)
    stds = np.where(stds > 1e-9, stds, 1.0)
    normalized = pred_matrix / stds
    weighted = normalized.sum(axis=1)
    consensus = np.sign(weighted).astype(np.int8)
    return consensus, weighted


def combinator_confgated_majority(pred_matrix: np.ndarray, dir_matrix: np.ndarray,
                                    threshold_per_tf: dict[str, float],
                                    tf_order: list[str]) -> np.ndarray:
    """Majority vote, but only count TFs whose |pred| > threshold for that TF."""
    n_bars = pred_matrix.shape[0]
    n_long = np.zeros(n_bars, dtype=np.int32)
    n_short = np.zeros(n_bars, dtype=np.int32)
    for j, tf in enumerate(tf_order):
        thr = threshold_per_tf.get(tf, 0.0)
        active = np.abs(pred_matrix[:, j]) > thr
        n_long += (active & (dir_matrix[:, j] == 1)).astype(np.int32)
        n_short += (active & (dir_matrix[:, j] == -1)).astype(np.int32)
    out = np.zeros(n_bars, dtype=np.int8)
    out[n_long > n_short] = 1
    out[n_short > n_long] = -1
    return out


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfs', nargs='+', default=DEFAULT_TFS)
    parser.add_argument('--common-tf', default=DEFAULT_COMMON_TF)
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--plots-base', default=PLOTS_BASE)
    parser.add_argument('--output-dir', default='reports/findings/v2_composite_l')
    parser.add_argument('--conf-threshold', type=float, default=10.0,
                        help='|pred| threshold for confgated_majority combinator')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"  V2 Composite L-Aggregator")
    print(f"  Voter TFs: {args.tfs}")
    print(f"  Common cadence: {args.common_tf}")
    print(f"  Conf threshold: {args.conf_threshold}")
    print(f"{'='*70}")

    # Load per-TF predictions
    per_tf_dfs = {}
    per_tf_rows = []
    for tf in args.tfs:
        df = load_per_tf_predictions(tf, args.plots_base)
        if df is None:
            print(f"  [skip] {tf}: no analysis_l_predictions.csv found")
            continue
        per_tf_dfs[tf] = df
        # Per-TF baseline summary (using its OWN ts range)
        actual_dir = np.sign(df['actual_signed_mfe'].values)
        pred_dir = df['pred_dir'].values
        valid = actual_dir != 0
        if valid.sum() > 0:
            acc = float((actual_dir[valid] == pred_dir[valid]).mean())
            base = float(max((actual_dir[valid] == 1).mean(), (actual_dir[valid] == -1).mean()))
            per_tf_rows.append({
                'tf': tf, 'n': int(valid.sum()),
                'accuracy': acc, 'baseline': base, 'lift': acc - base,
                'pred_std': float(df['pred_signed_mfe'].std()),
                'pred_mean_abs': float(df['pred_signed_mfe'].abs().mean()),
            })
            print(f"  {tf:>3}: n={valid.sum():>6}, "
                  f"acc={acc:.1%} (base={base:.1%}, lift={acc-base:+.1%}), "
                  f"|pred| mean={df['pred_signed_mfe'].abs().mean():.2f}")

    if len(per_tf_dfs) < 2:
        print(f"\nERROR: need >=2 TFs with predictions, got {len(per_tf_dfs)}.")
        return

    pd.DataFrame(per_tf_rows).to_csv(
        os.path.join(args.output_dir, 'per_tf_summary.csv'), index=False)

    # Common-cadence projection
    print(f"\n--- Projecting onto {args.common_tf} cadence ---")
    common_df = load_atlas_tf(args.data, args.common_tf)
    if common_df.empty:
        print(f"ERROR: no OHLC for common TF {args.common_tf}")
        return
    common_ts_all = common_df['timestamp'].values.astype(np.int64)
    if pd.api.types.is_datetime64_any_dtype(common_df['timestamp']):
        common_ts_all = common_ts_all // 10**9

    # Composite test window = intersection of per-TF prediction ranges
    starts = [int(df['timestamp'].iloc[0]) for df in per_tf_dfs.values()]
    ends = [int(df['timestamp'].iloc[-1]) for df in per_tf_dfs.values()]
    test_start = max(starts)
    test_end = min(ends)
    mask = (common_ts_all >= test_start) & (common_ts_all <= test_end)
    common_ts = common_ts_all[mask]
    print(f"  Test window: {test_start} -> {test_end}, {len(common_ts)} {args.common_tf} bars")

    proj = project_onto_common(per_tf_dfs, common_ts)
    tf_order = list(proj.keys())
    pred_matrix = np.column_stack([proj[tf][0] for tf in tf_order])  # (N, K) signed MFE preds
    dir_matrix = np.column_stack([proj[tf][1] for tf in tf_order]).astype(np.int8)  # (N, K) ±1

    # Actual direction at common cadence
    lookahead = ORACLE_LOOKAHEAD_BARS.get(args.common_tf, 16)
    actual_dir = compute_actual_direction(common_df, mask, lookahead)
    valid_y = actual_dir != 0
    baseline_acc = float(max((actual_dir[valid_y] == 1).mean(),
                              (actual_dir[valid_y] == -1).mean())) if valid_y.sum() > 0 else 0.5
    print(f"  Actual direction: {valid_y.sum()} non-flat bars, baseline={baseline_acc:.1%}")

    # Per-TF accuracy at the COMMON cadence (so rows are comparable)
    print(f"\n--- Per-TF accuracy at {args.common_tf} cadence ---")
    per_tf_common_rows = []
    for j, tf in enumerate(tf_order):
        d = dir_matrix[:, j]
        m = (d != 0) & valid_y
        if m.sum() == 0:
            continue
        sub_acc = float((d[m] == actual_dir[m]).mean())
        per_tf_common_rows.append({
            'tf': tf, 'n_active': int(m.sum()),
            'pct_data': float(m.sum() / valid_y.sum() * 100),
            'accuracy': sub_acc, 'lift': sub_acc - baseline_acc,
        })
        print(f"  {tf:>3}: n={m.sum():>5}, acc={sub_acc:.1%}, "
              f"lift={sub_acc-baseline_acc:+.1%}, %data={m.sum()/valid_y.sum()*100:.1f}%")

    # Combinator: majority vote, stratified by min agreement
    print(f"\n--- Composite: majority vote (stratified by agreement) ---")
    consensus_maj = combinator_majority(dir_matrix)
    n_long_voters = (dir_matrix == 1).sum(axis=1)
    n_short_voters = (dir_matrix == -1).sum(axis=1)
    consensus_strength = np.where(consensus_maj == 1, n_long_voters,
                                    np.where(consensus_maj == -1, n_short_voters, 0))
    maj_rows = []
    for k in range(1, len(tf_order) + 1):
        m = (consensus_strength >= k) & valid_y & (consensus_maj != 0)
        if m.sum() == 0:
            continue
        sub_acc = float((consensus_maj[m] == actual_dir[m]).mean())
        row = {'min_agree': k, 'n': int(m.sum()),
               'pct_data': float(m.sum() / valid_y.sum() * 100),
               'accuracy': sub_acc, 'lift': sub_acc - baseline_acc}
        maj_rows.append(row)
        print(f"    >= {k} agree: n={row['n']:>5}, acc={row['accuracy']:.1%}, "
              f"lift={row['lift']:+.1%}, %data={row['pct_data']:.1f}%")
    pd.DataFrame(maj_rows).to_csv(
        os.path.join(args.output_dir, 'composite_majority.csv'), index=False)

    # Combinator: strict-all
    print(f"\n--- Composite: strict-all (no dissent allowed) ---")
    consensus_strict = combinator_strict(dir_matrix)
    strict_score = evaluate_combinator(consensus_strict, actual_dir, valid_y,
                                         baseline_acc, 'strict-all')
    print(f"    n={strict_score['n_active']}, acc={strict_score['accuracy']:.1%}, "
          f"lift={strict_score['lift']:+.1%}, %data={strict_score['pct_data']:.1f}%")
    pd.DataFrame([strict_score]).to_csv(
        os.path.join(args.output_dir, 'composite_strict.csv'), index=False)

    # Combinator: magnitude-weighted
    print(f"\n--- Composite: magnitude-weighted ---")
    consensus_w, weighted_sum = combinator_weighted(pred_matrix)
    w_score = evaluate_combinator(consensus_w, actual_dir, valid_y,
                                   baseline_acc, 'magnitude-weighted')
    print(f"    n={w_score['n_active']}, acc={w_score['accuracy']:.1%}, "
          f"lift={w_score['lift']:+.1%}, %data={w_score['pct_data']:.1f}%")
    # Stratified by |weighted_sum|
    print(f"    Stratified by |weighted_sum|:")
    w_rows = []
    for thr in [0.0, 1.0, 2.0, 3.0, 5.0]:
        m = (np.abs(weighted_sum) > thr) & valid_y & (consensus_w != 0)
        if m.sum() == 0:
            continue
        sub_acc = float((consensus_w[m] == actual_dir[m]).mean())
        row = {'threshold_w': thr, 'n': int(m.sum()),
               'pct_data': float(m.sum() / valid_y.sum() * 100),
               'accuracy': sub_acc, 'lift': sub_acc - baseline_acc}
        w_rows.append(row)
        print(f"      |w|>{thr}: n={row['n']:>5}, acc={row['accuracy']:.1%}, "
              f"lift={row['lift']:+.1%}, %data={row['pct_data']:.1f}%")
    pd.DataFrame(w_rows).to_csv(
        os.path.join(args.output_dir, 'composite_weighted.csv'), index=False)

    # Combinator: confidence-gated majority
    print(f"\n--- Composite: confidence-gated majority (|pred|>{args.conf_threshold} per TF) ---")
    threshold_per_tf = {tf: args.conf_threshold for tf in tf_order}
    consensus_cg = combinator_confgated_majority(pred_matrix, dir_matrix,
                                                   threshold_per_tf, tf_order)
    cg_score = evaluate_combinator(consensus_cg, actual_dir, valid_y,
                                     baseline_acc, f'confgated_majority_{args.conf_threshold}')
    print(f"    n={cg_score['n_active']}, acc={cg_score['accuracy']:.1%}, "
          f"lift={cg_score['lift']:+.1%}, %data={cg_score['pct_data']:.1f}%")
    pd.DataFrame([cg_score]).to_csv(
        os.path.join(args.output_dir, 'composite_confgated.csv'), index=False)

    # Markdown summary
    md_path = os.path.join(args.output_dir, 'summary.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# V2 Composite Directional (L-Aggregator) — "
                f"{pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(f"**Voter TFs:** {args.tfs}\n\n")
        f.write(f"**Common cadence:** `{args.common_tf}` "
                f"({len(common_ts)} bars in test window)\n\n")
        f.write(f"**Baseline (majority class):** {baseline_acc:.1%}\n\n")

        f.write("## Per-TF (own cadence)\n\n")
        f.write(pd.DataFrame(per_tf_rows).to_markdown(index=False))
        f.write("\n\n")

        f.write(f"## Per-TF (projected onto {args.common_tf} cadence)\n\n")
        f.write(pd.DataFrame(per_tf_common_rows).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Composite: majority vote (stratified)\n\n")
        f.write(pd.DataFrame(maj_rows).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Composite: strict-all\n\n")
        f.write(pd.DataFrame([strict_score]).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Composite: magnitude-weighted (stratified by |weighted_sum|)\n\n")
        f.write(pd.DataFrame(w_rows).to_markdown(index=False))
        f.write("\n\n")

        f.write(f"## Composite: confgated-majority (|pred|>{args.conf_threshold})\n\n")
        f.write(pd.DataFrame([cg_score]).to_markdown(index=False))
        f.write("\n")
    print(f"\n  [saved] {md_path}")


if __name__ == '__main__':
    main()
