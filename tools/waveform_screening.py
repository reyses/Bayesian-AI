"""
Waveform Screening Tool
=======================
Standalone analysis module — does NOT modify production code.

Purpose: Determine which of the 16F x 12TF dimensions actually explain
outcome (MFE/MAE) variation. Cause-and-effect screening before building
the regression model.

Usage:
    python tools/waveform_screening.py [--data DATA/ATLAS_1WEEK]

Reads patterns from a trained checkpoint, extracts the full 16x12
hypervolume matrix for each, and runs factor screening against oracle MFE.

Output: tools/screening_report.txt + console summary
"""

import sys, os
import numpy as np
import pickle
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -- Feature names (16D) -----------------------------------------------------
FEATURE_NAMES = [
    'abs_z', 'log1p_vol', 'log1p_mom', 'coherence', 'tf_scale', 'depth',
    'parent_ctx', 'self_adx', 'self_hurst', 'self_dmi_diff',
    'parent_z', 'parent_dmi_diff', 'root_is_roche', 'tf_alignment',
    'self_pid', 'osc_coh'
]

# -- TF labels (12 rows in hypervolume matrix, depth 0=macro -> 11=micro) ----
TF_LABELS = [
    'd0_macro', 'd1', 'd2', 'd3', 'd4', 'd5',
    'd6', 'd7', 'd8', 'd9', 'd10', 'd11_micro'
]


def load_templates(checkpoint_dir='checkpoints'):
    """Load templates from the latest training checkpoint."""
    tmpl_path = os.path.join(checkpoint_dir, 'templates.pkl')
    if not os.path.exists(tmpl_path):
        print(f"ERROR: No templates found at {tmpl_path}")
        print("  Run training first: python -m training.orchestrator --train-only")
        sys.exit(1)

    with open(tmpl_path, 'rb') as f:
        templates = pickle.load(f)

    print(f"Loaded {len(templates)} templates from {tmpl_path}")
    return templates


def extract_matrices(templates):
    """Extract 16x12 hypervolume matrices and oracle MFE for all patterns."""
    from training.fractal_clustering import FractalClusteringEngine

    matrices = []  # Each is (depth, 16) — variable depth
    mfes = []
    maes = []
    meta = []  # (template_id, pattern_index, depth)

    for tmpl in templates:
        for i, p in enumerate(tmpl.patterns):
            mat = FractalClusteringEngine.build_hypervolume_matrix(p)
            if mat is None:
                continue

            oracle = getattr(p, 'oracle_meta', {})
            mfe = oracle.get('mfe', 0.0)
            mae = oracle.get('mae', 0.0)
            if mfe == 0.0 and mae == 0.0:
                continue

            matrices.append(mat)
            mfes.append(mfe)
            maes.append(mae)
            meta.append({
                'tid': tmpl.template_id,
                'idx': i,
                'depth': mat.shape[0] - 1,
                'dmi_diff': getattr(p, 'state', None) and
                            (getattr(p.state, 'dmi_plus', 0) - getattr(p.state, 'dmi_minus', 0)),
            })

    print(f"Extracted {len(matrices)} patterns with oracle data")
    return matrices, np.array(mfes), np.array(maes), meta


def pad_to_fixed_depth(matrices, max_depth=12):
    """Pad variable-depth matrices to fixed (max_depth, 16) with zeros."""
    n = len(matrices)
    padded = np.zeros((n, max_depth, 16))

    for i, mat in enumerate(matrices):
        d = min(mat.shape[0], max_depth)
        padded[i, :d, :] = mat[:d, :]

    return padded  # (n, 12, 16)


def flatten_matrices(padded):
    """Flatten (n, 12, 16) -> (n, 192) with named columns."""
    n = padded.shape[0]
    flat = padded.reshape(n, -1)

    col_names = []
    for d in range(padded.shape[1]):
        tf_lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        for f in range(padded.shape[2]):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            col_names.append(f"{tf_lbl}__{f_lbl}")

    return flat, col_names


def screen_factors(flat, col_names, mfes):
    """
    Cause-and-effect screening: correlate each of 192 dims with MFE.

    Returns sorted list of (col_name, correlation, abs_correlation).
    """
    results = []
    for j, name in enumerate(col_names):
        col = flat[:, j]

        # Skip dead columns (zero variance)
        if np.std(col) < 1e-12:
            results.append((name, 0.0, 0.0))
            continue

        corr = float(np.corrcoef(col, mfes)[0, 1])
        if np.isnan(corr):
            corr = 0.0

        results.append((name, corr, abs(corr)))

    # Sort by absolute correlation (strongest first)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def regression_r2(flat, col_names, mfes, top_k=20):
    """
    Fit OLS on top-K factors and report adj-R2.
    Stepwise: add one factor at a time, track R2 improvement.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    # Get top-K column indices
    screening = screen_factors(flat, col_names, mfes)
    top_names = [s[0] for s in screening[:top_k]]
    top_indices = [col_names.index(n) for n in top_names]

    print(f"\n{'='*70}")
    print(f"  STEPWISE REGRESSION (top {top_k} factors -> MFE)")
    print(f"{'='*70}")
    print(f"  {'Step':>4}  {'Factor':<35} {'R2':>8}  {'dR2':>8}  {'adj-R2':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}  {'-'*8}")

    scaler = StandardScaler()
    prev_r2 = 0.0
    steps = []

    for step, idx in enumerate(top_indices, 1):
        selected = top_indices[:step]
        X = scaler.fit_transform(flat[:, selected])
        reg = LinearRegression().fit(X, mfes)
        r2 = reg.score(X, mfes)

        n, k = X.shape
        adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)
        delta = r2 - prev_r2

        print(f"  {step:>4}  {col_names[idx]:<35} {r2:>8.4f}  {delta:>+8.4f}  {adj_r2:>8.4f}")
        steps.append((col_names[idx], r2, delta, adj_r2))
        prev_r2 = r2

    return steps


def print_screening_report(results, mfes, maes, meta, top_n=30):
    """Print the screening report."""
    print(f"\n{'='*70}")
    print(f"  WAVEFORM SCREENING REPORT")
    print(f"{'='*70}")
    print(f"  Patterns: {len(mfes):,}")
    print(f"  MFE: mean={np.mean(mfes):.2f}, std={np.std(mfes):.2f}")
    print(f"  MAE: mean={np.mean(maes):.2f}, std={np.std(maes):.2f}")

    # Depth distribution
    depths = [m['depth'] for m in meta]
    print(f"  Depths: min={min(depths)}, max={max(depths)}, "
          f"median={int(np.median(depths))}")

    # Top correlations
    print(f"\n  TOP {top_n} FACTORS (correlation with MFE):")
    print(f"  {'Rank':>4}  {'Factor':<35} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")

    for i, (name, corr, abs_corr) in enumerate(results[:top_n], 1):
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

    # Dead factors (zero correlation)
    dead = [r for r in results if r[2] < 0.01]
    print(f"\n  Dead factors (|corr| < 0.01): {len(dead)} / {len(results)}")

    # Group by timeframe: which depth matters most?
    print(f"\n  FACTOR IMPORTANCE BY TIMEFRAME DEPTH:")
    print(f"  {'Depth':<12} {'Mean |corr|':>12}  {'Max |corr|':>12}  {'# active':>10}")
    print(f"  {'-'*12} {'-'*12}  {'-'*12}  {'-'*10}")

    for d in range(12):
        prefix = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        depth_factors = [(n, c, a) for n, c, a in results if n.startswith(prefix + '__')]
        if depth_factors:
            abs_corrs = [a for _, _, a in depth_factors]
            active = sum(1 for a in abs_corrs if a >= 0.01)
            print(f"  {prefix:<12} {np.mean(abs_corrs):>12.4f}  {max(abs_corrs):>12.4f}  {active:>10}")

    # Group by feature: which feature matters most?
    print(f"\n  FACTOR IMPORTANCE BY FEATURE:")
    print(f"  {'Feature':<20} {'Mean |corr|':>12}  {'Max |corr|':>12}  {'Best depth':<12}")
    print(f"  {'-'*20} {'-'*12}  {'-'*12}  {'-'*12}")

    for f_name in FEATURE_NAMES:
        feat_factors = [(n, c, a) for n, c, a in results if n.endswith(f'__{f_name}')]
        if feat_factors:
            abs_corrs = [a for _, _, a in feat_factors]
            best_idx = np.argmax(abs_corrs)
            best_depth = feat_factors[best_idx][0].split('__')[0]
            print(f"  {f_name:<20} {np.mean(abs_corrs):>12.4f}  {max(abs_corrs):>12.4f}  {best_depth:<12}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Waveform factor screening')
    parser.add_argument('--checkpoint', default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top factors to display')
    args = parser.parse_args()

    # 1. Load
    templates = load_templates(args.checkpoint)

    # 2. Extract
    matrices, mfes, maes, meta = extract_matrices(templates)
    if len(matrices) < 50:
        print("ERROR: Too few patterns with oracle data for screening")
        sys.exit(1)

    # 3. Pad + flatten
    padded = pad_to_fixed_depth(matrices, max_depth=12)
    flat, col_names = flatten_matrices(padded)

    # 4. Screen
    results = screen_factors(flat, col_names, mfes)

    # 5. Report
    print_screening_report(results, mfes, maes, meta, top_n=args.top)

    # 6. Stepwise regression
    regression_r2(flat, col_names, mfes, top_k=20)

    # 7. Save
    report_path = os.path.join(os.path.dirname(__file__), 'screening_report.txt')
    with open(report_path, 'w') as f:
        f.write("WAVEFORM SCREENING RESULTS\n")
        f.write(f"Patterns: {len(mfes)}\n\n")
        f.write(f"{'Rank':>4}  {'Factor':<35} {'Corr':>8}  {'|Corr|':>8}\n")
        for i, (name, corr, abs_corr) in enumerate(results, 1):
            f.write(f"{i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}\n")
    print(f"\n  Full report saved: {report_path}")


if __name__ == '__main__':
    main()
