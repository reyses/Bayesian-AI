"""Research screening functions: I-MR features, factor correlation, stepwise regression.

Extracted from tools/standalone_research.py (lines 1459-1660).
"""

import numpy as np

from .data import TF_HIERARCHY, TF_LABELS, FEATURE_NAMES
from .imr import D4


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def pad_to_fixed_depth(matrices, max_depth=12):
    """Pad variable-depth matrices to fixed (max_depth, 16) with zeros."""
    n = len(matrices)
    padded = np.zeros((n, max_depth, 16))
    for i, mat in enumerate(matrices):
        d = min(mat.shape[0], max_depth)
        padded[i, :d, :] = mat[:d, :]
    return padded


def compute_moving_range(padded):
    """Compute I-MR segmentation features from (n, 12, 16) TF state matrix.

    Returns: mr_flat (n, 448), mr_col_names
    """
    n, n_depths, n_feat = padded.shape

    # MR: depth-to-depth differences
    mr = np.diff(padded, axis=1)  # (n, 11, 16)

    # UCL per feature column (D4 from imr.py, SPC n=2 subgroup)
    mr_abs = np.abs(mr)
    mr_bar_global = mr_abs.mean(axis=(0, 1))  # (16,)
    ucl = D4 * mr_bar_global
    ucl_flags = (mr_abs > ucl[None, None, :]).astype(float)

    # Column summaries
    slopes = np.zeros((n, n_feat))
    mr_bar_local = np.zeros((n, n_feat))
    n_breaks = np.zeros((n, n_feat))

    depth_x = np.arange(n_depths, dtype=float)
    depth_x_centered = depth_x - depth_x.mean()
    denom = (depth_x_centered ** 2).sum()

    for f in range(n_feat):
        col_vals = padded[:, :, f]
        slopes[:, f] = (col_vals * depth_x_centered[None, :]).sum(axis=1) / max(denom, 1e-12)
        mr_bar_local[:, f] = mr_abs[:, :, f].mean(axis=1)
        n_breaks[:, f] = ucl_flags[:, :, f].sum(axis=1)

    # Flatten
    mr_flat_parts = []
    mr_col_names = []

    # MR values (11 x 16 = 176)
    mr_flat_parts.append(mr.reshape(n, -1))
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"MR_{d_from}>{d_to}__{f_lbl}")

    # UCL flags (11 x 16 = 176)
    mr_flat_parts.append(ucl_flags.reshape(n, -1))
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"UCL_{d_from}>{d_to}__{f_lbl}")

    # Column summaries (3 x 16 = 48)
    mr_flat_parts.append(slopes)
    for f in range(n_feat):
        mr_col_names.append(f"slope__{FEATURE_NAMES[f]}")
    mr_flat_parts.append(mr_bar_local)
    for f in range(n_feat):
        mr_col_names.append(f"mr_bar__{FEATURE_NAMES[f]}")
    mr_flat_parts.append(n_breaks)
    for f in range(n_feat):
        mr_col_names.append(f"n_breaks__{FEATURE_NAMES[f]}")

    mr_flat = np.hstack(mr_flat_parts)
    print(f"  MR features: {mr_flat.shape[1]} columns "
          f"(176 MR + 176 UCL + 48 summaries)")
    return mr_flat, mr_col_names


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
    """Correlate each column with MFE, return sorted by |corr|."""
    results = []
    for j, name in enumerate(col_names):
        col = flat[:, j]
        if np.std(col) < 1e-12:
            results.append((name, 0.0, 0.0))
            continue
        corr = float(np.corrcoef(col, mfes)[0, 1])
        if np.isnan(corr):
            corr = 0.0
        results.append((name, corr, abs(corr)))
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def regression_r2(flat, col_names, mfes, top_k=20, return_model=False):
    """Stepwise OLS on top-K factors, report adj-R².
    If return_model=True, also returns (model, scaler, top_indices) for the final step."""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

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

    if return_model:
        # Refit final model cleanly for reuse
        final_scaler = StandardScaler()
        X_final = final_scaler.fit_transform(flat[:, top_indices])
        final_model = LinearRegression().fit(X_final, mfes)
        return steps, (final_model, final_scaler, top_indices)

    return steps


def print_screening_report(results, mfes, maes, meta, top_n=30):
    """Print the screening report."""
    print(f"\n{'='*70}")
    print(f"  STANDALONE RESEARCH SCREENING REPORT")
    print(f"{'='*70}")
    print(f"  Data points: {len(mfes):,}")
    print(f"  MFE: mean={np.mean(mfes):.2f}, std={np.std(mfes):.2f}")
    print(f"  MAE: mean={np.mean(maes):.2f}, std={np.std(maes):.2f}")
    print(f"  Win rate (MFE > MAE): {(mfes > maes).mean():.1%}")

    # Top correlations
    print(f"\n  TOP {top_n} FACTORS (correlation with MFE):")
    print(f"  {'Rank':>4}  {'Factor':<35} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results[:top_n], 1):
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

    # Dead factors
    dead = [r for r in results if r[2] < 0.01]
    print(f"\n  Dead factors (|corr| < 0.01): {len(dead)} / {len(results)}")

    # Group by TF depth
    print(f"\n  FACTOR IMPORTANCE BY TIMEFRAME DEPTH:")
    print(f"  {'Depth':<12} {'TF':>6} {'Mean |corr|':>12}  {'Max |corr|':>12}  {'# active':>10}")
    print(f"  {'-'*12} {'-'*6} {'-'*12}  {'-'*12}  {'-'*10}")
    for d in range(12):
        prefix = TF_LABELS[d]
        depth_factors = [(n, c, a) for n, c, a in results if n.startswith(prefix + '__')]
        if depth_factors:
            abs_corrs = [a for _, _, a in depth_factors]
            active = sum(1 for a in abs_corrs if a >= 0.01)
            print(f"  {prefix:<12} {TF_HIERARCHY[d]:>6} {np.mean(abs_corrs):>12.4f}  "
                  f"{max(abs_corrs):>12.4f}  {active:>10}")

    # Group by feature
    print(f"\n  FACTOR IMPORTANCE BY FEATURE:")
    print(f"  {'Feature':<20} {'Mean |corr|':>12}  {'Max |corr|':>12}  {'Best TF':<12}")
    print(f"  {'-'*20} {'-'*12}  {'-'*12}  {'-'*12}")
    for f_name in FEATURE_NAMES:
        feat_factors = [(n, c, a) for n, c, a in results if n.endswith(f'__{f_name}')]
        if feat_factors:
            abs_corrs = [a for _, _, a in feat_factors]
            best_idx = np.argmax(abs_corrs)
            best_depth = feat_factors[best_idx][0].split('__')[0]
            print(f"  {f_name:<20} {np.mean(abs_corrs):>12.4f}  {max(abs_corrs):>12.4f}  "
                  f"{best_depth:<12}")
