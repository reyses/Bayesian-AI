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

import sys, os, io
import numpy as np
import pickle
from pathlib import Path


class _Tee:
    """Write to both stdout and a StringIO buffer simultaneously."""
    def __init__(self, stream, buffer):
        self._stream = stream
        self._buffer = buffer

    def write(self, data):
        self._stream.write(data)
        self._buffer.write(data)

    def flush(self):
        self._stream.flush()

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


def extract_matrices(templates, warmup_days=30, window_days=7):
    """Extract 16x12 hypervolume matrices and oracle MFE for all patterns.

    Temporal windowing:
        warmup_days: skip this many days from the start (indicator cold-start)
        window_days: collect only this many days after warmup (screening sample)
        If warmup_days=0 and window_days=0, use ALL patterns (no filtering).
    """
    from training.fractal_clustering import FractalClusteringEngine

    matrices = []  # Each is (depth, 16) — variable depth
    mfes = []
    maes = []
    meta = []  # (template_id, pattern_index, depth)
    timestamps = []

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

            ts = getattr(p, 'timestamp', 0.0)
            if hasattr(p, 'state') and ts == 0.0:
                ts = getattr(p.state, 'timestamp', 0.0)

            matrices.append(mat)
            mfes.append(mfe)
            maes.append(mae)
            timestamps.append(ts)
            meta.append({
                'tid': tmpl.template_id,
                'idx': i,
                'depth': mat.shape[0] - 1,
                'ts': ts,
                'dmi_diff': getattr(p, 'state', None) and
                            (getattr(p.state, 'dmi_plus', 0) - getattr(p.state, 'dmi_minus', 0)),
            })

    ts_arr = np.array(timestamps)
    print(f"Extracted {len(matrices)} patterns with oracle data")

    # --- Temporal windowing: 1-month warmup + 1-week collection ---
    use_windowing = warmup_days > 0 or window_days > 0
    if use_windowing and len(ts_arr) > 0 and ts_arr.max() > 0:
        t0 = ts_arr[ts_arr > 0].min()
        t_warmup = t0 + warmup_days * 86400
        t_end = t_warmup + window_days * 86400 if window_days > 0 else ts_arr.max() + 1

        from datetime import datetime, timezone
        t0_dt = datetime.fromtimestamp(t0, tz=timezone.utc)
        tw_dt = datetime.fromtimestamp(t_warmup, tz=timezone.utc)
        te_dt = datetime.fromtimestamp(t_end, tz=timezone.utc)
        print(f"  Data range:     {t0_dt:%Y-%m-%d} to {datetime.fromtimestamp(ts_arr.max(), tz=timezone.utc):%Y-%m-%d}")
        print(f"  Warmup cutoff:  {t0_dt:%Y-%m-%d} -> {tw_dt:%Y-%m-%d}  ({warmup_days}d skipped)")
        print(f"  Collection:     {tw_dt:%Y-%m-%d} -> {te_dt:%Y-%m-%d}  ({window_days}d window)")

        mask = (ts_arr >= t_warmup) & (ts_arr < t_end)
        n_before = len(matrices)
        matrices = [matrices[i] for i in range(n_before) if mask[i]]
        mfes = [mfes[i] for i in range(n_before) if mask[i]]
        maes = [maes[i] for i in range(n_before) if mask[i]]
        meta = [meta[i] for i in range(n_before) if mask[i]]

        print(f"  After windowing: {len(matrices)} / {n_before} patterns "
              f"({len(matrices)/n_before*100:.1f}%)")

    return matrices, np.array(mfes), np.array(maes), meta


def pad_to_fixed_depth(matrices, max_depth=12):
    """Pad variable-depth matrices to fixed (max_depth, 16) with zeros."""
    n = len(matrices)
    padded = np.zeros((n, max_depth, 16))

    for i, mat in enumerate(matrices):
        d = min(mat.shape[0], max_depth)
        padded[i, :d, :] = mat[:d, :]

    return padded  # (n, 12, 16)


def compute_moving_range(padded):
    """Compute I-MR segmentation features from (n, 12, 16) hypervolume.

    For each pattern's 12-depth column per feature, compute:
      MR:  depth-to-depth difference (11 values per feature = 176 cols)
      UCL: flag where |MR| exceeds control limit (11 × 16 = 176 cols)
      Column summaries per feature (16 cols each):
        - slope: linear trend across depths
        - mr_bar: mean |MR| (average movement)
        - n_breaks: count of UCL violations (regime complexity)

    Returns: mr_flat (n, 448), mr_col_names
    """
    n, n_depths, n_feat = padded.shape  # (n, 12, 16)

    # --- MR: depth-to-depth differences ---
    mr = np.diff(padded, axis=1)  # (n, 11, 16)

    # --- UCL per feature column (D4=3.267 for n=2 subgroup) ---
    D4 = 3.267
    mr_abs = np.abs(mr)  # (n, 11, 16)
    # UCL = D4 * MR_bar (computed per-feature across all patterns)
    mr_bar_global = mr_abs.mean(axis=(0, 1))  # (16,) global MR_bar per feature
    ucl = D4 * mr_bar_global  # (16,)
    ucl_flags = (mr_abs > ucl[None, None, :]).astype(float)  # (n, 11, 16)

    # --- Column summaries (per feature, across 12 depths) ---
    slopes = np.zeros((n, n_feat))
    mr_bar_local = np.zeros((n, n_feat))
    n_breaks = np.zeros((n, n_feat))

    depth_x = np.arange(n_depths, dtype=float)
    depth_x_centered = depth_x - depth_x.mean()
    denom = (depth_x_centered ** 2).sum()

    for f in range(n_feat):
        col_vals = padded[:, :, f]  # (n, 12)
        # Slope: linear regression across depths
        slopes[:, f] = (col_vals * depth_x_centered[None, :]).sum(axis=1) / max(denom, 1e-12)
        # Local MR bar
        mr_bar_local[:, f] = mr_abs[:, :, f].mean(axis=1)
        # Number of UCL breaks
        n_breaks[:, f] = ucl_flags[:, :, f].sum(axis=1)

    # --- Flatten all MR features ---
    mr_flat_parts = []
    mr_col_names = []

    # 1. MR values (11 × 16 = 176)
    mr_reshaped = mr.reshape(n, -1)
    mr_flat_parts.append(mr_reshaped)
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"MR_{d_from}>{d_to}__{f_lbl}")

    # 2. UCL flags (11 × 16 = 176)
    ucl_reshaped = ucl_flags.reshape(n, -1)
    mr_flat_parts.append(ucl_reshaped)
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"UCL_{d_from}>{d_to}__{f_lbl}")

    # 3. Column summaries (3 × 16 = 48)
    mr_flat_parts.append(slopes)
    for f in range(n_feat):
        f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
        mr_col_names.append(f"slope__{f_lbl}")

    mr_flat_parts.append(mr_bar_local)
    for f in range(n_feat):
        f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
        mr_col_names.append(f"mr_bar__{f_lbl}")

    mr_flat_parts.append(n_breaks)
    for f in range(n_feat):
        f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
        mr_col_names.append(f"n_breaks__{f_lbl}")

    mr_flat = np.hstack(mr_flat_parts)  # (n, 576)
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
    parser.add_argument('--warmup', type=int, default=30,
                        help='Warmup period in days (skip cold-start data)')
    parser.add_argument('--window', type=int, default=7,
                        help='Collection window in days (after warmup)')
    parser.add_argument('--all', action='store_true',
                        help='Use ALL patterns (no warmup/window filtering)')
    args = parser.parse_args()

    warmup = 0 if args.all else args.warmup
    window = 0 if args.all else args.window

    # 1. Load
    templates = load_templates(args.checkpoint)

    # 2. Extract with temporal windowing
    matrices, mfes, maes, meta = extract_matrices(
        templates, warmup_days=warmup, window_days=window
    )
    if len(matrices) < 50:
        print("ERROR: Too few patterns with oracle data for screening")
        sys.exit(1)

    # 3. Pad + flatten
    padded = pad_to_fixed_depth(matrices, max_depth=12)
    flat_i, col_names_i = flatten_matrices(padded)

    # 4. Compute MR segmentation features (Y)
    flat_mr, col_names_mr = compute_moving_range(padded)

    # 5. Combine X (raw I values) + Y (MR segmentation) = Z
    flat_z = np.hstack([flat_i, flat_mr])
    col_names_z = col_names_i + col_names_mr
    print(f"  Combined: {len(col_names_i)} I + {len(col_names_mr)} MR "
          f"= {len(col_names_z)} total features")

    # 6. Screen all three: I-only, MR-only, combined Z
    # --- Start capturing all output to buffer ---
    _report_buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _report_buf)

    print(f"\n{'='*70}")
    print(f"  SCREENING X: Raw I values ({len(col_names_i)} features)")
    print(f"{'='*70}")
    results_i = screen_factors(flat_i, col_names_i, mfes)
    print_screening_report(results_i, mfes, maes, meta, top_n=args.top)
    steps_i = regression_r2(flat_i, col_names_i, mfes, top_k=20)

    print(f"\n{'='*70}")
    print(f"  SCREENING Y: MR Segmentation ({len(col_names_mr)} features)")
    print(f"{'='*70}")
    results_mr = screen_factors(flat_mr, col_names_mr, mfes)
    # Top MR factors only
    print(f"\n  TOP {args.top} MR FACTORS:")
    print(f"  {'Rank':>4}  {'Factor':<40} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*40} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results_mr[:args.top], 1):
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  {name:<40} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")
    steps_mr = regression_r2(flat_mr, col_names_mr, mfes, top_k=20)

    print(f"\n{'='*70}")
    print(f"  SCREENING Z: X + Y Combined ({len(col_names_z)} features)")
    print(f"{'='*70}")
    results_z = screen_factors(flat_z, col_names_z, mfes)
    print(f"\n  TOP {args.top} COMBINED FACTORS:")
    print(f"  {'Rank':>4}  {'Factor':<40} {'Corr':>8}  {'|Corr|':>8}")
    print(f"  {'-'*4}  {'-'*40} {'-'*8}  {'-'*8}")
    for i, (name, corr, abs_corr) in enumerate(results_z[:args.top], 1):
        src = 'I' if name in col_names_i else 'MR'
        bar = '#' * int(abs_corr * 40)
        print(f"  {i:>4}  [{src}] {name:<37} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")
    steps_z = regression_r2(flat_z, col_names_z, mfes, top_k=20)

    # 7. Summary comparison
    r2_i = steps_i[-1][3] if steps_i else 0
    r2_mr = steps_mr[-1][3] if steps_mr else 0
    r2_z = steps_z[-1][3] if steps_z else 0
    print(f"\n{'='*70}")
    print(f"  COMPARISON: adj-R2 @ 20 factors")
    print(f"{'='*70}")
    print(f"  X (I values only):     {r2_i:.4f}")
    print(f"  Y (MR segments only):  {r2_mr:.4f}")
    print(f"  Z (X + Y combined):    {r2_z:.4f}")
    print(f"  Lift from MR:          {r2_z - r2_i:+.4f}")

    # 8. Segmented screening with DIRECTIONAL SPLIT
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    tids = np.array([m['tid'] for m in meta])
    unique_tids = np.unique(tids)

    # --- Directional split: LONG (DMI >= 0) vs SHORT (DMI < 0) ---
    dmi_vals = np.array([m.get('dmi_diff', None) for m in meta], dtype=object)
    # Convert to float, treat None as 0
    dmi_float = np.array([float(d) if d is not None else 0.0 for d in dmi_vals])
    long_mask = dmi_float >= 0
    short_mask = ~long_mask

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    wr_long = float((mfes[long_mask] > maes[long_mask]).mean()) if n_long > 0 else 0
    wr_short = float((mfes[short_mask] > maes[short_mask]).mean()) if n_short > 0 else 0
    mfe_long = float(np.mean(mfes[long_mask])) if n_long > 0 else 0
    mfe_short = float(np.mean(mfes[short_mask])) if n_short > 0 else 0

    print(f"\n{'='*70}")
    print(f"  DIRECTIONAL SPLIT")
    print(f"{'='*70}")
    print(f"  LONG  (DMI >= 0): {n_long:>5} patterns, "
          f"WR={wr_long:.1%}, MFE={mfe_long:+.0f}")
    print(f"  SHORT (DMI <  0): {n_short:>5} patterns, "
          f"WR={wr_short:.1%}, MFE={mfe_short:+.0f}")
    print(f"  Mixed WR:         {float((mfes > maes).mean()):.1%}")

    print(f"\n{'='*70}")
    print(f"  SEGMENTED SCREENING: {len(unique_tids)} templates x 2 directions")
    print(f"{'='*70}")

    # Global stats
    global_mfe_mean = float(np.mean(mfes))
    global_mfe_std = float(np.std(mfes))

    seg_results = []
    for tid in unique_tids:
        for dir_name, dir_mask in [('LONG', long_mask), ('SHORT', short_mask)]:
            seg_mask = (tids == tid) & dir_mask
            n_seg = seg_mask.sum()
            if n_seg < 15:
                continue

            seg_mfes = mfes[seg_mask]
            seg_maes = maes[seg_mask]
            seg_flat = flat_z[seg_mask]

            # Screen within segment
            seg_screening = screen_factors(seg_flat, col_names_z, seg_mfes)
            top1_name, top1_corr, top1_abs = seg_screening[0]

            # Quick R2 with top 5 factors
            top5_names = [s[0] for s in seg_screening[:5]]
            top5_idx = [col_names_z.index(n) for n in top5_names]
            scaler = StandardScaler()
            X = scaler.fit_transform(seg_flat[:, top5_idx])
            reg = LinearRegression().fit(X, seg_mfes)
            r2 = reg.score(X, seg_mfes)
            n, k = X.shape
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)

            # --- MR entry/exit signals ---
            mr_factors = [(nm, c, a) for nm, c, a in seg_screening
                          if nm.startswith('MR_') and a > 0.20]
            entry_signals = [(nm, c) for nm, c, a in mr_factors if c > 0]
            exit_signals = [(nm, c) for nm, c, a in mr_factors if c < 0]

            # --- Cpk / Ppk ---
            seg_mean = float(np.mean(seg_mfes))
            seg_std = float(np.std(seg_mfes))
            cpk = seg_mean / (3 * seg_std) if seg_std > 1e-6 else 0.0
            ppk = seg_mean / (3 * global_mfe_std) if global_mfe_std > 1e-6 else 0.0

            # --- Probability metrics ---
            win_rate = float((seg_mfes > seg_maes).mean())
            p_positive = float((seg_mfes > 0).mean())
            p_good = float((seg_mfes > seg_mean).mean())

            # --- Good vs Bad pattern split ---
            median_mfe = float(np.median(seg_mfes))
            good_mask = seg_mfes >= median_mfe
            bad_mask = ~good_mask

            good_bad_diff = []
            if good_mask.sum() >= 5 and bad_mask.sum() >= 5:
                for j, cname in enumerate(col_names_z):
                    col = seg_flat[:, j]
                    if np.std(col) < 1e-12:
                        continue
                    good_mean = float(np.mean(col[good_mask]))
                    bad_mean = float(np.mean(col[bad_mask]))
                    pooled_std = float(np.std(col))
                    if pooled_std > 1e-12:
                        effect_size = (good_mean - bad_mean) / pooled_std
                        good_bad_diff.append((cname, effect_size, abs(effect_size)))
                good_bad_diff.sort(key=lambda x: x[2], reverse=True)

            seg_results.append({
                'tid': tid, 'dir': dir_name, 'n': n_seg,
                'seg_id': f"{dir_name[0]}_{tid}",
                'mfe_mean': seg_mean, 'mfe_std': seg_std,
                'mae_mean': float(np.mean(seg_maes)),
                'top1': top1_name, 'top1_corr': top1_corr,
                'adj_r2_5': adj_r2,
                'top5': [(s[0], s[1]) for s in seg_screening[:5]],
                'entry_signals': entry_signals[:5],
                'exit_signals': exit_signals[:5],
                'cpk': cpk, 'ppk': ppk,
                'win_rate': win_rate, 'p_positive': p_positive, 'p_good': p_good,
                'good_bad_top5': good_bad_diff[:5],
                'median_mfe': median_mfe,
                'good_mfe_mean': float(np.mean(seg_mfes[good_mask])),
                'bad_mfe_mean': float(np.mean(seg_mfes[bad_mask])),
            })

    # --- Extract dominant context feature for each segment ---
    def _extract_feature(factor_name):
        """Extract the base feature name from a factor like 'MR_d6>d7__self_pid'."""
        parts = factor_name.split('__')
        return parts[-1] if parts else factor_name

    def _extract_depth(factor_name):
        """Extract the depth info from a factor name."""
        if factor_name.startswith('MR_'):
            # MR_d6>d7__self_pid -> d6>d7
            return factor_name.split('__')[0].replace('MR_', '')
        elif factor_name.startswith('UCL_'):
            return factor_name.split('__')[0].replace('UCL_', '')
        elif factor_name.startswith('slope__') or factor_name.startswith('mr_bar__') or factor_name.startswith('n_breaks__'):
            return 'all'
        else:
            # d8__self_pid -> d8
            return factor_name.split('__')[0]

    for s in seg_results:
        # Dominant feature = most common feature in top 5
        top5_features = [_extract_feature(fn) for fn, _ in s['top5']]
        from collections import Counter
        feat_counts = Counter(top5_features)
        s['dominant_feature'] = feat_counts.most_common(1)[0][0]
        s['top1_feature'] = _extract_feature(s['top1'])
        s['top1_depth'] = _extract_depth(s['top1'])
        s['top1_src'] = 'I' if s['top1'] in col_names_i else 'MR'
        # Feature profile: which features appear in top 5
        s['feature_profile'] = list(dict.fromkeys(top5_features))  # unique, ordered

    # --- Group by dominant context feature ---
    from collections import defaultdict
    context_groups = defaultdict(list)
    for s in seg_results:
        context_groups[s['dominant_feature']].append(s)

    # Sort groups by Cpk (process capability)
    group_order = []
    for feat, segs in context_groups.items():
        total_n = sum(s['n'] for s in segs)
        wtd_r2 = np.average([s['adj_r2_5'] for s in segs], weights=[s['n'] for s in segs])
        wtd_cpk = np.average([s['cpk'] for s in segs], weights=[s['n'] for s in segs])
        wtd_ppk = np.average([s['ppk'] for s in segs], weights=[s['n'] for s in segs])
        avg_mfe = np.average([s['mfe_mean'] for s in segs], weights=[s['n'] for s in segs])
        wtd_wr = np.average([s['win_rate'] for s in segs], weights=[s['n'] for s in segs])
        group_order.append((feat, segs, total_n, wtd_r2, avg_mfe, wtd_cpk, wtd_ppk, wtd_wr))
    group_order.sort(key=lambda x: x[5], reverse=True)  # sort by Cpk

    # === FISSION TABLE: sort by P(success), classify KEEP / SPLIT / DROP ===
    all_sorted = sorted(seg_results, key=lambda x: x['win_rate'], reverse=True)

    # Signal-to-noise: mean/std (higher = more consistent)
    for s in all_sorted:
        s['snr'] = s['mfe_mean'] / s['mfe_std'] if s['mfe_std'] > 1e-6 else 0.0

    print(f"\n{'='*70}")
    print(f"  MODEL FISSION: Segment x Direction (sorted by P(success))")
    print(f"  P(win) = MFE > MAE.  SNR = mean/std.  Action = KEEP/SPLIT/DROP")
    print(f"{'='*70}")
    print(f"  {'Seg ID':<10} {'Ctx':<12} {'N':>4} {'P(win)':>7} {'P(>0)':>6} "
          f"{'SNR':>5} {'R2':>5} {'MFE':>6} {'MAE':>5} {'Action':<7}")
    print(f"  {'-'*10} {'-'*12} {'-'*4} {'-'*7} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*6} {'-'*5} {'-'*7}")

    keep_segs, split_segs, drop_segs = [], [], []
    for s in all_sorted:
        if s['win_rate'] >= 0.65 and s['snr'] >= 0.5:
            action = 'KEEP'
            keep_segs.append(s)
        elif s['win_rate'] >= 0.50:
            action = 'SPLIT'
            split_segs.append(s)
        else:
            action = 'DROP'
            drop_segs.append(s)

        wr_bar = '#' * int(s['win_rate'] * 20)
        print(f"  {s['seg_id']:<10} {s['dominant_feature']:<12} {s['n']:>4} "
              f"{s['win_rate']:>7.1%} {s['p_positive']:>6.0%} "
              f"{s['snr']:>5.2f} {s['adj_r2_5']:>5.2f} "
              f"{s['mfe_mean']:>+6.0f} {s['mae_mean']:>5.0f} "
              f"{action:<7} {wr_bar}")

    # === KEEP segments: what makes them work ===
    if keep_segs:
        keep_n = sum(s['n'] for s in keep_segs)
        keep_wr = np.average([s['win_rate'] for s in keep_segs],
                             weights=[s['n'] for s in keep_segs])
        keep_mfe = np.average([s['mfe_mean'] for s in keep_segs],
                              weights=[s['n'] for s in keep_segs])
        print(f"\n  KEEP ({len(keep_segs)} segments, {keep_n} patterns, "
              f"WR={keep_wr:.1%}, avg MFE={keep_mfe:+.0f}):")
        for s in keep_segs:
            # Entry/exit summary
            entry_str = ', '.join(
                f"{nm.split('__')[-1]}@{nm.split('__')[0].replace('MR_','')}"
                for nm, c in s['entry_signals'][:2]) or 'none'
            exit_str = ', '.join(
                f"{nm.split('__')[-1]}@{nm.split('__')[0].replace('MR_','')}"
                for nm, c in s['exit_signals'][:2]) or 'none'
            print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                  f"WR={s['win_rate']:.0%} MFE={s['mfe_mean']:+.0f}")
            print(f"      Entry: {entry_str}")
            print(f"      Exit:  {exit_str}")
            if s['good_bad_top5']:
                top_diff = s['good_bad_top5'][0]
                src = 'I' if top_diff[0] in col_names_i else 'MR'
                dir_str = 'higher' if top_diff[1] > 0 else 'lower'
                print(f"      Good vs Bad: {top_diff[0]} {dir_str} in winners (d={top_diff[1]:+.2f})")

    # === SPLIT segments: have signal but noisy, need finer segmentation ===
    if split_segs:
        split_n = sum(s['n'] for s in split_segs)
        split_wr = np.average([s['win_rate'] for s in split_segs],
                              weights=[s['n'] for s in split_segs])
        print(f"\n  SPLIT ({len(split_segs)} segments, {split_n} patterns, "
              f"WR={split_wr:.1%}) -- signal exists but noisy, need finer cuts:")
        for s in split_segs:
            if s['good_bad_top5']:
                split_feature = s['good_bad_top5'][0][0]
                print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                      f"WR={s['win_rate']:.0%} -- split on: {split_feature}")
            else:
                print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                      f"WR={s['win_rate']:.0%}")

    # === DROP segments: noise generators ===
    if drop_segs:
        drop_n = sum(s['n'] for s in drop_segs)
        drop_mfe = np.average([s['mfe_mean'] for s in drop_segs],
                              weights=[s['n'] for s in drop_segs])
        print(f"\n  DROP ({len(drop_segs)} segments, {drop_n} patterns, "
              f"avg MFE={drop_mfe:+.0f}) -- net noise, remove from model:")
        for s in drop_segs:
            print(f"    {s['seg_id']} [{s['dominant_feature']}] n={s['n']} "
                  f"WR={s['win_rate']:.0%} MFE={s['mfe_mean']:+.0f} "
                  f"MAE={s['mae_mean']:.0f}")

    # === EXPORT GATE CONFIG for orchestrator forward pass ===
    _fission_map = {}
    for s in keep_segs:
        _dir = 'long' if s['seg_id'].startswith('L_') else 'short'
        _tid = s['seg_id'].split('_', 1)[1]
        _fission_map[f"{_tid}_{_dir}"] = 'KEEP'
    for s in split_segs:
        _dir = 'long' if s['seg_id'].startswith('L_') else 'short'
        _tid = s['seg_id'].split('_', 1)[1]
        _fission_map[f"{_tid}_{_dir}"] = 'SPLIT'
    _gate_config = {
        'fission_map': _fission_map,
        'good_hours_utc': [0, 5, 17, 18, 19, 20],
        'default_class': 'DROP',
    }
    _ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'checkpoints')
    os.makedirs(_ckpt_dir, exist_ok=True)
    _gate_path = os.path.join(_ckpt_dir, 'screening_gates.json')
    import json as _json
    with open(_gate_path, 'w') as _gf:
        _json.dump(_gate_config, _gf, indent=2)
    print(f"\n  >> Exported screening gates to {_gate_path}")
    print(f"     KEEP: {sum(1 for v in _fission_map.values() if v == 'KEEP')}, "
          f"SPLIT: {sum(1 for v in _fission_map.values() if v == 'SPLIT')}, "
          f"hours: {_gate_config['good_hours_utc']}")

    # === WHAT-IF: model with only KEEP segments ===
    total_n = sum(s['n'] for s in seg_results)
    total_wr = np.average([s['win_rate'] for s in seg_results],
                          weights=[s['n'] for s in seg_results])
    total_mfe = np.average([s['mfe_mean'] for s in seg_results],
                           weights=[s['n'] for s in seg_results])

    print(f"\n{'='*70}")
    print(f"  WHAT-IF: Fission Impact")
    print(f"{'='*70}")
    print(f"  CURRENT (all segments):")
    print(f"    Segments: {len(seg_results)}, Patterns: {total_n}, "
          f"WR: {total_wr:.1%}, MFE: {total_mfe:+.0f}")

    if keep_segs:
        keep_total_n = sum(s['n'] for s in keep_segs)
        keep_total_wr = np.average([s['win_rate'] for s in keep_segs],
                                   weights=[s['n'] for s in keep_segs])
        keep_total_mfe = np.average([s['mfe_mean'] for s in keep_segs],
                                    weights=[s['n'] for s in keep_segs])
        print(f"  KEEP ONLY:")
        print(f"    Segments: {len(keep_segs)}, Patterns: {keep_total_n}, "
              f"WR: {keep_total_wr:.1%}, MFE: {keep_total_mfe:+.0f}")
        print(f"    Dropped: {total_n - keep_total_n} patterns "
              f"({(total_n - keep_total_n)/total_n:.0%} of volume)")
        print(f"    WR lift: {keep_total_wr - total_wr:+.1%}")

    if keep_segs or split_segs:
        ks = keep_segs + split_segs
        ks_n = sum(s['n'] for s in ks)
        ks_wr = np.average([s['win_rate'] for s in ks], weights=[s['n'] for s in ks])
        ks_mfe = np.average([s['mfe_mean'] for s in ks], weights=[s['n'] for s in ks])
        print(f"  KEEP + SPLIT (before refining splits):")
        print(f"    Segments: {len(ks)}, Patterns: {ks_n}, "
              f"WR: {ks_wr:.1%}, MFE: {ks_mfe:+.0f}")

    # Store for save section
    weighted_seg_r2 = np.average(
        [s['adj_r2_5'] for s in seg_results],
        weights=[s['n'] for s in seg_results]
    )

    # === PID DRILL-DOWN: I-MR through directional split ===
    pid_idx = FEATURE_NAMES.index('self_pid')  # 14

    print(f"\n{'='*70}")
    print(f"  PID DRILL-DOWN: I-MR x Direction")
    print(f"{'='*70}")

    # 1. PID I-chart: raw values at each depth, LONG vs SHORT
    print(f"\n  PID I-CHART (mean value at each depth):")
    print(f"  {'Depth':<12} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'r(MFE)L':>9} {'r(MFE)S':>9}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    for d in range(12):
        pid_col = padded[:, d, pid_idx]
        l_mean = float(np.mean(pid_col[long_mask]))
        s_mean = float(np.mean(pid_col[short_mask]))
        corr_l = float(np.corrcoef(pid_col[long_mask], mfes[long_mask])[0, 1]) \
            if long_mask.sum() > 10 and np.std(pid_col[long_mask]) > 1e-12 else 0.0
        corr_s = float(np.corrcoef(pid_col[short_mask], mfes[short_mask])[0, 1]) \
            if short_mask.sum() > 10 and np.std(pid_col[short_mask]) > 1e-12 else 0.0
        if np.isnan(corr_l): corr_l = 0.0
        if np.isnan(corr_s): corr_s = 0.0
        lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        print(f"  {lbl:<12} {l_mean:>+8.2f} {s_mean:>+8.2f} {l_mean - s_mean:>+8.2f} "
              f"{corr_l:>+9.4f} {corr_s:>+9.4f}")

    # 2. PID MR: depth-to-depth gradients, LONG vs SHORT
    mr_pid = np.diff(padded[:, :, pid_idx], axis=1)  # (n, 11)

    print(f"\n  PID MR (depth-to-depth gradient):")
    print(f"  {'Transition':<16} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'r(MFE)L':>9} {'r(MFE)S':>9}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    pid_mr_key_transitions = []
    for d in range(11):
        mr_col = mr_pid[:, d]
        l_mean = float(np.mean(mr_col[long_mask]))
        s_mean = float(np.mean(mr_col[short_mask]))
        corr_l = float(np.corrcoef(mr_col[long_mask], mfes[long_mask])[0, 1]) \
            if long_mask.sum() > 10 and np.std(mr_col[long_mask]) > 1e-12 else 0.0
        corr_s = float(np.corrcoef(mr_col[short_mask], mfes[short_mask])[0, 1]) \
            if short_mask.sum() > 10 and np.std(mr_col[short_mask]) > 1e-12 else 0.0
        if np.isnan(corr_l): corr_l = 0.0
        if np.isnan(corr_s): corr_s = 0.0
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        tag = ''
        if abs(corr_l) > 0.15 or abs(corr_s) > 0.15:
            tag = ' ***'
            pid_mr_key_transitions.append((f"{d_from}>{d_to}", corr_l, corr_s))
        print(f"  {d_from}>{d_to:<9} {l_mean:>+8.3f} {s_mean:>+8.3f} "
              f"{l_mean - s_mean:>+8.3f} {corr_l:>+9.4f} {corr_s:>+9.4f}{tag}")

    # 3. PID UCL breaks by direction
    D4 = 3.267
    pid_mr_abs = np.abs(mr_pid)
    pid_mr_bar = float(pid_mr_abs.mean())
    pid_ucl = D4 * pid_mr_bar
    pid_ucl_breaks = (pid_mr_abs > pid_ucl).astype(float)

    print(f"\n  PID UCL BREAKS (% with control limit violation, UCL={pid_ucl:.3f}):")
    print(f"  {'Transition':<16} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'WR|brk L':>9} {'WR|brk S':>9}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    for d in range(11):
        brk = pid_ucl_breaks[:, d]
        l_pct = float(brk[long_mask].mean()) * 100
        s_pct = float(brk[short_mask].mean()) * 100
        # WR when UCL breaks (conditional)
        l_brk_mask = long_mask & (brk > 0.5)
        s_brk_mask = short_mask & (brk > 0.5)
        wr_l_brk = float((mfes[l_brk_mask] > maes[l_brk_mask]).mean()) * 100 \
            if l_brk_mask.sum() > 5 else float('nan')
        wr_s_brk = float((mfes[s_brk_mask] > maes[s_brk_mask]).mean()) * 100 \
            if s_brk_mask.sum() > 5 else float('nan')
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        wr_l_str = f"{wr_l_brk:>7.1f}%" if not np.isnan(wr_l_brk) else "    n/a "
        wr_s_str = f"{wr_s_brk:>7.1f}%" if not np.isnan(wr_s_brk) else "    n/a "
        print(f"  {d_from}>{d_to:<9} {l_pct:>7.1f}% {s_pct:>7.1f}% "
              f"{l_pct - s_pct:>+7.1f}% {wr_l_str} {wr_s_str}")

    # 4. PID profile by fission class (KEEP vs SPLIT vs DROP)
    print(f"\n  PID PROFILE BY FISSION CLASS:")
    for label, seg_list in [('KEEP', keep_segs), ('SPLIT', split_segs), ('DROP', drop_segs)]:
        if not seg_list:
            continue
        class_mask = np.zeros(len(mfes), dtype=bool)
        for s in seg_list:
            seg_m = (tids == s['tid']) & (long_mask if s['dir'] == 'LONG' else short_mask)
            class_mask |= seg_m
        if class_mask.sum() < 10:
            continue
        pid_vals = padded[class_mask, :, pid_idx]  # (n_class, 12)
        pid_mfes_class = mfes[class_mask]
        pid_maes_class = maes[class_mask]
        class_wr = float((pid_mfes_class > pid_maes_class).mean())

        print(f"\n  {label} ({class_mask.sum()} patterns, WR={class_wr:.1%}):")
        print(f"    {'Depth':<12} {'Mean PID':>10} {'Std':>8} {'r(MFE)':>8}")
        print(f"    {'-'*12} {'-'*10} {'-'*8} {'-'*8}")
        for d in range(12):
            col = pid_vals[:, d]
            corr = float(np.corrcoef(col, pid_mfes_class)[0, 1]) \
                if np.std(col) > 1e-12 else 0.0
            if np.isnan(corr): corr = 0.0
            lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
            print(f"    {lbl:<12} {np.mean(col):>+10.3f} {np.std(col):>8.3f} {corr:>+8.4f}")

        # PID MR for this class
        pid_mr_class = np.diff(pid_vals, axis=1)  # (n_class, 11)
        print(f"    MR transitions:")
        print(f"    {'Transition':<16} {'Mean MR':>10} {'r(MFE)':>8}")
        print(f"    {'-'*16} {'-'*10} {'-'*8}")
        for d in range(11):
            mr_col = pid_mr_class[:, d]
            corr = float(np.corrcoef(mr_col, pid_mfes_class)[0, 1]) \
                if np.std(mr_col) > 1e-12 else 0.0
            if np.isnan(corr): corr = 0.0
            d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
            d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
            tag = ' ***' if abs(corr) > 0.15 else ''
            print(f"    {d_from}>{d_to:<9} {np.mean(mr_col):>+10.3f} {corr:>+8.4f}{tag}")

    # 5. PID as directional confirmation: does PID sign match DMI direction?
    print(f"\n  PID x DIRECTION CONFIRMATION:")
    print(f"  (Does PID sign at each depth agree with DMI direction?)")
    print(f"  {'Depth':<12} {'Agree%':>8} {'WR|agree':>10} {'WR|disagr':>10} {'Lift':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for d in range(12):
        pid_col = padded[:, d, pid_idx]
        # Agreement: LONG + positive PID, or SHORT + negative PID
        agree = (long_mask & (pid_col > 0)) | (short_mask & (pid_col < 0))
        disagree = ~agree & ((pid_col != 0) | True)  # all non-agree
        n_agree = agree.sum()
        n_disagree = disagree.sum()
        agree_pct = float(n_agree) / len(mfes) * 100
        wr_agree = float((mfes[agree] > maes[agree]).mean()) if n_agree > 10 else float('nan')
        wr_disagree = float((mfes[disagree] > maes[disagree]).mean()) if n_disagree > 10 else float('nan')
        lift = wr_agree - wr_disagree if not (np.isnan(wr_agree) or np.isnan(wr_disagree)) else 0
        lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        wr_a_str = f"{wr_agree:.1%}" if not np.isnan(wr_agree) else "n/a"
        wr_d_str = f"{wr_disagree:.1%}" if not np.isnan(wr_disagree) else "n/a"
        print(f"  {lbl:<12} {agree_pct:>7.1f}% {wr_a_str:>10} {wr_d_str:>10} {lift:>+7.1%}")

    # === TEMPORAL SPECIAL CAUSE: Market sessions, day-of-week, month position ===
    from datetime import datetime, timezone

    ts_arr = np.array([m['ts'] for m in meta])
    valid_ts = ts_arr > 0
    n_valid = valid_ts.sum()

    print(f"\n{'='*70}")
    print(f"  TEMPORAL SPECIAL CAUSE ANALYSIS")
    print(f"  (Patterns with valid timestamps: {n_valid} / {len(meta)})")
    print(f"{'='*70}")

    if n_valid > 50:
        # Convert to datetime objects for valid timestamps
        dts = np.array([
            datetime.fromtimestamp(t, tz=timezone.utc) if t > 0 else None
            for t in ts_arr
        ])
        hours_utc = np.array([dt.hour if dt else -1 for dt in dts])
        dow = np.array([dt.weekday() if dt else -1 for dt in dts])  # 0=Mon
        dom = np.array([dt.day if dt else -1 for dt in dts])  # 1-31

        # --- Market Sessions (ES/MES, times in UTC) ---
        # Asia/overnight: 22:00-08:00 UTC (CME Globex evening + Asia)
        # Europe:         08:00-14:30 UTC (London + Frankfurt)
        # US RTH:         14:30-21:00 UTC (NYSE/CME regular hours)
        # US close:       21:00-22:00 UTC (settlement)

        def _session(h):
            if h >= 22 or h < 8:
                return 'ASIA'
            elif h < 14:
                return 'EUROPE'
            elif h < 21:
                return 'US_RTH'
            else:
                return 'US_CLOSE'

        sessions = np.array([_session(h) if h >= 0 else 'UNK' for h in hours_utc])

        print(f"\n  1. MARKET SESSION:")
        print(f"  {'Session':<12} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'PID_d7':>8}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

        for sess in ['ASIA', 'EUROPE', 'US_RTH', 'US_CLOSE']:
            smask = (sessions == sess) & valid_ts
            n_s = smask.sum()
            if n_s < 10:
                continue
            wr = float((mfes[smask] > maes[smask]).mean())
            mfe_m = float(np.mean(mfes[smask]))
            mae_m = float(np.mean(maes[smask]))
            # WR by direction
            sl = smask & long_mask
            ss = smask & short_mask
            wr_l = float((mfes[sl] > maes[sl]).mean()) if sl.sum() > 5 else float('nan')
            wr_s = float((mfes[ss] > maes[ss]).mean()) if ss.sum() > 5 else float('nan')
            pid_d7 = float(np.mean(padded[smask, 7, pid_idx]))
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            print(f"  {sess:<12} {n_s:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {pid_d7:>+8.2f}")

        # --- Hourly breakdown for finer granularity ---
        print(f"\n  2. HOURLY BREAKDOWN (UTC):")
        print(f"  {'Hour':>4} {'Session':<10} {'N':>5} {'WR':>7} {'MFE':>7} {'WR_L':>7} {'WR_S':>7}")
        print(f"  {'-'*4} {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

        for h in range(24):
            hmask = (hours_utc == h) & valid_ts
            n_h = hmask.sum()
            if n_h < 10:
                continue
            wr = float((mfes[hmask] > maes[hmask]).mean())
            mfe_m = float(np.mean(mfes[hmask]))
            hl = hmask & long_mask
            hs = hmask & short_mask
            wr_l = float((mfes[hl] > maes[hl]).mean()) if hl.sum() > 5 else float('nan')
            wr_s = float((mfes[hs] > maes[hs]).mean()) if hs.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            bar = '#' * int(wr * 20)
            print(f"  {h:>4} {_session(h):<10} {n_h:>5} {wr:>7.1%} {mfe_m:>+7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7}  {bar}")

        # --- Day of Week ---
        dow_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

        print(f"\n  3. DAY OF WEEK:")
        print(f"  {'Day':<5} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

        # Build KEEP mask for cross-reference
        keep_mask_all = np.zeros(len(mfes), dtype=bool)
        for s in keep_segs:
            seg_m = (tids == s['tid']) & (long_mask if s['dir'] == 'LONG' else short_mask)
            keep_mask_all |= seg_m

        for d_idx in range(7):
            dmask = (dow == d_idx) & valid_ts
            n_d = dmask.sum()
            if n_d < 10:
                continue
            wr = float((mfes[dmask] > maes[dmask]).mean())
            mfe_m = float(np.mean(mfes[dmask]))
            mae_m = float(np.mean(maes[dmask]))
            dl = dmask & long_mask
            ds = dmask & short_mask
            dk = dmask & keep_mask_all
            wr_l = float((mfes[dl] > maes[dl]).mean()) if dl.sum() > 5 else float('nan')
            wr_s = float((mfes[ds] > maes[ds]).mean()) if ds.sum() > 5 else float('nan')
            wr_k = float((mfes[dk] > maes[dk]).mean()) if dk.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            wr_k_str = f"{wr_k:.1%}" if not np.isnan(wr_k) else "n/a"
            print(f"  {dow_names[d_idx]:<5} {n_d:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {wr_k_str:>8}")

        # --- Month Position (first week / mid / last week) ---
        def _month_pos(day):
            if day <= 7:
                return 'FIRST_WK'
            elif day >= 23:
                return 'LAST_WK'
            else:
                return 'MID'

        month_pos = np.array([_month_pos(d) if d > 0 else 'UNK' for d in dom])

        print(f"\n  4. MONTH POSITION:")
        print(f"  {'Period':<10} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*10} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

        for pos in ['FIRST_WK', 'MID', 'LAST_WK']:
            pmask = (month_pos == pos) & valid_ts
            n_p = pmask.sum()
            if n_p < 10:
                continue
            wr = float((mfes[pmask] > maes[pmask]).mean())
            mfe_m = float(np.mean(mfes[pmask]))
            mae_m = float(np.mean(maes[pmask]))
            pl = pmask & long_mask
            ps = pmask & short_mask
            pk = pmask & keep_mask_all
            wr_l = float((mfes[pl] > maes[pl]).mean()) if pl.sum() > 5 else float('nan')
            wr_s = float((mfes[ps] > maes[ps]).mean()) if ps.sum() > 5 else float('nan')
            wr_k = float((mfes[pk] > maes[pk]).mean()) if pk.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            wr_k_str = f"{wr_k:.1%}" if not np.isnan(wr_k) else "n/a"
            print(f"  {pos:<10} {n_p:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {wr_k_str:>8}")

        # --- Open/Close proximity (first/last 30min of each session) ---
        print(f"\n  5. SESSION OPEN/CLOSE (first & last 30min):")
        print(f"  {'Marker':<20} {'N':>5} {'WR':>7} {'MFE':>7} {'vs Sess':>8}")
        print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*7} {'-'*8}")

        minutes_utc = np.array([
            dt.hour * 60 + dt.minute if dt else -1 for dt in dts
        ])

        markers = [
            ('ASIA open',       22 * 60, 22 * 60 + 30, 'ASIA'),
            ('ASIA close',      7 * 60 + 30, 8 * 60, 'ASIA'),
            ('EUROPE open',     8 * 60, 8 * 60 + 30, 'EUROPE'),
            ('EUROPE close',    14 * 60, 14 * 60 + 30, 'EUROPE'),
            ('US RTH open',     14 * 60 + 30, 15 * 60, 'US_RTH'),
            ('US RTH close',    20 * 60 + 30, 21 * 60, 'US_RTH'),
        ]

        for label, t_start, t_end, parent_sess in markers:
            if t_start < t_end:
                mmask = (minutes_utc >= t_start) & (minutes_utc < t_end) & valid_ts
            else:
                mmask = ((minutes_utc >= t_start) | (minutes_utc < t_end)) & valid_ts
            n_m = mmask.sum()
            if n_m < 5:
                continue
            wr = float((mfes[mmask] > maes[mmask]).mean())
            mfe_m = float(np.mean(mfes[mmask]))
            # Compare vs full session
            sess_mask = (sessions == parent_sess) & valid_ts
            sess_wr = float((mfes[sess_mask] > maes[sess_mask]).mean()) if sess_mask.sum() > 10 else wr
            delta = wr - sess_wr
            print(f"  {label:<20} {n_m:>5} {wr:>7.1%} {mfe_m:>+7.0f} {delta:>+7.1%}")

        # --- Weekly position (Mon-Tue = start, Wed = mid, Thu-Fri = end) ---
        print(f"\n  6. WEEK POSITION:")
        print(f"  {'Period':<12} {'N':>5} {'WR':>7} {'MFE':>7} {'WR_L':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*12} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

        week_pos = {
            'START(M-T)': (dow == 0) | (dow == 1),
            'MID(W)':     dow == 2,
            'END(T-F)':   (dow == 3) | (dow == 4),
        }
        for wlabel, wmask_base in week_pos.items():
            wmask = wmask_base & valid_ts
            n_w = wmask.sum()
            if n_w < 10:
                continue
            wr = float((mfes[wmask] > maes[wmask]).mean())
            mfe_m = float(np.mean(mfes[wmask]))
            wl = wmask & long_mask
            wk = wmask & keep_mask_all
            wr_l = float((mfes[wl] > maes[wl]).mean()) if wl.sum() > 5 else float('nan')
            wr_k = float((mfes[wk] > maes[wk]).mean()) if wk.sum() > 5 else float('nan')
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_k_str = f"{wr_k:.1%}" if not np.isnan(wr_k) else "n/a"
            print(f"  {wlabel:<12} {n_w:>5} {wr:>7.1%} {mfe_m:>+7.0f} "
                  f"{wr_l_str:>7} {wr_k_str:>8}")

        # --- 7. MR UCL breaks x Temporal: do big jumps cluster at specific times? ---
        # Use the full MR UCL flags (all features, all depths)
        # ucl_flags shape: from compute_moving_range -> (n, 11, 16) equivalent
        # We already have flat_mr which includes UCL columns
        # Simpler: count total UCL breaks per pattern across all MR features
        mr_ucl_start = 11 * 16  # UCL columns start after MR columns in flat_mr
        mr_ucl_end = mr_ucl_start + 11 * 16
        ucl_per_pattern = flat_mr[:, mr_ucl_start:mr_ucl_end].sum(axis=1)  # total breaks per pattern
        has_ucl = ucl_per_pattern > 0

        print(f"\n  7. MR UCL BREAKS x TEMPORAL:")
        print(f"  (Where do control limit violations cluster in time?)")
        print(f"  Patterns with any UCL break: {has_ucl.sum()} / {len(mfes)} "
              f"({has_ucl.mean():.1%})")

        # UCL breaks by session
        print(f"\n  UCL breaks by SESSION:")
        print(f"  {'Session':<12} {'N_brk':>6} {'%brk':>6} {'WR|brk':>8} {'WR|no':>8} {'Lift':>7}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")

        for sess in ['ASIA', 'EUROPE', 'US_RTH', 'US_CLOSE']:
            smask = (sessions == sess) & valid_ts
            n_s = smask.sum()
            if n_s < 10:
                continue
            brk_in_sess = smask & has_ucl
            no_brk_in_sess = smask & ~has_ucl
            n_brk = brk_in_sess.sum()
            pct_brk = n_brk / max(n_s, 1)
            wr_brk = float((mfes[brk_in_sess] > maes[brk_in_sess]).mean()) if n_brk > 5 else float('nan')
            wr_no = float((mfes[no_brk_in_sess] > maes[no_brk_in_sess]).mean()) \
                if no_brk_in_sess.sum() > 5 else float('nan')
            lift = wr_brk - wr_no if not (np.isnan(wr_brk) or np.isnan(wr_no)) else 0
            wr_b_str = f"{wr_brk:.1%}" if not np.isnan(wr_brk) else "n/a"
            wr_n_str = f"{wr_no:.1%}" if not np.isnan(wr_no) else "n/a"
            print(f"  {sess:<12} {n_brk:>6} {pct_brk:>5.1%} {wr_b_str:>8} {wr_n_str:>8} {lift:>+6.1%}")

        # UCL breaks by hour
        print(f"\n  UCL breaks by HOUR (top hours with most breaks):")
        print(f"  {'Hour':>4} {'Session':<10} {'N_brk':>6} {'%brk':>6} {'WR|brk':>8} {'Lift':>7}")
        print(f"  {'-'*4} {'-'*10} {'-'*6} {'-'*6} {'-'*8} {'-'*7}")

        hour_data = []
        for h in range(24):
            hmask = (hours_utc == h) & valid_ts
            n_h = hmask.sum()
            if n_h < 10:
                continue
            brk_h = hmask & has_ucl
            no_brk_h = hmask & ~has_ucl
            n_brk = brk_h.sum()
            pct_brk = n_brk / max(n_h, 1)
            wr_brk = float((mfes[brk_h] > maes[brk_h]).mean()) if n_brk > 5 else float('nan')
            wr_no = float((mfes[no_brk_h] > maes[no_brk_h]).mean()) if no_brk_h.sum() > 5 else float('nan')
            lift = wr_brk - wr_no if not (np.isnan(wr_brk) or np.isnan(wr_no)) else 0
            hour_data.append((h, n_brk, pct_brk, wr_brk, lift))

        # Sort by break count
        hour_data.sort(key=lambda x: x[1], reverse=True)
        for h, n_brk, pct_brk, wr_brk, lift in hour_data[:10]:
            wr_b_str = f"{wr_brk:.1%}" if not np.isnan(wr_brk) else "n/a"
            print(f"  {h:>4} {_session(h):<10} {n_brk:>6} {pct_brk:>5.1%} "
                  f"{wr_b_str:>8} {lift:>+6.1%}")

        # UCL breaks by day of week
        print(f"\n  UCL breaks by DAY OF WEEK:")
        print(f"  {'Day':<5} {'N_brk':>6} {'%brk':>6} {'WR|brk':>8} {'WR|no':>8} {'Lift':>7}")
        print(f"  {'-'*5} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*7}")

        for d_idx in range(7):
            dmask = (dow == d_idx) & valid_ts
            n_d = dmask.sum()
            if n_d < 10:
                continue
            brk_d = dmask & has_ucl
            no_brk_d = dmask & ~has_ucl
            n_brk = brk_d.sum()
            pct_brk = n_brk / max(n_d, 1)
            wr_brk = float((mfes[brk_d] > maes[brk_d]).mean()) if n_brk > 5 else float('nan')
            wr_no = float((mfes[no_brk_d] > maes[no_brk_d]).mean()) if no_brk_d.sum() > 5 else float('nan')
            lift = wr_brk - wr_no if not (np.isnan(wr_brk) or np.isnan(wr_no)) else 0
            wr_b_str = f"{wr_brk:.1%}" if not np.isnan(wr_brk) else "n/a"
            wr_n_str = f"{wr_no:.1%}" if not np.isnan(wr_no) else "n/a"
            print(f"  {dow_names[d_idx]:<5} {n_brk:>6} {pct_brk:>5.1%} "
                  f"{wr_b_str:>8} {wr_n_str:>8} {lift:>+6.1%}")

        # Which specific MR transitions break most at each session?
        print(f"\n  TOP MR BREAKS per SESSION (which features spike when):")
        mr_transitions = []
        for d in range(11):
            d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
            d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
            for f_i in range(16):
                f_lbl = FEATURE_NAMES[f_i] if f_i < len(FEATURE_NAMES) else f'f{f_i}'
                col_idx = mr_ucl_start + d * 16 + f_i
                mr_transitions.append((f"{d_from}>{d_to}__{f_lbl}", col_idx))

        for sess in ['ASIA', 'EUROPE', 'US_RTH']:
            smask = (sessions == sess) & valid_ts
            if smask.sum() < 20:
                continue
            # Count breaks per MR transition in this session
            break_counts = []
            for tr_name, col_idx in mr_transitions:
                col = flat_mr[smask, col_idx]
                n_brk = int((col > 0.5).sum())
                if n_brk > 0:
                    brk_mask_local = (col > 0.5)
                    wr_brk = float((mfes[smask][brk_mask_local] > maes[smask][brk_mask_local]).mean())
                    break_counts.append((tr_name, n_brk, wr_brk))
            break_counts.sort(key=lambda x: x[1], reverse=True)
            print(f"\n  {sess}:")
            for tr_name, n_brk, wr_brk in break_counts[:5]:
                print(f"    {tr_name:<30} breaks={n_brk:>4}, WR|brk={wr_brk:.1%}")

    # === STACKED GATE ANALYSIS: Progressive filter compound WR ===
    # Combines all discovered gates to answer: "what's the WR when we ride the wave?"
    print(f"\n{'='*70}")
    print(f"  STACKED GATE ANALYSIS: Compound Filters")
    print(f"  Each gate stacks on previous — progressive noise removal")
    print(f"{'='*70}")

    # Build KEEP template mask from seg_results
    keep_tids = set()
    for s in keep_segs:
        keep_tids.add((s['tid'], s['dir']))

    keep_mask = np.zeros(len(mfes), dtype=bool)
    for i, m in enumerate(meta):
        d = 'LONG' if long_mask[i] else 'SHORT'
        if (m['tid'], d) in keep_tids:
            keep_mask[i] = True

    # Session + temporal masks (reuse already-computed arrays)
    if n_valid > 50:
        # Avoid Europe session (noise factory) + session opens (first 30min of each session)
        europe_mask = np.array([s == 'EUROPE' for s in sessions]) & valid_ts
        us_rth_mask = np.array([s == 'US_RTH' for s in sessions]) & valid_ts

        # Session open = minutes 0-29 of each hour that starts a session
        # US RTH open = 14:xx UTC, Europe open = 08:xx UTC
        session_open_mask = np.zeros(len(mfes), dtype=bool)
        for i, dt in enumerate(dts):
            if dt is None:
                continue
            h, mn = dt.hour, dt.minute
            # US RTH open: 14:00-14:29
            if h == 14 and mn < 30:
                session_open_mask[i] = True
            # Europe open: 08:00-08:29
            elif h == 8 and mn < 30:
                session_open_mask[i] = True

        # Good hours: hours with WR > 60% from drill-down
        # (17, 18, 0, 5 were consistently high across sessions)
        good_hours = {0, 5, 17, 18, 19, 20}
        good_hour_mask = np.array([h in good_hours for h in hours_utc]) & valid_ts

        # PID contrarian: PID sign disagrees with DMI direction
        # PID > 0 while SHORT, or PID < 0 while LONG
        pid_d7_vals = padded[:, 7, pid_idx]  # depth 7 = transition zone
        pid_contrarian = ((pid_d7_vals > 0) & short_mask) | ((pid_d7_vals < 0) & long_mask)

        # Day-of-week: avoid worst days (from drill-down)
        # Tuesday and Thursday were consistently strong
        good_dow = {1, 3}  # 1=TUE, 3=THU
        good_dow_mask = np.array([d in good_dow for d in dow]) & valid_ts

        # First week of month (dom 1-7) showed strong KEEP WR
        first_week_mask = (dom >= 1) & (dom <= 7) & valid_ts

        # --- Progressive stacking ---
        gates = []

        # Gate 0: Baseline (all patterns)
        gates.append(('ALL patterns', np.ones(len(mfes), dtype=bool)))

        # Gate 1: KEEP templates only
        gates.append(('+ KEEP templates', keep_mask))

        # Gate 2: + LONG direction only (all KEEP were LONG in full-year)
        g2 = keep_mask & long_mask
        gates.append(('+ LONG direction', g2))

        # Gate 3: + Avoid session opens (14:00-14:29, 08:00-08:29)
        g3 = g2 & ~session_open_mask
        gates.append(('+ Skip session opens', g3))

        # Gate 4: + Avoid Europe session entirely
        g4 = g3 & ~europe_mask
        gates.append(('+ Skip Europe session', g4))

        # Gate 5: + Good hours only (17-20, 0, 5 UTC)
        g5 = g4 & good_hour_mask
        gates.append(('+ Best hours (17-20,0,5)', g5))

        # Gate 6: + PID contrarian confirmation
        g6 = g5 & pid_contrarian
        gates.append(('+ PID contrarian', g6))

        # Gate 7: + Good day-of-week (TUE, THU)
        g7 = g5 & good_dow_mask  # branch from g5, not g6 (PID can be optional)
        gates.append(('+ Best DOW (TUE,THU)', g7))

        # Gate 8: Full stack (best hours + best DOW + PID contrarian)
        g8 = g5 & good_dow_mask & pid_contrarian
        gates.append(('FULL STACK (all gates)', g8))

        print(f"\n  {'Gate':<32} {'N':>5} {'%vol':>6} {'WR':>7} {'MFE':>7} "
              f"{'MAE':>6} {'$/trade':>8} {'Lift':>7}")
        print(f"  {'-'*32} {'-'*5} {'-'*6} {'-'*7} {'-'*7} "
              f"{'-'*6} {'-'*8} {'-'*7}")

        base_wr = float((mfes > maes).mean())
        total_patterns = len(mfes)

        for label, gmask in gates:
            n_g = gmask.sum()
            if n_g < 5:
                print(f"  {label:<32} {n_g:>5} {'<5':>6} {'n/a':>7}")
                continue
            wr_g = float((mfes[gmask] > maes[gmask]).mean())
            mfe_g = float(np.mean(mfes[gmask]))
            mae_g = float(np.mean(maes[gmask]))
            vol_pct = n_g / total_patterns
            avg_pnl = mfe_g - mae_g  # rough $/trade proxy
            lift = wr_g - base_wr
            print(f"  {label:<32} {n_g:>5} {vol_pct:>5.1%} {wr_g:>7.1%} "
                  f"{mfe_g:>+7.0f} {mae_g:>6.0f} {avg_pnl:>+8.0f} {lift:>+6.1%}")

        # --- Daily trade count estimate ---
        # Use valid timestamp range to compute trading days
        if valid_ts.sum() > 0:
            t_min = ts_arr[valid_ts].min()
            t_max = ts_arr[valid_ts].max()
            days_span = max((t_max - t_min) / 86400, 1)

            print(f"\n  DAILY THROUGHPUT (over {days_span:.0f} calendar days):")
            for label, gmask in gates:
                n_g = gmask.sum()
                if n_g < 5:
                    continue
                per_day = n_g / days_span
                wr_g = float((mfes[gmask] > maes[gmask]).mean())
                print(f"  {label:<32} {per_day:>6.1f}/day  WR={wr_g:.1%}")

        # --- Ride-the-wave summary ---
        # Use the best practical gate (g5 = KEEP + LONG + no opens + no Europe + good hours)
        best_gate = g5
        best_label = "KEEP+LONG+GoodHours"
        n_best = best_gate.sum()
        if n_best >= 5:
            wr_best = float((mfes[best_gate] > maes[best_gate]).mean())
            mfe_best = float(np.mean(mfes[best_gate]))
            mae_best = float(np.mean(maes[best_gate]))
            per_day = n_best / max(days_span, 1)

            print(f"\n  {'='*60}")
            print(f"  RIDE THE WAVE — Practical Gate Summary")
            print(f"  {'='*60}")
            print(f"  Filter: {best_label}")
            print(f"  Patterns:   {n_best} ({n_best/total_patterns:.1%} of volume)")
            print(f"  Win Rate:   {wr_best:.1%}")
            print(f"  Avg MFE:    {mfe_best:+.0f} ticks")
            print(f"  Avg MAE:    {mae_best:.0f} ticks")
            print(f"  $/trade:    {mfe_best - mae_best:+.0f} ticks net")
            print(f"  Throughput: {per_day:.1f} trades/day")
            print(f"  WR lift:    {wr_best - base_wr:+.1%} vs baseline")

            # Contract scaling estimate (MES = $1.25/tick)
            tick_val = 1.25  # MES
            net_per_trade = (mfe_best - mae_best) * tick_val
            daily_pnl_1 = net_per_trade * per_day
            print(f"\n  MES CONTRACT SCALING:")
            print(f"    1 contract:  ${net_per_trade:+.2f}/trade, "
                  f"${daily_pnl_1:+.0f}/day")
            for contracts in [2, 5, 10]:
                print(f"    {contracts} contracts: ${net_per_trade*contracts:+.2f}/trade, "
                      f"${daily_pnl_1*contracts:+.0f}/day")

            # What the wave rider adds on top
            print(f"\n  WAVE RIDER UPSIDE (not captured in static screening):")
            print(f"    - Ebb-and-flow at 5min inside 15min patterns")
            print(f"    - PID throttle: accelerate on momentum, pause on exhaustion")
            print(f"    - Contrarian PID: disagree signal = mean-reversion edge")
            print(f"    - EXIT gates from MR drill-down (not just entry)")

        # === SPLIT SEGMENTS WITH TEMPORAL GATES ===
        # Same noise removal applied to SPLIT segments for blended revenue
        split_tids = set()
        for s in split_segs:
            split_tids.add((s['tid'], s['dir']))

        split_mask = np.zeros(len(mfes), dtype=bool)
        for i, m in enumerate(meta):
            d = 'LONG' if long_mask[i] else 'SHORT'
            if (m['tid'], d) in split_tids:
                split_mask[i] = True

        n_split_raw = split_mask.sum()
        if n_split_raw >= 10:
            print(f"\n  {'='*60}")
            print(f"  SPLIT SEGMENTS — Temporal Gate Cleanup")
            print(f"  {'='*60}")

            split_gates = []
            split_gates.append(('SPLIT raw', split_mask))

            sp1 = split_mask & ~session_open_mask
            split_gates.append(('+ Skip session opens', sp1))

            sp2 = sp1 & ~europe_mask
            split_gates.append(('+ Skip Europe', sp2))

            sp3 = sp2 & good_hour_mask
            split_gates.append(('+ Best hours', sp3))

            sp4 = sp2 & good_dow_mask
            split_gates.append(('+ Best DOW (TUE,THU)', sp4))

            sp5 = sp3 & good_dow_mask
            split_gates.append(('+ Best hours + DOW', sp5))

            print(f"\n  {'Gate':<32} {'N':>5} {'WR':>7} {'MFE':>7} "
                  f"{'MAE':>6} {'net':>6} {'$/day':>8}")
            print(f"  {'-'*32} {'-'*5} {'-'*7} {'-'*7} "
                  f"{'-'*6} {'-'*6} {'-'*8}")

            for label, gmask in split_gates:
                n_g = gmask.sum()
                if n_g < 5:
                    print(f"  {label:<32} {n_g:>5}  (too few)")
                    continue
                wr_g = float((mfes[gmask] > maes[gmask]).mean())
                mfe_g = float(np.mean(mfes[gmask]))
                mae_g = float(np.mean(maes[gmask]))
                net_ticks = mfe_g - mae_g
                per_day_g = n_g / max(days_span, 1)
                daily_1mes = net_ticks * tick_val * per_day_g
                print(f"  {label:<32} {n_g:>5} {wr_g:>7.1%} {mfe_g:>+7.0f} "
                      f"{mae_g:>6.0f} {net_ticks:>+6.0f} ${daily_1mes:>+7.0f}")

            # Best practical SPLIT gate = skip Europe + good hours
            sp_best = sp3
            n_sp = sp_best.sum()

            # === REVENUE MODEL: 1 contract unified ===
            # KEEP tier stats
            k_net = (mfe_best - mae_best)  # ticks
            k_per_day = n_best / max(days_span, 1)

            # SPLIT tier stats
            if n_sp >= 5:
                sp_wr = float((mfes[sp_best] > maes[sp_best]).mean())
                sp_mfe = float(np.mean(mfes[sp_best]))
                sp_mae = float(np.mean(maes[sp_best]))
                sp_net = sp_mfe - sp_mae
                sp_per_day = n_sp / max(days_span, 1)
            else:
                sp_wr, sp_net, sp_per_day = 0, 0, 0

            # Combined unified pool (1 contract trades both KEEP + SPLIT serially)
            unified_mask = best_gate | sp_best
            n_unified = unified_mask.sum()
            if n_unified >= 5:
                u_wr = float((mfes[unified_mask] > maes[unified_mask]).mean())
                u_mfe = float(np.mean(mfes[unified_mask]))
                u_mae = float(np.mean(maes[unified_mask]))
                u_net = u_mfe - u_mae
                u_per_day = n_unified / max(days_span, 1)
                u_daily_1 = u_net * tick_val * u_per_day
            else:
                u_wr, u_net, u_per_day, u_daily_1 = 0, 0, 0, 0

            print(f"\n  {'='*60}")
            print(f"  REVENUE MODEL — 1 Contract (KEEP + SPLIT unified)")
            print(f"  {'='*60}")

            print(f"\n  POOL BREAKDOWN (1 MES = $1.25/tick):")
            print(f"  {'Pool':<20} {'N':>5} {'WR':>7} {'net/t':>6} "
                  f"{'trades/d':>9} {'$/day':>8}")
            print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*6} "
                  f"{'-'*9} {'-'*8}")
            k_daily_1 = k_net * tick_val * k_per_day
            sp_daily_1 = sp_net * tick_val * sp_per_day
            print(f"  {'KEEP (best hrs)':<20} {n_best:>5} {wr_best:>7.1%} "
                  f"{k_net:>+6.0f} {k_per_day:>9.1f} ${k_daily_1:>7,.0f}")
            if n_sp >= 5:
                print(f"  {'SPLIT (best hrs)':<20} {n_sp:>5} {sp_wr:>7.1%} "
                      f"{sp_net:>+6.0f} {sp_per_day:>9.1f} ${sp_daily_1:>7,.0f}")
            print(f"  {'-'*20} {'-'*5} {'-'*7} {'-'*6} "
                  f"{'-'*9} {'-'*8}")
            print(f"  {'UNIFIED':<20} {n_unified:>5} {u_wr:>7.1%} "
                  f"{u_net:>+6.0f} {u_per_day:>9.1f} ${u_daily_1:>7,.0f}")

            # OOS degradation scenarios
            print(f"\n  OOS DEGRADATION SCENARIOS (1 MES contract):")
            print(f"  IS baseline: {u_per_day:.1f} trades/day, "
                  f"+{u_net:.0f} ticks/trade, ${u_daily_1:,.0f}/day")
            print(f"\n  {'Scenario':<25} {'net/t':>6} {'$/trade':>8} "
                  f"{'$/day':>8} {'$/month':>9} {'$800?':>6}")
            print(f"  {'-'*25} {'-'*6} {'-'*8} "
                  f"{'-'*8} {'-'*9} {'-'*6}")

            for pct, label in [(0, 'IS (no decay)'),
                               (10, '10% haircut'),
                               (20, '20% haircut'),
                               (30, '30% haircut'),
                               (40, '40% haircut'),
                               (50, '50% haircut')]:
                decay = 1.0 - pct / 100
                d_net = u_net * decay
                d_trade = d_net * tick_val
                d_daily = d_trade * u_per_day
                d_monthly = d_daily * 20
                hits = 'YES' if d_daily >= 800 else 'no'
                print(f"  {label:<25} {d_net:>+6.0f} ${d_trade:>7,.0f} "
                      f"${d_daily:>7,.0f} ${d_monthly:>8,.0f} {hits:>6}")

            # Breakeven analysis
            # At what WR does 1 contract still make $800/day?
            # Simplification: assume same trade frequency, find min net/trade
            min_net_800 = 800 / (u_per_day * tick_val) if u_per_day > 0 else 0
            max_decay_800 = (1 - min_net_800 / u_net) * 100 if u_net > 0 else 0

            print(f"\n  BREAKEVEN:")
            print(f"    $800/day needs +{min_net_800:.0f} ticks/trade "
                  f"@ {u_per_day:.1f} trades/day")
            print(f"    Max tolerable decay: {max_decay_800:.0f}% "
                  f"before dropping below $800")
            print(f"    IS net: +{u_net:.0f}t -> "
                  f"buffer of {u_net - min_net_800:.0f} ticks "
                  f"({max_decay_800:.0f}% margin of safety)")

            # Contract scaling for reference
            print(f"\n  CONTRACT SCALING (at IS rates, ${u_daily_1:,.0f}/day/MES):")
            margin_1 = 1320
            for cts in [1, 2, 3, 5]:
                d = u_daily_1 * cts
                m = margin_1 * cts
                print(f"    {cts} MES: ${d:>8,.0f}/day, "
                      f"${d*20:>9,.0f}/month  (margin: ${m:>6,.0f})")
    else:
        print(f"  (Skipped — insufficient valid timestamps)")

    # 9. Save — flush captured buffer to file
    sys.stdout = _orig_stdout  # restore before writing path message

    report_path = os.path.join(os.path.dirname(__file__), 'screening_report.txt')
    header = "WAVEFORM SCREENING: MODEL FISSION REPORT\n"
    if warmup > 0 or window > 0:
        header += f"Warmup: {warmup}d, Window: {window}d\n"
    header += f"Patterns: {len(mfes)}\n"

    with open(report_path, 'w') as f:
        f.write(header)
        f.write(_report_buf.getvalue())

    print(f"\n  Full report saved: {report_path}")


if __name__ == '__main__':
    main()
