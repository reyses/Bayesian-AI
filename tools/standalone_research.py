"""
Standalone Research Screening Tool  (orchestrator)
=====================================================
Thin CLI orchestrator — all helpers live in tools/research/ subpackage.

Usage:
    python tools/standalone_research.py --data DATA/ATLAS_1WEEK --base-tf 15m
    python tools/standalone_research.py --data DATA/ATLAS_1WEEK --base-tf 15m --full
    python tools/standalone_research.py --data DATA/ATLAS --context-days 30 --analysis-days 7

Output: tools/standalone_report.txt + tools/plots/standalone/
"""

import sys, os, io, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.oracle_config import ORACLE_LOOKAHEAD_BARS

# ── Imports from tools.research subpackage ──
from tools.research.data import (
    TF_HIERARCHY, TF_SECONDS, TF_LABELS, FEATURE_NAMES,
    load_atlas_tf, compute_tf_physics, extract_16d, build_stacked_matrices,
)
from tools.research.imr import (
    compute_price_imr, detect_regimes, compute_regime_oracle,
)
from tools.research.screening import (
    pad_to_fixed_depth, compute_moving_range, flatten_matrices,
    screen_factors, regression_r2, print_screening_report,
)
from tools.research.seeds import (
    SeedPrimitiveLibrary, _detect_inflections, _adaptive_split,
)
from tools.research import plots as _res_plots
from tools.research.plots import (
    plot_price_imr, plot_regime_summary, plot_imr_charts, plot_segmented_imr,
)
from tools.research.features_v2 import (
    TF_HIERARCHY_V2, TF_LABELS_V2, FEATURE_NAMES_V2,
    N_TFS_V2, N_FEATURES_PER_TF_V2,
    detect_v2_cache, load_v2_features, align_v2_to_base_tf, reshape_v2_to_stack,
)


# ── Active feature spec — set in main() based on cache mode ──
# Default values are v1 (preserves existing behavior when --cache is unset
# or points to an .npz file).
_ACTIVE_TF_HIERARCHY = TF_HIERARCHY        # 12 TFs
_ACTIVE_TF_LABELS = TF_LABELS               # ['d0_1W', ..., 'd11_15s']
_ACTIVE_FEATURE_NAMES = FEATURE_NAMES       # 16 v1 feature names
_ACTIVE_N_TFS = 12
_ACTIVE_N_FEATURES = 16
_ACTIVE_EXTRAS = ['current_MR']             # appended to flattened stack
_ACTIVE_PID_NAME = 'self_pid'               # feature used in PID drill-down (line 5459)


def _set_active_spec_v2():
    """Switch active feature spec to v2 (8 TFs × 23 features + L0).

    Rebinds the module-level TF_HIERARCHY / TF_LABELS / FEATURE_NAMES so
    existing analyses that reference these names by import pick up v2 values
    automatically. This avoids touching every analysis call site.
    """
    global _ACTIVE_TF_HIERARCHY, _ACTIVE_TF_LABELS, _ACTIVE_FEATURE_NAMES
    global _ACTIVE_N_TFS, _ACTIVE_N_FEATURES, _ACTIVE_EXTRAS, _ACTIVE_PID_NAME
    global TF_HIERARCHY, TF_LABELS, FEATURE_NAMES
    _ACTIVE_TF_HIERARCHY = TF_HIERARCHY_V2
    _ACTIVE_TF_LABELS = TF_LABELS_V2
    _ACTIVE_FEATURE_NAMES = FEATURE_NAMES_V2
    _ACTIVE_N_TFS = N_TFS_V2                # 8
    _ACTIVE_N_FEATURES = N_FEATURES_PER_TF_V2  # 23
    _ACTIVE_EXTRAS = ['current_MR', 'L0_time_of_day']
    _ACTIVE_PID_NAME = 'reversion_prob'     # v2 closest analog of self_pid
    # Rebind module-level names so all downstream analyses pick up v2 values
    TF_HIERARCHY = TF_HIERARCHY_V2
    TF_LABELS = TF_LABELS_V2
    FEATURE_NAMES = FEATURE_NAMES_V2


def _build_col_names(prefix=""):
    """Build column names from the active spec.

    Returns list of length (N_TFS * N_FEATURES) + len(EXTRAS) when prefix=='',
    else just (N_TFS * N_FEATURES) for the delta naming.
    """
    cols = []
    for d in range(_ACTIVE_N_TFS):
        tf_lbl = _ACTIVE_TF_LABELS[d] if d < len(_ACTIVE_TF_LABELS) else f'd{d}'
        for f in range(_ACTIVE_N_FEATURES):
            f_lbl = _ACTIVE_FEATURE_NAMES[f] if f < len(_ACTIVE_FEATURE_NAMES) else f'f{f}'
            cols.append(f"{prefix}{tf_lbl}__{f_lbl}")
    if not prefix:
        cols.extend(_ACTIVE_EXTRAS)
    return cols


# Module-level PLOTS_DIR — synced from plots module after resolve_plots_dir()
PLOTS_DIR = _res_plots.PLOTS_DIR


def _resolve_plots_dir(data_path, analysis_days=0):
    """Thin wrapper: resolve in plots module, sync to this module."""
    global PLOTS_DIR
    sub = _res_plots.resolve_plots_dir(data_path, analysis_days)
    PLOTS_DIR = _res_plots.PLOTS_DIR
    return sub


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


# =============================================================================
#  MATPLOTLIB (lazy import for analyses that need inline plots in main)
# =============================================================================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import matplotlib.lines as mlines


# =============================================================================
#  MAIN
# =============================================================================
# All helper functions (data loading, I-MR, screening, seeds, plots) now live
# in tools/research/ subpackage. Only the analysis pipeline (main) remains here
# because analyses A-R share extensive local state that doesn't factor cleanly.

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Standalone research screening from raw ATLAS data')
    parser.add_argument('--data', default='DATA/ATLAS',
                        help='ATLAS data directory (default: DATA/ATLAS)')
    parser.add_argument('--months', nargs='+', default=None,
                        help='Specific months to load (e.g., 2025_01 2025_02)')
    parser.add_argument('--base-tf', default=None,
                        help='Base timeframe for analysis points. '
                             'Default: 15m for v1/SFE mode, 1m for v2 mode.')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup days before analysis window (default: 21)')
    parser.add_argument('--analysis-days', type=int, default=7,
                        help='Analysis window in days (0 = all remaining, default: 7)')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top factors to display')
    parser.add_argument('--full', action='store_true',
                        help='Run full 16D fractal pipeline (physics + TF state matrices)')
    parser.add_argument('--start', default='A',
                        help='Start from this analysis letter (e.g. --start Q). Skips earlier analyses.')
    parser.add_argument('--skip', default='',
                        help='Comma-separated analysis letters to skip (e.g. --skip G,H,I,J,K)')
    parser.add_argument('--cache', default=None,
                        help='Polymorphic. (a) Path to .npz cache: load assembled matrices and '
                             'skip Steps 1-8. (b) Path to v2 features dir (e.g., '
                             'DATA/ATLAS/FEATURES_5s_v2): switch to v2 mode, load precomputed '
                             '185D features, fall back to live core_v2 SFE compute for missing '
                             'days. Default base-TF becomes 1m in v2 mode.')
    args = parser.parse_args()
    _start_at = args.start.upper()
    _skip_set = set(s.strip().upper() for s in args.skip.split(',') if s.strip())

    # ── Detect cache mode and switch active feature spec ──────────────────
    # cache_mode: 'v2_features' | 'npz' | 'none' (no --cache passed)
    cache_mode = 'none'
    if args.cache:
        if detect_v2_cache(args.cache):
            cache_mode = 'v2_features'
            _set_active_spec_v2()
        elif args.cache.endswith('.npz') or os.path.isfile(args.cache):
            cache_mode = 'npz'
        else:
            # Path exists but isn't .npz and isn't a v2 dir, OR doesn't exist.
            # Treat as npz target if it doesn't exist (compute-and-save path).
            if not os.path.exists(args.cache):
                cache_mode = 'npz'  # will be created below
            else:
                print(f"ERROR: --cache path '{args.cache}' is neither a v2 features dir "
                      f"nor an .npz file. Expected directory containing L0/, L1_*/, "
                      f"L2_*/, L3_*/ OR a path ending in .npz.")
                sys.exit(1)

    # Resolve base_tf default based on mode
    if args.base_tf is None:
        args.base_tf = '1m' if cache_mode == 'v2_features' else '15m'

    # Resolve plots dir based on data path
    sample_label = _resolve_plots_dir(args.data, getattr(args, 'analysis_days', 0))

    # Capture all output to report file
    _report_buf = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, _report_buf)

    print(f"{'='*70}")
    print(f"  STANDALONE RESEARCH")
    print(f"  Data: {args.data}")
    print(f"  Base TF: {args.base_tf}")
    print(f"  Context: {args.context_days}d warmup, {args.analysis_days}d analysis")
    print(f"  Mode: {'FULL (16D fractal)' if args.full else 'PRICE I-MR (default)'}")
    print(f"{'='*70}")

    # ── Cache load (skip Steps 1-8 if cached .npz exists) ────────────
    if cache_mode == 'npz' and os.path.exists(args.cache):
        print(f"\n--- LOADING CACHED DATA: {args.cache} ---")
        _cached = np.load(args.cache, allow_pickle=False)
        X = _cached['X']
        X_delta = _cached['X_delta']
        Y_p = _cached['Y_p']
        Y_d = _cached['Y_d']
        sample_ts = _cached['sample_ts'].tolist()
        # Infer feature spec from X width: v1 = 12*16+1 = 193, v2 = 8*23+2 = 186
        if X.shape[1] == N_TFS_V2 * N_FEATURES_PER_TF_V2 + 2:
            _set_active_spec_v2()
            print(f"  Detected v2 cache (X.shape[1]={X.shape[1]})")
        col_names = _build_col_names(prefix="")
        delta_col_names = _build_col_names(prefix="dt_")
        print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"  Cache size: {os.path.getsize(args.cache)/1e6:.1f} MB")
    else:

        # =====================================================================
        #  STEP 1: Load base TF data
        # =====================================================================
        print(f"\n--- STEP 1: Loading base TF data ({args.base_tf}) ---")
        base_df = load_atlas_tf(args.data, args.base_tf, months=args.months)
        if base_df.empty:
            print(f"ERROR: Base TF '{args.base_tf}' has no data in {args.data}")
            sys.exit(1)
        print(f"  {args.base_tf}: {len(base_df):,} bars loaded")

        # =====================================================================
        #  STEP 2: Price I-MR chart (pure price/time, no physics)
        # =====================================================================
        print(f"\n--- STEP 2: Price I-MR Chart ---")
        price_imr = compute_price_imr(base_df, args.context_days, args.analysis_days)

        # =====================================================================
        #  STEP 3: Detect regimes from MR UCL breaks
        # =====================================================================
        print(f"\n--- STEP 3: Regime Detection ---")
        regime_ids, regime_meta = detect_regimes(price_imr)

        # =====================================================================
        #  STEP 4: Oracle MFE/MAE with regime-based direction
        # =====================================================================
        print(f"\n--- STEP 4: Oracle MFE/MAE ---")
        lookahead = ORACLE_LOOKAHEAD_BARS.get(args.base_tf, 16)
        bar_indices, mfes, maes, directions = compute_regime_oracle(
            base_df, regime_ids, regime_meta, lookahead=lookahead)

        if len(mfes) < 10:
            print(f"ERROR: Only {len(mfes)} oracle bars (need >= 10)")
            sys.exit(1)

        # =====================================================================
        #  STEP 5: Price I-MR plot + Regime summary
        # =====================================================================
        print(f"\n--- STEP 5: Generating charts ---")
        plot_price_imr(price_imr, regime_ids, regime_meta, base_df)
        plot_regime_summary(regime_meta, mfes, maes, bar_indices, regime_ids)

        # =====================================================================
        #  STEP 6: Print regime summary table
        # =====================================================================

        print(f"\n{'='*70}")
        print(f"  REGIME SUMMARY")
        print(f"{'='*70}")
        print(f"  Analysis bars: {len(mfes)}")
        print(f"  Regimes: {len(regime_meta)}")
        print(f"  MFE: mean={np.mean(mfes):.2f}, std={np.std(mfes):.2f}")
        print(f"  MAE: mean={np.mean(maes):.2f}, std={np.std(maes):.2f}")
        print(f"  Win rate (MFE > MAE): {(mfes > maes).mean():.1%}")

        print(f"\n  {'Regime':<10} {'Dir':<6} {'N':>5} {'WR':>7} {'MFE':>8} "
              f"{'MAE':>8} {'Vol':>6} {'Price':>10}")
        print(f"  {'-'*10} {'-'*6} {'-'*5} {'-'*7} {'-'*8} "
              f"{'-'*8} {'-'*6} {'-'*10}")

        bar_regime = regime_ids[bar_indices]
        for rm in regime_meta:
            rid = rm['regime_id']
            mask = bar_regime == rid
            if mask.sum() == 0:
                continue
            wr = float((mfes[mask] > maes[mask]).mean())
            mean_mfe = float(np.mean(mfes[mask]))
            mean_mae = float(np.mean(maes[mask]))
            print(f"  R{rid:<9} {rm['direction']:<6} {mask.sum():>5} {wr:>7.1%} "
                  f"{mean_mfe:>+8.1f} {mean_mae:>8.1f} {rm['volatility']:>6.2f} "
                  f"{rm['mean_price']:>10.1f}")

        # Directional split
        long_mask = np.array([d == 'LONG' for d in directions])
        short_mask = ~long_mask
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        wr_long = float((mfes[long_mask] > maes[long_mask]).mean()) if n_long > 0 else 0
        wr_short = float((mfes[short_mask] > maes[short_mask]).mean()) if n_short > 0 else 0

        print(f"\n  DIRECTIONAL SPLIT:")
        print(f"  LONG:  {n_long:>5} bars, WR={wr_long:.1%}")
        print(f"  SHORT: {n_short:>5} bars, WR={wr_short:.1%}")

        # =====================================================================
        #  STEP 7: Load all 12 TFs + compute fractal context
        # =====================================================================
        if cache_mode == 'v2_features':
            # ─── V2 PATH: load precomputed features (or live-compute fallback) ───
            print(f"\n--- STEP 7 (v2): Loading precomputed features from {args.cache} ---")
            # Limit ts_range to the actual analysis window (context + analysis days
            # back from t_max). Without this we'd load 12+ months of v2 features for
            # nothing; only the trailing window is used for the I-MR / oracle / X build.
            ts_max_total = int(base_df['timestamp'].iloc[-1])
            window_days = max(args.context_days + args.analysis_days, 1) + 2  # +2 buffer
            ts_min_window = ts_max_total - window_days * 86400
            features_5s = load_v2_features(
                v2_dir=args.cache,
                atlas_root=args.data,
                day_strs=None,            # auto-pick days available, filtered by ts_range
                ts_range=(ts_min_window, ts_max_total),
                verbose=True,
            )
            print(f"  v2 features: {len(features_5s):,} 5s rows, "
                  f"{features_5s.shape[1]} cols")

            # =================================================================
            #  STEP 8 (v2): Reindex onto base_tf timestamps + flatten to X
            # =================================================================
            print(f"\n--- STEP 8 (v2): Reindex onto {args.base_tf} bars ---")
            analysis_idx = np.where(regime_ids >= 0)[0]
            timestamps = base_df['timestamp'].values.astype(float)
            close = base_df['close'].values.astype(float)
            mr_signed = price_imr['mr']
            n_bars = len(close)

            # Filter to valid analysis indices (need idx-1 and idx+1)
            valid_idx = analysis_idx[(analysis_idx >= 1) & (analysis_idx + 1 < n_bars)]
            base_ts_for_align = timestamps[valid_idx].astype(np.int64)
            base_ts_prev = timestamps[valid_idx - 1].astype(np.int64)

            # Reindex 5s features onto base TF timestamps (current bar + previous bar for delta)
            aligned_curr = align_v2_to_base_tf(features_5s, base_ts_for_align)
            aligned_prev = align_v2_to_base_tf(features_5s, base_ts_prev)

            # Reshape to (N, 8, 23) stacks + L0 globals
            stack_curr, l0_curr = reshape_v2_to_stack(aligned_curr)
            stack_prev, l0_prev = reshape_v2_to_stack(aligned_prev)

            # Flatten + append [current_MR, L0_time_of_day] → (N, 8*23+2) = (N, 186)
            mr_for_valid = mr_signed[valid_idx]
            stack_flat = stack_curr.reshape(len(valid_idx), -1)  # (N, 184)
            X = np.concatenate(
                [stack_flat, mr_for_valid[:, None], l0_curr[:, None]],
                axis=1,
            )
            # Delta: per-TF stack diff (no MR/L0 in delta to keep parity with v1)
            stack_delta = (stack_curr - stack_prev).reshape(len(valid_idx), -1)
            X_delta = stack_delta

            Y_p = close[valid_idx]
            next_change = close[valid_idx + 1] - close[valid_idx]
            Y_d = np.where(next_change > 0, 1.0, np.where(next_change < 0, -1.0, 0.0))
            sample_ts = base_ts_for_align.tolist()

            print(f"  Samples: {len(Y_p)}, Level features: {X.shape[1]}, "
                  f"Delta features: {X_delta.shape[1]}")
            print(f"  X: {_ACTIVE_N_TFS}*{_ACTIVE_N_FEATURES}={_ACTIVE_N_TFS*_ACTIVE_N_FEATURES} "
                  f"stack + current_MR + L0_time_of_day = {X.shape[1]} level features")
        else:
            print(f"\n--- STEP 7: Loading all TF data + fractal context ---")
            all_dfs = {args.base_tf: base_df}
            for tf in TF_HIERARCHY:
                if tf == args.base_tf:
                    continue
                df = load_atlas_tf(args.data, tf, months=args.months)
                if not df.empty:
                    all_dfs[tf] = df
                    print(f"  {tf:>4}: {len(df):>8,} bars")
                else:
                    print(f"  {tf:>4}:   (not found)")

            print(f"\n  Computing physics per TF...")
            all_tf_states = {}
            for tf in tqdm(TF_HIERARCHY, desc="Physics", unit="tf", ascii=True, dynamic_ncols=True):
                if tf not in all_dfs:
                    continue
                states = compute_tf_physics(tf, all_dfs[tf])
                if states:
                    all_tf_states[tf] = states
                    print(f"  {tf:>4}: {len(states):>8,} states computed")

            # =================================================================
            #  STEP 8: Build X (fractal context + current MR) for each bar
            #
            #  X = N_FLAT fractal features + signed MR[t]
            #  Two Y targets:
            #    Y_price     = close[t]              (can we explain the price?)
            #    Y_direction = sign(close[t+1]-close[t])  (can we explain the direction?)
            # =================================================================
            _N_FLAT = _ACTIVE_N_TFS * _ACTIVE_N_FEATURES
            print(f"\n--- STEP 8: Building context matrix ({_N_FLAT + len(_ACTIVE_EXTRAS)} features per bar) ---")

            analysis_idx = np.where(regime_ids >= 0)[0]
            timestamps = base_df['timestamp'].values.astype(float)
            close = base_df['close'].values.astype(float)
            mr_signed = price_imr['mr']

            # Pre-sort timestamps for each TF (for binary search alignment)
            tf_sorted_ts = {}
            for tf in TF_HIERARCHY:
                if tf in all_tf_states and all_tf_states[tf]:
                    tf_sorted_ts[tf] = np.array(sorted(all_tf_states[tf].keys()))

            X_rows = []
            X_delta_rows = []  # rate-of-change features
            Y_price = []
            Y_direction = []
            sample_ts = []
            base_secs = TF_SECONDS.get(args.base_tf, 900)

            def _build_mat(t):
                """Build (N_TFS, N_FEATURES) fractal fingerprint at timestamp t."""
                mat = np.zeros((_ACTIVE_N_TFS, _ACTIVE_N_FEATURES))
                n = 0
                for depth_idx, tf in enumerate(TF_HIERARCHY):
                    if tf not in tf_sorted_ts:
                        continue
                    tf_ts_list = tf_sorted_ts[tf]
                    tf_secs = TF_SECONDS.get(tf, 60)
                    if tf_secs > base_secs:
                        pos = np.searchsorted(tf_ts_list, t, side='right') - 2
                    else:
                        pos = np.searchsorted(tf_ts_list, t, side='right') - 1
                    if pos < 0:
                        continue
                    nearest_ts = tf_ts_list[pos]
                    state = all_tf_states[tf][nearest_ts]
                    mat[depth_idx, :] = extract_16d(state, tf)
                    n += 1
                return mat, n

            n_bars = len(close)
            for idx in tqdm(analysis_idx, desc="Fractal context", unit="bar", ascii=True, dynamic_ncols=True):
                if idx + 1 >= n_bars or idx < 1:
                    continue

                t = int(timestamps[idx])
                t_prev = int(timestamps[idx - 1])
                current_mr = mr_signed[idx]

                mat, has_data = _build_mat(t)
                if has_data < 3:
                    continue

                # Build previous bar's matrix for rate-of-change
                mat_prev, has_prev = _build_mat(t_prev)
                if has_prev < 3:
                    delta = np.zeros_like(mat)
                else:
                    delta = mat - mat_prev

                x_row = np.concatenate([mat.flatten(), [current_mr]])
                x_delta = delta.flatten()
                X_rows.append(x_row)
                X_delta_rows.append(x_delta)
                Y_price.append(close[idx])
                next_change = close[idx + 1] - close[idx]
                Y_direction.append(1.0 if next_change > 0 else (-1.0 if next_change < 0 else 0.0))
                sample_ts.append(t)

            X = np.array(X_rows)
            X_delta = np.array(X_delta_rows)
            Y_p = np.array(Y_price)
            Y_d = np.array(Y_direction)
            print(f"  Samples: {len(Y_p)}, Level features: {X.shape[1] if len(X) > 0 else 0}, "
                  f"Delta features: {X_delta.shape[1] if len(X_delta) > 0 else 0}")
            print(f"  X: {_N_FLAT} fractal + 1 current MR = {_N_FLAT + 1} level features")
            print(f"  X_delta: {_N_FLAT} rate-of-change (feature[t] - feature[t-1])")

        # ── Column names from the active spec ──
        col_names = _build_col_names(prefix="")
        delta_col_names = _build_col_names(prefix="dt_")[:_ACTIVE_N_TFS * _ACTIVE_N_FEATURES]

        # ── Cache save: only for .npz target (skip when --cache points to v2 dir) ──
        if cache_mode == 'npz' and args.cache and not os.path.exists(args.cache):
            np.savez_compressed(args.cache,
                                X=X, X_delta=X_delta, Y_p=Y_p, Y_d=Y_d,
                                sample_ts=np.array(sample_ts, dtype=np.float64))
            print(f"\n  Cache saved: {args.cache} ({os.path.getsize(args.cache)/1e6:.1f} MB)")

    if _start_at <= 'A' and 'A' not in _skip_set:
        # =====================================================================
        #  ANALYSIS A: PRICE EXPLANATION (independent)
        #
        #  Y = close[t] -- the actual price level
        #  Question: does the fractal fingerprint describe WHERE price is?
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS A: PRICE EXPLANATION")
        print(f"  Y = close[t] at each bar")
        print(f"  Y range: {Y_p.min():.1f} to {Y_p.max():.1f}, "
              f"mean={Y_p.mean():.1f}, std={Y_p.std():.1f}")
        print(f"  Samples: {len(Y_p)}")
        print(f"{'='*70}")

        results_price = screen_factors(X, col_names, Y_p)

        print(f"\n  TOP 20 FACTORS (correlation with price):")
        print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
        print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
        for i, (name, corr, abs_corr) in enumerate(results_price[:20], 1):
            bar = '#' * int(abs_corr * 50)
            print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

        print(f"\n  Stepwise regression: context -> price")
        result_a = regression_r2(X, col_names, Y_p, top_k=20, return_model=True)
        steps_price, (price_model, price_scaler, price_feat_idx) = result_a
        r2_p = steps_price[-1][3] if steps_price else 0
        print(f"\n  >> PRICE adj-R2 = {r2_p:.4f}")
        print(f"  >> Context explains {r2_p*100:.1f}% of price variance")

        # Price: BY TIMEFRAME
        print(f"\n  PRICE BY TIMEFRAME:")
        print(f"  {'Depth':<12} {'TF':>6} {'Mean |r|':>10} {'Max |r|':>10} {'Top Factor':<35}")
        print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*35}")
        for d in range(len(TF_HIERARCHY)):
            prefix = TF_LABELS[d]
            pf = [(n, c, a) for n, c, a in results_price if n.startswith(prefix + '__')]
            if pf:
                abs_vals = [a for _, _, a in pf]
                mp = np.mean(abs_vals)
                mx = max(abs_vals)
                best = max(pf, key=lambda x: x[2])
                print(f"  {prefix:<12} {TF_HIERARCHY[d]:>6} {mp:>10.4f} {mx:>10.4f} {best[0]:<35}")

        # Price: BY FEATURE
        print(f"\n  PRICE BY FEATURE:")
        print(f"  {'Feature':<20} {'Mean |r|':>10} {'Max |r|':>10} {'Best TF':<15}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15}")
        for f_name in FEATURE_NAMES:
            pf = [(n, c, a) for n, c, a in results_price if n.endswith(f'__{f_name}')]
            if pf:
                abs_vals = [a for _, _, a in pf]
                mp = np.mean(abs_vals)
                mx = max(abs_vals)
                best = max(pf, key=lambda x: x[2])
                best_tf = best[0].split('__')[0]
                print(f"  {f_name:<20} {mp:>10.4f} {mx:>10.4f} {best_tf:<15}")
        mr_r_p = next((a for n, c, a in results_price if n == 'current_MR'), 0)
        print(f"  {'current_MR':<20} {mr_r_p:>10.4f} {mr_r_p:>10.4f} {'(base TF)':<15}")

        # Price: CONCLUSION
        print(f"\n  PRICE CONCLUSION:")
        if r2_p > 0.80:
            print(f"  Strong: adj-R2 = {r2_p:.4f}. The fractal context reliably describes")
            print(f"  WHERE price is. The 193 features map to price level with high fidelity.")
        elif r2_p > 0.30:
            print(f"  Moderate: adj-R2 = {r2_p:.4f}. Context captures price structure")
            print(f"  but with meaningful residual noise.")
        else:
            print(f"  Weak: adj-R2 = {r2_p:.4f}. Context does not reliably explain price level.")

    else:
        print(f"  [SKIP] Analysis A (--start {_start_at})")

    if _start_at <= 'B' and 'B' not in _skip_set:
        # =====================================================================
        #  ANALYSIS B: DIRECTION EXPLANATION (independent)
        #
        #  Y = sign(close[t+1] - close[t]) -- will price go up or down?
        #  Question: does the fractal fingerprint tell us which way price moves?
        # =====================================================================
        # Build direction matrix: X + price anchor = 194 features
        X_dir = np.column_stack([X, Y_p])  # add close[t] as anchor
        col_names_dir = col_names + ['price_anchor']

        n_up = (Y_d > 0).sum()
        n_down = (Y_d < 0).sum()
        n_flat = (Y_d == 0).sum()

        print(f"\n{'='*70}")
        print(f"  ANALYSIS B: DIRECTION EXPLANATION (with price anchor)")
        print(f"  Y = sign(next change): +1=up, -1=down")
        print(f"  X = 192 fractal + current_MR + price[t] = {X_dir.shape[1]} features")
        print(f"  Distribution: {n_up} up ({n_up/len(Y_d):.0%}), "
              f"{n_down} down ({n_down/len(Y_d):.0%}), {n_flat} flat")
        print(f"  Samples: {len(Y_d)}")
        print(f"{'='*70}")

        results_dir = screen_factors(X_dir, col_names_dir, Y_d)

        print(f"\n  TOP 20 FACTORS (correlation with direction):")
        print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
        print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
        for i, (name, corr, abs_corr) in enumerate(results_dir[:20], 1):
            bar = '#' * int(abs_corr * 50)
            print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}  {bar}")

        print(f"\n  Stepwise regression: context + anchor -> direction")
        steps_dir = regression_r2(X_dir, col_names_dir, Y_d, top_k=20)
        r2_d = steps_dir[-1][3] if steps_dir else 0
        print(f"\n  >> DIRECTION adj-R2 = {r2_d:.4f}")
        print(f"  >> Context explains {r2_d*100:.1f}% of direction variance")

        # Direction: BY TIMEFRAME
        print(f"\n  DIRECTION BY TIMEFRAME:")
        print(f"  {'Depth':<12} {'TF':>6} {'Mean |r|':>10} {'Max |r|':>10} {'Top Factor':<35}")
        print(f"  {'-'*12} {'-'*6} {'-'*10} {'-'*10} {'-'*35}")
        for d in range(len(TF_HIERARCHY)):
            prefix = TF_LABELS[d]
            df = [(n, c, a) for n, c, a in results_dir if n.startswith(prefix + '__')]
            if df:
                abs_vals = [a for _, _, a in df]
                md = np.mean(abs_vals)
                mx = max(abs_vals)
                best = max(df, key=lambda x: x[2])
                print(f"  {prefix:<12} {TF_HIERARCHY[d]:>6} {md:>10.4f} {mx:>10.4f} {best[0]:<35}")

        # Direction: BY FEATURE
        print(f"\n  DIRECTION BY FEATURE:")
        print(f"  {'Feature':<20} {'Mean |r|':>10} {'Max |r|':>10} {'Best TF':<15}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*15}")
        for f_name in FEATURE_NAMES:
            df = [(n, c, a) for n, c, a in results_dir if n.endswith(f'__{f_name}')]
            if df:
                abs_vals = [a for _, _, a in df]
                md = np.mean(abs_vals)
                mx = max(abs_vals)
                best = max(df, key=lambda x: x[2])
                best_tf = best[0].split('__')[0]
                print(f"  {f_name:<20} {md:>10.4f} {mx:>10.4f} {best_tf:<15}")
        mr_r_d = next((a for n, c, a in results_dir if n == 'current_MR'), 0)
        print(f"  {'current_MR':<20} {mr_r_d:>10.4f} {mr_r_d:>10.4f} {'(base TF)':<15}")

        # Direction: sign analysis — are correlations mostly negative?
        top20_signs = [c for _, c, _ in results_dir[:20]]
        n_neg = sum(1 for s in top20_signs if s < 0)
        n_pos = sum(1 for s in top20_signs if s > 0)
        print(f"\n  SIGN PATTERN: {n_neg}/20 top factors have NEGATIVE correlation")
        if n_neg > 14:
            print(f"  Most direction factors point to MEAN REVERSION -- higher feature")
            print(f"  values predict DOWN moves, suggesting overbought/overextended states.")
        elif n_pos > 14:
            print(f"  Most direction factors point to TREND CONTINUATION -- higher")
            print(f"  feature values predict UP moves, suggesting momentum persistence.")
        else:
            print(f"  Mixed signs -- no dominant directional bias in the features.")

        # Direction: CONCLUSION
        print(f"\n  DIRECTION CONCLUSION:")
        if r2_d > 0.15:
            print(f"  Useful: adj-R2 = {r2_d:.4f}. The fractal context carries meaningful")
            print(f"  directional signal. Worth building a directional model from these features.")
        elif r2_d > 0.05:
            print(f"  Weak but present: adj-R2 = {r2_d:.4f}. Some directional signal exists")
            print(f"  but it is fragile. May need more data, different features, or")
            print(f"  non-linear methods to extract it reliably.")
        else:
            print(f"  Insufficient: adj-R2 = {r2_d:.4f}. The fractal context does not")
            print(f"  reliably explain direction. The next bar is essentially unpredictable")
            print(f"  from these features alone.")

    else:
        print(f"  [SKIP] Analysis B (--start {_start_at})")

    if _start_at <= 'C' and 'C' not in _skip_set:
        # =====================================================================
        #  ANALYSIS C: DIRECTION FROM PRICE MODEL (derived)
        #
        #  Since we can explain price (A=95%) but not direction standalone (B=8.7%),
        #  can we derive direction from consecutive price predictions?
        #  predicted_dir = sign( predict(features[t+1]) - predict(features[t]) )
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS C: DIRECTION DERIVED FROM PRICE MODEL")
        print(f"  If price model predicts price[t] and price[t+1],")
        print(f"  direction = sign(predicted[t+1] - predicted[t])")
        print(f"{'='*70}")

        # Predict price for every sample using the price model from Analysis A
        X_price_feat = price_scaler.transform(X[:, price_feat_idx])
        predicted_prices = price_model.predict(X_price_feat)

        # Build consecutive pairs: predicted[t] vs predicted[t+1]
        n_pairs = len(predicted_prices) - 1
        pred_dir = np.sign(predicted_prices[1:] - predicted_prices[:-1])
        actual_dir = Y_d[:-1]  # actual direction at each t (already sign of close[t+1]-close[t])

        # Filter out flat actuals
        mask = actual_dir != 0
        pred_dir_f = pred_dir[mask]
        actual_dir_f = actual_dir[mask]
        n_valid = mask.sum()

        correct = (pred_dir_f == actual_dir_f).sum()
        accuracy = correct / n_valid if n_valid > 0 else 0

        print(f"\n  Pairs: {n_pairs}, Valid (non-flat): {n_valid}")
        print(f"  Predicted direction accuracy: {correct}/{n_valid} = {accuracy:.1%}")

        # Breakdown by actual direction
        up_mask = actual_dir_f > 0
        down_mask = actual_dir_f < 0
        up_correct = (pred_dir_f[up_mask] > 0).sum() if up_mask.sum() > 0 else 0
        down_correct = (pred_dir_f[down_mask] < 0).sum() if down_mask.sum() > 0 else 0
        print(f"\n  When actual UP:   {up_correct}/{up_mask.sum()} = "
              f"{up_correct/up_mask.sum():.1%}" if up_mask.sum() > 0 else "")
        print(f"  When actual DOWN: {down_correct}/{down_mask.sum()} = "
              f"{down_correct/down_mask.sum():.1%}" if down_mask.sum() > 0 else "")

        # Residual analysis: how big are the prediction errors vs actual moves?
        residuals = predicted_prices - Y_p
        actual_moves = np.diff(Y_p)
        print(f"\n  Price model residuals: mean={residuals.mean():.2f}, std={residuals.std():.2f}")
        print(f"  Actual bar-to-bar moves: mean={np.mean(np.abs(actual_moves)):.2f}, "
              f"std={np.std(actual_moves):.2f}")
        snr = np.mean(np.abs(actual_moves)) / residuals.std() if residuals.std() > 0 else 0
        print(f"  Signal-to-noise ratio: {snr:.3f} "
              f"({'good' if snr > 1.5 else 'marginal' if snr > 0.8 else 'poor'}: "
              f"{'moves > noise' if snr > 1 else 'noise > moves'})")

        # Confidence: only count predictions where delta is large enough
        pred_deltas = predicted_prices[1:] - predicted_prices[:-1]
        for threshold in [0.0, 5.0, 10.0, 20.0]:
            conf_mask = (np.abs(pred_deltas) > threshold) & mask
            if conf_mask.sum() > 0:
                conf_correct = (pred_dir[conf_mask] == actual_dir[conf_mask]).sum()
                conf_acc = conf_correct / conf_mask.sum()
                print(f"  |predicted delta| > {threshold:>5.1f}: "
                      f"{conf_correct}/{conf_mask.sum()} = {conf_acc:.1%}")

        # CONCLUSION
        print(f"\n  DERIVED DIRECTION CONCLUSION:")
        if accuracy > 0.60:
            print(f"  Promising: {accuracy:.1%} accuracy. Deriving direction from consecutive")
            print(f"  price predictions works better than standalone direction modeling.")
            if snr > 1.0:
                print(f"  Signal-to-noise is favorable ({snr:.2f}) -- moves are larger than")
                print(f"  prediction residuals.")
            else:
                print(f"  However, signal-to-noise is low ({snr:.2f}) -- may improve with")
                print(f"  more data or better price features.")
        elif accuracy > 0.52:
            print(f"  Marginal: {accuracy:.1%} accuracy. Slightly better than chance but")
            print(f"  not reliable enough. The price model's residual noise ({residuals.std():.1f})")
            print(f"  is {'larger' if snr < 1 else 'comparable to'} the typical move ({np.mean(np.abs(actual_moves)):.1f}).")
        else:
            print(f"  No improvement: {accuracy:.1%}. The price model's residuals overwhelm")
            print(f"  the bar-to-bar signal. 95% R2 on level does not translate to")
            print(f"  directional accuracy at this resolution.")

    else:
        print(f"  [SKIP] Analysis C (--start {_start_at})")

    if _start_at <= 'D' and 'D' not in _skip_set:
        # =====================================================================
        #  ANALYSIS D: DOES RATE-OF-CHANGE IMPROVE PRICE & DIRECTION?
        #
        #  Add delta features (feature[t] - feature[t-1]) to test if
        #  temporal pattern recognition helps beyond the spatial snapshot.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS D: RATE-OF-CHANGE (PATTERN RECOGNITION)")
        print(f"  Level = 193 features (snapshot at t)")
        print(f"  Delta = 192 features (change from t-1 to t)")
        print(f"  Combined = {193 + 192} features (level + delta)")
        print(f"{'='*70}")

        # Combine level + delta features
        X_combined = np.column_stack([X, X_delta])
        combined_col_names = col_names + delta_col_names

        # D1: Does delta help PRICE explanation?
        print(f"\n  D1: PRICE with level+delta features")
        print(f"  Stepwise regression: {X_combined.shape[1]} features -> price")
        steps_price_d = regression_r2(X_combined, combined_col_names, Y_p, top_k=20)
        r2_pd = steps_price_d[-1][3] if steps_price_d else 0
        print(f"\n  >> PRICE adj-R2 (level only):  {r2_p:.4f}")
        print(f"  >> PRICE adj-R2 (level+delta): {r2_pd:.4f}")
        print(f"  >> Delta contribution: {r2_pd - r2_p:+.4f} ({(r2_pd - r2_p)*100:+.1f}%)")

        # How many delta features made it into the model?
        n_delta_in_price = sum(1 for step in steps_price_d if step[0].startswith('dt_'))
        print(f"  >> Delta features in price model: {n_delta_in_price}/{len(steps_price_d)}")

        # Top delta features for price
        price_d_results = screen_factors(X_delta, delta_col_names, Y_p)
        print(f"\n  TOP 10 DELTA FACTORS for price:")
        print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
        print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
        for i, (name, corr, abs_corr) in enumerate(price_d_results[:10], 1):
            print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}")

        # D2: Does delta help DIRECTION explanation?
        print(f"\n  D2: DIRECTION with level+delta features")
        X_dir_combined = np.column_stack([X, X_delta, Y_p])  # add price anchor too
        dir_combined_names = col_names + delta_col_names + ['price_anchor']
        print(f"  Stepwise regression: {X_dir_combined.shape[1]} features -> direction")
        steps_dir_d = regression_r2(X_dir_combined, dir_combined_names, Y_d, top_k=20)
        r2_dd = steps_dir_d[-1][3] if steps_dir_d else 0
        print(f"\n  >> DIRECTION adj-R2 (level only):  {r2_d:.4f}")
        print(f"  >> DIRECTION adj-R2 (level+delta): {r2_dd:.4f}")
        print(f"  >> Delta contribution: {r2_dd - r2_d:+.4f} ({(r2_dd - r2_d)*100:+.1f}%)")

        n_delta_in_dir = sum(1 for step in steps_dir_d if step[0].startswith('dt_'))
        print(f"  >> Delta features in direction model: {n_delta_in_dir}/{len(steps_dir_d)}")

        # Top delta features for direction
        dir_d_results = screen_factors(X_delta, delta_col_names, Y_d)
        print(f"\n  TOP 10 DELTA FACTORS for direction:")
        print(f"  {'Rank':>4}  {'Factor':<35} {'r':>8}  {'|r|':>8}")
        print(f"  {'-'*4}  {'-'*35} {'-'*8}  {'-'*8}")
        for i, (name, corr, abs_corr) in enumerate(dir_d_results[:10], 1):
            print(f"  {i:>4}  {name:<35} {corr:>+8.4f}  {abs_corr:>8.4f}")

        # D3: Derived direction from combined price model
        print(f"\n  D3: DERIVED DIRECTION from level+delta price model")
        result_pd = regression_r2(X_combined, combined_col_names, Y_p, top_k=20, return_model=True)
        steps_pd, (pd_model, pd_scaler, pd_feat_idx) = result_pd
        X_pd_feat = pd_scaler.transform(X_combined[:, pd_feat_idx])
        pred_prices_d = pd_model.predict(X_pd_feat)

        pred_dir_d = np.sign(pred_prices_d[1:] - pred_prices_d[:-1])
        actual_dir_d = Y_d[:-1]
        mask_d = actual_dir_d != 0
        correct_d = (pred_dir_d[mask_d] == actual_dir_d[mask_d]).sum()
        n_valid_d = mask_d.sum()
        accuracy_d = correct_d / n_valid_d if n_valid_d > 0 else 0

        print(f"  Derived direction (level only):  {accuracy:.1%}")
        print(f"  Derived direction (level+delta): {accuracy_d:.1%}")
        print(f"  Delta contribution: {accuracy_d - accuracy:+.1%}")

        # Confidence gates for combined model
        pred_deltas_d = pred_prices_d[1:] - pred_prices_d[:-1]
        for threshold in [0.0, 5.0, 10.0, 20.0]:
            conf_mask = (np.abs(pred_deltas_d) > threshold) & mask_d
            if conf_mask.sum() > 0:
                conf_correct = (pred_dir_d[conf_mask] == actual_dir_d[conf_mask]).sum()
                conf_acc = conf_correct / conf_mask.sum()
                print(f"  |predicted delta| > {threshold:>5.1f}: "
                      f"{conf_correct}/{conf_mask.sum()} = {conf_acc:.1%}")

        # ANALYSIS D CONCLUSION
        print(f"\n  ANALYSIS D CONCLUSION:")
        price_gain = r2_pd - r2_p
        dir_gain = r2_dd - r2_d
        derived_gain = accuracy_d - accuracy
        if price_gain > 0.01 or dir_gain > 0.01 or derived_gain > 0.03:
            print(f"  Pattern recognition HELPS:")
            if price_gain > 0.01:
                print(f"    Price R2: {r2_p:.4f} -> {r2_pd:.4f} (+{price_gain:.4f})")
            if dir_gain > 0.01:
                print(f"    Direction R2: {r2_d:.4f} -> {r2_dd:.4f} (+{dir_gain:.4f})")
            if derived_gain > 0.03:
                print(f"    Derived accuracy: {accuracy:.1%} -> {accuracy_d:.1%} (+{derived_gain:.1%})")
        else:
            print(f"  Rate-of-change features do NOT meaningfully improve results.")
            print(f"    Price R2:  {r2_p:.4f} -> {r2_pd:.4f} ({price_gain:+.4f})")
            print(f"    Dir R2:    {r2_d:.4f} -> {r2_dd:.4f} ({dir_gain:+.4f})")
            print(f"    Derived:   {accuracy:.1%} -> {accuracy_d:.1%} ({derived_gain:+.1%})")
            print(f"  The spatial snapshot already captures what matters. Adding temporal")
            print(f"  deltas does not reveal hidden directional signal.")

    else:
        print(f"  [SKIP] Analysis D (--start {_start_at})")

    if _start_at <= 'E' and 'E' not in _skip_set:
        # =====================================================================
        #  ANALYSIS E: dP/dT-GROUPED DIRECTION (signal amplification)
        #
        #  Group bars by signed dP/dT (= signed MR = close[t] - close[t-1]).
        #  Within each group, bars have similar price behavior, preventing
        #  signal dilution from mixing different market characters.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS E: dP/dT-GROUPED ANALYSIS (SIGNAL AMPLIFICATION)")
        print(f"  Group bars by signed price change rate, run Three Questions per group.")
        print(f"  Hypothesis: homogeneous groups prevent signal dilution.")
        print(f"{'='*70}")

        # current_MR is the last column of X (index -1), which is signed dP/dT
        sample_mr = X[:, -1]  # signed MR for each sample

        # Bin into groups by signed MR: DOWN / FLAT / UP (terciles)
        # Use 3 bins to keep groups large enough for regression
        bin_edges = np.percentile(sample_mr, [33, 67])
        bin_labels = ['DOWN', 'FLAT', 'UP']
        bin_ids = np.digitize(sample_mr, bin_edges)  # 0-2

        print(f"\n  dP/dT bins (quintiles of signed MR):")
        print(f"  {'Bin':<15} {'Range':>20} {'N':>6} {'Mean MR':>10}")
        print(f"  {'-'*15} {'-'*20} {'-'*6} {'-'*10}")
        n_bins = len(bin_labels)
        for b in range(n_bins):
            mask_b = bin_ids == b
            n_b = mask_b.sum()
            if n_b > 0:
                mr_b = sample_mr[mask_b]
                print(f"  {bin_labels[b]:<15} [{mr_b.min():>+8.1f}, {mr_b.max():>+8.1f}] {n_b:>6} {mr_b.mean():>+10.2f}")

        # Run Three Questions per bin
        print(f"\n  PER-GROUP RESULTS:")
        print(f"  {'Bin':<15} {'N':>5} {'Price R2':>10} {'Dir R2':>10} {'Derived':>10} {'Dir>20':>10}")
        print(f"  {'-'*15} {'-'*5} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

        group_results = []
        for b in range(n_bins):
            mask_b = bin_ids == b
            n_b = mask_b.sum()
            if n_b < 20:  # need minimum samples for regression
                print(f"  {bin_labels[b]:<15} {n_b:>5}   (too few samples)")
                group_results.append((bin_labels[b], n_b, 0, 0, 0, 0))
                continue

            X_b = X[mask_b]
            Y_p_b = Y_p[mask_b]
            Y_d_b = Y_d[mask_b]

            # A: Price R2 per group
            steps_p_b = regression_r2(X_b, col_names, Y_p_b, top_k=20)
            r2_p_b = steps_p_b[-1][3] if steps_p_b else 0

            # B: Direction R2 per group
            X_dir_b = np.column_stack([X_b, Y_p_b])
            steps_d_b = regression_r2(X_dir_b, col_names + ['price_anchor'], Y_d_b, top_k=20)
            r2_d_b = steps_d_b[-1][3] if steps_d_b else 0

            # C: Derived direction per group
            result_b = regression_r2(X_b, col_names, Y_p_b, top_k=20, return_model=True)
            steps_b, (model_b, scaler_b, feat_idx_b) = result_b
            X_feat_b = scaler_b.transform(X_b[:, feat_idx_b])
            pred_p_b = model_b.predict(X_feat_b)

            pred_dir_b = np.sign(pred_p_b[1:] - pred_p_b[:-1])
            actual_dir_b = Y_d_b[:-1]
            mask_nf = actual_dir_b != 0
            n_valid_b = mask_nf.sum()
            if n_valid_b > 0:
                correct_b = (pred_dir_b[mask_nf] == actual_dir_b[mask_nf]).sum()
                acc_b = correct_b / n_valid_b
            else:
                acc_b = 0

            # Confidence gate >20
            pred_deltas_b = pred_p_b[1:] - pred_p_b[:-1]
            conf20_mask = (np.abs(pred_deltas_b) > 20) & mask_nf
            if conf20_mask.sum() > 5:
                conf20_acc = (pred_dir_b[conf20_mask] == actual_dir_b[conf20_mask]).sum() / conf20_mask.sum()
                conf20_str = f"{conf20_acc:.1%}({conf20_mask.sum()})"
            else:
                conf20_acc = 0
                conf20_str = "n/a"

            print(f"  {bin_labels[b]:<15} {n_b:>5} {r2_p_b:>10.4f} {r2_d_b:>10.4f} {acc_b:>9.1%} {conf20_str:>10}")
            group_results.append((bin_labels[b], n_b, r2_p_b, r2_d_b, acc_b, conf20_acc))

        # Compare vs global
        print(f"\n  {'GLOBAL':<15} {len(Y_p):>5} {r2_p:>10.4f} {r2_d:>10.4f} {accuracy:>9.1%}")

        # Summary statistics
        valid_groups = [(lbl, n, rp, rd, da, c20) for lbl, n, rp, rd, da, c20 in group_results if n >= 20]
        if valid_groups:
            avg_dir_r2 = np.mean([rd for _, _, _, rd, _, _ in valid_groups])
            avg_derived = np.mean([da for _, _, _, _, da, _ in valid_groups])
            best_group = max(valid_groups, key=lambda x: x[3])
            worst_group = min(valid_groups, key=lambda x: x[3])

            print(f"\n  ANALYSIS E CONCLUSION:")
            print(f"  Average per-group direction R2: {avg_dir_r2:.4f} (vs global {r2_d:.4f})")
            print(f"  Average per-group derived dir:  {avg_derived:.1%} (vs global {accuracy:.1%})")
            print(f"  Best group:  {best_group[0]} (dir R2={best_group[3]:.4f}, derived={best_group[4]:.1%})")
            print(f"  Worst group: {worst_group[0]} (dir R2={worst_group[3]:.4f}, derived={worst_group[4]:.1%})")

            if avg_dir_r2 > r2_d * 1.5:
                print(f"\n  SIGNAL AMPLIFICATION CONFIRMED: grouping by dP/dT improves")
                print(f"  direction R2 by {avg_dir_r2/max(r2_d,0.001):.1f}x on average.")
                print(f"  Homogeneous groups preserve directional signal that drowns")
                print(f"  in the global model. This validates the clustering approach.")
            else:
                print(f"\n  Grouping by dP/dT does not significantly amplify the signal.")
                print(f"  The direction problem may be fundamental, not a grouping issue.")

    else:
        print(f"  [SKIP] Analysis E (--start {_start_at})")

    if _start_at <= 'F' and 'F' not in _skip_set:
        # =====================================================================
        #  ANALYSIS F: REGIME SIGNATURE PLOT
        #
        #  Like fractal dimension vs SNR plots: each regime gets ONE mean
        #  trajectory line on a shared chart. Shapes normalized to entry=0
        #  so we compare MOVEMENT, not price level. Separation between
        #  regime lines = clustering captures distinct behavior.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS F: REGIME SIGNATURE PLOT")
        print(f"{'='*70}")

        lookback_bars = 8    # 8 bars before entry (2 hours at 15m)
        lookahead_bars = 16  # 16 bars after entry (4 hours at 15m)
        shape_len = lookback_bars + 1 + lookahead_bars  # 25 total points

        # Map sample timestamps back to base_df indices
        _ts_col = timestamps.astype(int)
        _ts_to_idx = {int(t): i for i, t in enumerate(_ts_col)}
        sample_indices = []
        for ts in sample_ts:
            if ts in _ts_to_idx:
                sample_indices.append(_ts_to_idx[ts])
            else:
                sample_indices.append(-1)

        # Collect shapes per regime (normalized to entry=0)
        shapes_by_regime = {rm['regime_id']: [] for rm in regime_meta}
        raw_shapes_by_regime = {rm['regime_id']: [] for rm in regime_meta}
        outcomes_by_regime = {rm['regime_id']: [] for rm in regime_meta}

        for i, bar_idx in enumerate(sample_indices):
            if bar_idx < 0 or bar_idx < lookback_bars:
                continue
            if bar_idx + lookahead_bars >= len(close):
                continue

            rid = regime_ids[bar_idx]
            if rid < 0:
                continue  # warmup bar

            # Extract price shape: lookback + entry + lookahead
            shape_raw = close[bar_idx - lookback_bars : bar_idx + lookahead_bars + 1]
            if len(shape_raw) != shape_len:
                continue

            entry_price = close[bar_idx]
            shape_norm = shape_raw - entry_price  # normalize: entry = 0

            shapes_by_regime[rid].append(shape_norm)
            raw_shapes_by_regime[rid].append(shape_raw)

            # Win/loss: MFE > MAE from oracle
            future = close[bar_idx + 1 : bar_idx + 1 + lookahead_bars]
            if len(future) > 0:
                max_up = future.max() - entry_price
                max_down = entry_price - future.min()
                outcomes_by_regime[rid].append(1 if max_up > max_down else 0)

        # Filter to regimes with enough shapes (min 5)
        active_rids = [rm['regime_id'] for rm in regime_meta
                       if len(shapes_by_regime.get(rm['regime_id'], [])) >= 5]
        n_active = len(active_rids)

        if n_active == 0:
            print("  No regimes with enough shapes.")
        else:
            # Convert raw shapes to delta-from-entry for each regime
            delta_by_regime = {}
            for rid in active_rids:
                raw = np.array(shapes_by_regime[rid])
                entry_prices = raw[:, lookback_bars]  # price at bar 0
                delta_by_regime[rid] = raw - entry_prices[:, np.newaxis]

            # Distinct colors + line styles + markers (like fractal dim reference)
            cmap = plt.cm.tab10
            line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']
            markers     = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
            regime_colors = {rid: cmap(k % 10) for k, rid in enumerate(active_rids)}
            x_axis = np.arange(-lookback_bars, lookahead_bars + 1)

            # ==== CHART 1: Signature overlay (all regime means, one plot) ====
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))

            legend_handles = []
            for k, rid in enumerate(active_rids):
                delta = delta_by_regime[rid]
                n_shapes = len(delta)
                color = regime_colors[rid]
                ls = line_styles[k % len(line_styles)]
                mk = markers[k % len(markers)]
                rm = next(m for m in regime_meta if m['regime_id'] == rid)
                wr = np.mean(outcomes_by_regime[rid]) * 100 if outcomes_by_regime[rid] else 0

                mean_d = delta.mean(axis=0)
                std_d = delta.std(axis=0)

                # Mean line with markers every 2 bars
                ax.plot(x_axis, mean_d, color=color, linewidth=2.5,
                        linestyle=ls, marker=mk, markevery=2, markersize=6)
                # +/- 1 std band
                ax.fill_between(x_axis, mean_d - std_d, mean_d + std_d,
                                color=color, alpha=0.08)

                label = f"R{rid} ({rm['direction']}, n={n_shapes}, WR={wr:.0f}%)"
                legend_handles.append(mlines.Line2D(
                    [], [], color=color, linestyle=ls, marker=mk,
                    markersize=6, linewidth=2.5, label=label))

            ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Bars from entry (15m)', fontsize=12)
            ax.set_ylabel('Price change from entry (ticks)', fontsize=12)
            ax.set_title('Regime Signatures: Mean Price Trajectory per I-MR Regime\n'
                          '(delta from entry, +/-1 std bands)', fontsize=14)
            ax.legend(handles=legend_handles, fontsize=10, loc='best',
                      framealpha=0.9, edgecolor='gray')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            sig_path = os.path.join(PLOTS_DIR, '0c_stacked_shapes.png')
            fig.savefig(sig_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"  Saved signature overlay: {sig_path}")

            # ==== CHART 2: Per-regime spaghetti audit (delta, vertical stack) ====
            fig2, axes2 = plt.subplots(n_active, 1,
                                       figsize=(12, 3.0 * n_active),
                                       squeeze=False)

            for row, rid in enumerate(active_rids):
                ax2 = axes2[row, 0]
                delta = delta_by_regime[rid]
                n_shapes = len(delta)
                color = regime_colors[rid]
                rm = next(m for m in regime_meta if m['regime_id'] == rid)
                wr = np.mean(outcomes_by_regime[rid]) * 100 if outcomes_by_regime[rid] else 0

                # Individual traces
                max_to_plot = min(n_shapes, 300)
                alpha = max(0.05, min(0.25, 20.0 / max_to_plot))
                for j in range(max_to_plot):
                    ax2.plot(x_axis, delta[j], color=color, alpha=alpha, linewidth=0.5)

                # Mean + std envelope
                mean_d = delta.mean(axis=0)
                std_d = delta.std(axis=0)
                ax2.plot(x_axis, mean_d, color=color, linewidth=3, label='Mean')
                ax2.fill_between(x_axis, mean_d - std_d, mean_d + std_d,
                                 color=color, alpha=0.15)

                ax2.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

                ax2.set_title(f"R{rid}: {n_shapes} shapes | dir={rm['direction']}, "
                              f"vol={rm['volatility']:.2f} | WR={wr:.0f}%",
                              fontsize=11, loc='left')
                ax2.set_ylabel('dPrice (ticks)')
                if row == n_active - 1:
                    ax2.set_xlabel('Bars from entry (15m)')
                ax2.legend(fontsize=8, loc='upper right')
                ax2.grid(True, alpha=0.2)

            fig2.suptitle('Per-Regime Shape Audit (delta from entry, individual traces)',
                          fontsize=14, y=1.01)
            plt.tight_layout()
            audit_path = os.path.join(PLOTS_DIR, '0d_regime_audit.png')
            fig2.savefig(audit_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig2)
            print(f"  Saved per-regime audit: {audit_path}")

            # Summary table
            for rid in active_rids:
                n_s = len(shapes_by_regime[rid])
                rm = next(m for m in regime_meta if m['regime_id'] == rid)
                delta = delta_by_regime[rid]
                mean_end = delta[:, -1].mean()   # mean endpoint delta
                std_end = delta[:, -1].std()
                wr = np.mean(outcomes_by_regime[rid]) * 100 if outcomes_by_regime[rid] else 0
                print(f"  R{rid}: {n_s:>4} shapes, dir={rm['direction']:>5}, "
                      f"mean_end={mean_end:>+7.1f}, std_end={std_end:>6.1f}, "
                      f"WR={wr:.0f}%")

    else:
        print(f"  [SKIP] Analysis F (--start {_start_at})")

    if _start_at <= 'G' and 'G' not in _skip_set:
        # =====================================================================
        #  ANALYSIS G: LAPLACIAN SUB-SEGMENTATION
        #
        #  d2p/dt2 (curvature) = the missing acceleration layer.
        #  I-MR segments by velocity breaks. Laplacian segments by SHAPE
        #  changes: inflection points, momentum shifts, deceleration.
        #  Sub-segment each I-MR regime by curvature sign runs, then check
        #  if sub-segments produce tighter shape overlay (the Shi ideal).
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS G: LAPLACIAN SUB-SEGMENTATION (d2p/dt2)")
        print(f"{'='*70}")

        # --- G1: Compute curvature (discrete Laplacian) ---
        # d2p/dt2[t] = close[t+1] - 2*close[t] + close[t-1]
        curvature = np.zeros(len(close))
        curvature[1:-1] = close[2:] - 2 * close[1:-1] + close[:-2]
        curvature[0] = curvature[1]   # pad edges
        curvature[-1] = curvature[-2]

        # Curvature I-MR: same SPC approach on d2p/dt2
        curv_abs = np.abs(curvature)
        analysis_curv = curvature[price_imr['analysis_mask']]
        analysis_curv_abs = np.abs(analysis_curv)

        # MR of curvature (moving range of curvature)
        curv_mr = np.zeros(len(curvature))
        curv_mr[1:] = np.abs(curvature[1:] - curvature[:-1])

        analysis_curv_mr = curv_mr[price_imr['analysis_mask']]
        mr_bar_curv = np.mean(analysis_curv_mr[1:]) if len(analysis_curv_mr) > 1 else 1.0
        ucl_curv = 3.267 * mr_bar_curv  # D4 for n=2

        print(f"  Curvature stats: mean={np.mean(analysis_curv):.4f}, "
              f"std={np.std(analysis_curv):.4f}")
        print(f"  Curvature MR: mean={mr_bar_curv:.4f}, UCL={ucl_curv:.4f}")

        # --- G2: Sub-segment by curvature sign + UCL breaks ---
        # Within each I-MR regime, create sub-segments where:
        #   - curvature sign flips (convex <-> concave)
        #   - OR curvature MR exceeds UCL (acceleration shock)
        # Minimum sub-segment size: 4 bars
        MIN_SUB = 4

        sub_ids = np.full(len(close), -1, dtype=int)
        sub_meta = []
        current_sub = 0
        analysis_indices_g = np.where(price_imr['analysis_mask'])[0]

        for rm in regime_meta:
            # Bars in this regime
            r_mask = (regime_ids == rm['regime_id'])
            r_indices = np.where(r_mask)[0]
            if len(r_indices) < MIN_SUB:
                for idx in r_indices:
                    sub_ids[idx] = current_sub
                current_sub += 1
                continue

            # Walk through regime bars, break on sign flip or UCL
            seg_start = 0
            prev_sign = 1 if curvature[r_indices[0]] >= 0 else -1

            for j in range(1, len(r_indices)):
                idx = r_indices[j]
                cur_sign = 1 if curvature[idx] >= 0 else -1
                mr_break = curv_mr[idx] > ucl_curv

                if (cur_sign != prev_sign or mr_break) and (j - seg_start >= MIN_SUB):
                    # Close current sub-segment
                    for k in range(seg_start, j):
                        sub_ids[r_indices[k]] = current_sub
                    current_sub += 1
                    seg_start = j

                prev_sign = cur_sign

            # Close final sub-segment
            for k in range(seg_start, len(r_indices)):
                sub_ids[r_indices[k]] = current_sub
            current_sub += 1

        # Merge tiny sub-segments
        n_subs_raw = current_sub
        for s in range(n_subs_raw):
            mask_s = (sub_ids == s)
            if 0 < mask_s.sum() < MIN_SUB:
                # Merge into previous or next
                idxs = np.where(mask_s)[0]
                if idxs[0] > 0 and sub_ids[idxs[0] - 1] >= 0:
                    sub_ids[mask_s] = sub_ids[idxs[0] - 1]
                elif idxs[-1] < len(sub_ids) - 1 and sub_ids[idxs[-1] + 1] >= 0:
                    sub_ids[mask_s] = sub_ids[idxs[-1] + 1]

        # Re-compact
        unique_subs = sorted([s for s in np.unique(sub_ids) if s >= 0])
        remap_s = {old: new for new, old in enumerate(unique_subs)}
        for i in range(len(sub_ids)):
            if sub_ids[i] >= 0:
                sub_ids[i] = remap_s[sub_ids[i]]
        n_subs = len(unique_subs)

        # Build sub-segment metadata
        for sid in range(n_subs):
            mask_s = (sub_ids == sid)
            indices_s = np.where(mask_s)[0]
            s_close = close[mask_s]
            s_curv = curvature[mask_s]
            parent_rid = regime_ids[indices_s[0]]

            sub_meta.append({
                'sub_id': sid,
                'parent_regime': int(parent_rid),
                'n_bars': int(mask_s.sum()),
                'mean_price': float(np.mean(s_close)),
                'mean_curvature': float(np.mean(s_curv)),
                'curv_sign': 'CONVEX' if np.mean(s_curv) >= 0 else 'CONCAVE',
                'price_change': float(s_close[-1] - s_close[0]) if len(s_close) > 1 else 0.0,
                'direction': 'LONG' if (s_close[-1] > s_close[0]) else 'SHORT',
                'start_idx': int(indices_s[0]),
                'end_idx': int(indices_s[-1]),
            })

        print(f"\n  I-MR regimes: {len(regime_meta)} -> Laplacian sub-segments: {n_subs}")
        print(f"  Avg sub-segment size: {np.mean([sm['n_bars'] for sm in sub_meta]):.1f} bars")
        print(f"\n  {'Sub':>4} {'Parent':>6} {'Bars':>5} {'Curv':>8} {'Shape':>8} "
              f"{'Dir':>6} {'Chg':>8}")
        print(f"  {'-'*4} {'-'*6} {'-'*5} {'-'*8} {'-'*8} {'-'*6} {'-'*8}")
        for sm in sub_meta:
            print(f"  S{sm['sub_id']:<3} R{sm['parent_regime']:<5} {sm['n_bars']:>5} "
                  f"{sm['mean_curvature']:>+8.3f} {sm['curv_sign']:>8} "
                  f"{sm['direction']:>6} {sm['price_change']:>+8.1f}")

        # --- G3: Collect shapes per sub-segment ---
        shapes_by_sub = {sm['sub_id']: [] for sm in sub_meta}
        outcomes_by_sub = {sm['sub_id']: [] for sm in sub_meta}

        for i, bar_idx in enumerate(sample_indices):
            if bar_idx < 0 or bar_idx < lookback_bars:
                continue
            if bar_idx + lookahead_bars >= len(close):
                continue

            sid = sub_ids[bar_idx]
            if sid < 0:
                continue

            shape_raw = close[bar_idx - lookback_bars : bar_idx + lookahead_bars + 1]
            if len(shape_raw) != shape_len:
                continue

            shapes_by_sub[sid].append(shape_raw)

            entry_price = close[bar_idx]
            future = close[bar_idx + 1 : bar_idx + 1 + lookahead_bars]
            if len(future) > 0:
                max_up = future.max() - entry_price
                max_down = entry_price - future.min()
                outcomes_by_sub[sid].append(1 if max_up > max_down else 0)

        # --- G4: Signature plot — sub-segments overlaid ---
        active_subs = [sm['sub_id'] for sm in sub_meta
                       if len(shapes_by_sub.get(sm['sub_id'], [])) >= 5]
        n_active_g = len(active_subs)

        print(f"\n  Sub-segments with >=5 shapes: {n_active_g}/{n_subs}")

        if n_active_g > 0:
            # Compute delta from entry
            delta_by_sub = {}
            for sid in active_subs:
                raw = np.array(shapes_by_sub[sid])
                entry_prices = raw[:, lookback_bars]
                delta_by_sub[sid] = raw - entry_prices[:, np.newaxis]

            # Color by parent regime, line style by curvature sign
            cmap_g = plt.cm.tab10
            x_axis_g = np.arange(-lookback_bars, lookahead_bars + 1)

            fig_g, ax_g = plt.subplots(1, 1, figsize=(14, 8))
            legend_handles_g = []

            for k, sid in enumerate(active_subs):
                sm = next(s for s in sub_meta if s['sub_id'] == sid)
                delta = delta_by_sub[sid]
                n_shapes = len(delta)
                color = cmap_g(sm['parent_regime'] % 10)
                ls = '-' if sm['curv_sign'] == 'CONVEX' else '--'
                mk = markers[k % len(markers)] if k < len(markers) else 'o'
                wr = np.mean(outcomes_by_sub[sid]) * 100 if outcomes_by_sub[sid] else 0

                mean_d = delta.mean(axis=0)
                std_d = delta.std(axis=0)

                ax_g.plot(x_axis_g, mean_d, color=color, linewidth=2.5,
                          linestyle=ls, marker=mk, markevery=2, markersize=5)
                ax_g.fill_between(x_axis_g, mean_d - std_d, mean_d + std_d,
                                  color=color, alpha=0.06)

                label = (f"S{sid} (R{sm['parent_regime']},{sm['curv_sign'][:3]}, "
                         f"n={n_shapes}, WR={wr:.0f}%)")
                legend_handles_g.append(mlines.Line2D(
                    [], [], color=color, linestyle=ls, marker=mk,
                    markersize=5, linewidth=2.5, label=label))

            ax_g.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_g.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax_g.set_xlabel('Bars from entry (15m)', fontsize=12)
            ax_g.set_ylabel('Price change from entry (ticks)', fontsize=12)
            ax_g.set_title('Laplacian Sub-Segment Signatures\n'
                            '(solid=CONVEX, dashed=CONCAVE, color=parent regime)',
                            fontsize=14)
            ax_g.legend(handles=legend_handles_g, fontsize=9, loc='best',
                        framealpha=0.9, edgecolor='gray')
            ax_g.grid(True, alpha=0.3)

            plt.tight_layout()
            g_sig_path = os.path.join(PLOTS_DIR, '0e_laplacian_signatures.png')
            fig_g.savefig(g_sig_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_g)
            print(f"  Saved: {g_sig_path}")

            # --- G5: Coherence comparison: I-MR vs Laplacian ---
            # Compute avg std at endpoint for I-MR regimes vs Laplacian sub-segments
            imr_stds = []
            for rid in active_rids:
                if rid in delta_by_regime:
                    imr_stds.append(delta_by_regime[rid][:, -1].std())

            lap_stds = []
            for sid in active_subs:
                lap_stds.append(delta_by_sub[sid][:, -1].std())

            avg_imr_std = np.mean(imr_stds) if imr_stds else 0
            avg_lap_std = np.mean(lap_stds) if lap_stds else 0

            print(f"\n  COHERENCE COMPARISON:")
            print(f"  I-MR regimes   -> avg endpoint std: {avg_imr_std:.1f} ticks "
                  f"({len(active_rids)} regimes)")
            print(f"  Laplacian subs -> avg endpoint std: {avg_lap_std:.1f} ticks "
                  f"({n_active_g} sub-segments)")
            if avg_imr_std > 0:
                improvement = (1 - avg_lap_std / avg_imr_std) * 100
                print(f"  Improvement: {improvement:+.1f}% "
                      f"({'TIGHTER' if improvement > 0 else 'WIDER'})")

            # Per sub-segment summary
            print(f"\n  {'Sub':>4} {'Parent':>6} {'Shape':>7} {'N':>4} "
                  f"{'MeanEnd':>8} {'StdEnd':>7} {'WR':>5}")
            for sid in active_subs:
                sm = next(s for s in sub_meta if s['sub_id'] == sid)
                delta = delta_by_sub[sid]
                wr = np.mean(outcomes_by_sub[sid]) * 100 if outcomes_by_sub[sid] else 0
                print(f"  S{sid:<3} R{sm['parent_regime']:<5} {sm['curv_sign'][:3]:>7} "
                      f"{len(delta):>4} {delta[:,-1].mean():>+8.1f} "
                      f"{delta[:,-1].std():>7.1f} {wr:>4.0f}%")

    else:
        print(f"  [SKIP] Analysis G (--start {_start_at})")

    if _start_at <= 'H' and 'H' not in _skip_set:
        # =====================================================================
        #  ANALYSIS H: ITERATIVE SHAPE CLUSTERING (delta from entry)
        #
        #  Every segment starts at 0, values = cumulative price change.
        #  e.g. [0, +40, +20, +30, +20, +40] = the movement pattern.
        #  Grid-search over segment length and cluster count.
        #  Score by silhouette, auto-select best, show top 10 clusters.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS H: ITERATIVE SHAPE CLUSTERING (delta from entry)")
        print(f"{'='*70}")

        from sklearn.cluster import KMeans
        from collections import Counter

        TOP_K = 10
        MIN_CLUSTER_SIZE = 5  # minimum members for a useful cluster

        analysis_idx_h = np.where(price_imr['analysis_mask'])[0]

        def _extract_segments(seg_len):
            """Cut segments, delta from entry (start=0, values=cumulative change)."""
            raws, feats, idxs = [], [], []
            for idx in analysis_idx_h:
                if idx + seg_len > len(close):
                    continue
                seg = close[idx : idx + seg_len]
                if len(seg) != seg_len:
                    continue
                feat = seg - seg[0]  # delta from entry: [0, +40, +20, ...]
                raws.append(seg)
                feats.append(feat)
                idxs.append(idx)
            if len(feats) == 0:
                return np.array([]), np.array([]), np.array([])
            return np.array(raws), np.array(feats), np.array(idxs)

        def _cluster_coherence(feats, labels, top_k=10):
            """Mean within-cluster std (lower = tighter overlays).
            Only considers top-k clusters by size with >= MIN_CLUSTER_SIZE."""
            counts = Counter(labels)
            top = [cid for cid, cnt in counts.most_common(top_k)
                   if cnt >= MIN_CLUSTER_SIZE]
            if not top:
                return 999.0, 0
            stds = []
            for cid in top:
                mask = (labels == cid)
                stds.append(feats[mask].std(axis=0).mean())
            return np.mean(stds), len(top)

        # --- Phase 1: Find best segment length ---
        seg_lens = [8, 12, 16, 24]
        best_len_score = 999.0
        best_seg_len = 16

        print(f"\n  Phase 1: Find best segment length (k=20, delta mode)")
        for seg_len in seg_lens:
            raws, feats, idxs = _extract_segments(seg_len)
            n_seg = len(feats)
            if n_seg < 40:
                continue
            k_test = min(20, n_seg // 3)
            km = KMeans(n_clusters=k_test, random_state=42, n_init=5)
            labels = km.fit_predict(feats)
            coh, n_valid = _cluster_coherence(feats, labels)
            is_best = coh < best_len_score
            if is_best:
                best_len_score = coh
                best_seg_len = seg_len
            marker = ' <--' if is_best else ''
            print(f"    len={seg_len:>2}: {n_seg} segs, k={k_test}, "
                  f"entropy_normalized={coh:.2f} ({n_valid} valid clusters){marker}")

        print(f"  Best length: {best_seg_len}")

        # --- Phase 2: Iterate k upward until clusters are tight ---
        raws, feats, idxs = _extract_segments(best_seg_len)
        n_seg = len(feats)

        max_k = min(n_seg // MIN_CLUSTER_SIZE, 100)
        k_candidates = [k for k in [10, 15, 20, 30, 40, 50, 75, 100]
                        if k <= max_k and k < n_seg]
        if not k_candidates:
            k_candidates = [max(2, n_seg // MIN_CLUSTER_SIZE)]

        print(f"\n  Phase 2: Iterate k (len={best_seg_len}, {n_seg} segments)")
        print(f"  {'K':>4} {'Coherence':>10} {'ValidClusters':>14} "
              f"{'MinSize':>8} {'Best?':>5}")
        print(f"  {'-'*4} {'-'*10} {'-'*14} {'-'*8} {'-'*5}")

        best_coh = 999.0
        best_config = None
        prev_coh = None

        for k in k_candidates:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(feats)
            coh, n_valid = _cluster_coherence(feats, labels)

            counts = Counter(labels)
            top_sizes = [cnt for _, cnt in counts.most_common(TOP_K)]
            min_top = min(top_sizes) if top_sizes else 0

            is_best = coh < best_coh and n_valid >= min(TOP_K, k)
            if is_best:
                best_coh = coh
                best_config = {
                    'seg_len': best_seg_len, 'k': k,
                    'labels': labels, 'raws': raws, 'feats': feats,
                    'idxs': idxs, 'entropy_normalized': coh,
                }

            marker = ' <--' if is_best else ''
            print(f"  {k:>4} {coh:>10.2f} {n_valid:>14} "
                  f"{min_top:>8}{marker}")

            # Stop if coherence stopped improving (< 5% gain)
            if prev_coh is not None and coh > prev_coh * 0.95 and not is_best:
                if k > 30:  # only stop early after trying enough
                    print(f"  (converged at k={k})")
                    break
            prev_coh = coh

        if best_config is None:
            print("  No valid configuration found. Skipping.")
        else:
            bc = best_config
            print(f"\n  BEST: len={bc['seg_len']}, k={bc['k']}, "
                  f"entropy_normalized={bc['entropy_normalized']:.2f}")

            # Build cluster stats
            labels = bc['labels']
            raws = bc['raws']
            feats = bc['feats']
            counts = Counter(labels)
            # Sort by size, filter to >= MIN_CLUSTER_SIZE
            top_clusters = [(cid, cnt) for cid, cnt in counts.most_common()
                            if cnt >= MIN_CLUSTER_SIZE][:TOP_K]

            print(f"\n  {'Clust':>5} {'N':>5} {'MeanChg':>8} {'StdChg':>7} "
                  f"{'WR':>5} {'Coh':>6}")
            print(f"  {'-'*5} {'-'*5} {'-'*8} {'-'*7} {'-'*5} {'-'*6}")

            cluster_stats = []
            for cid, count in top_clusters:
                mask_c = (labels == cid)
                raw_c = raws[mask_c]
                feat_c = feats[mask_c]

                changes = raw_c[:, -1] - raw_c[:, 0]
                mean_chg = changes.mean()
                std_chg = changes.std()
                wr = (changes > 0).sum() / len(changes) * 100
                coherence = feat_c.std(axis=0).mean()

                cluster_stats.append({
                    'cid': cid, 'count': count, 'mean_chg': mean_chg,
                    'std_chg': std_chg, 'wr': wr, 'entropy_normalized': coherence,
                    'feat': feat_c, 'raw': raw_c,
                })

                print(f"  C{cid:<4} {count:>5} {mean_chg:>+8.1f} {std_chg:>7.1f} "
                      f"{wr:>4.0f}% {coherence:>6.1f}")

            # Plot top 10 clusters
            n_plot = min(TOP_K, len(cluster_stats))
            n_cols = 5
            n_rows = (n_plot + n_cols - 1) // n_cols
            seg_len = bc['seg_len']
            x_seg = np.arange(seg_len)

            fig_h, axes_h = plt.subplots(n_rows, n_cols,
                                          figsize=(4 * n_cols, 3.5 * n_rows),
                                          squeeze=False)

            for k in range(n_plot):
                row, col = divmod(k, n_cols)
                ax = axes_h[row, col]
                cs = cluster_stats[k]
                feat_c = cs['feat']
                n_in = len(feat_c)
                color = cmap(k % 10)

                # Individual traces
                max_plot = min(n_in, 200)
                alpha = max(0.05, min(0.3, 20.0 / max_plot))
                for j in range(max_plot):
                    ax.plot(x_seg, feat_c[j], color=color, alpha=alpha, linewidth=0.5)

                # Mean + std envelope
                mean_f = feat_c.mean(axis=0)
                std_f = feat_c.std(axis=0)
                ax.plot(x_seg, mean_f, color=color, linewidth=3)
                ax.fill_between(x_seg, mean_f - std_f, mean_f + std_f,
                                color=color, alpha=0.15)

                ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

                # Auto-fit y-axis
                all_v = feat_c.flatten()
                y_c = np.mean(all_v)
                y_s = max(np.std(all_v) * 3, 5)
                ax.set_ylim(y_c - y_s, y_c + y_s)

                ax.set_title(f"C{cs['cid']}: n={n_in}, WR={cs['wr']:.0f}%\n"
                             f"coh={cs['entropy_normalized']:.1f}, chg={cs['mean_chg']:+.0f}",
                             fontsize=9)
                if col == 0:
                    ax.set_ylabel('Delta (ticks)')
                if row == n_rows - 1:
                    ax.set_xlabel('Bar')
                ax.grid(True, alpha=0.2)

            for k in range(n_plot, n_rows * n_cols):
                row, col = divmod(k, n_cols)
                axes_h[row, col].set_visible(False)

            fig_h.suptitle(f'Top {n_plot} Shape Clusters (len={seg_len}, k={bc["k"]})\n'
                            f'Delta from entry, entropy_normalized={bc["entropy_normalized"]:.1f}, '
                            f'{len(feats)} segments',
                            fontsize=13)
            plt.tight_layout()
            h_path = os.path.join(PLOTS_DIR, '0f_shape_clusters.png')
            fig_h.savefig(h_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_h)
            print(f"\n  Saved: {h_path}")

    else:
        print(f"  [SKIP] Analysis H (--start {_start_at})")

    if _start_at <= 'I' and 'I' not in _skip_set:
        # =====================================================================
        #  ANALYSIS I: SEED PRIMITIVE SHAPE CLASSIFICATION
        #
        #  Classify every segment against 20 mathematical seed shapes
        #  using Pearson correlation. Threshold 0.85 → shape or NOISE.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS I: SEED PRIMITIVE CLASSIFICATION (20 shapes)")
        print(f"{'='*70}")

        from collections import Counter as Counter_I

        # Use same segment length as Analysis H (best_seg_len), fallback 16
        seed_len = best_seg_len if 'best_seg_len' in dir() else 16
        library = SeedPrimitiveLibrary(N=seed_len)

        print(f"\n  Seed library: {len(library.shapes)} shapes, segment length={seed_len}")
        print(f"  Shapes: {', '.join(sorted(library.shapes.keys()))}")

        # Extract segments (same as Analysis H: every analysis bar)
        analysis_mask_i = price_imr['analysis_mask']
        analysis_idx_i = np.where(analysis_mask_i)[0]

        classifications = []  # (idx, shape_name, correlation, raw_segment)
        for idx in tqdm(analysis_idx_i, desc='  Classifying', ascii=True, dynamic_ncols=True):
            if idx + seed_len > len(close):
                continue
            seg = close[idx : idx + seed_len]
            if len(seg) != seed_len:
                continue
            shape_name, corr = library.classify_trajectory(seg)
            classifications.append((idx, shape_name, corr, seg))

        n_total = len(classifications)
        if n_total == 0:
            print("  No segments to classify. Skipping Analysis I.")
        else:
            # Tally
            shape_counts = Counter_I(c[1] for c in classifications)
            n_noise = shape_counts.get('NOISE', 0)
            n_matched = n_total - n_noise
            noise_pct = n_noise / n_total * 100

            print(f"\n  Total segments: {n_total}")
            print(f"  Matched (corr >= {library.CORR_THRESHOLD}): {n_matched} ({100 - noise_pct:.1f}%)")
            print(f"  NOISE (corr < {library.CORR_THRESHOLD}):    {n_noise} ({noise_pct:.1f}%)")

            # Shape breakdown table
            print(f"\n  {'Shape':<25} {'Count':>6} {'%':>6} {'MeanCorr':>9} "
                  f"{'MeanChg':>8} {'WR':>5}")
            print(f"  {'-'*25} {'-'*6} {'-'*6} {'-'*9} {'-'*8} {'-'*5}")

            shape_stats = []
            for shape_name in sorted(shape_counts.keys()):
                if shape_name == 'NOISE':
                    continue
                entries = [(idx, corr, seg) for idx, sn, corr, seg in classifications
                           if sn == shape_name]
                count = len(entries)
                pct = count / n_total * 100
                mean_corr = np.mean([e[1] for e in entries])
                changes = np.array([e[2][-1] - e[2][0] for e in entries])
                mean_chg = changes.mean()
                wr = (changes > 0).sum() / len(changes) * 100 if len(changes) > 0 else 0

                shape_stats.append({
                    'name': shape_name, 'count': count, 'pct': pct,
                    'mean_corr': mean_corr, 'mean_chg': mean_chg, 'wr': wr,
                    'entries': entries,
                })

                print(f"  {shape_name:<25} {count:>6} {pct:>5.1f}% {mean_corr:>9.3f} "
                      f"{mean_chg:>+8.1f} {wr:>4.0f}%")

            # NOISE deep-dive: what shapes were they closest to?
            if n_noise > 0:
                noise_entries = [(idx, corr, seg) for idx, sn, corr, seg in classifications
                                 if sn == 'NOISE']
                noise_corrs = np.array([e[1] for e in noise_entries])
                noise_chgs = np.array([e[2][-1] - e[2][0] for e in noise_entries])
                noise_wr = (noise_chgs > 0).sum() / len(noise_chgs) * 100
                print(f"  {'NOISE':<25} {n_noise:>6} {noise_pct:>5.1f}% "
                      f"{np.mean(noise_corrs):>9.3f} {noise_chgs.mean():>+8.1f} "
                      f"{noise_wr:>4.0f}%")

                # --- NOISE BREAKDOWN: what is the best-match shape for each noise segment? ---
                print(f"\n  NOISE BREAKDOWN (best-match shape for segments below 0.85):")

                # Re-classify noise to get their best-match shape name
                noise_best_shapes = []
                for idx, corr, seg in noise_entries:
                    mn, mx = seg.min(), seg.max()
                    if mx - mn < 1e-12:
                        noise_best_shapes.append('FLATLINE')
                        continue
                    normed = (seg - mn) / (mx - mn)
                    best_n, best_r = 'UNKNOWN', -999.0
                    for nm, tmpl in library.shapes.items():
                        if tmpl.std() < 1e-12:
                            continue
                        r = np.corrcoef(normed, tmpl)[0, 1]
                        if not np.isnan(r) and r > best_r:
                            best_r = r
                            best_n = nm
                    noise_best_shapes.append(best_n)

                noise_shape_counts = Counter_I(noise_best_shapes)
                print(f"\n  {'Nearest Shape':<25} {'Count':>6} {'%ofNoise':>9} "
                      f"{'MeanCorr':>9} {'MeanChg':>8} {'WR':>5}")
                print(f"  {'-'*25} {'-'*6} {'-'*9} {'-'*9} {'-'*8} {'-'*5}")

                for ns_name, ns_cnt in noise_shape_counts.most_common():
                    ns_mask = [i for i, s in enumerate(noise_best_shapes) if s == ns_name]
                    ns_corrs = noise_corrs[ns_mask]
                    ns_chgs = noise_chgs[ns_mask]
                    ns_wr = (ns_chgs > 0).sum() / len(ns_chgs) * 100 if len(ns_chgs) > 0 else 0
                    ns_pct = ns_cnt / n_noise * 100
                    print(f"  {ns_name:<25} {ns_cnt:>6} {ns_pct:>8.1f}% "
                          f"{ns_corrs.mean():>9.3f} {ns_chgs.mean():>+8.1f} {ns_wr:>4.0f}%")

                # Correlation band breakdown
                print(f"\n  NOISE BY CORRELATION BAND:")
                print(f"  {'Band':<15} {'Count':>6} {'%':>6} {'MeanChg':>8} {'WR':>5} "
                      f"{'StdChg':>7}")
                print(f"  {'-'*15} {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*7}")
                thr = library.CORR_THRESHOLD
                bands = [(thr - 0.05, thr, f'{thr-0.05:.2f}-{thr:.2f}'),
                         (thr - 0.15, thr - 0.05, f'{thr-0.15:.2f}-{thr-0.05:.2f}'),
                         (thr - 0.35, thr - 0.15, f'{thr-0.35:.2f}-{thr-0.15:.2f}'),
                         (-1.0, thr - 0.35, f'<{thr-0.35:.2f}')]
                for lo, hi, label in bands:
                    band_mask = (noise_corrs >= lo) & (noise_corrs < hi)
                    bc = band_mask.sum()
                    if bc == 0:
                        continue
                    b_chgs = noise_chgs[band_mask]
                    b_wr = (b_chgs > 0).sum() / bc * 100
                    print(f"  {label:<15} {bc:>6} {bc/n_noise*100:>5.1f}% "
                          f"{b_chgs.mean():>+8.1f} {b_wr:>4.0f}% {b_chgs.std():>7.1f}")

            # =================================================================
            #  SUB-CLASSIFICATION: within each shape, cluster timing variants
            # =================================================================
            from sklearn.cluster import KMeans as KMeans_sub

            MIN_FOR_SUB = 10   # need at least this many to sub-cluster
            SUB_K = 3           # number of timing sub-types per shape
            x_plot = np.arange(seed_len)

            sub_shapes = [s for s in shape_stats if s['count'] >= MIN_FOR_SUB]

            if sub_shapes:
                print(f"\n  {'='*60}")
                print(f"  SUB-CLASSIFICATION (shapes with n >= {MIN_FOR_SUB})")
                print(f"  {'='*60}")

                all_sub_stats = []  # collect for plotting

                for ss in sub_shapes:
                    entries = ss['entries']
                    # Normalize each segment to 0-1 for shape clustering
                    normed_segs = []
                    raw_segs = []
                    for idx, corr, seg in entries:
                        mn, mx = seg.min(), seg.max()
                        if mx - mn < 1e-12:
                            normed_segs.append(np.zeros(seed_len))
                        else:
                            normed_segs.append((seg - mn) / (mx - mn))
                        raw_segs.append(seg)
                    normed_arr = np.array(normed_segs)
                    raw_arr = np.array(raw_segs)

                    k_use = min(SUB_K, len(entries) // 3)
                    if k_use < 2:
                        k_use = 2

                    km = KMeans_sub(n_clusters=k_use, random_state=42, n_init=10)
                    sub_labels = km.fit_predict(normed_arr)

                    print(f"\n  {ss['name']} (n={ss['count']}) -> {k_use} sub-types:")
                    print(f"    {'Sub':<8} {'N':>4} {'MeanCorr':>9} {'MeanChg':>8} "
                          f"{'WR':>5} {'R2':>6} {'Timing':>20}")
                    print(f"    {'-'*8} {'-'*4} {'-'*9} {'-'*8} {'-'*5} {'-'*6} {'-'*20}")

                    shape_sub_stats = []
                    for si in range(k_use):
                        mask = (sub_labels == si)
                        n_sub = mask.sum()
                        if n_sub == 0:
                            continue
                        sub_normed = normed_arr[mask]
                        sub_raw = raw_arr[mask]
                        sub_corrs = [entries[j][1] for j in range(len(entries)) if mask[j]]

                        # Timing: where does main movement happen?
                        # Measure cumulative change at 33% and 66% of segment
                        centroid = km.cluster_centers_[si]
                        third = seed_len // 3
                        two_third = 2 * seed_len // 3

                        # Movement in first/mid/last third
                        move_early = abs(centroid[third] - centroid[0])
                        move_mid = abs(centroid[two_third] - centroid[third])
                        move_late = abs(centroid[-1] - centroid[two_third])
                        total_move = move_early + move_mid + move_late

                        if total_move < 1e-12:
                            timing = "FLAT"
                        else:
                            pct_early = move_early / total_move
                            pct_late = move_late / total_move
                            if pct_early > 0.45:
                                timing = "EARLY (front-loaded)"
                            elif pct_late > 0.45:
                                timing = "LATE (back-loaded)"
                            else:
                                timing = "STEADY (even)"

                        sub_chgs = sub_raw[:, -1] - sub_raw[:, 0]
                        sub_wr = (sub_chgs > 0).sum() / n_sub * 100
                        sub_mean_chg = sub_chgs.mean()
                        sub_mean_corr = np.mean(sub_corrs)

                        # Goodness of fit: R² = 1 - SS_res / SS_tot
                        ss_res = np.sum((sub_normed - centroid[np.newaxis, :]) ** 2)
                        ss_tot = np.sum((sub_normed - sub_normed.mean()) ** 2)
                        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

                        label = f"{ss['name']}_{si}"
                        shape_sub_stats.append({
                            'label': label, 'n': n_sub, 'timing': timing,
                            'mean_corr': sub_mean_corr, 'mean_chg': sub_mean_chg,
                            'wr': sub_wr, 'r2': r2, 'centroid': centroid,
                            'normed': sub_normed, 'raw': sub_raw,
                        })

                        print(f"    {label:<8} {n_sub:>4} {sub_mean_corr:>9.3f} "
                              f"{sub_mean_chg:>+8.1f} {sub_wr:>4.0f}% {r2:>5.2f} "
                              f"{timing:>20}")

                    all_sub_stats.append({
                        'parent': ss['name'], 'subs': shape_sub_stats,
                        'template': library.shapes[ss['name']],
                    })

                # --- Sub-classification plot: one row per parent shape ---
                if all_sub_stats:
                    n_parents = len(all_sub_stats)
                    max_subs = max(len(a['subs']) for a in all_sub_stats)
                    fig_sub, axes_sub = plt.subplots(
                        n_parents, max_subs,
                        figsize=(5 * max_subs, 4 * n_parents),
                        squeeze=False)

                    sub_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800']

                    for row_i, parent_info in enumerate(all_sub_stats):
                        for col_i, sub in enumerate(parent_info['subs']):
                            ax = axes_sub[row_i, col_i]
                            normed = sub['normed']
                            n_s = len(normed)
                            max_show = min(n_s, 150)
                            alpha = max(0.08, min(0.4, 20.0 / max_show))
                            color = sub_colors[col_i % len(sub_colors)]

                            for j in range(max_show):
                                ax.plot(x_plot, normed[j], color=color,
                                        alpha=alpha, linewidth=0.5)

                            # Centroid
                            ax.plot(x_plot, sub['centroid'], color=color,
                                    linewidth=3, label='Centroid')
                            # Template
                            ax.plot(x_plot, parent_info['template'], color='gray',
                                    linewidth=2, linestyle='--', alpha=0.6,
                                    label='Template')

                            ax.set_title(
                                f"{sub['label']}\n"
                                f"n={sub['n']}, WR={sub['wr']:.0f}%, "
                                f"chg={sub['mean_chg']:+.0f}\n"
                                f"{sub['timing']}",
                                fontsize=9)
                            ax.set_ylim(-0.05, 1.05)
                            ax.grid(True, alpha=0.15)
                            ax.axhline(y=0, color='gray', linewidth=0.3)
                            if col_i == 0:
                                ax.set_ylabel(parent_info['parent'], fontsize=9,
                                              fontweight='bold')
                            if row_i == 0 and col_i == 0:
                                ax.legend(fontsize=7, loc='best')

                        # Hide unused columns
                        for col_i in range(len(parent_info['subs']), max_subs):
                            axes_sub[row_i, col_i].set_visible(False)

                    fig_sub.suptitle(
                        f'Sub-Classification: Timing Variants Within Shapes\n'
                        f'{len(sub_shapes)} shapes sub-clustered (k={SUB_K})',
                        fontsize=13)
                    plt.tight_layout()
                    sub_path = os.path.join(PLOTS_DIR, '0k_sub_classification.png')
                    fig_sub.savefig(sub_path, dpi=150, bbox_inches='tight',
                                    facecolor='white')
                    plt.close(fig_sub)
                    print(f"\n  Saved: {sub_path}")

            # --- Plot 1: Seed template gallery (4x5 grid of all 20 shapes) ---
            fig_seeds, axes_seeds = plt.subplots(4, 5, figsize=(20, 12), squeeze=False)
            sorted_shapes = sorted(library.shapes.keys())
            x_plot = np.arange(seed_len)

            for i, name in enumerate(sorted_shapes):
                row, col = divmod(i, 5)
                ax = axes_seeds[row, col]
                template = library.shapes[name]
                cnt = shape_counts.get(name, 0)
                ax.plot(x_plot, template, color='#2196F3', linewidth=2.5)
                ax.fill_between(x_plot, 0, template, alpha=0.15, color='#2196F3')
                ax.set_title(f'{name}\nn={cnt}', fontsize=9, fontweight='bold')
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlim(0, seed_len - 1)
                ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                ax.axhline(y=1, color='gray', linewidth=0.5, alpha=0.5)
                ax.grid(True, alpha=0.15)
                if col == 0:
                    ax.set_ylabel('Normalized')
                if row == 3:
                    ax.set_xlabel('Bar')

            fig_seeds.suptitle(
                f'Seed Primitive Library ({len(library.shapes)} shapes, N={seed_len})\n'
                f'Matched: {n_matched}/{n_total} ({100 - noise_pct:.1f}%), '
                f'NOISE: {n_noise} ({noise_pct:.1f}%)',
                fontsize=13)
            plt.tight_layout()
            seeds_path = os.path.join(PLOTS_DIR, '0g_seed_templates.png')
            fig_seeds.savefig(seeds_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_seeds)
            print(f"\n  Saved: {seeds_path}")

            # --- Plot 2: Top matched shapes with actual price overlays ---
            # Show up to 10 most-populated shapes (excluding NOISE)
            top_shapes = sorted(shape_stats, key=lambda s: s['count'], reverse=True)[:10]
            n_top = len(top_shapes)

            if n_top > 0:
                n_cols_t = 5
                n_rows_t = (n_top + n_cols_t - 1) // n_cols_t
                fig_match, axes_match = plt.subplots(
                    n_rows_t, n_cols_t, figsize=(4 * n_cols_t, 3.5 * n_rows_t),
                    squeeze=False)

                for k, ss in enumerate(top_shapes):
                    row, col = divmod(k, n_cols_t)
                    ax = axes_match[row, col]

                    entries = ss['entries']
                    # Normalize each segment to 0-1 and overlay
                    max_show = min(len(entries), 200)
                    alpha = max(0.05, min(0.3, 20.0 / max_show))

                    all_normed = []
                    for j in range(max_show):
                        seg = entries[j][2]
                        mn, mx = seg.min(), seg.max()
                        if mx - mn < 1e-12:
                            normed = np.zeros(seed_len)
                        else:
                            normed = (seg - mn) / (mx - mn)
                        all_normed.append(normed)
                        ax.plot(x_plot, normed, color='#1976D2', alpha=alpha,
                                linewidth=0.5)

                    # Template
                    template = library.shapes[ss['name']]
                    ax.plot(x_plot, template, color='#F44336', linewidth=2.5,
                            label='Template')

                    # Mean of actual segments
                    if all_normed:
                        mean_n = np.mean(all_normed, axis=0)
                        ax.plot(x_plot, mean_n, color='#FF9800', linewidth=2,
                                linestyle='--', label='Mean actual')

                    ax.set_title(f"{ss['name']}\nn={ss['count']}, "
                                 f"r={ss['mean_corr']:.2f}, WR={ss['wr']:.0f}%",
                                 fontsize=8)
                    ax.set_ylim(-0.05, 1.05)
                    ax.grid(True, alpha=0.15)
                    if col == 0:
                        ax.set_ylabel('Normalized')
                    if row == n_rows_t - 1:
                        ax.set_xlabel('Bar')
                    if k == 0:
                        ax.legend(fontsize=7, loc='best')

                # Hide unused
                for k in range(n_top, n_rows_t * n_cols_t):
                    row, col = divmod(k, n_cols_t)
                    axes_match[row, col].set_visible(False)

                fig_match.suptitle(
                    f'Top {n_top} Matched Shapes — Actual Segments vs Templates\n'
                    f'{n_total} total segments, {n_matched} matched, '
                    f'{n_noise} NOISE ({noise_pct:.1f}%)',
                    fontsize=13)
                plt.tight_layout()
                match_path = os.path.join(PLOTS_DIR, '0h_seed_matches.png')
                fig_match.savefig(match_path, dpi=150, bbox_inches='tight',
                                  facecolor='white')
                plt.close(fig_match)
                print(f"  Saved: {match_path}")

            # --- Plot 3: Correlation distribution histogram ---
            all_corrs = [c[2] for c in classifications]
            fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
            ax_hist.hist(all_corrs, bins=50, color='#2196F3', alpha=0.7,
                         edgecolor='white')
            ax_hist.axvline(x=library.CORR_THRESHOLD, color='#F44336', linewidth=2,
                            linestyle='--', label=f'Threshold ({library.CORR_THRESHOLD})')
            ax_hist.set_xlabel('Best Pearson Correlation')
            ax_hist.set_ylabel('Segment Count')
            ax_hist.set_title(f'Correlation Distribution (n={n_total})\n'
                              f'Above 0.85: {n_matched} ({100 - noise_pct:.1f}%), '
                              f'Below: {n_noise} ({noise_pct:.1f}%)')
            ax_hist.legend()
            ax_hist.grid(True, alpha=0.15)
            plt.tight_layout()
            hist_path = os.path.join(PLOTS_DIR, '0i_corr_distribution.png')
            fig_hist.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_hist)
            print(f"  Saved: {hist_path}")

            # --- Plot 4: NOISE audit — spaghetti by nearest shape ---
            if n_noise > 0 and 'noise_entries' in dir() and 'noise_best_shapes' in dir():
                # Group noise by nearest shape, show top 6
                noise_by_shape = {}
                for i, (idx, corr, seg) in enumerate(noise_entries):
                    ns = noise_best_shapes[i]
                    if ns not in noise_by_shape:
                        noise_by_shape[ns] = []
                    noise_by_shape[ns].append((corr, seg))

                top_noise_shapes = sorted(noise_by_shape.keys(),
                                          key=lambda s: len(noise_by_shape[s]),
                                          reverse=True)[:6]
                n_ns = len(top_noise_shapes)

                if n_ns > 0:
                    n_cols_ns = min(3, n_ns)
                    n_rows_ns = (n_ns + n_cols_ns - 1) // n_cols_ns
                    fig_noise, axes_noise = plt.subplots(
                        n_rows_ns, n_cols_ns,
                        figsize=(5 * n_cols_ns, 4 * n_rows_ns),
                        squeeze=False)

                    for k, ns_name in enumerate(top_noise_shapes):
                        row, col = divmod(k, n_cols_ns)
                        ax = axes_noise[row, col]
                        ns_segs = noise_by_shape[ns_name]
                        ns_cnt = len(ns_segs)
                        ns_corrs_plot = [s[0] for s in ns_segs]
                        ns_raw = [s[1] for s in ns_segs]

                        # Delta from entry (not normalized — show raw movement)
                        max_show = min(ns_cnt, 150)
                        alpha_ns = max(0.08, min(0.4, 20.0 / max_show))

                        deltas = []
                        for j in range(max_show):
                            d = ns_raw[j] - ns_raw[j][0]
                            deltas.append(d)
                            ax.plot(x_plot, d, color='#9E9E9E', alpha=alpha_ns,
                                    linewidth=0.5)

                        # Mean delta
                        if deltas:
                            mean_d = np.mean(deltas, axis=0)
                            std_d = np.std(deltas, axis=0)
                            ax.plot(x_plot, mean_d, color='#F44336', linewidth=2.5,
                                    label='Mean')
                            ax.fill_between(x_plot, mean_d - std_d, mean_d + std_d,
                                            color='#F44336', alpha=0.12)

                        # Template overlay (normalized to match delta scale)
                        tmpl = library.shapes.get(ns_name)
                        if tmpl is not None and tmpl.std() > 1e-12 and deltas:
                            # Scale template to delta range for visual comparison
                            d_range = np.abs(mean_d).max()
                            if d_range > 0:
                                tmpl_scaled = (tmpl - tmpl.mean()) / tmpl.std() * np.std(deltas)
                                tmpl_scaled = tmpl_scaled - tmpl_scaled[0]
                                ax.plot(x_plot, tmpl_scaled, color='#2196F3',
                                        linewidth=2, linestyle='--', label='Template')

                        ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                        ns_chgs = np.array([s[1][-1] - s[1][0] for s in ns_segs])
                        ns_wr = (ns_chgs > 0).sum() / len(ns_chgs) * 100
                        ax.set_title(
                            f'NOISE nearest: {ns_name}\n'
                            f'n={ns_cnt}, r={np.mean(ns_corrs_plot):.2f}, '
                            f'WR={ns_wr:.0f}%',
                            fontsize=9)
                        ax.grid(True, alpha=0.15)
                        if col == 0:
                            ax.set_ylabel('Delta (ticks)')
                        if row == n_rows_ns - 1:
                            ax.set_xlabel('Bar')
                        if k == 0:
                            ax.legend(fontsize=7, loc='best')

                    for k in range(n_ns, n_rows_ns * n_cols_ns):
                        row, col = divmod(k, n_cols_ns)
                        axes_noise[row, col].set_visible(False)

                    fig_noise.suptitle(
                        f'NOISE Segments by Nearest Shape (n={n_noise}, '
                        f'corr < {library.CORR_THRESHOLD})\nDelta from entry (raw ticks)',
                        fontsize=13)
                    plt.tight_layout()
                    noise_path = os.path.join(PLOTS_DIR, '0j_noise_audit.png')
                    fig_noise.savefig(noise_path, dpi=150, bbox_inches='tight',
                                      facecolor='white')
                    plt.close(fig_noise)
                    print(f"  Saved: {noise_path}")

    else:
        print(f"  [SKIP] Analysis I (--start {_start_at})")

    if _start_at <= 'J' and 'J' not in _skip_set:
        # =====================================================================
        #  ANALYSIS J: RAW DELTA SUB-CLASSIFICATION (ADAPTIVE R² >= 0.80)
        #
        #  Recursive bisecting KMeans: keep splitting any sub-type with
        #  R² < 0.80 until it meets the target or runs out of segments.
        #  Raw delta from entry (ticks, not normalized).
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS J: ADAPTIVE RAW DELTA SUB-CLASSIFICATION (R\u00b2 >= 0.90)")
        print(f"{'='*70}")

        J_R2_TARGET = 0.90
        J_MIN_N = 3   # minimum segments per sub-type
        J_MIN_TOTAL = 10  # need at least this many segments to attempt
        x_j = np.arange(seed_len)
        shade_colors = {'RISE': '#4CAF50', 'DROP': '#F44336', 'HOLD': '#9E9E9E'}

        # Get all shapes with enough segments (sorted by count descending)
        j_shape_names = sorted(
            [sn for sn in set(c[1] for c in classifications) if sn != 'NOISE'],
            key=lambda sn: sum(1 for c in classifications if c[1] == sn),
            reverse=True)

        j_summary = []  # collect stats for console table
        subtype_map = {}  # idx → (shape_name, subtype_id) for Analysis K

        MISFIT_IQR_K = 1.0  # IQR multiplier for outlier detection (aggressive)

        def _norm01(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn) if (mx - mn) > 1e-12 else np.zeros_like(arr)

        def _plot_subtype(ax, sub_deltas, centroid, r2_val, title_str, x_arr,
                          shade_map, r2_target):
            """Plot a single sub-type panel: spaghetti + centroid + inflections."""
            n_sub = len(sub_deltas)
            if n_sub == 0:
                ax.set_visible(False)
                return

            max_show = min(n_sub, 150)
            alpha = max(0.1, min(0.5, 15.0 / max_show))
            for j in range(max_show):
                ax.plot(x_arr, sub_deltas[j], color='#90CAF9', alpha=alpha,
                        linewidth=0.7)

            ax.plot(x_arr, centroid, color='black', linewidth=3, zorder=5)

            inflections, segs_desc = _detect_inflections(centroid)
            for sd in segs_desc:
                clr = shade_map.get(sd['label'], '#9E9E9E')
                ax.axvspan(sd['start'], sd['end'], alpha=0.08, color=clr)
            for bi, lvl in inflections:
                ax.plot(bi, lvl, 'o', color='#F44336', markersize=10, zorder=6)
                ax.annotate(f'({bi},{lvl:+.0f})',
                            xy=(bi, lvl), xytext=(5, 10),
                            textcoords='offset points', fontsize=8,
                            fontweight='bold', color='#D32F2F',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      facecolor='white', alpha=0.8))

            ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
            ax.grid(True, alpha=0.15)

            sub_chgs = sub_deltas[:, -1]
            sub_wr = (sub_chgs > 0).sum() / n_sub * 100 if n_sub > 0 else 0
            r2_color = '#2E7D32' if r2_val >= r2_target else '#C62828'
            ax.set_title(f'{title_str}\n'
                         f'n={n_sub}, WR={sub_wr:.0f}%, '
                         f'chg={sub_chgs.mean():+.0f}t, '
                         f'R\u00b2={r2_val:.2f}',
                         fontsize=10, fontweight='bold', color=r2_color)
            ax.set_xlabel('Bar')

        for target_shape in j_shape_names:
            entries_j = [(idx, corr, seg) for idx, sn, corr, seg in classifications
                         if sn == target_shape]
            n_seg_j = len(entries_j)

            if n_seg_j < J_MIN_TOTAL:
                continue

            deltas_j = np.array([e[2] - e[2][0] for e in entries_j])  # raw ticks

            # --- Pass 1: Adaptive split ---
            j_labels, j_centroids, j_r2s = _adaptive_split(
                deltas_j, r2_target=J_R2_TARGET, min_n=J_MIN_N)

            n_clusters = len(j_centroids)

            # --- Quality gate: IQR outlier detection in raw tick space ---
            # Segments with RMSE > Q3 + 1.5*IQR from centroid are misfits
            misfit_mask = np.zeros(len(deltas_j), dtype=bool)
            for si in range(n_clusters):
                cl_mask = (j_labels == si)
                cl_indices = np.where(cl_mask)[0]
                if len(cl_indices) < 4:
                    continue  # need enough for IQR
                centroid = j_centroids[si]
                rmses = np.array([np.sqrt(np.mean((deltas_j[gi] - centroid) ** 2))
                                  for gi in cl_indices])
                q1, q3 = np.percentile(rmses, [25, 75])
                iqr = q3 - q1
                fence = q3 + MISFIT_IQR_K * iqr
                for i, gi in enumerate(cl_indices):
                    if rmses[i] > fence:
                        misfit_mask[gi] = True

            n_misfits = misfit_mask.sum()

            # Remove misfits from original clusters, recompute centroids/R²
            clean_labels = j_labels.copy()
            clean_labels[misfit_mask] = -1  # mark misfits

            # Rebuild clean centroids and R²s (shape-normalized)
            clean_centroids = []
            clean_r2s = []
            active_ids = []
            for si in range(n_clusters):
                cl_mask = (clean_labels == si)
                n_cl = cl_mask.sum()
                if n_cl < J_MIN_N:
                    clean_labels[cl_mask] = -1  # too small after filtering
                    continue
                sub = deltas_j[cl_mask]
                # Shape R²
                normed = np.array([_norm01(s) for s in sub])
                c_norm = normed.mean(axis=0)
                ss_res = np.sum((normed - c_norm[np.newaxis, :]) ** 2)
                ss_tot = np.sum((normed - normed.mean()) ** 2)
                sr2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
                clean_centroids.append(sub.mean(axis=0))
                clean_r2s.append(sr2)
                active_ids.append(si)

            # --- Pass 2: Reclassify misfits ---
            all_misfit_idx = np.where(clean_labels == -1)[0]
            n_total_misfits = len(all_misfit_idx)
            reclass_centroids = []
            reclass_r2s = []
            reclass_labels = None
            unclass_deltas = None
            n_unclass = 0

            if n_total_misfits >= 2 * J_MIN_N:
                misfit_deltas = deltas_j[all_misfit_idx]
                r_labels, r_centroids, r_r2s = _adaptive_split(
                    misfit_deltas, r2_target=J_R2_TARGET, min_n=J_MIN_N)
                reclass_labels = r_labels
                reclass_centroids = list(r_centroids)
                reclass_r2s = list(r_r2s)
            elif n_total_misfits > 0:
                # Too few to reclassify — all go to UNCLASSIFIED
                unclass_deltas = deltas_j[all_misfit_idx]
                n_unclass = len(unclass_deltas)

            # --- Save subtype mapping for Analysis K ---
            for ci_idx, si in enumerate(active_ids):
                cl_mask = (clean_labels == si)
                for li in np.where(cl_mask)[0]:
                    orig_idx = entries_j[li][0]  # bar index in base_df
                    subtype_map[orig_idx] = (target_shape, ci_idx)
            if reclass_labels is not None:
                n_clean_k = len(clean_centroids)
                for ri in range(len(reclass_centroids)):
                    ri_mask = (reclass_labels == ri)
                    for li in np.where(ri_mask)[0]:
                        orig_idx = entries_j[all_misfit_idx[li]][0]
                        subtype_map[orig_idx] = (target_shape, n_clean_k + ri)

            # --- Build combined plot ---
            n_clean = len(clean_centroids)
            n_reclass = len(reclass_centroids)
            has_unclass = n_unclass > 0 or (n_total_misfits > 0 and n_total_misfits < 2 * J_MIN_N)
            n_total_panels = n_clean + n_reclass + (1 if has_unclass else 0)

            if n_total_panels <= 4:
                n_cols = max(n_total_panels, 1)
                n_rows = 1
            else:
                n_cols = 4
                n_rows = (n_total_panels + 3) // 4

            fig_j, axes_j = plt.subplots(n_rows, n_cols,
                                          figsize=(6 * n_cols, 5 * n_rows),
                                          squeeze=False)

            panel_idx = 0

            # Plot clean sub-types
            for ci_idx, si in enumerate(active_ids):
                row, col = divmod(panel_idx, n_cols)
                ax = axes_j[row, col]
                cl_mask = (clean_labels == si)
                sub_d = deltas_j[cl_mask]
                _plot_subtype(ax, sub_d, clean_centroids[ci_idx],
                              clean_r2s[ci_idx],
                              f'{target_shape} sub-{ci_idx}',
                              x_j, shade_colors, J_R2_TARGET)
                if col == 0:
                    ax.set_ylabel('Delta from entry (ticks)')
                panel_idx += 1

            # Plot reclassified sub-types
            if n_reclass > 0:
                misfit_deltas = deltas_j[all_misfit_idx]
                for ri in range(n_reclass):
                    row, col = divmod(panel_idx, n_cols)
                    ax = axes_j[row, col]
                    ri_mask = (reclass_labels == ri)
                    sub_d = misfit_deltas[ri_mask]
                    _plot_subtype(ax, sub_d, reclass_centroids[ri],
                                  reclass_r2s[ri],
                                  f'{target_shape} reclass-{ri}',
                                  x_j, shade_colors, J_R2_TARGET)
                    # Orange border for reclassified panels
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#FF6F00')
                        spine.set_linewidth(3)
                    if col == 0:
                        ax.set_ylabel('Delta from entry (ticks)')
                    panel_idx += 1

            # Plot UNCLASSIFIED remainder
            if has_unclass:
                row, col = divmod(panel_idx, n_cols)
                ax = axes_j[row, col]
                if n_total_misfits > 0 and n_total_misfits < 2 * J_MIN_N:
                    unc_d = deltas_j[all_misfit_idx]
                else:
                    unc_d = unclass_deltas if unclass_deltas is not None else np.empty((0, seed_len))
                if len(unc_d) > 0:
                    for j in range(min(len(unc_d), 50)):
                        ax.plot(x_j, unc_d[j], color='#FFAB91', alpha=0.5,
                                linewidth=0.8)
                    ax.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
                    ax.grid(True, alpha=0.15)
                    ax.set_title(f'UNCLASSIFIED\nn={len(unc_d)}',
                                 fontsize=10, fontweight='bold', color='#BF360C')
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#BF360C')
                        spine.set_linewidth(3)
                    ax.set_xlabel('Bar')
                    if col == 0:
                        ax.set_ylabel('Delta from entry (ticks)')
                panel_idx += 1

            # Hide unused axes
            for idx in range(panel_idx, n_rows * n_cols):
                row, col = divmod(idx, n_cols)
                axes_j[row, col].set_visible(False)

            all_r2 = clean_r2s + reclass_r2s
            min_r2 = min(all_r2) if all_r2 else 0.0
            met_target = all(r2 >= J_R2_TARGET for r2 in all_r2)
            status = f'ALL >= {J_R2_TARGET}' if met_target else f'min R\u00b2={min_r2:.2f}'

            fig_j.suptitle(
                f'Analysis J: {target_shape} (k={n_clean}+{n_reclass}r'
                f'{f"+{n_unclass}u" if has_unclass else ""})\n'
                f'{n_seg_j} segments, {n_total_misfits} filtered | {status}',
                fontsize=13,
                color='#2E7D32' if met_target else '#C62828')
            plt.tight_layout()
            fname = target_shape.lower().replace(' ', '_')
            j_path = os.path.join(PLOTS_DIR, f'0l_{fname}_raw.png')
            fig_j.savefig(j_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_j)

            j_summary.append((target_shape, n_seg_j, n_clean, n_reclass,
                              n_total_misfits, min_r2, met_target))
            print(f"  {target_shape:<25} n={n_seg_j:>3}  k={n_clean}+{n_reclass}r  "
                  f"filt={n_total_misfits:>2}  min_R\u00b2={min_r2:.2f}  "
                  f"{'OK' if met_target else 'BELOW'}")

        # Summary table
        print(f"\n  {'Shape':<25} {'N':>4} {'k':>3} {'rcl':>4} {'filt':>5} "
              f"{'minR2':>6} {'Status':>8}")
        print(f"  {'-'*25} {'-'*4} {'-'*3} {'-'*4} {'-'*5} {'-'*6} {'-'*8}")
        for sn, n, k, rc, fl, mr2, ok in j_summary:
            print(f"  {sn:<25} {n:>4} {k:>3} {rc:>4} {fl:>5} "
                  f"{mr2:>6.2f} {'OK' if ok else 'BELOW':>8}")

    else:
        print(f"  [SKIP] Analysis J (--start {_start_at})")

    if _start_at <= 'K' and 'K' not in _skip_set:
        # =====================================================================
        #  ANALYSIS K: DIRECTION PREDICTION WITH FRACTAL CONTEXT
        #
        #  Blend 193D fractal properties at entry with shape classification
        #  to predict segment direction (UP/DOWN).
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS K: DIRECTION PREDICTION WITH FRACTAL CONTEXT")
        print(f"{'='*70}")

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, confusion_matrix

        # --- Step 1: Build direction dataset ---
        # Bridge segment bar indices to X rows via timestamp
        ts_to_xrow = {int(ts): i for i, ts in enumerate(sample_ts)}

        X_k_rows = []
        y_k = []
        shapes_k = []

        for idx, sn, corr, seg in classifications:
            if sn == 'NOISE':
                continue
            t = int(timestamps[idx])
            xrow_idx = ts_to_xrow.get(t, -1)
            if xrow_idx < 0 or xrow_idx >= len(X):
                continue
            direction = 1 if seg[-1] > seg[0] else 0
            X_k_rows.append(X[xrow_idx])
            y_k.append(direction)
            shapes_k.append(sn)

        X_k = np.array(X_k_rows)
        y_k = np.array(y_k)
        shapes_k = np.array(shapes_k)

        n_k = len(y_k)
        n_up = (y_k == 1).sum()
        n_down = (y_k == 0).sum()
        baseline = max(n_up, n_down) / n_k * 100 if n_k > 0 else 50.0

        print(f"\n  Dataset: {n_k} segments, {X_k.shape[1]} features")
        print(f"  UP: {n_up} ({n_up/n_k*100:.1f}%)  DOWN: {n_down} ({n_down/n_k*100:.1f}%)")
        print(f"  Baseline (majority class): {baseline:.1f}%")

        if n_k < 50:
            print(f"  SKIP: too few segments ({n_k}) for meaningful model")
        else:
            # --- Step 2: Train classifier ---
            X_train, X_test, y_train, y_test, sh_train, sh_test = train_test_split(
                X_k, y_k, shapes_k, test_size=0.30, random_state=42, stratify=y_k)

            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42)
            clf.fit(X_train, y_train)

            acc_train = accuracy_score(y_train, clf.predict(X_train)) * 100
            acc_test = accuracy_score(y_test, clf.predict(X_test)) * 100
            lift = acc_test - baseline

            print(f"\n  Model: GradientBoosting (200 trees, depth=4)")
            print(f"  Train accuracy: {acc_train:.1f}%")
            print(f"  Test accuracy:  {acc_test:.1f}%")
            print(f"  Lift vs baseline: {lift:+.1f}%")

            # --- Step 3: Per-shape breakdown ---
            y_pred_test = clf.predict(X_test)
            unique_shapes = sorted(set(shapes_k))

            print(f"\n  {'Shape':<25} {'N':>5} {'Base%':>7} {'Model%':>7} {'Lift':>7}")
            print(f"  {'-'*25} {'-'*5} {'-'*7} {'-'*7} {'-'*7}")

            shape_stats_k = []
            for sn in unique_shapes:
                # Baseline from full dataset
                sn_mask_all = (shapes_k == sn)
                sn_n = sn_mask_all.sum()
                sn_base = (y_k[sn_mask_all] == 1).sum() / sn_n * 100
                sn_base = max(sn_base, 100 - sn_base)  # majority class

                # Model accuracy on test set
                sn_mask_test = (sh_test == sn)
                sn_n_test = sn_mask_test.sum()
                if sn_n_test >= 3:
                    sn_acc = accuracy_score(y_test[sn_mask_test],
                                            y_pred_test[sn_mask_test]) * 100
                else:
                    sn_acc = float('nan')

                sn_lift = sn_acc - sn_base if not np.isnan(sn_acc) else float('nan')
                shape_stats_k.append((sn, sn_n, sn_base, sn_acc, sn_lift))

                if not np.isnan(sn_acc):
                    print(f"  {sn:<25} {sn_n:>5} {sn_base:>6.1f}% {sn_acc:>6.1f}% "
                          f"{sn_lift:>+6.1f}%")
                else:
                    print(f"  {sn:<25} {sn_n:>5} {sn_base:>6.1f}%     n/a     n/a")

            # --- Step 4: Feature importance ---
            importances = clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:20]

            print(f"\n  Top 20 fractal features for direction:")
            for rank, fi in enumerate(top_idx):
                fname = col_names[fi] if fi < len(col_names) else f'f{fi}'
                print(f"  {rank+1:>3}. {fname:<30} importance={importances[fi]:.4f}")

            # --- Step 5: Plot ---
            fig_k, axes_k = plt.subplots(2, 2, figsize=(16, 12))

            # Panel 1: Confusion matrix
            ax = axes_k[0, 0]
            cm = confusion_matrix(y_test, y_pred_test)
            im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['DOWN', 'UP'])
            ax.set_yticklabels(['DOWN', 'UP'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=16, fontweight='bold',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')
            ax.set_title(f'Confusion Matrix\nAccuracy={acc_test:.1f}%, '
                         f'Baseline={baseline:.1f}%, Lift={lift:+.1f}%',
                         fontsize=11, fontweight='bold')

            # Panel 2: Per-shape accuracy (baseline vs model)
            ax = axes_k[0, 1]
            valid_stats = [(sn, n, b, a, l) for sn, n, b, a, l in shape_stats_k
                           if not np.isnan(a)]
            valid_stats.sort(key=lambda x: x[4], reverse=True)  # sort by lift
            if valid_stats:
                y_pos = np.arange(len(valid_stats))
                names = [s[0] for s in valid_stats]
                bases = [s[2] for s in valid_stats]
                accs = [s[3] for s in valid_stats]
                ax.barh(y_pos - 0.15, bases, 0.3, color='#90CAF9', label='Baseline')
                ax.barh(y_pos + 0.15, accs, 0.3, color='#2E7D32', label='Model')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(names, fontsize=8)
                ax.set_xlabel('Accuracy %')
                ax.axvline(x=50, color='gray', linewidth=0.5, linestyle='--')
                ax.legend(fontsize=9)
            ax.set_title('Per-Shape Direction Accuracy', fontsize=11, fontweight='bold')

            # Panel 3: Top 20 feature importance
            ax = axes_k[1, 0]
            top20_names = [col_names[i] if i < len(col_names) else f'f{i}'
                           for i in top_idx]
            top20_imp = importances[top_idx]
            y_pos = np.arange(20)
            ax.barh(y_pos, top20_imp[::-1], color='#FF6F00')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top20_names[::-1], fontsize=8)
            ax.set_xlabel('Importance')
            ax.set_title('Top 20 Fractal Features', fontsize=11, fontweight='bold')

            # Panel 4: Per-TF contribution
            ax = axes_k[1, 1]
            _n_tfs = len(TF_HIERARCHY)
            _n_feats = len(FEATURE_NAMES)
            _n_flat = _n_tfs * _n_feats
            tf_contrib = np.zeros(_n_tfs)
            for fi in range(min(_n_flat, len(importances))):
                tf_idx = fi // _n_feats
                if tf_idx < _n_tfs:
                    tf_contrib[tf_idx] += importances[fi]
            # Trailing extras (current_MR, optional L0_time_of_day in v2)
            extras_contrib = importances[_n_flat:].tolist() if len(importances) > _n_flat else []

            tf_labels_plot = list(TF_LABELS[:_n_tfs]) if len(TF_LABELS) >= _n_tfs else \
                [f'TF{i}' for i in range(_n_tfs)]
            x_pos = np.arange(_n_tfs + len(extras_contrib))
            bars = list(tf_contrib) + list(extras_contrib)
            extra_labels = list(_ACTIVE_EXTRAS[:len(extras_contrib)])
            labels = list(tf_labels_plot) + extra_labels
            ax.bar(x_pos, bars, color='#1565C0')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Sum of Importances')
            ax.set_title('Per-Timeframe Contribution to Direction',
                         fontsize=11, fontweight='bold')

            fig_k.suptitle(
                f'Analysis K: Direction Prediction with Fractal Context\n'
                f'{n_k} segments, {X_k.shape[1]} features | '
                f'Accuracy={acc_test:.1f}%, Lift={lift:+.1f}%',
                fontsize=14, fontweight='bold')
            plt.tight_layout()
            k_path = os.path.join(PLOTS_DIR, '0m_direction_prediction.png')
            fig_k.savefig(k_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_k)
            print(f"\n  Saved: {k_path}")

    else:
        print(f"  [SKIP] Analysis K (--start {_start_at})")

    if _start_at <= 'L' and 'L' not in _skip_set:
        # =====================================================================
        #  ANALYSIS L: SIGNED MFE OLS (direction from price prediction)
        #
        #  Fit OLS: Y = signed_MFE = MFE * sign(direction)
        #  If we can predict signed MFE, sign gives direction and magnitude
        #  gives confidence. One model replaces direction + quality classifiers.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS L: SIGNED MFE OLS (DIRECTION FROM PRICE PREDICTION)")
        print(f"  Y = MFE * sign(direction)  |  positive=UP, negative=DOWN")
        print(f"  sign(prediction) -> direction,  |prediction| -> confidence")
        print(f"{'='*70}")

        from sklearn.linear_model import LinearRegression as _LR_L
        from sklearn.preprocessing import StandardScaler as _SS_L
        from sklearn.model_selection import train_test_split as _split_L

        # Bridge oracle bars to X rows via timestamp
        base_ts = base_df['timestamp'].values
        _oracle_ts_set = {}
        for i, bi in enumerate(bar_indices):
            _oracle_ts_set[int(base_ts[bi])] = i

        _l_xrows = []
        _l_smfe = []
        for xi, ts_val in enumerate(sample_ts):
            oi = _oracle_ts_set.get(int(ts_val), -1)
            if oi >= 0:
                _l_xrows.append(xi)
                _sign = 1.0 if directions[oi] == 'LONG' else -1.0
                _l_smfe.append(float(mfes[oi]) * _sign)

        n_l = len(_l_smfe)
        print(f"\n  Matched samples: {n_l} (oracle bars with fractal context)")

        if n_l >= 50:
            X_l = X[_l_xrows]
            Y_l = np.array(_l_smfe)

            n_pos = (Y_l > 0).sum()
            n_neg = (Y_l < 0).sum()
            print(f"  UP (positive): {n_pos} ({n_pos/n_l*100:.1f}%)  "
                  f"DOWN (negative): {n_neg} ({n_neg/n_l*100:.1f}%)")
            print(f"  Y range: [{Y_l.min():.1f}, {Y_l.max():.1f}], "
                  f"mean={Y_l.mean():.2f}, std={Y_l.std():.2f}")

            # Train/test split
            X_tr, X_te, y_tr, y_te = _split_L(X_l, Y_l, test_size=0.30, random_state=42)

            sc_l = _SS_L()
            X_tr_sc = sc_l.fit_transform(X_tr)
            X_te_sc = sc_l.transform(X_te)

            ols_l = _LR_L().fit(X_tr_sc, y_tr)
            pred_tr = ols_l.predict(X_tr_sc)
            pred_te = ols_l.predict(X_te_sc)

            # R-squared on train and test
            r2_tr = ols_l.score(X_tr_sc, y_tr)
            r2_te = ols_l.score(X_te_sc, y_te)
            n_te, k_te = X_te_sc.shape
            adj_r2_te = 1.0 - (1.0 - r2_te) * (n_te - 1) / max(1, n_te - k_te - 1)

            print(f"\n  OLS Signed MFE:")
            print(f"    Train R2:     {r2_tr:.4f}")
            print(f"    Test R2:      {r2_te:.4f}")
            print(f"    Test adj-R2:  {adj_r2_te:.4f}")

            # Direction accuracy: sign(predicted) vs sign(actual)
            dir_pred = np.sign(pred_te)
            dir_actual = np.sign(y_te)
            _nz = dir_actual != 0
            if _nz.sum() > 0:
                dir_correct = (dir_pred[_nz] == dir_actual[_nz]).sum()
                dir_acc = dir_correct / _nz.sum()
                _baseline_l = max((dir_actual[_nz] > 0).sum(), (dir_actual[_nz] < 0).sum()) / _nz.sum()
                _lift_l = dir_acc - _baseline_l
                print(f"\n  Direction from sign(prediction):")
                print(f"    Accuracy: {dir_correct}/{_nz.sum()} = {dir_acc:.1%}")
                print(f"    Baseline (majority): {_baseline_l:.1%}")
                print(f"    Lift: {_lift_l:+.1%}")

                # Confidence gates: only predict when |predicted| > threshold
                print(f"\n  Confidence gates (|predicted signed MFE| > threshold):")
                print(f"  {'Threshold':>10} {'N':>6} {'Accuracy':>10} {'Lift':>8} {'% of data':>10}")
                print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
                for thr in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
                    _cm = (np.abs(pred_te) > thr) & _nz
                    if _cm.sum() > 0:
                        _cc = (dir_pred[_cm] == dir_actual[_cm]).sum()
                        _ca = _cc / _cm.sum()
                        _pct = _cm.sum() / _nz.sum() * 100
                        print(f"  {thr:>10.1f} {_cm.sum():>6} {_ca:>10.1%} "
                              f"{_ca - _baseline_l:>+8.1%} {_pct:>9.1f}%")

                # LONG vs SHORT breakdown
                _pred_long = dir_pred[_nz] > 0
                _actual_long = dir_actual[_nz] > 0
                _long_correct = (_pred_long & _actual_long).sum()
                _long_total = _actual_long.sum()
                _short_correct = (~_pred_long & ~_actual_long).sum()
                _short_total = (~_actual_long).sum()
                print(f"\n  When actual LONG:  {_long_correct}/{_long_total} = "
                      f"{_long_correct/_long_total:.1%}" if _long_total > 0 else "")
                print(f"  When actual SHORT: {_short_correct}/{_short_total} = "
                      f"{_short_correct/_short_total:.1%}" if _short_total > 0 else "")

            # Top features by coefficient magnitude
            coeff_abs = np.abs(ols_l.coef_)
            top_idx = np.argsort(coeff_abs)[::-1][:20]
            all_names = col_names  # 193 features
            print(f"\n  TOP 20 FEATURES (by |coefficient| in signed MFE OLS):")
            print(f"  {'Rank':>4}  {'Feature':<40} {'Coeff':>10} {'|Coeff|':>10}")
            print(f"  {'-'*4}  {'-'*40} {'-'*10} {'-'*10}")
            for rank, fi in enumerate(top_idx, 1):
                fn = all_names[fi] if fi < len(all_names) else f'f{fi}'
                print(f"  {rank:>4}  {fn:<40} {ols_l.coef_[fi]:>+10.4f} {coeff_abs[fi]:>10.4f}")

            # CONCLUSION
            print(f"\n  ANALYSIS L CONCLUSION:")
            if _nz.sum() > 0 and dir_acc > 0.55:
                print(f"  PROMISING: {dir_acc:.1%} direction accuracy from signed MFE OLS.")
                print(f"  The 16D fractal context can predict not just WHERE price is,")
                print(f"  but which WAY it's going and how FAR. One regression gives")
                print(f"  direction (sign) + confidence (magnitude) + TP target (|pred|).")
            elif _nz.sum() > 0 and dir_acc > 0.52:
                print(f"  MARGINAL: {dir_acc:.1%} accuracy, slight lift over baseline.")
                print(f"  May improve with importance weighting or feature selection.")
            else:
                print(f"  INSUFFICIENT: {dir_acc:.1%} accuracy. Signed MFE is not reliably")
                print(f"  predictable from the 192D snapshot. Fall back to balanced")
                print(f"  direction classifier or template DMI side.")

            # ── Plot: Signed MFE — Predicted vs Actual ──────────────────────
            fig_l, axes_l = plt.subplots(2, 2, figsize=(16, 12),
                                          facecolor='white')

            # (0,0) Scatter: predicted vs actual signed MFE, color = actual direction
            ax = axes_l[0, 0]
            _c_long  = '#2196F3'  # blue = LONG (up)
            _c_short = '#F44336'  # red  = SHORT (down)
            _colors_te = np.where(y_te > 0, _c_long, _c_short)
            ax.scatter(y_te, pred_te, c=_colors_te, alpha=0.5, s=20, edgecolors='none')
            _lim = max(abs(y_te).max(), abs(pred_te).max()) * 1.1
            ax.plot([-_lim, _lim], [-_lim, _lim], 'k--', alpha=0.3, lw=1)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
            ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
            # Shade quadrants
            ax.fill_between([-_lim, 0], -_lim, 0, color=_c_short, alpha=0.04)  # correct SHORT
            ax.fill_between([0, _lim], 0, _lim, color=_c_long, alpha=0.04)     # correct LONG
            ax.set_xlabel('Actual Signed MFE', fontsize=10)
            ax.set_ylabel('Predicted Signed MFE', fontsize=10)
            ax.set_title(f'Predicted vs Actual (R\u00b2={r2_te:.3f})', fontsize=11, fontweight='bold')
            # Legend
            from matplotlib.lines import Line2D
            _leg = [Line2D([0], [0], marker='o', color='w', markerfacecolor=_c_long, markersize=8, label='LONG (actual)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=_c_short, markersize=8, label='SHORT (actual)')]
            ax.legend(handles=_leg, loc='upper left', fontsize=9)

            # (0,1) Histogram: predicted signed MFE distribution, stacked by actual direction
            ax = axes_l[0, 1]
            _pred_long_vals  = pred_te[y_te > 0]
            _pred_short_vals = pred_te[y_te < 0]
            _bins = np.linspace(-_lim, _lim, 40)
            ax.hist(_pred_long_vals, bins=_bins, alpha=0.7, color=_c_long, label='Actual LONG', edgecolor='white', lw=0.5)
            ax.hist(_pred_short_vals, bins=_bins, alpha=0.7, color=_c_short, label='Actual SHORT', edgecolor='white', lw=0.5)
            ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
            ax.set_xlabel('Predicted Signed MFE', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title('Prediction Distribution by Actual Direction', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.text(0.02, 0.95, f'LEFT of 0 = model says SHORT\nRIGHT of 0 = model says LONG',
                    transform=ax.transAxes, fontsize=8, va='top', color='gray')

            # (1,0) Confusion matrix as heatmap
            ax = axes_l[1, 0]
            if _nz.sum() > 0:
                _cm_labels = ['SHORT', 'LONG']
                _tp_short = (~_pred_long & ~_actual_long).sum()
                _fp_long  = (_pred_long & ~_actual_long).sum()
                _fn_long  = (~_pred_long & _actual_long).sum()
                _tp_long  = (_pred_long & _actual_long).sum()
                _cm = np.array([[_tp_short, _fp_long], [_fn_long, _tp_long]])
                _im = ax.imshow(_cm, cmap='Blues', aspect='auto')
                ax.set_xticks([0, 1]); ax.set_xticklabels(_cm_labels, fontsize=10)
                ax.set_yticks([0, 1]); ax.set_yticklabels(_cm_labels, fontsize=10)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                for _ri in range(2):
                    for _ci in range(2):
                        _val = _cm[_ri, _ci]
                        _clr = 'white' if _val > _cm.max() * 0.5 else 'black'
                        ax.text(_ci, _ri, str(_val), ha='center', va='center',
                                fontsize=16, fontweight='bold', color=_clr)
                ax.set_title(f'Direction Confusion Matrix\nAccuracy={dir_acc:.1%}, Lift={_lift_l:+.1%}',
                            fontsize=11, fontweight='bold')

            # (1,1) Confidence gate curve
            ax = axes_l[1, 1]
            _thrs = np.linspace(0, np.percentile(np.abs(pred_te), 95), 30)
            _accs = []
            _ns = []
            for _t in _thrs:
                _m = (np.abs(pred_te) > _t) & _nz
                if _m.sum() >= 5:
                    _accs.append((dir_pred[_m] == dir_actual[_m]).sum() / _m.sum() * 100)
                    _ns.append(_m.sum() / _nz.sum() * 100)
                else:
                    _accs.append(np.nan)
                    _ns.append(0)
            ax.plot(_thrs, _accs, color='#2196F3', lw=2, label='Accuracy %')
            ax.axhline(_baseline_l * 100, color='gray', ls='--', lw=1, alpha=0.7, label=f'Baseline {_baseline_l:.0%}')
            ax.set_xlabel('|Predicted Signed MFE| Threshold', fontsize=10)
            ax.set_ylabel('Direction Accuracy %', fontsize=10)
            ax.set_title('Confidence Gate: Accuracy vs Threshold', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax2 = ax.twinx()
            ax2.fill_between(_thrs, 0, _ns, alpha=0.15, color='orange')
            ax2.set_ylabel('% of Data Remaining', fontsize=9, color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            fig_l.suptitle(
                f'Analysis L: Signed MFE OLS — Direction from Price Prediction\n'
                f'{n_l} samples | Accuracy={dir_acc:.1%} | Lift={_lift_l:+.1%} | '
                f'LONG: {n_pos} ({n_pos/n_l*100:.0f}%)  SHORT: {n_neg} ({n_neg/n_l*100:.0f}%)',
                fontsize=13, fontweight='bold')
            plt.tight_layout()
            l_path = os.path.join(PLOTS_DIR, '0n_signed_mfe_direction.png')
            fig_l.savefig(l_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_l)
            print(f"\n  Saved: {l_path}")

            # ── Plot 2: Price chart with LONG/SHORT segment overlay ─────────
            # Predict signed MFE for ALL matched samples (not just test split)
            X_l_all_sc = sc_l.transform(X_l)
            pred_all = ols_l.predict(X_l_all_sc)
            pred_dir_all = np.sign(pred_all)

            # Map back to bar indices in base_df
            _matched_bar_idx = [bar_indices[_oracle_ts_set[int(sample_ts[xi])]]
                                for xi in _l_xrows]
            _matched_bar_idx = np.array(_matched_bar_idx)

            from datetime import datetime, timezone as _tz_l
            close_all = base_df['close'].values.astype(float)
            ts_all = base_df['timestamp'].values

            fig_p, ax_p = plt.subplots(1, 1, figsize=(20, 7), facecolor='white')

            # Plot full price line in gray
            _x_dates = [datetime.fromtimestamp(int(t), tz=_tz_l.utc) for t in ts_all]
            ax_p.plot(_x_dates, close_all, color='#BDBDBD', lw=0.8, alpha=0.6, zorder=1)

            # Overlay colored segments at each prediction point
            _seg_half = max(1, len(close_all) // 500)  # adaptive segment width
            for i, bi in enumerate(_matched_bar_idx):
                _s = max(0, bi - _seg_half)
                _e = min(len(close_all), bi + _seg_half + 1)
                _seg_x = _x_dates[_s:_e]
                _seg_y = close_all[_s:_e]
                if len(_seg_x) < 2:
                    continue
                _color = '#2196F3' if pred_dir_all[i] > 0 else '#F44336'
                _alpha = min(1.0, 0.3 + abs(pred_all[i]) / 100.0)  # stronger prediction = more opaque
                ax_p.plot(_seg_x, _seg_y, color=_color, lw=2.0, alpha=_alpha, zorder=2)

            # Mark correct/wrong with small dots
            for i, bi in enumerate(_matched_bar_idx):
                _actual_sign = 1.0 if directions[_oracle_ts_set[int(sample_ts[_l_xrows[i]])]] == 'LONG' else -1.0
                _correct = (pred_dir_all[i] == _actual_sign)
                if not _correct:
                    ax_p.plot(_x_dates[bi], close_all[bi], 'x', color='black',
                             markersize=4, alpha=0.5, zorder=3)

            ax_p.set_xlabel('Time', fontsize=10)
            ax_p.set_ylabel('Price', fontsize=10)
            ax_p.set_title(
                f'Price with Predicted Direction Overlay\n'
                f'Blue = LONG prediction | Red = SHORT prediction | X = wrong direction',
                fontsize=12, fontweight='bold')
            from matplotlib.lines import Line2D as _Line2D_p
            _leg_p = [_Line2D_p([0], [0], color='#2196F3', lw=2, label='Predicted LONG'),
                      _Line2D_p([0], [0], color='#F44336', lw=2, label='Predicted SHORT'),
                      _Line2D_p([0], [0], marker='x', color='black', lw=0, markersize=6, label='Wrong direction')]
            ax_p.legend(handles=_leg_p, loc='upper left', fontsize=9)
            fig_p.autofmt_xdate()
            plt.tight_layout()
            p_path = os.path.join(PLOTS_DIR, '0o_price_direction_overlay.png')
            fig_p.savefig(p_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_p)
            print(f"  Saved: {p_path}")

        else:
            print(f"  SKIP: too few matched samples ({n_l}) for meaningful analysis")

    else:
        print(f"  [SKIP] Analysis L (--start {_start_at})")

    if _start_at <= 'M' and 'M' not in _skip_set:
        # =====================================================================
        #  ANALYSIS M: NEXT-PRICE FORECAST (direction from delta)
        #
        #  Y = close[t+1]  given X = fractal context at time t
        #  direction = sign(predicted_next - actual_current)
        #  No regime labels, no oracle direction, no MFE lookahead.
        #  Just: "where will price be next?"
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS M: NEXT-PRICE FORECAST")
        print(f"  Y = close[t+1]  |  X = 192D fractal context at time t")
        print(f"  direction = sign(predicted_next - current_price)")
        print(f"{'='*70}")

        from sklearn.linear_model import LinearRegression as _LR_M
        from sklearn.preprocessing import StandardScaler as _SS_M

        # Build consecutive pairs: X[t] -> Y = close[t+1]
        n_m = len(X) - 1  # need t+1 for each t
        print(f"\n  Consecutive pairs: {n_m}")

        if n_m >= 50:
            X_m = X[:-1]             # features at time t
            Y_next_m = Y_p[1:]       # close[t+1]
            Y_curr_m = Y_p[:-1]      # close[t]
            Y_delta_m = Y_next_m - Y_curr_m  # actual price change

            n_up_m = (Y_delta_m > 0).sum()
            n_down_m = (Y_delta_m < 0).sum()
            n_flat_m = (Y_delta_m == 0).sum()
            print(f"  Actual direction: UP={n_up_m} ({n_up_m/n_m*100:.1f}%)  "
                  f"DOWN={n_down_m} ({n_down_m/n_m*100:.1f}%)  FLAT={n_flat_m}")
            print(f"  Delta range: [{Y_delta_m.min():.1f}, {Y_delta_m.max():.1f}], "
                  f"mean={Y_delta_m.mean():.2f}, std={Y_delta_m.std():.2f}")

            # Train/test split (chronological: first 70% train, last 30% test)
            _split_m = int(n_m * 0.70)
            Xm_tr, Xm_te = X_m[:_split_m], X_m[_split_m:]
            ym_next_tr, ym_next_te = Y_next_m[:_split_m], Y_next_m[_split_m:]
            ym_curr_tr, ym_curr_te = Y_curr_m[:_split_m], Y_curr_m[_split_m:]
            ym_delta_te = Y_delta_m[_split_m:]

            sc_m = _SS_M()
            Xm_tr_sc = sc_m.fit_transform(Xm_tr)
            Xm_te_sc = sc_m.transform(Xm_te)

            # Model: predict close[t+1] from features[t]
            ols_m = _LR_M().fit(Xm_tr_sc, ym_next_tr)
            pred_next_tr_m = ols_m.predict(Xm_tr_sc)
            pred_next_te_m = ols_m.predict(Xm_te_sc)

            r2_tr_m = ols_m.score(Xm_tr_sc, ym_next_tr)
            r2_te_m = ols_m.score(Xm_te_sc, ym_next_te)
            n_te_m, k_te_m = Xm_te_sc.shape
            adj_r2_te_m = 1.0 - (1.0 - r2_te_m) * (n_te_m - 1) / max(1, n_te_m - k_te_m - 1)

            print(f"\n  OLS Next-Price Forecast:")
            print(f"    Train R2:     {r2_tr_m:.4f}")
            print(f"    Test R2:      {r2_te_m:.4f}")
            print(f"    Test adj-R2:  {adj_r2_te_m:.4f}")

            # Direction = sign(predicted_next - actual_current)
            pred_delta_m = pred_next_te_m - ym_curr_te
            actual_delta_m = ym_delta_te

            # Residual analysis
            residuals_m = pred_next_te_m - ym_next_te
            print(f"\n  Residuals: mean={residuals_m.mean():.2f}, std={residuals_m.std():.2f}")
            print(f"  Actual moves: mean={np.mean(np.abs(actual_delta_m)):.2f}, std={np.std(actual_delta_m):.2f}")
            snr_m = np.mean(np.abs(actual_delta_m)) / residuals_m.std() if residuals_m.std() > 0 else 0
            print(f"  Signal-to-noise: {snr_m:.3f} "
                  f"({'good' if snr_m > 1.5 else 'marginal' if snr_m > 0.8 else 'poor'})")

            # Direction accuracy
            dir_pred_m = np.sign(pred_delta_m)
            dir_actual_m = np.sign(actual_delta_m)
            _nz_m = dir_actual_m != 0
            dir_acc_m = 0.0
            _baseline_m = 0.5
            _lift_m = 0.0
            if _nz_m.sum() > 0:
                dir_correct_m = (dir_pred_m[_nz_m] == dir_actual_m[_nz_m]).sum()
                dir_acc_m = dir_correct_m / _nz_m.sum()
                _baseline_m = max((dir_actual_m[_nz_m] > 0).sum(), (dir_actual_m[_nz_m] < 0).sum()) / _nz_m.sum()
                _lift_m = dir_acc_m - _baseline_m
                print(f"\n  Direction = sign(predicted_next - current):")
                print(f"    Accuracy: {dir_correct_m}/{_nz_m.sum()} = {dir_acc_m:.1%}")
                print(f"    Baseline (majority): {_baseline_m:.1%}")
                print(f"    Lift: {_lift_m:+.1%}")

                # Confidence gates
                print(f"\n  Confidence gates (|predicted delta| > threshold):")
                print(f"  {'Threshold':>10} {'N':>6} {'Accuracy':>10} {'Lift':>8} {'% of data':>10}")
                print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
                for thr in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
                    _cm_m = (np.abs(pred_delta_m) > thr) & _nz_m
                    if _cm_m.sum() > 0:
                        _cc_m = (dir_pred_m[_cm_m] == dir_actual_m[_cm_m]).sum()
                        _ca_m = _cc_m / _cm_m.sum()
                        _pct_m = _cm_m.sum() / _nz_m.sum() * 100
                        print(f"  {thr:>10.1f} {_cm_m.sum():>6} {_ca_m:>10.1%} "
                              f"{_ca_m - _baseline_m:>+8.1%} {_pct_m:>9.1f}%")

                # UP vs DOWN breakdown
                _pred_up_m = dir_pred_m[_nz_m] > 0
                _actual_up_m = dir_actual_m[_nz_m] > 0
                _up_correct_m = (_pred_up_m & _actual_up_m).sum()
                _up_total_m = _actual_up_m.sum()
                _down_correct_m = (~_pred_up_m & ~_actual_up_m).sum()
                _down_total_m = (~_actual_up_m).sum()
                if _up_total_m > 0:
                    print(f"\n  When actual UP:   {_up_correct_m}/{_up_total_m} = "
                          f"{_up_correct_m/_up_total_m:.1%}")
                if _down_total_m > 0:
                    print(f"  When actual DOWN: {_down_correct_m}/{_down_total_m} = "
                          f"{_down_correct_m/_down_total_m:.1%}")

            # CONCLUSION
            print(f"\n  ANALYSIS M CONCLUSION:")
            if _nz_m.sum() > 0 and dir_acc_m > 0.55:
                print(f"  PROMISING: {dir_acc_m:.1%} direction accuracy.")
                print(f"  Predicting next price from current fractal context works.")
                print(f"  direction = sign(predicted - current), confidence = |delta|")
            elif _nz_m.sum() > 0 and dir_acc_m > 0.52:
                print(f"  MARGINAL: {dir_acc_m:.1%} accuracy, slight lift over baseline.")
                if snr_m < 1.0:
                    print(f"  SNR={snr_m:.2f}: residual noise > typical moves.")
            else:
                print(f"  INSUFFICIENT: {dir_acc_m:.1%}. Next-bar prediction residuals")
                print(f"  overwhelm the directional signal at this timeframe.")

            # ── Table: Actual vs Predicted (test set) ───────────────────────
            import csv as _csv_m
            from datetime import datetime as _dt_m, timezone as _tz_mt

            # Build rows for test set
            _te_indices = list(range(_split_m, n_m))
            _table_rows = []
            for j, idx in enumerate(_te_indices):
                _ts_val = int(sample_ts[idx])
                _t_str = _dt_m.fromtimestamp(_ts_val, tz=_tz_mt.utc).strftime('%Y-%m-%d %H:%M')
                _curr = float(Y_curr_m[_split_m + j])
                _actual_next = float(Y_next_m[_split_m + j])
                _pred_next = float(pred_next_te_m[j])
                _act_delta = float(actual_delta_m[j])
                _prd_delta = float(pred_delta_m[j])
                _pred_dir = 'UP' if _prd_delta > 0 else 'DOWN'
                _act_dir = 'UP' if _act_delta > 0 else ('DOWN' if _act_delta < 0 else 'FLAT')
                _correct = 'Y' if _pred_dir == _act_dir else 'N'
                _table_rows.append((_t_str, _curr, _actual_next, _pred_next,
                                    _act_delta, _prd_delta, _pred_dir, _act_dir, _correct))

            # Console preview (first 30 + last 10)
            print(f"\n  {'='*95}")
            print(f"  PREDICTION TABLE (test set: {len(_table_rows)} rows)")
            print(f"  {'='*95}")
            _hdr = f"  {'Timestamp':<17} {'Current':>10} {'Actual':>10} {'Predicted':>10} {'Act.D':>8} {'Pred.D':>8} {'Dir':>5} {'Act':>5} {'OK':>3}"
            print(_hdr)
            print(f"  {'-'*95}")
            _show = _table_rows[:30] + ([('  ...', '', '', '', '', '', '', '', '')] if len(_table_rows) > 40 else []) + _table_rows[-10:] if len(_table_rows) > 40 else _table_rows
            for r in _show:
                if r[1] == '':
                    print(f"  {r[0]}")
                else:
                    print(f"  {r[0]:<17} {r[1]:>10.2f} {r[2]:>10.2f} {r[3]:>10.2f} "
                          f"{r[4]:>+8.2f} {r[5]:>+8.2f} {r[6]:>5} {r[7]:>5} {r[8]:>3}")

            # Save full CSV
            _csv_path = os.path.join(PLOTS_DIR, 'analysis_m_predictions.csv')
            with open(_csv_path, 'w', newline='', encoding='utf-8') as _cf:
                _w = _csv_m.writer(_cf)
                _w.writerow(['timestamp', 'current_price', 'actual_next', 'predicted_next',
                             'actual_delta', 'predicted_delta', 'predicted_dir', 'actual_dir', 'correct'])
                for r in _table_rows:
                    _w.writerow(r)
            print(f"\n  Full table saved: {_csv_path} ({len(_table_rows)} rows)")

            # Summary stats
            _n_correct = sum(1 for r in _table_rows if r[8] == 'Y')
            _mae = np.mean(np.abs(pred_next_te_m - ym_next_te))
            _rmse = np.sqrt(np.mean((pred_next_te_m - ym_next_te) ** 2))
            print(f"  MAE: {_mae:.2f}  |  RMSE: {_rmse:.2f}  |  "
                  f"Direction: {_n_correct}/{len(_table_rows)} correct")

            # ── Plot 1: 4-panel analysis ────────────────────────────────────
            fig_m, axes_m = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
            _c_up  = '#2196F3'
            _c_dn = '#F44336'

            # (0,0) Scatter: predicted delta vs actual delta
            ax = axes_m[0, 0]
            _colors_m = np.where(actual_delta_m > 0, _c_up, _c_dn)
            ax.scatter(actual_delta_m, pred_delta_m, c=_colors_m, alpha=0.4, s=15, edgecolors='none')
            _lim_m = max(np.percentile(np.abs(actual_delta_m), 99),
                         np.percentile(np.abs(pred_delta_m), 99)) * 1.1
            ax.plot([-_lim_m, _lim_m], [-_lim_m, _lim_m], 'k--', alpha=0.3, lw=1)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
            ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
            ax.fill_between([-_lim_m, 0], -_lim_m, 0, color=_c_dn, alpha=0.04)
            ax.fill_between([0, _lim_m], 0, _lim_m, color=_c_up, alpha=0.04)
            ax.set_xlabel('Actual Delta (close[t+1] - close[t])', fontsize=10)
            ax.set_ylabel('Predicted Delta (pred_next - current)', fontsize=10)
            ax.set_title(f'Predicted vs Actual Price Change', fontsize=11, fontweight='bold')
            from matplotlib.lines import Line2D as _L2M
            ax.legend(handles=[
                _L2M([0], [0], marker='o', color='w', markerfacecolor=_c_up, markersize=8, label='UP'),
                _L2M([0], [0], marker='o', color='w', markerfacecolor=_c_dn, markersize=8, label='DOWN'),
            ], loc='upper left', fontsize=9)

            # (0,1) Histogram: predicted delta distribution by actual direction
            ax = axes_m[0, 1]
            _bins_m = np.linspace(-_lim_m, _lim_m, 50)
            ax.hist(pred_delta_m[actual_delta_m > 0], bins=_bins_m, alpha=0.7, color=_c_up,
                    label='Actual UP', edgecolor='white', lw=0.5)
            ax.hist(pred_delta_m[actual_delta_m < 0], bins=_bins_m, alpha=0.7, color=_c_dn,
                    label='Actual DOWN', edgecolor='white', lw=0.5)
            ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
            ax.set_xlabel('Predicted Delta', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title('Prediction Distribution by Actual Direction', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)

            # (1,0) Confusion matrix
            ax = axes_m[1, 0]
            if _nz_m.sum() > 0:
                _tp_s_m = (~_pred_up_m & ~_actual_up_m).sum()
                _fp_u_m = (_pred_up_m & ~_actual_up_m).sum()
                _fn_u_m = (~_pred_up_m & _actual_up_m).sum()
                _tp_u_m = (_pred_up_m & _actual_up_m).sum()
                _cm_mat = np.array([[_tp_s_m, _fp_u_m], [_fn_u_m, _tp_u_m]])
                ax.imshow(_cm_mat, cmap='Blues', aspect='auto')
                ax.set_xticks([0, 1]); ax.set_xticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_yticks([0, 1]); ax.set_yticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                for _ri in range(2):
                    for _ci in range(2):
                        _v = _cm_mat[_ri, _ci]
                        ax.text(_ci, _ri, str(_v), ha='center', va='center',
                                fontsize=16, fontweight='bold',
                                color='white' if _v > _cm_mat.max() * 0.5 else 'black')
                ax.set_title(f'Direction Confusion Matrix\nAcc={dir_acc_m:.1%}, Lift={_lift_m:+.1%}',
                            fontsize=11, fontweight='bold')

            # (1,1) Confidence gate curve
            ax = axes_m[1, 1]
            _thrs_m = np.linspace(0, np.percentile(np.abs(pred_delta_m), 95), 30)
            _accs_m, _ns_m = [], []
            for _t in _thrs_m:
                _mm = (np.abs(pred_delta_m) > _t) & _nz_m
                if _mm.sum() >= 5:
                    _accs_m.append((dir_pred_m[_mm] == dir_actual_m[_mm]).sum() / _mm.sum() * 100)
                    _ns_m.append(_mm.sum() / _nz_m.sum() * 100)
                else:
                    _accs_m.append(np.nan); _ns_m.append(0)
            ax.plot(_thrs_m, _accs_m, color='#2196F3', lw=2, label='Accuracy %')
            ax.axhline(_baseline_m * 100, color='gray', ls='--', lw=1, alpha=0.7,
                       label=f'Baseline {_baseline_m:.0%}')
            ax.set_xlabel('|Predicted Delta| Threshold', fontsize=10)
            ax.set_ylabel('Direction Accuracy %', fontsize=10)
            ax.set_title('Confidence Gate: Accuracy vs Threshold', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax2_m = ax.twinx()
            ax2_m.fill_between(_thrs_m, 0, _ns_m, alpha=0.15, color='orange')
            ax2_m.set_ylabel('% Data Remaining', fontsize=9, color='orange')
            ax2_m.tick_params(axis='y', labelcolor='orange')

            fig_m.suptitle(
                f'Analysis M: Next-Price Forecast — Direction from Delta\n'
                f'{n_m} pairs | Acc={dir_acc_m:.1%} | Lift={_lift_m:+.1%} | SNR={snr_m:.2f} | '
                f'UP: {n_up_m} ({n_up_m/n_m*100:.0f}%)  DOWN: {n_down_m} ({n_down_m/n_m*100:.0f}%)',
                fontsize=13, fontweight='bold')
            plt.tight_layout()
            m_path = os.path.join(PLOTS_DIR, '0p_next_price_forecast.png')
            fig_m.savefig(m_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_m)
            print(f"\n  Saved: {m_path}")

            # ── Plot 2: Price with direction overlay ────────────────────────
            Xm_all_sc = sc_m.transform(X_m)
            pred_next_all_m = ols_m.predict(Xm_all_sc)
            pred_delta_all_m = pred_next_all_m - Y_curr_m

            from datetime import datetime, timezone as _tz_m
            close_all_m = base_df['close'].values.astype(float)
            ts_all_m = base_df['timestamp'].values

            _ts_to_bi_m = {int(ts_all_m[i]): i for i in range(len(ts_all_m))}
            _sample_bi_m = [_ts_to_bi_m.get(int(t), -1) for t in sample_ts[:-1]]

            fig_pm, ax_pm = plt.subplots(1, 1, figsize=(20, 7), facecolor='white')
            _x_dates_m = [datetime.fromtimestamp(int(t), tz=_tz_m.utc) for t in ts_all_m]
            ax_pm.plot(_x_dates_m, close_all_m, color='#BDBDBD', lw=0.8, alpha=0.6, zorder=1)

            _seg_half_m = max(1, len(close_all_m) // 500)
            for i, bi in enumerate(_sample_bi_m):
                if bi < 0:
                    continue
                _s = max(0, bi - _seg_half_m)
                _e = min(len(close_all_m), bi + _seg_half_m + 1)
                _seg_x = _x_dates_m[_s:_e]
                _seg_y = close_all_m[_s:_e]
                if len(_seg_x) < 2:
                    continue
                _color = _c_up if pred_delta_all_m[i] > 0 else _c_dn
                _alpha = min(1.0, 0.3 + abs(pred_delta_all_m[i]) / 50.0)
                ax_pm.plot(_seg_x, _seg_y, color=_color, lw=2.0, alpha=_alpha, zorder=2)

                # X for wrong direction
                if Y_delta_m[i] != 0 and np.sign(pred_delta_all_m[i]) != np.sign(Y_delta_m[i]):
                    ax_pm.plot(_x_dates_m[bi], close_all_m[bi], 'x', color='black',
                             markersize=4, alpha=0.5, zorder=3)

            ax_pm.set_xlabel('Time', fontsize=10)
            ax_pm.set_ylabel('Price', fontsize=10)
            ax_pm.set_title(
                f'Price with Predicted Direction (Next-Bar Forecast)\n'
                f'Blue = predicted UP | Red = predicted DOWN | X = wrong',
                fontsize=12, fontweight='bold')
            from matplotlib.lines import Line2D as _L2M2
            ax_pm.legend(handles=[
                _L2M2([0], [0], color=_c_up, lw=2, label='Predicted UP'),
                _L2M2([0], [0], color=_c_dn, lw=2, label='Predicted DOWN'),
                _L2M2([0], [0], marker='x', color='black', lw=0, markersize=6, label='Wrong'),
            ], loc='upper left', fontsize=9)
            fig_pm.autofmt_xdate()
            plt.tight_layout()
            pm_path = os.path.join(PLOTS_DIR, '0q_next_price_overlay.png')
            fig_pm.savefig(pm_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_pm)
            print(f"  Saved: {pm_path}")

        else:
            print(f"  SKIP: too few samples ({n_m}) for meaningful analysis")

    else:
        print(f"  [SKIP] Analysis M (--start {_start_at})")

    if _start_at <= 'N' and 'N' not in _skip_set:
        # =====================================================================
        #  ANALYSIS N: DELTA-DIRECT FORECAST (MR-centered)
        #
        #  Y = close[t+1] - close[t]  (the moving range / bar delta)
        #  Start from zero — predict the MOVEMENT, not the level.
        #  This removes the massive level variance that drowns direction signal.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS N: DELTA-DIRECT FORECAST (MR-CENTERED)")
        print(f"  Y = close[t+1] - close[t]  |  X = 192D fractal context at time t")
        print(f"  Predict the MOVEMENT directly, not the absolute price")
        print(f"{'='*70}")

        from sklearn.linear_model import LinearRegression as _LR_N
        from sklearn.preprocessing import StandardScaler as _SS_N

        n_n = len(X) - 1
        print(f"\n  Consecutive pairs: {n_n}")

        if n_n >= 50:
            X_n = X[:-1]                           # features at time t
            Y_delta_n = Y_p[1:] - Y_p[:-1]         # target = delta = close[t+1] - close[t]
            Y_curr_n = Y_p[:-1]                     # for reference only

            n_up_n = (Y_delta_n > 0).sum()
            n_down_n = (Y_delta_n < 0).sum()
            n_flat_n = (Y_delta_n == 0).sum()
            print(f"  Actual direction: UP={n_up_n} ({n_up_n/n_n*100:.1f}%)  "
                  f"DOWN={n_down_n} ({n_down_n/n_n*100:.1f}%)  FLAT={n_flat_n}")
            print(f"  Delta range: [{Y_delta_n.min():.2f}, {Y_delta_n.max():.2f}], "
                  f"mean={Y_delta_n.mean():.4f}, std={Y_delta_n.std():.2f}")

            # Chronological 70/30 split
            _split_n = int(n_n * 0.70)
            Xn_tr, Xn_te = X_n[:_split_n], X_n[_split_n:]
            yn_tr, yn_te = Y_delta_n[:_split_n], Y_delta_n[_split_n:]
            yn_curr_te = Y_curr_n[_split_n:]

            sc_n = _SS_N()
            Xn_tr_sc = sc_n.fit_transform(Xn_tr)
            Xn_te_sc = sc_n.transform(Xn_te)

            # OLS: predict delta directly
            ols_n = _LR_N().fit(Xn_tr_sc, yn_tr)
            pred_delta_n = ols_n.predict(Xn_te_sc)

            r2_tr_n = ols_n.score(Xn_tr_sc, yn_tr)
            r2_te_n = ols_n.score(Xn_te_sc, yn_te)
            n_te_n, k_te_n = Xn_te_sc.shape
            adj_r2_te_n = 1.0 - (1.0 - r2_te_n) * (n_te_n - 1) / max(1, n_te_n - k_te_n - 1)

            print(f"\n  OLS Delta-Direct:")
            print(f"    Train R2:     {r2_tr_n:.4f}")
            print(f"    Test R2:      {r2_te_n:.4f}")
            print(f"    Test adj-R2:  {adj_r2_te_n:.4f}")

            # Residual analysis
            residuals_n = pred_delta_n - yn_te
            print(f"\n  Residuals: mean={residuals_n.mean():.4f}, std={residuals_n.std():.2f}")
            print(f"  Actual deltas: mean={np.mean(np.abs(yn_te)):.2f}, std={np.std(yn_te):.2f}")
            snr_n = np.mean(np.abs(yn_te)) / residuals_n.std() if residuals_n.std() > 0 else 0
            print(f"  Signal-to-noise: {snr_n:.3f} "
                  f"({'good' if snr_n > 1.5 else 'marginal' if snr_n > 0.8 else 'poor'})")

            # Direction accuracy: sign(predicted_delta) vs sign(actual_delta)
            dir_pred_n = np.sign(pred_delta_n)
            dir_actual_n = np.sign(yn_te)
            _nz_n = dir_actual_n != 0
            dir_acc_n = 0.0
            _baseline_n = 0.5
            _lift_n = 0.0
            if _nz_n.sum() > 0:
                dir_correct_n = (dir_pred_n[_nz_n] == dir_actual_n[_nz_n]).sum()
                dir_acc_n = dir_correct_n / _nz_n.sum()
                _baseline_n = max((dir_actual_n[_nz_n] > 0).sum(), (dir_actual_n[_nz_n] < 0).sum()) / _nz_n.sum()
                _lift_n = dir_acc_n - _baseline_n
                print(f"\n  Direction = sign(predicted_delta):")
                print(f"    Accuracy: {dir_correct_n}/{_nz_n.sum()} = {dir_acc_n:.1%}")
                print(f"    Baseline (majority): {_baseline_n:.1%}")
                print(f"    Lift: {_lift_n:+.1%}")

                # Confidence gates
                print(f"\n  Confidence gates (|predicted delta| > threshold):")
                print(f"  {'Threshold':>10} {'N':>6} {'Accuracy':>10} {'Lift':>8} {'% of data':>10}")
                print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
                for thr in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
                    _cm_n = (np.abs(pred_delta_n) > thr) & _nz_n
                    if _cm_n.sum() > 0:
                        _cc_n = (dir_pred_n[_cm_n] == dir_actual_n[_cm_n]).sum()
                        _ca_n = _cc_n / _cm_n.sum()
                        _pct_n = _cm_n.sum() / _nz_n.sum() * 100
                        print(f"  {thr:>10.1f} {_cm_n.sum():>6} {_ca_n:>10.1%} "
                              f"{_ca_n - _baseline_n:>+8.1%} {_pct_n:>9.1f}%")

                # UP vs DOWN breakdown
                _pred_up_n = dir_pred_n[_nz_n] > 0
                _actual_up_n = dir_actual_n[_nz_n] > 0
                _up_correct_n = (_pred_up_n & _actual_up_n).sum()
                _up_total_n = _actual_up_n.sum()
                _dn_correct_n = (~_pred_up_n & ~_actual_up_n).sum()
                _dn_total_n = (~_actual_up_n).sum()
                if _up_total_n > 0:
                    print(f"\n  When actual UP:   {_up_correct_n}/{_up_total_n} = "
                          f"{_up_correct_n/_up_total_n:.1%}")
                if _dn_total_n > 0:
                    print(f"  When actual DOWN: {_dn_correct_n}/{_dn_total_n} = "
                          f"{_dn_correct_n/_dn_total_n:.1%}")

            # Top features by |coefficient|
            coeff_abs_n = np.abs(ols_n.coef_)
            top_idx_n = np.argsort(coeff_abs_n)[::-1][:20]
            print(f"\n  TOP 20 FEATURES (by |coefficient| in delta OLS):")
            print(f"  {'Rank':>4}  {'Feature':<40} {'Coeff':>10} {'|Coeff|':>10}")
            print(f"  {'-'*4}  {'-'*40} {'-'*10} {'-'*10}")
            for rank, fi in enumerate(top_idx_n, 1):
                fn = col_names[fi] if fi < len(col_names) else f'f{fi}'
                print(f"  {rank:>4}  {fn:<40} {ols_n.coef_[fi]:>+10.4f} {coeff_abs_n[fi]:>10.4f}")

            # CONCLUSION
            print(f"\n  ANALYSIS N CONCLUSION:")
            if _nz_n.sum() > 0 and dir_acc_n > 0.55:
                print(f"  PROMISING: {dir_acc_n:.1%} direction accuracy from delta-direct OLS.")
                print(f"  Training on the movement itself (not level) captures directional signal.")
            elif _nz_n.sum() > 0 and dir_acc_n > 0.52:
                print(f"  MARGINAL: {dir_acc_n:.1%} accuracy, slight lift over baseline.")
                if snr_n < 1.0:
                    print(f"  SNR={snr_n:.2f}: prediction noise still > typical delta magnitude.")
            else:
                print(f"  INSUFFICIENT: {dir_acc_n:.1%}. Even modeling delta directly,")
                print(f"  the fractal context cannot reliably predict next-bar direction.")

            # ── Table: Actual vs Predicted delta (test set) ─────────────────
            import csv as _csv_n
            from datetime import datetime as _dt_n, timezone as _tz_nt

            _te_idx_n = list(range(_split_n, n_n))
            _table_n = []
            for j, idx in enumerate(_te_idx_n):
                _ts_val = int(sample_ts[idx])
                _t_str = _dt_n.fromtimestamp(_ts_val, tz=_tz_nt.utc).strftime('%Y-%m-%d %H:%M')
                _curr = float(Y_curr_n[_split_n + j])
                _act_d = float(yn_te[j])
                _prd_d = float(pred_delta_n[j])
                _pred_dir = 'UP' if _prd_d > 0 else 'DOWN'
                _act_dir = 'UP' if _act_d > 0 else ('DOWN' if _act_d < 0 else 'FLAT')
                _correct = 'Y' if _pred_dir == _act_dir else 'N'
                _table_n.append((_t_str, _curr, _curr + _act_d, _curr + _prd_d,
                                 _act_d, _prd_d, _pred_dir, _act_dir, _correct))

            # Console preview
            print(f"\n  {'='*95}")
            print(f"  PREDICTION TABLE (test set: {len(_table_n)} rows)")
            print(f"  {'='*95}")
            _hdr_n = f"  {'Timestamp':<17} {'Current':>10} {'Actual':>10} {'Predicted':>10} {'Act.D':>8} {'Pred.D':>8} {'Dir':>5} {'Act':>5} {'OK':>3}"
            print(_hdr_n)
            print(f"  {'-'*95}")
            _show_n = _table_n[:30] + ([('  ...', '', '', '', '', '', '', '', '')] if len(_table_n) > 40 else []) + _table_n[-10:] if len(_table_n) > 40 else _table_n
            for r in _show_n:
                if r[1] == '':
                    print(f"  {r[0]}")
                else:
                    print(f"  {r[0]:<17} {r[1]:>10.2f} {r[2]:>10.2f} {r[3]:>10.2f} "
                          f"{r[4]:>+8.2f} {r[5]:>+8.2f} {r[6]:>5} {r[7]:>5} {r[8]:>3}")

            # Save CSV
            _csv_path_n = os.path.join(PLOTS_DIR, 'analysis_n_delta_predictions.csv')
            with open(_csv_path_n, 'w', newline='', encoding='utf-8') as _cf:
                _csv_n.writer(_cf).writerow(['timestamp', 'current_price', 'actual_next',
                    'predicted_next', 'actual_delta', 'predicted_delta',
                    'predicted_dir', 'actual_dir', 'correct'])
                _csv_n.writer(_cf).writerows(_table_n)
            print(f"\n  Full table saved: {_csv_path_n} ({len(_table_n)} rows)")

            _n_corr_n = sum(1 for r in _table_n if r[8] == 'Y')
            _mae_n = np.mean(np.abs(pred_delta_n - yn_te))
            _rmse_n = np.sqrt(np.mean((pred_delta_n - yn_te) ** 2))
            print(f"  MAE: {_mae_n:.2f}  |  RMSE: {_rmse_n:.2f}  |  "
                  f"Direction: {_n_corr_n}/{len(_table_n)} correct")

            # ── Plot 1: 4-panel analysis ────────────────────────────────────
            fig_n, axes_n = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
            _c_up_n  = '#2196F3'
            _c_dn_n = '#F44336'

            # (0,0) Scatter: predicted delta vs actual delta
            ax = axes_n[0, 0]
            _col_n = np.where(yn_te > 0, _c_up_n, _c_dn_n)
            ax.scatter(yn_te, pred_delta_n, c=_col_n, alpha=0.4, s=15, edgecolors='none')
            _lim_n = max(np.percentile(np.abs(yn_te), 99),
                         np.percentile(np.abs(pred_delta_n), 99)) * 1.1
            ax.plot([-_lim_n, _lim_n], [-_lim_n, _lim_n], 'k--', alpha=0.3, lw=1)
            ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
            ax.axvline(0, color='gray', lw=0.5, alpha=0.5)
            ax.fill_between([-_lim_n, 0], -_lim_n, 0, color=_c_dn_n, alpha=0.04)
            ax.fill_between([0, _lim_n], 0, _lim_n, color=_c_up_n, alpha=0.04)
            ax.set_xlabel('Actual Delta', fontsize=10)
            ax.set_ylabel('Predicted Delta', fontsize=10)
            ax.set_title(f'Predicted vs Actual Delta (R\u00b2={r2_te_n:.3f})', fontsize=11, fontweight='bold')
            from matplotlib.lines import Line2D as _L2N
            ax.legend(handles=[
                _L2N([0], [0], marker='o', color='w', markerfacecolor=_c_up_n, markersize=8, label='UP'),
                _L2N([0], [0], marker='o', color='w', markerfacecolor=_c_dn_n, markersize=8, label='DOWN'),
            ], loc='upper left', fontsize=9)

            # (0,1) Histogram: predicted delta by actual direction
            ax = axes_n[0, 1]
            _bins_n = np.linspace(-_lim_n, _lim_n, 50)
            ax.hist(pred_delta_n[yn_te > 0], bins=_bins_n, alpha=0.7, color=_c_up_n,
                    label='Actual UP', edgecolor='white', lw=0.5)
            ax.hist(pred_delta_n[yn_te < 0], bins=_bins_n, alpha=0.7, color=_c_dn_n,
                    label='Actual DOWN', edgecolor='white', lw=0.5)
            ax.axvline(0, color='black', lw=1.5, ls='--', alpha=0.7)
            ax.set_xlabel('Predicted Delta', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title('Predicted Delta Distribution by Actual Direction', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)

            # (1,0) Confusion matrix
            ax = axes_n[1, 0]
            if _nz_n.sum() > 0:
                _cm_nn = np.array([[_dn_correct_n, _dn_total_n - _dn_correct_n],
                                   [_up_total_n - _up_correct_n, _up_correct_n]])
                ax.imshow(_cm_nn, cmap='Blues', aspect='auto')
                ax.set_xticks([0, 1]); ax.set_xticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_yticks([0, 1]); ax.set_yticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                for _ri in range(2):
                    for _ci in range(2):
                        _v = _cm_nn[_ri, _ci]
                        ax.text(_ci, _ri, str(_v), ha='center', va='center',
                                fontsize=16, fontweight='bold',
                                color='white' if _v > _cm_nn.max() * 0.5 else 'black')
                ax.set_title(f'Direction Confusion Matrix\nAcc={dir_acc_n:.1%}, Lift={_lift_n:+.1%}',
                            fontsize=11, fontweight='bold')

            # (1,1) Confidence gate curve
            ax = axes_n[1, 1]
            _thrs_n = np.linspace(0, np.percentile(np.abs(pred_delta_n), 95), 30)
            _accs_n, _ns_n = [], []
            for _t in _thrs_n:
                _mn = (np.abs(pred_delta_n) > _t) & _nz_n
                if _mn.sum() >= 5:
                    _accs_n.append((dir_pred_n[_mn] == dir_actual_n[_mn]).sum() / _mn.sum() * 100)
                    _ns_n.append(_mn.sum() / _nz_n.sum() * 100)
                else:
                    _accs_n.append(np.nan); _ns_n.append(0)
            ax.plot(_thrs_n, _accs_n, color='#2196F3', lw=2, label='Accuracy %')
            ax.axhline(_baseline_n * 100, color='gray', ls='--', lw=1, alpha=0.7,
                       label=f'Baseline {_baseline_n:.0%}')
            ax.set_xlabel('|Predicted Delta| Threshold', fontsize=10)
            ax.set_ylabel('Direction Accuracy %', fontsize=10)
            ax.set_title('Confidence Gate: Accuracy vs Threshold', fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax2_n = ax.twinx()
            ax2_n.fill_between(_thrs_n, 0, _ns_n, alpha=0.15, color='orange')
            ax2_n.set_ylabel('% Data Remaining', fontsize=9, color='orange')
            ax2_n.tick_params(axis='y', labelcolor='orange')

            fig_n.suptitle(
                f'Analysis N: Delta-Direct Forecast (MR-Centered)\n'
                f'{n_n} pairs | Acc={dir_acc_n:.1%} | Lift={_lift_n:+.1%} | SNR={snr_n:.2f} | '
                f'UP: {n_up_n} ({n_up_n/n_n*100:.0f}%)  DOWN: {n_down_n} ({n_down_n/n_n*100:.0f}%)',
                fontsize=13, fontweight='bold')
            plt.tight_layout()
            n_path = os.path.join(PLOTS_DIR, '0r_delta_direct_forecast.png')
            fig_n.savefig(n_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_n)
            print(f"\n  Saved: {n_path}")

            # ── Plot 2: Price with delta-predicted direction overlay ────────
            Xn_all_sc = sc_n.transform(X_n)
            pred_delta_all_n = ols_n.predict(Xn_all_sc)

            from datetime import datetime as _dt_n2, timezone as _tz_n2
            close_all_n = base_df['close'].values.astype(float)
            ts_all_n = base_df['timestamp'].values

            _ts_to_bi_n = {int(ts_all_n[i]): i for i in range(len(ts_all_n))}
            _sample_bi_n = [_ts_to_bi_n.get(int(t), -1) for t in sample_ts[:-1]]

            fig_pn, ax_pn = plt.subplots(1, 1, figsize=(20, 7), facecolor='white')
            _x_dates_n = [_dt_n2.fromtimestamp(int(t), tz=_tz_n2.utc) for t in ts_all_n]
            ax_pn.plot(_x_dates_n, close_all_n, color='#BDBDBD', lw=0.8, alpha=0.6, zorder=1)

            _seg_half_n = max(1, len(close_all_n) // 500)
            for i, bi in enumerate(_sample_bi_n):
                if bi < 0:
                    continue
                _s = max(0, bi - _seg_half_n)
                _e = min(len(close_all_n), bi + _seg_half_n + 1)
                _sx = _x_dates_n[_s:_e]
                _sy = close_all_n[_s:_e]
                if len(_sx) < 2:
                    continue
                _color = _c_up_n if pred_delta_all_n[i] > 0 else _c_dn_n
                _alpha = min(1.0, 0.3 + abs(pred_delta_all_n[i]) / max(1, np.std(yn_te)))
                ax_pn.plot(_sx, _sy, color=_color, lw=2.0, alpha=_alpha, zorder=2)

                if Y_delta_n[i] != 0 and np.sign(pred_delta_all_n[i]) != np.sign(Y_delta_n[i]):
                    ax_pn.plot(_x_dates_n[bi], close_all_n[bi], 'x', color='black',
                             markersize=4, alpha=0.5, zorder=3)

            ax_pn.set_xlabel('Time', fontsize=10)
            ax_pn.set_ylabel('Price', fontsize=10)
            ax_pn.set_title(
                f'Price with Delta-Direct Predicted Direction\n'
                f'Blue = predicted UP | Red = predicted DOWN | X = wrong',
                fontsize=12, fontweight='bold')
            from matplotlib.lines import Line2D as _L2N2
            ax_pn.legend(handles=[
                _L2N2([0], [0], color=_c_up_n, lw=2, label='Predicted UP'),
                _L2N2([0], [0], color=_c_dn_n, lw=2, label='Predicted DOWN'),
                _L2N2([0], [0], marker='x', color='black', lw=0, markersize=6, label='Wrong'),
            ], loc='upper left', fontsize=9)
            fig_pn.autofmt_xdate()
            plt.tight_layout()
            pn_path = os.path.join(PLOTS_DIR, '0s_delta_direct_overlay.png')
            fig_pn.savefig(pn_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig_pn)
            print(f"  Saved: {pn_path}")

        else:
            print(f"  SKIP: too few samples ({n_n}) for meaningful analysis")

    else:
        print(f"  [SKIP] Analysis N (--start {_start_at})")

    if _start_at <= 'O' and 'O' not in _skip_set:
        # =====================================================================
        #  ANALYSIS O: STEPWISE DELTA DIRECTION
        #
        #  Forward stepwise selection: pick features one at a time, greedily
        #  maximizing DIRECTION ACCURACY (not R²) on a chronological holdout.
        #  Y = sign(close[t+1] - close[t])  — pure direction classification.
        #  Stops when no feature improves accuracy.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS O: STEPWISE DELTA DIRECTION")
        print(f"  Forward selection maximizing direction accuracy")
        print(f"  Y = sign(close[t+1] - close[t])  |  X selected from 192D")
        print(f"{'='*70}")

        from sklearn.linear_model import LogisticRegression as _LR_O
        from sklearn.preprocessing import StandardScaler as _SS_O

        n_o = len(X) - 1
        if n_o >= 50:
            X_o = X[:-1]
            Y_delta_o = Y_p[1:] - Y_p[:-1]

            # Remove flats for direction classification
            _nf_o = Y_delta_o != 0
            X_o = X_o[_nf_o]
            Y_dir_o = (Y_delta_o[_nf_o] > 0).astype(int)  # 1=UP, 0=DOWN
            n_o = len(Y_dir_o)

            n_up_o = Y_dir_o.sum()
            n_dn_o = n_o - n_up_o
            print(f"\n  Samples (excl. flat): {n_o}  |  UP={n_up_o} ({n_up_o/n_o*100:.1f}%)  DOWN={n_dn_o} ({n_dn_o/n_o*100:.1f}%)")

            # Chronological 60/20/20: train / validation (for selection) / test (final)
            _tr_end = int(n_o * 0.60)
            _va_end = int(n_o * 0.80)
            Xo_tr, Xo_va, Xo_te = X_o[:_tr_end], X_o[_tr_end:_va_end], X_o[_va_end:]
            yo_tr, yo_va, yo_te = Y_dir_o[:_tr_end], Y_dir_o[_tr_end:_va_end], Y_dir_o[_va_end:]

            sc_o = _SS_O()
            Xo_tr_sc = sc_o.fit_transform(Xo_tr)
            Xo_va_sc = sc_o.transform(Xo_va)
            Xo_te_sc = sc_o.transform(Xo_te)

            _baseline_o = max(yo_va.sum(), len(yo_va) - yo_va.sum()) / len(yo_va)
            print(f"  Validation baseline (majority): {_baseline_o:.1%}")
            print(f"  Train: {len(yo_tr)}, Validation: {len(yo_va)}, Test: {len(yo_te)}")

            n_feat = X_o.shape[1]
            _selected = []
            _remaining = list(range(n_feat))
            _best_acc_history = []
            _step_log = []

            print(f"\n  {'Step':>4}  {'Feature Added':<42} {'Val Acc':>8} {'Lift':>8} {'N feat':>6}")
            print(f"  {'-'*4}  {'-'*42} {'-'*8} {'-'*8} {'-'*6}")

            _prev_best_acc = 0.0
            _max_steps = min(50, n_feat)  # cap at 50 features

            for step in range(_max_steps):
                _best_feat = -1
                _best_acc = _prev_best_acc

                for fi in _remaining:
                    _try = _selected + [fi]
                    try:
                        _lr = _LR_O(max_iter=500, class_weight='balanced', C=1.0)
                        _lr.fit(Xo_tr_sc[:, _try], yo_tr)
                        _preds = _lr.predict(Xo_va_sc[:, _try])
                        _acc = (_preds == yo_va).sum() / len(yo_va)
                        if _acc > _best_acc:
                            _best_acc = _acc
                            _best_feat = fi
                    except Exception:
                        continue

                if _best_feat < 0 or _best_acc <= _prev_best_acc:
                    print(f"\n  STOPPED at step {step}: no feature improves validation accuracy")
                    break

                _selected.append(_best_feat)
                _remaining.remove(_best_feat)
                _prev_best_acc = _best_acc
                _best_acc_history.append(_best_acc)

                _fname = col_names[_best_feat] if _best_feat < len(col_names) else f'f{_best_feat}'
                _lift_s = _best_acc - _baseline_o
                _step_log.append((_fname, _best_acc, _lift_s, len(_selected)))
                print(f"  {step+1:>4}  {_fname:<42} {_best_acc:>8.1%} {_lift_s:>+8.1%} {len(_selected):>6}")

            # Final evaluation on TEST set with selected features
            if len(_selected) > 0:
                print(f"\n  {'='*60}")
                print(f"  FINAL MODEL: {len(_selected)} features")
                print(f"  {'='*60}")

                _lr_final = _LR_O(max_iter=500, class_weight='balanced', C=1.0)
                _lr_final.fit(Xo_tr_sc[:, _selected], yo_tr)

                # Validation performance
                _va_pred = _lr_final.predict(Xo_va_sc[:, _selected])
                _va_acc = (_va_pred == yo_va).sum() / len(yo_va)

                # Test performance
                _te_pred = _lr_final.predict(Xo_te_sc[:, _selected])
                _te_acc = (_te_pred == yo_te).sum() / len(yo_te)
                _te_baseline = max(yo_te.sum(), len(yo_te) - yo_te.sum()) / len(yo_te)
                _te_lift = _te_acc - _te_baseline

                _te_up_mask = yo_te == 1
                _te_dn_mask = yo_te == 0
                _te_up_correct = (_te_pred[_te_up_mask] == 1).sum()
                _te_dn_correct = (_te_pred[_te_dn_mask] == 0).sum()

                print(f"\n  Validation: {_va_acc:.1%} (baseline {_baseline_o:.1%}, lift {_va_acc - _baseline_o:+.1%})")
                print(f"  TEST:       {_te_acc:.1%} (baseline {_te_baseline:.1%}, lift {_te_lift:+.1%})")
                if _te_up_mask.sum() > 0:
                    print(f"  Test UP:    {_te_up_correct}/{_te_up_mask.sum()} = {_te_up_correct/_te_up_mask.sum():.1%}")
                if _te_dn_mask.sum() > 0:
                    print(f"  Test DOWN:  {_te_dn_correct}/{_te_dn_mask.sum()} = {_te_dn_correct/_te_dn_mask.sum():.1%}")

                # Confidence gates on test set using predict_proba
                _te_proba = _lr_final.predict_proba(Xo_te_sc[:, _selected])[:, 1]
                print(f"\n  Confidence gates (distance from 0.5):")
                print(f"  {'Threshold':>10} {'N':>6} {'Accuracy':>10} {'Lift':>8} {'% of data':>10}")
                print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
                for thr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
                    _conf_mask = np.abs(_te_proba - 0.5) > thr
                    if _conf_mask.sum() > 0:
                        _conf_pred = (_te_proba[_conf_mask] > 0.5).astype(int)
                        _conf_acc = (_conf_pred == yo_te[_conf_mask]).sum() / _conf_mask.sum()
                        _conf_pct = _conf_mask.sum() / len(yo_te) * 100
                        print(f"  {thr:>10.2f} {_conf_mask.sum():>6} {_conf_acc:>10.1%} "
                              f"{_conf_acc - _te_baseline:>+8.1%} {_conf_pct:>9.1f}%")

                # Selected features summary
                print(f"\n  SELECTED FEATURES ({len(_selected)}):")
                for i, fi in enumerate(_selected):
                    _fn = col_names[fi] if fi < len(col_names) else f'f{fi}'
                    _coef = _lr_final.coef_[0][i]
                    print(f"    {i+1:>3}. {_fn:<42} coeff={_coef:>+8.4f}")

                # CONCLUSION
                print(f"\n  ANALYSIS O CONCLUSION:")
                if _te_lift > 0.05:
                    print(f"  PROMISING: {_te_acc:.1%} test accuracy ({_te_lift:+.1%} lift) with only {len(_selected)} features.")
                    print(f"  Stepwise selection found a sparse, generalizable direction model.")
                elif _te_lift > 0.02:
                    print(f"  MARGINAL: {_te_acc:.1%} test accuracy ({_te_lift:+.1%} lift).")
                    print(f"  Some signal, but may not survive transaction costs.")
                else:
                    print(f"  INSUFFICIENT: {_te_acc:.1%} test accuracy ({_te_lift:+.1%} lift).")
                    print(f"  Even with feature selection, direction is not reliably predictable")
                    print(f"  from 192D fractal context at this timeframe/horizon.")

                # ── Table: test set predictions ─────────────────────────────
                import csv as _csv_o
                from datetime import datetime as _dt_o, timezone as _tz_ot

                _te_start_idx = _va_end  # index into the non-flat array
                # Map back to original sample indices (before flat removal)
                _nf_indices = np.where(_nf_o)[0]  # indices of non-flat in original
                _table_o = []
                for j in range(len(yo_te)):
                    _orig_idx = _nf_indices[_va_end + j]
                    _ts_val = int(sample_ts[_orig_idx])
                    _t_str = _dt_o.fromtimestamp(_ts_val, tz=_tz_ot.utc).strftime('%Y-%m-%d %H:%M')
                    _curr = float(Y_p[_orig_idx])
                    _next = float(Y_p[_orig_idx + 1])
                    _delta = _next - _curr
                    _pred_dir = 'UP' if _te_pred[j] == 1 else 'DOWN'
                    _act_dir = 'UP' if yo_te[j] == 1 else 'DOWN'
                    _prob = float(_te_proba[j])
                    _correct = 'Y' if _pred_dir == _act_dir else 'N'
                    _table_o.append((_t_str, _curr, _next, _delta, _prob, _pred_dir, _act_dir, _correct))

                # Console preview
                print(f"\n  {'='*100}")
                print(f"  PREDICTION TABLE (test set: {len(_table_o)} rows)")
                print(f"  {'='*100}")
                _hdr_o = f"  {'Timestamp':<17} {'Current':>10} {'Next':>10} {'Delta':>8} {'P(UP)':>7} {'Dir':>5} {'Act':>5} {'OK':>3}"
                print(_hdr_o)
                print(f"  {'-'*100}")
                _show_o = _table_o[:30] + ([('  ...', '', '', '', '', '', '', '')] if len(_table_o) > 40 else []) + _table_o[-10:] if len(_table_o) > 40 else _table_o
                for r in _show_o:
                    if r[1] == '':
                        print(f"  {r[0]}")
                    else:
                        print(f"  {r[0]:<17} {r[1]:>10.2f} {r[2]:>10.2f} {r[3]:>+8.2f} "
                              f"{r[4]:>7.3f} {r[5]:>5} {r[6]:>5} {r[7]:>3}")

                # Save CSV
                _csv_path_o = os.path.join(PLOTS_DIR, 'analysis_o_stepwise_predictions.csv')
                with open(_csv_path_o, 'w', newline='', encoding='utf-8') as _cf:
                    _csv_o.writer(_cf).writerow(['timestamp', 'current_price', 'next_price',
                        'delta', 'prob_up', 'predicted_dir', 'actual_dir', 'correct'])
                    _csv_o.writer(_cf).writerows(_table_o)
                print(f"\n  Full table saved: {_csv_path_o} ({len(_table_o)} rows)")

                # ── Plot: Stepwise accuracy curve + 4-panel ─────────────────
                fig_o, axes_o = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')

                # (0,0) Stepwise accuracy curve
                ax = axes_o[0, 0]
                _steps = list(range(1, len(_best_acc_history) + 1))
                ax.plot(_steps, [a * 100 for a in _best_acc_history], 'o-', color='#2196F3', lw=2, markersize=5)
                ax.axhline(_baseline_o * 100, color='gray', ls='--', lw=1, alpha=0.7, label=f'Baseline {_baseline_o:.0%}')
                ax.set_xlabel('Number of Features', fontsize=10)
                ax.set_ylabel('Validation Accuracy %', fontsize=10)
                ax.set_title('Stepwise Feature Selection Progress', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                # Annotate top features
                for si, (fn, acc, _, _) in enumerate(_step_log[:5]):
                    _short = fn.split('__')[-1] if '__' in fn else fn
                    ax.annotate(_short, (_steps[si], acc * 100), fontsize=7,
                               rotation=30, ha='left', va='bottom')

                # (0,1) Probability histogram by actual direction
                ax = axes_o[0, 1]
                _c_up_o = '#2196F3'
                _c_dn_o = '#F44336'
                _bins_o = np.linspace(0, 1, 30)
                ax.hist(_te_proba[yo_te == 1], bins=_bins_o, alpha=0.7, color=_c_up_o,
                        label='Actual UP', edgecolor='white', lw=0.5)
                ax.hist(_te_proba[yo_te == 0], bins=_bins_o, alpha=0.7, color=_c_dn_o,
                        label='Actual DOWN', edgecolor='white', lw=0.5)
                ax.axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.7)
                ax.set_xlabel('P(UP)', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title('Probability Distribution by Actual Direction', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)

                # (1,0) Confusion matrix
                ax = axes_o[1, 0]
                _tp_dn_o = (_te_pred[_te_dn_mask] == 0).sum()
                _fp_up_o = (_te_pred[_te_dn_mask] == 1).sum()
                _fn_dn_o = (_te_pred[_te_up_mask] == 0).sum()
                _tp_up_o = (_te_pred[_te_up_mask] == 1).sum()
                _cm_o = np.array([[_tp_dn_o, _fp_up_o], [_fn_dn_o, _tp_up_o]])
                ax.imshow(_cm_o, cmap='Blues', aspect='auto')
                ax.set_xticks([0, 1]); ax.set_xticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_yticks([0, 1]); ax.set_yticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                for _ri in range(2):
                    for _ci in range(2):
                        _v = _cm_o[_ri, _ci]
                        ax.text(_ci, _ri, str(_v), ha='center', va='center',
                                fontsize=16, fontweight='bold',
                                color='white' if _v > _cm_o.max() * 0.5 else 'black')
                ax.set_title(f'Test Confusion Matrix\nAcc={_te_acc:.1%}, Lift={_te_lift:+.1%}',
                            fontsize=11, fontweight='bold')

                # (1,1) Confidence gate curve on test
                ax = axes_o[1, 1]
                _thrs_o = np.linspace(0, 0.45, 25)
                _accs_o, _ns_o = [], []
                for _t in _thrs_o:
                    _cmask = np.abs(_te_proba - 0.5) > _t
                    if _cmask.sum() >= 5:
                        _cpred = (_te_proba[_cmask] > 0.5).astype(int)
                        _accs_o.append((_cpred == yo_te[_cmask]).sum() / _cmask.sum() * 100)
                        _ns_o.append(_cmask.sum() / len(yo_te) * 100)
                    else:
                        _accs_o.append(np.nan); _ns_o.append(0)
                ax.plot(_thrs_o, _accs_o, color='#2196F3', lw=2, label='Accuracy %')
                ax.axhline(_te_baseline * 100, color='gray', ls='--', lw=1, alpha=0.7,
                           label=f'Baseline {_te_baseline:.0%}')
                ax.set_xlabel('Confidence Threshold (|P(UP) - 0.5|)', fontsize=10)
                ax.set_ylabel('Test Accuracy %', fontsize=10)
                ax.set_title('Confidence Gate: Accuracy vs Threshold', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax2_o = ax.twinx()
                ax2_o.fill_between(_thrs_o, 0, _ns_o, alpha=0.15, color='orange')
                ax2_o.set_ylabel('% Data Remaining', fontsize=9, color='orange')
                ax2_o.tick_params(axis='y', labelcolor='orange')

                fig_o.suptitle(
                    f'Analysis O: Stepwise Delta Direction\n'
                    f'{n_o} samples | {len(_selected)} features selected | '
                    f'Test Acc={_te_acc:.1%} | Lift={_te_lift:+.1%}',
                    fontsize=13, fontweight='bold')
                plt.tight_layout()
                o_path = os.path.join(PLOTS_DIR, '0t_stepwise_direction.png')
                fig_o.savefig(o_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig_o)
                print(f"\n  Saved: {o_path}")

            else:
                print(f"  No features selected — direction not predictable from this feature set.")

        else:
            print(f"  SKIP: too few samples ({n_o}) for stepwise analysis")

    else:
        print(f"  [SKIP] Analysis O (--start {_start_at})")

    if _start_at <= 'P' and 'P' not in _skip_set:
        # =====================================================================
        #  ANALYSIS P: PAIRED DATA POINT DIRECTION
        #
        #  Nuclear option: use BOTH data points in each pair as features.
        #  X = [features_at_t CONCAT features_at_t+1] → 386D input
        #  Y = sign(close[t+1] - close[t])
        #  The model sees the "before and after" state — the TRANSITION.
        #  Stepwise selection finds which combination of both states matters.
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS P: PAIRED DATA POINT DIRECTION")
        print(f"  X = [features_t || features_t+1]  (both points concatenated)")
        print(f"  Y = sign(close[t+1] - close[t])  (direction between them)")
        print(f"  The model sees BOTH states to classify the transition")
        print(f"{'='*70}")

        from sklearn.linear_model import LogisticRegression as _LR_P
        from sklearn.preprocessing import StandardScaler as _SS_P

        n_p = len(X) - 1  # consecutive pairs
        print(f"\n  Consecutive data point pairs: {n_p}")

        if n_p >= 50:
            # Build paired feature matrix: [X[t] || X[t+1]] for each pair
            X_A = X[:-1]    # point A: features at time t       (193D)
            X_B = X[1:]      # point B: features at time t+1     (193D)
            Y_delta_p = Y_p[1:] - Y_p[:-1]  # price change A→B

            # Also compute the feature DIFFERENCE (B - A) as transition features
            X_diff = X_B - X_A  # how did the fractal context CHANGE? (193D)

            # Movement magnitude features (psychology + algo footprint)
            _abs_delta = np.abs(Y_delta_p).reshape(-1, 1)
            _pct_delta = (Y_delta_p / np.maximum(Y_p[:-1], 1e-6) * 100).reshape(-1, 1)
            _abs_pct = np.abs(_pct_delta)

            # Combined: [A || B || diff || |delta| || pct || |pct|] = 582D
            Xp_full = np.concatenate([X_A, X_B, X_diff, _abs_delta, _pct_delta, _abs_pct], axis=1)
            n_feat_p = Xp_full.shape[1]

            # Remove flats
            _nf_p = Y_delta_p != 0
            Xp_full = Xp_full[_nf_p]
            Yp_dir = (Y_delta_p[_nf_p] > 0).astype(int)  # 1=UP, 0=DOWN
            _nf_indices_p = np.where(_nf_p)[0]  # for mapping back
            n_p = len(Yp_dir)

            n_up_p = Yp_dir.sum()
            n_dn_p = n_p - n_up_p
            print(f"  Pairs (excl. flat): {n_p}  |  UP={n_up_p} ({n_up_p/n_p*100:.1f}%)  "
                  f"DOWN={n_dn_p} ({n_dn_p/n_p*100:.1f}%)")
            print(f"  Feature vector: {n_feat_p}D  (193 A + 193 B + 193 diff + 3 magnitude)")

            # Chronological 60/20/20
            _tr_p = int(n_p * 0.60)
            _va_p = int(n_p * 0.80)
            Xp_tr, Xp_va, Xp_te = Xp_full[:_tr_p], Xp_full[_tr_p:_va_p], Xp_full[_va_p:]
            yp_tr, yp_va, yp_te = Yp_dir[:_tr_p], Yp_dir[_tr_p:_va_p], Yp_dir[_va_p:]

            sc_p = _SS_P()
            Xp_tr_sc = sc_p.fit_transform(Xp_tr)
            Xp_va_sc = sc_p.transform(Xp_va)
            Xp_te_sc = sc_p.transform(Xp_te)

            _base_va = max(yp_va.sum(), len(yp_va) - yp_va.sum()) / len(yp_va)
            _base_te = max(yp_te.sum(), len(yp_te) - yp_te.sum()) / len(yp_te)
            print(f"\n  Train: {len(yp_tr)}, Validation: {len(yp_va)}, Test: {len(yp_te)}")
            print(f"  Validation baseline: {_base_va:.1%}, Test baseline: {_base_te:.1%}")

            # Build column names for paired features
            _pA_names = [f'A_{c}' for c in col_names]   # point A features
            _pB_names = [f'B_{c}' for c in col_names]   # point B features
            _pd_names = [f'diff_{c}' for c in col_names] # B-A difference
            _mag_names = ['abs_delta', 'pct_delta', 'abs_pct_delta']  # magnitude
            _pair_col_names = _pA_names + _pB_names + _pd_names + _mag_names

            # ── Stepwise forward selection ──────────────────────────────
            _selected_p = []
            _remaining_p = list(range(n_feat_p))
            _best_acc_hist_p = []
            _step_log_p = []
            _prev_best_p = 0.0
            _max_steps_p = min(50, n_feat_p)

            print(f"\n  STEPWISE SELECTION (maximizing validation direction accuracy):")
            print(f"  {'Step':>4}  {'Feature Added':<48} {'Val Acc':>8} {'Lift':>8} {'N feat':>6}")
            print(f"  {'-'*4}  {'-'*48} {'-'*8} {'-'*8} {'-'*6}")

            for step in range(_max_steps_p):
                _best_feat_p = -1
                _best_acc_p = _prev_best_p

                for fi in _remaining_p:
                    _try = _selected_p + [fi]
                    try:
                        _lr = _LR_P(max_iter=500, class_weight='balanced', C=1.0)
                        _lr.fit(Xp_tr_sc[:, _try], yp_tr)
                        _preds = _lr.predict(Xp_va_sc[:, _try])
                        _acc = (_preds == yp_va).sum() / len(yp_va)
                        if _acc > _best_acc_p:
                            _best_acc_p = _acc
                            _best_feat_p = fi
                    except Exception:
                        continue

                if _best_feat_p < 0 or _best_acc_p <= _prev_best_p:
                    print(f"\n  STOPPED at step {step}: no improvement")
                    break

                _selected_p.append(_best_feat_p)
                _remaining_p.remove(_best_feat_p)
                _prev_best_p = _best_acc_p
                _best_acc_hist_p.append(_best_acc_p)

                _fn_p = _pair_col_names[_best_feat_p] if _best_feat_p < len(_pair_col_names) else f'f{_best_feat_p}'
                _lift_sp = _best_acc_p - _base_va
                _step_log_p.append((_fn_p, _best_acc_p, _lift_sp, len(_selected_p)))
                print(f"  {step+1:>4}  {_fn_p:<48} {_best_acc_p:>8.1%} {_lift_sp:>+8.1%} {len(_selected_p):>6}")

            # Final model on test set
            if len(_selected_p) > 0:
                _lr_fp = _LR_P(max_iter=500, class_weight='balanced', C=1.0)
                _lr_fp.fit(Xp_tr_sc[:, _selected_p], yp_tr)

                _va_pred_p = _lr_fp.predict(Xp_va_sc[:, _selected_p])
                _va_acc_p = (_va_pred_p == yp_va).sum() / len(yp_va)

                _te_pred_p = _lr_fp.predict(Xp_te_sc[:, _selected_p])
                _te_acc_p = (_te_pred_p == yp_te).sum() / len(yp_te)
                _te_lift_p = _te_acc_p - _base_te

                _te_proba_p = _lr_fp.predict_proba(Xp_te_sc[:, _selected_p])[:, 1]

                _te_up_mask_p = yp_te == 1
                _te_dn_mask_p = yp_te == 0
                _te_up_corr = (_te_pred_p[_te_up_mask_p] == 1).sum() if _te_up_mask_p.sum() > 0 else 0
                _te_dn_corr = (_te_pred_p[_te_dn_mask_p] == 0).sum() if _te_dn_mask_p.sum() > 0 else 0

                print(f"\n  {'='*60}")
                print(f"  FINAL MODEL: {len(_selected_p)} features")
                print(f"  {'='*60}")
                print(f"\n  Validation: {_va_acc_p:.1%} (baseline {_base_va:.1%}, lift {_va_acc_p - _base_va:+.1%})")
                print(f"  TEST:       {_te_acc_p:.1%} (baseline {_base_te:.1%}, lift {_te_lift_p:+.1%})")
                if _te_up_mask_p.sum() > 0:
                    print(f"  Test UP:    {_te_up_corr}/{_te_up_mask_p.sum()} = {_te_up_corr/_te_up_mask_p.sum():.1%}")
                if _te_dn_mask_p.sum() > 0:
                    print(f"  Test DOWN:  {_te_dn_corr}/{_te_dn_mask_p.sum()} = {_te_dn_corr/_te_dn_mask_p.sum():.1%}")

                # Confidence gates
                print(f"\n  Confidence gates (|P(UP) - 0.5| > threshold):")
                print(f"  {'Threshold':>10} {'N':>6} {'Accuracy':>10} {'Lift':>8} {'% of data':>10}")
                print(f"  {'-'*10} {'-'*6} {'-'*10} {'-'*8} {'-'*10}")
                for thr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
                    _cmask = np.abs(_te_proba_p - 0.5) > thr
                    if _cmask.sum() > 0:
                        _cpred = (_te_proba_p[_cmask] > 0.5).astype(int)
                        _cacc = (_cpred == yp_te[_cmask]).sum() / _cmask.sum()
                        print(f"  {thr:>10.2f} {_cmask.sum():>6} {_cacc:>10.1%} "
                              f"{_cacc - _base_te:>+8.1%} {_cmask.sum()/len(yp_te)*100:>9.1f}%")

                # Selected features
                print(f"\n  SELECTED FEATURES ({len(_selected_p)}):")
                _row_width = X.shape[1]  # v1: 193, v2: 186
                for i, fi in enumerate(_selected_p):
                    _fn = _pair_col_names[fi] if fi < len(_pair_col_names) else f'f{fi}'
                    _coef = _lr_fp.coef_[0][i]
                    _source = 'point_A' if fi < _row_width else ('point_B' if fi < 2*_row_width else 'diff(B-A)')
                    print(f"    {i+1:>3}. {_fn:<48} coeff={_coef:>+8.4f}  [{_source}]")

                # CONCLUSION
                print(f"\n  ANALYSIS P CONCLUSION:")
                if _te_lift_p > 0.05:
                    print(f"  PROMISING: {_te_acc_p:.1%} test accuracy ({_te_lift_p:+.1%} lift).")
                    print(f"  Paired data points carry transition signal that single points do not.")
                elif _te_lift_p > 0.02:
                    print(f"  MARGINAL: {_te_acc_p:.1%} test ({_te_lift_p:+.1%} lift).")
                    print(f"  Some paired signal detected, needs refinement.")
                else:
                    print(f"  INSUFFICIENT: {_te_acc_p:.1%} test ({_te_lift_p:+.1%} lift).")
                    print(f"  Even with both data points visible, the transition direction")
                    print(f"  is not reliably classifiable from 192D fractal context.")

                # ── Table: test set predictions ──────────────────────────
                import csv as _csv_p
                from datetime import datetime as _dt_p, timezone as _tz_pt

                _table_p = []
                for j in range(len(yp_te)):
                    _orig_idx = _nf_indices_p[_va_p + j]
                    _ts_a = int(sample_ts[_orig_idx])
                    _ts_b = int(sample_ts[_orig_idx + 1])
                    _t_str_a = _dt_p.fromtimestamp(_ts_a, tz=_tz_pt.utc).strftime('%Y-%m-%d %H:%M')
                    _t_str_b = _dt_p.fromtimestamp(_ts_b, tz=_tz_pt.utc).strftime('%H:%M')
                    _price_a = float(Y_p[_orig_idx])
                    _price_b = float(Y_p[_orig_idx + 1])
                    _delta = _price_b - _price_a
                    _pred_dir = 'UP' if _te_pred_p[j] == 1 else 'DOWN'
                    _act_dir = 'UP' if yp_te[j] == 1 else 'DOWN'
                    _prob = float(_te_proba_p[j])
                    _correct = 'Y' if _pred_dir == _act_dir else 'N'
                    _table_p.append((_t_str_a, _t_str_b, _price_a, _price_b,
                                     _delta, _prob, _pred_dir, _act_dir, _correct))

                print(f"\n  {'='*105}")
                print(f"  PAIR PREDICTION TABLE (test: {len(_table_p)} pairs)")
                print(f"  {'='*105}")
                _hdr = (f"  {'Time A':<17} {'->B':>5} {'Price A':>10} {'Price B':>10} "
                        f"{'Delta':>8} {'P(UP)':>7} {'Pred':>5} {'Act':>5} {'OK':>3}")
                print(_hdr)
                print(f"  {'-'*105}")
                _show_p = _table_p[:30] + ([('  ...', '', '', '', '', '', '', '', '')] if len(_table_p) > 40 else []) + _table_p[-10:] if len(_table_p) > 40 else _table_p
                for r in _show_p:
                    if r[1] == '':
                        print(f"  {r[0]}")
                    else:
                        print(f"  {r[0]:<17} {r[1]:>5} {r[2]:>10.2f} {r[3]:>10.2f} "
                              f"{r[4]:>+8.2f} {r[5]:>7.3f} {r[6]:>5} {r[7]:>5} {r[8]:>3}")

                # Save CSV
                _csv_path_p = os.path.join(PLOTS_DIR, 'analysis_p_paired_points.csv')
                with open(_csv_path_p, 'w', newline='', encoding='utf-8') as _cf:
                    _w = _csv_p.writer(_cf)
                    _w.writerow(['time_a', 'time_b', 'price_a', 'price_b', 'delta',
                        'prob_up', 'predicted_dir', 'actual_dir', 'correct'])
                    _w.writerows(_table_p)
                print(f"\n  Saved: {_csv_path_p} ({len(_table_p)} pairs)")

                # ── Plot: 4-panel ───────────────────────────────────────
                fig_p2, axes_p2 = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')

                # (0,0) Stepwise accuracy curve
                ax = axes_p2[0, 0]
                _steps_p = list(range(1, len(_best_acc_hist_p) + 1))
                ax.plot(_steps_p, [a * 100 for a in _best_acc_hist_p], 'o-',
                        color='#2196F3', lw=2, markersize=6)
                ax.axhline(_base_va * 100, color='gray', ls='--', lw=1, alpha=0.7,
                           label=f'Baseline {_base_va:.0%}')
                ax.set_xlabel('Number of Features', fontsize=10)
                ax.set_ylabel('Validation Accuracy %', fontsize=10)
                ax.set_title('Stepwise Feature Selection', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                for si, (fn, acc, _, _) in enumerate(_step_log_p[:8]):
                    _short = fn.split('__')[-1] if '__' in fn else fn
                    if len(_short) > 20:
                        _short = _short[:18] + '..'
                    ax.annotate(_short, (_steps_p[si], acc * 100), fontsize=7,
                               rotation=30, ha='left', va='bottom')

                # (0,1) P(UP) histogram by actual direction
                ax = axes_p2[0, 1]
                _c_up_p = '#2196F3'
                _c_dn_p = '#F44336'
                _bins_p = np.linspace(0, 1, 30)
                ax.hist(_te_proba_p[yp_te == 1], bins=_bins_p, alpha=0.7, color=_c_up_p,
                        label='Actual UP', edgecolor='white', lw=0.5)
                ax.hist(_te_proba_p[yp_te == 0], bins=_bins_p, alpha=0.7, color=_c_dn_p,
                        label='Actual DOWN', edgecolor='white', lw=0.5)
                ax.axvline(0.5, color='black', lw=1.5, ls='--', alpha=0.7)
                ax.set_xlabel('P(UP)', fontsize=10)
                ax.set_ylabel('Count', fontsize=10)
                ax.set_title('Probability Distribution by Actual Direction', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)

                # (1,0) Confusion matrix
                ax = axes_p2[1, 0]
                _tp_dn = _te_dn_corr
                _fp_up = _te_dn_mask_p.sum() - _te_dn_corr
                _fn_dn = _te_up_mask_p.sum() - _te_up_corr
                _tp_up = _te_up_corr
                _cm_p = np.array([[_tp_dn, _fp_up], [_fn_dn, _tp_up]])
                ax.imshow(_cm_p, cmap='Blues', aspect='auto')
                ax.set_xticks([0, 1]); ax.set_xticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_yticks([0, 1]); ax.set_yticklabels(['DOWN', 'UP'], fontsize=10)
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
                for _ri in range(2):
                    for _ci in range(2):
                        _v = _cm_p[_ri, _ci]
                        ax.text(_ci, _ri, str(_v), ha='center', va='center',
                                fontsize=16, fontweight='bold',
                                color='white' if _v > _cm_p.max() * 0.5 else 'black')
                ax.set_title(f'Test Confusion Matrix\nAcc={_te_acc_p:.1%}, Lift={_te_lift_p:+.1%}',
                            fontsize=11, fontweight='bold')

                # (1,1) Confidence gate curve
                ax = axes_p2[1, 1]
                _thrs_p = np.linspace(0, 0.45, 25)
                _accs_p, _ns_p = [], []
                for _t in _thrs_p:
                    _cmask = np.abs(_te_proba_p - 0.5) > _t
                    if _cmask.sum() >= 5:
                        _cpred = (_te_proba_p[_cmask] > 0.5).astype(int)
                        _accs_p.append((_cpred == yp_te[_cmask]).sum() / _cmask.sum() * 100)
                        _ns_p.append(_cmask.sum() / len(yp_te) * 100)
                    else:
                        _accs_p.append(np.nan); _ns_p.append(0)
                ax.plot(_thrs_p, _accs_p, color='#2196F3', lw=2, label='Accuracy %')
                ax.axhline(_base_te * 100, color='gray', ls='--', lw=1, alpha=0.7,
                           label=f'Baseline {_base_te:.0%}')
                ax.set_xlabel('Confidence (|P(UP) - 0.5|)', fontsize=10)
                ax.set_ylabel('Test Accuracy %', fontsize=10)
                ax.set_title('Confidence Gate: Accuracy vs Threshold', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax2_p = ax.twinx()
                ax2_p.fill_between(_thrs_p, 0, _ns_p, alpha=0.15, color='orange')
                ax2_p.set_ylabel('% Data Remaining', fontsize=9, color='orange')
                ax2_p.tick_params(axis='y', labelcolor='orange')

                fig_p2.suptitle(
                    f'Analysis P: Paired Data Point Direction\n'
                    f'{n_p} pairs | {len(_selected_p)} features | '
                    f'Test Acc={_te_acc_p:.1%} | Lift={_te_lift_p:+.1%}',
                    fontsize=13, fontweight='bold')
                plt.tight_layout()
                p2_path = os.path.join(PLOTS_DIR, '0u_paired_point_direction.png')
                fig_p2.savefig(p2_path, dpi=150, bbox_inches='tight', facecolor='white')
                plt.close(fig_p2)
                print(f"  Saved: {p2_path}")

            else:
                print(f"  No features selected — paired points not predictable.")

        else:
            print(f"  SKIP: too few samples ({n_p}) for paired analysis")

    else:
        print(f"  [SKIP] Analysis P (--start {_start_at})")

    if _start_at <= 'Q' and 'Q' not in _skip_set:
        # =====================================================================
        #  ANALYSIS Q: SIGNED MAGNITUDE HISTOGRAM + PAIRED 192D PROFILES
        #
        #  1. Histogram |delta| to find natural magnitude scales (modes)
        #  2. Mirror into signed bins (+/- magnitude) from histogram valleys
        #  3. Each pair keeps its 192D features (point A + point B) for profiling
        #  4. Per bin: direction purity, 192D feature means, distinguishing features
        # =====================================================================
        print(f"\n{'='*70}")
        print(f"  ANALYSIS Q: SIGNED MAGNITUDE HISTOGRAM + PAIRED 192D PROFILES")
        print(f"  Histogram modes -> signed bins -> 192D pair profiles per bin")
        print(f"{'='*70}")

        n_q = len(Y_p) - 1
        if n_q >= 100:
            Y_delta_q = Y_p[1:] - Y_p[:-1]
            X_q_A = X[:-1]   # point A features (193D)
            X_q_B = X[1:]    # point B features (193D)

            # Remove flats
            _nf_q = Y_delta_q != 0
            Y_delta_q = Y_delta_q[_nf_q]
            X_q_A = X_q_A[_nf_q]
            X_q_B = X_q_B[_nf_q]
            n_q = len(Y_delta_q)
            _abs_d = np.abs(Y_delta_q)
            _dir_q = np.where(Y_delta_q > 0, 1, -1)

            # ── STEP 1: Split by sign FIRST, then histogram each side ───
            # UP and DN are separate populations with potentially different
            # magnitude structures. Analyze independently.
            _global_A_mean = X_q_A.mean(axis=0)
            _global_A_std = X_q_A.std(axis=0)
            _global_A_std[_global_A_std < 1e-10] = 1.0

            _base_up = (_dir_q > 0).sum() / n_q * 100
            print(f"\n  Total moves: {n_q} (flats removed)")
            print(f"  |delta| range: {_abs_d.min():.1f} to {_abs_d.max():.1f}, "
                  f"median={np.median(_abs_d):.1f}, mean={np.mean(_abs_d):.1f}")
            print(f"  Baseline: {_base_up:.1f}% UP / {100-_base_up:.1f}% DN")

            all_side_stats = {}  # for plotting later

            for _side, _side_label in [('UP', 'UP (delta > 0)'), ('DN', 'DN (delta < 0)')]:
                _s_mask = _dir_q > 0 if _side == 'UP' else _dir_q < 0
                _s_n = _s_mask.sum()
                _s_deltas = Y_delta_q[_s_mask]
                _s_abs = _abs_d[_s_mask]
                _s_A = X_q_A[_s_mask]
                _s_B = X_q_B[_s_mask]

                print(f"\n  {'='*70}")
                print(f"  {_side_label}: {_s_n} moves ({_s_n/n_q*100:.1f}%)")
                print(f"  |delta| range: {_s_abs.min():.1f} to {_s_abs.max():.1f}, "
                      f"median={np.median(_s_abs):.1f}, mean={np.mean(_s_abs):.1f}")
                print(f"  {'='*70}")

                # Log-histogram of this side's magnitudes
                _s_log = np.log1p(_s_abs)
                _s_nbins = min(200, max(50, _s_n // 20))
                _s_hist_c, _s_hist_e = np.histogram(_s_log, bins=_s_nbins)
                _s_centers = 0.5 * (_s_hist_e[:-1] + _s_hist_e[1:])

                # Gaussian smooth — adaptive sigma (lighter for fewer bins)
                _sigma = max(1.0, min(3.0, _s_nbins / 40))
                _kern_x = np.arange(-3 * int(np.ceil(_sigma)), 3 * int(np.ceil(_sigma)) + 1)
                _kern = np.exp(-0.5 * (_kern_x / _sigma) ** 2)
                _kern /= _kern.sum()
                _s_smooth = np.convolve(_s_hist_c.astype(float), _kern, mode='same')

                # Find modes
                _s_peaks = []
                for _i in range(1, len(_s_smooth) - 1):
                    if _s_smooth[_i] > _s_smooth[_i - 1] and _s_smooth[_i] > _s_smooth[_i + 1]:
                        _s_peaks.append(_i)

                # Find valleys
                _s_valleys_log = []
                for _pi in range(len(_s_peaks) - 1):
                    _seg = _s_smooth[_s_peaks[_pi]:_s_peaks[_pi + 1] + 1]
                    _vi = _s_peaks[_pi] + np.argmin(_seg)
                    _s_valleys_log.append(_s_centers[_vi])

                _s_mag_edges = sorted(np.expm1(np.array(_s_valleys_log))) if _s_valleys_log else []

                # Fallback: if no valleys found, use IQR-based splits
                # (small = below median, medium = median to P75, large = above P75)
                if not _s_mag_edges and _s_n >= 20:
                    _p25, _p50, _p75 = np.percentile(_s_abs, [25, 50, 75])
                    _s_mag_edges = sorted(set([_p25, _p50, _p75]))
                    # Remove duplicates that are too close (< 1 pt apart)
                    _deduped = [_s_mag_edges[0]]
                    for _e in _s_mag_edges[1:]:
                        if _e - _deduped[-1] >= 1.0:
                            _deduped.append(_e)
                    _s_mag_edges = _deduped
                    print(f"  (Fallback: IQR splits at {', '.join(f'{v:.1f}' for v in _s_mag_edges)})")

                print(f"  Log-histogram: sigma={_sigma:.1f}, {len(_s_peaks)} peaks")
                print(f"  Valleys (real): {', '.join(f'{v:.1f}' for v in _s_mag_edges)}")

                # Build magnitude bins for this side
                _s_bin_edges = np.array([0.0] + _s_mag_edges + [_s_abs.max() + 1])
                _s_n_bins = len(_s_bin_edges) - 1
                _s_bin_ids = np.digitize(_s_abs, _s_bin_edges) - 1
                _s_bin_ids = np.clip(_s_bin_ids, 0, _s_n_bins - 1)

                print(f"  Magnitude bins: {_s_n_bins}")
                print(f"\n  {'Bin':>4} {'Magnitude':>18} {'N':>6} {'%side':>6} "
                      f"{'Mean|d|':>8} {'Med|d|':>8}")
                print(f"  {'-'*4} {'-'*18} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")

                _s_bin_stats = []
                for b in range(_s_n_bins):
                    _bm = _s_bin_ids == b
                    _bn = _bm.sum()
                    if _bn == 0:
                        continue
                    _b_abs = _s_abs[_bm]
                    _lo = _s_bin_edges[b]
                    _hi = _s_bin_edges[b + 1]
                    if b == _s_n_bins - 1:
                        _bname = f'{_side}_{_lo:.0f}+'
                    else:
                        _bname = f'{_side}_{_lo:.0f}_{_hi:.0f}'

                    _b_A = _s_A[_bm]
                    _b_B = _s_B[_bm]
                    _b_A_mean = _b_A.mean(axis=0)
                    _b_B_mean = _b_B.mean(axis=0)
                    _b_diff = _b_B_mean - _b_A_mean

                    _s_bin_stats.append({
                        'name': _bname, 'n': _bn,
                        'range': f'[{_lo:.1f}, {_hi:.1f})',
                        'pct_side': _bn / _s_n * 100,
                        'mean_abs': float(np.mean(_b_abs)),
                        'median_abs': float(np.median(_b_abs)),
                        'feat_A_mean': _b_A_mean,
                        'feat_B_mean': _b_B_mean,
                        'feat_diff': _b_diff,
                        'mask': _bm,
                    })
                    print(f"  {b:>4} {_bname:>18} {_bn:>6} {_bn/_s_n*100:>5.1f}% "
                          f"{np.mean(_b_abs):>8.2f} {np.median(_b_abs):>8.2f}")

                # 192D profiles per magnitude bin
                print(f"\n  192D PROFILES ({_side}):")
                for bs in _s_bin_stats:
                    if bs['n'] < 5:
                        print(f"\n  {bs['name']:>18} (n={bs['n']}): too few for profile")
                        continue
                    _z_A = (bs['feat_A_mean'] - _global_A_mean) / _global_A_std
                    _top5 = np.argsort(np.abs(_z_A))[::-1][:5]
                    _feat_str = ', '.join([
                        f"{col_names[fi] if fi < len(col_names) else f'f{fi}'}({_z_A[fi]:+.2f}z)"
                        for fi in _top5
                    ])
                    _diff_abs = np.abs(bs['feat_diff'])
                    _top3_diff = np.argsort(_diff_abs)[::-1][:3]
                    _diff_str = ', '.join([
                        f"{col_names[fi] if fi < len(col_names) else f'f{fi}'}({bs['feat_diff'][fi]:+.3f})"
                        for fi in _top3_diff
                    ])
                    print(f"\n  {bs['name']:>18} (n={bs['n']}, {bs['pct_side']:.1f}% of {_side}):")
                    print(f"    Context (A):     {_feat_str}")
                    print(f"    Transition A->B: {_diff_str}")

                # Spike tier for this side
                if _s_mag_edges and _s_n_bins >= 2:
                    _sp_thresh = _s_mag_edges[-1]  # highest internal boundary
                    _sp_mask = _s_abs >= _sp_thresh
                    _sp_n = _sp_mask.sum()
                    if _sp_n >= 2:
                        print(f"\n  SPIKE {_side} (|delta| >= {_sp_thresh:.1f}, n={_sp_n}):")
                        _sp_A_local = _s_A[_sp_mask]
                        _sp_z = (_sp_A_local - _global_A_mean) / _global_A_std
                        _sp_deltas_local = _s_deltas[_sp_mask]
                        _sp_abs_local = _s_abs[_sp_mask]
                        for _si in range(_sp_n):
                            _z_row = _sp_z[_si]
                            _top3 = np.argsort(np.abs(_z_row))[::-1][:3]
                            _feat_str = ', '.join([
                                f"{col_names[fi] if fi < len(col_names) else f'f{fi}'}({_z_row[fi]:+.1f}z)"
                                for fi in _top3
                            ])
                            print(f"    {_sp_deltas_local[_si]:>+10.1f} "
                                  f"({_sp_abs_local[_si]:>7.1f})  {_feat_str}")

                # Store for plots
                all_side_stats[_side] = {
                    'n': _s_n, 'bin_stats': _s_bin_stats,
                    'hist_centers': _s_centers, 'hist_counts': _s_hist_c,
                    'smooth': _s_smooth, 'peaks': _s_peaks,
                    'valleys_log': _s_valleys_log, 'mag_edges': _s_mag_edges,
                }

            # ── Combined summary ──────────────────────────────────────────
            print(f"\n  {'='*70}")
            print(f"  CROSS-SIDE COMPARISON")
            print(f"  {'='*70}")
            for _side in ['UP', 'DN']:
                _ss = all_side_stats[_side]
                _n_v = len(_ss['mag_edges'])
                print(f"  {_side}: {_ss['n']} moves, {len(_ss['peaks'])} modes, "
                      f"{_n_v} valleys -> {_n_v + 1} magnitude tiers")
                if _ss['mag_edges']:
                    print(f"    Boundaries: {', '.join(f'{v:.1f}' for v in _ss['mag_edges'])}")

            # ── Plots: per-side log histograms + magnitude tier bars ────
            fig_q, axes_q = plt.subplots(2, 2, figsize=(16, 12), facecolor='white')
            _side_colors = {'UP': '#2196F3', 'DN': '#F44336'}

            for _pi, _side in enumerate(['UP', 'DN']):
                _ss = all_side_stats[_side]
                _clr = _side_colors[_side]

                # Top row: log histogram + smoothed + modes/valleys
                ax = axes_q[0, _pi]
                ax.bar(_ss['hist_centers'], _ss['hist_counts'],
                       width=(_ss['hist_centers'][1] - _ss['hist_centers'][0]) * 0.9
                       if len(_ss['hist_centers']) > 1 else 0.05,
                       alpha=0.4, color=_clr, label='raw')
                ax.plot(_ss['hist_centers'], _ss['smooth'], color=_clr,
                        lw=2, label='smoothed')
                for _pk in _ss['peaks']:
                    ax.axvline(_ss['hist_centers'][_pk], color='green',
                               ls='--', alpha=0.5)
                for _vl in _ss['valleys_log']:
                    ax.axvline(_vl, color='red', ls='-', alpha=0.7)
                    _real_v = np.expm1(_vl)
                    ax.annotate(f'{_real_v:.1f}', xy=(_vl, ax.get_ylim()[1]*0.9),
                                fontsize=7, color='red', ha='center')
                ax.set_title(f'{_side} side: log1p(|delta|) histogram (n={_ss["n"]})',
                             fontsize=10)
                ax.set_xlabel('log1p(|delta|)')
                ax.set_ylabel('count')
                ax.legend(fontsize=8)

                # Bottom row: magnitude tier bar chart
                ax2 = axes_q[1, _pi]
                _bstats = _ss['bin_stats']
                if _bstats:
                    _bnames = [bs['name'] for bs in _bstats]
                    _bns = [bs['n'] for bs in _bstats]
                    _bars = ax2.bar(range(len(_bnames)), _bns, color=_clr, alpha=0.6)
                    ax2.set_xticks(range(len(_bnames)))
                    ax2.set_xticklabels(_bnames, rotation=45, ha='right', fontsize=7)
                    ax2.set_ylabel('count')
                    ax2.set_title(f'{_side} magnitude tiers', fontsize=10)
                    # Annotate pct on bars
                    for _bi, _bar in enumerate(_bars):
                        _pct = _bstats[_bi]['pct_side']
                        ax2.text(_bar.get_x() + _bar.get_width()/2, _bar.get_height(),
                                 f'{_pct:.1f}%', ha='center', va='bottom', fontsize=7)

            fig_q.suptitle('Analysis Q: Sign-First Magnitude Binning', fontsize=13)
            fig_q.tight_layout(rect=[0, 0, 1, 0.96])
            fig_q.savefig(os.path.join(PLOTS_DIR, 'analysis_q_signed_histogram.png'),
                          dpi=150, bbox_inches='tight')
            plt.close(fig_q)
            print(f"\n  [saved] analysis_q_signed_histogram.png")

            # ── CSV export: one row per signed bin ────────────────────────
            _csv_rows = []
            for _side in ['UP', 'DN']:
                for bs in all_side_stats[_side]['bin_stats']:
                    _csv_rows.append({
                        'side': _side,
                        'bin': bs['name'],
                        'range': bs.get('range', ''),
                        'n': bs['n'],
                        'pct_side': round(bs['pct_side'], 2),
                        'mean_abs': round(bs['mean_abs'], 2),
                        'median_abs': round(bs['median_abs'], 2),
                    })
            import csv
            _csv_path = os.path.join(PLOTS_DIR, 'analysis_q_signed_bins.csv')
            with open(_csv_path, 'w', newline='') as _cf:
                _w = csv.DictWriter(_cf, fieldnames=['side','bin','range','n',
                                                      'pct_side','mean_abs','median_abs'])
                _w.writeheader()
                _w.writerows(_csv_rows)
            print(f"  [saved] analysis_q_signed_bins.csv ({len(_csv_rows)} bins)")


    # =====================================================================
    #  ANALYSIS R: CNN PATTERN DETECTION
    # =====================================================================
    if _start_at <= 'R' and 'R' not in _skip_set:
        print(f"\n{'='*70}")
        print(f"  ANALYSIS R: CNN PATTERN DETECTION (Conv1D, 7 classes)")
        print(f"{'='*70}")

        try:
            import torch
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            from tools.cnn_pattern_model import (
                PatternCNN, PATTERN_CLASSES, SEED_TO_CLASS, N_CLASSES,
                WINDOW_LEN, extract_ohlcv_windows, TORCH_AVAILABLE as _torch_ok,
                detect_peak_touch_levels, compute_level_context,
            )
        except ImportError:
            _torch_ok = False

        if not _torch_ok:
            print("  SKIP: PyTorch not available")
        else:
            _cl = base_df['close'].values.astype(float)
            _hi = base_df['high'].values.astype(float)
            _lo = base_df['low'].values.astype(float)
            _active_levels = []

            # ── Step 0: Detect liquidation levels via peak-touch scanning ──
            try:
                # Resample intraday to daily for level detection
                _daily_df = base_df.copy()
                if hasattr(_daily_df.index, 'date'):
                    _daily_df['_date'] = _daily_df.index.date
                else:
                    _daily_df['_date'] = range(len(_daily_df))
                _daily_agg = _daily_df.groupby('_date').agg({
                    'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'
                })
                _d_hi = _daily_agg['high'].values.astype(float)
                _d_lo = _daily_agg['low'].values.astype(float)

                print(f"  Daily bars for level detection: {len(_d_hi)}")

                # Detect levels from daily swing peaks
                _levels = detect_peak_touch_levels(
                    _d_hi, _d_lo,
                    lookback_bars=min(60, len(_d_hi)),
                    swing_order=3, tolerance=75.0,
                    merge_dist=150.0, top_k=7)

                if _levels:
                    print(f"\n  Detected liquidation levels ({len(_levels)}):")
                    for _lp, _lt in _levels:
                        print(f"    {_lp:>10.0f}  ({_lt} touches)")
                else:
                    print(f"  No levels detected (need more daily data)")

                # Also load manual levels from checkpoint if available
                _manual_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'checkpoints', 'price_levels.json')
                _manual_levels = []
                if os.path.exists(_manual_path):
                    import json
                    with open(_manual_path) as _f:
                        _ml = json.load(_f)
                    _manual_levels = [(float(p), 99) for p in _ml.get('levels', [])]
                    print(f"\n  Manual levels from checkpoint ({len(_manual_levels)}):")
                    for _mp, _ in _manual_levels:
                        print(f"    {_mp:>10.0f}  (manual)")

                # Use detected levels (fall back to manual if detection empty)
                _active_levels = _levels if _levels else _manual_levels

                # Visualization: price chart with horizontal level lines
                _aw = price_imr['analysis_mask']
                _ai = np.where(_aw)[0]
                if len(_ai) > 0 and _active_levels:
                    _s0, _s1 = _ai[0], _ai[-1] + 1
                    _x = np.arange(_s1 - _s0)
                    _cl_w = _cl[_s0:_s1]

                    _fig, _ax = plt.subplots(figsize=(20, 10))
                    _ax.plot(_x, _cl_w, color='black', linewidth=0.5,
                             label='Close', zorder=3)

                    # Draw horizontal level lines
                    _colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12',
                               '#9b59b6', '#1abc9c', '#e67e22']
                    for _li, (_lp, _lt) in enumerate(_active_levels):
                        _c = _colors[_li % len(_colors)]
                        _lw = 1.0 + min(_lt, 5) * 0.4
                        _ax.axhline(y=_lp, color=_c, linewidth=_lw,
                                    alpha=0.7, linestyle='--',
                                    label=f'{_lp:.0f} ({_lt}t)')

                    _ax.set_xlabel('Bar index (analysis window)')
                    _ax.set_ylabel('Price')
                    _ax.set_title('Liquidation Levels (peak-touch detection)')
                    _ax.legend(loc='upper left', fontsize=8)
                    _ax.grid(True, alpha=0.3)
                    _fig.tight_layout()
                    _lv_path = os.path.join(PLOTS_DIR, '0v_liquidation_levels.png')
                    _fig.savefig(_lv_path, dpi=150)
                    plt.close(_fig)
                    print(f"\n  [saved] {_lv_path}")

                    # Context stats preview
                    _ctx_preview = compute_level_context(
                        _cl, _active_levels, _ai)
                    _da = _ctx_preview[:, 0]
                    _db = _ctx_preview[:, 1]
                    print(f"\n  Level context (dist_above / dist_below):")
                    print(f"    dist_above: {_da.mean():.3f} +/- {_da.std():.3f}  "
                          f"[{_da.min():.3f}, {_da.max():.3f}]")
                    print(f"    dist_below: {_db.mean():.3f} +/- {_db.std():.3f}  "
                          f"[{_db.min():.3f}, {_db.max():.3f}]")

            except Exception as _e:
                import traceback
                print(f"  (level detection failed: {_e})")
                traceback.print_exc()
                _active_levels = []

            # ── Step 1: analysis bar indices with enough lookback ──
            _analysis_mask_r = price_imr['analysis_mask']
            _analysis_idx_r = np.where(_analysis_mask_r)[0]
            _analysis_idx_r = _analysis_idx_r[_analysis_idx_r >= WINDOW_LEN - 1]
            print(f"  Analysis bars with {WINDOW_LEN}-bar lookback: {len(_analysis_idx_r):,}")

            # ── Step 2: pseudo-label via seed primitives ──
            _seed_len_r = best_seg_len if 'best_seg_len' in dir() else 16
            _lib_r = SeedPrimitiveLibrary(N=_seed_len_r)

            _labels_r, _valid_idx_r = [], []
            for _idx in tqdm(_analysis_idx_r, desc='  Labeling', ascii=True,
                             dynamic_ncols=True):
                if _idx + _seed_len_r > len(close):
                    continue
                _seg = close[_idx : _idx + _seed_len_r]
                _shape_name, _corr = _lib_r.classify_trajectory(_seg)
                if _shape_name == 'NOISE':
                    continue
                _cnn_cls = SEED_TO_CLASS.get(_shape_name)
                if _cnn_cls is None:
                    continue
                _labels_r.append(_cnn_cls)
                _valid_idx_r.append(_idx)

            _valid_idx_r = np.array(_valid_idx_r)
            _labels_r = np.array(_labels_r)
            print(f"  Labeled samples: {len(_labels_r):,}  (NOISE excluded)")

            if len(_labels_r) < 100:
                print(f"  SKIP: too few samples ({len(_labels_r)}, need >= 100)")
            else:
                # ── Step 3: extract OHLCV windows + level context ──
                _windows_r, _vmask_r = extract_ohlcv_windows(
                    base_df, _valid_idx_r, window_len=WINDOW_LEN)
                _labels_r = _labels_r[_vmask_r]
                _valid_idx_r = _valid_idx_r[_vmask_r]
                print(f"  Valid windows: {len(_labels_r):,}")

                # Liquidation level context (dist_above, dist_below)
                _n_ctx = 2  # 2 level-distance scalars
                if _active_levels:
                    _ctx_r = compute_level_context(
                        _cl, _active_levels, _valid_idx_r)
                    print(f"\n  Level context (liquidation distances):")
                    print(f"    dist_above: {_ctx_r[:,0].mean():.3f} +/- {_ctx_r[:,0].std():.3f}")
                    print(f"    dist_below: {_ctx_r[:,1].mean():.3f} +/- {_ctx_r[:,1].std():.3f}")
                else:
                    _ctx_r = np.full((len(_valid_idx_r), 2), 0.5, dtype=np.float32)
                    print(f"\n  No levels detected — using neutral context (0.5, 0.5)")

                # Class distribution
                print(f"\n  Class distribution:")
                for _ci, _cn in enumerate(PATTERN_CLASSES):
                    _nc = int((_labels_r == _ci).sum())
                    print(f"    {_cn:<20} {_nc:>6} ({_nc/len(_labels_r)*100:>5.1f}%)")

                # ── Step 4: stratified train/test split ──
                from sklearn.model_selection import train_test_split
                _i_train, _i_test = train_test_split(
                    np.arange(len(_labels_r)), test_size=0.30,
                    random_state=42, stratify=_labels_r)

                _X_train = torch.FloatTensor(_windows_r[_i_train])
                _C_train = torch.FloatTensor(_ctx_r[_i_train])
                _y_train = torch.LongTensor(_labels_r[_i_train])
                _X_test = torch.FloatTensor(_windows_r[_i_test])
                _C_test = torch.FloatTensor(_ctx_r[_i_test])
                _y_test = torch.LongTensor(_labels_r[_i_test])

                _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"\n  Device: {_device}  |  Train: {len(_i_train):,}  Test: {len(_i_test):,}")

                # ── Step 5: train CNN (dual-path: conv shapes + macro context) ──
                # Class weights for imbalanced data
                _counts = np.bincount(_labels_r, minlength=N_CLASSES).astype(float)
                _counts[_counts == 0] = 1.0
                _class_w = torch.FloatTensor(len(_labels_r) / (N_CLASSES * _counts)).to(_device)

                _model = PatternCNN(n_context=_n_ctx).to(_device)
                _criterion = torch.nn.CrossEntropyLoss(weight=_class_w)
                _optimizer = optim.Adam(_model.parameters(), lr=1e-3, weight_decay=1e-4)

                _train_ds = TensorDataset(_X_train, _C_train, _y_train)
                _train_loader = DataLoader(_train_ds, batch_size=64, shuffle=True)

                _N_EPOCHS = 50
                for _epoch in tqdm(range(_N_EPOCHS), desc='  Training CNN',
                                   ascii=True, dynamic_ncols=True):
                    _model.train()
                    for _xb, _cb, _yb in _train_loader:
                        _xb, _cb, _yb = _xb.to(_device), _cb.to(_device), _yb.to(_device)
                        _optimizer.zero_grad()
                        _loss = _criterion(_model(_xb, _cb), _yb)
                        _loss.backward()
                        _optimizer.step()

                # ── Step 6: evaluate ──
                _model.eval()
                with torch.no_grad():
                    _preds_test = _model(_X_test.to(_device), _C_test.to(_device)).argmax(dim=1).cpu().numpy()
                    _preds_train = _model(_X_train.to(_device), _C_train.to(_device)).argmax(dim=1).cpu().numpy()

                _y_test_np = _y_test.numpy()
                _y_train_np = _y_train.numpy()

                from sklearn.metrics import accuracy_score, classification_report

                _acc_train = accuracy_score(_y_train_np, _preds_train) * 100
                _acc_test = accuracy_score(_y_test_np, _preds_test) * 100
                _baseline = max(np.bincount(_labels_r)) / len(_labels_r) * 100

                print(f"\n  CNN Results ({_N_EPOCHS} epochs, lr=1e-3):")
                print(f"    Train accuracy:  {_acc_train:.1f}%")
                print(f"    Test accuracy:   {_acc_test:.1f}%")
                print(f"    Majority baseline: {_baseline:.1f}%")
                print(f"    Lift over baseline: {_acc_test - _baseline:+.1f}%")

                # Per-class report
                _present = sorted(set(_y_test_np))
                _tgt_names = [PATTERN_CLASSES[i] for i in _present]
                print(f"\n  Per-class report:")
                print(classification_report(
                    _y_test_np, _preds_test,
                    labels=_present, target_names=_tgt_names, zero_division=0))

                # ── Step 7: direction accuracy ──
                _dir_map = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0}  # 6=neutral, skip
                _dir_mask = np.array([p in _dir_map for p in _preds_test])
                if _dir_mask.sum() >= 10:
                    _pred_dirs = np.array([_dir_map[p] for p in _preds_test[_dir_mask]])
                    _test_bar_idx = _valid_idx_r[_i_test][_dir_mask]
                    _actual_dirs = []
                    for _bi in _test_bar_idx:
                        if _bi + _seed_len_r < len(close):
                            _actual_dirs.append(1 if close[_bi + _seed_len_r] > close[_bi] else 0)
                        else:
                            _actual_dirs.append(-1)
                    _actual_dirs = np.array(_actual_dirs)
                    _ok = _actual_dirs >= 0
                    if _ok.sum() >= 10:
                        _dir_acc = (_pred_dirs[_ok] == _actual_dirs[_ok]).mean() * 100
                        print(f"  DIRECTION ACCURACY:")
                        print(f"    CNN direction: {_dir_acc:.1f}%  "
                              f"({_ok.sum()} directional predictions)")
                        print(f"    Ref: Analysis K = 70.6%")
                else:
                    print(f"  Direction: too few directional predictions ({_dir_mask.sum()})")

                # ── Step 8: confusion matrix plot ──
                try:
                    from sklearn.metrics import confusion_matrix as _cm_func

                    _cm = _cm_func(_y_test_np, _preds_test, labels=_present)
                    _fig, _ax = plt.subplots(figsize=(8, 6))
                    _im = _ax.imshow(_cm, cmap='Blues')
                    _ax.set_xticks(range(len(_tgt_names)))
                    _ax.set_yticks(range(len(_tgt_names)))
                    _ax.set_xticklabels(_tgt_names, rotation=45, ha='right', fontsize=8)
                    _ax.set_yticklabels(_tgt_names, fontsize=8)
                    for _ri in range(len(_tgt_names)):
                        for _ci2 in range(len(_tgt_names)):
                            _ax.text(_ci2, _ri, str(_cm[_ri, _ci2]),
                                     ha='center', va='center', fontsize=9)
                    _ax.set_xlabel('Predicted')
                    _ax.set_ylabel('Actual')
                    _ax.set_title(f'CNN Pattern Confusion (test acc={_acc_test:.1f}%)')
                    _fig.colorbar(_im)
                    _fig.tight_layout()
                    _cm_path = os.path.join(PLOTS_DIR, '0v_cnn_confusion.png')
                    _fig.savefig(_cm_path, dpi=150)
                    plt.close(_fig)
                    print(f"\n  [saved] {_cm_path}")
                except Exception as _e:
                    print(f"  (confusion matrix plot failed: {_e})")

    else:
        print(f"  [SKIP] Analysis R (--start {_start_at})")


    # =====================================================================
    if _start_at <= 'S' and 'S' not in _skip_set:
        print(f"\n{'='*70}")
        print(f"  ANALYSIS S: EXIT TREND GUARD — BAND CONFLICT STUDY")
        print(f"{'='*70}")

        try:
            # ------------------------------------------------------------------
            # Goal: measure how often fast-TF band resistance triggers tighten
            # during a trend that slow TFs still support, and what suppressing
            # those tightens would do to trade duration / capture rate.
            #
            # We simulate multi-scale z-scores from price data:
            #   "slow" = 240-bar rolling regression (proxies 4h at 1m base)
            #   "mid"  = 60-bar rolling regression  (proxies 1h)
            #   "fast" = 15-bar rolling regression   (proxies 15m)
            # Then for each oracle trade we check band conflicts.
            # ------------------------------------------------------------------

            _closes = base_df['close'].values.astype(float)
            _n = len(_closes)

            # Compute rolling z-scores at 3 scales
            def _rolling_z(prices, window):
                """Z-score: (price - rolling_mean) / rolling_std."""
                z = np.full(len(prices), np.nan)
                for i in range(window, len(prices)):
                    seg = prices[i - window:i + 1]
                    mu = seg.mean()
                    sd = seg.std()
                    if sd > 1e-12:
                        z[i] = (prices[i] - mu) / sd
                    else:
                        z[i] = 0.0
                return z

            _z_slow = _rolling_z(_closes, 240)
            _z_mid  = _rolling_z(_closes, 60)
            _z_fast = _rolling_z(_closes, 15)

            print(f"  Computed rolling z-scores: slow(240), mid(60), fast(15)")
            print(f"  Valid bars: slow={np.sum(~np.isnan(_z_slow))}, "
                  f"mid={np.sum(~np.isnan(_z_mid))}, fast={np.sum(~np.isnan(_z_fast))}")

            # Build simulated trades from oracle labels
            _n_trades = min(len(bar_indices), len(directions), len(mfes), len(maes))
            print(f"  Oracle trades available: {_n_trades}")

            # For each trade, look forward up to 60 bars and track band conflicts
            _HOLD_MAX = 60
            _trade_results = []  # list of dicts per trade

            for _ti in range(_n_trades):
                _entry_bar = int(bar_indices[_ti])
                _dir_raw = directions[_ti]
                _dir = 1 if (str(_dir_raw).upper() == 'LONG' or _dir_raw == 1) else -1
                _mfe = float(mfes[_ti])
                _mae = float(maes[_ti])

                if _entry_bar + _HOLD_MAX >= _n:
                    continue
                if np.isnan(_z_slow[_entry_bar]) or np.isnan(_z_fast[_entry_bar]):
                    continue

                _tightens_no_guard = 0
                _tightens_with_guard = 0
                _first_tighten_no_guard = None
                _first_tighten_with_guard = None
                _slow_trend_bars = 0
                _fast_resist_bars = 0

                for _b in range(_HOLD_MAX):
                    _idx = _entry_bar + _b
                    _zs = _z_slow[_idx]
                    _zm = _z_mid[_idx]
                    _zf = _z_fast[_idx]

                    if np.isnan(_zs) or np.isnan(_zf):
                        continue

                    # Determine slow TF trend direction
                    _slow_long = _zs < -0.5   # price below slow mean = trending up potential
                    _slow_short = _zs > 0.5   # price above slow mean = trending down potential

                    # Fast TF band hit (resistance for LONG, support for SHORT)
                    if _dir == 1:  # LONG trade
                        _fast_at_resist = _zf >= 1.0
                        _slow_supports = _slow_long or (_zs < 0.3)  # slow not at resistance
                    else:  # SHORT trade
                        _fast_at_resist = _zf <= -1.0
                        _slow_supports = _slow_short or (_zs > -0.3)

                    if _slow_supports:
                        _slow_trend_bars += 1
                    if _fast_at_resist:
                        _fast_resist_bars += 1

                    # Without guard: fast resistance = tighten
                    if _fast_at_resist:
                        _tightens_no_guard += 1
                        if _first_tighten_no_guard is None:
                            _first_tighten_no_guard = _b

                    # With guard: suppress tighten if slow TF supports trade
                    if _fast_at_resist and not _slow_supports:
                        _tightens_with_guard += 1
                        if _first_tighten_with_guard is None:
                            _first_tighten_with_guard = _b

                _trade_results.append({
                    'dir': _dir,
                    'mfe': _mfe,
                    'mae': _mae,
                    'tightens_no_guard': _tightens_no_guard,
                    'tightens_with_guard': _tightens_with_guard,
                    'first_tighten_no': _first_tighten_no_guard,
                    'first_tighten_with': _first_tighten_with_guard,
                    'slow_trend_bars': _slow_trend_bars,
                    'fast_resist_bars': _fast_resist_bars,
                })

            print(f"  Analyzed {len(_trade_results)} trades (skipped {_n_trades - len(_trade_results)} edge/NaN)")

            if len(_trade_results) > 0:
                _tr = _trade_results
                _tightens_no = [t['tightens_no_guard'] for t in _tr]
                _tightens_wi = [t['tightens_with_guard'] for t in _tr]
                _first_no = [t['first_tighten_no'] for t in _tr if t['first_tighten_no'] is not None]
                _first_wi = [t['first_tighten_with'] for t in _tr if t['first_tighten_with'] is not None]

                _total_tightens_no = sum(_tightens_no)
                _total_tightens_wi = sum(_tightens_wi)
                _suppressed = _total_tightens_no - _total_tightens_wi
                _suppress_pct = (_suppressed / _total_tightens_no * 100) if _total_tightens_no > 0 else 0

                print(f"\n  --- Tighten Signal Summary ---")
                print(f"  Total tighten events (no guard):   {_total_tightens_no}")
                print(f"  Total tighten events (with guard): {_total_tightens_wi}")
                print(f"  Suppressed by trend guard:         {_suppressed} ({_suppress_pct:.1f}%)")

                if _first_no:
                    print(f"\n  First tighten bar (no guard):   median={np.median(_first_no):.0f}, "
                          f"mean={np.mean(_first_no):.1f}, p25={np.percentile(_first_no,25):.0f}")
                if _first_wi:
                    print(f"  First tighten bar (with guard): median={np.median(_first_wi):.0f}, "
                          f"mean={np.mean(_first_wi):.1f}, p25={np.percentile(_first_wi,25):.0f}")
                    _delay = np.mean(_first_wi) - np.mean(_first_no) if _first_no else 0
                    print(f"  Average delay from guard:       {_delay:+.1f} bars")

                # Split by direction
                for _side_name, _side_val in [('LONG', 1), ('SHORT', -1)]:
                    _side_tr = [t for t in _tr if t['dir'] == _side_val]
                    if not _side_tr:
                        continue
                    _s_no = sum(t['tightens_no_guard'] for t in _side_tr)
                    _s_wi = sum(t['tightens_with_guard'] for t in _side_tr)
                    _s_sup = _s_no - _s_wi
                    _s_pct = (_s_sup / _s_no * 100) if _s_no > 0 else 0
                    _avg_mfe = np.mean([t['mfe'] for t in _side_tr])
                    print(f"\n  {_side_name} trades ({len(_side_tr)}): "
                          f"tightens {_s_no}->{_s_wi} (suppressed {_s_sup}, {_s_pct:.1f}%), "
                          f"avg MFE={_avg_mfe:.1f}")

                # MFE analysis: trades with high suppression vs low
                _high_suppress = [t for t in _tr if t['tightens_no_guard'] > 0 and
                                  (t['tightens_no_guard'] - t['tightens_with_guard']) / t['tightens_no_guard'] > 0.5]
                _low_suppress = [t for t in _tr if t['tightens_no_guard'] > 0 and
                                 (t['tightens_no_guard'] - t['tightens_with_guard']) / t['tightens_no_guard'] <= 0.5]

                if _high_suppress and _low_suppress:
                    _mfe_high = np.mean([t['mfe'] for t in _high_suppress])
                    _mfe_low = np.mean([t['mfe'] for t in _low_suppress])
                    print(f"\n  --- MFE by Suppression Rate ---")
                    print(f"  High suppression (>50% tightens blocked): {len(_high_suppress)} trades, "
                          f"avg MFE={_mfe_high:.1f}")
                    print(f"  Low suppression (<=50% blocked):          {len(_low_suppress)} trades, "
                          f"avg MFE={_mfe_low:.1f}")
                    print(f"  MFE difference: {_mfe_high - _mfe_low:+.1f} "
                          f"({'guard helps' if _mfe_high > _mfe_low else 'guard neutral/hurts'})")

                # ── Plot 1: Tighten timeline histogram ──
                _fig, _axes = plt.subplots(2, 2, figsize=(14, 10))

                # Top-left: first tighten bar distribution
                if _first_no:
                    _axes[0, 0].hist(_first_no, bins=30, alpha=0.6, label='No guard', color='red')
                if _first_wi:
                    _axes[0, 0].hist(_first_wi, bins=30, alpha=0.6, label='With guard', color='green')
                _axes[0, 0].set_xlabel('Bar # of first tighten')
                _axes[0, 0].set_ylabel('Count')
                _axes[0, 0].set_title('First Tighten Timing (later = better)')
                _axes[0, 0].legend()

                # Top-right: suppression rate by MFE bucket
                _mfe_vals = np.array([t['mfe'] for t in _tr])
                _supp_rates = np.array([
                    (t['tightens_no_guard'] - t['tightens_with_guard']) / max(t['tightens_no_guard'], 1)
                    for t in _tr
                ])
                _mfe_bins = np.percentile(_mfe_vals[~np.isnan(_mfe_vals)], [0, 20, 40, 60, 80, 100])
                _bin_labels = []
                _bin_rates = []
                for _bi in range(len(_mfe_bins) - 1):
                    _mask = (_mfe_vals >= _mfe_bins[_bi]) & (_mfe_vals < _mfe_bins[_bi + 1] + 0.01)
                    if _mask.sum() > 0:
                        _bin_labels.append(f"{_mfe_bins[_bi]:.0f}-{_mfe_bins[_bi+1]:.0f}")
                        _bin_rates.append(np.mean(_supp_rates[_mask]) * 100)
                _axes[0, 1].bar(_bin_labels, _bin_rates, color='steelblue')
                _axes[0, 1].set_xlabel('MFE bucket (ticks)')
                _axes[0, 1].set_ylabel('Suppression rate (%)')
                _axes[0, 1].set_title('Guard Suppression Rate by MFE')
                _axes[0, 1].tick_params(axis='x', rotation=30)

                # Bottom-left: slow trend bars vs fast resist bars scatter
                _stb = np.array([t['slow_trend_bars'] for t in _tr])
                _frb = np.array([t['fast_resist_bars'] for t in _tr])
                _colors = ['green' if t['dir'] == 1 else 'red' for t in _tr]
                _axes[1, 0].scatter(_stb, _frb, c=_colors, alpha=0.3, s=10)
                _axes[1, 0].set_xlabel('Slow TF trend-confirming bars (of 60)')
                _axes[1, 0].set_ylabel('Fast TF resistance-hit bars')
                _axes[1, 0].set_title('Band Conflict: Slow Support vs Fast Resist')

                # Bottom-right: cumulative tighten suppression over hold time
                _cum_no = np.zeros(_HOLD_MAX)
                _cum_wi = np.zeros(_HOLD_MAX)
                for _t in _tr:
                    if _t['first_tighten_no'] is not None:
                        _cum_no[_t['first_tighten_no']:] += 1
                    if _t['first_tighten_with'] is not None:
                        _cum_wi[_t['first_tighten_with']:] += 1
                _axes[1, 1].plot(range(_HOLD_MAX), _cum_no, 'r-', label='No guard')
                _axes[1, 1].plot(range(_HOLD_MAX), _cum_wi, 'g-', label='With guard')
                _axes[1, 1].set_xlabel('Bars held')
                _axes[1, 1].set_ylabel('Cumulative trades with first tighten')
                _axes[1, 1].set_title('Cumulative First-Tighten (gap = guard benefit)')
                _axes[1, 1].legend()

                _fig.suptitle('Analysis S: Exit Trend Guard — Band Conflict Study', fontsize=14)
                _fig.tight_layout()
                _s_path = os.path.join(PLOTS_DIR, 'analysis_s_exit_trend_guard.png')
                _fig.savefig(_s_path, dpi=150)
                plt.close(_fig)
                print(f"\n  [saved] {_s_path}")

                # ── Plot 2: Z-score snapshot at trade entries ──
                _fig2, _ax2 = plt.subplots(figsize=(12, 5))
                _entry_bars = [int(bar_indices[i]) for i in range(min(_n_trades, len(bar_indices)))
                               if int(bar_indices[i]) + _HOLD_MAX < _n and not np.isnan(_z_slow[int(bar_indices[i])])]
                _zs_at_entry = [_z_slow[b] for b in _entry_bars[:500]]
                _zm_at_entry = [_z_mid[b] for b in _entry_bars[:500]]
                _zf_at_entry = [_z_fast[b] for b in _entry_bars[:500]]
                _x_idx = range(len(_zs_at_entry))
                _ax2.scatter(_x_idx, _zs_at_entry, s=8, alpha=0.5, label='Slow (240)', c='blue')
                _ax2.scatter(_x_idx, _zm_at_entry, s=8, alpha=0.5, label='Mid (60)', c='orange')
                _ax2.scatter(_x_idx, _zf_at_entry, s=8, alpha=0.5, label='Fast (15)', c='red')
                _ax2.axhline(1.0, color='gray', ls='--', alpha=0.5, label='Resist (+1)')
                _ax2.axhline(-1.0, color='gray', ls='--', alpha=0.5, label='Support (-1)')
                _ax2.set_xlabel('Trade #')
                _ax2.set_ylabel('Z-score at entry')
                _ax2.set_title('Multi-Scale Z-Score at Trade Entry')
                _ax2.legend(fontsize=8)
                _fig2.tight_layout()
                _s2_path = os.path.join(PLOTS_DIR, 'analysis_s_entry_zscores.png')
                _fig2.savefig(_s2_path, dpi=150)
                plt.close(_fig2)
                print(f"  [saved] {_s2_path}")

            else:
                print("  No valid trades to analyze.")

        except Exception as _e:
            import traceback
            print(f"  [ERROR] Analysis S failed: {_e}")
            traceback.print_exc()

    else:
        print(f"  [SKIP] Analysis S (--start {_start_at})")


    # Save the assembled feature matrix before full pipeline may overwrite `X`
    # (v1: 193 cols = 12*16+1 ; v2: 186 cols = 8*23+2)
    _expected_widths = (193, 186)
    _X_193d = X.copy() if 'X' in dir() and X is not None and len(X.shape) == 2 and X.shape[1] in _expected_widths else None

    # =====================================================================
    if not args.full:
        print(f"\n  (Skipping full 16D pipeline -- use --full to enable)")
        # Save report + exit
        sys.stdout = _orig_stdout
        _report_path = os.path.join(PLOTS_DIR, 'research_report.txt')
        with open(_report_path, 'w', encoding='utf-8', errors='replace') as _rf:
            _rf.write(_report_buf.getvalue())
        print(f"  [saved] {_report_path}")
        return

    if cache_mode == 'v2_features':
        print(f"\n  [SKIP] --full pipeline not supported in v2 mode "
              f"(uses build_stacked_matrices() which is v1-only). "
              f"Run without --cache (v1 SFE path) for the full 16D pipeline.")
        sys.stdout = _orig_stdout
        _report_path = os.path.join(PLOTS_DIR, 'research_report.txt')
        with open(_report_path, 'w', encoding='utf-8', errors='replace') as _rf:
            _rf.write(_report_buf.getvalue())
        print(f"  [saved] {_report_path}")
        return

    print(f"\n{'='*70}")
    print(f"  FULL 16D FRACTAL PIPELINE")
    print(f"{'='*70}")

    # --- 7. Load all TFs + compute physics ---
    print(f"\n--- STEP 7: Loading all TF data + physics ---")
    all_dfs = {args.base_tf: base_df}
    for tf in TF_HIERARCHY:
        if tf == args.base_tf:
            continue
        df = load_atlas_tf(args.data, tf, months=args.months)
        if not df.empty:
            all_dfs[tf] = df
            print(f"  {tf:>4}: {len(df):>8,} bars")
        else:
            print(f"  {tf:>4}:   (not found)")

    print(f"\n  Computing physics per TF...")
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="Physics", unit="tf", ascii=True, dynamic_ncols=True):
        if tf not in all_dfs:
            continue
        states = compute_tf_physics(tf, all_dfs[tf])
        if states:
            all_tf_states[tf] = states
            print(f"  {tf:>4}: {len(states):>8,} states computed")

    # --- 8. Build stacked TF state matrices (regime-based segmentation) ---
    print(f"\n--- STEP 8: Building multi-TF state matrices ---")
    matrices, mfes_16d, maes_16d, meta = build_stacked_matrices(
        all_tf_states, args.base_tf, all_dfs[args.base_tf],
        context_days=args.context_days,
        analysis_days=args.analysis_days
    )

    if len(matrices) < 20:
        print(f"ERROR: Only {len(matrices)} matrices built (need >= 20)")
        sys.exit(1)

    # Replace ADX quartile bins with regime IDs in meta
    ts_to_regime = {}
    timestamps = base_df['timestamp'].values.astype(float)
    for i, ts in enumerate(timestamps):
        if regime_ids[i] >= 0:
            ts_to_regime[int(ts)] = regime_ids[i]

    for m in meta:
        rid = ts_to_regime.get(int(m['ts']), -1)
        if rid >= 0:
            m['tid'] = f'regime_{rid}'
        # keep existing tid if no regime match

    # --- 9. Pad + I-MR chart plots ---
    print(f"\n--- STEP 9: Fractal I-MR Charts ---")
    padded = pad_to_fixed_depth(matrices, max_depth=12)
    plot_imr_charts(padded, mfes_16d)

    # Use 16D mfes/maes for the full pipeline from here on
    mfes = mfes_16d
    maes = maes_16d

    # --- 10. Flatten + MR segmentation ---
    print(f"\n--- STEP 10: MR segmentation ---")
    flat_i, col_names_i = flatten_matrices(padded)
    flat_mr, col_names_mr = compute_moving_range(padded)

    flat_z = np.hstack([flat_i, flat_mr])
    col_names_z = col_names_i + col_names_mr
    print(f"  Combined: {len(col_names_i)} I + {len(col_names_mr)} MR "
          f"= {len(col_names_z)} total features")

    # --- 11. Screen all three: I-only, MR-only, combined ---
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

    # --- 12. Summary comparison ---
    r2_i = steps_i[-1][3] if steps_i else 0
    r2_mr = steps_mr[-1][3] if steps_mr else 0
    r2_z = steps_z[-1][3] if steps_z else 0
    print(f"\n{'='*70}")
    print(f"  COMPARISON: adj-R² @ 20 factors")
    print(f"{'='*70}")
    print(f"  X (I values only):     {r2_i:.4f}")
    print(f"  Y (MR segments only):  {r2_mr:.4f}")
    print(f"  Z (X + Y combined):    {r2_z:.4f}")
    print(f"  Lift from MR:          {r2_z - r2_i:+.4f}")

    # --- 13. Directional split (16D pipeline uses DMI from meta) ---
    dmi_float = np.array([float(m.get('dmi_diff', 0)) for m in meta])
    long_mask = dmi_float >= 0
    short_mask = ~long_mask

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    wr_long = float((mfes[long_mask] > maes[long_mask]).mean()) if n_long > 0 else 0
    wr_short = float((mfes[short_mask] > maes[short_mask]).mean()) if n_short > 0 else 0

    print(f"\n{'='*70}")
    print(f"  DIRECTIONAL SPLIT (16D)")
    print(f"{'='*70}")
    print(f"  LONG  (DMI >= 0): {n_long:>5} points, WR={wr_long:.1%}")
    print(f"  SHORT (DMI <  0): {n_short:>5} points, WR={wr_short:.1%}")
    print(f"  Mixed WR:         {float((mfes > maes).mean()):.1%}")

    # --- 14. Segmented screening ---
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from collections import Counter, defaultdict

    tids = np.array([m['tid'] for m in meta])
    unique_tids = sorted(set(tids))

    print(f"\n{'='*70}")
    print(f"  SEGMENTED SCREENING: {len(unique_tids)} segments x 2 directions")
    print(f"{'='*70}")

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

            seg_screening = screen_factors(seg_flat, col_names_z, seg_mfes)
            top1_name, top1_corr, top1_abs = seg_screening[0]

            top5_names = [s[0] for s in seg_screening[:5]]
            top5_idx = [col_names_z.index(n) for n in top5_names]
            scaler = StandardScaler()
            X = scaler.fit_transform(seg_flat[:, top5_idx])
            reg = LinearRegression().fit(X, seg_mfes)
            r2 = reg.score(X, seg_mfes)
            n, k = X.shape
            adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - k - 1)

            # MR entry/exit signals
            mr_factors = [(nm, c, a) for nm, c, a in seg_screening
                          if nm.startswith('MR_') and a > 0.20]
            entry_signals = [(nm, c) for nm, c, a in mr_factors if c > 0]
            exit_signals = [(nm, c) for nm, c, a in mr_factors if c < 0]

            # Cpk / Ppk
            seg_mean = float(np.mean(seg_mfes))
            seg_std = float(np.std(seg_mfes))
            cpk = seg_mean / (3 * seg_std) if seg_std > 1e-6 else 0.0
            ppk = seg_mean / (3 * global_mfe_std) if global_mfe_std > 1e-6 else 0.0

            win_rate = float((seg_mfes > seg_maes).mean())
            p_positive = float((seg_mfes > 0).mean())
            p_good = float((seg_mfes > seg_mean).mean())

            median_mfe = float(np.median(seg_mfes))
            good_mask_seg = seg_mfes >= median_mfe
            bad_mask_seg = ~good_mask_seg

            good_bad_diff = []
            if good_mask_seg.sum() >= 5 and bad_mask_seg.sum() >= 5:
                for j, cname in enumerate(col_names_z):
                    col = seg_flat[:, j]
                    if np.std(col) < 1e-12:
                        continue
                    good_mean = float(np.mean(col[good_mask_seg]))
                    bad_mean = float(np.mean(col[bad_mask_seg]))
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
                'good_mfe_mean': float(np.mean(seg_mfes[good_mask_seg])),
                'bad_mfe_mean': float(np.mean(seg_mfes[bad_mask_seg])),
            })

    # Extract dominant context feature for each segment
    def _extract_feature(factor_name):
        parts = factor_name.split('__')
        return parts[-1] if parts else factor_name

    def _extract_depth(factor_name):
        if factor_name.startswith('MR_'):
            return factor_name.split('__')[0].replace('MR_', '')
        elif factor_name.startswith('UCL_'):
            return factor_name.split('__')[0].replace('UCL_', '')
        elif factor_name.startswith('slope__') or factor_name.startswith('mr_bar__') or factor_name.startswith('n_breaks__'):
            return 'all'
        else:
            return factor_name.split('__')[0]

    for s in seg_results:
        top5_features = [_extract_feature(fn) for fn, _ in s['top5']]
        feat_counts = Counter(top5_features)
        s['dominant_feature'] = feat_counts.most_common(1)[0][0]
        s['top1_feature'] = _extract_feature(s['top1'])
        s['top1_depth'] = _extract_depth(s['top1'])
        s['top1_src'] = 'I' if s['top1'] in col_names_i else 'MR'
        s['feature_profile'] = list(dict.fromkeys(top5_features))

    # --- 15. Model fission ---
    all_sorted = sorted(seg_results, key=lambda x: x['win_rate'], reverse=True)

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

    # KEEP segments detail
    if keep_segs:
        keep_n = sum(s['n'] for s in keep_segs)
        keep_wr = np.average([s['win_rate'] for s in keep_segs],
                             weights=[s['n'] for s in keep_segs])
        keep_mfe = np.average([s['mfe_mean'] for s in keep_segs],
                              weights=[s['n'] for s in keep_segs])
        print(f"\n  KEEP ({len(keep_segs)} segments, {keep_n} patterns, "
              f"WR={keep_wr:.1%}, avg MFE={keep_mfe:+.0f}):")
        for s in keep_segs:
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

    # SPLIT segments
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

    # DROP segments
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

    # --- 15b. Segmented I-MR plots ---
    print(f"\n--- Generating segmented I-MR plots ---")
    plot_segmented_imr(padded, mfes, maes, meta, tids, long_mask,
                       keep_segs, split_segs, drop_segs,
                       base_df=all_dfs.get(args.base_tf))

    # --- 16. Export gate config ---
    import json as _json

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
    _gate_path = os.path.join(os.path.dirname(__file__), 'screening_gates.json')
    with open(_gate_path, 'w') as _gf:
        _json.dump(_gate_config, _gf, indent=2)
    print(f"\n  >> Exported screening gates to {_gate_path}")
    print(f"     KEEP: {sum(1 for v in _fission_map.values() if v == 'KEEP')}, "
          f"SPLIT: {sum(1 for v in _fission_map.values() if v == 'SPLIT')}, "
          f"hours: {_gate_config['good_hours_utc']}")

    # --- 17. What-if impact ---
    total_n = sum(s['n'] for s in seg_results)
    total_wr = np.average([s['win_rate'] for s in seg_results],
                          weights=[s['n'] for s in seg_results]) if seg_results else 0
    total_mfe = np.average([s['mfe_mean'] for s in seg_results],
                           weights=[s['n'] for s in seg_results]) if seg_results else 0

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

    # --- 18. PID drill-down ---
    # v1: 'self_pid' (PID-controller integral). v2: 'reversion_prob' (OU first-passage).
    # _ACTIVE_PID_NAME is set to the appropriate one in main()/_set_active_spec_v2().
    if _ACTIVE_PID_NAME in FEATURE_NAMES:
        pid_idx = FEATURE_NAMES.index(_ACTIVE_PID_NAME)
    else:
        # Fallback: if active spec doesn't have the named feature, use index 0
        # so analysis runs but produces meaningless numbers (logged below).
        print(f"  WARNING: feature '{_ACTIVE_PID_NAME}' not in active FEATURE_NAMES; "
              f"using index 0 for PID drill-down.")
        pid_idx = 0

    print(f"\n{'='*70}")
    print(f"  PID DRILL-DOWN: I-MR x Direction")
    print(f"{'='*70}")

    # PID I-chart
    print(f"\n  PID I-CHART (mean value at each depth):")
    print(f"  {'Depth':<12} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'r(MFE)L':>9} {'r(MFE)S':>9}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    for d in range(12):
        pid_col = padded[:, d, pid_idx]
        l_mean = float(np.mean(pid_col[long_mask])) if long_mask.sum() > 0 else 0
        s_mean = float(np.mean(pid_col[short_mask])) if short_mask.sum() > 0 else 0
        corr_l = float(np.corrcoef(pid_col[long_mask], mfes[long_mask])[0, 1]) \
            if long_mask.sum() > 10 and np.std(pid_col[long_mask]) > 1e-12 else 0.0
        corr_s = float(np.corrcoef(pid_col[short_mask], mfes[short_mask])[0, 1]) \
            if short_mask.sum() > 10 and np.std(pid_col[short_mask]) > 1e-12 else 0.0
        if np.isnan(corr_l): corr_l = 0.0
        if np.isnan(corr_s): corr_s = 0.0
        lbl = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        print(f"  {lbl:<12} {l_mean:>+8.2f} {s_mean:>+8.2f} {l_mean - s_mean:>+8.2f} "
              f"{corr_l:>+9.4f} {corr_s:>+9.4f}")

    # PID MR
    mr_pid = np.diff(padded[:, :, pid_idx], axis=1)  # (n, 11)

    print(f"\n  PID MR (depth-to-depth gradient):")
    print(f"  {'Transition':<16} {'LONG':>8} {'SHORT':>8} {'Delta':>8} "
          f"{'r(MFE)L':>9} {'r(MFE)S':>9}")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    pid_mr_key_transitions = []
    for d in range(11):
        mr_col = mr_pid[:, d]
        l_mean = float(np.mean(mr_col[long_mask])) if long_mask.sum() > 0 else 0
        s_mean = float(np.mean(mr_col[short_mask])) if short_mask.sum() > 0 else 0
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

    # PID UCL breaks
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
        l_pct = float(brk[long_mask].mean()) * 100 if long_mask.sum() > 0 else 0
        s_pct = float(brk[short_mask].mean()) * 100 if short_mask.sum() > 0 else 0
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

    # PID profile by fission class
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
        pid_vals = padded[class_mask, :, pid_idx]
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

        pid_mr_class = np.diff(pid_vals, axis=1)
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

    # PID x direction confirmation
    print(f"\n  PID x DIRECTION CONFIRMATION:")
    print(f"  (Does PID sign at each depth agree with DMI direction?)")
    print(f"  {'Depth':<12} {'Agree%':>8} {'WR|agree':>10} {'WR|disagr':>10} {'Lift':>8}")
    print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*8}")

    for d in range(12):
        pid_col = padded[:, d, pid_idx]
        agree = (long_mask & (pid_col > 0)) | (short_mask & (pid_col < 0))
        disagree = ~agree
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

    # --- 19. Temporal special cause analysis ---
    from datetime import datetime, timezone

    ts_arr = np.array([m['ts'] for m in meta])
    valid_ts = ts_arr > 0
    n_valid = valid_ts.sum()

    print(f"\n{'='*70}")
    print(f"  TEMPORAL SPECIAL CAUSE ANALYSIS")
    print(f"  (Patterns with valid timestamps: {n_valid} / {len(meta)})")
    print(f"{'='*70}")

    if n_valid > 50:
        dts = np.array([
            datetime.fromtimestamp(t, tz=timezone.utc) if t > 0 else None
            for t in ts_arr
        ])
        hours_utc = np.array([dt.hour if dt else -1 for dt in dts])
        dow = np.array([dt.weekday() if dt else -1 for dt in dts])
        dom = np.array([dt.day if dt else -1 for dt in dts])

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

        # 1. Market sessions
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
            sl = smask & long_mask
            ss_m = smask & short_mask
            wr_l = float((mfes[sl] > maes[sl]).mean()) if sl.sum() > 5 else float('nan')
            wr_s = float((mfes[ss_m] > maes[ss_m]).mean()) if ss_m.sum() > 5 else float('nan')
            pid_d7 = float(np.mean(padded[smask, 7, pid_idx]))
            wr_l_str = f"{wr_l:.1%}" if not np.isnan(wr_l) else "n/a"
            wr_s_str = f"{wr_s:.1%}" if not np.isnan(wr_s) else "n/a"
            print(f"  {sess:<12} {n_s:>5} {wr:>7.1%} {mfe_m:>+7.0f} {mae_m:>7.0f} "
                  f"{wr_l_str:>7} {wr_s_str:>7} {pid_d7:>+8.2f}")

        # 2. Hourly breakdown
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

        # 3. Day of week
        dow_names = ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']

        # Build KEEP mask for cross-reference
        keep_mask_all = np.zeros(len(mfes), dtype=bool)
        for s in keep_segs:
            seg_m = (tids == s['tid']) & (long_mask if s['dir'] == 'LONG' else short_mask)
            keep_mask_all |= seg_m

        print(f"\n  3. DAY OF WEEK:")
        print(f"  {'Day':<5} {'N':>5} {'WR':>7} {'MFE':>7} {'MAE':>7} "
              f"{'WR_L':>7} {'WR_S':>7} {'KEEP_WR':>8}")
        print(f"  {'-'*5} {'-'*5} {'-'*7} {'-'*7} {'-'*7} "
              f"{'-'*7} {'-'*7} {'-'*8}")

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

        # 4. Month position
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

        # 5. Session open/close proximity
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

        for marker_label, t_start, t_end, parent_sess in markers:
            if t_start < t_end:
                mmask = (minutes_utc >= t_start) & (minutes_utc < t_end) & valid_ts
            else:
                mmask = ((minutes_utc >= t_start) | (minutes_utc < t_end)) & valid_ts
            n_m = mmask.sum()
            if n_m < 5:
                continue
            wr = float((mfes[mmask] > maes[mmask]).mean())
            mfe_m = float(np.mean(mfes[mmask]))
            sess_mask = (sessions == parent_sess) & valid_ts
            sess_wr = float((mfes[sess_mask] > maes[sess_mask]).mean()) if sess_mask.sum() > 10 else wr
            delta = wr - sess_wr
            print(f"  {marker_label:<20} {n_m:>5} {wr:>7.1%} {mfe_m:>+7.0f} {delta:>+7.1%}")

        # 6. Week position
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

        # 7. MR UCL breaks x Temporal
        mr_ucl_start = 11 * 16
        mr_ucl_end = mr_ucl_start + 11 * 16
        ucl_per_pattern = flat_mr[:, mr_ucl_start:mr_ucl_end].sum(axis=1)
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

        # Top MR breaks per session
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
            break_counts = []
            for tr_name, col_idx in mr_transitions:
                if col_idx >= flat_mr.shape[1]:
                    continue
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

    else:
        print(f"  (Skipped — insufficient valid timestamps)")

    # =====================================================================
    #  ANALYSIS T: PARTIAL BAR ROBUSTNESS
    #
    #  Live trading problem: slow TF bars (4h, 1h, 30m) are incomplete
    #  mid-bar. Current research uses only completed bars — best case.
    #  This analysis tests: how much accuracy degrades when slow TF
    #  features are stale (simulating live conditions)?
    #
    #  Approach: for each slow TF, substitute its physics with the
    #  nearest completed sub-TF as a proxy for partial bar state.
    #  E.g., 4h slot uses latest 1h state, 1h uses 30m, etc.
    # =====================================================================
    if _start_at <= 'T' and 'T' not in _skip_set:
        print(f"\n{'='*70}")
        print(f"  ANALYSIS T: PARTIAL BAR ROBUSTNESS")
        print(f"  Does prediction survive when slow TFs are stale/partial?")
        print(f"{'='*70}")

        try:
            # Check prerequisites from earlier analyses
            if _X_193d is None or len(_X_193d) == 0:
                raise RuntimeError("No 193D feature matrix — need --full without --start past A")
            if '_l_xrows' not in dir() or len(_l_xrows) < 50:
                raise RuntimeError("Need Analysis L signed MFE data (>= 50 samples)")

            from sklearn.linear_model import LinearRegression as _LR_T
            from sklearn.preprocessing import StandardScaler as _SS_T

            base_secs_t = TF_SECONDS.get(args.base_tf, 900)

            # Proxy map: slow TF -> use nearest faster TF as partial substitute
            PARTIAL_PROXY = {
                '1W': '1D', '1D': '4h', '4h': '1h',
                '1h': '30m', '30m': '15m',
            }

            # Identify which TF depth indices are "slow" (longer than base TF)
            slow_depths = []
            for depth_idx, tf in enumerate(TF_HIERARCHY):
                tf_secs = TF_SECONDS.get(tf, 60)
                if tf_secs > base_secs_t:
                    proxy_tf = PARTIAL_PROXY.get(tf)
                    if proxy_tf and proxy_tf in tf_sorted_ts:
                        slow_depths.append((depth_idx, tf, proxy_tf))

            print(f"\n  Slow TFs (> {args.base_tf}): {len(slow_depths)}")
            for di, tf, proxy in slow_depths:
                print(f"    depth {di} ({tf}) -> proxy: {proxy}")

            if not slow_depths:
                print(f"  No slow TFs to test — all TFs <= base TF")
                raise RuntimeError("No slow TFs")

            # ── Build partial feature matrix ──────────────────────────────
            # For each sample, rebuild the 192D vector but substitute slow
            # TF slots with the proxy TF's latest completed state
            def _build_mat_partial(t):
                """Like _build_mat but slow TFs use sub-TF proxy."""
                mat = np.zeros((12, 16))
                n = 0
                for depth_idx, tf in enumerate(TF_HIERARCHY):
                    # Check if this TF should use a proxy
                    use_proxy = False
                    for di, slow_tf, proxy_tf in slow_depths:
                        if depth_idx == di:
                            use_proxy = True
                            actual_tf = proxy_tf
                            break
                    if not use_proxy:
                        actual_tf = tf

                    if actual_tf not in tf_sorted_ts:
                        continue

                    tf_ts_list = tf_sorted_ts[actual_tf]
                    actual_secs = TF_SECONDS.get(actual_tf, 60)

                    if actual_secs > base_secs_t:
                        pos = np.searchsorted(tf_ts_list, t, side='right') - 2
                    else:
                        pos = np.searchsorted(tf_ts_list, t, side='right') - 1

                    if pos < 0:
                        continue

                    nearest_ts = tf_ts_list[pos]
                    state = all_tf_states[actual_tf][nearest_ts]
                    # Extract 16D but keep original TF label for tf_scale/depth
                    vec = extract_16d(state, actual_tf)
                    mat[depth_idx, :] = vec
                    n += 1
                return mat, n

            # Build X_partial for all samples
            mr_signed_t = price_imr['mr']
            timestamps_t = base_df['timestamp'].values
            X_partial_rows = []
            _partial_valid = []

            for xi, ts_val in enumerate(sample_ts):
                mat_p, n_p = _build_mat_partial(ts_val)
                if n_p < 3:
                    X_partial_rows.append(np.zeros(193))
                    _partial_valid.append(False)
                    continue
                # Find matching bar index for current_MR
                _bi = np.searchsorted(timestamps_t.astype(np.int64), ts_val, side='right') - 1
                if 0 <= _bi < len(mr_signed_t):
                    _mr = mr_signed_t[_bi]
                else:
                    _mr = 0.0
                x_row = np.concatenate([mat_p.flatten(), [_mr]])
                X_partial_rows.append(x_row)
                _partial_valid.append(True)

            X_partial = np.array(X_partial_rows)
            _partial_valid = np.array(_partial_valid)
            print(f"\n  Partial matrix: {X_partial.shape[0]} samples, "
                  f"{_partial_valid.sum()} valid ({_partial_valid.mean():.1%})")

            # ── Compare: complete vs partial on signed MFE (Analysis L model) ──
            _l_xrows_arr = np.array(_l_xrows)
            Y_l_arr = np.array(_l_smfe)

            # Filter to samples valid in both complete and partial
            _both_valid = _partial_valid[_l_xrows_arr]
            _valid_mask = _both_valid
            _xr = _l_xrows_arr[_valid_mask]
            _yl = Y_l_arr[_valid_mask]

            n_valid = len(_yl)
            print(f"  Signed MFE samples (both valid): {n_valid}")

            if n_valid < 30:
                raise RuntimeError(f"Too few valid samples: {n_valid}")
            if n_valid < 500:
                print(f"  WARNING: only {n_valid} samples with 193 features — "
                      f"results may be noisy. Use --analysis-days 120 for reliable test.")

            # Train/test split (70/30)
            _split = int(n_valid * 0.7)
            _train_idx = np.arange(_split)
            _test_idx = np.arange(_split, n_valid)

            X_complete_train = _X_193d[_xr[_train_idx]]
            X_complete_test = _X_193d[_xr[_test_idx]]
            X_partial_train = X_partial[_xr[_train_idx]]
            X_partial_test = X_partial[_xr[_test_idx]]
            Y_train = _yl[_train_idx]
            Y_test = _yl[_test_idx]

            # Standardize
            _sc_c = _SS_T()
            X_ct = _sc_c.fit_transform(X_complete_train)
            X_cte = _sc_c.transform(X_complete_test)
            _sc_p = _SS_T()
            X_pt = _sc_p.fit_transform(X_partial_train)
            X_pte = _sc_p.transform(X_partial_test)

            # ── Model A: Train on complete, test on complete (baseline) ──
            _m_cc = _LR_T()
            _m_cc.fit(X_ct, Y_train)
            _pred_cc = _m_cc.predict(X_cte)
            _r2_cc = _m_cc.score(X_cte, Y_test)
            _dir_cc = np.sign(_pred_cc)
            _dir_actual = np.sign(Y_test)
            _nz_cc = (_dir_actual != 0) & (_dir_cc != 0)
            _acc_cc = (_dir_cc[_nz_cc] == _dir_actual[_nz_cc]).mean() if _nz_cc.sum() > 0 else 0

            # ── Model B: Train on complete, test on PARTIAL (live simulation) ──
            # This is the realistic scenario: model trained on complete bars,
            # but at inference time slow TFs are stale/partial
            _pred_cp = _m_cc.predict(_sc_c.transform(X_partial[_xr[_test_idx]]))
            _dir_cp = np.sign(_pred_cp)
            _nz_cp = (_dir_actual != 0) & (_dir_cp != 0)
            _acc_cp = (_dir_cp[_nz_cp] == _dir_actual[_nz_cp]).mean() if _nz_cp.sum() > 0 else 0
            # R2 on partial input
            _ss_res_cp = np.sum((Y_test - _pred_cp) ** 2)
            _ss_tot = np.sum((Y_test - Y_test.mean()) ** 2)
            _r2_cp = 1 - _ss_res_cp / _ss_tot if _ss_tot > 0 else 0

            # ── Model C: Train on partial, test on partial (adapted) ──
            _m_pp = _LR_T()
            _m_pp.fit(X_pt, Y_train)
            _pred_pp = _m_pp.predict(X_pte)
            _r2_pp = _m_pp.score(X_pte, Y_test)
            _dir_pp = np.sign(_pred_pp)
            _nz_pp = (_dir_actual != 0) & (_dir_pp != 0)
            _acc_pp = (_dir_pp[_nz_pp] == _dir_actual[_nz_pp]).mean() if _nz_pp.sum() > 0 else 0

            # ── Results table ─────────────────────────────────────────────
            print(f"\n  SIGNED MFE PREDICTION: COMPLETE vs PARTIAL FEATURES")
            print(f"  (test set: {len(Y_test)} samples)")
            print(f"")
            print(f"  {'Scenario':<45} {'R2':>8} {'Dir Acc':>8} {'N':>5}")
            print(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*5}")
            print(f"  {'A: Train complete, test complete (baseline)':<45} "
                  f"{_r2_cc:>8.4f} {_acc_cc:>7.1%} {_nz_cc.sum():>5}")
            print(f"  {'B: Train complete, test PARTIAL (live sim)':<45} "
                  f"{_r2_cp:>8.4f} {_acc_cp:>7.1%} {_nz_cp.sum():>5}")
            print(f"  {'C: Train partial, test partial (adapted)':<45} "
                  f"{_r2_pp:>8.4f} {_acc_pp:>7.1%} {_nz_pp.sum():>5}")

            # Degradation summary
            _deg_r2 = _r2_cp - _r2_cc
            _deg_acc = _acc_cp - _acc_cc
            print(f"\n  DEGRADATION (B vs A):")
            print(f"    R2:        {_deg_r2:+.4f}  ({_deg_r2/_r2_cc*100:+.1f}%)" if _r2_cc != 0
                  else f"    R2:        {_deg_r2:+.4f}")
            print(f"    Direction: {_deg_acc:+.1%}")

            _recov_r2 = _r2_pp - _r2_cp
            _recov_acc = _acc_pp - _acc_cp
            print(f"\n  RECOVERY (C vs B — does retraining on partial help?):")
            print(f"    R2:        {_recov_r2:+.4f}")
            print(f"    Direction: {_recov_acc:+.1%}")

            # ── Per-TF staleness analysis ─────────────────────────────────
            # Test each slow TF individually to find which one hurts most
            print(f"\n  PER-TF STALENESS IMPACT:")
            print(f"  (substitute ONE slow TF at a time, measure degradation)")
            print(f"")
            print(f"  {'TF':<8} {'Proxy':<8} {'R2':>8} {'Dir Acc':>8} {'dR2':>8} {'dAcc':>8}")
            print(f"  {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

            for di, tf, proxy_tf in slow_depths:
                # Build matrix with ONLY this TF substituted
                def _build_mat_single_sub(t, sub_depth=di, sub_proxy=proxy_tf):
                    mat = np.zeros((12, 16))
                    n = 0
                    for depth_idx, tf_i in enumerate(TF_HIERARCHY):
                        actual_tf = sub_proxy if depth_idx == sub_depth else tf_i
                        if actual_tf not in tf_sorted_ts:
                            continue
                        tf_ts_list = tf_sorted_ts[actual_tf]
                        actual_secs = TF_SECONDS.get(actual_tf, 60)
                        if actual_secs > base_secs_t:
                            pos = np.searchsorted(tf_ts_list, t, side='right') - 2
                        else:
                            pos = np.searchsorted(tf_ts_list, t, side='right') - 1
                        if pos < 0:
                            continue
                        nearest_ts = tf_ts_list[pos]
                        state = all_tf_states[actual_tf][nearest_ts]
                        mat[depth_idx, :] = extract_16d(state, actual_tf)
                        n += 1
                    return mat, n

                _X_single = []
                for xi_s in _xr[_test_idx]:
                    ts_s = sample_ts[xi_s]
                    _bi_s = np.searchsorted(timestamps_t.astype(np.int64), ts_s, side='right') - 1
                    _mr_s = mr_signed_t[_bi_s] if 0 <= _bi_s < len(mr_signed_t) else 0.0
                    mat_s, _ = _build_mat_single_sub(ts_s)
                    _X_single.append(np.concatenate([mat_s.flatten(), [_mr_s]]))
                _X_single = np.array(_X_single)
                _pred_s = _m_cc.predict(_sc_c.transform(_X_single))
                _ss_res_s = np.sum((Y_test - _pred_s) ** 2)
                _r2_s = 1 - _ss_res_s / _ss_tot if _ss_tot > 0 else 0
                _dir_s = np.sign(_pred_s)
                _nz_s = (_dir_actual != 0) & (_dir_s != 0)
                _acc_s = (_dir_s[_nz_s] == _dir_actual[_nz_s]).mean() if _nz_s.sum() > 0 else 0
                _dr2 = _r2_s - _r2_cc
                _dacc = _acc_s - _acc_cc

                flag = " ← CRITICAL" if _dacc < -0.05 else ""
                print(f"  {tf:<8} {proxy_tf:<8} {_r2_s:>8.4f} {_acc_s:>7.1%} "
                      f"{_dr2:>+8.4f} {_dacc:>+7.1%}{flag}")

            # ── Conclusion ────────────────────────────────────────────────
            print(f"\n  ANALYSIS T CONCLUSION:")
            if abs(_deg_acc) < 0.03:
                print(f"  ROBUST: partial bars degrade direction by only {_deg_acc:+.1%}.")
                print(f"  The model tolerates stale slow-TF features well.")
                print(f"  E[PnL] predictor can use completed-bar-trained model in live.")
            elif abs(_deg_acc) < 0.10:
                print(f"  MODERATE DEGRADATION: {_deg_acc:+.1%} direction loss with partial bars.")
                print(f"  Consider retraining on partial features (scenario C) for live use.")
                if _recov_acc > 0.02:
                    print(f"  Retraining recovers {_recov_acc:+.1%} — worth doing.")
            else:
                print(f"  FRAGILE: {_deg_acc:+.1%} direction loss — model relies on slow TF bars.")
                print(f"  Partial bar aggregation (maturity weighting) is critical for live.")
                if _recov_acc > 0.03:
                    print(f"  Retraining on partial features recovers {_recov_acc:+.1%}.")

            # ── Plot: complete vs partial accuracy comparison ──────────────
            fig_t, axes_t = plt.subplots(1, 2, figsize=(14, 5))

            # Left: scatter pred vs actual for complete and partial
            ax1 = axes_t[0]
            ax1.scatter(_pred_cc, Y_test, alpha=0.4, s=15, c='steelblue', label='Complete')
            ax1.scatter(_pred_cp, Y_test, alpha=0.4, s=15, c='coral', label='Partial (live)')
            _lim = max(abs(Y_test).max(), abs(_pred_cc).max(), abs(_pred_cp).max()) * 1.1
            ax1.plot([-_lim, _lim], [-_lim, _lim], 'k--', alpha=0.3)
            ax1.axhline(0, color='gray', alpha=0.3)
            ax1.axvline(0, color='gray', alpha=0.3)
            ax1.set_xlabel('Predicted Signed MFE')
            ax1.set_ylabel('Actual Signed MFE')
            ax1.set_title(f'Signed MFE: Complete R²={_r2_cc:.3f} vs Partial R²={_r2_cp:.3f}')
            ax1.legend(fontsize=8)

            # Right: per-TF degradation bar chart
            ax2 = axes_t[1]
            _tf_names = [tf for _, tf, _ in slow_depths]
            _tf_daccs = []
            for di, tf, proxy_tf in slow_depths:
                # Recompute per-TF degradation for chart (reuse values from loop)
                _X_s2 = []
                for xi_s in _xr[_test_idx]:
                    ts_s = sample_ts[xi_s]
                    _bi_s2 = np.searchsorted(timestamps_t.astype(np.int64), ts_s, side='right') - 1
                    _mr_s2 = mr_signed_t[_bi_s2] if 0 <= _bi_s2 < len(mr_signed_t) else 0.0
                    mat_s2, _ = _build_mat_single_sub(ts_s, sub_depth=di, sub_proxy=proxy_tf)
                    _X_s2.append(np.concatenate([mat_s2.flatten(), [_mr_s2]]))
                _X_s2 = np.array(_X_s2)
                _pred_s2 = _m_cc.predict(_sc_c.transform(_X_s2))
                _dir_s2 = np.sign(_pred_s2)
                _nz_s2 = (_dir_actual != 0) & (_dir_s2 != 0)
                _acc_s2 = (_dir_s2[_nz_s2] == _dir_actual[_nz_s2]).mean() if _nz_s2.sum() > 0 else 0
                _tf_daccs.append(_acc_s2 - _acc_cc)

            colors = ['#d32f2f' if d < -0.05 else '#ff9800' if d < -0.02 else '#4caf50'
                      for d in _tf_daccs]
            ax2.barh(_tf_names, [d * 100 for d in _tf_daccs], color=colors)
            ax2.axvline(0, color='black', linewidth=0.8)
            ax2.set_xlabel('Direction Accuracy Change (%)')
            ax2.set_title('Per-TF Staleness Impact')
            for i, (tf_n, d) in enumerate(zip(_tf_names, _tf_daccs)):
                ax2.text(d * 100 + (0.3 if d >= 0 else -0.3), i, f'{d:+.1%}',
                        va='center', ha='left' if d >= 0 else 'right', fontsize=8)

            fig_t.tight_layout()
            _plot_path_t = os.path.join(PLOTS_DIR, 'analysis_t_partial_bar_robustness.png')
            fig_t.savefig(_plot_path_t, dpi=150)
            plt.close(fig_t)
            print(f"\n  [saved] {_plot_path_t}")

        except Exception as _e_t:
            print(f"  [ERROR] Analysis T: {_e_t}")
            import traceback; traceback.print_exc()

    else:
        print(f"  [SKIP] Analysis T (--start {_start_at})")


    # =====================================================================
    #  ANALYSIS U: EXPECTED MOVE CONFIDENCE INTERVAL (dp/dt psychohistory)
    #
    #  For each bar, find k-nearest neighbors in 192D feature space.
    #  The neighbors' oracle MFE distribution gives a confidence interval
    #  for the expected move: [P10, P25, P50, P75, P90].
    #
    #  We don't predict the exact path — we predict the DESTINATION
    #  with a confidence interval.  Like psychohistory: individual bars
    #  are unpredictable, but patterns with similar characteristics
    #  (velocity, momentum, hurst, coherence) resolve predictably.
    #
    #  Outputs:
    #    1. CI coverage: how often does actual MFE fall within predicted CI?
    #    2. CI width vs accuracy tradeoff
    #    3. dp/dt signature scoring: which physics features tighten the CI?
    #    4. Counter-trend context: CI direction vs regime direction
    #    5. Ambition ratio: if TP were set to P50, how realistic?
    # =====================================================================
    if _start_at <= 'U' and 'U' not in _skip_set:
        print(f"\n{'='*70}")
        print(f"  ANALYSIS U: EXPECTED MOVE CONFIDENCE INTERVAL")
        print(f"  dp/dt psychohistory — predict the destination, not the path")
        print(f"  Method: k-NN in 192D feature space -> neighbor MFE distribution")
        print(f"{'='*70}")

        try:
            from sklearn.preprocessing import StandardScaler as _SS_U
            from sklearn.neighbors import NearestNeighbors as _KNN_U

            # ── Bridge: match full-pipeline oracle (meta) to step-8 X rows ──
            # meta[i]['ts'] = timestamp of the i-th stacked matrix
            # sample_ts[j] = timestamp of the j-th X row from step 8
            # We need pairs (j, i) where timestamps match
            _meta_ts_map = {int(m['ts']): _i for _i, m in enumerate(meta)}
            _sample_ts_map = {int(t): _j for _j, t in enumerate(sample_ts)}

            _u_xi = []       # index into _X_193d (step 8 rows)
            _u_oi = []       # index into mfes/maes/meta (full pipeline rows)
            for _ts_int, _oi in _meta_ts_map.items():
                _xi = _sample_ts_map.get(_ts_int, -1)
                if _xi >= 0:
                    _u_xi.append(_xi)
                    _u_oi.append(_oi)

            _n_u = len(_u_xi)
            print(f"\n  Matched samples: {_n_u} (oracle bars with fractal context)")

            if _n_u < 30:
                raise RuntimeError(f"Need >= 30 matched samples, got {_n_u}")

            # Use _X_193d (step 8, 193 features) indexed by _u_xi
            _X_u = _X_193d[_u_xi] if _X_193d is not None else X[_u_xi]
            _mfe_u = np.array([float(mfes[o]) for o in _u_oi])
            _mae_u = np.array([float(maes[o]) for o in _u_oi])
            # Direction from meta's dmi_diff (positive=LONG, negative=SHORT)
            _dir_u = np.array([
                'LONG' if meta[o].get('dmi_diff', 0) >= 0 else 'SHORT'
                for o in _u_oi
            ])
            _signed_mfe_u = np.array([
                float(mfes[o]) * (1.0 if meta[o].get('dmi_diff', 0) >= 0 else -1.0)
                for o in _u_oi
            ])

            print(f"  Feature matrix: {_X_u.shape}")
            print(f"  MFE: mean={_mfe_u.mean():.1f}, std={_mfe_u.std():.1f}")
            print(f"  Signed MFE: mean={_signed_mfe_u.mean():.1f}, std={_signed_mfe_u.std():.1f}")

            # ── Scale features and fit k-NN ──
            _k = min(50, _n_u // 5)  # 50 neighbors or 20% of data
            print(f"  k = {_k} neighbors")

            _sc_u = _SS_U()
            _X_u_sc = _sc_u.fit_transform(_X_u)

            _knn = _KNN_U(n_neighbors=_k + 1, metric='euclidean', n_jobs=-1)
            _knn.fit(_X_u_sc)

            # ── For each sample, get neighbor MFE distribution ──
            # Use leave-one-out: query all, exclude self (index 0 in results)
            _dists, _indices = _knn.kneighbors(_X_u_sc)

            # Percentiles for confidence interval
            _pcts = [10, 25, 50, 75, 90]
            _ci_results = []

            for _i in range(_n_u):
                # Exclude self (first neighbor is always self)
                _nbr_idx = _indices[_i, 1:]  # skip self
                _nbr_mfe = _mfe_u[_nbr_idx]
                _nbr_signed = _signed_mfe_u[_nbr_idx]
                _nbr_dir = _dir_u[_nbr_idx]

                # CI from neighbor signed MFE distribution
                _ci = {f'p{p}': float(np.percentile(_nbr_signed, p)) for p in _pcts}
                _ci['actual_mfe'] = float(_mfe_u[_i])
                _ci['actual_signed_mfe'] = float(_signed_mfe_u[_i])
                _ci['actual_dir'] = _dir_u[_i]
                _ci['predicted_dir'] = 'LONG' if _ci['p50'] > 0 else 'SHORT'
                _ci['ci_width'] = _ci['p75'] - _ci['p25']  # IQR
                _ci['ci_wide'] = _ci['p90'] - _ci['p10']   # 80% CI

                # Neighbor agreement on direction
                _nbr_long_pct = (_nbr_dir == 'LONG').sum() / len(_nbr_dir)
                _ci['nbr_long_pct'] = float(_nbr_long_pct)
                _ci['nbr_consensus'] = max(_nbr_long_pct, 1 - _nbr_long_pct)

                _ci_results.append(_ci)

            # ── 1. CI Coverage: how often does actual fall within predicted CI? ──
            print(f"\n  CONFIDENCE INTERVAL COVERAGE:")
            print(f"  {'CI Range':<20} {'Coverage':>10} {'Avg Width':>12} {'Interpretation'}")
            print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*30}")

            for _lo_p, _hi_p, _label in [
                ('p25', 'p75', '50% CI (IQR)'),
                ('p10', 'p90', '80% CI'),
            ]:
                _hits = sum(1 for c in _ci_results
                           if c[_lo_p] <= c['actual_signed_mfe'] <= c[_hi_p])
                _cov = _hits / _n_u
                _avg_w = np.mean([c[_hi_p] - c[_lo_p] for c in _ci_results])
                _ideal = int(_hi_p[1:]) - int(_lo_p[1:])
                _interp = "CALIBRATED" if abs(_cov * 100 - _ideal) < 10 else \
                          "OVER-CONFIDENT" if _cov * 100 < _ideal - 10 else "CONSERVATIVE"
                print(f"  {_label:<20} {_cov:>9.1%} {_avg_w:>11.1f}t {_interp}")

            # P50 as point prediction
            _p50_within_20 = sum(1 for c in _ci_results
                                if c['actual_signed_mfe'] != 0 and
                                abs(c['p50'] - c['actual_signed_mfe']) / max(abs(c['actual_signed_mfe']), 1) <= 0.20)
            print(f"\n  P50 within 20% of actual: {_p50_within_20}/{_n_u} = {_p50_within_20/_n_u:.1%}")

            # ── 2. Direction accuracy from CI ──
            _dir_correct = sum(1 for c in _ci_results if c['predicted_dir'] == c['actual_dir'])
            _dir_acc_u = _dir_correct / _n_u
            _baseline_u = max(sum(1 for c in _ci_results if c['actual_dir'] == 'LONG'),
                             sum(1 for c in _ci_results if c['actual_dir'] == 'SHORT')) / _n_u

            print(f"\n  DIRECTION FROM P50 (sign of median neighbor MFE):")
            print(f"    Accuracy:  {_dir_correct}/{_n_u} = {_dir_acc_u:.1%}")
            print(f"    Baseline:  {_baseline_u:.1%}")
            print(f"    Lift:      {_dir_acc_u - _baseline_u:+.1%}")

            # Direction accuracy by consensus strength
            print(f"\n  DIRECTION BY NEIGHBOR CONSENSUS:")
            print(f"  {'Consensus':>12} {'N':>6} {'Dir Acc':>9} {'Avg |P50|':>11} {'Avg IQR':>9}")
            print(f"  {'-'*12} {'-'*6} {'-'*9} {'-'*11} {'-'*9}")
            for _lo_cons, _hi_cons in [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]:
                _grp = [c for c in _ci_results if _lo_cons <= c['nbr_consensus'] < _hi_cons]
                if len(_grp) >= 10:
                    _ga = sum(1 for c in _grp if c['predicted_dir'] == c['actual_dir']) / len(_grp)
                    _gp50 = np.mean([abs(c['p50']) for c in _grp])
                    _giqr = np.mean([c['ci_width'] for c in _grp])
                    print(f"  {_lo_cons:.0%}-{_hi_cons:.0%}{' ':>5} {len(_grp):>6} {_ga:>9.1%} {_gp50:>10.1f}t {_giqr:>8.1f}t")

            # ── 3. dp/dt signature: which physics features correlate with CI tightness? ──
            print(f"\n  dp/dt SIGNATURE — FEATURES THAT TIGHTEN THE CI:")
            print(f"  (negative correlation with IQR width = tighter CI = more predictable)")

            _iqr_arr = np.array([c['ci_width'] for c in _ci_results])
            _feat_iqr_corr = []
            for _fi in range(_X_u.shape[1]):
                _col = _X_u[:, _fi]
                if np.std(_col) > 1e-10:
                    _r = float(np.corrcoef(_col, _iqr_arr)[0, 1])
                    _fname = col_names[_fi] if _fi < len(col_names) else f'f{_fi}'
                    _feat_iqr_corr.append((_fname, _r, abs(_r)))

            _feat_iqr_corr.sort(key=lambda x: -x[2])
            print(f"\n  {'Rank':>4} {'Feature':<40} {'r(IQR)':>8} {'Effect'}")
            print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*20}")
            for _ri, (_fn, _rv, _av) in enumerate(_feat_iqr_corr[:15], 1):
                _eff = "tightens CI" if _rv < 0 else "widens CI"
                print(f"  {_ri:>4} {_fn:<40} {_rv:>+8.4f} {_eff}")

            # ── 4. Counter-trend context ──
            # For each bar, compare CI direction vs regime direction
            # Map meta timestamps back to base_df indices for regime lookup
            _base_ts_arr = base_df['timestamp'].values.astype(float)
            _ts_to_baseidx = {int(t): _i for _i, t in enumerate(_base_ts_arr)}
            _regime_at_bar = np.array([
                regime_ids[_ts_to_baseidx[int(meta[o]['ts'])]]
                if int(meta[o]['ts']) in _ts_to_baseidx and
                   _ts_to_baseidx[int(meta[o]['ts'])] < len(regime_ids)
                else -1
                for o in _u_oi
            ])
            _regime_dir_at_bar = []
            _regime_meta_map = {rm['regime_id']: rm for rm in regime_meta}
            for _ri_val in _regime_at_bar:
                _rm = _regime_meta_map.get(_ri_val, None)
                _regime_dir_at_bar.append(_rm['direction'] if _rm else 'UNKNOWN')
            _regime_dir_at_bar = np.array(_regime_dir_at_bar)

            # Classify: with-trend vs counter-trend based on CI
            _with_trend = []
            _counter_trend = []
            for _i, _c in enumerate(_ci_results):
                if _regime_dir_at_bar[_i] == 'UNKNOWN':
                    continue
                if _c['predicted_dir'] == _regime_dir_at_bar[_i]:
                    _with_trend.append(_c)
                else:
                    _counter_trend.append(_c)

            print(f"\n  COUNTER-TREND CONTEXT (CI direction vs regime direction):")
            print(f"  {'Category':<20} {'N':>6} {'Dir Acc':>9} {'Avg |P50|':>11} {'Avg MFE':>10} {'Avg IQR':>9}")
            print(f"  {'-'*20} {'-'*6} {'-'*9} {'-'*11} {'-'*10} {'-'*9}")
            for _cat, _grp in [('With-trend', _with_trend), ('Counter-trend', _counter_trend)]:
                if _grp:
                    _ga = sum(1 for c in _grp if c['predicted_dir'] == c['actual_dir']) / len(_grp)
                    _gp50 = np.mean([abs(c['p50']) for c in _grp])
                    _gmfe = np.mean([c['actual_mfe'] for c in _grp])
                    _giqr = np.mean([c['ci_width'] for c in _grp])
                    print(f"  {_cat:<20} {len(_grp):>6} {_ga:>9.1%} {_gp50:>10.1f}t {_gmfe:>9.1f}t {_giqr:>8.1f}t")

            # ── 5. Ambition ratio: P50 as TP target ──
            # If TP = |P50|, what capture rate would we get?
            _ambition = []
            for _c in _ci_results:
                _pred_mag = abs(_c['p50'])
                _actual_mfe = _c['actual_mfe']
                if _pred_mag > 0 and _actual_mfe > 0:
                    _ratio = _pred_mag / _actual_mfe
                    _ambition.append({
                        'ratio': _ratio,
                        'pred_mag': _pred_mag,
                        'actual_mfe': _actual_mfe,
                        'dir_correct': _c['predicted_dir'] == _c['actual_dir'],
                        'achievable': _actual_mfe >= _pred_mag,
                    })

            if _ambition:
                _achievable = sum(1 for a in _ambition if a['achievable'])
                _avg_ratio = np.mean([a['ratio'] for a in _ambition])
                _med_pred = np.median([a['pred_mag'] for a in _ambition])
                _med_actual = np.median([a['actual_mfe'] for a in _ambition])

                print(f"\n  AMBITION RATIO (P50 as TP target):")
                print(f"    Avg ratio (|P50| / actual MFE):  {_avg_ratio:.2f}")
                print(f"    Median predicted magnitude:      {_med_pred:.1f} ticks")
                print(f"    Median actual MFE:               {_med_actual:.1f} ticks")
                print(f"    Achievable (MFE >= |P50|):       {_achievable}/{len(_ambition)} = {_achievable/len(_ambition):.1%}")

                # Bucket by ambition ratio
                print(f"\n  {'Ambition':>12} {'N':>6} {'Achievable':>12} {'Avg MFE':>10} {'Dir Acc':>9}")
                print(f"  {'-'*12} {'-'*6} {'-'*12} {'-'*10} {'-'*9}")
                for _lo_a, _hi_a in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 99)]:
                    _ab = [a for a in _ambition if _lo_a <= a['ratio'] < _hi_a]
                    if len(_ab) >= 5:
                        _ach = sum(1 for a in _ab if a['achievable']) / len(_ab)
                        _amfe = np.mean([a['actual_mfe'] for a in _ab])
                        _adir = sum(1 for a in _ab if a['dir_correct']) / len(_ab)
                        _label = f"{_lo_a:.1f}-{_hi_a:.1f}" if _hi_a < 99 else f">{_lo_a:.1f}"
                        print(f"  {_label:>12} {len(_ab):>6} {_ach:>11.1%} {_amfe:>9.1f}t {_adir:>9.1%}")

            # ── 6. CI quality by regime volatility ──
            print(f"\n  CI QUALITY BY REGIME VOLATILITY:")
            _regime_vols = []
            for _i, _c in enumerate(_ci_results):
                _rm = _regime_meta_map.get(_regime_at_bar[_i], None)
                if _rm:
                    _regime_vols.append((_rm['volatility'], _c))

            if _regime_vols:
                _vols = np.array([v for v, _ in _regime_vols])
                _vol_q = np.percentile(_vols, [25, 50, 75])
                _bounds = [(0, _vol_q[0], 'Low vol'), (_vol_q[0], _vol_q[1], 'Med-low'),
                           (_vol_q[1], _vol_q[2], 'Med-high'), (_vol_q[2], 999, 'High vol')]

                print(f"  {'Vol Regime':<12} {'N':>6} {'50%CI Cov':>11} {'80%CI Cov':>11} {'Dir Acc':>9} {'Avg IQR':>9}")
                print(f"  {'-'*12} {'-'*6} {'-'*11} {'-'*11} {'-'*9} {'-'*9}")
                for _vlo, _vhi, _vlbl in _bounds:
                    _vgrp = [c for v, c in _regime_vols if _vlo <= v < _vhi]
                    if len(_vgrp) >= 10:
                        _c50 = sum(1 for c in _vgrp if c['p25'] <= c['actual_signed_mfe'] <= c['p75']) / len(_vgrp)
                        _c80 = sum(1 for c in _vgrp if c['p10'] <= c['actual_signed_mfe'] <= c['p90']) / len(_vgrp)
                        _da = sum(1 for c in _vgrp if c['predicted_dir'] == c['actual_dir']) / len(_vgrp)
                        _iqr = np.mean([c['ci_width'] for c in _vgrp])
                        print(f"  {_vlbl:<12} {len(_vgrp):>6} {_c50:>10.1%} {_c80:>10.1%} {_da:>9.1%} {_iqr:>8.1f}t")

            # ── Plot ──
            fig_u, axes_u = plt.subplots(2, 2, figsize=(16, 12))
            fig_u.suptitle('Analysis U: Expected Move Confidence Interval', fontsize=14, fontweight='bold')

            # Plot 1: Actual vs P50 scatter
            ax1 = axes_u[0, 0]
            _p50s = np.array([c['p50'] for c in _ci_results])
            _actuals = np.array([c['actual_signed_mfe'] for c in _ci_results])
            ax1.scatter(_p50s, _actuals, alpha=0.15, s=8, c='steelblue')
            _lim = max(abs(_p50s.max()), abs(_actuals.max()), abs(_p50s.min()), abs(_actuals.min()))
            ax1.plot([-_lim, _lim], [-_lim, _lim], 'r--', alpha=0.5, label='Perfect')
            ax1.set_xlabel('Predicted (P50 signed MFE)')
            ax1.set_ylabel('Actual signed MFE')
            ax1.set_title('P50 vs Actual')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Plot 2: CI width vs prediction error
            ax2 = axes_u[0, 1]
            _errors = np.abs(_actuals - _p50s)
            _iqrs = np.array([c['ci_width'] for c in _ci_results])
            ax2.scatter(_iqrs, _errors, alpha=0.15, s=8, c='coral')
            ax2.set_xlabel('CI Width (IQR)')
            ax2.set_ylabel('|Prediction Error|')
            ax2.set_title('CI Width vs Error (wider CI = less certain)')
            ax2.grid(True, alpha=0.3)

            # Plot 3: Direction accuracy by consensus bin
            ax3 = axes_u[1, 0]
            _cons_bins = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
            _cons_accs = []
            _cons_labels = []
            _cons_ns = []
            for _lo, _hi in _cons_bins:
                _grp = [c for c in _ci_results if _lo <= c['nbr_consensus'] < _hi]
                if len(_grp) >= 5:
                    _acc = sum(1 for c in _grp if c['predicted_dir'] == c['actual_dir']) / len(_grp)
                    _cons_accs.append(_acc * 100)
                    _cons_labels.append(f'{_lo:.0%}-{_hi:.0%}')
                    _cons_ns.append(len(_grp))
            if _cons_accs:
                _bars = ax3.bar(range(len(_cons_accs)), _cons_accs, color='teal', alpha=0.7)
                ax3.set_xticks(range(len(_cons_labels)))
                ax3.set_xticklabels(_cons_labels, fontsize=9)
                ax3.set_ylabel('Direction Accuracy %')
                ax3.set_xlabel('Neighbor Consensus')
                ax3.set_title('Direction Accuracy by Consensus')
                ax3.axhline(50, color='red', linestyle='--', alpha=0.5, label='Coin flip')
                ax3.legend()
                ax3.grid(True, alpha=0.3, axis='y')
                for _b, _n in zip(_bars, _cons_ns):
                    ax3.text(_b.get_x() + _b.get_width()/2, _b.get_height() + 1,
                            f'n={_n}', ha='center', va='bottom', fontsize=8)

            # Plot 4: Ambition ratio histogram
            ax4 = axes_u[1, 1]
            if _ambition:
                _ratios = np.array([a['ratio'] for a in _ambition])
                _ratios_clipped = np.clip(_ratios, 0, 3)
                ax4.hist(_ratios_clipped, bins=30, color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=0.5)
                ax4.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='TP = actual MFE')
                ax4.set_xlabel('Ambition Ratio (|P50| / Actual MFE)')
                ax4.set_ylabel('Count')
                ax4.set_title('Ambition Ratio Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3, axis='y')

            fig_u.tight_layout()
            _plot_path_u = os.path.join(PLOTS_DIR, 'analysis_u_confidence_interval.png')
            fig_u.savefig(_plot_path_u, dpi=150)
            plt.close(fig_u)
            print(f"\n  [saved] {_plot_path_u}")

            # ── Conclusion ──
            print(f"\n  ANALYSIS U CONCLUSION:")
            _ci50_cov = sum(1 for c in _ci_results if c['p25'] <= c['actual_signed_mfe'] <= c['p75']) / _n_u
            _ci80_cov = sum(1 for c in _ci_results if c['p10'] <= c['actual_signed_mfe'] <= c['p90']) / _n_u
            if _ci50_cov >= 0.40 and _ci50_cov <= 0.60:
                print(f"  WELL-CALIBRATED: 50% CI covers {_ci50_cov:.0%} of actuals.")
            elif _ci50_cov < 0.40:
                print(f"  OVER-CONFIDENT: 50% CI only covers {_ci50_cov:.0%} — intervals too narrow.")
            else:
                print(f"  CONSERVATIVE: 50% CI covers {_ci50_cov:.0%} — intervals too wide.")

            if _dir_acc_u > 0.55:
                print(f"  DIRECTION USEFUL: {_dir_acc_u:.1%} from P50 sign ({_dir_acc_u - _baseline_u:+.1%} lift).")
            else:
                print(f"  DIRECTION MARGINAL: {_dir_acc_u:.1%} from P50 sign ({_dir_acc_u - _baseline_u:+.1%} lift).")

            _med_iqr = np.median(_iqrs)
            print(f"  Median CI width (IQR): {_med_iqr:.1f} ticks")
            print(f"  Practical use: set TP = |P50|, SL = P25 (for LONG) or P75 (for SHORT).")
            print(f"  Hook into pipeline as observation layer — log predictions, measure accuracy.")
            print(f"  Promote to decision layer only after proving coverage > 40% on OOS.")

        except Exception as _e_u:
            print(f"  [ERROR] Analysis U: {_e_u}")
            import traceback; traceback.print_exc()

    else:
        print(f"  [SKIP] Analysis U (--start {_start_at})")


    # --- 20. Stacked gate analysis ---
    print(f"\n{'='*70}")
    print(f"  STACKED GATE ANALYSIS: Compound Filters")
    print(f"  Each gate stacks on previous — progressive noise removal")
    print(f"{'='*70}")

    keep_tids = set()
    for s in keep_segs:
        keep_tids.add((s['tid'], s['dir']))

    keep_mask = np.zeros(len(mfes), dtype=bool)
    for i, m in enumerate(meta):
        d = 'LONG' if long_mask[i] else 'SHORT'
        if (m['tid'], d) in keep_tids:
            keep_mask[i] = True

    if n_valid > 50:
        europe_mask = np.array([s == 'EUROPE' for s in sessions]) & valid_ts

        session_open_mask = np.zeros(len(mfes), dtype=bool)
        for i, dt in enumerate(dts):
            if dt is None:
                continue
            h, mn = dt.hour, dt.minute
            if h == 14 and mn < 30:
                session_open_mask[i] = True
            elif h == 8 and mn < 30:
                session_open_mask[i] = True

        good_hours = {0, 5, 17, 18, 19, 20}
        good_hour_mask = np.array([h in good_hours for h in hours_utc]) & valid_ts

        pid_d7_vals = padded[:, 7, pid_idx]
        pid_contrarian = ((pid_d7_vals > 0) & short_mask) | ((pid_d7_vals < 0) & long_mask)

        good_dow = {1, 3}
        good_dow_mask = np.array([d in good_dow for d in dow]) & valid_ts

        # Progressive stacking
        gates = []
        gates.append(('ALL patterns', np.ones(len(mfes), dtype=bool)))
        gates.append(('+ KEEP segments', keep_mask))

        g2 = keep_mask & long_mask
        gates.append(('+ LONG direction', g2))

        g3 = g2 & ~session_open_mask
        gates.append(('+ Skip session opens', g3))

        g4 = g3 & ~europe_mask
        gates.append(('+ Skip Europe session', g4))

        g5 = g4 & good_hour_mask
        gates.append(('+ Best hours (17-20,0,5)', g5))

        g6 = g5 & pid_contrarian
        gates.append(('+ PID contrarian', g6))

        g7 = g5 & good_dow_mask
        gates.append(('+ Best DOW (TUE,THU)', g7))

        g8 = g5 & good_dow_mask & pid_contrarian
        gates.append(('FULL STACK (all gates)', g8))

        print(f"\n  {'Gate':<32} {'N':>5} {'%vol':>6} {'WR':>7} {'MFE':>7} "
              f"{'MAE':>6} {'$/trade':>8} {'Lift':>7}")
        print(f"  {'-'*32} {'-'*5} {'-'*6} {'-'*7} {'-'*7} "
              f"{'-'*6} {'-'*8} {'-'*7}")

        base_wr = float((mfes > maes).mean())
        total_patterns = len(mfes)

        for gate_label, gmask in gates:
            n_g = gmask.sum()
            if n_g < 5:
                print(f"  {gate_label:<32} {n_g:>5} {'<5':>6} {'n/a':>7}")
                continue
            wr_g = float((mfes[gmask] > maes[gmask]).mean())
            mfe_g = float(np.mean(mfes[gmask]))
            mae_g = float(np.mean(maes[gmask]))
            vol_pct = n_g / total_patterns
            avg_pnl = mfe_g - mae_g
            lift = wr_g - base_wr
            print(f"  {gate_label:<32} {n_g:>5} {vol_pct:>5.1%} {wr_g:>7.1%} "
                  f"{mfe_g:>+7.0f} {mae_g:>6.0f} {avg_pnl:>+8.0f} {lift:>+6.1%}")

        # Daily throughput
        t_min_ts = ts_arr[valid_ts].min()
        t_max_ts = ts_arr[valid_ts].max()
        days_span = max((t_max_ts - t_min_ts) / 86400, 1)

        print(f"\n  DAILY THROUGHPUT (over {days_span:.0f} calendar days):")
        for gate_label, gmask in gates:
            n_g = gmask.sum()
            if n_g < 5:
                continue
            per_day = n_g / days_span
            wr_g = float((mfes[gmask] > maes[gmask]).mean())
            print(f"  {gate_label:<32} {per_day:>6.1f}/day  WR={wr_g:.1%}")

        # Ride-the-wave summary
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

            # MES contract scaling
            tick_val = 1.25
            net_per_trade = (mfe_best - mae_best) * tick_val
            daily_pnl_1 = net_per_trade * per_day
            print(f"\n  MES CONTRACT SCALING:")
            print(f"    1 contract:  ${net_per_trade:+.2f}/trade, "
                  f"${daily_pnl_1:+.0f}/day")
            for contracts in [2, 5, 10]:
                print(f"    {contracts} contracts: ${net_per_trade*contracts:+.2f}/trade, "
                      f"${daily_pnl_1*contracts:+.0f}/day")

            # SPLIT segments with temporal gates
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

                for sp_label, gmask in split_gates:
                    n_g = gmask.sum()
                    if n_g < 5:
                        print(f"  {sp_label:<32} {n_g:>5}  (too few)")
                        continue
                    wr_g = float((mfes[gmask] > maes[gmask]).mean())
                    mfe_g = float(np.mean(mfes[gmask]))
                    mae_g = float(np.mean(maes[gmask]))
                    net_ticks = mfe_g - mae_g
                    per_day_g = n_g / max(days_span, 1)
                    daily_1mes = net_ticks * tick_val * per_day_g
                    print(f"  {sp_label:<32} {n_g:>5} {wr_g:>7.1%} {mfe_g:>+7.0f} "
                          f"{mae_g:>6.0f} {net_ticks:>+6.0f} ${daily_1mes:>+7.0f}")

                # Revenue model
                sp_best = sp3
                n_sp = sp_best.sum()

                k_net = (mfe_best - mae_best)
                k_per_day = n_best / max(days_span, 1)

                if n_sp >= 5:
                    sp_wr = float((mfes[sp_best] > maes[sp_best]).mean())
                    sp_mfe = float(np.mean(mfes[sp_best]))
                    sp_mae = float(np.mean(maes[sp_best]))
                    sp_net = sp_mfe - sp_mae
                    sp_per_day = n_sp / max(days_span, 1)
                else:
                    sp_wr, sp_net, sp_per_day = 0, 0, 0

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
                if u_net > 0 and u_per_day > 0:
                    print(f"\n  OOS DEGRADATION SCENARIOS (1 MES contract):")
                    print(f"  IS baseline: {u_per_day:.1f} trades/day, "
                          f"+{u_net:.0f} ticks/trade, ${u_daily_1:,.0f}/day")
                    print(f"\n  {'Scenario':<25} {'net/t':>6} {'$/trade':>8} "
                          f"{'$/day':>8} {'$/month':>9} {'$800?':>6}")
                    print(f"  {'-'*25} {'-'*6} {'-'*8} "
                          f"{'-'*8} {'-'*9} {'-'*6}")

                    for pct, decay_label in [(0, 'IS (no decay)'),
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
                        print(f"  {decay_label:<25} {d_net:>+6.0f} ${d_trade:>7,.0f} "
                              f"${d_daily:>7,.0f} ${d_monthly:>8,.0f} {hits:>6}")

                    # Breakeven
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

                    # Contract scaling
                    print(f"\n  CONTRACT SCALING (at IS rates, ${u_daily_1:,.0f}/day/MES):")
                    margin_1 = 1320
                    for cts in [1, 2, 3, 5]:
                        d_val = u_daily_1 * cts
                        m_val = margin_1 * cts
                        print(f"    {cts} MES: ${d_val:>8,.0f}/day, "
                              f"${d_val*20:>9,.0f}/month  (margin: ${m_val:>6,.0f})")

    else:
        print(f"  (Skipped — insufficient valid timestamps)")

    # --- Save report ---
    sys.stdout = _orig_stdout

    report_path = os.path.join(os.path.dirname(__file__), 'standalone_report.txt')
    header = f"STANDALONE RESEARCH REPORT (FULL 16D)\n"
    header += f"Data: {args.data}, Base TF: {args.base_tf}\n"
    header += f"Context: {args.context_days}d, Analysis: {args.analysis_days}d\n"
    header += f"Regimes: {len(regime_meta)}, Data points: {len(mfes)}\n"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.write(_report_buf.getvalue())

    print(f"\n  Full report saved: {report_path}")
    print(f"  Gates saved: {_gate_path}")
    print(f"  Charts: {PLOTS_DIR}/")
    print(f"  Done.")


if __name__ == '__main__':
    main()
