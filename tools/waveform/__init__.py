"""Waveform analysis subpackage.

Submodules:
    data       — ATLAS loading, 16D physics extraction, stacked matrices
    imr        — Price I-MR computation, regime detection, oracle
    screening  — Pad/flatten matrices, factor screening, R² regression
    seeds      — SeedPrimitiveLibrary (20 normalized primitives)
    plots      — All matplotlib plotting functions + PLOTS_DIR management
"""

from .data import (
    TF_HIERARCHY,
    TF_SECONDS,
    TF_LABELS,
    FEATURE_NAMES,
    load_atlas_tf,
    compute_tf_physics,
    extract_16d,
    build_stacked_matrices,
)

from .imr import (
    compute_price_imr,
    detect_regimes,
    compute_regime_oracle,
)

from .screening import (
    pad_to_fixed_depth,
    compute_moving_range,
    flatten_matrices,
    screen_factors,
    regression_r2,
    print_screening_report,
)

from .seeds import (
    SeedPrimitiveLibrary,
    _detect_inflections,
    _adaptive_split,
)

from .plots import (
    resolve_plots_dir,
    plot_price_imr,
    plot_regime_summary,
    plot_imr_charts,
    plot_segmented_imr,
)

__all__ = [
    # data
    'TF_HIERARCHY', 'TF_SECONDS', 'TF_LABELS', 'FEATURE_NAMES',
    'load_atlas_tf', 'compute_tf_physics', 'extract_16d', 'build_stacked_matrices',
    # imr
    'compute_price_imr', 'detect_regimes', 'compute_regime_oracle',
    # screening
    'pad_to_fixed_depth', 'compute_moving_range', 'flatten_matrices',
    'screen_factors', 'regression_r2', 'print_screening_report',
    # seeds
    'SeedPrimitiveLibrary', '_detect_inflections', '_adaptive_split',
    # plots
    'resolve_plots_dir', 'plot_price_imr', 'plot_regime_summary',
    'plot_imr_charts', 'plot_segmented_imr',
]
