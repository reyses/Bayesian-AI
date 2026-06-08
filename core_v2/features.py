"""
Feature names + loader for v2 (139D).
=====================================

Source of truth for:
  - FEATURE_NAMES     : ordered list of all 139 feature column names
  - N_FEATURES        : 139
  - TF_ORDER          : ['15s', '1m', '5m', '15m', '1h', '1D']
  - TF_SECONDS        : period in seconds per TF
  - N_BASE            : default window per TF (imported from SFE; Principle 2)
  - LAYER_FAMILIES    : structured metadata for per-layer-family parquet storage

Also provides `load_features(days, tfs, layers, root)` which joins the
decoupled per-layer-family parquets into a single training-ready DataFrame.

Storage layout (see research/feature_spec_v2.md):
    DATA/ATLAS/FEATURES_5s_v2/
    |-- L0/YYYY_MM_DD.parquet          (1 col: L0_time_of_day)
    |-- L1_15s/YYYY_MM_DD.parquet      (6 cols)
    |-- L1_1m/ ...
    |-- L2_15s/ ...
    |-- L3_15s/ ...

V1 name compatibility: module exposes FEATURE_NAMES, N_FEATURES, TF_ORDER,
TF_SECONDS as v1 did, so consumers that only read these symbol names can
swap `from core_v2` -> `from core_v2` with no other change (though the CONTENT
of these symbols is v2 and the CNN input shape changes from 91 to 139).

Supersedes: core/features.py (v1, 91D).
"""
from __future__ import annotations

import os
import glob
from typing import Iterable

import numpy as np
import pandas as pd

# Re-export the window defaults from the SFE (single source of truth)
from core_v2.statistical_field_engine import N_BASE, N_HURST_MULT, SWING_NOISE_WINDOW


# ─── TF order + periods ────────────────────────────────────────────────────

TF_ORDER = ['5s', '15s', '1m', '5m', '15m', '1h', '4h', '1D']
N_TFS = len(TF_ORDER)

TF_SECONDS = {
    '5s':   5,
    '15s':  15,
    '1m':   60,
    '5m':   300,
    '15m':  900,
    '1h':   3600,
    '4h':   14400,
    '1D':   86400,
}

# The hierarchical channel order for the CNN (from slowest to fastest).
# MUST be used when assembling the grid to prevent timeframe scrambling.
TF_HIERARCHY_V2 = ['1D', '4h', '1h', '15m', '5m', '1m', '15s', '5s']

# The ordered features per TF (25 total). This defines the feature axis in the CNN.
FEATURE_NAMES_V2 = [
    # L1 (8) — bar primitives
    'price_velocity_1b', 'price_accel_1b',
    'vol_velocity_1b',   'vol_accel_1b',
    'bar_range', 'body', 'upper_wick', 'lower_wick',
    # L2 (9) — rolling-window stats
    'price_velocity_w', 'price_accel_w',
    'vol_velocity_w',   'vol_accel_w',
    'price_mean_w', 'price_sigma_w',
    'vol_mean_w',   'vol_sigma_w',
    'vwap_w',
    # L3 (8) — approved exceptions
    'z_se',  'z_high', 'z_low',
    'SE_high', 'SE_low',
    'hurst', 'reversion_prob', 'swing_noise',
]

N_TFS_V2 = len(TF_HIERARCHY_V2)               # 8
N_FEATURES_PER_TF_V2 = len(FEATURE_NAMES_V2)  # 25
N_FLAT_FEATURES_V2 = N_TFS_V2 * N_FEATURES_PER_TF_V2  # 200

# ─── Feature name generators ───────────────────────────────────────────────

def _l0_names() -> list[str]:
    return ['L0_time_of_day']


def _l1_names(tf: str) -> list[str]:
    return [
        f'L1_{tf}_price_velocity_1b',
        f'L1_{tf}_price_accel_1b',
        f'L1_{tf}_vol_velocity_1b',
        f'L1_{tf}_vol_accel_1b',
        f'L1_{tf}_bar_range',
        f'L1_{tf}_body',
        f'L1_{tf}_upper_wick',
        f'L1_{tf}_lower_wick',
    ]


def _l2_names(tf: str, N: int | None = None) -> list[str]:
    if N is None:
        N = N_BASE[tf]
    return [
        f'L2_{tf}_price_velocity_{N}',
        f'L2_{tf}_price_accel_{N}',
        f'L2_{tf}_vol_velocity_{N}',
        f'L2_{tf}_vol_accel_{N}',
        f'L2_{tf}_price_mean_{N}',
        f'L2_{tf}_price_sigma_{N}',
        f'L2_{tf}_vol_mean_{N}',
        f'L2_{tf}_vol_sigma_{N}',
        f'L2_{tf}_vwap_{N}',
    ]


def _l3_names(tf: str, N: int | None = None) -> list[str]:
    if N is None:
        N = N_BASE[tf]
    return [
        f'L3_{tf}_z_se_{N}',
        f'L3_{tf}_z_high_{N}',
        f'L3_{tf}_z_low_{N}',
        f'L3_{tf}_SE_high_{N}',
        f'L3_{tf}_SE_low_{N}',
        f'L3_{tf}_hurst_{N}',
        f'L3_{tf}_reversion_prob_{N}',
        f'L3_{tf}_swing_noise_{N}',
    ]


# ─── Canonical feature name list ───────────────────────────────────────────
# Order (so columns in FEATURES_5s_v2 parquets are deterministic):
#   L0 global (1)
#   For each TF: L1 (6), L2 (9), L3 (8)

FEATURE_NAMES: list[str] = []
FEATURE_NAMES.extend(_l0_names())
for tf in TF_ORDER:
    FEATURE_NAMES.extend(_l1_names(tf))
    FEATURE_NAMES.extend(_l2_names(tf))
    FEATURE_NAMES.extend(_l3_names(tf))

# Expected: 1 (L0) + (8 + 9 + 8) * len(TF_ORDER) = 1 + 25 * N_TFS
# With TF_ORDER = ['5s','15s','1m','5m','15m','1h','4h','1D']  (N_TFS=8): 201
N_FEATURES = len(FEATURE_NAMES)
_expected = 1 + 25 * N_TFS
assert N_FEATURES == _expected, (
    f"Expected {_expected} features for {N_TFS} TFs, got {N_FEATURES}. "
    f"Check the _l0/_l1/_l2/_l3_names generators."
)


# ─── Layer-family metadata (for builder + loader) ──────────────────────────

# LAYER_FAMILIES[family_name] = (is_per_tf, tf_or_none, feature_name_list)
# Family name is also the subdirectory under FEATURES_5s_v2/.
LAYER_FAMILIES: dict[str, dict] = {}

# L0 is global (not per-TF)
LAYER_FAMILIES['L0'] = {
    'is_per_tf': False,
    'tf': None,
    'features': _l0_names(),
    'schema_version': 1,
}

for _tf in TF_ORDER:
    LAYER_FAMILIES[f'L1_{_tf}'] = {
        'is_per_tf': True,
        'tf': _tf,
        'features': _l1_names(_tf),
        'schema_version': 1,
    }
    LAYER_FAMILIES[f'L2_{_tf}'] = {
        'is_per_tf': True,
        'tf': _tf,
        'features': _l2_names(_tf),
        'schema_version': 1,
    }
    LAYER_FAMILIES[f'L3_{_tf}'] = {
        'is_per_tf': True,
        'tf': _tf,
        'features': _l3_names(_tf),
        'schema_version': 1,
    }


# ─── Layer-family parquet loader ───────────────────────────────────────────

DEFAULT_FEATURES_ROOT = 'DATA/ATLAS/FEATURES_5s_v2'


def layer_family_path(root: str, family: str, day: str) -> str:
    """Return the parquet path for a layer-family on a given day.

    Args:
        root: features root dir (default DATA/ATLAS/FEATURES_5s_v2).
        family: family name (e.g. 'L0', 'L1_1m', 'L3_1h').
        day: day label 'YYYY_MM_DD'.
    """
    return os.path.join(root, family, f'{day}.parquet')


def load_features(
    days: Iterable[str],
    tfs: Iterable[str] | None = None,
    layers: Iterable[str] | None = None,
    root: str = DEFAULT_FEATURES_ROOT,
    require_all: bool = True,
) -> pd.DataFrame:
    """Join per-layer-family parquets into a single training-ready DataFrame.

    Args:
        days: iterable of day labels 'YYYY_MM_DD'.
        tfs: iterable of TF labels to load. Default: all of TF_ORDER.
        layers: iterable of layer prefixes ('L0', 'L1', 'L2', 'L3'). Default: all.
        root: features root dir.
        require_all: if True, raise on missing files. If False, skip them silently.

    Returns:
        DataFrame with timestamp as index (monotonic) and the requested
        feature columns. NaN filled naturally during warmup (no forward-fill).
    """
    if tfs is None:
        tfs = list(TF_ORDER)
    if layers is None:
        layers = ['L0', 'L1', 'L2', 'L3']

    layers = list(layers)
    tfs = list(tfs)

    # Determine which families we need
    families_to_load: list[str] = []
    if 'L0' in layers:
        families_to_load.append('L0')
    for layer in layers:
        if layer == 'L0':
            continue
        for tf in tfs:
            families_to_load.append(f'{layer}_{tf}')

    per_day_frames: list[pd.DataFrame] = []
    for day in days:
        frames: list[pd.DataFrame] = []
        day_ts: pd.Series | None = None

        for family in families_to_load:
            path = layer_family_path(root, family, day)
            if not os.path.exists(path):
                if require_all:
                    raise FileNotFoundError(
                        f"Missing layer-family parquet: {path}. "
                        f"Re-run build_dataset_v2.py or set require_all=False."
                    )
                continue

            fdf = pd.read_parquet(path)
            if 'timestamp' not in fdf.columns:
                raise ValueError(
                    f"{path}: expected 'timestamp' column, got {list(fdf.columns)}"
                )

            # Anchor timestamp from first family we see
            if day_ts is None:
                day_ts = fdf['timestamp']
            elif not day_ts.equals(fdf['timestamp']):
                # Timestamp mismatch between families on the same day - bug
                raise ValueError(
                    f"Timestamp mismatch between {path} and earlier family on "
                    f"day {day}. All per-day layer-family parquets must share "
                    f"identical timestamps (anchor = 5s bars)."
                )

            frames.append(fdf.drop(columns=['timestamp']))

        if frames and day_ts is not None:
            day_df = pd.concat(frames, axis=1)
            day_df.insert(0, 'timestamp', day_ts.values)
            per_day_frames.append(day_df)

    if not per_day_frames:
        return pd.DataFrame()

    result = pd.concat(per_day_frames, axis=0, ignore_index=True)
    result = result.sort_values('timestamp').reset_index(drop=True)
    return result


# ─── Helpers for consumers ─────────────────────────────────────────────────

def get_feature_index(name: str) -> int:
    """Return the column index of a named feature in the canonical order."""
    try:
        return FEATURE_NAMES.index(name)
    except ValueError:
        raise ValueError(
            f"Unknown v2 feature: {name!r}. Use one of {len(FEATURE_NAMES)} "
            f"names from FEATURE_NAMES."
        )


def describe_feature_count() -> str:
    """Human-readable summary of the 139D layout, useful for logging."""
    l0 = len([n for n in FEATURE_NAMES if n.startswith('L0_')])
    l1 = len([n for n in FEATURE_NAMES if n.startswith('L1_')])
    l2 = len([n for n in FEATURE_NAMES if n.startswith('L2_')])
    l3 = len([n for n in FEATURE_NAMES if n.startswith('L3_')])
    return (
        f"v2 feature vector: {N_FEATURES} total\n"
        f"  L0 (global)       : {l0}\n"
        f"  L1 ({N_TFS} TFs x 8) : {l1}\n"
        f"  L2 ({N_TFS} TFs x 9) : {l2}\n"
        f"  L3 ({N_TFS} TFs x 8) : {l3}\n"
        f"  TF order          : {TF_ORDER}\n"
        f"  N_BASE windows    : {N_BASE}\n"
    )

# ─── Train/Serve Parity Anti-Scramble Assembler ────────────────────────────

def assemble_v2_grid(flat_v2_matrix: np.ndarray) -> np.ndarray:
    """Safely construct the (N, 8, 25) CNN grid from the flat (N, 201) _v2_matrix.
    
    CRITICAL: This implements the Anti-Scramble guarantee.
    The CNN channels MUST be ordered according to TF_HIERARCHY_V2 (descending) 
    and the feature axis MUST be ordered according to FEATURE_NAMES_V2.
    It builds the mapping via explicit name-keyed placement.
    
    Args:
        flat_v2_matrix: The (N, 201) array in FEATURE_NAMES order (ascending).
    
    Returns:
        grid: (N, 8, 25) array. Channel 0 == '1D'.
    """
    assert TF_HIERARCHY_V2[0] == '1D', "TF_HIERARCHY_V2 must be descending!"
    
    N = flat_v2_matrix.shape[0]
    grid = np.zeros((N, N_TFS_V2, N_FEATURES_PER_TF_V2), dtype=np.float32)
    
    # Pre-compute mapping from (tf_idx, feat_idx) -> flat index in FEATURE_NAMES
    # This provenance map guarantees we never misplace a feature.
    for tf_idx, tf in enumerate(TF_HIERARCHY_V2):
        for feat_idx, fname in enumerate(FEATURE_NAMES_V2):
            # Construct the canonical literal name as generated by _l1, _l2, _l3_names
            if fname.endswith('_1b') or fname in ('bar_range', 'body', 'upper_wick', 'lower_wick'):
                col = f'L1_{tf}_{fname}'
            elif fname.endswith('_w'):
                # L2 windowed features (strip _w, add N_BASE)
                col = f'L2_{tf}_{fname[:-2]}_{N_BASE[tf]}'
            else:
                # L3 features (add N_BASE)
                col = f'L3_{tf}_{fname}_{N_BASE[tf]}'
            
            try:
                flat_idx = FEATURE_NAMES.index(col)
                grid[:, tf_idx, feat_idx] = flat_v2_matrix[:, flat_idx]
            except ValueError:
                raise ValueError(f"CRITICAL PARITY ERROR: expected column '{col}' missing from flat FEATURE_NAMES.")
    
    return grid
