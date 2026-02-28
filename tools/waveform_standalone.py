"""
Standalone Waveform Screening Tool
====================================
Runs the I-MR factor screening DIRECTLY from raw ATLAS parquet data.
No templates.pkl, no checkpoints, no training dependency.

Genesis methodology:
  1. I-MR chart on 15m close → z_score per bar
  2. Make it fractal — one I-MR chart per timeframe
  3. Stack into (12 TF × 16D) hypervolume matrix per analysis point
  4. Screen factors against oracle MFE → rank by |correlation|
  5. Stepwise regression → adj-R² curve

Usage:
    python tools/waveform_standalone.py --data DATA/ATLAS_1DAY --base-tf 15m
    python tools/waveform_standalone.py --months 2025_01 --base-tf 15m

Output: tools/standalone_report.txt + console summary
"""

import sys, os, io, glob, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.quantum_field_engine import QuantumFieldEngine
from config.oracle_config import ORACLE_LOOKAHEAD_BARS


# -- TF hierarchy: 12 levels from macro (1W) to micro (15s) -------------------
# Drops 1s and 5s (too noisy, too much data for screening)
TF_HIERARCHY = ['1W', '1D', '4h', '1h', '30m', '15m', '5m', '3m', '2m', '1m', '30s', '15s']

# Seconds per bar for each TF (used for tf_scale feature = log2(seconds))
TF_SECONDS = {
    '1W': 604800, '1D': 86400, '4h': 14400, '1h': 3600,
    '30m': 1800, '15m': 900, '5m': 300, '3m': 180, '2m': 120,
    '1m': 60, '30s': 30, '15s': 15,
}

# TF labels for column naming (matches waveform_screening.py convention)
TF_LABELS = [
    'd0_1W', 'd1_1D', 'd2_4h', 'd3_1h', 'd4_30m', 'd5_15m',
    'd6_5m', 'd7_3m', 'd8_2m', 'd9_1m', 'd10_30s', 'd11_15s'
]

# 16D feature names
FEATURE_NAMES = [
    'z_score', 'log1p_vol', 'log1p_mom', 'coherence', 'tf_scale', 'depth',
    'parent_ctx', 'self_adx', 'self_hurst', 'self_dmi_diff',
    'parent_z', 'parent_dmi_diff', 'root_is_roche', 'tf_alignment',
    'self_pid', 'osc_coh'
]


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
#  DATA PIPELINE: Raw ATLAS → Physics → 16D → Hypervolume Matrices
# =============================================================================

def load_atlas_tf(data_dir, tf_name, months=None):
    """Load ATLAS parquet files for a single timeframe.

    Args:
        data_dir: Root ATLAS directory (e.g., 'DATA/ATLAS' or 'DATA/ATLAS_1DAY')
        tf_name: Timeframe string (e.g., '15m', '1h')
        months: Optional list of month strings (e.g., ['2025_01']). If None, load all.

    Returns:
        pd.DataFrame with [timestamp, open, high, low, close, volume], sorted by timestamp.
        Returns empty DataFrame if TF directory doesn't exist.
    """
    tf_dir = os.path.join(data_dir, tf_name)
    if not os.path.isdir(tf_dir):
        return pd.DataFrame()

    if months:
        files = [os.path.join(tf_dir, f'{m}.parquet') for m in months]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))

    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Ensure timestamp is numeric (seconds)
    if 'timestamp' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = df['timestamp'].astype('int64') // 10**9
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def compute_tf_physics(tf_name, df):
    """Run QuantumFieldEngine on a single TF's data.

    Returns:
        dict mapping timestamp (int) → ThreeBodyQuantumState
    """
    if df.empty or len(df) < 21:
        return {}

    engine = QuantumFieldEngine(regression_period=21)
    results = engine.batch_compute_states(df, use_cuda=False)

    states = {}
    for r in results:
        state = r['state']
        ts = int(state.timestamp) if hasattr(state, 'timestamp') and state.timestamp else 0
        if ts > 0:
            states[ts] = state

    return states


def extract_16d(state, tf_name):
    """Build 16D feature vector from a ThreeBodyQuantumState.

    Matches the feature layout in fractal_clustering.py:extract_features()
    but without ancestry (features 5-6, 10-13 set to 0).
    """
    z = state.z_score
    v = abs(state.particle_velocity) if hasattr(state, 'particle_velocity') else 0.0
    m = abs(state.momentum_strength) if hasattr(state, 'momentum_strength') else 0.0
    c = state.coherence if hasattr(state, 'coherence') else 0.0

    tf_scale = math.log2(max(TF_SECONDS.get(tf_name, 60), 1))

    adx = (state.adx_strength * 0.01) if hasattr(state, 'adx_strength') else 0.0
    hurst = state.hurst_exponent if hasattr(state, 'hurst_exponent') else 0.5
    dmi_diff = ((state.dmi_plus - state.dmi_minus) * 0.01) if hasattr(state, 'dmi_plus') else 0.0
    pid = state.term_pid if hasattr(state, 'term_pid') else 0.0
    osc = state.oscillation_coherence if hasattr(state, 'oscillation_coherence') else 0.0

    return [
        z,                          # [0]  signed z-score
        math.log1p(v),              # [1]  log1p(velocity)
        math.log1p(m),              # [2]  log1p(momentum)
        c,                          # [3]  coherence
        tf_scale,                   # [4]  tf_scale = log2(seconds)
        0.0,                        # [5]  depth (no ancestry)
        0.0,                        # [6]  parent_ctx (no ancestry)
        adx,                        # [7]  self_adx / 100
        hurst,                      # [8]  self_hurst
        dmi_diff,                   # [9]  self_dmi_diff / 100
        0.0,                        # [10] parent_z (no ancestry)
        0.0,                        # [11] parent_dmi_diff (no ancestry)
        0.0,                        # [12] root_is_roche (no ancestry)
        0.0,                        # [13] tf_alignment (no ancestry)
        pid,                        # [14] self_pid
        osc,                        # [15] osc_coh
    ]


def build_stacked_matrices(all_tf_states, base_tf, base_df,
                           context_days=21, analysis_days=7):
    """Build (12, 16) hypervolume matrices from stacked TF physics.

    For each bar in the base TF's analysis window:
      - Find the most recent state at each of the 12 TFs
      - Stack into (12, 16) matrix
      - Compute oracle MFE/MAE from the base TF's future bars

    Args:
        all_tf_states: dict {tf_name: {timestamp: ThreeBodyQuantumState}}
        base_tf: Base timeframe string (e.g., '15m')
        base_df: DataFrame for the base TF (for MFE/MAE lookahead)
        context_days: Days of warmup before analysis window
        analysis_days: Days of analysis window (0 = use all remaining)

    Returns:
        matrices: list of (12, 16) numpy arrays
        mfes: numpy array of MFE values
        maes: numpy array of MAE values
        meta: list of dicts with timestamp, dmi_diff, etc.
    """
    if base_tf not in all_tf_states or not all_tf_states[base_tf]:
        print("ERROR: No states computed for base TF")
        return [], np.array([]), np.array([]), []

    # Get sorted timestamps for the base TF
    base_states = all_tf_states[base_tf]
    base_timestamps = sorted(base_states.keys())

    if not base_timestamps:
        return [], np.array([]), np.array([]), []

    t_min = base_timestamps[0]
    t_max = base_timestamps[-1]

    # Define analysis window
    from datetime import datetime, timezone
    data_span_days = (t_max - t_min) / 86400

    # Auto-adjust if data is shorter than context window
    if context_days > 0 and data_span_days < context_days + 1:
        # Use first half for warmup, rest for analysis
        old_ctx = context_days
        context_days = max(0, int(data_span_days * 0.3))
        print(f"  Auto-adjusted context: {old_ctx}d → {context_days}d "
              f"(data span is only {data_span_days:.1f}d)")

    t_warmup_end = t_min + context_days * 86400
    if analysis_days > 0:
        t_analysis_end = t_warmup_end + analysis_days * 86400
    else:
        t_analysis_end = t_max + 1

    print(f"  Data range:     {datetime.fromtimestamp(t_min, tz=timezone.utc):%Y-%m-%d %H:%M} to "
          f"{datetime.fromtimestamp(t_max, tz=timezone.utc):%Y-%m-%d %H:%M}")
    print(f"  Data span:      {data_span_days:.1f} days")
    print(f"  Warmup:         {context_days}d → analysis starts "
          f"{datetime.fromtimestamp(t_warmup_end, tz=timezone.utc):%Y-%m-%d %H:%M}")
    print(f"  Analysis end:   {datetime.fromtimestamp(min(t_analysis_end, t_max), tz=timezone.utc):%Y-%m-%d %H:%M}")

    # Filter base timestamps to analysis window
    analysis_ts = [t for t in base_timestamps if t_warmup_end <= t < t_analysis_end]
    print(f"  Analysis bars:  {len(analysis_ts)} ({base_tf} bars in window)")

    if not analysis_ts:
        print("  WARNING: No bars in analysis window. Try reducing context_days.")
        return [], np.array([]), np.array([]), []

    # Pre-sort timestamps for each TF (for binary search alignment)
    tf_sorted_ts = {}
    for tf in TF_HIERARCHY:
        if tf in all_tf_states and all_tf_states[tf]:
            tf_sorted_ts[tf] = sorted(all_tf_states[tf].keys())

    # Build timestamp→index mapping for base_df (for MFE/MAE computation)
    if 'timestamp' in base_df.columns:
        ts_col = base_df['timestamp'].values
    else:
        ts_col = np.arange(len(base_df))

    ts_to_idx = {}
    for i, t in enumerate(ts_col):
        ts_to_idx[int(t)] = i

    # Oracle lookahead for base TF
    lookahead = ORACLE_LOOKAHEAD_BARS.get(base_tf, 16)

    matrices = []
    mfes = []
    maes = []
    meta = []

    for t in tqdm(analysis_ts, desc="Building hypervolumes", unit="bar"):
        # --- Stack 16D across all 12 TFs ---
        mat = np.zeros((12, 16))
        has_data = 0

        for depth_idx, tf in enumerate(TF_HIERARCHY):
            if tf not in tf_sorted_ts:
                continue  # TF not available, leave as zeros

            # Find most recent state at or before timestamp t
            tf_ts_list = tf_sorted_ts[tf]
            # Binary search for the largest timestamp <= t
            idx = np.searchsorted(tf_ts_list, t, side='right') - 1
            if idx < 0:
                continue  # No data before this timestamp

            nearest_ts = tf_ts_list[idx]
            state = all_tf_states[tf][nearest_ts]
            mat[depth_idx, :] = extract_16d(state, tf)
            has_data += 1

        if has_data < 3:
            continue  # Need at least 3 TFs with data

        # --- Compute oracle MFE/MAE from base TF future bars ---
        if t not in ts_to_idx:
            continue

        bar_idx = ts_to_idx[t]
        if bar_idx + lookahead >= len(base_df):
            continue  # Not enough future data

        entry_price = float(base_df.iloc[bar_idx]['close'])
        future = base_df.iloc[bar_idx + 1 : bar_idx + 1 + lookahead]

        if future.empty:
            continue

        max_up = float(future['high'].max() - entry_price)
        max_down = float(entry_price - future['low'].min())

        if max_up == 0 and max_down == 0:
            continue

        # Direction from z-score sign at base TF
        base_state = base_states[t]
        z = base_state.z_score
        dmi_diff = (base_state.dmi_plus - base_state.dmi_minus) \
            if hasattr(base_state, 'dmi_plus') else 0.0

        # MFE/MAE assignment based on direction
        # z < 0 → LONG setup (MFE = up, MAE = down)
        # z > 0 → SHORT setup (MFE = down, MAE = up)
        if z < 0:  # LONG
            mfe_val = max_up
            mae_val = max_down
        else:      # SHORT
            mfe_val = max_down
            mae_val = max_up

        matrices.append(mat)
        mfes.append(mfe_val)
        maes.append(mae_val)

        # ADX quartile for segmentation (proxy for template_id)
        adx = base_state.adx_strength if hasattr(base_state, 'adx_strength') else 0.0
        adx_bin = int(min(adx // 25, 3))  # 0-3 quartiles

        meta.append({
            'tid': f'adx_q{adx_bin}',
            'idx': len(matrices) - 1,
            'depth': 11,  # always full depth in standalone
            'ts': t,
            'dmi_diff': dmi_diff,
            'z_score': z,
        })

    print(f"  Built {len(matrices)} hypervolume matrices with oracle labels")
    return matrices, np.array(mfes), np.array(maes), meta


# =============================================================================
#  I-MR CHART PLOTS (matplotlib)
# =============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PLOTS_DIR = os.path.join(os.path.dirname(__file__), 'plots', 'standalone')


def plot_imr_charts(padded, mfes):
    """Generate Minitab-style I-MR charts for key features.

    Saves to tools/plots/standalone/:
      1_imr_key_features.png  — 6 key features, I + MR panel each
      2_i_heatmap.png         — full 12×16 I-chart heatmap
      3_mr_heatmap.png        — full 11×16 MR heatmap
      4_imr_correlation.png   — r(MFE) heatmap for I and MR
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n, n_depths, n_feat = padded.shape
    D4 = 3.267

    mr = np.diff(padded, axis=1)       # (n, 11, 16)
    mr_abs = np.abs(mr)

    tf_labels_short = [TF_HIERARCHY[d] for d in range(n_depths)]
    trans_labels = [f"{TF_HIERARCHY[d]}>{TF_HIERARCHY[d+1]}" for d in range(n_depths - 1)]

    key_feats = [
        (0,  'z_score',    'Signed z (fair value distance)'),
        (9,  'dmi_diff',   'DMI diff (directional bias)'),
        (14, 'self_pid',   'PID control force'),
        (3,  'coherence',  'Wave coherence'),
        (7,  'self_adx',   'ADX (trend strength)'),
        (8,  'self_hurst', 'Hurst exponent'),
    ]

    # ── PLOT 1: Key features I-MR panels ──
    fig = plt.figure(figsize=(20, 24))
    outer = gridspec.GridSpec(6, 1, hspace=0.35, figure=fig)

    for row, (f_idx, f_name, f_desc) in enumerate(key_feats):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[row],
                                                 wspace=0.25)
        ax_i = fig.add_subplot(inner[0])
        ax_mr = fig.add_subplot(inner[1])

        i_vals = padded[:, :, f_idx]
        mr_vals_f = mr[:, :, f_idx]
        mr_abs_f = mr_abs[:, :, f_idx]

        i_mean = i_vals.mean(axis=0)
        i_std = i_vals.std(axis=0)
        center = float(i_mean.mean())

        mr_abs_mean = mr_abs_f.mean(axis=0)
        mr_bar = float(mr_abs_f.mean())
        ucl_val = D4 * mr_bar

        # Correlation with MFE at each depth
        corr_i = np.zeros(n_depths)
        for d in range(n_depths):
            col = i_vals[:, d]
            if np.std(col) > 1e-12:
                c = np.corrcoef(col, mfes)[0, 1]
                corr_i[d] = c if not np.isnan(c) else 0.0

        # ── I chart ──
        x = np.arange(n_depths)
        colors_i = ['#F44336' if abs(c) > 0.15 else '#2196F3' for c in corr_i]

        ax_i.bar(x, i_mean, color=colors_i, alpha=0.7, edgecolor='white', linewidth=0.5)
        ax_i.errorbar(x, i_mean, yerr=i_std, fmt='none', ecolor='gray',
                       capsize=3, linewidth=1)
        ax_i.axhline(y=center, color='green', linestyle='-', linewidth=1.5,
                      label=f'Center={center:.3f}')
        ax_i.axhline(y=center + 3 * float(i_std.mean()), color='red',
                      linestyle='--', linewidth=1, alpha=0.7, label='UCL/LCL')
        ax_i.axhline(y=center - 3 * float(i_std.mean()), color='red',
                      linestyle='--', linewidth=1, alpha=0.7)
        ax_i.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.5)

        ax_i.set_xticks(x)
        ax_i.set_xticklabels(tf_labels_short, rotation=45, ha='right', fontsize=8)
        ax_i.set_title(f'I Chart: {f_name}\n{f_desc}', fontsize=10, fontweight='bold')
        ax_i.set_ylabel('Mean value', fontsize=9)
        ax_i.legend(fontsize=7, loc='best')
        ax_i.grid(True, alpha=0.2)

        # Annotate r(MFE) on bars
        for d in range(n_depths):
            if abs(corr_i[d]) > 0.05:
                ax_i.text(d, i_mean[d], f'r={corr_i[d]:+.2f}', ha='center',
                         va='bottom' if i_mean[d] >= 0 else 'top',
                         fontsize=6, color='#333')

        # ── MR chart ──
        x_mr = np.arange(n_depths - 1)
        breaks_pct = (mr_abs_f > ucl_val).mean(axis=0) * 100
        colors_mr = ['#FF5722' if bp > 5 else '#4CAF50' for bp in breaks_pct]

        ax_mr.bar(x_mr, mr_abs_mean, color=colors_mr, alpha=0.7,
                  edgecolor='white', linewidth=0.5)
        ax_mr.axhline(y=mr_bar, color='green', linestyle='-', linewidth=1.5,
                      label=f'MR_bar={mr_bar:.4f}')
        ax_mr.axhline(y=ucl_val, color='red', linestyle='--', linewidth=1.5,
                      label=f'UCL={ucl_val:.4f}')

        ax_mr.set_xticks(x_mr)
        ax_mr.set_xticklabels(trans_labels, rotation=45, ha='right', fontsize=7)
        ax_mr.set_title(f'MR Chart: {f_name}\nUCL breaks shown in red', fontsize=10,
                        fontweight='bold')
        ax_mr.set_ylabel('Mean |MR|', fontsize=9)
        ax_mr.legend(fontsize=7, loc='best')
        ax_mr.grid(True, alpha=0.2)

        # Annotate break %
        for d in range(n_depths - 1):
            if breaks_pct[d] > 1:
                ax_mr.text(d, mr_abs_mean[d], f'{breaks_pct[d]:.0f}%',
                          ha='center', va='bottom', fontsize=6, color='#333')

    fig.suptitle(f'FRACTAL I-MR CHART — {n} data points × {n_depths} TF depths\n'
                 f'I = feature value at each TF | MR = TF-to-TF transition\n'
                 f'Red I bars = |r(MFE)| > 0.15 | Red MR bars = >5% UCL breaks',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(PLOTS_DIR, '1_imr_key_features.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 1_imr_key_features.png")

    # ── PLOT 2: Full I-chart heatmap (12 TF × 16 features) ──
    i_matrix = padded.mean(axis=0)  # (12, 16)

    fig, ax = plt.subplots(figsize=(16, 8))
    vmax = max(abs(i_matrix.min()), abs(i_matrix.max()), 0.5)
    im = ax.imshow(i_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_depths))
    ax.set_yticklabels([f'{TF_LABELS[d]} ({TF_HIERARCHY[d]})' for d in range(n_depths)],
                        fontsize=9)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('TF Depth (macro → micro)', fontsize=12)
    ax.set_title(f'I-CHART HEATMAP: Mean feature value at each TF depth\n'
                 f'{n} data points | Red = positive, Blue = negative',
                 fontsize=13, fontweight='bold')

    for d in range(n_depths):
        for f in range(n_feat):
            val = i_matrix[d, f]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean value', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '2_i_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 2_i_heatmap.png")

    # ── PLOT 3: MR heatmap (11 transitions × 16 features) ──
    mr_matrix = mr_abs.mean(axis=0)  # (11, 16)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(mr_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_depths - 1))
    ax.set_yticklabels(trans_labels, fontsize=8)
    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('TF Transition', fontsize=12)
    ax.set_title(f'MR HEATMAP: Mean |MR| at each TF transition\n'
                 f'Higher = bigger regime change between adjacent TFs',
                 fontsize=13, fontweight='bold')

    for d in range(n_depths - 1):
        for f in range(n_feat):
            val = mr_matrix[d, f]
            if val > 0.01:
                color = 'white' if val > mr_matrix.max() * 0.5 else 'black'
                ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean |MR|', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '3_mr_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 3_mr_heatmap.png")

    # ── PLOT 4: Correlation with MFE heatmap (I + MR side by side) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # I correlations
    corr_i_matrix = np.zeros((n_depths, n_feat))
    for d in range(n_depths):
        for f in range(n_feat):
            col = padded[:, d, f]
            if np.std(col) > 1e-12:
                c = np.corrcoef(col, mfes)[0, 1]
                corr_i_matrix[d, f] = c if not np.isnan(c) else 0.0

    im1 = ax1.imshow(corr_i_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax1.set_xticks(range(n_feat))
    ax1.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
    ax1.set_yticks(range(n_depths))
    ax1.set_yticklabels([f'{TF_HIERARCHY[d]}' for d in range(n_depths)], fontsize=9)
    ax1.set_title('I values: r(MFE)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('TF Depth', fontsize=10)

    for d in range(n_depths):
        for f in range(n_feat):
            val = corr_i_matrix[d, f]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > 0.15 else 'black'
                ax1.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # MR correlations
    corr_mr_matrix = np.zeros((n_depths - 1, n_feat))
    for d in range(n_depths - 1):
        for f in range(n_feat):
            col = mr[:, d, f]
            if np.std(col) > 1e-12:
                c = np.corrcoef(col, mfes)[0, 1]
                corr_mr_matrix[d, f] = c if not np.isnan(c) else 0.0

    im2 = ax2.imshow(corr_mr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.3, vmax=0.3)
    ax2.set_xticks(range(n_feat))
    ax2.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(n_depths - 1))
    ax2.set_yticklabels(trans_labels, fontsize=8)
    ax2.set_title('MR values: r(MFE)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('TF Transition', fontsize=10)

    for d in range(n_depths - 1):
        for f in range(n_feat):
            val = corr_mr_matrix[d, f]
            if abs(val) > 0.05:
                color = 'white' if abs(val) > 0.15 else 'black'
                ax2.text(f, d, f'{val:.2f}', ha='center', va='center',
                        fontsize=6, color=color)
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    fig.suptitle(f'CORRELATION WITH MFE: Which (TF × Feature) dimensions predict outcome?\n'
                 f'Red = higher value → higher MFE | Blue = higher value → lower MFE',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '4_imr_correlation.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 4_imr_correlation.png")


def plot_segmented_imr(padded, mfes, maes, meta, tids, long_mask,
                       keep_segs, split_segs, drop_segs, base_df=None):
    """Plot 15m-anchored I-MR chart with price, z_score, and fission segments.

    Top panel = actual price through time (the thing we're trading).
    Then z_score I-chart, MR jump magnitude, MFE/MAE outcome, fission strip.
    Background bands = fission class. Line color/thickness = jump size.

    Saves to tools/plots/standalone/:
      5_segmented_imr_15m.png  — price + I-chart + MR with fission colors
      6_segmented_heatmap.png  — full fractal heatmap colored by segment
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)
    n = len(mfes)
    base_tf_depth = TF_HIERARCHY.index('15m')  # d5

    # Build fission class per data point
    keep_tids = set((s['tid'], s['dir']) for s in keep_segs)
    split_tids = set((s['tid'], s['dir']) for s in split_segs)

    fission_class = np.full(n, 2, dtype=int)  # 0=KEEP, 1=SPLIT, 2=DROP
    for i in range(n):
        d = 'LONG' if long_mask[i] else 'SHORT'
        tid = tids[i]
        if (tid, d) in keep_tids:
            fission_class[i] = 0
        elif (tid, d) in split_tids:
            fission_class[i] = 1

    fission_colors = {0: '#4CAF50', 1: '#FFC107', 2: '#F44336'}  # green/yellow/red
    fission_labels = {0: 'KEEP', 1: 'SPLIT', 2: 'DROP'}
    point_colors = [fission_colors[c] for c in fission_class]

    # Timestamps for x-axis
    from datetime import datetime, timezone as tz
    timestamps = [m['ts'] for m in meta]
    dt_labels = [datetime.fromtimestamp(t, tz=tz.utc) for t in timestamps]

    # Key features to plot on I-chart
    z_vals = padded[:, base_tf_depth, 0]       # z_score at 15m
    pid_vals = padded[:, base_tf_depth, 14]    # PID at 15m
    adx_vals = padded[:, base_tf_depth, 7]     # ADX at 15m
    dmi_vals = padded[:, base_tf_depth, 9]     # DMI diff at 15m

    # MR (bar-to-bar difference at 15m)
    z_mr = np.abs(np.diff(z_vals))
    D4 = 3.267
    mr_bar = float(z_mr.mean())
    ucl = D4 * mr_bar

    # ── PLOT 5: Segmented I-MR on 15m anchor ──
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    import matplotlib.cm as cm

    # Extract actual close prices aligned to each analysis point
    prices = np.zeros(n)
    if base_df is not None and 'close' in base_df.columns:
        ts_col = base_df['timestamp'].values if 'timestamp' in base_df.columns else np.arange(len(base_df))
        ts_to_idx = {}
        for idx_i, t in enumerate(ts_col):
            ts_to_idx[int(t)] = idx_i
        for i, m in enumerate(meta):
            bar_idx = ts_to_idx.get(int(m['ts']), -1)
            if bar_idx >= 0:
                prices[i] = float(base_df.iloc[bar_idx]['close'])
    has_price = prices.sum() > 0

    n_panels = 5 if has_price else 4
    ratios = [3, 2, 2, 2, 1] if has_price else [3, 2, 2, 1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 20 if has_price else 16),
                              sharex=True, gridspec_kw={'height_ratios': ratios})

    x = np.arange(n)
    panel_idx = 0

    # Panel 0: PRICE through time (the thing we're trading)
    if has_price:
        ax0 = axes[panel_idx]
        panel_idx += 1

        # Background bands
        for i in range(n):
            ax0.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i]], alpha=0.12)

        # Price line colored by jump magnitude
        price_mr = np.abs(np.diff(prices))
        price_mr_norm = Normalize(vmin=0, vmax=max(float(np.percentile(price_mr, 95)), 0.01))
        points_p = np.column_stack([x, prices]).reshape(-1, 1, 2)
        segments_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
        seg_widths_p = 0.8 + 3.0 * price_mr_norm(price_mr)
        lc_p = LineCollection(segments_p, cmap='hot_r', norm=price_mr_norm,
                              linewidths=seg_widths_p, alpha=0.9)
        lc_p.set_array(price_mr)
        ax0.add_collection(lc_p)
        ax0.set_xlim(0, n - 1)
        p_range = prices.max() - prices.min()
        ax0.set_ylim(prices.min() - p_range * 0.05, prices.max() + p_range * 0.05)

        cbar0 = plt.colorbar(lc_p, ax=ax0, shrink=0.6, pad=0.01)
        cbar0.set_label('|Price jump|', fontsize=8)

        for cls in [0, 1, 2]:
            n_cls = (fission_class == cls).sum()
            wr_cls = float((mfes[fission_class == cls] > maes[fission_class == cls]).mean()) \
                if n_cls > 0 else 0
            ax0.axvspan(0, 0, color=fission_colors[cls], alpha=0.3,
                        label=f'{fission_labels[cls]} (n={n_cls}, WR={wr_cls:.0%})')
        ax0.legend(fontsize=8, loc='upper right', ncol=3)
        ax0.set_ylabel('Price (15m close)', fontsize=10)
        ax0.set_title('PRICE: 15m close — thick/dark = big move | Background = Fission class',
                      fontsize=12, fontweight='bold')
        ax0.grid(True, alpha=0.15)

    # Panel 1: I-chart — z_score with jump-colored line, fission background
    ax1 = axes[panel_idx]
    panel_idx += 1

    # Background bands colored by fission class
    for i in range(n):
        ax1.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i]], alpha=0.12)

    # Line colored by jump magnitude — thick where big moves happen
    points = np.column_stack([x, z_vals]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Color by |MR| — hot = big jump (the ones you want to capture)
    mr_norm = Normalize(vmin=0, vmax=max(float(np.percentile(z_mr, 95)), 0.1))
    seg_widths = 0.8 + 3.0 * mr_norm(z_mr)  # thin=quiet, thick=big jump
    lc = LineCollection(segments, cmap='hot_r', norm=mr_norm,
                        linewidths=seg_widths, alpha=0.9)
    lc.set_array(z_mr)
    ax1.add_collection(lc)
    ax1.set_xlim(0, n - 1)
    ax1.set_ylim(z_vals.min() - 0.5, z_vals.max() + 0.5)

    center = float(np.mean(z_vals))
    std_z = float(np.std(z_vals))
    ax1.axhline(y=center, color='#888888', linewidth=1.2, linestyle='-', alpha=0.6, label=f'Center={center:.3f}')
    ax1.axhline(y=center + 3 * std_z, color='#AA0000', linewidth=1, linestyle='--', alpha=0.5, label='UCL/LCL')
    ax1.axhline(y=center - 3 * std_z, color='#AA0000', linewidth=1, linestyle='--', alpha=0.5)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle=':', alpha=0.5)

    # Colorbar for jump magnitude
    cbar1 = plt.colorbar(lc, ax=ax1, shrink=0.6, pad=0.01)
    cbar1.set_label('|Jump| (MR)', fontsize=8)

    # Legend for fission background
    for cls in [0, 1, 2]:
        n_cls = (fission_class == cls).sum()
        wr_cls = float((mfes[fission_class == cls] > maes[fission_class == cls]).mean()) \
            if n_cls > 0 else 0
        ax1.axvspan(0, 0, color=fission_colors[cls], alpha=0.3,
                    label=f'{fission_labels[cls]} (n={n_cls}, WR={wr_cls:.0%})')
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.set_ylabel('z_score (15m)', fontsize=10)
    ax1.set_title('I-CHART: 15m z_score — Line thickness/color = jump size | Background = Fission class',
                  fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.15)

    # Panel 2: MR chart — jump magnitude with fission background
    ax2 = axes[panel_idx]
    panel_idx += 1
    for i in range(len(z_mr)):
        ax2.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i + 1]], alpha=0.12)

    x_mr = np.arange(len(z_mr))
    points_mr = np.column_stack([x_mr, z_mr]).reshape(-1, 1, 2)
    segments_mr = np.concatenate([points_mr[:-1], points_mr[1:]], axis=1)
    lc_mr = LineCollection(segments_mr, cmap='hot_r', norm=mr_norm,
                           linewidths=1.2, alpha=0.9)
    lc_mr.set_array(z_mr[:-1])
    ax2.add_collection(lc_mr)
    ax2.set_xlim(0, len(z_mr) - 1)
    ax2.set_ylim(0, min(z_mr.max() * 1.2, ucl * 2))
    ax2.axhline(y=mr_bar, color='#888888', linewidth=1.2, alpha=0.6, label=f'MR_bar={mr_bar:.4f}')
    ax2.axhline(y=ucl, color='#AA0000', linewidth=1.2, linestyle='--', alpha=0.5, label=f'UCL={ucl:.4f}')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylabel('|MR| z_score', fontsize=10)
    ax2.set_title('MR CHART: Jump magnitude — big spikes = regime transitions worth capturing',
                  fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.15)

    # Panel 3: MFE/MAE with fission background — do big jumps = big MFE?
    ax3 = axes[panel_idx]
    panel_idx += 1
    for i in range(n):
        ax3.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i]], alpha=0.12)

    # MFE line colored by fission
    points_mfe = np.column_stack([x, mfes]).reshape(-1, 1, 2)
    seg_mfe = np.concatenate([points_mfe[:-1], points_mfe[1:]], axis=1)
    seg_colors_mfe = [fission_colors[c] for c in fission_class[:-1]]
    lc_mfe = LineCollection(seg_mfe, colors=seg_colors_mfe, linewidths=1.2, alpha=0.8)
    ax3.add_collection(lc_mfe)
    # -MAE line
    points_mae = np.column_stack([x, -maes]).reshape(-1, 1, 2)
    seg_mae = np.concatenate([points_mae[:-1], points_mae[1:]], axis=1)
    lc_mae = LineCollection(seg_mae, colors=seg_colors_mfe, linewidths=0.8, alpha=0.4)
    ax3.add_collection(lc_mae)
    ax3.set_xlim(0, n - 1)
    ax3.set_ylim(-maes.max() * 1.1, mfes.max() * 1.1)
    ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax3.axhline(y=float(np.mean(mfes)), color='blue', linewidth=1, linestyle='--',
                alpha=0.5, label=f'Mean MFE={np.mean(mfes):.0f}')
    ax3.plot([], [], color='gray', linewidth=1.2, label='MFE (top)')
    ax3.plot([], [], color='gray', linewidth=0.8, alpha=0.4, label='-MAE (bottom)')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_ylabel('MFE / -MAE (ticks)', fontsize=10)
    ax3.set_title('ORACLE OUTCOME: Do the big jumps produce big MFE?', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.15)

    # Panel 4: Fission class strip (categorical heatmap)
    ax4 = axes[panel_idx]
    for i in range(n):
        ax4.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i]], alpha=0.8)
    ax4.set_yticks([])
    ax4.set_ylabel('Class', fontsize=10)
    ax4.set_xlabel('Analysis bar index (15m)', fontsize=10)

    # X-axis: show timestamps at intervals
    n_ticks = min(20, n)
    tick_positions = np.linspace(0, n - 1, n_ticks, dtype=int)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([dt_labels[i].strftime('%m/%d %H:%M') for i in tick_positions],
                        rotation=45, ha='right', fontsize=7)

    fig.suptitle(f'SEGMENTED I-MR CHART — 15m Anchor ({n} data points)\n'
                 f'Green=KEEP (trade), Yellow=SPLIT (refine), Red=DROP (noise)\n'
                 f'Segmentation from ADX quartile x Direction across full fractal state',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '5_segmented_imr_15m.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 5_segmented_imr_15m.png")

    # ── PLOT 6: Fractal heatmap — mean feature value per (TF depth, feature), split by fission ──
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    for ax_idx, (cls, cls_label) in enumerate([(0, 'KEEP'), (1, 'SPLIT'), (2, 'DROP')]):
        ax = axes[ax_idx]
        cls_mask = fission_class == cls
        n_cls = cls_mask.sum()

        if n_cls < 2:
            ax.set_title(f'{cls_label} (n={n_cls})\n(too few)', fontsize=11)
            ax.axis('off')
            continue

        cls_padded = padded[cls_mask]
        cls_mean = cls_padded.mean(axis=0)  # (12, 16)

        # Correlation with MFE for this class
        cls_mfes = mfes[cls_mask]
        cls_maes = maes[cls_mask]
        cls_wr = float((cls_mfes > cls_maes).mean())

        vmax = max(abs(cls_mean.min()), abs(cls_mean.max()), 0.5)
        im = ax.imshow(cls_mean, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

        ax.set_xticks(range(16))
        ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(12))
        ax.set_yticklabels([f'{TF_HIERARCHY[d]}' for d in range(12)], fontsize=8)

        if ax_idx == 0:
            ax.set_ylabel('TF Depth (macro → micro)', fontsize=10)

        ax.set_title(f'{cls_label} (n={n_cls}, WR={cls_wr:.0%})',
                    fontsize=11, fontweight='bold',
                    color=fission_colors[cls])

        # Annotate cells with values
        for d in range(12):
            for f in range(16):
                val = cls_mean[d, f]
                if abs(val) > 0.01:
                    color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(f, d, f'{val:.2f}', ha='center', va='center',
                            fontsize=5, color=color)

        plt.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f'FRACTAL FINGERPRINT BY FISSION CLASS\n'
                 f'Mean (TF × Feature) value — what does each class look like?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '6_segmented_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 6_segmented_heatmap.png")


# =============================================================================
#  ANALYSIS PIPELINE (reused from waveform_screening.py)
# =============================================================================

def pad_to_fixed_depth(matrices, max_depth=12):
    """Pad variable-depth matrices to fixed (max_depth, 16) with zeros."""
    n = len(matrices)
    padded = np.zeros((n, max_depth, 16))
    for i, mat in enumerate(matrices):
        d = min(mat.shape[0], max_depth)
        padded[i, :d, :] = mat[:d, :]
    return padded


def compute_moving_range(padded):
    """Compute I-MR segmentation features from (n, 12, 16) hypervolume.

    Returns: mr_flat (n, 448), mr_col_names
    """
    n, n_depths, n_feat = padded.shape

    # MR: depth-to-depth differences
    mr = np.diff(padded, axis=1)  # (n, 11, 16)

    # UCL per feature column (D4=3.267 for n=2 subgroup)
    D4 = 3.267
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

    # MR values (11 × 16 = 176)
    mr_flat_parts.append(mr.reshape(n, -1))
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"MR_{d_from}>{d_to}__{f_lbl}")

    # UCL flags (11 × 16 = 176)
    mr_flat_parts.append(ucl_flags.reshape(n, -1))
    for d in range(n_depths - 1):
        d_from = TF_LABELS[d] if d < len(TF_LABELS) else f'd{d}'
        d_to = TF_LABELS[d + 1] if (d + 1) < len(TF_LABELS) else f'd{d+1}'
        for f in range(n_feat):
            f_lbl = FEATURE_NAMES[f] if f < len(FEATURE_NAMES) else f'f{f}'
            mr_col_names.append(f"UCL_{d_from}>{d_to}__{f_lbl}")

    # Column summaries (3 × 16 = 48)
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


def regression_r2(flat, col_names, mfes, top_k=20):
    """Stepwise OLS on top-K factors, report adj-R²."""
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

    return steps


def print_screening_report(results, mfes, maes, meta, top_n=30):
    """Print the screening report."""
    print(f"\n{'='*70}")
    print(f"  STANDALONE WAVEFORM SCREENING REPORT")
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


# =============================================================================
#  MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Standalone waveform screening from raw ATLAS data')
    parser.add_argument('--data', default='DATA/ATLAS',
                        help='ATLAS data directory (default: DATA/ATLAS)')
    parser.add_argument('--months', nargs='+', default=None,
                        help='Specific months to load (e.g., 2025_01 2025_02)')
    parser.add_argument('--base-tf', default='15m',
                        help='Base timeframe for analysis points (default: 15m)')
    parser.add_argument('--context-days', type=int, default=21,
                        help='Warmup days before analysis window (default: 21)')
    parser.add_argument('--analysis-days', type=int, default=7,
                        help='Analysis window in days (0 = all remaining, default: 7)')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top factors to display')
    args = parser.parse_args()

    print(f"{'='*70}")
    print(f"  STANDALONE WAVEFORM SCREENING")
    print(f"  Data: {args.data}")
    print(f"  Base TF: {args.base_tf}")
    print(f"  Context: {args.context_days}d warmup, {args.analysis_days}d analysis")
    print(f"{'='*70}")

    # --- 1. Load ATLAS data for all 12 TFs ---
    print(f"\n--- STEP 1: Loading ATLAS data ---")
    all_dfs = {}
    for tf in TF_HIERARCHY:
        df = load_atlas_tf(args.data, tf, months=args.months)
        if not df.empty:
            all_dfs[tf] = df
            print(f"  {tf:>4}: {len(df):>8,} bars")
        else:
            print(f"  {tf:>4}:   (not found)")

    if args.base_tf not in all_dfs:
        print(f"ERROR: Base TF '{args.base_tf}' has no data in {args.data}")
        sys.exit(1)

    # --- 2. Compute physics per TF ---
    print(f"\n--- STEP 2: Computing I-MR physics per timeframe ---")
    all_tf_states = {}
    for tf in tqdm(TF_HIERARCHY, desc="Physics", unit="tf"):
        if tf not in all_dfs:
            continue
        states = compute_tf_physics(tf, all_dfs[tf])
        if states:
            all_tf_states[tf] = states
            print(f"  {tf:>4}: {len(states):>8,} states computed")

    print(f"  Total: {len(all_tf_states)} TFs with physics data")

    # --- 3. Build stacked hypervolume matrices ---
    print(f"\n--- STEP 3: Building fractal I-MR hypervolume matrices ---")
    matrices, mfes, maes, meta = build_stacked_matrices(
        all_tf_states, args.base_tf, all_dfs[args.base_tf],
        context_days=args.context_days,
        analysis_days=args.analysis_days
    )

    if len(matrices) < 20:
        print(f"ERROR: Only {len(matrices)} matrices built (need ≥20)")
        sys.exit(1)

    # --- 4. Pad + I-MR chart plots ---
    print(f"\n--- STEP 4: I-MR Charts ---")
    padded = pad_to_fixed_depth(matrices, max_depth=12)
    plot_imr_charts(padded, mfes)

    # --- 5. Flatten + MR segmentation ---
    print(f"\n--- STEP 5: MR segmentation ---")
    flat_i, col_names_i = flatten_matrices(padded)
    flat_mr, col_names_mr = compute_moving_range(padded)

    flat_z = np.hstack([flat_i, flat_mr])
    col_names_z = col_names_i + col_names_mr
    print(f"  Combined: {len(col_names_i)} I + {len(col_names_mr)} MR "
          f"= {len(col_names_z)} total features")

    # --- 6. Screen all three: I-only, MR-only, combined ---
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

    # --- 6. Summary comparison ---
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

    # --- 7. Directional split ---
    dmi_float = np.array([float(m.get('dmi_diff', 0)) for m in meta])
    long_mask = dmi_float >= 0
    short_mask = ~long_mask

    n_long = long_mask.sum()
    n_short = short_mask.sum()
    wr_long = float((mfes[long_mask] > maes[long_mask]).mean()) if n_long > 0 else 0
    wr_short = float((mfes[short_mask] > maes[short_mask]).mean()) if n_short > 0 else 0

    print(f"\n{'='*70}")
    print(f"  DIRECTIONAL SPLIT")
    print(f"{'='*70}")
    print(f"  LONG  (DMI >= 0): {n_long:>5} points, WR={wr_long:.1%}")
    print(f"  SHORT (DMI <  0): {n_short:>5} points, WR={wr_short:.1%}")
    print(f"  Mixed WR:         {float((mfes > maes).mean()):.1%}")

    # --- 8. Segmented screening ---
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

    # --- 9. Model fission ---
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

    # --- 9b. Segmented I-MR plots ---
    print(f"\n--- Generating segmented I-MR plots ---")
    plot_segmented_imr(padded, mfes, maes, meta, tids, long_mask,
                       keep_segs, split_segs, drop_segs,
                       base_df=all_dfs.get(args.base_tf))

    # --- 10. Export gate config ---
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

    # --- 11. What-if impact ---
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

    # --- 12. PID drill-down ---
    pid_idx = FEATURE_NAMES.index('self_pid')  # 14

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

    # --- 13. Temporal special cause analysis ---
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

    # --- 14. Stacked gate analysis ---
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
    header = f"STANDALONE WAVEFORM SCREENING REPORT\n"
    header += f"Data: {args.data}, Base TF: {args.base_tf}\n"
    header += f"Context: {args.context_days}d, Analysis: {args.analysis_days}d\n"
    header += f"Data points: {len(mfes)}\n"

    with open(report_path, 'w') as f:
        f.write(header)
        f.write(_report_buf.getvalue())

    print(f"\n  Full report saved: {report_path}")
    print(f"  Gates saved: {_gate_path}")
    print(f"  Done.")


if __name__ == '__main__':
    main()
