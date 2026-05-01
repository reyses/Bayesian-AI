"""Research standalone plotting functions.

Extracted from tools/standalone_research.py. Contains:
  - plot_price_imr: Price I-MR chart with regime coloring (4 panels)
  - plot_regime_summary: 2x2 regime dashboard
  - plot_imr_charts: Minitab-style I-MR charts for key features + heatmaps
  - plot_segmented_imr: 15m-anchored I-MR with price, fission segments

PLOTS_DIR is a module-level default that the orchestrator can override at runtime.
"""

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .data import TF_HIERARCHY, TF_LABELS, FEATURE_NAMES

# Regime color palette (up to 20 distinct regimes)
_REGIME_COLORS = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0',
    '#00BCD4', '#E91E63', '#8BC34A', '#FF5722', '#3F51B5',
    '#CDDC39', '#795548', '#607D8B', '#009688', '#FFC107',
    '#673AB7', '#03A9F4', '#FFEB3B', '#FF6F00', '#1B5E20',
]

# Default output directory — orchestrator can override at runtime
PLOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'standalone')


def resolve_plots_dir(data_path, analysis_days=0, base_tf=None):
    """Set PLOTS_DIR subfolder based on data path + analysis window + base TF.
    ATLAS_1DAY -> 1d, ATLAS_1WEEK -> 1w, ATLAS+30d -> 1m, ATLAS full -> 1y.
    If base_tf is provided, suffix as <window>_<base_tf> (e.g., 1y_5m) so
    runs at different base TFs don't overwrite each other.
    Returns the subfolder name."""
    global PLOTS_DIR
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'plots', 'standalone')
    dp = data_path.upper().replace('\\', '/')
    if '1DAY' in dp or '1_DAY' in dp or (0 < analysis_days <= 1):
        sub = '1d'
    elif '1WEEK' in dp or '1_WEEK' in dp or (1 < analysis_days <= 7):
        sub = '1w'
    elif 'OOS' in dp:
        sub = '1y'
    elif 0 < analysis_days <= 60:
        sub = '1m'
    else:
        sub = '1y'
    if base_tf:
        sub = f"{sub}_{base_tf}"
    PLOTS_DIR = os.path.join(base, sub)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    return sub


def plot_price_imr(price_imr, regime_ids, regime_meta, base_df):
    """Plot the foundational price I-MR chart with regime coloring.

    4 panels: Price, I chart, MR chart, Regime map.
    Saves to tools/plots/standalone/0_price_imr.png
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    close = price_imr['close']
    mr = price_imr['mr']
    timestamps = price_imr['timestamps']
    center = price_imr['center']
    ucl_i = price_imr['ucl_i']
    lcl_i = price_imr['lcl_i']
    mr_bar = price_imr['mr_bar']
    ucl_mr = price_imr['ucl_mr']
    warmup_end = price_imr['warmup_end_idx']

    n = len(close)
    x = np.arange(n)

    # Convert timestamps to readable dates for x-axis
    from datetime import datetime, timezone
    date_labels = []
    date_positions = []
    prev_day = None
    for i, ts in enumerate(timestamps):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day = dt.strftime('%m/%d')
        if day != prev_day:
            date_labels.append(day)
            date_positions.append(i)
            prev_day = day

    fig, axes = plt.subplots(4, 1, figsize=(20, 16), sharex=True,
                              gridspec_kw={'height_ratios': [3, 2, 2, 1]})
    fig.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')

    n_regimes = len(regime_meta)

    # --- Panel 1: Price colored by regime ---
    ax = axes[0]
    # Draw warmup in gray
    if warmup_end > 1:
        ax.plot(x[:warmup_end], close[:warmup_end], color='#BBBBBB',
                linewidth=0.8, alpha=0.6)

    # Draw each regime segment with reference line at mean price
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.plot(x[s:e+1], close[s:e+1], color=color, linewidth=1.2,
                label=f"R{rm['regime_id']} ({rm['direction']})")
        # Regime mean price reference line
        ax.hlines(y=rm['mean_price'], xmin=s, xmax=e, color=color,
                  linestyle='--', linewidth=0.8, alpha=0.5)

    # Regime boundary vertical lines (on all panels)
    for rm in regime_meta[1:]:
        for a in axes:
            a.axvline(x=rm['start_idx'], color='#888888', linestyle=':',
                      linewidth=0.7, alpha=0.5)

    # Warmup boundary (on all panels)
    for a in axes:
        a.axvline(x=warmup_end, color='#333333', linestyle='--',
                  linewidth=1, alpha=0.6)
    ax.plot([], [], color='#333333', linestyle='--', label='Warmup end')

    ax.set_title('Price (15m Close) — Colored by Regime (dashed = mean price)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Price', fontsize=10)
    ax.legend(fontsize=7, loc='upper left', ncol=min(n_regimes + 1, 6))
    ax.grid(True, alpha=0.15)

    # --- Panel 2: I chart (close with control limits + data points) ---
    ax = axes[1]
    ax.plot(x, close, color='#333333', linewidth=0.6, alpha=0.5)
    # Individual data point for every bar
    inside = (close <= ucl_i) & (close >= lcl_i)
    ax.scatter(x[inside], close[inside], color='#333333', s=6,
               zorder=4, alpha=0.6, label='Data point')
    # Highlight points outside control limits in red
    outside = ~inside
    if outside.any():
        ax.scatter(x[outside], close[outside], color='#F44336', s=12,
                   zorder=5, label='Outside limits')

    ax.axhline(y=center, color='#888888', linestyle='-', linewidth=1.5,
               label=f'Center={center:.1f}')
    ax.axhline(y=ucl_i, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'UCL={ucl_i:.1f}')
    ax.axhline(y=lcl_i, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'LCL={lcl_i:.1f}')

    ax.set_title('I Chart — Individual Close Prices (each dot = 1 bar)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Close', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.15)

    # --- Panel 3: MR chart (signed) ---
    ax = axes[2]
    mr_abs = price_imr['mr_abs']

    # Color bars by sign: green=up, red=down
    colors_mr = np.where(mr >= 0, '#4CAF50', '#F44336')
    ax.bar(x, mr, color=colors_mr, width=1.0, alpha=0.7, edgecolor='none')

    # UCL / LCL (symmetric for signed MR)
    ax.axhline(y=0, color='#333333', linestyle='-', linewidth=0.8)
    ax.axhline(y=ucl_mr, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'+UCL={ucl_mr:.2f}')
    ax.axhline(y=-ucl_mr, color='#AA0000', linestyle='--', linewidth=1,
               alpha=0.7, label=f'-UCL={-ucl_mr:.2f}')
    ax.axhline(y=mr_bar, color='#888888', linestyle=':', linewidth=1,
               alpha=0.5, label=f'MR_bar={mr_bar:.2f}')
    ax.axhline(y=-mr_bar, color='#888888', linestyle=':', linewidth=1,
               alpha=0.5)

    # Mark UCL breaks
    mr_break = mr_abs > ucl_mr
    if mr_break.any():
        ax.scatter(x[mr_break], mr[mr_break], color='black', s=15,
                   zorder=5, marker='x', linewidths=1, label='UCL break')

    ax.set_title('MR Chart — Signed Moving Range (green=up, red=down)',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Price Change', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.grid(True, alpha=0.15)

    # --- Panel 4: Regime map ---
    ax = axes[3]
    for rm in regime_meta:
        s, e = rm['start_idx'], rm['end_idx']
        color = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.barh(0, e - s + 1, left=s, height=0.8, color=color, alpha=0.8,
                edgecolor='white', linewidth=0.5)
        # Label in center
        mid = (s + e) / 2
        ax.text(mid, 0, f"R{rm['regime_id']}\n{rm['direction']}\n{rm['n_bars']}b",
                ha='center', va='center', fontsize=7, fontweight='bold',
                color='white')

    ax.set_yticks([])
    ax.set_title('Regime Map — Natural price segments from MR UCL breaks',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Bar index', fontsize=10)

    # Date tick labels
    if date_positions:
        step = max(1, len(date_positions) // 15)
        ax.set_xticks([date_positions[i] for i in range(0, len(date_positions), step)])
        ax.set_xticklabels([date_labels[i] for i in range(0, len(date_labels), step)],
                           rotation=45, ha='right', fontsize=8)

    fig.suptitle(f'PRICE I-MR CHART — {n} bars, {n_regimes} regimes\n'
                 f'Warmup: {warmup_end} bars | UCL_MR={ucl_mr:.2f} '
                 f'(breaks trigger new regime)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '0_price_imr.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_regime_summary(regime_meta, mfes, maes, bar_indices, regime_ids):
    """2x2 regime dashboard.

    Saves to tools/plots/standalone/0b_regime_summary.png
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.set_facecolor('white')
    for row in axes:
        for ax in row:
            ax.set_facecolor('white')

    n_regimes = len(regime_meta)
    regime_labels = [f"R{rm['regime_id']}" for rm in regime_meta]
    colors = [_REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
              for rm in regime_meta]

    # Map bar_indices to their regime IDs
    bar_regime = regime_ids[bar_indices]

    # --- Top-left: MFE distribution by regime (box plot) ---
    ax = axes[0, 0]
    mfe_by_regime = []
    labels_used = []
    for rm in regime_meta:
        mask = bar_regime == rm['regime_id']
        if mask.any():
            mfe_by_regime.append(mfes[mask])
            labels_used.append(f"R{rm['regime_id']}")
    if mfe_by_regime:
        bp = ax.boxplot(mfe_by_regime, labels=labels_used, patch_artist=True)
        for patch, rm in zip(bp['boxes'], regime_meta):
            patch.set_facecolor(_REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)])
            patch.set_alpha(0.6)
    ax.set_title('MFE Distribution by Regime', fontsize=11, fontweight='bold')
    ax.set_ylabel('MFE (points)', fontsize=9)
    ax.grid(True, alpha=0.15)

    # --- Top-right: Regime volatility vs mean MFE (scatter) ---
    ax = axes[0, 1]
    for rm in regime_meta:
        mask = bar_regime == rm['regime_id']
        mean_mfe = float(mfes[mask].mean()) if mask.any() else 0
        c = _REGIME_COLORS[rm['regime_id'] % len(_REGIME_COLORS)]
        ax.scatter(rm['volatility'], mean_mfe, color=c, s=rm['n_bars'] * 2,
                   edgecolors='black', linewidth=0.5, zorder=5)
        ax.annotate(f"R{rm['regime_id']}", (rm['volatility'], mean_mfe),
                    fontsize=8, ha='left', va='bottom')
    ax.set_title('Regime Volatility vs Mean MFE', fontsize=11, fontweight='bold')
    ax.set_xlabel('Volatility (mean MR)', fontsize=9)
    ax.set_ylabel('Mean MFE', fontsize=9)
    ax.grid(True, alpha=0.15)

    # --- Bottom-left: Regime duration histogram ---
    ax = axes[1, 0]
    durations = [rm['n_bars'] for rm in regime_meta]
    ax.bar(range(n_regimes), durations, color=colors, edgecolor='white',
           linewidth=0.5)
    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels(regime_labels, fontsize=8)
    ax.set_title('Regime Duration (bars)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of bars', fontsize=9)
    ax.grid(True, alpha=0.15)

    # --- Bottom-right: Win rate by regime ---
    ax = axes[1, 1]
    win_rates = []
    for rm in regime_meta:
        mask = bar_regime == rm['regime_id']
        if mask.any():
            wins = (mfes[mask] > maes[mask]).sum()
            wr = float(wins) / float(mask.sum()) * 100
        else:
            wr = 0.0
        win_rates.append(wr)
    ax.bar(range(n_regimes), win_rates, color=colors, edgecolor='white',
           linewidth=0.5)
    ax.axhline(y=50, color='#888888', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xticks(range(n_regimes))
    ax.set_xticklabels(regime_labels, fontsize=8)
    ax.set_title('Win Rate by Regime (MFE > MAE)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Win Rate %', fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.15)

    fig.suptitle(f'REGIME SUMMARY — {n_regimes} regimes, '
                 f'{len(mfes)} analysis bars',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, '0b_regime_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_imr_charts(padded, mfes):
    """Generate Minitab-style I-MR charts for key features.

    Saves to tools/plots/standalone/:
      1_imr_key_features.png  — 6 key features, I + MR panel each
      2_i_heatmap.png         — full 12x16 I-chart heatmap
      3_mr_heatmap.png        — full 11x16 MR heatmap
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
        (3,  'entropy_normalized',  'Wave coherence'),
        (7,  'self_adx',   'ADX (trend strength)'),
        (8,  'self_hurst', 'Hurst exponent'),
    ]

    # -- PLOT 1: Key features I-MR panels --
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

        # -- I chart --
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

        # -- MR chart --
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

    # -- PLOT 2: Full I-chart heatmap (12 TF x 16 features) --
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

    # -- PLOT 3: MR heatmap (11 transitions x 16 features) --
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

    # -- PLOT 4: Correlation with MFE heatmap (I + MR side by side) --
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

    # -- PLOT 5: Segmented I-MR on 15m anchor --
    # White background. Line color = fission class only.
    # Green (#2E7D32) = KEEP/trade, Yellow (#F9A825) = SPLIT/mixed, Red (#C62828) = DROP/no trade
    from matplotlib.collections import LineCollection

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
    ratios = [3, 3, 2, 2, 1] if has_price else [3, 2, 2, 1]
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, 22 if has_price else 16),
                              sharex=True, gridspec_kw={'height_ratios': ratios})
    fig.patch.set_facecolor('white')

    x = np.arange(n)
    seg_colors = [fission_colors[c] for c in fission_class[:-1]]

    def _add_legend(ax):
        for cls in [0, 1, 2]:
            n_cls = (fission_class == cls).sum()
            wr_cls = float((mfes[fission_class == cls] > maes[fission_class == cls]).mean()) \
                if n_cls > 0 else 0
            ax.plot([], [], color=fission_colors[cls], linewidth=3,
                    label=f'{fission_labels[cls]} (n={n_cls}, WR={wr_cls:.0%})')

    panel_idx = 0

    # Panel 0: PRICE through time
    if has_price:
        ax0 = axes[panel_idx]
        ax0.set_facecolor('white')
        panel_idx += 1

        points_p = np.column_stack([x, prices]).reshape(-1, 1, 2)
        segments_p = np.concatenate([points_p[:-1], points_p[1:]], axis=1)
        lc_p = LineCollection(segments_p, colors=seg_colors, linewidths=1.5)
        ax0.add_collection(lc_p)
        ax0.set_xlim(0, n - 1)
        p_range = prices.max() - prices.min()
        ax0.set_ylim(prices.min() - p_range * 0.05, prices.max() + p_range * 0.05)

        _add_legend(ax0)
        ax0.legend(fontsize=8, loc='upper right', ncol=3)
        ax0.set_ylabel('Price (15m close)', fontsize=10)
        ax0.set_title('PRICE — Green=trade, Yellow=mixed, Red=no trade',
                      fontsize=12, fontweight='bold')
        ax0.grid(True, alpha=0.2)

    # Panel 1: I-chart (z_score)
    ax1 = axes[panel_idx]
    ax1.set_facecolor('white')
    panel_idx += 1

    points = np.column_stack([x, z_vals]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=seg_colors, linewidths=1.5)
    ax1.add_collection(lc)
    ax1.set_xlim(0, n - 1)
    ax1.set_ylim(z_vals.min() - 0.5, z_vals.max() + 0.5)

    center = float(np.mean(z_vals))
    std_z = float(np.std(z_vals))
    ax1.axhline(y=center, color='#888888', linewidth=1, linestyle='-', alpha=0.5, label=f'Center={center:.3f}')
    ax1.axhline(y=center + 3 * std_z, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4, label='UCL/LCL')
    ax1.axhline(y=center - 3 * std_z, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4)
    ax1.axhline(y=0, color='black', linewidth=0.5, linestyle=':', alpha=0.3)

    _add_legend(ax1)
    ax1.legend(fontsize=7, loc='upper right', ncol=3)
    ax1.set_ylabel('z_score (15m)', fontsize=10)
    ax1.set_title('I-CHART: 15m z_score', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.2)

    # Panel 2: MR chart
    ax2 = axes[panel_idx]
    ax2.set_facecolor('white')
    panel_idx += 1

    seg_colors_mr = [fission_colors[c] for c in fission_class[1:-1]]
    x_mr = np.arange(len(z_mr))
    points_mr = np.column_stack([x_mr, z_mr]).reshape(-1, 1, 2)
    segments_mr = np.concatenate([points_mr[:-1], points_mr[1:]], axis=1)
    lc_mr = LineCollection(segments_mr, colors=seg_colors_mr, linewidths=1.2)
    ax2.add_collection(lc_mr)
    ax2.set_xlim(0, len(z_mr) - 1)
    ax2.set_ylim(0, min(z_mr.max() * 1.2, ucl * 2))
    ax2.axhline(y=mr_bar, color='#888888', linewidth=1, alpha=0.5, label=f'MR_bar={mr_bar:.4f}')
    ax2.axhline(y=ucl, color='#888888', linewidth=0.8, linestyle='--', alpha=0.4, label=f'UCL={ucl:.4f}')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylabel('|MR| z_score', fontsize=10)
    ax2.set_title('MR CHART: Bar-to-bar jump magnitude', fontsize=10, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    # Panel 3: MFE/MAE
    ax3 = axes[panel_idx]
    ax3.set_facecolor('white')
    panel_idx += 1

    points_mfe = np.column_stack([x, mfes]).reshape(-1, 1, 2)
    seg_mfe = np.concatenate([points_mfe[:-1], points_mfe[1:]], axis=1)
    lc_mfe = LineCollection(seg_mfe, colors=seg_colors, linewidths=1.2)
    ax3.add_collection(lc_mfe)

    points_mae = np.column_stack([x, -maes]).reshape(-1, 1, 2)
    seg_mae = np.concatenate([points_mae[:-1], points_mae[1:]], axis=1)
    lc_mae = LineCollection(seg_mae, colors=seg_colors, linewidths=0.8, alpha=0.5)
    ax3.add_collection(lc_mae)
    ax3.set_xlim(0, n - 1)
    ax3.set_ylim(-maes.max() * 1.1, mfes.max() * 1.1)
    ax3.axhline(y=0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
    ax3.axhline(y=float(np.mean(mfes)), color='#2196F3', linewidth=0.8, linestyle='--',
                alpha=0.5, label=f'Mean MFE={np.mean(mfes):.0f}')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_ylabel('MFE / -MAE (ticks)', fontsize=10)
    ax3.set_title('ORACLE OUTCOME: MFE (solid) and -MAE (faded)', fontsize=10, fontweight='bold')
    ax3.grid(True, alpha=0.2)

    # Panel 4: Fission class strip
    ax4 = axes[panel_idx]
    ax4.set_facecolor('white')
    for i in range(n):
        ax4.axvspan(i - 0.5, i + 0.5, color=fission_colors[fission_class[i]], alpha=0.8)
    ax4.set_yticks([])
    ax4.set_ylabel('Class', fontsize=10)
    ax4.set_xlabel('Analysis bar index (15m)', fontsize=10)

    # X-axis timestamps
    n_ticks = min(20, n)
    tick_positions = np.linspace(0, n - 1, n_ticks, dtype=int)
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels([dt_labels[i].strftime('%m/%d %H:%M') for i in tick_positions],
                        rotation=45, ha='right', fontsize=7)

    fig.suptitle(f'SEGMENTED I-MR CHART — 15m Anchor ({n} data points)\n'
                 f'Green=KEEP (trade), Yellow=SPLIT (mixed), Red=DROP (no trade)',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, '5_segmented_imr_15m.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 5_segmented_imr_15m.png")

    # -- PLOT 6: Fractal heatmap -- mean feature value per (TF depth, feature), split by fission --
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
