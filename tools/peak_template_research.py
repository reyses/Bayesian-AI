"""
Peak Template Research Tool
============================
Finds all peaks in IS 1m data, extracts the 10-bar approach context as a
time series, labels the outcome (reversal / plateau / continuation), clusters
the results with UMAP + HDBSCAN, and reports which context features separate
the three outcome types.

Usage:
    python tools/peak_template_research.py
    python tools/peak_template_research.py --data DATA/ATLAS
"""

import sys
import io
import os
import argparse
import warnings
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.statistical_field_engine import StatisticalFieldEngine

# ── Optional clustering imports ─────────────────────────────────────────────
try:
    import umap
    import hdbscan
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    print("[WARN] umap-learn and/or hdbscan not installed.")
    print("       Install with: pip install umap-learn hdbscan")
    print("       Clustering will be skipped; CSV + console stats still generated.\n")

from sklearn.preprocessing import StandardScaler

# ── MNQ tick constants ──────────────────────────────────────────────────────
TICK_SIZE = 0.25
TICK_VALUE = 0.50

# ── Peak detection thresholds (mirrors advance_engine._detect_peak_reversal) ─
PC_RISE_THRESHOLD = 1.05        # P_center must exceed prev * 1.05
PC_MIN_PREV = 0.01              # Minimum prev P_center to test rise
FM_DECAY_THRESHOLD = 0.90       # |F_momentum| must drop below prev * 0.90
FM_MIN_PREV = 0.5               # Minimum prev |F_momentum| to test decay
COHERENCE_MIN = 0.55            # oscillation_entropy_normalized > 0.55

# ── Outcome labeling ────────────────────────────────────────────────────────
MFE_TICK_THRESHOLD = 10         # 10 ticks = 2.50 points MNQ
OUTCOME_LOOKAHEAD = 10          # bars after peak

# ── Lookback for context ────────────────────────────────────────────────────
LOOKBACK_BARS = 10

# ── Feature names ───────────────────────────────────────────────────────────
FEAT_16D_NAMES = [
    'abs_z', 'log_vol_delta', 'log_fm', 'osc_entropy',
    'tf_scale', 'depth', 'parent_ctx', 'adx',
    'hurst', 'dmi_diff', 'parent_z', 'parent_dmi_diff',
    'root_is_roche', 'tf_alignment', 'pid_output', 'osc_coh',
]
GEOM_6D_NAMES = ['slope', 'curvature', 'efficiency', 'norm_range', 'end_pos', 'monotonicity']

ALL_54D_NAMES = (
    [f'geom_{n}' for n in GEOM_6D_NAMES]
    + [f'peak_{n}' for n in FEAT_16D_NAMES]
    + [f'delta_{n}' for n in FEAT_16D_NAMES]
    + [f'slope_{n}' for n in FEAT_16D_NAMES]
)


# ═════════════════════════════════════════════════════════════════════════════
# Feature extraction helpers
# ═════════════════════════════════════════════════════════════════════════════

def extract_16d(state) -> np.ndarray:
    """Extract the canonical 16D feature vector from a MarketState."""
    return np.array([
        abs(getattr(state, 'z_score', 0.0)),
        np.log1p(abs(getattr(state, 'volume_delta', 0.0))),
        np.log1p(abs(getattr(state, 'F_momentum', 0.0))),
        getattr(state, 'oscillation_entropy_normalized', 0.0),
        0.0,   # tf_scale placeholder (single TF)
        0.0,   # depth placeholder
        0.0,   # parent_ctx placeholder
        getattr(state, 'adx_strength', 0.0),
        getattr(state, 'hurst_exponent', 0.5),
        getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0),
        0.0,   # parent_z placeholder
        0.0,   # parent_dmi_diff placeholder
        0.0,   # root_is_roche placeholder
        0.0,   # tf_alignment placeholder
        getattr(state, 'pid_output', 0.0) if hasattr(state, 'pid_output') else getattr(state, 'term_pid', 0.0),
        getattr(state, 'oscillation_entropy_normalized', 0.0),
    ], dtype=float)


def extract_6d_geometry(prices_10: np.ndarray) -> np.ndarray:
    """10 close prices -> 6D shape descriptor."""
    prices = np.array(prices_10, dtype=float)
    if len(prices) < 10 or prices.std() == 0:
        return np.zeros(6)

    n = len(prices)
    x = np.arange(n, dtype=float)

    rng = prices.max() - prices.min()
    if rng < 1e-8:
        rng = 1.0

    # slope: linreg slope normalized by range
    slope = np.polyfit(x, prices, 1)[0] / rng

    # curvature: mean 2nd derivative
    d2 = np.diff(prices, 2)
    curvature = d2.mean() / rng if len(d2) > 0 else 0.0

    # efficiency: net change / path length
    net = abs(prices[-1] - prices[0])
    path = np.sum(np.abs(np.diff(prices)))
    efficiency = net / path if path > 0 else 0.0

    # norm_range: range as bps of mean price
    norm_range = rng / prices.mean() * 10000 if prices.mean() > 0 else 0.0

    # end_position within range
    end_pos = (prices[-1] - prices.min()) / rng if rng > 0 else 0.5

    # monotonicity
    diffs = np.diff(prices)
    if len(diffs) > 0:
        pos_frac = (diffs > 0).sum() / len(diffs)
        monotonicity = max(pos_frac, 1 - pos_frac)
    else:
        monotonicity = 0.5

    return np.array([slope, curvature, efficiency, norm_range, end_pos, monotonicity])


def build_54d_vector(states, idx: int, prices: np.ndarray) -> np.ndarray:
    """
    Build the 54D context vector for a peak at bar `idx`.

    Components:
        6D  geometry  (10-bar lookback prices)
        16D at_peak   (features at bar idx)
        16D delta     (features[idx] - features[idx-LOOKBACK_BARS])
        16D slope     (linreg slope of each feature across 10 bars)
    """
    lb = LOOKBACK_BARS

    # --- 6D geometry from prices ---
    geom = extract_6d_geometry(prices[idx - lb: idx])

    # --- 16D at peak ---
    feat_peak = extract_16d(states[idx])

    # --- 16D delta ---
    feat_start = extract_16d(states[idx - lb])
    delta = feat_peak - feat_start

    # --- 16D slope (linreg across lookback window) ---
    feat_window = np.array([extract_16d(states[idx - lb + j]) for j in range(lb)], dtype=float)
    x = np.arange(lb, dtype=float)
    slopes = np.zeros(16)
    for d in range(16):
        col = feat_window[:, d]
        if col.std() > 1e-12:
            slopes[d] = np.polyfit(x, col, 1)[0]

    return np.concatenate([geom, feat_peak, delta, slopes])


# ═════════════════════════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════════════════════════

def load_1m_data(data_root: Path) -> pd.DataFrame:
    """Load all 1m parquet files from ATLAS, sorted by timestamp."""
    tf_dir = data_root / '1m'
    if not tf_dir.exists():
        raise FileNotFoundError(f"1m data directory not found: {tf_dir}")

    files = sorted(tf_dir.glob('*.parquet'))
    if not files:
        raise FileNotFoundError(f"No parquet files in {tf_dir}")

    print(f"Loading {len(files)} 1m parquet files from {tf_dir} ...")
    frames = []
    for f in tqdm(files, desc='Loading parquets'):
        frames.append(pd.read_parquet(f))

    df = pd.concat(frames, ignore_index=True)
    if 'timestamp' in df.columns:
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
    print(f"  Total bars: {len(df):,}")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# Compute states
# ═════════════════════════════════════════════════════════════════════════════

def compute_all_states(df: pd.DataFrame) -> list:
    """Run StatisticalFieldEngine.batch_compute_states on entire dataframe.

    The engine expects day-level data, so we split by date, compute per day,
    then stitch the results. Returns a list parallel to df rows (None for
    bars where the engine can't compute -- e.g. first regression_period bars).
    """
    engine = StatisticalFieldEngine()

    # Split by date for batch_compute_states
    if 'timestamp' in df.columns:
        dates = pd.to_datetime(df['timestamp'], unit='s').dt.date
    elif 'date' in df.columns:
        dates = pd.to_datetime(df['date']).dt.date
    else:
        # Treat whole frame as one chunk
        dates = pd.Series(['all'] * len(df))

    unique_dates = dates.unique()
    print(f"Computing MarketState for {len(unique_dates)} trading days ...")

    all_states = [None] * len(df)
    for d in tqdm(unique_dates, desc='Computing states'):
        mask = dates == d
        day_df = df.loc[mask].copy()
        day_indices = df.index[mask].tolist()

        if len(day_df) < engine.regression_period:
            continue

        results = engine.batch_compute_states(day_df)
        for j, r in enumerate(results):
            all_states[day_indices[j]] = r['state']

    n_valid = sum(1 for s in all_states if s is not None)
    print(f"  Valid states: {n_valid:,} / {len(df):,}")
    return all_states


# ═════════════════════════════════════════════════════════════════════════════
# Peak detection + labeling
# ═════════════════════════════════════════════════════════════════════════════

def detect_and_label_peaks(states: list, prices: np.ndarray):
    """
    Walk through states, detect peaks (same logic as advance_engine), label
    outcome from the 10 bars after the peak.

    Returns list of dicts (one per peak).
    """
    n = len(states)
    peaks = []

    prev_pc = None
    prev_fm = None

    for i in tqdm(range(n), desc='Detecting peaks'):
        state = states[i]
        if state is None:
            prev_pc = None
            prev_fm = None
            continue

        pc = getattr(state, 'P_at_center', 0.0)
        fm = abs(getattr(state, 'F_momentum', 0.0))
        coherence = getattr(state, 'oscillation_entropy_normalized', 0.0)

        is_peak = False
        if prev_pc is not None and prev_fm is not None:
            pc_up = pc > prev_pc * PC_RISE_THRESHOLD if prev_pc > PC_MIN_PREV else False
            fm_down = fm < prev_fm * FM_DECAY_THRESHOLD if prev_fm > FM_MIN_PREV else False
            if (pc_up or fm_down) and coherence > COHERENCE_MIN:
                is_peak = True

        prev_pc = pc
        prev_fm = fm

        if not is_peak:
            continue

        # Need LOOKBACK_BARS before and OUTCOME_LOOKAHEAD after
        if i < LOOKBACK_BARS:
            continue
        if i + OUTCOME_LOOKAHEAD >= n:
            continue

        # Check lookback states are all valid
        lookback_valid = all(states[i - LOOKBACK_BARS + j] is not None for j in range(LOOKBACK_BARS))
        if not lookback_valid:
            continue

        # Check lookahead states are all valid (need prices)
        lookahead_valid = all(states[i + j] is not None for j in range(1, OUTCOME_LOOKAHEAD + 1))
        if not lookahead_valid:
            continue

        # --- Direction implied by peak ---
        raw_fm = getattr(state, 'F_momentum', 0.0)
        if raw_fm > 0:
            implied_direction = 'SHORT'  # old move was UP, reversal = SHORT
        else:
            implied_direction = 'LONG'   # old move was DOWN, reversal = LONG

        # --- Outcome labeling ---
        peak_price = prices[i]
        mfe_reversal = 0.0
        mfe_continuation = 0.0

        for j in range(1, OUTCOME_LOOKAHEAD + 1):
            future_price = prices[i + j]
            if implied_direction == 'LONG':
                mfe_reversal = max(mfe_reversal, (future_price - peak_price) / TICK_SIZE)
                mfe_continuation = max(mfe_continuation, (peak_price - future_price) / TICK_SIZE)
            else:
                mfe_reversal = max(mfe_reversal, (peak_price - future_price) / TICK_SIZE)
                mfe_continuation = max(mfe_continuation, (future_price - peak_price) / TICK_SIZE)

        if mfe_reversal >= MFE_TICK_THRESHOLD:
            outcome = 'reversal'
        elif mfe_continuation >= MFE_TICK_THRESHOLD:
            outcome = 'continuation'
        else:
            outcome = 'plateau'

        # --- Build 54D feature vector ---
        vec_54d = build_54d_vector(states, i, prices)

        # --- Raw diagnostic values ---
        raw_vals = {
            'volume_at_peak': getattr(state, 'volume_delta', 0.0),
            'dmi_gap_at_peak': getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0),
            'fm_at_peak': raw_fm,
            'adx_at_peak': getattr(state, 'adx_strength', 0.0),
            'price_at_peak': peak_price,
        }

        peaks.append({
            'bar_idx': i,
            'timestamp': getattr(state, 'timestamp', 0.0),
            'price': peak_price,
            'implied_direction': implied_direction,
            'outcome': outcome,
            'mfe_reversal': mfe_reversal,
            'mfe_continuation': mfe_continuation,
            'features_54d': vec_54d,
            **raw_vals,
        })

    print(f"\n  Peaks detected: {len(peaks):,}")
    return peaks


# ═════════════════════════════════════════════════════════════════════════════
# Clustering
# ═════════════════════════════════════════════════════════════════════════════

def cluster_peaks(features: np.ndarray):
    """UMAP + HDBSCAN clustering. Returns labels, 2D embedding, 5D embedding."""
    if not CLUSTERING_AVAILABLE:
        return np.full(len(features), -1), np.zeros((len(features), 2)), np.zeros((len(features), 5))

    print("Scaling features ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    print("Running UMAP (5D) ...")
    reducer_5d = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=42)
    embedding_5d = reducer_5d.fit_transform(X_scaled)

    print("Running HDBSCAN ...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
    labels = clusterer.fit_predict(embedding_5d)

    print("Running UMAP (2D for visualization) ...")
    reducer_2d = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
    embedding_2d = reducer_2d.fit_transform(X_scaled)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Clusters: {n_clusters}, Noise points: {n_noise}")

    return labels, embedding_2d, embedding_5d


# ═════════════════════════════════════════════════════════════════════════════
# Analysis + reporting
# ═════════════════════════════════════════════════════════════════════════════

def print_and_save_report(peaks_df: pd.DataFrame, feature_names: list, report_path: Path):
    """Console summary + text report."""
    lines = []

    def log(msg=''):
        print(msg)
        lines.append(msg)

    log("=" * 72)
    log("PEAK TEMPLATE RESEARCH — SUMMARY")
    log("=" * 72)

    total = len(peaks_df)
    log(f"\nTotal peaks: {total:,}")

    # Outcome distribution
    log("\n--- Outcome Distribution ---")
    for outcome in ['reversal', 'plateau', 'continuation']:
        sub = peaks_df[peaks_df['outcome'] == outcome]
        pct = len(sub) / total * 100 if total > 0 else 0
        avg_mfe_r = sub['mfe_reversal'].mean() if len(sub) > 0 else 0
        avg_mfe_c = sub['mfe_continuation'].mean() if len(sub) > 0 else 0
        avg_vol = sub['volume_at_peak'].mean() if len(sub) > 0 else 0
        avg_dmi = sub['dmi_gap_at_peak'].mean() if len(sub) > 0 else 0
        log(f"  {outcome:14s}: {len(sub):5d} ({pct:5.1f}%)  "
            f"avg_mfe_rev={avg_mfe_r:6.1f}  avg_mfe_cont={avg_mfe_c:6.1f}  "
            f"avg_vol={avg_vol:7.1f}  avg_dmi={avg_dmi:+6.2f}")

    # Direction split
    log("\n--- Direction Split ---")
    for d in ['LONG', 'SHORT']:
        sub = peaks_df[peaks_df['implied_direction'] == d]
        if len(sub) == 0:
            continue
        rev_pct = (sub['outcome'] == 'reversal').sum() / len(sub) * 100
        log(f"  {d:6s}: {len(sub):5d} peaks, {rev_pct:.1f}% reversal rate")

    # Kruskal-Wallis feature ranking
    log("\n--- Top Features Separating Outcomes (Kruskal-Wallis) ---")
    groups = {o: peaks_df[peaks_df['outcome'] == o] for o in ['reversal', 'plateau', 'continuation']}

    kw_results = []
    for fname in feature_names:
        if fname not in peaks_df.columns:
            continue
        samples = [g[fname].dropna().values for g in groups.values() if len(g) > 0]
        samples = [s for s in samples if len(s) >= 5]
        if len(samples) < 2:
            continue
        try:
            stat, p = stats.kruskal(*samples)
            kw_results.append((fname, stat, p))
        except Exception:
            pass

    kw_results.sort(key=lambda x: x[1], reverse=True)
    for rank, (fname, stat, p) in enumerate(kw_results[:20], 1):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        log(f"  {rank:2d}. {fname:30s}  H={stat:10.2f}  p={p:.2e}  {sig}")

    # Cluster breakdown
    if 'cluster_label' in peaks_df.columns:
        labels = peaks_df['cluster_label'].values
        unique_labels = sorted(set(labels))
        n_clusters = len([l for l in unique_labels if l >= 0])
        log(f"\n--- Cluster Breakdown ({n_clusters} clusters) ---")
        for cl in unique_labels:
            sub = peaks_df[peaks_df['cluster_label'] == cl]
            outcome_dist = sub['outcome'].value_counts()
            dist_str = ', '.join(f"{o}={c}" for o, c in outcome_dist.items())
            label_str = f"Cluster {cl}" if cl >= 0 else "Noise"
            log(f"  {label_str:12s}: {len(sub):5d} members  [{dist_str}]")

    log("\n" + "=" * 72)

    # Write to file
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n[OK] Report saved to {report_path}")

    return kw_results


def make_chart(peaks_df: pd.DataFrame, embedding_2d: np.ndarray,
               kw_results: list, chart_path: Path):
    """2x2 chart: UMAP by outcome, UMAP by cluster, box plots, stacked bar."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Peak Template Research — Clustering & Feature Separation', fontsize=14)

    outcome_colors = {'reversal': 'green', 'plateau': 'gray', 'continuation': 'red'}

    # ── Top-left: UMAP colored by outcome ────────────────────────────────
    ax = axes[0, 0]
    for outcome, color in outcome_colors.items():
        mask = peaks_df['outcome'] == outcome
        if mask.sum() == 0:
            continue
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   c=color, label=outcome, alpha=0.4, s=8)
    ax.set_title('UMAP — Colored by Outcome')
    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')
    ax.legend(fontsize=8, markerscale=3)

    # ── Top-right: UMAP colored by cluster ───────────────────────────────
    ax = axes[0, 1]
    if 'cluster_label' in peaks_df.columns:
        labels = peaks_df['cluster_label'].values
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap('tab20', max(len(unique_labels), 1))
        for idx_l, cl in enumerate(unique_labels):
            mask = labels == cl
            color = 'lightgray' if cl == -1 else cmap(idx_l)
            label = 'Noise' if cl == -1 else f'C{cl}'
            ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                       c=[color], label=label, alpha=0.4, s=8)
        ax.set_title('UMAP — Colored by Cluster')
        ax.set_xlabel('UMAP-1')
        ax.set_ylabel('UMAP-2')
        if len(unique_labels) <= 15:
            ax.legend(fontsize=6, markerscale=3, ncol=2)
    else:
        ax.text(0.5, 0.5, 'Clustering unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title('UMAP — Colored by Cluster')

    # ── Bottom-left: Box plots of top 4 separating features ─────────────
    ax = axes[1, 0]
    top_features = [r[0] for r in kw_results[:4]] if kw_results else []
    if top_features:
        plot_data = []
        plot_labels = []
        plot_positions = []
        outcomes_ordered = ['reversal', 'plateau', 'continuation']
        colors_bp = ['green', 'gray', 'red']
        width = 0.25
        for fi, fname in enumerate(top_features):
            for oi, outcome in enumerate(outcomes_ordered):
                sub = peaks_df.loc[peaks_df['outcome'] == outcome, fname].dropna()
                if len(sub) > 0:
                    plot_data.append(sub.values)
                    plot_labels.append(f'{outcome[0].upper()}')
                    plot_positions.append(fi + (oi - 1) * width)

        if plot_data:
            bp = ax.boxplot(plot_data, positions=plot_positions, widths=width * 0.8,
                            patch_artist=True, showfliers=False)
            # Color the boxes
            box_idx = 0
            for fi, fname in enumerate(top_features):
                for oi, outcome in enumerate(outcomes_ordered):
                    sub = peaks_df.loc[peaks_df['outcome'] == outcome, fname].dropna()
                    if len(sub) > 0:
                        bp['boxes'][box_idx].set_facecolor(colors_bp[oi])
                        bp['boxes'][box_idx].set_alpha(0.5)
                        box_idx += 1
            ax.set_xticks(range(len(top_features)))
            ax.set_xticklabels([f.replace('peak_', '').replace('delta_', 'd_').replace('slope_', 's_')
                                for f in top_features], fontsize=7, rotation=15)
        ax.set_title('Top 4 Separating Features by Outcome')
        # Manual legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, alpha=0.5, label=o) for o, c in zip(outcomes_ordered, colors_bp)]
        ax.legend(handles=legend_elements, fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No features ranked', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    # ── Bottom-right: Outcome distribution per cluster (stacked bar) ────
    ax = axes[1, 1]
    if 'cluster_label' in peaks_df.columns:
        labels = peaks_df['cluster_label'].values
        unique_labels = sorted([l for l in set(labels) if l >= 0])
        if unique_labels:
            outcomes_ordered = ['reversal', 'plateau', 'continuation']
            x_pos = np.arange(len(unique_labels))
            bottoms = np.zeros(len(unique_labels))
            colors_stack = ['green', 'gray', 'red']
            for oi, outcome in enumerate(outcomes_ordered):
                counts = []
                for cl in unique_labels:
                    c = ((peaks_df['cluster_label'] == cl) & (peaks_df['outcome'] == outcome)).sum()
                    counts.append(c)
                counts = np.array(counts, dtype=float)
                ax.bar(x_pos, counts, bottom=bottoms, color=colors_stack[oi],
                       alpha=0.7, label=outcome, width=0.7)
                bottoms += counts
            ax.set_xticks(x_pos)
            ax.set_xticklabels([f'C{l}' for l in unique_labels], fontsize=7, rotation=45)
            ax.set_title('Outcome Distribution per Cluster')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No clusters found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
    else:
        ax.text(0.5, 0.5, 'Clustering unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(chart_path), dpi=150)
    plt.close(fig)
    print(f"[OK] Chart saved to {chart_path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Peak Template Research — cluster peaks by 10-bar context')
    parser.add_argument('--data', type=str, default='DATA/ATLAS',
                        help='Path to ATLAS data root (default: DATA/ATLAS)')
    args = parser.parse_args()

    data_root = PROJECT_ROOT / args.data

    # ── 1. Load data ────────────────────────────────────────────────────
    df = load_1m_data(data_root)

    # Get prices array
    price_col = 'price' if 'price' in df.columns else 'close'
    prices = df[price_col].values.astype(float)

    # ── 2. Compute states ───────────────────────────────────────────────
    states = compute_all_states(df)

    # ── 3. Detect peaks + label outcomes ────────────────────────────────
    peaks = detect_and_label_peaks(states, prices)

    if len(peaks) == 0:
        print("\n[WARN] No peaks detected. Check data or thresholds.")
        return

    # ── 4. Build feature matrix + dataframe ─────────────────────────────
    features_54d = np.array([p['features_54d'] for p in peaks], dtype=float)

    # Replace NaN/Inf with 0
    features_54d = np.nan_to_num(features_54d, nan=0.0, posinf=0.0, neginf=0.0)

    # Build dataframe
    peaks_df = pd.DataFrame({
        'peak_bar_idx': [p['bar_idx'] for p in peaks],
        'timestamp': [p['timestamp'] for p in peaks],
        'price': [p['price'] for p in peaks],
        'implied_direction': [p['implied_direction'] for p in peaks],
        'outcome': [p['outcome'] for p in peaks],
        'mfe_reversal': [p['mfe_reversal'] for p in peaks],
        'mfe_continuation': [p['mfe_continuation'] for p in peaks],
        'volume_at_peak': [p['volume_at_peak'] for p in peaks],
        'dmi_gap_at_peak': [p['dmi_gap_at_peak'] for p in peaks],
        'fm_at_peak': [p['fm_at_peak'] for p in peaks],
        'adx_at_peak': [p['adx_at_peak'] for p in peaks],
        'price_at_peak': [p['price_at_peak'] for p in peaks],
    })

    # Add 54D features as columns
    for i, name in enumerate(ALL_54D_NAMES):
        peaks_df[name] = features_54d[:, i]

    # ── 5. Cluster ──────────────────────────────────────────────────────
    labels, embedding_2d, embedding_5d = cluster_peaks(features_54d)
    peaks_df['cluster_label'] = labels

    # ── 6. Report ───────────────────────────────────────────────────────
    report_dir = PROJECT_ROOT / 'reports' / 'findings'
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / 'peak_template_features.csv'
    report_path = report_dir / 'peak_template_summary.txt'
    chart_path = report_dir / 'peak_template_clusters.png'

    # Save CSV
    peaks_df.to_csv(str(csv_path), index=False)
    print(f"\n[OK] CSV saved to {csv_path}  ({len(peaks_df):,} rows)")

    # Print + save text report
    kw_results = print_and_save_report(peaks_df, ALL_54D_NAMES, report_path)

    # Make chart
    make_chart(peaks_df, embedding_2d, kw_results, chart_path)

    print("\nDone.")


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    main()
