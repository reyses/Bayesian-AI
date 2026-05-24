"""Minimum Prediction Window — how many 1s bars to predict next-bar sign?

The fundamental question: given N bars of 1s data with grounded features
(velocity, std, volume), at what N does sign prediction exceed 50%?

Tests:
  - N = 2, 3, 5, 10, 20, 30, 60, 120, 180, 360
  - Predicts sign of next 1s price change
  - Splits by variance ratio bucket (trending/reverting/random)
  - Three prediction methods:
    1. Momentum: sign(mean velocity over N) → predicts continuation
    2. Reversion: -sign(mean velocity over N) → predicts reversal
    3. Best-of: pick whichever worked better in the prior window

Outputs:
  - tools/plots/min_prediction_window.png — accuracy by N and regime
  - reports/findings/min_prediction_window.txt — full results table

Usage:
    python tools/minimum_prediction_window.py
    python tools/minimum_prediction_window.py --data DATA/ATLAS_1WEEK --max-bars 50000
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25

# Window sizes to test (in 1s bars)
WINDOWS = [2, 3, 5, 10, 20, 30, 60, 120, 180, 360]

# Variance ratio windows for regime classification
VR_SHORT = 30   # ~30 seconds
VR_LONG = 120   # ~2 minutes


def load_1s_data(data_dir: str, max_bars: int = 0):
    """Load 1s ATLAS data."""
    print(f'Loading 1s data from {data_dir}...')
    files = sorted(glob.glob(os.path.join(data_dir, '1s', '*.parquet')))
    if not files:
        print(f'ERROR: No 1s parquet files in {data_dir}/1s/')
        sys.exit(1)

    from tqdm import tqdm
    dfs = []
    total = 0
    for f in tqdm(files, desc='Loading files'):
        df = pd.read_parquet(f)
        dfs.append(df)
        total += len(df)
        if max_bars > 0 and total >= max_bars:
            break

    data = pd.concat(dfs, ignore_index=True)
    if max_bars > 0 and len(data) > max_bars:
        data = data.iloc[:max_bars]

    print(f'  {len(data):,} bars from {len(files)} files')
    return data


def compute_grounded_features(closes: np.ndarray, volumes: np.ndarray,
                                highs: np.ndarray, lows: np.ndarray):
    """Compute grounded features from raw 1s data. No SFE needed.

    All features are level 1-3 from the feature tree:
    - velocity (level 2): dP/dt
    - std_price (level 2): rolling std of price changes
    - volume (level 2): raw volume per bar
    - acceleration (level 3): d(velocity)/dt
    - variance_ratio (level 3): var(short)/var(long)
    - dmi_diff_proxy (level 2): (high - prev_high) - (prev_low - low)
    """
    n = len(closes)

    # Price changes (in ticks)
    dp = np.diff(closes) / TICK
    dp = np.concatenate([[0], dp])

    # Velocity = price change per bar (already dp since dt=1s)
    velocity = dp

    # Acceleration = change in velocity
    accel = np.diff(velocity)
    accel = np.concatenate([[0], accel])

    # Rolling std of price changes (volatility)
    std_price = np.full(n, np.nan)
    for w in [20]:  # 20s rolling window
        for i in range(w, n):
            std_price[i] = np.std(dp[i-w:i], ddof=1)

    # Variance ratio: var(short) / var(long)
    var_ratio = np.full(n, np.nan)
    for i in range(VR_LONG, n):
        v_short = np.var(dp[i-VR_SHORT:i], ddof=1)
        v_long = np.var(dp[i-VR_LONG:i], ddof=1)
        if v_long > 1e-10:
            var_ratio[i] = v_short / v_long
        else:
            var_ratio[i] = 1.0

    # DMI diff proxy: directional movement from highs/lows
    up_move = np.diff(highs)
    down_move = np.diff(lows) * -1  # positive = lows moving down
    up_move = np.concatenate([[0], up_move])
    down_move = np.concatenate([[0], down_move])
    dmi_proxy = (up_move - down_move) / TICK

    return {
        'velocity': velocity,
        'acceleration': accel,
        'std_price': std_price,
        'var_ratio': var_ratio,
        'volume': volumes,
        'dmi_proxy': dmi_proxy,
        'dp': dp,
    }


def classify_regime(var_ratio: float) -> str:
    """Classify regime from variance ratio."""
    if np.isnan(var_ratio):
        return 'unknown'
    if var_ratio > 1.3:
        return 'trending'
    elif var_ratio < 0.7:
        return 'reverting'
    else:
        return 'random'


def run_prediction_test(dp: np.ndarray, velocity: np.ndarray,
                         var_ratio: np.ndarray, features: dict):
    """Test prediction accuracy at each window size."""
    n = len(dp)
    max_w = max(WINDOWS)

    results = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    # Progress
    from tqdm import tqdm

    # For each bar (starting after max window + VR_LONG warmup)
    start = max(max_w, VR_LONG) + 1

    for i in tqdm(range(start, n - 1), desc='Testing predictions'):
        # Target: sign of next bar's price change
        next_sign = 1 if dp[i + 1] > 0 else (-1 if dp[i + 1] < 0 else 0)
        if next_sign == 0:
            continue  # skip zero changes

        # Regime
        regime = classify_regime(var_ratio[i])

        for w in WINDOWS:
            if i < w:
                continue

            window = dp[i-w:i]
            vel_window = velocity[i-w:i]

            # Method 1: Momentum — predict continuation
            mean_vel = np.mean(vel_window)
            mom_pred = 1 if mean_vel > 0 else -1

            # Method 2: Reversion — predict reversal
            rev_pred = -mom_pred

            # Method 3: Weighted velocity — recent bars matter more
            weights = np.linspace(0.5, 1.5, w)
            weighted_vel = np.average(vel_window, weights=weights)
            weighted_pred = 1 if weighted_vel > 0 else -1

            # Method 4: Acceleration-based — if decelerating, predict flip
            accel_window = features['acceleration'][i-w:i]
            mean_accel = np.mean(accel_window[-min(5, w):])  # last 5 bars of accel
            if mean_vel > 0 and mean_accel < 0:
                accel_pred = -1  # moving up but slowing -> reversal
            elif mean_vel < 0 and mean_accel > 0:
                accel_pred = 1   # moving down but slowing -> reversal
            else:
                accel_pred = mom_pred  # no deceleration -> continuation

            # Method 5: Variance ratio adaptive
            if regime == 'trending':
                adaptive_pred = mom_pred
            elif regime == 'reverting':
                adaptive_pred = rev_pred
            else:
                adaptive_pred = mom_pred  # default to momentum in unknown

            # Score each method
            for method, pred in [('momentum', mom_pred),
                                  ('reversion', rev_pred),
                                  ('weighted', weighted_pred),
                                  ('acceleration', accel_pred),
                                  ('adaptive', adaptive_pred)]:
                key = (w, method)
                results[key][regime]['total'] += 1
                results[key]['all']['total'] += 1
                if pred == next_sign:
                    results[key][regime]['correct'] += 1
                    results[key]['all']['correct'] += 1

    return results


def format_results(results):
    """Format results into summary text."""
    lines = []
    lines.append('=' * 100)
    lines.append('MINIMUM PREDICTION WINDOW — How many 1s bars to predict next-bar sign?')
    lines.append('=' * 100)
    lines.append('')

    methods = ['momentum', 'reversion', 'weighted', 'acceleration', 'adaptive']
    regimes = ['all', 'trending', 'reverting', 'random']

    for method in methods:
        lines.append(f'\n{"="*80}')
        lines.append(f'METHOD: {method.upper()}')
        lines.append(f'{"="*80}')

        header = f'{"Window":>8}'
        for regime in regimes:
            header += f'  {regime:>12} (n)'
        lines.append(header)
        lines.append('-' * 90)

        for w in WINDOWS:
            key = (w, method)
            row = f'{w:>6}s '
            for regime in regimes:
                d = results[key][regime]
                total = d['total']
                if total > 0:
                    acc = d['correct'] / total * 100
                    row += f'  {acc:>6.2f}% ({total:>7,})'
                else:
                    row += f'  {"n/a":>6} ({"0":>7})'
            lines.append(row)

    # Best method per window
    lines.append(f'\n{"="*80}')
    lines.append('BEST METHOD PER WINDOW (all regimes)')
    lines.append(f'{"="*80}')
    lines.append(f'{"Window":>8}  {"Best Method":<15}  {"Accuracy":>8}  {"n":>10}  {"vs 50%":>8}')
    lines.append('-' * 60)

    for w in WINDOWS:
        best_method = ''
        best_acc = 0
        best_n = 0
        for method in methods:
            key = (w, method)
            d = results[key]['all']
            if d['total'] > 0:
                acc = d['correct'] / d['total'] * 100
                if acc > best_acc:
                    best_acc = acc
                    best_method = method
                    best_n = d['total']
        edge = best_acc - 50.0
        lines.append(f'{w:>6}s   {best_method:<15}  {best_acc:>7.2f}%  {best_n:>10,}  {edge:>+7.2f}%')

    # Regime breakdown for best method
    lines.append(f'\n{"="*80}')
    lines.append('REGIME BREAKDOWN (adaptive method)')
    lines.append(f'{"="*80}')
    lines.append(f'{"Window":>8}  {"Trending":>12}  {"Reverting":>12}  {"Random":>12}')
    lines.append('-' * 55)

    for w in WINDOWS:
        key = (w, 'adaptive')
        row = f'{w:>6}s '
        for regime in ['trending', 'reverting', 'random']:
            d = results[key][regime]
            if d['total'] > 100:
                acc = d['correct'] / d['total'] * 100
                row += f'  {acc:>6.2f}% n={d["total"]:>6,}'
            else:
                row += f'  {"thin":>6} n={d["total"]:>6,}'
        lines.append(row)

    lines.append('')
    lines.append('INTERPRETATION:')
    lines.append('  > 50% = better than coin flip (edge exists)')
    lines.append('  > 52% = tradable edge with enough volume')
    lines.append('  > 55% = strong edge')
    lines.append('  = 50% = random walk at this window (no prediction possible)')
    lines.append('  < 50% = contrarian signal (flip your prediction)')

    return '\n'.join(lines)


def plot_results(results):
    """Plot accuracy by window size and regime."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    methods = ['momentum', 'reversion', 'adaptive', 'acceleration']
    regimes = ['all', 'trending', 'reverting', 'random']
    colors = {'momentum': 'blue', 'reversion': 'red',
              'adaptive': 'green', 'acceleration': 'orange'}

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Minimum Prediction Window — 1s Bar Sign Prediction',
                 fontsize=16, fontweight='bold')

    for ax_idx, regime in enumerate(regimes):
        ax = axes[ax_idx // 2][ax_idx % 2]
        ax.axhline(50, color='gray', ls='--', lw=1, label='coin flip')

        for method in methods:
            accs = []
            ws = []
            for w in WINDOWS:
                key = (w, method)
                d = results[key][regime]
                if d['total'] > 100:
                    accs.append(d['correct'] / d['total'] * 100)
                    ws.append(w)

            if accs:
                ax.plot(ws, accs, 'o-', color=colors[method],
                        label=method, markersize=5)

        ax.set_xlabel('Window size (1s bars)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Regime: {regime}')
        ax.set_xscale('log')
        ax.set_xticks(WINDOWS)
        ax.set_xticklabels([str(w) for w in WINDOWS], rotation=45)
        ax.legend(fontsize=8)
        ax.set_ylim(45, 60)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join('tools', 'plots', 'min_prediction_window.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Plot: {out_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Minimum prediction window research')
    parser.add_argument('--data', default='DATA/ATLAS_OOS',
                        help='Data directory (default: DATA/ATLAS_OOS)')
    parser.add_argument('--max-bars', type=int, default=0,
                        help='Limit bars (0=all)')
    args = parser.parse_args()

    data = load_1s_data(args.data, args.max_bars)

    closes = data['close'].values
    volumes = data['volume'].values if 'volume' in data.columns else np.zeros(len(data))
    highs = data['high'].values
    lows = data['low'].values

    print('Computing grounded features...')
    features = compute_grounded_features(closes, volumes, highs, lows)

    print(f'Running prediction test ({len(closes):,} bars, {len(WINDOWS)} windows, 5 methods)...')
    results = run_prediction_test(
        features['dp'], features['velocity'],
        features['var_ratio'], features)

    # Format and save
    summary = format_results(results)

    out_path = os.path.join('reports', 'findings', 'min_prediction_window.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nSummary: {out_path}')

    # Print quick results
    print('\n' + summary)

    # Plot
    plot_results(results)


if __name__ == '__main__':
    main()
