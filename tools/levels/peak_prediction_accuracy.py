"""Peak Prediction Accuracy — is direction predictable at detected peaks?

Hypothesis: the 50.77% reversion accuracy on ALL bars jumps to 65%+ at
bars that are detected as peaks (velocity flip + magnitude + volume collapse).

Uses ONLY grounded features computed from raw 1s data. No SFE, no PhysicsEngine.
This is standalone research for the AdvanceEngine rebuild.

Tests:
  1. Detect peak candidates from grounded features
  2. Measure reversion accuracy AT peak bars vs random bars
  3. Vary detection sensitivity to find the sweet spot
  4. Measure at 1s, 10s, 30s, 60s lookaheads (does the peak signal persist?)

Outputs:
  - reports/findings/peak_prediction_accuracy.txt
  - tools/plots/peak_prediction_accuracy.png

Usage:
    python tools/peak_prediction_accuracy.py
    python tools/peak_prediction_accuracy.py --data DATA/ATLAS_1WEEK --max-bars 100000
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

# Lookahead windows: predict sign of price change over next N bars
LOOKAHEADS = [1, 3, 5, 10, 30, 60]

# Peak detection parameters to sweep
VELOCITY_WINDOWS = [5, 10, 20]       # bars to compute velocity
MAGNITUDE_PCTILES = [50, 75, 90]     # prior move must exceed this percentile
VOLUME_COLLAPSE_PCT = [0.3, 0.5]     # volume must drop below this fraction of recent avg


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


def compute_base_features(closes, volumes, highs, lows):
    """Compute grounded features from raw data."""
    n = len(closes)
    dp = np.diff(closes) / TICK
    dp = np.concatenate([[0], dp])

    # Rolling velocity at multiple windows
    velocities = {}
    for w in [5, 10, 20, 30]:
        vel = np.full(n, 0.0)
        for i in range(w, n):
            vel[i] = np.mean(dp[i-w:i])
        velocities[w] = vel

    # Rolling volume average (for collapse detection)
    vol_avg = np.full(n, 0.0)
    vol_w = 60  # 60-second average
    for i in range(vol_w, n):
        vol_avg[i] = np.mean(volumes[i-vol_w:i])

    # Rolling magnitude (absolute cumulative move over window)
    magnitudes = {}
    for w in [10, 20, 30, 60]:
        mag = np.full(n, 0.0)
        for i in range(w, n):
            mag[i] = abs(closes[i] - closes[i-w]) / TICK
        magnitudes[w] = mag

    # Rolling magnitude percentiles (causal — from prior data only)
    mag_pctiles = {}
    pctile_window = 3600  # 1 hour of history for percentile ranking
    for w in [10, 20, 30, 60]:
        pct = np.full(n, 0.0)
        for i in range(max(w, pctile_window), n):
            recent_mags = magnitudes[w][i-pctile_window:i]
            current_mag = magnitudes[w][i]
            pct[i] = np.sum(recent_mags < current_mag) / len(recent_mags)
        mag_pctiles[w] = pct

    # Velocity sign (direction of recent move)
    vel_signs = {}
    for w in velocities:
        vel_signs[w] = np.sign(velocities[w])

    # Velocity sign FLIP detection (sign changed from previous bar)
    vel_flips = {}
    for w in velocities:
        flips = np.zeros(n, dtype=bool)
        signs = vel_signs[w]
        for i in range(1, n):
            if signs[i] != 0 and signs[i-1] != 0 and signs[i] != signs[i-1]:
                flips[i] = True
        vel_flips[w] = flips

    return {
        'dp': dp,
        'velocities': velocities,
        'vel_signs': vel_signs,
        'vel_flips': vel_flips,
        'volumes': volumes,
        'vol_avg': vol_avg,
        'magnitudes': magnitudes,
        'mag_pctiles': mag_pctiles,
        'closes': closes,
    }


def detect_peaks(features, vel_window, mag_window, mag_pctile_thresh,
                 vol_collapse_thresh, require_volume=True):
    """Detect peak candidates from grounded features.

    A peak is: velocity just flipped sign + prior move was big + volume collapsed.
    Returns boolean mask of peak bars.
    """
    n = len(features['dp'])
    flips = features['vel_flips'][vel_window]
    mag_pct = features['mag_pctiles'].get(mag_window,
              features['mag_pctiles'].get(20))  # fallback

    peaks = np.zeros(n, dtype=bool)

    for i in range(1, n):
        if not flips[i]:
            continue

        # Magnitude filter: prior move must be large enough
        if mag_pct[i] < mag_pctile_thresh / 100.0:
            continue

        # Volume collapse filter (optional)
        if require_volume and features['vol_avg'][i] > 0:
            current_vol = features['volumes'][i]
            avg_vol = features['vol_avg'][i]
            if avg_vol > 0 and current_vol / avg_vol > vol_collapse_thresh:
                continue  # volume still high, not exhausted

        peaks[i] = True

    return peaks


def measure_accuracy(dp, closes, peaks, lookaheads, vel_sign_at_peak):
    """Measure reversion accuracy at peak bars vs random bars.

    At a peak, the velocity just flipped. Reversion = predict the NEW direction
    (the direction velocity flipped TO, not FROM).
    """
    n = len(dp)
    max_la = max(lookaheads)

    results = {'peak': {}, 'random': {}, 'peak_count': int(np.sum(peaks))}

    for la in lookaheads:
        peak_correct = 0
        peak_total = 0
        random_correct = 0
        random_total = 0

        for i in range(n - la):
            # Future price change over la bars
            future_change = closes[min(i + la, n-1)] - closes[i]
            future_sign = 1 if future_change > 0 else (-1 if future_change < 0 else 0)
            if future_sign == 0:
                continue

            if peaks[i]:
                # At peak: predict REVERSION (opposite of the old move direction)
                # vel_sign_at_peak[i] is the NEW direction after flip
                # So predicting the new direction = reversion of old
                pred = vel_sign_at_peak[i]
                if pred == 0:
                    continue
                peak_total += 1
                if pred == future_sign:
                    peak_correct += 1
            else:
                # Random bar: simple reversion (opposite of recent velocity)
                random_total += 1
                # Use dp sign as proxy
                if dp[i] != 0:
                    pred = -np.sign(dp[i])
                    if pred == future_sign:
                        random_correct += 1

        results['peak'][la] = {
            'correct': peak_correct,
            'total': peak_total,
            'accuracy': peak_correct / peak_total * 100 if peak_total > 0 else 0,
        }
        results['random'][la] = {
            'correct': random_correct,
            'total': random_total,
            'accuracy': random_correct / random_total * 100 if random_total > 0 else 0,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Peak prediction accuracy research')
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
    features = compute_base_features(closes, volumes, highs, lows)

    lines = []
    lines.append('=' * 90)
    lines.append('PEAK PREDICTION ACCURACY — Is direction predictable at detected peaks?')
    lines.append(f'Data: {len(closes):,} bars of 1s data')
    lines.append(f'Lookaheads: {LOOKAHEADS} bars (1s each)')
    lines.append('=' * 90)

    all_configs = []

    from tqdm import tqdm
    configs = []
    for vw in VELOCITY_WINDOWS:
        for mp in MAGNITUDE_PCTILES:
            for vc in VOLUME_COLLAPSE_PCT:
                configs.append((vw, mp, vc, True))
            configs.append((vw, mp, 0, False))  # no volume filter

    print(f'\nTesting {len(configs)} peak detection configurations...')

    for vw, mp, vc, use_vol in tqdm(configs, desc='Configurations'):
        mw = min(20, vw * 2)  # magnitude window = 2x velocity window
        peaks = detect_peaks(features, vw, mw, mp, vc, require_volume=use_vol)
        n_peaks = int(np.sum(peaks))

        if n_peaks < 50:
            continue

        # Get velocity sign at each peak (the NEW direction after flip)
        vel_sign_at_peak = features['vel_signs'][vw]

        results = measure_accuracy(
            features['dp'], closes, peaks, LOOKAHEADS, vel_sign_at_peak)

        config_label = f'vel_w={vw} mag_p{mp}'
        if use_vol:
            config_label += f' vol<{vc}'
        else:
            config_label += ' no_vol'

        config_result = {
            'label': config_label,
            'peaks': n_peaks,
            'pct_bars': n_peaks / len(closes) * 100,
            'results': results,
        }
        all_configs.append(config_result)

        # Print progress
        peak_acc_1 = results['peak'].get(1, {}).get('accuracy', 0)
        rand_acc_1 = results['random'].get(1, {}).get('accuracy', 0)
        edge = peak_acc_1 - rand_acc_1

    # Sort by peak accuracy at 10-bar lookahead
    all_configs.sort(
        key=lambda x: x['results']['peak'].get(10, {}).get('accuracy', 0),
        reverse=True)

    # Format results
    lines.append('')
    lines.append(f'{"Config":<30} {"Peaks":>7} {"% bars":>7}', )

    header = f'{"Config":<30} {"Peaks":>7} {"% bars":>7}'
    for la in LOOKAHEADS:
        header += f'  +{la}s peak'
        header += f'  +{la}s rand'
        header += f'  edge'
    lines.append('')
    lines.append(header)
    lines.append('-' * (40 + 21 * len(LOOKAHEADS)))

    for cfg in all_configs:
        row = f'{cfg["label"]:<30} {cfg["peaks"]:>7,} {cfg["pct_bars"]:>6.2f}%'
        for la in LOOKAHEADS:
            pa = cfg['results']['peak'].get(la, {}).get('accuracy', 0)
            ra = cfg['results']['random'].get(la, {}).get('accuracy', 0)
            edge = pa - ra
            row += f'  {pa:>5.1f}%  {ra:>5.1f}%  {edge:>+5.1f}'
        lines.append(row)

    # Best config analysis
    if all_configs:
        best = all_configs[0]
        lines.append('')
        lines.append('=' * 90)
        lines.append(f'BEST CONFIG: {best["label"]}')
        lines.append(f'  Peaks detected: {best["peaks"]:,} ({best["pct_bars"]:.2f}% of bars)')
        lines.append('')
        lines.append(f'  {"Lookahead":>10}  {"Peak Acc":>10}  {"Random Acc":>10}  {"Edge":>8}  {"Peak N":>8}')
        lines.append('  ' + '-' * 55)
        for la in LOOKAHEADS:
            p = best['results']['peak'].get(la, {})
            r = best['results']['random'].get(la, {})
            pa = p.get('accuracy', 0)
            ra = r.get('accuracy', 0)
            pn = p.get('total', 0)
            lines.append(f'  {la:>8}s  {pa:>9.2f}%  {ra:>9.2f}%  {pa-ra:>+7.2f}%  {pn:>8,}')

    lines.append('')
    lines.append('INTERPRETATION:')
    lines.append('  Peak accuracy >> Random accuracy = peaks ARE predictable')
    lines.append('  Edge > 5% at any lookahead = tradable detection signal')
    lines.append('  Edge increases with lookahead = structural reversal (not noise)')
    lines.append('  Edge decreases with lookahead = noise spike (not real peak)')

    summary = '\n'.join(lines)

    out_path = os.path.join('reports', 'findings', 'peak_prediction_accuracy.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nSummary: {out_path}')
    print('\n' + summary)

    # Plot
    if all_configs:
        _plot(all_configs)


def _plot(all_configs):
    """Plot peak vs random accuracy across lookaheads for top configs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    top = all_configs[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Peak vs Random Prediction Accuracy', fontsize=16, fontweight='bold')

    for idx, cfg in enumerate(top):
        ax = axes[idx // 3][idx % 3]

        peak_accs = [cfg['results']['peak'].get(la, {}).get('accuracy', 50) for la in LOOKAHEADS]
        rand_accs = [cfg['results']['random'].get(la, {}).get('accuracy', 50) for la in LOOKAHEADS]

        ax.plot(LOOKAHEADS, peak_accs, 'go-', label='peak bars', markersize=6)
        ax.plot(LOOKAHEADS, rand_accs, 'r^-', label='random bars', markersize=6)
        ax.axhline(50, color='gray', ls='--', lw=1)
        ax.set_xlabel('Lookahead (1s bars)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{cfg["label"]}\n({cfg["peaks"]:,} peaks, {cfg["pct_bars"]:.1f}%)')
        ax.legend(fontsize=8)
        ax.set_ylim(45, 65)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join('tools', 'plots', 'peak_prediction_accuracy.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'Plot: {out_path}')


if __name__ == '__main__':
    main()
