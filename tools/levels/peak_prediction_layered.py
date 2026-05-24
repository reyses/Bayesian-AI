"""Peak Prediction Layered — add DMI + variance ratio to peak detection.

Baseline: vel_w=5 mag_p75 vol<0.3 = 50.8% (+5.1% edge at 60s)
Question: does adding DMI exhaustion and/or variance ratio filter improve accuracy?

Layers tested:
  L0: velocity flip only (2-feature baseline)
  L1: + magnitude filter (prior move was big)
  L2: + volume collapse (participation drying up)
  L3: + DMI exhaustion (buyer/seller battle extreme + volume low)
  L4: + variance ratio filter (only trade in reverting regime)
  L5: L2 + L4 (best combo from L2 + regime filter)

Uses ONLY grounded features. No SFE, no PhysicsEngine.

Usage:
    python tools/peak_prediction_layered.py
    python tools/peak_prediction_layered.py --data DATA/ATLAS_1WEEK --max-bars 100000
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
LOOKAHEADS = [1, 5, 10, 30, 60]

# Variance ratio windows
VR_SHORT = 30
VR_LONG = 120


def load_1s_data(data_dir: str, max_bars: int = 0):
    """Load 1s ATLAS data."""
    files = sorted(glob.glob(os.path.join(data_dir, '1s', '*.parquet')))
    if not files:
        print(f'ERROR: No 1s parquet files in {data_dir}/1s/')
        sys.exit(1)
    from tqdm import tqdm
    dfs = []
    total = 0
    for f in tqdm(files, desc='Loading'):
        df = pd.read_parquet(f)
        dfs.append(df)
        total += len(df)
        if max_bars > 0 and total >= max_bars:
            break
    data = pd.concat(dfs, ignore_index=True)
    if max_bars > 0 and len(data) > max_bars:
        data = data.iloc[:max_bars]
    print(f'  {len(data):,} bars')
    return data


def compute_all_features(closes, volumes, highs, lows):
    """Compute ALL grounded features needed for layered testing."""
    n = len(closes)
    dp = np.diff(closes) / TICK
    dp = np.concatenate([[0], dp])

    # Velocity (5-bar)
    vel = np.full(n, 0.0)
    for i in range(5, n):
        vel[i] = np.mean(dp[i-5:i])
    vel_sign = np.sign(vel)

    # Velocity flip
    flips = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if vel_sign[i] != 0 and vel_sign[i-1] != 0 and vel_sign[i] != vel_sign[i-1]:
            flips[i] = True

    # Magnitude (20-bar move)
    magnitude = np.full(n, 0.0)
    for i in range(20, n):
        magnitude[i] = abs(closes[i] - closes[i-20]) / TICK

    # Magnitude percentile (forward pass, 1h lookback)
    mag_pct = np.full(n, 0.0)
    pctile_w = 3600
    for i in range(max(20, pctile_w), n):
        recent = magnitude[i-pctile_w:i]
        mag_pct[i] = np.sum(recent < magnitude[i]) / len(recent)

    # Volume average (60s)
    vol_avg = np.full(n, 0.0)
    for i in range(60, n):
        vol_avg[i] = np.mean(volumes[i-60:i])

    # DMI proxy: directional movement from highs/lows
    up_move = np.concatenate([[0], np.diff(highs)]) / TICK
    down_move = np.concatenate([[0], -np.diff(lows)]) / TICK
    # Use positive values only (Wilder's method)
    up_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    down_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed DMI (14-bar rolling average as simple proxy)
    dmi_window = 14
    dmi_plus = np.full(n, 0.0)
    dmi_minus = np.full(n, 0.0)
    for i in range(dmi_window, n):
        dmi_plus[i] = np.mean(up_dm[i-dmi_window:i])
        dmi_minus[i] = np.mean(down_dm[i-dmi_window:i])

    dmi_diff = dmi_plus - dmi_minus
    dmi_sum = dmi_plus + dmi_minus
    dmi_ratio = np.where(dmi_sum > 0, np.abs(dmi_diff) / dmi_sum, 0)

    # DMI exhaustion: one side dominant (ratio > 0.6) AND volume below average
    dmi_exhausted = np.zeros(n, dtype=bool)
    for i in range(60, n):
        if dmi_ratio[i] > 0.6 and vol_avg[i] > 0:
            if volumes[i] / vol_avg[i] < 0.5:
                dmi_exhausted[i] = True

    # Variance ratio
    var_ratio = np.full(n, 1.0)
    for i in range(VR_LONG, n):
        v_short = np.var(dp[i-VR_SHORT:i], ddof=1)
        v_long = np.var(dp[i-VR_LONG:i], ddof=1)
        if v_long > 1e-10:
            var_ratio[i] = v_short / v_long

    # Regime classification
    regime = np.full(n, 'random', dtype=object)
    for i in range(n):
        if var_ratio[i] > 1.3:
            regime[i] = 'trending'
        elif var_ratio[i] < 0.7:
            regime[i] = 'reverting'

    return {
        'dp': dp,
        'vel': vel,
        'vel_sign': vel_sign,
        'flips': flips,
        'magnitude': magnitude,
        'mag_pct': mag_pct,
        'volumes': volumes,
        'vol_avg': vol_avg,
        'dmi_plus': dmi_plus,
        'dmi_minus': dmi_minus,
        'dmi_diff': dmi_diff,
        'dmi_ratio': dmi_ratio,
        'dmi_exhausted': dmi_exhausted,
        'var_ratio': var_ratio,
        'regime': regime,
        'closes': closes,
    }


def test_layer(name, peak_mask, feat, lookaheads):
    """Measure prediction accuracy for a set of peaks."""
    n = len(feat['dp'])
    closes = feat['closes']
    vel_sign = feat['vel_sign']
    max_la = max(lookaheads)

    n_peaks = int(np.sum(peak_mask))
    results = {}

    for la in lookaheads:
        correct = 0
        total = 0
        for i in range(n - la):
            if not peak_mask[i]:
                continue
            future_change = closes[i + la] - closes[i]
            future_sign = 1 if future_change > 0 else (-1 if future_change < 0 else 0)
            if future_sign == 0:
                continue
            pred = vel_sign[i]  # new direction after flip
            if pred == 0:
                continue
            total += 1
            if pred == future_sign:
                correct += 1

        acc = correct / total * 100 if total > 0 else 0
        results[la] = {'correct': correct, 'total': total, 'accuracy': acc}

    return n_peaks, results


def main():
    parser = argparse.ArgumentParser(description='Layered peak prediction')
    parser.add_argument('--data', default='DATA/ATLAS_OOS')
    parser.add_argument('--max-bars', type=int, default=0)
    args = parser.parse_args()

    data = load_1s_data(args.data, args.max_bars)
    closes = data['close'].values
    volumes = data['volume'].values if 'volume' in data.columns else np.zeros(len(data))
    highs = data['high'].values
    lows = data['low'].values

    print('Computing features...')
    feat = compute_all_features(closes, volumes, highs, lows)
    n = len(closes)

    print(f'Building layer masks...')

    # L0: velocity flip only
    L0 = feat['flips'].copy()

    # L1: + magnitude > p75
    L1 = L0 & (feat['mag_pct'] >= 0.75)

    # L2: + volume collapse < 30% of avg (baseline from prior research)
    L2 = np.zeros(n, dtype=bool)
    for i in range(n):
        if L1[i] and feat['vol_avg'][i] > 0:
            if feat['volumes'][i] / feat['vol_avg'][i] < 0.3:
                L2[i] = True

    # L3: L1 + DMI exhaustion (instead of volume collapse)
    L3 = L1 & feat['dmi_exhausted']

    # L4: L2 + reverting regime only
    L4 = L2 & (feat['regime'] == 'reverting')

    # L5: L1 + DMI exhaustion + reverting regime
    L5 = L3 & (feat['regime'] == 'reverting')

    # L6: L2 + DMI exhaustion (both volume collapse AND DMI extreme)
    L6 = L2 & feat['dmi_exhausted']

    # L7: L1 + reverting regime (no volume filter, just regime)
    L7 = L1 & (feat['regime'] == 'reverting')

    # L8: flip + DMI exhaustion only (no magnitude)
    L8 = feat['flips'] & feat['dmi_exhausted']

    layers = [
        ('L0: vel flip only', L0),
        ('L1: + magnitude>p75', L1),
        ('L2: + vol collapse<30%', L2),
        ('L3: L1 + DMI exhaustion', L3),
        ('L4: L2 + reverting regime', L4),
        ('L5: L3 + reverting regime', L5),
        ('L6: L2 + DMI exhaustion', L6),
        ('L7: L1 + reverting regime', L7),
        ('L8: flip + DMI only', L8),
    ]

    from tqdm import tqdm
    lines = []
    lines.append('=' * 100)
    lines.append('PEAK PREDICTION — LAYERED FEATURE TEST')
    lines.append(f'Data: {n:,} bars of 1s | Lookaheads: {LOOKAHEADS}')
    lines.append('=' * 100)
    lines.append('')
    lines.append('Each layer adds ONE filter. Does accuracy improve?')
    lines.append('')

    header = f'{"Layer":<30} {"Peaks":>8} {"% bars":>7}'
    for la in LOOKAHEADS:
        header += f'  +{la}s acc'
    lines.append(header)
    lines.append('-' * (50 + 10 * len(LOOKAHEADS)))

    all_results = []
    for name, mask in tqdm(layers, desc='Testing layers'):
        n_peaks, results = test_layer(name, mask, feat, LOOKAHEADS)
        row = f'{name:<30} {n_peaks:>8,} {n_peaks/n*100:>6.2f}%'
        for la in LOOKAHEADS:
            acc = results[la]['accuracy']
            row += f'  {acc:>6.2f}%'
        lines.append(row)
        all_results.append((name, n_peaks, results))

    # Comparison: improvement over L0
    lines.append('')
    lines.append('IMPROVEMENT OVER L0 (velocity flip only):')
    lines.append('')
    header2 = f'{"Layer":<30} {"Peaks":>8}'
    for la in LOOKAHEADS:
        header2 += f'  +{la}s edge'
    lines.append(header2)
    lines.append('-' * (42 + 12 * len(LOOKAHEADS)))

    l0_results = all_results[0][2]
    for name, n_peaks, results in all_results:
        row = f'{name:<30} {n_peaks:>8,}'
        for la in LOOKAHEADS:
            edge = results[la]['accuracy'] - l0_results[la]['accuracy']
            row += f'  {edge:>+7.2f}%'
        lines.append(row)

    # Find best layer per lookahead
    lines.append('')
    lines.append('BEST LAYER PER LOOKAHEAD:')
    for la in LOOKAHEADS:
        best_name = ''
        best_acc = 0
        best_n = 0
        for name, n_peaks, results in all_results:
            if n_peaks < 50 and n_peaks < n * 0.0001:
                continue  # too few peaks to be reliable
            acc = results[la]['accuracy']
            if acc > best_acc:
                best_acc = acc
                best_name = name
                best_n = n_peaks
        lines.append(f'  +{la:>3}s: {best_name:<30} {best_acc:.2f}% (n={best_n:,})')

    lines.append('')
    lines.append('INTERPRETATION:')
    lines.append('  If adding a layer INCREASES accuracy: that feature catches real peaks')
    lines.append('  If adding a layer has NO effect: that feature is redundant')
    lines.append('  If adding a layer DECREASES accuracy: that feature is filtering out good peaks')
    lines.append('  Fewer peaks + higher accuracy = sharper detection (quality over quantity)')

    summary = '\n'.join(lines)

    out_path = os.path.join('reports', 'findings', 'peak_prediction_layered.txt')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(summary + '\n')
    print(f'\nSummary: {out_path}')
    print('\n' + summary)


if __name__ == '__main__':
    main()
