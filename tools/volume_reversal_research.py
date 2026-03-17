"""
Volume Reversal Pattern Research
================================
Three questions:
  1. Can we detect compression → spike → collapse sequence before reversals?
  2. Do fake reversions (FOMO) have weaker volume than real reversions (brick wall)?
  3. What does ADX look like as a function of volume?

Uses I-MR auto seeds as ground truth regime boundaries (real reversals).
Compares volume patterns at real reversal points vs non-reversal points.

Usage:
    python tools/volume_reversal_research.py --data DATA/ATLAS --month 2025_06
"""
import argparse
import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_tf_data(data_dir, tf, month=None):
    tf_dir = os.path.join(data_dir, tf)
    files = sorted(Path(tf_dir).glob('*.parquet'))
    if month:
        files = [f for f in files if month in f.stem]
    return pd.concat([pd.read_parquet(f) for f in files]).sort_values('timestamp').reset_index(drop=True)


def detect_volume_sequence(volumes, window=10, spike_mult=2.0, collapse_pct=0.5):
    """Detect compression → spike → collapse patterns in volume series.

    Returns list of (bar_index, compression_bars, spike_magnitude, collapse_depth)
    """
    n = len(volumes)
    if n < window + 5:
        return []

    patterns = []
    avg_vol = np.mean(volumes[:window])  # rolling baseline

    for i in range(window, n - 3):
        # Update rolling average
        avg_vol = np.mean(volumes[max(0, i - window):i])
        if avg_vol <= 0:
            continue

        # Phase 1: Compression — volume declining for N bars
        lookback = volumes[max(0, i - 5):i]
        if len(lookback) < 3:
            continue
        declining = all(lookback[j] <= lookback[j-1] * 1.1 for j in range(1, len(lookback)))
        compression = np.mean(lookback) < avg_vol * 0.7  # below 70% of average

        if not (declining or compression):
            continue

        # Phase 2: Spike — current bar or next bar is > spike_mult × average
        spike = False
        spike_bar = i
        spike_mag = 0
        for j in range(i, min(i + 3, n)):
            if volumes[j] > avg_vol * spike_mult:
                spike = True
                spike_bar = j
                spike_mag = volumes[j] / avg_vol
                break

        if not spike:
            continue

        # Phase 3: Collapse — volume after spike drops below collapse_pct of spike
        if spike_bar + 3 >= n:
            continue
        post_spike = volumes[spike_bar + 1: spike_bar + 4]
        post_avg = np.mean(post_spike)
        collapse = post_avg < volumes[spike_bar] * collapse_pct

        if collapse:
            patterns.append({
                'bar_idx': spike_bar,
                'compression_vol': np.mean(lookback),
                'spike_vol': volumes[spike_bar],
                'spike_magnitude': spike_mag,
                'post_vol': post_avg,
                'collapse_depth': post_avg / volumes[spike_bar],
            })

    return patterns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='DATA/ATLAS')
    parser.add_argument('--month', default='2025_06')
    args = parser.parse_args()

    print("=" * 70)
    print("  VOLUME REVERSAL PATTERN RESEARCH")
    print(f"  Data: {args.data} (month={args.month})")
    print("=" * 70)

    # Load 1m data (good balance of volume granularity and bar count)
    print("\n[1] Loading 1m data...")
    df_1m = load_tf_data(args.data, '1m', args.month)
    print(f"  {len(df_1m):,} 1m bars")

    # Also load 15s for price outcome measurement
    print("  Loading 15s data...")
    df_15s = load_tf_data(args.data, '15s', args.month)
    ts_15s = df_15s['timestamp'].values
    px_15s = df_15s['close'].values
    print(f"  {len(df_15s):,} 15s bars")

    # Compute states for ADX
    print("\n[2] Computing 1m market states...")
    from core.statistical_field_engine import StatisticalFieldEngine
    engine = StatisticalFieldEngine(regression_period=21, use_gpu=True)
    states = engine.batch_compute_states(df_1m, use_cuda=True)
    print(f"  {len(states):,} states")

    volumes = df_1m['volume'].values
    closes = df_1m['close'].values
    timestamps = df_1m['timestamp'].values
    adx_arr = np.array([s['state'].adx_strength for s in states])
    dmi_p = np.array([s['state'].dmi_plus for s in states])
    dmi_m = np.array([s['state'].dmi_minus for s in states])
    z_arr = np.array([s['state'].z_score for s in states])

    # =====================================================================
    # QUESTION 1: Compression → Spike → Collapse detection
    # =====================================================================
    print("\n[3] QUESTION 1: Compression-Spike-Collapse Sequences")

    patterns = detect_volume_sequence(volumes, window=10, spike_mult=2.0, collapse_pct=0.5)
    print(f"  Found {len(patterns)} C-S-C patterns")

    if patterns:
        # For each pattern: did price reverse within 10 bars?
        reversal_count = 0
        continuation_count = 0
        reversal_pnl = []
        continuation_pnl = []

        for p in patterns:
            idx = p['bar_idx']
            if idx + 10 >= len(closes) or idx < 5:
                continue

            # Price direction before spike
            pre_dir = closes[idx] - closes[idx - 5]
            # Price direction after spike
            post_dir = closes[min(idx + 10, len(closes)-1)] - closes[idx]

            # Reversal = post direction opposite to pre direction
            if pre_dir * post_dir < 0:
                reversal_count += 1
                # Measure reversal magnitude on 15s
                ts = timestamps[idx]
                idx_15s = np.searchsorted(ts_15s, ts)
                if idx_15s + 40 < len(px_15s):
                    move = abs(px_15s[idx_15s + 40] - px_15s[idx_15s]) / 0.25
                    reversal_pnl.append(move)
            else:
                continuation_count += 1
                ts = timestamps[idx]
                idx_15s = np.searchsorted(ts_15s, ts)
                if idx_15s + 40 < len(px_15s):
                    move = abs(px_15s[idx_15s + 40] - px_15s[idx_15s]) / 0.25
                    continuation_pnl.append(move)

        total = reversal_count + continuation_count
        print(f"  Reversal after C-S-C: {reversal_count}/{total} ({reversal_count/max(1,total)*100:.1f}%)")
        print(f"  Continuation after C-S-C: {continuation_count}/{total} ({continuation_count/max(1,total)*100:.1f}%)")
        if reversal_pnl:
            print(f"  Avg reversal magnitude: {np.mean(reversal_pnl):.1f} ticks")
        if continuation_pnl:
            print(f"  Avg continuation magnitude: {np.mean(continuation_pnl):.1f} ticks")

    # =====================================================================
    # QUESTION 2: Fake reversion (FOMO) vs real reversion (brick wall)
    # =====================================================================
    print("\n[4] QUESTION 2: Fake vs Real Reversals (Volume at DMI Cross)")

    # Find all DMI crossover points
    crosses = []
    for i in range(1, len(dmi_p)):
        prev_long = dmi_p[i-1] > dmi_m[i-1]
        curr_long = dmi_p[i] > dmi_m[i]
        if prev_long != curr_long:
            gap = abs(dmi_p[i] - dmi_m[i])
            direction = 'LONG' if curr_long else 'SHORT'

            # Was this reversal sustained (real) or temporary (fake)?
            if i + 20 < len(dmi_p):
                still_same_5 = (dmi_p[i+5] > dmi_m[i+5]) == curr_long
                still_same_20 = (dmi_p[i+20] > dmi_m[i+20]) == curr_long
                real = still_same_5 and still_same_20
            else:
                real = True

            # Volume at the cross
            vol_at = volumes[i]
            vol_before = np.mean(volumes[max(0, i-5):i]) if i > 0 else vol_at
            vol_after = np.mean(volumes[i+1:i+6]) if i + 5 < len(volumes) else vol_at

            crosses.append({
                'bar_idx': i,
                'direction': direction,
                'gap': gap,
                'real': real,
                'vol_at_cross': vol_at,
                'vol_before': vol_before,
                'vol_after': vol_after,
                'vol_ratio': vol_at / max(1, vol_before),
                'adx_at': adx_arr[i],
                'z_at': z_arr[i],
            })

    cdf = pd.DataFrame(crosses)
    real = cdf[cdf['real'] == True]
    fake = cdf[cdf['real'] == False]

    print(f"  Total DMI crosses: {len(cdf)}")
    print(f"  Real (sustained 20 bars): {len(real)} ({len(real)/len(cdf)*100:.1f}%)")
    print(f"  Fake (reversed within 20): {len(fake)} ({len(fake)/len(cdf)*100:.1f}%)")

    print(f"\n  {'Metric':25s}  {'Real med':>10s}  {'Fake med':>10s}  {'Gap':>10s}")
    print(f"  {'-'*60}")
    for col in ['vol_at_cross', 'vol_before', 'vol_after', 'vol_ratio', 'gap', 'adx_at']:
        r = real[col].median()
        f_val = fake[col].median()
        print(f"  {col:25s}  {r:>10.2f}  {f_val:>10.2f}  {r-f_val:>+10.2f}")

    # Volume ratio quartiles
    print(f"\n  VOLUME RATIO AT CROSS (vol_at / vol_before):")
    try:
        cdf['vrq'] = pd.qcut(cdf['vol_ratio'], 4, labels=['Q1(low)', 'Q2', 'Q3', 'Q4(high)'],
                              duplicates='drop')
        for q in ['Q1(low)', 'Q2', 'Q3', 'Q4(high)']:
            sub = cdf[cdf['vrq'] == q]
            if len(sub) == 0: continue
            real_pct = sub['real'].mean() * 100
            print(f"    {q:12s}  n={len(sub):>4d}  real={real_pct:.1f}%  avg_adx={sub['adx_at'].mean():.1f}")
    except Exception as e:
        print(f"    Quartile error: {e}")

    # =====================================================================
    # QUESTION 3: ADX as a function of volume
    # =====================================================================
    print("\n[5] QUESTION 3: ADX vs Volume Relationship")

    # Bin by volume quartile, show ADX distribution
    vol_df = pd.DataFrame({
        'volume': volumes[:len(adx_arr)],
        'adx': adx_arr,
        'z': z_arr,
    })
    vol_df = vol_df[vol_df['volume'] > 0]

    try:
        vol_df['vol_q'] = pd.qcut(vol_df['volume'], 5,
                                   labels=['P20(low)', 'P40', 'P60', 'P80', 'P100(high)'],
                                   duplicates='drop')
        print(f"\n  {'Volume Pct':15s}  {'Avg Volume':>12s}  {'Avg ADX':>10s}  {'Avg |z|':>10s}  {'ADX>30%':>10s}")
        print(f"  {'-'*60}")
        for q in ['P20(low)', 'P40', 'P60', 'P80', 'P100(high)']:
            sub = vol_df[vol_df['vol_q'] == q]
            if len(sub) == 0: continue
            avg_vol = sub['volume'].mean()
            avg_adx = sub['adx'].mean()
            avg_z = sub['z'].abs().mean()
            strong = (sub['adx'] > 30).mean() * 100
            print(f"  {q:15s}  {avg_vol:>12.0f}  {avg_adx:>10.1f}  {avg_z:>10.3f}  {strong:>9.1f}%")
    except Exception as e:
        print(f"  Quartile error: {e}")

    # Correlation
    from scipy import stats
    corr, pval = stats.spearmanr(vol_df['volume'], vol_df['adx'])
    print(f"\n  Spearman correlation (volume vs ADX): r={corr:+.4f}  p={pval:.6f}")

    corr2, pval2 = stats.spearmanr(vol_df['volume'], vol_df['z'].abs())
    print(f"  Spearman correlation (volume vs |z|): r={corr2:+.4f}  p={pval2:.6f}")

    # Volume at Z-score zones
    print(f"\n  VOLUME BY Z-SCORE ZONE:")
    for lo, hi, tag in [(-0.5, 0.5, 'Center (PID trance)'),
                         (0.5, 1.5, 'Approach zone'),
                         (1.5, 2.5, 'Roche limit'),
                         (2.5, 99, 'Event horizon')]:
        sub = vol_df[(vol_df['z'].abs() >= lo) & (vol_df['z'].abs() < hi)]
        if len(sub) < 10: continue
        print(f"    {tag:25s}  n={len(sub):>5d}  avg_vol={sub['volume'].mean():>8.0f}  "
              f"avg_adx={sub['adx'].mean():.1f}")

    # Save report
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f'reports/findings/volume_reversal_{ts}.txt'
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(f"Volume Reversal Research -- {datetime.now()}\n")
        f.write(f"See terminal output for full results\n")
    print(f"\n  Report saved: {report_path}")


if __name__ == '__main__':
    main()
