"""Pivot-to-pivot oracle seed scanner.

Finds price pivot highs/lows, measures the swing to the next opposing pivot,
and captures the full market state at each pivot point. Produces two seed
classes: REAL reversals (profitable swings) and FAKEOUTS (failed reversals).

This is an ORACLE tool — it uses N-bar lookahead to confirm pivots.
The output is training data, not a live trading signal.

Usage:
    python tools/pivot_seed_scanner.py --data DATA/ATLAS --tf 15s
    python tools/pivot_seed_scanner.py --data DATA/ATLAS_1WEEK --tf 1m --lookback 5 --min-swing 4

Output:
    reports/findings/pivot_seeds.csv         — all pivots with state + outcome
    reports/findings/pivot_seeds_summary.txt — breakdown of real vs fakeout
    checkpoints/pivot_seeds.pkl              — clustered seed library
"""

import argparse
import glob
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm


def find_pivots(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                lookback: int = 5) -> list[dict]:
    """Find pivot highs and pivot lows in price data.

    A pivot high at bar i: high[i] >= max(high[i-lookback:i]) AND
                           high[i] >= max(high[i+1:i+lookback+1])
    A pivot low at bar i:  low[i] <= min(low[i-lookback:i]) AND
                           low[i] <= min(low[i+1:i+lookback+1])

    Returns list of {index, type ('HIGH'/'LOW'), price}.
    """
    n = len(highs)
    pivots = []

    for i in range(lookback, n - lookback):
        # Check pivot high
        left_highs = highs[i - lookback:i]
        right_highs = highs[i + 1:i + lookback + 1]
        if highs[i] >= left_highs.max() and highs[i] >= right_highs.max():
            pivots.append({
                'index': i,
                'type': 'HIGH',
                'price': float(highs[i]),
            })

        # Check pivot low
        left_lows = lows[i - lookback:i]
        right_lows = lows[i + 1:i + lookback + 1]
        if lows[i] <= left_lows.min() and lows[i] <= right_lows.min():
            pivots.append({
                'index': i,
                'type': 'LOW',
                'price': float(lows[i]),
            })

    # Sort by index, deduplicate (a bar can be both high and low in rare cases)
    pivots.sort(key=lambda p: (p['index'], p['type']))
    return pivots


def match_pivot_trades(pivots: list[dict], tick_size: float = 0.25,
                       point_value: float = 2.0,
                       min_swing_ticks: int = 4) -> list[dict]:
    """Match consecutive opposing pivots into trades.

    Pivot LOW → next Pivot HIGH = LONG trade
    Pivot HIGH → next Pivot LOW = SHORT trade

    Also identifies fakeouts: pivot that gets breached before reaching
    the next opposing pivot.
    """
    trades = []
    i = 0

    while i < len(pivots) - 1:
        entry = pivots[i]

        # Find next opposing pivot
        j = i + 1
        while j < len(pivots) and pivots[j]['type'] == entry['type']:
            j += 1

        if j >= len(pivots):
            break

        exit_pivot = pivots[j]

        # Determine direction
        if entry['type'] == 'LOW':
            direction = 'LONG'
            pnl_ticks = (exit_pivot['price'] - entry['price']) / tick_size
        else:
            direction = 'SHORT'
            pnl_ticks = (entry['price'] - exit_pivot['price']) / tick_size

        pnl_dollars = pnl_ticks * tick_size * point_value
        hold_bars = exit_pivot['index'] - entry['index']

        # Classify: real reversal or fakeout
        is_fakeout = pnl_ticks < -min_swing_ticks  # lost more than min swing
        is_real = pnl_ticks >= min_swing_ticks       # gained at least min swing
        label = 'REAL' if is_real else ('FAKEOUT' if is_fakeout else 'MARGINAL')

        trades.append({
            'entry_index': entry['index'],
            'exit_index': exit_pivot['index'],
            'entry_price': entry['price'],
            'exit_price': exit_pivot['price'],
            'direction': direction,
            'pnl_ticks': pnl_ticks,
            'pnl_dollars': pnl_dollars,
            'hold_bars': hold_bars,
            'label': label,
            'entry_type': entry['type'],
            'exit_type': exit_pivot['type'],
        })

        i = j  # advance to exit pivot (it becomes next entry candidate)

    return trades


def enrich_with_state(trades: list[dict], states_map: dict,
                      df: pd.DataFrame) -> list[dict]:
    """Add market state features at entry to each trade."""
    enriched = []
    for t in trades:
        idx = t['entry_index']
        state = states_map.get(idx)
        row = df.iloc[idx] if idx < len(df) else None

        t['timestamp'] = float(row['timestamp']) if row is not None else 0.0

        if state is not None:
            t['F_momentum'] = float(getattr(state, 'F_momentum', 0.0))
            t['F_reversion'] = float(getattr(state, 'mean_reversion_force', 0.0))
            t['z_score'] = float(getattr(state, 'z_score', 0.0))
            t['sigma'] = float(getattr(state, 'regression_sigma', 0.0))
            t['velocity'] = float(getattr(state, 'velocity', 0.0))
            t['hurst'] = float(getattr(state, 'hurst_exponent', 0.0))
            t['adx'] = float(getattr(state, 'adx_strength', 0.0))
            t['dmi_plus'] = float(getattr(state, 'dmi_plus', 0.0))
            t['dmi_minus'] = float(getattr(state, 'dmi_minus', 0.0))
            t['dmi_diff'] = t['dmi_plus'] - t['dmi_minus']
            t['P_center'] = float(getattr(state, 'P_at_center', 0.0))
            t['entropy'] = float(getattr(state, 'entropy_normalized', 0.0))
            t['coherence'] = float(getattr(state, 'oscillation_entropy_normalized', 0.0))
            t['tunnel_prob'] = float(getattr(state, 'reversion_probability', 0.0))
            t['term_pid'] = float(getattr(state, 'term_pid', 0.0))
            t['volume_delta'] = float(getattr(state, 'volume_delta', 0.0))
            t['net_force'] = float(getattr(state, 'net_force', 0.0))
        else:
            # Fill with zeros
            for feat in ['F_momentum', 'F_reversion', 'z_score', 'sigma',
                         'velocity', 'hurst', 'adx', 'dmi_plus', 'dmi_minus',
                         'dmi_diff', 'P_center', 'entropy', 'coherence',
                         'tunnel_prob', 'term_pid', 'volume_delta', 'net_force']:
                t[feat] = 0.0

        enriched.append(t)
    return enriched


def cluster_seeds(trades_df: pd.DataFrame, n_clusters: int = 10,
                  min_members: int = 5) -> dict:
    """Cluster pivots into seed templates, separately for REAL and FAKEOUT."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    FEAT_COLS = ['F_momentum', 'F_reversion', 'z_score', 'sigma', 'velocity',
                 'hurst', 'adx', 'dmi_diff', 'P_center', 'entropy',
                 'coherence', 'tunnel_prob', 'term_pid', 'volume_delta']

    library = {'real': {}, 'fakeout': {}, 'feature_names': FEAT_COLS}

    for label, group_name in [('REAL', 'real'), ('FAKEOUT', 'fakeout')]:
        subset = trades_df[trades_df['label'] == label]
        if len(subset) < min_members * 2:
            print(f"  {label}: only {len(subset)} trades, skipping clustering")
            continue

        X = subset[FEAT_COLS].fillna(0.0).values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        nc = min(n_clusters, len(X) // max(min_members, 2))
        nc = max(nc, 1)

        km = KMeans(n_clusters=nc, random_state=42, n_init=10)
        labels_arr = km.fit_predict(X_scaled)

        seeds = {}
        for sid in range(nc):
            mask = labels_arr == sid
            cluster = subset.iloc[mask]
            if len(cluster) < min_members:
                continue

            pnls = cluster['pnl_ticks'].values
            seeds[sid] = {
                'centroid': X[mask].mean(axis=0).tolist(),
                'n': len(cluster),
                'wr': float((pnls > 0).mean()),
                'avg_pnl_ticks': float(pnls.mean()),
                'avg_hold_bars': float(cluster['hold_bars'].mean()),
                'long_pct': float((cluster['direction'] == 'LONG').mean()),
                'avg_fm': float(cluster['F_momentum'].mean()),
                'avg_z': float(cluster['z_score'].mean()),
                'avg_vol': float(cluster['volume_delta'].abs().mean()),
            }

        library[group_name] = seeds
        library[f'{group_name}_scaler_mean'] = scaler.mean_.tolist()
        library[f'{group_name}_scaler_std'] = scaler.scale_.tolist()
        print(f"  {label}: {len(seeds)} seeds from {len(subset)} trades")

    return library


def write_summary(trades_df: pd.DataFrame, output_path: str):
    """Write human-readable summary."""
    L = []
    W = 70
    L.append("=" * W)
    L.append("PIVOT SEED SCANNER — Oracle Reversal Analysis")
    L.append("=" * W)

    total = len(trades_df)
    real = trades_df[trades_df['label'] == 'REAL']
    fake = trades_df[trades_df['label'] == 'FAKEOUT']
    marg = trades_df[trades_df['label'] == 'MARGINAL']

    L.append(f"  Total pivots: {total}")
    L.append(f"  REAL reversals: {len(real)} ({len(real)/total*100:.0f}%) "
             f"avg PnL={real['pnl_ticks'].mean():.1f}t  avg hold={real['hold_bars'].mean():.0f} bars")
    L.append(f"  FAKEOUTS:       {len(fake)} ({len(fake)/total*100:.0f}%) "
             f"avg PnL={fake['pnl_ticks'].mean():.1f}t  avg hold={fake['hold_bars'].mean():.0f} bars")
    L.append(f"  MARGINAL:       {len(marg)} ({len(marg)/total*100:.0f}%) "
             f"avg PnL={marg['pnl_ticks'].mean():.1f}t  avg hold={marg['hold_bars'].mean():.0f} bars")
    L.append("")

    # Feature comparison: REAL vs FAKEOUT
    FEAT_COLS = ['F_momentum', 'z_score', 'sigma', 'velocity', 'hurst',
                 'adx', 'dmi_diff', 'P_center', 'entropy', 'coherence',
                 'volume_delta', 'term_pid']

    L.append(f"  {'Feature':<16} {'REAL mean':>10} {'FAKE mean':>10} {'Delta':>10} {'Signal?':>8}")
    L.append(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for feat in FEAT_COLS:
        r_mean = real[feat].mean() if len(real) > 0 else 0
        f_mean = fake[feat].mean() if len(fake) > 0 else 0
        delta = r_mean - f_mean
        r_std = real[feat].std() if len(real) > 1 else 1
        # Cohen's d effect size
        d = abs(delta) / max(r_std, 1e-6)
        signal = "***" if d > 0.5 else ("**" if d > 0.3 else ("*" if d > 0.1 else ""))
        L.append(f"  {feat:<16} {r_mean:>10.4f} {f_mean:>10.4f} {delta:>+10.4f} {signal:>8}")

    L.append("")

    # Direction breakdown
    L.append("DIRECTION BREAKDOWN:")
    for d in ['LONG', 'SHORT']:
        dt = trades_df[trades_df['direction'] == d]
        if len(dt) > 0:
            wr = (dt['pnl_ticks'] > 0).mean() * 100
            avg = dt['pnl_ticks'].mean()
            real_pct = (dt['label'] == 'REAL').mean() * 100
            L.append(f"  {d}: {len(dt)} trades  WR={wr:.0f}%  avg={avg:.1f}t  "
                     f"real={real_pct:.0f}%")

    L.append("")
    L.append("=" * W)

    text = '\n'.join(L)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)
    print(f"\nSummary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Pivot-to-pivot oracle seed scanner')
    parser.add_argument('--data', type=str, default='DATA/ATLAS',
                        help='ATLAS data root (default: DATA/ATLAS)')
    parser.add_argument('--tf', type=str, default='15s',
                        help='Timeframe to scan (default: 15s)')
    parser.add_argument('--lookback', type=int, default=5,
                        help='Bars each side to confirm pivot (default: 5)')
    parser.add_argument('--min-swing', type=int, default=4,
                        help='Minimum ticks for a real swing (default: 4)')
    parser.add_argument('--clusters', type=int, default=10,
                        help='Clusters per label class (default: 10)')
    parser.add_argument('--output', type=str, default='checkpoints/pivot_seeds.pkl',
                        help='Output path for seed library')
    args = parser.parse_args()

    # Find TF data files
    tf_dir = os.path.join(args.data, args.tf)
    if not os.path.isdir(tf_dir):
        print(f"ERROR: {tf_dir} not found")
        return 1

    files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    print(f"Found {len(files)} files in {tf_dir}")

    # Import engine for state computation
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_v2.statistical_field_engine import StatisticalFieldEngine

    engine = StatisticalFieldEngine()
    tick_size = 0.25
    point_value = 2.0

    all_trades = []

    for f_path in tqdm(files, desc='Scanning pivots'):
        df = pd.read_parquet(f_path)
        if df.empty or len(df) < args.lookback * 4:
            continue

        highs = df['high'].values.astype(np.float64)
        lows = df['low'].values.astype(np.float64)
        closes = df['close'].values.astype(np.float64)

        # Find pivots
        pivots = find_pivots(highs, lows, closes, lookback=args.lookback)
        if len(pivots) < 2:
            continue

        # Match into trades
        trades = match_pivot_trades(pivots, tick_size=tick_size,
                                    point_value=point_value,
                                    min_swing_ticks=args.min_swing)
        if not trades:
            continue

        # Compute states for enrichment
        try:
            states_raw = engine.batch_compute_states(df, use_cuda=True)
            states_map = {s['bar_idx']: s['state'] for s in states_raw}
        except Exception:
            states_map = {}

        # Enrich with market state
        trades = enrich_with_state(trades, states_map, df)
        all_trades.extend(trades)

    if not all_trades:
        print("ERROR: No pivot trades found")
        return 1

    trades_df = pd.DataFrame(all_trades)
    print(f"\nTotal pivot trades: {len(trades_df)}")
    print(f"  REAL: {(trades_df['label'] == 'REAL').sum()}")
    print(f"  FAKEOUT: {(trades_df['label'] == 'FAKEOUT').sum()}")
    print(f"  MARGINAL: {(trades_df['label'] == 'MARGINAL').sum()}")

    # Save raw CSV
    csv_path = 'reports/findings/pivot_seeds.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    trades_df.to_csv(csv_path, index=False)
    print(f"Raw data: {csv_path}")

    # Cluster into seeds
    print("\nClustering...")
    library = cluster_seeds(trades_df, n_clusters=args.clusters)

    # Save library
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'wb') as f_out:
        pickle.dump(library, f_out)
    print(f"Seed library: {args.output}")

    # Summary report
    write_summary(trades_df, 'reports/findings/pivot_seeds_summary.txt')

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
