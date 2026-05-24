"""Multi-timeframe pivot seed scanner.

Detect pivots at 1s precision, confirm with 1m exhaustion, capture 15s state.

The idea: 1s gives precise entry timing, 1m tells us if the move is actually
done (DMI crossing, volume dying), 15s gives the execution context features.

Usage:
    python tools/pivot_seed_scanner_mtf.py --data DATA/ATLAS_1WEEK
    python tools/pivot_seed_scanner_mtf.py --data DATA/ATLAS --lookback 60 --min-swing 8

Output:
    reports/findings/pivot_seeds_mtf.csv          — all pivots with multi-TF state
    reports/findings/pivot_seeds_mtf_summary.txt  — real vs fakeout analysis
    checkpoints/pivot_seeds_mtf.pkl               — clustered seed library
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


def find_pivots_1s(highs: np.ndarray, lows: np.ndarray,
                   lookback: int = 60) -> list[dict]:
    """Find pivot highs/lows in 1s data.

    lookback=60 means 60s (1 min) each side to confirm pivot.
    This matches the 1m confirmation scale.
    """
    n = len(highs)
    pivots = []

    for i in range(lookback, n - lookback):
        # Pivot high
        left_h = highs[i - lookback:i]
        right_h = highs[i + 1:i + lookback + 1]
        if highs[i] >= left_h.max() and highs[i] >= right_h.max():
            pivots.append({'index': i, 'type': 'HIGH', 'price': float(highs[i])})

        # Pivot low
        left_l = lows[i - lookback:i]
        right_l = lows[i + 1:i + lookback + 1]
        if lows[i] <= left_l.min() and lows[i] <= right_l.min():
            pivots.append({'index': i, 'type': 'LOW', 'price': float(lows[i])})

    pivots.sort(key=lambda p: (p['index'], p['type']))
    return pivots


def match_pivot_trades(pivots: list[dict], tick_size: float = 0.25,
                       point_value: float = 2.0,
                       min_swing_ticks: int = 4) -> list[dict]:
    """Match consecutive opposing pivots into trades."""
    trades = []
    i = 0

    while i < len(pivots) - 1:
        entry = pivots[i]
        j = i + 1
        while j < len(pivots) and pivots[j]['type'] == entry['type']:
            j += 1
        if j >= len(pivots):
            break

        exit_pivot = pivots[j]

        if entry['type'] == 'LOW':
            direction = 'LONG'
            pnl_ticks = (exit_pivot['price'] - entry['price']) / tick_size
        else:
            direction = 'SHORT'
            pnl_ticks = (entry['price'] - exit_pivot['price']) / tick_size

        pnl_dollars = pnl_ticks * tick_size * point_value
        hold_seconds = exit_pivot['index'] - entry['index']  # 1s bars = seconds

        is_fakeout = pnl_ticks < -min_swing_ticks
        is_real = pnl_ticks >= min_swing_ticks
        label = 'REAL' if is_real else ('FAKEOUT' if is_fakeout else 'MARGINAL')

        trades.append({
            'entry_index_1s': entry['index'],
            'exit_index_1s': exit_pivot['index'],
            'entry_price': entry['price'],
            'exit_price': exit_pivot['price'],
            'direction': direction,
            'pnl_ticks': pnl_ticks,
            'pnl_dollars': pnl_dollars,
            'hold_seconds': hold_seconds,
            'label': label,
            'entry_type': entry['type'],
        })

        i = j

    return trades


def lookup_tf_state(ts: float, tf_df: pd.DataFrame, tf_states_map: dict,
                    tf_ts: np.ndarray) -> dict:
    """Find the nearest completed bar in a slower TF and extract its state."""
    if tf_ts is None or len(tf_ts) == 0:
        return {}

    # Find last completed bar at or before timestamp
    idx = np.searchsorted(tf_ts, ts, side='right') - 1
    if idx < 0:
        return {}

    state = tf_states_map.get(idx)
    if state is None:
        return {}

    return {
        'F_momentum': float(getattr(state, 'F_momentum', 0.0)),
        'F_reversion': float(getattr(state, 'mean_reversion_force', 0.0)),
        'z_score': float(getattr(state, 'z_score', 0.0)),
        'sigma': float(getattr(state, 'regression_sigma', 0.0)),
        'velocity': float(getattr(state, 'velocity', 0.0)),
        'hurst': float(getattr(state, 'hurst_exponent', 0.0)),
        'adx': float(getattr(state, 'adx_strength', 0.0)),
        'dmi_plus': float(getattr(state, 'dmi_plus', 0.0)),
        'dmi_minus': float(getattr(state, 'dmi_minus', 0.0)),
        'dmi_diff': float(getattr(state, 'dmi_plus', 0.0) - getattr(state, 'dmi_minus', 0.0)),
        'P_center': float(getattr(state, 'P_at_center', 0.0)),
        'entropy': float(getattr(state, 'entropy_normalized', 0.0)),
        'coherence': float(getattr(state, 'oscillation_entropy_normalized', 0.0)),
        'tunnel_prob': float(getattr(state, 'reversion_probability', 0.0)),
        'term_pid': float(getattr(state, 'term_pid', 0.0)),
        'volume_delta': float(getattr(state, 'volume_delta', 0.0)),
        'net_force': float(getattr(state, 'net_force', 0.0)),
    }


def enrich_mtf(trades: list[dict], df_1s: pd.DataFrame,
               df_15s: pd.DataFrame, states_15s: dict,
               ts_15s: np.ndarray,
               df_1m: pd.DataFrame, states_1m: dict,
               ts_1m: np.ndarray) -> list[dict]:
    """Enrich trades with multi-TF state at entry."""
    ts_1s = df_1s['timestamp'].values.astype(np.float64)

    enriched = []
    for t in trades:
        idx = t['entry_index_1s']
        entry_ts = float(ts_1s[idx]) if idx < len(ts_1s) else 0.0
        t['timestamp'] = entry_ts

        # 15s state (execution TF context)
        s15 = lookup_tf_state(entry_ts, df_15s, states_15s, ts_15s)
        for k, v in s15.items():
            t[f's15_{k}'] = v

        # 1m state (confirmation TF)
        s1m = lookup_tf_state(entry_ts, df_1m, states_1m, ts_1m)
        for k, v in s1m.items():
            t[f's1m_{k}'] = v

        # Cross-TF features (the real discriminators)
        if s15 and s1m:
            # DMI agreement: are 15s and 1m moving in same direction?
            t['dmi_agreement'] = 1.0 if (s15['dmi_diff'] * s1m['dmi_diff']) > 0 else 0.0
            # Volume alignment: are both TFs showing same volume direction?
            t['vol_agreement'] = 1.0 if (s15['volume_delta'] * s1m['volume_delta']) > 0 else 0.0
            # Momentum alignment
            t['fm_agreement'] = 1.0 if (s15['F_momentum'] * s1m['F_momentum']) > 0 else 0.0
            # 1m exhaustion: is 1m momentum fading?
            t['s1m_fm_abs'] = abs(s1m['F_momentum'])
            # 1m volume exhaustion
            t['s1m_vol_abs'] = abs(s1m['volume_delta'])
        else:
            t['dmi_agreement'] = 0.0
            t['vol_agreement'] = 0.0
            t['fm_agreement'] = 0.0
            t['s1m_fm_abs'] = 0.0
            t['s1m_vol_abs'] = 0.0

        enriched.append(t)

    return enriched


def write_summary(trades_df: pd.DataFrame, output_path: str):
    """Write human-readable summary with multi-TF analysis."""
    L = []
    W = 70
    L.append("=" * W)
    L.append("MULTI-TF PIVOT SCANNER -- 1s detect / 1m confirm / 15s context")
    L.append("=" * W)

    total = len(trades_df)
    real = trades_df[trades_df['label'] == 'REAL']
    fake = trades_df[trades_df['label'] == 'FAKEOUT']
    marg = trades_df[trades_df['label'] == 'MARGINAL']

    L.append(f"  Total pivots: {total}")
    L.append(f"  REAL: {len(real)} ({len(real)/total*100:.0f}%) "
             f"avg={real['pnl_ticks'].mean():.1f}t  hold={real['hold_seconds'].mean():.0f}s")
    L.append(f"  FAKEOUT: {len(fake)} ({len(fake)/total*100:.0f}%) "
             f"avg={fake['pnl_ticks'].mean():.1f}t  hold={fake['hold_seconds'].mean():.0f}s")
    L.append(f"  MARGINAL: {len(marg)} ({len(marg)/total*100:.0f}%)")
    L.append("")

    # 1m confirmation features (the money signals)
    L.append("-- 1m CONFIRMATION (move exhaustion) --")
    CONF_FEATS = ['s1m_F_momentum', 's1m_dmi_diff', 's1m_volume_delta',
                  's1m_z_score', 's1m_adx', 's1m_coherence', 's1m_term_pid']
    L.append(f"  {'1m Feature':<22} {'REAL':>10} {'FAKE':>10} {'Delta':>10} {'Signal':>8}")
    L.append(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for feat in CONF_FEATS:
        if feat not in trades_df.columns:
            continue
        r_mean = real[feat].mean() if len(real) > 0 else 0
        f_mean = fake[feat].mean() if len(fake) > 0 else 0
        delta = r_mean - f_mean
        r_std = real[feat].std() if len(real) > 1 else 1
        d = abs(delta) / max(r_std, 1e-6)
        signal = "***" if d > 0.5 else ("**" if d > 0.3 else ("*" if d > 0.1 else ""))
        L.append(f"  {feat:<22} {r_mean:>10.4f} {f_mean:>10.4f} {delta:>+10.4f} {signal:>8}")

    L.append("")

    # 15s execution features
    L.append("-- 15s CONTEXT (entry state) --")
    EXEC_FEATS = ['s15_F_momentum', 's15_dmi_diff', 's15_volume_delta',
                  's15_z_score', 's15_P_center', 's15_coherence']
    L.append(f"  {'15s Feature':<22} {'REAL':>10} {'FAKE':>10} {'Delta':>10} {'Signal':>8}")
    L.append(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for feat in EXEC_FEATS:
        if feat not in trades_df.columns:
            continue
        r_mean = real[feat].mean() if len(real) > 0 else 0
        f_mean = fake[feat].mean() if len(fake) > 0 else 0
        delta = r_mean - f_mean
        r_std = real[feat].std() if len(real) > 1 else 1
        d = abs(delta) / max(r_std, 1e-6)
        signal = "***" if d > 0.5 else ("**" if d > 0.3 else ("*" if d > 0.1 else ""))
        L.append(f"  {feat:<22} {r_mean:>10.4f} {f_mean:>10.4f} {delta:>+10.4f} {signal:>8}")

    L.append("")

    # Cross-TF agreement
    L.append("-- CROSS-TF AGREEMENT --")
    for feat in ['dmi_agreement', 'vol_agreement', 'fm_agreement']:
        if feat not in trades_df.columns:
            continue
        r_mean = real[feat].mean() if len(real) > 0 else 0
        f_mean = fake[feat].mean() if len(fake) > 0 else 0
        L.append(f"  {feat:<22} REAL={r_mean:.2f}  FAKE={f_mean:.2f}")

    L.append("")

    # Direction
    L.append("DIRECTION:")
    for d in ['LONG', 'SHORT']:
        dt = trades_df[trades_df['direction'] == d]
        if len(dt) > 0:
            wr = (dt['pnl_ticks'] > 0).mean() * 100
            real_pct = (dt['label'] == 'REAL').mean() * 100
            L.append(f"  {d}: {len(dt)} trades  WR={wr:.0f}%  real={real_pct:.0f}%")

    L.append("")
    L.append("=" * W)

    text = '\n'.join(L)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text)
    print(f"\nSummary: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Multi-TF pivot seed scanner')
    parser.add_argument('--data', type=str, default='DATA/ATLAS',
                        help='ATLAS data root')
    parser.add_argument('--lookback', type=int, default=60,
                        help='1s bars each side to confirm pivot (default: 60 = 1 min)')
    parser.add_argument('--min-swing', type=int, default=8,
                        help='Min ticks for a real swing (default: 8 = $4)')
    parser.add_argument('--output', type=str, default='checkpoints/pivot_seeds_mtf.pkl',
                        help='Output path for seed library')
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core_v2.statistical_field_engine import StatisticalFieldEngine

    engine = StatisticalFieldEngine()

    # Find matching files across TFs
    dir_1s = os.path.join(args.data, '1s')
    dir_15s = os.path.join(args.data, '15s')
    dir_1m = os.path.join(args.data, '1m')

    files_1s = sorted(glob.glob(os.path.join(dir_1s, '*.parquet')))
    if not files_1s:
        print(f"ERROR: No 1s files in {dir_1s}")
        return 1

    print(f"Found {len(files_1s)} 1s files")
    all_trades = []

    for f_1s in tqdm(files_1s, desc='Scanning pivots (MTF)'):
        # Load 1s
        df_1s = pd.read_parquet(f_1s)
        if df_1s.empty or len(df_1s) < args.lookback * 4:
            continue

        # Find matching 15s and 1m files (same month)
        fname = os.path.basename(f_1s)
        f_15s = os.path.join(dir_15s, fname)
        f_1m = os.path.join(dir_1m, fname)

        df_15s = pd.read_parquet(f_15s) if os.path.exists(f_15s) else pd.DataFrame()
        df_1m = pd.read_parquet(f_1m) if os.path.exists(f_1m) else pd.DataFrame()

        # Find pivots on 1s
        highs = df_1s['high'].values.astype(np.float64)
        lows = df_1s['low'].values.astype(np.float64)
        pivots = find_pivots_1s(highs, lows, lookback=args.lookback)
        if len(pivots) < 2:
            continue

        trades = match_pivot_trades(pivots, min_swing_ticks=args.min_swing)
        if not trades:
            continue

        # Compute states for 15s and 1m
        states_15s_map = {}
        ts_15s = None
        if not df_15s.empty:
            try:
                raw_15s = engine.batch_compute_states(df_15s, use_cuda=True)
                states_15s_map = {s['bar_idx']: s['state'] for s in raw_15s}
                ts_15s = df_15s['timestamp'].values.astype(np.float64)
            except Exception:
                pass

        states_1m_map = {}
        ts_1m = None
        if not df_1m.empty:
            try:
                raw_1m = engine.batch_compute_states(df_1m, use_cuda=True)
                states_1m_map = {s['bar_idx']: s['state'] for s in raw_1m}
                ts_1m = df_1m['timestamp'].values.astype(np.float64)
            except Exception:
                pass

        # Enrich with multi-TF state
        trades = enrich_mtf(trades, df_1s,
                            df_15s, states_15s_map, ts_15s,
                            df_1m, states_1m_map, ts_1m)
        all_trades.extend(trades)

    if not all_trades:
        print("ERROR: No pivot trades found")
        return 1

    trades_df = pd.DataFrame(all_trades)
    print(f"\nTotal pivot trades: {len(trades_df)}")
    print(f"  REAL: {(trades_df['label'] == 'REAL').sum()}")
    print(f"  FAKEOUT: {(trades_df['label'] == 'FAKEOUT').sum()}")
    print(f"  MARGINAL: {(trades_df['label'] == 'MARGINAL').sum()}")

    # Save CSV
    csv_path = 'reports/findings/pivot_seeds_mtf.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    trades_df.to_csv(csv_path, index=False)
    print(f"Raw data: {csv_path}")

    # Summary
    write_summary(trades_df, 'reports/findings/pivot_seeds_mtf_summary.txt')

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
