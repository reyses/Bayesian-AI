"""
AdvanceEngine V2 Training Pipeline — Grounded Template System.

Phases:
  1. Extract 70D grounded features from multi-TF ATLAS data
  2. K-Means clustering into templates
  3. Label templates with trend/peak seed outcomes (full lookahead)
  4. Generate per-template configs (SL/TP/direction/hold)
  5. Validate on OOS bar-by-bar (no lookahead)

Usage:
    python -m training.advance_v2_trainer --phase 1         # extract features
    python -m training.advance_v2_trainer --phase 2         # cluster
    python -m training.advance_v2_trainer --phase 3         # label with seeds
    python -m training.advance_v2_trainer --phase all       # full pipeline
    python -m training.advance_v2_trainer --phase validate  # OOS only
"""
import argparse
import glob
import json
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm


# Paths
IS_ROOT = 'DATA/ATLAS'
OOS_ROOT = 'DATA/ATLAS_OOS'
CHECKPOINT_DIR = 'checkpoints/advance_v2'
SEED_DIR = 'DATA/regime_seeds'


def load_tf_data(root: str, tf: str) -> pd.DataFrame:
    """Load all parquet files for a given TF."""
    pattern = os.path.join(root, tf, '*.parquet')
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return df


def phase1_extract_features(data_root: str, output_path: str):
    """
    Phase 1: Extract 70D grounded features for every 1m bar.

    For each 1m bar, look up the corresponding state from each TF
    (latest completed bar at that TF before the 1m timestamp).
    """
    from core.statistical_field_engine import StatisticalFieldEngine
    from core.grounded_features import (
        GroundedFeatureExtractor, TEMPLATE_TFS, N_FEATURES, FEATURE_NAMES
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load 1m as the base TF (iteration TF)
    print("Loading 1m data...")
    df_1m = load_tf_data(data_root, '1m')
    print(f"  1m bars: {len(df_1m):,}")

    # Load all needed TFs and compute SFE states
    sfe = StatisticalFieldEngine()
    tf_states = {}
    tf_dfs = {}

    for tf in TEMPLATE_TFS:
        print(f"Loading {tf}...")
        df_tf = load_tf_data(data_root, tf)
        if df_tf.empty:
            print(f"  {tf}: NO DATA, skipping")
            continue
        print(f"  {tf}: {len(df_tf):,} bars, computing states...")
        states = sfe.batch_compute_states(df_tf, use_cuda=True)
        tf_states[tf] = states
        tf_dfs[tf] = df_tf
        print(f"  {tf}: {len(states)} states ready")

    # For each 1m bar, find the latest state from each TF
    print(f"\nExtracting {N_FEATURES}D features for {len(df_1m):,} bars...")
    extractor = GroundedFeatureExtractor()

    features = np.zeros((len(df_1m), N_FEATURES), dtype=np.float32)
    timestamps = df_1m['timestamp'].values
    prices_1m = df_1m['close'].values
    volumes_1m = df_1m['volume'].values if 'volume' in df_1m.columns else np.zeros(len(df_1m))

    # Build timestamp-sorted index for each TF for fast lookup
    tf_ts = {}
    for tf in TEMPLATE_TFS:
        if tf in tf_dfs:
            tf_ts[tf] = tf_dfs[tf]['timestamp'].values

    # Track latest index per TF (avoid re-searching from start)
    tf_idx = {tf: 0 for tf in TEMPLATE_TFS}

    for i in tqdm(range(len(df_1m)), desc="Extracting features"):
        ts = timestamps[i]
        states_by_tf = {}
        prices_by_tf = {}
        volumes_by_tf = {}

        for tf in TEMPLATE_TFS:
            if tf not in tf_states:
                continue

            # Find latest completed bar at this TF <= current 1m timestamp
            ts_arr = tf_ts[tf]
            idx = tf_idx[tf]

            # Advance index to latest bar before current timestamp
            while idx < len(ts_arr) - 1 and ts_arr[idx + 1] <= ts:
                idx += 1
            tf_idx[tf] = idx

            if idx < len(tf_states[tf]):
                state_entry = tf_states[tf][idx]
                st = state_entry['state'] if isinstance(state_entry, dict) else state_entry
                states_by_tf[tf] = st
                prices_by_tf[tf] = float(tf_dfs[tf].iloc[idx]['close'])
                if 'volume' in tf_dfs[tf].columns:
                    volumes_by_tf[tf] = float(tf_dfs[tf].iloc[idx]['volume'])

        features[i] = extractor.extract(states_by_tf, prices_by_tf, volumes_by_tf)

    # Save
    np.save(output_path, features)
    # Also save metadata
    meta = {
        'n_bars': len(df_1m),
        'n_features': N_FEATURES,
        'feature_names': FEATURE_NAMES,
        'tfs': TEMPLATE_TFS,
        'data_root': data_root,
        'timestamp_range': [float(timestamps[0]), float(timestamps[-1])],
    }
    meta_path = output_path.replace('.npy', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved: {output_path} ({features.shape})")
    print(f"Meta: {meta_path}")
    return features


def phase2_cluster(features_path: str, n_templates: int = 400):
    """Phase 2: K-Means clustering of 70D features into templates."""
    from core.template_matcher import TemplateMatcher

    features = np.load(features_path)
    print(f"Loaded features: {features.shape}")

    matcher = TemplateMatcher(n_templates=n_templates)
    matcher.fit(features)

    save_path = os.path.join(CHECKPOINT_DIR, 'templates')
    matcher.save(save_path)
    return matcher


def phase3_label_templates(features_path: str, data_root: str):
    """
    Phase 3: Label each template with outcomes using full lookahead.

    For each bar, look ahead 1-20 bars and compute:
    - Best direction (LONG or SHORT)
    - Best PnL (max favorable excursion)
    - Optimal hold time
    """
    from core.template_matcher import TemplateMatcher

    features = np.load(features_path)
    df_1m = load_tf_data(data_root, '1m')
    prices = df_1m['close'].values
    highs = df_1m['high'].values
    lows = df_1m['low'].values

    print(f"Labeling {len(features):,} bars with lookahead outcomes...")

    # For each bar, compute the optimal trade (full lookahead)
    TICK = 0.25
    MAX_LOOK = 20  # look ahead up to 20 bars
    outcomes = {}

    for i in tqdm(range(len(features) - MAX_LOOK), desc="Computing outcomes"):
        best_long_pnl = 0
        best_short_pnl = 0
        best_long_hold = 0
        best_short_hold = 0

        for j in range(1, MAX_LOOK + 1):
            # LONG: entry at close[i], check MFE using highs
            long_pnl = (highs[i + j] - prices[i]) / TICK
            if long_pnl > best_long_pnl:
                best_long_pnl = long_pnl
                best_long_hold = j

            # SHORT: entry at close[i], check MFE using lows
            short_pnl = (prices[i] - lows[i + j]) / TICK
            if short_pnl > best_short_pnl:
                best_short_pnl = short_pnl
                best_short_hold = j

        # Best direction
        if best_long_pnl > best_short_pnl:
            direction = 'LONG'
            pnl = best_long_pnl
            hold = best_long_hold
        else:
            direction = 'SHORT'
            pnl = best_short_pnl
            hold = best_short_hold

        outcomes[i] = {
            'direction': direction,
            'pnl_ticks': float(pnl),
            'hold_bars': int(hold),
            'long_pnl': float(best_long_pnl),
            'short_pnl': float(best_short_pnl),
        }

    # Save outcomes
    outcomes_path = os.path.join(CHECKPOINT_DIR, 'outcomes.json')
    with open(outcomes_path, 'w') as f:
        json.dump(outcomes, f)
    print(f"Saved {len(outcomes):,} outcomes to {outcomes_path}")

    # Re-fit templates WITH outcomes
    matcher = TemplateMatcher(n_templates=400)
    matcher.fit(features[:len(features) - MAX_LOOK], outcomes)

    save_path = os.path.join(CHECKPOINT_DIR, 'templates_labeled')
    matcher.save(save_path)
    print(f"Labeled templates saved to {save_path}")

    # Summary
    profitable = sum(1 for c in matcher.configs.values() if c.avg_pnl_ticks > 0)
    high_conf = sum(1 for c in matcher.configs.values() if c.confidence > 0.2)
    print(f"\nTemplate summary:")
    print(f"  Profitable: {profitable}/{len(matcher.configs)}")
    print(f"  High confidence (>0.2): {high_conf}/{len(matcher.configs)}")

    # Top 10 templates
    top = sorted(matcher.configs.values(), key=lambda c: c.avg_pnl_ticks, reverse=True)[:10]
    print(f"\n  Top 10 templates:")
    for c in top:
        print(f"    T{c.template_id:>3}: dir={c.direction:<5} wr={c.win_rate:.0%} "
              f"pnl={c.avg_pnl_ticks:+.1f}t n={c.n_samples} hold={c.hold_bars}b")

    return matcher


def phase_validate(data_root: str = OOS_ROOT):
    """
    Phase 5: OOS validation — bar-by-bar, no lookahead.
    """
    from core.statistical_field_engine import StatisticalFieldEngine
    from core.grounded_features import GroundedFeatureExtractor, TEMPLATE_TFS
    from core.template_matcher import TemplateMatcher

    # Load templates
    matcher = TemplateMatcher()
    matcher.load(os.path.join(CHECKPOINT_DIR, 'templates_labeled'))

    # Load OOS 1m data
    df_1m = load_tf_data(data_root, '1m')
    print(f"OOS 1m bars: {len(df_1m):,}")

    # Load all TFs
    sfe = StatisticalFieldEngine()
    tf_states = {}
    tf_dfs = {}
    for tf in TEMPLATE_TFS:
        df_tf = load_tf_data(data_root, tf)
        if df_tf.empty:
            continue
        states = sfe.batch_compute_states(df_tf, use_cuda=True)
        tf_states[tf] = states
        tf_dfs[tf] = df_tf

    # Extract features bar-by-bar
    extractor = GroundedFeatureExtractor()
    tf_ts = {tf: tf_dfs[tf]['timestamp'].values for tf in tf_dfs}
    tf_idx = {tf: 0 for tf in TEMPLATE_TFS}

    prices = df_1m['close'].values
    highs = df_1m['high'].values
    lows = df_1m['low'].values
    TICK = 0.25
    TP = 10
    SL = 40

    trades = []
    in_trade = False
    trade_dir = ''
    entry_price = 0
    entry_bar = 0
    tp_count = 0
    last_tp_price = 0

    for i in tqdm(range(len(df_1m)), desc="OOS validation"):
        ts = df_1m.iloc[i]['timestamp']
        price = prices[i]
        high = highs[i]
        low = lows[i]

        # Build multi-TF state
        states_by_tf = {}
        prices_by_tf = {}
        volumes_by_tf = {}
        for tf in TEMPLATE_TFS:
            if tf not in tf_states:
                continue
            ts_arr = tf_ts[tf]
            idx = tf_idx[tf]
            while idx < len(ts_arr) - 1 and ts_arr[idx + 1] <= ts:
                idx += 1
            tf_idx[tf] = idx
            if idx < len(tf_states[tf]):
                state_entry = tf_states[tf][idx]
                st = state_entry['state'] if isinstance(state_entry, dict) else state_entry
                states_by_tf[tf] = st
                prices_by_tf[tf] = float(tf_dfs[tf].iloc[idx]['close'])
                if 'volume' in tf_dfs[tf].columns:
                    volumes_by_tf[tf] = float(tf_dfs[tf].iloc[idx]['volume'])

        feat = extractor.extract(states_by_tf, prices_by_tf, volumes_by_tf)

        # SL/TP check
        if in_trade:
            ref = last_tp_price if tp_count > 0 else entry_price
            if trade_dir == 'LONG':
                if (low - entry_price) / TICK <= -SL:
                    total_pnl = -SL + tp_count * TP
                    trades.append({'pnl': total_pnl, 'tps': tp_count, 'exit': 'SL',
                                   'held': i - entry_bar, 'tid': last_tid})
                    in_trade = False
                    continue
                if (high - ref) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref + TP * TICK
            else:
                if (entry_price - high) / TICK <= -SL:
                    total_pnl = -SL + tp_count * TP
                    trades.append({'pnl': total_pnl, 'tps': tp_count, 'exit': 'SL',
                                   'held': i - entry_bar, 'tid': last_tid})
                    in_trade = False
                    continue
                if (ref - low) / TICK >= TP:
                    tp_count += 1
                    last_tp_price = ref - TP * TICK

        # Match template
        result = matcher.match(feat, bar_index=i, timestamp=ts, price=price)

        if result.template_id < 0 or not result.config or not result.config.active:
            continue
        if result.confidence < 0.1:
            continue
        if not result.direction or result.direction == 'BOTH':
            continue

        new_dir = result.direction
        last_tid = result.template_id

        # FLIP
        if in_trade and new_dir != trade_dir:
            pnl = (price - entry_price) / TICK if trade_dir == 'LONG' else (entry_price - price) / TICK
            trades.append({'pnl': pnl, 'tps': tp_count, 'exit': 'FLIP',
                           'held': i - entry_bar, 'tid': last_tid})
            trade_dir = new_dir
            entry_price = price
            entry_bar = i
            tp_count = 0
            last_tp_price = 0
        elif not in_trade:
            in_trade = True
            trade_dir = new_dir
            entry_price = price
            entry_bar = i
            tp_count = 0
            last_tp_price = 0

    # Results
    if not trades:
        print("No trades generated!")
        return

    tdf = pd.DataFrame(trades)
    total_pnl = tdf['pnl'].sum()
    winners = tdf[tdf['pnl'] > 0]
    n = len(tdf)
    trading_days = len(df_1m['timestamp'].apply(
        lambda t: pd.Timestamp(t, unit='s').date()).unique())

    print(f"\n{'='*60}")
    print(f"OOS VALIDATION — AdvanceEngine V2")
    print(f"{'='*60}")
    print(f"  Trades: {n}")
    print(f"  WR: {len(winners)/n*100:.1f}%")
    print(f"  PnL: {total_pnl:,.0f}t (${total_pnl*0.5:,.2f})")
    print(f"  $/day: ${total_pnl*0.5/trading_days:.2f}")
    print(f"  Trading days: {trading_days}")

    # Save markers
    markers_path = os.path.join(CHECKPOINT_DIR, 'oos_markers.csv')
    matcher.save_markers(markers_path)

    # Save results
    results_path = os.path.join(CHECKPOINT_DIR, 'oos_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'trades': n,
            'wr': len(winners) / n,
            'pnl_ticks': float(total_pnl),
            'pnl_dollars': float(total_pnl * 0.5),
            'per_day': float(total_pnl * 0.5 / trading_days),
            'trading_days': trading_days,
        }, f, indent=2)
    print(f"\nResults: {results_path}")


def main():
    parser = argparse.ArgumentParser(description='AdvanceEngine V2 Training')
    parser.add_argument('--phase', default='all', help='1, 2, 3, validate, or all')
    parser.add_argument('--data', default=IS_ROOT, help='ATLAS root for IS')
    parser.add_argument('--oos', default=OOS_ROOT, help='ATLAS root for OOS')
    parser.add_argument('--templates', type=int, default=400, help='Number of templates')
    args = parser.parse_args()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    features_path = os.path.join(CHECKPOINT_DIR, 'features_70d.npy')

    if args.phase in ('1', 'all'):
        print(f"\n{'='*60}\nPHASE 1: Extract 70D grounded features\n{'='*60}")
        phase1_extract_features(args.data, features_path)

    if args.phase in ('2', 'all'):
        print(f"\n{'='*60}\nPHASE 2: K-Means clustering\n{'='*60}")
        phase2_cluster(features_path, n_templates=args.templates)

    if args.phase in ('3', 'all'):
        print(f"\n{'='*60}\nPHASE 3: Label templates with lookahead outcomes\n{'='*60}")
        phase3_label_templates(features_path, args.data)

    if args.phase in ('validate', 'all'):
        print(f"\n{'='*60}\nVALIDATION: OOS bar-by-bar\n{'='*60}")
        phase_validate(args.oos)


if __name__ == '__main__':
    main()
