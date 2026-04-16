"""
Live Parity Check — build features from live bars and compare against
live-computed features, then run the blended pipeline for trade parity.

Step 1: Merge ATLAS_NT8 + live bar chunks → ATLAS_PARITY (temporary)
Step 2: Aggregate 5s → all higher TFs
Step 3: Build features → FEATURES_PARITY_5s
Step 4: Feature parity — compare FEATURES_PARITY_5s vs FEATURES_LIVE_5s
Step 5: Trade parity — run blended engine on FEATURES_PARITY_5s, compare
        against live trade log

Usage:
    python tools/live_parity_check.py                # full check
    python tools/live_parity_check.py --features     # feature parity only
    python tools/live_parity_check.py --trades       # trade parity only (assumes features built)
"""
import os
import sys
import glob
import shutil
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS_NT8 = 'DATA/ATLAS_NT8'
ATLAS_LIVE_CHUNKS = 'DATA/ATLAS_LIVE/5s/_chunks'
FEATURES_LIVE = 'DATA/FEATURES_LIVE_5s'
ATLAS_PARITY = 'DATA/ATLAS_PARITY'
FEATURES_PARITY = 'DATA/FEATURES_PARITY_5s'
LIVE_TRADE_LOG = 'reports/live'

TF_SECONDS = {
    '15s': 15, '30s': 30, '1m': 60, '5m': 300,
    '15m': 900, '1h': 3600, '1D': 86400,
}


def step1_build_atlas():
    """Build ATLAS_PARITY from live bar chunks only (no NT8 merge).

    Copies ATLAS_NT8 history for warmup context (prior days), then
    overwrites the live session day(s) with ONLY the live bar data.
    This ensures parity features are computed from the exact same
    bars the live engine received.
    """
    print('STEP 1: Build ATLAS_PARITY from live bars')

    chunk_files = sorted(glob.glob(os.path.join(ATLAS_LIVE_CHUNKS, '*.parquet')))
    if not chunk_files:
        print('  No live chunks found — nothing to build')
        return

    # Load live chunks and find the earliest live timestamp
    live_dfs = {}
    for cf in chunk_files:
        day_name = os.path.basename(cf).split('_0000')[0]
        df = pd.read_parquet(cf)
        df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
        live_dfs[day_name] = df

    all_live_bars = pd.concat(live_dfs.values(), ignore_index=True)
    live_start_ts = float(all_live_bars['timestamp'].min())
    live_days = set(live_dfs.keys())
    print(f'  Live days: {sorted(live_days)}')
    print(f'  Live start: ts={live_start_ts:.0f}')

    # Copy ATLAS_NT8 for warmup context (prior days stay as-is)
    src_5s = os.path.join(ATLAS_NT8, '5s')
    dst_5s = os.path.join(ATLAS_PARITY, '5s')
    os.makedirs(dst_5s, exist_ok=True)

    copied = 0
    for f in sorted(glob.glob(os.path.join(src_5s, '*.parquet'))):
        day = os.path.basename(f).replace('.parquet', '')
        if day not in live_days:
            shutil.copy2(f, dst_5s)
            copied += 1
    print(f'  Copied {copied} NT8 warmup parquets (prior days)')

    # For live days: NT8 bars BEFORE live start + live bars FROM live start
    # This matches what the live LFE had: ATLAS_NT8 warmup + live bars
    for day_name, live_df in live_dfs.items():
        nt8_path = os.path.join(src_5s, f'{day_name}.parquet')
        if os.path.exists(nt8_path):
            nt8_df = pd.read_parquet(nt8_path)
            # NT8 bars strictly before live start (warmup context)
            nt8_before = nt8_df[nt8_df['timestamp'] < live_start_ts]
            combined = pd.concat([nt8_before, live_df], ignore_index=True)
            combined = combined.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)
            print(f'  {day_name}: {len(nt8_before)} NT8 warmup + {len(live_df)} live = {len(combined)} total')
        else:
            combined = live_df
            print(f'  {day_name}: {len(live_df)} live bars (no NT8 warmup for this day)')

        dst_path = os.path.join(dst_5s, f'{day_name}.parquet')
        combined.to_parquet(dst_path, index=False)

    print(f'  ATLAS_PARITY built — NT8 before live start + live bars after')


def step2_aggregate():
    """Aggregate 5s bars to all higher TFs in ATLAS_PARITY."""
    print('\nSTEP 2: Aggregate 5s → higher TFs')

    src_5s = os.path.join(ATLAS_PARITY, '5s')
    all_5s = sorted(glob.glob(os.path.join(src_5s, '*.parquet')))
    df_5s = pd.concat([pd.read_parquet(f) for f in all_5s], ignore_index=True)
    df_5s = df_5s.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    for tf_name, tf_seconds in TF_SECONDS.items():
        tf_dir = os.path.join(ATLAS_PARITY, tf_name)
        os.makedirs(tf_dir, exist_ok=True)

        df = df_5s.copy()
        df['_tf_ts'] = (df['timestamp'] // tf_seconds) * tf_seconds
        df['_day'] = pd.to_datetime(df['timestamp'], unit='s', utc=True).dt.strftime('%Y_%m_%d')

        tf_bars = df.groupby(['_day', '_tf_ts']).agg(
            timestamp=('_tf_ts', 'first'),
            open=('open', 'first'),
            high=('high', 'max'),
            low=('low', 'min'),
            close=('close', 'last'),
            volume=('volume', 'sum'),
        ).reset_index(drop=True)
        tf_bars['_day'] = pd.to_datetime(tf_bars['timestamp'], unit='s', utc=True).dt.strftime('%Y_%m_%d')

        written = 0
        for day_name, day_df in tf_bars.groupby('_day'):
            out_df = day_df.drop(columns=['_day']).sort_values('timestamp').reset_index(drop=True)
            out_df['timestamp'] = out_df['timestamp'].astype(np.int64)
            out_path = os.path.join(tf_dir, f'{day_name}.parquet')
            out_df.to_parquet(out_path, index=False)
            written += 1

        print(f'  {tf_name}: {written} days, {len(tf_bars)} bars')


def step3_build_features():
    """Build features from ATLAS_PARITY → FEATURES_PARITY_5s."""
    print('\nSTEP 3: Build features')

    os.makedirs(FEATURES_PARITY, exist_ok=True)

    # Find the live session days
    live_files = sorted(glob.glob(os.path.join(FEATURES_LIVE, '*.parquet')))
    if not live_files:
        print('  No live features to compare against')
        return

    live_days = [os.path.basename(f).replace('.parquet', '') for f in live_files]
    # Build features for live days + 1 day before (for warmup context)
    start_date = min(live_days)
    print(f'  Building from {start_date} for days: {live_days}')

    import subprocess
    # --atlas DATA/ATLAS_PARITY auto-derives output to DATA/FEATURES_PARITY_5s
    result = subprocess.run(
        [sys.executable, 'training/build_dataset.py',
         '--resolution', '5s', '--atlas', ATLAS_PARITY,
         '--start', start_date.replace('_', '-')],
        capture_output=True, text=True, timeout=600)

    if result.returncode == 0:
        print(f'  Features built')
        # Show last few lines of output
        for line in result.stdout.strip().split('\n')[-5:]:
            print(f'    {line}')
    else:
        print(f'  Build failed: {result.stderr[-300:]}')


def step4_feature_parity():
    """Compare FEATURES_PARITY_5s vs FEATURES_LIVE_5s."""
    print('\nSTEP 4: Feature Parity Check')
    print('=' * 60)

    live_files = sorted(glob.glob(os.path.join(FEATURES_LIVE, '*.parquet')))
    if not live_files:
        print('  No live features')
        return

    from core.features import TF_ORDER

    for lf in live_files:
        day = os.path.basename(lf)
        pf = os.path.join(FEATURES_PARITY, day)
        if not os.path.exists(pf):
            print(f'  {day}: no parity features (skipped)')
            continue

        live = pd.read_parquet(lf)
        batch = pd.read_parquet(pf)

        overlap_ts = sorted(set(batch['timestamp'].values) & set(live['timestamp'].values))
        if not overlap_ts:
            print(f'  {day}: no overlap ({len(batch)} batch, {len(live)} live)')
            continue

        b = batch[batch['timestamp'].isin(overlap_ts)].sort_values('timestamp').reset_index(drop=True)
        l = live[live['timestamp'].isin(overlap_ts)].sort_values('timestamp').reset_index(drop=True)

        feat_cols = [c for c in b.columns if c != 'timestamp']
        total_cells = len(b) * len(feat_cols)
        exact = 0
        close = 0

        for col in feat_cols:
            bv = b[col].values.astype(np.float32)
            lv = l[col].values.astype(np.float32)
            diff = np.abs(bv - lv)
            exact += (diff == 0).sum()
            close += (diff < 1e-4).sum()

        pct = close / total_cells * 100
        print(f'  {day}: {len(overlap_ts)} bars | {pct:.1f}% parity ({close:,}/{total_cells:,})')

        # Per-TF breakdown
        for tf in TF_ORDER:
            tf_cols = [c for c in feat_cols if c.startswith(tf + '_')]
            tf_close = 0
            tf_total = 0
            for col in tf_cols:
                bv = b[col].values.astype(np.float32)
                lv = l[col].values.astype(np.float32)
                tf_close += (np.abs(bv - lv) < 1e-4).sum()
                tf_total += len(bv)
            tf_pct = tf_close / tf_total * 100 if tf_total > 0 else 0
            print(f'    {tf:>4}: {tf_pct:>6.1f}%')


def step5_trade_parity():
    """Run blended engine on FEATURES_PARITY_5s and compare vs live trades."""
    print('\nSTEP 5: Trade Parity Check')
    print('=' * 60)

    parity_files = sorted(glob.glob(os.path.join(FEATURES_PARITY, '*.parquet')))
    if not parity_files:
        print('  No parity features — run with --features first')
        return

    from training.sfe_ticker import FeatureTicker
    from training.nightmare_blended import BlendedEngine
    from core.ledger import Ledger
    from core import sim_executor

    engine = BlendedEngine(use_cnn=False, live_mode=True)
    ledger = Ledger()

    all_trades = []
    for fpath in parity_files:
        day_name = os.path.basename(fpath).replace('.parquet', '')
        # Price file from ATLAS_PARITY
        price_file = os.path.join(ATLAS_PARITY, '1m', f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        ledger.clear()
        ft = FeatureTicker(fpath, price_file=price_file)
        day_trades = sim_executor.run(ledger, engine, ft, eod_close=True)
        day_trades = sim_executor.adapt_trades(day_trades)
        for t in day_trades:
            t['day'] = day_name
        all_trades.extend(day_trades)

        day_pnl = sum(t['pnl'] for t in day_trades)
        print(f'  {day_name}: {len(day_trades)} trades, ${day_pnl:+,.0f}')

    if not all_trades:
        print('  No trades produced')
        return

    total_pnl = sum(t['pnl'] for t in all_trades)
    primaries = [t for t in all_trades if not t.get('is_chain', False)]
    chains = [t for t in all_trades if t.get('is_chain', False)]
    print(f'\n  PARITY TRADES: {len(primaries)} primary + {len(chains)} chain')
    print(f'  Total PnL: ${total_pnl:+,.0f}')

    # Compare against live trade log
    session_date = os.path.basename(parity_files[-1]).replace('.parquet', '')
    live_log_path = os.path.join(LIVE_TRADE_LOG, f'v2_trades_{session_date}.csv')
    if os.path.exists(live_log_path):
        live_log = pd.read_csv(live_log_path)
        live_entries = live_log[live_log['type'].str.contains('ENTRY', na=False)]
        live_exits = live_log[live_log['type'].str.contains('EXIT|FLATTEN', na=False)]
        print(f'\n  LIVE TRADES: {len(live_entries)} entries, {len(live_exits)} exits')

        # Tier comparison
        parity_tiers = Counter(t.get('entry_tier', '?') for t in primaries)
        live_tiers = Counter(live_entries['tier'].values)
        print(f'\n  {"Tier":<20} {"Parity":>8} {"Live":>8}')
        print(f'  {"-"*40}')
        all_tiers = sorted(set(list(parity_tiers.keys()) + list(live_tiers.keys())))
        for tier in all_tiers:
            print(f'  {tier:<20} {parity_tiers.get(tier, 0):>8} {live_tiers.get(tier, 0):>8}')
    else:
        print(f'\n  No live trade log at {live_log_path}')


def main():
    parser = argparse.ArgumentParser(description='Live Parity Check')
    parser.add_argument('--features', action='store_true', help='Feature parity only')
    parser.add_argument('--trades', action='store_true', help='Trade parity only')
    args = parser.parse_args()

    if args.trades:
        step5_trade_parity()
        return

    # Clean previous parity data
    if os.path.exists(ATLAS_PARITY):
        shutil.rmtree(ATLAS_PARITY)
    if os.path.exists(FEATURES_PARITY):
        shutil.rmtree(FEATURES_PARITY)

    step1_build_atlas()
    step2_aggregate()
    step3_build_features()
    step4_feature_parity()

    if not args.features:
        step5_trade_parity()


if __name__ == '__main__':
    main()
