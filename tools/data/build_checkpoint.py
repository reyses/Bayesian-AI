"""
Build a checkpoint.json at a specific end-date cutoff.

Use case: create a clean warm-start checkpoint for a downstream atlas.
  Example: IS atlas has data through Apr 8, NT8 starts Mar 20.
           Build an IS checkpoint ending Mar 19 → NT8 chains from
           that instead of Apr state.

Process:
  1. Find the last N days of 5s bars in <atlas> ending on --end (inclusive)
  2. Feed them into an Aggregator (just bar aggregation, no features)
  3. Extract velocities from the features parquet for the end day
     (FEATURES_5s/<end>.parquet, last row's *_velocity columns)
  4. Save via agg.save_checkpoint() → <atlas>/checkpoint.json
     (or --out PATH to write elsewhere)

Usage:
    python tools/build_checkpoint.py --atlas DATA/ATLAS --end 2026-03-19
    python tools/build_checkpoint.py --atlas DATA/ATLAS --end 2026-03-19 \
        --out DATA/ATLAS/checkpoint_mar19.json
    python tools/build_checkpoint.py --atlas DATA/ATLAS --end 2026-03-19 \
        --days 20   # 20-day warmup instead of default 10
"""
import os
import sys
import glob
import argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.utils.aggregator import Aggregator


DEFAULT_WARMUP_DAYS = 10


def _day_name_to_iso(d: str) -> str:
    return d.replace('_', '-')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', required=True, help='Atlas root (e.g. DATA/ATLAS)')
    ap.add_argument('--end', required=True, help='End date YYYY-MM-DD (inclusive)')
    ap.add_argument('--days', type=int, default=DEFAULT_WARMUP_DAYS,
                    help='Warmup window in days (default 10)')
    ap.add_argument('--out', default=None, help='Override output path')
    ap.add_argument('--anchor', default='5s', help='Anchor TF (default 5s)')
    args = ap.parse_args()

    # Resolve parquet files for anchor TF
    tf_dir = os.path.join(args.atlas, args.anchor)
    if not os.path.isdir(tf_dir):
        print(f'ERROR: {tf_dir} not found')
        return
    all_files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    # Filter to <= end
    end_iso = args.end
    eligible = [f for f in all_files
                if _day_name_to_iso(os.path.basename(f).replace('.parquet', '')) <= end_iso]
    if not eligible:
        print(f'ERROR: no days <= {end_iso} in {tf_dir}')
        return

    warmup = eligible[-args.days:]
    print(f'Warmup: {len(warmup)} days ({os.path.basename(warmup[0])} -> '
          f'{os.path.basename(warmup[-1])})')

    # Build aggregator from 5s bars
    agg = Aggregator(history_limit=2000)
    total_bars = 0
    for fpath in warmup:
        df = pd.read_parquet(fpath).sort_values('timestamp').reset_index(drop=True)
        for _, row in df.iterrows():
            agg.feed({
                'timestamp': row['timestamp'],
                'open': row['open'], 'high': row['high'],
                'low': row['low'], 'close': row['close'],
                'volume': row.get('volume', 0),
            })
            total_bars += 1

    # Extract velocities from FEATURES_5s at end date (if available)
    end_day_name = args.end.replace('-', '_')
    feat_path = os.path.join(args.atlas, 'FEATURES_5s', f'{end_day_name}.parquet')
    velocities = {}
    if os.path.exists(feat_path):
        fdf = pd.read_parquet(feat_path)
        last = fdf.iloc[-1]
        for tf in ('15s', '1m', '5m', '15m', '1h', '1D'):
            col = f'{tf}_velocity'
            if col in fdf.columns:
                velocities[tf] = float(last[col])
        print(f'Velocities at end of {end_day_name}:')
        for k, v in velocities.items():
            print(f'  {k:>4}: {v:+.4f}')
    else:
        print(f'WARNING: {feat_path} not found — velocities empty')

    # Save checkpoint
    out_path = args.out or os.path.join(args.atlas, 'checkpoint.json')
    agg.save_checkpoint(out_path, velocities=velocities)
    print()
    print(f'Checkpoint written: {out_path}')
    print(f'  Bars fed: {total_bars:,}')
    print(f'  bar_counts: ' + ', '.join(f'{k}={len(v)}' for k, v in agg.history.items() if v))


if __name__ == '__main__':
    main()
