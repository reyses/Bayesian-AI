#!/usr/bin/env python
"""
I-MR -> Seed Converter — convert auto-detected I-MR regimes into seed JSON files.

Uses human-marked seeds as calibration reference to filter I-MR regimes to
trade-quality entries. Generates lookback timestamps from 1m ATLAS data.

Usage:
    python tools/imr_to_seeds.py
    python tools/imr_to_seeds.py --human-seeds DATA/regime_seeds/seeds_2025-01-02_*.json
    python tools/imr_to_seeds.py --min-mfe 50 --max-dur 60 --min-rr 1.5
    python tools/imr_to_seeds.py --no-filter          # convert all I-MR regimes
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TICK_SIZE = 0.25
TICK_VALUE = 0.50
LOOKBACK_BARS = 10


def load_human_seeds(path: str) -> list:
    """Load human-marked seed JSON."""
    with open(path) as f:
        data = json.load(f)
    return data.get('seeds', [])


def compute_human_profile(seeds: list) -> dict:
    """Extract calibration thresholds from human seed data.

    Uses P25 for minimums (quality floor) and P75 for maximums (duration cap).
    """
    mfe = [s['mfe_ticks'] for s in seeds]
    mae = [s['mae_ticks'] for s in seeds]
    dur = [s['duration_mins'] for s in seeds]
    rr = [s['mfe_ticks'] / max(s['mae_ticks'], 1) for s in seeds]

    profile = {
        'n_seeds': len(seeds),
        'min_mfe_ticks': float(np.percentile(mfe, 10)),       # P10 = quality floor
        'max_duration_mins': float(np.percentile(dur, 90)),    # P90 = duration ceiling
        'min_rr': float(np.percentile(rr, 10)),                # P10 = minimum R:R
        'max_mae_ticks': float(np.percentile(mae, 90)),        # P90 = MAE ceiling
        # Distribution stats for reporting
        'mfe_median': float(np.median(mfe)),
        'mae_median': float(np.median(mae)),
        'dur_median': float(np.median(dur)),
        'rr_median': float(np.median(rr)),
    }
    return profile


def load_1m_timestamps(data_dir: str) -> pd.DataFrame:
    """Load 1m ATLAS data for lookback timestamp extraction."""
    from tools.research.data import load_atlas_tf
    df = load_atlas_tf(data_dir, '1m')
    if df.empty:
        print("WARNING: No 1m data found — lookback timestamps will be empty")
    return df


def get_lookback_timestamps(ts_start: float, df_1m: pd.DataFrame,
                            n_bars: int = LOOKBACK_BARS) -> list:
    """Get timestamps of N bars before regime start from 1m data."""
    if df_1m.empty:
        return []

    ts_col = df_1m['timestamp'].values
    # Find the bar at or just before regime start
    mask = ts_col <= ts_start
    valid_idx = np.where(mask)[0]
    if len(valid_idx) == 0:
        return []

    end_idx = valid_idx[-1]
    start_idx = max(0, end_idx - n_bars + 1)
    return ts_col[start_idx:end_idx + 1].tolist()


def convert_regimes(imr_csv: str, df_1m: pd.DataFrame,
                    min_mfe: float = 30, max_dur: float = 120,
                    min_rr: float = 0.5, max_mae: float = 200,
                    no_filter: bool = False) -> dict:
    """Convert I-MR regime CSV to per-day seed dictionaries.

    Returns: {date_str: {'seeds': [...], 'meta': {...}}}
    """
    df = pd.read_csv(imr_csv)
    n_raw = len(df)

    # Fix negative MAE bug (price never pulled back = 0 MAE, not negative)
    neg_mae = (df['mae_ticks_1s'] < 0).sum()
    if neg_mae > 0:
        print(f"  Fixed {neg_mae} negative MAE entries (clamped to 0)")
    df['mae_ticks_1s'] = df['mae_ticks_1s'].clip(lower=0)
    df['mae_dollars_1s'] = df['mae_dollars_1s'].clip(lower=0)

    # Drop FLAT
    n_flat = (df['direction'] == 'FLAT').sum()
    df = df[df['direction'] != 'FLAT'].copy()

    # Apply filters
    if not no_filter:
        mask = (
            (df['mfe_ticks_1s'] >= min_mfe) &
            (df['duration_mins'] <= max_dur) &
            (df['mae_ticks_1s'] <= max_mae)
        )
        # R:R filter (avoid division by zero)
        rr = df['mfe_ticks_1s'] / df['mae_ticks_1s'].clip(lower=1)
        mask = mask & (rr >= min_rr)
        df = df[mask].copy()

    n_filtered = len(df)
    print(f"  Raw: {n_raw} | Flat: {n_flat} | Filtered: {n_filtered} "
          f"({n_filtered / n_raw * 100:.1f}%)")

    # Extract date from start_time
    df['date'] = df['start_time'].str[:10]

    # Group by date and build seed JSONs
    days = {}
    for date_str, group in tqdm(df.groupby('date'), desc='Building seeds',
                                 unit='day', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        seeds = []
        for i, (_, row) in enumerate(group.iterrows()):
            ts_start = float(row['ts_start'])
            ts_end = float(row['ts_end'])
            entry_price = float(row['entry_price'])
            exit_price = float(row['exit_price'])
            mfe_t = float(row['mfe_ticks_1s'])
            mae_t = float(row['mae_ticks_1s'])

            # Lookback timestamps from 1m data
            lookback_ts = get_lookback_timestamps(ts_start, df_1m)

            # Find 1m bar index for start
            if not df_1m.empty:
                ts_col = df_1m['timestamp'].values
                idx_arr = np.where(ts_col <= ts_start)[0]
                start_idx = int(idx_arr[-1]) if len(idx_arr) > 0 else 0
                idx_arr2 = np.where(ts_col <= ts_end)[0]
                end_idx = int(idx_arr2[-1]) if len(idx_arr2) > 0 else start_idx
            else:
                start_idx = 0
                end_idx = 0

            seed = {
                'trade_id': i,
                'regime_id': int(row['regime_id']),
                'direction': row['direction'],
                'start_idx': start_idx,
                'end_idx': end_idx,
                'ts_start': ts_start,
                'ts_end': ts_end,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'change_ticks': float(row['price_change_ticks']),
                'change_dollars': float(row['price_change_dollars']),
                'mfe_ticks': mfe_t,
                'mae_ticks': mae_t,
                'mfe_dollars': mfe_t * TICK_VALUE,
                'mae_dollars': mae_t * TICK_VALUE,
                'duration_mins': float(row['duration_mins']),
                'time_to_mfe_mins': float(row['time_to_mfe_mins']),
                'n_bars': int(row['duration_bars']),
                'lookback_bars': LOOKBACK_BARS,
                'lookback_start_idx': max(0, start_idx - LOOKBACK_BARS),
                'lookback_timestamps': lookback_ts,
                'regime_start_idx': start_idx,
                # Extra I-MR fields
                'volatility': float(row['volatility']),
                'source': 'imr_auto',
            }
            seeds.append(seed)

        days[date_str] = {
            'date': date_str,
            'timeframe': '1m',
            'created': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'n_seeds': len(seeds),
            'source': 'imr_regime_auto',
            'seeds': seeds,
        }

    return days


def print_summary(days: dict, profile: dict = None):
    """Print conversion summary with optional human profile comparison."""
    total_seeds = sum(d['n_seeds'] for d in days.values())
    all_seeds = [s for d in days.values() for s in d['seeds']]
    n_days = len(days)

    print(f"\n{'=' * 70}")
    print(f"  I-MR -> SEED CONVERSION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Days:  {n_days}")
    print(f"  Seeds: {total_seeds} ({total_seeds / n_days:.1f}/day)")

    if all_seeds:
        mfes = [s['mfe_ticks'] for s in all_seeds]
        maes = [s['mae_ticks'] for s in all_seeds]
        durs = [s['duration_mins'] for s in all_seeds]
        rrs = [s['mfe_ticks'] / max(s['mae_ticks'], 1) for s in all_seeds]
        longs = sum(1 for s in all_seeds if s['direction'] == 'LONG')
        shorts = sum(1 for s in all_seeds if s['direction'] == 'SHORT')

        print(f"\n  Direction: LONG={longs} ({longs/total_seeds*100:.0f}%) "
              f"SHORT={shorts} ({shorts/total_seeds*100:.0f}%)")
        print(f"\n  {'Metric':<15s} {'P10':>8s} {'P25':>8s} {'Median':>8s} "
              f"{'P75':>8s} {'P90':>8s} {'Mean':>8s}")
        print(f"  {'-'*63}")
        for name, vals in [('MFE (ticks)', mfes), ('MAE (ticks)', maes),
                           ('Duration (m)', durs), ('R:R', rrs)]:
            print(f"  {name:<15s} {np.percentile(vals,10):>8.1f} "
                  f"{np.percentile(vals,25):>8.1f} {np.median(vals):>8.1f} "
                  f"{np.percentile(vals,75):>8.1f} {np.percentile(vals,90):>8.1f} "
                  f"{np.mean(vals):>8.1f}")

    if profile:
        print(f"\n  Human calibration ({profile['n_seeds']} seeds):")
        print(f"    MFE median: {profile['mfe_median']:.0f}t  "
              f"MAE median: {profile['mae_median']:.0f}t  "
              f"Dur median: {profile['dur_median']:.0f}m  "
              f"R:R median: {profile['rr_median']:.1f}")

    # Per-day distribution
    print(f"\n  Seeds/day distribution:")
    counts = sorted([d['n_seeds'] for d in days.values()])
    print(f"    Min={min(counts)}  P25={np.percentile(counts,25):.0f}  "
          f"Med={np.median(counts):.0f}  P75={np.percentile(counts,75):.0f}  "
          f"Max={max(counts)}")

    # Top 5 / Bottom 5 days
    by_count = sorted(days.items(), key=lambda x: x[1]['n_seeds'], reverse=True)
    print(f"\n  Top 5 days:    ", end='')
    print('  '.join(f"{d}({v['n_seeds']})" for d, v in by_count[:5]))
    print(f"  Bottom 5 days: ", end='')
    print('  '.join(f"{d}({v['n_seeds']})" for d, v in by_count[-5:]))


def save_seeds(days: dict, output_dir: str, mode: str = 'single'):
    """Save seeds to JSON files.

    mode='single': one big JSON with all days
    mode='daily': one JSON per day (matches trade_marker.py format)
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    if mode == 'daily':
        for date_str, day_data in days.items():
            path = os.path.join(output_dir, f'seeds_{date_str}_{ts}.json')
            with open(path, 'w') as f:
                json.dump(day_data, f, indent=2)
        print(f"\n  Saved {len(days)} daily seed files to {output_dir}/")
    else:
        # Single combined file
        combined = {
            'created': ts,
            'source': 'imr_regime_auto',
            'n_days': len(days),
            'n_seeds_total': sum(d['n_seeds'] for d in days.values()),
            'days': days,
        }
        path = os.path.join(output_dir, f'imr_seeds_all_{ts}.json')
        with open(path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"\n  Saved combined seed file: {path}")

    return path


def main():
    parser = argparse.ArgumentParser(description='I-MR -> Seed Converter')
    parser.add_argument('--imr-csv',
                        default='tools/plots/standalone/imr_regimes/regime_segments_full_year.csv',
                        help='I-MR regime CSV path')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root for 1m lookback extraction')
    parser.add_argument('--human-seeds', default=None,
                        help='Human seed JSON for calibration (auto-detects latest if omitted)')
    parser.add_argument('--output-dir', default='DATA/regime_seeds/imr_auto',
                        help='Output directory for seed JSONs')

    # Filter overrides (if human seeds not provided)
    parser.add_argument('--min-mfe', type=float, default=None,
                        help='Min MFE ticks (default: from human seeds or 30)')
    parser.add_argument('--max-dur', type=float, default=None,
                        help='Max duration mins (default: from human seeds or 120)')
    parser.add_argument('--min-rr', type=float, default=None,
                        help='Min R:R ratio (default: from human seeds or 0.5)')
    parser.add_argument('--max-mae', type=float, default=None,
                        help='Max MAE ticks (default: from human seeds or 200)')
    parser.add_argument('--no-filter', action='store_true',
                        help='Skip all filters (convert every regime)')
    parser.add_argument('--daily', action='store_true',
                        help='Save one JSON per day (vs single combined file)')
    args = parser.parse_args()

    print(f"I-MR -> Seed Converter")
    print(f"  I-MR CSV: {args.imr_csv}")

    # ── Human calibration ──
    profile = None
    if args.human_seeds:
        human_path = args.human_seeds
    else:
        # Auto-detect latest human seed file
        seed_dir = 'DATA/regime_seeds'
        candidates = sorted(Path(seed_dir).glob('seeds_*.json'), reverse=True)
        # Skip imr_auto subdirectory
        candidates = [c for c in candidates if 'imr_auto' not in str(c)]
        if candidates:
            human_path = str(candidates[0])
        else:
            human_path = None

    if human_path and os.path.isfile(human_path):
        print(f"  Human seeds: {human_path}")
        human_seeds = load_human_seeds(human_path)
        profile = compute_human_profile(human_seeds)
        print(f"  Human profile ({profile['n_seeds']} seeds): "
              f"MFE>={profile['min_mfe_ticks']:.0f}t, "
              f"dur<={profile['max_duration_mins']:.0f}m, "
              f"R:R>={profile['min_rr']:.1f}, "
              f"MAE<={profile['max_mae_ticks']:.0f}t")
    else:
        print(f"  No human seeds found — using default thresholds")

    # Resolve filter values: CLI override > human profile > defaults
    min_mfe = args.min_mfe or (profile['min_mfe_ticks'] if profile else 30)
    max_dur = args.max_dur or (profile['max_duration_mins'] if profile else 120)
    min_rr = args.min_rr or (profile['min_rr'] if profile else 0.5)
    max_mae = args.max_mae or (profile['max_mae_ticks'] if profile else 200)

    if not args.no_filter:
        print(f"\n  Filters: MFE>={min_mfe:.0f}t  dur<={max_dur:.0f}m  "
              f"R:R>={min_rr:.1f}  MAE<={max_mae:.0f}t")

    # ── Load 1m data for lookback ──
    print(f"\n  Loading 1m ATLAS data for lookback extraction...")
    df_1m = load_1m_timestamps(args.data_dir)
    if not df_1m.empty:
        print(f"  {len(df_1m)} bars loaded")

    # ── Convert ──
    print(f"\n  Converting regimes...")
    days = convert_regimes(
        args.imr_csv, df_1m,
        min_mfe=min_mfe, max_dur=max_dur,
        min_rr=min_rr, max_mae=max_mae,
        no_filter=args.no_filter,
    )

    if not days:
        print("No regimes passed filters.")
        return

    # ── Summary ──
    print_summary(days, profile)

    # ── Save ──
    mode = 'daily' if args.daily else 'single'
    path = save_seeds(days, args.output_dir, mode=mode)

    print(f"\n{'=' * 70}")
    print(f"  Done. {sum(d['n_seeds'] for d in days.values())} seeds across {len(days)} days")
    print(f"  Use with: python training/trainer.py --fresh --seeds {path}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
