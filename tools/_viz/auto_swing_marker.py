#!/usr/bin/env python
"""
Auto Swing Marker -- ZigZag-based swing detector calibrated from human seeds.

Detects every swing (ebb and flow) on 1m price data, measures MFE/MAE from 1s,
and outputs seed JSON files identical to trade_marker.py format.

Calibration from human continuous marking (Jan 3):
  - Minimum reversal: ~20 ticks (P10 of human |change|)
  - Minimum bars between pivots: 3 (P10 of human n_bars)

Usage:
    python tools/auto_swing_marker.py --date 2025-01-06
    python tools/auto_swing_marker.py --all                    # full ATLAS
    python tools/auto_swing_marker.py --date 2025-01-06 --min-reversal 30
    python tools/auto_swing_marker.py --calibrate DATA/regime_seeds/seeds_2025-01-03_*.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.golden_path import load_1s_index, load_1s_window

TICK_SIZE = 0.25
TICK_VALUE = 0.50
LOOKBACK_BARS = 10
SEEDS_DIR = 'DATA/regime_seeds'

# Default calibration (tuned from human seed analysis — target median ~7m duration)
DEFAULT_MIN_REVERSAL = 30   # ticks — minimum price change to trigger a pivot
DEFAULT_MIN_BARS = 5        # minimum bars between consecutive pivots
DEFAULT_MAX_BARS = 15       # maximum bars per swing — chop long trends into bite-size


def calibrate_from_seeds(seed_path: str) -> dict:
    """Extract auto-swing parameters from human seed data."""
    with open(seed_path) as f:
        data = json.load(f)
    seeds = data.get('seeds', [])
    if not seeds:
        return {}

    changes = [abs(s['change_ticks']) for s in seeds]
    nbars = [s['n_bars'] for s in seeds]
    mfes = [s['mfe_ticks'] for s in seeds]

    params = {
        'min_reversal': max(8, int(np.percentile(changes, 10))),
        'min_bars': max(2, int(np.percentile(nbars, 10))),
        'source': seed_path,
        'n_seeds': len(seeds),
        'median_change': float(np.median(changes)),
        'median_mfe': float(np.median(mfes)),
        'median_bars': float(np.median(nbars)),
    }
    return params


def detect_swings(close: np.ndarray, min_reversal: float = DEFAULT_MIN_REVERSAL,
                  min_bars: int = DEFAULT_MIN_BARS,
                  max_bars: int = DEFAULT_MAX_BARS) -> list:
    """Detect swing pivots using ZigZag logic on close prices.

    Tracks running high/low. When price reverses from the extreme by >= min_reversal
    ticks, that extreme becomes a pivot and direction flips.

    max_bars: if a swing exceeds this many bars, force a pivot at the current
    extreme and flip direction. Chops long trends into bite-size swings.
    0 = no cap.

    Returns list of pivot indices where direction changes.
    """
    n = len(close)
    if n < 3:
        return []

    ct = close / TICK_SIZE  # work in tick space

    pivots = [0]  # always start at bar 0

    # Find initial direction: first move of min_reversal from bar 0
    direction = 0  # 0=undecided, 1=up, -1=down
    extreme_idx = 0
    extreme_val = ct[0]

    for i in range(1, n):
        price = ct[i]
        last_pivot = pivots[-1]

        # Duration cap: force pivot at current extreme if swing too long
        # Only chop if the swing actually moved min_reversal ticks — otherwise
        # the trend is still going sideways and forcing a pivot creates a fake swing
        if max_bars > 0 and direction != 0 and i - last_pivot >= max_bars:
            pivot_val = ct[last_pivot]
            swing_move = abs(extreme_val - pivot_val)
            if swing_move >= min_reversal and extreme_idx > last_pivot:
                pivots.append(extreme_idx)
                direction = -direction  # flip
                extreme_val = price
                extreme_idx = i
                continue
            # else: swing hasn't moved enough — let it keep running

        if direction == 0:
            # Track both extremes from start
            if price > extreme_val:
                extreme_val = price
                extreme_idx = i
            if price < ct[0] and ct[0] - price >= min_reversal:
                # First significant down move — start was a high pivot
                direction = -1
                extreme_val = price
                extreme_idx = i
            elif price > ct[0] and price - ct[0] >= min_reversal:
                # First significant up move — start was a low pivot
                direction = 1
                extreme_val = price
                extreme_idx = i

        elif direction == 1:
            # Trending up — track the high
            if price >= extreme_val:
                extreme_val = price
                extreme_idx = i
            elif extreme_val - price >= min_reversal and i - extreme_idx >= min_bars:
                # Reversal down confirmed — the high was a pivot
                pivots.append(extreme_idx)
                direction = -1
                extreme_val = price
                extreme_idx = i

        elif direction == -1:
            # Trending down — track the low
            if price <= extreme_val:
                extreme_val = price
                extreme_idx = i
            elif price - extreme_val >= min_reversal and i - extreme_idx >= min_bars:
                # Reversal up confirmed — the low was a pivot
                pivots.append(extreme_idx)
                direction = 1
                extreme_val = price
                extreme_idx = i

    # Always end at last bar
    if pivots[-1] != n - 1:
        pivots.append(n - 1)

    # Merge pass: absorb tiny swings into their neighbors.
    # If a swing is < min_bars bars AND < min_reversal ticks, remove both its
    # pivots so the surrounding swings merge across it.
    changed = True
    while changed and len(pivots) > 2:
        changed = False
        new_pivots = [pivots[0]]
        i = 1
        while i < len(pivots) - 1:
            si = pivots[i - 1] if i > 0 else pivots[0]
            # Use the last kept pivot as the real start
            si = new_pivots[-1]
            ei = pivots[i]
            ni = pivots[i + 1]
            swing_bars = ei - si
            swing_ticks = abs(ct[ei] - ct[si])
            if swing_bars < min_bars and swing_ticks < min_reversal:
                # Skip this pivot AND the next one — merge across the tiny swing
                # But only skip the next if it's not the last bar
                if i + 2 < len(pivots):
                    i += 2  # drop both pivots of the tiny swing
                    changed = True
                    continue
                else:
                    # Near the end — just skip this one
                    i += 1
                    changed = True
                    continue
            new_pivots.append(pivots[i])
            i += 1
        # Always keep the last pivot
        if new_pivots[-1] != pivots[-1]:
            new_pivots.append(pivots[-1])
        pivots = new_pivots

    return pivots


def measure_1s(index_1s, ts_start, ts_end, direction, cache):
    """Measure MFE/MAE from 1s data."""
    try:
        df_1s = load_1s_window(index_1s, ts_start, ts_end, cache)
        if len(df_1s) < 5:
            return 0.0, 0.0, 0.0

        p1s = df_1s['close'].values.astype(float)
        ts1s = df_1s['timestamp'].values.astype(float)
        entry = p1s[0]

        if direction == 'LONG':
            fav = (p1s - entry) / TICK_SIZE
            adv = (entry - p1s) / TICK_SIZE
        else:
            fav = (entry - p1s) / TICK_SIZE
            adv = (p1s - entry) / TICK_SIZE

        mfe_idx = int(np.argmax(fav))
        mfe = float(fav[mfe_idx])
        mae = float(np.max(adv[:mfe_idx + 1])) if mfe_idx > 0 else 0.0
        mae = max(mae, 0.0)  # clamp negative
        time_to_mfe = float(ts1s[mfe_idx] - ts1s[0]) / 60.0

        return mfe, mae, time_to_mfe
    except Exception:
        return 0.0, 0.0, 0.0


def _extract_bar_physics(state):
    """Extract physics dict from a MarketState object."""
    if state is None:
        return None
    return {
        'fm': round(float(state.F_momentum), 2),
        'z': round(float(state.z_score), 3),
        'dmi_p': round(float(state.dmi_plus), 2),
        'dmi_m': round(float(state.dmi_minus), 2),
        'adx': round(float(state.adx_strength), 2),
        'vel': round(float(state.velocity), 3),
        'vol': round(float(state.volume_delta), 1),
        'hurst': round(float(state.hurst_exponent), 3),
        'P_center': round(float(state.P_at_center), 4),
        'coherence': round(float(getattr(state, 'oscillation_entropy_normalized', 0)), 4),
        'sigma': round(float(state.regression_sigma), 3),
        'pid': round(float(state.term_pid), 4),
        # Additional physics
        'entropy': round(float(state.entropy_normalized), 4),
        'net_force': round(float(state.net_force), 2),
        'mr_force': round(float(state.mean_reversion_force), 3),
        'rev_prob': round(float(state.reversion_probability), 4),
        'P_upper': round(float(state.P_near_upper), 4),
        'P_lower': round(float(state.P_near_lower), 4),
        'dmi_p_prev': round(float(state.di_plus_prev), 2),
        'dmi_m_prev': round(float(state.di_minus_prev), 2),
        'adx_prev': round(float(state.adx_prev), 2),
        'noise': round(float(state.swing_noise_ticks), 1),
    }


def mark_day(df_day, timestamps, close, index_1s, date_str,
             min_reversal, min_bars, max_bars=0, states=None) -> list:
    """Run auto-swing detection on one day, return seed list.

    Args:
        states: optional list of MarketState objects (same length as close).
                When provided, seeds are enriched with full physics for both
                lookback (10 bars before entry) and trade segment (entry to exit).
    """
    pivots = detect_swings(close, min_reversal=min_reversal, min_bars=min_bars,
                           max_bars=max_bars)

    if len(pivots) < 2:
        return []

    cache = {}
    seeds = []

    for j in range(len(pivots) - 1):
        si = pivots[j]
        ei = pivots[j + 1]

        if ei <= si:
            continue

        direction = 'LONG' if close[ei] > close[si] else 'SHORT'
        ts_s = float(timestamps[si])
        ts_e = float(timestamps[ei])

        mfe, mae, time_to_mfe = measure_1s(index_1s, ts_s, ts_e, direction, cache)

        change = close[ei] - close[si]
        change_ticks = change / TICK_SIZE
        duration_mins = (ts_e - ts_s) / 60.0

        seed = {
            'trade_id': len(seeds),
            'direction': direction,
            'start_idx': si,
            'end_idx': ei,
            'ts_start': ts_s,
            'ts_end': ts_e,
            'entry_price': round(float(close[si]), 2),
            'exit_price': round(float(close[ei]), 2),
            'change_ticks': round(change_ticks, 1),
            'change_dollars': round(change_ticks * TICK_VALUE, 2),
            'mfe_ticks': round(mfe, 1),
            'mae_ticks': round(mae, 1),
            'mfe_dollars': round(mfe * TICK_VALUE, 2),
            'mae_dollars': round(mae * TICK_VALUE, 2),
            'duration_mins': round(duration_mins, 1),
            'time_to_mfe_mins': round(time_to_mfe, 1),
            'n_bars': ei - si + 1,
            'lookback_bars': LOOKBACK_BARS,
            'lookback_start_idx': max(0, si - LOOKBACK_BARS),
            'lookback_timestamps': [float(timestamps[k])
                                    for k in range(max(0, si - LOOKBACK_BARS), si)],
            'regime_start_idx': si,
            'source': 'auto_swing',
        }

        # Enrich with physics if states available
        if states is not None:
            # Lookback: 10 bars before entry (the approach)
            lb_start = max(0, si - LOOKBACK_BARS)
            lookback_physics = []
            for k in range(lb_start, si):
                if k < len(states) and states[k] is not None:
                    ph = _extract_bar_physics(states[k])
                    if ph is not None:
                        ph['close'] = round(float(close[k]), 2)
                        ph['ts'] = float(timestamps[k])
                        lookback_physics.append(ph)
            seed['lookback'] = lookback_physics

            # Trade segment: entry to exit
            trade_physics = []
            for k in range(si, min(ei + 1, len(states))):
                if states[k] is not None:
                    ph = _extract_bar_physics(states[k])
                    if ph is not None:
                        ph['close'] = round(float(close[k]), 2)
                        ph['ts'] = float(timestamps[k])
                        trade_physics.append(ph)
            seed['trade'] = trade_physics

            # Entry state summary (for quick filtering)
            if si < len(states) and states[si] is not None:
                es = states[si]
                seed['entry_fm'] = round(float(es.F_momentum), 2)
                seed['entry_dmi_diff'] = round(float(es.dmi_plus - es.dmi_minus), 2)
                seed['entry_vol'] = round(float(es.volume_delta), 1)
                seed['entry_adx'] = round(float(es.adx_strength), 2)
                seed['entry_z'] = round(float(es.z_score), 3)

            # Exit state summary
            if ei < len(states) and states[ei] is not None:
                xs = states[ei]
                seed['exit_fm'] = round(float(xs.F_momentum), 2)
                seed['exit_dmi_diff'] = round(float(xs.dmi_plus - xs.dmi_minus), 2)
                seed['exit_vol'] = round(float(xs.volume_delta), 1)
                seed['exit_adx'] = round(float(xs.adx_strength), 2)

        seeds.append(seed)

    return seeds


def compare_with_human(auto_seeds, human_path, date_str):
    """Compare auto-detected swings with human seeds for validation."""
    with open(human_path) as f:
        human = json.load(f)
    h_seeds = human.get('seeds', [])

    if not h_seeds or not auto_seeds:
        return

    h_mfe = [s['mfe_ticks'] for s in h_seeds]
    a_mfe = [s['mfe_ticks'] for s in auto_seeds]
    h_dur = [s['duration_mins'] for s in h_seeds]
    a_dur = [s['duration_mins'] for s in auto_seeds]
    h_change = [abs(s['change_ticks']) for s in h_seeds]
    a_change = [abs(s['change_ticks']) for s in auto_seeds]

    print(f"\n  === VALIDATION: Auto vs Human ({date_str}) ===")
    print(f"  {'':20s} {'Human':>10s} {'Auto':>10s}")
    print(f"  {'Seeds':20s} {len(h_seeds):>10d} {len(auto_seeds):>10d}")
    print(f"  {'Median MFE (t)':20s} {np.median(h_mfe):>10.0f} {np.median(a_mfe):>10.0f}")
    print(f"  {'Median duration (m)':20s} {np.median(h_dur):>10.0f} {np.median(a_dur):>10.0f}")
    print(f"  {'Median |change| (t)':20s} {np.median(h_change):>10.0f} {np.median(a_change):>10.0f}")
    print(f"  {'Total MFE ($)':20s} {sum(s['mfe_dollars'] for s in h_seeds):>10.0f} "
          f"{sum(s['mfe_dollars'] for s in auto_seeds):>10.0f}")

    h_long = sum(1 for s in h_seeds if s['direction'] == 'LONG')
    a_long = sum(1 for s in auto_seeds if s['direction'] == 'LONG')
    print(f"  {'LONG %':20s} {h_long/len(h_seeds)*100:>9.0f}% {a_long/len(auto_seeds)*100:>9.0f}%")


def main():
    parser = argparse.ArgumentParser(description='Auto Swing Marker (ZigZag)')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--date', default=None,
                        help='Single day (YYYY-MM-DD)')
    parser.add_argument('--all', action='store_true',
                        help='Process all trading days in ATLAS')
    parser.add_argument('--start', default=None,
                        help='Start date for --all (YYYY-MM-DD)')
    parser.add_argument('--end', default=None,
                        help='End date for --all (YYYY-MM-DD)')
    parser.add_argument('--min-reversal', type=float, default=None,
                        help=f'Min reversal ticks (default: {DEFAULT_MIN_REVERSAL})')
    parser.add_argument('--min-bars', type=int, default=None,
                        help=f'Min bars between pivots (default: {DEFAULT_MIN_BARS})')
    parser.add_argument('--max-bars', type=int, default=None,
                        help=f'Max bars per swing — chop long trends (default: {DEFAULT_MAX_BARS}, 0=off)')
    parser.add_argument('--calibrate', default=None,
                        help='Human seed JSON for parameter calibration')
    parser.add_argument('--output-dir', default=SEEDS_DIR,
                        help='Output directory')
    parser.add_argument('--validate', default=None,
                        help='Human seed JSON for comparison (single day)')
    args = parser.parse_args()

    print("Auto Swing Marker")

    # Calibration
    if args.calibrate:
        params = calibrate_from_seeds(args.calibrate)
        print(f"  Calibrated from {params['n_seeds']} human seeds:")
        print(f"    min_reversal={params['min_reversal']}t, min_bars={params['min_bars']}")
        print(f"    Human median: {params['median_change']:.0f}t change, "
              f"{params['median_mfe']:.0f}t MFE, {params['median_bars']:.0f} bars")
        min_reversal = args.min_reversal or params['min_reversal']
        min_bars = args.min_bars or params['min_bars']
    else:
        min_reversal = args.min_reversal or DEFAULT_MIN_REVERSAL
        min_bars = args.min_bars or DEFAULT_MIN_BARS

    max_bars = args.max_bars if args.max_bars is not None else DEFAULT_MAX_BARS
    print(f"  Parameters: min_reversal={min_reversal}t, min_bars={min_bars}, max_bars={max_bars}")

    # Load data
    print(f"\n  Loading 1m data...")
    df_1m = load_atlas_tf(args.data_dir, '1m')
    if df_1m.empty:
        print("ERROR: No 1m data")
        sys.exit(1)
    print(f"  {len(df_1m)} bars")

    print(f"  Loading 1s index...")
    index_1s = load_1s_index(args.data_dir)

    timestamps_all = df_1m['timestamp'].values.astype(float)
    close_all = df_1m['close'].values.astype(float)

    # Determine days to process
    if args.date:
        dates = [args.date]
    elif args.all:
        # Get all unique trading days
        day_strs = sorted(set(
            datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y-%m-%d')
            for t in timestamps_all
        ))
        if args.start:
            day_strs = [d for d in day_strs if d >= args.start]
        if args.end:
            day_strs = [d for d in day_strs if d <= args.end]
        dates = day_strs
    else:
        print("ERROR: Specify --date or --all")
        sys.exit(1)

    # Compute physics states for enrichment
    print(f"  Computing 1m physics states...")
    from core_v2.statistical_field_engine import StatisticalFieldEngine
    _sfe = StatisticalFieldEngine()
    _raw_states = _sfe.batch_compute_states(df_1m)
    _states_all = [s['state'] if s and isinstance(s, dict) and 'state' in s else None
                   for s in _raw_states]
    print(f"  States: {len(_states_all)}")

    print(f"\n  Processing {len(dates)} days...")

    all_seeds = {}
    total_seeds = 0

    for date_str in tqdm(dates, desc='Marking days', unit='day',
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
        dt = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        t_start = dt.timestamp()
        t_end = t_start + 86400

        mask = (timestamps_all >= t_start) & (timestamps_all < t_end)
        idx = np.where(mask)[0]
        if len(idx) < 10:
            continue

        close_day = close_all[idx]
        ts_day = timestamps_all[idx]

        states_day = [_states_all[i] for i in idx]
        seeds = mark_day(df_1m.iloc[idx], ts_day, close_day, index_1s,
                         date_str, min_reversal, min_bars, max_bars,
                         states=states_day)

        if seeds:
            all_seeds[date_str] = {
                'date': date_str,
                'timeframe': '1m',
                'created': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'n_seeds': len(seeds),
                'source': 'auto_swing',
                'params': {'min_reversal': min_reversal, 'min_bars': min_bars, 'max_bars': max_bars},
                'seeds': seeds,
            }
            total_seeds += len(seeds)

    # Summary
    print(f"\n{'='*60}")
    print(f"  AUTO SWING MARKER RESULTS")
    print(f"{'='*60}")
    print(f"  Days processed: {len(dates)}")
    print(f"  Days with seeds: {len(all_seeds)}")
    print(f"  Total seeds: {total_seeds}")
    if all_seeds:
        print(f"  Avg seeds/day: {total_seeds / len(all_seeds):.1f}")

        all_s = [s for d in all_seeds.values() for s in d['seeds']]
        mfes = [s['mfe_ticks'] for s in all_s]
        durs = [s['duration_mins'] for s in all_s]
        changes = [abs(s['change_ticks']) for s in all_s]
        longs = sum(1 for s in all_s if s['direction'] == 'LONG')

        print(f"\n  Median MFE: {np.median(mfes):.0f}t (${np.median(mfes)*TICK_VALUE:.0f})")
        print(f"  Median duration: {np.median(durs):.0f}m")
        print(f"  Median |change|: {np.median(changes):.0f}t")
        print(f"  Direction: {longs} LONG ({longs/len(all_s)*100:.0f}%), "
              f"{len(all_s)-longs} SHORT ({(len(all_s)-longs)/len(all_s)*100:.0f}%)")

    # Save
    if all_seeds:
        os.makedirs(args.output_dir, exist_ok=True)
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')

        if len(dates) == 1:
            # Single day — flat format
            date_str = dates[0]
            out_path = os.path.join(args.output_dir,
                                    f'auto_seeds_{date_str}_{ts_tag}.json')
            with open(out_path, 'w') as f:
                json.dump(all_seeds[date_str], f, indent=2)
        else:
            # Multi-day — combined format
            out_path = os.path.join(args.output_dir,
                                    f'auto_seeds_all_{ts_tag}.json')
            combined = {
                'created': ts_tag,
                'source': 'auto_swing',
                'params': {'min_reversal': min_reversal, 'min_bars': min_bars, 'max_bars': max_bars},
                'n_days': len(all_seeds),
                'n_seeds_total': total_seeds,
                'days': all_seeds,
            }
            with open(out_path, 'w') as f:
                json.dump(combined, f, indent=2)

        print(f"\n  Saved: {out_path}")

    # Validate against human seeds if provided
    if args.validate and args.date:
        compare_with_human(
            all_seeds.get(args.date, {}).get('seeds', []),
            args.validate, args.date
        )


if __name__ == '__main__':
    main()
