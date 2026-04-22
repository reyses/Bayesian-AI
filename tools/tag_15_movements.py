"""
Tag every $15 movement opportunity in the raw 5s price data.

For each 5s bar i, looks forward up to MAX_BARS (default 96 = 8 min) and
checks:
  - LONG:  did HIGH reach close[i] + 7.5 points within window?
  - SHORT: did LOW reach close[i] - 7.5 points within window?

A bar is tagged if EITHER direction fires. For each tagged bar we record:
  - Direction (LONG / SHORT / BOTH — which side hit first)
  - Bars-to-target
  - Starting price

This gives us the universe of "$15 in 8 min" opportunities independent
of any entry logic. From this we can measure:
  - Density per day: how many opportunities exist?
  - Time-of-day distribution
  - Bars-to-target distribution (how quickly does $15 happen?)
  - Correlation with what days produce what kind of opportunities

$15 is configurable. 7.5 points is the math: MNQ = $2/point × 7.5 = $15.
8 min default matches the user's "saturation" frame.

Usage:
    python tools/tag_15_movements.py                    # all IS days
    python tools/tag_15_movements.py --day 2025_04_09   # single day
    python tools/tag_15_movements.py --target 20        # $20 moves
    python tools/tag_15_movements.py --timeout 5        # 5 min window

Output: reports/findings/tag_movements_<target>_<timeout>.md
        training_iso/output/trades/movements_<target>_<timeout>.pkl
"""
import os
import sys
import glob
import pickle
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


ATLAS_5S_DIR = 'DATA/ATLAS/5s'
OUT_PKL_TEMPLATE = 'training_iso/output/trades/movements_${target}_{timeout}m.pkl'
OUT_MD_TEMPLATE  = 'reports/findings/tag_movements_${target}_{timeout}m.md'

DOLLAR_PER_POINT = 2.0  # MNQ: $2 per 1.0 point
BARS_PER_MINUTE = 12    # 5s bars


def tag_day(df, target_dollars, timeout_bars):
    """Return list of tagged events per bar.

    Each event has:
        bar_idx, timestamp, start_price,
        long_hit_bar, long_hit_seconds,
        short_hit_bar, short_hit_seconds,
        first_direction  ('LONG' / 'SHORT' / 'BOTH' / 'NEITHER'),
        bars_to_first
    """
    target_pts = target_dollars / DOLLAR_PER_POINT
    n = len(df)
    if n < 2:
        return []
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    ts = df['timestamp'].values

    events = []
    for i in range(n - 1):
        start = closes[i]
        long_tgt = start + target_pts
        short_tgt = start - target_pts
        window_end = min(i + 1 + timeout_bars, n)

        long_hit = None
        short_hit = None
        for j in range(i + 1, window_end):
            if long_hit is None and highs[j] >= long_tgt:
                long_hit = j - i
            if short_hit is None and lows[j] <= short_tgt:
                short_hit = j - i
            if long_hit is not None and short_hit is not None:
                break

        if long_hit is None and short_hit is None:
            first = 'NEITHER'
            bars_first = None
        elif long_hit is None:
            first = 'SHORT'
            bars_first = short_hit
        elif short_hit is None:
            first = 'LONG'
            bars_first = long_hit
        elif long_hit < short_hit:
            first = 'LONG'  # or BOTH_LONG_FIRST
            bars_first = long_hit
        elif short_hit < long_hit:
            first = 'SHORT'
            bars_first = short_hit
        else:
            first = 'BOTH'   # tied (same bar both sides)
            bars_first = long_hit

        events.append({
            'bar_idx': i,
            'timestamp': int(ts[i]),
            'start_price': float(start),
            'long_hit_bar': long_hit,
            'long_hit_sec': long_hit * 5 if long_hit else None,
            'short_hit_bar': short_hit,
            'short_hit_sec': short_hit * 5 if short_hit else None,
            'first_direction': first,
            'bars_to_first': bars_first,
        })
    return events


def day_summary(events, day_name):
    if not events:
        return None
    total = len(events)
    long_any = sum(1 for e in events if e['long_hit_bar'] is not None)
    short_any = sum(1 for e in events if e['short_hit_bar'] is not None)
    both = sum(1 for e in events
               if e['long_hit_bar'] is not None
               and e['short_hit_bar'] is not None)
    neither = sum(1 for e in events if e['first_direction'] == 'NEITHER')
    long_first = sum(1 for e in events if e['first_direction'] == 'LONG')
    short_first = sum(1 for e in events if e['first_direction'] == 'SHORT')
    tied = sum(1 for e in events if e['first_direction'] == 'BOTH')

    # Bars-to-target distribution (for FIRST hits)
    first_bars = [e['bars_to_first'] for e in events
                  if e['bars_to_first'] is not None]
    if first_bars:
        arr = np.asarray(first_bars)
        b_mean = float(arr.mean())
        b_median = float(np.median(arr))
        b_p10 = float(np.percentile(arr, 10))
        b_p90 = float(np.percentile(arr, 90))
    else:
        b_mean = b_median = b_p10 = b_p90 = None

    return {
        'day': day_name,
        'total_bars': total,
        'long_any': long_any,
        'short_any': short_any,
        'both': both,
        'neither': neither,
        'long_first': long_first,
        'short_first': short_first,
        'tied': tied,
        'any_hit': long_any + short_any - both,   # union (not double-count)
        'bars_to_first_mean': b_mean,
        'bars_to_first_median': b_median,
        'bars_to_first_p10': b_p10,
        'bars_to_first_p90': b_p90,
    }


def hourly_breakdown(events):
    """Return {hour_of_day_utc: {long_first, short_first, neither, total}}."""
    hourly = {}
    for e in events:
        t = datetime.fromtimestamp(e['timestamp'], tz=timezone.utc)
        hr = t.hour
        if hr not in hourly:
            hourly[hr] = {'long_first': 0, 'short_first': 0,
                          'neither': 0, 'total': 0, 'both': 0}
        hourly[hr]['total'] += 1
        if e['first_direction'] == 'LONG':
            hourly[hr]['long_first'] += 1
        elif e['first_direction'] == 'SHORT':
            hourly[hr]['short_first'] += 1
        elif e['first_direction'] == 'BOTH':
            hourly[hr]['both'] += 1
        else:
            hourly[hr]['neither'] += 1
    return hourly


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None,
                    help='Specific day stem like 2025_04_09 (default: all IS days)')
    ap.add_argument('--target', type=float, default=15.0,
                    help='Target $ move (default 15)')
    ap.add_argument('--timeout', type=float, default=8.0,
                    help='Max lookahead in minutes (default 8)')
    ap.add_argument('--max-days', type=int, default=None,
                    help='Cap N days (debug)')
    args = ap.parse_args()

    timeout_bars = int(round(args.timeout * BARS_PER_MINUTE))

    if args.day:
        paths = [os.path.join(ATLAS_5S_DIR, f'{args.day}.parquet')]
    else:
        paths = sorted(glob.glob(os.path.join(ATLAS_5S_DIR, '2025_*.parquet')))
        if args.max_days:
            paths = paths[:args.max_days]

    print(f'Scanning {len(paths)} day(s) for ${args.target} moves in '
          f'{args.timeout} min ({timeout_bars} bars)...')

    all_events = []
    day_summaries = []
    hourly_agg = {}

    for p in paths:
        day_name = os.path.basename(p).replace('.parquet', '')
        if not os.path.exists(p):
            print(f'  MISSING: {p}')
            continue
        df = pd.read_parquet(p)
        events = tag_day(df, args.target, timeout_bars)
        summary = day_summary(events, day_name)
        if summary is None:
            continue
        day_summaries.append(summary)
        # Aggregate hourly
        h = hourly_breakdown(events)
        for hr, stats in h.items():
            if hr not in hourly_agg:
                hourly_agg[hr] = {'long_first': 0, 'short_first': 0,
                                  'neither': 0, 'total': 0, 'both': 0}
            for k in stats:
                hourly_agg[hr][k] += stats[k]
        # Save events (minimal — keep only hit events to limit size)
        for e in events:
            e['day'] = day_name
            if e['long_hit_bar'] is not None or e['short_hit_bar'] is not None:
                all_events.append(e)
        if len(day_summaries) % 20 == 0:
            print(f'  ...{len(day_summaries)} days done')

    print(f'\nProcessed {len(day_summaries)} days, '
          f'{len(all_events):,} tagged events.')

    # Global stats
    df_sum = pd.DataFrame(day_summaries)
    total_bars = df_sum['total_bars'].sum()
    total_long = df_sum['long_any'].sum()
    total_short = df_sum['short_any'].sum()
    total_both = df_sum['both'].sum()
    total_any = total_long + total_short - total_both
    total_neither = df_sum['neither'].sum()

    print('\n=== GLOBAL ===')
    print(f'  Bars scanned:           {total_bars:,}')
    print(f'  Bars with LONG hit:     {total_long:,}  ({total_long/total_bars*100:.1f}%)')
    print(f'  Bars with SHORT hit:    {total_short:,}  ({total_short/total_bars*100:.1f}%)')
    print(f'  Bars with BOTH:         {total_both:,}  ({total_both/total_bars*100:.1f}%)')
    print(f'  Bars with ANY $15 hit:  {total_any:,}  ({total_any/total_bars*100:.1f}%)')
    print(f'  Bars NEITHER:           {total_neither:,}  ({total_neither/total_bars*100:.1f}%)')
    print(f'  Days processed:         {len(df_sum)}')
    print(f'  Mean tagged bars/day:   {total_any/len(df_sum):.0f}')
    print(f'  Median tagged bars/day: {df_sum["any_hit"].median():.0f}')

    # Per-day distribution
    print(f'\nTagged bars/day distribution:')
    for pct in [5, 25, 50, 75, 95]:
        v = np.percentile(df_sum['any_hit'], pct)
        print(f'  p{pct}: {v:.0f}')

    # Bars-to-first distribution
    b_means = df_sum['bars_to_first_mean'].dropna()
    print(f'\nMean bars-to-first (across days):')
    print(f'  p10 / p50 / p90: {np.percentile(b_means, 10):.1f} / '
          f'{np.percentile(b_means, 50):.1f} / '
          f'{np.percentile(b_means, 90):.1f}')

    # Hourly
    print(f'\nHourly breakdown (UTC):')
    print(f'{"Hr":>3} {"Total":>8} {"Long":>7} {"Short":>7} {"Neither":>8}')
    for hr in sorted(hourly_agg.keys()):
        s = hourly_agg[hr]
        print(f'{hr:>3} {s["total"]:>8,} {s["long_first"]:>7,} '
              f'{s["short_first"]:>7,} {s["neither"]:>8,}')

    # Write MD
    out_md_path = OUT_MD_TEMPLATE.replace('${target}',
                                           str(int(args.target))).replace(
                                           '{timeout}', str(int(args.timeout)))
    out = [f'# $15 movement tagging — target ${args.target} in {args.timeout} min', '']
    out.append(f'Scanned {len(df_sum)} days ({total_bars:,} bars, 5s granularity).')
    out.append('')
    out.append('## Global summary')
    out.append('')
    out.append('| Metric | Value | % |')
    out.append('|---|---:|---:|')
    out.append(f'| Bars scanned | {total_bars:,} | — |')
    out.append(f'| Bars with LONG ${args.target} | {total_long:,} | '
               f'{total_long/total_bars*100:.1f}% |')
    out.append(f'| Bars with SHORT ${args.target} | {total_short:,} | '
               f'{total_short/total_bars*100:.1f}% |')
    out.append(f'| Bars with BOTH | {total_both:,} | '
               f'{total_both/total_bars*100:.1f}% |')
    out.append(f'| Bars with ANY hit | {total_any:,} | '
               f'{total_any/total_bars*100:.1f}% |')
    out.append(f'| NEITHER (dead zones) | {total_neither:,} | '
               f'{total_neither/total_bars*100:.1f}% |')
    out.append('')
    out.append(f'**Mean tagged bars/day: {total_any/len(df_sum):.0f}**')
    out.append(f'**Median tagged bars/day: {df_sum["any_hit"].median():.0f}**')
    out.append('')

    # Per-day distribution
    out.append('## Per-day distribution (tagged bars)')
    out.append('')
    out.append('| Percentile | Tagged bars |')
    out.append('|---:|---:|')
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        v = np.percentile(df_sum['any_hit'], pct)
        out.append(f'| p{pct} | {v:.0f} |')
    out.append('')

    # Hourly
    out.append('## Hourly distribution (UTC)')
    out.append('')
    out.append('| Hour | Total | Long first | Short first | Neither | '
               'Long% | Short% | Neither% |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for hr in sorted(hourly_agg.keys()):
        s = hourly_agg[hr]
        if s['total'] == 0:
            continue
        out.append(f'| {hr:02d} | {s["total"]:,} | {s["long_first"]:,} | '
                   f'{s["short_first"]:,} | {s["neither"]:,} | '
                   f'{s["long_first"]/s["total"]*100:.0f}% | '
                   f'{s["short_first"]/s["total"]*100:.0f}% | '
                   f'{s["neither"]/s["total"]*100:.0f}% |')
    out.append('')

    # Top 10 highest-density days
    out.append('## Top-10 highest-density days (most tagged bars)')
    out.append('')
    top = df_sum.nlargest(10, 'any_hit')
    out.append('| Day | Total bars | Tagged | Tag % | Long first | Short first | Median bars-to-first |')
    out.append('|---|---:|---:|---:|---:|---:|---:|')
    for _, row in top.iterrows():
        out.append(f'| {row["day"]} | {row["total_bars"]:,} | '
                   f'{row["any_hit"]:,} | '
                   f'{row["any_hit"]/row["total_bars"]*100:.1f}% | '
                   f'{row["long_first"]:,} | {row["short_first"]:,} | '
                   f'{row["bars_to_first_median"]:.0f} |')
    out.append('')

    # Bottom 10
    out.append('## Bottom-10 lowest-density days (quietest)')
    out.append('')
    bot = df_sum.nsmallest(10, 'any_hit')
    out.append('| Day | Total bars | Tagged | Tag % |')
    out.append('|---|---:|---:|---:|')
    for _, row in bot.iterrows():
        out.append(f'| {row["day"]} | {row["total_bars"]:,} | '
                   f'{row["any_hit"]:,} | '
                   f'{row["any_hit"]/row["total_bars"]*100:.1f}% |')
    out.append('')

    os.makedirs(os.path.dirname(out_md_path), exist_ok=True)
    with open(out_md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote MD: {out_md_path}')

    # Write events pickle
    out_pkl_path = OUT_PKL_TEMPLATE.replace('${target}',
                                             str(int(args.target))).replace(
                                             '{timeout}', str(int(args.timeout)))
    os.makedirs(os.path.dirname(out_pkl_path), exist_ok=True)
    with open(out_pkl_path, 'wb') as f:
        pickle.dump({
            'events': all_events,
            'day_summaries': day_summaries,
            'hourly': hourly_agg,
            'config': {'target': args.target, 'timeout_min': args.timeout,
                       'timeout_bars': timeout_bars,
                       'dollar_per_point': DOLLAR_PER_POINT},
        }, f)
    print(f'Wrote events pkl: {out_pkl_path}')


if __name__ == '__main__':
    main()
