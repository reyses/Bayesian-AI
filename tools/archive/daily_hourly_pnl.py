"""
Daily + hourly PnL distribution — the revenue-stream view.

Mean is misleading (one big day inflates it). What matters for a readable
income stream:
  - MODE of daily PnL: what's the most common day look like?
  - Spread: how tight is the distribution?
  - Daily WR: what fraction of days are winning?
  - Hourly contribution: which hours pay the bills, which bleed?

Outputs:
  - Daily distribution with mode (bucketed in $50 increments)
  - Daily WR
  - Per-hour aggregate PnL contribution (all days combined)
  - Per-hour WR on trades that started in that hour
  - Best/worst hours as ranked tables

Usage:
    python tools/daily_hourly_pnl.py
    python tools/daily_hourly_pnl.py --trades path/to/some.pkl
    python tools/daily_hourly_pnl.py --tier TREND_FOLLOWER  # filter one tier
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def hour_of_ts(ts):
    return datetime.utcfromtimestamp(ts).hour


def bucket_pnl(p, step=50):
    """Bucket to nearest $step (rounded down to bucket start)."""
    return int(np.floor(p / step) * step)


def daily_breakdown(trades):
    """Group trades by day, aggregate to per-day PnL + WR."""
    by_day = defaultdict(list)
    for t in trades:
        by_day[t.get('day', '?')].append(t)
    days = []
    for day, ts in sorted(by_day.items()):
        pnls = np.array([t['pnl'] for t in ts])
        wins = int((pnls > 0).sum())
        days.append({
            'day': day,
            'n_trades': len(ts),
            'pnl': float(pnls.sum()),
            'wr': wins / max(len(ts), 1) * 100,
            'winners': wins,
            'losers': len(ts) - wins,
        })
    return days


def hourly_breakdown(trades):
    """Group trades by entry hour (UTC), aggregate PnL + WR + N."""
    by_hour = defaultdict(list)
    for t in trades:
        ts = t.get('timestamp', 0)
        h = hour_of_ts(int(ts))
        by_hour[h].append(t)
    hours = []
    for h in range(24):
        bucket = by_hour.get(h, [])
        if not bucket:
            hours.append({'hour': h, 'n': 0, 'pnl': 0.0, 'wr': 0.0,
                          'mean': 0.0, 'median': 0.0})
            continue
        pnls = np.array([t['pnl'] for t in bucket])
        wins = int((pnls > 0).sum())
        hours.append({
            'hour': h,
            'n': len(bucket),
            'pnl': float(pnls.sum()),
            'wr': wins / len(bucket) * 100,
            'mean': float(pnls.mean()),
            'median': float(np.median(pnls)),
        })
    return hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default='training_iso/output/trades/iso_is.pkl')
    ap.add_argument('--tier', default=None, help='Filter to one tier')
    ap.add_argument('--bucket', type=int, default=50, help='$bucket size for mode')
    args = ap.parse_args()

    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    if args.tier:
        trades = [t for t in trades if t.get('entry_tier') == args.tier]
        print(f'Filtered to {args.tier}: {len(trades)} trades')
    else:
        print(f'All trades: {len(trades)}')
        from collections import Counter
        tiers = Counter(t.get('entry_tier', '?') for t in trades)
        print(f'Tiers: {dict(tiers.most_common())}')

    days = daily_breakdown(trades)
    n_days = len(days)
    if n_days == 0:
        print('No trades loaded.')
        return

    daily_pnls = np.array([d['pnl'] for d in days])
    total = float(daily_pnls.sum())
    winning_days = sum(1 for d in days if d['pnl'] > 0)
    breakeven_days = sum(1 for d in days if d['pnl'] == 0)
    losing_days = n_days - winning_days - breakeven_days

    print()
    print(f'=== DAILY PnL DISTRIBUTION (n_days={n_days}) ===')
    print(f'  Total: ${total:+,.0f}  |  Mean $/day: ${daily_pnls.mean():+.2f}')
    print(f'  Median: ${np.median(daily_pnls):+.2f}  |  Std: ${daily_pnls.std():.2f}')
    print(f'  Winning days: {winning_days}/{n_days} ({winning_days/n_days*100:.0f}%)')
    print(f'  Losing days:  {losing_days}/{n_days} ({losing_days/n_days*100:.0f}%)')
    print(f'  Flat days:    {breakeven_days}/{n_days}')
    print()
    print(f'  Percentiles:  p10={np.percentile(daily_pnls, 10):+7.0f}  '
          f'p25={np.percentile(daily_pnls, 25):+7.0f}  '
          f'p50={np.percentile(daily_pnls, 50):+7.0f}  '
          f'p75={np.percentile(daily_pnls, 75):+7.0f}  '
          f'p90={np.percentile(daily_pnls, 90):+7.0f}')
    print(f'  Extremes:     min=${daily_pnls.min():+7.0f}  '
          f'max=${daily_pnls.max():+7.0f}')

    # Mode via bucket
    buckets = defaultdict(int)
    for p in daily_pnls:
        buckets[bucket_pnl(p, args.bucket)] += 1
    sorted_buckets = sorted(buckets.items())
    mode_bucket = max(buckets.items(), key=lambda kv: kv[1])
    print()
    print(f'=== DAILY PnL BUCKETS (${args.bucket} each) ===')
    print(f'  MODE bucket: [${mode_bucket[0]}..${mode_bucket[0]+args.bucket}) '
          f'— {mode_bucket[1]} days ({mode_bucket[1]/n_days*100:.0f}%)')
    print()
    print(f'  {"Bucket":<20} {"N":>5} {"% days":>8} {"Bar":<40}')
    max_count = max(buckets.values())
    for b, count in sorted_buckets:
        bar = '#' * int(40 * count / max_count)
        print(f'  [${b:>+6}..${b+args.bucket:>+6})  {count:>5} {count/n_days*100:>7.1f}% {bar}')

    # $300/day target analysis
    target = 300
    days_above = (daily_pnls >= target).sum()
    days_above_100 = (daily_pnls >= 100).sum()
    days_above_0 = (daily_pnls > 0).sum()
    print()
    print(f'=== REVENUE TARGETS ===')
    print(f'  Days >= $300: {days_above}/{n_days} ({days_above/n_days*100:.0f}%)')
    print(f'  Days >= $100: {days_above_100}/{n_days} ({days_above_100/n_days*100:.0f}%)')
    print(f'  Days >  $0:   {days_above_0}/{n_days} ({days_above_0/n_days*100:.0f}%)')
    print(f'  Avg on positive days: ${daily_pnls[daily_pnls>0].mean():+.2f}')
    print(f'  Avg on negative days: ${daily_pnls[daily_pnls<0].mean():+.2f}' if (daily_pnls<0).any() else '')

    # Hourly breakdown
    hours = hourly_breakdown(trades)
    print()
    print(f'=== HOURLY PnL (entry hour, UTC) ===')
    print(f'  {"Hour":>4} {"N":>6} {"Total":>10} {"Mean":>9} {"Median":>9} {"WR":>5}')
    for h in hours:
        if h['n'] == 0:
            continue
        print(f'  {h["hour"]:>4}h {h["n"]:>6} ${h["pnl"]:>+8,.0f} ${h["mean"]:>+7.2f} '
              f'${h["median"]:>+7.2f} {h["wr"]:>4.0f}%')

    # Best/worst hours
    print()
    active_hours = [h for h in hours if h['n'] >= 10]
    if active_hours:
        by_pnl = sorted(active_hours, key=lambda h: -h['pnl'])
        print(f'=== BEST 5 HOURS (by total PnL) ===')
        for h in by_pnl[:5]:
            print(f'  {h["hour"]:>2}h UTC: n={h["n"]:>4}  total=${h["pnl"]:>+6,.0f}  '
                  f'mean=${h["mean"]:>+6.2f}  WR={h["wr"]:>4.0f}%')
        print(f'=== WORST 5 HOURS (by total PnL) ===')
        for h in by_pnl[-5:][::-1]:
            print(f'  {h["hour"]:>2}h UTC: n={h["n"]:>4}  total=${h["pnl"]:>+6,.0f}  '
                  f'mean=${h["mean"]:>+6.2f}  WR={h["wr"]:>4.0f}%')


if __name__ == '__main__':
    main()
