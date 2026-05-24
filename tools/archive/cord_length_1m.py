"""
1m cord-length analysis — the upper bound on extractable PnL.

Cord length = sum of |price_change| between zigzag reversal points filtered
at threshold R. This is the theoretical maximum PnL a perfect oracle could
capture — every turn, no slippage, unlimited trades.

Computed at multiple R thresholds so we see the full "amplitude → capacity"
curve:
  R = $0   : total variation (all 1m changes, noise-inflated)
  R = $5   : micro legs
  R = $10  : small legs
  R = $15  : saturation-frame legs (our target)
  R = $20  : medium legs
  R = $30  : major swings only

For each R, per day:
  - N legs
  - Avg leg size
  - Cord length ($ × 2)
  - Legs/hour

Aggregated IS + OOS comparison. Plus a chart for one day showing the
zigzag at R=$15.

Usage:
    python tools/cord_length_1m.py                      # full IS + OOS
    python tools/cord_length_1m.py --day 2025_06_09     # single day chart

Output:
    reports/findings/cord_length_1m.md
    charts/cord_length_<day>_R{R}.png
"""
import os
import sys
import glob
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm


ATLAS_1M_DIR = 'DATA/ATLAS/1m'
OUT_MD = 'reports/findings/cord_length_1m.md'
DOLLAR_PER_POINT = 2.0
THRESHOLDS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0]  # $


def zigzag_pivots(closes, threshold_dollars):
    """Return indices of zigzag pivot points given $ threshold.

    threshold=0 means every bar is a pivot (total variation).
    """
    n = len(closes)
    if n < 2:
        return [0]
    if threshold_dollars == 0:
        return list(range(n))

    threshold_pts = threshold_dollars / DOLLAR_PER_POINT
    pivots = [0]
    last_pivot_price = closes[0]
    last_dir = None  # None until first qualifying move

    for i in range(1, n):
        price = closes[i]
        if last_dir is None:
            move = price - last_pivot_price
            if abs(move) >= threshold_pts:
                last_dir = 'up' if move > 0 else 'down'
                pivots.append(i)
                last_pivot_price = price
        elif last_dir == 'up':
            # We're in an up leg; update pivot if new high
            if price > closes[pivots[-1]]:
                pivots[-1] = i
                last_pivot_price = price
            elif (last_pivot_price - price) >= threshold_pts:
                # Reversal confirmed
                last_dir = 'down'
                pivots.append(i)
                last_pivot_price = price
        elif last_dir == 'down':
            if price < closes[pivots[-1]]:
                pivots[-1] = i
                last_pivot_price = price
            elif (price - last_pivot_price) >= threshold_pts:
                last_dir = 'up'
                pivots.append(i)
                last_pivot_price = price
    return pivots


def cord_stats(closes, pivots):
    """Given a pivot list, compute cord length and leg stats."""
    if len(pivots) < 2:
        return {
            'n_legs': 0, 'cord_pts': 0.0, 'cord_dollars': 0.0,
            'mean_leg_pts': 0.0, 'mean_leg_dollars': 0.0,
        }
    legs_pts = [abs(closes[pivots[i + 1]] - closes[pivots[i]])
                for i in range(len(pivots) - 1)]
    cord_pts = float(sum(legs_pts))
    return {
        'n_legs': len(legs_pts),
        'cord_pts': cord_pts,
        'cord_dollars': cord_pts * DOLLAR_PER_POINT,
        'mean_leg_pts': float(np.mean(legs_pts)) if legs_pts else 0.0,
        'mean_leg_dollars': float(np.mean(legs_pts) * DOLLAR_PER_POINT) if legs_pts else 0.0,
        'max_leg_pts': float(max(legs_pts)),
        'median_leg_pts': float(np.median(legs_pts)),
    }


def analyze_day(df, thresholds):
    """Return {threshold: stats} for one day."""
    closes = df['close'].values.astype(np.float64)
    n_bars = len(closes)
    active_hours = n_bars / 60.0  # 1m bars
    out = {}
    for R in thresholds:
        pivots = zigzag_pivots(closes, R)
        s = cord_stats(closes, pivots)
        s['legs_per_hour'] = s['n_legs'] / active_hours if active_hours else 0
        out[R] = s
    out['_n_bars'] = n_bars
    out['_active_hours'] = active_hours
    out['_price_range_pts'] = float(closes.max() - closes.min())
    out['_price_range_dollars'] = out['_price_range_pts'] * DOLLAR_PER_POINT
    return out


def aggregate_days(paths, thresholds, label):
    per_day = []
    for p in tqdm(paths, desc=label, unit='day'):
        day_name = os.path.basename(p).replace('.parquet', '')
        df = pd.read_parquet(p)
        stats = analyze_day(df, thresholds)
        stats['day'] = day_name
        per_day.append(stats)
    return per_day


def aggregate_summary(per_day, thresholds):
    """Return per-threshold aggregate stats."""
    out = {}
    n_days = len(per_day)
    total_hours = sum(d['_active_hours'] for d in per_day)
    out['_n_days'] = n_days
    out['_total_hours'] = total_hours
    for R in thresholds:
        cord_sum = sum(d[R]['cord_dollars'] for d in per_day)
        legs_sum = sum(d[R]['n_legs'] for d in per_day)
        per_day_array = np.array([d[R]['cord_dollars'] for d in per_day])
        out[R] = {
            'total_cord_dollars': cord_sum,
            'total_legs': legs_sum,
            'mean_per_day': cord_sum / n_days if n_days else 0,
            'median_per_day': float(np.median(per_day_array)),
            'p25_per_day': float(np.percentile(per_day_array, 25)),
            'p75_per_day': float(np.percentile(per_day_array, 75)),
            'p95_per_day': float(np.percentile(per_day_array, 95)),
            'legs_per_day': legs_sum / n_days if n_days else 0,
            'dollars_per_leg': cord_sum / legs_sum if legs_sum else 0,
            'dollars_per_hour': cord_sum / total_hours if total_hours else 0,
            'dollars_per_8min': cord_sum / (total_hours * 60 / 8) if total_hours else 0,
        }
    return out


def render_chart(df, R, out_path):
    """Plot price + zigzag at threshold R."""
    closes = df['close'].values.astype(np.float64)
    ts = df['timestamp'].values.astype(np.float64)
    dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]
    pivots = zigzag_pivots(closes, R)
    s = cord_stats(closes, pivots)
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(dts, closes, color='black', linewidth=0.8, alpha=0.5,
            label='Price (1m close)')
    if len(pivots) >= 2:
        piv_dts = [dts[i] for i in pivots]
        piv_prices = [closes[i] for i in pivots]
        ax.plot(piv_dts, piv_prices, color='tab:red', linewidth=1.5,
                marker='o', markersize=5,
                label=f'Zigzag ${R:.0f} — {len(pivots)-1} legs, cord=${s["cord_dollars"]:,.0f}')
    ax.set_ylabel('MNQ price', fontsize=11)
    ax.set_xlabel('Time (UTC)', fontsize=11)
    ax.set_title(f'1m zigzag at ${R:.0f} threshold — cord length overlay', fontsize=13)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None,
                    help='Single day to chart (in addition to full analysis)')
    args = ap.parse_args()

    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))

    print(f'IS: {len(is_paths)} days  |  OOS: {len(oos_paths)} days')

    is_per_day = aggregate_days(is_paths, THRESHOLDS, 'IS')
    oos_per_day = aggregate_days(oos_paths, THRESHOLDS, 'OOS')

    is_agg = aggregate_summary(is_per_day, THRESHOLDS)
    oos_agg = aggregate_summary(oos_per_day, THRESHOLDS)

    # Console
    print('\n=== IS aggregate (277 days) ===')
    print(f'{"R":>6} {"legs/day":>10} {"$/leg":>8} {"$/day":>10} '
          f'{"$/hour":>8} {"$/8min":>8}')
    for R in THRESHOLDS:
        a = is_agg[R]
        print(f'${R:>4.0f} {a["legs_per_day"]:>10.0f} '
              f'${a["dollars_per_leg"]:>6.1f} '
              f'${a["mean_per_day"]:>8,.0f} '
              f'${a["dollars_per_hour"]:>6.0f} '
              f'${a["dollars_per_8min"]:>6.1f}')

    print('\n=== OOS aggregate ===')
    print(f'{"R":>6} {"legs/day":>10} {"$/leg":>8} {"$/day":>10} '
          f'{"$/hour":>8} {"$/8min":>8}')
    for R in THRESHOLDS:
        a = oos_agg[R]
        print(f'${R:>4.0f} {a["legs_per_day"]:>10.0f} '
              f'${a["dollars_per_leg"]:>6.1f} '
              f'${a["mean_per_day"]:>8,.0f} '
              f'${a["dollars_per_hour"]:>6.0f} '
              f'${a["dollars_per_8min"]:>6.1f}')

    # Markdown
    out = ['# 1m cord-length analysis — theoretical PnL ceiling', '']
    out.append('**Cord length** = sum of absolute price movements between '
               'zigzag reversal points. It is the theoretical maximum PnL a '
               'perfect oracle could extract (every turn caught, no slippage).')
    out.append('')
    out.append('All dollars use MNQ contract spec: $2 per 1.0 point movement.')
    out.append('')
    out.append(f'**Data**: {len(is_paths)} IS days + {len(oos_paths)} OOS days '
               'of 1m bars.')
    out.append('')

    out.append('## IS aggregate')
    out.append('')
    out.append('| R | Mean legs/day | Mean leg $ | **Mean $/day** | $/hour | $/8min | Median $/day | p95 $/day |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for R in THRESHOLDS:
        a = is_agg[R]
        label = 'all bars' if R == 0 else f'${R:.0f}'
        out.append(f'| {label} | {a["legs_per_day"]:.0f} | '
                   f'${a["dollars_per_leg"]:.2f} | '
                   f'**${a["mean_per_day"]:,.0f}** | '
                   f'${a["dollars_per_hour"]:.0f} | '
                   f'${a["dollars_per_8min"]:.1f} | '
                   f'${a["median_per_day"]:,.0f} | '
                   f'${a["p95_per_day"]:,.0f} |')
    out.append('')

    out.append('## OOS aggregate')
    out.append('')
    out.append('| R | Mean legs/day | Mean leg $ | **Mean $/day** | $/hour | $/8min | Median $/day | p95 $/day |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|')
    for R in THRESHOLDS:
        a = oos_agg[R]
        label = 'all bars' if R == 0 else f'${R:.0f}'
        out.append(f'| {label} | {a["legs_per_day"]:.0f} | '
                   f'${a["dollars_per_leg"]:.2f} | '
                   f'**${a["mean_per_day"]:,.0f}** | '
                   f'${a["dollars_per_hour"]:.0f} | '
                   f'${a["dollars_per_8min"]:.1f} | '
                   f'${a["median_per_day"]:,.0f} | '
                   f'${a["p95_per_day"]:,.0f} |')
    out.append('')

    # Capture-rate context
    out.append('## Capture-rate context')
    out.append('')
    out.append('Current iso engine: IS **$311/day** · OOS **$67/day**.')
    out.append('')
    out.append('| R | IS cord $/day | IS capture % | OOS cord $/day | OOS capture % |')
    out.append('|---:|---:|---:|---:|---:|')
    for R in THRESHOLDS:
        is_cord = is_agg[R]['mean_per_day']
        oos_cord = oos_agg[R]['mean_per_day']
        is_cap = 311 / is_cord * 100 if is_cord else 0
        oos_cap = 67 / oos_cord * 100 if oos_cord else 0
        label = 'all bars' if R == 0 else f'${R:.0f}'
        out.append(f'| {label} | ${is_cord:,.0f} | {is_cap:.1f}% | '
                   f'${oos_cord:,.0f} | {oos_cap:.1f}% |')
    out.append('')

    # Best/worst days by cord length at R=$15
    out.append('## Top-10 highest cord-length days (R=$15 IS)')
    out.append('')
    is_by_cord = sorted(is_per_day, key=lambda d: -d[15.0]['cord_dollars'])
    out.append('| Day | Cord $ | Legs | Avg leg $ | Price range $ |')
    out.append('|---|---:|---:|---:|---:|')
    for d in is_by_cord[:10]:
        s = d[15.0]
        out.append(f'| {d["day"]} | ${s["cord_dollars"]:,.0f} | {s["n_legs"]} | '
                   f'${s["mean_leg_dollars"]:.1f} | ${d["_price_range_dollars"]:,.0f} |')
    out.append('')

    out.append('## Bottom-10 quietest days (R=$15 IS)')
    out.append('')
    out.append('| Day | Cord $ | Legs | Avg leg $ |')
    out.append('|---|---:|---:|---:|')
    for d in is_by_cord[-10:]:
        s = d[15.0]
        out.append(f'| {d["day"]} | ${s["cord_dollars"]:,.0f} | {s["n_legs"]} | '
                   f'${s["mean_leg_dollars"]:.1f} |')
    out.append('')

    # Chart a specific day if requested
    if args.day:
        day_path = os.path.join(ATLAS_1M_DIR, f'{args.day}.parquet')
        if os.path.exists(day_path):
            df = pd.read_parquet(day_path)
            for R in [5.0, 10.0, 15.0, 30.0]:
                out_chart = f'charts/cord_length_{args.day}_R{int(R)}.png'
                render_chart(df, R, out_chart)
                print(f'Wrote chart: {out_chart}')
            out.append(f'## Chart of {args.day}')
            out.append('')
            out.append(f'Zigzag overlays at R=$5/$10/$15/$30 saved to `charts/`.')
            out.append('')

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print(f'\nWrote: {OUT_MD}')


if __name__ == '__main__':
    main()
