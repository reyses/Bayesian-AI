#!/usr/bin/env python
"""
Session Overlay — Map trades onto 1h + 1m price structure with adaptive Fibs.

Panels:
  1) 1h candlesticks  + adaptive Fib zones + trade entry dots
  2) 1m price close   + H/L fill + trade vertical lines + entry dots
  3) 1m Moving Range   |close_i - close_{i-1}|  + 60-bar MA

Fibs auto-detect from zigzag swings on 1h data. When price breaks 0% or 100%
the old levels dim and a new set is drawn from the next swing pair.

Usage:
  python tools/session_overlay.py
  python tools/session_overlay.py --data DATA/ATLAS_OOS --trades checkpoints/oracle_trade_log.csv
  python tools/session_overlay.py --data DATA/ATLAS --trades reports/is/oracle_trade_log.csv
  python tools/session_overlay.py --fib-pct 1.5   # larger swings only
"""

import argparse, glob, os, sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ── Fib ratios & colors ─────────────────────────────────────────────────
FIB_RATIOS = [
    ('0%',    0.000),
    ('23.6%', 0.236),
    ('38.2%', 0.382),
    ('50%',   0.500),
    ('61.8%', 0.618),
    ('76.4%', 0.764),
    ('100%',  1.000),
]
FIB_COLORS = {
    '0%': '#777777', '23.6%': '#FF8800', '38.2%': '#FF0044',
    '50%': '#CC00CC', '61.8%': '#0088FF', '76.4%': '#00CCBB', '100%': '#777777',
}


# ── Data loaders ─────────────────────────────────────────────────────────

def load_parquet(data_dir, tf):
    """Load + concat all parquet files for a timeframe."""
    tf_dir = os.path.join(data_dir, tf)
    files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    if not files:
        sys.exit(f"  No parquet in {tf_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = (pd.concat(dfs, ignore_index=True)
            .sort_values('timestamp')
            .drop_duplicates('timestamp')
            .reset_index(drop=True))
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    return df


def load_trades(path):
    """Load trade log CSV, convert timestamps."""
    df = pd.read_csv(path)
    for col in ['entry_time', 'exit_time']:
        if col in df.columns:
            df[col + '_dt'] = pd.to_datetime(df[col], unit='s', utc=True)
    return df


# ── Zigzag swing detection ──────────────────────────────────────────────

def detect_zigzag(df, min_pct=1.0):
    """Detect significant swing highs/lows using percentage threshold.

    Returns list of dicts: {idx, dt, price, type='H'|'L'}
    """
    highs = df['high'].values
    lows = df['low'].values
    dts = df['dt'].values

    # Seed with the first extreme in the first 10 bars
    first_h_idx = int(np.argmax(highs[:min(10, len(highs))]))
    first_l_idx = int(np.argmin(lows[:min(10, len(lows))]))

    if first_h_idx <= first_l_idx:
        swings = [{'idx': first_h_idx, 'dt': dts[first_h_idx],
                    'price': float(highs[first_h_idx]), 'type': 'H'}]
    else:
        swings = [{'idx': first_l_idx, 'dt': dts[first_l_idx],
                    'price': float(lows[first_l_idx]), 'type': 'L'}]

    for i in range(swings[0]['idx'] + 1, len(highs)):
        last = swings[-1]
        h = float(highs[i])
        l = float(lows[i])

        if last['type'] == 'H':
            pct_drop = (last['price'] - l) / last['price'] * 100
            if pct_drop >= min_pct:
                swings.append({'idx': i, 'dt': dts[i], 'price': l, 'type': 'L'})
            elif h > last['price']:
                swings[-1] = {'idx': i, 'dt': dts[i], 'price': h, 'type': 'H'}
        else:
            pct_rise = (h - last['price']) / last['price'] * 100
            if pct_rise >= min_pct:
                swings.append({'idx': i, 'dt': dts[i], 'price': h, 'type': 'H'})
            elif l < last['price']:
                swings[-1] = {'idx': i, 'dt': dts[i], 'price': l, 'type': 'L'}

    return swings


def compute_fib_zones(swings, end_dt):
    """Build Fib retracement zones from consecutive swing pairs.

    Each zone has:
      - levels: dict of fib_name -> price
      - start_dt / end_dt: time validity window
      - active: True if this is the last (current) zone
      - anchor_pts: the two swing points that define this zone
    """
    zones = []
    for i in range(len(swings) - 1):
        s1, s2 = swings[i], swings[i + 1]

        # Determine high/low of the swing pair
        if s1['type'] == 'H':
            high, low = s1['price'], s2['price']
        else:
            high, low = s2['price'], s1['price']

        span = high - low
        if span < 1:
            continue

        # Zone is valid from the second swing until the next swing (or end)
        zone_start = s2['dt']
        zone_end = swings[i + 2]['dt'] if (i + 2) < len(swings) else end_dt
        is_active = (i == len(swings) - 2)

        levels = {}
        for name, ratio in FIB_RATIOS:
            levels[name] = low + span * ratio

        zones.append({
            'levels': levels,
            'high': high,
            'low': low,
            'start_dt': zone_start,
            'end_dt': zone_end,
            'active': is_active,
            'anchor_pts': (s1, s2),
        })

    return zones


def draw_fib_zones(ax, zones, show_labels=True, max_expired=2):
    """Draw adaptive Fib zones — active bright, last N expired dim, rest hidden."""
    # Only show active + last N expired zones to avoid clutter
    expired = [z for z in zones if not z['active']]
    visible_expired = expired[-max_expired:] if len(expired) > max_expired else expired
    visible = visible_expired + [z for z in zones if z['active']]

    for zone in visible:
        a_line = 0.60 if zone['active'] else 0.12
        lw = 1.0 if zone['active'] else 0.4
        ls = '-' if zone['active'] else ':'

        x0, x1 = zone['start_dt'], zone['end_dt']

        for name, price in zone['levels'].items():
            c = FIB_COLORS.get(name, '#888')
            ax.plot([x0, x1], [price, price],
                    color=c, linewidth=lw, alpha=a_line, linestyle=ls)

        # Labels for active zone only
        if zone['active'] and show_labels:
            for name, price in zone['levels'].items():
                c = FIB_COLORS.get(name, '#888')
                ax.text(x1, price + 2,
                        f' {name}  {price:,.0f}',
                        color=c, fontsize=8, va='bottom', ha='right',
                        bbox=dict(facecolor='#080808', edgecolor='none',
                                  alpha=0.8, pad=1))


def draw_zigzag(ax, swings, color='#FFFF00', alpha=0.4, lw=0.8):
    """Draw the zigzag line connecting swing points."""
    if len(swings) < 2:
        return
    xs = [s['dt'] for s in swings]
    ys = [s['price'] for s in swings]
    ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw, linestyle='-', zorder=3)
    # Swing markers
    for s in swings:
        marker = 'v' if s['type'] == 'H' else '^'
        mc = '#FF4444' if s['type'] == 'H' else '#44FF44'
        ax.scatter([s['dt']], [s['price']], marker=marker, color=mc,
                   s=40, alpha=0.7, zorder=4, edgecolors='white', linewidths=0.3)


# ── Candle renderer ──────────────────────────────────────────────────────

def plot_candles_1h(ax, df):
    """Efficient 1h candlestick plot using vectorised bar + vlines."""
    up = df[df['close'] >= df['open']]
    dn = df[df['close'] <  df['open']]
    w = 0.032  # bar width in days (~46 min)

    ax.vlines(up['dt'].values, up['low'].values, up['high'].values,
              color='#00DD00', linewidth=0.7)
    ax.vlines(dn['dt'].values, dn['low'].values, dn['high'].values,
              color='#DD0000', linewidth=0.7)
    ax.bar(up['dt'].values, (up['close'] - up['open']).values,
           bottom=up['open'].values, width=w, color='#00DD00',
           edgecolor='#00DD00', linewidth=0.3)
    ax.bar(dn['dt'].values, (dn['open'] - dn['close']).values,
           bottom=dn['close'].values, width=w, color='#DD0000',
           edgecolor='#DD0000', linewidth=0.3)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Session overlay: trades on 1h + 1m + MR')
    ap.add_argument('--data',    default='DATA/ATLAS_OOS', help='ATLAS data directory')
    ap.add_argument('--trades',  default='checkpoints/oracle_trade_log.csv', help='Trade log CSV')
    ap.add_argument('--no-fib',  dest='fib', action='store_false', default=True)
    ap.add_argument('--fib-pct', type=float, default=2.0,
                    help='Zigzag minimum %% move for swing detection (default 2.0)')
    ap.add_argument('--output',  default=None, help='Output PNG path')
    ap.add_argument('--dpi',     type=int, default=150)
    args = ap.parse_args()

    # ── Load price data ──────────────────────────────────────────────────
    print("Loading 1h data ...")
    df_1h = load_parquet(args.data, '1h')
    print(f"  {len(df_1h):,} bars  ({df_1h['dt'].iloc[0]:%Y-%m-%d} "
          f"-> {df_1h['dt'].iloc[-1]:%Y-%m-%d})")

    print("Loading 1m data ...")
    df_1m = load_parquet(args.data, '1m')
    print(f"  {len(df_1m):,} bars")

    # ── Adaptive Fibonacci ───────────────────────────────────────────────
    fib_zones = []
    swings = []
    if args.fib:
        print(f"Detecting zigzag swings (min {args.fib_pct:.1f}% move) ...")
        swings = detect_zigzag(df_1h, min_pct=args.fib_pct)
        print(f"  {len(swings)} swings detected")
        for s in swings:
            tag = 'v' if s['type'] == 'H' else '^'
            print(f"    {tag} {pd.Timestamp(s['dt']):%Y-%m-%d %H:%M}  "
                  f"{s['price']:,.2f}  ({s['type']})")
        fib_zones = compute_fib_zones(swings, df_1h['dt'].iloc[-1])
        print(f"  {len(fib_zones)} Fib zones  "
              f"(active: {sum(1 for z in fib_zones if z['active'])})")

    # ── Load trades (optional) ───────────────────────────────────────────
    trades = None
    wins = losses = pd.DataFrame()
    if os.path.exists(args.trades):
        print(f"Loading trades from {args.trades} ...")
        trades = load_trades(args.trades)
        t_min, t_max = df_1h['timestamp'].min(), df_1h['timestamp'].max()
        if 'entry_time' in trades.columns:
            trades = trades[(trades['entry_time'] >= t_min) &
                            (trades['entry_time'] <= t_max)].copy()
        print(f"  {len(trades):,} trades in range")
        if 'result' in trades.columns:
            wins   = trades[trades['result'] == 'WIN']
            losses = trades[trades['result'] != 'WIN']
            print(f"  WIN={len(wins)}  LOSS={len(losses)}")
    else:
        print(f"  Trade file not found: {args.trades} -- plotting price only")

    # ── Figure ───────────────────────────────────────────────────────────
    print("Rendering ...")
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(60, 30), dpi=args.dpi,
        gridspec_kw={'height_ratios': [4, 4, 2]},
        sharex=True,
    )
    fig.patch.set_facecolor('#080808')
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('#080808')
        ax.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#222222')
        ax.grid(True, alpha=0.12, color='#444444', linewidth=0.4)

    # ── Panel 1: 1h Candles + Adaptive Fibs ──────────────────────────────
    ax1.set_title('1h Candlesticks  +  Adaptive Fibonacci  +  Trade Clusters',
                  color='white', fontsize=16, fontweight='bold', pad=12)
    plot_candles_1h(ax1, df_1h)
    ax1.set_ylabel('Price', color='#aaaaaa', fontsize=10)

    if fib_zones:
        draw_fib_zones(ax1, fib_zones, show_labels=True)
        draw_zigzag(ax1, swings)

    # Trade dots on 1h
    if len(wins):
        ax1.scatter(wins['entry_time_dt'].values, wins['entry_price'].values,
                    s=22, color='#00FF44', alpha=0.75, zorder=5, marker='o',
                    label=f'WIN  ({len(wins)})')
    if len(losses):
        ax1.scatter(losses['entry_time_dt'].values, losses['entry_price'].values,
                    s=28, color='#FF2222', alpha=0.85, zorder=5, marker='x',
                    linewidths=0.8, label=f'LOSS ({len(losses)})')
    if trades is not None and len(trades):
        ax1.legend(loc='upper left', fontsize=9, facecolor='#111111',
                   edgecolor='#333333', labelcolor='white', framealpha=0.85)

    # ── Panel 2: 1m Close + H/L fill + trades ───────────────────────────
    ax2.set_title('1m Price  (close line + H/L range)  +  Trade Entries',
                  color='white', fontsize=16, fontweight='bold', pad=12)
    ax2.fill_between(df_1m['dt'].values, df_1m['low'].values, df_1m['high'].values,
                     color='#2244AA', alpha=0.20, linewidth=0)
    ax2.plot(df_1m['dt'].values, df_1m['close'].values,
             color='#5588FF', linewidth=0.25, alpha=0.85)
    ax2.set_ylabel('Price', color='#aaaaaa', fontsize=10)

    # Adaptive Fibs on 1m (no labels — too dense)
    if fib_zones:
        draw_fib_zones(ax2, fib_zones, show_labels=False)

    # Trade vertical lines + dots
    y_lo = df_1m['low'].min() - 30
    y_hi = df_1m['high'].max() + 30
    if len(wins):
        ax2.vlines(wins['entry_time_dt'].values, y_lo, y_hi,
                   color='#00FF44', alpha=0.06, linewidth=0.4)
        ax2.scatter(wins['entry_time_dt'].values, wins['entry_price'].values,
                    s=14, color='#00FF44', alpha=0.65, zorder=5, marker='o')
    if len(losses):
        ax2.vlines(losses['entry_time_dt'].values, y_lo, y_hi,
                   color='#FF2222', alpha=0.10, linewidth=0.5)
        ax2.scatter(losses['entry_time_dt'].values, losses['entry_price'].values,
                    s=20, color='#FF2222', alpha=0.80, zorder=5, marker='x',
                    linewidths=0.7)
    ax2.set_ylim(y_lo, y_hi)

    # ── Panel 3: Moving Range ────────────────────────────────────────────
    ax3.set_title('1m Moving Range   |close_i - close_{i-1}|',
                  color='white', fontsize=14, fontweight='bold', pad=10)
    mr = df_1m['close'].diff().abs()
    ax3.fill_between(df_1m['dt'].values, 0, mr.values,
                     color='#FF8800', alpha=0.30, linewidth=0)
    mr_ma = mr.rolling(60, min_periods=1).mean()
    ax3.plot(df_1m['dt'].values, mr_ma.values,
             color='#FFCC00', linewidth=0.7, alpha=0.9, label='60-bar MA')
    ax3.set_ylabel('MR (points)', color='#aaaaaa', fontsize=10)
    ax3.legend(loc='upper left', fontsize=8, facecolor='#111111',
               edgecolor='#333333', labelcolor='white', framealpha=0.85)

    # Trade lines on MR
    if len(wins):
        ax3.vlines(wins['entry_time_dt'].values, 0, mr.max() * 0.3,
                   color='#00FF44', alpha=0.06, linewidth=0.4)
    if len(losses):
        ax3.vlines(losses['entry_time_dt'].values, 0, mr.max() * 0.3,
                   color='#FF2222', alpha=0.10, linewidth=0.5)

    # ── X-axis ───────────────────────────────────────────────────────────
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax3.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    ax3.xaxis.set_minor_locator(mdates.DayLocator())
    ax3.tick_params(axis='x', rotation=45, labelsize=8)

    # ── Save ─────────────────────────────────────────────────────────────
    out = args.output or os.path.join('reports', 'plots', 'session_overlay.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.tight_layout(h_pad=2.0)
    print(f"  Saving {out} ({len(df_1m):,} 1m bars, {len(fib_zones)} fib zones) ...")
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight', facecolor='#080808')
    plt.close(fig)
    _mb = os.path.getsize(out) / 1024 / 1024
    print(f"  Saved: {out}  ({_mb:.1f} MB)")


if __name__ == '__main__':
    main()
