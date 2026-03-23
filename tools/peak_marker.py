#!/usr/bin/env python
"""
Peak Marker — Manually mark peaks on a price chart.

Single click = mark a peak. Direction auto-detected from local price action.
Press D to delete last mark, Q to save+quit.

Saves to DATA/regime_seeds/human_peaks_{date}.json

After marking, run peak_marker_analysis.py to analyze
the 1s data before each marked peak.

Usage:
    python tools/peak_marker.py --date 2025-07-14
    python tools/peak_marker.py --date 2025-07-14 --tf 5m
    python tools/peak_marker.py --date 2025-07-14 --tf 1h
    python tools/peak_marker.py --data DATA/ATLAS_OOS --date 2026-02-05 --tf 5m
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf

# Force interactive backend
plt.switch_backend('TkAgg')

TICK_SIZE = 0.25
SEEDS_DIR = 'DATA/regime_seeds'


class PeakMarker:
    """Manual peak marker with crosshair."""

    def __init__(self, df, date_str, tf='5m'):
        self.df = df
        self.date_str = date_str
        self.tf = tf

        self.close = df['close'].values.astype(float)
        self.high = df['high'].values.astype(float)
        self.low = df['low'].values.astype(float)
        self.timestamps = df['timestamp'].values.astype(float)
        self.dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in self.timestamps]

        # Marked peaks
        self.peaks = []
        self._peak_markers = []

    def _find_nearest_bar(self, x_date):
        """Find nearest bar index to click."""
        click_num = mdates.date2num(x_date) if not isinstance(x_date, float) else x_date
        bar_nums = mdates.date2num(self.dt_stamps)
        idx = int(np.argmin(np.abs(bar_nums - click_num)))
        return idx

    def _detect_direction(self, idx, lookback=5, lookahead=5):
        """Auto-detect peak direction from surrounding price action.

        If price was RISING before and FALLING after -> peak HIGH -> SHORT (reversal down)
        If price was FALLING before and RISING after -> peak LOW -> LONG (reversal up)
        """
        start = max(0, idx - lookback)
        end = min(len(self.close) - 1, idx + lookahead)

        price_before = self.close[start:idx]
        price_after = self.close[idx:end + 1]

        if len(price_before) < 2 or len(price_after) < 2:
            return 'UNKNOWN'

        trend_before = price_before[-1] - price_before[0]  # positive = was rising
        trend_after = price_after[-1] - price_after[0]     # negative = now falling

        if trend_before > 0 and trend_after < 0:
            return 'SHORT'  # peak high, reversal down
        elif trend_before < 0 and trend_after > 0:
            return 'LONG'   # peak low, reversal up
        elif trend_before > 0:
            return 'SHORT'  # was rising, assume peak
        elif trend_before < 0:
            return 'LONG'   # was falling, assume bottom
        else:
            return 'UNKNOWN'

    def _on_click(self, event):
        """Handle click — mark peak."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # left click only
            return

        click_num = event.xdata
        if click_num is None:
            return

        idx = self._find_nearest_bar(click_num)
        direction = self._detect_direction(idx)

        # Snap to bar extreme: use high or low (whichever is the peak)
        # Click above close = peak high, click below close = peak low
        click_price = event.ydata if event.ydata is not None else self.close[idx]
        dist_to_high = abs(click_price - self.high[idx])
        dist_to_low = abs(click_price - self.low[idx])
        if dist_to_high < dist_to_low:
            snap_price = float(self.high[idx])
            snap_label = 'H'
        else:
            snap_price = float(self.low[idx])
            snap_label = 'L'

        peak = {
            'bar_index': int(idx),
            'timestamp': float(self.timestamps[idx]),
            'time_utc': self.dt_stamps[idx].strftime('%H:%M:%S'),
            'price': snap_price,
            'close': float(self.close[idx]),
            'high': float(self.high[idx]),
            'low': float(self.low[idx]),
            '_snap': snap_label,
            '_direction_hint': direction,
            'tf': self.tf,
        }
        self.peaks.append(peak)

        # Draw marker at the snapped extreme
        m = self.ax.scatter(self.dt_stamps[idx], snap_price,
                            marker='D', c='cyan', s=150, zorder=10,
                            edgecolors='black', lw=1.5)
        label = self.ax.text(self.dt_stamps[idx], snap_price + 2,
                             f'#{len(self.peaks)} {snap_label}\n{snap_price:.2f}',
                             fontsize=7, ha='center', color='cyan', fontweight='bold')
        self._peak_markers.append((m, label))

        print(f'  Peak #{len(self.peaks)}: {self.dt_stamps[idx].strftime("%H:%M:%S")} '
              f'@ {self.close[idx]:.2f}')

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        """Handle keypress — D=delete last, Q=save+quit, S=flip direction."""
        if event.key == 'd' and self.peaks:
            # Delete last peak
            removed = self.peaks.pop()
            m, label = self._peak_markers.pop()
            m.remove()
            label.remove()
            print(f'  Deleted peak @ {removed["time_utc"]}')
            self.fig.canvas.draw_idle()

        elif event.key == 'q':
            self._save()
            plt.close(self.fig)

    def _save(self):
        """Save marked peaks to JSON."""
        os.makedirs(SEEDS_DIR, exist_ok=True)
        path = os.path.join(SEEDS_DIR, f'human_peaks_{self.date_str}_{self.tf}.json')
        out = {
            'date': self.date_str,
            'tf': self.tf,
            'n_peaks': len(self.peaks),
            'peaks': self.peaks,
        }
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'\nSaved {len(self.peaks)} peaks to {path}')

    def run(self):
        """Show interactive chart."""
        self.fig, self.ax = plt.subplots(figsize=(24, 10))

        # Candlestick-style (simplified: just bars)
        for i in range(len(self.close)):
            color = 'green' if self.close[i] >= (self.close[i-1] if i > 0 else self.close[i]) else 'red'
            self.ax.plot([self.dt_stamps[i], self.dt_stamps[i]],
                         [self.low[i], self.high[i]], color=color, lw=1, alpha=0.6)
            self.ax.plot(self.dt_stamps[i], self.close[i], '.', color=color, markersize=3)

        self.ax.set_title(f'{self.date_str} ({self.tf}) — Click to mark peaks | '
                          f'D=delete last | Q=save+quit',
                          fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Price')
        self.ax.set_xlabel('Time (UTC)')
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax.grid(True, alpha=0.3)

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        print(f'\nPeak Marker — {self.date_str} ({self.tf})')
        print(f'  Click to mark peaks (location only, no direction)')
        print(f'  D = delete last mark')
        print(f'  Q = save and quit')
        print()

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Manual peak marker')
    parser.add_argument('--date', required=True, help='Date (YYYY-MM-DD) or range (YYYY-MM-DD:YYYY-MM-DD)')
    parser.add_argument('--tf', default='5m',
                        choices=['1m', '3m', '5m', '15m', '30m', '1h', '4h'],
                        help='Timeframe for chart (default: 5m)')
    parser.add_argument('--data', default='DATA/ATLAS',
                        help='ATLAS directory (default: DATA/ATLAS)')
    args = parser.parse_args()

    # Load data for the date (supports single date or range with ':')
    if ':' in args.date:
        _start, _end = args.date.split(':')
    else:
        _start = _end = args.date

    # Collect unique months from date range
    _months = set()
    _s = pd.Timestamp(_start)
    _e = pd.Timestamp(_end)
    while _s <= _e:
        _months.add(f'{_s.year}_{_s.month:02d}')
        _s += pd.DateOffset(months=1)
    _months = sorted(_months)

    df = load_atlas_tf(args.data, args.tf, _months)
    if df is None or len(df) == 0:
        df = load_atlas_tf('DATA/ATLAS_OOS', args.tf, _months)
    if df is None or len(df) == 0:
        print(f'No data for {args.date} at {args.tf}')
        sys.exit(1)

    # Filter to requested date range
    df['_dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
    df = df[(df['_dt'].dt.strftime('%Y-%m-%d') >= _start) &
            (df['_dt'].dt.strftime('%Y-%m-%d') <= _end)].copy()
    df = df.drop(columns=['_dt']).reset_index(drop=True)
    if len(df) == 0:
        print(f'No bars for {args.date} after date filter')
        sys.exit(1)

    print(f'Loaded {len(df)} bars for {args.date} at {args.tf}')

    _label = _start if _start == _end else f'{_start}_to_{_end}'
    marker = PeakMarker(df, _label, args.tf)
    marker.run()


if __name__ == '__main__':
    main()
