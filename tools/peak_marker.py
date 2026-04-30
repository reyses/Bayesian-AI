#!/usr/bin/env python
"""
Peak Marker — Manually mark peaks on a price chart.

Single click = mark a peak. Direction auto-detected from local price action.
Press D to delete last mark, Q to save+quit.

Saves to DATA/regime_seeds/human_peaks_{date}_{tf}.json

After marking, run peak_marker_analysis.py to analyze
the 1s data before each marked peak.

Macro-scale TFs supported (added 2026-04-29):
    --tf 1W   weekly bars resampled from 1D (W-FRI close convention, ~60 bars / 14mo)
    --tf 1D   daily bars (uses DATA/ATLAS/1D/*.parquet directly, ~348 bars / 14mo)
    --tf 4h   4-hour bars (uses DATA/ATLAS/4h/*.parquet directly, ~1,743 bars / 14mo)

When --tf is 1W, 1D or 4h and --date is omitted, the full ATLAS range loads
automatically — convenient for marking the macro picture across all data.
You can zoom (Z key) and pan (X key) within the full chart.

Human level overlay (added 2026-04-29):
    Hand-marked support/resistance levels from DATA/levels/levels_*.json
    are drawn as horizontal lines on the chart. Disable with --no-levels.

Usage:
    # Macro marking (full dataset)
    python tools/peak_marker.py --tf 1W
    python tools/peak_marker.py --tf 1D
    python tools/peak_marker.py --tf 1W --no-levels

    # Day/week-specific
    python tools/peak_marker.py --date 2025-07-14
    python tools/peak_marker.py --date 2025-07-14 --tf 5m
    python tools/peak_marker.py --date 2025-07-14 --tf 1h
    python tools/peak_marker.py --data DATA/ATLAS_OOS --date 2026-02-05 --tf 5m
    python tools/peak_marker.py --date 2025-01-01:2026-03-21 --tf 1D
"""

import argparse
import glob
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
LEVELS_DIR = 'DATA/levels'


# ─── Macro TF loaders (added 2026-04-29) ─────────────────────────────────────

def _load_full_1d(data_dir: str) -> pd.DataFrame:
    """Load every 1D parquet under data_dir/1D/ — sorted, deduped.
    Used for both --tf 1D directly AND as the source for 1W resampling."""
    tf_dir = os.path.join(data_dir, '1D')
    if not os.path.isdir(tf_dir):
        return pd.DataFrame()
    files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f))
        except Exception as e:
            print(f'  WARN read fail {f}: {e}')
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = df['timestamp'].astype('int64') // 10**9
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    return df


def _resample_to_1w(df_1d: pd.DataFrame) -> pd.DataFrame:
    """Resample 1D OHLCV to 1W bars using W-FRI convention (Friday close)."""
    if df_1d is None or df_1d.empty:
        return pd.DataFrame()
    work = df_1d.copy()
    work['dt'] = pd.to_datetime(work['timestamp'], unit='s', utc=True)
    work = work.set_index('dt')
    weekly = work.resample('W-FRI').agg(
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    ).dropna(subset=['open', 'close']).reset_index()
    weekly['timestamp'] = (weekly['dt'].astype('int64') // 10**9).astype(int)
    weekly = weekly[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    return weekly.reset_index(drop=True)


def load_macro_tf(data_dir: str, tf: str) -> pd.DataFrame:
    """Load full-range OHLCV for macro TFs: '1W' resamples from 1D, '1D' loads directly."""
    df_1d = _load_full_1d(data_dir)
    if df_1d.empty:
        return df_1d
    if tf == '1D':
        return df_1d
    if tf == '1W':
        return _resample_to_1w(df_1d)
    raise ValueError(f'load_macro_tf only handles 1W/1D, got {tf}')


# ─── Human levels loader (added 2026-04-29) ──────────────────────────────────

def load_levels_in_range(start_dt: datetime, end_dt: datetime,
                          levels_dir: str = LEVELS_DIR) -> list[dict]:
    """Load all hand-marked support/resistance levels from monthly JSON files
    whose date falls within or adjacent to [start_dt, end_dt].

    Each levels file is dated for the START of a month (levels_YYYY-MM-01.json).
    Returns flat list of {'price': float, 'type': 'support'|'resistance',
                          'src_date': 'YYYY-MM-DD', 'color': str}.
    """
    if not os.path.isdir(levels_dir):
        return []
    out = []
    for f in sorted(glob.glob(os.path.join(levels_dir, 'levels_*.json'))):
        try:
            with open(f) as fh:
                data = json.load(fh)
        except Exception:
            continue
        src_date_str = data.get('date', '')
        try:
            src_dt = pd.Timestamp(src_date_str).tz_localize('UTC')
        except Exception:
            continue
        # Include if file date is within range or within 1 month either side
        if src_dt < start_dt - pd.Timedelta(days=31):
            continue
        if src_dt > end_dt + pd.Timedelta(days=31):
            continue
        for lvl in data.get('levels', []):
            out.append({
                'price': float(lvl['price']),
                'type': lvl.get('type', 'unknown'),
                'src_date': src_date_str,
                'color': lvl.get('color', '#888888'),
            })
    return out


class PeakMarker:
    """Manual peak marker with crosshair."""

    def __init__(self, df, date_str, tf='5m', levels=None):
        self.df = df
        self.date_str = date_str
        self.tf = tf
        self.levels = levels or []  # list of {'price', 'type', 'src_date', 'color'}

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
        """Handle click — mark peak. Ignores clicks when toolbar is active (zoom/pan)."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # left click only
            return
        # Skip if toolbar zoom/pan mode is active
        toolbar = self.fig.canvas.toolbar
        if toolbar and hasattr(toolbar, 'mode') and toolbar.mode:
            return  # toolbar is in zoom or pan mode

        click_num = event.xdata
        if click_num is None:
            return

        idx = self._find_nearest_bar(click_num)

        # Check if clicking near an existing peak — toggle delete
        for pi, existing in enumerate(self.peaks):
            if existing['bar_index'] == idx:
                # Remove this peak
                self.peaks.pop(pi)
                m, label = self._peak_markers.pop(pi)
                m.remove()
                label.remove()
                print(f'  Removed peak @ {existing["time_utc"]}')
                self.fig.canvas.draw_idle()
                return

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

        elif event.key == 'z':
            # Zoom in 50% centered on current view
            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()
            xc = (xl[0] + xl[1]) / 2
            yc = (yl[0] + yl[1]) / 2
            xr = (xl[1] - xl[0]) * 0.25
            yr = (yl[1] - yl[0]) * 0.25
            self.ax.set_xlim(xc - xr, xc + xr)
            self.ax.set_ylim(yc - yr, yc + yr)
            self.fig.canvas.draw_idle()

        elif event.key == 'a':
            # Zoom out 2x centered on current view
            xl = self.ax.get_xlim()
            yl = self.ax.get_ylim()
            xc = (xl[0] + xl[1]) / 2
            yc = (yl[0] + yl[1]) / 2
            xr = (xl[1] - xl[0])
            yr = (yl[1] - yl[0])
            self.ax.set_xlim(xc - xr, xc + xr)
            self.ax.set_ylim(yc - yr, yc + yr)
            self.fig.canvas.draw_idle()

        elif event.key == 'x':
            # Toggle pan mode
            self.fig.canvas.toolbar.pan()

        elif event.key == 'h':
            # Reset view to full extent
            self.ax.autoscale()
            self.fig.canvas.draw_idle()

        elif event.key == 'q':
            self._save()
            plt.close(self.fig)

    def _save(self):
        """Save marked peaks to JSON + screenshot of the chart."""
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

        # Save screenshot
        ss_dir = os.path.join('reports', 'screenshots')
        os.makedirs(ss_dir, exist_ok=True)
        ss_path = os.path.join(ss_dir, f'peak_marker_{self.date_str}_{self.tf}.png')
        self.fig.savefig(ss_path, dpi=150)
        print(f'Screenshot: {ss_path}')

    def _load_existing(self):
        """Load previously saved peaks for this date+tf."""
        path = os.path.join(SEEDS_DIR, f'human_peaks_{self.date_str}_{self.tf}.json')
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        prev = data.get('peaks', [])
        if not prev:
            return
        print(f'  Loaded {len(prev)} existing peaks from {path}')
        for peak in prev:
            self.peaks.append(peak)
            idx = peak['bar_index']
            if idx >= len(self.close):
                continue
            snap_price = peak.get('price', self.close[idx])
            snap_label = peak.get('_snap', '?')
            m = self.ax.scatter(self.dt_stamps[idx], snap_price,
                                marker='D', c='cyan', s=150, zorder=10,
                                edgecolors='black', lw=1.5)
            label = self.ax.text(self.dt_stamps[idx], snap_price + 2,
                                 f'#{len(self.peaks)} {snap_label}\n{snap_price:.2f}',
                                 fontsize=7, ha='center', color='cyan', fontweight='bold')
            self._peak_markers.append((m, label))

    def _date_formatter_for_tf(self):
        """Pick the right date axis format for the chart's TF.
        Macro TFs (1W/1D/4h) need wider date labels; sub-hour shows time-only."""
        if self.tf == '1W':
            return mdates.DateFormatter('%Y-%m-%d')
        if self.tf == '1D':
            return mdates.DateFormatter('%Y-%m-%d')
        if self.tf == '4h':
            return mdates.DateFormatter('%m-%d %H:%M')
        if self.tf in ('1h', '30m'):
            return mdates.DateFormatter('%m-%d %H:%M')
        return mdates.DateFormatter('%H:%M')

    def _draw_levels(self):
        """Overlay hand-marked support/resistance levels as horizontal lines."""
        if not self.levels:
            return
        # Determine x-axis bounds for line annotation (just first/last bar dt)
        if not self.dt_stamps:
            return
        x_left = self.dt_stamps[0]
        x_right = self.dt_stamps[-1]
        # Get y-range so we can clip levels far outside view
        y_min = float(np.min(self.low))
        y_max = float(np.max(self.high))
        y_pad = (y_max - y_min) * 0.1
        y_min -= y_pad; y_max += y_pad
        n_drawn = 0
        for lvl in self.levels:
            p = lvl['price']
            if p < y_min or p > y_max:
                continue
            color = '#CC0000' if lvl['type'] == 'resistance' else (
                '#0066CC' if lvl['type'] == 'support' else '#888888'
            )
            self.ax.axhline(p, color=color, lw=0.8, alpha=0.45, ls='-', zorder=1)
            # Tag price + source month at right edge
            self.ax.text(x_right, p, f' {p:.0f} {lvl["src_date"][:7]}',
                          fontsize=7, color=color, alpha=0.6, va='center', ha='left', zorder=1)
            n_drawn += 1
        print(f'  Drew {n_drawn}/{len(self.levels)} levels in chart range')

    def run(self):
        """Show interactive chart."""
        self.fig, self.ax = plt.subplots(figsize=(24, 10))

        # Candlestick-style (simplified: just bars)
        for i in range(len(self.close)):
            color = 'green' if self.close[i] >= (self.close[i-1] if i > 0 else self.close[i]) else 'red'
            self.ax.plot([self.dt_stamps[i], self.dt_stamps[i]],
                         [self.low[i], self.high[i]], color=color, lw=1, alpha=0.6)
            self.ax.plot(self.dt_stamps[i], self.close[i], '.', color=color, markersize=3)

        # Overlay human levels (after bars, before peak markers)
        self._draw_levels()

        title_levels = f' | {len(self.levels)} levels' if self.levels else ''
        self.ax.set_title(f'{self.date_str} ({self.tf}){title_levels} — Click to mark | '
                          f'Z=zoom in | A=zoom out | X=pan | H=reset | D=del | Q=save',
                          fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Price')
        self.ax.set_xlabel('Date / Time (UTC)')
        self.ax.xaxis.set_major_formatter(self._date_formatter_for_tf())
        if self.tf in ('1W', '1D'):
            # Tilt date labels for readability on macro charts
            for label in self.ax.get_xticklabels():
                label.set_rotation(30)
                label.set_ha('right')
        self.ax.grid(True, alpha=0.3)

        # Load previous marks if re-running
        self._load_existing()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        print(f'\nPeak Marker — {self.date_str} ({self.tf})')
        print(f'  Bars: {len(self.close)}  |  Levels overlaid: {len(self.levels)}')
        print(f'  Click = mark peak (location only)')
        print(f'  Z = zoom in | A = zoom out | X = pan | H = reset view')
        print(f'  D = delete last mark')
        print(f'  Q = save and quit')
        print()

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Manual peak marker')
    parser.add_argument('--date', default=None,
                        help='Date YYYY-MM-DD or range YYYY-MM-DD:YYYY-MM-DD. '
                             'Optional for --tf 1W/1D (defaults to full ATLAS range).')
    parser.add_argument('--tf', default='5m',
                        choices=['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1D', '1W'],
                        help='Timeframe for chart (default: 5m). 1W resamples from 1D.')
    parser.add_argument('--data', default='DATA/ATLAS',
                        help='ATLAS directory (default: DATA/ATLAS)')
    parser.add_argument('--no-levels', action='store_true',
                        help='Disable human levels overlay')
    args = parser.parse_args()

    # Full-range TFs: auto-load full ATLAS range when --date omitted.
    # 1W resamples from 1D; 1D and 4h load their own parquet folders directly.
    full_range_tfs = ('1D', '1W', '4h')
    is_macro = args.tf in full_range_tfs

    # ── Load OHLCV data ─────────────────────────────────────────────────
    if is_macro:
        # Macro TFs: load full ATLAS range; date filter optional
        if args.tf == '1W':
            df = load_macro_tf(args.data, '1W')
            if df is None or len(df) == 0:
                df = load_macro_tf('DATA/ATLAS_OOS', '1W')
        elif args.tf == '1D':
            df = load_macro_tf(args.data, '1D')
            if df is None or len(df) == 0:
                df = load_macro_tf('DATA/ATLAS_OOS', '1D')
        else:  # 4h
            df = load_atlas_tf(args.data, args.tf, months=None)
            if df is None or len(df) == 0:
                df = load_atlas_tf('DATA/ATLAS_OOS', args.tf, months=None)
        if df is None or len(df) == 0:
            print(f'No {args.tf} data found in {args.data}/{args.tf} (and OOS fallback)')
            sys.exit(1)

        # Optional date filter at macro
        if args.date:
            if ':' in args.date:
                _start, _end = args.date.split(':')
            else:
                _start = _end = args.date
            df['_dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            df = df[(df['_dt'].dt.strftime('%Y-%m-%d') >= _start) &
                    (df['_dt'].dt.strftime('%Y-%m-%d') <= _end)].copy()
            df = df.drop(columns=['_dt']).reset_index(drop=True)
        else:
            # Default range = data extent
            _all_dt = pd.to_datetime(df['timestamp'], unit='s', utc=True)
            _start = _all_dt.min().strftime('%Y-%m-%d')
            _end = _all_dt.max().strftime('%Y-%m-%d')

        if len(df) == 0:
            print(f'No bars after date filter')
            sys.exit(1)

        print(f'Loaded {len(df)} {args.tf} bars  ({_start} -> {_end})')

    else:
        # Sub-day TFs: original behavior, --date is required
        if not args.date:
            print(f'--date is required for --tf {args.tf} (only 1D/1W support full-range default)')
            sys.exit(1)

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

    # ── Load human levels in range (unless disabled) ───────────────────
    levels = []
    if not args.no_levels:
        try:
            start_dt = pd.Timestamp(_start, tz='UTC')
            end_dt = pd.Timestamp(_end, tz='UTC')
            levels = load_levels_in_range(start_dt, end_dt, LEVELS_DIR)
            print(f'Loaded {len(levels)} human-marked levels for overlay')
        except Exception as e:
            print(f'  WARN levels load failed: {e}')

    _label = _start if _start == _end else f'{_start}_to_{_end}'
    marker = PeakMarker(df, _label, args.tf, levels=levels)
    marker.run()


if __name__ == '__main__':
    main()
