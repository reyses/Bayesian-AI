"""
Interactive Level Drawing Tool.

Displays 1m price chart for a given date. Click to draw horizontal
support/resistance lines. Right-click to remove the nearest line.
Close the window to save.

Lines are saved to JSON for the CNN to learn from.

Usage:
  python tools/draw_levels.py --date 2026-03-18
  python tools/draw_levels.py --date 2026-03-18 --lookback 2026-03-16

Output:
  reports/findings/levels_YYYY-MM-DD.json
  reports/findings/levels_YYYY-MM-DD.png
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TICK = 0.25
ATLAS_ROOT = 'DATA/ATLAS'


def load_data(date_str, lookback_str=None, trade_end=None):
    """Load 1m price data for the trade week + lookback context."""
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    trade_start_ts = pd.Timestamp(date_str).timestamp()
    trade_end_ts = pd.Timestamp(trade_end).timestamp() + 86400 if trade_end else trade_start_ts + 86400

    # Trade week
    df_day = df[(df['timestamp'] >= trade_start_ts) & (df['timestamp'] < trade_end_ts)].reset_index(drop=True)

    # Lookback context (previous week)
    df_context = None
    if lookback_str:
        lb_start = pd.Timestamp(lookback_str).timestamp()
        df_context = df[(df['timestamp'] >= lb_start) & (df['timestamp'] < trade_start_ts)].reset_index(drop=True)

    return df_day, df_context


class LevelDrawer:
    """Interactive matplotlib figure for drawing horizontal levels.

    Left-click: add level. Right-click: remove nearest.
    Zoom button cycles through timeframes (1m -> 5m -> 15m -> 1h -> 4h -> 1m).
    Candlestick display with wicks.
    """

    TF_CYCLE = ['4h', '1h', '15m', '5m', '1m']

    def __init__(self, df_day, df_context=None, date_str='', lookback_str=None, trade_end=None):
        self.date_str = date_str
        self.lookback_str = lookback_str
        self.trade_end = trade_end or date_str
        self.levels = []
        self.line_objects = []
        self.label_objects = []
        self.current_tf_idx = 0  # start at 1m

        # Store raw data per TF for cycling
        self.tf_data = {}
        # Always store 1m as fallback
        self.tf_data['1m'] = {'day': df_day, 'context': df_context}
        self._load_all_tfs(date_str, lookback_str)

        # Start with 4h (first in TF_CYCLE)
        start_tf = self.TF_CYCLE[0]
        if start_tf in self.tf_data:
            self._build_display_data(start_tf)
        else:
            self._build_display_data('1m', df_day, df_context)

        # Create figure (30% smaller, centered)
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 7))

        # Set default save directory for the toolbar save button
        examples_dir = os.path.abspath('examples')
        os.makedirs(examples_dir, exist_ok=True)
        matplotlib.rcParams['savefig.directory'] = examples_dir

        mng = self.fig.canvas.manager
        try:
            mng.window.wm_geometry('+200+100')
        except Exception:
            pass

        # Override zoom button to cycle TFs
        toolbar = self.fig.canvas.manager.toolbar
        if hasattr(toolbar, '_actions'):
            # Try to find and replace zoom action
            pass
        # Interaction state
        self._dragging = False
        self._drag_start = None
        self._selected_idx = None
        self._preview_line = None

        # Connect events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_drag)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)

        self._draw_chart()

    def _load_all_tfs(self, date_str, lookback_str):
        """Pre-load all TF data for cycling. Loads full week range."""
        lb_start_ts = pd.Timestamp(lookback_str).timestamp() if lookback_str else 0
        trade_end_ts = pd.Timestamp(self.trade_end).timestamp() + 86400

        for tf in self.TF_CYCLE:
            try:
                # May span multiple months — load all relevant parquets
                files = sorted(glob.glob(os.path.join(ATLAS_ROOT, tf, '*.parquet')))
                if not files:
                    continue
                dfs = [pd.read_parquet(f) for f in files]
                df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

                # Filter to lookback+trade range
                df = df[(df['timestamp'] >= lb_start_ts) &
                        (df['timestamp'] < trade_end_ts)].reset_index(drop=True)

                if len(df) == 0:
                    continue

                # All bars are analysis week — no context/trade split
                if len(df) > 0:
                    self.tf_data[tf] = {'day': df, 'context': None}
                    print(f"  {tf}: {len(df)} bars")

                del df
            except Exception as e:
                print(f"  {tf}: failed to load ({e})")

    def _build_display_data(self, tf, df_day=None, df_context=None):
        """Set current display data from TF."""
        if tf in self.tf_data:
            df_day = self.tf_data[tf]['day']
            df_context = self.tf_data[tf]['context']
        if df_day is None:
            return

        if df_context is not None and len(df_context) > 0:
            self.df_full = pd.concat([df_context, df_day], ignore_index=True)
            self.trade_start_idx = len(df_context)
        else:
            self.df_full = df_day
            self.trade_start_idx = 0

        self.opens = self.df_full['open'].values
        self.prices = self.df_full['close'].values
        self.highs = self.df_full['high'].values
        self.lows = self.df_full['low'].values
        self.timestamps = self.df_full['timestamp'].values
        self.n = len(self.prices)

        # Use bar index for x-axis (no gaps) — timestamps used for labels only
        self.x = np.arange(self.n, dtype=float)

    def _draw_chart(self):
        """Draw candlestick chart with current TF data."""
        self.ax.clear()
        tf = self.TF_CYCLE[self.current_tf_idx]

        self.fig.suptitle(
            f'Draw Levels — {self.date_str} [{tf.upper()}] | '
            f'left-click=add, right-click=remove, Z=cycle TF, close=save',
            fontsize=13, fontweight='bold')

        # Draw candlesticks
        candle_width = np.median(np.diff(self.x)) * 0.6 if len(self.x) > 1 else 0.5

        for i in range(self.n):
            is_context = (i < self.trade_start_idx)
            o, c, h, l = self.opens[i], self.prices[i], self.highs[i], self.lows[i]
            x = self.x[i]

            if is_context:
                body_color = '#CCCCCC' if c >= o else '#AAAAAA'
                wick_color = '#BBBBBB'
                alpha = 0.4
            else:
                body_color = '#26A69A' if c >= o else '#EF5350'  # green / red
                wick_color = '#555555'
                alpha = 0.9

            # Wick (high-low line)
            self.ax.plot([x, x], [l, h], color=wick_color, linewidth=0.5, alpha=alpha)

            # Body (open-close rectangle)
            body_bottom = min(o, c)
            body_height = abs(c - o)
            if body_height < TICK:
                body_height = TICK  # minimum visible body
            self.ax.bar(x, body_height, bottom=body_bottom, width=candle_width,
                        color=body_color, edgecolor=wick_color, linewidth=0.3, alpha=alpha)

        # Trade day separator
        if self.trade_start_idx > 0:
            self.ax.axvline(x=self.x[self.trade_start_idx], color='black',
                            linestyle=':', alpha=0.3, linewidth=1)

        # Redraw all saved levels
        self.line_objects = []
        self.label_objects = []
        for level in self.levels:
            line = self.ax.axhline(y=level['price'], color=level['color'],
                                    linewidth=1.5, alpha=0.7,
                                    linestyle='--' if level['type'] == 'resistance' else '-')
            txt = self.ax.text(self.x[-1] + (self.x[-1] - self.x[0]) * 0.01,
                               level['price'], f'{level["price"]:.2f}',
                               fontsize=8, color=level['color'], va='center')
            self.line_objects.append(line)
            self.label_objects.append(txt)

        self.ax.set_xlabel('')
        self.ax.set_ylabel('Price')

        # Custom x-axis: show timestamps at regular intervals, no gaps
        n_ticks = min(20, self.n)
        tick_step = max(1, self.n // n_ticks)
        tick_positions = list(range(0, self.n, tick_step))
        tick_labels = []
        tf = self.TF_CYCLE[self.current_tf_idx]
        for i in tick_positions:
            dt = pd.to_datetime(self.timestamps[i], unit='s')
            if tf in ('4h', '1h'):
                tick_labels.append(dt.strftime('%m/%d %H:%M'))
            else:
                tick_labels.append(dt.strftime('%H:%M'))
        self.ax.set_xticks(tick_positions)
        self.ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)

        # Day boundaries: vertical lines where the date changes
        for i in range(1, self.n):
            dt_prev = pd.to_datetime(self.timestamps[i-1], unit='s').date()
            dt_curr = pd.to_datetime(self.timestamps[i], unit='s').date()
            if dt_curr != dt_prev:
                self.ax.axvline(x=i, color='navy', linewidth=0.8, alpha=0.3, linestyle='-')

        self.ax.grid(True, which='major', alpha=0.2, linewidth=0.4)

        # Status
        self.status_text = self.ax.text(
            0.02, 0.02, '', transform=self.ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self._update_status()

        self.fig.canvas.draw()

    def _on_key(self, event):
        if event.key == 'z':
            # Cycle to next TF
            self.current_tf_idx = (self.current_tf_idx + 1) % len(self.TF_CYCLE)
            tf = self.TF_CYCLE[self.current_tf_idx]
            attempts = 0
            while tf not in self.tf_data and attempts < len(self.TF_CYCLE):
                self.current_tf_idx = (self.current_tf_idx + 1) % len(self.TF_CYCLE)
                tf = self.TF_CYCLE[self.current_tf_idx]
                attempts += 1
            if tf in self.tf_data:
                self._build_display_data(tf)
                self._draw_chart()
                print(f"  Switched to {tf.upper()} ({self.n} bars)")

        elif event.key == 'd' or event.key == 'delete':
            # Delete: if a level is selected, delete it. Otherwise delete nearest to cursor.
            if not self.levels:
                return
            if self._selected_idx is not None:
                idx = self._selected_idx
            elif event.inaxes == self.ax and event.ydata is not None:
                dists = [abs(l['price'] - event.ydata) for l in self.levels]
                idx = np.argmin(dists)
                threshold = (self.prices.max() - self.prices.min()) * 0.02
                if dists[idx] > threshold:
                    return
            else:
                return

            # Remove
            removed = self.levels.pop(idx)
            self.line_objects[idx].remove()
            self.line_objects.pop(idx)
            self.label_objects[idx].remove()
            self.label_objects.pop(idx)
            self._selected_idx = None
            print(f"  Deleted {removed['type']} at {removed['price']:.2f}")
            self._update_status()
            self.fig.canvas.draw()

    def _on_click(self, event):
        """Click to start drawing or select existing level."""
        if event.inaxes != self.ax:
            return
        if event.button != MouseButton.LEFT:
            return

        price = event.ydata
        self._drag_start = price

        # Check if clicking near an existing level (select it)
        if self.levels:
            dists = [abs(l['price'] - price) for l in self.levels]
            nearest = np.argmin(dists)
            threshold = (self.prices.max() - self.prices.min()) * 0.01
            if dists[nearest] < threshold:
                # Select this level for dragging
                self._selected_idx = nearest
                self._dragging = True
                # Highlight selected
                self.line_objects[nearest].set_linewidth(3)
                self.line_objects[nearest].set_alpha(1.0)
                self._update_status()
                self.fig.canvas.draw()
                return

        # Not near existing — start new line
        self._dragging = True
        self._selected_idx = None
        self._preview_line = self.ax.axhline(y=price, color='gray',
                                              linewidth=1, alpha=0.5, linestyle=':')
        self.fig.canvas.draw()

    def _on_drag(self, event):
        """Drag to position level."""
        if not self._dragging or event.inaxes != self.ax:
            return

        price = event.ydata

        if self._selected_idx is not None:
            # Moving existing level
            snapped = round(price / TICK) * TICK
            idx = self._selected_idx
            self.levels[idx]['price'] = snapped
            self.line_objects[idx].set_ydata([snapped, snapped])
            self.label_objects[idx].set_position(
                (self.x[-1] + (self.x[-1] - self.x[0]) * 0.01, snapped))
            self.label_objects[idx].set_text(f'{snapped:.2f}')
        elif hasattr(self, '_preview_line') and self._preview_line:
            # Preview new line
            snapped = round(price / TICK) * TICK
            self._preview_line.set_ydata([snapped, snapped])

        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        """Release to place level."""
        if not self._dragging:
            return
        self._dragging = False

        if event.inaxes != self.ax:
            # Released outside — cancel
            if hasattr(self, '_preview_line') and self._preview_line:
                self._preview_line.remove()
                self._preview_line = None
            if self._selected_idx is not None:
                self.line_objects[self._selected_idx].set_linewidth(1.5)
                self.line_objects[self._selected_idx].set_alpha(0.7)
                self._selected_idx = None
            self.fig.canvas.draw()
            return

        price = event.ydata
        snapped = round(price / TICK) * TICK

        if self._selected_idx is not None:
            # Finished moving existing level — update type based on position
            mid = (self.prices[self.trade_start_idx:].max() +
                   self.prices[self.trade_start_idx:].min()) / 2
            idx = self._selected_idx
            if snapped > mid:
                self.levels[idx]['type'] = 'resistance'
                self.levels[idx]['color'] = '#CC0000'
            else:
                self.levels[idx]['type'] = 'support'
                self.levels[idx]['color'] = '#0066CC'
            self.line_objects[idx].set_color(self.levels[idx]['color'])
            self.line_objects[idx].set_linestyle('--' if self.levels[idx]['type'] == 'resistance' else '-')
            self.line_objects[idx].set_linewidth(1.5)
            self.line_objects[idx].set_alpha(0.7)
            self.label_objects[idx].set_color(self.levels[idx]['color'])
            self._selected_idx = None
        else:
            # Place new level
            if hasattr(self, '_preview_line') and self._preview_line:
                self._preview_line.remove()
                self._preview_line = None

            mid = (self.prices[self.trade_start_idx:].max() +
                   self.prices[self.trade_start_idx:].min()) / 2
            if snapped > mid:
                level_type = 'resistance'
                color = '#CC0000'
            else:
                level_type = 'support'
                color = '#0066CC'

            self.levels.append({'price': snapped, 'type': level_type, 'color': color})
            line = self.ax.axhline(y=snapped, color=color, linewidth=1.5, alpha=0.7,
                                    linestyle='--' if level_type == 'resistance' else '-')
            txt = self.ax.text(self.x[-1] + (self.x[-1] - self.x[0]) * 0.01,
                               snapped, f'{snapped:.2f}',
                               fontsize=8, color=color, va='center')
            self.line_objects.append(line)
            self.label_objects.append(txt)

        self._update_status()
        self.fig.canvas.draw()

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return

        price = event.ydata

        if event.button == MouseButton.LEFT:
            # Add level — determine if support or resistance based on
            # whether price is above or below current levels center
            mid = (self.prices[self.trade_start_idx:].max() +
                   self.prices[self.trade_start_idx:].min()) / 2
            if price > mid:
                level_type = 'resistance'
                color = '#CC0000'  # red
            else:
                level_type = 'support'
                color = '#0066CC'  # blue

            self.levels.append({'price': round(price / TICK) * TICK,
                                'type': level_type, 'color': color})
            line = self.ax.axhline(y=round(price / TICK) * TICK, color=color,
                                    linewidth=1.5, alpha=0.7,
                                    linestyle='--' if level_type == 'resistance' else '-')
            # Add price label
            self.ax.text(self.x[-1] + 5, round(price / TICK) * TICK,
                         f'{round(price / TICK) * TICK:.2f}',
                         fontsize=8, color=color, va='center')
            self.line_objects.append(line)

        elif event.button == MouseButton.RIGHT:
            # Remove nearest level
            if self.levels:
                dists = [abs(l['price'] - price) for l in self.levels]
                nearest = np.argmin(dists)
                if dists[nearest] < (self.prices.max() - self.prices.min()) * 0.02:
                    self.levels.pop(nearest)
                    self.line_objects[nearest].remove()
                    self.line_objects.pop(nearest)

        self._update_status()
        self.fig.canvas.draw()

    def _update_status(self):
        n_r = sum(1 for l in self.levels if l['type'] == 'resistance')
        n_s = sum(1 for l in self.levels if l['type'] == 'support')
        self.status_text.set_text(f'Levels: {len(self.levels)} (R={n_r} S={n_s}) | '
                                   f'Left-click=add, Right-click=remove')

    def save(self):
        """Save levels to JSON (data dir) and PNG (examples dir)."""
        # PNG to examples/
        png_dir = 'examples'
        os.makedirs(png_dir, exist_ok=True)
        png_path = os.path.join(png_dir, f'levels_{self.date_str}.png')
        self.fig.savefig(png_path, dpi=150, bbox_inches='tight')

        # JSON to DATA/levels/
        json_dir = 'DATA/levels'
        os.makedirs(json_dir, exist_ok=True)
        data = {
            'date': self.date_str,
            'levels': sorted(self.levels, key=lambda l: l['price'], reverse=True),
            'n_resistance': sum(1 for l in self.levels if l['type'] == 'resistance'),
            'n_support': sum(1 for l in self.levels if l['type'] == 'support'),
        }
        json_path = os.path.join(json_dir, f'levels_{self.date_str}.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(self.levels)} levels:")
        print(f"  PNG:  {png_path}")
        print(f"  JSON: {json_path}")
        for l in sorted(self.levels, key=lambda x: x['price'], reverse=True):
            print(f"    {l['type']:>10}: {l['price']:.2f}")

    def show(self):
        """Show interactive window."""
        plt.show()
        self.save()


def main():
    parser = argparse.ArgumentParser(description='Interactive level drawing tool')
    parser.add_argument('--date', default='2026-03-16', help='Any date in the trade week (YYYY-MM-DD)')
    args = parser.parse_args()

    # CME session: Sunday 5PM CT (23:00 UTC) to Friday 4PM CT (21:00 UTC)
    CME_OPEN_HOUR = 23   # Sunday 23:00 UTC
    CME_CLOSE_HOUR = 21  # Friday 21:00 UTC

    trade_dt = pd.Timestamp(args.date)

    # Find Sunday that starts the trade week containing args.date
    days_since_sunday = (trade_dt.dayofweek + 1) % 7  # Sun=0
    trade_sunday = trade_dt - pd.Timedelta(days=days_since_sunday)
    trade_open = pd.Timestamp(f'{trade_sunday.date()} {CME_OPEN_HOUR}:00:00')
    trade_friday = trade_sunday + pd.Timedelta(days=5)
    trade_close = pd.Timestamp(f'{trade_friday.date()} {CME_CLOSE_HOUR}:00:00')

    # Analysis = previous CME week
    prev_sunday = trade_sunday - pd.Timedelta(days=7)
    analysis_open = pd.Timestamp(f'{prev_sunday.date()} {CME_OPEN_HOUR}:00:00')
    prev_friday = prev_sunday + pd.Timedelta(days=5)
    analysis_close = pd.Timestamp(f'{prev_friday.date()} {CME_CLOSE_HOUR}:00:00')

    analysis_start = analysis_open.strftime('%Y-%m-%d')
    trade_end = trade_close.strftime('%Y-%m-%d')

    print(f"Analysis week: {analysis_open.strftime('%a %m/%d %H:%M')} -> {analysis_close.strftime('%a %m/%d %H:%M')} UTC")
    print(f"Trade week:    {trade_open.strftime('%a %m/%d %H:%M')} -> {trade_close.strftime('%a %m/%d %H:%M')} UTC")

    # Load analysis week
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
    df_all = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    start_ts = analysis_open.timestamp()
    end_ts = analysis_close.timestamp()
    df_day = df_all[(df_all['timestamp'] >= start_ts) & (df_all['timestamp'] <= end_ts)].reset_index(drop=True)
    df_context = None

    print(f"  Bars: {len(df_day)}")

    print(f"\nOpening interactive chart...")
    print(f"  Click+drag: draw level | Click+drag near line: move it")
    print(f"  D/Delete: remove nearest level")
    print(f"  Z: cycle TF (4h -> 1h -> 15m -> 5m -> 1m)")
    print(f"  Close window or toolbar save: save to examples/")

    drawer = LevelDrawer(df_day, None, analysis_start,
                         lookback_str=analysis_start, trade_end=trade_end)
    drawer.show()


if __name__ == '__main__':
    main()
