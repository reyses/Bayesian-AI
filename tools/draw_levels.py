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
        self.current_tf_idx = 0

        # Load previously saved levels if they exist
        json_dir = 'DATA/levels'
        json_path = os.path.join(json_dir, f'levels_{date_str}.json')
        if os.path.exists(json_path):
            import json as _json
            with open(json_path) as f:
                saved = _json.load(f)
            self.levels = saved.get('levels', [])
            print(f"  Loaded {len(self.levels)} previously saved levels from {json_path}")

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
            f'click+drag=draw/move, D=delete, Z=micro, X=macro',
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

    def _shift_week(self, direction):
        """Move to next (+1) or previous (-1) week. Always shows 4-week window."""
        import json as _json

        current_dt = pd.Timestamp(self.date_str)
        new_dt = current_dt + pd.Timedelta(days=7 * direction)
        new_date_str = new_dt.strftime('%Y-%m-%d')

        print(f"  Moving to week of {new_date_str}...")
        self.date_str = new_date_str

        # 4-week window: 3 weeks before + current week
        window_start = new_dt - pd.Timedelta(days=21)
        window_end = new_dt + pd.Timedelta(days=7)

        self.lookback_str = window_start.strftime('%Y-%m-%d')
        self.trade_end = window_end.strftime('%Y-%m-%d')

        # Reload all TF data for 4-week window
        self.tf_data = {}
        files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
        df_all = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        df_all = df_all.sort_values('timestamp').reset_index(drop=True)
        start_ts = window_start.timestamp()
        end_ts = window_end.timestamp()
        df_window = df_all[(df_all['timestamp'] >= start_ts) & (df_all['timestamp'] < end_ts)].reset_index(drop=True)
        self.tf_data['1m'] = {'day': df_window, 'context': None}
        del df_all

        self._load_all_tfs(self.date_str, self.lookback_str)

        # Load saved levels for this week, or carry forward from previous
        json_path = os.path.join('DATA', 'levels', f'levels_{new_date_str}.json')
        if os.path.exists(json_path):
            with open(json_path) as f:
                saved = _json.load(f)
            self.levels = saved.get('levels', [])
            print(f"  Loaded {len(self.levels)} saved levels")
        else:
            # Carry forward: keep current levels as starting point
            import copy
            self.levels = copy.deepcopy(self.levels)
            print(f"  Carrying forward {len(self.levels)} levels from previous week")

        # Rebuild display
        tf = self.TF_CYCLE[self.current_tf_idx]
        if tf in self.tf_data:
            self._build_display_data(tf)
        elif '1m' in self.tf_data:
            self._build_display_data('1m')
        else:
            print(f"  No data for this period — press arrow to try another week")
            return
        if self.n == 0:
            print(f"  No bars for this period — press arrow to try another week")
            return
        self._draw_chart()
        print(f"  4-week window: {window_start.strftime('%m/%d')} - {window_end.strftime('%m/%d')}")

    def _cycle_tf(self, direction):
        """Cycle TF forward (micro) or backward (macro), skipping missing data."""
        attempts = 0
        while attempts < len(self.TF_CYCLE):
            self.current_tf_idx = (self.current_tf_idx + direction) % len(self.TF_CYCLE)
            tf = self.TF_CYCLE[self.current_tf_idx]
            if tf in self.tf_data:
                self._build_display_data(tf)
                self._draw_chart()
                print(f"  Switched to {tf.upper()} ({self.n} bars)")
                return
            attempts += 1

    def _on_key(self, event):
        if event.key == 'z':
            # Z = drill down to micro (4h -> 1h -> 15m -> 5m -> 1m)
            self._cycle_tf(+1)
        elif event.key == 'x':
            # X = zoom out to macro (1m -> 5m -> 15m -> 1h -> 4h)
            self._cycle_tf(-1)

        elif event.key == 'right':
            # Right arrow = next week (saves current, loads 4-week view)
            self.save()
            self._shift_week(+1)

        elif event.key == 'left':
            # Left arrow = previous week (saves current, loads 4-week view)
            self.save()
            self._shift_week(-1)

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
        """Click near line = toggle select. Click empty = place new level."""
        if event.inaxes != self.ax:
            return
        if event.button != MouseButton.LEFT:
            return

        price = event.ydata

        # Check if clicking near an existing level
        if self.levels:
            dists = [abs(l['price'] - price) for l in self.levels]
            nearest = np.argmin(dists)
            threshold = (self.prices.max() - self.prices.min()) * 0.01
            if dists[nearest] < threshold:
                if self._selected_idx == nearest:
                    # Already selected — deselect
                    self.line_objects[nearest].set_linewidth(1.5)
                    self.line_objects[nearest].set_alpha(0.7)
                    self._selected_idx = None
                    self._dragging = False
                else:
                    # Deselect previous
                    if self._selected_idx is not None and self._selected_idx < len(self.line_objects):
                        self.line_objects[self._selected_idx].set_linewidth(1.5)
                        self.line_objects[self._selected_idx].set_alpha(0.7)
                    # Select this one
                    self._selected_idx = nearest
                    self._dragging = True
                    self.line_objects[nearest].set_linewidth(3)
                    self.line_objects[nearest].set_alpha(1.0)
                self._update_status()
                self.fig.canvas.draw()
                return

        # Not near existing — place new level immediately
        snapped = round(price / TICK) * TICK
        mid = (self.prices.max() + self.prices.min()) / 2
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

    def _on_drag(self, event):
        """Drag to move selected level."""
        if self._selected_idx is None or event.inaxes != self.ax:
            return

        price = event.ydata
        snapped = round(price / TICK) * TICK
        idx = self._selected_idx
        self.levels[idx]['price'] = snapped
        self.line_objects[idx].set_ydata([snapped, snapped])
        self.label_objects[idx].set_position(
            (self.x[-1] + (self.x[-1] - self.x[0]) * 0.01, snapped))
        self.label_objects[idx].set_text(f'{snapped:.2f}')

        self.fig.canvas.draw_idle()

    def _on_release(self, event):
        """Release after dragging a selected level — update its type."""
        if self._selected_idx is None:
            return
        idx = self._selected_idx
        snapped = self.levels[idx]['price']
        mid = (self.prices.max() + self.prices.min()) / 2
        if snapped > mid:
            self.levels[idx]['type'] = 'resistance'
            self.levels[idx]['color'] = '#CC0000'
        else:
            self.levels[idx]['type'] = 'support'
            self.levels[idx]['color'] = '#0066CC'
        self.line_objects[idx].set_color(self.levels[idx]['color'])
        self.line_objects[idx].set_linestyle('--' if self.levels[idx]['type'] == 'resistance' else '-')
        self.label_objects[idx].set_color(self.levels[idx]['color'])
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
        if not self.levels:
            print(f"  No levels to save for {self.date_str}")
            return

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

        print(f"  Saved {len(self.levels)} levels for {self.date_str}")

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

    # 4-week window centered on the given date's week
    # 3 weeks before + current week
    window_start = trade_dt - pd.Timedelta(days=21)
    window_end = trade_dt + pd.Timedelta(days=7)
    date_str = trade_dt.strftime('%Y-%m-%d')

    print(f"4-week window: {window_start.strftime('%m/%d/%Y')} -> {window_end.strftime('%m/%d/%Y')}")

    # Load 4-week window
    files = sorted(glob.glob(os.path.join(ATLAS_ROOT, '1m', '*.parquet')))
    df_all = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    start_ts = window_start.timestamp()
    end_ts = window_end.timestamp()
    df_day = df_all[(df_all['timestamp'] >= start_ts) & (df_all['timestamp'] < end_ts)].reset_index(drop=True)
    del df_all

    print(f"  Bars: {len(df_day)}")

    print(f"\nOpening interactive chart...")
    print(f"  Click: select line (toggle) | Drag selected: move it")
    print(f"  Click empty: draw new level")
    print(f"  D/Delete: remove selected/nearest level")
    print(f"  Z: drill down (4h -> 1h -> 15m -> 5m -> 1m)")
    print(f"  X: zoom out  (1m -> 5m -> 15m -> 1h -> 4h)")
    print(f"  Right/Left arrow: next/previous week (4-week view, auto-saves)")
    print(f"  Close window: save")

    drawer = LevelDrawer(df_day, None, date_str,
                         lookback_str=window_start.strftime('%Y-%m-%d'),
                         trade_end=window_end.strftime('%Y-%m-%d'))
    drawer.show()


if __name__ == '__main__':
    main()
