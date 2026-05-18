#!/usr/bin/env python
"""
Trade Marker — Manually mark trades on a price chart with crosshair.

Click to mark START, click again to mark END. Direction auto-detected.
Press S to flip direction, D to delete last mark, Q to save+quit.

Usage:
    python tools/trade_marker.py --date 2025-07-14
    python tools/trade_marker.py --date 2025-07-14 --tf 1m
    python tools/trade_marker.py --date 2025-07-14 --tf 5m
    python tools/trade_marker.py --date 2025-07-14 --tf 1h --burn-hours 10
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf
from tools.golden_path import load_1s_index, load_1s_window

# Force interactive backend AFTER imports (plots.py sets Agg)
plt.switch_backend('TkAgg')

TICK_SIZE = 0.25
TICK_VALUE = 0.50
SEEDS_DIR = 'DATA/regime_seeds'


class TradeMarker:
    """Manual trade marker with crosshair on price chart."""

    def __init__(self, df, index_1s, date_str, tf='1m', continuous=False):
        self.df = df
        self.index_1s = index_1s
        self.date_str = date_str
        self.tf = tf
        self.continuous = continuous

        self.close = df['close'].values.astype(float)
        self.high = df['high'].values.astype(float)
        self.low = df['low'].values.astype(float)
        self.timestamps = df['timestamp'].values.astype(float)
        self.dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc) for t in self.timestamps]

        # State machine
        self.state = 'IDLE'  # IDLE -> START_SET -> IDLE
        self.start_idx = None
        self.start_line = None

        # Live preview elements (shown while START is set, cursor moves)
        self._preview_line = None
        self._preview_label = None
        self._preview_span = None

        # Completed trades
        self.trades = []

        # Plot elements to clean up
        self._trade_patches = []

    def _find_nearest_bar(self, x_date):
        """Find nearest bar index to a matplotlib date click."""
        import matplotlib.dates as mdates
        click_num = mdates.date2num(x_date) if not isinstance(x_date, float) else x_date
        bar_nums = mdates.date2num(self.dt_stamps)
        idx = int(np.argmin(np.abs(bar_nums - click_num)))
        return idx

    def _measure_1s(self, ts_start, ts_end, direction):
        """Measure MFE/MAE from 1s data between start and end."""
        cache = {}
        try:
            df_1s = load_1s_window(self.index_1s, ts_start, ts_end, cache)
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
            time_to_mfe = float(ts1s[mfe_idx] - ts1s[0]) / 60.0  # minutes

            return mfe, mae, time_to_mfe
        except Exception:
            return 0.0, 0.0, 0.0

    def _draw_trade(self, trade, idx):
        """Draw a completed trade on the chart."""
        si, ei = trade['start_idx'], trade['end_idx']
        direction = trade['direction']
        color = '#00C853' if direction == 'LONG' else '#FF1744'

        # Shade the trade region
        span = self.ax.axvspan(self.dt_stamps[si], self.dt_stamps[ei],
                               alpha=0.15, color=color)

        # Entry/exit markers
        marker_entry = '^' if direction == 'LONG' else 'v'
        m1 = self.ax.scatter([self.dt_stamps[si]], [self.close[si]],
                             color=color, s=150, zorder=5, marker=marker_entry)
        m2 = self.ax.scatter([self.dt_stamps[ei]], [self.close[ei]],
                             color='black', s=100, zorder=5, marker='s')

        # Label
        mfe_str = f"${trade['mfe_dollars']:.0f}" if trade['mfe_dollars'] > 0 else "?"
        mae_str = f"${trade['mae_dollars']:.0f}" if trade['mae_dollars'] > 0 else "?"
        mid_idx = (si + ei) // 2
        y_pos = max(self.close[si], self.close[ei]) + 2
        label = self.ax.text(self.dt_stamps[mid_idx], y_pos,
                             f"T{idx+1} {direction}\nMFE:{mfe_str} MAE:{mae_str}",
                             fontsize=8, ha='center', va='bottom', fontweight='bold',
                             color=color,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                       alpha=0.8, edgecolor=color))

        self._trade_patches.append((span, m1, m2, label))

    def _update_title(self):
        """Update the title bar with current state."""
        n_trades = len(self.trades)
        n_long = sum(1 for t in self.trades if t['direction'] == 'LONG')
        n_short = n_trades - n_long

        mode_str = "CONTINUOUS" if self.continuous else "TWO-CLICK"

        if self.continuous:
            if self.start_idx is not None:
                action = f"Click to mark swing END (start={self.start_idx})"
            else:
                action = "Click to set first pivot"
        elif self.state == 'IDLE':
            action = "Click to mark trade START"
        elif self.state == 'START_SET':
            action = f"START set at bar {self.start_idx} -- Click to mark END"

        total_mfe = sum(t['mfe_dollars'] for t in self.trades)

        self.ax.set_title(
            f"{self.date_str} | {self.tf} | {len(self.df)} bars | [{mode_str}]\n"
            f"Trades: {n_trades} (L:{n_long} S:{n_short}) | "
            f"Total MFE: ${total_mfe:.0f}\n"
            f"[{action}]  C=toggle mode, S=flip, D=delete last, Q=save+quit",
            fontsize=10, fontweight='bold'
        )
        self.fig.canvas.draw_idle()

    def _clear_preview(self):
        """Remove live preview elements."""
        for obj in (self._preview_line, self._preview_label, self._preview_span):
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
        self._preview_line = None
        self._preview_label = None
        self._preview_span = None

    def _on_motion(self, event):
        """Show live preview: duration, ticks, direction while START is set."""
        if self.state != 'START_SET' or event.inaxes != self.ax:
            return

        import matplotlib.dates as mdates
        if event.xdata is None:
            return

        bar_nums = mdates.date2num(self.dt_stamps)
        hover_idx = int(np.argmin(np.abs(bar_nums - event.xdata)))

        if hover_idx == self.start_idx:
            return

        si = min(self.start_idx, hover_idx)
        ei = max(self.start_idx, hover_idx)

        # Compute preview stats
        change = self.close[ei] - self.close[si]
        change_ticks = change / TICK_SIZE
        direction = 'LONG' if change > 0 else 'SHORT'
        duration_mins = (self.timestamps[ei] - self.timestamps[si]) / 60.0
        n_bars = ei - si + 1

        # Clear old preview
        self._clear_preview()

        # Draw preview span
        color = '#00C853' if direction == 'LONG' else '#FF1744'
        self._preview_span = self.ax.axvspan(
            self.dt_stamps[si], self.dt_stamps[ei],
            alpha=0.06, color=color)

        # End line
        self._preview_line = self.ax.axvline(
            x=self.dt_stamps[hover_idx], color=color,
            linewidth=1.5, linestyle=':', alpha=0.6)

        # Duration/ticks label at cursor position
        mid_idx = (si + ei) // 2
        y_top = self.ax.get_ylim()[1]
        self._preview_label = self.ax.text(
            self.dt_stamps[mid_idx], y_top,
            f"{direction} | {duration_mins:.0f}m | {n_bars} bars | {abs(change_ticks):.0f}t",
            fontsize=9, ha='center', va='top', fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      alpha=0.9, edgecolor=color))

        self.fig.canvas.draw_idle()

    def _complete_trade(self, si, ei):
        """Build a trade dict from start/end indices, measure with 1s, draw it."""
        direction = 'LONG' if self.close[ei] > self.close[si] else 'SHORT'

        ts_s = float(self.timestamps[si])
        ts_e = float(self.timestamps[ei])
        mfe, mae, time_to_mfe = self._measure_1s(ts_s, ts_e, direction)

        change = self.close[ei] - self.close[si]
        change_ticks = change / TICK_SIZE
        duration_mins = (ts_e - ts_s) / 60.0

        trade = {
            'trade_id': len(self.trades),
            'timeframe': self.tf,
            'direction': direction,
            'start_idx': si,
            'end_idx': ei,
            'ts_start': ts_s,
            'ts_end': ts_e,
            'entry_price': round(float(self.close[si]), 2),
            'exit_price': round(float(self.close[ei]), 2),
            'change_ticks': round(change_ticks, 1),
            'change_dollars': round(change_ticks * TICK_VALUE, 2),
            'mfe_ticks': round(mfe, 1),
            'mae_ticks': round(mae, 1),
            'mfe_dollars': round(mfe * TICK_VALUE, 2),
            'mae_dollars': round(mae * TICK_VALUE, 2),
            'duration_mins': round(duration_mins, 1),
            'time_to_mfe_mins': round(time_to_mfe, 1),
            'n_bars': ei - si + 1,
            'lookback_bars': 10,
            'lookback_start_idx': max(0, si - 10),
            'lookback_timestamps': [float(self.timestamps[j])
                                    for j in range(max(0, si - 10), si)],
            'regime_start_idx': si,
        }

        self.trades.append(trade)
        self._draw_trade(trade, len(self.trades) - 1)

        # Print to console
        surv = "YES" if mae <= 20 else "NO"
        rr = mfe / mae if mae > 0 else float('inf')
        print(f"  T{trade['trade_id']+1}: {direction} | "
              f"{duration_mins:.0f}m | "
              f"MFE: ${mfe*TICK_VALUE:.0f} ({mfe:.0f}t) | "
              f"MAE: ${mae*TICK_VALUE:.0f} ({mae:.0f}t) | "
              f"R:R 1:{rr:.1f} | "
              f"$10 SL: {surv}")

        return trade

    def _on_click(self, event):
        """Handle mouse click — mark start or end."""
        if event.inaxes != self.ax:
            return
        if event.button != 1:  # left click only
            return

        import matplotlib.dates as mdates
        click_num = event.xdata
        if click_num is None:
            return

        bar_nums = mdates.date2num(self.dt_stamps)
        idx = int(np.argmin(np.abs(bar_nums - click_num)))

        if self.continuous:
            # Continuous mode: every click = end of current swing + start of next
            if self.start_idx is None:
                # Should not happen (auto-started at bar 0), but handle it
                self.start_idx = idx
                self.start_line = self.ax.axvline(x=self.dt_stamps[idx], color='blue',
                                                   linewidth=2, linestyle='--', alpha=0.7)
                self.state = 'START_SET'
                self._update_title()
                return

            if idx == self.start_idx:
                return

            si = min(self.start_idx, idx)
            ei = max(self.start_idx, idx)

            # Complete current trade
            self._complete_trade(si, ei)

            # Remove old start line + preview
            if self.start_line:
                self.start_line.remove()
                self.start_line = None
            self._clear_preview()

            # Immediately start next trade from this end point
            self.start_idx = ei
            self.start_line = self.ax.axvline(x=self.dt_stamps[ei], color='blue',
                                               linewidth=2, linestyle='--', alpha=0.7)
            self.state = 'START_SET'
            self._update_title()

            # Auto-pan: if click is near right edge, pan right
            xlim = self.ax.get_xlim()
            click_frac = (click_num - xlim[0]) / (xlim[1] - xlim[0])
            if click_frac > 0.75:
                self._pan(1)

        else:
            # Original two-click mode
            if self.state == 'IDLE':
                self.start_idx = idx
                self.start_line = self.ax.axvline(x=self.dt_stamps[idx], color='blue',
                                                   linewidth=2, linestyle='--', alpha=0.7)
                self.state = 'START_SET'
                self._update_title()

            elif self.state == 'START_SET':
                end_idx = idx
                if end_idx == self.start_idx:
                    return

                si = min(self.start_idx, end_idx)
                ei = max(self.start_idx, end_idx)

                self._complete_trade(si, ei)

                if self.start_line:
                    self.start_line.remove()
                    self.start_line = None
                self._clear_preview()

                self.state = 'IDLE'
                self.start_idx = None
                self._update_title()

    def _on_key(self, event):
        """Handle key press."""
        if event.key in ('s', 'S'):
            # Flip direction of last trade
            if self.trades:
                t = self.trades[-1]
                old_dir = t['direction']
                new_dir = 'SHORT' if old_dir == 'LONG' else 'LONG'
                t['direction'] = new_dir

                # Re-measure with flipped direction
                mfe, mae, ttm = self._measure_1s(t['ts_start'], t['ts_end'], new_dir)
                t['mfe_ticks'] = round(mfe, 1)
                t['mae_ticks'] = round(mae, 1)
                t['mfe_dollars'] = round(mfe * TICK_VALUE, 2)
                t['mae_dollars'] = round(mae * TICK_VALUE, 2)
                t['time_to_mfe_mins'] = round(ttm, 1)

                # Redraw
                if self._trade_patches:
                    for obj in self._trade_patches[-1]:
                        obj.remove()
                    self._trade_patches.pop()
                self._draw_trade(t, len(self.trades) - 1)
                self._update_title()
                print(f"  T{t['trade_id']+1}: flipped to {new_dir}")

        elif event.key in ('d', 'D'):
            # Delete last trade
            if self.trades:
                t = self.trades.pop()
                if self._trade_patches:
                    for obj in self._trade_patches[-1]:
                        obj.remove()
                    self._trade_patches.pop()
                print(f"  Deleted T{t['trade_id']+1}")

                if self.continuous:
                    # Snap start back to previous trade's end (or bar 0)
                    if self.start_line:
                        self.start_line.remove()
                    prev_end = self.trades[-1]['end_idx'] if self.trades else 0
                    self.start_idx = prev_end
                    self.start_line = self.ax.axvline(
                        x=self.dt_stamps[prev_end], color='blue',
                        linewidth=2, linestyle='--', alpha=0.7)
                    self.state = 'START_SET'
                else:
                    # Also cancel pending start
                    if self.state == 'START_SET' and self.start_line:
                        self.start_line.remove()
                        self.start_line = None
                        self.state = 'IDLE'

                self._update_title()
            elif not self.continuous:
                # No trades, cancel pending start in two-click mode
                if self.state == 'START_SET' and self.start_line:
                    self.start_line.remove()
                    self.start_line = None
                    self.state = 'IDLE'
                    self._update_title()

        elif event.key in ('c', 'C'):
            # Toggle continuous mode
            self.continuous = not self.continuous
            mode = "CONTINUOUS" if self.continuous else "TWO-CLICK"
            print(f"  Mode: {mode}")

            if self.continuous and self.start_idx is None and self.trades:
                # Entering continuous: auto-set start from last trade's end
                last_end = self.trades[-1]['end_idx']
                self.start_idx = last_end
                if self.start_line:
                    self.start_line.remove()
                self.start_line = self.ax.axvline(
                    x=self.dt_stamps[last_end], color='blue',
                    linewidth=2, linestyle='--', alpha=0.7)
                self.state = 'START_SET'
            elif not self.continuous:
                # Leaving continuous: clear pending start
                if self.start_line:
                    self.start_line.remove()
                    self.start_line = None
                self._clear_preview()
                self.state = 'IDLE'
                self.start_idx = None

            self._update_title()

        elif event.key == 'escape':
            # Cancel pending start
            if self.state == 'START_SET':
                if self.start_line:
                    self.start_line.remove()
                    self.start_line = None
                self._clear_preview()
                self.state = 'IDLE'
                self.start_idx = None
                self._update_title()

        elif event.key == 'left':
            self._pan(-1)
        elif event.key == 'right':
            self._pan(1)
        elif event.key == 'up':
            self._zoom(0.5)
        elif event.key == 'down':
            self._zoom(2.0)

        elif event.key in ('q', 'Q'):
            self._save()
            plt.close(self.fig)

    def _save(self):
        """Save marked trades. Merges with existing seeds for the same date."""
        if not self.trades:
            print("\n  No trades marked. Nothing saved.")
            return

        os.makedirs(SEEDS_DIR, exist_ok=True)

        # Check for existing multi-TF seed file for this date
        existing_path = os.path.join(SEEDS_DIR, f'seeds_{self.date_str}_multi.json')
        existing_seeds = []
        existing_tfs = set()
        if os.path.exists(existing_path):
            with open(existing_path) as f:
                existing_data = json.load(f)
            existing_seeds = existing_data.get('seeds', [])
            existing_tfs = set(existing_data.get('marked_timeframes', []))
            # Remove any prior seeds at this TF (re-marking replaces)
            existing_seeds = [s for s in existing_seeds
                              if s.get('timeframe', '1m') != self.tf]
            print(f"  Merging with existing: {len(existing_seeds)} seeds from {existing_tfs}")

        # Merge: existing (other TFs) + new (this TF)
        all_seeds = existing_seeds + self.trades
        all_tfs = existing_tfs | {self.tf}

        # Re-number trade IDs globally
        for i, s in enumerate(all_seeds):
            s['trade_id'] = i

        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save multi-TF file (canonical, overwritten on each TF pass)
        with open(existing_path, 'w') as f:
            json.dump({
                'date': self.date_str,
                'marked_timeframes': sorted(all_tfs),
                'created': ts_tag,
                'n_seeds': len(all_seeds),
                'seeds': all_seeds,
            }, f, indent=2)

        # Also save a per-TF snapshot (for reference)
        tf_path = os.path.join(SEEDS_DIR, f'seeds_{self.date_str}_{self.tf}_{ts_tag}.json')
        with open(tf_path, 'w') as f:
            json.dump({
                'date': self.date_str,
                'timeframe': self.tf,
                'created': ts_tag,
                'n_seeds': len(self.trades),
                'seeds': self.trades,
            }, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  MARKING COMPLETE")
        print(f"{'='*60}")
        print(f"  This TF:    {len(self.trades)} seeds @ {self.tf}")
        print(f"  Multi-TF:   {len(all_seeds)} total seeds across {sorted(all_tfs)}")
        print(f"  Merged:     {existing_path}")
        print(f"  Snapshot:   {tf_path}")

        if self.trades:
            total_mfe = sum(t['mfe_dollars'] for t in self.trades)
            avg_mfe = total_mfe / len(self.trades)
            n_long = sum(1 for t in self.trades if t['direction'] == 'LONG')
            n_short = len(self.trades) - n_long
            print(f"\n  Stats ({self.tf}):")
            print(f"    Avg MFE:   ${avg_mfe:.2f}")
            print(f"    Total MFE: ${total_mfe:.2f}")
            print(f"    LONG:      {n_long}")
            print(f"    SHORT:     {n_short}")

    def _pan(self, direction):
        """Pan the view left or right by half the current window width."""
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        window = xlim[1] - xlim[0]
        shift = window * 0.5 * direction  # -1 = left, +1 = right

        # Clamp to data bounds
        x_min = mdates.date2num(self.dt_stamps[0])
        x_max = mdates.date2num(self.dt_stamps[-1])

        new_left = xlim[0] + shift
        new_right = xlim[1] + shift

        if new_left < x_min:
            new_left = x_min
            new_right = x_min + window
        if new_right > x_max:
            new_right = x_max
            new_left = x_max - window

        self.ax.set_xlim(new_left, new_right)
        # Auto-fit Y to visible data
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _zoom(self, factor):
        """Zoom in/out by factor (>1 = zoom out, <1 = zoom in)."""
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        half_w = (xlim[1] - xlim[0]) / 2 * factor

        x_min = mdates.date2num(self.dt_stamps[0])
        x_max = mdates.date2num(self.dt_stamps[-1])

        new_left = max(x_min, center - half_w)
        new_right = min(x_max, center + half_w)

        self.ax.set_xlim(new_left, new_right)
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _autofit_y(self):
        """Auto-fit Y axis to visible data range."""
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        bar_nums = mdates.date2num(self.dt_stamps)
        mask = (bar_nums >= xlim[0]) & (bar_nums <= xlim[1])
        if mask.any():
            vis_low = self.low[mask].min()
            vis_high = self.high[mask].max()
            pad = (vis_high - vis_low) * 0.05
            self.ax.set_ylim(vis_low - pad, vis_high + pad)

    def run(self):
        """Show the chart and start marking."""
        import matplotlib.dates as mdates
        from matplotlib.widgets import Cursor, Button

        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 8))
        self.fig.subplots_adjust(bottom=0.12)  # room for buttons
        self.ax.set_facecolor('#FAFAFA')

        # Plot price with high/low range
        self.ax.plot(self.dt_stamps, self.close, color='#1565C0', linewidth=1.2, alpha=0.9)
        self.ax.fill_between(self.dt_stamps, self.low, self.high,
                             alpha=0.08, color='#1565C0')

        self.ax.set_ylabel('Price', fontsize=11)
        self.ax.grid(True, alpha=0.2)

        # Draw session dividers (midnight UTC lines) for multi-day views
        seen_days = set()
        for i, dt_s in enumerate(self.dt_stamps):
            day_str = dt_s.strftime('%Y-%m-%d')
            if day_str not in seen_days:
                seen_days.add(day_str)
                if i > 0:  # skip first bar
                    self.ax.axvline(x=dt_s, color='#FF9800', linewidth=1.5,
                                    linestyle='-', alpha=0.4)
                    self.ax.text(dt_s, self.ax.get_ylim()[1], f' {day_str}',
                                 fontsize=8, color='#FF9800', alpha=0.7,
                                 va='top', ha='left')

        # Start zoomed to first 1 hour of data
        n_bars = len(self.dt_stamps)
        _tf_bph = {'15s': 240, '30s': 120, '1m': 60, '2m': 30, '3m': 20,
                    '5m': 12, '15m': 4, '30m': 2, '1h': 1}
        bars_per_hour = _tf_bph.get(self.tf, 60)

        zoom_end = min(bars_per_hour, n_bars - 1)
        self.ax.set_xlim(mdates.date2num(self.dt_stamps[0]),
                         mdates.date2num(self.dt_stamps[zoom_end]))
        self._autofit_y()

        # Crosshair cursor
        self.ax.format_coord = lambda x, y: f'Price: {y:.2f}'
        self.cursor = Cursor(self.ax, useblit=True, color='gray',
                             linewidth=0.5, linestyle='--')

        # Navigation buttons
        btn_color = '#E0E0E0'
        ax_left = self.fig.add_axes([0.15, 0.02, 0.08, 0.04])
        ax_right = self.fig.add_axes([0.24, 0.02, 0.08, 0.04])
        ax_zin = self.fig.add_axes([0.37, 0.02, 0.08, 0.04])
        ax_zout = self.fig.add_axes([0.46, 0.02, 0.08, 0.04])
        ax_fit = self.fig.add_axes([0.59, 0.02, 0.08, 0.04])

        self.btn_left = Button(ax_left, '<< Pan Left', color=btn_color)
        self.btn_right = Button(ax_right, 'Pan Right >>', color=btn_color)
        self.btn_zin = Button(ax_zin, 'Zoom In', color=btn_color)
        self.btn_zout = Button(ax_zout, 'Zoom Out', color=btn_color)
        self.btn_fit = Button(ax_fit, 'Fit All', color=btn_color)

        self.btn_left.on_clicked(lambda e: self._pan(-1))
        self.btn_right.on_clicked(lambda e: self._pan(1))
        self.btn_zin.on_clicked(lambda e: self._zoom(0.5))
        self.btn_zout.on_clicked(lambda e: self._zoom(2.0))
        self.btn_fit.on_clicked(lambda e: (
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[0]),
                             mdates.date2num(self.dt_stamps[-1])),
            self._autofit_y(),
            self.fig.canvas.draw_idle()
        ))

        # In continuous mode, auto-start at bar 0
        if self.continuous:
            self.start_idx = 0
            self.start_line = self.ax.axvline(x=self.dt_stamps[0], color='blue',
                                               linewidth=2, linestyle='--', alpha=0.7)
            self.state = 'START_SET'

        self._update_title()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        mode = "CONTINUOUS (C to toggle)" if self.continuous else "TWO-CLICK (C to toggle)"
        print(f"\n  Trade Marker ready -- {len(self.df)} bars loaded [{mode}]")
        if self.continuous:
            print(f"  Auto-started at bar 0. Each click = end swing + start next.")
        else:
            print(f"  Click START, click END. Direction auto-detected.")
        print(f"  C=toggle mode, S=flip direction, D=delete last, Esc=cancel, Q=save+quit")
        print(f"  Arrow keys: Left/Right=pan, Up=zoom in, Down=zoom out\n")

        plt.tight_layout(rect=[0, 0.07, 1, 1])
        plt.show(block=True)


def main():
    parser = argparse.ArgumentParser(description='Trade Marker (manual crosshair)')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root directory')
    parser.add_argument('--date', required=True,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, default=1,
                        help='Number of calendar days to load (default: 1)')
    parser.add_argument('--tf', default='1m',
                        help='Timeframe for chart (default: 1m)')
    parser.add_argument('--continuous', action='store_true',
                        help='Continuous mode: auto-start at bar 0, each click = pivot')
    parser.add_argument('--burn-hours', type=float, default=0,
                        help='Skip first N hours (regression warmup, e.g. 10 for day 1)')
    args = parser.parse_args()

    date_label = args.date if args.days == 1 else f"{args.date} (+{args.days-1}d)"
    print(f"Trade Marker -- {date_label} ({args.tf})")

    # Load chart data
    print("Loading data...")
    df = load_atlas_tf(args.data_dir, args.tf)
    if df.empty:
        print(f"ERROR: No {args.tf} data")
        sys.exit(1)

    # Filter to date range
    dt = datetime.strptime(args.date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    t_start = dt.timestamp()
    t_end = t_start + 86400 * args.days
    mask = (df['timestamp'] >= t_start) & (df['timestamp'] < t_end)
    df_day = df[mask].reset_index(drop=True)

    # Burn first N hours (regression warmup for slow TFs)
    if args.burn_hours > 0:
        burn_until = t_start + args.burn_hours * 3600
        burn_mask = df_day['timestamp'] >= burn_until
        n_burned = (~burn_mask).sum()
        df_day = df_day[burn_mask].reset_index(drop=True)
        print(f"  Burned first {args.burn_hours}h ({n_burned} bars skipped)")

    if df_day.empty:
        print(f"ERROR: No data for {date_label}")
        sys.exit(1)

    # Show time span
    ts_first = datetime.fromtimestamp(float(df_day['timestamp'].iloc[0]), tz=timezone.utc)
    ts_last = datetime.fromtimestamp(float(df_day['timestamp'].iloc[-1]), tz=timezone.utc)
    span_hours = (float(df_day['timestamp'].iloc[-1]) - float(df_day['timestamp'].iloc[0])) / 3600
    print(f"  {len(df_day)} bars | {ts_first.strftime('%a %b %d %H:%M')} → {ts_last.strftime('%a %b %d %H:%M')} UTC ({span_hours:.1f}h)")

    # Load 1s index for MFE/MAE measurement
    print("Loading 1s index...")
    index_1s = load_1s_index(args.data_dir)

    marker = TradeMarker(df_day, index_1s, date_label, tf=args.tf,
                         continuous=args.continuous)
    marker.run()


if __name__ == '__main__':
    main()
