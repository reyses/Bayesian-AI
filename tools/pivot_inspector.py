#!/usr/bin/env python
"""
Pivot Inspector -- visualize real vs fakeout pivots on price chart.

Shows 1s price action with pivot points color-coded:
  GREEN triangle = REAL reversal (profitable swing)
  RED triangle = FAKEOUT (failed reversal)
  YELLOW triangle = MARGINAL

Overlays 1m DMI diff and volume delta for confirmation analysis.

Keys:
    Left/Right = prev/next day
    Up/Down    = zoom in/out
    F          = fit full day
    R          = show only REAL pivots
    K          = show only FAKEOUT pivots
    A          = show ALL pivots
    Q          = quit

Usage:
    python tools/pivot_inspector.py
    python tools/pivot_inspector.py --csv reports/findings/pivot_seeds_mtf.csv
    python tools/pivot_inspector.py --data DATA/ATLAS_1WEEK
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

matplotlib.use('TkAgg')


class PivotInspector:
    def __init__(self, pivots_df: pd.DataFrame, data_root: str = 'DATA/ATLAS_1WEEK'):
        self.pivots = pivots_df
        self.data_root = data_root
        self.filter_label = 'ALL'

        # Group pivots by day
        self.pivots['date'] = pd.to_datetime(self.pivots['timestamp'], unit='s').dt.date
        self.days = sorted(self.pivots['date'].unique())
        self.day_idx = 0

        # Load 1s price data
        self._load_price_data()

        # Setup figure
        self.fig, (self.ax_price, self.ax_dmi, self.ax_vol) = plt.subplots(
            3, 1, figsize=(16, 10), height_ratios=[3, 1, 1], sharex=True)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.suptitle('Pivot Inspector -- Arrow keys: navigate, R/K/A: filter, Q: quit')

        self._draw()
        plt.tight_layout()
        plt.show()

    def _load_price_data(self):
        """Load 1s and 1m data for overlay."""
        import glob
        self.df_1s = pd.DataFrame()
        self.df_1m = pd.DataFrame()

        for tf, attr in [('1s', 'df_1s'), ('1m', 'df_1m')]:
            tf_dir = os.path.join(self.data_root, tf)
            if not os.path.isdir(tf_dir):
                continue
            files = sorted(glob.glob(os.path.join(tf_dir, '*.parquet')))
            if files:
                dfs = [pd.read_parquet(f) for f in files]
                df = pd.concat(dfs, ignore_index=True).sort_values('timestamp')
                setattr(self, attr, df)

    def _draw(self):
        """Draw current day."""
        for ax in [self.ax_price, self.ax_dmi, self.ax_vol]:
            ax.clear()

        if self.day_idx >= len(self.days):
            return

        day = self.days[self.day_idx]
        day_pivots = self.pivots[self.pivots['date'] == day]

        if self.filter_label != 'ALL':
            shown = day_pivots[day_pivots['label'] == self.filter_label]
        else:
            shown = day_pivots

        # Day timestamp range
        day_ts_min = day_pivots['timestamp'].min() - 300
        day_ts_max = day_pivots['timestamp'].max() + 300

        # Price chart (1s)
        if not self.df_1s.empty:
            mask = (self.df_1s['timestamp'] >= day_ts_min) & (self.df_1s['timestamp'] <= day_ts_max)
            day_1s = self.df_1s[mask]
            if not day_1s.empty:
                t = day_1s['timestamp'].values
                self.ax_price.plot(t, day_1s['close'].values, color='gray', linewidth=0.5, alpha=0.7)
                self.ax_price.fill_between(t, day_1s['low'].values, day_1s['high'].values,
                                           color='lightblue', alpha=0.3)

        # Pivot markers
        colors = {'REAL': 'green', 'FAKEOUT': 'red', 'MARGINAL': 'orange'}
        markers_up = {'REAL': '^', 'FAKEOUT': '^', 'MARGINAL': '^'}
        markers_dn = {'REAL': 'v', 'FAKEOUT': 'v', 'MARGINAL': 'v'}

        for _, row in shown.iterrows():
            ts = row['timestamp']
            price = row['entry_price']
            label = row['label']
            direction = row['direction']
            c = colors.get(label, 'gray')
            m = markers_up.get(label, '^') if direction == 'LONG' else markers_dn.get(label, 'v')
            size = 80 if label == 'FAKEOUT' else 40

            self.ax_price.scatter(ts, price, color=c, marker=m, s=size, zorder=5,
                                  edgecolors='black', linewidths=0.5)

            # Draw swing line to exit
            if 'exit_price' in row and pd.notna(row.get('exit_price')):
                exit_ts = row['timestamp'] + row.get('hold_seconds', 60)
                self.ax_price.plot([ts, exit_ts], [price, row['exit_price']],
                                   color=c, alpha=0.4, linewidth=1)

        # 1m DMI overlay
        if not self.df_1m.empty:
            mask = (self.df_1m['timestamp'] >= day_ts_min) & (self.df_1m['timestamp'] <= day_ts_max)
            day_1m = self.df_1m[mask]
            if not day_1m.empty and 'high' in day_1m.columns:
                t = day_1m['timestamp'].values
                # Use close as proxy — actual DMI needs state computation
                self.ax_dmi.plot(t, day_1m['close'].diff().fillna(0).values,
                                 color='blue', linewidth=0.8, label='1m price delta')
                self.ax_dmi.axhline(0, color='gray', linewidth=0.5, linestyle='--')
                self.ax_dmi.set_ylabel('1m delta')
                self.ax_dmi.legend(fontsize=8)

        # 1m volume overlay
        if not self.df_1m.empty and 'volume' in self.df_1m.columns:
            mask = (self.df_1m['timestamp'] >= day_ts_min) & (self.df_1m['timestamp'] <= day_ts_max)
            day_1m = self.df_1m[mask]
            if not day_1m.empty:
                t = day_1m['timestamp'].values
                vol = day_1m['volume'].values
                closes = day_1m['close'].values
                opens = day_1m['open'].values if 'open' in day_1m.columns else closes
                vol_signed = np.where(closes >= opens, vol, -vol)
                colors_bar = ['green' if v >= 0 else 'red' for v in vol_signed]
                self.ax_vol.bar(t, vol_signed, width=50, color=colors_bar, alpha=0.6)
                self.ax_vol.axhline(0, color='gray', linewidth=0.5, linestyle='--')
                self.ax_vol.set_ylabel('1m volume')

        n_real = len(day_pivots[day_pivots['label'] == 'REAL'])
        n_fake = len(day_pivots[day_pivots['label'] == 'FAKEOUT'])
        n_marg = len(day_pivots[day_pivots['label'] == 'MARGINAL'])
        pnl = day_pivots['pnl_ticks'].sum()

        self.ax_price.set_title(
            f'{day}  |  {len(shown)} shown (filter: {self.filter_label})  |  '
            f'R:{n_real} F:{n_fake} M:{n_marg}  |  PnL: {pnl:.0f}t  |  '
            f'Day {self.day_idx+1}/{len(self.days)}')
        self.ax_price.set_ylabel('Price')

        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if event.key == 'right':
            self.day_idx = min(self.day_idx + 1, len(self.days) - 1)
            self._draw()
        elif event.key == 'left':
            self.day_idx = max(self.day_idx - 1, 0)
            self._draw()
        elif event.key == 'r':
            self.filter_label = 'REAL'
            self._draw()
        elif event.key == 'k':
            self.filter_label = 'FAKEOUT'
            self._draw()
        elif event.key == 'a':
            self.filter_label = 'ALL'
            self._draw()
        elif event.key == 'q':
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description='Pivot Inspector')
    parser.add_argument('--csv', type=str, default='reports/findings/pivot_seeds_mtf.csv',
                        help='Path to pivot seeds CSV')
    parser.add_argument('--data', type=str, default='DATA/ATLAS_1WEEK',
                        help='ATLAS data root for price overlay')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found. Run pivot_seed_scanner_mtf.py first.")
        return 1

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} pivots ({(df['label']=='REAL').sum()} real, "
          f"{(df['label']=='FAKEOUT').sum()} fakeout)")

    PivotInspector(df, data_root=args.data)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
