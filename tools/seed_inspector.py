#!/usr/bin/env python
"""
Seed Inspector -- step through I-MR auto-seeds on a price chart for visual QA.

Shows each seed with lookback (blue) and regime (green/red) highlighted.
Accept/reject each seed, then save the filtered set.

Keys:
    Y / Enter   = ACCEPT seed (keep it)
    N / Delete  = REJECT seed (drop it)
    S           = SKIP (decide later)
    Left/Right  = navigate to prev/next seed without deciding
    Up/Down     = zoom in/out
    F           = fit full day
    Q           = save + quit (undecided seeds kept by default)

Usage:
    python tools/seed_inspector.py
    python tools/seed_inspector.py --seeds DATA/regime_seeds/imr_auto/imr_seeds_all_*.json
    python tools/seed_inspector.py --start-day 2025-03-01
    python tools/seed_inspector.py --min-mfe 50   # only inspect seeds above threshold
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.research.data import load_atlas_tf

# Force interactive backend AFTER imports (plots.py sets Agg)
plt.switch_backend('TkAgg')

TICK_SIZE = 0.25
TICK_VALUE = 0.50
CONTEXT_BARS = 30  # bars before/after seed to show


class SeedInspector:
    """Step-through inspector for I-MR auto-seeds."""

    def __init__(self, seeds, df_1m, source_path):
        self.all_seeds = seeds
        self.df_1m = df_1m
        self.source_path = source_path

        self.close = df_1m['close'].values.astype(float)
        self.high = df_1m['high'].values.astype(float)
        self.low = df_1m['low'].values.astype(float)
        self.timestamps = df_1m['timestamp'].values.astype(float)
        self.dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc)
                          for t in self.timestamps]

        # Decisions: 'accept', 'reject', None (undecided)
        self.decisions = [None] * len(seeds)
        self.current_idx = 0

        # Plot references to clean up between seeds
        self._patches = []

    def _find_bar_idx(self, ts):
        """Find the 1m bar index closest to a timestamp."""
        idx_arr = np.where(self.timestamps <= ts)[0]
        if len(idx_arr) == 0:
            return 0
        return int(idx_arr[-1])

    def _draw_seed(self):
        """Draw the current seed on the chart."""
        # Clear previous seed drawings
        for p in self._patches:
            try:
                p.remove()
            except Exception:
                pass
        self._patches = []

        seed = self.all_seeds[self.current_idx]
        ts_start = seed['ts_start']
        ts_end = seed['ts_end']
        direction = seed['direction']

        # Find bar indices
        si = self._find_bar_idx(ts_start)
        ei = self._find_bar_idx(ts_end)

        # Lookback window (10 bars before start)
        lb_start = max(0, si - seed.get('lookback_bars', 10))
        lb_end = si

        # Context window for zoom
        ctx_start = max(0, lb_start - CONTEXT_BARS)
        ctx_end = min(len(self.dt_stamps) - 1, ei + CONTEXT_BARS)

        # Colors
        regime_color = '#00C853' if direction == 'LONG' else '#FF1744'

        # Draw lookback zone (blue)
        if lb_start < lb_end and lb_end < len(self.dt_stamps):
            p = self.ax.axvspan(self.dt_stamps[lb_start], self.dt_stamps[lb_end],
                                alpha=0.15, color='#2196F3', label='Lookback')
            self._patches.append(p)

            # Lookback label
            mid_lb = (lb_start + lb_end) // 2
            if mid_lb < len(self.dt_stamps):
                t = self.ax.text(self.dt_stamps[mid_lb], self.ax.get_ylim()[1],
                                 'LOOKBACK', fontsize=8, ha='center', va='top',
                                 color='#1565C0', fontweight='bold', alpha=0.7)
                self._patches.append(t)

        # Draw regime zone (green/red)
        if si < ei and ei < len(self.dt_stamps):
            p = self.ax.axvspan(self.dt_stamps[si], self.dt_stamps[ei],
                                alpha=0.12, color=regime_color, label=f'Regime ({direction})')
            self._patches.append(p)

            # Regime price path highlighted
            regime_slice = slice(si, ei + 1)
            line, = self.ax.plot(self.dt_stamps[regime_slice],
                                 self.close[regime_slice],
                                 color=regime_color, linewidth=2.5, alpha=0.8)
            self._patches.append(line)

            # Entry/exit markers
            marker_entry = '^' if direction == 'LONG' else 'v'
            m1 = self.ax.scatter([self.dt_stamps[si]], [self.close[si]],
                                 color=regime_color, s=200, zorder=5,
                                 marker=marker_entry, edgecolors='black', linewidths=0.5)
            m2 = self.ax.scatter([self.dt_stamps[ei]], [self.close[ei]],
                                 color='black', s=120, zorder=5, marker='s')
            self._patches.extend([m1, m2])

        # Zoom to context window
        import matplotlib.dates as mdates
        if ctx_start < ctx_end:
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[ctx_start]),
                             mdates.date2num(self.dt_stamps[ctx_end]))
            self._autofit_y()

        # Update title with seed stats
        self._update_title()
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
            pad = (vis_high - vis_low) * 0.08
            self.ax.set_ylim(vis_low - pad, vis_high + pad)

    def _update_title(self):
        """Update title with seed stats and progress."""
        seed = self.all_seeds[self.current_idx]
        i = self.current_idx
        n = len(self.all_seeds)

        direction = seed['direction']
        mfe = seed['mfe_ticks']
        mae = seed['mae_ticks']
        dur = seed['duration_mins']
        rr = mfe / max(mae, 1)

        # Decision counts
        accepted = sum(1 for d in self.decisions if d == 'accept')
        rejected = sum(1 for d in self.decisions if d == 'reject')
        undecided = n - accepted - rejected

        # Current decision status
        dec = self.decisions[i]
        if dec == 'accept':
            status = '[ACCEPTED]'
            status_color = 'green'
        elif dec == 'reject':
            status = '[REJECTED]'
            status_color = 'red'
        else:
            status = '[UNDECIDED]'
            status_color = 'gray'

        # Date from seed
        date_str = seed.get('start_time', '')
        if not date_str:
            dt = datetime.fromtimestamp(seed['ts_start'], tz=timezone.utc)
            date_str = dt.strftime('%Y-%m-%d %H:%M')

        dir_color = 'green' if direction == 'LONG' else 'red'

        self.ax.set_title(
            f"Seed {i+1}/{n}  |  {date_str}  |  "
            f"{direction} {dur:.0f}m  |  "
            f"MFE: ${mfe*TICK_VALUE:.0f} ({mfe:.0f}t)  "
            f"MAE: ${mae*TICK_VALUE:.0f} ({mae:.0f}t)  "
            f"R:R 1:{rr:.1f}  {status}\n"
            f"Progress: {accepted} accepted, {rejected} rejected, {undecided} remaining  |  "
            f"[Y=accept  N=reject  S=skip  Left/Right=nav  Q=save+quit]",
            fontsize=10, fontweight='bold'
        )

    def _on_key(self, event):
        """Handle key press for seed decisions and navigation."""
        if event.key in ('y', 'Y', 'enter', 'return'):
            self.decisions[self.current_idx] = 'accept'
            self._print_decision('ACCEPT')
            self._advance(1)

        elif event.key in ('n', 'N', 'delete', 'backspace'):
            self.decisions[self.current_idx] = 'reject'
            self._print_decision('REJECT')
            self._advance(1)

        elif event.key in ('s', 'S'):
            self.decisions[self.current_idx] = None
            self._print_decision('SKIP')
            self._advance(1)

        elif event.key == 'left':
            self._advance(-1)

        elif event.key == 'right':
            self._advance(1)

        elif event.key == 'up':
            self._zoom(0.5)

        elif event.key == 'down':
            self._zoom(2.0)

        elif event.key in ('f', 'F'):
            self._fit_day()

        elif event.key in ('q', 'Q'):
            self._save()
            plt.close(self.fig)

    def _print_decision(self, action):
        """Print current seed decision to console."""
        seed = self.all_seeds[self.current_idx]
        i = self.current_idx
        mfe = seed['mfe_ticks']
        mae = seed['mae_ticks']
        rr = mfe / max(mae, 1)
        date_str = datetime.fromtimestamp(seed['ts_start'], tz=timezone.utc).strftime('%Y-%m-%d %H:%M')
        print(f"  [{i+1}/{len(self.all_seeds)}] {action}: "
              f"{seed['direction']} {seed['duration_mins']:.0f}m | "
              f"MFE ${mfe*TICK_VALUE:.0f} MAE ${mae*TICK_VALUE:.0f} R:R 1:{rr:.1f} | "
              f"{date_str}")

    def _advance(self, delta):
        """Move to next/prev seed."""
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.all_seeds):
            self.current_idx = new_idx
            self._draw_seed()
        elif new_idx >= len(self.all_seeds):
            # Reached the end — check if all decided
            undecided = sum(1 for d in self.decisions if d is None)
            if undecided == 0:
                print("\n  All seeds reviewed!")
                self._save()
                plt.close(self.fig)
            else:
                print(f"  End of list ({undecided} undecided). Press Q to save.")

    def _zoom(self, factor):
        """Zoom in/out."""
        import matplotlib.dates as mdates
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        half_w = (xlim[1] - xlim[0]) / 2 * factor

        x_min = mdates.date2num(self.dt_stamps[0])
        x_max = mdates.date2num(self.dt_stamps[-1])

        self.ax.set_xlim(max(x_min, center - half_w),
                         min(x_max, center + half_w))
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _fit_day(self):
        """Fit view to the full day of the current seed."""
        import matplotlib.dates as mdates
        seed = self.all_seeds[self.current_idx]
        ts = seed['ts_start']
        # Find day boundaries
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        day_start = dt.replace(hour=0, minute=0, second=0).timestamp()
        day_end = day_start + 86400

        mask = (self.timestamps >= day_start) & (self.timestamps < day_end)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[idx[0]]),
                             mdates.date2num(self.dt_stamps[idx[-1]]))
            self._autofit_y()
            self.fig.canvas.draw_idle()

    def _save(self):
        """Save accepted seeds to a new JSON file."""
        accepted = [self.all_seeds[i] for i, d in enumerate(self.decisions)
                    if d == 'accept']
        rejected = [self.all_seeds[i] for i, d in enumerate(self.decisions)
                    if d == 'reject']
        undecided = [self.all_seeds[i] for i, d in enumerate(self.decisions)
                     if d is None]

        # Undecided seeds are kept (conservative — don't discard without review)
        kept = accepted + undecided

        n_total = len(self.all_seeds)
        print(f"\n{'='*60}")
        print(f"  INSPECTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Total:     {n_total}")
        print(f"  Accepted:  {len(accepted)}")
        print(f"  Rejected:  {len(rejected)}")
        print(f"  Undecided: {len(undecided)} (kept)")
        print(f"  Final:     {len(kept)} seeds")

        if not kept:
            print("  No seeds to save.")
            return

        # Save reviewed seeds
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.dirname(self.source_path)
        out_name = f"imr_seeds_reviewed_{ts_tag}.json"
        out_path = os.path.join(out_dir, out_name)

        # Group by date for compatibility with seed_loader
        days = {}
        for seed in kept:
            dt = datetime.fromtimestamp(seed['ts_start'], tz=timezone.utc)
            date_str = dt.strftime('%Y-%m-%d')
            if date_str not in days:
                days[date_str] = {
                    'date': date_str,
                    'timeframe': '1m',
                    'created': ts_tag,
                    'n_seeds': 0,
                    'source': 'imr_reviewed',
                    'seeds': [],
                }
            days[date_str]['seeds'].append(seed)
            days[date_str]['n_seeds'] = len(days[date_str]['seeds'])

        output = {
            'created': ts_tag,
            'source': 'imr_reviewed',
            'n_days': len(days),
            'n_seeds_total': len(kept),
            'review_stats': {
                'total_inspected': n_total,
                'accepted': len(accepted),
                'rejected': len(rejected),
                'undecided': len(undecided),
            },
            'days': days,
        }

        with open(out_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"  Saved: {out_path}")

        # Also save rejection log (useful for tuning filters)
        if rejected:
            rej_path = os.path.join(out_dir, f"imr_seeds_rejected_{ts_tag}.json")
            with open(rej_path, 'w') as f:
                json.dump({'seeds': rejected, 'n': len(rejected)}, f, indent=2)
            print(f"  Rejections: {rej_path}")

    def run(self):
        """Launch the inspector window."""
        import matplotlib.dates as mdates
        from matplotlib.widgets import Cursor

        self.fig, self.ax = plt.subplots(1, 1, figsize=(20, 8))
        self.fig.subplots_adjust(bottom=0.06)
        self.ax.set_facecolor('#FAFAFA')

        # Plot full price series (light, in background)
        self.ax.plot(self.dt_stamps, self.close, color='#90A4AE',
                     linewidth=0.8, alpha=0.6)
        self.ax.fill_between(self.dt_stamps, self.low, self.high,
                             alpha=0.04, color='#90A4AE')

        self.ax.set_ylabel('Price', fontsize=11)
        self.ax.grid(True, alpha=0.2)

        # Crosshair
        self.cursor = Cursor(self.ax, useblit=True, color='gray',
                             linewidth=0.5, linestyle='--')

        # Connect keys
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # Draw first seed
        self._draw_seed()

        n = len(self.all_seeds)
        accepted_pre = sum(1 for d in self.decisions if d == 'accept')
        print(f"\n  Seed Inspector ready -- {n} seeds to review")
        print(f"  Y/Enter=accept  N/Del=reject  S=skip  Left/Right=nav  Q=save+quit")
        print(f"  Up/Down=zoom  F=fit day\n")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show(block=True)


def load_seeds_from_file(path: str) -> list:
    """Load seeds from combined I-MR JSON (or daily format)."""
    with open(path) as f:
        data = json.load(f)

    seeds = []
    if 'days' in data:
        # Combined format: {days: {date: {seeds: [...]}}}
        for date_str, day_data in sorted(data['days'].items()):
            if isinstance(day_data, dict) and 'seeds' in day_data:
                for s in day_data['seeds']:
                    # Add start_time if missing
                    if 'start_time' not in s:
                        dt = datetime.fromtimestamp(s['ts_start'], tz=timezone.utc)
                        s['start_time'] = dt.strftime('%Y-%m-%d %H:%M')
                    seeds.append(s)
    elif 'seeds' in data:
        # Single-day format
        seeds = data['seeds']
    else:
        print(f"ERROR: Unrecognized seed format in {path}")

    return seeds


def main():
    parser = argparse.ArgumentParser(description='Seed Inspector (visual QA)')
    parser.add_argument('--seeds', default=None,
                        help='Seed JSON file (auto-detects latest imr_auto if omitted)')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root for 1m chart data')
    parser.add_argument('--start-day', default=None,
                        help='Start from this date (YYYY-MM-DD), skip earlier seeds')
    parser.add_argument('--min-mfe', type=float, default=0,
                        help='Only inspect seeds with MFE >= this (ticks)')
    parser.add_argument('--max-seeds', type=int, default=0,
                        help='Limit number of seeds to inspect (0 = all)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Randomize seed order (sample across days)')
    args = parser.parse_args()

    # Find seed file
    if args.seeds:
        seed_path = args.seeds
    else:
        # Auto-detect latest in imr_auto
        auto_dir = Path('DATA/regime_seeds/imr_auto')
        if auto_dir.exists():
            candidates = sorted(auto_dir.glob('imr_seeds_all_*.json'), reverse=True)
            if candidates:
                seed_path = str(candidates[0])
            else:
                print("ERROR: No imr_auto seed files found. Run imr_to_seeds.py first.")
                sys.exit(1)
        else:
            print("ERROR: DATA/regime_seeds/imr_auto/ not found. Run imr_to_seeds.py first.")
            sys.exit(1)

    print(f"Seed Inspector")
    print(f"  Seeds: {seed_path}")

    # Load seeds
    seeds = load_seeds_from_file(seed_path)
    if not seeds:
        print("ERROR: No seeds loaded")
        sys.exit(1)
    print(f"  Loaded: {len(seeds)} seeds")

    # Apply filters
    if args.start_day:
        dt_cutoff = datetime.strptime(args.start_day, '%Y-%m-%d').replace(tzinfo=timezone.utc)
        ts_cutoff = dt_cutoff.timestamp()
        seeds = [s for s in seeds if s['ts_start'] >= ts_cutoff]
        print(f"  After start-day filter ({args.start_day}): {len(seeds)}")

    if args.min_mfe > 0:
        seeds = [s for s in seeds if s['mfe_ticks'] >= args.min_mfe]
        print(f"  After MFE filter (>={args.min_mfe}t): {len(seeds)}")

    if args.shuffle:
        import random
        random.shuffle(seeds)
        print(f"  Shuffled order")

    if args.max_seeds > 0:
        seeds = seeds[:args.max_seeds]
        print(f"  Limited to {len(seeds)} seeds")

    if not seeds:
        print("No seeds to inspect after filters.")
        sys.exit(0)

    # Load 1m chart data
    print(f"\n  Loading 1m ATLAS data...")
    df_1m = load_atlas_tf(args.data_dir, '1m')
    if df_1m.empty:
        print("ERROR: No 1m data found")
        sys.exit(1)
    print(f"  {len(df_1m)} bars loaded")

    # Launch inspector
    inspector = SeedInspector(seeds, df_1m, seed_path)
    inspector.run()


if __name__ == '__main__':
    main()
