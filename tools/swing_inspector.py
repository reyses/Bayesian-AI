#!/usr/bin/env python
"""
Swing Inspector -- grade continuous swing groups on a price chart.

Shows groups of consecutive trades (~10 at a time) as a snapshot.
User grades each snapshot for quality of swing marking.

Keys:
    A / 1       = Grade A (excellent swings)
    B / 2       = Grade B (good, minor issues)
    C / 3       = Grade C (acceptable, some bad swings)
    D / 4       = Grade D (poor, many wrong pivots)
    F / 5       = Grade F (fail, unusable)
    Left/Right  = navigate to prev/next group without grading
    Up/Down     = zoom in/out
    +/-         = show more/fewer trades per group
    R           = toggle showing individual trade stats
    Q           = save + quit

Usage:
    python tools/swing_inspector.py --seeds DATA/regime_seeds/auto_swing/auto_seeds_all_*.json
    python tools/swing_inspector.py --seeds DATA/regime_seeds/auto_swing/auto_seeds_all_*.json --group-size 15
    python tools/swing_inspector.py --seeds DATA/regime_seeds/auto_swing/auto_seeds_all_*.json --start-day 2025-03-01
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

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
TICK_VALUE = 0.50

# Grade colors for the title bar
GRADE_COLORS = {
    'A': '#4CAF50',  # green
    'B': '#8BC34A',  # light green
    'C': '#FFC107',  # amber
    'D': '#FF9800',  # orange
    'F': '#F44336',  # red
}

GRADE_LABELS = {
    'A': 'Excellent',
    'B': 'Good',
    'C': 'Acceptable',
    'D': 'Poor',
    'F': 'Fail',
}


class SwingInspector:
    """Group-based continuous swing inspector with pivot editing."""

    def __init__(self, day_groups, df_1m, source_path, group_size=10):
        self.day_groups = day_groups       # {date_str: [seeds...]} sorted by time
        self.df_1m = df_1m
        self.source_path = source_path
        self.group_size = group_size
        self.show_stats = False
        self.edit_mode = False
        self.edits_made = 0

        self.close = df_1m['close'].values.astype(float)
        self.high = df_1m['high'].values.astype(float)
        self.low = df_1m['low'].values.astype(float)
        self.timestamps = df_1m['timestamp'].values.astype(float)
        self.dt_stamps = [datetime.fromtimestamp(t, tz=timezone.utc)
                          for t in self.timestamps]

        # Build flat list of groups: (date, start_trade_idx, seeds_in_group)
        self.groups = []
        self._build_groups()

        # Grades: per-group decision
        self.grades = [None] * len(self.groups)
        self.current_idx = 0
        self._patches = []
        self._pivot_markers = []  # edit mode: clickable pivot dots

    def _build_groups(self):
        """Split each day's seeds into chunks of group_size."""
        self.groups = []
        for date_str in sorted(self.day_groups.keys()):
            seeds = self.day_groups[date_str]
            for i in range(0, len(seeds), self.group_size):
                chunk = seeds[i:i + self.group_size]
                self.groups.append({
                    'date': date_str,
                    'offset': i,
                    'seeds': chunk,
                })

    def _find_bar_idx(self, ts):
        """Find the 1m bar index closest to a timestamp."""
        idx_arr = np.where(self.timestamps <= ts)[0]
        if len(idx_arr) == 0:
            return 0
        return int(idx_arr[-1])

    def _draw_group(self):
        """Draw the current group of consecutive trades."""
        for p in self._patches:
            try:
                p.remove()
            except Exception:
                pass
        self._patches = []

        group = self.groups[self.current_idx]
        seeds = group['seeds']

        if not seeds:
            return

        # Find overall bar range for this group
        first_ts = seeds[0]['ts_start']
        last_ts = seeds[-1]['ts_end']
        first_bi = self._find_bar_idx(first_ts)
        last_bi = self._find_bar_idx(last_ts)

        # Add context padding (5% of range on each side, min 5 bars)
        span = max(last_bi - first_bi, 10)
        pad = max(int(span * 0.08), 5)
        ctx_start = max(0, first_bi - pad)
        ctx_end = min(len(self.dt_stamps) - 1, last_bi + pad)

        # Draw each trade as alternating colored zones
        colors_long = ['#4CAF50', '#81C784']    # green shades
        colors_short = ['#F44336', '#E57373']   # red shades

        for j, seed in enumerate(seeds):
            si = self._find_bar_idx(seed['ts_start'])
            ei = self._find_bar_idx(seed['ts_end'])
            direction = seed['direction']

            if si >= len(self.dt_stamps) or ei >= len(self.dt_stamps) or ei <= si:
                continue

            # Alternating shade for visual separation
            if direction == 'LONG':
                color = colors_long[j % 2]
            else:
                color = colors_short[j % 2]

            # Shaded zone
            p = self.ax.axvspan(self.dt_stamps[si], self.dt_stamps[ei],
                                alpha=0.12, color=color)
            self._patches.append(p)

            # Highlighted price path
            slc = slice(si, ei + 1)
            line, = self.ax.plot(self.dt_stamps[slc], self.close[slc],
                                 color=color, linewidth=2.0, alpha=0.85)
            self._patches.append(line)

            # Entry marker (triangle up/down)
            marker = '^' if direction == 'LONG' else 'v'
            m = self.ax.scatter([self.dt_stamps[si]], [self.close[si]],
                                color=color, s=100, zorder=5,
                                marker=marker, edgecolors='black', linewidths=0.5)
            self._patches.append(m)

            # Trade number label at entry
            mid_x = self.dt_stamps[(si + ei) // 2]
            y_pos = self.low[si:ei+1].min() if direction == 'SHORT' else self.high[si:ei+1].max()
            va = 'top' if direction == 'SHORT' else 'bottom'
            t = self.ax.text(mid_x, y_pos,
                             f'{j+1}', fontsize=7, ha='center', va=va,
                             color=color, fontweight='bold', alpha=0.8,
                             bbox=dict(boxstyle='round,pad=0.1',
                                       facecolor='white', alpha=0.6, edgecolor='none'))
            self._patches.append(t)

            # Optional: show individual trade stats
            if self.show_stats:
                mfe = seed.get('mfe_ticks', 0)
                dur = seed.get('duration_mins', 0)
                stat_text = f'{dur:.0f}m\n{mfe:.0f}t'
                mid_bi = (si + ei) // 2
                if mid_bi < len(self.dt_stamps):
                    y_mid = (self.close[si] + self.close[ei]) / 2
                    t2 = self.ax.text(self.dt_stamps[mid_bi], y_mid,
                                      stat_text, fontsize=6, ha='center', va='center',
                                      color='#333', alpha=0.7,
                                      bbox=dict(boxstyle='round,pad=0.15',
                                                facecolor='white', alpha=0.7,
                                                edgecolor='#999', linewidth=0.5))
                    self._patches.append(t2)

        # Exit marker for last trade
        last_seed = seeds[-1]
        last_ei = self._find_bar_idx(last_seed['ts_end'])
        if last_ei < len(self.dt_stamps):
            m2 = self.ax.scatter([self.dt_stamps[last_ei]], [self.close[last_ei]],
                                 color='black', s=80, zorder=5, marker='s')
            self._patches.append(m2)

        # Edit mode: show pivot dots at each trade boundary
        self._pivot_markers = []
        if self.edit_mode:
            pivot_indices = set()
            for seed in seeds:
                pivot_indices.add(self._find_bar_idx(seed['ts_start']))
                pivot_indices.add(self._find_bar_idx(seed['ts_end']))
            for pi in sorted(pivot_indices):
                if pi < len(self.dt_stamps):
                    m = self.ax.scatter([self.dt_stamps[pi]], [self.close[pi]],
                                        color='#FFD600', s=180, zorder=10,
                                        marker='D', edgecolors='black', linewidths=1.5)
                    self._patches.append(m)
                    self._pivot_markers.append(pi)

        # Set view
        if ctx_start < ctx_end:
            self.ax.set_xlim(mdates.date2num(self.dt_stamps[ctx_start]),
                             mdates.date2num(self.dt_stamps[ctx_end]))
            self._autofit_y()

        self._update_title()
        self.fig.canvas.draw_idle()

    def _autofit_y(self):
        """Auto-fit Y axis to visible data range."""
        xlim = self.ax.get_xlim()
        bar_nums = mdates.date2num(self.dt_stamps)
        mask = (bar_nums >= xlim[0]) & (bar_nums <= xlim[1])
        if mask.any():
            vis_low = self.low[mask].min()
            vis_high = self.high[mask].max()
            pad = (vis_high - vis_low) * 0.08
            self.ax.set_ylim(vis_low - pad, vis_high + pad)

    def _update_title(self):
        """Update title with group stats and progress."""
        group = self.groups[self.current_idx]
        seeds = group['seeds']
        i = self.current_idx
        n = len(self.groups)

        # Group stats
        n_trades = len(seeds)
        n_long = sum(1 for s in seeds if s['direction'] == 'LONG')
        n_short = n_trades - n_long
        total_mfe = sum(s.get('mfe_ticks', 0) for s in seeds)
        avg_mfe = total_mfe / max(n_trades, 1)
        total_dur = sum(s.get('duration_mins', 0) for s in seeds)
        avg_dur = total_dur / max(n_trades, 1)
        span_mins = total_dur

        # Alternation rate for this group
        if n_trades > 1:
            alts = sum(1 for k in range(1, n_trades)
                       if seeds[k]['direction'] != seeds[k-1]['direction'])
            alt_pct = alts / (n_trades - 1) * 100
        else:
            alt_pct = 0

        # Grade info
        grade = self.grades[i]
        if grade:
            grade_text = f'[Grade: {grade} - {GRADE_LABELS[grade]}]'
        else:
            grade_text = '[UNGRADED]'

        # Progress
        graded = sum(1 for g in self.grades if g is not None)
        grade_dist = defaultdict(int)
        for g in self.grades:
            if g:
                grade_dist[g] += 1
        dist_str = ' '.join(f'{k}:{v}' for k, v in sorted(grade_dist.items()))

        date_str = group['date']
        offset = group['offset']

        mode_str = 'EDIT MODE (click=del pivot, right-click=add)' if self.edit_mode else 'GRADE MODE'
        edits_str = f'  [{self.edits_made} edits]' if self.edits_made > 0 else ''

        self.ax.set_title(
            f"Group {i+1}/{n}  |  {date_str} trades {offset+1}-{offset+n_trades}  |  "
            f"{n_trades} trades ({n_long}L/{n_short}S)  |  "
            f"Span: {span_mins:.0f}m  Avg: {avg_dur:.1f}m/trade  "
            f"Avg MFE: {avg_mfe:.0f}t  Alt: {alt_pct:.0f}%  "
            f"{grade_text}\n"
            f"[{mode_str}]{edits_str}  Graded: {graded}/{n}  [{dist_str}]  |  "
            f"[E=edit  A-F=grade  Left/Right=nav  +/-=size({self.group_size})  "
            f"R=stats  Q=save+quit]",
            fontsize=9, fontweight='bold'
        )

    def _on_key(self, event):
        """Handle key press."""
        key = event.key

        # Edit mode toggle
        if key in ('e', 'E'):
            self.edit_mode = not self.edit_mode
            mode = 'EDIT' if self.edit_mode else 'GRADE'
            print(f"  -> {mode} mode")
            self._draw_group()
            return

        # Grading (only in grade mode)
        if not self.edit_mode:
            if key in ('a', 'A', '1'):
                self._set_grade('A')
            elif key in ('b', 'B', '2'):
                self._set_grade('B')
            elif key in ('c', 'C', '3'):
                self._set_grade('C')
            elif key in ('d', 'D', '4'):
                self._set_grade('D')
            elif key in ('f', 'F', '5'):
                self._set_grade('F')

        # Navigation
        if key == 'left':
            self._advance(-1)
        elif key == 'right':
            self._advance(1)

        # Zoom
        elif key == 'up':
            self._zoom(0.5)
        elif key == 'down':
            self._zoom(2.0)

        # Group size
        elif key in ('+', '='):
            self.group_size = min(50, self.group_size + 2)
            self._rebuild_groups()
        elif key in ('-', '_'):
            self.group_size = max(4, self.group_size - 2)
            self._rebuild_groups()

        # Toggle stats
        elif key in ('r', 'R'):
            self.show_stats = not self.show_stats
            self._draw_group()

        # Undo last edit
        elif key in ('z', 'Z') and self.edit_mode:
            # Simple undo: not implemented yet, just redraw
            pass

        # Save + quit
        elif key in ('q', 'Q'):
            self._save()
            plt.close(self.fig)

    def _on_click(self, event):
        """Handle mouse click for pivot editing."""
        if not self.edit_mode or event.inaxes != self.ax:
            return

        # Convert click x to bar index
        click_date = mdates.num2date(event.xdata)
        click_ts = click_date.replace(tzinfo=timezone.utc).timestamp()
        click_bi = self._find_bar_idx(click_ts)

        group = self.groups[self.current_idx]
        seeds = group['seeds']
        date_str = group['date']

        if event.button == 1:
            # LEFT CLICK = delete nearest pivot (merge two swings)
            self._delete_pivot(click_bi, seeds, date_str)
        elif event.button == 3:
            # RIGHT CLICK = add pivot here (split a swing)
            self._add_pivot(click_bi, seeds, date_str)

    def _delete_pivot(self, click_bi, seeds, date_str):
        """Delete the pivot nearest to click, merging two swings."""
        if len(seeds) < 2:
            return

        # Find which pivot boundary is closest to the click
        best_dist = 999999
        best_idx = -1  # index in seeds where this is the END pivot

        for j, seed in enumerate(seeds):
            ei = self._find_bar_idx(seed['ts_end'])
            dist = abs(ei - click_bi)
            # Don't allow deleting the very first start or very last end
            if j < len(seeds) - 1 and dist < best_dist:
                best_dist = dist
                best_idx = j

        if best_idx < 0 or best_dist > 10:
            return  # too far from any pivot

        # Merge seed[best_idx] and seed[best_idx+1] into one seed
        s1 = seeds[best_idx]
        s2 = seeds[best_idx + 1]

        merged = dict(s1)  # copy first seed
        merged['ts_end'] = s2['ts_end']
        merged['end_idx'] = s2.get('end_idx', 0)
        merged['exit_price'] = s2['exit_price']

        # Recalculate direction from start to end
        entry_p = merged['entry_price']
        exit_p = merged['exit_price']
        merged['direction'] = 'LONG' if exit_p > entry_p else 'SHORT'

        # Recalculate basic stats
        change = exit_p - entry_p
        merged['change_ticks'] = round(change / TICK_SIZE, 1)
        merged['change_dollars'] = round(change / TICK_SIZE * TICK_VALUE, 2)
        merged['duration_mins'] = round((merged['ts_end'] - merged['ts_start']) / 60.0, 1)
        merged['n_bars'] = s1.get('n_bars', 0) + s2.get('n_bars', 0)

        # MFE/MAE: take the max of both (conservative — not re-measured from 1s)
        merged['mfe_ticks'] = max(s1.get('mfe_ticks', 0), s2.get('mfe_ticks', 0))
        merged['mae_ticks'] = max(s1.get('mae_ticks', 0), s2.get('mae_ticks', 0))
        merged['mfe_dollars'] = round(merged['mfe_ticks'] * TICK_VALUE, 2)
        merged['mae_dollars'] = round(merged['mae_ticks'] * TICK_VALUE, 2)

        # Replace in seeds list
        seeds[best_idx] = merged
        seeds.pop(best_idx + 1)

        # Also update in day_groups
        self._update_day_seeds(date_str, seeds, self.groups[self.current_idx]['offset'])

        # Renumber trade_ids
        for k, s in enumerate(seeds):
            s['trade_id'] = k

        self.edits_made += 1
        print(f"  DELETED pivot between trades {best_idx+1}-{best_idx+2} -> merged "
              f"({merged['direction']} {merged['duration_mins']:.0f}m)")
        self._draw_group()

    def _add_pivot(self, click_bi, seeds, date_str):
        """Add a pivot at click position, splitting a swing in two."""
        if click_bi >= len(self.timestamps):
            return

        # Find which seed contains this bar
        target_idx = -1
        for j, seed in enumerate(seeds):
            si = self._find_bar_idx(seed['ts_start'])
            ei = self._find_bar_idx(seed['ts_end'])
            if si <= click_bi <= ei:
                target_idx = j
                break

        if target_idx < 0:
            return

        seed = seeds[target_idx]
        si = self._find_bar_idx(seed['ts_start'])
        ei = self._find_bar_idx(seed['ts_end'])

        # Don't split if too close to edges
        if click_bi - si < 2 or ei - click_bi < 2:
            return

        # Create two new seeds from the split
        split_ts = float(self.timestamps[click_bi])
        split_price = round(float(self.close[click_bi]), 2)
        entry_price = seed['entry_price']
        exit_price = seed['exit_price']

        s1 = dict(seed)
        s1['ts_end'] = split_ts
        s1['exit_price'] = split_price
        s1['end_idx'] = click_bi
        change1 = split_price - entry_price
        s1['direction'] = 'LONG' if change1 > 0 else 'SHORT'
        s1['change_ticks'] = round(change1 / TICK_SIZE, 1)
        s1['change_dollars'] = round(change1 / TICK_SIZE * TICK_VALUE, 2)
        s1['duration_mins'] = round((split_ts - seed['ts_start']) / 60.0, 1)
        s1['n_bars'] = click_bi - si + 1

        s2 = dict(seed)
        s2['ts_start'] = split_ts
        s2['entry_price'] = split_price
        s2['start_idx'] = click_bi
        s2['regime_start_idx'] = click_bi
        s2['lookback_start_idx'] = max(0, click_bi - 10)
        s2['lookback_timestamps'] = [float(self.timestamps[k])
                                      for k in range(max(0, click_bi - 10), click_bi)]
        change2 = exit_price - split_price
        s2['direction'] = 'LONG' if change2 > 0 else 'SHORT'
        s2['change_ticks'] = round(change2 / TICK_SIZE, 1)
        s2['change_dollars'] = round(change2 / TICK_SIZE * TICK_VALUE, 2)
        s2['duration_mins'] = round((seed['ts_end'] - split_ts) / 60.0, 1)
        s2['n_bars'] = ei - click_bi + 1

        # Replace the original with two new seeds
        seeds[target_idx] = s1
        seeds.insert(target_idx + 1, s2)

        # Update in day_groups
        self._update_day_seeds(date_str, seeds, self.groups[self.current_idx]['offset'])

        # Renumber
        for k, s in enumerate(seeds):
            s['trade_id'] = k

        self.edits_made += 1
        print(f"  ADDED pivot at bar {click_bi} -> split trade {target_idx+1} "
              f"into {s1['direction']} {s1['duration_mins']:.0f}m + "
              f"{s2['direction']} {s2['duration_mins']:.0f}m")
        self._draw_group()

    def _update_day_seeds(self, date_str, group_seeds, offset):
        """Push edited group seeds back into day_groups."""
        day_seeds = self.day_groups[date_str]
        # Replace the slice that corresponds to this group
        old_end = offset + len(self.groups[self.current_idx]['seeds'])
        # The group seeds reference is already the same list, but after add/delete
        # the count changed, so rebuild the day list
        new_day = day_seeds[:offset] + group_seeds + day_seeds[old_end:]
        self.day_groups[date_str] = new_day
        # Update the group reference
        self.groups[self.current_idx]['seeds'] = group_seeds

    def _set_grade(self, grade):
        """Set grade for current group and advance."""
        self.grades[self.current_idx] = grade
        group = self.groups[self.current_idx]
        n_trades = len(group['seeds'])
        print(f"  [{self.current_idx+1}/{len(self.groups)}] Grade {grade} "
              f"({GRADE_LABELS[grade]}): {group['date']} "
              f"trades {group['offset']+1}-{group['offset']+n_trades}")
        self._advance(1)

    def _advance(self, delta):
        """Move to next/prev group."""
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.groups):
            self.current_idx = new_idx
            self._draw_group()
        elif new_idx >= len(self.groups):
            ungraded = sum(1 for g in self.grades if g is None)
            if ungraded == 0:
                print("\n  All groups graded!")
                self._save()
                plt.close(self.fig)
            else:
                print(f"  End of list ({ungraded} ungraded). Press Q to save.")

    def _zoom(self, factor):
        """Zoom in/out."""
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        half_w = (xlim[1] - xlim[0]) / 2 * factor

        x_min = mdates.date2num(self.dt_stamps[0])
        x_max = mdates.date2num(self.dt_stamps[-1])

        self.ax.set_xlim(max(x_min, center - half_w),
                         min(x_max, center + half_w))
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _rebuild_groups(self):
        """Rebuild groups after group_size change."""
        old_group = self.groups[self.current_idx]
        old_date = old_group['date']
        old_offset = old_group['offset']

        self._build_groups()
        self.grades = [None] * len(self.groups)

        # Try to find the closest group to where we were
        best = 0
        for i, g in enumerate(self.groups):
            if g['date'] == old_date and g['offset'] <= old_offset:
                best = i
        self.current_idx = best
        self._draw_group()
        print(f"  Group size -> {self.group_size} ({len(self.groups)} groups)")

    def _save(self):
        """Save grading results."""
        graded = [(i, self.grades[i]) for i in range(len(self.grades))
                  if self.grades[i] is not None]

        n_total = len(self.groups)
        grade_dist = defaultdict(int)
        for _, g in graded:
            grade_dist[g] += 1

        print(f"\n{'='*60}")
        print(f"  INSPECTION COMPLETE")
        print(f"{'='*60}")
        print(f"  Groups:  {n_total}")
        print(f"  Graded:  {len(graded)}")
        for grade in 'ABCDF':
            if grade in grade_dist:
                print(f"    {grade} ({GRADE_LABELS[grade]}): {grade_dist[grade]}")

        # Compute quality score
        grade_scores = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
        if graded:
            avg_score = sum(grade_scores[g] for _, g in graded) / len(graded)
            print(f"\n  Quality Score: {avg_score:.2f}/4.00")
            # Acceptance rate (A+B+C)
            acceptable = sum(1 for _, g in graded if g in ('A', 'B', 'C'))
            print(f"  Acceptable (A+B+C): {acceptable}/{len(graded)} "
                  f"({acceptable/len(graded)*100:.0f}%)")

        if not graded:
            print("  No grades to save.")
            return

        # Save results
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.dirname(self.source_path)
        out_path = os.path.join(out_dir, f'swing_grades_{ts_tag}.json')

        results = {
            'created': ts_tag,
            'source': self.source_path,
            'group_size': self.group_size,
            'n_groups': n_total,
            'n_graded': len(graded),
            'grade_distribution': dict(grade_dist),
            'quality_score': round(avg_score, 2) if graded else None,
            'groups': [],
        }

        for i, grade in graded:
            group = self.groups[i]
            results['groups'].append({
                'date': group['date'],
                'offset': group['offset'],
                'n_trades': len(group['seeds']),
                'grade': grade,
                'first_ts': group['seeds'][0]['ts_start'],
                'last_ts': group['seeds'][-1]['ts_end'],
            })

        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {out_path}")

        # If edits were made, also save the edited seeds
        if self.edits_made > 0:
            edited_path = os.path.join(out_dir, f'auto_seeds_edited_{ts_tag}.json')
            edited_data = {
                'created': ts_tag,
                'source': 'swing_inspector_edited',
                'original': self.source_path,
                'edits_made': self.edits_made,
                'n_days': len(self.day_groups),
                'n_seeds_total': sum(len(v) for v in self.day_groups.values()),
                'days': {},
            }
            for date_str, seeds in sorted(self.day_groups.items()):
                edited_data['days'][date_str] = {
                    'date': date_str,
                    'timeframe': '1m',
                    'n_seeds': len(seeds),
                    'source': 'auto_swing_edited',
                    'seeds': seeds,
                }
            with open(edited_path, 'w') as f:
                json.dump(edited_data, f, indent=2)
            print(f"  Edited seeds: {edited_path} ({self.edits_made} edits)")

    def run(self):
        """Launch the inspector window."""
        from matplotlib.widgets import Cursor

        self.fig, self.ax = plt.subplots(1, 1, figsize=(22, 8))
        self.fig.subplots_adjust(bottom=0.06)
        self.ax.set_facecolor('#FAFAFA')

        # Plot full price series (light background)
        self.ax.plot(self.dt_stamps, self.close, color='#90A4AE',
                     linewidth=0.7, alpha=0.5)
        self.ax.fill_between(self.dt_stamps, self.low, self.high,
                             alpha=0.03, color='#90A4AE')

        self.ax.set_ylabel('Price', fontsize=11)
        self.ax.grid(True, alpha=0.2)

        # Crosshair
        self.cursor = Cursor(self.ax, useblit=True, color='gray',
                             linewidth=0.5, linestyle='--')

        # Connect keys + clicks
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

        # Draw first group
        self._draw_group()

        n = len(self.groups)
        total_trades = sum(len(g['seeds']) for g in self.groups)
        print(f"\n  Swing Inspector ready -- {n} groups ({total_trades} trades)")
        print(f"  Group size: {self.group_size} trades/snapshot")
        print(f"  E = edit mode   A-F = grade   Left/Right = nav   +/- = group size")
        print(f"  Edit: left-click pivot = delete   right-click = add pivot")
        print(f"  R = toggle stats    Q = save+quit\n")

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.show(block=True)


def load_seeds_by_day(path: str) -> dict:
    """Load seeds grouped by day from combined JSON."""
    with open(path) as f:
        data = json.load(f)

    day_groups = {}
    if 'days' in data:
        for date_str, day_data in sorted(data['days'].items()):
            if isinstance(day_data, dict) and 'seeds' in day_data:
                seeds = day_data['seeds']
                # Sort by start time within day
                seeds.sort(key=lambda s: s['ts_start'])
                day_groups[date_str] = seeds
    elif 'seeds' in data:
        # Single-day format — group by date
        for s in data['seeds']:
            dt = datetime.fromtimestamp(s['ts_start'], tz=timezone.utc)
            d = dt.strftime('%Y-%m-%d')
            if d not in day_groups:
                day_groups[d] = []
            day_groups[d].append(s)
        for d in day_groups:
            day_groups[d].sort(key=lambda s: s['ts_start'])

    return day_groups


def main():
    parser = argparse.ArgumentParser(description='Swing Inspector (continuous group QA)')
    parser.add_argument('--seeds', default=None,
                        help='Seed JSON file (auto-detects latest auto_swing if omitted)')
    parser.add_argument('--data-dir', default='DATA/ATLAS',
                        help='ATLAS root for 1m chart data')
    parser.add_argument('--group-size', type=int, default=20,
                        help='Number of consecutive trades per snapshot (default: 20)')
    parser.add_argument('--start-day', default=None,
                        help='Start from this date (YYYY-MM-DD)')
    parser.add_argument('--end-day', default=None,
                        help='End at this date (YYYY-MM-DD)')
    parser.add_argument('--shuffle-days', action='store_true',
                        help='Randomize day order (AQL sampling)')
    parser.add_argument('--max-groups', type=int, default=0,
                        help='Limit number of groups to inspect')
    args = parser.parse_args()

    # Find seed file
    if args.seeds:
        seed_path = args.seeds
    else:
        # Auto-detect latest auto_swing
        auto_dir = Path('DATA/regime_seeds/auto_swing')
        if auto_dir.exists():
            candidates = sorted(auto_dir.glob('auto_seeds_all_*.json'), reverse=True)
            if candidates:
                seed_path = str(candidates[0])
            else:
                print("ERROR: No auto_swing seed files found. Run auto_swing_marker.py first.")
                sys.exit(1)
        else:
            # Try flat regime_seeds
            auto_dir = Path('DATA/regime_seeds')
            candidates = sorted(auto_dir.glob('auto_seeds_all_*.json'), reverse=True)
            if candidates:
                seed_path = str(candidates[0])
            else:
                print("ERROR: No auto seed files found.")
                sys.exit(1)

    print(f"Swing Inspector")
    print(f"  Seeds: {seed_path}")

    # Load seeds grouped by day
    day_groups = load_seeds_by_day(seed_path)
    if not day_groups:
        print("ERROR: No seeds loaded")
        sys.exit(1)

    total = sum(len(v) for v in day_groups.values())
    print(f"  Loaded: {total} seeds across {len(day_groups)} days")

    # Date filters
    if args.start_day:
        day_groups = {d: s for d, s in day_groups.items() if d >= args.start_day}
        print(f"  After start-day filter: {len(day_groups)} days")

    if args.end_day:
        day_groups = {d: s for d, s in day_groups.items() if d <= args.end_day}
        print(f"  After end-day filter: {len(day_groups)} days")

    if args.shuffle_days:
        import random
        keys = list(day_groups.keys())
        random.shuffle(keys)
        day_groups = {k: day_groups[k] for k in keys}
        print(f"  Shuffled day order (AQL sampling)")

    if not day_groups:
        print("No days to inspect.")
        sys.exit(0)

    # Load 1m chart data
    print(f"\n  Loading 1m ATLAS data...")
    df_1m = load_atlas_tf(args.data_dir, '1m')
    if df_1m.empty:
        print("ERROR: No 1m data found")
        sys.exit(1)
    print(f"  {len(df_1m)} bars loaded")

    # Launch inspector
    inspector = SwingInspector(day_groups, df_1m, seed_path,
                                group_size=args.group_size)

    if args.max_groups > 0 and len(inspector.groups) > args.max_groups:
        inspector.groups = inspector.groups[:args.max_groups]
        inspector.grades = [None] * len(inspector.groups)
        print(f"  Limited to {args.max_groups} groups")

    inspector.run()


if __name__ == '__main__':
    main()
