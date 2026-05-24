"""
Swing Inspector Plugin
Renders swing groups and handles A-F grading and interactive pivot editing.
"""
import argparse
import json
import pandas as pd
import numpy as np

from tools.viz.core.plugin import VizPlugin

class SwingInspectorPlugin(VizPlugin):
    
    def __init__(self, args):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--seeds', required=True, help='Path to auto_seeds.json')
        parser.add_argument('--group-size', type=int, default=15)
        self.args = parser.parse_args(args)
        
        self.day_groups = {} # {date_str: [seeds]}
        self.group_offset = 0 # Offset within current day's seeds
        self.edit_mode = False
        self.grades = {} # { (date_str, group_idx): grade }

    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)
        try:
            with open(self.args.seeds, 'r') as f:
                data = json.load(f)
            # Group by day
            if 'days' in data:
                for date_str, day_data in data['days'].items():
                    if 'seeds' in day_data:
                        self.day_groups[date_str] = day_data['seeds']
            elif 'seeds' in data:
                for s in data['seeds']:
                    d = pd.to_datetime(s['ts_start'], unit='s').strftime('%Y-%m-%d')
                    self.day_groups.setdefault(d, []).append(s)
            
            # Sort all days
            for d in self.day_groups:
                self.day_groups[d].sort(key=lambda x: x['ts_start'])
                
            print(f"[Plugin] Loaded {sum(len(v) for v in self.day_groups.values())} seeds across {len(self.day_groups)} days.")
        except Exception as e:
            print(f"[Plugin] Failed to load seeds: {e}")

    def draw(self, ax, time_range, patches_list):
        day_str = self.engine.days[self.engine.day_idx]
        seeds = self.day_groups.get(day_str, [])
        if not seeds:
            return
            
        # Get current group
        chunk = seeds[self.group_offset : self.group_offset + self.args.group_size]
        if not chunk:
            return
            
        # Fit engine view to this chunk
        ts_start = chunk[0]['ts_start']
        ts_end = chunk[-1]['ts_end']
        
        dt_start = pd.to_datetime(ts_start, unit='s', utc=True).tz_convert('America/New_York').tz_localize(None)
        dt_end = pd.to_datetime(ts_end, unit='s', utc=True).tz_convert('America/New_York').tz_localize(None)
        
        # Add 15min padding
        pad = pd.Timedelta(minutes=15)
        ax.set_xlim(dt_start - pad, dt_end + pad)
        
        # Draw alternating zones
        colors_long = ['#4CAF50', '#81C784']
        colors_short = ['#F44336', '#E57373']
        
        for j, s in enumerate(chunk):
            s_dt = pd.to_datetime(s['ts_start'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None)
            e_dt = pd.to_datetime(s['ts_end'], unit='s', utc=True).tz_convert('America/New_York').tz_localize(None)
            
            direction = s.get('direction', 'LONG')
            color = colors_long[j % 2] if direction == 'LONG' else colors_short[j % 2]
            
            p = ax.axvspan(s_dt, e_dt, alpha=0.15, color=color)
            patches_list.append(p)
            
            # Entry marker
            marker = '^' if direction == 'LONG' else 'v'
            en_px = s.get('entry_price', 0)
            m = ax.scatter([s_dt], [en_px], color=color, s=150, zorder=5, marker=marker, edgecolors='black')
            patches_list.append(m)
            
            # Trade label
            mid_x = s_dt + (e_dt - s_dt) / 2
            t = ax.text(mid_x, en_px, f'{j+1}', fontsize=8, ha='center',
                        color=color, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.6))
            patches_list.append(t)

    def on_key(self, event):
        day_str = self.engine.days[self.engine.day_idx]
        seeds = self.day_groups.get(day_str, [])
        
        k = event.key.lower() if event.key else ''
        
        # Navigation inside day
        if k == 'right':
            if self.group_offset + self.args.group_size < len(seeds):
                self.group_offset += self.args.group_size
                self.engine.draw()
                return True
            else:
                self.group_offset = 0 # Reset for next day
                return False # Let engine handle page down
                
        elif k == 'left':
            if self.group_offset - self.args.group_size >= 0:
                self.group_offset -= self.args.group_size
                self.engine.draw()
                return True
            else:
                return False
                
        elif k == 'e':
            self._toggle_edit()
            return True
            
        elif k in ('a', 'b', 'c', 'd', 'f') and not self.edit_mode:
            self._grade(k.upper())
            return True
            
        return False

    def on_click(self, event):
        if not self.edit_mode:
            return False
        # Stub for pivot editing
        print("[Plugin] Clicked in Edit Mode (implement merge/split logic here)")
        return True

    def get_title_stats(self) -> str:
        day_str = self.engine.days[self.engine.day_idx]
        seeds = self.day_groups.get(day_str, [])
        group_idx = self.group_offset // max(1, self.args.group_size)
        grade = self.grades.get((day_str, group_idx), "UNGRADED")
        
        mode = "EDIT" if self.edit_mode else "GRADE"
        return f"[{mode}]  |  Group {group_idx+1}  |  Grade: {grade}  |  (A-F to grade, E to edit)"

    def _toggle_edit(self):
        self.edit_mode = not self.edit_mode
        self.engine.draw()
        
    def _grade(self, grade):
        day_str = self.engine.days[self.engine.day_idx]
        seeds = self.day_groups.get(day_str, [])
        group_idx = self.group_offset // self.args.group_size
        self.grades[(day_str, group_idx)] = grade
        print(f"[Plugin] Grade {grade} saved for {day_str} group {group_idx}")
        
        # Auto advance
        if self.group_offset + self.args.group_size < len(seeds):
            self.group_offset += self.args.group_size
            self.engine.draw()

    def get_buttons(self) -> list:
        return [
            {'label': 'A', 'action': lambda: self._grade('A')},
            {'label': 'B', 'action': lambda: self._grade('B')},
            {'label': 'C', 'action': lambda: self._grade('C')},
            {'label': 'D', 'action': lambda: self._grade('D')},
            {'label': 'F', 'action': lambda: self._grade('F')},
            {'label': 'Edit', 'action': self._toggle_edit},
        ]

def get_plugin(unknown_args):
    return SwingInspectorPlugin(unknown_args)
