"""
Manual Peak Marker Plugin
Allows the user to manually click on the chart to mark peaks/troughs.
Saves the results to DATA/regime_seeds/human_peaks_{date}_{tf}.json.
Also overlays manual support/resistance levels from DATA/levels/levels_*.json.
"""
import argparse
import json
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

from tools.viz.core.plugin import VizPlugin

SEEDS_DIR = 'DATA/regime_seeds'
LEVELS_DIR = 'DATA/levels'

class ManualPeakMarkerPlugin(VizPlugin):
    
    def __init__(self, args):
        super().__init__()
        parser = argparse.ArgumentParser()
        parser.add_argument('--no-levels', action='store_true', help='Disable human levels overlay')
        self.args = parser.parse_args(args)
        
        self.peaks = []
        self.levels = []
        
    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)
        
        # Load human levels if enabled
        if not self.args.no_levels:
            self._load_human_levels()
            
        # Try to load existing peaks for the starting day
        self._load_existing_peaks()

    def _load_human_levels(self):
        if not os.path.isdir(LEVELS_DIR):
            return
            
        try:
            start_dt = pd.Timestamp(self.engine.dt.iloc[0]).tz_localize('UTC')
            end_dt = pd.Timestamp(self.engine.dt.iloc[-1]).tz_localize('UTC')
        except:
            return

        for f in sorted(glob.glob(os.path.join(LEVELS_DIR, 'levels_*.json'))):
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
            if src_dt < start_dt - pd.Timedelta(days=31) or src_dt > end_dt + pd.Timedelta(days=31):
                continue
                
            for lvl in data.get('levels', []):
                self.levels.append({
                    'price': float(lvl['price']),
                    'type': lvl.get('type', 'unknown'),
                    'src_date': src_date_str,
                    'color': lvl.get('color', '#888888'),
                })
                
        print(f"[Plugin] Loaded {len(self.levels)} manual support/resistance levels.")

    def _load_existing_peaks(self):
        day_str = self.engine.days[self.engine.day_idx]
        path = os.path.join(SEEDS_DIR, f'human_peaks_{day_str}_{self.engine.tf}.json')
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self.peaks = data.get('peaks', [])
            print(f"[Plugin] Loaded {len(self.peaks)} existing peaks from {path}")
        except Exception as e:
            print(f"[Plugin] Failed to load existing peaks: {e}")

    def draw(self, ax, time_range, patches_list):
        if self.engine.dt is None or len(self.engine.dt) == 0:
            return
            
        # Draw levels
        if self.levels:
            x_right = self.engine.dt.iloc[-1]
            y_min, y_max = self.engine.lows.min(), self.engine.highs.max()
            y_pad = (y_max - y_min) * 0.1
            
            for lvl in self.levels:
                p = lvl['price']
                if p < y_min - y_pad or p > y_max + y_pad:
                    continue
                color = '#CC0000' if lvl['type'] == 'resistance' else ('#0066CC' if lvl['type'] == 'support' else '#888888')
                line = ax.axhline(p, color=color, lw=0.8, alpha=0.45, ls='-', zorder=1)
                patches_list.append(line)
                txt = ax.text(x_right, p, f' {p:.0f} {lvl["src_date"][:7]}',
                              fontsize=7, color=color, alpha=0.6, va='center', ha='left', zorder=1)
                patches_list.append(txt)

        # Draw peaks
        for pi, peak in enumerate(self.peaks):
            idx = peak['bar_index']
            if idx >= len(self.engine.dt):
                continue
                
            snap_price = peak.get('price', 0)
            snap_label = peak.get('_snap', '?')
            dt_val = self.engine.dt.iloc[idx]
            
            m = ax.scatter([dt_val], [snap_price], marker='D', c='cyan', s=150, zorder=10, edgecolors='black', lw=1.5)
            patches_list.append(m)
            
            label = ax.text(dt_val, snap_price + 2, f'#{pi+1} {snap_label}\n{snap_price:.2f}',
                            fontsize=7, ha='center', color='cyan', fontweight='bold')
            patches_list.append(label)

    def _detect_direction(self, idx, lookback=5, lookahead=5):
        start = max(0, idx - lookback)
        end = min(len(self.engine.closes) - 1, idx + lookahead)
        
        price_before = self.engine.closes[start:idx]
        price_after = self.engine.closes[idx:end + 1]
        
        if len(price_before) < 2 or len(price_after) < 2:
            return 'UNKNOWN'
            
        trend_before = price_before[-1] - price_before[0]
        trend_after = price_after[-1] - price_after[0]
        
        if trend_before > 0 and trend_after < 0:
            return 'SHORT'
        elif trend_before < 0 and trend_after > 0:
            return 'LONG'
        elif trend_before > 0:
            return 'SHORT'
        elif trend_before < 0:
            return 'LONG'
        else:
            return 'UNKNOWN'

    def on_click(self, event):
        if event.button != 1:
            return False
            
        click_num = event.xdata
        if click_num is None:
            return False
            
        # Find nearest bar
        bar_nums = mdates.date2num(self.engine.dt)
        idx = int(np.argmin(np.abs(bar_nums - click_num)))
        
        # Check if clicking near an existing peak -> delete it
        for pi, existing in enumerate(self.peaks):
            if existing['bar_index'] == idx:
                self.peaks.pop(pi)
                self.engine.draw()
                return True
                
        # Auto detect direction
        direction = self._detect_direction(idx)
        
        # Snap to high/low
        click_price = event.ydata if event.ydata is not None else self.engine.closes[idx]
        dist_to_high = abs(click_price - self.engine.highs[idx])
        dist_to_low = abs(click_price - self.engine.lows[idx])
        
        if dist_to_high < dist_to_low:
            snap_price = float(self.engine.highs[idx])
            snap_label = 'H'
        else:
            snap_price = float(self.engine.lows[idx])
            snap_label = 'L'
            
        peak = {
            'bar_index': int(idx),
            'timestamp': float(self.engine.timestamps[idx]),
            'price': snap_price,
            'close': float(self.engine.closes[idx]),
            'high': float(self.engine.highs[idx]),
            'low': float(self.engine.lows[idx]),
            '_snap': snap_label,
            '_direction_hint': direction,
            'tf': self.engine.tf,
        }
        
        self.peaks.append(peak)
        self.engine.draw()
        return True

    def _save(self):
        day_str = self.engine.days[self.engine.day_idx]
        os.makedirs(SEEDS_DIR, exist_ok=True)
        path = os.path.join(SEEDS_DIR, f'human_peaks_{day_str}_{self.engine.tf}.json')
        
        out = {
            'date': day_str,
            'tf': self.engine.tf,
            'n_peaks': len(self.peaks),
            'peaks': self.peaks,
        }
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"[Plugin] Saved {len(self.peaks)} peaks to {path}")

    def on_key(self, event):
        k = event.key.lower() if event.key else ''
        if k == 'd' and self.peaks:
            self.peaks.pop()
            self.engine.draw()
            return True
        elif k == 's':
            self._save()
            return True
        return False

    def get_buttons(self):
        return [
            {'label': 'Save Peaks', 'action': self._save, 'tooltip': 'Save manually marked peaks to JSON'},
            {'label': 'Del Last', 'action': lambda: self.on_key(type('obj', (object,), {'key': 'd'})), 'tooltip': 'Delete the last marked peak'}
        ]

    def get_title_stats(self):
        return f"Manual Peaks: {len(self.peaks)}  |  Levels: {len(self.levels)}  |  Click to Mark  |  [S] Save  [D] Del"

def get_plugin(unknown_args):
    return ManualPeakMarkerPlugin(unknown_args)
