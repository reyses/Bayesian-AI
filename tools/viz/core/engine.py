"""
Core visualization engine for interactive price chart inspection.
Handles the main UI, panning, zooming, day navigation, and delegates drawing to plugins.
"""
import os
import glob
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
from matplotlib.widgets import Cursor

# Force interactive backend
for _bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
    try:
        import matplotlib.pyplot as plt
        plt.switch_backend(_bk)
        break
    except Exception:
        continue

import matplotlib.dates as mdates

REPO = Path(__file__).resolve().parent.parent.parent.parent
EXAMPLES_DIR = REPO / 'examples'
EXAMPLES_DIR.mkdir(exist_ok=True)
matplotlib.rcParams['savefig.directory'] = str(EXAMPLES_DIR)

RAW_NT8 = REPO / 'DATA/ATLAS_NT8'
RAW_ATLAS = REPO / 'DATA/ATLAS'
SETTINGS_PATH = REPO / 'DATA/viz_engine_settings.json'
TZ = 'America/New_York'


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(d, indent=2))
    except Exception as e:
        print(f'[Engine] settings save failed: {e}')


def list_days(tf='1m') -> list:
    """Find all days available in the given timeframe."""
    days = set()
    for root in (RAW_ATLAS, RAW_NT8):
        tf_dir = root / tf
        if tf_dir.exists():
            for p in glob.glob(str(tf_dir / '*.parquet')):
                days.add(Path(p).stem)
    return sorted(list(days))


def bars_path(day: str, tf: str) -> Path:
    """Try to load bars from ATLAS_NT8 first, then ATLAS."""
    nt8 = RAW_NT8 / tf / f'{day}.parquet'
    return nt8 if nt8.exists() else RAW_ATLAS / tf / f'{day}.parquet'


class VizEngine:
    """The central interactive visualization framework."""
    
    def __init__(self, plugin, initial_day=None, tf='1m'):
        self.plugin = plugin
        self.tf = tf
        self.days = list_days(tf=self.tf)
        if not self.days:
            raise FileNotFoundError(f"No {self.tf} ATLAS data found.")
            
        self._settings = load_settings()
        
        # Load day from args or fall back to last saved setting
        target_day = initial_day or self._settings.get('day')
        self.day_idx = self.days.index(target_day) if target_day in self.days else 0
        
        self.dt = None
        self.closes = None
        self.highs = None
        self.lows = None
        self.timestamps = None
        self.plugin = plugin
        
        # UI State
        self.show_ohlc = True
        self._tool_patches = []
        
        has_ind = getattr(self.plugin, 'requires_indicator_panel', False)
        
        if has_ind:
            self.fig, (self.ax, self.ax_ind) = plt.subplots(
                2, 1, figsize=(18, 8), sharex=True,
                gridspec_kw={'height_ratios': [3, 1]}
            )
            self.fig.subplots_adjust(bottom=0.20, top=0.92, left=0.05, right=0.98, hspace=0.05)
            self.ax_ind.set_facecolor('#FAFAFA')
            self.ax_ind.grid(True, alpha=0.2)
        else:
            self.fig, self.ax = plt.subplots(figsize=(18, 8))
            self.fig.subplots_adjust(bottom=0.20, top=0.92, left=0.05, right=0.98)
            self.ax_ind = None
            
        try:
            self.fig.canvas.manager.set_window_title(f"VizEngine - {plugin.__class__.__name__}")
        except Exception:
            pass
            
        self.ax.set_ylabel('Price', fontsize=11)
        self.ax.grid(True, alpha=0.2)
        self.ax.set_facecolor('#FAFAFA')
        
        # Crosshair cursor
        self.cursor = Cursor(self.ax, useblit=True, color='gray', linewidth=0.5, linestyle='--')
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('close_event', self._on_close)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_hover)
        
        # Tooltip text at the very bottom
        self.tooltip_text = self.fig.text(0.5, 0.005, '', ha='center', va='bottom', fontsize=9, color='gray')
        self._tooltips = {}
        
        # Load plugin
        self.plugin.setup(self)
        
        # Build UI Buttons
        self._buttons = []
        self._build_buttons()
        
        self._apply_window_geometry()
        
        self.load_day(fit=True)

    def _nav_to(self, idx):
        if 0 <= idx < len(self.days):
            self.day_idx = idx
            self.load_day(fit=True)

    def _nav_by(self, delta):
        self._nav_to(self.day_idx + delta)
        
    def _nav_fit(self):
        self._fit_all()
        self.draw()

    def _build_buttons(self):
        engine_actions = [
            ('|<', lambda: self._nav_to(0), 'First Day (Home)'),
            ('< Day', lambda: self._nav_by(-1), 'Previous Day (PgDn)'),
            ('Day >', lambda: self._nav_by(1), 'Next Day (PgUp)'),
            ('>|', lambda: self._nav_to(len(self.days)-1), 'Last Day (End)'),
            ('< Pan', lambda: self._pan(-1), 'Pan Left (Left Arrow)'),
            ('Pan >', lambda: self._pan(1), 'Pan Right (Right Arrow)'),
            ('Zoom +', lambda: self._zoom(0.5), 'Zoom In (Up Arrow)'),
            ('Zoom -', lambda: self._zoom(2.0), 'Zoom Out (Down Arrow)'),
            ('Fit All', lambda: self._nav_fit(), 'Fit View (R)')
        ]
        
        try:
            mgr = self.fig.canvas.manager
            toolbar = getattr(mgr, 'toolbar', None)
            if not toolbar:
                return
                
            import tkinter as tk
            
            tk.Label(toolbar, text=" | ").pack(side=tk.LEFT)
            
            def make_on_enter(text):
                def _enter(e):
                    self.tooltip_text.set_text(text)
                    self.fig.canvas.draw_idle()
                return _enter
                
            def on_leave(e):
                self.tooltip_text.set_text('')
                self.fig.canvas.draw_idle()
            
            # Engine buttons
            for lbl, act, tooltip in engine_actions:
                btn = tk.Button(toolbar, text=lbl, command=act)
                btn.pack(side=tk.LEFT, padx=1, pady=2)
                
                btn.bind("<Enter>", make_on_enter(tooltip))
                btn.bind("<Leave>", on_leave)
                
            # Plugin buttons
            if hasattr(self.plugin, 'get_buttons'):
                p_btns = self.plugin.get_buttons()
                if p_btns:
                    tk.Label(toolbar, text=" || Plugin: ").pack(side=tk.LEFT)
                    for pb in p_btns:
                        btn = tk.Button(toolbar, text=pb['label'], command=pb['action'])
                        btn.pack(side=tk.LEFT, padx=1, pady=2)
                        
                        if 'tooltip' in pb:
                            btn.bind("<Enter>", make_on_enter(pb['tooltip']))
                            btn.bind("<Leave>", on_leave)
                            
            if hasattr(self.plugin, 'get_sliders'):
                sliders = self.plugin.get_sliders()
                if sliders:
                    tk.Label(toolbar, text=" | ").pack(side=tk.LEFT)
                    for sl in sliders:
                        frame = tk.Frame(toolbar)
                        frame.pack(side=tk.LEFT, padx=5)
                        tk.Label(frame, text=sl['label'], font=("Arial", 7)).pack(side=tk.TOP)
                        scale = tk.Scale(frame, from_=sl['min'], to=sl['max'], resolution=sl['step'], 
                                         orient=tk.HORIZONTAL, length=80, showvalue=True, font=("Arial", 7))
                        scale.set(sl['valinit'])
                        scale.bind("<ButtonRelease-1>", lambda e, act=sl['action'], sc=scale: act(sc.get()))
                        scale.pack(side=tk.BOTTOM)
                            
        except Exception as e:
            print(f"[Engine] Could not build native toolbar widgets: {e}")

    def load_day(self, fit=False):
        """Resets the view to the current day and loads data."""
        self.loaded_day_indices = {self.day_idx}
        self._reload_data()
        
        day = self.days[self.day_idx]
        saved_day = self._settings.get('day')
        saved_xlim = self._settings.get('xlim')
        saved_ylim = self._settings.get('ylim')
        
        if day == saved_day and saved_xlim and saved_ylim and not fit:
            self.draw(fit=False)
            try:
                self.ax.set_xlim(saved_xlim)
                self.ax.set_ylim(saved_ylim)
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        else:
            self.draw(fit=True)
        
    def _reload_data(self):
        """Loads and concatenates all currently visible days."""
        dfs = []
        for idx in sorted(self.loaded_day_indices):
            day = self.days[idx]
            try:
                df = pd.read_parquet(bars_path(day, self.tf)).sort_values('timestamp').reset_index(drop=True)
                dfs.append(df)
            except Exception as e:
                print(f"[Engine] Could not load {day}: {e}")
                
        if not dfs:
            return
            
        df = pd.concat(dfs, ignore_index=True)
        self.dt = df['timestamp']
        self.closes = df['close'].values.astype(float)
        self.highs = df['high'].values.astype(float)
        self.lows = df['low'].values.astype(float)
        self.timestamps = df['timestamp'].values.astype(float)
        
        # Convert to local timezone
        self.dt = (pd.to_datetime(df['timestamp'], unit='s', utc=True)
                   .dt.tz_convert(TZ).dt.tz_localize(None))

    def draw(self, fit=False):
        """Redraws the engine base UI and invokes plugin drawing."""
        # Clear base lines
        for ln in self._price_lines:
            try:
                ln.remove()
            except Exception:
                pass
        self._price_lines.clear()
        
        # Draw base price
        line, = self.ax.plot(self.dt, self.closes, color='#90A4AE', linewidth=1.0, alpha=0.8, zorder=1)
        fill = self.ax.fill_between(self.dt, self.lows, self.highs, alpha=0.1, color='#90A4AE', zorder=0)
        self._price_lines.extend([line, fill])
        
        # Clear old plugin patches
        for p in self._tool_patches:
            try:
                p.remove()
            except Exception:
                pass
        self._tool_patches.clear()
        
        if fit:
            self._fit_all()
            
        view_l, view_r = self.ax.get_xlim()
        
        # Call plugin draw hook
        if hasattr(self.plugin, 'draw'):
            import inspect
            sig = inspect.signature(self.plugin.draw)
            if 'ax_ind' in sig.parameters:
                self.plugin.draw(self.ax, self.ax_ind, (self.dt.iloc[0], self.dt.iloc[-1]), self._tool_patches)
            else:
                self.plugin.draw(self.ax, (self.dt.iloc[0], self.dt.iloc[-1]), self._tool_patches)
                
        self.ax.set_xlim(view_l, view_r)
        self._update_title()
        self.fig.canvas.draw_idle()

    def _update_title(self):
        day = self.days[self.day_idx]
        plugin_title = self.plugin.get_title_stats()
        
        self.ax.set_title(
            f"{day} [{self.day_idx + 1}/{len(self.days)}]  |  {plugin_title}",
            fontsize=11, fontweight='bold', loc='left'
        )

    def _on_hover(self, ev):
        txt = self._tooltips.get(ev.inaxes, '')
        if self.tooltip_text.get_text() != txt:
            self.tooltip_text.set_text(txt)
            self.fig.canvas.draw_idle()

    def _on_key(self, ev):
        if ev.key is None:
            return
            
        # Give plugin first right of refusal
        if self.plugin.on_key(ev):
            return
            
        # Default engine handlers
        k = ev.key.lower()
        if k == 'pageup':
            self._nav_by(1)
        elif k == 'pagedown':
            self._nav_by(-1)
        elif ev.key == 'home':
            self._nav_to(0)
        elif ev.key == 'end':
            self._nav_to(len(self.days)-1)
        elif ev.key == 'left':
            self._pan(-1)
        elif ev.key == 'right':
            self._pan(1)
        elif ev.key == 'up':
            self._zoom(0.5)
        elif ev.key == 'down':
            self._zoom(2.0)
        elif k == 'r':
            self._nav_fit()
        elif k == 'p':
            self._screenshot()
        elif k == 'q':
            plt.close(self.fig)

    def _on_close(self, ev):
        self._persist()

    def _on_click(self, ev):
        if ev.inaxes != self.ax:
            return
        # Pass to plugin
        self.plugin.on_click(ev)

    def _screenshot(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        day = self.days[self.day_idx]
        path = EXAMPLES_DIR / f'viz_{day}_{ts}.png'
        self.fig.savefig(path, dpi=130, bbox_inches='tight')
        print(f'[Engine] Screenshot saved -> {path}')

    def _ensure_data(self, new_lo, new_hi):
        if self.dt is None or len(self.dt) == 0:
            return
            
        current_lo = mdates.date2num(self.dt.iloc[0])
        current_hi = mdates.date2num(self.dt.iloc[-1])
        changed = False
        
        # Load past days if zooming left
        if new_lo < current_lo:
            days_needed = int(np.ceil(current_lo - new_lo))
            min_idx = min(self.loaded_day_indices)
            for i in range(1, days_needed + 2): # +2 for buffer
                if min_idx - i >= 0:
                    self.loaded_day_indices.add(min_idx - i)
                    changed = True
                    
        # Load future days if zooming right
        if new_hi > current_hi:
            days_needed = int(np.ceil(new_hi - current_hi))
            max_idx = max(self.loaded_day_indices)
            for i in range(1, days_needed + 2):
                if max_idx + i < len(self.days):
                    self.loaded_day_indices.add(max_idx + i)
                    changed = True
                    
        if changed:
            self._reload_data()

    def _pan(self, direction):
        xlo, xhi = self.ax.get_xlim()
        window = xhi - xlo
        shift = window * 0.5 * direction
        new_l, new_r = xlo + shift, xhi + shift
        
        self._ensure_data(new_l, new_r)
        
        self.ax.set_xlim(new_l, new_r)
        self._autofit_y()
        self.draw(fit=False)

    def _zoom(self, factor):
        xlo, xhi = self.ax.get_xlim()
        center = (xlo + xhi) / 2
        half = (xhi - xlo) / 2 * factor
        new_l, new_r = center - half, center + half
        
        self._ensure_data(new_l, new_r)
        
        self.ax.set_xlim(new_l, new_r)
        self._autofit_y()
        self.draw(fit=False)

    def _fit_all(self):
        if self.dt is None or len(self.dt) == 0:
            return
        self.ax.set_xlim(self.dt.iloc[0], self.dt.iloc[-1])
        lo, hi = float(self.lows.min()), float(self.highs.max())
        pad = (hi - lo) * 0.05
        self.ax.set_ylim(lo - pad, hi + pad)

    def _autofit_y(self):
        if self.dt is None or len(self.dt) == 0:
            return
        xlo, xhi = self.ax.get_xlim()
        xn = mdates.date2num(self.dt.values)
        m = (xn >= xlo) & (xn <= xhi)
        if not m.any():
            return
        lo, hi = float(self.lows[m].min()), float(self.highs[m].max())
        pad = (hi - lo) * 0.05 or 1.0
        self.ax.set_ylim(lo - pad, hi + pad)

    def _apply_window_geometry(self):
        geom = self._settings.get('window_geometry')
        if not geom:
            return
        try:
            mgr = self.fig.canvas.manager
            if hasattr(mgr, 'window') and hasattr(mgr.window, 'geometry'):
                def _apply_and_refresh():
                    mgr.window.geometry(geom)
                    mgr.window.update_idletasks()
                    
                    # Simulates the manual resize workaround to snap layout into place
                    try:
                        import re
                        m = re.match(r'(\d+)x(\d+)(.*)', geom)
                        if m:
                            w, h, pos = int(m.group(1)), int(m.group(2)), m.group(3)
                            # Resize by 20 pixels to force Tk layout cascade
                            mgr.window.geometry(f"{w+20}x{h+20}{pos}")
                            mgr.window.update_idletasks()
                            
                            # Revert back to exact saved size after a tiny delay
                            # so the OS window manager actually processes the change
                            def _snap_back():
                                mgr.window.geometry(f"{w}x{h}{pos}")
                                mgr.window.update_idletasks()
                                self.fig.canvas.draw_idle()
                                
                            if hasattr(mgr.window, 'after'):
                                mgr.window.after(100, _snap_back)
                            else:
                                _snap_back()
                    except Exception:
                        pass
                    
                if hasattr(mgr.window, 'after'):
                    mgr.window.after(100, _apply_and_refresh)
                else:
                    _apply_and_refresh()
        except Exception:
            pass

    def _persist(self):
        out = {
            'day': self.days[self.day_idx],
            'tf': self.tf
        }
        
        # Save exact zoom and pan positions
        try:
            out['xlim'] = self.ax.get_xlim()
            out['ylim'] = self.ax.get_ylim()
        except Exception:
            pass
            
        try:
            mgr = self.fig.canvas.manager
            if hasattr(mgr, 'window') and hasattr(mgr.window, 'geometry'):
                out['window_geometry'] = str(mgr.window.geometry())
        except Exception as e:
            print(f"[Engine] Warning: Could not get window geometry: {e}")
        save_settings(out)

    def run(self):
        plt.show(block=True)
