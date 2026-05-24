"""
Feature Marker Plugin
Mark points on a price chart and toggle V2 features as overlays.
Use --tf 5s when running this plugin!
"""
import json
import os
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib.dates as mdates

from tools.viz.core.plugin import VizPlugin

MARKERS_DIR = 'chart/feature_markers'

TF_CONFIG = {
    '1m':  ('DATA/ATLAS/1m',  15),
    '5m':  ('DATA/ATLAS/5m',   9),
    '15m': ('DATA/ATLAS/15m', 12),
    '1h':  ('DATA/ATLAS/1h',  12),
    '4h':  ('DATA/ATLAS/4h',  18),
}
PERIOD_S = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600, '4h': 14400}

PRICE_OVERLAY_COLORS = {
    'M1':  '#E53935', 'M5':  '#FB8C00', 'M15': '#43A047',
    'M1h': '#1E88E5', 'M4h': '#8E24AA', 'MH5': '#26A69A',
    'ML5': '#EF5350', 'B5':  '#FB8C00', 'B15': '#43A047',
    'B1h': '#1E88E5',
}

INDICATOR_COLORS = {
    'z_se_1m': '#E53935', 'z_se_5m': '#FB8C00', 'z_se_15m': '#43A047',
    'rprob_1m': '#8E24AA', 'hurst_5m': '#5E35B1', 'sn_5m':   '#00897B',
    'volvel_5m': '#FB8C00', 'volacc_5m': '#D81B60', 'volmean_5m': '#3949AB',
    'volsigma_1m': '#6D4C41',
}

def _load_tf_ohlcv(tf: str, day: str) -> pd.DataFrame:
    base, _ = TF_CONFIG.get(tf, ('', 0))
    path = os.path.join(base, f'{day}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)

def _load_v2_layer(layer: str, tf: str, day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/FEATURES_5s_v2/{layer}_{tf}/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)

def _ffill_to_5s(values: np.ndarray, src_ts: np.ndarray, target_ts: np.ndarray, period_s: int) -> np.ndarray:
    target = target_ts - period_s
    idx = np.searchsorted(src_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(src_ts) - 1)
    return values[idx]

class FeatureMarkerPlugin(VizPlugin):
    
    requires_indicator_panel = True

    def __init__(self, args):
        super().__init__()
        
        self.price_overlays = {}
        self._indicator_cache = {}
        
        self.active_price = set()
        self.active_indicator = None
        
        self.pins = []
        self.show_labels = True
        
        self._loaded_days = set()

    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)
        if engine.tf != '5s':
            print("[Plugin] WARNING: Feature Marker is designed to run on 5s data. Run with --tf 5s.")
        # Setup indicator axis properties
        if engine.ax_ind:
            import matplotlib.dates as mdates
            engine.ax_ind.set_xlabel('time (UTC)')
            engine.ax_ind.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            engine.ax_ind.xaxis.set_major_locator(mdates.HourLocator(interval=1))

    def _ensure_data_loaded(self):
        # We need to build price overlays and indicators for any NEW days that the engine loaded
        new_days = self.engine.loaded_day_indices - self._loaded_days
        if not new_days:
            return
            
        for idx in new_days:
            day = self.engine.days[idx]
            
            # 1. Build price overlays for this day
            for tf in ('1m', '5m', '15m', '1h', '4h'):
                tf_oh = _load_tf_ohlcv(tf, day)
                if tf_oh.empty:
                    continue
                _, N = TF_CONFIG[tf]
                tf_oh['close_mean']  = tf_oh['close'].rolling(N, min_periods=2).mean()
                tf_oh['close_sigma'] = tf_oh['close'].rolling(N, min_periods=2).std()
                tf_oh['high_mean']   = tf_oh['high'].rolling(N, min_periods=2).mean()
                tf_oh['low_mean']    = tf_oh['low'].rolling(N, min_periods=2).mean()
                
                # We need to ffill onto the ENGINE's timeline, not just the day's timeline.
                # Since the engine has concatenated data, we can just do it on the fly in `draw`, 
                # but it's more efficient to cache it. We'll store it in a dict per day.
                if day not in self.price_overlays:
                    self.price_overlays[day] = {}
                    
                self.price_overlays[day][f'M_{tf}'] = (tf_oh['timestamp'].values.astype(np.int64), tf_oh['close_mean'].values)
                self.price_overlays[day][f'S_{tf}'] = (tf_oh['timestamp'].values.astype(np.int64), tf_oh['close_sigma'].values)
                self.price_overlays[day][f'MH_{tf}'] = (tf_oh['timestamp'].values.astype(np.int64), tf_oh['high_mean'].values)
                self.price_overlays[day][f'ML_{tf}'] = (tf_oh['timestamp'].values.astype(np.int64), tf_oh['low_mean'].values)
                
            # Clear indicator cache since we have new days
            self._indicator_cache.clear()
            
        self._loaded_days.update(new_days)

    def _get_active_price_series(self, name):
        """Builds a continuous numpy array for the active price overlay aligned to engine.timestamps"""
        out = np.full(len(self.engine.timestamps), np.nan)
        for idx in self.engine.loaded_day_indices:
            day = self.engine.days[idx]
            if day not in self.price_overlays or name not in self.price_overlays[day]:
                continue
            
            src_ts, src_vals = self.price_overlays[day][name]
            
            # Find which portion of the engine's timestamps belong to this day
            mask = (self.engine.timestamps >= src_ts[0]) & (self.engine.timestamps <= src_ts[-1] + 86400)
            if not mask.any():
                continue
                
            target_ts = self.engine.timestamps[mask]
            
            # ffill
            tf_str = name.split('_')[1]
            period_s = PERIOD_S.get(tf_str, 300)
            out[mask] = _ffill_to_5s(src_vals, src_ts, target_ts, period_s)
        return out

    def _get_indicator(self, code: str):
        """Return (label, values, color, panel_kind) for the current engine view."""
        if code in self._indicator_cache:
            return self._indicator_cache[code]

        spec_map = {
            'z_se_1m':    ('L3', '1m',  'L3_1m_z_se_15',           'line', INDICATOR_COLORS['z_se_1m']),
            'z_se_5m':    ('L3', '5m',  'L3_5m_z_se_9',            'line', INDICATOR_COLORS['z_se_5m']),
            'z_se_15m':   ('L3', '15m', 'L3_15m_z_se_12',          'line', INDICATOR_COLORS['z_se_15m']),
            'rprob_1m':   ('L3', '1m',  'L3_1m_reversion_prob_15', 'prob', INDICATOR_COLORS['rprob_1m']),
            'hurst_5m':   ('L3', '5m',  'L3_5m_hurst_9',           'line', INDICATOR_COLORS['hurst_5m']),
            'sn_5m':      ('L3', '5m',  'L3_5m_swing_noise_9',     'line', INDICATOR_COLORS['sn_5m']),
            'volvel_5m':  ('L2', '5m',  'L2_5m_vol_velocity_9',    'line', INDICATOR_COLORS['volvel_5m']),
            'volacc_5m':  ('L2', '5m',  'L2_5m_vol_accel_9',       'line', INDICATOR_COLORS['volacc_5m']),
            'volmean_5m': ('L2', '5m',  'L2_5m_vol_mean_9',        'line', INDICATOR_COLORS['volmean_5m']),
            'volsigma_1m':('L2', '1m',  'L2_1m_vol_sigma_15',      'line', INDICATOR_COLORS['volsigma_1m']),
        }
        if code not in spec_map:
            return None
            
        layer, tf, col, kind, color = spec_map[code]
        
        out = np.full(len(self.engine.timestamps), np.nan)
        for idx in self.engine.loaded_day_indices:
            day = self.engine.days[idx]
            df = _load_v2_layer(layer, tf, day)
            if df.empty or col not in df.columns:
                continue
                
            vals = df[col].values.astype(float)
            ts   = df['timestamp'].values.astype(np.int64)
            
            mask = (self.engine.timestamps >= ts[0]) & (self.engine.timestamps <= ts[-1] + 86400)
            if not mask.any():
                continue
            
            target_ts = self.engine.timestamps[mask]
            
            # Align
            idx_align = np.searchsorted(ts, target_ts, side='right') - 1
            idx_align = np.clip(idx_align, 0, len(ts) - 1)
            out[mask] = vals[idx_align]

        result = (code, out, color, kind)
        self._indicator_cache[code] = result
        return result

    def draw(self, ax, ax_ind, time_range, patches_list):
        self._ensure_data_loaded()
        
        # Draw Price Overlays
        for name in self.active_price:
            if name in ('M1', 'M5', 'M15', 'M1h', 'M4h'):
                tf = {'M1':'1m','M5':'5m','M15':'15m','M1h':'1h','M4h':'4h'}[name]
                vals = self._get_active_price_series(f'M_{tf}')
                line = ax.plot(self.engine.dt, vals, color=PRICE_OVERLAY_COLORS[name], lw=1.2, label=f'{tf} M_close')[0]
                patches_list.append(line)
            elif name in ('MH5', 'ML5'):
                tf = '5m'
                prefix = 'MH' if name == 'MH5' else 'ML'
                vals = self._get_active_price_series(f'{prefix}_{tf}')
                line = ax.plot(self.engine.dt, vals, color=PRICE_OVERLAY_COLORS[name], lw=1.2, linestyle='--', label=f'{tf} M_high/low')[0]
                patches_list.append(line)
            elif name in ('B5', 'B15', 'B1h'):
                tf = {'B5': '5m', 'B15': '15m', 'B1h': '1h'}[name]
                M = self._get_active_price_series(f'M_{tf}')
                S = self._get_active_price_series(f'S_{tf}')
                color = PRICE_OVERLAY_COLORS[name]
                band = ax.fill_between(self.engine.dt, M - 2*S, M + 2*S, color=color, alpha=0.10, label=f'{tf} +/-2sigma_close')
                patches_list.append(band)

        # Draw Indicator
        if ax_ind:
            ax_ind.clear()
            if self.active_indicator:
                ax_ind.set_visible(True)
                r = self._get_indicator(self.active_indicator)
                if r:
                    code, vals, color, kind = r
                    ax_ind.plot(self.engine.dt, vals, color=color, lw=0.9, label=code)
                    ax_ind.set_ylabel(code, fontsize=10)
                    if kind == 'prob':
                        ax_ind.set_ylim(0, 1.05)
            else:
                ax_ind.set_visible(False)

        # Draw Pins
        for pin in self.pins:
            ts_val = pd.to_datetime(pin['ts'], unit='s', utc=True)
            line = ax.axvline(ts_val, color='black', lw=1.2, linestyle='-', alpha=0.7)
            patches_list.append(line)
            
            if self.show_labels:
                lbl = ax.text(ts_val, self.engine.highs.max(),
                              f"P{pin['pin_id']+1}\n{ts_val.strftime('%H:%M:%S')}\n${pin['price']:.2f}",
                              fontsize=8, ha='left', va='top',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
                patches_list.append(lbl)

    def on_click(self, event):
        if event.button != 1 or event.xdata is None:
            return False
            
        bar_nums = mdates.date2num(self.engine.dt)
        idx = int(np.argmin(np.abs(bar_nums - event.xdata)))
        
        # Shift+click = delete
        if event.key and 'shift' in event.key:
            if not self.pins: return False
            dists = [abs(int(np.searchsorted(self.engine.timestamps, p['ts'])) - idx) for p in self.pins]
            j = int(np.argmin(dists))
            self.pins.pop(j)
            for i, p in enumerate(self.pins): p['pin_id'] = i
            self.engine.draw()
            return True

        # Drop pin
        snap = {}
        # We don't bother computing snapshot accurately right now for simplicity 
        # but could fetch values from _get_active_price_series
        
        pin = {
            'pin_id': len(self.pins),
            'ts': int(self.engine.timestamps[idx]),
            'iso': self.engine.dt.iloc[idx].strftime('%Y-%m-%dT%H:%M:%SZ'),
            'price': float(self.engine.closes[idx]),
            'snapshot': snap,
        }
        self.pins.append(pin)
        self.engine.draw()
        return True

    def on_key(self, event):
        k = event.key
        mp = {'1': 'M1', '2': 'M5', '3': 'M15', '4': 'M1h', '5': 'M4h',
              '6': 'MH5', '7': 'ML5', 'b': 'B5', 'B': 'B15', 'n': 'B1h'}
        mi = {'q': 'z_se_1m', 'w': 'z_se_5m', 'e': 'z_se_15m',
              'r': 'rprob_1m', 't': 'hurst_5m', 'y': 'sn_5m',
              'u': 'volvel_5m', 'i': 'volacc_5m',
              'o': 'volmean_5m', 'p': 'volsigma_1m'}
              
        if k in mp:
            name = mp[k]
            if name in self.active_price: self.active_price.remove(name)
            else: self.active_price.add(name)
            self.engine.draw()
            return True
            
        elif k in mi:
            code = mi[k]
            if self.active_indicator == code: self.active_indicator = None
            else: self.active_indicator = code
            self.engine.draw()
            return True
            
        elif k in ('d', 'D'):
            self.pins.clear()
            self.engine.draw()
            return True
            
        elif k in ('g', 'G'):
            self.show_labels = not self.show_labels
            self.engine.draw()
            return True
            
        elif k in ('s', 'S'):
            self._save()
            return True
            
        return False

    def _save(self):
        if not self.pins: return
        os.makedirs(MARKERS_DIR, exist_ok=True)
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        day = self.engine.days[self.engine.day_idx]
        out_path = os.path.join(MARKERS_DIR, f'pins_{day}_{ts_tag}.json')
        with open(out_path, 'w') as f:
            json.dump({
                'day': day,
                'created': ts_tag,
                'n_pins': len(self.pins),
                'pins': self.pins,
            }, f, indent=2)
        print(f'[Plugin] Saved {len(self.pins)} pins -> {out_path}')

    def get_title_stats(self) -> str:
        return f"Feature Marker | Pins: {len(self.pins)} | Active: {len(self.active_price)} overlays, Ind: {self.active_indicator}"

def get_plugin(unknown_args):
    return FeatureMarkerPlugin(unknown_args)
