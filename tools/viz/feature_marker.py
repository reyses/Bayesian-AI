#!/usr/bin/env python
"""Feature Marker - Mark points on a price chart and toggle V2 features as overlays.

Adapted from tools/trade_marker.py. Same UX (click to mark, crosshair, pan/zoom)
but instead of marking trade legs you drop pins and toggle V2 features ON/OFF
as overlays so you can SEE which features react at each pin.

Two overlay surfaces:
  - PRICE PANEL (top, always shown): price-space overlays
        regression means at multiple TFs, +/-2sigma bands, M_high/M_low
  - INDICATOR PANEL (bottom, hides when nothing active): dimensionless features
        z_se, reversion_prob, hurst, swing_noise, vol_velocity, vol_accel ...

Click to drop a pin. Each pin records timestamp + close price + a snapshot of
EVERY active feature value at that bar so you can compare across pins later.

Keyboard
--------
  PRICE-PANEL OVERLAYS (toggle ON/OFF)
    1   1m  M_close (regression mean)
    2   5m  M_close
    3   15m M_close
    4   1h  M_close
    5   4h  M_close
    6   5m  M_high  (3-body upper anchor)
    7   5m  M_low   (3-body lower anchor)
    b   5m  +/-2sigma_close bands
    B   15m +/-2sigma_close bands
    n   1h  +/-2sigma_close bands

  INDICATOR PANEL (one feature at a time; press again to clear)
    q   L3_1m_z_se
    w   L3_5m_z_se
    e   L3_15m_z_se
    r   L3_1m_reversion_prob
    t   L3_5m_hurst
    y   L3_5m_swing_noise
    u   L2_5m_vol_velocity   (LEADING pre-pivot signal)
    i   L2_5m_vol_accel
    o   L2_5m_vol_mean
    p   L2_1m_vol_sigma

  MARKING
    LEFT CLICK   drop a pin at the nearest bar
    shift+click  delete the nearest pin
    d            delete all pins
    g            toggle pin labels (timestamp + price)

  NAVIGATION
    LEFT/RIGHT   pan
    UP/DOWN      zoom in / out
    f            fit all
    s            save pins JSON
    Q            save and quit

Usage
-----
    python tools/feature_marker.py --day 2026_02_12
    python tools/feature_marker.py --day 2026_03_03
    python tools/feature_marker.py --day 2026_02_12 --start-hour 13 --end-hour 18
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

plt.switch_backend('TkAgg')

TICK_SIZE = 0.25
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
    'M1':  '#E53935',
    'M5':  '#FB8C00',
    'M15': '#43A047',
    'M1h': '#1E88E5',
    'M4h': '#8E24AA',
    'MH5': '#26A69A',
    'ML5': '#EF5350',
    'B5':  '#FB8C00',
    'B15': '#43A047',
    'B1h': '#1E88E5',
}

INDICATOR_COLORS = {
    'z_se_1m': '#E53935',
    'z_se_5m': '#FB8C00',
    'z_se_15m': '#43A047',
    'rprob_1m': '#8E24AA',
    'hurst_5m': '#5E35B1',
    'sn_5m':   '#00897B',
    'volvel_5m': '#FB8C00',
    'volacc_5m': '#D81B60',
    'volmean_5m': '#3949AB',
    'volsigma_1m': '#6D4C41',
}


def _load_5s_ohlcv(day: str) -> pd.DataFrame:
    path = f'DATA/ATLAS/5s/{day}.parquet'
    if not os.path.exists(path):
        raise FileNotFoundError(f'5s OHLCV missing: {path}')
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _load_tf_ohlcv(tf: str, day: str) -> pd.DataFrame:
    base, _ = TF_CONFIG[tf]
    path = os.path.join(base, f'{day}.parquet')
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _load_v2_layer(layer: str, tf: str, day: str) -> pd.DataFrame:
    """Load a feature layer like L3_1m for one day. Returns empty on miss."""
    path = f'DATA/ATLAS/FEATURES_5s_v2/{layer}_{tf}/{day}.parquet'
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df = df.copy()
        df['timestamp'] = (df['timestamp'].astype('int64') // 10**9)
    return df.sort_values('timestamp').reset_index(drop=True)


def _ffill_to_5s(values: np.ndarray, src_ts: np.ndarray, target_ts: np.ndarray,
                 period_s: int) -> np.ndarray:
    """Forward-fill TF-cadence series onto 5s grid (no lookahead)."""
    target = target_ts - period_s
    idx = np.searchsorted(src_ts, target, side='right') - 1
    idx = np.clip(idx, 0, len(src_ts) - 1)
    return values[idx]


class FeatureMarker:

    def __init__(self, day: str, start_hour: float = 0.0, end_hour: float = 24.0):
        self.day = day
        self.start_hour = start_hour
        self.end_hour = end_hour

        print(f'Loading 5s OHLCV for {day}...')
        self.ohlcv = _load_5s_ohlcv(day)

        # Crop by hour window
        day_dt = datetime.strptime(day.replace('_', '-'), '%Y-%m-%d').replace(
            tzinfo=timezone.utc)
        t_start = day_dt.timestamp() + start_hour * 3600
        t_end = day_dt.timestamp() + end_hour * 3600
        m = (self.ohlcv['timestamp'] >= t_start) & (self.ohlcv['timestamp'] < t_end)
        self.ohlcv = self.ohlcv[m].reset_index(drop=True)
        if self.ohlcv.empty:
            raise RuntimeError(f'No 5s data in window {start_hour}-{end_hour}h for {day}')

        self.ts_5s = self.ohlcv['timestamp'].values.astype(np.int64)
        self.dt_5s = [datetime.fromtimestamp(int(t), tz=timezone.utc) for t in self.ts_5s]
        self.close = self.ohlcv['close'].values.astype(float)
        self.high  = self.ohlcv['high'].values.astype(float)
        self.low   = self.ohlcv['low'].values.astype(float)

        # Build price-space overlays from TF OHLCV
        self.price_overlays = {}
        self._build_price_overlays()

        # Load indicator features lazily; cache on first toggle
        self._indicator_cache = {}

        # Active overlays
        self.active_price = set()         # which price-overlays are visible
        self.active_indicator = None      # which indicator is in lower panel
        self._price_artists = {}          # name -> [Line2D, ...]
        self._indicator_artist = None
        self._indicator_band_artist = None  # for filled bands

        # Markers
        self.pins = []                    # list of dicts
        self._pin_artists = []            # [(line, label), ...]
        self.show_labels = True

        # Fig handles set in run()
        self.fig = None
        self.ax = None
        self.ax_ind = None
        self.cursor = None

    def _build_price_overlays(self):
        """Compute regression mean + sigma at each TF on the 5s grid."""
        for tf in ('1m', '5m', '15m', '1h', '4h'):
            tf_oh = _load_tf_ohlcv(tf, self.day)
            if tf_oh.empty:
                continue
            _, N = TF_CONFIG[tf]
            tf_oh['close_mean']  = tf_oh['close'].rolling(N, min_periods=2).mean()
            tf_oh['close_sigma'] = tf_oh['close'].rolling(N, min_periods=2).std()
            tf_oh['high_mean']   = tf_oh['high'].rolling(N, min_periods=2).mean()
            tf_oh['low_mean']    = tf_oh['low'].rolling(N, min_periods=2).mean()
            tf_ts = tf_oh['timestamp'].values.astype(np.int64)
            period_s = PERIOD_S[tf]
            self.price_overlays[f'M_{tf}'] = _ffill_to_5s(
                tf_oh['close_mean'].values, tf_ts, self.ts_5s, period_s)
            self.price_overlays[f'S_{tf}'] = _ffill_to_5s(
                tf_oh['close_sigma'].values, tf_ts, self.ts_5s, period_s)
            self.price_overlays[f'MH_{tf}'] = _ffill_to_5s(
                tf_oh['high_mean'].values, tf_ts, self.ts_5s, period_s)
            self.price_overlays[f'ML_{tf}'] = _ffill_to_5s(
                tf_oh['low_mean'].values, tf_ts, self.ts_5s, period_s)

    def _get_indicator(self, code: str):
        """Return (label, values_5s, color, panel_kind) for a given code.
        panel_kind: 'line' (z, vel, accel, hurst etc) or 'prob' (rprob 0-1)."""
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
        df = _load_v2_layer(layer, tf, self.day)
        if df.empty or col not in df.columns:
            print(f'  [warn] {code} feature column {col} missing for {self.day}')
            self._indicator_cache[code] = None
            return None
        vals = df[col].values.astype(float)
        ts   = df['timestamp'].values.astype(np.int64)
        # All V2 layers are at 5s cadence already; align by ts
        idx = np.searchsorted(ts, self.ts_5s, side='right') - 1
        idx = np.clip(idx, 0, len(ts) - 1)
        aligned = vals[idx]
        result = (code, aligned, color, kind)
        self._indicator_cache[code] = result
        return result

    # ---- Overlay toggles -----------------------------------------------------

    def _toggle_price_overlay(self, name: str):
        if name in self.active_price:
            for art in self._price_artists.pop(name, []):
                try:
                    art.remove()
                except Exception:
                    pass
            self.active_price.discard(name)
            print(f'  [off] {name}')
        else:
            self._draw_price_overlay(name)
            self.active_price.add(name)
            print(f'  [on]  {name}')
        self.ax.legend(loc='upper left', fontsize=8)
        self.fig.canvas.draw_idle()

    def _draw_price_overlay(self, name: str):
        artists = []
        if name == 'M1':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['M_1m'],
                color=PRICE_OVERLAY_COLORS['M1'], lw=1.0, label='1m M_close')[0])
        elif name == 'M5':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['M_5m'],
                color=PRICE_OVERLAY_COLORS['M5'], lw=1.2, label='5m M_close')[0])
        elif name == 'M15':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['M_15m'],
                color=PRICE_OVERLAY_COLORS['M15'], lw=1.4, label='15m M_close')[0])
        elif name == 'M1h':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['M_1h'],
                color=PRICE_OVERLAY_COLORS['M1h'], lw=1.6, label='1h M_close')[0])
        elif name == 'M4h':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['M_4h'],
                color=PRICE_OVERLAY_COLORS['M4h'], lw=1.8, label='4h M_close')[0])
        elif name == 'MH5':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['MH_5m'],
                color=PRICE_OVERLAY_COLORS['MH5'], lw=1.2, linestyle='--',
                label='5m M_high')[0])
        elif name == 'ML5':
            artists.append(self.ax.plot(self.dt_5s, self.price_overlays['ML_5m'],
                color=PRICE_OVERLAY_COLORS['ML5'], lw=1.2, linestyle='--',
                label='5m M_low')[0])
        elif name in ('B5', 'B15', 'B1h'):
            tf = {'B5': '5m', 'B15': '15m', 'B1h': '1h'}[name]
            M = self.price_overlays[f'M_{tf}']
            S = self.price_overlays[f'S_{tf}']
            color = PRICE_OVERLAY_COLORS[name]
            up = M + 2 * S
            dn = M - 2 * S
            band = self.ax.fill_between(self.dt_5s, dn, up, color=color, alpha=0.10,
                                        label=f'{tf} +/-2sigma_close')
            artists.append(band)
        self._price_artists[name] = artists

    def _set_indicator(self, code: str):
        """Show one indicator in the lower panel; pressing the same code clears."""
        # Clear current
        if self._indicator_artist is not None:
            try:
                self._indicator_artist.remove()
            except Exception:
                pass
            self._indicator_artist = None
        if self._indicator_band_artist is not None:
            try:
                self._indicator_band_artist.remove()
            except Exception:
                pass
            self._indicator_band_artist = None

        if self.active_indicator == code:
            self.active_indicator = None
            self.ax_ind.set_ylabel('')
            self.ax_ind.set_visible(False)
            print(f'  [off] indicator: {code}')
            self.fig.canvas.draw_idle()
            return

        self.ax_ind.set_visible(True)
        result = self._get_indicator(code)
        if result is None:
            print(f'  [skip] {code} unavailable')
            return
        label, vals, color, kind = result
        line, = self.ax_ind.plot(self.dt_5s, vals, color=color, lw=0.9, label=label)
        self._indicator_artist = line
        self.ax_ind.set_ylabel(label, fontsize=10)
        self.ax_ind.relim()
        self.ax_ind.autoscale_view(scaley=True)
        if kind == 'prob':
            self.ax_ind.set_ylim(0, 1.05)
        self.ax_ind.legend(loc='upper right', fontsize=8)
        self.active_indicator = code
        print(f'  [on]  indicator: {code}')
        self.fig.canvas.draw_idle()

    # ---- Pins ----------------------------------------------------------------

    def _drop_pin(self, idx: int):
        """Drop a pin at bar idx; capture snapshot of all active features."""
        snap = {}
        for name in self.active_price:
            key_map = {'M1': 'M_1m', 'M5': 'M_5m', 'M15': 'M_15m',
                       'M1h': 'M_1h', 'M4h': 'M_4h', 'MH5': 'MH_5m', 'ML5': 'ML_5m'}
            if name in key_map:
                snap[name] = float(self.price_overlays[key_map[name]][idx])
            elif name in ('B5', 'B15', 'B1h'):
                tf = {'B5': '5m', 'B15': '15m', 'B1h': '1h'}[name]
                M = self.price_overlays[f'M_{tf}'][idx]
                S = self.price_overlays[f'S_{tf}'][idx]
                snap[name] = {'M': float(M), 'sigma': float(S),
                              '+2s': float(M + 2 * S), '-2s': float(M - 2 * S)}
        if self.active_indicator:
            r = self._get_indicator(self.active_indicator)
            if r is not None:
                _, vals, _, _ = r
                snap[self.active_indicator] = float(vals[idx])

        pin = {
            'pin_id': len(self.pins),
            'ts':    int(self.ts_5s[idx]),
            'iso':   self.dt_5s[idx].strftime('%Y-%m-%dT%H:%M:%SZ'),
            'price': float(self.close[idx]),
            'snapshot': snap,
        }
        self.pins.append(pin)

        line = self.ax.axvline(self.dt_5s[idx], color='black', lw=1.2,
                               linestyle='-', alpha=0.7)
        label = None
        if self.show_labels:
            label = self.ax.text(self.dt_5s[idx],
                                 self.ax.get_ylim()[1] * 0.999,
                                 f"P{pin['pin_id']+1}\n{self.dt_5s[idx].strftime('%H:%M:%S')}\n${self.close[idx]:.2f}",
                                 fontsize=8, ha='left', va='top',
                                 bbox=dict(boxstyle='round,pad=0.2',
                                           facecolor='yellow', alpha=0.7))
        self._pin_artists.append((line, label))
        print(f'  Pin {pin["pin_id"]+1}  '
              f'{pin["iso"]}  ${pin["price"]:.2f}  snap_keys={list(snap.keys())}')
        self.fig.canvas.draw_idle()

    def _delete_nearest_pin(self, idx: int):
        if not self.pins:
            return
        # Closest by bar index
        dists = [abs(int(np.searchsorted(self.ts_5s, p['ts'])) - idx)
                 for p in self.pins]
        j = int(np.argmin(dists))
        pin = self.pins.pop(j)
        line, label = self._pin_artists.pop(j)
        try:
            line.remove()
            if label is not None:
                label.remove()
        except Exception:
            pass
        # Renumber
        for i, p in enumerate(self.pins):
            p['pin_id'] = i
        print(f'  Deleted pin at {pin["iso"]}')
        self.fig.canvas.draw_idle()

    def _clear_all_pins(self):
        for line, label in self._pin_artists:
            try:
                line.remove()
                if label is not None:
                    label.remove()
            except Exception:
                pass
        self._pin_artists.clear()
        self.pins.clear()
        print('  Cleared all pins')
        self.fig.canvas.draw_idle()

    def _toggle_labels(self):
        self.show_labels = not self.show_labels
        for (line, label), pin in zip(self._pin_artists, self.pins):
            if label is not None:
                label.set_visible(self.show_labels)
        print(f'  Labels: {"ON" if self.show_labels else "OFF"}')
        self.fig.canvas.draw_idle()

    # ---- Save ----------------------------------------------------------------

    def _save(self):
        if not self.pins:
            print('  No pins to save')
            return
        os.makedirs(MARKERS_DIR, exist_ok=True)
        ts_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(MARKERS_DIR, f'pins_{self.day}_{ts_tag}.json')
        with open(out_path, 'w') as f:
            json.dump({
                'day': self.day,
                'created': ts_tag,
                'n_pins': len(self.pins),
                'pins': self.pins,
            }, f, indent=2)
        print(f'  Saved {len(self.pins)} pins -> {out_path}')

    # ---- Navigation ----------------------------------------------------------

    def _autofit_y(self):
        xlim = self.ax.get_xlim()
        bar_nums = mdates.date2num(self.dt_5s)
        mask = (bar_nums >= xlim[0]) & (bar_nums <= xlim[1])
        if mask.any():
            lo = self.low[mask].min()
            hi = self.high[mask].max()
            pad = (hi - lo) * 0.05
            self.ax.set_ylim(lo - pad, hi + pad)

    def _pan(self, direction: int):
        xlim = self.ax.get_xlim()
        w = xlim[1] - xlim[0]
        shift = w * 0.5 * direction
        x_min = mdates.date2num(self.dt_5s[0])
        x_max = mdates.date2num(self.dt_5s[-1])
        new_l = max(x_min, xlim[0] + shift)
        new_r = min(x_max, xlim[1] + shift)
        if new_r - new_l < w * 0.5:
            return
        self.ax.set_xlim(new_l, new_r)
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _zoom(self, factor: float):
        xlim = self.ax.get_xlim()
        c = (xlim[0] + xlim[1]) / 2
        half = (xlim[1] - xlim[0]) / 2 * factor
        x_min = mdates.date2num(self.dt_5s[0])
        x_max = mdates.date2num(self.dt_5s[-1])
        self.ax.set_xlim(max(x_min, c - half), min(x_max, c + half))
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _fit_all(self):
        self.ax.set_xlim(mdates.date2num(self.dt_5s[0]),
                         mdates.date2num(self.dt_5s[-1]))
        self._autofit_y()
        self.fig.canvas.draw_idle()

    # ---- Event handlers ------------------------------------------------------

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None:
            return
        bar_nums = mdates.date2num(self.dt_5s)
        idx = int(np.argmin(np.abs(bar_nums - event.xdata)))
        # Shift+click = delete nearest
        if event.key is not None and 'shift' in event.key:
            self._delete_nearest_pin(idx)
        else:
            self._drop_pin(idx)

    def _on_key(self, event):
        k = event.key
        # Price overlays
        mp = {'1': 'M1', '2': 'M5', '3': 'M15', '4': 'M1h', '5': 'M4h',
              '6': 'MH5', '7': 'ML5', 'b': 'B5', 'B': 'B15', 'n': 'B1h'}
        # Indicators
        mi = {'q': 'z_se_1m', 'w': 'z_se_5m', 'e': 'z_se_15m',
              'r': 'rprob_1m', 't': 'hurst_5m', 'y': 'sn_5m',
              'u': 'volvel_5m', 'i': 'volacc_5m',
              'o': 'volmean_5m', 'p': 'volsigma_1m'}
        if k in mp:
            self._toggle_price_overlay(mp[k])
        elif k in mi:
            self._set_indicator(mi[k])
        elif k in ('d', 'D'):
            self._clear_all_pins()
        elif k in ('g', 'G'):
            self._toggle_labels()
        elif k in ('s', 'S'):
            self._save()
        elif k == 'left':
            self._pan(-1)
        elif k == 'right':
            self._pan(1)
        elif k == 'up':
            self._zoom(0.5)
        elif k == 'down':
            self._zoom(2.0)
        elif k in ('f', 'F'):
            self._fit_all()
        elif k == 'Q':
            self._save()
            plt.close(self.fig)

    # ---- Run -----------------------------------------------------------------

    def run(self):
        from matplotlib.widgets import Cursor
        self.fig, (self.ax, self.ax_ind) = plt.subplots(
            2, 1, figsize=(22, 11), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]})
        self.fig.subplots_adjust(hspace=0.05)

        # Price plot
        self.ax.plot(self.dt_5s, self.close, color='black', lw=0.6, alpha=0.85,
                     label='5s close')
        self.ax.fill_between(self.dt_5s, self.low, self.high,
                             color='black', alpha=0.05)
        self.ax.set_facecolor('#FAFAFA')
        self.ax.set_ylabel('price', fontsize=11)
        self.ax.grid(True, alpha=0.25)
        self.ax.legend(loc='upper left', fontsize=8)

        # Indicator plot (hidden until activated)
        self.ax_ind.set_facecolor('#FAFAFA')
        self.ax_ind.grid(True, alpha=0.25)
        self.ax_ind.axhline(0, color='gray', lw=0.5, alpha=0.5)
        self.ax_ind.set_xlabel('time (UTC)')
        self.ax_ind.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        self.ax_ind.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        self.ax_ind.set_visible(False)

        self._fit_all()

        title = (f'{self.day}  feature marker  '
                 f'[{self.start_hour:.1f}-{self.end_hour:.1f}h]\n'
                 f'1-7 price means/HL  b/B/n bands  q-p indicators  '
                 f'click=pin shift+click=del  d=clear g=labels s=save Q=save+quit')
        self.ax.set_title(title, fontsize=10)

        self.cursor = Cursor(self.ax, useblit=True, color='gray',
                             linewidth=0.5, linestyle='--')

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        print(f'\n  Feature marker ready -- {len(self.ohlcv)} 5s bars loaded')
        print(f'  Price overlays:  1=1m  2=5m  3=15m  4=1h  5=4h  '
              f'6=M_high(5m)  7=M_low(5m)  b=5m bands  B=15m bands  n=1h bands')
        print(f'  Indicators:      q=z_se_1m  w=z_se_5m  e=z_se_15m  r=rprob_1m  '
              f't=hurst_5m  y=sn_5m  u=volvel_5m  i=volacc_5m  o=volmean_5m  '
              f'p=volsigma_1m')
        print(f'  Click=drop pin  shift+click=delete  d=clear all  g=toggle labels')
        print(f'  Arrows=pan/zoom  f=fit all  s=save pins  Q=save+quit\n')

        plt.show(block=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True, help='YYYY_MM_DD')
    ap.add_argument('--start-hour', type=float, default=0.0)
    ap.add_argument('--end-hour', type=float, default=24.0)
    args = ap.parse_args()

    fm = FeatureMarker(day=args.day,
                       start_hour=args.start_hour,
                       end_hour=args.end_hour)
    fm.run()


if __name__ == '__main__':
    main()
