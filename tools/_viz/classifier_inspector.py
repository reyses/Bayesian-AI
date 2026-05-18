#!/usr/bin/env python
"""Classifier inspector — visual debugger for the entry-timing + direction
classifier pipeline.

Shows:
  - 5s price candles for one day
  - GROUND TRUTH oracle golden moments (diamond markers, green=LONG, red=SHORT)
  - CLASSIFIER fire candidates at current threshold (circle markers, by direction)
  - Two interactive sliders: T_timing and T_dir
  - Stats panel: fires, precision (overlap with oracle), at current thresholds

Keyboard:
  PgUp/PgDn   — next/previous day
  HOME/END    — first/last day
  LEFT/RIGHT  — pan time axis
  UP/DOWN     — zoom in/out time axis
  r           — reset view (fit all)
  s           — toggle signal source: raw <-> smoothed
  c           — toggle dot color: direction <-> correctness (vs zigzag)
  b / B       — toggle B1 pivot-imminent ribbon / cycle K (1,3,5,10 min)
  f / F       — toggle B2 fakeout ring on pivots / cycle K (3,5,10 min)
  z           — toggle composite zone color bar (B1+B4+B5 cloud)
  d / D       — toggle B6 directional pivot triangles / cycle K (1,3,5,10 min)
  p           — screenshot to examples/
  q           — quit (persists settings)

Usage:
  python tools/classifier_inspector.py --day 2026_01_02
  python tools/classifier_inspector.py --target oos
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

# Allow `python tools/classifier_inspector.py` to find the `tools` package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
import matplotlib.pyplot as plt
# NOTE: do NOT call plt.switch_backend('TkAgg') here.
# `tools.auto_swing_marker` -> `tools.research` -> `tools.research.plots`
# runs `matplotlib.use('Agg')` at import time, overriding any setting here.
# The switch must happen AFTER all imports — see end-of-imports block below.
from matplotlib.widgets import Slider
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# Reuse the canonical ZigZag swing detector
# (this import chain runs `matplotlib.use('Agg')` via tools.research.plots —
# we'll switch back to TkAgg AFTER all imports finish)
from tools._viz.auto_swing_marker import detect_swings, TICK_SIZE

# NOW (after the Agg-setting import chain) force back to interactive.
plt.switch_backend('TkAgg')

# Default screenshot save location → project_root/examples/
EXAMPLES_DIR = str(Path(__file__).resolve().parent.parent.parent / 'examples')
os.makedirs(EXAMPLES_DIR, exist_ok=True)
matplotlib.rcParams['savefig.directory'] = EXAMPLES_DIR


RAW_5S_DIR = Path('DATA/ATLAS/5s')
RAW_1M_DIR = Path('DATA/ATLAS/1m')
RAW_5S_DIR_NT8 = Path('DATA/ATLAS_NT8/5s')
RAW_1M_DIR_NT8 = Path('DATA/ATLAS_NT8/1m')


def _bars_path(day: str, tf: str):
    """Return the right OHLC parquet path for this day, trying ATLAS_NT8
    first for NT8-era days, falling back to ATLAS."""
    name = f'{day}.parquet'
    nt8 = (RAW_1M_DIR_NT8 if tf == '1m' else RAW_5S_DIR_NT8) / name
    if nt8.exists():
        return nt8
    return (RAW_1M_DIR if tf == '1m' else RAW_5S_DIR) / name
TIMING_CACHE = Path('reports/findings/regret_oracle/zz_timing_cache_OOS_NT8_atr4_gbm.parquet')
DIR_CACHE = Path('reports/findings/regret_oracle/direction_proba_cache_OOS_NT8.parquet')
TREND3_CACHE = Path('reports/findings/regret_oracle/trend3_cache_OOS_NT8.parquet')
TREND3_SMOOTHED_CACHE = Path('reports/findings/regret_oracle/trend3_smoothed_OOS_NT8.parquet')
B1_PROBA_CACHE = Path('reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
B2_PROBA_CACHE = Path('reports/findings/regret_oracle/b2_proba_OOS_NT8.parquet')
B5_PROBA_CACHE = Path('reports/findings/regret_oracle/b5_leg_phase_OOS_NT8.parquet')
B6_PROBA_CACHE = Path('reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')
CLOUD_CACHE = Path('reports/findings/regret_oracle/pivot_probability_cloud.parquet')
ORACLE_CSV = Path('reports/findings/regret_oracle/daisy_chain_OOS_2026.csv')
PICKS_DIR = Path('DATA/cusp_picks')
SETTINGS_PATH = Path('DATA/cusp_picks/inspector_settings.json')


def load_settings() -> dict:
    """Persisted window geometry + slider values + day index."""
    if not SETTINGS_PATH.exists():
        return {}
    try:
        import json
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d: dict):
    try:
        import json
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SETTINGS_PATH, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        print(f'[warn] settings save failed: {e}')


def load_user_picks(day: str) -> list:
    """Load manual picks from cusp_marker for this day. Returns list of
    {timestamp, price, direction} dicts."""
    import json
    date_key = day.replace('_', '-')
    p = PICKS_DIR / f'picks_{date_key}_multi.json'
    if not p.exists():
        return []
    with open(p) as f:
        d = json.load(f)
    out = []
    for pk in d.get('picks', []):
        out.append({
            'ts': int(pk['timestamp']),
            'price': float(pk['price']),
            'direction': pk['direction'],
            'mfe_ticks': pk.get('mfe_ticks'),
            'snap': pk.get('snap', ''),
        })
    return out


def load_day_data(day: str, oracle: pd.DataFrame, timing_cache: pd.DataFrame,
                  dir_cache: pd.DataFrame):
    # 1m OHLC for candle rendering (auto-fallback ATLAS_NT8 → ATLAS)
    bars_path = _bars_path(day, '1m')
    if not bars_path.exists():
        return None
    bars = pd.read_parquet(bars_path).sort_values('timestamp').reset_index(drop=True)
    bars['ts_dt'] = pd.to_datetime(bars['timestamp'], unit='s')

    # 5s closes for zigzag oracle
    bars5s_path = _bars_path(day, '5s')
    bars5s = None
    if bars5s_path.exists():
        bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
        bars5s['ts_dt'] = pd.to_datetime(bars5s['timestamp'], unit='s')

    oracle_day = oracle[oracle['session_date_key'] == day].copy() if 'session_date_key' in oracle.columns else oracle.iloc[0:0]
    tc = timing_cache[timing_cache['day'] == day].copy()
    dc = dir_cache[dir_cache['day'] == day].copy()
    # Tolerate missing classifier caches (e.g., IS days when only OOS was cached)
    if len(tc) > 0:
        cache = tc.merge(dc[['timestamp','p_long']], on='timestamp', how='left')
        cache['ts_dt'] = pd.to_datetime(cache['timestamp'], unit='s')
    else:
        cache = pd.DataFrame(columns=['timestamp','day','p_timing','p_long','ts_dt'])

    # 3-class trend predictions (optional overlay)
    trend3 = None
    try:
        if TREND3_CACHE.exists():
            t3_all = pd.read_parquet(TREND3_CACHE)
            t3 = t3_all[t3_all['day'] == day].copy()
            if len(t3) > 0:
                t3['ts_dt'] = pd.to_datetime(t3['timestamp'], unit='s')
                trend3 = t3
    except Exception:
        trend3 = None

    # DMI-smoothed 3-class predictions (optional overlay for A/B vs raw)
    trend3_sm = None
    try:
        if TREND3_SMOOTHED_CACHE.exists():
            t3s_all = pd.read_parquet(TREND3_SMOOTHED_CACHE)
            t3s = t3s_all[t3s_all['day'] == day].copy()
            if len(t3s) > 0:
                t3s['ts_dt'] = pd.to_datetime(t3s['timestamp'], unit='s')
                trend3_sm = t3s
    except Exception:
        trend3_sm = None

    # B1 pivot-imminent probabilities (per 1m bar)
    b1_proba = None
    try:
        if B1_PROBA_CACHE.exists():
            b1_all = pd.read_parquet(B1_PROBA_CACHE)
            b1d = b1_all[b1_all['day'] == day].copy()
            if len(b1d) > 0:
                b1d['ts_dt'] = pd.to_datetime(b1d['timestamp'], unit='s')
                b1_proba = b1d
    except Exception:
        b1_proba = None

    # B2 fakeout probabilities (per pivot)
    b2_proba = None
    try:
        if B2_PROBA_CACHE.exists():
            b2_all = pd.read_parquet(B2_PROBA_CACHE)
            b2d = b2_all[b2_all['day'] == day].copy()
            if len(b2d) > 0:
                b2d['ts_dt'] = pd.to_datetime(b2d['timestamp'], unit='s')
                b2_proba = b2d
    except Exception:
        b2_proba = None

    # Cloud composite cache (zone + cloud_state per bar)
    cloud_data = None
    try:
        if CLOUD_CACHE.exists():
            cl_all = pd.read_parquet(CLOUD_CACHE)
            cl_d = cl_all[cl_all['day'] == day].copy()
            if len(cl_d) > 0:
                cl_d['ts_dt'] = pd.to_datetime(cl_d['timestamp'], unit='s')
                cloud_data = cl_d
    except Exception:
        cloud_data = None

    # B6 directional pivot probabilities (per bar)
    b6_proba = None
    try:
        if B6_PROBA_CACHE.exists():
            b6_all = pd.read_parquet(B6_PROBA_CACHE)
            b6d = b6_all[b6_all['day'] == day].copy()
            if len(b6d) > 0:
                b6d['ts_dt'] = pd.to_datetime(b6d['timestamp'], unit='s')
                b6_proba = b6d
    except Exception:
        b6_proba = None

    # User's manual picks (cusp_marker output) for this day
    user_picks = load_user_picks(day)

    return {
        'day': day,
        'bars': bars,         # 1m for candles
        'bars5s': bars5s,     # 5s for zigzag
        'oracle': oracle_day,
        'cache': cache,
        'user_picks': user_picks,
        'trend3': trend3,     # 3-class trend predictions (optional)
        'trend3_sm': trend3_sm,  # DMI-smoothed trend3 (optional)
        'b1_proba': b1_proba,   # B1 pivot-imminent (per-1m-bar P_K)
        'b2_proba': b2_proba,   # B2 fakeout (per-pivot P_K)
        'b6_proba': b6_proba,   # B6 directional pivot (per-1m-bar P_LONG / P_SHORT)
        'cloud_data': cloud_data,  # composite cloud: zone + cloud_state per bar
    }


def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Average True Range on 1m bars. Returns mean of last `period` TRs in PRICE units."""
    h = bars['high'].values; l = bars['low'].values; c = bars['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    # Day-typical ATR: median of all TRs (robust to opening spike) over the period window
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())


# Multiplier the trend3 classifier was TRAINED at. Correctness mode in the
# inspector always uses this value, regardless of the visual slider.
TRAIN_ATR_MULT = 4.0


def _compute_zigzag(bars5s_or_1m, atr_pts, atr_mult, zigzag_min_bars):
    """Return (pivots, ts_unix, closes, min_rev_ticks) for one zigzag pass."""
    closes = bars5s_or_1m['close'].values.astype(np.float64)
    ts_unix = bars5s_or_1m['timestamp'].values
    atr_ticks = atr_pts / TICK_SIZE
    min_rev_ticks = max(4, int(round(atr_ticks * atr_mult)))
    pivots = detect_swings(
        closes, min_reversal=min_rev_ticks,
        min_bars=zigzag_min_bars, max_bars=0,
    )
    if len(pivots) >= 2 and pivots[-1] != len(closes) - 1:
        pivots = list(pivots) + [len(closes) - 1]
    return pivots, ts_unix, closes, min_rev_ticks


def _leg_dir_truth_at_bars(pivots, piv_ts_unix, piv_closes, bar_ts_arr):
    """Given a pivot set + per-bar timestamps, return leg_direction per bar."""
    truth = np.full(len(bar_ts_arr), '', dtype=object)
    if len(pivots) < 2:
        return truth
    piv_dir = np.array([
        'LONG' if piv_closes[k+1] > piv_closes[k] else 'SHORT'
        for k in range(len(pivots) - 1)
    ])
    idx = np.searchsorted(piv_ts_unix[pivots], bar_ts_arr, side='right') - 1
    for i, k in enumerate(idx):
        if 0 <= k < len(piv_dir):
            truth[i] = piv_dir[k]
    return truth


def draw_candles(ax, bars: pd.DataFrame, body_color_up='lightgray',
                 body_color_down='dimgray', wick_color='black',
                 width_seconds=50):
    """Lightweight candlestick renderer. Wick = vline(low-high), body =
    rectangle(open-close), neutral gray colors so oracle/clf overlays stand out.
    """
    from matplotlib.patches import Rectangle
    w_days = width_seconds / 86400.0   # matplotlib date width
    half_w = w_days / 2
    ts = bars['ts_dt'].values
    o = bars['open'].values; c = bars['close'].values
    h = bars['high'].values; l = bars['low'].values
    up = c >= o
    # Wicks (vertical line low to high)
    ax.vlines(ts, l, h, color=wick_color, linewidth=0.5, alpha=0.7)
    # Body rectangles
    import matplotlib.dates as mdates
    ts_num = mdates.date2num(ts)
    for i in range(len(ts)):
        x = ts_num[i] - half_w
        bottom = min(o[i], c[i])
        height = abs(c[i] - o[i])
        if height < 0.01:
            height = 0.01
        color = body_color_up if up[i] else body_color_down
        rect = Rectangle((x, bottom), w_days, height,
                          facecolor=color, edgecolor=wick_color,
                          linewidth=0.3, alpha=0.7, zorder=2)
        ax.add_patch(rect)


class Inspector:
    def __init__(self, days, oracle, timing_cache, dir_cache):
        self.days = days
        self.day_idx = 0
        self.oracle = oracle
        self.timing_cache = timing_cache
        self.dir_cache = dir_cache

        # Load persisted settings (geometry + sliders + day)
        self._settings = load_settings()
        self.t_timing         = float(self._settings.get('t_timing', 0.50))
        self.t_dir            = float(self._settings.get('t_dir', 0.65))
        self.zigzag_atr_mult  = float(self._settings.get('zigzag_atr_mult', 4.0))
        self.zigzag_min_bars  = int(self._settings.get('zigzag_min_bars', 3))
        # Direction-signal viz toggles
        #   signal_source: 'raw' | 'smoothed'  (key 's' toggles)
        #   color_mode:    'direction' | 'correctness'  (key 'c' toggles)
        self.signal_source    = self._settings.get('signal_source', 'raw')
        self.color_mode       = self._settings.get('color_mode', 'direction')
        # B1/B2 overlay toggles
        #   b1_overlay: bool — show pivot-imminent ribbon above price (key 'b')
        #   b1_K:       int  — which K minute window (1,3,5,10) (key 'B' cycles)
        #   b2_overlay: bool — show fakeout color ring on pivots (key 'f')
        #   b2_K:       int  — which K minute window for fakeout (3,5,10) (key 'F' cycles)
        self.b1_overlay = bool(self._settings.get('b1_overlay', True))
        self.b1_K       = int(self._settings.get('b1_K', 10))
        self.b2_overlay = bool(self._settings.get('b2_overlay', True))
        self.b2_K       = int(self._settings.get('b2_K', 10))
        # Composite zone color bar (B1+B4+B5 cloud composite) — 'z' toggles
        self.zone_overlay = bool(self._settings.get('zone_overlay', True))
        # B6 directional pivot triangles — 'd' toggles
        self.b6_overlay = bool(self._settings.get('b6_overlay', True))
        self.b6_K       = int(self._settings.get('b6_K', 10))

        # Build figure — single price panel (bottom proba panel removed)
        self.fig, self.ax_price = plt.subplots(1, 1, figsize=(16, 9))
        self.ax_proba = self.ax_price   # alias for backward-compat
        plt.subplots_adjust(left=0.05, right=0.97, top=0.94, bottom=0.18)

        # Sliders — three stacked rows
        ax_s1 = plt.axes([0.10, 0.10, 0.80, 0.022])
        ax_s2 = plt.axes([0.10, 0.07, 0.80, 0.022])
        ax_s3 = plt.axes([0.10, 0.04, 0.80, 0.022])
        self.slider_oracle = Slider(ax_s1, 'zigzag ATR mult',
                                     0.5, 30.0, valinit=self.zigzag_atr_mult,
                                     valstep=0.5)
        self.slider_timing = Slider(ax_s2, 'T_timing', 0.0, 1.0,
                                     valinit=self.t_timing, valstep=0.01)
        self.slider_dir = Slider(ax_s3, 'T_dir', 0.50, 0.99,
                                  valinit=self.t_dir, valstep=0.01)
        self.slider_oracle.on_changed(self._on_thr)
        self.slider_timing.on_changed(self._on_thr)
        self.slider_dir.on_changed(self._on_thr)

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        # Persist on window close
        self.fig.canvas.mpl_connect('close_event', lambda _e: self._persist_settings())

        # Stats text
        self.stats_text = self.fig.text(0.05, 0.01, '', fontsize=8, family='monospace')

        self._load_and_draw()
        self._apply_window_geometry()

    def _load_and_draw(self):
        day = self.days[self.day_idx]
        data = load_day_data(day, self.oracle, self.timing_cache, self.dir_cache)
        if data is None:
            self.day_idx = min(self.day_idx + 1, len(self.days) - 1)
            return
        self.data = data
        # Day changed → fit-all (don't preserve previous day's zoom)
        self._draw(preserve_view=False)

    def _draw(self, preserve_view=True):
        d = self.data
        # Capture current zoom/pan before clearing so we can restore it after.
        # Sliders change overlays only — view should NOT reset.
        saved_xlim = self.ax_price.get_xlim()
        saved_ylim = self.ax_price.get_ylim()
        # Default matplotlib axis range is (0, 1) → treat as "no view set yet".
        view_was_set = not (abs(saved_xlim[0]) < 1e-6 and abs(saved_xlim[1] - 1.0) < 1e-6)

        self.ax_price.clear()

        bars = d['bars']
        # Candlestick 1m bars (neutral gray so oracle/clf markers pop)
        draw_candles(self.ax_price, bars)
        # Default view = full day
        self.ax_price.set_xlim(bars['ts_dt'].iloc[0], bars['ts_dt'].iloc[-1])
        ylo = bars['low'].min(); yhi = bars['high'].max()
        ypad = (yhi - ylo) * 0.02
        self.ax_price.set_ylim(ylo - ypad, yhi + ypad)

        # ATR-adaptive zigzag: compute day's ATR on 1m bars, scale by multiplier
        atr_pts = compute_atr(bars, period=14)
        atr_ticks = atr_pts / TICK_SIZE
        min_rev_ticks = max(4, int(round(atr_ticks * self.zigzag_atr_mult)))
        d['atr_pts'] = atr_pts
        d['min_rev_ticks'] = min_rev_ticks

        # Zigzag oracle — detect swing pivots on 5s closes (precise)
        bars_zz = d.get('bars5s') if d.get('bars5s') is not None else bars
        closes = bars_zz['close'].values.astype(np.float64)
        ts_arr = bars_zz['ts_dt'].values
        ts_unix = bars_zz['timestamp'].values
        # On 5s, scale min_bars proportionally (1m's 3 bars ≈ 5s's 36 bars)
        zz_min_bars = self.zigzag_min_bars * 12 if d.get('bars5s') is not None else self.zigzag_min_bars
        pivots = detect_swings(
            closes,
            min_reversal=min_rev_ticks,
            min_bars=zz_min_bars,
            max_bars=0,
        )
        # Append final bar as terminal pivot so the last leg is rendered
        if len(pivots) >= 2 and pivots[-1] != len(closes) - 1:
            pivots = list(pivots) + [len(closes) - 1]
        n_swings = max(len(pivots) - 1, 0)
        R_ticks = min_rev_ticks
        R_price = R_ticks * TICK_SIZE
        truth_label = (f'ATR={atr_pts:.2f}pt  ×{self.zigzag_atr_mult:.1f}  '
                       f'→ min_rev={R_ticks}t ({R_price:.1f}pt)  swings={n_swings}')

        # Build zigzag-extreme trail (orange stairstep) + R-trigger line (cyan)
        running_extreme = np.full_like(closes, np.nan)
        r_trigger = np.full_like(closes, np.nan)
        for k in range(len(pivots) - 1):
            i0 = pivots[k]; i1 = pivots[k+1]
            seg = closes[i0:i1+1]
            if closes[i1] > closes[i0]:
                # Uptrend leg — running max + R-trigger = ext - R below
                ext = np.maximum.accumulate(seg)
                running_extreme[i0:i1+1] = ext
                r_trigger[i0:i1+1] = ext - R_price
            else:
                ext = np.minimum.accumulate(seg)
                running_extreme[i0:i1+1] = ext
                r_trigger[i0:i1+1] = ext + R_price
        self.ax_price.plot(ts_arr, running_extreme, color='orange',
                            linewidth=1.0, alpha=0.9, label='zigzag extreme',
                            zorder=3)
        self.ax_price.plot(ts_arr, r_trigger, color='cyan',
                            linewidth=1.0, alpha=0.8, label='R-trigger',
                            zorder=3)

        # Render each swing as a trade: entry → exit with PnL $ label
        swings = []
        DOLLAR_PER_POINT = 2.0   # MNQ
        for k in range(len(pivots) - 1):
            i0 = pivots[k]; i1 = pivots[k+1]
            p0 = closes[i0]; p1 = closes[i1]
            ts0 = ts_arr[i0]; ts1 = ts_arr[i1]
            ts0_u = int(ts_unix[i0]); ts1_u = int(ts_unix[i1])
            direction = 'LONG' if p1 > p0 else 'SHORT'
            color = 'green' if direction == 'LONG' else 'red'
            pnl_usd = abs(p1 - p0) * DOLLAR_PER_POINT   # always positive (oracle)
            swings.append((ts0_u, ts1_u, direction, pnl_usd))

            # Entry X (solid) + Exit X (faded) + connecting line
            self.ax_price.scatter([ts0], [p0], marker='x', s=45,
                                   color=color, linewidths=2.0, zorder=6)
            self.ax_price.scatter([ts1], [p1], marker='x', s=45,
                                   color=color, linewidths=2.0, alpha=0.6, zorder=6)
            self.ax_price.plot([ts0, ts1], [p0, p1],
                                color=color, linewidth=1.2, alpha=0.55, zorder=5)
            # $ PnL badge at exit
            self.ax_price.annotate(
                f'${pnl_usd:+.0f}',
                xy=(ts1, p1), xytext=(5, 0),
                textcoords='offset points', fontsize=7, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color,
                          edgecolor='none', alpha=0.7),
                zorder=7,
            )
        d['swings'] = swings

        # ── B2 fakeout ring on confirmed pivots ───────────────────────────
        # For each zigzag swing's entry pivot, draw a colored ring whose
        # color depends on P(fakeout) — green = real leg, red = predicted fake.
        b2_proba = d.get('b2_proba')
        if self.b2_overlay and b2_proba is not None and len(b2_proba) > 0:
            col = f'p_fakeout_{self.b2_K}m'
            b2_ts = b2_proba['timestamp'].values
            b2_p  = b2_proba[col].values
            for (s_ts, _, s_dir, _) in swings:
                j = np.argmin(np.abs(b2_ts - s_ts))
                if abs(b2_ts[j] - s_ts) > 90:
                    continue   # no matching pivot record
                p_fake = float(b2_p[j])
                # Color: green->yellow->red as p_fake rises from 0 to 1
                if p_fake < 0.30:
                    ring = '#00e676'   # bright green (likely real)
                elif p_fake < 0.50:
                    ring = '#ffeb3b'   # yellow (uncertain)
                elif p_fake < 0.70:
                    ring = '#ff9100'   # orange (likely fake)
                else:
                    ring = '#ff1744'   # red (very likely fake)
                # Find entry-pivot price (same as scatter location above)
                # We need the actual y-position — easiest: lookup in 5s closes
                i_pivot = int(np.searchsorted(ts_unix, s_ts, side='left'))
                if i_pivot >= len(closes):
                    i_pivot = len(closes) - 1
                y_pivot = float(closes[i_pivot])
                ts_pivot = pd.to_datetime(s_ts, unit='s')
                self.ax_price.scatter([ts_pivot], [y_pivot], marker='o',
                                       s=350, facecolor='none', edgecolor=ring,
                                       linewidths=2.5, alpha=0.85, zorder=8)
                # Probability label
                self.ax_price.annotate(
                    f'{p_fake*100:.0f}%',
                    xy=(ts_pivot, y_pivot), xytext=(-22, 14),
                    textcoords='offset points', fontsize=6,
                    color='white',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=ring,
                              edgecolor='none', alpha=0.85),
                    zorder=9,
                )

        # User's manual cusp_marker picks (purple stars — ground truth)
        user_picks = d.get('user_picks', [])
        for pk in user_picks:
            ts_pk = pd.to_datetime(pk['ts'], unit='s')
            color = '#a020f0' if pk['direction'] == 'LONG' else '#ff00aa'
            self.ax_price.scatter([ts_pk], [pk['price']], marker='*', s=180,
                                   facecolor=color, edgecolor='white',
                                   linewidths=0.8, zorder=8, alpha=0.95)

        # 3-class trend predictions — paint EVERY bar with either:
        #   color_mode='direction'   : lime=LONG / orangered=SHORT predicted dir
        #   color_mode='correctness' : green=matches zigzag truth / red=wrong
        # signal_source='raw'      : argmax(p_long, p_short, p_neutral) cache
        # signal_source='smoothed' : regime_dir from DMI windowed-EMA cache
        # T_timing slider = confidence floor (raw uses p_dir; smoothed uses
        # p_long_ema/p_short_ema as the "strength" proxy).
        if self.signal_source == 'smoothed':
            trend3 = d.get('trend3_sm')
        else:
            trend3 = d.get('trend3')

        # Build per-1m-bar leg_direction TRUTH. ALWAYS uses ATR×TRAIN_ATR_MULT
        # (the multiplier trend3 was trained against), regardless of slider —
        # otherwise correctness coloring would score the model against a label
        # set it never saw at training time.
        bar_ts_arr = bars['timestamp'].values
        atr_mult_truth = TRAIN_ATR_MULT
        if abs(self.zigzag_atr_mult - TRAIN_ATR_MULT) < 1e-6:
            # Slider already at training mult — reuse the visual pivots
            piv_truth = pivots
            ts_unix_truth = ts_unix
            closes_truth = closes
        else:
            # Recompute pivots at train mult (separate cheap pass)
            piv_truth, ts_unix_truth, closes_truth, _ = _compute_zigzag(
                bars_zz, atr_pts, atr_mult_truth, zz_min_bars,
            )
        truth_per_bar = _leg_dir_truth_at_bars(
            piv_truth, ts_unix_truth, closes_truth, bar_ts_arr,
        )

        n_pred = 0; n_correct = 0; n_wrong = 0; n_below_floor = 0
        if trend3 is not None and len(trend3) > 0:
            T = self.t_timing   # confidence floor
            xs, ys, colors, alphas = [], [], [], []
            for _, r in trend3.iterrows():
                p_long  = float(r['p_long'])
                p_short = float(r['p_short'])
                p_neut  = float(r['p_neutral'])
                # Choose predicted direction. For smoothed, prefer regime_dir
                # if available; otherwise argmax (raw).
                if self.signal_source == 'smoothed' and 'regime_dir' in r:
                    rd = str(r['regime_dir'])
                    if rd == 'LONG':
                        pred = 'LONG';  dir_conf = float(r.get('p_long_ema', p_long))
                    elif rd == 'SHORT':
                        pred = 'SHORT'; dir_conf = float(r.get('p_short_ema', p_short))
                    else:
                        # NEUTRAL regime — skip the dot
                        continue
                else:
                    if p_long >= p_short:
                        pred = 'LONG';  dir_conf = p_long
                    else:
                        pred = 'SHORT'; dir_conf = p_short
                if dir_conf < T:
                    n_below_floor += 1
                    continue
                i = np.searchsorted(bar_ts_arr, r['timestamp'], side='right') - 1
                if i < 0 or i >= len(bars):
                    continue
                truth = truth_per_bar[i]
                # Decide color
                if self.color_mode == 'correctness':
                    if truth == '':
                        color = 'gold'   # no truth available
                    elif truth == pred:
                        color = '#00e676'   # bright green = correct
                        n_correct += 1
                    else:
                        color = '#ff1744'   # bright red = wrong
                        n_wrong += 1
                else:
                    color = 'lime' if pred == 'LONG' else 'orangered'
                    if truth == pred and truth != '':
                        n_correct += 1
                    elif truth != '' and truth != pred:
                        n_wrong += 1
                n_pred += 1
                xs.append(r['ts_dt'])
                lo = float(bars['low'].iloc[i])
                offset = max(0.25, atr_pts * 0.1)
                ys.append(lo - offset)
                colors.append(color)
                # Strength = directional probability above NEUTRAL mass
                # For smoothed we proxy with dir_conf - p_neut (raw p_neut)
                strength = dir_conf - p_neut
                alphas.append(max(0.25, min(1.0, strength + 0.5)))
            if xs:
                self.ax_price.scatter(xs, ys, marker='o', s=60,
                                        c=colors, alpha=alphas, zorder=10,
                                        edgecolors='white', linewidths=0.6)
            fires = trend3[
                (trend3['p_long'] > self.t_timing) |
                (trend3['p_short'] > self.t_timing)
            ] if 'p_long' in trend3.columns else trend3
        else:
            fires = pd.DataFrame()
        cache = d['cache']   # keep for bottom panel (kept for backward-compat)
        # (Old 2-class fire-rendering loop removed — trend3 block above draws them)

        # ── B1 pivot-imminent ribbon ───────────────────────────────────────
        # A row of colored markers ABOVE the price candles, one per 1m bar.
        # Color intensity = P(pivot_within_K_min).  Threshold colors at
        # 0.50 (yellow), 0.70 (orange), 0.85 (red — "fire warning").
        b1_proba = d.get('b1_proba')
        if self.b1_overlay and b1_proba is not None and len(b1_proba) > 0:
            col = f'p_pivot_{self.b1_K}m'
            yhi_cur = self.ax_price.get_ylim()[1]
            ylo_cur = self.ax_price.get_ylim()[0]
            ribbon_y = yhi_cur + (yhi_cur - ylo_cur) * 0.005   # just above
            # Extend ylim slightly to fit ribbon
            self.ax_price.set_ylim(ylo_cur, yhi_cur + (yhi_cur - ylo_cur) * 0.025)
            xs, ys, colors, alphas = [], [], [], []
            for ts_dt, p in zip(b1_proba['ts_dt'].values,
                                 b1_proba[col].values):
                p = float(p)
                if p < 0.30:
                    continue   # below visual threshold — keep ribbon clean
                if p < 0.50:
                    c = '#ffeb3b'   # yellow
                elif p < 0.70:
                    c = '#ff9100'   # orange
                elif p < 0.85:
                    c = '#ff5252'   # light red
                else:
                    c = '#ff1744'   # red — fire-warning
                xs.append(ts_dt); ys.append(ribbon_y)
                colors.append(c)
                alphas.append(min(1.0, 0.4 + p * 0.6))   # alpha grows with p
            if xs:
                self.ax_price.scatter(xs, ys, marker='s', s=35,
                                       c=colors, alpha=alphas, zorder=11,
                                       edgecolors='none')

        # ── Composite zone color bar ────────────────────────────────────
        # Paints a row of zone-colored squares ABOVE the B1 ribbon. Each
        # 1m bar gets a color from the pivot probability cloud zone:
        #   CLEAR     -> green     (no pivot proximity)
        #   WATCH     -> yellow
        #   NEAR_5m   -> light orange
        #   NEAR_3m   -> orange
        #   IMMINENT  -> red
        #   NEAR_PIVOT-> red-orange (B4 W=120 high)
        #   AT_PIVOT  -> bright red (B4 W=60 high)
        #   WIDE_ZONE -> tan       (B4 W=300 high)
        cloud_data = d.get('cloud_data')
        if self.zone_overlay and cloud_data is not None and len(cloud_data) > 0:
            ZONE_COLORS = {
                'CLEAR':       '#a0e0a0',
                'WATCH':       '#ffe0a0',
                'WIDE_ZONE':   '#deb887',
                'NEAR_5m':     '#ffb060',
                'NEAR_3m':     '#ff8050',
                'IMMINENT':    '#ff4030',
                'NEAR_PIVOT':  '#ff6040',
                'AT_PIVOT':    '#ff1010',
            }
            ylo_cur, yhi_cur = self.ax_price.get_ylim()
            zone_y = yhi_cur + (yhi_cur - ylo_cur) * 0.005
            self.ax_price.set_ylim(ylo_cur, yhi_cur + (yhi_cur - ylo_cur) * 0.020)
            xs, ys, colors = [], [], []
            for ts_dt, z in zip(cloud_data['ts_dt'].values,
                                  cloud_data['zone'].values):
                xs.append(ts_dt); ys.append(zone_y)
                colors.append(ZONE_COLORS.get(str(z), '#cccccc'))
            if xs:
                self.ax_price.scatter(xs, ys, marker='s', s=30,
                                       c=colors, alpha=0.85, zorder=12,
                                       edgecolors='none')

        # ── B6 directional pivot triangles ──────────────────────────────
        # ▲ green below low = "LONG pivot coming" (bottom expected)
        # ▼ red above high  = "SHORT pivot coming" (top expected)
        # Only fires above P >= 0.50 threshold.
        b6_proba = d.get('b6_proba')
        if self.b6_overlay and b6_proba is not None and len(b6_proba) > 0:
            K = self.b6_K
            p_long_col  = f'p_PIVOT_TO_LONG_{K}m'
            p_short_col = f'p_PIVOT_TO_SHORT_{K}m'
            if p_long_col in b6_proba.columns and p_short_col in b6_proba.columns:
                bar_ts_arr = bars['timestamp'].values
                for _, r in b6_proba.iterrows():
                    p_long  = float(r[p_long_col])
                    p_short = float(r[p_short_col])
                    if max(p_long, p_short) < 0.50:
                        continue
                    i = np.searchsorted(bar_ts_arr, r['timestamp'], side='right') - 1
                    if i < 0 or i >= len(bars):
                        continue
                    if p_long > p_short:
                        y = float(bars['low'].iloc[i]) - 0.5 * atr_pts * 0.1
                        alpha = min(1.0, 0.4 + p_long * 0.6)
                        self.ax_price.scatter([r['ts_dt']], [y],
                                               marker='^', s=70,
                                               color='#00d050',
                                               alpha=alpha, zorder=13,
                                               edgecolors='white', linewidths=0.5)
                    else:
                        y = float(bars['high'].iloc[i]) + 0.5 * atr_pts * 0.1
                        alpha = min(1.0, 0.4 + p_short * 0.6)
                        self.ax_price.scatter([r['ts_dt']], [y],
                                               marker='v', s=70,
                                               color='#ff4030',
                                               alpha=alpha, zorder=13,
                                               edgecolors='white', linewidths=0.5)

        # Title + axes — include day $ total + signal/color modes
        day_pnl = sum(pnl for *_, pnl in swings)
        n_evald = n_correct + n_wrong
        acc = (n_correct / n_evald) if n_evald > 0 else float('nan')
        # Warn if visual ATR mult differs from training-truth ATR mult
        atr_warn = ''
        if (self.color_mode == 'correctness'
                and abs(self.zigzag_atr_mult - TRAIN_ATR_MULT) > 1e-6):
            atr_warn = f'  [WARN viz_atr={self.zigzag_atr_mult:.1f} != train_atr={TRAIN_ATR_MULT:.0f}]'
        self.ax_price.set_title(
            f'{d["day"]}   day {self.day_idx+1}/{len(self.days)}   '
            f'{truth_label}   day $: ${day_pnl:+.0f}   '
            f'src={self.signal_source} color={self.color_mode}   '
            f'preds={n_pred}  acc={acc*100:.1f}% (vs ATR×{TRAIN_ATR_MULT:.0f} zigzag){atr_warn}',
            fontsize=9
        )
        self.ax_price.set_ylabel('price')
        self.ax_price.grid(True, alpha=0.3)
        # Legend with marker shapes
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], color='gray', linewidth=2, alpha=0.7, label='1m candles'),
            Line2D([0], [0], color='orange', linewidth=1.5, label='zigzag extreme'),
            Line2D([0], [0], color='cyan', linewidth=1.5, label='R-trigger'),
            Line2D([0], [0], marker='x', color='green', linewidth=1.0,
                   markersize=7, markeredgewidth=2, label='zz LONG entry-exit'),
            Line2D([0], [0], marker='x', color='red', linewidth=1.0,
                   markersize=7, markeredgewidth=2, label='zz SHORT entry-exit'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#a020f0',
                   markersize=10, linestyle='', label='your LONG pick'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='#ff00aa',
                   markersize=10, linestyle='', label='your SHORT pick'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lime',
                   markersize=6, linestyle='', label='clf LONG fire'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orangered',
                   markersize=6, linestyle='', label='clf SHORT fire'),
        ]
        self.ax_price.legend(handles=legend_handles, loc='upper left', fontsize=6,
                              ncol=2)

        # (Bottom probability time-series panel removed per user request —
        # only the price panel + sliders are shown now.)

        # Stats — precision = TP fires / total fires (TP = fire whose direction
        # matches a swing's entry pivot within ±120s).
        # recall = unique swings hit by at least one matching fire / total swings.
        n_oracle = len(swings)
        n_fires = len(fires)
        if n_fires > 0 and n_oracle > 0 and 'p_long' in fires.columns:
            tp_fires = 0
            hit_swings = set()
            for _, r in fires.iterrows():
                ft = int(r['timestamp'])
                pred_dir = 'LONG' if r['p_long'] >= 0.5 else 'SHORT'
                for s_idx, (s_entry_ts, _s_exit_ts, s_dir, _pnl) in enumerate(swings):
                    if abs(ft - s_entry_ts) <= 120 and s_dir == pred_dir:
                        tp_fires += 1
                        hit_swings.add(s_idx)
                        break
            prec   = tp_fires / n_fires
            recall = len(hit_swings) / n_oracle
            tp = tp_fires   # for display (compatibility)
        else:
            tp = 0; prec = 0.0; recall = 0.0
        b1_on = 'on' if self.b1_overlay else 'off'
        b2_on = 'on' if self.b2_overlay else 'off'
        stats = (f'ATR={d["atr_pts"]:.2f}pt x{self.zigzag_atr_mult:.1f} '
                 f'= {d["min_rev_ticks"]}t   '
                 f'T_timing={self.t_timing:.2f}   T_dir={self.t_dir:.2f}   |   '
                 f'src={self.signal_source} color={self.color_mode}   '
                 f'acc={(acc*100 if not np.isnan(acc) else 0):.1f}%   |   '
                 f'B1[{b1_on} K={self.b1_K}m]  B2[{b2_on} K={self.b2_K}m]   '
                 f'(s c b B f F)')
        self.stats_text.set_text(stats)

        # Restore zoom/pan after slider-triggered redraws
        if preserve_view and view_was_set:
            self.ax_price.set_xlim(saved_xlim)
            self.ax_price.set_ylim(saved_ylim)

        self.fig.canvas.draw_idle()

    def _on_thr(self, _val):
        self.t_timing = self.slider_timing.val
        self.t_dir = self.slider_dir.val
        self.zigzag_atr_mult = float(self.slider_oracle.val)
        self._draw()

    def _on_key(self, event):
        # Day navigation
        if event.key == 'pageup':
            self.day_idx = min(self.day_idx + 1, len(self.days) - 1)
            self._load_and_draw()
        elif event.key == 'pagedown':
            self.day_idx = max(self.day_idx - 1, 0)
            self._load_and_draw()
        elif event.key == 'home':
            self._fit_all()
        elif event.key == 'end':
            self.day_idx = len(self.days) - 1
            self._load_and_draw()
        # Pan / zoom (matplotlib viewport)
        elif event.key == 'left':
            self._pan(-1)
        elif event.key == 'right':
            self._pan(+1)
        elif event.key == 'up':
            self._zoom(0.5)
        elif event.key == 'down':
            self._zoom(2.0)
        elif event.key == 'r':
            self._fit_all()
        elif event.key in ('p', 'P'):
            self._screenshot()
        elif event.key in ('s', 'S'):
            self.signal_source = 'smoothed' if self.signal_source == 'raw' else 'raw'
            print(f'  [toggle] signal_source -> {self.signal_source}')
            self._draw()
        elif event.key in ('c', 'C'):
            self.color_mode = 'correctness' if self.color_mode == 'direction' else 'direction'
            print(f'  [toggle] color_mode -> {self.color_mode}')
            self._draw()
        elif event.key == 'b':
            self.b1_overlay = not self.b1_overlay
            print(f'  [toggle] B1 overlay -> {self.b1_overlay}')
            self._draw()
        elif event.key == 'B':
            ks = [1, 3, 5, 10]
            i = ks.index(self.b1_K) if self.b1_K in ks else 0
            self.b1_K = ks[(i + 1) % len(ks)]
            print(f'  [cycle] B1 K -> {self.b1_K} min')
            self._draw()
        elif event.key == 'f':
            self.b2_overlay = not self.b2_overlay
            print(f'  [toggle] B2 overlay -> {self.b2_overlay}')
            self._draw()
        elif event.key == 'F':
            ks = [3, 5, 10]
            i = ks.index(self.b2_K) if self.b2_K in ks else 2
            self.b2_K = ks[(i + 1) % len(ks)]
            print(f'  [cycle] B2 K -> {self.b2_K} min')
            self._draw()
        elif event.key == 'z':
            self.zone_overlay = not self.zone_overlay
            print(f'  [toggle] composite zone overlay -> {self.zone_overlay}')
            self._draw()
        elif event.key == 'd':
            self.b6_overlay = not self.b6_overlay
            print(f'  [toggle] B6 directional overlay -> {self.b6_overlay}')
            self._draw()
        elif event.key == 'D':
            ks = [1, 3, 5, 10]
            i = ks.index(self.b6_K) if self.b6_K in ks else 3
            self.b6_K = ks[(i + 1) % len(ks)]
            print(f'  [cycle] B6 K -> {self.b6_K} min')
            self._draw()
        elif event.key == 'q':
            self._persist_settings()
            plt.close(self.fig)

    def _screenshot(self):
        """Save the current figure straight to examples/ — no file dialog."""
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        day = self.data['day'] if hasattr(self, 'data') and self.data else 'unknown'
        path = os.path.join(EXAMPLES_DIR, f'inspector_{day}_{ts}.png')
        self.fig.savefig(path, dpi=130, bbox_inches='tight')
        print(f'  [screenshot] saved -> {path}')

    # ── Persistence ────────────────────────────────────────────────────
    def _capture_settings(self) -> dict:
        out = {
            't_timing': float(self.t_timing),
            't_dir': float(self.t_dir),
            'zigzag_atr_mult': float(self.zigzag_atr_mult),
            'zigzag_min_bars': int(self.zigzag_min_bars),
            'day_idx': int(self.day_idx),
            'signal_source': self.signal_source,
            'color_mode': self.color_mode,
            'b1_overlay': bool(self.b1_overlay),
            'b1_K': int(self.b1_K),
            'b2_overlay': bool(self.b2_overlay),
            'b2_K': int(self.b2_K),
            'zone_overlay': bool(self.zone_overlay),
            'b6_overlay': bool(self.b6_overlay),
            'b6_K': int(self.b6_K),
        }
        try:
            mgr = self.fig.canvas.manager
            if hasattr(mgr, 'window') and hasattr(mgr.window, 'geometry'):
                out['window_geometry'] = str(mgr.window.geometry())
        except Exception:
            pass
        return out

    def _persist_settings(self):
        save_settings(self._capture_settings())

    def _apply_window_geometry(self):
        geom = self._settings.get('window_geometry')
        if not geom:
            return
        try:
            mgr = self.fig.canvas.manager
            if hasattr(mgr, 'window') and hasattr(mgr.window, 'geometry'):
                mgr.window.geometry(geom)
        except Exception:
            pass

    # ── Pan / zoom (cusp_marker pattern) ──────────────────────────────
    def _pan(self, direction):
        d = self.data
        bars = d['bars']
        ts_dt = bars['ts_dt'].values
        xlim = self.ax_price.get_xlim()
        window = xlim[1] - xlim[0]
        shift = window * 0.5 * direction
        import matplotlib.dates as mdates
        x_min = mdates.date2num(pd.Timestamp(ts_dt[0]))
        x_max = mdates.date2num(pd.Timestamp(ts_dt[-1]))
        new_left = xlim[0] + shift
        new_right = xlim[1] + shift
        if new_left < x_min:
            new_left = x_min; new_right = x_min + window
        if new_right > x_max:
            new_right = x_max; new_left = max(x_min, x_max - window)
        self.ax_price.set_xlim(new_left, new_right)
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _zoom(self, factor):
        d = self.data
        bars = d['bars']
        ts_dt = bars['ts_dt'].values
        import matplotlib.dates as mdates
        xlim = self.ax_price.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        half_w = (xlim[1] - xlim[0]) / 2 * factor
        x_min = mdates.date2num(pd.Timestamp(ts_dt[0]))
        x_max = mdates.date2num(pd.Timestamp(ts_dt[-1]))
        new_left = max(x_min, center - half_w)
        new_right = min(x_max, center + half_w)
        self.ax_price.set_xlim(new_left, new_right)
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _fit_all(self):
        d = self.data
        bars = d['bars']
        self.ax_price.set_xlim(bars['ts_dt'].iloc[0], bars['ts_dt'].iloc[-1])
        ylo = bars['low'].min(); yhi = bars['high'].max()
        ypad = (yhi - ylo) * 0.02
        self.ax_price.set_ylim(ylo - ypad, yhi + ypad)
        self.fig.canvas.draw_idle()

    def _autofit_y(self):
        d = self.data
        bars = d['bars']
        import matplotlib.dates as mdates
        xlim = self.ax_price.get_xlim()
        bar_nums = mdates.date2num(bars['ts_dt'].values)
        mask = (bar_nums >= xlim[0]) & (bar_nums <= xlim[1])
        if not mask.any():
            return
        vis_low = float(bars['low'].values[mask].min())
        vis_high = float(bars['high'].values[mask].max())
        pad = (vis_high - vis_low) * 0.02
        self.ax_price.set_ylim(vis_low - pad, vis_high + pad)

    def run(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None, help='YYYY_MM_DD (default: first OOS)')
    ap.add_argument('--timing-cache', default=str(TIMING_CACHE))
    ap.add_argument('--dir-cache', default=str(DIR_CACHE))
    ap.add_argument('--oracle-csv', default=str(ORACLE_CSV),
                    help='Daisy oracle CSV (used for context only — primary '
                         'oracle is now zigzag-detected from 5s closes)')
    args = ap.parse_args()

    print(f'Loading caches...')
    tc = pd.read_parquet(args.timing_cache)
    dc = pd.read_parquet(args.dir_cache)
    oracle = pd.read_csv(args.oracle_csv)
    if 'session_date_key' not in oracle.columns:
        oracle['session_date_key'] = pd.to_datetime(oracle['session_date']).dt.strftime('%Y_%m_%d')
    print(f'  timing cache: {len(tc)} rows, {tc.day.nunique()} days')
    print(f'  dir cache:    {len(dc)} rows')
    print(f'  oracle:       {len(oracle)} bars, {oracle.session_date_key.nunique()} days')

    # Days = union of OOS cache days + any day with manual picks
    cache_days = set(tc['day'].unique())
    pick_days = set()
    for p in PICKS_DIR.glob('picks_*_multi.json'):
        # filename: picks_YYYY-MM-DD_multi.json
        stem = p.stem.replace('picks_', '').replace('_multi', '')
        if len(stem) == 10 and stem[4] == '-' and stem[7] == '-':
            pick_days.add(stem.replace('-', '_'))
    all_days = sorted(cache_days | pick_days)
    days = all_days
    print(f'  total days available: {len(days)} '
          f'(cache: {len(cache_days)}, pick-only: {len(pick_days - cache_days)})')

    if args.day:
        day_norm = args.day.replace('-', '_')
        if day_norm not in days:
            print(f'Day {day_norm} not available. Sample: {days[:5]}...')
            sys.exit(1)
        start_idx = days.index(day_norm)
    else:
        # Restore from settings if available; else pick-day; else 0
        sett = load_settings()
        if 'day_idx' in sett and 0 <= int(sett['day_idx']) < len(days):
            start_idx = int(sett['day_idx'])
        elif pick_days:
            start_idx = days.index(sorted(pick_days)[0])
        else:
            start_idx = 0

    insp = Inspector(days, oracle, tc, dc)
    insp.day_idx = start_idx
    insp._load_and_draw()
    insp.run()


if __name__ == '__main__':
    main()
