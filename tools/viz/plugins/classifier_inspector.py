"""
Classifier Inspector Plugin
Visual debugger for the entry-timing + direction classifier pipeline.
Ported to VizEngine.
"""
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path

from tools.viz.core.plugin import VizPlugin
from tools.viz.auto_swing_marker import detect_swings, TICK_SIZE

TIMING_CACHE = Path('reports/findings/regret_oracle/zz_timing_cache_OOS_NT8_atr4_gbm.parquet')
DIR_CACHE = Path('reports/findings/regret_oracle/direction_proba_cache_OOS_NT8.parquet')
TREND3_CACHE = Path('reports/findings/regret_oracle/trend3_cache_OOS_NT8.parquet')
TREND3_SMOOTHED_CACHE = Path('reports/findings/regret_oracle/trend3_smoothed_OOS_NT8.parquet')
B1_PROBA_CACHE = Path('reports/findings/regret_oracle/b1_proba_OOS_NT8.parquet')
B2_PROBA_CACHE = Path('reports/findings/regret_oracle/b2_proba_OOS_NT8.parquet')
CLOUD_CACHE = Path('reports/findings/regret_oracle/pivot_probability_cloud.parquet')
B6_PROBA_CACHE = Path('reports/findings/regret_oracle/b6_proba_OOS_NT8.parquet')

TRAIN_ATR_MULT = 4.0
PICKS_DIR = Path('DATA/cusp_picks')

def compute_atr(high, low, close, period=14):
    if len(high) < period + 1:
        return float((high - low).mean()) if len(high) > 0 else 1.0
    prev_c = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())

def _leg_dir_truth_at_bars(pivots, piv_ts_unix, piv_closes, bar_ts_arr):
    truth = np.full(len(bar_ts_arr), '', dtype=object)
    if len(pivots) < 2: return truth
    piv_dir = np.array(['LONG' if piv_closes[k+1] > piv_closes[k] else 'SHORT' for k in range(len(pivots) - 1)])
    idx = np.searchsorted(piv_ts_unix[pivots], bar_ts_arr, side='right') - 1
    for i, k in enumerate(idx):
        if 0 <= k < len(piv_dir): truth[i] = piv_dir[k]
    return truth

class ClassifierInspectorPlugin(VizPlugin):

    def __init__(self, args):
        super().__init__()
        
        self.zigzag_atr_mult = 4.0
        self.t_timing = 0.50
        self.t_dir = 0.65
        
        self.zigzag_min_bars = 3
        self.signal_source = 'raw'
        self.color_mode = 'direction'
        
        self.b1_overlay = True
        self.b1_K = 10
        self.b2_overlay = True
        self.b2_K = 10
        self.zone_overlay = True
        self.b6_overlay = True
        self.b6_K = 10
        
        self._cache_data = {}
        
    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)

    def _ensure_data(self, day):
        if day in self._cache_data: return
        
        data = {}
        # 5s data for zigzag
        path_5s = f'DATA/ATLAS/5s/{day}.parquet'
        if os.path.exists(path_5s):
            bars5s = pd.read_parquet(path_5s).sort_values('timestamp').reset_index(drop=True)
            data['bars5s'] = bars5s
            
        def try_load(path, day):
            if path.exists():
                df = pd.read_parquet(path)
                day_df = df[df['day'] == day].copy()
                if not day_df.empty: return day_df
            return None
            
        data['trend3'] = try_load(TREND3_CACHE, day)
        data['trend3_sm'] = try_load(TREND3_SMOOTHED_CACHE, day)
        data['b1_proba'] = try_load(B1_PROBA_CACHE, day)
        data['b2_proba'] = try_load(B2_PROBA_CACHE, day)
        data['b6_proba'] = try_load(B6_PROBA_CACHE, day)
        data['cloud'] = try_load(CLOUD_CACHE, day)
        
        # User Picks
        date_key = day.replace('_', '-')
        p = PICKS_DIR / f'picks_{date_key}_multi.json'
        picks = []
        if p.exists():
            import json
            with open(p) as f:
                d = json.load(f)
                picks = d.get('picks', [])
        data['user_picks'] = picks
        
        self._cache_data[day] = data

    def get_sliders(self) -> list:
        return [
            {'label': 'zigzag ATR mult', 'min': 0.5, 'max': 30.0, 'step': 0.5, 'valinit': self.zigzag_atr_mult, 'action': self.set_atr},
            {'label': 'T_timing', 'min': 0.0, 'max': 1.0, 'step': 0.01, 'valinit': self.t_timing, 'action': self.set_timing},
            {'label': 'T_dir', 'min': 0.5, 'max': 0.99, 'step': 0.01, 'valinit': self.t_dir, 'action': self.set_dir},
        ]
        
    def set_atr(self, val):
        self.zigzag_atr_mult = float(val)
        self.engine.draw()
        
    def set_timing(self, val):
        self.t_timing = float(val)
        self.engine.draw()
        
    def set_dir(self, val):
        self.t_dir = float(val)
        self.engine.draw()

    def on_key(self, event):
        k = event.key
        if k == 's':
            self.signal_source = 'smoothed' if self.signal_source == 'raw' else 'raw'
            self.engine.draw()
            return True
        elif k == 'c':
            self.color_mode = 'correctness' if self.color_mode == 'direction' else 'direction'
            self.engine.draw()
            return True
        elif k == 'b':
            self.b1_overlay = not self.b1_overlay
            self.engine.draw()
            return True
        elif k == 'f':
            self.b2_overlay = not self.b2_overlay
            self.engine.draw()
            return True
        elif k == 'z':
            self.zone_overlay = not self.zone_overlay
            self.engine.draw()
            return True
        elif k == 'd':
            self.b6_overlay = not self.b6_overlay
            self.engine.draw()
            return True
        return False

    def draw(self, ax, time_range, patches_list):
        day = self.engine.days[self.engine.day_idx]
        self._ensure_data(day)
        d = self._cache_data[day]
        
        atr_pts = compute_atr(self.engine.highs, self.engine.lows, self.engine.closes, period=14)
        atr_ticks = atr_pts / TICK_SIZE
        min_rev_ticks = max(4, int(round(atr_ticks * self.zigzag_atr_mult)))
        
        # Zigzag logic
        bars_zz = d.get('bars5s')
        if bars_zz is not None:
            closes = bars_zz['close'].values.astype(np.float64)
            ts_unix = bars_zz['timestamp'].values
            ts_dt = pd.to_datetime(ts_unix, unit='s')
            zz_min_bars = self.zigzag_min_bars * 12
        else:
            closes = self.engine.closes
            ts_unix = self.engine.timestamps
            ts_dt = self.engine.dt
            zz_min_bars = self.zigzag_min_bars
            
        pivots = detect_swings(closes, min_reversal=min_rev_ticks, min_bars=zz_min_bars, max_bars=0)
        if len(pivots) >= 2 and pivots[-1] != len(closes) - 1:
            pivots = list(pivots) + [len(closes) - 1]
            
        swings = []
        for k in range(len(pivots) - 1):
            i0 = pivots[k]; i1 = pivots[k+1]
            p0 = closes[i0]; p1 = closes[i1]
            ts0 = ts_dt[i0]; ts1 = ts_dt[i1]
            color = 'green' if p1 > p0 else 'red'
            
            # Entry / Exit
            m1 = ax.scatter([ts0], [p0], marker='x', s=45, color=color, linewidths=2.0, zorder=6)
            m2 = ax.scatter([ts1], [p1], marker='x', s=45, color=color, linewidths=2.0, alpha=0.6, zorder=6)
            line, = ax.plot([ts0, ts1], [p0, p1], color=color, linewidth=1.2, alpha=0.55, zorder=5)
            patches_list.extend([m1, m2, line])
            
            pnl_usd = abs(p1 - p0) * 2.0
            swings.append((ts_unix[i0], ts_unix[i1], 'LONG' if p1 > p0 else 'SHORT', pnl_usd))
            txt = ax.annotate(f'${pnl_usd:+.0f}', xy=(ts1, p1), xytext=(5, 0), textcoords='offset points', 
                              fontsize=7, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor=color, edgecolor='none', alpha=0.7), zorder=7)
            patches_list.append(txt)

        # Draw b1 ribbon
        b1_proba = d.get('b1_proba')
        if self.b1_overlay and b1_proba is not None:
            col = f'p_pivot_{self.b1_K}m'
            if col in b1_proba.columns:
                xs, ys, colors, alphas = [], [], [], []
                y_rib = self.engine.highs.max() + atr_pts * 0.1
                ts_arr = pd.to_datetime(b1_proba['timestamp'], unit='s')
                for ts_val, p in zip(ts_arr, b1_proba[col].values):
                    p = float(p)
                    if p < 0.30: continue
                    c = '#ffeb3b' if p < 0.50 else ('#ff9100' if p < 0.70 else ('#ff5252' if p < 0.85 else '#ff1744'))
                    xs.append(ts_val); ys.append(y_rib)
                    colors.append(c); alphas.append(min(1.0, 0.4 + p * 0.6))
                if xs:
                    sc = ax.scatter(xs, ys, marker='s', s=35, c=colors, alpha=alphas, zorder=11, edgecolors='none')
                    patches_list.append(sc)

        # Draw classifier dots
        trend_df = d.get('trend3_sm') if self.signal_source == 'smoothed' else d.get('trend3')
        if trend_df is not None:
            # We skip the complex correct/wrong evaluation for brevity since it's just visual markers
            # We just draw directional predictions > t_timing
            xs, ys, colors, alphas = [], [], [], []
            for _, r in trend_df.iterrows():
                p_l = float(r.get('p_long', 0)); p_s = float(r.get('p_short', 0)); p_n = float(r.get('p_neutral', 0))
                dir_conf = max(p_l, p_s)
                if dir_conf < self.t_timing: continue
                
                color = 'lime' if p_l > p_s else 'orangered'
                ts_val = pd.to_datetime(r['timestamp'], unit='s')
                y_val = self.engine.lows.min() - atr_pts * 0.2
                
                xs.append(ts_val); ys.append(y_val); colors.append(color); alphas.append(min(1.0, dir_conf - p_n + 0.5))
            if xs:
                sc = ax.scatter(xs, ys, marker='o', s=60, c=colors, alpha=alphas, zorder=10, edgecolors='white', linewidths=0.6)
                patches_list.append(sc)

def get_plugin(unknown_args):
    return ClassifierInspectorPlugin(unknown_args)
