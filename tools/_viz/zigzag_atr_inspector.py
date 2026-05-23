#!/usr/bin/env python
"""Interactive ATR-slider zigzag inspector — drag the ATR multiplier and watch
the zigzag re-tile. The dynamic sweet-spot tuner.

One day's 5s price with the zigzag overlaid; a slider sets the ATR multiplier
(zigzag threshold = ATR(14) x mult). On every move the zigzag is recomputed
and the title updates R-trigger size, leg count, median swing, give-up tax.

No classifier caches — works on ANY day with 5s + 1m parquet (IS or OOS).

Usage:  python tools/_viz/zigzag_atr_inspector.py --day 2026_05_05
Keys:
  PgUp/PgDn   next / previous day
  HOME/END    first / last day
  LEFT/RIGHT  pan        UP/DOWN  zoom        r  fit view
  p           screenshot -> examples/
  q           quit (persists ATR mult, day, window geometry)
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.widgets import Slider

# This import chain runs matplotlib.use('Agg') via tools.research.plots —
# switch back to an interactive backend afterwards.
from tools._viz.auto_swing_marker import detect_swings, TICK_SIZE
for _bk in ('TkAgg', 'QtAgg', 'Qt5Agg', 'MacOSX'):
    try:
        plt.switch_backend(_bk)
        break
    except Exception:
        continue

REPO = Path(__file__).resolve().parent.parent.parent
EXAMPLES_DIR = REPO / 'examples'
EXAMPLES_DIR.mkdir(exist_ok=True)
matplotlib.rcParams['savefig.directory'] = str(EXAMPLES_DIR)
SETTINGS_PATH = REPO / 'DATA/cusp_picks/zigzag_atr_inspector_settings.json'
RAW_NT8 = REPO / 'DATA/ATLAS_NT8'
RAW_ATLAS = REPO / 'DATA/ATLAS'
ATR_PERIOD = 14
MIN_BARS = 36
TZ = 'America/New_York'
GREEN, RED = '#1a9850', '#d73027'


def bars_path(day: str, tf: str) -> Path:
    """5s/1m parquet — try ATLAS_NT8 (2026) then ATLAS (2025)."""
    nt8 = RAW_NT8 / tf / f'{day}.parquet'
    return nt8 if nt8.exists() else RAW_ATLAS / tf / f'{day}.parquet'


def list_days() -> list:
    """Every day with both a 5s and a 1m parquet (ATLAS + ATLAS_NT8), sorted."""
    days = set()
    for root in (RAW_ATLAS, RAW_NT8):
        for p in glob.glob(str(root / '5s' / '*.parquet')):
            days.add(Path(p).stem)
    return sorted(d for d in days if bars_path(d, '1m').exists())


def compute_atr(b1: pd.DataFrame) -> float:
    h, l, c = (b1[x].values.astype(float) for x in ('high', 'low', 'close'))
    if len(c) < 2:
        return 1.0
    prev = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev), np.abs(l - prev)])
    return (float(np.median(tr[-ATR_PERIOD * 3:])) if len(tr) >= ATR_PERIOD
            else float(tr.mean()))


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
        print(f'[warn] settings save failed: {e}')


class ZigzagATRInspector:
    def __init__(self, days, day_idx):
        self.days = days
        self.day_idx = day_idx
        self._settings = load_settings()
        self.atr_mult = float(self._settings.get('atr_mult', 4.0))
        self.dt = self.closes = self.atr_pts = None
        self.price_line = None
        self._zz = []

        self.fig, self.ax = plt.subplots(figsize=(16, 8))
        try:
            self.fig.canvas.manager.set_window_title('Zigzag ATR Inspector')
        except Exception:
            pass
        plt.subplots_adjust(bottom=0.16, top=0.93)
        self.ax.set_ylabel('price')
        self.ax.grid(alpha=0.25)

        ax_sl = plt.axes([0.13, 0.05, 0.74, 0.03])
        self.slider = Slider(ax_sl, 'zigzag ATR mult', 0.5, 10.0,
                             valinit=self.atr_mult, valstep=0.25)
        self.slider.on_changed(self._on_slider)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self._apply_window_geometry()
        self._load_day(fit=True)

    # ── data ───────────────────────────────────────────────────────────
    def _load_day(self, fit=False):
        day = self.days[self.day_idx]
        try:
            b5 = pd.read_parquet(bars_path(day, '5s')).sort_values(
                'timestamp').reset_index(drop=True)
            b1 = pd.read_parquet(bars_path(day, '1m')).sort_values(
                'timestamp').reset_index(drop=True)
        except Exception as e:
            print(f'[warn] could not load {day}: {e}')
            return
        self.closes = b5['close'].values.astype(float)
        self.dt = (pd.to_datetime(b5['timestamp'], unit='s', utc=True)
                   .dt.tz_convert(TZ).dt.tz_localize(None))
        self.atr_pts = compute_atr(b1)
        if self.price_line is not None:
            self.price_line.remove()
        self.price_line, = self.ax.plot(self.dt, self.closes, color='0.72',
                                        lw=0.7, zorder=1)
        self._draw(fit=fit)

    def _draw(self, fit=False):
        for ln in self._zz:
            ln.remove()
        self._zz.clear()
        min_rev = max(4, int(round(self.atr_pts / TICK_SIZE * self.atr_mult)))
        r_pt = min_rev * TICK_SIZE
        piv = detect_swings(self.closes, min_reversal=min_rev,
                            min_bars=MIN_BARS, max_bars=0)
        amps = [abs(self.closes[piv[k + 1]] - self.closes[piv[k]])
                for k in range(len(piv) - 1)]
        med = float(np.median(amps)) if amps else float('nan')
        giveup = (2 * r_pt / med * 100.0
                  if med and np.isfinite(med) and med > 0 else float('nan'))
        for k in range(len(piv) - 1):
            a, b = piv[k], piv[k + 1]
            c = GREEN if self.closes[b] > self.closes[a] else RED
            ln, = self.ax.plot([self.dt.iloc[a], self.dt.iloc[b]],
                               [self.closes[a], self.closes[b]],
                               color=c, lw=1.5, marker='o', ms=3, zorder=3)
            self._zz.append(ln)
        self.ax.set_title(
            f'{self.days[self.day_idx]}  [{self.day_idx + 1}/{len(self.days)}]'
            f'    ATR x{self.atr_mult:g}  (ATR(14)={self.atr_pts:.2f}pt)    '
            f'R-trigger {r_pt:.1f}pt    {len(piv) - 1} legs    '
            f'median swing {med:.1f}pt    give-up 2R/swing = {giveup:.0f}%',
            fontsize=10)
        if fit:
            self._fit_all()
        self.fig.canvas.draw_idle()

    # ── callbacks ──────────────────────────────────────────────────────
    def _on_slider(self, val):
        self.atr_mult = float(val)
        self._draw()

    def _on_key(self, ev):
        if ev.key == 'pageup':
            self.day_idx = min(self.day_idx + 1, len(self.days) - 1)
            self._load_day(fit=True)
        elif ev.key == 'pagedown':
            self.day_idx = max(self.day_idx - 1, 0)
            self._load_day(fit=True)
        elif ev.key == 'home':
            self.day_idx = 0
            self._load_day(fit=True)
        elif ev.key == 'end':
            self.day_idx = len(self.days) - 1
            self._load_day(fit=True)
        elif ev.key == 'left':
            self._pan(-1)
        elif ev.key == 'right':
            self._pan(+1)
        elif ev.key == 'up':
            self._zoom(0.5)
        elif ev.key == 'down':
            self._zoom(2.0)
        elif ev.key == 'r':
            self._fit_all()
            self.fig.canvas.draw_idle()
        elif ev.key in ('p', 'P'):
            self._screenshot()
        elif ev.key == 'q':
            self._persist()
            plt.close(self.fig)

    # ── screenshot ─────────────────────────────────────────────────────
    def _screenshot(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        day = self.days[self.day_idx]
        path = EXAMPLES_DIR / f'zigzag_atr_{day}_x{self.atr_mult:g}_{ts}.png'
        self.fig.savefig(path, dpi=130, bbox_inches='tight')
        print(f'  [screenshot] saved -> {path}')

    # ── pan / zoom ─────────────────────────────────────────────────────
    def _xnum_bounds(self):
        return (mdates.date2num(self.dt.iloc[0]),
                mdates.date2num(self.dt.iloc[-1]))

    def _pan(self, direction):
        xlo, xhi = self.ax.get_xlim()
        window = xhi - xlo
        shift = window * 0.5 * direction
        x_min, x_max = self._xnum_bounds()
        new_l, new_r = xlo + shift, xhi + shift
        if new_l < x_min:
            new_l, new_r = x_min, x_min + window
        if new_r > x_max:
            new_r, new_l = x_max, max(x_min, x_max - window)
        self.ax.set_xlim(new_l, new_r)
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _zoom(self, factor):
        xlo, xhi = self.ax.get_xlim()
        center = (xlo + xhi) / 2
        half = (xhi - xlo) / 2 * factor
        x_min, x_max = self._xnum_bounds()
        self.ax.set_xlim(max(x_min, center - half), min(x_max, center + half))
        self._autofit_y()
        self.fig.canvas.draw_idle()

    def _fit_all(self):
        self.ax.set_xlim(self.dt.iloc[0], self.dt.iloc[-1])
        lo, hi = float(self.closes.min()), float(self.closes.max())
        pad = (hi - lo) * 0.03
        self.ax.set_ylim(lo - pad, hi + pad)

    def _autofit_y(self):
        xlo, xhi = self.ax.get_xlim()
        xn = mdates.date2num(self.dt.values)
        m = (xn >= xlo) & (xn <= xhi)
        if not m.any():
            return
        lo, hi = float(self.closes[m].min()), float(self.closes[m].max())
        pad = (hi - lo) * 0.03 or 1.0
        self.ax.set_ylim(lo - pad, hi + pad)

    # ── settings persistence ───────────────────────────────────────────
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

    def _persist(self):
        out = {'atr_mult': float(self.atr_mult),
               'day': self.days[self.day_idx]}
        try:
            mgr = self.fig.canvas.manager
            if hasattr(mgr, 'window') and hasattr(mgr.window, 'geometry'):
                out['window_geometry'] = str(mgr.window.geometry())
        except Exception:
            pass
        save_settings(out)

    def run(self):
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', default=None,
                    help='YYYY_MM_DD (default: last session, else first)')
    args = ap.parse_args()
    days = list_days()
    if not days:
        raise SystemExit('No days found under DATA/ATLAS{,_NT8}/5s')
    target = args.day or load_settings().get('day')
    day_idx = days.index(target) if target in days else 0
    if args.day and args.day not in days:
        print(f'[warn] {args.day} not found; starting at {days[day_idx]}')
    ZigzagATRInspector(days, day_idx).run()


if __name__ == '__main__':
    main()
