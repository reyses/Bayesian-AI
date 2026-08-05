"""CHART REPLAY RECORDER — capture the human's turn/entry calls on a causal chart
(owner 2026-07-28). Reads the day BAR-BY-BAR with the future FOGGED. Full toolkit:
candlesticks + cubic (orange, 7.5min state) + sigma bands (±1/2σ envelope) +
current-bar horizontal σ levels + last-3-cusp S/R levels. TIMEFRAME TOGGLE (1m
structure ↔ 5s entry) — read structure on 1m, drop to 5s to time the entry.

Run:  python -m tools.viz.run --plugin chart_replay_recorder --day 2025_01_21 --who moises
Keys: f/F step +1/+10 · v back · z fit · -/= zoom · 1/5 timeframe(1m/5s) ·
      [ / ] shrink/grow σ sample (telescopes the horizontal σ levels through the
      level hierarchy — small=local, large=big shelf) · t/l/s/e click-type
      (TURN/LONG/SHORT/EXIT) · u undo · then CLICK the price to mark.
Log:  research/dojo_forge/reports/human_dojo/replay_<day>.<who>.jsonl
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

from tools.viz.core.plugin import VizPlugin

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOGDIR = os.path.join(REPO, 'research', 'dojo_forge', 'reports', 'human_dojo')
sys.path.insert(0, os.path.join(REPO, 'research', 'dojo_forge', 'tools'))
try:
    import cubic_regression as _cub
except Exception:
    _cub = None
try:
    import level_coordinate_system as _lcs   # TF-telescope level frame (density peaks)
except Exception:
    _lcs = None

TF_SEC = {'1m': 60, '5s': 5}
VIEW_BARS = {'1m': 45, '5s': 150}      # window shown (last ~45min / ~12.5min)
CUSP_R = {'1m': 25.0, '5s': 12.0}
MAJOR_R = {'1m': 70.0, '5s': 40.0}     # bigger-swing S/R (the "other levels" to check proximity)
AHEAD = {'1m': 5, '5s': 15}
PROX = 6.0                             # pts: "close proximity" to a major level
TYPES = {'t': ('TURN', '#455A64', 'D'), 'l': ('LONG', '#1565C0', '^'),
         's': ('SHORT', '#AD1457', 'v'), 'e': ('EXIT', '#C62828', 'x')}
WARMUP_MIN = 30


def _zigzag(x, R):
    n = len(x); hi = lo = x[0]; hii = loi = 0; d = 0; out = []
    for i in range(1, n):
        p = x[i]
        if not np.isfinite(p):
            continue
        if d >= 0 and p > hi: hi, hii = p, i
        if d <= 0 and p < lo: lo, loi = p, i
        if d >= 0 and hi - p >= R: out.append((hii, hi)); d = -1; lo, loi = p, i
        elif d <= 0 and p - lo >= R: out.append((loi, lo)); d = 1; hi, hii = p, i
    return out


def _roll_std(res, W=20):
    n = len(res); s = np.full(n, np.nan)
    for i in range(W - 1, n):
        w = res[i - W + 1:i + 1]; w = w[np.isfinite(w)]
        if len(w) >= 5:
            s[i] = w.std()
    return s


class ChartReplayRecorder(VizPlugin):
    def __init__(self, args):
        super().__init__()
        a = args or []
        self.who = a[a.index('--who') + 1] if '--who' in a and a.index('--who') + 1 < len(a) else 'owner'
        self.tf = '1m'
        self.mode = 't'
        self.zoom = 1.0          # view-width multiplier (-/= to zoom out/in)
        self.sw = {'1m': 20, '5s': 20}   # sigma sample size per TF ([ / ] to sweep — telescopes levels)
        self.marks = []          # dicts: ts, price, type, tf
        self._cts = None         # cursor timestamp (epoch s)
        self._day = None
        self._tf = {}            # tf -> dict(ep, xnum, o,h,l,c, cubic, sigma)
        self._log = None

    # ---- data ----
    def _load(self, day):
        d = {}
        # 5s first (cubic base)
        p5 = os.path.join(REPO, 'DATA', 'ATLAS', '5s', f'{day}.parquet')
        cubic5 = ep5 = None
        if os.path.exists(p5):
            m5 = pd.read_parquet(p5, columns=['timestamp', 'open', 'high', 'low', 'close']).sort_values('timestamp')
            ep5 = m5['timestamp'].astype('int64').to_numpy()
            c5 = m5['close'].astype(float).to_numpy()
            cubic5 = _cub.rolling(c5, 90, 5)[0] if _cub else np.full(len(c5), np.nan)
            d['5s'] = dict(ep=ep5, xnum=mdates.date2num(pd.to_datetime(ep5, unit='s')),
                           o=m5['open'].astype(float).to_numpy(), h=m5['high'].astype(float).to_numpy(),
                           l=m5['low'].astype(float).to_numpy(), c=c5,
                           cubic=cubic5, res=c5 - cubic5, sigma=_roll_std(c5 - cubic5))
        # 1m
        p1 = os.path.join(REPO, 'DATA', 'ATLAS', '1m', f'{day}.parquet')
        if os.path.exists(p1):
            m1 = pd.read_parquet(p1, columns=['timestamp', 'open', 'high', 'low', 'close']).sort_values('timestamp')
            ep1 = m1['timestamp'].astype('int64').to_numpy(); c1 = m1['close'].astype(float).to_numpy()
            if cubic5 is not None:
                k = np.searchsorted(ep5, ep1, side='right') - 1
                cub1 = np.where(k >= 0, cubic5[np.clip(k, 0, len(cubic5) - 1)], np.nan)
            else:
                cub1 = np.full(len(c1), np.nan)
            d['1m'] = dict(ep=ep1, xnum=mdates.date2num(pd.to_datetime(ep1, unit='s')),
                           o=m1['open'].astype(float).to_numpy(), h=m1['high'].astype(float).to_numpy(),
                           l=m1['low'].astype(float).to_numpy(), c=c1,
                           cubic=cub1, res=c1 - cub1, sigma=_roll_std(c1 - cub1))
        return d

    def _ensure_day(self):
        day = self.engine.days[self.engine.day_idx]
        if day == self._day:
            return
        self._day = day
        self._tf = self._load(day)
        if '1m' not in self._tf:
            return
        self.tf = '1m'
        self._cts = int(self._tf['1m']['ep'][min(WARMUP_MIN, len(self._tf['1m']['ep']) - 1)])
        self.marks = []
        os.makedirs(LOGDIR, exist_ok=True)
        if self._log:
            self._log.close()
        self._log = open(os.path.join(LOGDIR, f'replay_{day}.{self.who}.jsonl'), 'a', encoding='utf-8')
        self._rec('start')

    def _D(self):
        return self._tf.get(self.tf)

    def _cur(self, D):
        return int(np.clip(np.searchsorted(D['ep'], self._cts, side='right') - 1, 0, len(D['ep']) - 1))

    def _rec(self, ev, **kw):
        if self._log:
            self._log.write(json.dumps(dict(wall=round(time.time(), 1), day=self._day, who=self.who,
                                            tf=self.tf, event=ev, **kw)) + '\n')
            self._log.flush()

    # ---- view ----
    def _fit_view(self):
        D = self._D()
        if D is None:
            return
        cur = self._cur(D); back = int(VIEW_BARS[self.tf] * self.zoom); ahead = AHEAD[self.tf]
        lo = max(0, cur - back); hi = min(len(D['ep']) - 1, cur + ahead)
        ax = self.engine.ax
        ax.set_xlim(mdates.num2date(D['xnum'][lo]), mdates.num2date(D['xnum'][hi]))
        hh = D['h'][lo:cur + 1]; ll = D['l'][lo:cur + 1]
        hi_v = np.nanmax(hh) if np.isfinite(hh).any() else D['c'][cur]
        lo_v = np.nanmin(ll) if np.isfinite(ll).any() else D['c'][cur]
        for _, lvl in _zigzag(D['c'][:cur + 1], CUSP_R[self.tf])[-3:]:
            hi_v = max(hi_v, lvl); lo_v = min(lo_v, lvl)
        sig = self._sigma(D)
        if np.isfinite(D['cubic'][cur]) and np.isfinite(sig[cur]):
            hi_v = max(hi_v, D['cubic'][cur] + 2 * sig[cur])
            lo_v = min(lo_v, D['cubic'][cur] - 2 * sig[cur])
        pad = max(2.0, (hi_v - lo_v) * 0.08)
        ax.set_ylim(lo_v - pad, hi_v + pad)

    # ---- draw ----
    def draw(self, ax, ax_ind, time_range, patches):
        self._ensure_day()
        D = self._D()
        if D is None:
            patches.append(ax.text(0.5, 0.5, f'no data for {self._day}', transform=ax.transAxes,
                                   ha='center', color='gray')); return
        # cover the engine's own price line so we fully own the chart (any TF)
        patches.append(ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                               facecolor='white', edgecolor='none', zorder=1.9)))
        cur = self._cur(D); n = len(D['ep'])
        back = int(VIEW_BARS[self.tf] * self.zoom); ahead = AHEAD[self.tf]
        vlo = max(0, cur - back); vhi = min(n - 1, cur + ahead)
        xd = mdates.num2date(D['xnum'][vlo:cur + 1])
        cx = mdates.num2date(D['xnum'][cur])
        sigf = self._sigma(D)
        cb = D['cubic'][vlo:cur + 1]; sg = sigf[vlo:cur + 1]
        # sigma envelope
        if np.isfinite(cb).any():
            patches.append(ax.fill_between(xd, cb - 2 * sg, cb + 2 * sg, color='#5C6BC0', alpha=0.08, zorder=2, linewidth=0))
            patches.append(ax.fill_between(xd, cb - sg, cb + sg, color='#5C6BC0', alpha=0.12, zorder=2.1, linewidth=0))
            for s in (2, -2):
                ln, = ax.plot(xd, cb + s * sg, color='#5C6BC0', lw=0.9, ls='--', alpha=0.6, zorder=2.2)
                patches.append(ln)
        # candles
        oo, hh, ll, cc = D['o'][vlo:cur + 1], D['h'][vlo:cur + 1], D['l'][vlo:cur + 1], D['c'][vlo:cur + 1]
        up = cc >= oo
        wpx = (D['xnum'][1] - D['xnum'][0]) * 0.68 if len(D['xnum']) > 1 else 0.0004
        patches.append(ax.vlines(xd, ll, hh, color=np.where(up, '#2E7D32', '#C62828'), linewidth=0.9, zorder=3))
        bc = ax.bar(xd, height=np.abs(cc - oo), bottom=np.minimum(oo, cc), width=wpx,
                    color=np.where(up, '#2E7D32', '#C62828'), edgecolor='none', zorder=3.5)
        patches.extend(bc.patches)
        # cubic
        if np.isfinite(cb).any():
            ln, = ax.plot(xd, cb, color='#E8833A', lw=1.8, zorder=4); patches.append(ln)
        # current-bar horizontal sigma levels
        c0, s0 = D['cubic'][cur], sigf[cur]
        if np.isfinite(c0) and np.isfinite(s0):
            patches.append(ax.axhline(c0, color='#E8833A', ls=':', lw=1.0, alpha=0.7, zorder=4))
            patches.append(ax.text(cx, c0, f' µ {c0:.1f}', color='#E8833A', fontsize=8, va='center', ha='left', zorder=9))
            for mult, lab in [(2, '+2σ'), (1, '+1σ'), (-1, '−1σ'), (-2, '−2σ')]:
                lvl = c0 + mult * s0
                patches.append(ax.axhline(lvl, color='#3949AB', lw=1.0, alpha=0.75 if abs(mult) == 2 else 0.5, zorder=4))
                patches.append(ax.text(cx, lvl, f' {lab} {lvl:.1f}', color='#3949AB', fontsize=8, va='center', ha='left', zorder=9))
        # TF-TELESCOPE level frame (causal, refit at current bar — the frame-
        # stability gate proved overnight-static frames DECAY; rolling refit is
        # load-bearing). Falls back to legacy zigzag levels if _lcs missing.
        frame = self._frame_lines(D, cur)
        vhi_p = np.nanmax(hh) if np.isfinite(hh).any() else D['c'][cur]
        vlo_p = np.nanmin(ll) if np.isfinite(ll).any() else D['c'][cur]
        majors = []
        for scname, lvl, touches, colr, lwd in frame:
            if scname == 'day':
                majors.append(lvl)
            if vlo_p - 12 <= lvl <= vhi_p + 12:
                patches.append(ax.axhline(lvl, color=colr, ls='-', lw=lwd, alpha=0.55, zorder=3))
                patches.append(ax.text(mdates.num2date(D['xnum'][vlo]), lvl,
                                       f' {scname} {lvl:.0f} ({touches}t)',
                                       color=colr, fontsize=8, fontweight='bold' if scname == 'day' else 'normal',
                                       va='center', zorder=8))
        if not majors:                                   # coarse scale empty early in day
            majors = [lv for _, lv in _zigzag(D['c'][:cur + 1], MAJOR_R[self.tf])[-6:]]
        # proximity readout: current price vs nearest MAJOR level (the exit trigger)
        if majors:
            px_now = D['c'][cur]; near = min(majors, key=lambda L: abs(L - px_now)); dpx = px_now - near
            close = abs(dpx) <= PROX
            patches.append(ax.text(0.008, 0.90,
                                   f'top→nearest major {near:.0f} : Δ{dpx:+.1f}pt'
                                   + ('  ◀ CLOSE (exit zone)' if close else ''),
                                   transform=ax.transAxes, va='top', ha='left', fontsize=10,
                                   color='#C62828' if close else '#4A148C', fontweight='bold', zorder=10,
                                   bbox=dict(boxstyle='round', fc='#FFF3F3' if close else 'white',
                                             ec='#C62828' if close else '#4A148C', alpha=0.9)))
        # fog just ahead
        patches.append(ax.axvspan(cx, mdates.num2date(D['xnum'][vhi]), facecolor='#FAFAFA',
                                  edgecolor='none', alpha=1.0, zorder=5))
        patches.append(ax.axvline(cx, color='#455A64', lw=1.3, alpha=0.85, zorder=6))
        # marks (any tf, by timestamp)
        for mk in self.marks:
            _, col, sym = TYPES[[k for k, v in TYPES.items() if v[0] == mk['type']][0]]
            patches.append(ax.scatter(mdates.num2date(mdates.date2num(pd.to_datetime(mk['ts'], unit='s'))),
                                      mk['price'], marker=sym, s=110, color=col,
                                      edgecolor='white', linewidths=1.2, zorder=9))
        nm, col, _ = TYPES[self.mode]
        patches.append(ax.text(0.008, 0.985,
                               f'REPLAY {self.tf} ×{self.zoom:.1f} · [{nm}] click · f/F step · v back · '
                               f'1/5 TF · −/= zoom · z reset · t/l/s/e · u undo',
                               transform=ax.transAxes, va='top', ha='left', fontsize=10,
                               color=col, fontweight='bold', zorder=10,
                               bbox=dict(boxstyle='round', fc='white', ec=col, alpha=0.9)))
        self._fit_view()

    def _redraw(self):
        """Set the fitted view BEFORE the engine redraws, so the engine's xlim-restore
        keeps our per-TF window (else a TF switch keeps the old TF's x-scale)."""
        self._fit_view()
        self.engine.draw()

    def _sigma(self, D):
        """Live sigma at the current sample size (from stored residual). Sweeping
        the window telescopes the horizontal σ levels through the level hierarchy."""
        return _roll_std(D['res'], self.sw[self.tf])

    def _frame_lines(self, D, cur):
        """Causal TF-telescope frame at the current cursor: fit on 1m bars up to
        NOW (structure lives on 1m regardless of view TF). Cached per 1m bar.
        Returns [(scale_name, price, touches, color, lw)]."""
        if _lcs is None or '1m' not in self._tf:
            return []
        D1 = self._tf['1m']
        c1 = int(np.searchsorted(D1['ep'], self._cts, side='right')) - 1
        if c1 < 30:
            return []
        if getattr(self, '_frame_cache_bar', None) == c1:
            return self._frame_cache
        df = pd.DataFrame({'open': D1['o'][:c1 + 1], 'high': D1['h'][:c1 + 1],
                           'low': D1['l'][:c1 + 1], 'close': D1['c'][:c1 + 1]})
        out = []
        try:
            for sc in _lcs.telescope(df):
                for L in sc['lines']:
                    out.append((sc['name'], L['price'], L['touches'], sc['color'], sc['lw']))
        except Exception:
            out = []
        self._frame_cache_bar, self._frame_cache = c1, out
        return out

    # ---- interaction ----
    def on_click(self, event):
        D = self._D()
        if D is None or event.inaxes is None or event.xdata is None or event.ydata is None:
            return False
        bar = int(np.clip(np.searchsorted(D['xnum'], event.xdata), 0, len(D['xnum']) - 1))
        if bar > self._cur(D):                       # causal: no clicking the fog
            return True
        ts = int(D['ep'][bar]); typ = TYPES[self.mode][0]
        self.marks.append(dict(ts=ts, price=float(event.ydata), type=typ, tf=self.tf))
        self._rec('mark', type=typ, ts=ts, price=round(float(event.ydata), 2),
                  bars_early=self._cur(D) - bar)
        self._redraw(); return True

    def on_key(self, event):
        D = self._D()
        k = (event.key or '')
        if D is None:
            return False
        cur = self._cur(D)
        if k in ('f', 'right'):
            self._cts = int(D['ep'][min(cur + 1, len(D['ep']) - 1)]); self._redraw(); return True
        if k in ('F', 'g'):
            self._cts = int(D['ep'][min(cur + 10, len(D['ep']) - 1)]); self._redraw(); return True
        if k in ('v', 'left'):
            self._cts = int(D['ep'][max(WARMUP_MIN if self.tf == '1m' else 0, cur - 1)]); self._redraw(); return True
        if k in ('1', '5'):
            newtf = '1m' if k == '1' else '5s'
            if newtf in self._tf:
                self.tf = newtf; self._redraw()
            return True
        if k == 'z':
            self.zoom = 1.0; self._redraw(); return True
        if k in ('[', '{'):                     # shrink σ sample -> tight local levels
            self.sw[self.tf] = max(5, int(self.sw[self.tf] / 1.5)); self._redraw(); return True
        if k in (']', '}'):                     # grow σ sample -> telescope to big shelf
            self.sw[self.tf] = min(600, int(self.sw[self.tf] * 1.5)); self._redraw(); return True
        if k in ('-', '_'):
            self.zoom = min(8.0, self.zoom * 1.7); self._redraw(); return True
        if k in ('=', '+'):
            self.zoom = max(0.5, self.zoom / 1.7); self._redraw(); return True
        if k.lower() in TYPES:
            self.mode = k.lower(); self._redraw(); return True
        if k == 'u' and self.marks:
            m = self.marks.pop(); self._rec('undo', type=m['type'], ts=m['ts']); self._redraw(); return True
        return False

    def get_title_stats(self):
        D = self._D()
        if D is None:
            return f'{self._day} — no data'
        cx = pd.to_datetime(self._cts, unit='s')
        return (f'REPLAY {self._day} · TF={self.tf} · {cx:%H:%M:%S} · σW={self.sw[self.tf]} · '
                f'mode={TYPES[self.mode][0]} · marks={len(self.marks)}  [{self.who}]')


def get_plugin(args):
    return ChartReplayRecorder(args)
