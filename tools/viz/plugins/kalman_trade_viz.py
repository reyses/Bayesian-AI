"""Kalman Trade Visualizer — inspect the GA-Kalman trades interactively on the real chart.

Price panel: entry ▲ / exit ▼ markers colored by ARCHETYPE (consensus labels), a faint span
line per trade, the MFE peak (★), and the Kalman smoothed-position line.
Indicator panel: the Kalman VELOCITY (the entry signal) with ±entry-threshold lines, so you
SEE the trigger fire. Step through days with the VizEngine.

Interactive:
  python -m tools.viz.run --plugin kalman_trade_viz --day 2025_03_10 --tf 1s \
         --trade-log reports/findings/kalman_clean_trades.csv
Headless PNG (verify a day):
  python -m tools.viz.plugins.kalman_trade_viz --day 2025_03_10 --out reports/findings/kalman_viz_2025_03_10.png
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tools.viz.core.plugin import VizPlugin

REPO = Path(__file__).resolve().parent.parent.parent.parent
ATLAS_1S = REPO / 'DATA/ATLAS/1s'
TZ = 'America/New_York'
DEF_LOG = 'reports/findings/kalman_clean_trades.csv'
Q, R, ENTRY_VEL = 1.81e-09, 12.5533, 0.06626
ARCH_C = {'CLEAN_RIDE': '#00b050', 'GAVE_BACK': '#ff8f00', 'STOPPED': '#e00000',
          'CHOP': '#9e9e9e', 'GAP_TRUNCATED': '#6a1b9a', 'SMALL_WIN': '#80c89a',
          'SMALL_LOSS': '#e3a0a0'}


def _ny(ts):
    return pd.to_datetime(np.asarray(ts, dtype='float64'), unit='s', utc=True).tz_convert(TZ).tz_localize(None)


def _archetype(r, p75):
    g = r['net_usd'] / 2 + 1.25
    kept = g / r['mfe_pts'] if r['mfe_pts'] > 1e-6 else 0
    if r.get('gap_close', 0) == 1 and (r['exit_ts'] - r['entry_ts']) <= 120: return 'GAP_TRUNCATED'
    if r['net_usd'] <= -90: return 'STOPPED'
    if r['mfe_pts'] >= 20 and kept < 0.4: return 'GAVE_BACK'
    if r['mfe_pts'] < 10: return 'CHOP'
    if r['mfe_pts'] >= p75 and r['net_usd'] > 0 and kept >= 0.6: return 'CLEAN_RIDE'
    return 'SMALL_WIN' if r['net_usd'] > 0 else 'SMALL_LOSS'


def _kalman_vel_pos(px, q=Q, r=R, dt=1.0):
    n = len(px); vel = np.empty(n); pos = np.empty(n)
    x0, x1, x2 = px[0], 0.0, 0.0
    P = np.eye(3) * 1e3
    F = np.array([[1, dt, .5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    Qm = q * np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2], [dt**3/6, dt**2/2, dt]])
    x = np.array([x0, x1, x2])
    for i in range(n):
        x = F @ x; P = F @ P @ F.T + Qm
        y = px[i] - x[0]; S = P[0, 0] + r; K = P[:, 0] / S
        x = x + K * y; P = (np.eye(3) - np.outer(K, [1, 0, 0])) @ P
        pos[i], vel[i] = x[0], x[1]
    return pos, vel


def _load_trades(path):
    df = pd.read_csv(path)
    p75 = df['mfe_pts'].quantile(0.75)
    df['arch'] = df.apply(lambda r: _archetype(r, p75), axis=1)
    df['entry_dt'] = _ny(df['entry_ts']); df['exit_dt'] = _ny(df['exit_ts'])
    df['day'] = df['day'].astype(str)
    return df


_KCACHE = {}
def _kalman_for_day(day):
    if day in _KCACHE:
        return _KCACHE[day]
    f = ATLAS_1S / f'{day}.parquet'
    if not f.exists():
        return None
    d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
    px = d['close'].to_numpy(np.float64); ts = d['timestamp'].to_numpy(np.int64)
    pos, vel = _kalman_vel_pos(px)
    out = (_ny(ts), px, pos, vel, ts)
    _KCACHE[day] = out
    return out


def _draw(ax, ax_ind, day, lo, hi, trades, patches=None):
    push = (lambda o: patches.append(o)) if patches is not None else (lambda o: None)
    k = _kalman_for_day(day)
    tr = trades[(trades['day'] == day) & (trades['entry_dt'] >= lo) & (trades['entry_dt'] <= hi)]
    if k is not None:
        kdt, px, pos, vel, ts = k
        push(ax.plot(kdt, pos, color='#1565c0', lw=1.0, alpha=0.8, zorder=4, label='Kalman pos')[0])
        if ax_ind is not None:
            push(ax_ind.plot(kdt, vel, color='#37474f', lw=0.7, zorder=2)[0])
            for s in (ENTRY_VEL, -ENTRY_VEL):
                push(ax_ind.axhline(s, color='#ff8f00', lw=1.0, ls='--', zorder=1))
            push(ax_ind.axhline(0, color='#b0bec5', lw=0.6, ls=':', zorder=1))
            ax_ind.set_ylabel('Kalman velocity\n(±entry thr)', fontsize=9)
    # per-trade: span line + entry/exit markers + MFE peak, colored by archetype
    for _, t in tr.iterrows():
        c = ARCH_C.get(t['arch'], '#000')
        push(ax.plot([t['entry_dt'], t['exit_dt']], [t['entry_price'], t['exit_price']],
                     color=c, lw=0.8, alpha=0.5, zorder=5)[0])
        push(ax.scatter([t['entry_dt']], [t['entry_price']], marker='^', color=c, s=55,
                        edgecolors='black', linewidths=0.4, zorder=6))
        push(ax.scatter([t['exit_dt']], [t['exit_price']], marker='v', color=c, s=45,
                        edgecolors='black', linewidths=0.4, alpha=0.85, zorder=6))
        # MFE peak from 1s within the trade
        if k is not None:
            m = (ts >= t['entry_ts']) & (ts <= t['exit_ts'])
            if m.any():
                seg = px[m]; sdt = kdt[m]
                pk = np.argmax(seg) if t['dir'] == 'LONG' else np.argmin(seg)
                push(ax.scatter([sdt[pk]], [seg[pk]], marker='*', color=c, s=70, zorder=6, alpha=0.7))
    handles = [Line2D([], [], color='#1565c0', label='Kalman pos')] + \
              [Line2D([], [], marker='^', ls='', mfc=c, mec='k', ms=8, label=a) for a, c in ARCH_C.items()]
    handles += [Line2D([], [], marker='*', ls='', mfc='gray', mec='none', ms=10, label='MFE peak')]
    ax.legend(handles=handles, loc='upper left', fontsize=7, ncol=2, framealpha=0.8)


class KalmanTradeVizPlugin(VizPlugin):
    requires_indicator_panel = True

    def __init__(self, args):
        super().__init__()
        ap = argparse.ArgumentParser()
        ap.add_argument('--trade-log', default=DEF_LOG)
        self.args = ap.parse_args(args)
        self.trades = _load_trades(self.args.trade_log)
        self._stats = {}

    def setup(self, engine, **kwargs):
        super().setup(engine, **kwargs)
        print(f"[kalman_trade_viz] {len(self.trades)} trades from {self.args.trade_log}")

    def draw(self, ax, ax_ind, time_range, patches):
        day = str(self.engine.days[self.engine.day_idx])
        lo, hi = self.engine.dt.iloc[0], self.engine.dt.iloc[-1]
        _draw(ax, ax_ind, day, lo, hi, self.trades, patches)
        d = self.trades[(self.trades['day'] == day) & (self.trades['entry_dt'] >= lo) & (self.trades['entry_dt'] <= hi)]
        self._stats = {'n': len(d), 'net': float(d['net_usd'].sum()),
                       'arch': d['arch'].value_counts().to_dict()}

    def get_title_stats(self):
        s = self._stats
        a = " ".join(f"{k}:{v}" for k, v in s.get('arch', {}).items())
        return f"Kalman trades: {s.get('n',0)} | net ${s.get('net',0):,.0f} | {a}"


def get_plugin(unknown_args):
    return KalmanTradeVizPlugin(unknown_args)


def render_png(day, trade_log, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = _kalman_for_day(day)
    if k is None:
        print("no 1s data for", day); return
    kdt, px, pos, vel, ts = k
    trades = _load_trades(trade_log)
    fig, (ax, ax_ind) = plt.subplots(2, 1, figsize=(18, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.06, left=0.05, right=0.98, top=0.93, bottom=0.06)
    ax.plot(kdt, px, color='#90a4ae', lw=0.4, alpha=0.9, zorder=1)
    ax.set_ylabel('Price (1s) + Kalman + trades'); ax.grid(True, alpha=0.2)
    _draw(ax, ax_ind, day, kdt[0], kdt[-1], trades)
    ax_ind.grid(True, alpha=0.2)
    d = trades[trades['day'] == day]
    ax.set_title(f"Kalman trades {day} | {len(d)} trades net ${d['net_usd'].sum():,.0f}  "
                 f"[^entry vexit *MFE, color=archetype]", fontsize=12, fontweight='bold', loc='left')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"saved {out} ({len(d)} trades on {day})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--trade-log', default=DEF_LOG)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    render_png(a.day, a.trade_log, a.out or f'reports/findings/kalman_viz_{a.day}.png')
