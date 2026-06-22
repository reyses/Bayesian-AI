"""Single-trade inspector — shows ONE Kalman trade at a time, zoomed to it.
Standalone (no VizEngine), so it runs even if the full engine is fussy.

Interactive (a window opens; step with keys):
  python research/inspect_one_trade.py
  python research/inspect_one_trade.py --arch GAVE_BACK     # only that archetype
Keys: → / n = next trade, ← / p = prev, s = save PNG, q = quit.

Headless single PNG (no window):
  python research/inspect_one_trade.py --i 0 --save reports/findings/one_trade.png
"""
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
ATLAS_1S = REPO / 'DATA/ATLAS/1s'
TZ = 'America/New_York'
ENTRY_VEL = 0.06626
ARCH_C = {'CLEAN_RIDE': '#00b050', 'GAVE_BACK': '#ff8f00', 'STOPPED': '#e00000',
          'CHOP': '#9e9e9e', 'GAP_TRUNCATED': '#6a1b9a', 'SMALL_WIN': '#80c89a', 'SMALL_LOSS': '#e3a0a0'}


def _ny(ts):
    return pd.to_datetime(np.asarray(ts, float), unit='s', utc=True).tz_convert(TZ).tz_localize(None)


def _arch(r, p75):
    g = r['net_usd'] / 2 + 1.25
    kept = g / r['mfe_pts'] if r['mfe_pts'] > 1e-6 else 0
    if r.get('gap_close', 0) == 1 and (r['exit_ts'] - r['entry_ts']) <= 120: return 'GAP_TRUNCATED'
    if r['net_usd'] <= -90: return 'STOPPED'
    if r['mfe_pts'] >= 20 and kept < 0.4: return 'GAVE_BACK'
    if r['mfe_pts'] < 10: return 'CHOP'
    if r['mfe_pts'] >= p75 and r['net_usd'] > 0 and kept >= 0.6: return 'CLEAN_RIDE'
    return 'SMALL_WIN' if r['net_usd'] > 0 else 'SMALL_LOSS'


def _kalman(px, q=1.81e-09, r=12.5533, dt=1.0):
    n = len(px); vel = np.empty(n); pos = np.empty(n)
    x = np.array([px[0], 0.0, 0.0]); P = np.eye(3) * 1e3
    F = np.array([[1, dt, .5 * dt * dt], [0, 1, dt], [0, 0, 1]])
    Qm = q * np.array([[dt**5/20, dt**4/8, dt**3/6], [dt**4/8, dt**3/3, dt**2/2], [dt**3/6, dt**2/2, dt]])
    H = np.array([1.0, 0, 0])
    for i in range(n):
        x = F @ x; P = F @ P @ F.T + Qm
        S = P[0, 0] + r; K = P[:, 0] / S
        x = x + K * (px[i] - x[0]); P = (np.eye(3) - np.outer(K, H)) @ P
        pos[i], vel[i] = x[0], x[1]
    return pos, vel


_DAY = {}
def _day1s(day):
    if day not in _DAY:
        f = ATLAS_1S / f'{day}.parquet'
        d = pd.read_parquet(f, columns=['timestamp', 'close']).sort_values('timestamp')
        _DAY[day] = (d['timestamp'].to_numpy(np.int64), d['close'].to_numpy(np.float64))
    return _DAY[day]


def draw(fig, axp, axv, tr, i):
    axp.clear(); axv.clear()
    t = tr.iloc[i]
    ts, px = _day1s(t['day'])
    pad = int(max(300, 0.6 * (t['exit_ts'] - t['entry_ts'])))
    m = (ts >= t['entry_ts'] - pad) & (ts <= t['exit_ts'] + pad)
    wts, wpx = ts[m], px[m]
    if len(wpx) < 5:
        axp.set_title("no data"); fig.canvas.draw_idle(); return
    pos, vel = _kalman(wpx)
    dt = _ny(wts)
    c = ARCH_C.get(t['arch'], 'k')
    axp.plot(dt, wpx, color='#90a4ae', lw=0.7, alpha=0.9, label='price 1s')
    axp.plot(dt, pos, color='#1565c0', lw=1.3, label='Kalman pos')
    ed, xd = _ny([t['entry_ts']])[0], _ny([t['exit_ts']])[0]
    axp.scatter([ed], [t['entry_price']], marker='^', color=c, s=130, edgecolors='k', zorder=6, label='entry')
    axp.scatter([xd], [t['exit_price']], marker='v', color=c, s=130, edgecolors='k', zorder=6, label='exit')
    seg = (wts >= t['entry_ts']) & (wts <= t['exit_ts'])
    if seg.any():
        s_px, s_dt = wpx[seg], dt[seg]
        pk = np.argmax(s_px) if t['dir'] == 'LONG' else np.argmin(s_px)
        axp.scatter([s_dt[pk]], [s_px[pk]], marker='*', color=c, s=220, zorder=6, label='MFE peak')
    axp.axvspan(ed, xd, color=c, alpha=0.08)
    axp.set_ylabel('price'); axp.legend(loc='upper left', fontsize=8)
    axp.set_title(f"#{i} {t['arch']} | {t['dir']} {t['split']} | net ${t['net_usd']:.0f}  "
                  f"MFE {t['mfe_pts']:.0f}pt  MAE {t['mae_pts']:.0f}pt  dur {(t['exit_ts']-t['entry_ts'])/60:.0f}m  "
                  f"| {t['day']} | trade {i+1}/{len(tr)}   [→/n next  ←/p prev  s save  q quit]",
                  fontsize=10, fontweight='bold', loc='left')
    axv.plot(dt, vel, color='#37474f', lw=0.8)
    for s in (ENTRY_VEL, -ENTRY_VEL):
        axv.axhline(s, color='#ff8f00', lw=1.0, ls='--')
    axv.axhline(0, color='#b0bec5', lw=0.6, ls=':')
    axv.axvline(ed, color='green', lw=1, ls='--'); axv.axvline(xd, color='red', lw=1, ls='--')
    axv.set_ylabel('Kalman vel\n(±entry thr)'); axv.grid(True, alpha=0.2); axp.grid(True, alpha=0.2)
    fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trade-log', default='reports/findings/kalman_clean_trades.csv')
    ap.add_argument('--arch', default=None, help='filter to one archetype')
    ap.add_argument('--i', type=int, default=0, help='start index')
    ap.add_argument('--save', default=None, help='render this one to PNG headless and exit')
    a = ap.parse_args()
    tr = pd.read_csv(a.trade_log)
    p75 = tr['mfe_pts'].quantile(0.75)
    tr['arch'] = tr.apply(lambda r: _arch(r, p75), axis=1)
    if a.arch:
        tr = tr[tr['arch'] == a.arch].reset_index(drop=True)
    tr = tr.sort_values(['day', 'entry_ts']).reset_index(drop=True)
    print(f"{len(tr)} trades" + (f" of {a.arch}" if a.arch else ""))

    if a.save:
        matplotlib.use('Agg')
    fig, (axp, axv) = plt.subplots(2, 1, figsize=(15, 7), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    state = {'i': max(0, min(a.i, len(tr) - 1))}
    draw(fig, axp, axv, tr, state['i'])

    if a.save:
        os.makedirs(os.path.dirname(a.save) or '.', exist_ok=True)
        fig.savefig(a.save, dpi=130, bbox_inches='tight'); print(f"saved {a.save}"); return

    def on_key(e):
        if e.key in ('right', 'n'): state['i'] = (state['i'] + 1) % len(tr)
        elif e.key in ('left', 'p'): state['i'] = (state['i'] - 1) % len(tr)
        elif e.key == 's':
            p = f"reports/findings/one_trade_{state['i']}.png"; fig.savefig(p, dpi=130, bbox_inches='tight')
            print("saved", p); return
        elif e.key == 'q': plt.close(fig); return
        else: return
        draw(fig, axp, axv, tr, state['i'])

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show(block=True)


if __name__ == '__main__':
    main()
