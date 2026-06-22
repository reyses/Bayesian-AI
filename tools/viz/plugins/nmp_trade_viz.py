"""NMP Trade Visualizer plugin.

Main indicator NMP follows = the z-score (L3_1m_z_se_15) shown in the indicator
panel with the +/-Z_ENTRY threshold lines. Price on top. Entry markers colored
by DIRECTION: GREEN = long, RED = short (^ long / v short).

Interactive:
  python -m tools.viz.run --plugin nmp_trade_viz --day 2024_02_20 --tf 1m \
         --trade-log reports/findings/nmp_fade_2024_02_smoke.csv

Headless PNG (for a quick look / verification):
  python -m tools.viz.plugins.nmp_trade_viz --day 2024_02_20 \
         --trade-log reports/findings/nmp_fade_2024_02_smoke.csv \
         --out reports/findings/nmp_trades_2024_02_20.png
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from tools.viz.core.plugin import VizPlugin
from core_v2.statistical_field_engine import _ols_fit_kernel, N_BASE

REPO = Path(__file__).resolve().parent.parent.parent.parent
TZ = 'America/New_York'                 # match VizEngine's display tz
Z_ENTRY = 1.8481                        # NMP entry threshold (|z| > Z_ENTRY)
Z_EXIT = 0.4752
LONG_C, SHORT_C = '#00b050', '#e00000'  # green long / red short
DEF_LOG = 'reports/findings/nmp_fade_2024_02_smoke.csv'


def _ny(ts):
    """unix seconds -> tz-naive America/New_York (matches the engine x-axis)."""
    return pd.to_datetime(np.asarray(ts, dtype='float64'), unit='s', utc=True
                          ).tz_convert(TZ).tz_localize(None)


def _load_trades(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['entry_ts', 'leg_dir'])
    df['entry_dt'] = _ny(df['entry_ts'])
    if 'exit_ts' in df.columns:
        df['exit_dt'] = _ny(df['exit_ts'])
    df['is_long'] = df['leg_dir'].str.upper().eq('LONG')
    return df


def _load_zline(day):
    """The indicator NMP follows: L3_1m_z_se_15 over the day, as (dt, z)."""
    p = REPO / 'DATA/ATLAS/FEATURES_5s_v2/L3_1m' / f'{day}.parquet'
    if not p.exists():
        return None, None
    z = pd.read_parquet(p, columns=['timestamp', 'L3_1m_z_se_15'])
    return _ny(z['timestamp']), z['L3_1m_z_se_15'].to_numpy()


def _load_bands(day):
    """The 3 z-bands NMP follows, in PRICE space, from the 1m trailing OLS fit
    (same kernel as compute_L3): center = regression mean (z=0); upper/lower =
    rm +/- Z_ENTRY*se (where |z| == Z_ENTRY, i.e. the snap-back trigger level)."""
    p = REPO / 'DATA/ATLAS/1m' / f'{day}.parquet'
    if not p.exists():
        return None
    m = pd.read_parquet(p, columns=['timestamp', 'close'])
    rm, se = _ols_fit_kernel(m['close'].to_numpy(np.float64), N_BASE['1m'])
    return _ny(m['timestamp']), rm, se


def _load_rm(day, tf):
    """Regression mean (trailing OLS fitted endpoint) for a TF, in price space."""
    p = REPO / 'DATA/ATLAS' / tf / f'{day}.parquet'
    if not p.exists():
        return None
    m = pd.read_parquet(p, columns=['timestamp', 'close'])
    rm, _se = _ols_fit_kernel(m['close'].to_numpy(np.float64), N_BASE[tf])
    return _ny(m['timestamp']), rm


def _draw_nmp(ax, ax_ind, day, dt_lo, dt_hi, trades, patches=None):
    """Markers + 3 z-bands on the price ax; z indicator on ax_ind. Shared by the
    plugin (engine draws base price) and the headless renderer."""
    push = (lambda o: patches.append(o)) if patches is not None else (lambda o: None)
    tr = trades[(trades['entry_dt'] >= dt_lo) & (trades['entry_dt'] <= dt_hi)]

    # --- the 3 z-bands NMP follows, in PRICE space (1m trailing OLS) ---
    b = _load_bands(day)
    if b is not None:
        bdt, rm, se = b
        push(ax.plot(bdt, rm, color='#1565c0', lw=1.0, ls='--', alpha=0.9, zorder=4)[0])   # mean (z=0)
        for k, a in [(1, 0.55), (2, 0.40), (3, 0.28)]:                                      # 1/2/3 sigma
            push(ax.plot(bdt, rm + k * se, color='#455a64', lw=0.8, alpha=a, zorder=3)[0])
            push(ax.plot(bdt, rm - k * se, color='#455a64', lw=0.8, alpha=a, zorder=3)[0])
        push(ax.plot(bdt, rm + Z_ENTRY * se, color='#ff8f00', lw=1.3, alpha=0.9, zorder=4)[0])  # entry trigger
        push(ax.plot(bdt, rm - Z_ENTRY * se, color='#ff8f00', lw=1.3, alpha=0.9, zorder=4)[0])

    # --- slower, steadier anchor: the 5m regression mean (contrast to the chasing 1m mean) ---
    r5 = _load_rm(day, '5m')
    if r5 is not None:
        push(ax.plot(r5[0], r5[1], color='#6a1b9a', lw=1.7, alpha=0.95, zorder=5, label='5m reg mean')[0])

    # --- markers: SHAPE = entry(^)/exit(v), COLOR = direction (green long / red short) ---
    for is_long, sub in tr.groupby('is_long'):
        c = LONG_C if is_long else SHORT_C
        push(ax.scatter(sub['entry_dt'], sub['entry_price'], marker='^', color=c, s=55,
                        zorder=6, edgecolors='black', linewidths=0.4))
        if 'exit_dt' in sub.columns:
            push(ax.scatter(sub['exit_dt'], sub['exit_price'], marker='v', color=c, s=42,
                            zorder=6, edgecolors='black', linewidths=0.4, alpha=0.8))
            for _, row in sub.iterrows():
                if pd.notnull(row['exit_dt']):
                    push(ax.axvspan(row['entry_dt'], row['exit_dt'], color=c, alpha=0.15, zorder=0))
    handles = [
        Line2D([], [], color='#1565c0', ls='--', label='1m reg mean'),
        Line2D([], [], color='#455a64', label='1m +/-1,2,3 sigma'),
        Line2D([], [], color='#ff8f00', label='+/-Z_ENTRY (1.85sig)'),
        Line2D([], [], color='#6a1b9a', lw=1.7, label='5m reg mean'),
        Line2D([], [], marker='^', ls='', mfc='gray', mec='k', ms=8, label='entry'),
        Line2D([], [], marker='v', ls='', mfc='gray', mec='k', ms=8, label='exit'),
        Line2D([], [], marker='s', ls='', mfc=LONG_C, mec='none', ms=9, label='long'),
        Line2D([], [], marker='s', ls='', mfc=SHORT_C, mec='none', ms=9, label='short'),
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=7.5, framealpha=0.75, ncol=2)

    # --- indicator panel: the z-score NMP follows ---
    if ax_ind is not None:
        zdt, zval = _load_zline(day)
        if zdt is not None:
            ln, = ax_ind.plot(zdt, zval, color='#37474F', lw=0.8, zorder=2)
            push(ln)
        for k in (1, 2, 3):                       # +/- 1,2,3 sigma grid
            for s in (k, -k):
                push(ax_ind.axhline(s, color='#90a4ae', lw=0.6, ls='-', alpha=0.5, zorder=1))
        push(ax_ind.axhline(0.0, color='#b0bec5', lw=0.7, ls=':', zorder=1))
        for s in (Z_ENTRY, -Z_ENTRY):             # entry trigger (1.85 sigma)
            push(ax_ind.axhline(s, color='#ff8f00', lw=1.1, ls='--', zorder=1))
        for s in (Z_EXIT, -Z_EXIT):               # exit band
            push(ax_ind.axhline(s, color='#c0ca33', lw=0.7, ls=':', zorder=1))
        # entries on the z line (extra_z_se = z at entry), colored by direction
        if 'extra_z_se' in tr.columns:
            for is_long, sub in tr.groupby('is_long'):
                c = LONG_C if is_long else SHORT_C
                push(ax_ind.scatter(sub['entry_dt'], sub['extra_z_se'], marker='^', color=c,
                                    s=36, zorder=5, edgecolors='black', linewidths=0.3))
        ax_ind.set_ylabel('z_se (1m)  +/-Z_ENTRY', fontsize=9)


# ---------------- interactive plugin ----------------
class NMPTradeVizPlugin(VizPlugin):
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
        print(f"[nmp_trade_viz] {len(self.trades)} trades from {self.args.trade_log}")

    def draw(self, ax, ax_ind, time_range, patches):
        day = self.engine.days[self.engine.day_idx]
        lo, hi = self.engine.dt.iloc[0], self.engine.dt.iloc[-1]
        _draw_nmp(ax, ax_ind, day, lo, hi, self.trades, patches)
        d = self.trades[(self.trades['entry_dt'] >= lo) & (self.trades['entry_dt'] <= hi)]
        pnl_col = 'net_usd' if 'net_usd' in d.columns else 'pnl_usd'
        pnl_val = float(d[pnl_col].sum()) if pnl_col in d.columns else 0.0
        self._stats = {'n': len(d), 'L': int(d['is_long'].sum()),
                       'S': int((~d['is_long']).sum()), 'pnl': pnl_val}

    def get_title_stats(self):
        s = self._stats
        return (f"NMP trades: {s.get('n',0)}  (L {s.get('L',0)} / S {s.get('S',0)})  "
                f"|  day PnL ${s.get('pnl',0):,.0f}")


def get_plugin(unknown_args):
    return NMPTradeVizPlugin(unknown_args)


# ---------------- headless PNG renderer ----------------
def render_png(day, trade_log, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    px = pd.read_parquet(REPO / 'DATA/ATLAS/1s' / f'{day}.parquet')   # 1s granular price
    dt = pd.Series(_ny(px['timestamp']))   # Series so .iloc matches engine.dt
    trades = _load_trades(trade_log)

    fig, (ax, ax_ind) = plt.subplots(2, 1, figsize=(18, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.06, left=0.05, right=0.98, top=0.93, bottom=0.06)
    # 1s price (granular path) under the 1m z-bands
    ax.plot(dt, px['close'], color='#78909c', lw=0.4, alpha=0.9, zorder=1)
    ax.set_ylabel('Price (1s) + 1m z-bands')
    ax.grid(True, alpha=0.2)

    _draw_nmp(ax, ax_ind, day, dt.iloc[0], dt.iloc[-1], trades)
    ax_ind.grid(True, alpha=0.2)

    d = trades[(trades['entry_dt'] >= dt.iloc[0]) & (trades['entry_dt'] <= dt.iloc[-1])]
    pnl_str = f"  |  day PnL ${d['pnl_usd'].sum():,.0f}" if 'pnl_usd' in d.columns else ""
    ax.set_title(f"NMP entries {day}  |  {len(d)} entries "
                 f"(L {int(d['is_long'].sum())} / S {int((~d['is_long']).sum())}){pnl_str}   "
                 f"[^ entry / v exit  -  green=long, red=short]", fontsize=12, fontweight='bold', loc='left')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"saved {out}  ({len(d)} trades on {day})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--day', required=True)
    ap.add_argument('--trade-log', default=DEF_LOG)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    out = a.out or f'reports/findings/nmp_trades_{a.day}.png'
    render_png(a.day, a.trade_log, out)
