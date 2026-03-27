"""I-MR Overlay for TradeCNN trades on price chart.

4 panels: Price + SE bands + trades, DMI, Volume, PnL/Equity
Works with any TradeCNN trade log CSV.

Usage:
    python -m tools.trade_cnn_imr_overlay                          # last day OOS
    python -m tools.trade_cnn_imr_overlay --date 2026-02-15        # specific date
    python -m tools.trade_cnn_imr_overlay --data DATA/ATLAS_OOS    # data source
    python -m tools.trade_cnn_imr_overlay --log path/to/trades.csv # custom trade log
    python -m tools.trade_cnn_imr_overlay --dpi 1000               # ultra HD
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def main():
    parser = argparse.ArgumentParser(description='TradeCNN I-MR Overlay')
    parser.add_argument('--date', default=None, help='Date (YYYY-MM-DD), default=last day')
    parser.add_argument('--data', default='DATA/ATLAS_OOS', help='ATLAS data root')
    parser.add_argument('--log', default='checkpoints/trade_cnn/oos_trade_log.csv',
                        help='Trade log CSV')
    parser.add_argument('--dpi', type=int, default=300, help='Output DPI')
    parser.add_argument('--output', default=None, help='Output path (default: examples/)')
    parser.add_argument('--se-window', type=int, default=60, help='SE band window')
    args = parser.parse_args()

    # Load price data
    files = sorted(glob.glob(os.path.join(args.data, '1m', '*.parquet')))
    if not files:
        print(f"No 1m data in {args.data}")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

    # Select date
    if args.date:
        _target = pd.Timestamp(args.date).date()
    else:
        _target = df['dt'].dt.date.max()

    day_mask = df['dt'].dt.date == _target
    if not day_mask.any():
        print(f"No data for {_target}")
        return
    df_day = df[day_mask].reset_index(drop=True)
    day_start = df[day_mask].index[0]
    n = len(df_day)
    print(f"Date: {_target}, bars: {n}")

    # Load trades
    if not os.path.exists(args.log):
        print(f"No trade log: {args.log}")
        return
    log = pd.read_csv(args.log)
    day_trades = log[(log['bar'] >= day_start) & (log['bar'] < day_start + n)].reset_index(drop=True)
    print(f"Trades: {len(day_trades)}")

    times = df_day['dt'].values
    prices = df_day['close'].values
    highs = df_day['high'].values
    lows = df_day['low'].values
    volumes = df_day['volume'].values if 'volume' in df_day.columns else np.zeros(n)

    # SE bands
    _w = args.se_window
    reg_mean = pd.Series(prices).rolling(_w, min_periods=_w//2).mean().values
    reg_std = pd.Series(prices).rolling(_w, min_periods=_w//2).std().values
    se = reg_std / np.sqrt(_w)
    _v = ~np.isnan(reg_mean)

    # SFE for DMI
    from core.statistical_field_engine import StatisticalFieldEngine
    sfe = StatisticalFieldEngine()
    # Use wider window for warmup
    _warm_start = max(0, day_start - 500)
    _warm_df = df.iloc[_warm_start:day_start + n].reset_index(drop=True)
    states_warm = sfe.batch_compute_states(_warm_df)
    _offset = day_start - _warm_start
    states_day = states_warm[_offset:]
    dmi_p = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_plus', 0) for s in states_day[:n]])
    dmi_m = np.array([getattr(s['state'] if isinstance(s, dict) else s, 'dmi_minus', 0) for s in states_day[:n]])

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(48, 28), sharex=True,
                             gridspec_kw={'height_ratios': [4, 1.5, 1, 1]})
    ax1, ax2, ax3, ax4 = axes

    # Panel 1: Price + SE + trades
    for i in range(n):
        c = '#00cc00' if prices[i] >= (prices[i-1] if i > 0 else prices[i]) else '#cc0000'
        ax1.plot([times[i], times[i]], [lows[i], highs[i]], color=c, lw=1.5)
        ax1.plot(times[i], prices[i], '.', color=c, ms=2)

    ax1.plot(times[_v], reg_mean[_v], '--', color='yellow', lw=1, alpha=0.8, label='Reg Mean')
    for _k, _alpha in [(1, 0.1), (2, 0.07), (3, 0.04)]:
        ax1.fill_between(times[_v], (reg_mean - _k * se)[_v], (reg_mean + _k * se)[_v],
                         alpha=_alpha, color='cyan', label=f'{_k} SE' if _k == 1 else '')

    for _, t in day_trades.iterrows():
        _local = int(t['bar']) - day_start
        if _local < 0 or _local >= n:
            continue
        _c = '#00ff00' if t['dir'] == 'LONG' else '#ff4444'
        _mk = '^' if t['dir'] == 'LONG' else 'v'
        ax1.scatter(times[_local], t['entry'], marker=_mk, c=_c, s=100, zorder=5, edgecolors='white', lw=1)
        _ec = '#00ff00' if t['pnl'] > 0 else '#ff4444'
        ax1.scatter(times[_local], t['price'], marker='x', c=_ec, s=60, zorder=5, lw=1.5)
        _held = int(t['held'])
        _exit_local = min(_local + _held, n - 1)
        if _held > 0:
            ax1.plot([times[_local], times[_exit_local]], [t['entry'], t['price']],
                     color=_c, lw=0.5, alpha=0.3)

    _total = day_trades['pnl'].sum()
    ax1.set_title(f'{_target} TradeCNN I-MR: {len(day_trades)} trades, {_total:.0f}t (${_total*0.5:,.0f})',
                  fontsize=16, fontweight='bold', color='white')
    ax1.set_ylabel('Price', fontsize=12, color='white')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.set_facecolor('#1a1a2e')
    ax1.grid(True, alpha=0.15)

    # Panel 2: DMI
    ax2.plot(times[:len(dmi_p)], dmi_p, '--', color='#00cc00', lw=1, label='DMI+')
    ax2.plot(times[:len(dmi_m)], dmi_m, '--', color='#cc0000', lw=1, label='DMI-')
    _smooth = pd.Series(dmi_p - dmi_m).rolling(3).mean().values
    _vs = ~np.isnan(_smooth)
    ax2.plot(times[_vs], _smooth[_vs], color='cyan', lw=1.5, label='Smooth diff')
    ax2.axhline(0, color='white', lw=0.5, alpha=0.3)
    ax2.set_ylabel('DMI', fontsize=12, color='white')
    ax2.legend(fontsize=8)
    ax2.set_facecolor('#1a1a2e')
    ax2.grid(True, alpha=0.15)

    # Panel 3: Volume
    vol_colors = ['#00cc00' if prices[i] >= (prices[i-1] if i > 0 else prices[i]) else '#cc0000'
                  for i in range(n)]
    ax3.bar(times, volumes, width=np.timedelta64(50, 's'), color=vol_colors, alpha=0.5)
    _vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values
    ax3.plot(times, _vol_avg, '--', color='yellow', lw=1, label='30-bar avg')
    ax3.set_ylabel('Volume', fontsize=12, color='white')
    ax3.legend(fontsize=8)
    ax3.set_facecolor('#1a1a2e')
    ax3.grid(True, alpha=0.15)

    # Panel 4: PnL + equity
    if len(day_trades) > 0:
        _locals = (day_trades['bar'].values - day_start).astype(int)
        _valid = (_locals >= 0) & (_locals < n)
        if _valid.any():
            _pnls = day_trades['pnl'].values[_valid] * 0.5
            _cols = ['#00cc00' if p > 0 else '#cc0000' for p in _pnls]
            ax4.bar(times[_locals[_valid]], _pnls, width=np.timedelta64(50, 's'), color=_cols, alpha=0.7)
            _eq = np.cumsum(_pnls)
            ax4.plot(times[_locals[_valid]], _eq, color='cyan', lw=2, label='Equity')
    ax4.axhline(0, color='white', lw=0.5)
    ax4.set_ylabel('PnL / Equity ($)', fontsize=12, color='white')
    ax4.legend(fontsize=8)
    ax4.set_facecolor('#1a1a2e')
    ax4.grid(True, alpha=0.15)

    for ax in axes:
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#333')
    axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes[-1].set_xlabel('Time (UTC)', fontsize=10, color='white')
    fig.patch.set_facecolor('#0d0d1a')
    plt.tight_layout()

    _out = args.output or f'examples/trade_cnn_imr_{_target}.png'
    os.makedirs(os.path.dirname(_out), exist_ok=True)
    fig.savefig(_out, dpi=args.dpi, facecolor=fig.get_facecolor())
    plt.close()
    print(f'Chart: {_out} ({len(day_trades)} trades, ${_total*0.5:,.0f})')


if __name__ == '__main__':
    main()
