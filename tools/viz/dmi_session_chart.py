"""Generate I-chart of a live session with trades, DMI, and volume."""
import pandas as pd, numpy as np, os, sys
from core_v2.statistical_field_engine import StatisticalFieldEngine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

df = pd.read_parquet('checkpoints/live/bars_MNQ_60s.parquet')
df['dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)

# Use last 500 bars for warmup, display last 350
df_warm = df.tail(500).reset_index(drop=True)
sfe = StatisticalFieldEngine()
states = sfe.batch_compute_states(df_warm)

disp_start = max(0, len(df_warm) - 350)
times = df_warm['dt'].values[disp_start:]
prices = df_warm['close'].values[disp_start:]
highs = df_warm['high'].values[disp_start:]
lows = df_warm['low'].values[disp_start:]
volumes = df_warm['volume'].values[disp_start:]
n = len(times)

dmi_p = np.array([getattr(states[disp_start+i]['state'] if isinstance(states[disp_start+i],dict) else states[disp_start+i], 'dmi_plus', 0) for i in range(n)])
dmi_m = np.array([getattr(states[disp_start+i]['state'] if isinstance(states[disp_start+i],dict) else states[disp_start+i], 'dmi_minus', 0) for i in range(n)])
smooth_dmi = pd.Series(dmi_p - dmi_m).rolling(3).mean().values
reg_mean = pd.Series(df_warm['close'].values).rolling(60, min_periods=30).mean().values[disp_start:]

# Trade log from session report
trades_raw = [
    (6, 'SHORT', 24325.00, 24323.75, +2.50, 'MANUAL_FLATTEN'),
    (7, 'LONG', 24325.00, 24328.75, +7.50, 'dmi_tp'),
    (8, 'LONG', 24329.25, 24331.00, +3.50, 'dmi_tp'),
    (9, 'SHORT', 24331.25, 24329.75, +3.00, 'hold_expire'),
    (10, 'SHORT', 24380.00, 24379.75, +0.50, 'dmi_tp'),
    (11, 'SHORT', 24379.25, 24374.75, +9.00, 'dmi_tp'),
    (12, 'SHORT', 24374.00, 24376.00, -4.00, 'dmi_tp'),
    (13, 'SHORT', 24375.75, 24376.00, -0.50, 'hold_expire'),
    (14, 'LONG', 24368.25, 24360.00, -16.50, 'dmi_sl'),
    (15, 'SHORT', 24357.25, 24358.75, -3.00, 'dmi_tp'),
    (16, 'SHORT', 24358.00, 24352.25, +11.50, 'dmi_tp'),
    (17, 'SHORT', 24351.75, 24356.75, -10.00, 'dmi_tp'),
    (18, 'SHORT', 24356.00, 24359.00, -6.00, 'physics_flip'),
    (19, 'LONG', 24359.00, 24360.00, +2.00, 'dmi_tp'),
    (20, 'SHORT', 24360.50, 24364.50, -8.00, 'dmi_tp'),
    (21, 'SHORT', 24364.25, 24361.50, +5.50, 'hold_expire'),
    (22, 'SHORT', 24342.75, 24343.00, -0.50, 'dmi_tp'),
    (23, 'SHORT', 24342.25, 24353.25, -22.00, 'dmi_sl'),
    (24, 'LONG', 24346.50, 24344.00, -5.00, 'dmi_sl'),
    (25, 'SHORT', 24346.75, 24344.25, +5.00, 'dmi_tp'),
]

# Match trades to bars by entry price
trade_markers = []
for t in trades_raw:
    num, side, entry_p, exit_p, pnl, reason = t
    idx = np.argmin(np.abs(prices - entry_p))
    trade_markers.append({'bar': idx, 'side': side, 'entry': entry_p, 'exit': exit_p,
                          'pnl': pnl, 'reason': reason, 'num': num})

# PLOT
fig, axes = plt.subplots(3, 1, figsize=(28, 18), sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.5, 1]})

# Panel 1: Price + regression mean + trades
ax1 = axes[0]
for i in range(n):
    c = '#00cc00' if prices[i] >= (prices[i-1] if i > 0 else prices[i]) else '#cc0000'
    ax1.plot([times[i], times[i]], [lows[i], highs[i]], color=c, lw=1.5)
    ax1.plot(times[i], prices[i], '.', color=c, ms=3)

valid = ~np.isnan(reg_mean)
ax1.plot(times[valid], reg_mean[valid], '--', color='yellow', lw=1.5, alpha=0.8, label='Reg Mean (60)')

for m in trade_markers:
    i = m['bar']
    if i >= n: continue
    c_entry = '#00ff00' if m['side'] == 'LONG' else '#ff4444'
    mk = '^' if m['side'] == 'LONG' else 'v'
    ax1.scatter(times[i], m['entry'], marker=mk, c=c_entry, s=200, zorder=5, edgecolors='white', lw=1.5)
    c_exit = '#00ff00' if m['pnl'] > 0 else '#ff4444'
    ax1.scatter(times[i], m['exit'], marker='x', c=c_exit, s=120, zorder=5, lw=2)
    ax1.annotate(f"#{m['num']} {m['reason']}\n{m['pnl']:+.1f}",
                 (times[i], m['exit']), fontsize=6, color=c_exit,
                 textcoords='offset points', xytext=(8, -10 if m['pnl'] < 0 else 10))

ax1.set_ylabel('Price', fontsize=12, color='white')
ax1.set_title('2026-03-25 Session (16:58-18:34 PST) - DMI Flipper v2', fontsize=14, fontweight='bold', color='white')
ax1.legend(fontsize=9, loc='upper left')
ax1.set_facecolor('#1a1a2e')
ax1.grid(True, alpha=0.15)

# Panel 2: DMI
ax2 = axes[1]
ax2.plot(times, dmi_p, '--', color='#00cc00', lw=1, label='DMI+')
ax2.plot(times, dmi_m, '--', color='#cc0000', lw=1, label='DMI-')
vs = ~np.isnan(smooth_dmi)
ax2.plot(times[vs], smooth_dmi[vs], color='cyan', lw=2, label='Smooth DMI diff')
ax2.axhline(0, color='white', lw=0.5, alpha=0.3)
for i in range(1, len(smooth_dmi)):
    if np.isnan(smooth_dmi[i]) or np.isnan(smooth_dmi[i-1]): continue
    if smooth_dmi[i-1] < 0 and smooth_dmi[i] > 0:
        ax2.axvline(times[i], color='#00ff00', lw=0.5, alpha=0.3)
    elif smooth_dmi[i-1] > 0 and smooth_dmi[i] < 0:
        ax2.axvline(times[i], color='#ff4444', lw=0.5, alpha=0.3)
ax2.set_ylabel('DMI', fontsize=12, color='white')
ax2.legend(fontsize=8)
ax2.set_facecolor('#1a1a2e')
ax2.grid(True, alpha=0.15)

# Panel 3: Volume
ax3 = axes[2]
vol_avg = pd.Series(volumes).rolling(30, min_periods=1).mean().values
colors = ['#00cc00' if prices[i] >= (prices[i-1] if i > 0 else prices[i]) else '#cc0000' for i in range(n)]
ax3.bar(times, volumes, width=np.timedelta64(50, 's'), color=colors, alpha=0.6)
ax3.plot(times, vol_avg, '--', color='yellow', lw=1, label='30-bar avg')
ax3.axhline(np.nanmean(volumes)*2, color='orange', lw=0.5, ls=':', alpha=0.5, label='2x avg')
ax3.set_ylabel('Volume', fontsize=12, color='white')
ax3.legend(fontsize=8)
ax3.set_facecolor('#1a1a2e')
ax3.grid(True, alpha=0.15)

for ax in axes:
    ax.tick_params(colors='white', labelsize=9)
    for sp in ax.spines.values(): sp.set_color('#333')
axes[-1].xaxis.set_major_formatter(DateFormatter('%H:%M'))
axes[-1].set_xlabel('Time (UTC)', fontsize=10, color='white')
fig.patch.set_facecolor('#0d0d1a')
plt.tight_layout()

out = 'examples/dmi_session_20260325_ichart.png'
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f'Chart: {out}')
