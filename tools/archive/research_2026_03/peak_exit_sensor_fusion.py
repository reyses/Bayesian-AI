"""
Peak Exit Sensor Fusion Research -- multi-TF trajectory analysis.

Uses the ACTUAL peak detection signals (P_at_center, F_momentum, coherence)
plus multi-TF sensors:
  - 1s velocity  (fast/noisy -- detects the turn instantly)
  - 1m volume    (slow/accurate -- institutional flow confirmation)
  - 15s state    (execution TF -- peak detection + DMI)

For each peak trade, replays all three TFs and captures per-bar state.
Generates I-MR charts showing when sensors agree the peak has passed.

Usage:
    python tools/peak_exit_sensor_fusion.py [--oos] [--max-trades 1000]

Output:
    reports/findings/sensor_fusion_trajectories.csv
    reports/findings/sensor_fusion_charts.png
    reports/findings/sensor_fusion_summary.txt
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.statistical_field_engine import StatisticalFieldEngine


def load_tf_states(data_root, tf_label):
    """Load one TF's parquets and compute states."""
    tf_dir = os.path.join(data_root, tf_label)
    if not os.path.isdir(tf_dir):
        print(f"  WARNING: {tf_dir} not found, skipping")
        return None, None

    files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))
    if not files:
        return None, None

    engine = StatisticalFieldEngine()
    all_states = []
    all_bars = []

    for fname in files:
        df = pd.read_parquet(os.path.join(tf_dir, fname))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        raw = engine.batch_compute_states(df)
        for r in raw:
            st = r['state'] if isinstance(r, dict) and 'state' in r else r
            all_states.append(st)
        all_bars.append(df)

    df_all = pd.concat(all_bars, ignore_index=True)
    print(f"  {tf_label}: {len(df_all)} bars, {len(all_states)} states")
    return df_all, all_states


def find_state_at_ts(timestamps, states, target_ts):
    """Find the state at or just before target_ts."""
    idx = np.searchsorted(timestamps, target_ts, side='right') - 1
    if idx < 0:
        idx = 0
    if idx >= len(states):
        idx = len(states) - 1
    return idx, states[idx]


def extract_trajectory(trade, ts_15s, states_15s, ts_1m, states_1m,
                       ts_1s, states_1s, pre=2, post=10):
    """Extract multi-TF trajectory for one trade."""
    entry_ts = int(trade['entry_time'])
    hold = int(trade.get('hold_bars', 1))
    direction = trade['direction'].upper()
    entry_price = trade['entry_price']

    # Find entry bar in 15s
    entry_idx = int(np.searchsorted(ts_15s, entry_ts))
    if entry_idx >= len(states_15s) or entry_idx < 1:
        return []

    exit_idx = min(entry_idx + hold, len(states_15s) - 1)
    start = max(entry_idx - pre, 0)
    end = min(exit_idx + post, len(states_15s) - 1)

    # Entry state snapshots
    entry_st = states_15s[entry_idx]
    entry_pac = getattr(entry_st, 'P_at_center', 0.0) or 0.0
    entry_fm = abs(getattr(entry_st, 'F_momentum', 0.0) or 0.0)

    rows = []
    prev_pac = entry_pac
    prev_fm_abs = entry_fm

    for i in range(start, end + 1):
        if i >= len(states_15s):
            break

        st_15s = states_15s[i]
        offset = i - entry_idx
        in_trade = (i >= entry_idx and i <= exit_idx)
        bar_ts = int(ts_15s[i]) if i < len(ts_15s) else entry_ts

        # ── 15s state (execution TF -- peak detection) ──
        pac = getattr(st_15s, 'P_at_center', 0.0) or 0.0
        fm = getattr(st_15s, 'F_momentum', 0.0) or 0.0
        fm_abs = abs(fm)
        coh = getattr(st_15s, 'oscillation_entropy_normalized', 0.0) or 0.0
        z = getattr(st_15s, 'z_score', 0.0) or 0.0
        price = getattr(st_15s, 'price', 0.0) or 0.0
        dmi_plus = getattr(st_15s, 'dmi_plus', 0.0) or 0.0
        dmi_minus = getattr(st_15s, 'dmi_minus', 0.0) or 0.0
        adx = getattr(st_15s, 'adx_strength', 0.0) or 0.0
        vel_15s = getattr(st_15s, 'velocity', 0.0) or 0.0
        p_upper = getattr(st_15s, 'P_near_upper', 0.0) or 0.0
        p_lower = getattr(st_15s, 'P_near_lower', 0.0) or 0.0
        reversion = getattr(st_15s, 'mean_reversion_force', 0.0) or 0.0

        # Peak detection (same logic as advance_engine)
        pac_up = pac > prev_pac * 1.05 if prev_pac > 0.01 else False
        fm_down = fm_abs < prev_fm_abs * 0.90 if prev_fm_abs > 0.5 else False
        peak_fires = (pac_up or fm_down) and coh > 0.55
        prev_pac = pac
        prev_fm_abs = fm_abs

        # dP_center/dt (rate of change -- research says this is the leading signal)
        d_pac = pac - (getattr(states_15s[i-1], 'P_at_center', 0.0) or 0.0) if i > 0 else 0.0

        # ── 1m state (volume sensor -- institutional flow) ──
        vol_1m = 0.0
        vel_1m = 0.0
        fm_1m = 0.0
        dmi_diff_1m = 0.0
        if states_1m is not None:
            _, st_1m = find_state_at_ts(ts_1m, states_1m, bar_ts)
            vol_1m = getattr(st_1m, 'volume_delta', 0.0) or 0.0
            vel_1m = getattr(st_1m, 'velocity', 0.0) or 0.0
            fm_1m = getattr(st_1m, 'F_momentum', 0.0) or 0.0
            dmi_plus_1m = getattr(st_1m, 'dmi_plus', 0.0) or 0.0
            dmi_minus_1m = getattr(st_1m, 'dmi_minus', 0.0) or 0.0
            dmi_diff_1m = dmi_plus_1m - dmi_minus_1m

        # ── 1s state (velocity sensor -- fast turn detection) ──
        vel_1s = 0.0
        fm_1s = 0.0
        if states_1s is not None:
            _, st_1s = find_state_at_ts(ts_1s, states_1s, bar_ts)
            vel_1s = getattr(st_1s, 'velocity', 0.0) or 0.0
            fm_1s = getattr(st_1s, 'F_momentum', 0.0) or 0.0

        # ── Sensor fusion: do sensors agree on direction? ──
        # Positive = bullish, negative = bearish
        vel_1s_dir = 1 if vel_1s > 0 else (-1 if vel_1s < 0 else 0)
        vol_1m_dir = 1 if vol_1m > 0 else (-1 if vol_1m < 0 else 0)
        trade_dir = 1 if direction == 'LONG' else -1

        vel_1s_agrees = (vel_1s_dir == trade_dir)
        vol_1m_against = (vol_1m_dir == -trade_dir)  # volume flowing AGAINST trade

        # Unrealized PnL
        if in_trade and price > 0:
            unreal = ((price - entry_price) / 0.25) * trade_dir
        else:
            unreal = np.nan

        rows.append({
            'trade_idx': 0,
            'bar_offset': offset,
            'in_trade': in_trade,
            'at_exit': (i == exit_idx),
            'direction': direction,
            'price': price,
            # 15s peak detection
            'P_at_center': pac,
            'P_near_upper': p_upper,
            'P_near_lower': p_lower,
            'F_momentum_15s': fm,
            'F_momentum_abs_15s': fm_abs,
            'coherence': coh,
            'd_P_center': d_pac,
            'peak_fires': peak_fires,
            'z_score': z,
            'velocity_15s': vel_15s,
            'mean_reversion': reversion,
            # 15s DMI
            'dmi_plus': dmi_plus,
            'dmi_minus': dmi_minus,
            'dmi_diff_15s': dmi_plus - dmi_minus,
            'adx_15s': adx,
            # 1m sensors (institutional)
            'volume_1m': vol_1m,
            'velocity_1m': vel_1m,
            'F_momentum_1m': fm_1m,
            'dmi_diff_1m': dmi_diff_1m,
            # 1s sensors (fast)
            'velocity_1s': vel_1s,
            'F_momentum_1s': fm_1s,
            # Sensor fusion
            'vel_1s_agrees': vel_1s_agrees,
            'vol_1m_against': vol_1m_against,
            'sensors_both_bad': (not vel_1s_agrees and vol_1m_against),
            # Trade
            'unreal_ticks': unreal,
            'actual_pnl': trade['actual_pnl'],
            'exit_reason': trade['exit_reason'],
            'hold_bars': hold,
            'result': trade['result'],
        })

    return rows


def plot_charts(traj, output_path):
    """Generate I-MR charts for multi-TF sensor fusion."""
    in_trade = traj[traj['in_trade'] == True].copy()
    wins = in_trade[in_trade['result'] == 'WIN']
    losses = in_trade[in_trade['result'] == 'LOSS']

    fig = plt.figure(figsize=(18, 28))
    gs = GridSpec(7, 2, figure=fig, hspace=0.4, wspace=0.25)
    fig.suptitle('Peak Trade Sensor Fusion: Winners vs Losers (OOS)\n'
                 '15s peak state + 1m volume + 1s velocity',
                 fontsize=14, fontweight='bold', y=0.99)

    metrics = [
        ('P_at_center', 'P_at_center (probability)', '15s -- peak detection primary'),
        ('F_momentum_abs_15s', '|F_momentum| 15s', '15s -- momentum decay = peak forming'),
        ('d_P_center', 'dP_center/dt (rate of change)', '15s -- leading indicator of peak'),
        ('volume_1m', 'Volume Delta (1m)', '1m -- institutional flow (collapse = peak confirmed)'),
        ('velocity_1s', 'Velocity (1s)', '1s -- fast turn detection'),
        ('dmi_diff_15s', 'DMI Diff (15s)', '15s -- direction confirmation'),
        ('unreal_ticks', 'Unrealized PnL (ticks)', 'Where the money is'),
    ]

    for row, (col, title, subtitle) in enumerate(metrics):
        win_agg = wins.groupby('bar_offset')[col].agg(['mean', 'std', 'count'])
        loss_agg = losses.groupby('bar_offset')[col].agg(['mean', 'std', 'count'])

        # I chart
        ax_i = fig.add_subplot(gs[row, 0])
        for agg, color, label in [(win_agg, 'green', 'WIN'), (loss_agg, 'red', 'LOSS')]:
            offsets = agg.index.values
            ax_i.plot(offsets, agg['mean'], f'{color[0]}-o', markersize=3,
                      label=label, linewidth=1.5)
            ax_i.fill_between(offsets,
                              agg['mean'] - agg['std'],
                              agg['mean'] + agg['std'],
                              alpha=0.12, color=color)
        ax_i.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Entry')
        ax_i.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_i.set_title(f'{title}\n({subtitle})', fontsize=10)
        ax_i.set_xlabel('Bar offset from entry')
        ax_i.legend(fontsize=8)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_xlim(-2, 12)

        # MR chart
        ax_mr = fig.add_subplot(gs[row, 1])
        for agg, color, label in [(win_agg, 'green', 'WIN'), (loss_agg, 'red', 'LOSS')]:
            means = agg['mean'].values
            if len(means) > 1:
                mr = np.abs(np.diff(means))
                mr_offsets = agg.index.values[1:]
                ax_mr.plot(mr_offsets, mr, f'{color[0]}-o', markersize=3,
                           label=f'{label} MR', linewidth=1.5)
                if len(mr) > 2:
                    mr_mean = mr.mean()
                    mr_ucl = 3.267 * mr_mean
                    ax_mr.axhline(y=mr_mean, color=color, linestyle='--', alpha=0.5)
                    ax_mr.axhline(y=mr_ucl, color=color, linestyle=':', alpha=0.3)
        ax_mr.axvline(x=0, color='blue', linestyle='--', alpha=0.5)
        ax_mr.set_title(f'{title} -- Moving Range', fontsize=10)
        ax_mr.set_xlabel('Bar offset from entry')
        ax_mr.legend(fontsize=8)
        ax_mr.grid(True, alpha=0.3)
        ax_mr.set_xlim(-2, 12)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Chart saved: {output_path}")


def write_summary(traj, output_path):
    """Write text summary."""
    in_trade = traj[traj['in_trade'] == True]
    wins = in_trade[in_trade['result'] == 'WIN']
    losses = in_trade[in_trade['result'] == 'LOSS']

    lines = []
    lines.append("=" * 80)
    lines.append("PEAK EXIT SENSOR FUSION ANALYSIS")
    lines.append(f"15s peak state + 1m volume + 1s velocity")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("=" * 80)

    n_trades = in_trade['trade_idx'].nunique()
    n_wins = wins['trade_idx'].nunique()
    n_losses = losses['trade_idx'].nunique()
    lines.append(f"\n  Trades: {n_trades} ({n_wins} W, {n_losses} L)")

    # Sensor fusion agreement
    lines.append(f"\n  SENSOR FUSION (vel_1s agrees with trade AND vol_1m against trade):")
    for offset in range(0, 11):
        w = wins[wins['bar_offset'] == offset]
        l = losses[losses['bar_offset'] == offset]
        if len(w) < 5 or len(l) < 5:
            continue
        w_bad = w['sensors_both_bad'].mean() * 100
        l_bad = l['sensors_both_bad'].mean() * 100
        lines.append(f"    bar +{offset}: WIN both_bad={w_bad:>5.1f}%  "
                     f"LOSS both_bad={l_bad:>5.1f}%  "
                     f"sep={l_bad - w_bad:>+5.1f}pp")

    # Per-metric tables
    for metric, label in [
        ('P_at_center', 'P_at_center'),
        ('F_momentum_abs_15s', '|F_momentum| 15s'),
        ('d_P_center', 'dP_center/dt'),
        ('volume_1m', 'Volume 1m'),
        ('velocity_1s', 'Velocity 1s'),
        ('velocity_1m', 'Velocity 1m'),
        ('dmi_diff_15s', 'DMI Diff 15s'),
        ('dmi_diff_1m', 'DMI Diff 1m'),
        ('F_momentum_1m', 'F_momentum 1m'),
        ('F_momentum_1s', 'F_momentum 1s'),
    ]:
        lines.append(f"\n  {label} BY BAR OFFSET:")
        lines.append(f"  {'Off':>4} | {'W mean':>10} {'W std':>10} | {'L mean':>10} {'L std':>10} | {'Sep':>8}")
        lines.append(f"  {'-'*4} | {'-'*10} {'-'*10} | {'-'*10} {'-'*10} | {'-'*8}")
        for offset in range(0, 11):
            w = wins[wins['bar_offset'] == offset][metric]
            l = losses[losses['bar_offset'] == offset][metric]
            if len(w) < 5 or len(l) < 5:
                continue
            sep = w.mean() - l.mean()
            lines.append(f"  {offset:>4} | {w.mean():>+10.2f} {w.std():>10.2f} | "
                         f"{l.mean():>+10.2f} {l.std():>10.2f} | {sep:>+8.2f}")

    # Volume at entry/exit
    entry = in_trade[in_trade['bar_offset'] == 0]
    exit_b = in_trade[in_trade['at_exit'] == True]
    lines.append(f"\n  1m VOLUME AT ENTRY vs EXIT:")
    for lbl, sub in [('WIN entry', entry[entry['result']=='WIN']),
                      ('LOSS entry', entry[entry['result']=='LOSS']),
                      ('WIN exit', exit_b[exit_b['result']=='WIN']),
                      ('LOSS exit', exit_b[exit_b['result']=='LOSS'])]:
        lines.append(f"    {lbl:<12}: vol_1m={sub['volume_1m'].mean():>+8.1f}  "
                     f"vel_1s={sub['velocity_1s'].mean():>+6.2f}  "
                     f"vel_1m={sub['velocity_1m'].mean():>+6.2f}")

    report = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oos', action='store_true')
    parser.add_argument('--max-trades', type=int, default=1000)
    args = parser.parse_args()

    data_root = os.path.join('DATA', 'ATLAS_OOS') if args.oos else os.path.join('DATA', 'ATLAS')
    trade_path = 'checkpoints/oos_trade_log.csv'

    trades = pd.read_csv(trade_path)
    peak = trades[trades['template_id'] == -100].copy()
    print(f"  {len(peak)} peak trades")

    if len(peak) > args.max_trades:
        peak = peak.sample(n=args.max_trades, random_state=42)
        print(f"  Sampled {args.max_trades}")

    # Load all three TFs
    print(f"\n  Loading states from {data_root}...")
    df_15s, states_15s = load_tf_states(data_root, '15s')
    df_1m, states_1m = load_tf_states(data_root, '1m')
    df_1s, states_1s = load_tf_states(data_root, '1s')

    ts_15s = df_15s['timestamp'].values.astype(np.int64) if df_15s is not None else np.array([])
    ts_1m = df_1m['timestamp'].values.astype(np.int64) if df_1m is not None else np.array([])
    ts_1s = df_1s['timestamp'].values.astype(np.int64) if df_1s is not None else np.array([])

    # Extract trajectories
    all_rows = []
    for ti, (_, trade) in enumerate(tqdm(peak.iterrows(), total=len(peak),
                                          desc="Extracting")):
        rows = extract_trajectory(trade, ts_15s, states_15s, ts_1m, states_1m,
                                  ts_1s, states_1s)
        for r in rows:
            r['trade_idx'] = ti
        all_rows.extend(rows)

    traj = pd.DataFrame(all_rows)
    print(f"  {len(traj)} trajectory rows")

    out_dir = os.path.join('reports', 'findings')
    traj.to_csv(os.path.join(out_dir, 'sensor_fusion_trajectories.csv'), index=False)
    plot_charts(traj, os.path.join(out_dir, 'sensor_fusion_charts.png'))
    write_summary(traj, os.path.join(out_dir, 'sensor_fusion_summary.txt'))


if __name__ == '__main__':
    main()
