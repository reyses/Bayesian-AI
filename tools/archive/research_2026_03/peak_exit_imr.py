"""
Peak Exit I-MR Charts -- regression center, F_momentum, DMI, volume trajectories.

For each peak trade, captures per-bar statistical state and plots I-MR control
charts to visualize when the move exhausts. This is the evidence base for
the stateful peak exit.

Usage:
    python tools/peak_exit_imr.py [--oos] [--max-trades 500]

Output:
    reports/findings/peak_exit_imr_trajectories.csv
    reports/findings/peak_exit_imr_charts.png
    reports/findings/peak_exit_imr_summary.txt
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


def load_states(data_root):
    """Load 15s parquets and compute full MarketState for each bar."""
    tf_dir = os.path.join(data_root, '15s')
    files = sorted(f for f in os.listdir(tf_dir) if f.endswith('.parquet'))

    engine = StatisticalFieldEngine()
    all_states = []
    all_bars = []

    for fname in tqdm(files, desc="Computing states"):
        df = pd.read_parquet(os.path.join(tf_dir, fname))
        if 'timestamp' in df.columns and not np.issubdtype(df['timestamp'].dtype, np.number):
            df['timestamp'] = df['timestamp'].apply(lambda x: x.timestamp())
        raw = engine.batch_compute_states(df)
        for r in raw:
            st = r['state'] if isinstance(r, dict) and 'state' in r else r
            all_states.append(st)
        all_bars.append(df)

    df_all = pd.concat(all_bars, ignore_index=True)
    print(f"  {len(df_all)} bars, {len(all_states)} states")
    return df_all, all_states


def extract_trajectory(trade, timestamps, states, pre=2, post=10):
    """Extract per-bar trajectory for one trade with full state."""
    entry_ts = int(trade['entry_time'])
    hold = int(trade.get('hold_bars', 1))
    direction = trade['direction'].upper()
    entry_price = trade['entry_price']

    entry_idx = int(np.searchsorted(timestamps, entry_ts))
    if entry_idx >= len(states) or entry_idx < 1:
        return []

    exit_idx = min(entry_idx + hold, len(states) - 1)
    start = max(entry_idx - pre, 0)
    end = min(exit_idx + post, len(states) - 1)

    entry_st = states[entry_idx]
    entry_pc = getattr(entry_st, 'P_at_center', 0.0) or 0.0
    entry_fm = abs(getattr(entry_st, 'F_momentum', 0.0) or 0.0)
    entry_rc = getattr(entry_st, 'regression_center', 0.0) or 0.0

    rows = []
    peak_pc = entry_pc  # track best P_at_center since entry

    for i in range(start, end + 1):
        if i >= len(states):
            break
        st = states[i]
        offset = i - entry_idx
        in_trade = (i >= entry_idx and i <= exit_idx)

        # Peak detection uses P_at_center (probability, 0-1) not regression_center (price)
        p_at_center = getattr(st, 'P_at_center', 0.0) or 0.0
        fm = getattr(st, 'F_momentum', 0.0) or 0.0
        fm_abs = abs(fm)
        sigma = getattr(st, 'regression_sigma', 1.0) or 1.0
        rc = getattr(st, 'regression_center', 0.0) or 0.0
        price = getattr(st, 'price', 0.0) or 0.0
        dmi_plus = getattr(st, 'dmi_plus', 0.0) or 0.0
        dmi_minus = getattr(st, 'dmi_minus', 0.0) or 0.0
        adx = getattr(st, 'adx_strength', 0.0) or 0.0
        volume = getattr(st, 'volume_delta', 0.0) or 0.0
        coherence = getattr(st, 'oscillation_entropy_normalized', 0.0) or 0.0
        entropy = getattr(st, 'entropy_normalized', 0.5) or 0.5
        z = getattr(st, 'z_score', 0.0) or 0.0
        velocity = getattr(st, 'velocity', 0.0) or 0.0
        reversion = getattr(st, 'mean_reversion_force', 0.0) or 0.0
        momentum_str = getattr(st, 'momentum_strength', 0.0) or 0.0
        # Probability weights (wave function)
        p_upper = getattr(st, 'P_near_upper', 0.0) or 0.0
        p_lower = getattr(st, 'P_near_lower', 0.0) or 0.0

        # Track peak P_at_center (probability peak since entry)
        if i >= entry_idx:
            peak_pc = max(peak_pc, p_at_center)

        # P_at_center giveback from peak
        pc_giveback = (peak_pc - p_at_center) / max(peak_pc, 0.001) if peak_pc > 0.01 else 0.0

        # RC delta from entry in sigma units
        rc_delta_sigma = (rc - entry_rc) / max(sigma, 0.01)

        # Peak detection: would it fire? (same logic as advance_engine)
        _fires = False
        _peak_dir = ''
        if i > 0 and i < len(states):
            _prev_st = states[i - 1]
            _prev_pac = getattr(_prev_st, 'P_at_center', 0.0) or 0.0
            _prev_fma = abs(getattr(_prev_st, 'F_momentum', 0.0) or 0.0)
            _pc_up = p_at_center > _prev_pac * 1.05 if _prev_pac > 0.01 else False
            _fm_down = fm_abs < _prev_fma * 0.90 if _prev_fma > 0.5 else False
            if (_pc_up or _fm_down) and coherence > 0.55:
                _fires = True
                # Direction: peak fires = state change = reversal
                # High P_at_center = price near band center = potential reversal
                _peak_dir = 'peak'

        # Unrealized PnL
        if in_trade:
            if direction == 'LONG':
                unreal = (price - entry_price) / 0.25
            else:
                unreal = (entry_price - price) / 0.25
        else:
            unreal = np.nan

        rows.append({
            'trade_idx': 0,
            'bar_offset': offset,
            'in_trade': in_trade,
            'at_exit': (i == exit_idx),
            'direction': direction,
            'price': price,
            # Peak detection state (what actually drives entry/exit)
            'P_at_center': p_at_center,
            'P_near_upper': p_upper,
            'P_near_lower': p_lower,
            'F_momentum': fm,
            'F_momentum_abs': fm_abs,
            'coherence': coherence,
            'entropy': entropy,
            'peak_fires': _fires,
            'pc_giveback': pc_giveback,
            'peak_pc': peak_pc,
            # Regression band
            'regression_center': rc,
            'rc_delta_sigma': rc_delta_sigma,
            'regression_sigma': sigma,
            'z_score': z,
            # DMI
            'dmi_plus': dmi_plus,
            'dmi_minus': dmi_minus,
            'dmi_diff': dmi_plus - dmi_minus,
            'adx': adx,
            # Volume
            'volume_delta': volume,
            # Dynamics
            'velocity': velocity,
            'mean_reversion': reversion,
            'momentum_strength': momentum_str,
            # Trade
            'unreal_ticks': unreal,
            'actual_pnl': trade['actual_pnl'],
            'exit_reason': trade['exit_reason'],
            'hold_bars': hold,
            'result': trade['result'],
        })

    return rows


def compute_imr(series):
    """Compute I-MR (Individual-Moving Range) control limits."""
    vals = series.dropna().values
    if len(vals) < 3:
        return {'mean': np.nan, 'ucl': np.nan, 'lcl': np.nan,
                'mr_mean': np.nan, 'mr_ucl': np.nan}
    mr = np.abs(np.diff(vals))
    mr_mean = mr.mean()
    x_mean = vals.mean()
    # d2 = 1.128 for n=2 (standard I-MR constant)
    ucl = x_mean + 2.66 * mr_mean
    lcl = x_mean - 2.66 * mr_mean
    mr_ucl = 3.267 * mr_mean
    return {'mean': x_mean, 'ucl': ucl, 'lcl': lcl,
            'mr_mean': mr_mean, 'mr_ucl': mr_ucl}


def plot_imr_charts(traj, output_path):
    """Generate I-MR charts for peak trade trajectories."""
    in_trade = traj[traj['in_trade'] == True].copy()

    # Average trajectory by bar offset and result
    wins = in_trade[in_trade['result'] == 'WIN']
    losses = in_trade[in_trade['result'] == 'LOSS']

    fig = plt.figure(figsize=(18, 24))
    gs = GridSpec(6, 2, figure=fig, hspace=0.35, wspace=0.25)
    fig.suptitle('Peak Trade Trajectories: Winners vs Losers (OOS)',
                 fontsize=14, fontweight='bold', y=0.98)

    metrics = [
        ('P_at_center', 'P_at_center (probability at band center)', 'Peak detection primary signal -- high = near center'),
        ('F_momentum_abs', '|F_momentum| (absolute)', 'Peak detection secondary -- decay = peak forming'),
        ('coherence', 'Coherence (oscillation entropy)', 'Peak detection gate -- must be > 0.55'),
        ('dmi_diff', 'DMI Diff (DI+ minus DI-)', 'Direction signal -- positive = bullish'),
        ('volume_delta', 'Volume Delta', 'Volume flow -- collapse after peak = exit signal'),
        ('unreal_ticks', 'Unrealized PnL (ticks)', 'Where the money is'),
    ]

    for row, (col, title, subtitle) in enumerate(metrics):
        # Aggregate by bar_offset
        win_agg = wins.groupby('bar_offset')[col].agg(['mean', 'std', 'count'])
        loss_agg = losses.groupby('bar_offset')[col].agg(['mean', 'std', 'count'])

        # I chart (left)
        ax_i = fig.add_subplot(gs[row, 0])
        offsets_w = win_agg.index.values
        offsets_l = loss_agg.index.values

        ax_i.plot(offsets_w, win_agg['mean'], 'g-o', markersize=3, label='WIN', linewidth=1.5)
        ax_i.fill_between(offsets_w,
                          win_agg['mean'] - win_agg['std'],
                          win_agg['mean'] + win_agg['std'],
                          alpha=0.15, color='green')
        ax_i.plot(offsets_l, loss_agg['mean'], 'r-o', markersize=3, label='LOSS', linewidth=1.5)
        ax_i.fill_between(offsets_l,
                          loss_agg['mean'] - loss_agg['std'],
                          loss_agg['mean'] + loss_agg['std'],
                          alpha=0.15, color='red')

        ax_i.axvline(x=0, color='blue', linestyle='--', alpha=0.5, label='Entry')
        ax_i.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax_i.set_title(f'{title}\n({subtitle})', fontsize=10)
        ax_i.set_xlabel('Bar offset from entry')
        ax_i.legend(fontsize=8)
        ax_i.grid(True, alpha=0.3)
        ax_i.set_xlim(-2, 12)

        # MR chart (right) -- moving range of the metric
        ax_mr = fig.add_subplot(gs[row, 1])

        for lbl, agg, color in [('WIN', win_agg, 'green'), ('LOSS', loss_agg, 'red')]:
            means = agg['mean'].values
            if len(means) > 1:
                mr = np.abs(np.diff(means))
                mr_offsets = agg.index.values[1:]
                ax_mr.plot(mr_offsets, mr, f'{color[0]}-o', markersize=3,
                           label=f'{lbl} MR', linewidth=1.5)
                # MR control limit
                if len(mr) > 2:
                    mr_mean = mr.mean()
                    mr_ucl = 3.267 * mr_mean
                    ax_mr.axhline(y=mr_mean, color=color, linestyle='--', alpha=0.5)
                    ax_mr.axhline(y=mr_ucl, color=color, linestyle=':', alpha=0.3,
                                  label=f'{lbl} UCL')

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
    """Write text summary with I-MR statistics."""
    in_trade = traj[traj['in_trade'] == True]
    wins = in_trade[in_trade['result'] == 'WIN']
    losses = in_trade[in_trade['result'] == 'LOSS']

    lines = []
    lines.append("=" * 80)
    lines.append("PEAK EXIT I-MR ANALYSIS")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("=" * 80)

    n_trades = in_trade['trade_idx'].nunique()
    n_wins = wins['trade_idx'].nunique()
    n_losses = losses['trade_idx'].nunique()
    lines.append(f"\n  Trades: {n_trades} ({n_wins} wins, {n_losses} losses)")

    # Per-bar-offset stats for key metrics
    for metric, label in [
        ('P_at_center', 'P_at_center (probability)'),
        ('F_momentum_abs', '|F_momentum|'),
        ('coherence', 'Coherence'),
        ('dmi_diff', 'DMI Diff'),
        ('volume_delta', 'Volume Delta'),
        ('unreal_ticks', 'Unrealized PnL'),
    ]:
        lines.append(f"\n  {label} BY BAR OFFSET:")
        lines.append(f"  {'Offset':>6} | {'WIN mean':>10} {'WIN std':>10} | {'LOSS mean':>10} {'LOSS std':>10} | {'Separation':>10}")
        lines.append(f"  {'-'*6} | {'-'*10} {'-'*10} | {'-'*10} {'-'*10} | {'-'*10}")
        for offset in range(0, 11):
            w = wins[wins['bar_offset'] == offset][metric]
            l = losses[losses['bar_offset'] == offset][metric]
            if len(w) < 5 or len(l) < 5:
                continue
            sep = w.mean() - l.mean()
            lines.append(f"  {offset:>6} | {w.mean():>+10.2f} {w.std():>10.2f} | {l.mean():>+10.2f} {l.std():>10.2f} | {sep:>+10.2f}")

    # DMI at entry vs exit
    entry_bars = in_trade[in_trade['bar_offset'] == 0]
    exit_bars = in_trade[in_trade['at_exit'] == True]

    lines.append(f"\n  DMI AT ENTRY:")
    for lbl, sub in [('WIN', entry_bars[entry_bars['result'] == 'WIN']),
                      ('LOSS', entry_bars[entry_bars['result'] == 'LOSS'])]:
        lines.append(f"    {lbl}: dmi_diff={sub['dmi_diff'].mean():>+.2f}  "
                     f"adx={sub['adx'].mean():.1f}  "
                     f"DI+={sub['dmi_plus'].mean():.1f}  DI-={sub['dmi_minus'].mean():.1f}")

    lines.append(f"\n  DMI AT EXIT:")
    for lbl, sub in [('WIN', exit_bars[exit_bars['result'] == 'WIN']),
                      ('LOSS', exit_bars[exit_bars['result'] == 'LOSS'])]:
        lines.append(f"    {lbl}: dmi_diff={sub['dmi_diff'].mean():>+.2f}  "
                     f"adx={sub['adx'].mean():.1f}  "
                     f"DI+={sub['dmi_plus'].mean():.1f}  DI-={sub['dmi_minus'].mean():.1f}")

    # Volume at entry vs exit
    lines.append(f"\n  VOLUME AT ENTRY:")
    for lbl, sub in [('WIN', entry_bars[entry_bars['result'] == 'WIN']),
                      ('LOSS', entry_bars[entry_bars['result'] == 'LOSS'])]:
        lines.append(f"    {lbl}: vol_delta={sub['volume_delta'].mean():>+.1f}")

    lines.append(f"\n  VOLUME AT EXIT:")
    for lbl, sub in [('WIN', exit_bars[exit_bars['result'] == 'WIN']),
                      ('LOSS', exit_bars[exit_bars['result'] == 'LOSS'])]:
        lines.append(f"    {lbl}: vol_delta={sub['volume_delta'].mean():>+.1f}")

    # I-MR control limits for F_momentum
    lines.append(f"\n  I-MR CONTROL LIMITS (F_momentum, in-trade bars):")
    for lbl, sub in [('WIN', wins), ('LOSS', losses)]:
        fm_vals = sub.groupby('trade_idx')['F_momentum'].apply(list)
        all_fm = []
        for tlist in fm_vals:
            all_fm.extend(tlist)
        imr = compute_imr(pd.Series(all_fm))
        lines.append(f"    {lbl}: mean={imr['mean']:>+.1f}  UCL={imr['ucl']:>+.1f}  "
                     f"LCL={imr['lcl']:>+.1f}  MR_mean={imr['mr_mean']:.1f}")

    report = '\n'.join(lines) + '\n'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(report)
    print(f"\n  Report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Peak exit I-MR charts")
    parser.add_argument('--data', default=os.path.join('DATA', 'ATLAS'))
    parser.add_argument('--oos', action='store_true')
    parser.add_argument('--max-trades', type=int, default=2000)
    args = parser.parse_args()

    if args.oos:
        trade_path = 'checkpoints/oos_trade_log.csv'
        data_root = os.path.join('DATA', 'ATLAS_OOS')
    else:
        trade_path = 'checkpoints/oos_trade_log.csv'
        data_root = args.data

    trades = pd.read_csv(trade_path)
    peak = trades[trades['template_id'] == -100].copy()
    print(f"  {len(peak)} peak trades from {trade_path}")

    if len(peak) > args.max_trades:
        peak = peak.sample(n=args.max_trades, random_state=42)
        print(f"  Sampled {args.max_trades}")

    df_bars, states = load_states(data_root if args.oos else os.path.join('DATA', 'ATLAS_OOS'))
    timestamps = df_bars['timestamp'].values.astype(np.int64)

    all_rows = []
    for ti, (_, trade) in enumerate(tqdm(peak.iterrows(), total=len(peak),
                                          desc="Extracting trajectories")):
        rows = extract_trajectory(trade, timestamps, states, pre=2, post=10)
        for r in rows:
            r['trade_idx'] = ti
        all_rows.extend(rows)

    traj = pd.DataFrame(all_rows)
    print(f"  {len(traj)} trajectory rows")

    # Save CSV
    out_dir = os.path.join('reports', 'findings')
    traj.to_csv(os.path.join(out_dir, 'peak_exit_imr_trajectories.csv'), index=False)

    # Charts
    plot_imr_charts(traj, os.path.join(out_dir, 'peak_exit_imr_charts.png'))

    # Summary
    write_summary(traj, os.path.join(out_dir, 'peak_exit_imr_summary.txt'))


if __name__ == '__main__':
    main()
