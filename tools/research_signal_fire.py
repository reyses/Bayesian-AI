#!/usr/bin/env python
"""
Signal Fire Rate Research — gate cascade analysis and loosening experiments.

Answers: "If the gates were looser, how many more trades would the trainer try?"

Data sources:
  - signal_log.csv     — every candidate evaluated (traded + blocked)
  - fn_log.csv         — missed real moves (oracle says profitable, we skipped)
  - trade_log.csv      — what we actually traded

Usage:
  python tools/research_signal_fire.py                  # OOS (default)
  python tools/research_signal_fire.py --is             # IS
  python tools/research_signal_fire.py --save           # save report
  python tools/research_signal_fire.py --scenario all   # simulate all gates open
"""
import argparse
import os
import sys
import glob

import numpy as np
import pandas as pd

# --- Gate metadata -----------------------------------------------------------------------
GATE_ORDER = [
    'gate0_noise',
    'gate0_r3_snap', 'gate0_r3_struct',
    'gate0_r4_nightmare', 'gate0_r4_struct',
    'gate0_hurst', 'gate0_momentum', 'gate0_tunnel',
    'gate0_5',
    'gate1',
    'gate2',
    'gate3',
    'gate4_momentum_align', 'gate4_confidence',
    'score_loser',
    'traded',
]

GATE_LABELS = {
    'gate0_noise':          'G0  Noise zone (|z| < 0.5)',
    'gate0_r3_snap':        'G0  Approach: BAND_REVERSAL no tmpl',
    'gate0_r3_struct':      'G0  Approach: MOMENTUM_BREAK weak',
    'gate0_r4_nightmare':   'G0  Extreme: nightmare field',
    'gate0_r4_struct':      'G0  Extreme: momentum break',
    'gate0_hurst':          'G0  Physics: Hurst < 0.5',
    'gate0_momentum':       'G0  Physics: momentum override',
    'gate0_tunnel':         'G0  Physics: reversion prob < 40%',
    'gate0_5':              'G0.5  Depth filter (depth<3 / blacklist)',
    'gate1':                'G1  Template match (dist > 4.5)',
    'gate2':                'G2  Brain reject (unprofitable)',
    'gate3':                'G3  Low conviction',
    'gate4_momentum_align': 'G4  Momentum misalign',
    'gate4_confidence':     'G4  Confidence too low',
    'score_loser':          'Competition: lost to better candidate',
    'traded':               'TRADED',
}

# Which gates are "looseable" (vs structural like score_loser)
LOOSEABLE_GATES = [
    'gate0_noise', 'gate0_r3_snap', 'gate0_r3_struct',
    'gate0_r4_nightmare', 'gate0_r4_struct',
    'gate0_hurst', 'gate0_momentum', 'gate0_tunnel',
    'gate0_5', 'gate1', 'gate2', 'gate3',
    'gate4_momentum_align', 'gate4_confidence',
]


def load_data(mode='oos'):
    """Load signal log, FN log, and trade log.

    Signal log + FN log: reports/{mode}/ (single file or shards/ subdir).
    Trade log: checkpoints/ (consumed by analytics suite).
    """
    _mode_dir = os.path.join('reports', mode)
    tl_path = os.path.join('checkpoints',
                           'oos_trade_log.csv' if mode in ('oos', 'oos2') else 'oracle_trade_log.csv')
    if mode == 'is' and not os.path.exists(tl_path):
        tl_path = 'checkpoints/oracle_trade_log_old.csv'

    data = {}
    # Signal log + FN log: try single file, then shards
    for name, base_name in [('signal', 'signal_log.csv'), ('fn', 'fn_oracle_log.csv')]:
        single = os.path.join(_mode_dir, base_name)
        shard_pattern = os.path.join(_mode_dir, 'shards',
                                     base_name.replace('.csv', '_*.csv'))
        shards = sorted(glob.glob(shard_pattern))
        if shards:
            data[name] = pd.concat([pd.read_csv(s, low_memory=False) for s in shards])
            print(f"  Loaded {name}: {len(data[name]):,} rows from {len(shards)} shards")
        elif os.path.exists(single):
            data[name] = pd.read_csv(single, low_memory=False)
            print(f"  Loaded {name}: {len(data[name]):,} rows from {single}")
        else:
            print(f"  WARNING: no {base_name} found in {_mode_dir}")
            data[name] = pd.DataFrame()

    # Trade log: checkpoints/
    if os.path.exists(tl_path):
        data['trade'] = pd.read_csv(tl_path, low_memory=False)
        print(f"  Loaded trade: {len(data['trade']):,} rows from {tl_path}")
    else:
        print(f"  WARNING: {tl_path} not found")
        data['trade'] = pd.DataFrame()

    return data.get('signal', pd.DataFrame()), data.get('fn', pd.DataFrame()), data.get('trade', pd.DataFrame())


def gate_funnel(sl, fn, tl):
    """Section 1: Full gate cascade funnel."""
    lines = []
    lines.append("=" * 90)
    lines.append("1. GATE CASCADE FUNNEL")
    lines.append("=" * 90)

    # Signal log breakdown
    if not sl.empty and 'gate' in sl.columns:
        total = len(sl)
        traded = (sl.gate == 'traded').sum()
        blocked = total - traded

        lines.append(f"\n  Total candidates evaluated:  {total:>9,}")
        lines.append(f"  Traded:                      {traded:>9,}  ({traded/total*100:.2f}%)")
        lines.append(f"  Blocked:                     {blocked:>9,}  ({blocked/total*100:.2f}%)")

        lines.append(f"\n  {'Gate':<45} {'Count':>8} {'%Total':>8} {'OraclePnL':>14} {'%Real':>7}")
        lines.append(f"  {'-'*45} {'-'*8} {'-'*8} {'-'*14} {'-'*7}")

        for gate in GATE_ORDER:
            sub = sl[sl.gate == gate]
            if len(sub) == 0:
                continue
            n = len(sub)
            pct = n / total * 100
            opnl = pd.to_numeric(sub.get('oracle_pnl', 0), errors='coerce').sum()
            # % real moves (non-noise)
            if 'oracle_label' in sub.columns:
                real = sub[sub.oracle_label.astype(str) != 'NOISE']
                real_pct = len(real) / n * 100 if n else 0
            else:
                real_pct = 0
            label = GATE_LABELS.get(gate, gate)
            lines.append(f"  {label:<45} {n:>8,} {pct:>7.1f}% ${opnl:>12,.0f} {real_pct:>6.1f}%")

        # Unknown gates
        known = set(GATE_ORDER)
        unknown = sl[~sl.gate.isin(known)]
        if len(unknown) > 0:
            for g in unknown.gate.unique():
                sub = unknown[unknown.gate == g]
                lines.append(f"  {str(g):<45} {len(sub):>8,}")

    # FN log breakdown (missed real moves)
    if not fn.empty and 'gate_blocked' in fn.columns:
        lines.append(f"\n  --MISSED REAL MOVES (FN oracle log) --")
        total_fn = len(fn)
        fn_pnl = fn['fn_potential_pnl'].sum() if 'fn_potential_pnl' in fn.columns else 0
        lines.append(f"  Total missed real moves: {total_fn:,}  worth ${fn_pnl:,.0f}")
        lines.append(f"\n  {'Gate that blocked':>40} {'Count':>8} {'%FN':>7} {'Missed$':>14}")
        lines.append(f"  {'-'*40} {'-'*8} {'-'*7} {'-'*14}")

        for gate in GATE_ORDER + ['unknown']:
            sub = fn[fn.gate_blocked == gate]
            if len(sub) == 0:
                continue
            n = len(sub)
            pct = n / total_fn * 100
            mpnl = sub['fn_potential_pnl'].sum() if 'fn_potential_pnl' in sub.columns else 0
            label = GATE_LABELS.get(gate, gate)
            lines.append(f"  {label:>40} {n:>8,} {pct:>6.1f}% ${mpnl:>12,.0f}")

    # Trade log summary
    if not tl.empty:
        n_trades = len(tl)
        total_pnl = tl['actual_pnl'].sum() if 'actual_pnl' in tl.columns else 0
        avg_pnl = total_pnl / n_trades if n_trades else 0
        lines.append(f"\n  --ACTUAL TRADES --")
        lines.append(f"  Trades: {n_trades:,}  PnL: ${total_pnl:,.2f}  Avg: ${avg_pnl:.2f}/trade")

    return lines


def loosening_scenarios(sl, fn):
    """Section 2: What if we loosened each gate?"""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("2. GATE LOOSENING SCENARIOS")
    lines.append("   'If we removed gate X, how many more signals would fire?'")
    lines.append("=" * 90)

    if sl.empty and fn.empty:
        lines.append("  No data available")
        return lines

    # For each looseable gate, show what we'd gain
    lines.append(f"\n  {'Gate removed':<42} {'Extra signals':>13} {'Extra real':>11} "
                 f"{'Oracle$':>12} {'Avg Oracle$':>12} {'%Noise':>7}")
    lines.append(f"  {'-'*42} {'-'*13} {'-'*11} {'-'*12} {'-'*12} {'-'*7}")

    scenario_data = []

    for gate in LOOSEABLE_GATES:
        # From signal log: signals blocked by this gate
        sl_extra = sl[sl.gate == gate] if not sl.empty and 'gate' in sl.columns else pd.DataFrame()
        # From FN log: real moves blocked by this gate
        fn_extra = fn[fn.gate_blocked == gate] if not fn.empty and 'gate_blocked' in fn.columns else pd.DataFrame()

        n_sl = len(sl_extra)
        n_fn = len(fn_extra)
        n_total = n_sl + n_fn

        if n_total == 0:
            continue

        # Oracle PnL from signal log
        sl_opnl = pd.to_numeric(sl_extra.get('oracle_pnl', 0), errors='coerce').sum() if n_sl else 0
        fn_opnl = fn_extra['fn_potential_pnl'].sum() if n_fn and 'fn_potential_pnl' in fn_extra.columns else 0
        total_opnl = sl_opnl + fn_opnl

        # Real move count from signal log
        if n_sl and 'oracle_label' in sl_extra.columns:
            sl_real = len(sl_extra[sl_extra.oracle_label.astype(str) != 'NOISE'])
        else:
            sl_real = 0
        n_real = sl_real + n_fn  # FN are all real moves by definition
        n_noise = n_total - n_real
        noise_pct = n_noise / n_total * 100 if n_total else 0
        avg_opnl = total_opnl / n_real if n_real else 0

        label = GATE_LABELS.get(gate, gate)
        lines.append(f"  {label:<42} {n_total:>13,} {n_real:>11,} "
                     f"${total_opnl:>11,.0f} ${avg_opnl:>11,.0f} {noise_pct:>6.1f}%")

        scenario_data.append({
            'gate': gate, 'label': label,
            'extra': n_total, 'real': n_real, 'noise': n_noise,
            'oracle_pnl': total_opnl, 'avg_oracle': avg_opnl,
        })

    # Cumulative: open gates from most impactful to least
    lines.append(f"\n  --CUMULATIVE LOOSENING (ordered by oracle $ recovered) --")
    lines.append(f"  {'Open gate':<42} {'Cumul signals':>14} {'Cumul real':>11} {'Cumul Oracle$':>14}")
    lines.append(f"  {'-'*42} {'-'*14} {'-'*11} {'-'*14}")

    scenario_data.sort(key=lambda x: -x['oracle_pnl'])
    cumul_signals = 0
    cumul_real = 0
    cumul_pnl = 0.0
    for s in scenario_data:
        cumul_signals += s['extra']
        cumul_real += s['real']
        cumul_pnl += s['oracle_pnl']
        lines.append(f"  + {s['label']:<40} {cumul_signals:>14,} {cumul_real:>11,} ${cumul_pnl:>13,.0f}")

    return lines


def fn_deep_dive(fn):
    """Section 3: Deep dive into missed real moves — what would we gain?"""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("3. MISSED REAL MOVES — DEEP DIVE")
    lines.append("   'Of the profitable signals we missed, what do they look like?'")
    lines.append("=" * 90)

    if fn.empty:
        lines.append("  No FN data available")
        return lines

    total = len(fn)
    total_pnl = fn['fn_potential_pnl'].sum()

    # By depth
    if 'depth' in fn.columns:
        lines.append(f"\n  --BY DEPTH --")
        lines.append(f"  {'Depth':>6} {'Count':>8} {'%Total':>8} {'Avg Potential$':>15} {'Total$':>14}")
        for d in sorted(fn['depth'].unique()):
            sub = fn[fn.depth == d]
            n = len(sub)
            avg = sub['fn_potential_pnl'].mean()
            tot = sub['fn_potential_pnl'].sum()
            lines.append(f"  {d:>6} {n:>8,} {n/total*100:>7.1f}% ${avg:>14,.2f} ${tot:>13,.0f}")

    # By oracle label
    if 'oracle_label_name' in fn.columns:
        lines.append(f"\n  --BY ORACLE LABEL --")
        for label in fn['oracle_label_name'].unique():
            sub = fn[fn.oracle_label_name == label]
            n = len(sub)
            avg = sub['fn_potential_pnl'].mean()
            tot = sub['fn_potential_pnl'].sum()
            lines.append(f"  {str(label):<18} n={n:>6,}  avg=${avg:>8,.2f}  total=${tot:>12,.0f}")

    # Physics characteristics of FN signals
    physics_cols = ['hurst', 'tunnel_prob', 'velocity', 'mom_rev_ratio']
    available = [c for c in physics_cols if c in fn.columns]
    if available:
        lines.append(f"\n  --PHYSICS PROFILE OF MISSED SIGNALS --")
        lines.append(f"  (Compare to traded signals to understand what the gates are rejecting)")
        lines.append(f"  {'Feature':<20} {'Mean':>10} {'Median':>10} {'P25':>10} {'P75':>10}")
        lines.append(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
        for col in available:
            vals = pd.to_numeric(fn[col], errors='coerce').dropna()
            if len(vals) == 0:
                continue
            lines.append(f"  {col:<20} {vals.mean():>10.3f} {vals.median():>10.3f} "
                         f"{vals.quantile(0.25):>10.3f} {vals.quantile(0.75):>10.3f}")

    # Top-impact gate: if we opened just the #1 bottleneck, what trades would we add?
    if 'gate_blocked' in fn.columns:
        top_gate = fn['gate_blocked'].value_counts().index[0]
        top_sub = fn[fn.gate_blocked == top_gate]
        lines.append(f"\n  --TOP BOTTLENECK: {GATE_LABELS.get(top_gate, top_gate)} --")
        lines.append(f"  Missed: {len(top_sub):,} real moves  worth ${top_sub['fn_potential_pnl'].sum():,.0f}")
        lines.append(f"  Avg potential per signal: ${top_sub['fn_potential_pnl'].mean():.2f}")

        # What depths are these?
        if 'depth' in top_sub.columns:
            lines.append(f"  Depth distribution: {dict(top_sub['depth'].value_counts().sort_index())}")

    return lines


def fire_rate_over_time(sl, tl):
    """Section 4: Fire rate over time + theoretical comparison."""
    from datetime import datetime

    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("4. FIRE RATE vs THEORETICAL CAPACITY")
    lines.append("=" * 90)

    # Derive day from entry_time (unix timestamp) or 'day' column
    if tl.empty:
        lines.append("  No trade data available")
        return lines

    if 'day' in tl.columns:
        tl = tl.copy()
        tl['_day'] = tl['day'].astype(str)
    elif 'entry_time' in tl.columns:
        tl = tl.copy()
        tl['_day'] = pd.to_datetime(tl['entry_time'], unit='s').dt.strftime('%Y-%m-%d')
    else:
        lines.append("  No time data available")
        return lines

    # Daily trades
    daily = tl.groupby('_day').agg(
        trades=('actual_pnl', 'count'),
        pnl=('actual_pnl', 'sum'),
    ).reset_index()

    n_days = len(daily)
    avg_daily = daily['trades'].mean()
    total_trades = daily['trades'].sum()

    # Theoretical: 8-min peak oscillation, 22h session (CME futures 6PM-4PM)
    SESSION_HOURS = 22.0
    PEAK_INTERVAL_MIN = 8.0
    theoretical_per_day = SESSION_HOURS * 60 / PEAK_INTERVAL_MIN
    fire_pct = avg_daily / theoretical_per_day * 100

    lines.append(f"\n  --THEORETICAL CAPACITY --")
    lines.append(f"  Peak oscillation period:  {PEAK_INTERVAL_MIN:.0f} min")
    lines.append(f"  Session length:           {SESSION_HOURS:.0f} hours")
    lines.append(f"  Theoretical trades/day:   {theoretical_per_day:.0f}")
    lines.append(f"")
    lines.append(f"  --ACTUAL FIRE RATE --")
    lines.append(f"  Trading days:             {n_days}")
    lines.append(f"  Total trades:             {total_trades:,}")
    lines.append(f"  Avg trades/day:           {avg_daily:.1f}")
    lines.append(f"  Fire rate:                {fire_pct:.1f}% of theoretical capacity")
    lines.append(f"  Gap:                      {theoretical_per_day - avg_daily:.0f} trades/day left on table")

    # Daily breakdown
    lines.append(f"\n  {'Day':<12} {'Trades':>7} {'PnL':>12} {'vs Theory':>10}")
    lines.append(f"  {'-'*12} {'-'*7} {'-'*12} {'-'*10}")
    for _, row in daily.iterrows():
        pct = row['trades'] / theoretical_per_day * 100
        lines.append(f"  {str(row['_day']):<12} {row['trades']:>7,} ${row['pnl']:>10,.2f} {pct:>9.1f}%")

    # Signal log: candidates evaluated per day
    if not sl.empty:
        if 'day' in sl.columns:
            sl_daily = sl.groupby('day').size()
        elif 'ts' in sl.columns:
            sl = sl.copy()
            sl['_day'] = pd.to_datetime(pd.to_numeric(sl['ts'], errors='coerce'), unit='s').dt.strftime('%Y-%m-%d')
            sl_daily = sl.groupby('_day').size()
        else:
            sl_daily = pd.Series(dtype=int)

        if len(sl_daily) > 0:
            lines.append(f"\n  --SIGNAL VOLUME --")
            lines.append(f"  Avg candidates evaluated/day: {sl_daily.mean():,.0f}")
            lines.append(f"  Conversion rate (traded/evaluated): {avg_daily / sl_daily.mean() * 100:.3f}%")

    return lines


def slot_utilization(tl):
    """Section 5: How much time are we spending in-trade vs available?"""
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("5. SLOT UTILIZATION -- Time Spent In Trade")
    lines.append("=" * 90)

    if tl.empty or 'entry_time' not in tl.columns or 'exit_time' not in tl.columns:
        lines.append("  No entry/exit time data available")
        return lines

    tl = tl.copy()
    tl['_day'] = pd.to_datetime(tl['entry_time'], unit='s').dt.strftime('%Y-%m-%d')

    # Session = 22 hours = 79,200 seconds = 5,280 bars at 15s
    SESSION_SECS = 22.0 * 3600
    SESSION_BARS = int(SESSION_SECS / 15)
    PEAK_INTERVAL_MIN = 8.0
    THEORETICAL_PER_DAY = SESSION_SECS / 60 / PEAK_INTERVAL_MIN

    daily_stats = []
    for day, grp in tl.groupby('_day'):
        n_trades = len(grp)
        total_hold_bars = grp['hold_bars'].sum()
        total_hold_secs = total_hold_bars * 15  # 15s bars
        total_hold_min = total_hold_secs / 60
        pct_in_trade = total_hold_bars / SESSION_BARS * 100

        # Time available (not in trade)
        avail_bars = SESSION_BARS - total_hold_bars
        avail_min = avail_bars * 15 / 60

        # How many theoretical peaks occur while we're in trade?
        peaks_missed_slot = total_hold_min / PEAK_INTERVAL_MIN

        daily_stats.append({
            'day': day,
            'trades': n_trades,
            'hold_bars': int(total_hold_bars),
            'hold_min': total_hold_min,
            'pct_in_trade': pct_in_trade,
            'avail_min': avail_min,
            'peaks_missed': peaks_missed_slot,
        })

    ds = pd.DataFrame(daily_stats)

    lines.append(f"\n  Session: {SESSION_SECS/3600:.0f}h = {SESSION_BARS:,} bars (15s)")
    lines.append(f"")
    lines.append(f"  -- AVERAGES ACROSS {len(ds)} TRADING DAYS --")
    lines.append(f"  Avg time in trade:     {ds['hold_min'].mean():>7.1f} min/day  ({ds['pct_in_trade'].mean():.1f}%)")
    lines.append(f"  Avg time available:    {ds['avail_min'].mean():>7.1f} min/day  ({100 - ds['pct_in_trade'].mean():.1f}%)")
    lines.append(f"  Avg hold per trade:    {ds['hold_bars'].sum() / ds['trades'].sum():.1f} bars = {ds['hold_bars'].sum() / ds['trades'].sum() * 15 / 60:.1f} min")
    lines.append(f"  Avg peaks missed (slot): {ds['peaks_missed'].mean():.1f} / {THEORETICAL_PER_DAY:.0f} theoretical")
    lines.append(f"")

    # Breakdown of the theoretical gap
    avg_traded = ds['trades'].mean()
    avg_slot_missed = ds['peaks_missed'].mean()
    gate_and_detection_missed = THEORETICAL_PER_DAY - avg_traded - avg_slot_missed
    lines.append(f"  -- THEORETICAL GAP BREAKDOWN --")
    lines.append(f"  Theoretical capacity:        {THEORETICAL_PER_DAY:>6.0f} trades/day")
    lines.append(f"  Actually traded:             {avg_traded:>6.1f}  ({avg_traded/THEORETICAL_PER_DAY*100:.1f}%)")
    lines.append(f"  Lost to slot blocking:       {avg_slot_missed:>6.1f}  ({avg_slot_missed/THEORETICAL_PER_DAY*100:.1f}%)")
    lines.append(f"  Lost to gates/no detection:  {gate_and_detection_missed:>6.1f}  ({gate_and_detection_missed/THEORETICAL_PER_DAY*100:.1f}%)")
    lines.append(f"")

    # Hold time distribution
    all_hold = tl['hold_bars'].values
    lines.append(f"  -- HOLD TIME DISTRIBUTION --")
    for label, lo, hi in [('< 1 min', 0, 4), ('1-5 min', 4, 20), ('5-15 min', 20, 60),
                           ('15-60 min', 60, 240), ('> 60 min', 240, 999999)]:
        n = int(((all_hold >= lo) & (all_hold < hi)).sum())
        pct = n / len(all_hold) * 100
        bar = '#' * min(40, int(pct / 2))
        lines.append(f"    {label:<10} {n:>6,}  ({pct:>5.1f}%)  {bar}")

    # Worst slot hogs (days with highest % in trade)
    top5 = ds.nlargest(5, 'pct_in_trade')
    lines.append(f"\n  -- TOP 5 BUSIEST DAYS --")
    lines.append(f"  {'Day':<12} {'Trades':>7} {'In-Trade':>10} {'% Session':>10}")
    lines.append(f"  {'-'*12} {'-'*7} {'-'*10} {'-'*10}")
    for _, row in top5.iterrows():
        lines.append(f"  {row['day']:<12} {row['trades']:>7} {row['hold_min']:>8.1f}m {row['pct_in_trade']:>9.1f}%")

    return lines


def capacity_decomposition(sl, fn, tl):
    """Section 6: Capacity Utilization Decomposition.

    Multiplicative decomposition of throughput gap:

      Actual/Theoretical = P(available) x P(detected | available) x P(traded | detected)

    Each conditional probability isolates one bottleneck:
      P(available)            = 1 - (time_in_trade / session_time)
      P(detected | available) = signal_bars / available_peaks
      P(traded | detected)    = trades / signal_bars
    """
    lines = []
    lines.append("\n" + "=" * 90)
    lines.append("6. CAPACITY UTILIZATION DECOMPOSITION")
    lines.append("=" * 90)

    if tl.empty or 'entry_time' not in tl.columns:
        lines.append("  No trade data available")
        return lines

    tl = tl.copy()
    tl['_day'] = pd.to_datetime(tl['entry_time'], unit='s').dt.strftime('%Y-%m-%d')
    n_days = tl['_day'].nunique()

    SESSION_SECS = 22.0 * 3600
    SESSION_BARS = int(SESSION_SECS / 15)
    PEAK_INTERVAL_MIN = 8.0
    THEORETICAL_PER_DAY = SESSION_SECS / 60 / PEAK_INTERVAL_MIN

    # --- P(available): fraction of session NOT slot-blocked ---
    avg_hold_bars_per_day = tl.groupby('_day')['hold_bars'].sum().mean()
    p_available = (SESSION_BARS - avg_hold_bars_per_day) / SESSION_BARS

    # --- P(detected | available): detection rate given open slot ---
    bars_with_signal = 0
    p_detected = 0
    if not sl.empty:
        if 'day' in sl.columns:
            sl_copy = sl.copy()
            sl_copy['_day'] = sl_copy['day'].astype(str)
        elif 'ts' in sl.columns:
            sl_copy = sl.copy()
            sl_copy['_day'] = pd.to_datetime(pd.to_numeric(sl_copy['ts'], errors='coerce'),
                                              unit='s').dt.strftime('%Y-%m-%d')
        else:
            sl_copy = pd.DataFrame()

        if not sl_copy.empty and '_day' in sl_copy.columns:
            bars_with_signal = sl_copy.groupby('_day')['ts'].nunique().mean()
            available_peaks = THEORETICAL_PER_DAY * p_available
            p_detected = min(1.0, bars_with_signal / available_peaks) if available_peaks > 0 else 0

    # --- P(traded | detected): conversion rate through gates ---
    avg_trades_per_day = tl.groupby('_day').size().mean()
    p_traded = avg_trades_per_day / bars_with_signal if bars_with_signal > 0 else 0

    # --- Joint probability ---
    p_joint = p_available * p_detected * p_traded

    lines.append(f"\n  Actual / Theoretical = P(avail) x P(detect|avail) x P(trade|detect)")
    lines.append(f"")
    lines.append(f"  +---------------------------+--------+---------------------------------------+")
    lines.append(f"  | P(available)              | {p_available:>5.3f}  | 1 - slot_blocked_fraction             |")
    lines.append(f"  |   Session:                | {SESSION_SECS/3600:.0f}h    | {SESSION_BARS:,} bars at 15s                |")
    lines.append(f"  |   Avg bars in trade/day:  | {avg_hold_bars_per_day:>5.0f}  |                                       |")
    lines.append(f"  |   Avg bars available/day: | {SESSION_BARS - avg_hold_bars_per_day:>5.0f}  |                                       |")
    lines.append(f"  +---------------------------+--------+---------------------------------------+")
    lines.append(f"  | P(detected | available)   | {p_detected:>5.3f}  | signal_bars / available_peaks         |")
    lines.append(f"  |   Theoretical peaks/day:  |   {THEORETICAL_PER_DAY:>3.0f}  | {PEAK_INTERVAL_MIN:.0f}-min oscillation period          |")
    lines.append(f"  |   Adj. for availability:  |   {THEORETICAL_PER_DAY * p_available:>3.0f}  |                                       |")
    lines.append(f"  |   Bars with candidates:   | {bars_with_signal:>5.0f}  | unique ts with signal/day             |")
    lines.append(f"  +---------------------------+--------+---------------------------------------+")
    lines.append(f"  | P(traded | detected)      | {p_traded:>5.3f}  | trades / signal_bars                  |")
    lines.append(f"  |   Signal bars/day:        | {bars_with_signal:>5.0f}  |                                       |")
    lines.append(f"  |   Trades/day:             | {avg_trades_per_day:>5.1f}  |                                       |")
    lines.append(f"  +---------------------------+--------+---------------------------------------+")
    lines.append(f"  | Joint probability         | {p_joint:>5.3f}  |                                       |")
    lines.append(f"  +---------------------------+--------+---------------------------------------+")
    lines.append(f"")
    lines.append(f"  Theoretical capacity:  {THEORETICAL_PER_DAY:.0f} trades/day")
    lines.append(f"  Expected (T x P_joint): {THEORETICAL_PER_DAY * p_joint:.1f} trades/day")
    lines.append(f"  Actual:                {avg_trades_per_day:.1f} trades/day")
    lines.append(f"")

    # Marginal contribution to the gap (conditional decomposition)
    slot_loss = (1 - p_available) * THEORETICAL_PER_DAY
    detect_loss = p_available * (1 - p_detected) * THEORETICAL_PER_DAY
    gate_loss = p_available * p_detected * (1 - p_traded) * THEORETICAL_PER_DAY
    lines.append(f"  -- THROUGHPUT LOSS DECOMPOSITION --")
    lines.append(f"  Theoretical capacity:                {THEORETICAL_PER_DAY:>6.0f} trades/day")
    lines.append(f"  - Slot blocking  (1-P_avail):        {slot_loss:>6.1f}  ({slot_loss/THEORETICAL_PER_DAY*100:>5.1f}%)")
    lines.append(f"  - No detection   (1-P_detect|avail): {detect_loss:>6.1f}  ({detect_loss/THEORETICAL_PER_DAY*100:>5.1f}%)")
    lines.append(f"  - Gate rejection (1-P_trade|detect): {gate_loss:>6.1f}  ({gate_loss/THEORETICAL_PER_DAY*100:>5.1f}%)")
    lines.append(f"  = Actual throughput:                  {avg_trades_per_day:>6.1f}")
    lines.append(f"")
    biggest = max([('Slot blocking', slot_loss), ('Detection', detect_loss),
                   ('Gate rejection', gate_loss)], key=lambda x: x[1])
    lines.append(f"  Largest bottleneck: {biggest[0]} ({biggest[1]:.0f} trades/day lost)")

    return lines


def main():
    parser = argparse.ArgumentParser(description='Signal fire rate research')
    parser.add_argument('--is', dest='is_mode', action='store_true', help='Analyze IS data (shorthand for --mode is)')
    parser.add_argument('--mode', choices=['is', 'oos', 'oos2'], default=None,
                        help='Data mode: is, oos, or oos2')
    parser.add_argument('--save', action='store_true', help='Save report to reports/findings/')
    args = parser.parse_args()

    mode = args.mode or ('is' if args.is_mode else 'oos')
    print(f"\n{'='*90}")
    print(f"SIGNAL FIRE RATE RESEARCH — {mode.upper()}")
    print(f"{'='*90}\n")

    sl, fn, tl = load_data(mode)
    print()

    report = []
    report.extend(gate_funnel(sl, fn, tl))
    report.extend(loosening_scenarios(sl, fn))
    report.extend(fn_deep_dive(fn))
    report.extend(fire_rate_over_time(sl, tl))
    report.extend(slot_utilization(tl))
    report.extend(capacity_decomposition(sl, fn, tl))

    # Print
    for line in report:
        print(line)

    # Save
    if args.save:
        from datetime import datetime
        os.makedirs('reports/findings', exist_ok=True)
        fname = f"reports/findings/{datetime.now().strftime('%Y-%m-%d')}_signal_fire_{mode}.md"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report) + '\n')
        print(f"\n  Report saved: {fname}")


if __name__ == '__main__':
    main()
