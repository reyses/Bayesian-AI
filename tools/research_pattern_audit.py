#!/usr/bin/env python
"""
Pattern Library Audit Tool
==========================
End-to-end audit of the pattern recognition pipeline:
  1. Template statistics (MFE/MAE/bar expectations)
  2. Oracle horizon analysis (what TF/lookahead generated the stats)
  3. Scale comparison: template expectations vs actual trade outcomes
  4. Anchor patience impact quantification
  5. Per-trade matching walkthrough (sample trades)

Usage:
    python tools/research_pattern_audit.py                  # full audit, IS mode
    python tools/research_pattern_audit.py --mode oos       # OOS data
    python tools/research_pattern_audit.py --trace 5        # trace 5 sample trades
    python tools/research_pattern_audit.py --template 21    # audit single template
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

# ── Oracle config (lookahead horizons) ──────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.oracle_config import ORACLE_LOOKAHEAD_BARS

# MNQ tick
TICK_SIZE = 0.25
TICK_VALUE = 0.50

# TF → seconds mapping
TF_SECONDS = {
    '1s': 1, '5s': 5, '15s': 15, '30s': 30,
    '1m': 60, '2m': 120, '3m': 180, '5m': 300,
    '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400,
    '1D': 86400, '1W': 604800,
}

# Depth → approximate root TF (from training pipeline)
DEPTH_TO_TF = {
    1: '4h', 2: '1h', 3: '30m', 4: '15m', 5: '5m',
    6: '3m', 7: '1m', 8: '30s', 9: '15s', 10: '5s',
    11: '1s', 12: '1s',
}


def load_pattern_library():
    path = 'checkpoints/pattern_library.pkl'
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        sys.exit(1)
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_trade_log(mode='is'):
    name = 'oos_trade_log.csv' if mode in ('oos', 'oos2') else 'oracle_trade_log.csv'
    path = os.path.join('checkpoints', name)
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping trade comparison")
        return None
    return pd.read_csv(path)


# ======================================================================
#  SECTION 1: Template Library Overview
# ======================================================================
def section_template_overview(lib):
    print(f"\n{'='*74}")
    print(f"  SECTION 1: Template Library Overview ({len(lib)} templates)")
    print(f"{'='*74}")

    rows = []
    for tid, entry in lib.items():
        rows.append({
            'tid': tid,
            'members': entry.get('member_count', 0),
            'p75_mfe_ticks': entry.get('p75_mfe_ticks', 0),
            'mean_mfe_ticks': entry.get('mean_mfe_ticks', 0),
            'mean_mae_ticks': entry.get('mean_mae_ticks', 0),
            'p25_mae_ticks': entry.get('p25_mae_ticks', 0),
            'avg_mfe_bar': entry.get('avg_mfe_bar', 0),
            'p75_mfe_bar': entry.get('p75_mfe_bar', 0),
            'sigma_ticks': entry.get('regression_sigma_ticks', 0),
            'win_rate': entry.get('stats_win_rate', 0),
            'long_bias': entry.get('long_bias', 0.5),
        })
    df = pd.DataFrame(rows).sort_values('tid')

    print(f"\n  {'TID':>4} {'Members':>7} {'p75 MFE':>8} {'mean MFE':>9} "
          f"{'mean MAE':>9} {'sig tks':>8} {'avgMFEbar':>9} {'WR%':>5}")
    print(f"  {'':>4} {'':>7} {'(ticks)':>8} {'(ticks)':>9} {'(ticks)':>9} "
          f"{'':>8} {'(bars)':>9} {'':>5}")
    print(f"  {'-'*70}")

    for _, r in df.iterrows():
        print(f"  {int(r.tid):>4} {int(r.members):>7} {r.p75_mfe_ticks:>8.0f} "
              f"{r.mean_mfe_ticks:>9.0f} {r.mean_mae_ticks:>9.0f} "
              f"{r.sigma_ticks:>8.0f} {r.avg_mfe_bar:>9.1f} "
              f"{r.win_rate:>5.0%}")

    # Summary stats
    print(f"\n  AGGREGATE STATISTICS:")
    print(f"    p75_mfe_ticks: min={df.p75_mfe_ticks.min():.0f}  "
          f"median={df.p75_mfe_ticks.median():.0f}  "
          f"mean={df.p75_mfe_ticks.mean():.0f}  "
          f"max={df.p75_mfe_ticks.max():.0f}")
    print(f"    avg_mfe_bar:   min={df.avg_mfe_bar.min():.0f}  "
          f"median={df.avg_mfe_bar.median():.0f}  "
          f"mean={df.avg_mfe_bar.mean():.0f}  "
          f"max={df.avg_mfe_bar.max():.0f}")

    return df


# ======================================================================
#  SECTION 2: Oracle Horizon Analysis
# ======================================================================
def section_oracle_horizons(lib):
    print(f"\n{'='*74}")
    print(f"  SECTION 2: Oracle Horizon Analysis")
    print(f"{'='*74}")
    print(f"\n  Oracle lookahead windows (from config/oracle_config.py):")
    print(f"    {'TF':>5} {'Bars':>5} {'Duration':>12} {'Max MFE horizon':>20}")
    print(f"    {'-'*45}")
    for tf in ['1s','5s','15s','30s','1m','3m','5m','15m','30m','1h','4h']:
        bars = ORACLE_LOOKAHEAD_BARS.get(tf, 0)
        if bars == 0:
            continue
        secs = TF_SECONDS.get(tf, 0)
        dur_min = bars * secs / 60
        if dur_min >= 60:
            dur_str = f"{dur_min/60:.1f}h"
        else:
            dur_str = f"{dur_min:.0f}min"
        max_ticks_str = f"price range over {dur_str}"
        print(f"    {tf:>5} {bars:>5} {dur_str:>12} {max_ticks_str:>20}")

    print(f"\n  CRITICAL INSIGHT:")
    print(f"    Template MFE is the 75th percentile of MAX price excursion")
    print(f"    over the oracle's FULL lookahead window.")
    print(f"    A 15m pattern looks ahead 16 bars × 15 min = 4 HOURS.")
    print(f"    But actual trades hold for ~1-5 minutes (3-20 bars × 15s).")
    print(f"    The template 'expects' a 4-hour move; the trade captures 1 minute.")


# ======================================================================
#  SECTION 3: Scale Mismatch — Template vs Actual
# ======================================================================
def section_scale_mismatch(lib, tl):
    if tl is None:
        print(f"\n  (skipped — no trade log)")
        return

    print(f"\n{'='*74}")
    print(f"  SECTION 3: Scale Mismatch — Template Expectations vs Reality")
    print(f"{'='*74}")

    rows = []
    for tid, grp in tl.groupby('template_id'):
        if tid not in lib:
            continue
        entry = lib[tid]
        p75_mfe = entry.get('p75_mfe_ticks', 0)
        avg_mfe_bar = entry.get('avg_mfe_bar', 0)
        if p75_mfe == 0:
            continue

        actual_mfe = grp['trade_mfe_ticks'].mean()
        actual_bars = grp['hold_bars'].mean()
        ratio_mfe = actual_mfe / p75_mfe if p75_mfe > 0 else 0
        ratio_bars = actual_bars / avg_mfe_bar if avg_mfe_bar > 0 else 0

        rows.append({
            'tid': tid,
            'n_trades': len(grp),
            'template_mfe': p75_mfe,
            'actual_mfe': actual_mfe,
            'mfe_ratio': ratio_mfe,
            'template_bars': avg_mfe_bar,
            'actual_bars': actual_bars,
            'bar_ratio': ratio_bars,
            'avg_pnl': grp['actual_pnl'].mean(),
            'template_mfe_pts': p75_mfe * TICK_SIZE,
            'actual_mfe_pts': actual_mfe * TICK_SIZE,
        })

    df = pd.DataFrame(rows).sort_values('n_trades', ascending=False)

    print(f"\n  {'TID':>4} {'Trades':>6} {'Tmpl MFE':>9} {'Act MFE':>8} {'Ratio':>6} "
          f"{'Tmpl bars':>9} {'Act bars':>9} {'Ratio':>6} {'Avg$':>6}")
    print(f"  {'':>4} {'':>6} {'(ticks)':>9} {'(ticks)':>8} {'':>6} "
          f"{'':>9} {'':>9} {'':>6} {'':>6}")
    print(f"  {'-'*70}")

    for _, r in df.head(30).iterrows():
        print(f"  {int(r.tid):>4} {int(r.n_trades):>6} "
              f"{r.template_mfe:>9.0f} {r.actual_mfe:>8.1f} {r.mfe_ratio:>6.1%} "
              f"{r.template_bars:>9.1f} {r.actual_bars:>9.1f} {r.bar_ratio:>6.1%} "
              f"${r.avg_pnl:>5.1f}")

    # Aggregate
    all_mfe_ratios = df['mfe_ratio']
    all_bar_ratios = df['bar_ratio']
    print(f"\n  SCALE MISMATCH SUMMARY:")
    print(f"    MFE ratio (actual/template): "
          f"median={all_mfe_ratios.median():.1%}  mean={all_mfe_ratios.mean():.1%}")
    print(f"    Bar ratio (actual/template): "
          f"median={all_bar_ratios.median():.1%}  mean={all_bar_ratios.mean():.1%}")
    print(f"    --> Trades capture ~{all_mfe_ratios.median():.0%} of template MFE")
    print(f"    --> Trades hold ~{all_bar_ratios.median():.0%} of template expected time")

    # In dollar terms
    total_left = df['n_trades'].sum() * (df['template_mfe'].mean() - df['actual_mfe'].mean()) * TICK_VALUE
    print(f"\n    Weighted avg template MFE: {df['template_mfe'].mean():,.0f} ticks "
          f"= ${df['template_mfe'].mean() * TICK_VALUE:,.0f}")
    print(f"    Weighted avg actual MFE:   {df['actual_mfe'].mean():.0f} ticks "
          f"= ${df['actual_mfe'].mean() * TICK_VALUE:.0f}")

    return df


# ======================================================================
#  SECTION 4: Anchor Patience Impact
# ======================================================================
def section_anchor_patience(lib, tl):
    if tl is None:
        return

    print(f"\n{'='*74}")
    print(f"  SECTION 4: Anchor Patience Impact on Exits")
    print(f"{'='*74}")

    # All correct-direction trades that gave back >90%
    gb = tl[(tl['trade_class'] == 'correct_dir') &
            (tl['capture_rate'] <= 0.10) &
            (tl['actual_pnl'] >= 0) &
            (tl['oracle_mfe'] > 0)].copy()

    if gb.empty:
        print(f"  No giveback trades found.")
        return

    total_trades = len(tl[tl['trade_class'] == 'correct_dir'])
    print(f"\n  Correct-direction trades: {total_trades}")
    print(f"  Gave back >90% of MFE:   {len(gb)}  ({100*len(gb)/total_trades:.1f}%)")
    print(f"  Total PnL left on table:  ${gb['oracle_mfe'].sum() - gb['actual_pnl'].sum():,.0f}")

    # Break down by WHY giveback didn't fire
    gb_sl = gb[gb['exit_reason'] == 'stop_loss'].copy()
    gb_env = gb[gb['exit_reason'] == 'envelope_decay'].copy()
    gb_pgb = gb[gb['exit_reason'] == 'peak_giveback'].copy()

    print(f"\n  Exit reason breakdown (giveback trades):")
    print(f"    stop_loss:      {len(gb_sl):>4}  (BE lock → full retrace)")
    print(f"    envelope_decay: {len(gb_env):>4}")
    print(f"    peak_giveback:  {len(gb_pgb):>4}")
    print(f"    other:          {len(gb) - len(gb_sl) - len(gb_env) - len(gb_pgb):>4}")

    if gb_sl.empty:
        return

    # Anchor patience analysis on SL givebacks
    print(f"\n  ANCHOR PATIENCE on {len(gb_sl)} stop_loss givebacks:")
    print(f"    Avg anchor_mfe_ticks:  {gb_sl['anchor_mfe_ticks'].mean():>8,.0f} "
          f"= ${gb_sl['anchor_mfe_ticks'].mean() * TICK_VALUE:>8,.0f} expected move")
    print(f"    Avg actual trade MFE:  {gb_sl['trade_mfe_ticks'].mean():>8.0f} "
          f"= ${gb_sl['trade_mfe_ticks'].mean() * TICK_VALUE:>8.0f} actual move")
    print(f"    Avg 30% threshold:     {(gb_sl['anchor_mfe_ticks'] * 0.3).mean():>8,.0f} "
          f"= ${(gb_sl['anchor_mfe_ticks'] * 0.3).mean() * TICK_VALUE:>8,.0f} needed to enable giveback")
    print(f"    Avg anchor_mfe_bars:   {gb_sl['anchor_mfe_bars'].mean():>8.0f}")
    print(f"    Avg hold_bars at exit:  {gb_sl['hold_bars'].mean():>8.1f}")
    print(f"    --> Patience active:    always (hold << anchor_bars AND mfe << 30% threshold)")

    # What if we used realistic anchor values?
    print(f"\n  HYPOTHETICAL: realistic anchor thresholds")
    for cap in [20, 40, 80, 160]:
        capped = gb_sl.copy()
        capped['effective_anchor'] = capped['anchor_mfe_ticks'].clip(upper=cap)
        would_fire = capped[capped['trade_mfe_ticks'] >= capped['effective_anchor'] * 0.3]
        still_blocked = capped[capped['trade_mfe_ticks'] < capped['effective_anchor'] * 0.3]
        saved_pnl = (would_fire['trade_mfe_ticks'] * TICK_VALUE * 0.5).sum()  # est: capture 50% of MFE
        print(f"    Cap anchor at {cap:>3} ticks: "
              f"{len(would_fire):>4} get giveback (+${saved_pnl:>6,.0f} est), "
              f"{len(still_blocked):>4} still blocked")


# ======================================================================
#  SECTION 5: TP/SL Sizing Audit
# ======================================================================
def section_sizing_audit(lib, tl):
    if tl is None:
        return

    print(f"\n{'='*74}")
    print(f"  SECTION 5: TP/SL Sizing Audit")
    print(f"{'='*74}")

    print(f"\n  How TP/SL are computed from template stats:")
    print(f"    TP = p75_mfe_ticks × 0.85  (85% of 75th percentile MFE)")
    print(f"    SL = mean_mae_ticks × 2.0  (2× mean MAE)")
    print(f"    Trail = mean_mae_ticks × 1.1")
    print()

    rows = []
    for tid, grp in tl.groupby('template_id'):
        if tid not in lib:
            continue
        entry = lib[tid]
        p75_mfe = entry.get('p75_mfe_ticks', 0)
        mean_mae = entry.get('mean_mae_ticks', 0)
        if p75_mfe == 0:
            continue

        computed_tp = max(5, int(round(p75_mfe * 0.85)))
        computed_sl = max(3, int(round(mean_mae * 2.0))) if mean_mae > 1.0 else 20
        actual_tp = grp['tp_ticks'].mean()
        actual_sl = grp['sl_ticks'].mean()
        actual_mfe = grp['trade_mfe_ticks'].mean()
        tp_reachable = actual_mfe / computed_tp if computed_tp > 0 else 0

        rows.append({
            'tid': tid,
            'n': len(grp),
            'tp_ticks': computed_tp,
            'sl_ticks': computed_sl,
            'actual_tp': actual_tp,
            'actual_sl': actual_sl,
            'actual_mfe': actual_mfe,
            'tp_reachable': tp_reachable,
            'p75_mfe': p75_mfe,
            'tp_hits': (grp['exit_reason'] == 'take_profit').sum(),
        })

    df = pd.DataFrame(rows).sort_values('n', ascending=False)

    print(f"  {'TID':>4} {'Trades':>6} {'Comp TP':>8} {'Comp SL':>8} "
          f"{'MFE':>6} {'MFE/TP':>7} {'TP hits':>7}")
    print(f"  {'':>4} {'':>6} {'(ticks)':>8} {'(ticks)':>8} "
          f"{'(ticks)':>6} {'ratio':>7} {'':>7}")
    print(f"  {'-'*55}")

    for _, r in df.head(25).iterrows():
        print(f"  {int(r.tid):>4} {int(r.n):>6} "
              f"{int(r.tp_ticks):>8} {int(r.sl_ticks):>8} "
              f"{r.actual_mfe:>6.0f} {r.tp_reachable:>7.1%} "
              f"{int(r.tp_hits):>7}")

    # TP reachability summary
    print(f"\n  TP REACHABILITY SUMMARY:")
    print(f"    Avg MFE/TP ratio: {df.tp_reachable.mean():.1%}")
    print(f"    Templates where MFE/TP > 50%: {(df.tp_reachable > 0.5).sum()}/{len(df)}")
    print(f"    Templates where MFE/TP > 100% (reachable): {(df.tp_reachable >= 1.0).sum()}/{len(df)}")
    print(f"    --> TP is set from 4-hour oracle horizon but trades last ~1 minute")
    print(f"    --> TPs are virtually unreachable by design")

    total_tp_hits = df['tp_hits'].sum()
    total_trades = df['n'].sum()
    print(f"    Total TP exits: {total_tp_hits}/{total_trades} "
          f"({100*total_tp_hits/total_trades:.2f}%)")


# ======================================================================
#  SECTION 6: Per-Depth Oracle Horizon
# ======================================================================
def section_depth_horizon(lib, tl):
    if tl is None:
        return

    print(f"\n{'='*74}")
    print(f"  SECTION 6: Oracle Horizon by Depth")
    print(f"{'='*74}")

    print(f"\n  Each depth maps to a root TF. Oracle looks ahead N bars of THAT TF.")
    print(f"  But trades are executed on 15s bars and last ~1 minute.")
    print()
    print(f"  {'Depth':>5} {'Root TF':>7} {'Oracle LA':>10} {'Horizon':>10} "
          f"{'p75 MFE':>8} {'Act MFE':>8} {'Act hold':>9} {'Ratio':>6}")
    print(f"  {'':>5} {'':>7} {'(bars)':>10} {'(minutes)':>10} "
          f"{'(ticks)':>8} {'(ticks)':>8} {'(minutes)':>9} {'':>6}")
    print(f"  {'-'*70}")

    for depth in range(3, 13):
        tf = DEPTH_TO_TF.get(depth, '15s')
        la_bars = ORACLE_LOOKAHEAD_BARS.get(tf, 60)
        tf_secs = TF_SECONDS.get(tf, 15)
        horizon_min = la_bars * tf_secs / 60

        # Template stats for this depth
        depth_tids = [tid for tid, e in lib.items()
                      if 'centroid' in e and len(e['centroid']) > 5
                      and abs(e['centroid'][5] - depth) < 1.5]
        if depth_tids:
            avg_p75 = np.mean([lib[t].get('p75_mfe_ticks', 0) for t in depth_tids])
        else:
            avg_p75 = 0

        # Actual trade stats
        d_trades = tl[tl['entry_depth'] == depth]
        if len(d_trades) > 0:
            act_mfe = d_trades['trade_mfe_ticks'].mean()
            act_hold_min = d_trades['hold_bars'].mean() * 15 / 60
        else:
            act_mfe = 0
            act_hold_min = 0

        ratio = act_mfe / avg_p75 if avg_p75 > 0 else 0

        if horizon_min >= 60:
            horizon_str = f"{horizon_min/60:.1f}h"
        else:
            horizon_str = f"{horizon_min:.0f}min"

        print(f"  {depth:>5} {tf:>7} {la_bars:>10} {horizon_str:>10} "
              f"{avg_p75:>8.0f} {act_mfe:>8.0f} {act_hold_min:>8.1f}m "
              f"{ratio:>6.1%}")


# ======================================================================
#  SECTION 7: Trade Matching Trace (step-by-step)
# ======================================================================
def section_trade_trace(lib, tl, n_traces=5, target_tid=None):
    if tl is None:
        return

    print(f"\n{'='*74}")
    print(f"  SECTION 7: Trade Matching Trace (step-by-step)")
    print(f"{'='*74}")

    if target_tid is not None:
        sample = tl[tl['template_id'] == target_tid].head(n_traces)
    else:
        # Pick a mix: some winners, some givebacks
        winners = tl[tl['actual_pnl'] > 20].sample(min(n_traces // 2 + 1, len(tl[tl['actual_pnl'] > 20])),
                                                     random_state=42)
        givebacks = tl[(tl['capture_rate'] <= 0.10) & (tl['actual_pnl'] >= 0)].sample(
            min(n_traces // 2 + 1, len(tl[(tl['capture_rate'] <= 0.10) & (tl['actual_pnl'] >= 0)])),
            random_state=42)
        sample = pd.concat([winners, givebacks]).head(n_traces)

    for i, (_, trade) in enumerate(sample.iterrows()):
        tid = int(trade['template_id'])
        entry = lib.get(tid, {})
        print(f"\n  ── Trade {i+1}/{n_traces} ──────────────────────────────────")

        # Step 1: Pattern detection
        print(f"  STEP 1: Pattern Detected")
        print(f"    Depth:      {int(trade['entry_depth'])}")
        print(f"    Root TF:    {trade.get('root_tf', '?')}")
        print(f"    Direction:  {trade['direction']}")
        print(f"    Entry:      ${trade['entry_price']:.2f} at ts={int(trade['entry_time'])}")

        # Step 2: Feature extraction + template match
        print(f"  STEP 2: Template Match")
        print(f"    Matched TID:  {tid}")
        print(f"    Members:      {entry.get('member_count', '?')}")
        if 'centroid' in entry:
            c = entry['centroid']
            print(f"    Centroid z:   {c[0]:.2f}  vel: {c[1]:.2f}  mom: {c[2]:.2f}  "
                  f"depth: {c[5]:.1f}")

        # Step 3: Template expectations
        p75_mfe = entry.get('p75_mfe_ticks', 0)
        mean_mae = entry.get('mean_mae_ticks', 0)
        avg_bar = entry.get('avg_mfe_bar', 0)
        tf = trade.get('root_tf', '15s')
        la = ORACLE_LOOKAHEAD_BARS.get(tf, 60)
        tf_sec = TF_SECONDS.get(tf, 15)
        horizon = la * tf_sec / 60

        print(f"  STEP 3: Template Expectations (from oracle)")
        print(f"    p75_mfe_ticks:  {p75_mfe:.0f}  = ${p75_mfe * TICK_VALUE:.0f}  "
              f"(over {horizon:.0f}min oracle horizon)")
        print(f"    mean_mae_ticks: {mean_mae:.0f}  = ${mean_mae * TICK_VALUE:.0f}")
        print(f"    avg_mfe_bar:    {avg_bar:.0f}  "
              f"(bars in {tf} timeframe = {avg_bar * tf_sec / 60:.1f}min)")

        # Step 4: Position sizing from template
        computed_tp = max(5, int(round(p75_mfe * 0.85))) if p75_mfe > 2 else 0
        computed_sl = max(3, int(round(mean_mae * 2.0))) if mean_mae > 1 else 20
        print(f"  STEP 4: Position Sizing")
        print(f"    TP = p75_mfe × 0.85 = {computed_tp} ticks = ${computed_tp * TICK_VALUE:.0f}")
        print(f"    SL = mean_mae × 2.0 = {computed_sl} ticks = ${computed_sl * TICK_VALUE:.0f}")
        print(f"    Actual TP used:  {trade['tp_ticks']:.0f}  SL used: {trade['sl_ticks']:.0f}")

        # Step 5: Anchor patience parameters
        anc_mfe = trade.get('anchor_mfe_ticks', 0)
        anc_bars = trade.get('anchor_mfe_bars', 0)
        print(f"  STEP 5: Anchor Patience")
        print(f"    anchor_mfe_ticks: {anc_mfe:.0f}  (giveback disabled until MFE > {anc_mfe*0.3:.0f})")
        print(f"    anchor_mfe_bars:  {anc_bars:.0f}  (patience active for {anc_bars} bars)")

        # Step 6: What actually happened
        print(f"  STEP 6: Trade Outcome")
        print(f"    Hold bars:     {int(trade['hold_bars'])}  ({trade['hold_bars'] * 15 / 60:.1f} min)")
        print(f"    Trade MFE:     {trade['trade_mfe_ticks']:.0f} ticks = ${trade['trade_mfe_ticks'] * TICK_VALUE:.0f}")
        print(f"    Oracle MFE:    ${trade['oracle_mfe']:.1f}")
        print(f"    Exit reason:   {trade['exit_reason']}")
        print(f"    Actual PnL:    ${trade['actual_pnl']:.1f}")
        print(f"    Capture rate:  {trade['capture_rate']:.1%}")

        # Step 7: Diagnosis
        print(f"  STEP 7: Diagnosis")
        if trade['capture_rate'] <= 0.10 and trade['actual_pnl'] >= 0:
            if trade['exit_reason'] == 'stop_loss':
                if anc_mfe > 0 and trade['trade_mfe_ticks'] < anc_mfe * 0.3:
                    print(f"    ⚠ ANCHOR PATIENCE BLOCKED GIVEBACK")
                    print(f"      Trade MFE ({trade['trade_mfe_ticks']:.0f}) < "
                          f"30% of anchor ({anc_mfe*0.3:.0f})")
                    print(f"      Giveback was disabled → price retraced to BE stop")
                elif trade['trade_mfe_ticks'] < 16:
                    print(f"    ⚠ MFE TOO LOW FOR GIVEBACK")
                    print(f"      Trade MFE ({trade['trade_mfe_ticks']:.0f}) < "
                          f"giveback_min_mfe (16 ticks)")
                else:
                    print(f"    ⚠ UNKNOWN — giveback should have triggered")
            else:
                print(f"    Low capture despite non-SL exit — possible noise/reversal")
        elif trade['actual_pnl'] > 0:
            print(f"    ✓ Profitable trade, capture rate: {trade['capture_rate']:.0%}")
        else:
            print(f"    ✗ Loss: ${trade['actual_pnl']:.1f}")


# ======================================================================
#  SECTION 8: Normalization Recommendations
# ======================================================================
def section_recommendations(lib, tl):
    if tl is None:
        return

    print(f"\n{'='*74}")
    print(f"  SECTION 8: Normalization Recommendations")
    print(f"{'='*74}")

    # Compute realistic MFE per template from actual trades
    print(f"\n  Option A: Use ACTUAL trade MFE percentiles (from forward pass history)")
    rows = []
    for tid, grp in tl.groupby('template_id'):
        if tid not in lib or len(grp) < 3:
            continue
        rows.append({
            'tid': tid,
            'n': len(grp),
            'template_p75_mfe': lib[tid].get('p75_mfe_ticks', 0),
            'actual_p75_mfe': grp['trade_mfe_ticks'].quantile(0.75),
            'actual_p50_mfe': grp['trade_mfe_ticks'].median(),
            'actual_mean_mfe': grp['trade_mfe_ticks'].mean(),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"    Template p75_mfe (oracle):  median={df.template_p75_mfe.median():>6.0f} ticks")
        print(f"    Actual p75 trade MFE:       median={df.actual_p75_mfe.median():>6.0f} ticks")
        print(f"    Actual p50 trade MFE:       median={df.actual_p50_mfe.median():>6.0f} ticks")
        print(f"    Ratio actual/template:      {df.actual_p75_mfe.median() / df.template_p75_mfe.median():.1%}")

    # Option B: normalize by oracle horizon ratio
    print(f"\n  Option B: Scale by horizon ratio (trade_hold / oracle_lookahead)")
    trading_horizon_bars = tl['hold_bars'].median()  # in 15s bars
    trading_horizon_min = trading_horizon_bars * 15 / 60
    print(f"    Median trade hold: {trading_horizon_bars:.0f} bars = {trading_horizon_min:.1f} min")
    for tf in ['1m', '5m', '15m']:
        la = ORACLE_LOOKAHEAD_BARS.get(tf, 60)
        tf_sec = TF_SECONDS.get(tf, 60)
        oracle_min = la * tf_sec / 60
        ratio = trading_horizon_min / oracle_min
        print(f"    {tf} oracle: {oracle_min:.0f}min → scale factor: {ratio:.3f}")

    # Option C: fixed cap
    print(f"\n  Option C: Cap anchor_mfe_ticks to realistic maximum")
    for cap in [40, 80, 120]:
        gb_sl = tl[(tl['trade_class'] == 'correct_dir') &
                   (tl['capture_rate'] <= 0.10) &
                   (tl['actual_pnl'] >= 0) &
                   (tl['exit_reason'] == 'stop_loss') &
                   (tl['trade_mfe_ticks'] >= 16)].copy()
        if gb_sl.empty:
            continue
        gb_sl['eff'] = gb_sl['anchor_mfe_ticks'].clip(upper=cap)
        freed = gb_sl[gb_sl['trade_mfe_ticks'] >= gb_sl['eff'] * 0.3]
        est_save = (freed['trade_mfe_ticks'] * TICK_VALUE * 0.5).sum()
        print(f"    Cap={cap:>3}: {len(freed):>3} trades freed from patience → "
              f"est +${est_save:>6,.0f}")


# ======================================================================
#  MAIN
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Pattern Library Audit')
    parser.add_argument('--mode', choices=['is', 'oos', 'oos2'], default='oos',
                        help='Which trade log to analyze (default: oos)')
    parser.add_argument('--trace', type=int, default=5,
                        help='Number of trades to trace step-by-step')
    parser.add_argument('--template', type=int, default=None,
                        help='Audit a specific template ID')
    args = parser.parse_args()

    print(f"  Pattern Library Audit ({args.mode.upper()} mode)")
    print(f"  {'='*50}")

    lib = load_pattern_library()
    print(f"  Loaded {len(lib)} templates from pattern_library.pkl")

    tl = load_trade_log(args.mode)
    if tl is not None:
        print(f"  Loaded {len(tl):,} trades from trade log")

    section_template_overview(lib)
    section_oracle_horizons(lib)
    section_scale_mismatch(lib, tl)
    section_anchor_patience(lib, tl)
    section_sizing_audit(lib, tl)
    section_depth_horizon(lib, tl)
    section_trade_trace(lib, tl, n_traces=args.trace, target_tid=args.template)
    section_recommendations(lib, tl)

    print(f"\n{'='*74}")
    print(f"  Audit complete.")
    print(f"{'='*74}")


if __name__ == '__main__':
    main()
