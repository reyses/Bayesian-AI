"""
Comprehensive trade review tool — exhaustive post-run analysis.

Usage:
    python tools/trade_review.py                          # OOS trade log (default)
    python tools/trade_review.py --is                     # IS trade log
    python tools/trade_review.py --file path/to/log.csv   # Custom file
    python tools/trade_review.py --save                   # Save report to reports/findings/

Sections:
    1. Executive Summary
    2. Scratch vs Real Trade Decomposition
    3. PnL Distribution & Percentiles
    4. Risk Metrics (Sharpe, Sortino, Calmar, PF, payoff)
    5. Drawdown Analysis (intraday + cumulative)
    6. Streak Analysis (win/loss runs)
    7. Exit Reason Cross-Tab (exit × direction × oracle)
    8. Tail Risk (worst trades, VaR, expected shortfall)
    9. Template Concentration (Pareto, top earners)
   10. Time Analysis (hour, DOW, session)
   11. Direction Audit (LONG vs SHORT breakdown)
   12. Bootstrap Confidence Intervals (WR, avg PnL, Sharpe)
   13. Anchor Exit Effectiveness
"""
import argparse
import csv
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TICK = 0.25
TICK_VAL = 0.50

def _f(val, default=0.0):
    try: return float(val)
    except (ValueError, TypeError): return default

def _i(val, default=0):
    try: return int(float(val))
    except (ValueError, TypeError): return default

def _ts_to_et(ts):
    """Unix timestamp -> ET datetime."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone(ZoneInfo('US/Eastern'))
    except Exception:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc)

def _session(hour):
    if hour < 4: return 'Overnight'
    if hour < 9: return 'Europe/PreMkt'
    if hour < 12: return 'US_Open'
    if hour < 15: return 'US_Mid'
    return 'US_Close'

def _pct(n, total):
    return n / total * 100 if total else 0.0

def _avg(vals):
    return sum(vals) / len(vals) if vals else 0.0

def _std(vals):
    if len(vals) < 2: return 0.0
    m = _avg(vals)
    return math.sqrt(sum((v - m)**2 for v in vals) / (len(vals) - 1))

def _median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0: return 0.0
    if n % 2 == 1: return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def _percentile(vals, p):
    s = sorted(vals)
    if not s: return 0.0
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    return s[f] + (k - f) * (s[c] - s[f])

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_trades(path):
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_executive(trades, out):
    pnls = [_f(t['actual_pnl']) for t in trades]
    n = len(trades)
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p < 0)
    scratches = sum(1 for p in pnls if p == 0)
    be_wins = sum(1 for p in pnls if p == 0.50)  # breakeven SL at entry+1tick
    total = sum(pnls)
    out.append('=' * 90)
    out.append('1. EXECUTIVE SUMMARY')
    out.append('=' * 90)
    out.append(f'  Trades: {n:,}  |  Wins: {wins:,}  |  Losses: {losses:,}  |  Scratch ($0): {scratches:,}  |  BE ($0.50): {be_wins:,}')
    out.append(f'  Win Rate: {_pct(wins, n):.1f}%  |  Win+BE Rate: {_pct(wins + be_wins - (be_wins if be_wins <= wins else 0), n):.1f}%')
    out.append(f'  Total PnL: ${total:,.2f}  |  Avg/trade: ${_avg(pnls):.2f}  |  Median: ${_median(pnls):.2f}')
    out.append(f'  Profit Factor: {sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)):.2f}' if sum(p for p in pnls if p < 0) != 0 else '  Profit Factor: INF')
    out.append('')


def section_scratch(trades, out):
    pnls = [_f(t['actual_pnl']) for t in trades]
    n = len(pnls)
    scratch = [t for t, p in zip(trades, pnls) if abs(p) <= 0.50]
    real = [t for t, p in zip(trades, pnls) if abs(p) > 0.50]
    real_pnls = [_f(t['actual_pnl']) for t in real]
    real_wins = sum(1 for p in real_pnls if p > 0)
    real_losses = sum(1 for p in real_pnls if p < 0)

    out.append('=' * 90)
    out.append('2. SCRATCH vs REAL TRADE DECOMPOSITION')
    out.append('=' * 90)
    out.append(f'  Scratch trades (|PnL| <= $0.50): {len(scratch):,} ({_pct(len(scratch), n):.1f}%)')
    out.append(f'  Real trades    (|PnL| >  $0.50): {len(real):,} ({_pct(len(real), n):.1f}%)')
    out.append('')
    if real:
        out.append(f'  REAL TRADE STATS:')
        out.append(f'    Win Rate: {_pct(real_wins, len(real)):.1f}%  ({real_wins}W / {real_losses}L)')
        out.append(f'    Total PnL: ${sum(real_pnls):,.2f}  |  Avg: ${_avg(real_pnls):.2f}  |  Median: ${_median(real_pnls):.2f}')
        win_pnls = [p for p in real_pnls if p > 0]
        loss_pnls = [p for p in real_pnls if p < 0]
        out.append(f'    Avg Winner: ${_avg(win_pnls):.2f}  |  Avg Loser: ${_avg(loss_pnls):.2f}  |  Payoff: {abs(_avg(win_pnls) / _avg(loss_pnls)):.2f}x' if loss_pnls else f'    Avg Winner: ${_avg(win_pnls):.2f}  |  No losses')

    # Scratch breakdown by exit reason
    scratch_reasons = defaultdict(int)
    for t in scratch:
        scratch_reasons[t.get('exit_reason', '?')] += 1
    if scratch_reasons:
        out.append(f'  SCRATCH EXIT REASONS:')
        for reason, cnt in sorted(scratch_reasons.items(), key=lambda x: -x[1]):
            out.append(f'    {reason:<25} {cnt:>5} ({_pct(cnt, len(scratch)):.1f}%)')
    out.append('')


def section_pnl_dist(trades, out):
    pnls = [_f(t['actual_pnl']) for t in trades]
    out.append('=' * 90)
    out.append('3. PnL DISTRIBUTION & PERCENTILES')
    out.append('=' * 90)
    # Histogram bins
    bins = [(-1e9, -100), (-100, -50), (-50, -10), (-10, -0.01), (0, 0.51),
            (0.51, 10), (10, 50), (50, 100), (100, 500), (500, 1e9)]
    labels = ['< -$100', '-$100..-$50', '-$50..-$10', '-$10..-$0', '$0-$0.50',
              '$0.51-$10', '$10-$50', '$50-$100', '$100-$500', '> $500']
    out.append(f'  {"Bin":<18} {"Count":>7} {"Pct":>7} {"CumPnL":>12}  Bar')
    out.append(f'  {"-"*18} {"-"*7} {"-"*7} {"-"*12}  {"-"*30}')
    for (lo, hi), label in zip(bins, labels):
        in_bin = [p for p in pnls if lo <= p < hi]
        cnt = len(in_bin)
        bar = '#' * min(50, max(1, cnt * 50 // len(pnls))) if cnt else ''
        out.append(f'  {label:<18} {cnt:>7} {_pct(cnt, len(pnls)):>6.1f}% ${sum(in_bin):>11,.2f}  {bar}')

    out.append('')
    out.append(f'  Percentiles:')
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        out.append(f'    P{p:<3}: ${_percentile(pnls, p):>10,.2f}')

    # Pareto / concentration
    sorted_pnls = sorted(pnls, reverse=True)
    n = len(sorted_pnls)
    top5_pnl = sum(sorted_pnls[:max(1, n // 20)])
    top10_pnl = sum(sorted_pnls[:max(1, n // 10)])
    top20_pnl = sum(sorted_pnls[:max(1, n // 5)])
    total = sum(pnls)
    out.append('')
    out.append(f'  PROFIT CONCENTRATION:')
    out.append(f'    Top  5% ({n // 20:>4} trades): ${top5_pnl:>10,.2f} = {_pct(top5_pnl, total):.1f}% of total')
    out.append(f'    Top 10% ({n // 10:>4} trades): ${top10_pnl:>10,.2f} = {_pct(top10_pnl, total):.1f}% of total')
    out.append(f'    Top 20% ({n // 5:>4} trades): ${top20_pnl:>10,.2f} = {_pct(top20_pnl, total):.1f}% of total')
    out.append('')


def section_risk(trades, out):
    pnls = [_f(t['actual_pnl']) for t in trades]
    # Group by day for Sharpe
    daily = defaultdict(float)
    for t in trades:
        et = _ts_to_et(t.get('entry_time', 0))
        daily[et.strftime('%Y-%m-%d')] += _f(t['actual_pnl'])
    daily_pnls = list(daily.values())

    avg_daily = _avg(daily_pnls)
    std_daily = _std(daily_pnls)
    neg_daily = [d for d in daily_pnls if d < 0]
    downside_std = _std(neg_daily) if len(neg_daily) >= 2 else _std(daily_pnls)

    sharpe = (avg_daily / std_daily * math.sqrt(252)) if std_daily > 0 else 0
    sortino = (avg_daily / downside_std * math.sqrt(252)) if downside_std > 0 else 0

    # Max DD (cumulative)
    cumul = 0
    peak = 0
    max_dd = 0
    for dp in daily_pnls:
        cumul += dp
        peak = max(peak, cumul)
        dd = peak - cumul
        max_dd = max(max_dd, dd)
    calmar = (sum(daily_pnls) / max_dd) if max_dd > 0 else 0

    # Profit factor
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')

    # Payoff ratio
    win_pnls = [p for p in pnls if p > 0]
    loss_pnls = [p for p in pnls if p < 0]
    payoff = abs(_avg(win_pnls) / _avg(loss_pnls)) if loss_pnls else float('inf')

    out.append('=' * 90)
    out.append('4. RISK METRICS')
    out.append('=' * 90)
    out.append(f'  Trading days: {len(daily_pnls)}  |  Avg daily PnL: ${avg_daily:.2f}  |  Std: ${std_daily:.2f}')
    out.append(f'  Sharpe (annualized):  {sharpe:.2f}')
    out.append(f'  Sortino (annualized): {sortino:.2f}')
    out.append(f'  Calmar ratio:         {calmar:.2f}')
    out.append(f'  Profit Factor:        {pf:.2f}')
    out.append(f'  Payoff ratio:         {payoff:.2f}x (avg win / avg loss)')
    out.append(f'  Max cumul drawdown:   ${max_dd:,.2f}')
    out.append(f'  Winning days: {sum(1 for d in daily_pnls if d > 0)}/{len(daily_pnls)} ({_pct(sum(1 for d in daily_pnls if d > 0), len(daily_pnls)):.1f}%)')
    out.append(f'  Losing days:  {sum(1 for d in daily_pnls if d < 0)}/{len(daily_pnls)} ({_pct(sum(1 for d in daily_pnls if d < 0), len(daily_pnls)):.1f}%)')
    out.append('')


def section_drawdown(trades, out):
    # Intraday worst dips per day
    daily_trades = defaultdict(list)
    for t in trades:
        et = _ts_to_et(t.get('entry_time', 0))
        daily_trades[et.strftime('%Y-%m-%d')].append(_f(t['actual_pnl']))

    worst_dips = []
    for date, pnls in sorted(daily_trades.items()):
        cumul = 0
        worst = 0
        for p in pnls:
            cumul += p
            worst = min(worst, cumul)
        worst_dips.append((date, worst, sum(pnls), len(pnls)))

    out.append('=' * 90)
    out.append('5. DRAWDOWN ANALYSIS')
    out.append('=' * 90)
    worst_dips.sort(key=lambda x: x[1])
    out.append(f'  TOP 10 WORST INTRADAY DIPS:')
    out.append(f'    {"Date":<12} {"Dip":>10} {"Day PnL":>10} {"Trades":>7}')
    for date, dip, day_pnl, n_trades in worst_dips[:10]:
        out.append(f'    {date:<12} ${dip:>9,.2f} ${day_pnl:>9,.2f} {n_trades:>7}')

    # Cumulative equity curve stats
    cumul = 0
    peak = 0
    max_dd = 0
    dd_start = None
    dd_end = None
    current_dd_start = None
    for date, pnls in sorted(daily_trades.items()):
        cumul += sum(pnls)
        if cumul > peak:
            peak = cumul
            current_dd_start = date
        dd = peak - cumul
        if dd > max_dd:
            max_dd = dd
            dd_start = current_dd_start
            dd_end = date

    out.append(f'\n  CUMULATIVE EQUITY:')
    out.append(f'    Final: ${cumul:,.2f}  |  Peak: ${peak:,.2f}  |  Max DD: ${max_dd:,.2f}')
    if dd_start and dd_end:
        out.append(f'    Max DD period: {dd_start} -> {dd_end}')
    out.append('')


def section_streaks(trades, out):
    pnls = [_f(t['actual_pnl']) for t in trades]
    # Win/loss streaks
    max_win_streak = 0
    max_loss_streak = 0
    cur_win = 0
    cur_loss = 0
    for p in pnls:
        if p >= 0:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    # Consecutive loss PnL
    worst_consec = 0
    cur_consec = 0
    for p in pnls:
        if p < 0:
            cur_consec += p
        else:
            cur_consec = 0
        worst_consec = min(worst_consec, cur_consec)

    out.append('=' * 90)
    out.append('6. STREAK ANALYSIS')
    out.append('=' * 90)
    out.append(f'  Max consecutive wins:   {max_win_streak}')
    out.append(f'  Max consecutive losses: {max_loss_streak}')
    out.append(f'  Worst consecutive loss PnL: ${worst_consec:,.2f}')
    out.append('')


def section_exit_crosstab(trades, out):
    out.append('=' * 90)
    out.append('7. EXIT REASON CROSS-TAB')
    out.append('=' * 90)

    # exit_reason × result
    tab = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        reason = t.get('exit_reason', '?')
        pnl = _f(t['actual_pnl'])
        tab[reason]['n'] += 1
        tab[reason]['pnl'] += pnl
        if pnl > 0:
            tab[reason]['wins'] += 1

    out.append(f'  {"Exit Reason":<25} {"N":>6} {"WR%":>7} {"Total PnL":>12} {"Avg PnL":>10} {"% of PnL":>8}')
    out.append(f'  {"-"*25} {"-"*6} {"-"*7} {"-"*12} {"-"*10} {"-"*8}')
    total_pnl = sum(v['pnl'] for v in tab.values())
    for reason in sorted(tab.keys(), key=lambda r: -tab[r]['pnl']):
        v = tab[reason]
        out.append(f'  {reason:<25} {v["n"]:>6} {_pct(v["wins"], v["n"]):>6.1f}% ${v["pnl"]:>11,.2f} ${v["pnl"]/v["n"]:>9.2f} {_pct(v["pnl"], total_pnl):>7.1f}%')

    # exit_reason × direction
    out.append('')
    out.append(f'  EXIT × DIRECTION:')
    dir_tab = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'pnl': 0}))
    for t in trades:
        reason = t.get('exit_reason', '?')
        d = t.get('direction', '?')
        pnl = _f(t['actual_pnl'])
        dir_tab[reason][d]['n'] += 1
        dir_tab[reason][d]['pnl'] += pnl

    out.append(f'  {"Exit Reason":<25} {"LONG n":>8} {"LONG avg":>10} {"SHORT n":>9} {"SHORT avg":>10}')
    out.append(f'  {"-"*25} {"-"*8} {"-"*10} {"-"*9} {"-"*10}')
    for reason in sorted(dir_tab.keys(), key=lambda r: -sum(dir_tab[r][d]['n'] for d in dir_tab[r])):
        l = dir_tab[reason].get('LONG', {'n': 0, 'pnl': 0})
        s = dir_tab[reason].get('SHORT', {'n': 0, 'pnl': 0})
        la = l['pnl'] / l['n'] if l['n'] else 0
        sa = s['pnl'] / s['n'] if s['n'] else 0
        out.append(f'  {reason:<25} {l["n"]:>8} ${la:>9.2f} {s["n"]:>9} ${sa:>9.2f}')

    # exit_reason × oracle label
    out.append('')
    out.append(f'  EXIT × ORACLE CLASS:')
    class_tab = defaultdict(lambda: defaultdict(lambda: {'n': 0, 'pnl': 0}))
    for t in trades:
        reason = t.get('exit_reason', '?')
        tc = t.get('trade_class', '?')
        pnl = _f(t['actual_pnl'])
        class_tab[reason][tc]['n'] += 1
        class_tab[reason][tc]['pnl'] += pnl

    classes = sorted({t.get('trade_class', '?') for t in trades})
    hdr = f'  {"Exit":<20}'
    for c in classes:
        hdr += f' {c[:12]:>14}'
    out.append(hdr)
    for reason in sorted(class_tab.keys()):
        row = f'  {reason:<20}'
        for c in classes:
            v = class_tab[reason].get(c, {'n': 0, 'pnl': 0})
            row += f' {v["n"]:>6}/${v["pnl"]:>6.0f}'
        out.append(row)
    out.append('')


def section_tail_risk(trades, out):
    pnls = [_f(t['actual_pnl']) for t in trades]
    n = len(pnls)
    out.append('=' * 90)
    out.append('8. TAIL RISK ANALYSIS')
    out.append('=' * 90)

    # Worst N trades
    worst = sorted(zip(pnls, trades), key=lambda x: x[0])[:15]
    out.append(f'  WORST 15 TRADES:')
    out.append(f'    {"PnL":>10} {"Dir":>6} {"Exit":>20} {"Depth":>6} {"Hold":>6} {"TID":>5} {"Oracle":>12} {"Class":>15}')
    for p, t in worst:
        out.append(f'    ${p:>9,.2f} {t.get("direction","?"):>6} {t.get("exit_reason","?"):>20} {t.get("entry_depth","?"):>6} {t.get("hold_bars","?"):>5}b {t.get("template_id","?"):>5} {t.get("oracle_label_name","?"):>12} {t.get("trade_class","?"):>15}')

    # VaR and Expected Shortfall
    sorted_pnls = sorted(pnls)
    var_1 = sorted_pnls[max(0, int(n * 0.01))]
    var_5 = sorted_pnls[max(0, int(n * 0.05))]
    es_1 = _avg(sorted_pnls[:max(1, int(n * 0.01))])
    es_5 = _avg(sorted_pnls[:max(1, int(n * 0.05))])

    out.append(f'\n  VALUE AT RISK:')
    out.append(f'    VaR  1%: ${var_1:>10,.2f}  (1 in 100 trades worse than this)')
    out.append(f'    VaR  5%: ${var_5:>10,.2f}  (1 in 20 trades worse than this)')
    out.append(f'    ES   1%: ${es_1:>10,.2f}  (avg of worst 1%)')
    out.append(f'    ES   5%: ${es_5:>10,.2f}  (avg of worst 5%)')
    out.append('')


def section_template(trades, out):
    out.append('=' * 90)
    out.append('9. TEMPLATE CONCENTRATION')
    out.append('=' * 90)

    by_tid = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    for t in trades:
        tid = t.get('template_id', '?')
        pnl = _f(t['actual_pnl'])
        by_tid[tid]['n'] += 1
        by_tid[tid]['pnl'] += pnl
        if pnl > 0:
            by_tid[tid]['wins'] += 1

    total_pnl = sum(v['pnl'] for v in by_tid.values())
    sorted_tids = sorted(by_tid.items(), key=lambda x: -x[1]['pnl'])

    out.append(f'  TOP 15 TEMPLATES BY PnL:')
    out.append(f'    {"TID":>5} {"Trades":>7} {"WR%":>7} {"Total PnL":>12} {"Avg":>9} {"% of Total":>11} {"Cumul%":>8}')
    cumul_pct = 0
    for tid, v in sorted_tids[:15]:
        pct = _pct(v['pnl'], total_pnl)
        cumul_pct += pct
        out.append(f'    {tid:>5} {v["n"]:>7} {_pct(v["wins"], v["n"]):>6.1f}% ${v["pnl"]:>11,.2f} ${v["pnl"]/v["n"]:>8.2f} {pct:>10.1f}% {cumul_pct:>7.1f}%')

    # Bottom 5 (money losers)
    bottom = sorted_tids[-5:] if len(sorted_tids) >= 5 else sorted_tids
    bottom.reverse()
    out.append(f'\n  BOTTOM 5 TEMPLATES (money losers):')
    for tid, v in bottom:
        if v['pnl'] >= 0:
            continue
        out.append(f'    TID {tid:>5}: {v["n"]} trades, ${v["pnl"]:,.2f} total, ${v["pnl"]/v["n"]:.2f}/trade')

    out.append(f'\n  TEMPLATE STATS: {len(by_tid)} unique templates, '
               f'{sum(1 for v in by_tid.values() if v["pnl"] > 0)} profitable, '
               f'{sum(1 for v in by_tid.values() if v["pnl"] < 0)} unprofitable, '
               f'{sum(1 for v in by_tid.values() if v["pnl"] == 0)} breakeven')
    out.append('')


def section_time(trades, out):
    out.append('=' * 90)
    out.append('10. TIME ANALYSIS')
    out.append('=' * 90)

    by_hour = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    by_dow = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})
    by_session = defaultdict(lambda: {'n': 0, 'pnl': 0, 'wins': 0})

    for t in trades:
        et = _ts_to_et(t.get('entry_time', 0))
        h = et.hour
        dow = et.strftime('%a')
        sess = _session(h)
        pnl = _f(t['actual_pnl'])
        for bucket, key in [(by_hour, h), (by_dow, dow), (by_session, sess)]:
            bucket[key]['n'] += 1
            bucket[key]['pnl'] += pnl
            if pnl > 0:
                bucket[key]['wins'] += 1

    out.append(f'  BY HOUR (ET):')
    out.append(f'    {"Hour":>5} {"N":>6} {"WR%":>7} {"Avg PnL":>10} {"Total PnL":>12}')
    for h in sorted(by_hour.keys()):
        v = by_hour[h]
        out.append(f'    {h:>5} {v["n"]:>6} {_pct(v["wins"], v["n"]):>6.1f}% ${v["pnl"]/v["n"]:>9.2f} ${v["pnl"]:>11,.2f}')

    out.append(f'\n  BY SESSION:')
    for sess in ['Overnight', 'Europe/PreMkt', 'US_Open', 'US_Mid', 'US_Close']:
        v = by_session.get(sess, {'n': 0, 'pnl': 0, 'wins': 0})
        if v['n']:
            out.append(f'    {sess:<15} {v["n"]:>6} trades  {_pct(v["wins"], v["n"]):>6.1f}% WR  ${v["pnl"]/v["n"]:>9.2f}/trade  ${v["pnl"]:>11,.2f} total')

    out.append(f'\n  BY DAY OF WEEK:')
    for dow in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
        v = by_dow.get(dow, {'n': 0, 'pnl': 0, 'wins': 0})
        if v['n']:
            out.append(f'    {dow:<5} {v["n"]:>6} trades  {_pct(v["wins"], v["n"]):>6.1f}% WR  ${v["pnl"]/v["n"]:>9.2f}/trade  ${v["pnl"]:>11,.2f} total')
    out.append('')


def section_direction(trades, out):
    out.append('=' * 90)
    out.append('11. DIRECTION AUDIT')
    out.append('=' * 90)

    longs = [t for t in trades if t.get('direction') == 'LONG']
    shorts = [t for t in trades if t.get('direction') == 'SHORT']

    for label, subset in [('LONG', longs), ('SHORT', shorts)]:
        pnls = [_f(t['actual_pnl']) for t in subset]
        if not pnls:
            continue
        wins = sum(1 for p in pnls if p > 0)
        real = [p for p in pnls if abs(p) > 0.50]
        real_wins = sum(1 for p in real if p > 0)
        correct = sum(1 for t in subset if t.get('trade_class') == 'correct_dir')
        out.append(f'  {label}: {len(subset)} trades ({_pct(len(subset), len(trades)):.1f}%)')
        out.append(f'    Headline WR: {_pct(wins, len(pnls)):.1f}%  |  Real WR: {_pct(real_wins, len(real)):.1f}%  |  Oracle correct: {_pct(correct, len(subset)):.1f}%')
        out.append(f'    Total: ${sum(pnls):,.2f}  |  Avg: ${_avg(pnls):.2f}  |  PF: {sum(p for p in pnls if p > 0) / abs(sum(p for p in pnls if p < 0)):.2f}' if sum(p for p in pnls if p < 0) != 0 else f'    Total: ${sum(pnls):,.2f}  |  Avg: ${_avg(pnls):.2f}  |  PF: INF')
    out.append('')


def section_bootstrap(trades, out, n_boot=10000):
    random.seed(42)
    pnls = [_f(t['actual_pnl']) for t in trades]
    n = len(pnls)

    # Daily PnL for Sharpe bootstrap
    daily = defaultdict(float)
    for t in trades:
        et = _ts_to_et(t.get('entry_time', 0))
        daily[et.strftime('%Y-%m-%d')] += _f(t['actual_pnl'])
    daily_pnls = list(daily.values())

    boot_wr = []
    boot_avg = []
    boot_sharpe = []

    for _ in range(n_boot):
        sample = random.choices(pnls, k=n)
        boot_wr.append(sum(1 for p in sample if p > 0) / n * 100)
        boot_avg.append(_avg(sample))

        d_sample = random.choices(daily_pnls, k=len(daily_pnls))
        m = _avg(d_sample)
        s = _std(d_sample)
        boot_sharpe.append(m / s * math.sqrt(252) if s > 0 else 0)

    out.append('=' * 90)
    out.append('12. BOOTSTRAP CONFIDENCE INTERVALS (10,000 resamples)')
    out.append('=' * 90)
    for label, vals in [('Win Rate (%)', boot_wr), ('Avg PnL ($)', boot_avg), ('Sharpe (ann)', boot_sharpe)]:
        lo = _percentile(vals, 2.5)
        hi = _percentile(vals, 97.5)
        med = _percentile(vals, 50)
        out.append(f'  {label:<18}  median: {med:>8.2f}  95% CI: [{lo:>8.2f}, {hi:>8.2f}]')
    out.append('')


def section_anchor(trades, out):
    """Anchor exit effectiveness — how do anchor fields modulate exits."""
    anchor_trades = [t for t in trades if _f(t.get('anchor_mfe_ticks', 0)) > 0]
    if not anchor_trades:
        return

    out.append('=' * 90)
    out.append('13. ANCHOR EXIT EFFECTIVENESS')
    out.append('=' * 90)

    # Split: exited before vs after anchor_mfe_bars
    early = []  # exited before expected time
    late = []   # exited after expected time
    for t in anchor_trades:
        held = _i(t.get('hold_bars', 0))
        anchor_bars = _f(t.get('anchor_mfe_bars', 0))
        if anchor_bars > 0 and held < anchor_bars:
            early.append(t)
        else:
            late.append(t)

    early_pnls = [_f(t['actual_pnl']) for t in early]
    late_pnls = [_f(t['actual_pnl']) for t in late]

    out.append(f'  Trades with anchor data: {len(anchor_trades)}')
    out.append(f'  Exited BEFORE expected MFE time: {len(early)} ({_pct(len(early), len(anchor_trades)):.1f}%)  avg ${_avg(early_pnls):.2f}')
    out.append(f'  Exited AFTER  expected MFE time: {len(late)} ({_pct(len(late), len(anchor_trades)):.1f}%)  avg ${_avg(late_pnls):.2f}')

    # MFE capture: did trades reach anchor_mfe_ticks?
    reached = sum(1 for t in anchor_trades if _f(t.get('trade_mfe_ticks', 0)) >= _f(t.get('anchor_mfe_ticks', 0)))
    out.append(f'  Reached expected MFE: {reached}/{len(anchor_trades)} ({_pct(reached, len(anchor_trades)):.1f}%)')

    # By exit reason
    by_reason = defaultdict(lambda: {'n': 0, 'avg_anchor': 0, 'avg_held': 0, 'avg_pnl': 0})
    for t in anchor_trades:
        reason = t.get('exit_reason', '?')
        by_reason[reason]['n'] += 1
        by_reason[reason]['avg_anchor'] += _f(t.get('anchor_mfe_bars', 0))
        by_reason[reason]['avg_held'] += _i(t.get('hold_bars', 0))
        by_reason[reason]['avg_pnl'] += _f(t['actual_pnl'])

    out.append(f'\n  ANCHOR × EXIT REASON:')
    out.append(f'    {"Reason":<22} {"N":>6} {"Avg Anchor":>12} {"Avg Held":>10} {"Avg PnL":>10}')
    for reason in sorted(by_reason.keys(), key=lambda r: -by_reason[r]['n']):
        v = by_reason[reason]
        out.append(f'    {reason:<22} {v["n"]:>6} {v["avg_anchor"]/v["n"]:>11.1f}b {v["avg_held"]/v["n"]:>9.1f}b ${v["avg_pnl"]/v["n"]:>9.2f}')
    out.append('')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_review(path, save_path=None):
    trades = load_trades(path)
    print(f'Loaded {len(trades):,} trades from {path}')

    out = []
    out.append(f'COMPREHENSIVE TRADE REVIEW — {len(trades):,} trades')
    out.append(f'Source: {os.path.basename(path)}')
    out.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    out.append('')

    section_executive(trades, out)
    section_scratch(trades, out)
    section_pnl_dist(trades, out)
    section_risk(trades, out)
    section_drawdown(trades, out)
    section_streaks(trades, out)
    section_exit_crosstab(trades, out)
    section_tail_risk(trades, out)
    section_template(trades, out)
    section_time(trades, out)
    section_direction(trades, out)
    section_bootstrap(trades, out)
    section_anchor(trades, out)

    text = '\n'.join(out)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f'Saved: {save_path}')

    return text


def main():
    parser = argparse.ArgumentParser(description='Comprehensive trade review')
    parser.add_argument('--file', type=str, help='Path to trade log CSV')
    parser.add_argument('--is', dest='is_mode', action='store_true', help='Use IS trade log')
    parser.add_argument('--save', action='store_true', help='Save report to reports/findings/')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Checkpoint dir')
    args = parser.parse_args()

    if args.file:
        path = args.file
    elif args.is_mode:
        path = os.path.join(args.checkpoint_dir, 'oracle_trade_log.csv')
    else:
        path = os.path.join(args.checkpoint_dir, 'oos_trade_log.csv')

    if not os.path.exists(path):
        print(f'ERROR: {path} not found')
        sys.exit(1)

    save_path = None
    if args.save:
        label = 'is' if args.is_mode else 'oos'
        save_path = f'reports/findings/{datetime.now().strftime("%Y-%m-%d")}_trade_review_{label}.md'

    text = run_review(path, save_path)
    print(text)


if __name__ == '__main__':
    main()
