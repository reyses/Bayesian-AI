"""Shared trade report generator for IS, OOS, and live.

All three paths produce trade dicts from BarProcessor. This module
computes standard breakdowns and formats a consistent scorecard.

Trade dict minimum fields (from BarProcessor._handle_exit / force_close):
    pnl, side, exit_reason, bars_held, trade_mfe_ticks, entry_price, exit_price, tid

Optional enrichment (from trainer oracle tracking):
    direction, actual_pnl, capture_rate, hold_bars, playbook, entry_depth, root_tf
"""

import os
import time
from collections import defaultdict


def _pf(gross_win: float, gross_loss: float) -> float:
    """Profit factor: gross win / |gross loss|."""
    return gross_win / abs(gross_loss) if gross_loss != 0 else 0.0


def _wr(wins: int, total: int) -> float:
    """Win rate as percentage."""
    return wins / total * 100 if total > 0 else 0.0


def compute_stats(trades: list[dict]) -> dict:
    """Compute standard statistics from a list of trade dicts.

    Normalizes field names: supports both BarProcessor format (pnl, side, exit_reason)
    and oracle format (actual_pnl, direction, exit_reason).
    """
    n = len(trades)
    if n == 0:
        return {'n': 0, 'pnl': 0, 'wins': 0, 'wr': 0, 'pf': 0, 'avg': 0,
                'gross_win': 0, 'gross_loss': 0, 'max_dd': 0}

    def _get_pnl(t):
        return t.get('actual_pnl', t.get('pnl', 0.0))

    def _get_side(t):
        s = t.get('direction', t.get('side', '?'))
        return s.upper() if isinstance(s, str) else '?'

    pnls = [_get_pnl(t) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = sum(p for p in pnls if p < 0)
    total_pnl = sum(pnls)

    # Drawdown
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    return {
        'n': n,
        'pnl': total_pnl,
        'wins': wins,
        'wr': _wr(wins, n),
        'pf': _pf(gross_win, gross_loss),
        'avg': total_pnl / n,
        'gross_win': gross_win,
        'gross_loss': gross_loss,
        'max_dd': max_dd,
    }


def exit_breakdown(trades: list[dict]) -> dict[str, dict]:
    """Exit reason breakdown: {reason: {n, wins, pnl, gross_win, gross_loss, pf, wr, avg}}."""
    buckets = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0,
                                    'gross_win': 0.0, 'gross_loss': 0.0})
    for t in trades:
        reason = t.get('exit_reason', 'unknown')
        pnl = t.get('actual_pnl', t.get('pnl', 0.0))
        buckets[reason]['n'] += 1
        buckets[reason]['pnl'] += pnl
        if pnl > 0:
            buckets[reason]['wins'] += 1
            buckets[reason]['gross_win'] += pnl
        elif pnl < 0:
            buckets[reason]['gross_loss'] += pnl

    # Add derived fields
    for v in buckets.values():
        v['pf'] = _pf(v['gross_win'], v['gross_loss'])
        v['wr'] = _wr(v['wins'], v['n'])
        v['avg'] = v['pnl'] / v['n'] if v['n'] > 0 else 0
    return dict(buckets)


def direction_breakdown(trades: list[dict]) -> dict[str, dict]:
    """Direction breakdown: {LONG/SHORT: {n, wins, pnl, wr, avg}}."""
    buckets = defaultdict(lambda: {'n': 0, 'wins': 0, 'pnl': 0.0})
    for t in trades:
        side = t.get('direction', t.get('side', '?'))
        if isinstance(side, str):
            side = side.upper()
        pnl = t.get('actual_pnl', t.get('pnl', 0.0))
        buckets[side]['n'] += 1
        buckets[side]['pnl'] += pnl
        if pnl > 0:
            buckets[side]['wins'] += 1
    for v in buckets.values():
        v['wr'] = _wr(v['wins'], v['n'])
        v['avg'] = v['pnl'] / v['n'] if v['n'] > 0 else 0
    return dict(buckets)


def hold_duration_breakdown(trades: list[dict], bar_seconds: float = 15.0) -> dict[str, dict]:
    """Hold duration buckets: {label: {n, pnl, avg}}."""
    # Buckets in bars (15s default)
    BUCKETS = [(0, 2, '<30s'), (2, 4, '30s-1m'), (4, 8, '1-2m'),
               (8, 20, '2-5m'), (20, 40, '5-10m'), (40, 9999, '10m+')]
    result = {}
    for lo, hi, lbl in BUCKETS:
        bucket = [t for t in trades
                  if lo <= t.get('hold_bars', t.get('bars_held', 0)) < hi]
        if bucket:
            pnls = [t.get('actual_pnl', t.get('pnl', 0.0)) for t in bucket]
            result[lbl] = {
                'n': len(bucket),
                'pnl': sum(pnls),
                'avg': sum(pnls) / len(bucket),
            }
    return result


def format_scorecard(trades: list[dict], *,
                     mode: str = 'IS',
                     start_date: str = '',
                     end_date: str = '',
                     daily_ledger: list[dict] = None,
                     max_dd: float = 0.0,
                     peak_stats: dict = None,
                     git_hash: str = '',
                     run_ts: str = '') -> str:
    """Format a complete scorecard report. Returns multi-line string."""
    stats = compute_stats(trades)
    exits = exit_breakdown(trades)
    dirs = direction_breakdown(trades)
    holds = hold_duration_breakdown(trades)

    n = stats['n']
    if n == 0:
        return f"SCORECARD: {mode} — no trades"

    # Daily stats
    n_days = len(daily_ledger) if daily_ledger else 1
    if daily_ledger:
        day_pnls = sorted(d.get('pnl', 0) for d in daily_ledger)
        win_days = sum(1 for p in day_pnls if p > 0)
        worst_day = day_pnls[0] if day_pnls else 0
        best_day = day_pnls[-1] if day_pnls else 0
        med_day = day_pnls[len(day_pnls) // 2] if day_pnls else 0
    else:
        win_days = 0
        worst_day = best_day = med_day = 0

    W = 70
    L = []
    L.append('=' * W)
    L.append(f'SCORECARD: {mode} {start_date} to {end_date}')
    if git_hash or run_ts:
        L.append(f'  Commit: {git_hash}  |  Generated: {run_ts}')
    L.append(f'  PnL: ${stats["pnl"]:,.2f}  |  Trades: {n:,}  |  '
             f'WR: {stats["wr"]:.1f}%  |  PF: {stats["pf"]:.2f}')
    L.append(f'  $/trade: ${stats["avg"]:.2f}  |  $/day: ${stats["pnl"]/n_days:,.0f}  |  '
             f'Max DD: ${max_dd or stats["max_dd"]:,.0f}')
    if daily_ledger:
        L.append(f'  Win days: {win_days}/{n_days} ({_wr(win_days, n_days):.0f}%)  |  '
                 f'Best: ${best_day:,.0f}  |  Worst: ${worst_day:,.0f}  |  Median: ${med_day:,.0f}')
    L.append('=' * W)

    # What works
    L.append('')
    L.append('WHAT WORKS (PF > 1.5):')
    good = [(k, v) for k, v in exits.items() if v['pnl'] > 0]
    good.sort(key=lambda x: -x[1]['pnl'])
    for reason, v in good[:6]:
        L.append(f'  {reason:22s} {v["n"]:>5} trades  PF={v["pf"]:>5.2f}  '
                 f'${v["pnl"]:>+10,.2f}  ${v["avg"]:>+7.2f}/tr  WR={v["wr"]:.0f}%')

    # What hurts
    L.append('')
    L.append('WHAT HURTS (PF < 1.0):')
    bad = [(k, v) for k, v in exits.items() if v['pnl'] < 0]
    bad.sort(key=lambda x: x[1]['pnl'])
    for reason, v in bad[:5]:
        L.append(f'  {reason:22s} {v["n"]:>5} trades  PF={v["pf"]:>5.2f}  '
                 f'${v["pnl"]:>+10,.2f}  ${v["avg"]:>+7.2f}/tr  WR={v["wr"]:.0f}%')
    fixable = sum(-v['pnl'] * 0.5 for _, v in bad)
    L.append(f'  FIXABLE (50% recovery): ~${fixable:,.0f} -> PnL could be ${stats["pnl"] + fixable:,.0f}')

    # Hold duration
    L.append('')
    L.append('HOLD DURATION:')
    for lbl, v in holds.items():
        flag = ''
        if v['avg'] > stats['avg'] * 1.5 and v['n'] > 50:
            flag = ' <-- SWEET SPOT'
        elif v['avg'] < -5:
            flag = ' <-- OVER-HOLDING'
        L.append(f'  {lbl:>6s}: {v["n"]:>5} trades  ${v["avg"]:>+7.2f}/tr{flag}')

    # Direction
    L.append('')
    L.append('DIRECTION:')
    for d in sorted(dirs.keys()):
        v = dirs[d]
        pct = v['n'] / n * 100
        L.append(f'  {d}: {v["n"]:>5} ({pct:.0f}%)  WR={v["wr"]:.0f}%  PnL=${v["pnl"]:>+10,.2f}')

    # Sensor gate (optional)
    if peak_stats:
        ps = peak_stats
        L.append('')
        L.append('SENSOR GATE:')
        det = ps.get('peak_detected', 0)
        ent = ps.get('peak_entered', 0)
        blk_sensor = ps.get('blocked_1m_sensor', 0)
        blk_cat = ps.get('blocked_cat', 0)
        blk_adx = ps.get('blocked_adx_chop', 0)
        blk_cool = ps.get('blocked_cooldown', 0)
        blk_build = ps.get('blocked_no_buildup', 0)
        blk_total = blk_sensor + blk_cat + blk_adx + blk_cool + blk_build
        ratio = f'{blk_total}:{ent}' if ent > 0 else 'N/A'
        L.append(f'  Detected: {det:,}  |  Entered: {ent:,}  |  Blocked: {blk_total:,}  ({ratio} ratio)')
        if blk_sensor > 0: L.append(f'    1m_sensor: {blk_sensor:,}')
        if blk_cat > 0:    L.append(f'    cat_regime: {blk_cat:,}')
        if blk_adx > 0:    L.append(f'    adx_chop: {blk_adx:,}')
        if blk_cool > 0:   L.append(f'    cooldown: {blk_cool:,}')
        if blk_build > 0:  L.append(f'    no_buildup: {blk_build:,}')

    L.append('')
    L.append('=' * W)
    return '\n'.join(L)


def write_scorecard(trades: list[dict], path: str, **kwargs) -> str:
    """Format scorecard and write to file. Returns the report text."""
    text = format_scorecard(trades, **kwargs)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    return text
