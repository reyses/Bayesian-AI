"""
Saturation simulator — fixed TP / SL / timeout on every trade.

Answers: "can we build a saturation strategy that yields $X every T minutes?"

Applies a rigid rule to every trade's full 5s-granularity `path`:
  1. Walk path chronologically.
  2. TP fires first time pnl >= TP_DOLLARS  → exit at TP.
  3. SL fires first time pnl <= -SL_DOLLARS → exit at -SL.
  4. If neither by TIMEOUT_MINUTES, exit at that bar's pnl.
  5. If path ends first (tier exited earlier), use original exit.

TP / SL are hit-first-wins; precedence at same bar: TP if path[i]>=TP, else SL.
(Path granularity is 5s, so a bar with both TP and SL-touch is rare.)

Reports per tier per dataset:
  - Baseline $/trade, new $/trade, Δ
  - New WR
  - Trade duration distribution under the rule

Usage:
    python tools/saturation_sim.py                         # default TP=10, SL=5, 8 min
    python tools/saturation_sim.py --tp 15 --sl 7 --timeout 10
    python tools/saturation_sim.py --sweep                 # grid sweep

Output: reports/findings/saturation_sim_<source>.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH_TEMPLATE = 'reports/findings/saturation_sim_{source}.md'

SOURCES = {
    'iso':     {'IS': 'iso_is.pkl',     'OOS': 'iso_oos.pkl'},
    'blended': {'IS': 'blended_is.pkl', 'OOS': 'blended_oos.pkl'},
}

BARS_PER_MINUTE = 12   # 5s bars
MIN_TIER_N = 20


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def apply_rule(trade, tp, sl, timeout_bars_5s):
    """Return (new_pnl, exit_reason, exit_bar_idx).

    tp: take-profit in $ (positive). None disables.
    sl: stop-loss in $ (positive, will exit at -sl). None disables.
    timeout_bars_5s: max bars before timeout exit. None disables.
    """
    path = trade.get('path', [])
    if not path:
        return float(trade.get('pnl', 0.0)), 'original', -1
    for i, pt in enumerate(path):
        if timeout_bars_5s is not None and i >= timeout_bars_5s:
            # Timeout at bar N — exit at this bar's pnl
            return float(pt.get('pnl', 0.0)), 'sat_timeout', i
        p = float(pt.get('pnl', 0.0))
        if tp is not None and p >= tp:
            return float(tp), 'sat_tp', i
        if sl is not None and p <= -abs(sl):
            return -float(abs(sl)), 'sat_sl', i
    # Path ended first (original exit reached before timeout)
    return float(trade.get('pnl', 0.0)), 'original', len(path) - 1


def simulate(trades, tp, sl, timeout_min):
    """Run saturation rule on all trades. Return per-tier result dict +
    global aggregate."""
    timeout_bars = int(timeout_min * BARS_PER_MINUTE) if timeout_min else None
    by_tier = defaultdict(lambda: {
        'n': 0, 'orig_pnl': 0.0, 'new_pnl': 0.0,
        'wins': 0, 'losses': 0, 'flats': 0,
        'sat_tp': 0, 'sat_sl': 0, 'sat_timeout': 0, 'original': 0,
        'exit_bars': [],
    })
    for t in trades:
        tier = t.get('entry_tier', '?')
        r = by_tier[tier]
        r['n'] += 1
        orig = float(t.get('pnl', 0.0))
        r['orig_pnl'] += orig
        new_pnl, reason, bar_idx = apply_rule(t, tp, sl, timeout_bars)
        r['new_pnl'] += new_pnl
        r[reason] += 1
        if new_pnl > 0:
            r['wins'] += 1
        elif new_pnl < 0:
            r['losses'] += 1
        else:
            r['flats'] += 1
        if bar_idx >= 0:
            r['exit_bars'].append(bar_idx)
    out = {}
    for tier, r in by_tier.items():
        n = r['n']
        bars = r['exit_bars']
        out[tier] = {
            'n': n,
            'orig_pnl': r['orig_pnl'],
            'new_pnl': r['new_pnl'],
            'delta': r['new_pnl'] - r['orig_pnl'],
            'new_per_trade': r['new_pnl'] / n if n else 0,
            'orig_per_trade': r['orig_pnl'] / n if n else 0,
            'wr': r['wins'] / (r['wins'] + r['losses']) * 100
                   if (r['wins'] + r['losses']) else 0,
            'wins': r['wins'], 'losses': r['losses'],
            'sat_tp': r['sat_tp'], 'sat_sl': r['sat_sl'],
            'sat_timeout': r['sat_timeout'], 'original': r['original'],
            'tp_rate':      r['sat_tp'] / n * 100 if n else 0,
            'sl_rate':      r['sat_sl'] / n * 100 if n else 0,
            'timeout_rate': r['sat_timeout'] / n * 100 if n else 0,
            'mean_exit_bar': float(np.mean(bars)) if bars else 0,
            'median_exit_bar': float(np.median(bars)) if bars else 0,
        }
    return out


def render_table(label, result, tp, sl, timeout_min, out):
    out.append(f'## {label} — TP ${tp} / SL ${sl} / timeout {timeout_min} min')
    out.append('')
    out.append('| Tier | N | Baseline $/tr | New $/tr | Δ $/tr | '
               'New WR | TP% | SL% | Timeout% | Orig% | Median exit (bars 5s) |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    tiers = sorted(result.keys(),
                   key=lambda t: -result[t]['new_pnl'])
    total_new = total_orig = 0
    for tier in tiers:
        r = result[tier]
        if r['n'] < MIN_TIER_N:
            continue
        total_new += r['new_pnl']
        total_orig += r['orig_pnl']
        out.append(
            f'| {tier} | {r["n"]:,} | '
            f'${r["orig_per_trade"]:+.2f} | '
            f'${r["new_per_trade"]:+.2f} | '
            f'${r["new_per_trade"] - r["orig_per_trade"]:+.2f} | '
            f'{r["wr"]:.0f}% | '
            f'{r["tp_rate"]:.0f}% | {r["sl_rate"]:.0f}% | '
            f'{r["timeout_rate"]:.0f}% | {(r["original"]/r["n"]*100):.0f}% | '
            f'{r["median_exit_bar"]:.0f} |')
    out.append(f'| **TOTAL** | — | ${total_orig:+,.0f} | ${total_new:+,.0f} | '
               f'${total_new - total_orig:+,.0f} | — | — | — | — | — | — |')
    out.append('')


def print_summary(label, result, tp, sl, timeout_min):
    print(f'\n=== {label} — TP ${tp} / SL ${sl} / {timeout_min}min ===')
    total_new = sum(r['new_pnl'] for r in result.values())
    total_orig = sum(r['orig_pnl'] for r in result.values())
    print(f'  Total orig: ${total_orig:+,.0f}')
    print(f'  Total new:  ${total_new:+,.0f}   (delta ${total_new-total_orig:+,.0f})')
    print(f'{"Tier":<18} {"N":>6} {"Orig/tr":>8} {"New/tr":>8} {"WR":>4} '
          f'{"TP%":>4} {"SL%":>4} {"TO%":>4} {"OEx%":>5}')
    for tier in sorted(result.keys(), key=lambda t: -result[t]['new_pnl']):
        r = result[tier]
        if r['n'] < MIN_TIER_N:
            continue
        print(f'{tier:<18} {r["n"]:>6,} ${r["orig_per_trade"]:>+6.2f} '
              f'${r["new_per_trade"]:>+6.2f} {r["wr"]:>3.0f}% '
              f'{r["tp_rate"]:>3.0f}% {r["sl_rate"]:>3.0f}% '
              f'{r["timeout_rate"]:>3.0f}% '
              f'{(r["original"]/r["n"]*100):>4.0f}%')


def run_single(tp, sl, timeout_min, is_trades, oos_trades, out, label_suffix=''):
    is_res  = simulate(is_trades,  tp, sl, timeout_min)
    oos_res = simulate(oos_trades, tp, sl, timeout_min)
    print_summary(f'IS{label_suffix}', is_res, tp, sl, timeout_min)
    print_summary(f'OOS{label_suffix}', oos_res, tp, sl, timeout_min)
    render_table(f'IS{label_suffix}',  is_res,  tp, sl, timeout_min, out)
    render_table(f'OOS{label_suffix}', oos_res, tp, sl, timeout_min, out)
    return is_res, oos_res


def sweep(is_trades, oos_trades, out):
    """Grid sweep across (tp, sl, timeout). Report top configs."""
    out.append('## Sweep — full grid')
    out.append('')
    out.append('For each (TP, SL, timeout) combo: IS total $, OOS total $, '
               'combined $. Walk-forward safe = both positive.')
    out.append('')
    out.append('| TP | SL | Timeout (min) | IS $ | IS $/day | OOS $ | '
               'OOS $/day | Combined $ | Combined $/day | Walk-fwd? |')
    out.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')

    rows = []
    for tp in [5, 7.5, 10, 12.5, 15, 20]:
        for sl in [3, 5, 7.5, 10]:
            for timeout in [3, 5, 8, 12, 15]:
                is_res = simulate(is_trades, tp, sl, timeout)
                oos_res = simulate(oos_trades, tp, sl, timeout)
                is_total = sum(r['new_pnl'] for r in is_res.values())
                oos_total = sum(r['new_pnl'] for r in oos_res.values())
                combined = is_total + oos_total
                rows.append({
                    'tp': tp, 'sl': sl, 'timeout': timeout,
                    'is': is_total, 'oos': oos_total,
                    'combined': combined,
                })
    rows.sort(key=lambda r: -r['combined'])
    IS_DAYS = 277
    OOS_DAYS = 71
    for r in rows[:20]:
        is_day = r['is'] / IS_DAYS
        oos_day = r['oos'] / OOS_DAYS
        comb_day = r['combined'] / (IS_DAYS + OOS_DAYS)
        walk = '✓' if r['is'] > 0 and r['oos'] > 0 else '—'
        out.append(
            f'| ${r["tp"]:.1f} | ${r["sl"]:.1f} | {r["timeout"]} | '
            f'${r["is"]:+,.0f} | ${is_day:+.0f} | '
            f'${r["oos"]:+,.0f} | ${oos_day:+.0f} | '
            f'${r["combined"]:+,.0f} | ${comb_day:+.0f} | {walk} |')
    out.append('')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=list(SOURCES.keys()), default='iso')
    ap.add_argument('--tp', type=float, default=10.0,
                    help='Take-profit in $ (default 10)')
    ap.add_argument('--sl', type=float, default=5.0,
                    help='Stop-loss abs in $ (default 5)')
    ap.add_argument('--timeout', type=float, default=8.0,
                    help='Timeout in minutes (default 8)')
    ap.add_argument('--sweep', action='store_true',
                    help='Grid sweep over (TP, SL, timeout)')
    args = ap.parse_args()

    files = SOURCES[args.source]
    out_path = OUT_PATH_TEMPLATE.format(source=args.source)

    is_path  = os.path.join(TRADES_DIR, files['IS'])
    oos_path = os.path.join(TRADES_DIR, files['OOS'])
    print(f'Loading {is_path}...')
    is_trades = load(is_path)
    print(f'Loading {oos_path}...')
    oos_trades = load(oos_path)

    out = [f'# Saturation simulator — {args.source}', '']
    out.append(f'Rule: TP ${args.tp} / SL -${args.sl} / timeout {args.timeout} min. '
               'Applied post-hoc to every trade\'s 5s-granularity path.')
    out.append('')

    if args.sweep:
        sweep(is_trades, oos_trades, out)
    else:
        run_single(args.tp, args.sl, args.timeout, is_trades, oos_trades, out)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
