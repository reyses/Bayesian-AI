"""
Tier segment diagnostic — split IS chronologically and measure per-tier
stability across segments. Overfit rules should show divergence within IS.

Hypothesis: if a tier's Apr 19 cluster/milestone rules are IS-overfit,
sub-samples of IS should disagree about its performance the same way IS
and OOS disagree. Walk-forward from H1→H2, or Q1→Q2→Q3→Q4, reveals:
  - DRIFT:   $/trade monotonically decays across segments (pattern of decay)
  - VOLATILE: $/trade jumps wildly between segments (noise)
  - STABLE:  $/trade consistent across segments (real edge)

For each tier, report per-segment N, $, $/trade, WR, $WR, then compute
cross-segment $/trade std and range. High std/mean ratio = unstable tier.

Also reports OOS for reference (no segmentation — too few days).

Usage:
    python tools/tier_segment_diagnostic.py
    python tools/tier_segment_diagnostic.py --segments 4

Output: reports/findings/tier_segment_diagnostic.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH = 'reports/findings/tier_segment_diagnostic.md'


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def segment_trades(trades, n_segments):
    """Split trades into n equal-day segments (chronological)."""
    by_day = defaultdict(list)
    for t in trades:
        by_day[t.get('day', 'unknown')].append(t)
    days = sorted(by_day.keys())
    n_days = len(days)
    bounds = np.linspace(0, n_days, n_segments + 1, dtype=int)
    segments = []
    for i in range(n_segments):
        seg_days = days[bounds[i]: bounds[i + 1]]
        seg_trades = []
        for d in seg_days:
            seg_trades.extend(by_day[d])
        segments.append({
            'idx': i + 1,
            'n_days': len(seg_days),
            'day_first': seg_days[0] if seg_days else '',
            'day_last': seg_days[-1] if seg_days else '',
            'trades': seg_trades,
        })
    return segments


def tier_stats(trades):
    """Return {tier: {n, pnl, per_trade, wr, dwr}}."""
    by_tier = defaultdict(lambda: {'n': 0, 'pnl': 0.0,
                                    'wins': 0, 'losses': 0,
                                    'win_d': 0.0, 'loss_d': 0.0})
    for t in trades:
        pnl = float(t.get('pnl', 0.0))
        tier = t.get('entry_tier', '?')
        r = by_tier[tier]
        r['n'] += 1
        r['pnl'] += pnl
        if pnl > 0:
            r['wins'] += 1
            r['win_d'] += pnl
        elif pnl < 0:
            r['losses'] += 1
            r['loss_d'] += -pnl
    out = {}
    for tier, r in by_tier.items():
        out[tier] = {
            'n': r['n'],
            'pnl': r['pnl'],
            'per_trade': r['pnl'] / r['n'] if r['n'] else 0,
            'wr': r['wins'] / (r['wins'] + r['losses']) * 100
                  if (r['wins'] + r['losses']) else 0,
            'dwr': ((r['win_d'] / r['loss_d']) - 1) * 100
                    if r['loss_d'] > 0 else 0,
            'win_d': r['win_d'],
            'loss_d': r['loss_d'],
        }
    return out


def render_segment_table(label, segments, all_tiers, out):
    """Per-tier $/trade row across segments."""
    out.append(f'## {label} — $/trade across segments')
    out.append('')
    out.append('Header shows segment index and day range.')
    out.append('')
    # Segment header
    hdr = ['Tier', 'Overall']
    for s in segments:
        hdr.append(f'S{s["idx"]} ({s["day_first"]}..{s["day_last"]})')
    hdr.append('std')
    hdr.append('range')
    hdr.append('pattern')
    out.append('| ' + ' | '.join(hdr) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr) - 1)) + '|')

    # Full-period stats
    full_trades = []
    for s in segments:
        full_trades.extend(s['trades'])
    full_stats = tier_stats(full_trades)

    # Per-segment stats
    seg_stats = [tier_stats(s['trades']) for s in segments]

    for tier in all_tiers:
        full = full_stats.get(tier, {'per_trade': 0, 'n': 0})
        row = [tier, f'${full["per_trade"]:+.2f} (n={full["n"]:,})']
        pts = []
        for ss in seg_stats:
            r = ss.get(tier)
            if r is None or r['n'] == 0:
                row.append('—')
                continue
            row.append(f'${r["per_trade"]:+.2f} (n={r["n"]})')
            pts.append(r['per_trade'])
        if len(pts) >= 2:
            std = float(np.std(pts, ddof=0))
            rng = max(pts) - min(pts)
            # pattern classification
            sign_changes = sum(1 for i in range(len(pts) - 1)
                               if pts[i] * pts[i + 1] < 0)
            if sign_changes == 0:
                if all(pts[i] >= pts[i + 1] for i in range(len(pts) - 1)):
                    pat = 'DECAY'
                elif all(pts[i] <= pts[i + 1] for i in range(len(pts) - 1)):
                    pat = 'IMPROVING'
                else:
                    pat = 'stable'
            else:
                pat = f'FLIPS ({sign_changes})'
            row.append(f'${std:.2f}')
            row.append(f'${rng:.2f}')
            row.append(pat)
        else:
            row.extend(['—', '—', '—'])
        out.append('| ' + ' | '.join(row) + ' |')
    out.append('')


def render_segment_pnl_table(label, segments, all_tiers, out):
    """Per-tier total $ row across segments."""
    out.append(f'## {label} — total $ across segments')
    out.append('')
    hdr = ['Tier', 'Overall $']
    for s in segments:
        hdr.append(f'S{s["idx"]}')
    hdr.append('sum check')
    out.append('| ' + ' | '.join(hdr) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr) - 1)) + '|')

    full_trades = []
    for s in segments:
        full_trades.extend(s['trades'])
    full_stats = tier_stats(full_trades)
    seg_stats = [tier_stats(s['trades']) for s in segments]

    grand_pnl = 0.0
    for tier in all_tiers:
        full = full_stats.get(tier, {'pnl': 0})
        grand_pnl += full['pnl']
        row = [tier, f'${full["pnl"]:+,.0f}']
        seg_sum = 0
        for ss in seg_stats:
            r = ss.get(tier)
            if r is None or r['n'] == 0:
                row.append('—')
            else:
                row.append(f'${r["pnl"]:+,.0f}')
                seg_sum += r['pnl']
        # sum check
        diff = seg_sum - full['pnl']
        row.append('✓' if abs(diff) < 0.5 else f'⚠ {diff:+.0f}')
        out.append('| ' + ' | '.join(row) + ' |')
    # Totals row
    tot_row = ['**TOTAL**', f'**${grand_pnl:+,.0f}**']
    for ss in seg_stats:
        stot = sum(r['pnl'] for r in ss.values())
        tot_row.append(f'**${stot:+,.0f}**')
    tot_row.append('')
    out.append('| ' + ' | '.join(tot_row) + ' |')
    out.append('')


def render_wr_table(label, segments, all_tiers, out):
    out.append(f'## {label} — WR across segments')
    out.append('')
    hdr = ['Tier', 'Overall WR']
    for s in segments:
        hdr.append(f'S{s["idx"]}')
    out.append('| ' + ' | '.join(hdr) + ' |')
    out.append('|' + '|'.join(['---'] + ['---:'] * (len(hdr) - 1)) + '|')

    full_trades = []
    for s in segments:
        full_trades.extend(s['trades'])
    full_stats = tier_stats(full_trades)
    seg_stats = [tier_stats(s['trades']) for s in segments]

    for tier in all_tiers:
        full = full_stats.get(tier, {'wr': 0, 'n': 0})
        row = [tier, f'{full["wr"]:.0f}%']
        for ss in seg_stats:
            r = ss.get(tier)
            if r is None or r['n'] == 0:
                row.append('—')
            else:
                row.append(f'{r["wr"]:.0f}%')
        out.append('| ' + ' | '.join(row) + ' |')
    out.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--segments', type=int, default=2,
                    help='Number of chronological segments (default 2 = halves)')
    ap.add_argument('--also-quarters', action='store_true',
                    help='Also run a 4-segment view for finer resolution')
    args = ap.parse_args()

    is_trades = load(os.path.join(TRADES_DIR, 'iso_is.pkl'))
    oos_trades = load(os.path.join(TRADES_DIR, 'iso_oos.pkl'))

    print(f'IS trades: {len(is_trades):,}')
    print(f'OOS trades: {len(oos_trades):,}')

    out = ['# Tier segment diagnostic — IS chronological stability', '']
    out.append(f'IS split into {args.segments} chronological segments; '
               'same tier measured in each. Overfit rules should show '
               'divergence within IS (decay pattern, sign flips, or high '
               'variance).')
    out.append('')
    out.append('Legend: **DECAY** = monotonic decline. **FLIPS(N)** = N sign '
               'changes across segments. **stable** = consistent.')
    out.append('')

    # Halves
    is_segments = segment_trades(is_trades, args.segments)
    is_tiers = sorted(set(t.get('entry_tier', '?') for t in is_trades))

    render_segment_table(f'IS ({args.segments} segments)',
                         is_segments, is_tiers, out)
    render_segment_pnl_table(f'IS ({args.segments} segments)',
                             is_segments, is_tiers, out)
    render_wr_table(f'IS ({args.segments} segments)',
                    is_segments, is_tiers, out)

    if args.also_quarters and args.segments != 4:
        q_segments = segment_trades(is_trades, 4)
        out.append('---')
        out.append('')
        render_segment_table('IS (4 quarters)', q_segments, is_tiers, out)
        render_segment_pnl_table('IS (4 quarters)', q_segments, is_tiers, out)

    # OOS reference (no segmentation)
    out.append('---')
    out.append('')
    out.append('## OOS (reference, no segmentation — too few days)')
    out.append('')
    oos_stats = tier_stats(oos_trades)
    total_pnl = sum(r['pnl'] for r in oos_stats.values())
    out.append(f'**{len(oos_trades):,} trades · ${total_pnl:+,.0f} total**')
    out.append('')
    out.append('| Tier | N | $ | $/trade | WR |')
    out.append('|---|---:|---:|---:|---:|')
    for tier in sorted(oos_stats.keys(), key=lambda k: -oos_stats[k]['pnl']):
        r = oos_stats[tier]
        out.append(f'| {tier} | {r["n"]:,} | ${r["pnl"]:+,.0f} | '
                   f'${r["per_trade"]:+.2f} | {r["wr"]:.0f}% |')
    out.append('')

    # Console summary — print the $/trade per-segment view (most actionable)
    print()
    print(f'{"Tier":<18} {"Overall":>9} ', end='')
    for s in is_segments:
        print(f'{"S" + str(s["idx"]):>9} ', end='')
    print('Pattern')
    print('-' * (30 + 11 * len(is_segments) + 10))
    full_stats = tier_stats(is_trades)
    seg_stats = [tier_stats(s['trades']) for s in is_segments]
    for tier in sorted(is_tiers, key=lambda k: -full_stats.get(k, {'pnl': 0})['pnl']):
        full = full_stats.get(tier, {'per_trade': 0, 'n': 0})
        print(f'{tier:<18} ${full["per_trade"]:>+7.2f} ', end='')
        pts = []
        for ss in seg_stats:
            r = ss.get(tier)
            if r is None or r['n'] == 0:
                print(f'{"—":>9} ', end='')
                continue
            print(f'${r["per_trade"]:>+7.2f} ', end='')
            pts.append(r['per_trade'])
        if len(pts) >= 2:
            sign_changes = sum(1 for i in range(len(pts) - 1)
                               if pts[i] * pts[i + 1] < 0)
            if sign_changes == 0:
                if all(pts[i] >= pts[i + 1] for i in range(len(pts) - 1)):
                    pat = 'DECAY'
                elif all(pts[i] <= pts[i + 1] for i in range(len(pts) - 1)):
                    pat = 'IMPROVING'
                else:
                    pat = 'stable'
            else:
                pat = f'FLIPS({sign_changes})'
            print(pat)
        else:
            print()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
