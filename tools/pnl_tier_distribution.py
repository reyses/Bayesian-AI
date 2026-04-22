"""
PnL distribution by tier + mode of daily PnL.

Two views on the engine:
  1. Per-day summary (overall): N days, winning/losing, accumulated,
     avg/best/worst/median, mode bucket, histogram.
  2. Per-tier summary: same block for each tier using tier-appropriate
     bucket edges (tier daily PnL is 5-10x smaller than system daily PnL).

Output format mirrors the pipeline's summary block — ASCII histogram
with '#' bars, one row per bucket.

Usage:
    python tools/pnl_tier_distribution.py                 # iso (default)
    python tools/pnl_tier_distribution.py --source blended

Output: reports/findings/pnl_tier_distribution_<source>.md (+ console)
"""
import os
import sys
import argparse
import pickle
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH = 'reports/findings/pnl_tier_distribution.md'

# Default: iso engine pickles (the active experimental frame).
# Switch to blended via CLI: --source blended
SOURCES = {
    'iso':     {'IS': 'iso_is.pkl',     'OOS': 'iso_oos.pkl'},
    'blended': {'IS': 'blended_is.pkl', 'OOS': 'blended_oos.pkl'},
}

# Overall system daily PnL buckets — wide, asymmetric, catches fat tails.
SYSTEM_BUCKETS = [
    ('<-$500',       float('-inf'),  -500.0),
    ('-$500:-$200',  -500.0,         -200.0),
    ('-$200:-$50',   -200.0,          -50.0),
    ('-$50:$0',       -50.0,            0.0),
    ('$0:$50',          0.0,           50.0),
    ('$50:$200',       50.0,          200.0),
    ('$200:$500',     200.0,          500.0),
    ('$500:$1K',      500.0,         1000.0),
    ('>$1K',         1000.0,  float('inf')),
]

# Per-tier daily PnL buckets — narrower since single tiers are ~5-10x smaller.
TIER_BUCKETS = [
    ('<-$200',       float('-inf'),  -200.0),
    ('-$200:-$100',  -200.0,         -100.0),
    ('-$100:-$50',   -100.0,          -50.0),
    ('-$50:-$20',     -50.0,          -20.0),
    ('-$20:$0',       -20.0,            0.0),
    ('$0:$20',          0.0,           20.0),
    ('$20:$50',        20.0,           50.0),
    ('$50:$100',       50.0,          100.0),
    ('$100:$200',     100.0,          200.0),
    ('>$200',         200.0,  float('inf')),
]


def bucket_of(pnl, buckets):
    for name, lo, hi in buckets:
        if lo <= pnl < hi:
            return name
    return buckets[-1][0] if pnl >= buckets[-1][1] else buckets[0][0]


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def aggregate_daily(trades):
    """Return {day: total_pnl}, {day: {tier: pnl}}, sorted day list."""
    day_total = defaultdict(float)
    day_tier = defaultdict(lambda: defaultdict(float))
    tier_n = defaultdict(int)
    tier_pnl = defaultdict(float)
    for t in trades:
        pnl = float(t.get('pnl', 0.0))
        tier = t.get('entry_tier', '?')
        day = t.get('day', 'unknown')
        day_total[day] += pnl
        day_tier[day][tier] += pnl
        tier_n[tier] += 1
        tier_pnl[tier] += pnl
    days = sorted(day_total.keys())
    return day_total, day_tier, days, tier_n, tier_pnl


def summarize(daily_pnls, buckets):
    """Given a list of daily PnLs, return summary stats dict."""
    if not daily_pnls:
        return None
    n = len(daily_pnls)
    accumulated = sum(daily_pnls)
    avg = accumulated / n
    wins = sum(1 for v in daily_pnls if v > 0)
    losses = sum(1 for v in daily_pnls if v < 0)
    flats = sum(1 for v in daily_pnls if v == 0)
    best = max(daily_pnls)
    worst = min(daily_pnls)
    med = median(daily_pnls)
    counts = defaultdict(int)
    for v in daily_pnls:
        counts[bucket_of(v, buckets)] += 1
    mode_name = max(((name, counts[name]) for name, _, _ in buckets),
                    key=lambda kv: kv[1])
    return {
        'n': n,
        'accumulated': accumulated,
        'avg': avg,
        'wins': wins,
        'losses': losses,
        'flats': flats,
        'best': best,
        'worst': worst,
        'median': med,
        'counts': dict(counts),
        'mode_name': mode_name[0],
        'mode_count': mode_name[1],
    }


def render_summary_block(label_title, summary, buckets, bar_width=40,
                         n_trades=None):
    """Return list of lines matching the blended pipeline format."""
    lines = []
    lines.append('=' * 60)
    lines.append(f'SUMMARY: {summary["n"]} days' +
                 (f' | {n_trades:,} trades' if n_trades else ''))
    lines.append('=' * 60)
    lines.append(f'  Winning days: {summary["wins"]}/{summary["n"]} '
                 f'({summary["wins"] / summary["n"] * 100:.0f}%)')
    lines.append(f'  Losing days:  {summary["losses"]}/{summary["n"]} '
                 f'({summary["losses"] / summary["n"] * 100:.0f}%)')
    if summary['flats']:
        lines.append(f'  Flat days:    {summary["flats"]}/{summary["n"]} '
                     f'({summary["flats"] / summary["n"] * 100:.0f}%)')
    lines.append(f'  Accumulated:  ${summary["accumulated"]:>12,.0f}')
    lines.append(f'  Avg $/day:    ${summary["avg"]:>12,.0f}')
    lines.append(f'  Best day:     ${summary["best"]:>12,.0f}')
    lines.append(f'  Worst day:    ${summary["worst"]:>12,.0f}')
    lines.append(f'  Median day:   ${summary["median"]:>12,.0f}')
    lines.append(f'  Mode bucket:  {summary["mode_name"]} '
                 f'({summary["mode_count"]} days)')
    lines.append('  Distribution:')
    max_count = max((summary['counts'].get(name, 0)
                     for name, _, _ in buckets), default=1)
    max_count = max(max_count, 1)
    for name, _, _ in buckets:
        c = summary['counts'].get(name, 0)
        bar_len = int(round(c / max_count * bar_width))
        bar = '#' * bar_len
        lines.append(f'    {name:>16}: {c:>3} {bar}')
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=list(SOURCES.keys()), default='iso')
    args = ap.parse_args()

    files = SOURCES[args.source]
    datasets = {k: os.path.join(TRADES_DIR, v) for k, v in files.items()}
    out_path = OUT_PATH.replace('.md', f'_{args.source}.md')

    out_md = [f'# PnL distribution by tier + mode of daily PnL — {args.source}', '']
    out_md.append(f'Source: `{args.source}` ({files["IS"]} / {files["OOS"]})')
    out_md.append('')
    out_md.append('Two views on the engine:')
    out_md.append('1. Overall daily PnL — where does a typical day cluster?')
    out_md.append('2. Per-tier daily PnL — how does each tier contribute?')
    out_md.append('')

    for label, path in datasets.items():
        if not os.path.exists(path):
            print(f'{label}: file not found, skipping')
            continue
        print(f'Loading {path}...')
        trades = load(path)
        day_total, day_tier, days, tier_n, tier_pnl = aggregate_daily(trades)

        # 1) Overall daily summary
        overall_vals = [day_total[d] for d in days]
        overall_summary = summarize(overall_vals, SYSTEM_BUCKETS)

        block_lines = render_summary_block(
            f'{label} — OVERALL', overall_summary, SYSTEM_BUCKETS,
            n_trades=len(trades))
        print()
        print(f'### {label} — OVERALL ###')
        for line in block_lines:
            print(line)

        out_md.append(f'## {label} — overall daily PnL')
        out_md.append('')
        out_md.append('```')
        out_md.extend(block_lines)
        out_md.append('```')
        out_md.append('')

        # 2) Per-tier daily summary (order by total PnL descending so heroes on top)
        tiers_sorted = sorted(tier_pnl.keys(),
                              key=lambda t: tier_pnl[t],
                              reverse=True)

        out_md.append(f'## {label} — per tier')
        out_md.append('')

        # Compact tier contribution table
        total_pnl = overall_summary['accumulated']
        out_md.append('| Tier | N trades | Total $ | % of system | $/trade | '
                      'Days traded | $/day* | Day mode |')
        out_md.append('|---|---:|---:|---:|---:|---:|---:|---|')
        for tier in tiers_sorted:
            vals = [day_tier[d].get(tier, 0.0) for d in days]
            vals_traded = [v for v in vals if v != 0]
            tsum = summarize(vals_traded, TIER_BUCKETS) if vals_traded else None
            n_traded = len(vals_traded)
            share = (tier_pnl[tier] / total_pnl * 100) if total_pnl else 0
            per_trade = tier_pnl[tier] / tier_n[tier] if tier_n[tier] else 0
            per_day = tsum['avg'] if tsum else 0
            mode = tsum['mode_name'] if tsum else '—'
            out_md.append(f'| {tier} | {tier_n[tier]:,} | '
                          f'${tier_pnl[tier]:+,.0f} | {share:+.0f}% | '
                          f'${per_trade:+.2f} | {n_traded} | '
                          f'${per_day:+.2f} | {mode} |')
        out_md.append('')

        # Full summary block per tier
        for tier in tiers_sorted:
            vals = [day_tier[d].get(tier, 0.0) for d in days]
            vals_traded = [v for v in vals if v != 0]
            if not vals_traded:
                continue
            tsum = summarize(vals_traded, TIER_BUCKETS)
            tier_block = render_summary_block(
                f'{label} — {tier}', tsum, TIER_BUCKETS,
                n_trades=tier_n[tier])
            print()
            print(f'### {label} — {tier} ###')
            for line in tier_block:
                print(line)
            out_md.append(f'### {label} — {tier}')
            out_md.append('')
            out_md.append('```')
            out_md.extend(tier_block)
            out_md.append('```')
            out_md.append('')

        out_md.append('---')
        out_md.append('')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_md))
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
