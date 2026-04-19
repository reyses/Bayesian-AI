"""
PnL Mode Buckets — distribution of final $/trade across the engine.

Bucket trades by realized PnL into 10 tick-aligned ranges. Identify the
MODE (most populous bucket) and how $ contributes across buckets.

Bucket scheme (mirror of peak-bucket concept, signed):
    BIG_LOSS       < -$50
    MED_LOSS       -$50 to -$25
    REAL_LOSS      -$25 to -$10
    MARGINAL_LOSS  -$10 to -$5
    NOISE_LOSS     -$5 to 0
    NOISE_WIN      0 to +$5
    MARGINAL_WIN   +$5 to +$10
    REAL_WIN       +$10 to +$25
    STRONG_WIN     +$25 to +$50
    BIG_WIN        > +$50

Usage:
    python tools/pnl_mode_buckets.py                         # full iso_is.pkl
    python tools/pnl_mode_buckets.py --trades path/to.pkl
    python tools/pnl_mode_buckets.py --tier NMP_FADE         # filter to one tier

Output: reports/findings/pnl_mode_buckets_<tag>.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_DIR = 'reports/findings'

TV = 0.50

BUCKETS = [
    ('BIG_LOSS',      float('-inf'),  -50.0),
    ('MED_LOSS',      -50.0,          -25.0),
    ('REAL_LOSS',     -25.0,          -10.0),
    ('MARGINAL_LOSS', -10.0,           -5.0),
    ('NOISE_LOSS',     -5.0,            0.0),
    ('NOISE_WIN',       0.0,            5.0),
    ('MARGINAL_WIN',    5.0,           10.0),
    ('REAL_WIN',       10.0,           25.0),
    ('STRONG_WIN',     25.0,           50.0),
    ('BIG_WIN',        50.0, float('inf')),
]


def bucket_of(pnl):
    for name, lo, hi in BUCKETS:
        if lo <= pnl < hi:
            return name
    return 'BIG_WIN' if pnl >= 50 else 'BIG_LOSS'


def analyze(trades, label):
    total = len(trades)
    if total == 0:
        return None
    counts = Counter()
    pnl_sum = defaultdict(float)
    per_tier = defaultdict(lambda: Counter())
    for t in trades:
        pnl = t.get('pnl', 0.0)
        b = bucket_of(pnl)
        counts[b] += 1
        pnl_sum[b] += pnl
        per_tier[t.get('entry_tier', '?')][b] += 1
    mode_bucket = counts.most_common(1)[0][0]
    mode_count = counts[mode_bucket]
    total_pnl = sum(pnl_sum.values())
    return {
        'label': label,
        'n': total,
        'total_pnl': total_pnl,
        'counts': counts,
        'pnl_sum': pnl_sum,
        'per_tier': per_tier,
        'mode_bucket': mode_bucket,
        'mode_count': mode_count,
        'mode_pct': mode_count / total * 100,
    }


def write_report(result, out_path):
    L = []
    L.append(f'# PnL Mode Buckets — {result["label"]}')
    L.append('')
    L.append(f'**{result["n"]:,} trades · ${result["total_pnl"]:+,.0f} total PnL**')
    L.append('')
    L.append(f'**MODE bucket:** `{result["mode_bucket"]}` — '
             f'{result["mode_count"]:,} trades ({result["mode_pct"]:.1f}% of all).')
    L.append('')

    L.append('## Overall bucket distribution')
    L.append('')
    L.append('| Bucket | Range | N | % | Bucket PnL | $/trade |')
    L.append('|---|---|---:|---:|---:|---:|')
    for name, lo, hi in BUCKETS:
        n = result['counts'].get(name, 0)
        if n == 0:
            continue
        pct = n / result['n'] * 100
        pnl = result['pnl_sum'][name]
        per = pnl / n
        lo_s = f'${lo:.0f}' if lo != float('-inf') else '-inf'
        hi_s = f'${hi:.0f}' if hi != float('inf') else '+inf'
        mark = ' **MODE**' if name == result['mode_bucket'] else ''
        L.append(f'| {name}{mark} | {lo_s} → {hi_s} | {n:,} | '
                 f'{pct:.1f}% | ${pnl:+,.0f} | ${per:+.2f} |')
    L.append('')

    L.append('## Per-tier bucket distribution (count)')
    L.append('')
    tiers_sorted = sorted(result['per_tier'].keys())
    hdr = ['Tier'] + [n for n, _, _ in BUCKETS]
    L.append('| ' + ' | '.join(hdr) + ' |')
    L.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
    for tier in tiers_sorted:
        buckets = result['per_tier'][tier]
        total_tier = sum(buckets.values())
        cells = [tier]
        for name, _, _ in BUCKETS:
            n = buckets.get(name, 0)
            if n == 0:
                cells.append('—')
            else:
                pct = n / total_tier * 100
                cells.append(f'{n} ({pct:.0f}%)')
        L.append('| ' + ' | '.join(cells) + ' |')
    L.append('')

    L.append('## Interpretation')
    L.append('')
    mb = result['mode_bucket']
    L.append(f'- Most trades land in **{mb}** — tells us the typical '
             f'$/trade is around that range.')
    # Find largest $ contributor bucket
    largest = max(result['pnl_sum'].items(), key=lambda kv: kv[1])
    smallest = min(result['pnl_sum'].items(), key=lambda kv: kv[1])
    L.append(f'- **Biggest $ contributor bucket:** `{largest[0]}` '
             f'(${largest[1]:+,.0f}).')
    L.append(f'- **Worst $ contributor bucket:** `{smallest[0]}` '
             f'(${smallest[1]:+,.0f}).')
    L.append('')
    L.append('If the MODE bucket is NOISE_WIN/NOISE_LOSS, most trades are '
             'near-breakeven — edge comes from tail winners. If MODE is in '
             'REAL_WIN or higher, the tier systematically captures real '
             'moves. Losing-side modes (MARGINAL_LOSS, REAL_LOSS) signal '
             'a tier bleeding more often than winning.')
    L.append('')
    L.append('---')
    L.append(f'_Generated by `tools/pnl_mode_buckets.py`_')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=os.path.join(TRADES_DIR, 'iso_is.pkl'))
    ap.add_argument('--tier', default=None, help='filter to a single tier')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    print(f'Loading {args.trades}...')
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)

    if args.tier:
        trades = [t for t in trades if t.get('entry_tier') == args.tier]
        label = f'{args.tier} ({len(trades):,} trades)'
        tag = args.tier
    else:
        label = f'ALL TIERS ({len(trades):,} trades)'
        tag = 'all'

    result = analyze(trades, label)
    if result is None:
        print('No trades.')
        return

    # Console summary
    print()
    print(f'{"Bucket":<16} {"Range":<17} {"N":>6} {"%":>6} {"PnL":>12} {"$/tr":>8}')
    print('-' * 70)
    for name, lo, hi in BUCKETS:
        n = result['counts'].get(name, 0)
        if n == 0:
            continue
        pct = n / result['n'] * 100
        pnl = result['pnl_sum'][name]
        per = pnl / n
        lo_s = f'${lo:.0f}' if lo != float('-inf') else '-inf'
        hi_s = f'${hi:.0f}' if hi != float('inf') else '+inf'
        rng = f'{lo_s} -> {hi_s}'
        mark = ' *MODE*' if name == result['mode_bucket'] else ''
        print(f'{name:<16} {rng:<17} {n:>6,} {pct:>5.1f}% '
              f'${pnl:>+10,.0f} ${per:>+7.2f}{mark}')
    print()
    print(f'MODE: {result["mode_bucket"]} '
          f'({result["mode_count"]:,}/{result["n"]:,} = {result["mode_pct"]:.1f}%)')

    out_path = args.out or os.path.join(OUT_DIR, f'pnl_mode_buckets_{tag}.md')
    write_report(result, out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
