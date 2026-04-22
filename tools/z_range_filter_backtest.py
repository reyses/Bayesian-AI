"""
z_range_filter research — validate whether filtering trades by 1h z-range
at entry improves blended forward-pass PnL.

Research question: does rejecting entries when (1h_z_high - 1h_z_low) is
wide, AND scaling contract size when moderate, actually produce positive
lift on IS + OOS historical data?

Rules tested:
  1. HARD FILTER: reject trades where 1h_z_range >= reject_threshold
  2. SIZING: for kept trades, cap contracts based on range
  3. COMBINED: both applied

Bucket scheme: PER-TIER PERCENTILE DECILES.
For each tier, the baseline PnL distribution is split into 10 equal-count
buckets (D1 = worst 10% for THAT tier, D10 = best 10%). Same $ boundaries
are applied to filter+sizing trades so we can read bucket shifts as
"percent of originally-bottom-10% trades the filter rejected" etc.

This is more honest than fixed $ cutoffs because a -$50 loss is a
catastrophe for FADE_CALM (avg -$0.10/trade) but routine for FREIGHT_TRAIN
(avg -$14.50/trade).

Tiers with fewer than MIN_TIER_N baseline trades are skipped from the
decile analysis (not enough data to define deciles reliably).

Usage:
    python tools/z_range_filter_backtest.py
    python tools/z_range_filter_backtest.py --reject 2.5 --size1 2.0 --size2 1.5

Output: reports/findings/z_range_filter_backtest.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import FEATURE_NAMES


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH = 'reports/findings/z_range_filter_backtest.md'

DECILE_LABELS = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10']
MIN_TIER_N = 40  # below this the deciles aren't meaningful


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_z_range(trades):
    """Return list of (z_range, pnl, tier, original_contracts) per trade."""
    idx_hi = None
    idx_lo = None
    for i, name in enumerate(FEATURE_NAMES[:91]):
        if name == '1h_z_high':
            idx_hi = i
        elif name == '1h_z_low':
            idx_lo = i
    assert idx_hi is not None and idx_lo is not None, '1h z bounds not found'

    rows = []
    for t in trades:
        ef = t.get('entry_79d')
        if ef is None or len(ef) < 91:
            continue
        z_hi = float(ef[idx_hi])
        z_lo = float(ef[idx_lo])
        z_range = z_hi - z_lo
        rows.append({
            'z_range': z_range,
            'pnl': float(t.get('pnl', 0.0)),
            'tier': t.get('entry_tier', '?'),
            'contracts': int(t.get('contracts', 1)),
        })
    return rows


def compute_tier_deciles(rows):
    """Given baseline rows, compute 9 decile thresholds per tier.
    Returns {tier: np.array of 9 thresholds (P10..P90)}."""
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r['tier']].append(r['pnl'])
    out = {}
    for tier, pnls in by_tier.items():
        if len(pnls) < MIN_TIER_N:
            continue
        arr = np.asarray(pnls, dtype=np.float64)
        thresholds = np.percentile(arr, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        out[tier] = thresholds
    return out


def decile_index(pnl, thresholds):
    """Return 0..9 decile index for a trade PnL given tier's 9 thresholds."""
    # thresholds: P10, P20, ..., P90 (length 9)
    # D1 = pnl <= P10, D2 = P10 < pnl <= P20, ..., D10 = pnl > P90
    for i, t in enumerate(thresholds):
        if pnl <= t:
            return i
    return 9


def simulate(rows, tier_thresholds, reject_thr=None, size_tiers=None):
    """Apply filter + sizing. Classify trades into per-tier deciles using
    the supplied tier_thresholds (always the BASELINE thresholds so shifts
    are comparable)."""
    per_tier_decile_n = defaultdict(lambda: [0] * 10)
    per_tier_decile_pnl = defaultdict(lambda: [0.0] * 10)
    per_tier_pnl = defaultdict(float)
    per_tier_kept = Counter()
    total_pnl = 0.0
    kept = 0
    rejected = 0
    wins = 0
    skipped_tiers = Counter()
    for r in rows:
        z = r['z_range']
        if reject_thr is not None and z >= reject_thr:
            rejected += 1
            continue
        scale = 1.0
        if size_tiers is not None:
            orig = max(1, r['contracts'])
            rec_max = orig
            for thr, cap in size_tiers:
                if z >= thr:
                    rec_max = min(rec_max, cap)
            scale = rec_max / orig
        adj_pnl = r['pnl'] * scale
        tier = r['tier']
        if tier in tier_thresholds:
            d = decile_index(adj_pnl, tier_thresholds[tier])
            per_tier_decile_n[tier][d] += 1
            per_tier_decile_pnl[tier][d] += adj_pnl
        else:
            skipped_tiers[tier] += 1
        per_tier_pnl[tier] += adj_pnl
        per_tier_kept[tier] += 1
        total_pnl += adj_pnl
        kept += 1
        if adj_pnl > 0:
            wins += 1
    return {
        'total_pnl': total_pnl,
        'kept': kept,
        'rejected': rejected,
        'wins': wins,
        'per_tier_decile_n': {k: list(v) for k, v in per_tier_decile_n.items()},
        'per_tier_decile_pnl': {k: list(v) for k, v in per_tier_decile_pnl.items()},
        'per_tier_pnl': dict(per_tier_pnl),
        'per_tier_kept': dict(per_tier_kept),
        'skipped_tiers': dict(skipped_tiers),
    }


def headline(label, baseline, result):
    n = result['kept']
    rej = result['rejected']
    pnl = result['total_pnl']
    w = result['wins']
    wr = (w / n * 100) if n else 0
    delta = pnl - baseline['total_pnl']
    return (f'{label:<22} kept={n:>5}  rej={rej:>5}  '
            f'$={pnl:>+10,.0f}  d={delta:>+9,.0f}  WR={wr:.0f}%')


def fmt_range(lo, hi):
    lo_s = '-inf' if lo is None else f'${lo:+.2f}'
    hi_s = '+inf' if hi is None else f'${hi:+.2f}'
    return f'{lo_s} -> {hi_s}'


def render_tier_decile_legend(tier_thresholds, out_lines, title):
    """Show the $ boundaries per tier so the reader knows what D1..D10 mean."""
    out_lines.append(f'### {title}')
    out_lines.append('')
    out_lines.append('Per-tier $ boundaries for deciles D1 (worst 10%) .. D10 (best 10%), '
                     'computed from that tier\'s BASELINE PnL distribution.')
    out_lines.append('')
    hdr = ['Tier'] + DECILE_LABELS
    out_lines.append('| ' + ' | '.join(hdr) + ' |')
    out_lines.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
    for tier in sorted(tier_thresholds.keys()):
        thr = tier_thresholds[tier]
        # Edges: (-inf, thr[0]], (thr[0], thr[1]], ..., (thr[8], +inf)
        edges = [(-np.inf, thr[0])]
        for i in range(len(thr) - 1):
            edges.append((thr[i], thr[i + 1]))
        edges.append((thr[-1], np.inf))
        cells = [tier]
        for lo, hi in edges:
            if lo == -np.inf:
                cells.append(f'<=${hi:+.1f}')
            elif hi == np.inf:
                cells.append(f'>${lo:+.1f}')
            else:
                cells.append(f'${lo:+.1f}..${hi:+.1f}')
        out_lines.append('| ' + ' | '.join(cells) + ' |')
    out_lines.append('')


def render_tier_decile_table(result, out_lines, title, show_pnl=False):
    """Per-tier % distribution OR $ contribution across the 10 deciles."""
    out_lines.append(f'### {title}')
    out_lines.append('')
    if show_pnl:
        out_lines.append('Dollar contribution per decile.')
    else:
        out_lines.append('Percent of kept trades in each decile (using the BASELINE '
                         'thresholds). Baseline is 10% per decile by construction.')
    out_lines.append('')
    hdr = ['Tier', 'N', '$'] + DECILE_LABELS
    out_lines.append('| ' + ' | '.join(hdr) + ' |')
    out_lines.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
    for tier in sorted(result['per_tier_decile_n'].keys()):
        dn = result['per_tier_decile_n'][tier]
        dp = result['per_tier_decile_pnl'][tier]
        total_n = sum(dn)
        tier_pnl = result['per_tier_pnl'].get(tier, 0.0)
        cells = [tier, f'{total_n:,}', f'${tier_pnl:+,.0f}']
        for i in range(10):
            if show_pnl:
                cells.append('—' if dp[i] == 0 else f'${dp[i]:+,.0f}')
            else:
                pct = (dn[i] / total_n * 100) if total_n else 0
                cells.append('—' if pct == 0 else f'{pct:.0f}%')
        out_lines.append('| ' + ' | '.join(cells) + ' |')
    out_lines.append('')


def render_decile_delta_table(baseline, filtered, out_lines, title):
    """Percentage-point and $ shift per decile (filtered vs baseline)."""
    out_lines.append(f'### {title}')
    out_lines.append('')
    out_lines.append('Decile-share shift in pp (kept trades) and $ shift. '
                     'Negative pp on D1 = filter removed bottom-10% trades '
                     '(good). Negative pp on D10 = filter removed top-10% '
                     'trades (bad). A sign-asymmetric row is a real signal; '
                     'a symmetric row (all small shifts, or D1 and D10 both '
                     'down) is a variance cutter, not a loss filter.')
    out_lines.append('')
    hdr = ['Tier', 'dN', 'd$'] + DECILE_LABELS
    out_lines.append('| ' + ' | '.join(hdr) + ' |')
    out_lines.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
    tiers = sorted(set(baseline['per_tier_decile_n'].keys())
                   | set(filtered['per_tier_decile_n'].keys()))
    for tier in tiers:
        b_dn = baseline['per_tier_decile_n'].get(tier, [0] * 10)
        f_dn = filtered['per_tier_decile_n'].get(tier, [0] * 10)
        b_total = sum(b_dn)
        f_total = sum(f_dn)
        dn = f_total - b_total
        b_pnl = baseline['per_tier_pnl'].get(tier, 0.0)
        f_pnl = filtered['per_tier_pnl'].get(tier, 0.0)
        d_pnl = f_pnl - b_pnl
        cells = [tier, f'{dn:+,}', f'${d_pnl:+,.0f}']
        for i in range(10):
            b_p = (b_dn[i] / b_total * 100) if b_total else 0
            f_p = (f_dn[i] / f_total * 100) if f_total else 0
            d_p = f_p - b_p
            if abs(d_p) < 0.5:
                cells.append('.')
            else:
                cells.append(f'{d_p:+.0f}pp')
        out_lines.append('| ' + ' | '.join(cells) + ' |')
    out_lines.append('')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reject', type=float, default=2.5)
    ap.add_argument('--size1',  type=float, default=2.0, help='>=this: 1 contract')
    ap.add_argument('--size2',  type=float, default=1.5, help='>=this: 2 contracts')
    args = ap.parse_args()

    size_tiers = [(args.size1, 1), (args.size2, 2)]
    size_tiers.sort(key=lambda kv: -kv[0])

    datasets = {
        'IS':  os.path.join(TRADES_DIR, 'blended_is.pkl'),
        'OOS': os.path.join(TRADES_DIR, 'blended_oos.pkl'),
    }

    out_md = ['# z_range filter backtest — per-tier decile view', '']
    out_md.append(f'Thresholds: reject >= {args.reject}, size1 >= {args.size1}, '
                  f'size2 >= {args.size2}')
    out_md.append('')
    out_md.append('**Bucket scheme**: per-tier percentile deciles. For each tier, '
                  'baseline PnL is split into 10 equal-count buckets:')
    out_md.append('- **D1** = worst 10% of trades for THAT tier')
    out_md.append('- **D10** = best 10% of trades for THAT tier')
    out_md.append('- Baseline uniformly 10% per decile by construction')
    out_md.append('- Filter runs reuse the baseline thresholds so shifts are comparable')
    out_md.append('')
    out_md.append(f'Tiers with fewer than {MIN_TIER_N} baseline trades are skipped.')
    out_md.append('')

    for label, path in datasets.items():
        if not os.path.exists(path):
            print(f'{label}: file not found, skipping')
            continue
        print(f'\n=== {label} ({path}) ===')
        trades = load(path)
        rows = extract_z_range(trades)
        print(f'{len(rows):,} trades with entry features')

        tier_thresholds = compute_tier_deciles(rows)
        skipped = [t for t in set(r['tier'] for r in rows) if t not in tier_thresholds]

        baseline = simulate(rows, tier_thresholds)
        print(headline('BASELINE (no filter)', baseline, baseline))

        filt_only = simulate(rows, tier_thresholds, reject_thr=args.reject)
        print(headline(f'Filter >= {args.reject}', baseline, filt_only))

        size_only = simulate(rows, tier_thresholds, size_tiers=size_tiers)
        print(headline('Sizing only', baseline, size_only))

        combined = simulate(rows, tier_thresholds, reject_thr=args.reject,
                            size_tiers=size_tiers)
        print(headline('Filter + Sizing', baseline, combined))

        out_md.append(f'## {label}')
        out_md.append('')
        out_md.append('| Strategy | Kept | Rejected | PnL | d vs Baseline | WR |')
        out_md.append('|---|---:|---:|---:|---:|---:|')
        for name, res in [('Baseline', baseline),
                          (f'Filter >= {args.reject}', filt_only),
                          ('Sizing only', size_only),
                          ('Filter + Sizing', combined)]:
            delta = res['total_pnl'] - baseline['total_pnl']
            wr = (res['wins'] / max(res['kept'], 1)) * 100
            out_md.append(f'| {name} | {res["kept"]:,} | {res["rejected"]:,} | '
                          f'${res["total_pnl"]:+,.0f} | ${delta:+,.0f} | '
                          f'{wr:.0f}% |')
        out_md.append('')

        if skipped:
            out_md.append(f'Skipped (N < {MIN_TIER_N}): ' + ', '.join(skipped))
            out_md.append('')

        render_tier_decile_legend(tier_thresholds, out_md,
                                  f'{label} — per-tier decile $ boundaries')
        render_tier_decile_table(baseline, out_md,
                                 f'{label} — baseline decile share (sanity check: ~10% each)')
        render_tier_decile_table(combined, out_md,
                                 f'{label} — filter+sizing decile share')
        render_tier_decile_table(combined, out_md,
                                 f'{label} — filter+sizing decile $ contribution',
                                 show_pnl=True)
        render_decile_delta_table(baseline, combined, out_md,
                                  f'{label} — per-tier decile SHIFT (filter+sizing vs baseline)')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out_md))
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
