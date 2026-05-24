"""
Tier daily concentration diagnostic — per-tier, per-day PnL Pareto.

Question: is each tier's edge spread across many days, or concentrated
in a handful of mega-days? If concentrated, the tier is regime-fragile
(a few hot days carried the whole IS number, OOS won't reproduce them).

For each tier:
  - Build the full daily-PnL series (one point per day the tier traded)
  - Compute: N days traded, mean/median/max/min per-day
  - Concentration: Top-N days' $ as % of total tier $
  - **Pareto positive**: what % of POSITIVE days produces 80% of
    positive $?  Lower = more concentrated = more fragile.
  - **Pareto abs**: what % of ALL days produces 80% of |$|?

Heuristic (rules of thumb, not hard lines):
  - Top-10 days > 50% of total $ → HIGHLY concentrated (regime-fragile)
  - Pareto 80% < 20% of days → "Pareto holds" (edge is tail-driven)
  - Pareto 80% > 40% of days → broadly distributed (robust)

Usage:
    python tools/tier_daily_concentration.py
    python tools/tier_daily_concentration.py --trades path/to.pkl

Output: reports/findings/tier_daily_concentration.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_PATH = 'reports/findings/tier_daily_concentration.md'


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_daily_series(trades):
    """Return {tier: [(day, pnl), ...]} sorted by day."""
    by_tier_day = defaultdict(lambda: defaultdict(float))
    for t in trades:
        tier = t.get('entry_tier', '?')
        day = t.get('day', 'unknown')
        by_tier_day[tier][day] += float(t.get('pnl', 0.0))
    out = {}
    for tier, day_map in by_tier_day.items():
        out[tier] = [(d, day_map[d]) for d in sorted(day_map.keys())]
    return out


def concentration_stats(daily):
    """Given list of (day, pnl), return concentration diagnostics."""
    if not daily:
        return None
    pnls = np.array([p for _, p in daily], dtype=np.float64)
    total = float(pnls.sum())
    n = len(pnls)
    pos = pnls[pnls > 0]
    neg = pnls[pnls < 0]
    pos_total = float(pos.sum()) if pos.size else 0
    neg_total = float(-neg.sum()) if neg.size else 0

    # Sort descending by $ contribution for Top-N
    sorted_desc = np.sort(pnls)[::-1]
    cum = np.cumsum(sorted_desc)
    top10_pnl = float(cum[min(9, n - 1)]) if n else 0
    top20_pnl = float(cum[min(19, n - 1)]) if n else 0

    # Pareto on positive days: fraction of positive days to reach 80% of positive $
    if pos.size > 1 and pos_total > 0:
        pos_sorted = np.sort(pos)[::-1]
        cum_pos = np.cumsum(pos_sorted)
        idx = int(np.searchsorted(cum_pos, 0.8 * pos_total, side='left'))
        pareto_pos_pct = (idx + 1) / len(pos_sorted) * 100
    else:
        pareto_pos_pct = None

    # Pareto abs: fraction of ALL days to reach 80% of abs $
    abs_pnls = np.abs(pnls)
    abs_total = float(abs_pnls.sum())
    if n > 1 and abs_total > 0:
        abs_sorted = np.sort(abs_pnls)[::-1]
        cum_abs = np.cumsum(abs_sorted)
        idx = int(np.searchsorted(cum_abs, 0.8 * abs_total, side='left'))
        pareto_abs_pct = (idx + 1) / n * 100
    else:
        pareto_abs_pct = None

    return {
        'n_days': n,
        'total': total,
        'mean': float(pnls.mean()),
        'median': float(np.median(pnls)),
        'std': float(pnls.std(ddof=0)),
        'max': float(pnls.max()),
        'min': float(pnls.min()),
        'pos_days': int((pnls > 0).sum()),
        'neg_days': int((pnls < 0).sum()),
        'flat_days': int((pnls == 0).sum()),
        'pos_total': pos_total,
        'neg_total': neg_total,
        'top10_pnl': top10_pnl,
        'top10_pct': (top10_pnl / total * 100) if total > 0 else float('nan'),
        'top20_pnl': top20_pnl,
        'top20_pct': (top20_pnl / total * 100) if total > 0 else float('nan'),
        'pareto_pos_pct': pareto_pos_pct,
        'pareto_abs_pct': pareto_abs_pct,
    }


def classify_robustness(s):
    """Human label for concentration."""
    if s is None or s['n_days'] < 10:
        return 'too-small'
    # top10_pct: if total is small or negative, this is unreliable
    if s['total'] <= 0:
        return 'net-negative'
    t10 = s['top10_pct']
    pp = s['pareto_pos_pct'] or 100.0
    if t10 >= 70 or pp <= 15:
        return 'FRAGILE'
    if t10 >= 50 or pp <= 25:
        return 'concentrated'
    if t10 >= 30 or pp <= 40:
        return 'moderate'
    return 'ROBUST'


def render_tier_table(label, per_tier_stats, out):
    out.append(f'## {label} — per-tier daily concentration')
    out.append('')
    out.append('- **Top-10 %** = top 10 profit days as % of tier total. High = fragile.')
    out.append('- **Pareto (+)** = % of *positive-$* days needed to reach 80% of positive $. Low = tail-driven.')
    out.append('- **Pareto (|$|)** = % of ALL days needed to reach 80% of |$|. Low = concentrated.')
    out.append('- **Robustness** flag: FRAGILE / concentrated / moderate / ROBUST.')
    out.append('')
    out.append('| Tier | Days | Total $ | Mean $/day | Median | Max | Min | Pos days | Top-10 $ | Top-10 % | Top-20 $ | Top-20 % | Pareto (+) | Pareto (\\|$\\|) | Robust |')
    out.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    # Sort by absolute $ impact for easier scanning
    tiers_sorted = sorted(per_tier_stats.keys(),
                          key=lambda t: -abs(per_tier_stats[t]['total']))
    for tier in tiers_sorted:
        s = per_tier_stats[tier]
        cls = classify_robustness(s)
        def f(v, fmt='+,.0f', prefix='$'):
            return '—' if v is None else f'{prefix}{v:{fmt}}'
        def fp(v):
            return '—' if v is None else f'{v:.1f}%'
        out.append(f'| {tier} | {s["n_days"]} | {f(s["total"])} | '
                   f'{f(s["mean"], "+,.2f")} | {f(s["median"], "+,.2f")} | '
                   f'{f(s["max"])} | {f(s["min"])} | '
                   f'{s["pos_days"]}/{s["n_days"]} ({s["pos_days"]/s["n_days"]*100:.0f}%) | '
                   f'{f(s["top10_pnl"])} | {fp(s["top10_pct"])} | '
                   f'{f(s["top20_pnl"])} | {fp(s["top20_pct"])} | '
                   f'{fp(s["pareto_pos_pct"])} | {fp(s["pareto_abs_pct"])} | '
                   f'**{cls}** |')
    out.append('')


def render_top_days_per_tier(label, daily_by_tier, out, top_n=10):
    out.append(f'## {label} — top-{top_n} days per tier (sanity check)')
    out.append('')
    tiers_sorted = sorted(daily_by_tier.keys(),
                          key=lambda t: -sum(p for _, p in daily_by_tier[t]))
    for tier in tiers_sorted:
        daily = daily_by_tier[tier]
        if len(daily) < top_n:
            continue
        sorted_days = sorted(daily, key=lambda kv: -kv[1])
        out.append(f'**{tier}** — top {top_n} days:')
        out.append('')
        out.append('| Day | $ |')
        out.append('|---|---:|')
        for d, p in sorted_days[:top_n]:
            out.append(f'| {d} | ${p:+,.0f} |')
        total = sum(p for _, p in daily)
        top_sum = sum(p for _, p in sorted_days[:top_n])
        pct = (top_sum / total * 100) if total > 0 else float('nan')
        out.append('')
        out.append(f'_Top-{top_n} = ${top_sum:+,.0f} of ${total:+,.0f} '
                   f'total ({pct:.0f}%)._')
        out.append('')


def print_console(label, per_tier_stats):
    print(f'\n=== {label} ===')
    print(f'{"Tier":<18} {"Days":>5} {"Total":>10} {"Mean":>8} {"Top10%":>7} '
          f'{"Par+":>5} {"Par|$|":>6} {"Robust":<12}')
    print('-' * 80)
    tiers_sorted = sorted(per_tier_stats.keys(),
                          key=lambda t: -abs(per_tier_stats[t]['total']))
    for tier in tiers_sorted:
        s = per_tier_stats[tier]
        cls = classify_robustness(s)
        def ff(v, fmt='+6.1f'):
            return '  —  ' if v is None else f'{v:{fmt}}'
        print(f'{tier:<18} {s["n_days"]:>5} ${s["total"]:>+9,.0f} '
              f'${s["mean"]:>+7.2f} {s["top10_pct"]:>6.1f}% '
              f'{ff(s["pareto_pos_pct"])}% {ff(s["pareto_abs_pct"])}% {cls}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trades', default=None,
                    help='Single pickle path. If omitted, runs both IS and OOS.')
    args = ap.parse_args()

    datasets = []
    if args.trades:
        datasets = [('DATA', args.trades)]
    else:
        datasets = [('IS',  os.path.join(TRADES_DIR, 'iso_is.pkl')),
                    ('OOS', os.path.join(TRADES_DIR, 'iso_oos.pkl'))]

    out = ['# Tier daily concentration diagnostic', '']
    out.append('For each tier, this analysis reveals how "tail-driven" the '
               'edge is. Robust tiers show consistent small-win contributions '
               'across many days; fragile tiers concentrate most of their '
               '$ in a small number of hot days.')
    out.append('')

    for label, path in datasets:
        if not os.path.exists(path):
            print(f'{label}: not found, skipping')
            continue
        print(f'Loading {path}...')
        trades = load(path)
        daily_by_tier = build_daily_series(trades)
        per_tier_stats = {t: concentration_stats(d)
                          for t, d in daily_by_tier.items()}
        print_console(label, per_tier_stats)
        render_tier_table(label, per_tier_stats, out)
        render_top_days_per_tier(label, daily_by_tier, out, top_n=10)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
