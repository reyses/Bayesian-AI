"""
Template Inspector — Load checkpoint templates and dump a human-readable report.

Usage:
    python tools/inspect_templates.py                    # default: checkpoints/templates.pkl
    python tools/inspect_templates.py --path other.pkl   # custom path
    python tools/inspect_templates.py --top 20           # show top N by member count
    python tools/inspect_templates.py --sort mfe         # sort by mean_mfe_ticks (default: members)
"""

import pickle
import sys
import os
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.fractal_discovery_agent import TIMEFRAME_SECONDS


def load_templates(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def inspect(templates, top_n=None, sort_by='members'):
    n = len(templates)
    total_members = sum(t.member_count for t in templates)

    # Sort
    if sort_by == 'mfe':
        templates = sorted(templates, key=lambda t: t.mean_mfe_ticks, reverse=True)
    elif sort_by == 'wr':
        templates = sorted(templates, key=lambda t: t.stats_win_rate, reverse=True)
    elif sort_by == 'mega':
        templates = sorted(templates, key=lambda t: t.stats_mega_rate, reverse=True)
    else:
        templates = sorted(templates, key=lambda t: t.member_count, reverse=True)

    if top_n:
        templates = templates[:top_n]

    # Global stats
    print(f"\n{'='*100}")
    print(f"TEMPLATE INSPECTION -- {n} templates, {total_members} total members")
    print(f"{'='*100}")

    # TF distribution across all templates
    tf_counter = Counter()
    for t in templates:
        for p in (t.patterns or []):
            tf = getattr(p, 'timeframe', '?')
            tf_counter[tf] += 1

    if tf_counter:
        print(f"\nMember TF distribution:")
        for tf, count in sorted(tf_counter.items(),
                                key=lambda x: TIMEFRAME_SECONDS.get(x[0], 0)):
            pct = count / total_members * 100
            bar = '#' * int(pct / 2)
            print(f"  [{tf:>4s}] {count:>6,} ({pct:5.1f}%) {bar}")

    # Direction summary
    long_templates = sum(1 for t in templates if t.long_bias > t.short_bias)
    short_templates = sum(1 for t in templates if t.short_bias > t.long_bias)
    neutral = n - long_templates - short_templates
    print(f"\nDirection: {long_templates} LONG-biased, {short_templates} SHORT-biased, {neutral} neutral")

    # Oracle quality
    wr_values = [t.stats_win_rate for t in templates if t.member_count >= 5]
    mfe_values = [t.mean_mfe_ticks for t in templates if t.member_count >= 5]
    if wr_values:
        import numpy as np
        print(f"\nOracle quality (templates with >= 5 members):")
        print(f"  Win rate:  min={min(wr_values):.1%}  med={np.median(wr_values):.1%}  max={max(wr_values):.1%}")
        print(f"  Mean MFE:  min={min(mfe_values):.1f}  med={np.median(mfe_values):.1f}  max={max(mfe_values):.1f} ticks")

    # Per-template table
    print(f"\n{'='*100}")
    header = (f"  {'ID':>4s}  {'Name':<20s}  {'N':>5s}  {'WR':>5s}  {'Mega%':>5s}  "
              f"{'MFE':>6s}  {'MAE':>6s}  {'Bias':>6s}  {'TF':>5s}  "
              f"{'AvgBar':>6s}  {'EV':>7s}  {'Risk':>5s}")
    print(header)
    print(f"  {'-'*96}")

    for t in templates:
        # Dominant TF from members
        member_tfs = Counter(getattr(p, 'timeframe', '?') for p in (t.patterns or []))
        dom_tf = member_tfs.most_common(1)[0][0] if member_tfs else '?'

        # Direction
        if t.long_bias > t.short_bias:
            bias = f'L{t.long_bias:.0%}'
        elif t.short_bias > t.long_bias:
            bias = f'S{t.short_bias:.0%}'
        else:
            bias = '  --  '

        name = (t.semantic_name or 'Unknown')[:20]

        print(f"  {t.template_id:>4d}  {name:<20s}  {t.member_count:>5d}  "
              f"{t.stats_win_rate:>5.1%}  {t.stats_mega_rate:>5.1%}  "
              f"{t.mean_mfe_ticks:>6.1f}  {t.mean_mae_ticks:>6.1f}  "
              f"{bias:>6s}  {dom_tf:>5s}  "
              f"{t.avg_mfe_bar:>6.1f}  {t.expected_value:>7.2f}  {t.risk_score:>5.2f}")

    print(f"\n{'='*100}")
    print(f"Showing {len(templates)} / {n} templates (sort: {sort_by})")

    # Duration analysis (mfe_bar * tf_seconds)
    print(f"\nDuration analysis (avg_mfe_bar * TF seconds):")
    dur_mins = []
    for t in templates:
        if t.patterns:
            tf = getattr(t.patterns[0], 'timeframe', '1m')
            tf_secs = TIMEFRAME_SECONDS.get(tf, 60)
            d = (t.avg_mfe_bar * tf_secs) / 60.0
            dur_mins.append(d)

    if dur_mins:
        import numpy as np
        print(f"  Min: {min(dur_mins):.1f}m  P25: {np.percentile(dur_mins, 25):.1f}m  "
              f"Med: {np.median(dur_mins):.1f}m  P75: {np.percentile(dur_mins, 75):.1f}m  "
              f"Max: {max(dur_mins):.1f}m")
        short = sum(1 for d in dur_mins if d < 1.0)
        medium = sum(1 for d in dur_mins if 1.0 <= d < 10.0)
        long_ = sum(1 for d in dur_mins if d >= 10.0)
        print(f"  < 1 min: {short}  |  1-10 min: {medium}  |  > 10 min: {long_}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='checkpoints/templates.pkl')
    parser.add_argument('--top', type=int, default=None, help='Show top N templates')
    parser.add_argument('--sort', default='members', choices=['members', 'mfe', 'wr', 'mega'])
    args = parser.parse_args()

    if not os.path.isfile(args.path):
        print(f"ERROR: {args.path} not found. Run --fresh first.")
        sys.exit(1)

    templates = load_templates(args.path)
    print(f"Loaded {len(templates)} templates from {args.path}")
    inspect(templates, top_n=args.top, sort_by=args.sort)
