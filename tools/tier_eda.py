"""
Tier-by-tier surgical EDA — what separates winners from losers per tier.

Consumes iso_is.pkl (trades tagged with entry_tier via the no-interference
multi-engine run). Runs the same "segment / separator / peak / regime
shift" EDA we ran on KILL_SHOT, parameterized by tier name.

Segments:
  winner       pnl >=  $5
  small_loser  -$5 < pnl < $5   (break-even ± noise)
  mid_loser   -$15 <= pnl <= -$5
  tail_loser   pnl < -$15       (the "what makes bad really bad" bucket)

Outputs:
  - Console summary
  - reports/findings/tier_eda_<TIER>_<ts>.md

Usage:
    python tools/tier_eda.py --tier KILL_SHOT
    python tools/tier_eda.py --tier NMP_FADE --trades training_iso/output/trades/iso_is.pkl
    python tools/tier_eda.py --all   # loop every tier in the pickle
"""
import os
import sys
import pickle
import argparse
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core_v2.features import FEATURE_NAMES


def segment_trade(pnl):
    if pnl >= 5:    return 'winner'
    if pnl < -15:   return 'tail_loser'
    if pnl <= -5:   return 'mid_loser'
    return 'small_loser'


def summarize_segment(trades):
    if not trades:
        return None
    pnls = np.array([t['pnl'] for t in trades])
    peaks = np.array([t['peak'] for t in trades])
    held = np.array([t['held'] for t in trades])
    return {
        'n': len(trades),
        'mean_pnl': pnls.mean(),
        'total_pnl': pnls.sum(),
        'mean_peak': peaks.mean(),
        'median_peak': float(np.median(peaks)),
        'mean_held': held.mean(),
        'median_held': float(np.median(held)),
        'mean_giveback': (peaks - pnls).mean(),
    }


def feature_separation(segmented):
    winners = segmented.get('winner', [])
    tails = segmented.get('tail_loser', [])
    if len(winners) < 10 or len(tails) < 10:
        return []

    win_X = np.stack([np.asarray(t['entry_79d'][:91], dtype=np.float64) for t in winners])
    tail_X = np.stack([np.asarray(t['entry_79d'][:91], dtype=np.float64) for t in tails])

    rows = []
    for i, name in enumerate(FEATURE_NAMES[:91]):
        w, t = win_X[:, i], tail_X[:, i]
        pooled = np.sqrt((w.std()**2 + t.std()**2) / 2) + 1e-9
        d = (w.mean() - t.mean()) / pooled
        rows.append({
            'feature': name,
            'winner_median': float(np.median(w)),
            'tail_median': float(np.median(t)),
            'cohen_d': float(d),
        })
    rows.sort(key=lambda r: -abs(r['cohen_d']))
    return rows


def path_regime_shift(segmented, key_feature_indices, feature_names):
    """For each segment + each feature, mean value at entry / peak / exit."""
    out_rows = []
    for seg_name in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
        trades = segmented.get(seg_name, [])
        if not trades:
            continue
        # Collect feature vectors at entry, peak, exit
        entries, peaks, exits = [], [], []
        peak_bars = []
        peak_to_close = []
        for t in trades:
            path = t.get('path', [])
            if not path:
                continue
            pnls = np.array([p.get('pnl', 0.0) for p in path])
            peak_idx = int(np.argmax(pnls))
            peak_bars.append(peak_idx)
            peak_to_close.append(len(path) - 1 - peak_idx)

            entry_f = np.asarray(t['entry_79d'][:91], dtype=np.float64)
            entries.append(entry_f)
            pk = path[peak_idx].get('features')
            if pk is not None and len(pk) >= 91:
                peaks.append(np.asarray(pk[:91], dtype=np.float64))
            ex = path[-1].get('features')
            if ex is not None and len(ex) >= 91:
                exits.append(np.asarray(ex[:91], dtype=np.float64))
        if not entries:
            continue
        E = np.stack(entries); P = np.stack(peaks) if peaks else None
        X = np.stack(exits) if exits else None
        for idx in key_feature_indices:
            row = {
                'segment': seg_name,
                'feature': feature_names[idx],
                'entry': float(E[:, idx].mean()),
                'peak': float(P[:, idx].mean()) if P is not None else np.nan,
                'exit': float(X[:, idx].mean()) if X is not None else np.nan,
            }
            row['delta_entry_exit'] = row['exit'] - row['entry']
            out_rows.append(row)
        out_rows.append({
            'segment': seg_name,
            'feature': '__timing__',
            'entry': np.mean(peak_bars),
            'peak': np.median(peak_bars),
            'exit': np.mean(peak_to_close),
            'delta_entry_exit': np.nan,
        })
    return out_rows


def _idx(tf, slot, is_helper=False):
    if is_helper:
        return 72 + tf * 3 + slot
    return tf * 12 + slot


TF_15S, TF_1M, TF_5M, TF_15M, TF_1H, TF_1D = 0, 1, 2, 3, 4, 5


def key_feature_indices():
    """Indices to watch for regime-shift analysis: wicks, vr, p_center, z, velocity."""
    return [
        _idx(TF_1M, 2, True), _idx(TF_5M, 2, True),
        _idx(TF_15M, 2, True), _idx(TF_1H, 2, True),
        _idx(TF_1M, 2), _idx(TF_5M, 2),
        _idx(TF_1M, 9), _idx(TF_5M, 9),
        _idx(TF_1M, 0), _idx(TF_5M, 0), _idx(TF_15M, 0), _idx(TF_1H, 0),
        _idx(TF_1M, 3), _idx(TF_5M, 3),
    ]


def run_tier(trades, tier_name, out_dir):
    sub = [t for t in trades if t.get('entry_tier') == tier_name]
    print(f'\n{"="*70}')
    print(f'TIER: {tier_name}  (n={len(sub):,})')
    print(f'{"="*70}')
    if len(sub) < 30:
        print(f'  (too few trades for meaningful EDA)')
        return

    # Segment
    segmented = defaultdict(list)
    for t in sub:
        segmented[segment_trade(t['pnl'])].append(t)

    print('\nSegments:')
    print(f'  {"Segment":<13} {"N":>5} {"MeanPnL":>9} {"Total":>10} '
          f'{"MeanPeak":>10} {"MedPeak":>9} {"MeanGiveback":>13} {"MedHeld":>8}')
    for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
        s = summarize_segment(segmented.get(seg, []))
        if s is None:
            print(f'  {seg:<13} 0')
            continue
        print(f'  {seg:<13} {s["n"]:>5,} ${s["mean_pnl"]:>+7.2f} '
              f'${s["total_pnl"]:>+9,.0f} ${s["mean_peak"]:>+8.2f} '
              f'${s["median_peak"]:>+8.2f} ${s["mean_giveback"]:>+12.2f} '
              f'{s["median_held"]:>7.0f}m')

    # Exit reasons
    print('\nExit reason by segment:')
    for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
        bucket = segmented.get(seg, [])
        if not bucket:
            continue
        reasons = Counter(t.get('exit_reason', '?') for t in bucket)
        top = ', '.join(f'{r}:{c}' for r, c in reasons.most_common(4))
        print(f'  {seg:<13}: {top}')

    # Feature separators
    seps = feature_separation(segmented)
    if seps:
        print('\nTop 20 feature separators (|Cohen d| winner vs tail):')
        print(f'  {"feature":<28} {"winner_med":>12} {"tail_med":>12} {"d":>8}')
        for r in seps[:20]:
            print(f'  {r["feature"]:<28} {r["winner_median"]:>12.4f} '
                  f'{r["tail_median"]:>12.4f} {r["cohen_d"]:>+8.3f}')

    # Regime shift
    shift = path_regime_shift(segmented, key_feature_indices(), FEATURE_NAMES)
    print('\nRegime shift (entry -> peak -> exit):')
    print(f'  {"segment":<13} {"feature":<24} {"entry":>8} {"peak":>8} '
          f'{"exit":>8} {"delta":>8}')
    for r in shift:
        if r['feature'] == '__timing__':
            print(f'  {r["segment"]:<13} {"(peak_bar_median, p2c_mean)":<24} '
                  f'{r["entry"]:>8.1f} {r["peak"]:>8.1f} {r["exit"]:>8.1f}')
            continue
        print(f'  {r["segment"]:<13} {r["feature"]:<24} {r["entry"]:>8.4f} '
              f'{r["peak"]:>8.4f} {r["exit"]:>8.4f} {r["delta_entry_exit"]:>+8.4f}')

    # Write markdown report
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    md_path = os.path.join(out_dir, f'tier_eda_{tier_name}_{ts}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f'# Tier EDA — {tier_name} — {ts}\n\n')
        f.write(f'N={len(sub):,} trades.\n\n')
        f.write('## Segments\n\n')
        f.write('| Segment | N | MeanPnL | Total | MeanPeak | MedPeak | MeanGiveback | MedHeld |\n')
        f.write('|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
            s = summarize_segment(segmented.get(seg, []))
            if s is None:
                continue
            f.write(f'| {seg} | {s["n"]:,} | ${s["mean_pnl"]:+.2f} | ${s["total_pnl"]:+,.0f} | '
                    f'${s["mean_peak"]:+.2f} | ${s["median_peak"]:+.2f} | '
                    f'${s["mean_giveback"]:+.2f} | {s["median_held"]:.0f}m |\n')
        f.write('\n## Exit reasons\n\n')
        for seg in ('winner', 'small_loser', 'mid_loser', 'tail_loser'):
            bucket = segmented.get(seg, [])
            if not bucket:
                continue
            reasons = Counter(t.get('exit_reason', '?') for t in bucket)
            top = ', '.join(f'`{r}`:{c}' for r, c in reasons.most_common(5))
            f.write(f'- **{seg}**: {top}\n')
        if seps:
            f.write('\n## Top 30 separators (winner vs tail)\n\n')
            f.write('| Feature | Winner median | Tail median | Cohen d |\n')
            f.write('|---|---:|---:|---:|\n')
            for r in seps[:30]:
                f.write(f'| `{r["feature"]}` | {r["winner_median"]:.4f} | '
                        f'{r["tail_median"]:.4f} | {r["cohen_d"]:+.3f} |\n')
        f.write('\n## Regime shift (entry -> peak -> exit)\n\n')
        f.write('| Segment | Feature | Entry | Peak | Exit | Δ |\n')
        f.write('|---|---|---:|---:|---:|---:|\n')
        for r in shift:
            if r['feature'] == '__timing__':
                continue
            f.write(f'| {r["segment"]} | `{r["feature"]}` | {r["entry"]:.4f} | '
                    f'{r["peak"]:.4f} | {r["exit"]:.4f} | {r["delta_entry_exit"]:+.4f} |\n')
    print(f'\nWrote {md_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', type=str, default=None,
                    help='Tier name (KILL_SHOT, NMP_FADE, etc). Omit + use --all.')
    ap.add_argument('--all', action='store_true',
                    help='Loop every tier present in the pickle')
    ap.add_argument('--trades', default='training_iso/output/trades/iso_is.pkl')
    ap.add_argument('--out', default='reports/findings')
    args = ap.parse_args()

    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    print(f'Loaded {len(trades):,} trades from {args.trades}')
    tier_counts = Counter(t.get('entry_tier', '?') for t in trades)
    print(f'Tiers: {dict(tier_counts.most_common())}')

    if args.all:
        for tier in tier_counts.keys():
            run_tier(trades, tier, args.out)
    elif args.tier:
        run_tier(trades, args.tier, args.out)
    else:
        ap.error('need --tier NAME or --all')


if __name__ == '__main__':
    main()
