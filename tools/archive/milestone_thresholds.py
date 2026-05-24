"""
Milestone Thresholds — data-defined peak-ticks gates per bar.

Premise: rather than picking arbitrary bucket-promotion deadlines
("by bar 5 must be MARGINAL"), let the trade data define the threshold
at each bar N by finding where the winner vs loser peak-ticks
distributions separate cleanest.

Method: at each bar checkpoint N, compute the running peak_ticks for
every still-open trade. Sweep candidate thresholds T. Maximize
Youden's J = winner-retention − loser-retention (equivalently TPR − FPR).
The argmax T at each bar = physics-defined milestone.

Secondary output: temporal-interaction view showing how the winner vs
loser peak distributions evolve bar-by-bar (ASCII histogram + summary
statistics). This lets us SEE the trajectories of the two cohorts and
confirm the separator threshold is real, not an artifact of noise.

Usage:
    python tools/milestone_thresholds.py --tier KILL_SHOT
    python tools/milestone_thresholds.py --tier KILL_SHOT_INVERSE \
        --trades training_iso/output/trades/iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl

Output: reports/findings/milestone_thresholds_{TIER}.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_DIR = 'training_iso/output/trades'
OUT_DIR = 'reports/findings'

TV = 0.50

# Bar checkpoints to analyze
BARS = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30]

# Thresholds to sweep (in ticks). Coarse enough to be fast, fine enough
# to resolve the optimum.
THRESHOLD_SWEEP = list(range(0, 101, 1))

# Histogram buckets for distribution view
HIST_BUCKETS = [
    ('NOISE', 0, 5),
    ('FAKE', 5, 10),
    ('MARGINAL', 10, 20),
    ('REAL', 20, 40),
    ('STRONG', 40, 80),
    ('DOMINANT', 80, 10000),
]


def resolve_pickle(tier, explicit):
    if explicit:
        return explicit
    for cand in (
        os.path.join(TRADES_DIR, f'iso_is_{tier}.pkl'),
        os.path.join(TRADES_DIR, 'iso_is_KILL_SHOT+KILL_SHOT_INVERSE.pkl'),
        os.path.join(TRADES_DIR, 'iso_is.pkl'),
    ):
        if os.path.exists(cand):
            return cand
    return cand


def running_peak_ticks_at_bar(trade, bar_n):
    """Max peak_pnl in ticks, at bar <= bar_n. None if trade closed before bar_n."""
    if trade.get('held', 0) < bar_n:
        return None
    best = 0.0
    for p in trade.get('path', []):
        if p.get('bar', 0) <= bar_n:
            best = max(best, p.get('pnl', 0.0))
    return best / TV


def classify_bucket(peak_ticks):
    for name, lo, hi in HIST_BUCKETS:
        if lo <= peak_ticks < hi:
            return name
    return 'UNKNOWN'


def find_separator(winners_peaks, losers_peaks):
    """Sweep thresholds; return threshold that maximizes J = TPR - FPR.

    TPR = fraction of winners with peak >= T (winner retention)
    FPR = fraction of losers with peak >= T (loser retention we'd miss)

    Goal: high TPR, low FPR. Youden's J captures that balance.
    """
    if not winners_peaks or not losers_peaks:
        return None
    w = np.array(winners_peaks)
    l = np.array(losers_peaks)
    best = None
    for T in THRESHOLD_SWEEP:
        tpr = (w >= T).mean()
        fpr = (l >= T).mean()
        j = tpr - fpr
        if best is None or j > best['j']:
            best = {
                'T': T, 'j': j, 'tpr': tpr, 'fpr': fpr,
                'winners_keep': int((w >= T).sum()),
                'losers_keep':  int((l >= T).sum()),
                'winners_cut':  int((w < T).sum()),
                'losers_cut':   int((l < T).sum()),
            }
    return best


def distribution_summary(peaks_arr):
    if not peaks_arr:
        return None
    a = np.array(peaks_arr)
    return {
        'n': len(a),
        'mean': float(a.mean()),
        'median': float(np.median(a)),
        'p25': float(np.percentile(a, 25)),
        'p75': float(np.percentile(a, 75)),
        'p90': float(np.percentile(a, 90)),
    }


def bucket_distribution(peaks_arr):
    """Return %-of-trades in each bucket."""
    if not peaks_arr:
        return {}
    cnts = Counter()
    for p in peaks_arr:
        cnts[classify_bucket(p)] += 1
    total = len(peaks_arr)
    return {b: cnts[b] / total * 100 for b in [n for n, _, _ in HIST_BUCKETS]}


def analyze(trades):
    winners = [t for t in trades if t.get('pnl', 0) > 0]
    losers = [t for t in trades if t.get('pnl', 0) < 0]
    rows = []
    for bar in BARS:
        w_peaks = [running_peak_ticks_at_bar(t, bar) for t in winners]
        l_peaks = [running_peak_ticks_at_bar(t, bar) for t in losers]
        w_peaks = [p for p in w_peaks if p is not None]
        l_peaks = [p for p in l_peaks if p is not None]
        sep = find_separator(w_peaks, l_peaks)
        rows.append({
            'bar': bar,
            'n_w': len(w_peaks),
            'n_l': len(l_peaks),
            'separator': sep,
            'w_dist': distribution_summary(w_peaks),
            'l_dist': distribution_summary(l_peaks),
            'w_buckets': bucket_distribution(w_peaks),
            'l_buckets': bucket_distribution(l_peaks),
        })
    return {'rows': rows, 'n_winners': len(winners), 'n_losers': len(losers)}


def write_report(tier, result, out_path):
    L = []
    L.append(f'# Milestone Thresholds (data-defined) — {tier}')
    L.append('')
    L.append(f'**{result["n_winners"]} winners / {result["n_losers"]} losers**. '
             'For each bar N, swept peak_ticks thresholds 0-100 and picked '
             'the value that maximizes winner-retention − loser-retention '
             "(Youden's J). That threshold is the data-defined milestone.")
    L.append('')

    # ── Part 1: Milestone table ────────────────────────────────────
    L.append('## Part 1 — Physics-defined milestones')
    L.append('')
    L.append('| bar | n_w | n_l | milestone (ticks) | $ equiv | W keep % | L cut % | J | cuts |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for r in result['rows']:
        sep = r['separator']
        if sep is None:
            continue
        dollar = sep['T'] * TV
        wkeep = sep['tpr'] * 100
        lcut = (1 - sep['fpr']) * 100
        cuts = f"{sep['losers_cut']}L / {sep['winners_cut']}W cut"
        L.append(f'| {r["bar"]} | {r["n_w"]} | {r["n_l"]} | '
                 f'**{sep["T"]}** | ${dollar:.2f} | {wkeep:.0f}% | '
                 f'{lcut:.0f}% | {sep["j"]:.2f} | {cuts} |')
    L.append('')
    L.append('**Read**: at bar N, if peak_ticks < milestone, cut. Keeps the '
             'W keep % of winners, cuts the L cut % of losers.')
    L.append('')

    # ── Part 2: Temporal interaction view ──────────────────────────
    L.append('## Part 2 — Peak distribution by bar (temporal slices)')
    L.append('')
    L.append('Bucket % of still-open trades at each bar. Winners rise through '
             'buckets over time; losers stagnate in NOISE/FAKE.')
    L.append('')
    for cohort_name, key in [('WINNERS', 'w_buckets'), ('LOSERS', 'l_buckets')]:
        L.append(f'### {cohort_name}')
        L.append('')
        hdr = ['bar', 'n'] + [n for n, _, _ in HIST_BUCKETS]
        L.append('| ' + ' | '.join(hdr) + ' |')
        L.append('|' + '|'.join(['---:'] * len(hdr)) + '|')
        for r in result['rows']:
            n = r['n_w'] if cohort_name == 'WINNERS' else r['n_l']
            if n == 0:
                continue
            cells = [str(r['bar']), str(n)]
            for bucket_name, _, _ in HIST_BUCKETS:
                pct = r[key].get(bucket_name, 0)
                cells.append(f'{pct:.0f}%')
            L.append('| ' + ' | '.join(cells) + ' |')
        L.append('')

    # ── Part 3: Distribution statistics ────────────────────────────
    L.append('## Part 3 — Peak-ticks distribution statistics')
    L.append('')
    L.append('| bar | W p25 | W p50 | W p75 | L p25 | L p50 | L p75 |')
    L.append('|---:|---:|---:|---:|---:|---:|---:|')
    for r in result['rows']:
        w = r['w_dist']
        l = r['l_dist']
        if w is None or l is None:
            continue
        L.append(f'| {r["bar"]} | {w["p25"]:.0f} | {w["median"]:.0f} | '
                 f'{w["p75"]:.0f} | {l["p25"]:.0f} | {l["median"]:.0f} | '
                 f'{l["p75"]:.0f} |')
    L.append('')

    # ── Part 4: Proposed rule set ──────────────────────────────────
    L.append('## Part 4 — Proposed multi-milestone exit rules')
    L.append('')
    L.append('```python')
    L.append('# Physics-defined milestones — at each bar, if peak_ticks is')
    L.append('# below the data-separator threshold, the trade is statistically')
    L.append('# unlikely to become a winner. Cut it.')
    for r in result['rows']:
        sep = r['separator']
        if sep is None:
            continue
        if sep['j'] < 0.10:
            note = '  # weak separator — probably skip'
        elif sep['j'] < 0.20:
            note = '  # moderate — consider'
        else:
            note = '  # strong separator'
        L.append(f"if bars_held >= {r['bar']} and peak_ticks < {sep['T']}:"
                 f"{note}")
        L.append(f"    return 'milestone_bar{r['bar']}_cut'")
    L.append('```')
    L.append('')
    L.append('Strong rules (J ≥ 0.20) should be wired. Weak rules (J < 0.10) '
             'are noise. Moderate rules (0.10-0.20) go in via A/B if '
             'portfolio impact is positive.')
    L.append('')
    L.append('---')
    L.append(f'_Generated by `tools/milestone_thresholds.py --tier {tier}`_')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', required=True)
    ap.add_argument('--trades', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    pkl = resolve_pickle(args.tier, args.trades)
    print(f'Loading {pkl}...')
    with open(pkl, 'rb') as f:
        trades = pickle.load(f)
    sub = [t for t in trades if t.get('entry_tier') == args.tier]
    if not sub:
        print(f'No trades for {args.tier}')
        return
    print(f'  {args.tier}: {len(sub)} trades')

    print('Sweeping thresholds per bar...')
    result = analyze(sub)
    for r in result['rows']:
        sep = r['separator']
        if sep:
            print(f"  bar {r['bar']:>3}: T={sep['T']:>3} ticks  "
                  f"J={sep['j']:.2f}  W keep {sep['tpr']*100:.0f}%  "
                  f"L cut {(1-sep['fpr'])*100:.0f}%")

    out_path = args.out or os.path.join(OUT_DIR, f'milestone_thresholds_{args.tier}.md')
    write_report(args.tier, result, out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
