"""
Loser Cliff EDA — Q2 of three-question method applied to a tier's LOSERS.

For a given tier, split trades into winners and losers, then walk through
bar-N path points to find the natural "dead" timescale: the bar where
loser peak_pnl stalls while winners have clearly tipped their hand.

This is the rule-generator for no-progress cuts:
    if bars_held > N AND peak_pnl < $X → exit

Why this matters: the three-question peak-arrival rule works on WINNERS
(exits at feature signature of a peak). It does nothing to cut losers that
never had a peak. For tiers like KILL_SHOT (61% WR + negative $/trade),
losers run ~2× the size of winners. We need a mechanism to cut them early.

Usage:
    python tools/loser_cliff_eda.py                   # default KILL_SHOT
    python tools/loser_cliff_eda.py --tier KILL_SHOT
    python tools/loser_cliff_eda.py --tier MTF_BREAKOUT
    python tools/loser_cliff_eda.py --tier RIDE_AGAINST --trades training_iso/output/trades/iso_is.pkl

Output:
    reports/findings/loser_cliff_{TIER}.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter

import numpy as np
import pandas as pd


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES

TRADES_PATH = 'training_iso/output/trades/iso_is.pkl'
OUT_DIR = 'reports/findings'

# Bar-N sampling points (in minutes, assuming 1m entries).
BAR_NS = [5, 10, 15, 20, 30, 45, 60, 90]

# Peak thresholds to check (in $).
PEAK_THRESHOLDS = [1, 3, 5, 10, 15]

# Cliff trigger: smallest N where (loser% - winner%) >= CLIFF_DELTA_PP
# on some threshold. 15pp is the heuristic from TREND_FOLLOWER Q2 work.
CLIFF_DELTA_PP = 15.0


def peak_by_bar(trade: dict, bar_n: int) -> float | None:
    """Return running peak_pnl at bar `bar_n` in the trade path.

    Returns None if the trade closed before that bar (we don't fabricate
    extrapolated peaks).
    """
    path = trade.get('path', [])
    if not path:
        return None
    # path[i]['bar'] is the bars_held at that tick. We want the max peak_pnl
    # seen at bar <= bar_n.
    seen_bars = [p for p in path if p.get('bar', 0) <= bar_n]
    if not seen_bars:
        return None
    # Did the trade actually reach bar_n (or close after that)?
    max_bar = max(p.get('bar', 0) for p in path)
    if max_bar < bar_n:
        # Trade closed before reaching bar_n. For "fraction with peak < $X
        # at bar N" questions, these trades don't count — we exclude them.
        return None
    # Return the running peak at bar_n (the stored running max).
    return max(p.get('peak_pnl', 0.0) for p in seen_bars)


def bar_of_first_cross(trade: dict, threshold: float) -> int | None:
    """Return the earliest bar at which pnl >= threshold."""
    for p in trade.get('path', []):
        if p.get('pnl', 0.0) >= threshold:
            return p.get('bar', 0)
    return None


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Standardized mean difference. Positive = a higher than b."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def entry_feature_discrimination(winners: list, losers: list,
                                 top_k: int = 15) -> list:
    """Cohen d at entry for each of 91 features: winners vs losers.

    Returns list of (feat_name, d, winner_mean, loser_mean, split) sorted
    by |d| descending. Positive d = feature value HIGHER in winners.

    This is the entry-filter question: is there a feature that at ENTRY
    time already distinguishes which trades will win? If |d| > 0.5 on
    something clean, it's a candidate entry gate.
    """
    W = []
    L = []
    for t in winners:
        ef = t.get('entry_79d')
        if ef is not None and len(ef) >= 91:
            W.append(ef[:91])
    for t in losers:
        ef = t.get('entry_79d')
        if ef is not None and len(ef) >= 91:
            L.append(ef[:91])
    if not W or not L:
        return []
    W = np.array(W, dtype=float)
    L = np.array(L, dtype=float)

    ranked = []
    for i, name in enumerate(FEATURE_NAMES[:91]):
        w_vals = W[:, i]
        l_vals = L[:, i]
        d = cohen_d(w_vals, l_vals)
        w_mean = float(np.mean(w_vals))
        l_mean = float(np.mean(l_vals))
        # Suggested split: midpoint between means (directional gate)
        split = (w_mean + l_mean) / 2.0
        ranked.append((name, d, w_mean, l_mean, split))
    ranked.sort(key=lambda x: abs(x[1]), reverse=True)
    return ranked[:top_k]


def analyze(trades: list, tier: str) -> dict:
    sub = [t for t in trades if t.get('entry_tier') == tier]
    if not sub:
        return {'tier': tier, 'n': 0}
    winners = [t for t in sub if t.get('pnl', 0) > 0]
    losers = [t for t in sub if t.get('pnl', 0) < 0]

    held_winners = np.array([t.get('held', 0) for t in winners])
    held_losers = np.array([t.get('held', 0) for t in losers])
    peak_winners = np.array([t.get('peak', 0.0) for t in winners])
    peak_losers = np.array([t.get('peak', 0.0) for t in losers])

    # Final-realized pnl distributions (sanity).
    pnl_winners = np.array([t.get('pnl', 0.0) for t in winners])
    pnl_losers = np.array([t.get('pnl', 0.0) for t in losers])

    # Asymmetry: if |mean loser pnl| > 1.5x mean winner pnl, exits are broken.
    avg_win = float(pnl_winners.mean()) if len(pnl_winners) else 0.0
    avg_los = float(pnl_losers.mean()) if len(pnl_losers) else 0.0
    asymmetry = abs(avg_los) / max(avg_win, 1e-9) if avg_win > 0 else 0.0

    # Bar-N cliff analysis.
    rows = []
    for N in BAR_NS:
        row = {'bar_N': N}
        # Which subset reached bar N?
        reached_winners = [t for t in winners if (t.get('held', 0) >= N
                                                  and peak_by_bar(t, N) is not None)]
        reached_losers = [t for t in losers if (t.get('held', 0) >= N
                                                and peak_by_bar(t, N) is not None)]
        row['winners_reached'] = len(reached_winners)
        row['losers_reached'] = len(reached_losers)

        for thr in PEAK_THRESHOLDS:
            pct_w = (sum(1 for t in reached_winners
                         if peak_by_bar(t, N) < thr)
                     / max(len(reached_winners), 1)) * 100
            pct_l = (sum(1 for t in reached_losers
                         if peak_by_bar(t, N) < thr)
                     / max(len(reached_losers), 1)) * 100
            row[f'pct_w_lt_{thr}'] = pct_w
            row[f'pct_l_lt_{thr}'] = pct_l
            row[f'delta_lt_{thr}'] = pct_l - pct_w
        rows.append(row)

    # Winner time-to-first-cross (when do winners tip their hand?)
    first_cross_stats = {}
    for thr in PEAK_THRESHOLDS:
        crosses = [bar_of_first_cross(t, thr) for t in winners]
        crosses = [c for c in crosses if c is not None]
        if crosses:
            arr = np.array(crosses)
            first_cross_stats[thr] = {
                'n': len(arr),
                'pct_of_winners': len(arr) / len(winners) * 100,
                'p50': int(np.percentile(arr, 50)),
                'p75': int(np.percentile(arr, 75)),
                'p90': int(np.percentile(arr, 90)),
            }

    # Identify cliff: smallest N where any threshold shows delta >= CLIFF_DELTA_PP
    cliff = None
    for row in rows:
        for thr in PEAK_THRESHOLDS:
            if row[f'delta_lt_{thr}'] >= CLIFF_DELTA_PP:
                cliff = {
                    'bar_N': row['bar_N'],
                    'threshold': thr,
                    'pct_losers_below': row[f'pct_l_lt_{thr}'],
                    'pct_winners_below': row[f'pct_w_lt_{thr}'],
                    'delta_pp': row[f'delta_lt_{thr}'],
                }
                break
        if cliff:
            break

    entry_disc = entry_feature_discrimination(winners, losers)

    return {
        'tier': tier,
        'n': len(sub),
        'n_winners': len(winners),
        'n_losers': len(losers),
        'wr': len(winners) / len(sub) * 100,
        'avg_win_pnl': avg_win,
        'avg_los_pnl': avg_los,
        'asymmetry': asymmetry,
        'avg_peak_win': float(peak_winners.mean()) if len(peak_winners) else 0.0,
        'avg_peak_los': float(peak_losers.mean()) if len(peak_losers) else 0.0,
        'med_held_win': int(np.median(held_winners)) if len(held_winners) else 0,
        'med_held_los': int(np.median(held_losers)) if len(held_losers) else 0,
        'rows': rows,
        'first_cross': first_cross_stats,
        'cliff': cliff,
        'entry_discrimination': entry_disc,
    }


def print_report(res: dict):
    print()
    print(f'{"="*72}')
    print(f'LOSER CLIFF EDA — {res["tier"]}')
    print(f'{"="*72}')
    print(f'  N: {res["n"]:,}  WR: {res["wr"]:.1f}%  '
          f'winners={res["n_winners"]:,}  losers={res["n_losers"]:,}')
    print(f'  avg winner pnl: ${res["avg_win_pnl"]:+.2f}   '
          f'avg loser pnl: ${res["avg_los_pnl"]:+.2f}   '
          f'asymmetry: {res["asymmetry"]:.2f}x')
    print(f'  avg peak winner: ${res["avg_peak_win"]:+.2f}   '
          f'avg peak loser: ${res["avg_peak_los"]:+.2f}')
    print(f'  median held (m): win={res["med_held_win"]}  los={res["med_held_los"]}')

    print()
    print('Winner time-to-first-cross:')
    print(f'{"thr $":>6} {"%win":>6} {"p50":>5} {"p75":>5} {"p90":>5}')
    print('-' * 32)
    for thr, s in res['first_cross'].items():
        print(f'{thr:>6} {s["pct_of_winners"]:>5.0f}% {s["p50"]:>5} {s["p75"]:>5} {s["p90"]:>5}')

    print()
    print('Bar-N "never crossed $X" fractions (loser% - winner% delta):')
    # Header
    hdr = f'{"bar_N":>6} {"W":>5} {"L":>5} '
    for thr in PEAK_THRESHOLDS:
        hdr += f'{"<$" + str(thr):>16} '
    print(hdr)
    print('-' * len(hdr))
    for row in res['rows']:
        line = f'{row["bar_N"]:>6} {row["winners_reached"]:>5} {row["losers_reached"]:>5} '
        for thr in PEAK_THRESHOLDS:
            w = row[f'pct_w_lt_{thr}']
            l = row[f'pct_l_lt_{thr}']
            d = row[f'delta_lt_{thr}']
            marker = '*' if d >= CLIFF_DELTA_PP else ' '
            line += f'{w:>4.0f}/{l:>4.0f}({d:>+4.0f}){marker} '
        print(line)

    print()
    if res['cliff']:
        c = res['cliff']
        print(f'CLIFF DETECTED: bar >= {c["bar_N"]} AND peak_pnl < ${c["threshold"]}')
        print(f'  Captures {c["pct_losers_below"]:.0f}% of losers, '
              f'only {c["pct_winners_below"]:.0f}% of winners '
              f'({c["delta_pp"]:+.0f}pp separation)')
    else:
        print('NO CLIFF (no bar-N / threshold hit CLIFF_DELTA_PP separation).')
        print('Signal too weak — a no-progress cut likely won\'t help this tier.')

    print()
    print('Entry-feature discrimination (winners vs losers, Cohen d):')
    print(f'  Positive d = feature value HIGHER in winners at entry.')
    print(f'  |d| > 0.5 = strong separator. |d| > 0.3 = moderate.')
    print()
    print(f'{"feature":<26} {"d":>7} {"W mean":>10} {"L mean":>10} {"split":>10}')
    print('-' * 68)
    for name, d, w_mu, l_mu, split in res.get('entry_discrimination', [])[:15]:
        flag = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
        print(f'{name:<26} {d:>+7.3f} {w_mu:>10.3f} {l_mu:>10.3f} {split:>10.3f}{flag}')


def write_report(res: dict, out_path: str):
    lines = []
    tier = res['tier']
    lines.append(f'# Loser Cliff EDA — {tier}')
    lines.append('')
    lines.append(f'**Purpose:** find the bar N at which a losing trade has clearly '
                 f'failed to make progress (peak_pnl stays flat) while winners '
                 f'have tipped their hand. This gives a no-progress cut rule: '
                 f'`bars_held > N AND peak_pnl < $X → exit`.')
    lines.append('')
    lines.append('## Tier summary')
    lines.append('')
    lines.append(f'- **N:** {res["n"]:,}  (WR: {res["wr"]:.1f}%)')
    lines.append(f'- **Winners:** {res["n_winners"]:,} @ avg ${res["avg_win_pnl"]:+.2f} '
                 f'(peak avg ${res["avg_peak_win"]:+.2f})')
    lines.append(f'- **Losers:** {res["n_losers"]:,} @ avg ${res["avg_los_pnl"]:+.2f} '
                 f'(peak avg ${res["avg_peak_los"]:+.2f})')
    lines.append(f'- **Asymmetry:** {res["asymmetry"]:.2f}x '
                 f'(|loser pnl| / winner pnl — anything > 1.5 is broken exits)')
    lines.append(f'- **Median hold time:** winners {res["med_held_win"]}m, '
                 f'losers {res["med_held_los"]}m')
    lines.append('')

    lines.append('## Winner time-to-first-cross')
    lines.append('')
    lines.append('When do winners tip their hand? Bar at which `pnl >= threshold` '
                 'is first hit.')
    lines.append('')
    lines.append('| threshold | %winners | p50 | p75 | p90 |')
    lines.append('|---:|---:|---:|---:|---:|')
    for thr, s in res['first_cross'].items():
        lines.append(f'| ${thr} | {s["pct_of_winners"]:.0f}% | '
                     f'{s["p50"]}m | {s["p75"]}m | {s["p90"]}m |')
    lines.append('')

    lines.append('## Bar-N no-progress fraction')
    lines.append('')
    lines.append('For each (bar N, threshold $X), % of trades with running peak_pnl < $X '
                 'at that bar. `*` = delta ≥ 15pp (cliff candidate).')
    lines.append('')
    hdr_cells = ['bar N', 'winners reached', 'losers reached']
    for thr in PEAK_THRESHOLDS:
        hdr_cells.append(f'< ${thr} W%/L%/Δ')
    lines.append('| ' + ' | '.join(hdr_cells) + ' |')
    lines.append('|' + '|'.join(['---:'] * len(hdr_cells)) + '|')
    for row in res['rows']:
        cells = [str(row['bar_N']), str(row['winners_reached']), str(row['losers_reached'])]
        for thr in PEAK_THRESHOLDS:
            w = row[f'pct_w_lt_{thr}']
            l = row[f'pct_l_lt_{thr}']
            d = row[f'delta_lt_{thr}']
            mark = ' *' if d >= CLIFF_DELTA_PP else ''
            cells.append(f'{w:.0f}%/{l:.0f}%/{d:+.0f}pp{mark}')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines.append('')

    lines.append('## Cliff rule candidate')
    lines.append('')
    if res['cliff']:
        c = res['cliff']
        lines.append(f'**Rule:** `if bars_held >= {c["bar_N"]} AND peak_pnl < ${c["threshold"]} → exit`')
        lines.append('')
        lines.append(f'- Captures {c["pct_losers_below"]:.0f}% of losers')
        lines.append(f'- Only cuts {c["pct_winners_below"]:.0f}% of winners')
        lines.append(f'- Separation: {c["delta_pp"]:+.0f}pp')
        lines.append('')
        lines.append('Expected effect: trim the avg loser hold time, reducing '
                     'avg loser magnitude without sacrificing winner tail.')
    else:
        lines.append('**No cliff detected.** No bar N / threshold combination hit '
                     f'the {CLIFF_DELTA_PP}pp separator. Loser path is too similar '
                     'to winner path — a no-progress cut will not help this tier.')
        lines.append('')
        lines.append('Try a different Q2 axis (MAE-based cut, feature-based '
                     'thesis-validity check, or winner-exit refinement instead).')
    lines.append('')
    lines.append('## Entry-feature discrimination (Cohen d: winners vs losers)')
    lines.append('')
    lines.append('For each of the 91 features at ENTRY time, standardized mean '
                 'difference between future-winners and future-losers. '
                 'Positive d = feature value HIGHER in winners at entry. '
                 '`**` |d| ≥ 0.5 (strong, entry-gate candidate). '
                 '`*` |d| ≥ 0.3 (moderate).')
    lines.append('')
    lines.append('| feature | Cohen d | winner mean | loser mean | midpoint split |')
    lines.append('|---|---:|---:|---:|---:|')
    for name, d, w_mu, l_mu, split in res.get('entry_discrimination', [])[:15]:
        flag = ' **' if abs(d) >= 0.5 else ('  *' if abs(d) >= 0.3 else '')
        lines.append(f'| {name} | {d:+.3f}{flag} | {w_mu:.3f} | {l_mu:.3f} | {split:.3f} |')
    lines.append('')
    lines.append('**Interpretation:**')
    top = res.get('entry_discrimination', [])
    if top and abs(top[0][1]) >= 0.5:
        name, d, w_mu, l_mu, _ = top[0]
        side = 'higher' if d > 0 else 'lower'
        lines.append(f'- `{name}` separates winners from losers at entry '
                     f'(d={d:+.2f}). Winner mean ({w_mu:.3f}) vs loser mean '
                     f'({l_mu:.3f}). **Entry-gate candidate.**')
    elif top and abs(top[0][1]) >= 0.3:
        name, d, _, _, _ = top[0]
        lines.append(f'- Top separator is `{name}` at d={d:+.2f} — moderate '
                     f'signal. Could help as an entry gate but probably not a '
                     f'clean one-feature filter.')
    else:
        lines.append('- No strong entry-feature separator (|d| < 0.3 on all 91). '
                     'Winners & losers look identical at entry time. The fix '
                     'is an **exit rule**, not an entry filter.')
    lines.append('')
    lines.append('---')
    lines.append(f'_Generated by `tools/loser_cliff_eda.py --tier {tier}`_')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='KILL_SHOT', help='tier to analyze')
    ap.add_argument('--trades', default=TRADES_PATH, help='trade pickle path')
    ap.add_argument('--out', default=None, help='custom output md path')
    args = ap.parse_args()

    print(f'Loading {args.trades}...')
    with open(args.trades, 'rb') as f:
        trades = pickle.load(f)
    print(f'  {len(trades):,} total trades')
    tier_counts = Counter(t.get('entry_tier', '?') for t in trades)
    print(f'  Tier counts: {dict(tier_counts)}')
    print()

    res = analyze(trades, args.tier)
    if res['n'] == 0:
        print(f'No trades for tier {args.tier!r}. Exiting.')
        return
    print_report(res)

    out_path = args.out or os.path.join(OUT_DIR, f'loser_cliff_{args.tier}.md')
    write_report(res, out_path)
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
