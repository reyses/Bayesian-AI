"""
BIG_LOSS Physics — when do catastrophic losers (< -$50) tip their hand?

BIG_LOSS (pnl < -$50) is the largest single drain on the engine:
~2K trades costing ~$350K. If we can detect them early via MAE or running
PnL, cutting at -$50 saves hundreds of $K vs letting them run to -$160 avg.

Analysis:
  1. Tier breakdown — which tiers produce most BIG_LOSS?
  2. Bar-N MAE distribution for BIG_LOSS vs winners vs mild losers
  3. Identify bar where BIG_LOSS MAE crosses actionable thresholds
     (e.g., -$20, -$30, -$40)
  4. Compare against non-BIG_LOSS trade MAE at same bars — the separation
     is where we can cut without hurting winners

Usage:
    python tools/big_loss_physics.py

Output: reports/findings/big_loss_physics.md
"""
import os
import sys
import pickle
import argparse
from collections import Counter
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


TRADES_PATH = 'training_iso/output/trades/iso_is.pkl'
OUT_PATH = 'reports/findings/big_loss_physics.md'

# Bar checkpoints (minutes)
BARS = [1, 2, 3, 5, 7, 10, 15, 20, 30]

# PnL thresholds to track when crossed (ticks / dollars)
MAE_THRESHOLDS = [-10, -20, -30, -40, -50]


def running_min_at_bar(trade, bar_n):
    """Min PnL seen in [bar 0, bar_n]. None if trade closed before bar_n."""
    if trade.get('held', 0) < bar_n:
        return None
    vals = [p.get('pnl', 0.0) for p in trade.get('path', []) if p.get('bar', 0) <= bar_n]
    return min(vals) if vals else None


def bar_of_first_cross_down(trade, threshold):
    """Earliest bar at which pnl <= threshold."""
    for p in trade.get('path', []):
        if p.get('pnl', 0.0) <= threshold:
            return p.get('bar', 0)
    return None


def classify(pnl):
    if pnl < -50: return 'BIG_LOSS'
    if pnl < -25: return 'MED_LOSS'
    if pnl < -10: return 'REAL_LOSS'
    if pnl < -5:  return 'MARGINAL_LOSS'
    if pnl < 0:   return 'NOISE_LOSS'
    if pnl < 5:   return 'NOISE_WIN'
    if pnl < 10:  return 'MARGINAL_WIN'
    if pnl < 25:  return 'REAL_WIN'
    if pnl < 50:  return 'STRONG_WIN'
    return 'BIG_WIN'


def main():
    print(f'Loading {TRADES_PATH}...')
    with open(TRADES_PATH, 'rb') as f:
        trades = pickle.load(f)
    print(f'  {len(trades):,} total trades')

    big_loss = [t for t in trades if t.get('pnl', 0) < -50]
    winners = [t for t in trades if t.get('pnl', 0) > 0]
    mild_loss = [t for t in trades if -50 <= t.get('pnl', 0) < 0]
    print(f'  BIG_LOSS: {len(big_loss):,}  winners: {len(winners):,}  '
          f'mild losers: {len(mild_loss):,}')

    # ── Tier breakdown ───────────────────────────────────────
    tier_counts = Counter(t.get('entry_tier', '?') for t in big_loss)
    tier_pnl = {}
    for t in big_loss:
        tier_pnl.setdefault(t.get('entry_tier', '?'), []).append(t.get('pnl', 0))

    # ── Bar-N MAE distribution per cohort ────────────────────
    def med_mae(cohort, bar_n):
        vals = [running_min_at_bar(t, bar_n) for t in cohort]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        arr = np.array(vals)
        return {
            'n': len(arr),
            'p25': float(np.percentile(arr, 25)),
            'p50': float(np.percentile(arr, 50)),
            'p75': float(np.percentile(arr, 75)),
            'p90': float(np.percentile(arr, 90)),
        }

    rows = []
    for bar in BARS:
        rows.append({
            'bar': bar,
            'big_loss': med_mae(big_loss, bar),
            'winners':  med_mae(winners, bar),
            'mild_loss': med_mae(mild_loss, bar),
        })

    # ── "When does BIG_LOSS commit" analysis ─────────────────
    # For each BIG_LOSS trade, find the earliest bar where pnl <= threshold.
    commit_stats = {}
    for thr in MAE_THRESHOLDS:
        bars_arr = [bar_of_first_cross_down(t, thr) for t in big_loss]
        bars_arr = [b for b in bars_arr if b is not None]
        pct = len(bars_arr) / max(len(big_loss), 1) * 100
        if bars_arr:
            arr = np.array(bars_arr)
            commit_stats[thr] = {
                'pct_ever_crossed': pct,
                'p25': int(np.percentile(arr, 25)),
                'p50': int(np.percentile(arr, 50)),
                'p75': int(np.percentile(arr, 75)),
                'p90': int(np.percentile(arr, 90)),
            }

    # Same analysis on winners — what % of winners dip to those thresholds?
    winner_crosses = {}
    for thr in MAE_THRESHOLDS:
        bars_arr = [bar_of_first_cross_down(t, thr) for t in winners]
        bars_arr = [b for b in bars_arr if b is not None]
        pct = len(bars_arr) / max(len(winners), 1) * 100
        winner_crosses[thr] = {'pct_ever_crossed': pct, 'n': len(bars_arr)}

    # ── Console summary ───────────────────────────────────────
    print()
    print('Tier breakdown of BIG_LOSS:')
    for tier, cnt in tier_counts.most_common():
        pnls = tier_pnl.get(tier, [0])
        print(f'  {tier:<22} {cnt:>4} trades  avg ${np.mean(pnls):+.0f}  '
              f'total ${sum(pnls):+,.0f}')

    print()
    print('Median MAE (running min) at bar N:')
    print(f'{"bar":>4} {"big_loss p50":>14} {"winners p50":>13} {"mild_loss p50":>15}')
    print('-' * 50)
    for r in rows:
        def fmt(d):
            return f'${d["p50"]:+.0f}' if d else '—'
        print(f'{r["bar"]:>4} {fmt(r["big_loss"]):>14} '
              f'{fmt(r["winners"]):>13} {fmt(r["mild_loss"]):>15}')

    print()
    print('When do BIG_LOSS trades cross MAE thresholds?')
    for thr in MAE_THRESHOLDS:
        s = commit_stats.get(thr)
        w = winner_crosses[thr]
        if s:
            print(f'  Cross ${thr}: {s["pct_ever_crossed"]:.0f}% of BIG_LOSS '
                  f'(p50 bar {s["p50"]}), '
                  f'{w["pct_ever_crossed"]:.0f}% of winners')

    # ── Write markdown ────────────────────────────────────────
    L = []
    L.append('# BIG_LOSS Physics — when does the bleed commit?')
    L.append('')
    L.append(f'BIG_LOSS (pnl < -$50): **{len(big_loss):,} trades**, total '
             f'${sum(t["pnl"] for t in big_loss):+,.0f}. Largest single drain '
             f'on engine PnL.')
    L.append('')

    L.append('## Tier source of BIG_LOSS')
    L.append('')
    L.append('| Tier | N | avg $/trade | total |')
    L.append('|---|---:|---:|---:|')
    for tier, cnt in tier_counts.most_common():
        pnls = tier_pnl.get(tier, [0])
        L.append(f'| {tier} | {cnt} | ${np.mean(pnls):+.0f} | ${sum(pnls):+,.0f} |')
    L.append('')

    L.append('## Median MAE by bar (big_loss vs winners vs mild_loss)')
    L.append('')
    L.append('| bar | BIG_LOSS MAE | Winners MAE | Mild Losers MAE | Gap (BL − W) |')
    L.append('|---:|---:|---:|---:|---:|')
    for r in rows:
        bl = r['big_loss']
        w = r['winners']
        ml = r['mild_loss']
        if bl and w:
            gap = bl['p50'] - w['p50']
            bl_s = f'${bl["p50"]:+.0f}'
            w_s = f'${w["p50"]:+.0f}'
            ml_s = f'${ml["p50"]:+.0f}' if ml else '—'
            L.append(f'| {r["bar"]} | {bl_s} | {w_s} | {ml_s} | ${gap:+.0f} |')
    L.append('')

    L.append('## When does BIG_LOSS commit? (MAE-threshold crossing)')
    L.append('')
    L.append('| MAE threshold | % BIG_LOSS crossed | Median bar | % winners crossed |')
    L.append('|---:|---:|---:|---:|')
    for thr in MAE_THRESHOLDS:
        s = commit_stats.get(thr, {'pct_ever_crossed': 0, 'p50': '—'})
        w = winner_crosses[thr]
        L.append(f'| ${thr} | {s.get("pct_ever_crossed",0):.0f}% | '
                 f'{s.get("p50","—")} | {w["pct_ever_crossed"]:.0f}% |')
    L.append('')
    L.append('**Rule candidate**: if winners rarely dip to -$X but BIG_LOSS '
             'regularly does, a hard MAE stop at -$X catches the bleeders '
             'without hitting winners.')
    L.append('')

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))
    print()
    print(f'Wrote: {OUT_PATH}')


if __name__ == '__main__':
    main()
