"""Run regret analysis on all 8 isolated tier pickles.

For each tier (from tools/run_tier_isolated.py output), computes:
  - actual $/trade, $/day
  - optimal $/trade (perfect counterfactual)
  - capture % = actual / optimal
  - % trades where best_action = counter* (direction edge test)
  - if we FOLLOW regret labels blindly, what's the upper-bound $/day?

This tells us which tiers deserve CNN training (real direction edge,
low capture) vs which are dead (50/50 coin flip, no edge to extract).

Outputs:
  - reports/findings/2026-04-17_iso_regret.md
  - reports/findings/iso_regret_per_tier.csv
  - training/output/isolated/regret/{TIER}.csv  (for later CNN training)

Usage:
    python tools/regret_on_isolated.py
    python tools/regret_on_isolated.py KILL_SHOT FADE_CALM
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.regret import compute_all_regrets

ISO_DIR = 'training/output/isolated'
OUT_DIR = 'training/output/isolated/regret'
OUT_MD = 'reports/findings/2026-04-17_iso_regret.md'
OUT_CSV = 'reports/findings/iso_regret_per_tier.csv'

ALL_TIERS = ['FADE_CALM', 'RIDE_AGAINST', 'KILL_SHOT', 'CASCADE',
             'FADE_AGAINST', 'MTF_BREAKOUT', 'MTF_EXHAUSTION', 'FREIGHT_TRAIN']


def prep_trades(trades):
    """regret.py reads t['timestamp']; our isolated trades use entry_ts."""
    for t in trades:
        if 'timestamp' not in t or not t['timestamp']:
            t['timestamp'] = t.get('entry_ts', 0)
    return trades


def tier_summary(tier, trades, regret_df):
    """Per-tier rollup dict."""
    n = len(trades)
    if n == 0:
        return None
    days = len(set(t.get('day', '') for t in trades))
    actual = regret_df['actual_pnl'].sum()
    optimal = regret_df['best_pnl'].sum()
    capture = actual / max(abs(optimal), 1) * 100 if optimal > 0 else 0.0

    counter_mask = regret_df['best_action'].str.contains('counter', na=False)
    n_counter = int(counter_mask.sum())
    pct_counter = n_counter / n * 100

    # What PnL would we get if we FLIPPED the losers that regret says flip?
    # i.e., follow regret labels blindly (oracle upper bound)
    # best_pnl is the best counterfactual magnitude — summing it gives optimal
    # More useful: if we kept SAME-labeled trades as-is and flipped COUNTER-labeled
    same_df = regret_df[~counter_mask]
    counter_df = regret_df[counter_mask]
    # SAME kept at actual, COUNTER flipped to counter_best (magnitude from entry)
    if_flipped_losers = same_df['actual_pnl'].sum() + counter_df['counter_best'].sum()

    # More conservative: instead of counter_best (peak), use counter_at_exit
    # (just flip direction, keep same hold time)
    if_flipped_at_exit = same_df['actual_pnl'].sum() + counter_df['counter_at_exit'].sum()

    # WR on COUNTER trades (are they actual losers?)
    counter_trades = [trades[i] for i in range(n) if counter_mask.iloc[i]]
    counter_wr = (sum(1 for t in counter_trades if t['pnl'] > 0) /
                  max(len(counter_trades), 1) * 100)

    # avg regret by action
    action_breakdown = regret_df['best_action'].value_counts().to_dict()

    return {
        'tier': tier,
        'n_trades': n,
        'n_days': days,
        'actual_total': actual,
        'actual_per_day': actual / max(days, 1),
        'actual_per_trade': actual / max(n, 1),
        'optimal_total': optimal,
        'optimal_per_day': optimal / max(days, 1),
        'capture_pct': capture,
        'n_counter': n_counter,
        'pct_counter': pct_counter,
        'counter_wr': counter_wr,
        'if_flipped_peak': if_flipped_losers,
        'if_flipped_peak_per_day': if_flipped_losers / max(days, 1),
        'if_flipped_at_exit': if_flipped_at_exit,
        'if_flipped_at_exit_per_day': if_flipped_at_exit / max(days, 1),
        'avg_regret': regret_df['regret'].mean(),
        'action_breakdown': action_breakdown,
    }


def main():
    tiers_arg = sys.argv[1:] if len(sys.argv) > 1 else ALL_TIERS
    tiers = [t for t in tiers_arg if t in ALL_TIERS]
    if not tiers:
        print(f'No valid tiers. Options: {ALL_TIERS}')
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)

    summaries = []

    for tier in tiers:
        pkl = os.path.join(ISO_DIR, f'{tier}.pkl')
        if not os.path.exists(pkl):
            print(f'  [skip] {tier}: no pkl at {pkl}')
            continue

        with open(pkl, 'rb') as f:
            trades = pickle.load(f)

        # Primary only — chains aren't entries, they're re-entries
        trades = [t for t in trades if not t.get('is_chain', False)]
        if len(trades) < 20:
            print(f'  [skip] {tier}: only {len(trades)} primary trades')
            continue

        trades = prep_trades(trades)
        print(f'\n{"="*60}')
        print(f'{tier}: {len(trades):,} primary trades')
        print(f'{"="*60}')

        regret_df = compute_all_regrets(trades, price_dir='DATA/ATLAS/1m')

        # Save per-tier regret df
        out_csv = os.path.join(OUT_DIR, f'{tier}.csv')
        regret_df.to_csv(out_csv, index=False)
        print(f'  Regret CSV: {out_csv}')

        s = tier_summary(tier, trades, regret_df)
        summaries.append(s)

        print(f'  Actual:  ${s["actual_total"]:+,.0f}  '
              f'(${s["actual_per_day"]:+.0f}/day, '
              f'${s["actual_per_trade"]:+.2f}/trade)')
        print(f'  Optimal: ${s["optimal_total"]:+,.0f}  '
              f'(${s["optimal_per_day"]:+.0f}/day)')
        print(f'  Capture: {s["capture_pct"]:.1f}%')
        print(f'  % best=counter: {s["pct_counter"]:.1f}%  '
              f'(n={s["n_counter"]})')
        print(f'  WR of counter-labeled trades: {s["counter_wr"]:.0f}% '
              f'(should be <<50% if regret is real)')
        print(f'  If we flipped counter-labeled at peak: '
              f'${s["if_flipped_peak"]:+,.0f}  '
              f'(${s["if_flipped_peak_per_day"]:+.0f}/day)')
        print(f'  If we flipped counter-labeled at exit: '
              f'${s["if_flipped_at_exit"]:+,.0f}  '
              f'(${s["if_flipped_at_exit_per_day"]:+.0f}/day)')

    if not summaries:
        print('No tiers analyzed.')
        return

    # --- Write MD report ---
    df = pd.DataFrame([{k: v for k, v in s.items() if k != 'action_breakdown'}
                       for s in summaries])
    df = df.sort_values('if_flipped_peak_per_day', ascending=False)
    df.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append('# Regret Analysis on Isolated Tiers\n')
    lines.append(f'Date: 2026-04-17\n')
    lines.append(f'Source: `{ISO_DIR}/*.pkl` (from `tools/run_tier_isolated.py`)\n\n')
    lines.append('## How to read this\n')
    lines.append('- **Actual**: what the tier made as-is (from isolated run).\n')
    lines.append('- **Optimal**: sum of best counterfactual per trade — perfect oracle.\n')
    lines.append('- **Capture**: actual/optimal. Low = exits leak, or direction wrong.\n')
    lines.append('- **% counter**: share of trades where regret says flip direction.\n'
                 '  - ~50% = tier has no directional edge (coin flip). DEAD.\n'
                 '  - 25-35% = real edge, CNN can extract it.\n'
                 '  - <20% = direction mostly right; problem is exits.\n')
    lines.append('- **Counter WR**: actual win rate of counter-labeled trades. '
                 'If regret is sound, should be much lower than 50%.\n')
    lines.append('- **If flipped at peak**: upper-bound oracle $/day if we obeyed '
                 'regret labels perfectly (peak capture on counters).\n')
    lines.append('- **If flipped at exit**: realistic bound — flip direction but '
                 "keep the same exit timing (no peak-chasing).\n\n")

    lines.append('## Per-tier summary\n\n')
    lines.append('| Tier | N | $/day actual | $/day optimal | Capture % | % counter | Counter WR | $/day if flip@peak | $/day if flip@exit |\n')
    lines.append('|---|---|---|---|---|---|---|---|---|\n')
    for _, r in df.iterrows():
        lines.append(f'| {r["tier"]} | {r["n_trades"]:,} | '
                     f'${r["actual_per_day"]:+.0f} | '
                     f'${r["optimal_per_day"]:+.0f} | '
                     f'{r["capture_pct"]:.0f}% | '
                     f'{r["pct_counter"]:.0f}% | '
                     f'{r["counter_wr"]:.0f}% | '
                     f'${r["if_flipped_peak_per_day"]:+.0f} | '
                     f'${r["if_flipped_at_exit_per_day"]:+.0f} |\n')

    lines.append('\n## Verdict per tier (my read)\n\n')
    for _, r in df.iterrows():
        verdict = _verdict(r)
        lines.append(f'- **{r["tier"]}**: {verdict}\n')

    lines.append('\n## Best-action breakdown per tier\n\n')
    for s in summaries:
        lines.append(f'### {s["tier"]}\n')
        total = sum(s['action_breakdown'].values())
        for action, cnt in sorted(s['action_breakdown'].items(),
                                   key=lambda x: -x[1]):
            lines.append(f'- {action}: {cnt} ({cnt/total*100:.0f}%)\n')
        lines.append('\n')

    lines.append('\n## Aggregate — all tiers pooled\n\n')
    total_trades = sum(s['n_trades'] for s in summaries)
    total_days = max(s['n_days'] for s in summaries)
    total_actual = sum(s['actual_total'] for s in summaries)
    total_optimal = sum(s['optimal_total'] for s in summaries)
    total_if_flip_peak = sum(s['if_flipped_peak'] for s in summaries)
    total_if_flip_exit = sum(s['if_flipped_at_exit'] for s in summaries)
    lines.append(f'- Trades: {total_trades:,} across {total_days} days\n')
    lines.append(f'- Actual: ${total_actual:+,.0f} (${total_actual/max(total_days,1):+.0f}/day)\n')
    lines.append(f'- Optimal: ${total_optimal:+,.0f} (${total_optimal/max(total_days,1):+.0f}/day)\n')
    lines.append(f'- If flipped at peak (oracle): ${total_if_flip_peak:+,.0f} '
                 f'(${total_if_flip_peak/max(total_days,1):+.0f}/day)\n')
    lines.append(f'- If flipped at exit (realistic): ${total_if_flip_exit:+,.0f} '
                 f'(${total_if_flip_exit/max(total_days,1):+.0f}/day)\n')

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(''.join(lines))
    print(f'\nReport: {OUT_MD}')
    print(f'Per-tier CSV: {OUT_CSV}')
    print(f'Per-tier regret CSVs: {OUT_DIR}/')


def _verdict(r):
    """Human-readable verdict for each tier."""
    if r['pct_counter'] > 45:
        return (f'DEAD — {r["pct_counter"]:.0f}% counter-flip means no '
                f'directional edge. Kill or rebuild entry.')
    if r['pct_counter'] < 20:
        return (f'Direction solid ({r["pct_counter"]:.0f}% counter). '
                f'Exits leak {100-r["capture_pct"]:.0f}% — tune exits, not entries.')
    if r['if_flipped_at_exit_per_day'] > 0:
        return (f'Real edge — CNN flip target. '
                f'{r["pct_counter"]:.0f}% counter-flip; '
                f'flipping at exit alone = ${r["if_flipped_at_exit_per_day"]:+.0f}/day.')
    return (f'Marginal — {r["pct_counter"]:.0f}% counter, '
            f'realistic flip gain only ${r["if_flipped_at_exit_per_day"]:+.0f}/day.')


if __name__ == '__main__':
    main()
