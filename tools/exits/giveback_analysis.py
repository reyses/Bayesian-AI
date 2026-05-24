"""
Giveback Analysis — after peak PnL, how many bars until break-even?

For each trade with peak_pnl > 0:
- Find the bar where peak occurred
- Count bars from peak until pnl <= 0 (or trade ends)
- Categorize: never_broke_even, broke_even_in_N_bars, etc.

Output: reports/findings/giveback_analysis.txt

Usage:
    python tools/giveback_analysis.py
    python tools/giveback_analysis.py --tier RIDE_AGAINST
    python tools/giveback_analysis.py --days 60
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import BlendedEngine

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'reports/findings'


def collect_trades(tier_filter=None, max_days=None):
    feat_files = sorted(f for f in glob.glob(os.path.join(FEATURES_DIR, '*.parquet'))
                        if '2025_' in os.path.basename(f))
    if max_days:
        feat_files = feat_files[:max_days]

    print(f'Processing {len(feat_files)} IS days...')
    engine = BlendedEngine(use_cnn=False)
    rows = []

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades:
            tier = t.get('entry_tier', '?')
            if tier_filter and tier != tier_filter:
                continue
            path = t.get('path', [])
            if len(path) < 2:
                continue

            pnls = [p.get('pnl', 0) for p in path]
            peak_pnl = max(pnls)
            if peak_pnl <= 0:
                continue  # never went positive

            peak_bar = pnls.index(peak_pnl)
            # Bars from peak until pnl <= 0
            never_broke = True
            bars_to_zero = -1
            for i in range(peak_bar + 1, len(pnls)):
                if pnls[i] <= 0:
                    bars_to_zero = i - peak_bar
                    never_broke = False
                    break

            # If trade never broke even, but final PnL is negative, that's giveback past zero
            final_pnl = pnls[-1]
            broke_even = not never_broke

            rows.append({
                'tier': tier,
                'pnl': t['pnl'],
                'final_pnl': final_pnl,
                'peak_pnl': peak_pnl,
                'peak_bar': peak_bar,
                'total_bars': len(pnls),
                'bars_after_peak': len(pnls) - 1 - peak_bar,
                'broke_even_after_peak': broke_even,
                'bars_to_zero': bars_to_zero,
                'is_winner': t['pnl'] > 0,
                'giveback': peak_pnl - final_pnl,
                'giveback_pct': (peak_pnl - final_pnl) / peak_pnl * 100 if peak_pnl > 0 else 0,
            })

    return pd.DataFrame(rows)


def analyze(df):
    lines = []
    lines.append('=' * 80)
    lines.append('GIVEBACK ANALYSIS — after peak, how long until break-even?')
    lines.append(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    lines.append('=' * 80)
    lines.append('')
    lines.append(f'Trades analyzed: {len(df):,} (peak PnL > 0)')
    lines.append('')

    # Overall stats
    broke = df[df['broke_even_after_peak']]
    held = df[~df['broke_even_after_peak']]

    lines.append('OVERALL:')
    lines.append(f'  Held above break-even: {len(held):,} ({len(held)/len(df)*100:.0f}%)')
    lines.append(f'  Returned to zero:      {len(broke):,} ({len(broke)/len(df)*100:.0f}%)')
    lines.append('')

    if len(broke) > 0:
        b = broke['bars_to_zero'].values
        lines.append('BARS FROM PEAK TO BREAK-EVEN (returned to zero):')
        lines.append(f'  Mean:   {b.mean():.1f} bars')
        lines.append(f'  Median: {np.median(b):.0f} bars')
        lines.append(f'  P10:    {np.percentile(b, 10):.0f} bars')
        lines.append(f'  P25:    {np.percentile(b, 25):.0f} bars')
        lines.append(f'  P50:    {np.percentile(b, 50):.0f} bars')
        lines.append(f'  P75:    {np.percentile(b, 75):.0f} bars')
        lines.append(f'  P90:    {np.percentile(b, 90):.0f} bars')
        lines.append(f'  Max:    {b.max():.0f} bars')
        lines.append('')

        # Cumulative distribution
        lines.append('CUMULATIVE: % of returned-to-zero trades that broke down within N bars:')
        for n in [1, 2, 3, 5, 10, 15, 20, 30, 60]:
            pct = (b <= n).mean() * 100
            lines.append(f'  Within {n:>3} bars: {pct:.0f}%')
        lines.append('')

    # Winners vs losers split
    winners = df[df['is_winner']]
    losers = df[~df['is_winner']]
    lines.append(f'WINNERS ({len(winners):,}): peaked then ended positive')
    lines.append(f'  Avg peak: ${winners["peak_pnl"].mean():.1f}')
    lines.append(f'  Avg final: ${winners["pnl"].mean():.1f}')
    lines.append(f'  Avg giveback: ${winners["giveback"].mean():.1f} '
                 f'({winners["giveback_pct"].mean():.0f}%)')
    lines.append(f'  Returned to zero before recovery: '
                 f'{winners["broke_even_after_peak"].sum():,} '
                 f'({winners["broke_even_after_peak"].mean()*100:.0f}%)')
    lines.append('')

    if len(losers) > 0:
        lines.append(f'LOSERS ({len(losers):,}): peaked positive but ended negative')
        lines.append(f'  Avg peak: ${losers["peak_pnl"].mean():.1f}')
        lines.append(f'  Avg final: ${losers["pnl"].mean():.1f}')
        lines.append(f'  Avg giveback: ${losers["giveback"].mean():.1f} '
                     f'({losers["giveback_pct"].mean():.0f}%)')
        lines.append('')

    # Per tier
    lines.append('=' * 80)
    lines.append('PER-TIER GIVEBACK PROFILE')
    lines.append('=' * 80)
    lines.append(f'{"Tier":<18} {"N":>6} {"%held":>8} {"med→0":>8} {"avg pk":>10} {"avg gb":>10} {"gb%":>6}')
    lines.append('-' * 75)

    for tier in sorted(df['tier'].unique()):
        sub = df[df['tier'] == tier]
        if len(sub) < 20:
            continue
        held_pct = (~sub['broke_even_after_peak']).mean() * 100
        broke_sub = sub[sub['broke_even_after_peak']]
        med = np.median(broke_sub['bars_to_zero']) if len(broke_sub) > 0 else 0
        avg_peak = sub['peak_pnl'].mean()
        avg_giveback = sub['giveback'].mean()
        gb_pct = sub['giveback_pct'].mean()

        lines.append(f'{tier:<18} {len(sub):>6} {held_pct:>7.0f}% '
                     f'{med:>7.0f}b {avg_peak:>+9.1f} {avg_giveback:>+9.1f} '
                     f'{gb_pct:>5.0f}%')

    lines.append('')

    # Implication
    lines.append('=' * 80)
    lines.append('IMPLICATION')
    lines.append('=' * 80)
    if len(broke) > 0:
        median_bars = int(np.median(broke['bars_to_zero']))
        pct_within_3 = (broke['bars_to_zero'] <= 3).mean() * 100
        lines.append(f'After peak, {pct_within_3:.0f}% of giveback trades return to zero within 3 bars.')
        lines.append(f'Median bars from peak to break-even: {median_bars}.')
        lines.append('')
        lines.append('If you set a "trail from peak" exit at N bars after peak:')
        for n in [3, 5, 10]:
            saved = (broke['bars_to_zero'] > n).sum()
            lost = (broke['bars_to_zero'] <= n).sum()
            lines.append(f'  N={n}: would catch {saved:,} trades before zero, '
                         f'miss {lost:,} (still profitable)')

    lines.append('')
    lines.append('=' * 80)

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tier', type=str, default=None)
    parser.add_argument('--days', type=int, default=None)
    args = parser.parse_args()

    df = collect_trades(tier_filter=args.tier, max_days=args.days)
    if len(df) < 50:
        print('Not enough peaked trades collected.')
        return

    report = analyze(df)
    print(report)

    suffix = f'_{args.tier}' if args.tier else ''
    out_path = os.path.join(OUTPUT_DIR, f'giveback_analysis{suffix}.txt')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
