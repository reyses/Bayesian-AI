"""Mid-leg entry  --  E4: leg overlap-structure analysis (mechanism behind E3).

E3 found a 1-contract greedy engine catches 2922/2926 OOS legs -- almost no
'busy and missed a parallel leg' population. This script quantifies WHY: the
hardened zigzag legs are a SEQUENTIAL partition of the price path (leg N ends
at a pivot, leg N+1 is born at that pivot), so consecutive legs barely
overlap and the engine is essentially never forced to skip a leg.

Computes, over the 51 sealed OOS days:
  - gap between consecutive legs (entry[N+1] - exit[N]) distribution
  - how many legs overlap ANY earlier leg
  - greedy 1-contract engine utilisation (busy time / session span)
  - characterisation of the genuinely-missable legs

Output: reports/findings/regret_oracle/2026-05-20_midleg_overlap.txt
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RD = 'reports/findings/regret_oracle'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oos', default=f'{RD}/trade_trajectory_OOS_full.parquet')
    ap.add_argument('--out', default=f'{RD}/2026-05-20_midleg_overlap.txt')
    args = ap.parse_args()

    lines = []
    def out(s=''):
        print(s)
        lines.append(s)

    traj = pd.read_parquet(args.oos)
    legs5 = traj[traj['K'] == 5][['leg_id', 'day', 'entry_ts', 'exit_ts',
                                  'leg_dir', 'exit_pnl_usd']]
    days = sorted(legs5['day'].unique())

    out('=' * 80)
    out('MID-LEG ENTRY  --  E4: leg overlap-structure analysis')
    out('=' * 80)
    out(f'{len(legs5):,} legs over {len(days)} sealed OOS days')
    out('')

    gaps = []                 # entry[N+1] - exit[N], seconds
    overlap_any = 0           # legs overlapping any earlier leg
    missed = []               # greedy-missed legs
    utils = []                # per-day engine utilisation
    durations = []

    for day in days:
        L = (legs5[legs5['day'] == day]
             .sort_values('entry_ts')
             .reset_index(drop=True))
        ent = L['entry_ts'].values.astype(np.int64)
        ext = L['exit_ts'].values.astype(np.int64)
        lid = L['leg_id'].values
        n = len(L)
        durations.extend((ext - ent).tolist())

        gaps.extend((ent[1:] - ext[:-1]).tolist())

        run_max = -1 << 62
        for i in range(n):
            if run_max > ent[i]:
                overlap_any += 1
            run_max = max(run_max, ext[i])

        free = -1 << 62
        busy = 0
        for i in range(n):
            if ent[i] >= free:
                free = ext[i]
                busy += int(ext[i] - ent[i])
            else:
                missed.append((day, int(lid[i]), int(ent[i]), int(ext[i]),
                               float(L['exit_pnl_usd'].values[i])))
        span = int(ext.max() - ent.min()) if n else 0
        if span > 0:
            utils.append(busy / span)

    gaps = np.array(gaps, dtype=np.float64)
    durations = np.array(durations, dtype=np.float64)

    out('-' * 80)
    out('CONSECUTIVE-LEG GAP  (entry[N+1] - exit[N], seconds)')
    out('-' * 80)
    out(f'  consecutive pairs : {len(gaps):,}')
    out(f'  gap < 0 (overlap) : {(gaps < 0).sum():,}  ({(gaps < 0).mean()*100:.1f}%)')
    out(f'  gap == 0          : {(gaps == 0).sum():,}  ({(gaps == 0).mean()*100:.1f}%)')
    out(f'  gap > 0           : {(gaps > 0).sum():,}  ({(gaps > 0).mean()*100:.1f}%)')
    out(f'  percentiles (s)   : min {gaps.min():.0f}  p5 {np.percentile(gaps,5):.0f}  '
        f'p25 {np.percentile(gaps,25):.0f}  med {np.median(gaps):.0f}  '
        f'p75 {np.percentile(gaps,75):.0f}  p95 {np.percentile(gaps,95):.0f}  '
        f'max {gaps.max():.0f}')
    out('')
    out('-' * 80)
    out('LEG DURATION  (exit - entry, seconds)')
    out('-' * 80)
    out(f'  median {np.median(durations):.0f}s  mean {durations.mean():.0f}s  '
        f'p90 {np.percentile(durations,90):.0f}s  max {durations.max():.0f}s')
    out('')
    out('-' * 80)
    out('OVERLAP  &  ENGINE UTILISATION')
    out('-' * 80)
    out(f'  legs overlapping any earlier leg : {overlap_any:,} / {len(legs5):,}  '
        f'({overlap_any/len(legs5)*100:.2f}%)')
    out(f'  greedy 1-contract utilisation    : mean {np.mean(utils)*100:.1f}%  '
        f'(busy time / session span, per day)')
    out(f'  greedy-missed legs (engine busy) : {len(missed)} in {len(days)} days')
    out('')
    if missed:
        out('  the genuinely-missable legs:')
        for day, lid, ent, ext, pnl in missed:
            out(f'    {day}  leg {lid:>5}  entry {ent}  exit {ext}  '
                f'dur {ext-ent}s  exit_pnl ${pnl:+.0f}')
    out('')
    out('=' * 80)
    out('INTERPRETATION')
    out('=' * 80)
    out('Hardened zigzag legs are a near-perfect SEQUENTIAL PARTITION of the')
    out('price path: leg N exits at a pivot, leg N+1 is born from that pivot,')
    out('its R-trigger fires shortly after. Consecutive gaps are almost all')
    out('>= 0, so a 1-contract engine riding leg N to its exit is free and')
    out('ready before leg N+1 triggers. There is no meaningful population of')
    out('legs missed because the engine was busy -> mid-leg / late-join entry')
    out('has nothing to act on. "Lost signals" seen live are cold-start (fix:')
    out('zigzag-state priming) or deliberate B7 skips -- NOT busy-missed legs.')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
