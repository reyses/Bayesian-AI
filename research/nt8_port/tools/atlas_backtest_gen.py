#!/usr/bin/env python3
"""Run the FROZEN reference decider (golden_vector_gen, verbatim) over the full
DATA/ATLAS history (owner 2026-07-26: "run it on the atlas data, not NT8").
Writes per-day decision vectors to research/nt8_port/atlas_backtest/ ; the
trade ledger is built from these by build_trade_ledger.py --gb atlas_backtest.
"""
import argparse
import glob
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import golden_vector_gen as gvg          # noqa: E402
import dossier_signal_pipeline as dsp    # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), 'atlas_backtest')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--last', type=int, default=None, help='last N days')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(os.path.join(dsp.D5, '*.parquet')))
    print(f'D5={dsp.D5}  {len(files)} 5s day files', flush=True)
    names = [os.path.basename(f)[:10] for f in files]
    lo, hi = 20, len(files)                 # need >=20 prior days for ctx
    if args.start:
        lo = max(lo, next((i for i, n in enumerate(names) if n >= args.start), lo))
    if args.end:
        hi = next((i for i, n in enumerate(names) if n > args.end), hi)
    if args.last:
        lo = max(lo, hi - args.last)
    model = gvg.fit_combiner()
    print(f'combiner fit ok; running days [{names[lo]} .. {names[hi-1]}] '
          f'= {hi-lo} days', flush=True)
    t0 = time.time()
    done = 0
    for j in range(lo, hi):
        day, F, zz, G = gvg.process_day(files, j, model)
        G.to_parquet(os.path.join(OUT, f'{day}.parquet'))
        done += 1
        if done <= 3 or done % 25 == 0:
            print(f'  [{done}/{hi-lo}] {day}: {len(G)} bars, '
                  f'entries={int(G["entry"].sum())} '
                  f'({time.time()-t0:.0f}s)', flush=True)
    print(f'DONE {done} days in {time.time()-t0:.0f}s -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
