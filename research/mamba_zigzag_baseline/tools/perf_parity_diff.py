"""Compare two --loss-dump .npz parity dumps from train_mamba_rl.py.

Reports: length, first step where actions diverge, first step where |loss A - loss B|
exceeds tol, and max |delta| over the common identical-action prefix.

Usage:
  python research/mamba_zigzag_baseline/tools/perf_parity_diff.py A.npz B.npz [--tol 1e-4]
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tol', type=float, default=1e-4)
    args = ap.parse_args()

    A = np.load(args.a)
    B = np.load(args.b)
    la, lb = A['losses'], B['losses']
    aa, ab = A['actions'], B['actions']
    ra, rb = A['rewards'], B['rewards']
    n = min(len(la), len(lb))
    print(f'A: {len(la)} steps | B: {len(lb)} steps | comparing first {n}')

    act_div = np.nonzero(aa[:n] != ab[:n])[0]
    first_act = int(act_div[0]) if len(act_div) else None
    print(f'action trajectories: '
          + ('IDENTICAL' if first_act is None else f'diverge at step {first_act}'))

    # loss comparison only meaningful over the identical-action prefix
    prefix = n if first_act is None else first_act
    dl = np.abs(la[:prefix] - lb[:prefix])
    dr = np.abs(ra[:prefix] - rb[:prefix])
    over = np.nonzero(dl > args.tol)[0]
    print(f'identical-action prefix: {prefix} steps')
    if prefix:
        print(f'max |loss delta| over prefix: {dl.max():.3e} '
              f'(tol {args.tol:g}) -> {"PASS" if not len(over) else f"FAIL at step {int(over[0])}"}')
        print(f'max |reward delta| over prefix: {dr.max():.3e}')
    if first_act is not None:
        print('NOTE: post-divergence losses are not comparable (different trajectories).')


if __name__ == '__main__':
    main()
