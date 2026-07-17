"""
Compile parity checker for the seq-window trainer's --compile_act path.

Compares two --loss-dump npz files (per-bar losses, actions, rewards) from
seed-matched runs. Gates (per the speed-pass discipline: causality supreme,
parity before trusting any compiled run):
  ACTIONS  : bitwise identical (same seed + same logits => same samples)
  LOSSES   : max |delta| <= --tol   (1e-6 default; fp32/--no_autocast runs)
  REWARDS  : max |delta| <= --tol   (env is deterministic given actions)
  BITWISE  : with --bitwise, arrays must be byte-identical (self-determinism
             gate: two runs of the SAME compiled binary + seed)

Run:
  python tools/compile_parity_check.py A.npz B.npz [--tol 1e-6] [--bitwise]
Exit 0 = PASS, 1 = FAIL. Prints a verdict block either way.
"""
import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--tol', type=float, default=1e-6)
    ap.add_argument('--bitwise', action='store_true',
                    help='require byte-identical arrays (self-determinism gate)')
    args = ap.parse_args()

    A, B = np.load(args.a), np.load(args.b)
    keys = sorted(set(A.files) & set(B.files))
    if not keys:
        print('FAIL: no common arrays')
        sys.exit(1)

    ok = True
    for k in keys:
        a, b = A[k], B[k]
        if a.shape != b.shape:
            print(f'FAIL {k}: shape {a.shape} vs {b.shape} '
                  f'(runs diverged — an action flipped; inspect first mismatch)')
            ok = False
            continue
        if args.bitwise or a.dtype.kind in 'iub':
            same = np.array_equal(a, b)
            n_diff = int((a != b).sum()) if not same else 0
            first = int(np.argmax(a != b)) if not same else -1
            print(f"{'PASS' if same else 'FAIL'} {k}: "
                  f"{'byte-identical' if same else f'{n_diff} diffs, first at idx {first}'}")
            ok &= same
        else:
            d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            print(f"{'PASS' if d <= args.tol else 'FAIL'} {k}: max|delta|={d:.3e} "
                  f"(tol {args.tol:.1e})")
            ok &= (d <= args.tol)

    print('VERDICT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
