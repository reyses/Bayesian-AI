"""
Apply trained pivot-direction NN as a post-hoc filter on existing trades.

For each trade in rm_is.pkl (TEST split only — last 15% of days to avoid
leakage), predict P(win) and assign regime:
  - P >= take_threshold → KEEP original trade as-is
  - P <= flip_threshold → FLIP: multiply pnl by −1 (as if trade was opposite dir)
  - Else → SKIP (pnl = 0)

Report $/day in each regime and aggregate.

Usage:
    python tools/apply_pivot_nn_filter.py
"""
import os
import sys
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.train_pivot_direction_nn import feat_to_grid, PivotCNN, GRID_H, GRID_W


TRADES_PKL = 'training_RM_physics/output/trades/rm_is.pkl'
MODEL_PATH = 'training_RM_physics/output/pivot_direction_cnn.pt'

TAKE_THRESHOLD = 0.55
FLIP_THRESHOLD = 0.45


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(TRADES_PKL, 'rb') as f:
        trades = pickle.load(f)
    valid = [t for t in trades if len(t.get('entry_79d', [])) >= 90 and t['pnl'] != 0]
    # Use only TEST split (last 15% of days) — same as training script
    valid.sort(key=lambda t: t['day'])
    days = sorted(set(t['day'] for t in valid))
    n_days = len(days)
    test_days = set(days[int(n_days * 0.85):])
    test_trades = [t for t in valid if t['day'] in test_days]
    print(f'Test trades: {len(test_trades)} over {len(test_days)} days')

    # Load model
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = PivotCNN().to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    mean = ckpt['mean']
    std = ckpt['std']

    # Build grids + predict
    grids = np.stack([feat_to_grid(t['entry_79d']) for t in test_trades])
    grids = (grids - mean) / std
    with torch.no_grad():
        logits = model(torch.from_numpy(grids).unsqueeze(1).to(device)).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))

    # Attach predictions, classify regime
    regimes = []
    for p in probs:
        if p >= TAKE_THRESHOLD:
            regimes.append('TAKE')
        elif p <= FLIP_THRESHOLD:
            regimes.append('FLIP')
        else:
            regimes.append('SKIP')
    regimes = np.array(regimes)

    pnls_orig = np.array([t['pnl'] for t in test_trades])
    pnls_filtered = pnls_orig.copy()
    pnls_filtered[regimes == 'FLIP'] = -pnls_filtered[regimes == 'FLIP']
    pnls_filtered[regimes == 'SKIP'] = 0.0

    # Per-day aggregates
    day_of = np.array([t['day'] for t in test_trades])
    uniq = sorted(set(day_of))
    day_orig = defaultdict(float)
    day_filt = defaultdict(float)
    for t, p_o, p_f in zip(test_trades, pnls_orig, pnls_filtered):
        day_orig[t['day']] += p_o
        day_filt[t['day']] += p_f

    orig_daily = np.array([day_orig[d] for d in uniq])
    filt_daily = np.array([day_filt[d] for d in uniq])

    def pnl_ratio_wr(arr):
        profit = float(arr[arr > 0].sum())
        loss = float(abs(arr[arr < 0].sum()))
        return profit / max(loss, 1e-9) - 1

    print()
    print('=== Regime breakdown ===')
    for r in ['TAKE', 'FLIP', 'SKIP']:
        m = regimes == r
        if m.sum() == 0:
            continue
        p_o = pnls_orig[m]
        p_f = pnls_filtered[m]
        print(f'{r}: {m.sum()} trades  orig=${p_o.sum():+,.0f}  filtered=${p_f.sum():+,.0f}')

    print()
    print('=== Original vs NN-filtered (TEST set) ===')
    print(f'{"":<15}{"trades":>8}{"net $":>12}{"$/day":>10}{"TradeWR":>10}{"DayWR":>8}')
    def row(name, arr, trades_count, days):
        wr = pnl_ratio_wr(arr)
        day_wr = (np.array([day_orig[d] if name == 'orig' else day_filt[d] for d in uniq]) > 0).mean() * 100
        return f'{name:<15}{trades_count:>8}${arr.sum():>+10,.0f}${arr.sum()/len(days):>+8,.0f}{wr:>+10.2f}{day_wr:>7.0f}%'
    # Need to recompute day_wr correctly
    orig_day_wr = (orig_daily > 0).mean() * 100
    filt_day_wr = (filt_daily > 0).mean() * 100
    orig_wr = pnl_ratio_wr(pnls_orig)
    filt_wr = pnl_ratio_wr(pnls_filtered[pnls_filtered != 0])  # WR on active only
    print(f'{"original":<15}{len(test_trades):>8}${pnls_orig.sum():>+10,.0f}${pnls_orig.sum()/len(uniq):>+8,.0f}{orig_wr:>+10.2f}{orig_day_wr:>7.0f}%')
    active = pnls_filtered[regimes != 'SKIP']
    print(f'{"nn-filtered":<15}{(regimes!="SKIP").sum():>8}${pnls_filtered.sum():>+10,.0f}${pnls_filtered.sum()/len(uniq):>+8,.0f}{filt_wr:>+10.2f}{filt_day_wr:>7.0f}%')

    print()
    # Write report
    out_dir = 'reports/findings'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'pivot_nn_filter.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Pivot NN filter — post-hoc uplift on TEST set\n\n')
        f.write(f'Generated: {datetime.now().isoformat(timespec="seconds")}\n\n')
        f.write(f'Test trades: {len(test_trades)} over {len(uniq)} days\n')
        f.write(f'Thresholds: TAKE ≥ {TAKE_THRESHOLD}, FLIP ≤ {FLIP_THRESHOLD}\n\n')
        f.write('## Regime counts\n\n')
        for r in ['TAKE', 'FLIP', 'SKIP']:
            m = regimes == r
            f.write(f'- {r}: {int(m.sum())} trades ({int(m.sum())/len(test_trades)*100:.0f}%)\n')
        f.write('\n')
        f.write('## Aggregate comparison\n\n')
        f.write('| Variant | Trades used | Net $ | $/day | Trade WR | Day WR |\n')
        f.write('|---|---:|---:|---:|---:|---:|\n')
        f.write(f'| original | {len(test_trades)} | ${pnls_orig.sum():+,.0f} | ${pnls_orig.sum()/len(uniq):+.0f} | {orig_wr:+.2f} | {orig_day_wr:.0f}% |\n')
        f.write(f'| nn-filtered | {int((regimes != "SKIP").sum())} | ${pnls_filtered.sum():+,.0f} | ${pnls_filtered.sum()/len(uniq):+.0f} | {filt_wr:+.2f} | {filt_day_wr:.0f}% |\n')
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
