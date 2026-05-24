"""Live zigzag baseline — forward pass ZZ direction vs hindsight leg_direction.

Purpose: settle whether the trend3 ML adds anything over a 30-line zigzag
indicator. If the indicator alone matches the hindsight leg direction at a
rate comparable to trend3, the ML is approximating the indicator.

Architecture (per user 2026-05-17):
  - ZIGZAG THRESHOLD lives at 1m TF — min_reversal_ticks comes from
    ATR(14) of 1m bars × the training multiplier (×4).
  - FLIP DETECTION resolves at 5s precision — the moment 5s close crosses
    the R-trigger price, direction flips. Slower 1m-bar checks would
    delay the flip by up to 60s.
  - min_bars between pivots = 36 in 5s units = 3 minutes (matches the
    training-label generator `build_zigzag_pivot_dataset.py`).

Method:
  - For each NT8 OOS day, load 5s closes
  - Compute day ATR(14) on 1m, get min_rev_ticks at ATR×4 (training match)
  - Run STREAMING detect_swings on 5s closes: emit per-5s-bar direction
    state at the time that bar is processed (forward pass, no future info)
  - Sample the per-5s direction at each 1m close timestamp
  - Compare to leg_direction (hindsight) from the truth dataset

Per user 2026-05-17: this is the baseline floor. trend3 must beat it
significantly to justify the ML.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

TICK_SIZE = 0.25
TRAIN_ATR_MULT = 4.0
NT8_5S_DIR = Path('DATA/ATLAS_NT8/5s')
NT8_1M_DIR = Path('DATA/ATLAS_NT8/1m')


def compute_atr(bars1m: pd.DataFrame, period: int = 14) -> float:
    h = bars1m['high'].values; l = bars1m['low'].values; c = bars1m['close'].values
    if len(h) < period + 1:
        return float((h - l).mean()) if len(h) > 0 else 1.0
    prev_c = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - prev_c), np.abs(l - prev_c)])
    return float(np.median(tr[-period * 3:])) if len(tr) >= period else float(tr.mean())


def live_zigzag_direction(closes: np.ndarray, min_reversal_ticks: int,
                           min_bars: int = 36) -> np.ndarray:
    """Streaming zigzag: emit per-bar direction state at time-of-bar.

    Direction state:
       0 = undecided (haven't moved min_reversal yet)
      +1 = current leg is LONG (last confirmed pivot was a low)
      -1 = current leg is SHORT (last confirmed pivot was a high)

    Returns int8 array same length as closes. NO future info used per bar.
    """
    n = len(closes)
    out = np.zeros(n, dtype=np.int8)
    if n < 3:
        return out
    ct = closes / TICK_SIZE
    direction = 0
    extreme_val = ct[0]
    extreme_idx = 0
    for i in range(1, n):
        price = ct[i]
        if direction == 0:
            # Undecided — find first move of min_reversal from start
            if price > extreme_val:
                extreme_val = price; extreme_idx = i
            if price < ct[0] and (ct[0] - price) >= min_reversal_ticks:
                direction = -1
                extreme_val = price; extreme_idx = i
            elif price > ct[0] and (price - ct[0]) >= min_reversal_ticks:
                direction = 1
                extreme_val = price; extreme_idx = i
        elif direction == 1:
            # In up-leg — track running high
            if price >= extreme_val:
                extreme_val = price; extreme_idx = i
            elif (extreme_val - price) >= min_reversal_ticks and \
                 (i - extreme_idx) >= min_bars:
                # Reversal confirmed — flip to down-leg
                direction = -1
                extreme_val = price; extreme_idx = i
        elif direction == -1:
            # In down-leg — track running low
            if price <= extreme_val:
                extreme_val = price; extreme_idx = i
            elif (price - extreme_val) >= min_reversal_ticks and \
                 (i - extreme_idx) >= min_bars:
                direction = 1
                extreme_val = price; extreme_idx = i
        out[i] = direction
    return out


def evaluate_day(day: str, atr_mult: float, min_bars_5s: int = 36) -> dict | None:
    """Compute live-zigzag direction per 5s bar, sample at 1m closes, compare
    to hindsight leg_direction. Returns dict of metrics for this day."""
    bars1m_path = NT8_1M_DIR / f'{day}.parquet'
    bars5s_path = NT8_5S_DIR / f'{day}.parquet'
    if not bars1m_path.exists() or not bars5s_path.exists():
        return None
    bars1m = pd.read_parquet(bars1m_path).sort_values('timestamp').reset_index(drop=True)
    bars5s = pd.read_parquet(bars5s_path).sort_values('timestamp').reset_index(drop=True)
    atr_pts = compute_atr(bars1m, 14)
    min_rev_ticks = max(4, int(round(atr_pts / TICK_SIZE * atr_mult)))
    closes5s = bars5s['close'].values.astype(np.float64)
    ts5s = bars5s['timestamp'].values.astype(np.int64)
    live_dir = live_zigzag_direction(closes5s, min_rev_ticks, min_bars_5s)
    return {
        'day': day,
        'atr_pts': atr_pts,
        'min_rev_ticks': min_rev_ticks,
        'ts5s': ts5s,
        'live_dir': live_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--truth',
                    default='reports/findings/regret_oracle/zigzag_pivot_dataset_NT8_OOS_atr4.parquet')
    ap.add_argument('--atr-mult', type=float, default=TRAIN_ATR_MULT)
    ap.add_argument('--min-bars-5s', type=int, default=36,
                    help='Min bars between pivots on 5s (default 36 = 3 min)')
    ap.add_argument('--out', default='reports/findings/regret_oracle/live_zigzag_baseline.txt')
    args = ap.parse_args()

    print(f'Loading truth: {args.truth}')
    truth_df = pd.read_parquet(args.truth)
    print(f'  {len(truth_df)} bars / {truth_df["day"].nunique()} days')

    # Derive leg_direction from is_pivot/pivot_dir if not present
    if 'leg_direction' not in truth_df.columns:
        from tools.direction_signal_accuracy import derive_leg_direction
        truth_df = derive_leg_direction(truth_df)

    truth_df = truth_df[truth_df['leg_direction'].isin(['LONG', 'SHORT'])].copy()
    print(f'  with valid leg_direction: {len(truth_df)}')

    days = sorted(truth_df['day'].unique())
    print(f'  evaluating {len(days)} days at ATR×{args.atr_mult}')

    # Process each day
    per_day = []
    all_pred = []
    all_truth = []
    all_day = []
    for day in tqdm(days, desc='days'):
        info = evaluate_day(day, args.atr_mult, args.min_bars_5s)
        if info is None:
            continue
        day_truth = truth_df[truth_df['day'] == day].sort_values('timestamp')
        if len(day_truth) == 0:
            continue
        # Sample live_dir at each 1m close timestamp (find last 5s ≤ ts1m)
        ts1m = day_truth['timestamp'].values.astype(np.int64)
        idx5s = np.searchsorted(info['ts5s'], ts1m, side='right') - 1
        idx5s = np.clip(idx5s, 0, len(info['ts5s']) - 1)
        live_at_1m = info['live_dir'][idx5s]
        # Map int8 to string for comparison: +1->LONG, -1->SHORT, 0->NEUTRAL
        live_str = np.where(live_at_1m == 1, 'LONG',
                  np.where(live_at_1m == -1, 'SHORT', 'NEUTRAL'))
        truth_str = day_truth['leg_direction'].values

        # Per-day stats
        signal_mask = (live_str != 'NEUTRAL')
        n_total = len(live_str)
        n_signal = int(signal_mask.sum())
        n_correct = int(((live_str == truth_str) & signal_mask).sum())
        n_wrong = n_signal - n_correct
        per_day.append({
            'day': day,
            'n_bars': n_total,
            'coverage': n_signal / max(n_total, 1),
            'acc': n_correct / max(n_signal, 1) if n_signal > 0 else float('nan'),
            'atr_pts': info['atr_pts'],
            'min_rev_ticks': info['min_rev_ticks'],
            'n_correct': n_correct,
            'n_wrong': n_wrong,
        })
        all_pred.extend(live_str.tolist())
        all_truth.extend(truth_str.tolist())
        all_day.extend([day] * n_total)

    all_pred = np.array(all_pred)
    all_truth = np.array(all_truth)
    per_day_df = pd.DataFrame(per_day)

    # === Headline ===
    signal_mask = (all_pred != 'NEUTRAL')
    n_total = len(all_pred)
    n_signal = int(signal_mask.sum())
    n_correct = int(((all_pred == all_truth) & signal_mask).sum())
    acc_overall = n_correct / max(n_signal, 1)

    # Per-class precision/recall
    tp_long  = int(((all_pred == 'LONG')  & (all_truth == 'LONG')).sum())
    fp_long  = int(((all_pred == 'LONG')  & (all_truth == 'SHORT')).sum())
    fn_long  = int(((all_pred != 'LONG')  & (all_truth == 'LONG')).sum())
    tp_short = int(((all_pred == 'SHORT') & (all_truth == 'SHORT')).sum())
    fp_short = int(((all_pred == 'SHORT') & (all_truth == 'LONG')).sum())
    fn_short = int(((all_pred != 'SHORT') & (all_truth == 'SHORT')).sum())
    prec_long  = tp_long  / max(tp_long + fp_long, 1)
    rec_long   = tp_long  / max(tp_long + fn_long, 1)
    prec_short = tp_short / max(tp_short + fp_short, 1)
    rec_short  = tp_short / max(tp_short + fn_short, 1)

    # 95% bootstrap CI on per-day accuracy
    accs = per_day_df['acc'].dropna().values
    rng = np.random.default_rng(42)
    boots = np.array([accs[rng.integers(0, len(accs), len(accs))].mean()
                      for _ in range(4000)])
    mean_per_day = float(accs.mean())
    ci_lo = float(np.percentile(boots, 2.5))
    ci_hi = float(np.percentile(boots, 97.5))

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 72)
    out('LIVE ZIGZAG BASELINE  (forward pass indicator vs hindsight leg_direction)')
    out('=' * 72)
    out(f'Days: {len(per_day_df)}   bars: {n_total:,}')
    out(f'ATR multiplier (matched to training): x{args.atr_mult}')
    out(f'5s min-bars between pivots: {args.min_bars_5s}')
    out('')
    out(f'Coverage (non-NEUTRAL):   {n_signal/max(n_total,1)*100:.1f}%   '
        f'({n_signal:,} / {n_total:,})')
    out(f'Accuracy ON SIGNAL bars:  {acc_overall*100:.2f}%   '
        f'({n_correct:,} correct)')
    out(f'Per-day acc mean:         {mean_per_day*100:.2f}%   '
        f'95% CI [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]')
    out('')
    out(f'LONG :  prec={prec_long*100:.1f}%  rec={rec_long*100:.1f}%   '
        f'tp={tp_long} fp={fp_long} fn={fn_long}')
    out(f'SHORT:  prec={prec_short*100:.1f}%  rec={rec_short*100:.1f}%   '
        f'tp={tp_short} fp={fp_short} fn={fn_short}')
    out('')
    out('--- HEAD-TO-HEAD vs trend3 (NT8 OOS, same days) ---')
    out(f'  RAW trend3      : 65.86% on 66.0% coverage   per-day 66.19% [63.41, 68.79]')
    out(f'  SMOOTHED trend3 : 61.48% on 99.9% coverage   per-day 60.68% [57.76, 63.35]')
    out(f'  LIVE zigzag     : {acc_overall*100:.2f}% on {n_signal/max(n_total,1)*100:.1f}% coverage   '
        f'per-day {mean_per_day*100:.2f}% [{ci_lo*100:.2f}, {ci_hi*100:.2f}]')

    # === Flip lag at zigzag truth pivots ===
    # When leg_direction flips, how many bars until live signal catches up?
    out('')
    out('FLIP LAG at zigzag truth flips (bars from truth-flip -> live ZZ catch-up):')
    lags = []
    for day in per_day_df['day']:
        mask = (np.array(all_day) == day)
        t = all_truth[mask]; p = all_pred[mask]
        for i in range(1, len(t)):
            if t[i] != t[i-1] and t[i] in ('LONG', 'SHORT'):
                new_dir = t[i]
                lag = None
                for j in range(i, len(t)):
                    if p[j] == new_dir:
                        lag = j - i; break
                    if t[j] != new_dir:
                        break
                if lag is not None:
                    lags.append(lag)
    lags = np.array(lags)
    if len(lags):
        out(f'  n={len(lags)}  median={np.median(lags):.1f}  mean={lags.mean():.1f}  '
            f'p25={np.percentile(lags,25):.1f}  p75={np.percentile(lags,75):.1f}  '
            f'p90={np.percentile(lags,90):.1f}')

    # === Save ===
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    per_day_df.to_csv(out_path.with_suffix('.per_day.csv'), index=False)
    print(f'\nWrote: {out_path}')


if __name__ == '__main__':
    main()
