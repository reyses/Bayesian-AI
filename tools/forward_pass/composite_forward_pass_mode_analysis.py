"""Mode-centric analysis of the hardened forward pass.

Per CLAUDE.md metric standard:
  $/trade -> histogram mode, bin width $2
  $/day   -> histogram mode, bin width $25
  Mean + 95% bootstrap CI always shown alongside

Loads composite_forward_pass_hardened.csv, computes sizing multipliers
per scheme, reports mode + mean + CI per leg AND per day.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


DOLLAR_PER_POINT = 2.0


def hist_mode(arr, bin_width=2.0):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0: return 0.0
    lo = np.floor(a.min() / bin_width) * bin_width
    hi = np.ceil(a.max() / bin_width) * bin_width
    if hi <= lo: return float(a.mean())
    bins = np.arange(lo, hi + bin_width, bin_width)
    counts, edges = np.histogram(a, bins=bins)
    if counts.sum() == 0: return float(a.mean())
    k = int(counts.argmax())
    return float((edges[k] + edges[k + 1]) / 2.0)


def gbm_ev(pred_R):
    return float(np.clip(max(pred_R - 1.0, 0.0), 0.0, 3.0))


def gbm_quantile(pred_R, rank_pct):
    if rank_pct >= 0.80: return 2.0
    if rank_pct >= 0.50: return 1.5
    if rank_pct >= 0.20: return 0.8
    return 0.5


def hand_aggressive(zone, b6_match):
    if zone == 'AT_PIVOT' or b6_match >= 0.70: return 2.0
    elif zone in ('IMMINENT', 'NEAR_PIVOT', 'NEAR_3m', 'NEAR_5m'): return 1.2
    elif zone == 'CLEAR' and b6_match < 0.50: return 0.0
    else: return 0.8


def boot_ci(arr, n=4000, seed=42):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2: return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    boots = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n)])
    return float(arr.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened.csv')
    ap.add_argument('--out',
                    default='reports/findings/regret_oracle/composite_forward_pass_hardened_mode.txt')
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f'Loaded {len(df):,} legs across {df["day"].nunique()} days')

    # Build size multipliers per scheme
    pred = df['pred_amp_R_hardened'].values
    rank_pct = pd.Series(pred).rank(pct=True).values
    schemes = {
        'flat': np.ones(len(df)),
        'gbm_ev': np.array([gbm_ev(p) for p in pred]),
        'gbm_quantile': np.array([gbm_quantile(p, r) for p, r in zip(pred, rank_pct)]),
        'hand_aggressive': np.array([hand_aggressive(z, b)
                                       for z, b in zip(df['entry_zone'], df['entry_p_b6_match'])]),
    }

    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('=' * 78)
    out('HARDENED FORWARD PASS — mode-centric metrics (CLAUDE.md standard)')
    out('=' * 78)
    out(f'Legs: {len(df):,}   Days: {df["day"].nunique()}')
    out('')

    out('=== PER-LEG P&L (bin width $2) ===')
    out(f'{"scheme":<18}  {"mode":>9}  {"median":>9}  {"mean":>9}  '
        f'{"CI lo":>9}  {"CI hi":>9}  {"% pos":>6}')
    pnl_raw = df['pnl_usd'].values
    for name, sizes in schemes.items():
        weighted = pnl_raw * sizes
        # When size=0, the leg wasn't taken — exclude from mode/median
        taken = sizes > 0
        wpnl = weighted[taken]
        mode = hist_mode(wpnl, bin_width=2.0)
        median = float(np.median(wpnl)) if len(wpnl) else float('nan')
        mean, lo, hi = boot_ci(wpnl)
        pct_pos = float((wpnl > 0).mean() * 100) if len(wpnl) else 0
        out(f'{name:<18}  ${mode:>+7.2f}  ${median:>+7.2f}  ${mean:>+7.2f}  '
            f'${lo:>+7.2f}  ${hi:>+7.2f}  {pct_pos:>5.1f}%')

    out('')
    out('=== PER-DAY P&L (bin width $25) ===')
    out(f'{"scheme":<18}  {"mode":>10}  {"median":>10}  {"mean":>10}  '
        f'{"CI lo":>10}  {"CI hi":>10}  {"% pos":>6}')
    for name, sizes in schemes.items():
        weighted = pnl_raw * sizes
        df_copy = df.copy()
        df_copy['wpnl'] = weighted
        per_day = df_copy.groupby('day')['wpnl'].sum().values
        mode = hist_mode(per_day, bin_width=25.0)
        median = float(np.median(per_day)) if len(per_day) else float('nan')
        mean, lo, hi = boot_ci(per_day)
        pct_pos = float((per_day > 0).mean() * 100) if len(per_day) else 0
        out(f'{name:<18}  ${mode:>+8.2f}  ${median:>+8.2f}  ${mean:>+8.2f}  '
            f'${lo:>+8.2f}  ${hi:>+8.2f}  {pct_pos:>5.1f}%')

    out('')
    out('=== HISTOGRAM SHAPE for gbm_ev per-leg P&L ===')
    sizes = schemes['gbm_ev']
    wpnl = (pnl_raw * sizes)[sizes > 0]
    bin_w = 5.0   # wider bins for shape readability
    lo = np.floor(wpnl.min() / bin_w) * bin_w
    hi = np.ceil(wpnl.max() / bin_w) * bin_w
    edges = np.arange(lo, hi + bin_w, bin_w)
    counts, _ = np.histogram(wpnl, bins=edges)
    total = counts.sum()
    # Show top 12 bins by count
    top_idx = np.argsort(counts)[::-1][:12]
    out(f'  bin (USD)            count    % of legs   (top 12 by count)')
    for k in sorted(top_idx, key=lambda j: edges[j]):
        bin_lo = edges[k]; bin_hi = edges[k+1]
        c = int(counts[k])
        pct = c / max(total, 1) * 100
        bar = '#' * int(pct * 1.0)
        out(f'  [${bin_lo:>+5.0f}, ${bin_hi:>+5.0f})   {c:>5}   {pct:>5.1f}%  {bar}')

    out('')
    out('=== HISTOGRAM SHAPE for gbm_ev per-day P&L ===')
    df_copy = df.copy()
    df_copy['wpnl'] = pnl_raw * sizes
    per_day = df_copy.groupby('day')['wpnl'].sum().values
    bin_w = 100.0
    lo = np.floor(per_day.min() / bin_w) * bin_w
    hi = np.ceil(per_day.max() / bin_w) * bin_w
    edges = np.arange(lo, hi + bin_w, bin_w)
    counts, _ = np.histogram(per_day, bins=edges)
    out(f'  bin (USD/day)            count   days  bar')
    for k in range(len(counts)):
        if counts[k] == 0: continue
        bin_lo = edges[k]; bin_hi = edges[k+1]
        c = int(counts[k])
        bar = '#' * (c * 2)
        out(f'  [${bin_lo:>+7.0f}, ${bin_hi:>+7.0f})   {c:>5}   {bar}')

    Path(args.out).write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
