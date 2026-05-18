"""Mode of oracle MFE at SEED LEVEL — the unfiltered regret analysis output.

Per user 2026-05-12: "what's the mode of the trades at seed level?" — meaning
the distribution of mfe_dollars across ALL oracle entries (M_1m local extrema
firing with forward-60m MFE), no filter, no model gating.

Per CLAUDE.md:
  - $/trade reported as MODE (histogram bin $2) AND mean with 95% bootstrap CI
  - Also report median for asymmetric tails
  - Stratify by direction (LONG vs SHORT) — they often differ
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ORACLE_PATHS = [
    Path('reports/findings/regret_oracle/oracle_entries_jul2025.csv'),
    Path('reports/findings/regret_oracle/oracle_entries_2025-09-08.csv'),
]
BIN_W = 2.0
B = 4000


def histogram_mode(pnls: np.ndarray, bin_w: float = BIN_W):
    if len(pnls) == 0:
        return None
    lo = np.floor(pnls.min() / bin_w) * bin_w
    hi = np.ceil(pnls.max() / bin_w) * bin_w
    bins = np.arange(lo, hi + bin_w, bin_w)
    counts, edges = np.histogram(pnls, bins=bins)
    mode_idx = int(np.argmax(counts))
    return {
        'mode_lo': float(edges[mode_idx]),
        'mode_hi': float(edges[mode_idx + 1]),
        'mode_center': float((edges[mode_idx] + edges[mode_idx + 1]) / 2),
        'mode_count': int(counts[mode_idx]),
        'mode_pct': 100.0 * counts[mode_idx] / len(pnls),
        'counts': counts,
        'edges': edges,
    }


def bootstrap_mean_ci(pnls: np.ndarray, n_boot: int = B):
    rng = np.random.default_rng(42)
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = rng.choice(pnls, size=len(pnls), replace=True).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(pnls.mean()), float(lo), float(hi)


# Duration buckets — premise: 1-hour lapse to take the best trade.
# A trade's time-to-MFE categorizes WHAT KIND of trade it is:
#   FLASH  = sure-shot, edge resolves immediately
#   FAST   = quick convergence
#   MEDIUM = needs patience but moves
#   SLOW   = full-window grind (per user: "waiting is a bad idea")
DURATION_BUCKETS = [
    ('FLASH  (0-3m)',   0,    3),
    ('FAST   (3-10m)',  3,   10),
    ('MEDIUM (10-25m)', 10,  25),
    ('SLOW   (25-60m)', 25,  61),
]


def categorize_by_duration(df: pd.DataFrame):
    """Split oracle trades by time-to-MFE; report $ + velocity + signature
    per bucket. Velocity ($/min) is the direct measure of the user's
    'speed of convergence' idea."""
    ttm = df['time_to_mfe_min'].astype(float)
    mfe = df['mfe_dollars'].astype(float)
    print('\n\n' + '=' * 72)
    print('TRADES CATEGORIZED BY DURATION (time-to-MFE)')
    print('=' * 72)
    print(f'  time_to_mfe distribution: '
          f'q25={ttm.quantile(.25):.0f}m  median={ttm.median():.0f}m  '
          f'q75={ttm.quantile(.75):.0f}m  max={ttm.max():.0f}m')

    for name, lo, hi in DURATION_BUCKETS:
        mask = (ttm >= lo) & (ttm < hi)
        sub = df[mask]
        if len(sub) == 0:
            print(f'\n  {name}:  (none)')
            continue
        p = sub['mfe_dollars'].astype(float).values
        t = sub['time_to_mfe_min'].astype(float).values
        vel = p / np.maximum(t, 0.5)            # $/min, floor t at 0.5m
        m = histogram_mode(p)
        n_long = int((sub['direction'] == 'LONG').sum())
        n_short = int((sub['direction'] == 'SHORT').sum())
        print(f'\n  {name}   n={len(sub)} ({100*len(sub)/len(df):.0f}%)   '
              f'L:{n_long} S:{n_short}')
        print(f'    $ MFE       : mode=${m["mode_center"]:+.0f}  '
              f'median=${np.median(p):+.0f}  mean=${p.mean():+.0f}')
        print(f'    velocity    : median=${np.median(vel):.1f}/min  '
              f'mean=${vel.mean():.1f}/min  '
              f'(>${np.percentile(vel,75):.1f}/min top quartile)')
        # Dominant categorical signature
        for col in ('crm_stack', 'd_rail_position'):
            if col in sub.columns:
                vc = sub[col].value_counts(dropna=False)
                top = vc.index[0]
                print(f'    {col:16s}: {str(top)} ({100*vc.iloc[0]/len(sub):.0f}%)')
        # Mean extension state
        for col in ('z_15m', 'z_1h_high', 'z_1h_low'):
            if col in sub.columns:
                v = pd.to_numeric(sub[col], errors='coerce')
                print(f'    mean {col:11s}: {v.mean():+.2f}', end='')
        print()


def summarize(label: str, pnls: np.ndarray):
    if len(pnls) == 0:
        print(f'\n=== {label} ===  (no data)')
        return
    m = histogram_mode(pnls)
    mean, ci_lo, ci_hi = bootstrap_mean_ci(pnls)
    print(f'\n=== {label}  (n={len(pnls)}) ===')
    print(f'  mode bin    : [${m["mode_lo"]:.0f}, ${m["mode_hi"]:.0f})  center=${m["mode_center"]:+.1f}')
    print(f'                count={m["mode_count"]}  ({m["mode_pct"]:.1f}% of trades)')
    print(f'  median      : ${np.median(pnls):+.2f}')
    print(f'  mean (95%CI): ${mean:+.2f}  [${ci_lo:+.2f}, ${ci_hi:+.2f}]')
    print(f'  min / max   : ${pnls.min():+.0f} / ${pnls.max():+.0f}')
    print(f'  q25/q50/q75 : ${np.percentile(pnls,25):+.0f} / ${np.percentile(pnls,50):+.0f} / ${np.percentile(pnls,75):+.0f}')
    print(f'  top 12 bins (ASCII histogram, bin=${BIN_W:.0f}):')
    counts, edges = m['counts'], m['edges']
    top = np.argsort(counts)[::-1][:12]
    top_sorted = sorted(top, key=lambda k: edges[k])
    max_c = max(counts[top])
    for k in top_sorted:
        bar = '#' * int(60 * counts[k] / max_c)
        marker = ' <-- MODE' if k == int(np.argmax(counts)) else ''
        print(f'    [${edges[k]:+5.0f}, ${edges[k+1]:+5.0f}) : {counts[k]:5d}  {bar}{marker}')


def main():
    frames = []
    for p in ORACLE_PATHS:
        if p.exists():
            df = pd.read_csv(p)
            df['source'] = p.name
            frames.append(df)
            print(f'Loaded {len(df)} from {p.name}')
    if not frames:
        print('No oracle CSVs found.'); return
    df = pd.concat(frames, ignore_index=True)
    print(f'\nTotal oracle entries (seed level): {len(df)}')

    pnls_all = df['mfe_dollars'].astype(float).values
    pnls_long  = df[df['direction'] == 'LONG']['mfe_dollars'].astype(float).values
    pnls_short = df[df['direction'] == 'SHORT']['mfe_dollars'].astype(float).values

    summarize('ALL ORACLE ENTRIES (seed level)', pnls_all)
    summarize('LONG side', pnls_long)
    summarize('SHORT side', pnls_short)

    categorize_by_duration(df)

    # Stratify by direction-anchor categorical (d_direction) if present
    if 'd_direction' in df.columns:
        print('\n\n--- Stratified by anchor regime category ---')
        for col in ['crm_stack', 'd_stack', 'd_rail_position']:
            if col in df.columns:
                print(f'\nBy {col}:')
                for v, sub in df.groupby(col, dropna=False):
                    p = sub['mfe_dollars'].astype(float).values
                    if len(p) < 20:
                        continue
                    m = histogram_mode(p)
                    mean, _, _ = bootstrap_mean_ci(p, n_boot=1000)
                    print(f'  {str(v):>30s}  n={len(p):4d}  '
                          f'mode_center=${m["mode_center"]:+5.1f}  '
                          f'median=${np.median(p):+5.0f}  '
                          f'mean=${mean:+5.1f}')


if __name__ == '__main__':
    main()
