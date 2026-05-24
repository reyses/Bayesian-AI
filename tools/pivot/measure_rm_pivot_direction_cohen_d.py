"""
Measure direction signal at regression-mean (RM) zigzag pivots.

Reproduces (and stress-tests) the journal claim (2026-04-21):
  "Direction at zigzag pivots via 1m_z_se residual, Cohen d = -2.46
   walk-forward, 86% oracle accuracy"

Method:
  1. Per day: rolling 60-bar OLS on 1m closes → RM series
  2. Live-safe zigzag (confirmation-only, no lookahead) on RM series
  3. At each confirmed pivot bar: record residual (1m_z_se), pivot type,
     and next-leg direction (known once next pivot confirms)
  4. Metrics per R ∈ {$2, $4, $6, $10}:
     - Cohen d of residuals between HIGH vs LOW pivot groups
     - Accuracy: residual-sign predicts next-leg direction
     - N pivots/day (sanity)
  5. Walk-forward:
     - Monthly: Feb–Dec 2025
     - IS pooled: all 2025
     - OOS: 2026 Jan–Feb

Output:
  - research/rm_pivot/findings/YYYY-MM-DD_cohen_d_verify.md
  - research/rm_pivot/findings/YYYY-MM-DD_cohen_d_verify.png

Reproduction:
  python tools/measure_rm_pivot_direction_cohen_d.py
"""
import os
import sys
import glob
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Paths ──────────────────────────────────────────────────────────────
ATLAS_1M_DIR = 'DATA/ATLAS/1m'
FEATURES_5S_DIR = 'DATA/ATLAS/FEATURES_5s'
FINDINGS_DIR = 'research/rm_pivot/findings'

# ── Parameters ─────────────────────────────────────────────────────────
REG_WINDOW = 60            # rolling OLS bars (1m = 60 min)
DOLLAR_PER_POINT = 2.0     # MNQ
R_SWEEP_DOLLARS = [2.0, 4.0, 6.0, 10.0]
RES_COL = '1m_z_se'


# ══════════════════════════════════════════════════════════════════════
# Math helpers
# ══════════════════════════════════════════════════════════════════════

def rolling_rm(closes: np.ndarray, window: int = REG_WINDOW) -> np.ndarray:
    """Rolling-OLS fitted-value (last-bar) series. NaN for warm-up bars."""
    n = len(closes)
    rm = np.full(n, np.nan)
    x = np.arange(window, dtype=np.float64)
    xm = x.mean()
    dx = x - xm
    denom = float((dx * dx).sum())
    if denom < 1e-12:
        return rm
    for i in range(window - 1, n):
        y = closes[i - window + 1: i + 1]
        ym = y.mean()
        slope = float((dx * (y - ym)).sum() / denom)
        intercept = ym - slope * xm
        rm[i] = intercept + slope * (window - 1)
    return rm


def zigzag_confirmations(series: np.ndarray, r_points: float):
    """Live-safe zigzag. Returns list of (confirm_idx, pivot_type, extreme_idx).

    Pivot is emitted at the CONFIRMATION bar (where retracement crosses
    r_points). No lookahead — the confirmation bar is the earliest bar
    at which the prior extreme can be known as a pivot in live data.
    """
    out = []
    leg_dir = None
    extreme_val = None
    extreme_idx = -1
    for i in range(len(series)):
        v = series[i]
        if np.isnan(v):
            continue
        if extreme_val is None:
            extreme_val, extreme_idx = v, i
            continue
        if leg_dir is None:
            if v - extreme_val >= r_points:
                leg_dir = 'up'
                extreme_val, extreme_idx = v, i
            elif extreme_val - v >= r_points:
                leg_dir = 'down'
                extreme_val, extreme_idx = v, i
            elif v > extreme_val or v < extreme_val:
                extreme_val, extreme_idx = v, i
        elif leg_dir == 'up':
            if v > extreme_val:
                extreme_val, extreme_idx = v, i
            elif extreme_val - v >= r_points:
                out.append((i, 'HIGH', extreme_idx))
                leg_dir = 'down'
                extreme_val, extreme_idx = v, i
        else:
            if v < extreme_val:
                extreme_val, extreme_idx = v, i
            elif v - extreme_val >= r_points:
                out.append((i, 'LOW', extreme_idx))
                leg_dir = 'up'
                extreme_val, extreme_idx = v, i
    return out


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled-std effect size)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float('nan')
    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return float('nan')
    return float((a.mean() - b.mean()) / pooled)


# ══════════════════════════════════════════════════════════════════════
# Per-day + accumulation
# ══════════════════════════════════════════════════════════════════════

def residual_at_bar(res_ts: np.ndarray, res_vals: np.ndarray,
                    target_ts: int) -> float:
    """Residual (1m_z_se) at-or-before the 1m bar's timestamp.
    Uses searchsorted on the 5s feature file. No lookahead."""
    if len(res_ts) == 0:
        return float('nan')
    idx = np.searchsorted(res_ts, target_ts, side='right') - 1
    if idx < 0:
        return float('nan')
    return float(res_vals[idx])


def collect_pivots_per_day(day_path: str):
    """Return a dict: R_dollars → list of {pivot_idx, pivot_type, residual,
    next_leg_dir} for that day's RM zigzag pivots."""
    day = os.path.basename(day_path).replace('.parquet', '')
    feat_path = os.path.join(FEATURES_5S_DIR, f'{day}.parquet')
    if not os.path.exists(feat_path):
        return None, day

    df_min = (pd.read_parquet(day_path)
              .sort_values('timestamp').reset_index(drop=True))
    closes = df_min['close'].values.astype(np.float64)
    ts_1m = df_min['timestamp'].values.astype(np.int64)

    df_feat = (pd.read_parquet(feat_path)
               .sort_values('timestamp').reset_index(drop=True))
    if RES_COL not in df_feat.columns:
        return None, day
    res_ts = df_feat['timestamp'].values.astype(np.int64)
    res_vals = df_feat[RES_COL].values.astype(np.float64)

    rm = rolling_rm(closes, REG_WINDOW)

    results = {}
    for r_dollars in R_SWEEP_DOLLARS:
        r_points = r_dollars / DOLLAR_PER_POINT
        confs = zigzag_confirmations(rm, r_points)
        rows = []
        for k in range(len(confs)):
            conf_idx, ptype, ex_idx = confs[k]
            # Next-leg direction: at a HIGH pivot, the new leg goes DOWN; at
            # a LOW pivot, the new leg goes UP. This is a direct consequence
            # of the zigzag definition, not an extra prediction.
            next_leg = 'DOWN' if ptype == 'HIGH' else 'UP'
            # Residual at the confirmation bar (live-safe: uses the feature
            # value at the 1m bar's close timestamp).
            target_ts = int(ts_1m[conf_idx])
            res = residual_at_bar(res_ts, res_vals, target_ts)
            rows.append({
                'day': day,
                'conf_idx': conf_idx,
                'pivot_type': ptype,
                'residual': res,
                'next_leg': next_leg,
            })
        results[r_dollars] = rows
    return results, day


# ══════════════════════════════════════════════════════════════════════
# Metric computation on a pooled row set
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(rows, label):
    """rows: list of dicts. Returns a metrics dict for one R × one cohort."""
    rows = [r for r in rows if not np.isnan(r['residual'])]
    n = len(rows)
    if n < 10:
        return {'label': label, 'n': n, 'd': float('nan'),
                'acc': float('nan'), 'mean_high': float('nan'),
                'mean_low': float('nan')}
    high_res = np.array([r['residual'] for r in rows
                          if r['pivot_type'] == 'HIGH'])
    low_res = np.array([r['residual'] for r in rows
                         if r['pivot_type'] == 'LOW'])
    d = cohen_d(high_res, low_res)

    # Accuracy of residual-sign predicting next-leg direction
    # Mean-reversion rule: residual > 0 (price above RM) → next leg DOWN
    #                     residual < 0 (price below RM) → next leg UP
    correct = 0
    total = 0
    for r in rows:
        if r['residual'] == 0:
            continue
        pred = 'DOWN' if r['residual'] > 0 else 'UP'
        if pred == r['next_leg']:
            correct += 1
        total += 1
    acc = correct / total * 100 if total else float('nan')

    return {
        'label': label,
        'n': n,
        'n_high': len(high_res),
        'n_low': len(low_res),
        'mean_high': float(high_res.mean()) if len(high_res) else float('nan'),
        'mean_low': float(low_res.mean()) if len(low_res) else float('nan'),
        'd': d,
        'acc': acc,
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    is_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2025_*.parquet')))
    oos_paths = sorted(glob.glob(os.path.join(ATLAS_1M_DIR, '2026_*.parquet')))
    print(f'IS: {len(is_paths)} days  |  OOS: {len(oos_paths)} days')

    # Collect all pivot rows tagged with day + month + dataset
    all_rows = []   # each row has R_dollars, month, dataset, ...
    def _ingest(paths, dataset_tag):
        for p in tqdm(paths, desc=dataset_tag, unit='day'):
            results, day = collect_pivots_per_day(p)
            if results is None:
                continue
            month = day[:7]  # e.g. '2025_06'
            for r_dollars, rows in results.items():
                for row in rows:
                    row['r_dollars'] = r_dollars
                    row['month'] = month
                    row['dataset'] = dataset_tag
                all_rows.extend(rows)

    _ingest(is_paths, 'IS')
    _ingest(oos_paths, 'OOS')

    df = pd.DataFrame(all_rows)
    if df.empty:
        print('No rows collected — check data paths.')
        return

    # ── Compute metrics across R × cohort ─────────────────────────────
    report_lines = []
    report_lines.append('# Cohen d verification at RM zigzag pivots')
    report_lines.append('')
    report_lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    report_lines.append(f'Ref: `research/rm_pivot/cycle_01.md`')
    report_lines.append('')
    report_lines.append('## Method')
    report_lines.append('')
    report_lines.append('- Rolling 60-bar OLS on 1m closes → RM series')
    report_lines.append('- Live-safe zigzag on RM (confirmation-only, no lookahead)')
    report_lines.append('- Pivot type + residual (`1m_z_se`) at confirmation bar + next-leg direction')
    report_lines.append('- Cohen d: residual distribution HIGH pivots vs LOW pivots')
    report_lines.append('- Accuracy: residual-sign predicts next-leg (mean-reversion rule)')
    report_lines.append('')

    # Pooled IS / OOS per R
    report_lines.append('## Pooled results (all days)')
    report_lines.append('')
    report_lines.append('| R ($) | Cohort | N pivots | N HIGH | N LOW | mean_HIGH | mean_LOW | Cohen d | Accuracy |')
    report_lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for r in R_SWEEP_DOLLARS:
        for ds in ['IS', 'OOS']:
            sub = df[(df.r_dollars == r) & (df.dataset == ds)].to_dict('records')
            m = compute_metrics(sub, f'{ds} R=${r:.0f}')
            report_lines.append(
                f'| ${r:.0f} | {ds} | {m["n"]} | {m["n_high"]} | {m["n_low"]} '
                f'| {m["mean_high"]:+.3f} | {m["mean_low"]:+.3f} '
                f'| **{m["d"]:+.2f}** | {m["acc"]:.1f}% |'
            )
    report_lines.append('')

    # Monthly walk-forward (IS 2025 only, per R)
    report_lines.append('## IS monthly walk-forward')
    report_lines.append('')
    report_lines.append('Stability check: does |d| hold month-to-month, or is it a pooled-average illusion?')
    report_lines.append('')
    for r in R_SWEEP_DOLLARS:
        report_lines.append(f'### R = ${r:.0f}')
        report_lines.append('')
        report_lines.append('| Month | N | Cohen d | Accuracy |')
        report_lines.append('|---|---:|---:|---:|')
        monthly = []
        for month in sorted(df[df.dataset == 'IS']['month'].unique()):
            sub = df[(df.r_dollars == r) & (df.dataset == 'IS')
                     & (df.month == month)].to_dict('records')
            m = compute_metrics(sub, f'{month} R=${r:.0f}')
            monthly.append(m)
            report_lines.append(f'| {month} | {m["n"]} | {m["d"]:+.2f} | {m["acc"]:.1f}% |')
        ds_vals = [m['d'] for m in monthly if not np.isnan(m['d'])]
        if ds_vals:
            report_lines.append(f'| **std(d)** |  | **{np.std(ds_vals):.2f}** |  |')
            report_lines.append(f'| **mean(d)** |  | **{np.mean(ds_vals):+.2f}** |  |')
        report_lines.append('')

    # ── Chart: distributions + monthly trend at R=$4 ──────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: residual distribution at pivots (R=$4 IS)
    ax = axes[0]
    sub = df[(df.r_dollars == 4.0) & (df.dataset == 'IS')
             & df.residual.notna()]
    if len(sub) > 0:
        hi = sub[sub.pivot_type == 'HIGH']['residual']
        lo = sub[sub.pivot_type == 'LOW']['residual']
        bins = np.linspace(-4, 4, 50)
        ax.hist(hi, bins=bins, alpha=0.6, color='tab:red', label=f'HIGH (n={len(hi)})')
        ax.hist(lo, bins=bins, alpha=0.6, color='tab:green', label=f'LOW (n={len(lo)})')
        ax.axvline(0, color='black', linewidth=0.6)
        ax.set_title(f'IS R=$4: residual at pivots')
        ax.set_xlabel('1m_z_se at pivot')
        ax.legend()
        ax.grid(alpha=0.3)

    # Panel 2: d vs R (pooled IS & OOS)
    ax = axes[1]
    for ds, color in [('IS', 'tab:blue'), ('OOS', 'tab:orange')]:
        ds_ds = []
        for r in R_SWEEP_DOLLARS:
            sub = df[(df.r_dollars == r) & (df.dataset == ds)].to_dict('records')
            m = compute_metrics(sub, '')
            ds_ds.append(m['d'])
        ax.plot(R_SWEEP_DOLLARS, ds_ds, 'o-', label=ds, color=color)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axhline(-1.5, color='tab:green', linestyle='--', linewidth=0.8, label='|d|=1.5 gate')
    ax.axhline(1.5, color='tab:green', linestyle='--', linewidth=0.8)
    ax.set_xlabel('R ($)')
    ax.set_ylabel('Cohen d (HIGH − LOW)')
    ax.set_title('Cohen d by retracement threshold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: monthly d trend at R=$4 IS
    ax = axes[2]
    months = sorted(df[df.dataset == 'IS']['month'].unique())
    ds_monthly = []
    for m in months:
        sub = df[(df.r_dollars == 4.0) & (df.dataset == 'IS')
                 & (df.month == m)].to_dict('records')
        mm = compute_metrics(sub, '')
        ds_monthly.append(mm['d'])
    ax.plot(range(len(months)), ds_monthly, 'o-', color='tab:blue')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.axhline(-1.5, color='tab:green', linestyle='--', linewidth=0.8)
    ax.axhline(1.5, color='tab:green', linestyle='--', linewidth=0.8)
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels([m.replace('2025_', '') for m in months], rotation=45)
    ax.set_ylabel('Cohen d (R=$4)')
    ax.set_title('IS monthly d trend (stability)')
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # ── Write outputs ─────────────────────────────────────────────────
    os.makedirs(FINDINGS_DIR, exist_ok=True)
    date_tag = datetime.now().strftime('%Y-%m-%d')
    md_path = os.path.join(FINDINGS_DIR, f'{date_tag}_cohen_d_verify.md')
    png_path = os.path.join(FINDINGS_DIR, f'{date_tag}_cohen_d_verify.png')

    report_lines.append('## Chart')
    report_lines.append('')
    report_lines.append(f'![distributions]({os.path.basename(png_path)})')
    report_lines.append('')

    report_lines.append('## Reproduction')
    report_lines.append('')
    report_lines.append('```')
    report_lines.append('python tools/measure_rm_pivot_direction_cohen_d.py')
    report_lines.append('```')
    report_lines.append('')

    plt.savefig(png_path, dpi=110, bbox_inches='tight')
    plt.close()

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f'Wrote: {md_path}')
    print(f'Wrote: {png_path}')


if __name__ == '__main__':
    main()
