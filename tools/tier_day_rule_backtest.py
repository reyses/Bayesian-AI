"""
Tier day rule backtest — apply a combined-feature decision rule from the
walk-forward-stable shortlist and measure $ lift.

Rule shape: score = signed sum of z-scored day features
  +z(mean_5m_variance_ratio)    (bleed days have HIGHER 5m vr)
  -z(mean_1h_variance_ratio)    (bleed days have LOWER 1h vr)
  +z(mean_1h_z_range)           (bleed days have WIDER 1h z_range)
  [-z(day_entry_range)]         (bleed days have SMALLER entry range)

Two variants:
  A — LIVE-READABLE: uses the first 3 features only. All three are rolling
      features of market state, computable in real-time at any bar. Can be
      evaluated pre-market-open (first trade of day) as a kill switch.
  B — INTRADAY: includes day_entry_range (only known after accumulation).
      Evaluable once the day has enough trades. Shows the separation
      ceiling if we used all four features.

Calibration: z-scores and thresholds computed on IS. OOS uses the IS
normalization and a frozen IS-calibrated threshold (honest walk-forward).

Threshold sweep: skip the top-X% IS days by bleed score, measure net $
delta vs baseline. X in {10, 15, 20, 25, 30, 35, 40, 45, 50}.

For each X, we report:
  - Bleed days caught (of total bleed days)
  - Harvest days false-flagged (of total harvest)
  - $ on skipped days (signed — should be very negative)
  - $ delta vs baseline (= -$ on skipped days)

Usage:
    python tools/tier_day_rule_backtest.py
    python tools/tier_day_rule_backtest.py --tier FADE_CALM

Output: reports/findings/tier_day_rule_backtest_<TIER>.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tier_day_classifier import classify_days


TRADES_DIR = 'training_iso/output/trades'
THRESHOLD_SWEEP_PCT = [10, 15, 20, 25, 30, 35, 40, 45, 50]

# Feature set + sign (positive sign = higher feature pushes toward bleed)
RULE_A_FEATURES = [
    ('mean_5m_variance_ratio',  +1),
    ('mean_1h_variance_ratio',  -1),
    ('mean_1h_z_range',         +1),
]

RULE_B_FEATURES = RULE_A_FEATURES + [
    ('day_entry_range',         -1),
]


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_day_frame(records, features):
    """Return X (n_days, n_feats), y (n_days tier_pnl), classes, days."""
    n = len(records)
    nf = len(features)
    X = np.zeros((n, nf), dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    classes = []
    days = []
    for i, r in enumerate(records):
        for j, (name, _sign) in enumerate(features):
            X[i, j] = r['feats'].get(name, 0.0)
        y[i] = r['tier_pnl']
        classes.append(r['cls'])
        days.append(r['day'])
    return X, y, classes, days


def z_normalize(X, mean=None, std=None):
    """Return (Z, mean, std). If mean/std supplied, use those (for OOS)."""
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        std = np.where(std < 1e-9, 1.0, std)
    Z = (X - mean) / std
    return Z, mean, std


def compute_score(Z, features):
    """Return signed-sum bleed score per day."""
    signs = np.array([s for _, s in features], dtype=np.float64)
    return Z @ signs


def sweep_thresholds(score, y, classes, pcts):
    """For each percentile (top-X% skip), compute rule stats.
    Returns list of dicts."""
    n = len(score)
    order = np.argsort(-score)  # highest bleed score first
    classes = np.asarray(classes)
    out = []
    for pct in pcts:
        k = max(1, int(round(n * pct / 100.0)))
        skip_idx = order[:k]
        keep_idx = order[k:]
        skip_pnl = float(y[skip_idx].sum())
        keep_pnl = float(y[keep_idx].sum())
        baseline_pnl = float(y.sum())
        delta = keep_pnl - baseline_pnl  # = -skip_pnl
        cls_skip = classes[skip_idx]
        cls_keep = classes[keep_idx]
        bleed_total = int((classes == 'bleed').sum())
        harv_total = int((classes == 'harvest').sum())
        bleed_caught = int((cls_skip == 'bleed').sum())
        harv_fp = int((cls_skip == 'harvest').sum())
        neut_skipped = int((cls_skip == 'neutral').sum())
        score_threshold = float(score[skip_idx].min()) if k else float('inf')
        out.append({
            'pct': pct,
            'k_skipped': k,
            'skip_pnl': skip_pnl,
            'keep_pnl': keep_pnl,
            'baseline_pnl': baseline_pnl,
            'delta': delta,
            'threshold_score': score_threshold,
            'bleed_caught': bleed_caught,
            'bleed_total': bleed_total,
            'harv_fp': harv_fp,
            'harv_total': harv_total,
            'neut_skipped': neut_skipped,
            'bleed_catch_rate': (bleed_caught / bleed_total * 100) if bleed_total else 0,
            'harv_fp_rate': (harv_fp / harv_total * 100) if harv_total else 0,
        })
    return out


def apply_frozen_threshold(score, y, classes, threshold_score):
    """Apply a pre-computed score threshold (from IS) to a new dataset."""
    classes = np.asarray(classes)
    mask_skip = score >= threshold_score
    mask_keep = ~mask_skip
    k = int(mask_skip.sum())
    skip_pnl = float(y[mask_skip].sum())
    keep_pnl = float(y[mask_keep].sum())
    baseline_pnl = float(y.sum())
    delta = keep_pnl - baseline_pnl
    cls_skip = classes[mask_skip]
    bleed_total = int((classes == 'bleed').sum())
    harv_total = int((classes == 'harvest').sum())
    bleed_caught = int((cls_skip == 'bleed').sum())
    harv_fp = int((cls_skip == 'harvest').sum())
    neut_skipped = int((cls_skip == 'neutral').sum())
    return {
        'k_skipped': k,
        'skip_pnl': skip_pnl,
        'keep_pnl': keep_pnl,
        'baseline_pnl': baseline_pnl,
        'delta': delta,
        'threshold_score': threshold_score,
        'bleed_caught': bleed_caught,
        'bleed_total': bleed_total,
        'harv_fp': harv_fp,
        'harv_total': harv_total,
        'neut_skipped': neut_skipped,
        'bleed_catch_rate': (bleed_caught / bleed_total * 100) if bleed_total else 0,
        'harv_fp_rate': (harv_fp / harv_total * 100) if harv_total else 0,
    }


def render_sweep_table(sweep, label):
    lines = []
    lines.append(f'### {label}')
    lines.append('')
    lines.append('| Top-X% skip | Days skipped | Bleed caught | Bleed catch % | '
                 'Harvest FP | Harv FP % | $ on skipped | $ delta vs baseline | '
                 'Threshold score |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for r in sweep:
        lines.append(f'| {r["pct"]}% | {r["k_skipped"]} | '
                     f'{r["bleed_caught"]}/{r["bleed_total"]} | '
                     f'{r["bleed_catch_rate"]:.0f}% | '
                     f'{r["harv_fp"]}/{r["harv_total"]} | '
                     f'{r["harv_fp_rate"]:.0f}% | '
                     f'${r["skip_pnl"]:+,.0f} | '
                     f'${r["delta"]:+,.0f} | '
                     f'{r["threshold_score"]:+.2f} |')
    lines.append('')
    return lines


def render_frozen_table(frozen_results, label):
    """Apply IS-calibrated thresholds to OOS. frozen_results: list of
    (pct, is_threshold, oos_stats)."""
    lines = []
    lines.append(f'### {label}')
    lines.append('')
    lines.append('IS-calibrated threshold applied to OOS (no OOS tuning).')
    lines.append('')
    lines.append('| IS Top-X% | IS threshold | OOS days skipped | '
                 'OOS bleed caught | OOS Bleed catch % | OOS harvest FP | '
                 'OOS FP % | OOS $ on skipped | OOS $ delta vs baseline |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for pct, is_thr, r in frozen_results:
        lines.append(f'| {pct}% | {is_thr:+.2f} | {r["k_skipped"]} | '
                     f'{r["bleed_caught"]}/{r["bleed_total"]} | '
                     f'{r["bleed_catch_rate"]:.0f}% | '
                     f'{r["harv_fp"]}/{r["harv_total"]} | '
                     f'{r["harv_fp_rate"]:.0f}% | '
                     f'${r["skip_pnl"]:+,.0f} | '
                     f'${r["delta"]:+,.0f} |')
    lines.append('')
    return lines


def print_sweep(label, sweep):
    print(f'\n--- {label} ---')
    print(f'{"X%":>4} {"Skip":>5} {"BCaught":>9} {"BCatch":>7} '
          f'{"FP":>5} {"FP%":>5} {"$Skip":>10} {"$Delta":>10}')
    for r in sweep:
        print(f'{r["pct"]:>3}% {r["k_skipped"]:>5} '
              f'{r["bleed_caught"]:>3}/{r["bleed_total"]:<5} '
              f'{r["bleed_catch_rate"]:>5.0f}%  '
              f'{r["harv_fp"]:>2}/{r["harv_total"]:<3} '
              f'{r["harv_fp_rate"]:>4.0f}% '
              f'${r["skip_pnl"]:>+9,.0f} ${r["delta"]:>+9,.0f}')


def run_rule(records_is, records_oos, features, rule_label,
             out_lines, tier, threshold):
    X_is, y_is, cls_is, days_is = build_day_frame(records_is, features)
    X_oos, y_oos, cls_oos, days_oos = build_day_frame(records_oos, features)

    Z_is, mu, sigma = z_normalize(X_is)
    Z_oos, _, _ = z_normalize(X_oos, mean=mu, std=sigma)

    score_is = compute_score(Z_is, features)
    score_oos = compute_score(Z_oos, features)

    # IS sweep
    is_sweep = sweep_thresholds(score_is, y_is, cls_is, THRESHOLD_SWEEP_PCT)

    # Apply IS-calibrated thresholds to OOS
    frozen = []
    for r in is_sweep:
        oos_r = apply_frozen_threshold(score_oos, y_oos, cls_oos,
                                       r['threshold_score'])
        frozen.append((r['pct'], r['threshold_score'], oos_r))

    out_lines.append(f'## {rule_label}')
    out_lines.append('')
    out_lines.append(f'Features (name, sign):')
    for name, sign in features:
        sgn = '+' if sign > 0 else '-'
        out_lines.append(f'  - `{sgn}z({name})`')
    out_lines.append('')
    out_lines.append(f'Baseline {tier} PnL IS: ${y_is.sum():+,.0f} '
                     f'across {len(y_is)} days.')
    out_lines.append(f'Baseline {tier} PnL OOS: ${y_oos.sum():+,.0f} '
                     f'across {len(y_oos)} days.')
    out_lines.append('')
    out_lines.append('Bleed days (<-$' + str(int(threshold))
                     + ') IS: ' + str(sum(1 for c in cls_is if c == 'bleed'))
                     + ' | OOS: ' + str(sum(1 for c in cls_oos if c == 'bleed')))
    out_lines.append('Harvest days (>+$' + str(int(threshold))
                     + ') IS: ' + str(sum(1 for c in cls_is if c == 'harvest'))
                     + ' | OOS: ' + str(sum(1 for c in cls_oos if c == 'harvest')))
    out_lines.append('')

    out_lines.extend(render_sweep_table(is_sweep,
                                        'IS sweep — threshold calibrated on IS'))
    out_lines.extend(render_frozen_table(frozen,
                                         'OOS — IS-calibrated threshold frozen'))

    print(f'\n=== {rule_label} ===')
    print(f'  Baseline {tier} IS:  ${y_is.sum():+,.0f} ({len(y_is)} days)')
    print(f'  Baseline {tier} OOS: ${y_oos.sum():+,.0f} ({len(y_oos)} days)')
    print_sweep(f'IS — {rule_label}', is_sweep)
    oos_sweep_view = [
        {'pct': p, 'k_skipped': r['k_skipped'],
         'bleed_caught': r['bleed_caught'],
         'bleed_total': r['bleed_total'],
         'bleed_catch_rate': r['bleed_catch_rate'],
         'harv_fp': r['harv_fp'],
         'harv_total': r['harv_total'],
         'harv_fp_rate': r['harv_fp_rate'],
         'skip_pnl': r['skip_pnl'],
         'delta': r['delta']}
        for (p, _, r) in frozen]
    print_sweep(f'OOS (frozen IS thresholds) — {rule_label}', oos_sweep_view)

    return is_sweep, frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='RIDE_AGAINST')
    ap.add_argument('--threshold', type=float, default=200.0)
    args = ap.parse_args()

    out_path = (f'reports/findings/tier_day_rule_backtest_{args.tier}'
                f'_thr{int(args.threshold)}.md')

    is_trades = load(os.path.join(TRADES_DIR, 'blended_is.pkl'))
    oos_trades = load(os.path.join(TRADES_DIR, 'blended_oos.pkl'))

    records_is = classify_days(is_trades, args.tier, args.threshold)
    records_oos = classify_days(oos_trades, args.tier, args.threshold)

    out = [f'# Tier day rule backtest — {args.tier}', '']
    out.append(f'Threshold: BLEED = tier PnL <= -${args.threshold}, '
               f'HARVEST = tier PnL >= +${args.threshold}.')
    out.append('')
    out.append('Z-scores and decision thresholds are calibrated on IS. '
               'OOS uses the same IS normalization and the same IS-calibrated '
               'threshold (honest walk-forward).')
    out.append('')
    out.append('**Rule A** is live-readable: three rolling market-state '
               'features computable at any bar. Useful as morning kill switch '
               'and intraday re-check.')
    out.append('')
    out.append('**Rule B** adds `day_entry_range` — only measurable once a '
               'day has accumulated trades. Evaluable intraday, not at open.')
    out.append('')

    run_rule(records_is, records_oos, RULE_A_FEATURES,
             'Rule A — 3 live-readable features',
             out, args.tier, args.threshold)

    run_rule(records_is, records_oos, RULE_B_FEATURES,
             'Rule B — 4 features incl. intraday entry range',
             out, args.tier, args.threshold)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    print()
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
