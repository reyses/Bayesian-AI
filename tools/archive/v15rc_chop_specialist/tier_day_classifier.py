"""
Tier day classifier — find day-level features that separate a tier's
BLEED days from its HARVEST days.

Motivation: RIDE_AGAINST on IS has 92 days losing >$200 AND 76 days
winning >$200 — same tier, same entry logic, opposite outcomes. Something
about the MARKET STATE of each day distinguishes which side it lands on.
If we find that signal, we can kill the tier on bleed days and keep it on
harvest days — a day-level regime filter, not an entry-level filter.

Process:
  1. Load blended IS + OOS trade pickles.
  2. Group trades by day. For target tier, sum tier PnL per day.
     - BLEED  day: tier PnL <= -threshold (default $200)
     - HARVEST day: tier PnL >= +threshold
     - NEUTRAL (excluded from separation analysis)
  3. Compute day-level features for each day:
     - Mean of each of 91 entry features (averaged across ALL trades
       that day, so the day signature reflects the market state)
     - + derived: day N trades, day entry_price range, % of trades
       that are the target tier, mean & stdev of trade-level PnL,
       first-trade-of-day time
  4. Cohen d on each day-feature between bleed and harvest cohorts.
  5. Walk-forward: shortlist features where sign(d_IS) == sign(d_OOS)
     AND min(|d_IS|, |d_OOS|) >= 0.3.

Usage:
    python tools/tier_day_classifier.py
    python tools/tier_day_classifier.py --tier FADE_CALM
    python tools/tier_day_classifier.py --threshold 150

Output: reports/findings/tier_day_classifier_<TIER>.md
"""
import os
import sys
import pickle
import argparse
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_v2.features import FEATURE_NAMES


TRADES_DIR = 'training_iso/output/trades'


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_day_features(day_trades, target_tier):
    """Return (feat_dict, tier_pnl, tier_trade_count).

    Day features:
      - mean_<feat> for each of the 91 entry features (averaged across
        ALL trades on this day, regardless of tier)
      - day_n_trades
      - day_entry_range = max(entry_price) - min(entry_price)
      - day_entry_mid = mean(entry_price)
      - day_ride_against_share = <target>_trades / n_trades
      - day_mean_trade_pnl
      - day_std_trade_pnl
      - day_first_hour_frac (fraction of trades in first hour)
    """
    n = len(day_trades)
    if n == 0:
        return None, 0.0, 0
    prices = np.array([t.get('entry_price', 0.0) for t in day_trades], dtype=np.float64)
    pnls = np.array([t.get('pnl', 0.0) for t in day_trades], dtype=np.float64)
    timestamps = np.array([t.get('timestamp', 0.0) for t in day_trades], dtype=np.float64)

    # Mean each entry feature across day
    feats_stack = []
    for t in day_trades:
        ef = t.get('entry_79d')
        if ef is None or len(ef) < 91:
            continue
        feats_stack.append(np.asarray(ef, dtype=np.float64)[:91])
    if not feats_stack:
        return None, 0.0, 0
    feats_stack = np.vstack(feats_stack)
    feat_means = feats_stack.mean(axis=0)

    out = {}
    for i, name in enumerate(FEATURE_NAMES[:91]):
        out[f'mean_{name}'] = float(feat_means[i])
    # Also a derived z_range day-signature
    out['mean_1h_z_range'] = float(feat_means[FEATURE_NAMES.index('1h_z_high')]
                                   - feat_means[FEATURE_NAMES.index('1h_z_low')])

    # Derived
    out['day_n_trades'] = float(n)
    out['day_entry_range'] = float(prices.max() - prices.min())
    out['day_entry_mid'] = float(prices.mean())
    tier_count = sum(1 for t in day_trades if t.get('entry_tier') == target_tier)
    out['day_target_tier_share'] = tier_count / n
    out['day_target_tier_n'] = float(tier_count)
    out['day_mean_trade_pnl'] = float(pnls.mean())
    out['day_std_trade_pnl'] = float(pnls.std(ddof=0))
    # First-hour fraction: trades within 3600s of first trade
    if len(timestamps) > 1:
        t0 = timestamps.min()
        first_hour_n = int((timestamps - t0 <= 3600).sum())
        out['day_first_hour_frac'] = first_hour_n / n
    else:
        out['day_first_hour_frac'] = 1.0

    tier_pnl = sum(t['pnl'] for t in day_trades if t.get('entry_tier') == target_tier)
    return out, float(tier_pnl), tier_count


def classify_days(trades, target_tier, threshold):
    """Return list of dicts: {day, tier_pnl, cls, feats}."""
    by_day = defaultdict(list)
    for t in trades:
        by_day[t.get('day', 'unknown')].append(t)

    out = []
    for day in sorted(by_day.keys()):
        dt = by_day[day]
        feats, tier_pnl, tier_n = compute_day_features(dt, target_tier)
        if feats is None or tier_n == 0:
            continue  # skip days where the target tier didn't trade
        if tier_pnl <= -threshold:
            cls = 'bleed'
        elif tier_pnl >= threshold:
            cls = 'harvest'
        else:
            cls = 'neutral'
        out.append({
            'day': day,
            'tier_pnl': tier_pnl,
            'tier_n': tier_n,
            'cls': cls,
            'feats': feats,
        })
    return out


def cohen_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb)
                     / max(len(a) + len(b) - 2, 1))
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def rank_features(day_records):
    """Cohen d per feature between bleed cohort and harvest cohort.
    Positive d => feature is HIGHER on bleed days than harvest days.
    Returns list of {name, d, bleed_mean, harvest_mean, n_bleed, n_harvest}.
    """
    bleed = [r for r in day_records if r['cls'] == 'bleed']
    harvest = [r for r in day_records if r['cls'] == 'harvest']
    if len(bleed) < 5 or len(harvest) < 5:
        return [], len(bleed), len(harvest)
    feat_names = list(bleed[0]['feats'].keys())
    rows = []
    for name in feat_names:
        a = [r['feats'][name] for r in bleed if name in r['feats']]
        b = [r['feats'][name] for r in harvest if name in r['feats']]
        d = cohen_d(a, b)
        if np.isnan(d):
            d = 0.0
        rows.append({
            'name': name,
            'd': float(d),
            'abs_d': float(abs(d)),
            'bleed_mean': float(np.mean(a)) if a else 0.0,
            'harvest_mean': float(np.mean(b)) if b else 0.0,
            'n_bleed': len(a),
            'n_harvest': len(b),
        })
    rows.sort(key=lambda r: -r['abs_d'])
    return rows, len(bleed), len(harvest)


def cohort_stats(day_records, cls):
    days = [r for r in day_records if r['cls'] == cls]
    pnls = [r['tier_pnl'] for r in days]
    trades = [r['tier_n'] for r in days]
    return {
        'n_days': len(days),
        'total_pnl': float(sum(pnls)),
        'mean_pnl': float(np.mean(pnls)) if pnls else 0.0,
        'median_pnl': float(np.median(pnls)) if pnls else 0.0,
        'total_trades': int(sum(trades)),
    }


def write_report(is_records, oos_records, is_rank, oos_rank,
                 is_bleed_n, is_harv_n, oos_bleed_n, oos_harv_n,
                 target_tier, threshold, out_path):
    L = []
    L.append(f'# Tier day classifier — {target_tier}')
    L.append('')
    L.append(f'Threshold: BLEED = tier PnL <= -${threshold}, '
             f'HARVEST = tier PnL >= +${threshold}.')
    L.append('')

    # Section 1: cohort stats
    L.append('## 1. Cohort stats')
    L.append('')
    L.append('| Dataset | Class | Days | Total $ | Mean $ | Median $ | Trades |')
    L.append('|---|---|---:|---:|---:|---:|---:|')
    for label, records in [('IS', is_records), ('OOS', oos_records)]:
        for cls in ('bleed', 'harvest', 'neutral'):
            s = cohort_stats(records, cls)
            L.append(f'| {label} | {cls} | {s["n_days"]} | '
                     f'${s["total_pnl"]:+,.0f} | ${s["mean_pnl"]:+,.0f} | '
                     f'${s["median_pnl"]:+,.0f} | {s["total_trades"]:,} |')
    L.append('')
    L.append('_The bleed/harvest $ totals are what\'s at stake if we can '
             'separate them with a day-level filter._')
    L.append('')

    # Section 2: top IS discriminators
    L.append('## 2. Top-20 day features — IS')
    L.append('')
    L.append(f'(N bleed={is_bleed_n}, N harvest={is_harv_n}) '
             'Positive d => feature is HIGHER on bleed days.')
    L.append('')
    L.append('| Rank | Feature | d | bleed mean | harvest mean |')
    L.append('|---:|---|---:|---:|---:|')
    for i, r in enumerate(is_rank[:20]):
        sgn = '+' if r['d'] >= 0 else ''
        L.append(f'| {i+1} | `{r["name"]}` | {sgn}{r["d"]:.2f} | '
                 f'{r["bleed_mean"]:+.3f} | {r["harvest_mean"]:+.3f} |')
    L.append('')

    # Section 3: top OOS discriminators
    L.append('## 3. Top-20 day features — OOS')
    L.append('')
    L.append(f'(N bleed={oos_bleed_n}, N harvest={oos_harv_n})')
    L.append('')
    L.append('| Rank | Feature | d | bleed mean | harvest mean |')
    L.append('|---:|---|---:|---:|---:|')
    for i, r in enumerate(oos_rank[:20]):
        sgn = '+' if r['d'] >= 0 else ''
        L.append(f'| {i+1} | `{r["name"]}` | {sgn}{r["d"]:.2f} | '
                 f'{r["bleed_mean"]:+.3f} | {r["harvest_mean"]:+.3f} |')
    L.append('')

    # Section 4: walk-forward stable shortlist
    L.append('## 4. Walk-forward stable shortlist')
    L.append('')
    L.append('Features where sign(d_IS) matches sign(d_OOS) AND '
             'min(|d_IS|, |d_OOS|) >= 0.30. Sorted by min |d| descending.')
    L.append('')
    is_by_name = {r['name']: r for r in is_rank}
    oos_by_name = {r['name']: r for r in oos_rank}
    shortlist = []
    for name in is_by_name:
        r_is = is_by_name[name]
        r_oos = oos_by_name.get(name)
        if r_oos is None:
            continue
        if r_is['d'] * r_oos['d'] <= 0:
            continue
        mn = min(abs(r_is['d']), abs(r_oos['d']))
        if mn < 0.30:
            continue
        shortlist.append({
            'name': name,
            'd_is': r_is['d'],
            'd_oos': r_oos['d'],
            'min_abs_d': mn,
            'is_bleed_mean': r_is['bleed_mean'],
            'is_harvest_mean': r_is['harvest_mean'],
            'oos_bleed_mean': r_oos['bleed_mean'],
            'oos_harvest_mean': r_oos['harvest_mean'],
        })
    shortlist.sort(key=lambda s: -s['min_abs_d'])

    if shortlist:
        L.append('| Feature | d_IS | d_OOS | min |d| | IS bleed mean | '
                 'IS harv mean | OOS bleed mean | OOS harv mean |')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for s in shortlist[:30]:
            sgn_is = '+' if s['d_is'] >= 0 else ''
            sgn_oos = '+' if s['d_oos'] >= 0 else ''
            L.append(f'| `{s["name"]}` | {sgn_is}{s["d_is"]:.2f} | '
                     f'{sgn_oos}{s["d_oos"]:.2f} | {s["min_abs_d"]:.2f} | '
                     f'{s["is_bleed_mean"]:+.3f} | '
                     f'{s["is_harvest_mean"]:+.3f} | '
                     f'{s["oos_bleed_mean"]:+.3f} | '
                     f'{s["oos_harvest_mean"]:+.3f} |')
    else:
        L.append('_No features clear the bar. Bleed vs harvest days are '
                 'not statically separable by any single day-level feature '
                 'with walk-forward stability._')
    L.append('')

    # Interpretation hints
    L.append('## 5. How to read this')
    L.append('')
    L.append('- **Non-empty shortlist** = there is a real day-level signal '
             'to classify on. Next step: build a decision rule from the '
             'top 2-3 features (e.g. quantile split of d_IS>0.5 features) '
             'and backtest it on OOS.')
    L.append('- **Empty shortlist** = day outcomes are not predictable '
             'from aggregated market state alone. The bleed/harvest split '
             'comes from path-dependent noise within the day, not entry-time '
             'regime. If so, day-level classification is the wrong frame — '
             'move to path-based signals (e.g. drawdown in first N trades '
             'of the day as an intraday kill switch).')
    L.append('')

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


def print_summary(label, records, rank, bleed_n, harv_n, tier, threshold):
    s_bleed = cohort_stats(records, 'bleed')
    s_harv = cohort_stats(records, 'harvest')
    s_neut = cohort_stats(records, 'neutral')
    print(f'\n=== {label} — {tier} (threshold ${threshold}) ===')
    print(f'  BLEED:   {s_bleed["n_days"]:>3} days  '
          f'${s_bleed["total_pnl"]:>+9,.0f}  '
          f'median ${s_bleed["median_pnl"]:>+6,.0f}')
    print(f'  HARVEST: {s_harv["n_days"]:>3} days  '
          f'${s_harv["total_pnl"]:>+9,.0f}  '
          f'median ${s_harv["median_pnl"]:>+6,.0f}')
    print(f'  NEUTRAL: {s_neut["n_days"]:>3} days  '
          f'${s_neut["total_pnl"]:>+9,.0f}  '
          f'median ${s_neut["median_pnl"]:>+6,.0f}')
    print()
    print(f'  Top-10 day discriminators (|d| rank):')
    for i, r in enumerate(rank[:10]):
        sgn = '+' if r['d'] >= 0 else ''
        print(f'    {i+1:>2}. {r["name"]:<30} d={sgn}{r["d"]:>5.2f}  '
              f'bleed={r["bleed_mean"]:>+8.3f}  '
              f'harv={r["harvest_mean"]:>+8.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tier', default='RIDE_AGAINST')
    ap.add_argument('--threshold', type=float, default=200.0)
    args = ap.parse_args()

    out_path = (f'reports/findings/tier_day_classifier_{args.tier}'
                f'_thr{int(args.threshold)}.md')

    datasets = {
        'IS':  os.path.join(TRADES_DIR, 'blended_is.pkl'),
        'OOS': os.path.join(TRADES_DIR, 'blended_oos.pkl'),
    }

    results = {}
    for label, path in datasets.items():
        if not os.path.exists(path):
            print(f'{label}: file not found, skipping')
            continue
        print(f'Loading {path}...')
        trades = load(path)
        records = classify_days(trades, args.tier, args.threshold)
        rank, bleed_n, harv_n = rank_features(records)
        results[label] = {
            'records': records,
            'rank': rank,
            'bleed_n': bleed_n,
            'harv_n': harv_n,
        }
        print_summary(label, records, rank, bleed_n, harv_n,
                      args.tier, args.threshold)

    if 'IS' in results and 'OOS' in results:
        write_report(
            results['IS']['records'], results['OOS']['records'],
            results['IS']['rank'], results['OOS']['rank'],
            results['IS']['bleed_n'], results['IS']['harv_n'],
            results['OOS']['bleed_n'], results['OOS']['harv_n'],
            args.tier, args.threshold, out_path,
        )
        print()
        print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
