"""
Multi-TF Feature Exploratory Data Analysis.

Modules E1-E7: understand how each of the 13D features behaves
within level zones, at boundaries, before reversals, and across TFs.

Usage:
  python -m tools.feature_eda --module E1
  python -m tools.feature_eda --module all
"""
import argparse
import gc
import glob
import json
import os
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['savefig.directory'] = os.path.abspath('examples')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ATLAS = 'DATA/ATLAS'
TICK = 0.25
OUT_DIR = 'reports/findings/eda'

FEATURE_NAMES = [
    'dmi_diff', 'dmi_gap', 'vol_rel', 'dir_vol', 'velocity', 'z_se', 'price_accel',
    'std_price', 'variance_ratio', 'bar_range', 'wick_ratio',
    'vwap_distance', 'time_of_day',
]


def load_features(tf, month_str=None):
    """Load OHLCV + compute 13D features for a TF. Returns df, feats."""
    from core_v2.statistical_field_engine import StatisticalFieldEngine
    from training.train_trade_cnn import extract_features_13d

    files = sorted(glob.glob(os.path.join(ATLAS, tf, '*.parquet')))
    if not files:
        return None, None
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    if month_str:
        m_start = pd.Timestamp(f'{month_str}-01').timestamp()
        m = int(month_str[5:7])
        y = int(month_str[:4])
        if m == 12:
            m_end = pd.Timestamp(f'{y+1}-01-01').timestamp()
        else:
            m_end = pd.Timestamp(f'{y}-{m+1:02d}-01').timestamp()
        df = df[(df['timestamp'] >= m_start) & (df['timestamp'] < m_end)].reset_index(drop=True)

    if len(df) < 20:
        return None, None

    sfe = StatisticalFieldEngine()
    states = sfe.batch_compute_states(df)
    feats = extract_features_13d(states, df)
    del states; gc.collect()

    return df, feats


def get_levels_for_month(month_str):
    """Find the level file that covers this month."""
    files = sorted(glob.glob('DATA/levels/levels_*.json'))
    best = None
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        if d['date'][:7] <= month_str:
            best = d['levels']
    return best


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# E1: Feature Distributions Per Zone
# ============================================================
def run_e1(months):
    """Feature distributions within each level zone."""
    ensure_dir()
    print(f"\n{'='*60}\nE1: Feature Distributions Per Zone\n{'='*60}")

    all_rows = []
    for month in tqdm(months, desc="E1"):
        levels = get_levels_for_month(month)
        if not levels or len(levels) < 2:
            continue
        prices_lev = sorted([l['price'] for l in levels])

        df, feats = load_features('1h', month)
        if df is None:
            continue
        closes = df['close'].values

        for zi in range(len(prices_lev) - 1):
            lo, hi = prices_lev[zi], prices_lev[zi + 1]
            mask = (closes >= lo) & (closes <= hi)
            if mask.sum() < 10:
                continue

            zone_feats = feats[mask]
            for fi, fname in enumerate(FEATURE_NAMES):
                col = zone_feats[:, fi]
                all_rows.append({
                    'month': month, 'zone': f'{lo:.0f}-{hi:.0f}',
                    'zone_lo': lo, 'zone_hi': hi,
                    'feature': fname,
                    'mean': col.mean(), 'std': col.std(),
                    'median': np.median(col),
                    'skew': sp_stats.skew(col),
                    'p25': np.percentile(col, 25),
                    'p75': np.percentile(col, 75),
                    'n': len(col),
                })
        del df, feats; gc.collect()

    df_r = pd.DataFrame(all_rows)
    df_r.to_csv(os.path.join(OUT_DIR, 'e1_distributions.csv'), index=False)
    print(f"  Saved: {OUT_DIR}/e1_distributions.csv ({len(df_r)} rows)")

    # Summary: which features vary most across zones?
    print(f"\n  Feature variance ACROSS zones (higher = more zone-dependent):")
    for fname in FEATURE_NAMES:
        sub = df_r[df_r['feature'] == fname]
        if len(sub) > 1:
            cross_zone_var = sub['mean'].std()
            print(f"    {fname:<16} cross-zone std of mean: {cross_zone_var:.4f}")


# ============================================================
# E2: Features at Level Boundaries
# ============================================================
def run_e2(months):
    """Compare features at level boundaries vs mid-zone."""
    ensure_dir()
    print(f"\n{'='*60}\nE2: Features at Level Boundaries\n{'='*60}")

    at_boundary = {f: [] for f in FEATURE_NAMES}
    mid_zone = {f: [] for f in FEATURE_NAMES}

    for month in tqdm(months, desc="E2"):
        levels = get_levels_for_month(month)
        if not levels:
            continue
        level_prices = [l['price'] for l in levels]

        df, feats = load_features('1h', month)
        if df is None:
            continue
        closes = df['close'].values

        for i in range(len(closes)):
            dist_to_nearest = min(abs(closes[i] - lp) for lp in level_prices)
            for fi, fname in enumerate(FEATURE_NAMES):
                if dist_to_nearest < 20:  # within 20 points = at boundary
                    at_boundary[fname].append(feats[i, fi])
                elif dist_to_nearest > 100:  # 100+ points away = mid zone
                    mid_zone[fname].append(feats[i, fi])

        del df, feats; gc.collect()

    # T-test for each feature
    print(f"\n  {'Feature':<16} {'At Boundary':>12} {'Mid Zone':>12} {'T-stat':>8} {'P-value':>10} {'Sig?':>5}")
    print(f"  {'-'*65}")
    results = []
    for fname in FEATURE_NAMES:
        ab = np.array(at_boundary[fname])
        mz = np.array(mid_zone[fname])
        if len(ab) > 10 and len(mz) > 10:
            t, p = sp_stats.ttest_ind(ab, mz)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {fname:<16} {ab.mean():>12.4f} {mz.mean():>12.4f} {t:>8.2f} {p:>10.4f} {sig:>5}")
            results.append({'feature': fname, 'boundary_mean': ab.mean(), 'midzone_mean': mz.mean(),
                           't_stat': t, 'p_value': p, 'significant': p < 0.05,
                           'n_boundary': len(ab), 'n_midzone': len(mz)})

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(OUT_DIR, 'e2_boundary_vs_midzone.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/e2_boundary_vs_midzone.csv")


# ============================================================
# E3: Features Before Reversals
# ============================================================
def run_e3(months):
    """What do features look like before price reverses at a level?"""
    ensure_dir()
    print(f"\n{'='*60}\nE3: Features Before Reversals\n{'='*60}")

    lookback = 4  # bars before reversal to examine
    reversal_feats = {f: {t: [] for t in range(-lookback, 1)} for f in FEATURE_NAMES}
    breakout_feats = {f: {t: [] for t in range(-lookback, 1)} for f in FEATURE_NAMES}

    for month in tqdm(months, desc="E3"):
        levels = get_levels_for_month(month)
        if not levels:
            continue
        level_prices = [l['price'] for l in levels]

        df, feats = load_features('1h', month)
        if df is None:
            continue
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        n = len(closes)

        for lp in level_prices:
            for i in range(lookback, n - 2):
                # Touch: bar's wick reached within 10 points of level
                touched = (highs[i] >= lp - 10 and lows[i] <= lp + 10)
                if not touched:
                    continue

                # Reversal: price moved AWAY from level after touch
                # Breakout: price moved THROUGH level after touch
                if lp > closes[i]:  # level is above = resistance
                    reversed_here = closes[i + 1] < closes[i]  # price went down
                    broke_through = closes[i + 1] > lp + 10
                else:  # level is below = support
                    reversed_here = closes[i + 1] > closes[i]  # price went up
                    broke_through = closes[i + 1] < lp - 10

                target = reversal_feats if reversed_here else (breakout_feats if broke_through else None)
                if target is None:
                    continue

                for t in range(-lookback, 1):
                    idx = i + t
                    if 0 <= idx < n:
                        for fi, fname in enumerate(FEATURE_NAMES):
                            target[fname][t].append(feats[idx, fi])

        del df, feats; gc.collect()

    # Report: which features diverge before reversal vs breakout?
    print(f"\n  Features that DIVERGE before reversal vs breakout (at t=0, touch bar):")
    print(f"  {'Feature':<16} {'Reversal':>10} {'Breakout':>10} {'Diff':>8} {'P-value':>10} {'Sig?':>5}")
    print(f"  {'-'*60}")

    results = []
    for fname in FEATURE_NAMES:
        rev = np.array(reversal_feats[fname][0])
        brk = np.array(breakout_feats[fname][0])
        if len(rev) > 10 and len(brk) > 10:
            t, p = sp_stats.ttest_ind(rev, brk)
            diff = rev.mean() - brk.mean()
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
            print(f"  {fname:<16} {rev.mean():>10.4f} {brk.mean():>10.4f} {diff:>8.4f} {p:>10.4f} {sig:>5}")
            results.append({'feature': fname, 'reversal_mean': rev.mean(), 'breakout_mean': brk.mean(),
                           'diff': diff, 'p_value': p, 'n_rev': len(rev), 'n_brk': len(brk)})

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(OUT_DIR, 'e3_reversal_vs_breakout.csv'), index=False)

    # Chart: feature trajectory before reversal vs breakout
    sig_feats = [r['feature'] for r in results if r['p_value'] < 0.05][:6]
    if sig_feats:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Features Before Reversal (green) vs Breakout (red)', fontsize=14, fontweight='bold')
        for idx, fname in enumerate(sig_feats):
            ax = axes[idx // 3][idx % 3]
            t_range = range(-lookback, 1)
            rev_means = [np.mean(reversal_feats[fname][t]) if reversal_feats[fname][t] else 0 for t in t_range]
            brk_means = [np.mean(breakout_feats[fname][t]) if breakout_feats[fname][t] else 0 for t in t_range]
            ax.plot(list(t_range), rev_means, 'g-o', linewidth=2, label='Reversal')
            ax.plot(list(t_range), brk_means, 'r-s', linewidth=2, label='Breakout')
            ax.set_title(fname, fontsize=11)
            ax.set_xlabel('Bars before touch')
            ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'e3_reversal_features.png'), dpi=150, bbox_inches='tight')
        print(f"  Chart: {OUT_DIR}/e3_reversal_features.png")

    print(f"  Saved: {OUT_DIR}/e3_reversal_vs_breakout.csv")


# ============================================================
# E4: Cross-TF Feature Correlation
# ============================================================
def run_e4(months):
    """Correlation of each feature across timeframes."""
    ensure_dir()
    print(f"\n{'='*60}\nE4: Cross-TF Feature Correlation\n{'='*60}")

    tf_pairs = [('1h', '15m'), ('1h', '1m'), ('15m', '1m')]
    results = []

    for tf_high, tf_low in tf_pairs:
        print(f"\n  {tf_high} vs {tf_low}:")
        all_corrs = {fname: [] for fname in FEATURE_NAMES}

        for month in tqdm(months, desc=f"  {tf_high}-{tf_low}"):
            df_h, feats_h = load_features(tf_high, month)
            df_l, feats_l = load_features(tf_low, month)
            if df_h is None or df_l is None:
                continue

            # Align: for each higher TF bar, find the lower TF bar at the same time
            ts_h = df_h['timestamp'].values
            ts_l = df_l['timestamp'].values

            for fi, fname in enumerate(FEATURE_NAMES):
                # Sample: take lower TF value at each higher TF timestamp
                aligned_l = []
                aligned_h = []
                for i in range(len(ts_h)):
                    idx = np.searchsorted(ts_l, ts_h[i], side='right') - 1
                    if 0 <= idx < len(feats_l):
                        aligned_h.append(feats_h[i, fi])
                        aligned_l.append(feats_l[idx, fi])

                if len(aligned_h) > 20:
                    r, p = sp_stats.spearmanr(aligned_h, aligned_l)
                    all_corrs[fname].append(r)

            del df_h, feats_h, df_l, feats_l; gc.collect()

        print(f"  {'Feature':<16} {'Avg Corr':>10} {'Std':>8} {'Interpretation':>20}")
        print(f"  {'-'*55}")
        for fname in FEATURE_NAMES:
            corrs = all_corrs[fname]
            if corrs:
                avg = np.mean(corrs)
                std = np.std(corrs)
                interp = 'REDUNDANT' if abs(avg) > 0.8 else ('RELATED' if abs(avg) > 0.5 else 'INDEPENDENT')
                print(f"  {fname:<16} {avg:>10.3f} {std:>8.3f} {interp:>20}")
                results.append({'tf_high': tf_high, 'tf_low': tf_low, 'feature': fname,
                               'avg_corr': avg, 'std_corr': std})

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(OUT_DIR, 'e4_cross_tf_correlation.csv'), index=False)

    # Heatmap
    if len(results) > 0:
        fig, axes = plt.subplots(1, len(tf_pairs), figsize=(6 * len(tf_pairs), 8))
        fig.suptitle('Cross-TF Feature Correlation', fontsize=14, fontweight='bold')
        for idx, (tf_h, tf_l) in enumerate(tf_pairs):
            ax = axes[idx] if len(tf_pairs) > 1 else axes
            sub = df_r[(df_r['tf_high'] == tf_h) & (df_r['tf_low'] == tf_l)]
            if len(sub) == 0:
                continue
            corrs = sub.set_index('feature')['avg_corr']
            colors = ['#CC0000' if c < -0.5 else '#0066CC' if c > 0.5 else '#888888' for c in corrs.values]
            ax.barh(range(len(corrs)), corrs.values, color=colors, alpha=0.7)
            ax.set_yticks(range(len(corrs)))
            ax.set_yticklabels(corrs.index, fontsize=9)
            ax.set_xlabel('Spearman Correlation')
            ax.set_title(f'{tf_h} vs {tf_l}', fontsize=11)
            ax.axvline(x=0, color='black', linewidth=0.5)
            ax.set_xlim(-1, 1)
            ax.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, 'e4_cross_tf_heatmap.png'), dpi=150, bbox_inches='tight')
        print(f"\n  Chart: {OUT_DIR}/e4_cross_tf_heatmap.png")

    print(f"  Saved: {OUT_DIR}/e4_cross_tf_correlation.csv")


# ============================================================
# E5: Cross-TF Lead/Lag at Levels
# ============================================================
def run_e5(months):
    """Which TF's features change first at level reversals?"""
    ensure_dir()
    print(f"\n{'='*60}\nE5: Cross-TF Lead/Lag at Levels\n{'='*60}")

    # Use 1h and 1m: does 1h DMI flip before or after 1m DMI at levels?
    lookback = 5
    lead_counts = {fname: {'1h_leads': 0, '1m_leads': 0, 'simultaneous': 0} for fname in FEATURE_NAMES[:7]}

    for month in tqdm(months, desc="E5"):
        levels = get_levels_for_month(month)
        if not levels:
            continue
        level_prices = [l['price'] for l in levels]

        df_1h, feats_1h = load_features('1h', month)
        df_1m, feats_1m = load_features('1m', month)
        if df_1h is None or df_1m is None:
            continue

        closes_1m = df_1m['close'].values
        ts_1h = df_1h['timestamp'].values
        ts_1m = df_1m['timestamp'].values

        # Find reversals in 1m at levels
        for lp in level_prices:
            for i in range(lookback + 1, len(closes_1m) - 2):
                touched = abs(closes_1m[i] - lp) < 20
                if not touched:
                    continue
                reversed_1m = (closes_1m[i+1] - closes_1m[i]) * (closes_1m[i] - closes_1m[i-1]) < 0
                if not reversed_1m:
                    continue

                # Find corresponding 1h bar
                h_idx = np.searchsorted(ts_1h, ts_1m[i], side='right') - 1
                if h_idx < lookback or h_idx >= len(feats_1h) - 1:
                    continue

                # For each directional feature: did 1h change before 1m?
                for fi, fname in enumerate(FEATURE_NAMES[:7]):
                    # 1m feature change at reversal
                    m_change = feats_1m[i, fi] - feats_1m[i - lookback, fi]
                    # 1h feature change at same time
                    h_change = feats_1h[h_idx, fi] - feats_1h[max(0, h_idx - 2), fi]

                    if abs(m_change) < 1e-8 and abs(h_change) < 1e-8:
                        continue
                    elif abs(h_change) > abs(m_change) * 1.5:
                        lead_counts[fname]['1h_leads'] += 1
                    elif abs(m_change) > abs(h_change) * 1.5:
                        lead_counts[fname]['1m_leads'] += 1
                    else:
                        lead_counts[fname]['simultaneous'] += 1

        del df_1h, feats_1h, df_1m, feats_1m; gc.collect()

    print(f"\n  {'Feature':<16} {'1h Leads':>10} {'1m Leads':>10} {'Simult':>10} {'Leader':>10}")
    print(f"  {'-'*55}")
    results = []
    for fname in FEATURE_NAMES[:7]:
        c = lead_counts[fname]
        total = c['1h_leads'] + c['1m_leads'] + c['simultaneous']
        if total > 0:
            leader = '1H' if c['1h_leads'] > c['1m_leads'] * 1.2 else ('1M' if c['1m_leads'] > c['1h_leads'] * 1.2 else 'TIE')
            print(f"  {fname:<16} {c['1h_leads']:>10} {c['1m_leads']:>10} {c['simultaneous']:>10} {leader:>10}")
            results.append({**c, 'feature': fname, 'leader': leader, 'total': total})

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(OUT_DIR, 'e5_lead_lag.csv'), index=False)
    print(f"\n  Saved: {OUT_DIR}/e5_lead_lag.csv")


# ============================================================
# E6: Daily Range Within Zones
# ============================================================
def run_e6(months):
    """Daily range within each level zone."""
    ensure_dir()
    print(f"\n{'='*60}\nE6: Daily Range Within Zones\n{'='*60}")

    all_rows = []
    for month in tqdm(months, desc="E6"):
        levels = get_levels_for_month(month)
        if not levels or len(levels) < 2:
            continue
        prices_lev = sorted([l['price'] for l in levels])

        df, _ = load_features('1h', month)
        if df is None:
            continue

        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        df['hour'] = pd.to_datetime(df['timestamp'], unit='s').dt.hour

        for zi in range(len(prices_lev) - 1):
            lo, hi = prices_lev[zi], prices_lev[zi + 1]
            zone_range = (hi - lo) / TICK

            for date, grp in df.groupby('date'):
                in_zone = grp[(grp['close'] >= lo) & (grp['close'] <= hi)]
                if len(in_zone) < 3:
                    continue
                daily_range = (in_zone['high'].max() - in_zone['low'].min()) / TICK
                pct_of_zone = daily_range / zone_range * 100 if zone_range > 0 else 0

                # Peak hour
                hourly = in_zone.groupby('hour').apply(
                    lambda h: (h['high'].max() - h['low'].min()) / TICK)
                peak_hour = hourly.idxmax() if len(hourly) > 0 else 0

                all_rows.append({
                    'month': month, 'date': str(date),
                    'zone': f'{lo:.0f}-{hi:.0f}',
                    'zone_range': zone_range,
                    'daily_range': daily_range,
                    'pct_of_zone': pct_of_zone,
                    'peak_hour': peak_hour,
                    'bars_in_zone': len(in_zone),
                })

        del df; gc.collect()

    df_r = pd.DataFrame(all_rows)
    df_r.to_csv(os.path.join(OUT_DIR, 'e6_daily_ranges.csv'), index=False)

    if len(df_r) > 0:
        print(f"\n  Avg daily range: {df_r['daily_range'].mean():.0f} ticks")
        print(f"  Avg % of zone:   {df_r['pct_of_zone'].mean():.1f}%")
        print(f"  Peak hour dist:  {df_r['peak_hour'].value_counts().head(5).to_dict()}")

    print(f"  Saved: {OUT_DIR}/e6_daily_ranges.csv ({len(df_r)} rows)")


# ============================================================
# E7: Feature-Level Interaction Matrix
# ============================================================
def run_e7(months):
    """Correlation between each feature and distance to nearest level."""
    ensure_dir()
    print(f"\n{'='*60}\nE7: Feature-Level Interaction Matrix\n{'='*60}")

    results = []
    for month in tqdm(months, desc="E7"):
        levels = get_levels_for_month(month)
        if not levels:
            continue
        level_prices = [l['price'] for l in levels]

        df, feats = load_features('1h', month)
        if df is None:
            continue
        closes = df['close'].values

        # Distance to nearest level for each bar
        dist = np.array([min(abs(c - lp) for lp in level_prices) for c in closes])

        for fi, fname in enumerate(FEATURE_NAMES):
            col = feats[:, fi]
            if col.std() < 1e-8:
                continue
            r, p = sp_stats.spearmanr(dist, col)
            results.append({'month': month, 'feature': fname, 'corr': r, 'p_value': p})

        del df, feats; gc.collect()

    df_r = pd.DataFrame(results)
    df_r.to_csv(os.path.join(OUT_DIR, 'e7_feature_level_interaction.csv'), index=False)

    # Average correlation per feature
    print(f"\n  {'Feature':<16} {'Avg Corr':>10} {'Interpretation':>25}")
    print(f"  {'-'*50}")
    for fname in FEATURE_NAMES:
        sub = df_r[df_r['feature'] == fname]
        if len(sub) > 0:
            avg = sub['corr'].mean()
            interp = 'LEVEL-AWARE' if abs(avg) > 0.15 else 'OBLIVIOUS'
            print(f"  {fname:<16} {avg:>10.3f} {interp:>25}")

    print(f"\n  Saved: {OUT_DIR}/e7_feature_level_interaction.csv")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='all',
                        choices=['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'all'])
    parser.add_argument('--months', default=None, help='Comma-separated YYYY-MM or all')
    args = parser.parse_args()

    if args.months:
        months = args.months.split(',')
    else:
        # All months with level data
        level_files = sorted(glob.glob('DATA/levels/levels_*.json'))
        months = sorted(set(
            json.load(open(f))['date'][:7]
            for f in level_files
        ))

    print(f"EDA across {len(months)} months: {months[0]} to {months[-1]}")

    modules = {
        'E1': run_e1, 'E2': run_e2, 'E3': run_e3,
        'E4': run_e4, 'E5': run_e5, 'E6': run_e6, 'E7': run_e7,
    }

    if args.module == 'all':
        for name, func in modules.items():
            func(months)
    else:
        modules[args.module](months)

    print(f"\n{'='*60}")
    print(f"EDA COMPLETE — results in {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
