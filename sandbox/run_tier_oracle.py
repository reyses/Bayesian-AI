"""
Tier Oracle Pipeline — physics forward pass + regret per tier + oracle ranking.

One script: runs IS forward pass with all tiers at max-fill, computes regret
on each tier's trades, ranks tiers by oracle potential, dumps EDA.

Run from project root:
    python sandbox/run_tier_oracle.py
    python sandbox/run_tier_oracle.py --days 50    # quick test
    python sandbox/run_tier_oracle.py --target oos
"""
import os
import sys
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from tqdm import tqdm

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from training.sfe_ticker import FeatureTicker
from core.features_79d import N_CORE, N_HELPER, N_TFS

FEATURES_DIR = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUT_DIR = 'sandbox/output'
TICK = 0.25
TV = 0.50
LOOKAHEAD_1M = 60  # 1 hour lookahead for regret

# ── Feature indices ───────────────────────────────────────────────────
_1M = N_CORE           # 12
_5M = 2 * N_CORE       # 24
_1H = 4 * N_CORE       # 48
_Z, _DMI, _VR, _VEL, _ACCEL, _VOL_REL, _BAR_RANGE, _HURST = range(8)

# Helper offsets
_H_START = N_CORE * N_TFS
_5M_WICK = _H_START + 2 * N_HELPER + 2
_15M_WICK = _H_START + 3 * N_HELPER + 2
_1M_WICK = _H_START + 1 * N_HELPER + 2

# ── Tier thresholds ───────────────────────────────────────────────────
ROCHE = 2.0
VR_ENTRY = 1.0
WICK_5M_MIN = 0.3
WICK_15M_MIN = 0.25
H1_Z_MIN = 0.5
H1_AGAINST_Z_MIN = 1.5
VELOCITY_THRESHOLD = 50.0
FREIGHT_TRAIN_THRESHOLD = 100.0
REGIME_VR_MAX = 0.35
REGIME_HURST_MAX = 0.45
MTF_5M_VEL_MIN = 30.0
MTF_1M_VEL_ALIVE = 10.0
EXHAUST_BAR_RANGE_MIN = 80.0
EXHAUST_ACCEL_MIN = 20.0
ABSORB_VOL_MIN = 1.5
ABSORB_RANGE_MAX = 40.0
ABSORB_WICK_MIN = 0.3


def check_all_tiers(feat, z):
    """Check which tiers this bar qualifies for. Returns list of (tier, direction)."""
    results = []
    abs_vel = abs(feat[_1M + _VEL])
    direction = 'short' if z > 0 else 'long'

    wick_5m = feat[_5M_WICK] if len(feat) > _5M_WICK else 0
    wick_15m = feat[_15M_WICK] if len(feat) > _15M_WICK else 0
    has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN

    h1_z = feat[_1H + _Z] if len(feat) > _1H + _Z else 0
    h1_vel = feat[_1H + _VEL] if len(feat) > _1H + _VEL else 0
    h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                  (direction == 'short' and h1_z > H1_Z_MIN))

    vr = feat[_1M + _VR]
    hurst = feat[_1M + _HURST]
    v5 = abs(feat[_5M + _VEL]) if len(feat) > _5M + _VEL else 0
    v5_accel = feat[_5M + _ACCEL] if len(feat) > _5M + _ACCEL else 0
    v1 = abs(feat[_1M + _VEL])
    bar_range = feat[_1M + _BAR_RANGE]
    accel_1m = feat[_1M + _ACCEL]
    vol_rel = feat[_1M + _VOL_REL]
    wick_1m = feat[_1M_WICK] if len(feat) > _1M_WICK else 0

    # NMP tiers (require z extreme + vr < 1)
    is_nmp = abs(z) > ROCHE and vr < VR_ENTRY
    if is_nmp:
        if has_wick and h1_aligned:
            results.append(('CASCADE', direction))
        if has_wick and not h1_aligned:
            results.append(('KILL_SHOT', direction))
        if abs_vel >= FREIGHT_TRAIN_THRESHOLD:
            ft_dir = 'long' if feat[_1M + _VEL] > 0 else 'short'
            results.append(('FREIGHT_TRAIN', ft_dir))
        h1_against = ((direction == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                       (direction == 'short' and h1_z < -H1_AGAINST_Z_MIN))
        if h1_against:
            results.append(('FADE_AGAINST', direction))
        h1_vel_against = ((direction == 'long' and h1_vel < -H1_AGAINST_Z_MIN) or
                          (direction == 'short' and h1_vel > H1_AGAINST_Z_MIN))
        if h1_vel_against and not h1_against:
            ride_dir = 'long' if h1_vel > 0 else 'short'
            results.append(('RIDE_AGAINST', ride_dir))
        if abs_vel >= VELOCITY_THRESHOLD:
            results.append(('FADE_MOMENTUM', direction))
        results.append(('FADE_CALM', direction))

    # ISO tiers (independent conditions)
    if vr < REGIME_VR_MAX and hurst < REGIME_HURST_MAX:
        results.append(('REGIME_FLIP', direction))
    if v5_accel < 0 and v5 > MTF_5M_VEL_MIN and v1 > MTF_1M_VEL_ALIVE:
        mtf_dir = 'short' if feat[_5M + _VEL] > 0 else 'long'
        results.append(('MTF_EXHAUSTION', mtf_dir))
    if (bar_range > EXHAUST_BAR_RANGE_MIN and
            abs(accel_1m) > EXHAUST_ACCEL_MIN and
            accel_1m * feat[_1M + _VEL] < 0):
        ex_dir = 'short' if feat[_1M + _VEL] > 0 else 'long'
        results.append(('EXHAUSTION_BAR', ex_dir))
    if vol_rel > ABSORB_VOL_MIN and bar_range < ABSORB_RANGE_MAX and wick_1m > ABSORB_WICK_MIN:
        results.append(('ABSORPTION', direction))

    return results


# ── STEP 1: Max-Fill Forward Pass ─────────────────────────────────────

def run_maxfill(target='is', max_days=None):
    """Each tier independently captures trades from all eligible bars."""
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    if target == 'is':
        feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        feat_files = [f for f in feat_files if '2026_' in os.path.basename(f)]
    if max_days:
        feat_files = feat_files[:max_days]

    tier_entries = defaultdict(list)  # {tier: [(day, ts, direction, entry_price, entry_idx, feat)]}

    print(f'STEP 1: Max-Fill Forward Pass — {len(feat_files)} days')
    for fpath in tqdm(feat_files, desc='  Scanning', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        ft = FeatureTicker(fpath, price_file=os.path.join(ATLAS_1M, f'{day_name}.parquet'))
        bars = list(ft)

        for i, state in enumerate(bars):
            feat = state['features_79d']
            price = state['price']
            ts = state['timestamp']
            if price < 100:
                continue
            is_1m = (int(ts) % 60) < 5
            if not is_1m:
                continue

            z = feat[_1M + _Z]
            eligible = check_all_tiers(feat, z)
            for tier, direction in eligible:
                tier_entries[tier].append({
                    'day': day_name, 'timestamp': ts, 'dir': direction,
                    'entry_price': price, 'entry_79d': feat.tolist() if hasattr(feat, 'tolist') else list(feat),
                })

    total = sum(len(v) for v in tier_entries.values())
    print(f'  {total:,} entries across {len(tier_entries)} tiers')
    for tier in sorted(tier_entries, key=lambda t: -len(tier_entries[t])):
        print(f'    {tier:<20} {len(tier_entries[tier]):>7}')
    return tier_entries


# ── STEP 2: Regret Per Tier ───────────────────────────────────────────

def run_regret(tier_entries):
    """Compute FULL regret for every entry in every tier.

    Uses training.regret.compute_regret — same oracle as the pipeline:
    SAME early/exit/extended, COUNTER early/exit/extended, entry lookback.
    """
    from training.regret import compute_regret

    print(f'\nSTEP 2: Full Regret Analysis (5 curves + entry lookback)')

    # Cache day prices
    day_cache = {}

    all_regret = []

    for tier in sorted(tier_entries.keys()):
        entries = tier_entries[tier]
        tier_results = []

        # Group by day for efficiency
        by_day = defaultdict(list)
        for e in entries:
            by_day[e['day']].append(e)

        for day_name in tqdm(sorted(by_day.keys()), desc=f'  {tier:<20}', unit='day', leave=False):
            if day_name not in day_cache:
                path = os.path.join(ATLAS_1M, f'{day_name}.parquet')
                if os.path.exists(path):
                    day_cache[day_name] = pd.read_parquet(path).sort_values('timestamp')
                else:
                    day_cache[day_name] = None

            day_df = day_cache[day_name]
            if day_df is None:
                continue

            closes = day_df['close'].values
            ts_arr = day_df['timestamp'].values

            for e in by_day[day_name]:
                entry_idx = int(np.searchsorted(ts_arr, e['timestamp'], side='right')) - 1
                if entry_idx < 0 or entry_idx >= len(closes):
                    continue

                # Build trade dict for compute_regret
                trade = {
                    'dir': e['dir'],
                    'entry_price': e['entry_price'],
                    'pnl': 0,  # no actual exit yet — regret computes all options
                    'held': 20,  # default hold for "actual exit" reference
                    'peak': 0,
                }

                r = compute_regret(trade, closes, entry_idx)

                tier_results.append({
                    'tier': tier, 'day': day_name, 'timestamp': e['timestamp'],
                    'dir': e['dir'], 'entry_price': e['entry_price'],
                    # Full regret fields
                    'best_action': r['best_action'],
                    'best_pnl': r['best_pnl'],
                    'regret': r['regret'],
                    'same_early_best': r['same_early_best'],
                    'same_early_bar': r['same_early_bar'],
                    'same_at_exit': r['same_at_exit'],
                    'same_ext_best': r['same_ext_best'],
                    'same_ext_bar': r['same_ext_bar'],
                    'same_best': r['same_best'],
                    'same_best_bar': r['same_best_bar'],
                    'counter_early_best': r['counter_early_best'],
                    'counter_early_bar': r['counter_early_bar'],
                    'counter_at_exit': r['counter_at_exit'],
                    'counter_ext_best': r['counter_ext_best'],
                    'counter_ext_bar': r['counter_ext_bar'],
                    'counter_best': r['counter_best'],
                    'counter_best_bar': r['counter_best_bar'],
                    'early_entry_gain': r.get('early_entry_gain', 0),
                })

        all_regret.extend(tier_results)
        if tier_results:
            avg = np.mean([r['best_pnl'] for r in tier_results])
            same_pct = sum(1 for r in tier_results if 'same' in r['best_action']) / len(tier_results) * 100
            tqdm.write(f'  {tier:<20} {len(tier_results):>7} trades  oracle=${avg:.1f}/tr  same={same_pct:.0f}%')

    return pd.DataFrame(all_regret)


# ── STEP 3: Oracle Ranking + EDA ──────────────────────────────────────

def oracle_eda(regret_df, n_days=277):
    """Rank tiers by oracle potential. Full EDA."""
    print(f'\n{"="*80}')
    print(f'ORACLE TIER RANKING — perfect exits, per tier')
    print(f'{"="*80}')
    print(f'{"Tier":<20} {"Entries":>8} {"Oracle$/tr":>11} {"Oracle$/day":>12} '
          f'{"Same%":>6} {"OracleBar":>10} {"Oracle Total":>13}')
    print(f'{"-"*80}')

    rows = []
    for tier, grp in regret_df.groupby('tier'):
        n = len(grp)
        oracle_avg = grp['best_pnl'].mean()
        oracle_total = grp['best_pnl'].sum()
        oracle_day = oracle_total / n_days
        same_pct = grp['best_action'].str.contains('same').mean() * 100
        avg_bar = grp['same_best_bar'].mean()
        rows.append((tier, n, oracle_avg, oracle_day, same_pct, avg_bar, oracle_total))

    rows.sort(key=lambda r: -r[3])  # sort by $/day
    for tier, n, avg, day, same, bar, total in rows:
        print(f'{tier:<20} {n:>8} {avg:>11.1f} {day:>12.0f} '
              f'{same:>5.0f}% {bar:>10.0f} {total:>13,.0f}')

    grand_total = regret_df['best_pnl'].sum()
    print(f'{"-"*80}')
    print(f'{"TOTAL":<20} {len(regret_df):>8} {"":>11} {grand_total/n_days:>12.0f} '
          f'{"":>6} {"":>10} {grand_total:>13,.0f}')

    print(f'\nOracle ceiling: ${grand_total/n_days:,.0f}/day')

    # Best action breakdown per tier
    print(f'\nBEST ACTION BREAKDOWN:')
    for tier, grp in regret_df.groupby('tier'):
        actions = grp['best_action'].value_counts()
        parts = [f'{act}={cnt}' for act, cnt in actions.items()]
        print(f'  {tier:<20} {" | ".join(parts)}')

    # Direction analysis per tier
    print(f'\nDIRECTION ORACLE (should physics flip?):')
    for tier, grp in regret_df.groupby('tier'):
        same = grp[grp['best_action'].str.contains('same')]
        counter = grp[grp['best_action'].str.contains('counter')]
        s_avg = same['best_pnl'].mean() if len(same) > 0 else 0
        c_avg = counter['best_pnl'].mean() if len(counter) > 0 else 0
        print(f'  {tier:<20} SAME={len(same):>6} (${s_avg:.0f}/tr)  '
              f'COUNTER={len(counter):>6} (${c_avg:.0f}/tr)')

    # Duration distribution per tier (using same_best_bar / counter_best_bar)
    print(f'\nDURATION ORACLE (optimal hold bars):')
    for tier, grp in regret_df.groupby('tier'):
        bars = grp['same_best_bar'].values  # same direction optimal bar
        short = (bars < 10).sum()
        medium = ((bars >= 10) & (bars < 40)).sum()
        long_count = (bars >= 40).sum()
        print(f'  {tier:<20} SHORT(<10)={short:>5}  MED(10-40)={medium:>5}  LONG(40+)={long_count:>5}  '
              f'avg={bars.mean():.0f} bars')

    # Early entry potential
    print(f'\nEARLY ENTRY GAIN (entering earlier would have helped by):')
    if 'early_entry_gain' in regret_df.columns:
        for tier, grp in regret_df.groupby('tier'):
            gain = grp['early_entry_gain']
            pct_helps = (gain > 5).sum() / len(gain) * 100
            print(f'  {tier:<20} avg=${gain.mean():.0f}  helps(>$5)={pct_helps:.0f}%')

    # Same vs Counter PnL comparison
    print(f'\nSAME vs COUNTER PEAK PNL:')
    for tier, grp in regret_df.groupby('tier'):
        s = grp['same_best'].mean()
        c = grp['counter_best'].mean()
        winner = 'SAME' if s > c else 'COUNTER'
        print(f'  {tier:<20} same_peak=${s:.0f}  counter_peak=${c:.0f}  -> {winner}')

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='is', choices=['is', 'oos', 'all'])
    parser.add_argument('--days', type=int, default=None)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Step 1: Max-fill
    tier_entries = run_maxfill(args.target, args.days)

    # Save entries
    entries_path = os.path.join(OUT_DIR, 'tier_entries.pkl')
    with open(entries_path, 'wb') as f:
        pickle.dump(dict(tier_entries), f)

    # Step 2: Regret
    regret_df = run_regret(tier_entries)
    regret_path = os.path.join(OUT_DIR, 'tier_regret.csv')
    regret_df.to_csv(regret_path, index=False)
    print(f'\nSaved: {regret_path}')

    # Step 3: Oracle EDA
    n_days = len(set(regret_df['day'])) or 277
    oracle_eda(regret_df, n_days)


if __name__ == '__main__':
    main()
