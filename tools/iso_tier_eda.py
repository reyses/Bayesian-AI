"""
Max-Fill Tier Forward Pass + EDA — each tier gets independent access to ALL data.

No waterfall competition. Each tier evaluates every bar independently.
CASCADE and FADE_MOMENTUM can both trigger on the same bar.
Result: maximum possible trades per tier for CNN training.

Usage:
    python tools/iso_tier_eda.py                          # all tiers, max fill
    python tools/iso_tier_eda.py --tier FREIGHT_TRAIN     # single tier
    python tools/iso_tier_eda.py --target oos             # OOS only
"""
import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from core.features_79d import N_CORE, N_HELPER, N_TFS

FEATURES_DIR_5S = 'DATA/FEATURES_79D_5s'
ATLAS_1M = 'DATA/ATLAS/1m'

TICK = 0.25
TV = 0.50

# Feature indices (1m offset = 1 * N_CORE = 12)
_1M = N_CORE  # 12
_Z = 0
_VR = 2
_VEL = 3
_ACCEL = 4
_HURST = 7
_DMI = 1

# 5m indices
_5M = 2 * N_CORE  # 24
_5M_WICK = N_CORE * N_TFS + 2 * N_HELPER + 2  # helper: wick_ratio for 5m

# 15m indices
_15M_WICK = N_CORE * N_TFS + 3 * N_HELPER + 2

# 1h indices
_1H = 4 * N_CORE  # 48
_1H_Z = _1H + _Z
_1H_VEL = _1H + _VEL

# Additional feature indices for ISO tiers
_1M_BAR_RANGE = _1M + 6        # bar_range
_1M_ACCEL = _1M + 4            # acceleration
_1M_VOL_REL = _1M + 5          # vol_rel
_1M_WICK = N_CORE * N_TFS + 1 * N_HELPER + 2  # helper: wick_ratio for 1m
_5M_VEL = _5M + _VEL           # 5m velocity
_5M_ACCEL = _5M + 4            # 5m acceleration

# Tier thresholds (from nightmare_blended.py)
# NMP entry
ROCHE = 2.0
VR_ENTRY = 1.0

# NMP tier thresholds
WICK_5M_MIN = 0.3
WICK_15M_MIN = 0.25
H1_Z_MIN = 0.5
H1_AGAINST_Z_MIN = 1.5
VELOCITY_THRESHOLD = 50.0
FREIGHT_TRAIN_THRESHOLD = 100.0

# ISO tier thresholds (disabled in blended, active here for max-fill)
REGIME_VR_MAX = 0.35
REGIME_HURST_MAX = 0.45
MTF_5M_VEL_MIN = 30.0
MTF_1M_VEL_ALIVE = 10.0
EXHAUST_BAR_RANGE_MIN = 80.0
EXHAUST_ACCEL_MIN = 20.0
ABSORB_VOL_MIN = 1.5
ABSORB_RANGE_MAX = 40.0
ABSORB_WICK_MIN = 0.3


def check_tier(feat, z):
    """Check which tiers this bar qualifies for. Returns list of (tier, direction)."""
    results = []
    abs_vel = abs(feat[_1M + _VEL])
    direction = 'short' if z > 0 else 'long'

    # Wick rejection
    wick_5m = feat[_5M_WICK] if len(feat) > _5M_WICK else 0
    wick_15m = feat[_15M_WICK] if len(feat) > _15M_WICK else 0
    has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN

    h1_z = feat[_1H_Z] if len(feat) > _1H_Z else 0
    h1_vel = feat[_1H_VEL] if len(feat) > _1H_VEL else 0
    h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                  (direction == 'short' and h1_z > H1_Z_MIN))

    # CASCADE: wick + 1h aligned
    if has_wick and h1_aligned:
        results.append(('CASCADE', direction))

    # KILL_SHOT: wick, no 1h
    if has_wick and not h1_aligned:
        results.append(('KILL_SHOT', direction))

    # FREIGHT_TRAIN: extreme velocity
    if abs_vel >= FREIGHT_TRAIN_THRESHOLD:
        ft_dir = 'long' if feat[_1M + _VEL] > 0 else 'short'
        results.append(('FREIGHT_TRAIN', ft_dir))

    # FADE_AGAINST: 1h z extreme against fade direction
    h1_against = ((direction == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                  (direction == 'short' and h1_z < -H1_AGAINST_Z_MIN))
    if h1_against:
        results.append(('FADE_AGAINST', direction))

    # RIDE_AGAINST: 1h velocity opposes
    h1_vel_against = ((direction == 'long' and h1_vel < -H1_AGAINST_Z_MIN) or
                      (direction == 'short' and h1_vel > H1_AGAINST_Z_MIN))
    if h1_vel_against and not h1_against:
        ride_dir = 'long' if h1_vel > 0 else 'short'
        results.append(('RIDE_AGAINST', ride_dir))

    # FADE_MOMENTUM: velocity > threshold
    if abs_vel >= VELOCITY_THRESHOLD:
        results.append(('FADE_MOMENTUM', direction))

    # FADE_CALM: default (always qualifies for NMP)
    results.append(('FADE_CALM', direction))

    # === ISO TIERS (non-NMP: don't require z extreme) ===
    # These check independently — don't need abs(z) > ROCHE
    vr = feat[_1M + _VR]
    hurst = feat[_1M + _HURST]
    v5 = abs(feat[_5M_VEL]) if len(feat) > _5M_VEL else 0
    v5_accel = feat[_5M_ACCEL] if len(feat) > _5M_ACCEL else 0
    v1 = abs(feat[_1M + _VEL])
    bar_range = feat[_1M_BAR_RANGE] if len(feat) > _1M_BAR_RANGE else 0
    accel_1m = abs(feat[_1M_ACCEL]) if len(feat) > _1M_ACCEL else 0
    vol_rel = feat[_1M_VOL_REL] if len(feat) > _1M_VOL_REL else 0
    wick_1m = feat[_1M_WICK] if len(feat) > _1M_WICK else 0

    # REGIME_FLIP: vr low + hurst low = just shifted to mean-reverting
    if vr < REGIME_VR_MAX and hurst < REGIME_HURST_MAX:
        results.append(('REGIME_FLIP', direction))

    # MTF_EXHAUSTION: 5m decelerating + 1m still alive
    if v5_accel < 0 and v5 > MTF_5M_VEL_MIN and v1 > MTF_1M_VEL_ALIVE:
        mtf_dir = 'short' if feat[_5M_VEL] > 0 else 'long'
        results.append(('MTF_EXHAUSTION', mtf_dir))

    # EXHAUSTION_BAR: bar_range climax + velocity decelerating
    if (bar_range > EXHAUST_BAR_RANGE_MIN and
            accel_1m > EXHAUST_ACCEL_MIN and
            feat[_1M_ACCEL] * feat[_1M + _VEL] < 0):  # decel
        ex_dir = 'short' if feat[_1M + _VEL] > 0 else 'long'
        results.append(('EXHAUSTION_BAR', ex_dir))

    # ABSORPTION: high volume + low range + wicks
    if vol_rel > ABSORB_VOL_MIN and bar_range < ABSORB_RANGE_MAX and wick_1m > ABSORB_WICK_MIN:
        results.append(('ABSORPTION', direction))

    return results


def run_max_fill(tier_filter=None, target='is', max_days=None):
    """Run each tier independently on all data. No waterfall."""
    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR_5S, '*.parquet')))
    if target == 'is':
        feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    elif target == 'oos':
        feat_files = [f for f in feat_files if '2026_' in os.path.basename(f)]
    if max_days:
        feat_files = feat_files[:max_days]

    if not feat_files:
        print(f'No feature files for {target}')
        return {}

    # Collect trades per tier
    tier_trades = {}  # {tier: [trade_dicts]}

    print(f'MAX-FILL FORWARD PASS — {len(feat_files)} days')

    for fpath in tqdm(feat_files, desc='Days', unit='day'):
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')

        ft = FeatureTicker(fpath, price_file=price_file)
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
            vr = feat[_1M + _VR]

            # NMP tiers need z extreme + vr < 1
            # ISO tiers have their own conditions (checked inside check_tier)
            is_nmp_eligible = abs(z) > ROCHE and vr < VR_ENTRY

            # Check all tiers this bar qualifies for
            eligible = check_tier(feat, z)

            # Filter: NMP tiers only if NMP eligible, ISO tiers always checked
            nmp_tiers = {'CASCADE', 'KILL_SHOT', 'FREIGHT_TRAIN', 'FADE_AGAINST',
                         'RIDE_AGAINST', 'FADE_MOMENTUM', 'FADE_CALM'}
            if not is_nmp_eligible:
                eligible = [(t, d) for t, d in eligible if t not in nmp_tiers]

            if not eligible:
                continue

            for tier, direction in eligible:
                if tier_filter and tier not in tier_filter:
                    continue

                # Simulate trade with REAL per-tier exit physics
                entry_price = price
                entry_abs_z = abs(z)
                entry_velocity = abs(feat[_1M + _VEL])
                entry_vol_rel = feat[_1M_VOL_REL] if len(feat) > _1M_VOL_REL else 1.0
                peak_pnl = 0
                held = 0
                exit_pnl = 0
                exit_reason = 'max_bars'
                p_center_bars = 0

                for j in range(i + 1, min(i + 720, len(bars))):  # max 60 min at 5s
                    bar_j = bars[j]
                    p = bar_j['price']
                    fj = bar_j['features_79d']
                    if p < 100:
                        continue

                    if direction == 'long':
                        pnl = (p - entry_price) / TICK * TV
                    else:
                        pnl = (entry_price - p) / TICK * TV

                    peak_pnl = max(peak_pnl, pnl)
                    held += 1
                    bars_1m = held // 12

                    z_j = fj[_1M + _Z]
                    abs_z_j = abs(z_j)
                    vr_j = fj[_1M + _VR]
                    vel_j = fj[_1M + _VEL]
                    accel_j = fj[_1M + _ACCEL] if len(fj) > _1M + _ACCEL else 0
                    vol_j = fj[_1M_VOL_REL] if len(fj) > _1M_VOL_REL else 1.0
                    v5_j = fj[_5M_VEL] if len(fj) > _5M_VEL else 0
                    v5a_j = fj[_5M_ACCEL] if len(fj) > _5M_ACCEL else 0

                    exited = False

                    # Hard stop
                    if pnl <= -150:
                        exit_reason = 'hard_stop'; exited = True

                    # Giveback stop
                    elif peak_pnl >= 30 and pnl < peak_pnl * 0.5:
                        exit_reason = 'giveback'; exited = True

                    # ── Per-tier exits ──
                    elif tier in ('CASCADE', 'KILL_SHOT'):
                        # p_center confirmation + z conviction at bar 24
                        p_ctr = fj[_1M + 9] if len(fj) > _1M + 9 else 0  # p_at_center
                        if p_ctr > 0.4:
                            p_center_bars += 1
                        else:
                            p_center_bars = 0
                        req = 3 if tier == 'CASCADE' else 2
                        if p_center_bars >= req:
                            exit_reason = f'{tier.lower()}_center'; exited = True
                        elif bars_1m >= 2 and bars_1m < 3:
                            z_shrink = (entry_abs_z - abs_z_j) / max(entry_abs_z, 0.01)
                            if z_shrink < 0.20:
                                exit_reason = f'{tier.lower()}_no_conviction'; exited = True

                    elif tier == 'FREIGHT_TRAIN':
                        abs_vel_j = abs(vel_j)
                        vel_ratio = abs_vel_j / max(entry_velocity, 1.0)
                        if vel_ratio < 0.50 and vel_j * accel_j < 0:
                            exit_reason = 'freight_decel'; exited = True

                    elif tier == 'REGIME_FLIP':
                        if bars_1m >= 1 and bars_1m < 2:
                            if abs_z_j > entry_abs_z:
                                exit_reason = 'regime_no_conviction'; exited = True
                        if vr_j > 0.30:
                            exit_reason = 'regime_vr_rising'; exited = True
                        elif abs_z_j < 0.3:
                            exit_reason = 'regime_mean'; exited = True

                    elif tier == 'ABSORPTION':
                        if bars_1m >= 1 and bars_1m < 2:
                            z_shrink = (entry_abs_z - abs_z_j) / max(entry_abs_z, 0.01)
                            if z_shrink < 0.10:
                                exit_reason = 'absorb_no_conviction'; exited = True
                        if bars_1m >= 2 and vol_j > 1.5:
                            exit_reason = 'absorb_vol_persistent'; exited = True
                        elif vr_j > 0.65:
                            exit_reason = 'absorb_vr_rising'; exited = True
                        elif abs_z_j < 0.3:
                            exit_reason = 'absorb_mean'; exited = True

                    elif tier == 'EXHAUSTION_BAR':
                        if bars_1m >= 1 and bars_1m < 2:
                            z_shrink = (entry_abs_z - abs_z_j) / max(entry_abs_z, 0.01)
                            if z_shrink < 0.20:
                                exit_reason = 'exhaust_no_conviction'; exited = True
                        elif abs_z_j < 0.3:
                            exit_reason = 'exhaust_mean'; exited = True

                    elif tier == 'MTF_EXHAUSTION':
                        if bars_1m >= 1 and bars_1m < 2:
                            z_shrink = (entry_abs_z - abs_z_j) / max(entry_abs_z, 0.01)
                            if z_shrink < 0.10:
                                exit_reason = 'mtf_no_conviction'; exited = True
                        if v5a_j > 0 and abs(v5_j) > 30:
                            exit_reason = 'mtf_5m_reaccel'; exited = True
                        elif abs_z_j < 0.3:
                            exit_reason = 'mtf_mean'; exited = True

                    else:
                        # FADE_AGAINST, FADE_CALM, FADE_MOMENTUM, RIDE_AGAINST
                        # Standard fade/ride physics
                        if abs_z_j < 0.3:
                            exit_reason = 'mean_reached'; exited = True

                    if exited:
                        exit_pnl = pnl
                        break
                else:
                    if held > 0:
                        exit_pnl = pnl

                if held == 0:
                    continue

                trade = {
                    'day': day_name,
                    'entry_tier': tier,
                    'dir': direction,
                    'entry_price': entry_price,
                    'pnl': exit_pnl,
                    'peak': peak_pnl,
                    'held': held,
                    'exit_reason': exit_reason,
                    'timestamp': ts,
                    'entry_79d': feat.tolist() if hasattr(feat, 'tolist') else list(feat),
                }

                if tier not in tier_trades:
                    tier_trades[tier] = []
                tier_trades[tier].append(trade)

    return tier_trades


def print_eda(tier_trades):
    """Print EDA per tier."""
    print(f'\n{"="*70}')
    print(f'MAX-FILL TIER REPORT')
    print(f'{"="*70}')
    print(f'{"Tier":<20} {"Trades":>7} {"WR":>6} {"Avg$":>8} {"Med$":>8} {"Total$":>10} {"AvgHeld":>8}')
    print(f'{"-"*70}')

    all_trades = []
    for tier in sorted(tier_trades.keys()):
        trades = tier_trades[tier]
        all_trades.extend(trades)
        pnls = [t['pnl'] for t in trades]
        held = [t['held'] for t in trades]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f'{tier:<20} {len(trades):>7} {wr:>5.0f}% {np.mean(pnls):>8.1f} '
              f'{np.median(pnls):>8.1f} {sum(pnls):>10,.0f} {np.mean(held):>8.0f}')

    print(f'{"-"*70}')
    total = sum(len(v) for v in tier_trades.values())
    print(f'{"TOTAL":<20} {total:>7}')

    # Per-tier direction breakdown
    print(f'\nDIRECTION BREAKDOWN:')
    for tier in sorted(tier_trades.keys()):
        trades = tier_trades[tier]
        for d in ['long', 'short']:
            sub = [t for t in trades if t['dir'] == d]
            if sub:
                pnls = [t['pnl'] for t in sub]
                wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
                print(f'  {tier:<20} {d:<6} {len(sub):>5} trades  WR={wr:.0f}%  avg=${np.mean(pnls):.1f}')

    # Per-tier exit reason breakdown
    print(f'\nEXIT REASONS:')
    for tier in sorted(tier_trades.keys()):
        trades = tier_trades[tier]
        exits = Counter(t.get('exit_reason', '?') for t in trades)
        print(f'  {tier}:')
        for reason, count in exits.most_common(8):
            sub_pnls = [t['pnl'] for t in trades if t.get('exit_reason') == reason]
            avg = np.mean(sub_pnls)
            wr = sum(1 for p in sub_pnls if p > 0) / len(sub_pnls) * 100
            print(f'    {reason:<30} {count:>6}  WR={wr:.0f}%  avg=${avg:.1f}')

    # Save
    out_dir = 'training/output/trades'
    os.makedirs(out_dir, exist_ok=True)
    for tier, trades in tier_trades.items():
        pkl_path = os.path.join(out_dir, f'maxfill_{tier.lower()}.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(trades, f)

        csv_rows = [{k: v for k, v in t.items()
                     if not isinstance(v, (list, dict, np.ndarray))} for t in trades]
        csv_path = os.path.join(out_dir, f'maxfill_{tier.lower()}.csv')
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f'\nSaved per-tier pickles + CSVs to {out_dir}/')


def parse_args():
    p = argparse.ArgumentParser(description='Max-fill tier forward pass + EDA')
    p.add_argument('--tier', type=str, nargs='*', default=None,
                   help='Specific tiers (default: all)')
    p.add_argument('--target', type=str, default='is', choices=['is', 'oos', 'all'])
    p.add_argument('--days', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    tier_filter = set(args.tier) if args.tier else None
    tier_trades = run_max_fill(tier_filter, args.target, args.days)
    if tier_trades:
        print_eda(tier_trades)


if __name__ == '__main__':
    main()
