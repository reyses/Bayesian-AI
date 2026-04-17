"""Run each tier in isolation — no chains, no negative exits, no catch-all.

For each tier listed, runs Phase 1 physics with ONLY that tier able to
enter. Everything else is disabled. This gives the cleanest possible
read on each tier's individual edge.

Output: per-tier $/day, WR, trade count on the honest features.

Usage:
    python tools/run_tier_isolated.py                    # all tiers
    python tools/run_tier_isolated.py FADE_CALM KILL_SHOT  # specific tiers
"""
import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_blended import (
    BlendedEngine,
    _1M_OFFSET, _Z, _VR, _1M_VELOCITY_IDX, _1H_VELOCITY_IDX, _1H_Z_IDX,
    _5M_WICK_IDX, _15M_WICK_IDX, _5M_ACCEL_IDX, _5M_VELOCITY_IDX,
    _1M_HURST_IDX, _1M_VOL_REL_IDX, _1M_DMI_IDX, _5M_BAR_RANGE_IDX,
    H1_AGAINST_Z_MIN, H1_Z_MIN, WICK_5M_MIN, WICK_15M_MIN,
    FREIGHT_TRAIN_THRESHOLD, FREIGHT_TRAIN_VR_MAX,
    MTF_Z_MIN, MTF_VR_MIN, MTF_VOL_MIN, MTF_5M_VEL_MIN, MTF_1M_VEL_ALIVE,
    _N_CORE,
)
from core.ledger import Ledger
from core import sim_executor

FEATURES_DIR = 'DATA/ATLAS/FEATURES_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
ALL_TIERS = ['FADE_CALM', 'RIDE_AGAINST', 'KILL_SHOT', 'CASCADE',
             'FADE_AGAINST', 'MTF_BREAKOUT', 'MTF_EXHAUSTION', 'FREIGHT_TRAIN']


def tier_classifier(tier):
    """Return a function(feat, z) -> (direction, tier, cnn_flipped) that
    only fires for the given tier. Returns (None, None, False) otherwise."""
    def classify(feat, z):
        direction = 'short' if z > 0 else 'long'

        # Read all features (copied from _classify_full_tier)
        wick_5m = feat[_5M_WICK_IDX]
        wick_15m = feat[_15M_WICK_IDX]
        h1_z = feat[_1H_Z_IDX]
        velocity = feat[_1M_VELOCITY_IDX]
        h1_vel = feat[_1H_VELOCITY_IDX]
        abs_vel = abs(velocity)
        acceleration = feat[_1M_OFFSET + 4]
        vr = feat[_1M_OFFSET + _VR]
        v5_vel = feat[_5M_VELOCITY_IDX]
        v5_accel = feat[_5M_ACCEL_IDX]
        hurst = feat[_1M_HURST_IDX]
        vol_rel = feat[_1M_VOL_REL_IDX]
        dmi = feat[_1M_DMI_IDX]
        v1 = abs(velocity)
        has_wick = wick_5m > WICK_5M_MIN and wick_15m > WICK_15M_MIN
        h1_against_fade = ((direction == 'long' and h1_z > H1_AGAINST_Z_MIN) or
                           (direction == 'short' and h1_z < -H1_AGAINST_Z_MIN))
        h1_aligned = ((direction == 'long' and h1_z < -H1_Z_MIN) or
                      (direction == 'short' and h1_z > H1_Z_MIN))

        if tier == 'FREIGHT_TRAIN':
            if (abs_vel >= FREIGHT_TRAIN_THRESHOLD and
                    velocity * acceleration > 0 and vr < FREIGHT_TRAIN_VR_MAX):
                ft_dir = 'long' if velocity > 0 else 'short'
                return ft_dir, 'FREIGHT_TRAIN', True
            return None, None, False

        if tier == 'KILL_SHOT':
            if has_wick and not h1_aligned:
                return direction, 'KILL_SHOT', False
            return None, None, False

        if tier == 'CASCADE':
            if has_wick and h1_aligned:
                return direction, 'CASCADE', False
            return None, None, False

        if tier == 'RIDE_AGAINST':
            h1_vel_against = ((direction == 'long' and h1_vel < -3.0) or
                              (direction == 'short' and h1_vel > 3.0))
            if h1_vel_against and not h1_against_fade:
                ride_dir = 'long' if h1_vel > 0 else 'short'
                return ride_dir, 'RIDE_AGAINST', False
            return None, None, False

        if tier == 'FADE_AGAINST':
            if h1_against_fade and abs(v5_vel) < 10.0:
                return direction, 'FADE_AGAINST', False
            return None, None, False

        if tier == 'MTF_EXHAUSTION':
            if (v5_accel < 0 and abs(v5_vel) > MTF_5M_VEL_MIN and
                    v1 > MTF_1M_VEL_ALIVE and abs(z) > MTF_Z_MIN and
                    vr > MTF_VR_MIN and vol_rel > MTF_VOL_MIN):
                mtf_dir = 'long' if v5_vel > 0 else 'short'
                return mtf_dir, 'MTF_EXHAUSTION', True
            return None, None, False

        if tier == 'MTF_BREAKOUT':
            z_5m = abs(feat[2 * _N_CORE + _Z])
            z_15m = abs(feat[3 * _N_CORE + _Z])
            if z_5m > 1.3 and z_15m > 1.3:
                breakout_dir = 'long' if z > 0 else 'short'
                dmi_aligned = ((breakout_dir == 'long' and dmi > -5) or
                               (breakout_dir == 'short' and dmi < 5))
                if dmi_aligned:
                    return breakout_dir, 'MTF_BREAKOUT', True
            return None, None, False

        if tier == 'FADE_CALM':
            # Only fade_calm, but gated — skip if higher_tf_opposing
            higher_tf_opposing = False
            if direction == 'long' and v5_vel < -3 and h1_vel < -3:
                higher_tf_opposing = True
            if direction == 'short' and v5_vel > 3 and h1_vel > 3:
                higher_tf_opposing = True
            if higher_tf_opposing:
                return None, None, False
            # Skip if ANY other tier would have fired (they take priority)
            if has_wick:
                return None, None, False
            h1_vel_against = ((direction == 'long' and h1_vel < -3.0) or
                              (direction == 'short' and h1_vel > 3.0))
            if h1_vel_against and not h1_against_fade:
                return None, None, False
            if h1_against_fade and abs(v5_vel) < 10.0:
                return None, None, False
            return direction, 'FADE_CALM', False

        return None, None, False

    return classify


def run_tier(tier, feat_files):
    """Run one tier in isolation across all days. Returns (trades, total_pnl)."""
    engine = BlendedEngine(use_cnn=False)
    # Patch the classifier
    engine._classify_full_tier = tier_classifier(tier)

    all_trades = []
    for fpath in feat_files:
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        ledger = Ledger()
        ft = FeatureTicker(fpath, price_file=price_file)
        trades = sim_executor.run(ledger, engine, ft, eod_close=True)
        for t in trades:
            t['day'] = day_name
        all_trades.extend(trades)

    return all_trades


def main():
    tiers = sys.argv[1:] if len(sys.argv) > 1 else ALL_TIERS
    tiers = [t for t in tiers if t in ALL_TIERS]

    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    print(f'Feature files: {len(feat_files)}')
    print(f'Tiers to test: {tiers}')
    print()

    # Save per-tier results
    os.makedirs('training/output/isolated', exist_ok=True)
    all_results = {}

    print(f'{"Tier":<18} {"Trades":>8} {"Primary":>8} {"Chain":>8} '
          f'{"WR":>5} {"Total $":>10} {"$/day":>8}')
    print('-' * 75)

    for tier in tiers:
        trades = run_tier(tier, feat_files)
        primaries = [t for t in trades if not t.get('is_chain', False)]
        chains = [t for t in trades if t.get('is_chain', False)]
        total_pnl = sum(t['pnl'] for t in trades)
        wr = (sum(1 for t in primaries if t['pnl'] > 0) /
              max(len(primaries), 1) * 100)
        days = len(set(t.get('day', '') for t in trades))
        per_day = total_pnl / max(days, 1)

        print(f'{tier:<18} {len(trades):>8,} {len(primaries):>8,} '
              f'{len(chains):>8,} {wr:>4.0f}% ${total_pnl:>+9,.0f} ${per_day:>+7,.0f}')

        # Save
        out = f'training/output/isolated/{tier}.pkl'
        with open(out, 'wb') as f:
            pickle.dump(trades, f)
        all_results[tier] = {'trades': len(trades), 'primaries': len(primaries),
                             'chains': len(chains), 'pnl': total_pnl,
                             'wr': wr, 'per_day': per_day}

    # Summary
    print()
    total_pnl = sum(r['pnl'] for r in all_results.values())
    print(f'Total across all tiers: ${total_pnl:+,.0f}')


if __name__ == '__main__':
    main()
