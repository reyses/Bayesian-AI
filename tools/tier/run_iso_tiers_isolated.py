"""
Run each IsoEngine tier in isolation across 277 days.

For each of the 5 tiers (NMP_FADE, NMP_RIDE, NMP_FADE_FLIP_VR,
NMP_FADE_FLIP_WICK, NMP_RIDE_FLIP), spin up an IsoEngine with
`only_tier=<tier>` so only that tier's condition can fire. Run the full
IS dataset through each and report the standalone WR / $/day / $/trade.

This is the honest per-tier measurement — no priority-ordering contamination,
no competing tier stealing volume. Each tier gets its full universe of
candidate bars and we see what it actually captures alone.

Usage:
    python tools/run_iso_tiers_isolated.py
    python tools/run_iso_tiers_isolated.py NMP_FADE NMP_FADE_FLIP_VR
"""
import os
import sys
import glob
import pickle
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.sfe_ticker import FeatureTicker
from training.nightmare_iso import IsoEngine, TIER_MAP

FEATURES_DIR = 'DATA/ATLAS/FEATURES_5s'
ATLAS_1M = 'DATA/ATLAS/1m'
OUTPUT_DIR = 'training_iso/output/isolated'


def run_tier(tier: str, feat_files):
    """Run one tier across all days with only_tier=tier."""
    engine = IsoEngine(only_tier=tier)
    all_trades = []

    for fpath in feat_files:
        day_name = os.path.basename(fpath).replace('.parquet', '')
        price_file = os.path.join(ATLAS_1M, f'{day_name}.parquet')
        if not os.path.exists(price_file):
            price_file = None

        engine.reset()
        ft = FeatureTicker(fpath, price_file=price_file)
        for state in ft:
            engine.on_state(state)
        engine.force_close()

        for t in engine.trades[-len(engine.trades):]:
            t['day'] = day_name
        all_trades.extend(engine.trades)
        engine.trades = []   # fresh per-day collection

    return all_trades


def main():
    tiers_to_run = sys.argv[1:] or list(TIER_MAP.keys())
    tiers_to_run = [t for t in tiers_to_run if t in TIER_MAP]

    feat_files = sorted(glob.glob(os.path.join(FEATURES_DIR, '*.parquet')))
    # IS only (2025)
    feat_files = [f for f in feat_files if '2025_' in os.path.basename(f)]
    print(f'Feature files: {len(feat_files)} IS days')
    print(f'Tiers to run: {tiers_to_run}')
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'{"Tier":<22} {"N":>7} {"WR":>6} {"Total $":>10} {"$/day":>8} {"$/trade":>8}')
    print('-' * 75)

    rows = []
    for tier in tiers_to_run:
        t0 = datetime.now()
        trades = run_tier(tier, feat_files)
        dt = (datetime.now() - t0).total_seconds()
        n = len(trades)
        if n == 0:
            print(f'{tier:<22} {n:>7} (no trades)  [{dt:.0f}s]')
            continue
        total = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / n * 100
        days = len(set(t.get('day', '') for t in trades))
        per_day = total / max(days, 1)
        per_trade = total / n

        print(f'{tier:<22} {n:>7,} {wr:>5.1f}% ${total:>+9,.0f} '
              f'${per_day:>+7,.1f} ${per_trade:>+7,.2f}  [{dt:.0f}s]')

        out_path = os.path.join(OUTPUT_DIR, f'{tier}.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(trades, f)
        rows.append({
            'tier': tier, 'n': n, 'wr': wr, 'total': total,
            'per_day': per_day, 'per_trade': per_trade, 'days': days,
        })

    if rows:
        print()
        print(f'Saved per-tier trade pickles to {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
