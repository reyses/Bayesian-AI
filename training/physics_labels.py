"""
Physics Labels — extract per-tier CNN training labels from oracle (regret) analysis.

Takes trades from physics forward pass + regret analysis, produces labels for 5 jobs:
  1. Entry gate:  is this a good entry? (winner/loser from actual outcome)
  2. Direction:   long or short? (from regret optimal direction)
  3. Duration:    SHORT/MEDIUM/LONG? (from regret optimal exit bar, binned)
  4. Exit:        HOLD or EXIT? (per bar, from regret optimal exit bar)
  5. Loser ID:    RECOVER or DEAD? (per bar when pnl < 0, from final outcome)

Labels are deterministic — same input trades → same labels.

Usage:
    from training.physics_labels import generate_all_labels

    labels = generate_all_labels(trades, regret_df)
    # labels = {tier: {job: DataFrame}}
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import defaultdict

# Duration bins (in 1m bars — bars_held is timestamp-based, 1 = 1 minute)
DURATION_SHORT_MAX = 10    # < 10 min = quick scalp
DURATION_MEDIUM_MAX = 40   # 10-40 min = standard trade
# >= 40 min = extended hold (LONG)

# Minimum trades per tier for CNN training
MIN_TRADES_PER_TIER = 100


def _bin_duration(bars: int) -> int:
    """Bin optimal exit bar into SHORT(0)/MEDIUM(1)/LONG(2)."""
    if bars < DURATION_SHORT_MAX:
        return 0
    elif bars < DURATION_MEDIUM_MAX:
        return 1
    else:
        return 2


def generate_entry_labels(trades: List[Dict]) -> pd.DataFrame:
    """Job 1: Entry gate labels.

    Label = 1 if trade was a winner (pnl > 0), 0 if loser.
    """
    rows = []
    for t in trades:
        rows.append({
            'tier': t.get('entry_tier', 'UNKNOWN'),
            'direction': t.get('dir', 'long'),
            'pnl': t.get('pnl', 0),
            'label': 1 if t.get('pnl', 0) > 0 else 0,
            'entry_79d': t.get('entry_79d', []),
        })
    return pd.DataFrame(rows)


def generate_direction_labels(trades: List[Dict], regret_df: pd.DataFrame) -> pd.DataFrame:
    """Job 2: Direction labels from regret optimal action.

    Label = absolute direction (0=short, 1=long) that the oracle says is best.
    """
    rows = []
    for i, t in enumerate(trades):
        if i >= len(regret_df):
            break

        r = regret_df.iloc[i]
        physics_dir = t.get('dir', 'long')
        best_action = r.get('best_action', 'same_at_exit')

        # Convert regret's SAME/COUNTER to absolute long/short
        is_counter = 'counter' in best_action
        if is_counter:
            oracle_dir = 'short' if physics_dir == 'long' else 'long'
        else:
            oracle_dir = physics_dir

        rows.append({
            'tier': t.get('entry_tier', 'UNKNOWN'),
            'physics_dir': physics_dir,
            'oracle_dir': oracle_dir,
            'label': 1 if oracle_dir == 'long' else 0,
            'best_pnl': r.get('best_pnl', 0),
            'regret': r.get('regret', 0),
            'entry_79d': t.get('entry_79d', []),
        })
    return pd.DataFrame(rows)


def generate_duration_labels(trades: List[Dict], regret_df: pd.DataFrame) -> pd.DataFrame:
    """Job 3: Duration labels from regret optimal exit bar.

    Label = SHORT(0) / MEDIUM(1) / LONG(2) based on oracle's optimal exit bar.
    """
    rows = []
    for i, t in enumerate(trades):
        if i >= len(regret_df):
            break

        r = regret_df.iloc[i]
        best_action = r.get('best_action', 'same_at_exit')

        # Find the optimal bar count from regret
        if 'same' in best_action:
            optimal_bar = int(r.get('same_best_bar', t.get('held', 20)))
        else:
            optimal_bar = int(r.get('counter_best_bar', t.get('held', 20)))

        rows.append({
            'tier': t.get('entry_tier', 'UNKNOWN'),
            'optimal_bar': optimal_bar,
            'label': _bin_duration(optimal_bar),
            'actual_held': t.get('held', 0),
            'pnl': t.get('pnl', 0),
            'entry_79d': t.get('entry_79d', []),
        })
    return pd.DataFrame(rows)


def generate_exit_labels(trades: List[Dict], regret_df: pd.DataFrame) -> pd.DataFrame:
    """Job 4: Exit labels — per bar during each trade.

    For each bar in the trade path:
      HOLD(1) if bar < oracle's optimal exit bar
      EXIT(0) if bar >= oracle's optimal exit bar
    """
    rows = []
    for i, t in enumerate(trades):
        if i >= len(regret_df):
            break

        r = regret_df.iloc[i]
        path = t.get('path', [])
        if not path:
            continue

        best_action = r.get('best_action', 'same_at_exit')
        if 'same' in best_action:
            optimal_bar = int(r.get('same_best_bar', len(path)))
        else:
            optimal_bar = int(r.get('counter_best_bar', len(path)))

        tier = t.get('entry_tier', 'UNKNOWN')
        entry_z = t.get('entry_79d', [0] * 91)[10] if len(t.get('entry_79d', [])) > 10 else 0

        for bar_idx, bar in enumerate(path):
            feat = bar.get('features', None)
            if feat is None:
                continue

            pnl = bar.get('pnl', 0) if 'pnl' in bar else 0
            peak = bar.get('peak_pnl', 0) if 'peak_pnl' in bar else 0

            rows.append({
                'tier': tier,
                'trade_idx': i,
                'bar_idx': bar_idx,
                'label': 1 if bar_idx < optimal_bar else 0,  # HOLD=1, EXIT=0
                'features': feat if isinstance(feat, list) else feat.tolist(),
                'bars_held': bar_idx,
                'pnl': pnl,
                'peak_pnl': peak,
                'entry_z': entry_z,
            })
    return pd.DataFrame(rows)


def generate_loser_labels(trades: List[Dict]) -> pd.DataFrame:
    """Job 5: Loser ID labels — per bar when pnl < 0.

    For bars where trade is underwater:
      RECOVER(1) if the trade eventually ended positive
      DEAD(0) if the trade ended negative
    """
    rows = []
    for i, t in enumerate(trades):
        path = t.get('path', [])
        if not path:
            continue

        final_pnl = t.get('pnl', 0)
        label = 1 if final_pnl > 0 else 0  # RECOVER or DEAD
        tier = t.get('entry_tier', 'UNKNOWN')
        entry_z = t.get('entry_79d', [0] * 91)[10] if len(t.get('entry_79d', [])) > 10 else 0

        for bar_idx, bar in enumerate(path):
            pnl = bar.get('pnl', 0) if 'pnl' in bar else 0
            if pnl >= 0:
                continue  # only label bars where trade is underwater

            feat = bar.get('features', None)
            if feat is None:
                continue

            peak = bar.get('peak_pnl', 0) if 'peak_pnl' in bar else 0

            rows.append({
                'tier': tier,
                'trade_idx': i,
                'bar_idx': bar_idx,
                'label': label,
                'features': feat if isinstance(feat, list) else feat.tolist(),
                'bars_held': bar_idx,
                'pnl': pnl,
                'peak_pnl': peak,
                'entry_z': entry_z,
            })
    return pd.DataFrame(rows)


def generate_all_labels(trades: List[Dict], regret_df: pd.DataFrame) -> Dict:
    """Generate all 5 label types, split by tier.

    Returns:
        {tier: {
            'entry': DataFrame,
            'direction': DataFrame,
            'duration': DataFrame,
            'exit': DataFrame,
            'loser': DataFrame,
            'n_trades': int,
            'trainable': bool,  # True if n_trades >= MIN_TRADES_PER_TIER
        }}
    """
    # Generate global labels
    entry_df = generate_entry_labels(trades)
    direction_df = generate_direction_labels(trades, regret_df)
    duration_df = generate_duration_labels(trades, regret_df)
    exit_df = generate_exit_labels(trades, regret_df)
    loser_df = generate_loser_labels(trades)

    # Split by tier
    tiers = sorted(entry_df['tier'].unique())
    result = {}

    for tier in tiers:
        tier_entry = entry_df[entry_df['tier'] == tier].reset_index(drop=True)
        tier_dir = direction_df[direction_df['tier'] == tier].reset_index(drop=True)
        tier_dur = duration_df[duration_df['tier'] == tier].reset_index(drop=True)
        tier_exit = exit_df[exit_df['tier'] == tier].reset_index(drop=True)
        tier_loser = loser_df[loser_df['tier'] == tier].reset_index(drop=True)

        n_trades = len(tier_entry)
        trainable = n_trades >= MIN_TRADES_PER_TIER

        result[tier] = {
            'entry': tier_entry,
            'direction': tier_dir,
            'duration': tier_dur,
            'exit': tier_exit,
            'loser': tier_loser,
            'n_trades': n_trades,
            'trainable': trainable,
        }

        wr = tier_entry['label'].mean() * 100 if n_trades > 0 else 0
        status = 'TRAIN' if trainable else 'SKIP (too few)'
        print(f'  {tier:<20} {n_trades:>5} trades  WR={wr:4.0f}%  {status}')

        if trainable:
            # Label distribution summary
            dir_long = (tier_dir['label'] == 1).sum()
            dur_counts = tier_dur['label'].value_counts().to_dict()
            n_exit = len(tier_exit)
            n_loser = len(tier_loser)
            print(f'    direction: {dir_long}/{n_trades} long ({dir_long/n_trades*100:.0f}%)')
            print(f'    duration:  S={dur_counts.get(0,0)} M={dur_counts.get(1,0)} L={dur_counts.get(2,0)}')
            print(f'    exit bars: {n_exit}  loser bars: {n_loser}')

    return result
