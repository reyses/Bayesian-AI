"""
Regret Analysis — full counterfactual for every trade.

For each trade (winner or loser), computes:
  SAME EARLY:       what if exited 1-3 bars before actual exit?
  SAME EXTENDED:    what if held past exit?
  COUNTER EARLY:    what if flipped direction at bar 1-3?
  COUNTER AT EXIT:  what if flipped at exit point?
  COUNTER EXTENDED: what if flipped at entry and held?

The BEST ACTION tells the tree what it SHOULD have done.
The REGRET tells the tree how much it left on the table.

Usage:
    from nn_v2.regret import compute_regret, compute_all_regrets

    regret = compute_regret(trade, prices, entry_idx)
    all_regrets = compute_all_regrets(trades, price_df)
"""
import numpy as np
import pandas as pd
from typing import Dict, List

TICK = 0.25
TV = 0.50
LOOKAHEAD = 360  # 30 min at 5s resolution (360 bars) — full inverse peak window
LOOKBACK = 120   # 10 min at 5s resolution (120 bars) — early entry window


def compute_regret(trade: Dict, all_closes: np.ndarray, entry_bar_idx: int) -> Dict:
    """Compute full counterfactual for one trade.

    Args:
        trade: dict with 'dir', 'pnl', 'held', 'entry_price', 'peak'
        all_closes: full day's close prices (1m bars)
        entry_bar_idx: index into all_closes where trade entered

    Returns:
        dict with all counterfactual PnLs and best action
    """
    direction = trade['dir']
    entry_price = trade['entry_price']
    held = trade['held']
    actual_pnl = trade['pnl']
    exit_bar = entry_bar_idx + held
    n = len(all_closes)

    # Full PnL curve: from entry to entry + held + LOOKAHEAD
    end_bar = min(entry_bar_idx + held + LOOKAHEAD, n)
    if entry_bar_idx >= n:
        return _empty_regret(actual_pnl)

    # === ENTRY LOOKBACK: what if entered 1-10 bars earlier? ===
    early_entries = []
    lookback_start = max(0, entry_bar_idx - LOOKBACK)
    for lb_idx in range(lookback_start, entry_bar_idx):
        lb_price = all_closes[lb_idx]
        # If entered at this earlier bar, what's the PnL at actual exit bar?
        if direction == 'short':
            lb_same_at_exit = (lb_price - all_closes[min(exit_bar, n - 1)]) / TICK * TV
            lb_counter_at_exit = (all_closes[min(exit_bar, n - 1)] - lb_price) / TICK * TV
        else:
            lb_same_at_exit = (all_closes[min(exit_bar, n - 1)] - lb_price) / TICK * TV
            lb_counter_at_exit = (lb_price - all_closes[min(exit_bar, n - 1)]) / TICK * TV

        # Best PnL if entered here and held to peak
        lb_end = min(lb_idx + held + LOOKAHEAD, n)
        lb_prices = all_closes[lb_idx:lb_end]
        if direction == 'short':
            lb_same_peak = float(np.max((lb_price - lb_prices) / TICK * TV))
            lb_counter_peak = float(np.max((lb_prices - lb_price) / TICK * TV))
        else:
            lb_same_peak = float(np.max((lb_prices - lb_price) / TICK * TV))
            lb_counter_peak = float(np.max((lb_price - lb_prices) / TICK * TV))

        bars_before = entry_bar_idx - lb_idx
        early_entries.append({
            'bars_before': bars_before,
            'price': float(lb_price),
            'same_at_exit': lb_same_at_exit,
            'counter_at_exit': lb_counter_at_exit,
            'same_peak': lb_same_peak,
            'counter_peak': lb_counter_peak,
        })

    # Best early entry
    if early_entries:
        best_early = max(early_entries, key=lambda e: max(e['same_peak'], e['counter_peak']))
        early_entry_gain = max(best_early['same_peak'], best_early['counter_peak']) - max(actual_pnl, 0)
    else:
        best_early = None
        early_entry_gain = 0.0

    # === PnL curves from actual entry ===
    same_pnls = []
    counter_pnls = []
    for i in range(entry_bar_idx, end_bar):
        price = all_closes[i]
        if direction == 'short':
            same = (entry_price - price) / TICK * TV
            counter = (price - entry_price) / TICK * TV
        else:
            same = (price - entry_price) / TICK * TV
            counter = (entry_price - price) / TICK * TV
        same_pnls.append(same)
        counter_pnls.append(counter)

    same_pnls = np.array(same_pnls)
    counter_pnls = np.array(counter_pnls)
    bars_from_entry = np.arange(len(same_pnls))

    # === SAME DIRECTION ===
    # Early: bars before actual exit
    same_early_pnls = same_pnls[:held] if held > 0 else np.array([0.0])
    same_early_best_bar = int(np.argmax(same_early_pnls))
    same_early_best = float(same_early_pnls[same_early_best_bar])

    # At exit
    same_at_exit = float(same_pnls[held]) if held < len(same_pnls) else actual_pnl

    # Extended: bars after actual exit
    same_ext_pnls = same_pnls[held:] if held < len(same_pnls) else np.array([actual_pnl])
    same_ext_best_bar = held + int(np.argmax(same_ext_pnls))
    same_ext_best = float(np.max(same_ext_pnls))

    # Overall same best
    same_best_bar = int(np.argmax(same_pnls))
    same_best = float(same_pnls[same_best_bar])

    # === COUNTER DIRECTION ===
    # Early: flip at bar 1, 2, 3
    counter_early_pnls = counter_pnls[:min(4, len(counter_pnls))]
    counter_early_best_bar = int(np.argmax(counter_early_pnls))
    counter_early_best = float(counter_early_pnls[counter_early_best_bar]) if len(counter_early_pnls) > 0 else 0.0

    # At exit: flip at the exit bar
    counter_at_exit = float(counter_pnls[held]) if held < len(counter_pnls) else 0.0

    # Extended: flip at entry and hold
    counter_ext_best_bar = int(np.argmax(counter_pnls))
    counter_ext_best = float(np.max(counter_pnls))

    # Overall counter best
    counter_best_bar = int(np.argmax(counter_pnls))
    counter_best = float(counter_pnls[counter_best_bar])

    # === BEST ACTION ===
    options = {
        'same_early': same_early_best,
        'same_at_exit': same_at_exit,
        'same_extended': same_ext_best,
        'counter_early': counter_early_best,
        'counter_at_exit': counter_at_exit,
        'counter_extended': counter_ext_best,
    }

    best_action = max(options, key=options.get)
    best_pnl = options[best_action]
    regret = best_pnl - actual_pnl

    return {
        # Actual
        'actual_pnl': actual_pnl,
        'actual_held': held,
        'direction': direction,

        # Same direction
        'same_early_best': same_early_best,
        'same_early_bar': same_early_best_bar,
        'same_at_exit': same_at_exit,
        'same_ext_best': same_ext_best,
        'same_ext_bar': same_ext_best_bar,
        'same_best': same_best,
        'same_best_bar': same_best_bar,

        # Counter direction
        'counter_early_best': counter_early_best,
        'counter_early_bar': counter_early_best_bar,
        'counter_at_exit': counter_at_exit,
        'counter_ext_best': counter_ext_best,
        'counter_ext_bar': counter_ext_best_bar,
        'counter_best': counter_best,
        'counter_best_bar': counter_best_bar,

        # Decision
        'best_action': best_action,
        'best_pnl': best_pnl,
        'regret': regret,

        # Entry timing
        'early_entry_gain': early_entry_gain,
        'best_early_bars_before': best_early['bars_before'] if best_early else 0,
        'best_early_same_peak': best_early['same_peak'] if best_early else 0,
        'best_early_counter_peak': best_early['counter_peak'] if best_early else 0,

        # Full curves (for tree training)
        'same_curve': same_pnls.tolist(),
        'counter_curve': counter_pnls.tolist(),
        'early_entries': early_entries,
    }


def _empty_regret(actual_pnl):
    return {
        'actual_pnl': actual_pnl, 'actual_held': 0, 'direction': '',
        'same_early_best': 0, 'same_early_bar': 0, 'same_at_exit': actual_pnl,
        'same_ext_best': 0, 'same_ext_bar': 0, 'same_best': actual_pnl, 'same_best_bar': 0,
        'counter_early_best': 0, 'counter_early_bar': 0, 'counter_at_exit': 0,
        'counter_ext_best': 0, 'counter_ext_bar': 0, 'counter_best': 0, 'counter_best_bar': 0,
        'best_action': 'same_at_exit', 'best_pnl': actual_pnl, 'regret': 0,
        'same_curve': [], 'counter_curve': [],
    }


def compute_all_regrets(trades: List[Dict], price_dir: str = 'DATA/ATLAS/1m') -> pd.DataFrame:
    """Compute regret for all trades. Returns DataFrame with one row per trade.

    Loads price data per day, finds entry bar index, computes regret.
    """
    import os, glob
    from tqdm import tqdm

    # Group trades by day
    by_day = {}
    for i, t in enumerate(trades):
        day = t.get('day', '')
        if day not in by_day:
            by_day[day] = []
        by_day[day].append((i, t))

    results = [None] * len(trades)

    for day, day_trades in tqdm(by_day.items(), desc='Regret analysis', unit='day'):
        # Load day's price data
        price_file = os.path.join(price_dir, f'{day}.parquet')
        if not os.path.exists(price_file):
            for i, t in day_trades:
                results[i] = _empty_regret(t['pnl'])
            continue

        df = pd.read_parquet(price_file).sort_values('timestamp')
        closes = df['close'].values
        timestamps = df['timestamp'].values

        for i, t in day_trades:
            # Find entry bar index
            entry_ts = t.get('timestamp', 0)
            if entry_ts > 0:
                entry_idx = int(np.searchsorted(timestamps, entry_ts, side='left'))
                entry_idx = min(entry_idx, len(closes) - 1)
            else:
                entry_idx = 0

            regret = compute_regret(t, closes, entry_idx)
            regret['trade_id'] = t.get('trade_id', i)
            regret['day'] = day
            regret['entry_tier'] = t.get('entry_tier', 'NMP')
            results[i] = regret

    # Fill any None results
    for i in range(len(results)):
        if results[i] is None:
            results[i] = _empty_regret(trades[i]['pnl'])

    # Build DataFrame (without curve/array columns)
    flat = []
    for r in results:
        row = {k: v for k, v in r.items()
               if k not in ('same_curve', 'counter_curve', 'early_entries', 'early_approach_79d')}
        flat.append(row)

    return pd.DataFrame(flat)


def summarize_regret_by_branch(regret_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize regret per tree branch. The KEY output for tree exit calibration."""
    if 'leaf_id' not in regret_df.columns:
        return pd.DataFrame()

    rows = []
    for lid, group in regret_df.groupby('leaf_id'):
        n = len(group)
        actual_total = group['actual_pnl'].sum()
        same_best_total = group['same_best'].sum()
        counter_best_total = group['counter_best'].sum()
        avg_regret = group['regret'].mean()

        # What's the dominant best action for this branch?
        action_counts = group['best_action'].value_counts()
        dominant_action = action_counts.index[0]
        dominant_pct = action_counts.iloc[0] / n * 100

        # Optimal PnL if we followed the best action
        optimal_total = group['best_pnl'].sum()

        rows.append({
            'leaf_id': lid,
            'n_trades': n,
            'actual_total': actual_total,
            'actual_avg': actual_total / n,
            'same_best_total': same_best_total,
            'counter_best_total': counter_best_total,
            'optimal_total': optimal_total,
            'avg_regret': avg_regret,
            'total_regret': group['regret'].sum(),
            'dominant_action': dominant_action,
            'dominant_pct': dominant_pct,
            # Optimal exit timing
            'avg_same_best_bar': group['same_best_bar'].mean(),
            'avg_counter_best_bar': group['counter_best_bar'].mean(),
        })

    return pd.DataFrame(rows).sort_values('total_regret', ascending=False)


def correct_trades(trades: List[Dict], price_dir: str = 'DATA/ATLAS/1m') -> List[Dict]:
    """Produce corrected trades from regret analysis.

    For each NMP trade, computes the optimal action and produces a new trade
    record with the CORRECT direction, exit bar, and real PnL from prices.

    The corrected trades are what SHOULD have happened — ground truth for
    training the tree on natural patterns instead of error predictions.

    Returns list of corrected trade dicts (same format as NMP trades).
    """
    import os
    from tqdm import tqdm

    # Group trades by day
    by_day = {}
    for i, t in enumerate(trades):
        day = t.get('day', '')
        if day not in by_day:
            by_day[day] = []
        by_day[day].append((i, t))

    corrected = []

    for day, day_trades in tqdm(by_day.items(), desc='Correcting trades', unit='day'):
        price_file = os.path.join(price_dir, f'{day}.parquet')
        if not os.path.exists(price_file):
            continue

        df = pd.read_parquet(price_file).sort_values('timestamp')
        closes = df['close'].values
        timestamps = df['timestamp'].values
        n = len(closes)

        for idx, t in day_trades:
            entry_ts = t.get('timestamp', 0)
            entry_bar = int(np.searchsorted(timestamps, entry_ts, side='left'))
            entry_bar = min(entry_bar, n - 1)
            entry_price = closes[entry_bar]

            # Compute regret for this trade
            r = compute_regret(t, closes, entry_bar)
            best_action = r['best_action']
            best_pnl = r['best_pnl']

            # Determine corrected direction
            original_dir = t['dir']
            if 'counter' in best_action:
                corrected_dir = 'long' if original_dir == 'short' else 'short'
            else:
                corrected_dir = original_dir

            # Early entry: use approach buffer 79D if entering earlier is better
            early_bars = int(r.get('best_early_bars_before', 0))
            approach = t.get('approach', [])
            corrected_entry_79d = t.get('entry_79d', [])
            corrected_entry_bar = entry_bar
            corrected_entry_price = entry_price

            if early_bars > 0 and approach and early_bars <= len(approach):
                # Approach buffer is newest-last: approach[-1] = bar before entry
                # approach[-early_bars] = the earlier entry point
                earlier_state = approach[-early_bars]
                if 'features_79d' in earlier_state:
                    corrected_entry_79d = earlier_state['features_79d']
                    if hasattr(corrected_entry_79d, 'tolist'):
                        corrected_entry_79d = corrected_entry_79d.tolist()
                corrected_entry_price = earlier_state.get('price', entry_price)
                corrected_entry_bar = max(0, entry_bar - early_bars)

            # Exit bar is absolute — regret computed from original entry
            # Early entry extends segment backward, exit stays where regret said
            if 'same' in best_action:
                corrected_exit_bar = entry_bar + r['same_best_bar']
            else:
                corrected_exit_bar = entry_bar + r['counter_best_bar']
            corrected_exit_bar = min(corrected_exit_bar, n - 1)

            # Real PnL from actual prices at corrected entry/exit
            exit_price = closes[corrected_exit_bar]
            if corrected_dir == 'long':
                corrected_pnl = (exit_price - corrected_entry_price) / TICK * TV
            else:
                corrected_pnl = (corrected_entry_price - exit_price) / TICK * TV

            corrected_held = corrected_exit_bar - corrected_entry_bar

            # Build corrected trade (same format as NMP trade)
            ct = {
                'trade_id': len(corrected),
                'day': day,
                'timestamp': entry_ts,
                'time': t.get('time', ''),
                'dir': corrected_dir,
                'entry_price': corrected_entry_price,
                'exit_price': exit_price,
                'pnl': corrected_pnl,
                'held': corrected_held,
                'peak': best_pnl,
                'exit': best_action,
                # Early entry info
                'early_bars': early_bars,
                # Original trade data (for context)
                'original_dir': original_dir,
                'original_pnl': t['pnl'],
                'original_held': t['held'],
                'entry_tier': t.get('entry_tier', 'NMP'),
                'original_exit_reason': t.get('exit_reason', t.get('exit', '')),
                'best_action': best_action,
                'regret': r['regret'],
                # 79D data — uses approach buffer for early entries
                'entry_79d': corrected_entry_79d,
                'exit_79d': t.get('exit_79d', []),
                'approach': approach,
                'approach_length': len(approach),
                'path': t.get('path', []),
                'path_length': len(t.get('path', [])),
            }
            corrected.append(ct)

    # Summary
    n_flipped = sum(1 for c in corrected if c['dir'] != c['original_dir'])
    n_early = sum(1 for c in corrected if c.get('early_bars', 0) > 0)
    total_original = sum(c['original_pnl'] for c in corrected)
    total_corrected = sum(c['pnl'] for c in corrected)
    print(f'  Corrected {len(corrected)} trades: {n_flipped} direction-flipped, {n_early} early-entry')
    print(f'  Original PnL: ${total_original:,.0f} -> Corrected PnL: ${total_corrected:,.0f}')

    return corrected
