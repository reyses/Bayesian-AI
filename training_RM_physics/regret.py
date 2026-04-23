"""
Regret Analysis — bounded counterfactual with symmetric peak-validity gates.

For each trade, computes counterfactuals within a TIME-bounded window:

  LOOKBACK = 10 minutes before entry
  LOOKAHEAD = 30 minutes after exit

Branches:
  SAME EARLY:       close at in-trade peak (= shorter trade). GATED.
  SAME EXTENDED:    held past exit (= longer trade). GATED.
  COUNTER EARLY:    flipped direction at bar 1-3 (entry-timing regret)
  COUNTER IN-TRADE: flipped at entry, close at in-trade counter peak. GATED.
  COUNTER EXTENDED: flipped at entry, held past exit. GATED.
  SAME AT EXIT / COUNTER AT EXIT: baselines (always candidates)

Peak-validity gates (symmetric, 2026-04-17):

  EXTENDED is valid only if post-horizon peak > in-trade peak.
    Otherwise holding longer was dominated by in-trade action — that's
    an exit-timing problem disguised as a horizon problem.

  EARLY / IN-TRADE is valid only if in-trade peak >= post-horizon peak.
    Otherwise the optimal close point is OUTSIDE the trade window, not
    inside — crediting "shorter trade at peak" when a bigger peak was
    available post-exit would be giving up the bigger regret signal.

  At most ONE of (early, extended) can be strictly valid. On ties the
  in-trade / shorter option wins (stricter, less exposure).

Cadence note: windows are specified in MINUTES and converted to bars
using the price data's bar period (default 60s for 1m data). Before
this fix, LOOKAHEAD was 360 bars labeled "30 min at 5s" but the code
loaded 1m data → effective window was 6 hours.

Usage:
    from training_RM_physics.regret import compute_regret, compute_all_regrets

    regret = compute_regret(trade, prices, entry_idx, bar_period_sec=60)
    all_regrets = compute_all_regrets(trades)   # defaults to 1m bars
"""
import numpy as np
import pandas as pd
from typing import Dict, List

TICK = 0.25
TV = 0.50

# Time-bounded counterfactual windows (in minutes)
LOOKAHEAD_MIN = 30.0   # 30 min after trade exit
LOOKBACK_MIN = 10.0    # 10 min before trade entry

# Default bar period — ATLAS/1m uses 60s. Override per call for 5s data.
DEFAULT_BAR_PERIOD_SEC = 60


def _mins_to_bars(minutes: float, bar_period_sec: int) -> int:
    """Convert a time window to a bar count based on data cadence."""
    return max(1, int((minutes * 60) / bar_period_sec))


def compute_regret(trade: Dict, all_closes: np.ndarray, entry_bar_idx: int,
                   bar_period_sec: int = DEFAULT_BAR_PERIOD_SEC) -> Dict:
    """Compute bounded counterfactual for one trade.

    Args:
        trade: dict with 'dir', 'pnl', 'held', 'entry_price', 'peak'
        all_closes: full day's close prices (cadence = bar_period_sec)
        entry_bar_idx: index into all_closes where trade entered
        bar_period_sec: seconds per bar (60 for 1m, 5 for 5s). Used to
            convert LOOKAHEAD_MIN / LOOKBACK_MIN to bar counts.

    Returns:
        dict with counterfactual PnLs, peak-validity flags, best action.
    """
    direction = trade['dir']
    entry_price = trade['entry_price']
    held = trade['held']
    actual_pnl = trade['pnl']
    exit_bar = entry_bar_idx + held
    n = len(all_closes)

    # Convert time-bounded windows to bar counts for this cadence
    lookback_bars = _mins_to_bars(LOOKBACK_MIN, bar_period_sec)
    lookahead_bars = _mins_to_bars(LOOKAHEAD_MIN, bar_period_sec)

    # Full PnL curve: from entry to exit + LOOKAHEAD
    end_bar = min(entry_bar_idx + held + lookahead_bars, n)
    if entry_bar_idx >= n:
        return _empty_regret(actual_pnl)

    # === ENTRY LOOKBACK: what if entered up to LOOKBACK_MIN earlier? ===
    early_entries = []
    lookback_start = max(0, entry_bar_idx - lookback_bars)
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
        lb_end = min(lb_idx + held + lookahead_bars, n)
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

    # In-trade peak (same direction): max gain during [entry, exit)
    in_trade_peak_same = float(np.max(same_pnls[:max(held, 1)])) if len(same_pnls) > 0 else actual_pnl
    in_trade_peak_same_bar = int(np.argmax(same_pnls[:max(held, 1)])) if len(same_pnls) > 0 else 0

    # Extended: bars after actual exit
    same_ext_pnls = same_pnls[held:] if held < len(same_pnls) else np.array([actual_pnl])
    same_ext_best_bar_local = int(np.argmax(same_ext_pnls))
    same_ext_best_bar = held + same_ext_best_bar_local
    post_horizon_peak_same = float(np.max(same_ext_pnls))

    # Symmetric peak-validity gates (same direction):
    #   EXTENDED valid: post-horizon peak strictly dominates in-trade peak.
    #   EARLY / SHORTER valid: in-trade peak dominates post-horizon (>=).
    # At most one is strictly valid. Ties favor the shorter trade (lower
    # exposure, no horizon-dependent assumption).
    same_extended_valid = post_horizon_peak_same > in_trade_peak_same
    same_shorter_valid = in_trade_peak_same >= post_horizon_peak_same

    # Extended best is just the post-horizon peak — used as a reporting
    # field regardless of validity (the gate decides candidacy below).
    same_ext_best = post_horizon_peak_same

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

    # In-trade peak (counter direction): max flipped-position gain during
    # [entry, exit). This is what a counter-direction entry at bar 0 would
    # have captured WITHIN the original trade's time window.
    in_trade_peak_counter = float(np.max(counter_pnls[:max(held, 1)])) if len(counter_pnls) > 0 else 0.0
    in_trade_peak_counter_bar = int(np.argmax(counter_pnls[:max(held, 1)])) if len(counter_pnls) > 0 else 0

    # Extended: counter PnL past the actual exit bar
    counter_ext_pnls = counter_pnls[held:] if held < len(counter_pnls) else np.array([0.0])
    counter_ext_best_bar_local = int(np.argmax(counter_ext_pnls))
    counter_ext_best_bar = held + counter_ext_best_bar_local
    post_horizon_peak_counter = float(np.max(counter_ext_pnls))

    # Symmetric peak-validity gates (counter direction):
    counter_extended_valid = post_horizon_peak_counter > in_trade_peak_counter
    counter_in_trade_valid = in_trade_peak_counter >= post_horizon_peak_counter

    counter_ext_best = post_horizon_peak_counter

    # Overall counter best
    counter_best_bar = int(np.argmax(counter_pnls))
    counter_best = float(counter_pnls[counter_best_bar])

    # === BEST ACTION ===
    # Baseline options: always candidates.
    #   SAME_AT_EXIT / COUNTER_AT_EXIT = what happened / what flipping at
    #   exit would have yielded.
    #   COUNTER_EARLY = entry-timing regret for the counter direction (max
    #   counter PnL in bars 1-3, not a peak-validity target).
    options = {
        'same_at_exit': same_at_exit,
        'counter_at_exit': counter_at_exit,
        'counter_early': counter_early_best,
    }
    # Peak-validity gated (symmetric):
    #   SAME_EARLY is a "close at in-trade peak / shorter trade" signal.
    #   It competes only if the in-trade peak dominates the post-horizon.
    if same_shorter_valid:
        options['same_early'] = same_early_best
    if same_extended_valid:
        options['same_extended'] = post_horizon_peak_same
    #   COUNTER_IN_TRADE = flip at entry, close at in-trade counter peak.
    #   New option mirroring SAME_EARLY for the flipped direction. Gated
    #   on counter_in_trade_valid. Also requires a non-trivial in-trade
    #   counter peak (> 0) to be a meaningful candidate.
    if counter_in_trade_valid and in_trade_peak_counter > 0:
        options['counter_in_trade'] = in_trade_peak_counter
    if counter_extended_valid:
        options['counter_extended'] = post_horizon_peak_counter

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

        # Symmetric peak-validity gates (2026-04-17):
        # Where does the global best live, inside the trade window or
        # outside it? At most one of (*_shorter_valid, *_extended_valid)
        # is strictly true; ties go to SHORTER / IN_TRADE.
        'in_trade_peak_same': in_trade_peak_same,
        'in_trade_peak_same_bar': in_trade_peak_same_bar,
        'post_horizon_peak_same': post_horizon_peak_same,
        'same_shorter_valid': bool(same_shorter_valid),
        'same_extended_valid': bool(same_extended_valid),
        'in_trade_peak_counter': in_trade_peak_counter,
        'in_trade_peak_counter_bar': in_trade_peak_counter_bar,
        'post_horizon_peak_counter': post_horizon_peak_counter,
        'counter_in_trade_valid': bool(counter_in_trade_valid),
        'counter_extended_valid': bool(counter_extended_valid),

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
        'in_trade_peak_same': 0.0, 'in_trade_peak_same_bar': 0,
        'post_horizon_peak_same': 0.0,
        'same_shorter_valid': False, 'same_extended_valid': False,
        'in_trade_peak_counter': 0.0, 'in_trade_peak_counter_bar': 0,
        'post_horizon_peak_counter': 0.0,
        'counter_in_trade_valid': False, 'counter_extended_valid': False,
        'counter_early_best': 0, 'counter_early_bar': 0, 'counter_at_exit': 0,
        'counter_ext_best': 0, 'counter_ext_bar': 0, 'counter_best': 0, 'counter_best_bar': 0,
        'best_action': 'same_at_exit', 'best_pnl': actual_pnl, 'regret': 0,
        'early_entry_gain': 0.0, 'best_early_bars_before': 0,
        'best_early_same_peak': 0.0, 'best_early_counter_peak': 0.0,
        'same_curve': [], 'counter_curve': [], 'early_entries': [],
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


def correct_trades(trades: List[Dict], price_dir: str = 'DATA/ATLAS/1m',
                   bar_period_sec: int = DEFAULT_BAR_PERIOD_SEC) -> List[Dict]:
    """Produce corrected trades from regret analysis.

    For each trade, computes the gated best_action from compute_regret()
    and produces a new trade record with:
      * the direction matching best_action (same or counter-flipped)
      * the exit bar matching the SPECIFIC peak the best_action refers to
        (not the overall argmax — that would ignore peak-validity gates)
      * the real PnL from actual prices at that bar

    Per-action exit-bar map:
      same_early        -> in_trade_peak_same_bar      (peak inside trade)
      same_at_exit      -> actual held                 (no change)
      same_extended     -> same_ext_bar                (peak post-horizon)
      counter_early     -> counter_early_bar           (bar 0-3, flip early)
      counter_at_exit   -> actual held                 (flip at exit)
      counter_in_trade  -> in_trade_peak_counter_bar   (flip, close at in-trade
                                                        counter peak)
      counter_extended  -> counter_ext_bar             (flip, peak post-horizon)

    The corrected trades are the ORACLE ground truth — what should have
    happened. They feed any future learning layer (CNN direction flip,
    exit timing, entry filter).
    """
    import os
    from tqdm import tqdm

    # Group trades by day for efficient price-data loading
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

            # Compute gated regret for this trade
            r = compute_regret(t, closes, entry_bar, bar_period_sec=bar_period_sec)
            best_action = r['best_action']
            best_pnl = r['best_pnl']
            held = int(t.get('held', 0))

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
                earlier_state = approach[-early_bars]
                if 'features' in earlier_state:
                    corrected_entry_79d = earlier_state['features']
                    if hasattr(corrected_entry_79d, 'tolist'):
                        corrected_entry_79d = corrected_entry_79d.tolist()
                corrected_entry_price = earlier_state.get('price', entry_price)
                corrected_entry_bar = max(0, entry_bar - early_bars)

            # ── Per-action exit-bar mapping (respects peak-validity gates) ─
            # Bar offsets are from entry_bar; some are < held (intra-trade),
            # some are >= held (post-horizon). The mapping uses the SPECIFIC
            # peak bar the gated best_action points at.
            action_to_offset = {
                'same_early':       r.get('in_trade_peak_same_bar', 0),
                'same_at_exit':     held,
                'same_extended':    r.get('same_ext_bar', held),
                'counter_early':    r.get('counter_early_bar', 0),
                'counter_at_exit':  held,
                'counter_in_trade': r.get('in_trade_peak_counter_bar', 0),
                'counter_extended': r.get('counter_ext_bar', held),
            }
            offset = action_to_offset.get(best_action, held)
            corrected_exit_bar = min(entry_bar + int(offset), n - 1)

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
