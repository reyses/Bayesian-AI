"""
Sim Executor — thin bar-loop driver for training.

Walks bars from a FeatureTicker, calls ledger.update_bar → engine.evaluate →
applies DecisionBatch to the ledger. Produces a closed-trade list in the same
format as BlendedEngine.trades.

Single responsibility: loop + glue. No signal logic, no position state.

Spec: docs/JULES_ENGINE_DECOUPLE_ORDERS.md (Phase 3)
"""
import numpy as np
from typing import Iterator, Dict, List

from core.ledger import Ledger
from core.engine_signals import DecisionBatch

# Feature indices for entry context (same constants as nightmare_blended.py).
# Duplicated here to avoid a circular import — the executor must not import
# the engine module at module level (engine imports are the caller's job).
_1M_OFFSET = 12
_Z = 0
_VR = 2
_1M_VELOCITY_IDX = 15   # TF1*12 + 3
_1H_Z_IDX = 48          # TF4*12
_1M_VOL_REL_IDX = 17    # TF1*12 + 5
_5M_VELOCITY_IDX = 27   # TF2*12 + 3

# RIDE exit patience tiers (must match nightmare_blended.py constants)
RIDE_EXIT_BARS_TIERS = {'strong': 5, 'medium': 3, 'weak': 2}


def _compute_entry_context(feat: np.ndarray, direction: str) -> dict:
    """Extract entry-time context fields from the feature vector.

    These are frozen at entry and passed to ledger.add_position so the
    Position carries the same entry snapshots _open_trade used to set.
    """
    entry_h1_z = abs(float(feat[_1H_Z_IDX]))
    v5 = float(feat[_5M_VELOCITY_IDX])
    v5_aligned = ((direction == 'long' and v5 > 0) or
                  (direction == 'short' and v5 < 0))

    if entry_h1_z > 2.0:
        ride_exit_bars = RIDE_EXIT_BARS_TIERS['strong']
    elif entry_h1_z > 1.5:
        ride_exit_bars = RIDE_EXIT_BARS_TIERS['medium']
    else:
        ride_exit_bars = RIDE_EXIT_BARS_TIERS['weak']

    return {
        'entry_abs_z': abs(float(feat[_1M_OFFSET + _Z])),
        'entry_velocity': abs(float(feat[_1M_VELOCITY_IDX])),
        'entry_h1_z': entry_h1_z,
        'entry_vol_rel': float(feat[_1M_VOL_REL_IDX]),
        'v5_aligned': v5_aligned,
        'ride_exit_bars': ride_exit_bars,
    }


def apply_decision(ledger: Ledger, batch: DecisionBatch,
                   feat: np.ndarray, price: float, ts: float):
    """Apply a DecisionBatch to the ledger. Shared logic for sim.

    Order of operations (spec §sim_executor):
      1. Apply counter updates to every open position.
      2. Process per-position exits (from position_decisions).
      3. Process negative_exit (close primary + all chains).
      4. Process chain_entry (scale-in, same direction).
      5. Process entry (fresh primary, only if flat after exits).
    """
    # 1. Counter updates
    for pd in batch.position_decisions:
        ledger.apply_position_decision(pd)

    # 2. Per-position exits
    for pd in batch.position_decisions:
        if pd.exit_reason is not None:
            pos = ledger.get(pd.contract_id)
            if pos is not None:
                ledger.remove_position(
                    pd.contract_id, price, ts, pd.exit_reason, feat)

    # 3. Negative exit — close primary and all chains
    if batch.negative_exit is not None:
        reason = batch.negative_exit.reason
        # Close chains first, then primary (same order as _flatten_all_chains)
        for chain in ledger.chains:
            ledger.remove_position(chain.contract_id, price, ts,
                                   f'chain_{reason}', feat)
        if ledger.primary is not None:
            ledger.remove_position(
                ledger.primary.contract_id, price, ts, reason, feat)

    # 4. Chain entry (only if still in position after exits)
    if batch.chain_entry is not None and not ledger.is_flat:
        sig = batch.chain_entry
        ctx = _compute_entry_context(feat, sig.direction)
        ledger.add_position(
            direction=sig.direction,
            entry_price=price,
            entry_ts=ts,
            entry_tier=sig.tier,
            entry_features=feat.copy(),
            is_chain=True,
            cnn_flipped=sig.cnn_flipped,
            **ctx,
        )

    # 5. Fresh entry (only if flat)
    if batch.entry is not None and ledger.is_flat:
        sig = batch.entry
        ctx = _compute_entry_context(feat, sig.direction)
        ledger.add_position(
            direction=sig.direction,
            entry_price=price,
            entry_ts=ts,
            entry_tier=sig.tier,
            entry_features=feat.copy(),
            is_chain=False,
            cnn_flipped=sig.cnn_flipped,
            **ctx,
        )


def run(ledger: Ledger, engine, bars: Iterator[Dict],
        eod_close: bool = False) -> List[dict]:
    """Drive a full simulation over a bar sequence.

    Args:
        ledger:  A fresh (or continuing) Ledger instance.
        engine:  A BlendedEngine (or any object with evaluate(state)->DecisionBatch).
        bars:    Iterator of state dicts from FeatureTicker. Each dict must
                 have 'features', 'price', 'timestamp'. The key 'features'
                 is mapped to 'features_79d' for evaluate().
        eod_close: If True, call force_close after the last bar. Matches the
                 training pattern where every day ends with force_close().

    Returns:
        List of closed-trade dicts from the ledger (same format as
        BlendedEngine.trades).
    """
    trades_before = len(ledger.closed_trades)
    last_feat, last_price, last_ts = None, 0.0, 0.0

    for state in bars:
        feat = state['features']
        price = state['price']
        ts = state['timestamp']
        volume = state.get('bar_data', {}).get('volume', 0.0) if state.get('bar_data') else 0.0

        last_feat, last_price, last_ts = feat, price, ts

        # 1. Advance per-bar state on every open position
        ledger.update_bar(feat, price, ts, current_volume=volume)

        # 2. Engine evaluates: pure function of (features, positions)
        was_in_pos = not ledger.is_flat
        eval_state = {
            'features_79d': feat,
            'price': price,
            'timestamp': ts,
            'positions': ledger.snapshot(),
        }
        batch = engine.evaluate(eval_state)

        # 3. Apply decisions to the ledger
        apply_decision(ledger, batch, feat, price, ts)

        # 4. Fast re-evaluation: if exits made us flat this bar, re-evaluate
        #    immediately so the engine can enter on the same bar. The old
        #    on_state() path did exit+entry sequentially in one call; this
        #    preserves that behavior without giving the engine write access.
        if was_in_pos and ledger.is_flat:
            eval_state['positions'] = ledger.snapshot()  # now flat
            batch2 = engine.evaluate(eval_state)
            if batch2.entry is not None:
                apply_decision(ledger, batch2, feat, price, ts)

    if eod_close and last_feat is not None:
        force_close(ledger, last_price, last_ts, last_feat)

    return ledger.closed_trades[trades_before:]


def adapt_trades(trades: List[dict]) -> List[dict]:
    """Add backward-compatible field aliases to trade records.

    The ledger produces records with 'entry_features' / 'exit_features'.
    The downstream pipeline (regret.py, cnn_entry.py, cnn_flip.py,
    cnn_exit.py) reads 'entry_79d' / 'exit_79d' / 'v5_aligned'.

    This shim bridges the gap so the pipeline works unchanged.
    """
    for t in trades:
        if 'entry_79d' not in t and 'entry_features' in t:
            t['entry_79d'] = t['entry_features']
        if 'exit_79d' not in t and 'exit_features' in t:
            t['exit_79d'] = t['exit_features']
        if 'v5_aligned' not in t:
            t['v5_aligned'] = True  # safe default
        if 'approach' not in t:
            t['approach'] = []
        if 'path' not in t:
            t['path'] = []
        if 'exit_price' not in t:
            # Should already be there from ledger.remove_position
            pass
    return trades


def force_close(ledger: Ledger, price: float, ts: float,
                feat: np.ndarray, reason: str = 'end_of_day'):
    """Close all open positions at end-of-day (or session end).

    Mirrors BlendedEngine.force_close() but through the ledger.
    """
    # Close chains first, then primary
    for chain in ledger.chains:
        ledger.remove_position(chain.contract_id, price, ts,
                               f'chain_{reason}', feat)
    if ledger.primary is not None:
        ledger.remove_position(
            ledger.primary.contract_id, price, ts, reason, feat)
