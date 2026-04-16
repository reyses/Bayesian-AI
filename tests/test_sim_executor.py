"""
Unit tests for core/sim_executor.py — Phase 3 bar-loop driver.

These tests verify that the sim executor correctly glues together:
  ledger.update_bar → engine.evaluate → apply_decision → ledger mutations

They do NOT attempt bit-exact parity with the old on_state() path (the new
path fixes the silent-flip bug — see spec §Phase 3). Instead they verify:

  1. Entry from flat: FeatureTicker-style bars → ledger opens a position.
  2. Exit on physics: a known-exiting scenario closes the position.
  3. Chain entry: a second tier fires while in position → ledger adds chain.
  4. Negative exit: opposing stronger tier → closes primary + chains.
  5. force_close: end-of-day flushes all positions.
  6. Counter persistence: counters survive across bars via apply_position_decision.
  7. Trade record format: closed trades have the fields the training pipeline reads.

Run with:  pytest tests/test_sim_executor.py -v
"""
import numpy as np
import pytest

from core.ledger import Ledger
from core.engine_signals import (
    DecisionBatch, EntrySignal, ExitSignal, PositionDecision, PositionsView,
)
from core import sim_executor
from core.sim_executor import apply_decision, force_close, _compute_entry_context
from core.features import N_FEATURES

# Reuse the feature-index constants from the engine (for building test features)
from training.nightmare_blended import (
    BlendedEngine,
    _1M_OFFSET, _Z, _VR,
    _1M_VELOCITY_IDX, _1H_VELOCITY_IDX, _1H_Z_IDX,
    _1M_P_CENTER_IDX, _5M_WICK_IDX, _15M_WICK_IDX,
    _5M_VELOCITY_IDX, _1M_VOL_REL_IDX,
    P_CENTER_EXIT, P_CENTER_EXIT_BARS_CASCADE,
    RIDE_EXIT_BARS_TIERS,
)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_features(**overrides) -> np.ndarray:
    """Build a zero feature vector with optional field overrides."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    field_map = {
        'z_1m':        _1M_OFFSET + _Z,
        'vr_1m':       _1M_OFFSET + _VR,
        'velocity_1m': _1M_VELOCITY_IDX,
        'p_center_1m': _1M_P_CENTER_IDX,
        'h1_z':        _1H_Z_IDX,
        'h1_vel':      _1H_VELOCITY_IDX,
        'wick_5m':     _5M_WICK_IDX,
        'wick_15m':    _15M_WICK_IDX,
        'v5_vel':      _5M_VELOCITY_IDX,
        'vol_rel_1m':  _1M_VOL_REL_IDX,
    }
    for k, v in overrides.items():
        if k not in field_map:
            raise KeyError(f'unknown override: {k}')
        feat[field_map[k]] = v
    return feat


# 1m boundary timestamps (ts % 60 < 5)
TS_1M_0 = 1776000000    # bar 0
TS_1M_1 = 1776000060    # bar 1 (1 min later)
TS_1M_2 = 1776000120    # bar 2
TS_1M_3 = 1776000180    # bar 3
TS_1M_4 = 1776000240    # bar 4

PRICE = 26000.0


def make_bar(ts, feat, price=PRICE, volume=0.0):
    """Build a FeatureTicker-style state dict."""
    return {
        'timestamp': ts,
        'price': price,
        'features': feat,
        'bar_idx': 0,
        'bar_data': {'volume': volume} if volume else None,
    }


@pytest.fixture
def engine():
    """A fresh BlendedEngine with CNNs disabled."""
    return BlendedEngine(use_cnn=False)


@pytest.fixture
def ledger():
    return Ledger()


# ═══════════════════════════════════════════════════════════════════════
# Test: _compute_entry_context
# ═══════════════════════════════════════════════════════════════════════

class TestEntryContext:
    def test_entry_abs_z(self):
        feat = make_features(z_1m=-2.5)
        ctx = _compute_entry_context(feat, 'long')
        assert ctx['entry_abs_z'] == pytest.approx(2.5)

    def test_entry_velocity(self):
        feat = make_features(velocity_1m=-80.0)
        ctx = _compute_entry_context(feat, 'short')
        assert ctx['entry_velocity'] == pytest.approx(80.0)

    def test_ride_exit_bars_strong(self):
        feat = make_features(h1_z=2.5)
        ctx = _compute_entry_context(feat, 'long')
        assert ctx['ride_exit_bars'] == RIDE_EXIT_BARS_TIERS['strong']

    def test_ride_exit_bars_medium(self):
        feat = make_features(h1_z=1.7)
        ctx = _compute_entry_context(feat, 'long')
        assert ctx['ride_exit_bars'] == RIDE_EXIT_BARS_TIERS['medium']

    def test_ride_exit_bars_weak(self):
        feat = make_features(h1_z=0.5)
        ctx = _compute_entry_context(feat, 'long')
        assert ctx['ride_exit_bars'] == RIDE_EXIT_BARS_TIERS['weak']

    def test_v5_aligned_long(self):
        feat = make_features(v5_vel=5.0)
        ctx = _compute_entry_context(feat, 'long')
        assert ctx['v5_aligned'] is True

    def test_v5_not_aligned_long(self):
        feat = make_features(v5_vel=-5.0)
        ctx = _compute_entry_context(feat, 'long')
        assert ctx['v5_aligned'] is False


# ═══════════════════════════════════════════════════════════════════════
# Test: apply_decision
# ═══════════════════════════════════════════════════════════════════════

class TestApplyDecision:
    def test_entry_from_flat(self, ledger):
        """A batch with an entry signal opens a primary position."""
        feat = make_features(z_1m=-2.5, vr_1m=0.5)
        batch = DecisionBatch(
            entry=EntrySignal(tier='FADE_CALM', direction='long', cnn_flipped=False),
        )
        apply_decision(ledger, batch, feat, PRICE, TS_1M_0)
        assert not ledger.is_flat
        assert ledger.primary.direction == 'long'
        assert ledger.primary.entry_tier == 'FADE_CALM'
        assert ledger.primary.entry_price == PRICE

    def test_entry_ignored_when_not_flat(self, ledger):
        """A batch with entry is ignored if the ledger already has a primary."""
        feat = make_features(z_1m=-2.5)
        ledger.add_position('long', PRICE, TS_1M_0, 'FADE_CALM', feat)
        batch = DecisionBatch(
            entry=EntrySignal(tier='CASCADE', direction='short', cnn_flipped=False),
        )
        apply_decision(ledger, batch, feat, PRICE, TS_1M_1)
        # Still the original position
        assert ledger.primary.entry_tier == 'FADE_CALM'

    def test_exit_closes_position(self, ledger):
        """A position decision with exit_reason closes the position."""
        feat = make_features(z_1m=-2.5)
        pos = ledger.add_position('long', PRICE, TS_1M_0, 'CASCADE', feat)
        batch = DecisionBatch(
            position_decisions=[
                PositionDecision(contract_id=pos.contract_id,
                                 exit_reason='cascade_center'),
            ],
        )
        apply_decision(ledger, batch, feat, PRICE + 10, TS_1M_1)
        assert ledger.is_flat
        assert len(ledger.closed_trades) == 1
        assert ledger.closed_trades[0]['exit_reason'] == 'cascade_center'
        assert ledger.closed_trades[0]['pnl'] > 0

    def test_counter_update_persists(self, ledger):
        """Counter updates from position_decisions are applied to the position."""
        feat = make_features(z_1m=-2.5)
        pos = ledger.add_position('long', PRICE, TS_1M_0, 'CASCADE', feat)
        batch = DecisionBatch(
            position_decisions=[
                PositionDecision(contract_id=pos.contract_id,
                                 tier_p_center_bars=2),
            ],
        )
        apply_decision(ledger, batch, feat, PRICE, TS_1M_1)
        assert ledger.primary.tier_p_center_bars == 2

    def test_chain_entry(self, ledger):
        """A chain_entry signal adds a chain to an existing primary."""
        feat = make_features(z_1m=-2.5)
        ledger.add_position('long', PRICE, TS_1M_0, 'KILL_SHOT', feat)
        batch = DecisionBatch(
            chain_entry=EntrySignal(tier='FADE_CALM', direction='long',
                                    cnn_flipped=False),
        )
        apply_decision(ledger, batch, feat, PRICE + 5, TS_1M_1)
        assert len(ledger.chains) == 1
        assert ledger.chains[0].entry_tier == 'FADE_CALM'
        assert ledger.chains[0].is_chain is True

    def test_chain_ignored_when_flat(self, ledger):
        """A chain_entry is dropped if the ledger became flat from exits."""
        feat = make_features(z_1m=-2.5)
        # No position — chain entry should be ignored
        batch = DecisionBatch(
            chain_entry=EntrySignal(tier='FADE_CALM', direction='long',
                                    cnn_flipped=False),
        )
        apply_decision(ledger, batch, feat, PRICE, TS_1M_0)
        assert ledger.is_flat

    def test_negative_exit_closes_primary_and_chains(self, ledger):
        """Negative exit closes primary + all chains."""
        feat = make_features(z_1m=-2.5)
        primary = ledger.add_position('long', PRICE, TS_1M_0, 'FADE_CALM', feat)
        chain = ledger.add_position('long', PRICE, TS_1M_1, 'KILL_SHOT', feat,
                                    is_chain=True)
        batch = DecisionBatch(
            # Counter updates for both positions (no exits from physics)
            position_decisions=[
                PositionDecision(contract_id=primary.contract_id),
                PositionDecision(contract_id=chain.contract_id),
            ],
            negative_exit=ExitSignal(contract_id=primary.contract_id,
                                     reason='negative_exit_FREIGHT_TRAIN'),
        )
        apply_decision(ledger, batch, feat, PRICE - 5, TS_1M_2)
        assert ledger.is_flat
        assert len(ledger.closed_trades) == 2
        reasons = {t['exit_reason'] for t in ledger.closed_trades}
        assert 'negative_exit_FREIGHT_TRAIN' in reasons
        assert 'chain_negative_exit_FREIGHT_TRAIN' in reasons

    def test_exit_then_entry_same_bar(self, ledger):
        """If an exit fires, a new entry in the same batch can open (if flat)."""
        feat = make_features(z_1m=-2.5)
        pos = ledger.add_position('long', PRICE, TS_1M_0, 'CASCADE', feat)
        batch = DecisionBatch(
            position_decisions=[
                PositionDecision(contract_id=pos.contract_id,
                                 exit_reason='cascade_center'),
            ],
            entry=EntrySignal(tier='FADE_CALM', direction='short',
                              cnn_flipped=False),
        )
        apply_decision(ledger, batch, feat, PRICE, TS_1M_1)
        # Old position closed, new position opened
        assert not ledger.is_flat
        assert ledger.primary.entry_tier == 'FADE_CALM'
        assert ledger.primary.direction == 'short'
        assert len(ledger.closed_trades) == 1


# ═══════════════════════════════════════════════════════════════════════
# Test: force_close
# ═══════════════════════════════════════════════════════════════════════

class TestForceClose:
    def test_force_close_primary_only(self, ledger):
        feat = make_features(z_1m=-2.5)
        ledger.add_position('long', PRICE, TS_1M_0, 'FADE_CALM', feat)
        force_close(ledger, PRICE + 10, TS_1M_4, feat, reason='end_of_day')
        assert ledger.is_flat
        assert len(ledger.closed_trades) == 1
        assert ledger.closed_trades[0]['exit_reason'] == 'end_of_day'

    def test_force_close_primary_and_chains(self, ledger):
        feat = make_features(z_1m=-2.5)
        ledger.add_position('long', PRICE, TS_1M_0, 'FADE_CALM', feat)
        ledger.add_position('long', PRICE, TS_1M_1, 'KILL_SHOT', feat,
                            is_chain=True)
        force_close(ledger, PRICE, TS_1M_4, feat)
        assert ledger.is_flat
        assert len(ledger.closed_trades) == 2
        reasons = [t['exit_reason'] for t in ledger.closed_trades]
        assert reasons[0] == 'chain_end_of_day'  # chain closed first
        assert reasons[1] == 'end_of_day'

    def test_force_close_when_flat_is_noop(self, ledger):
        feat = make_features()
        force_close(ledger, PRICE, TS_1M_0, feat)
        assert len(ledger.closed_trades) == 0


# ═══════════════════════════════════════════════════════════════════════
# Test: full run() integration
# ═══════════════════════════════════════════════════════════════════════

class TestRun:
    def test_run_single_trade_lifecycle(self, engine, ledger):
        """Walk a sequence that enters CASCADE and exits on p_center."""
        # Bar 0: NMP conditions fire → CASCADE entry (z > ROCHE, vr < 1, wick)
        entry_feat = make_features(
            z_1m=-2.5, vr_1m=0.5,
            wick_5m=0.90, wick_15m=0.85,
            h1_z=-1.5,  # aligned for CASCADE (direction=long, h1_z < -1.0)
        )
        bars = [make_bar(TS_1M_0, entry_feat)]

        # Bars 1-3: p_center > 0.60 sustained for 3 bars → cascade_center exit.
        # Exit features suppress re-entry (higher TFs opposing for direction=long:
        # v5<-3, h1_vel<-3, h1_z>1.5 suppresses RIDE_AGAINST, v5 abs>10 kills FADE_AGAINST).
        for i in range(1, 4):
            exit_feat = make_features(
                z_1m=-0.3, vr_1m=0.5,
                p_center_1m=0.75,
                v5_vel=-15.0, h1_vel=-5.0, h1_z=2.0,
            )
            bars.append(make_bar(TS_1M_0 + 60 * i, exit_feat))

        trades = sim_executor.run(ledger, engine, iter(bars))
        assert len(trades) == 1
        assert trades[0]['entry_tier'] == 'CASCADE'
        assert trades[0]['exit_reason'] == 'cascade_center'
        assert trades[0]['dir'] == 'long'
        assert ledger.is_flat

    def test_run_no_entry_when_all_tiers_suppressed(self, engine, ledger):
        """Bars where _classify_full_tier returns (None, None, False).

        Getting no entry requires threading the needle through all tier checks:
        - z=0.5 (direction=short) + no wick → no CASCADE/KILL_SHOT
        - h1_z=-2.0 → h1_against_fade=True → suppresses RIDE_AGAINST
        - v5_vel=15.0 → abs > 10 → suppresses FADE_AGAINST
        - v5_vel>3 + h1_vel>3 → higher_tf_opposing=True → suppresses FADE_CALM
        """
        opposing = make_features(z_1m=0.5, vr_1m=0.8, v5_vel=15.0,
                                 h1_vel=5.0, h1_z=-2.0)
        bars = [make_bar(TS_1M_0 + 60 * i, opposing) for i in range(5)]
        trades = sim_executor.run(ledger, engine, iter(bars))
        assert len(trades) == 0
        assert ledger.is_flat

    def test_run_force_close_after_bars(self, engine, ledger):
        """Position left open after bars → force_close flushes it."""
        entry_feat = make_features(
            z_1m=-2.5, vr_1m=0.5,
            wick_5m=0.90, wick_15m=0.85,
            h1_z=-1.5,
        )
        bars = [make_bar(TS_1M_0, entry_feat)]
        # Only one bar — no exit conditions met
        trades = sim_executor.run(ledger, engine, iter(bars))
        assert len(trades) == 0
        assert not ledger.is_flat

        # force_close to end the day
        force_close(ledger, PRICE, TS_1M_4, entry_feat)
        assert ledger.is_flat
        assert len(ledger.closed_trades) == 1
        assert ledger.closed_trades[0]['exit_reason'] == 'end_of_day'

    def test_trade_record_has_required_fields(self, engine, ledger):
        """Closed trade records have all fields the training pipeline reads."""
        entry_feat = make_features(
            z_1m=-2.5, vr_1m=0.5,
            wick_5m=0.90, wick_15m=0.85,
            h1_z=-1.5,
        )
        bars = [make_bar(TS_1M_0, entry_feat)]
        for i in range(1, 4):
            exit_feat = make_features(z_1m=-0.3, vr_1m=0.5, p_center_1m=0.75,
                                      v5_vel=-15.0, h1_vel=-5.0, h1_z=2.0)
            bars.append(make_bar(TS_1M_0 + 60 * i, exit_feat))

        sim_executor.run(ledger, engine, iter(bars))
        assert len(ledger.closed_trades) >= 1

        required_keys = {
            'contract_id', 'dir', 'entry_price', 'exit_price',
            'entry_ts', 'exit_ts', 'pnl', 'held', 'peak',
            'entry_tier', 'exit_reason', 'cnn_flipped', 'is_chain',
            'entry_features', 'exit_features',
        }
        actual_keys = set(ledger.closed_trades[0].keys())
        missing = required_keys - actual_keys
        assert not missing, f'missing keys in trade record: {missing}'

    def test_run_accumulates_across_multiple_calls(self, engine, ledger):
        """Calling run() multiple times appends to the same ledger.

        Note: exit-phase features suppress higher-TF opposing so the engine
        doesn't open chain contracts mid-trade. force_close between days
        ensures a clean slate (mirrors the training loop pattern).
        """
        # Day 1: enter + exit
        entry_feat = make_features(
            z_1m=-2.5, vr_1m=0.5,
            wick_5m=0.90, wick_15m=0.85, h1_z=-1.5,
        )
        # Exit features: suppress chain entries AND re-entry after exit
        # (direction=long: v5<-3, h1_vel<-3 → opposing; h1_z>1.5 kills RIDE_AGAINST;
        # abs(v5)>10 kills FADE_AGAINST)
        exit_feat = make_features(z_1m=-0.3, vr_1m=0.5, p_center_1m=0.75,
                                  v5_vel=-15.0, h1_vel=-5.0, h1_z=2.0)
        bars_day1 = [make_bar(TS_1M_0, entry_feat)]
        for i in range(1, 4):
            bars_day1.append(make_bar(TS_1M_0 + 60 * i, exit_feat))

        trades1 = sim_executor.run(ledger, engine, iter(bars_day1))
        force_close(ledger, PRICE, TS_1M_0 + 300, exit_feat)
        assert len(trades1) == 1

        # Day 2: same thing
        ts_day2 = TS_1M_0 + 86400  # next day
        bars_day2 = [make_bar(ts_day2, entry_feat)]
        for i in range(1, 4):
            bars_day2.append(make_bar(ts_day2 + 60 * i, exit_feat))

        trades2 = sim_executor.run(ledger, engine, iter(bars_day2))
        force_close(ledger, PRICE, ts_day2 + 300, exit_feat)
        assert len(trades2) == 1

        # Total in ledger = 2 trades (ignoring any force_close empties)
        primary_trades = [t for t in ledger.closed_trades
                          if not t['is_chain']]
        assert len(primary_trades) == 2

    def test_counter_persistence_across_bars(self, engine, ledger):
        """Counters updated by evaluate() survive to the next bar's snapshot."""
        # Enter a CASCADE trade
        entry_feat = make_features(
            z_1m=-2.5, vr_1m=0.5,
            wick_5m=0.90, wick_15m=0.85, h1_z=-1.5,
        )
        bars = [make_bar(TS_1M_0, entry_feat)]
        sim_executor.run(ledger, engine, iter(bars))
        assert not ledger.is_flat

        # Bar 1: p_center above threshold → counter should be 1
        feat1 = make_features(z_1m=-1.5, vr_1m=0.5, p_center_1m=0.75)
        bars1 = [make_bar(TS_1M_1, feat1)]
        sim_executor.run(ledger, engine, iter(bars1))
        assert ledger.primary.tier_p_center_bars == 1

        # Bar 2: p_center above threshold → counter should be 2
        feat2 = make_features(z_1m=-1.5, vr_1m=0.5, p_center_1m=0.75)
        bars2 = [make_bar(TS_1M_2, feat2)]
        sim_executor.run(ledger, engine, iter(bars2))
        assert ledger.primary.tier_p_center_bars == 2

        # Bar 3: counter hits 3 → CASCADE exit fires
        # Suppress re-entry after exit (higher TFs opposing for direction=long)
        feat3 = make_features(z_1m=-1.0, vr_1m=0.5, p_center_1m=0.75,
                              v5_vel=-15.0, h1_vel=-5.0, h1_z=2.0)
        bars3 = [make_bar(TS_1M_3, feat3)]
        trades = sim_executor.run(ledger, engine, iter(bars3))
        assert len(trades) == 1
        assert trades[0]['exit_reason'] == 'cascade_center'
        assert ledger.is_flat
