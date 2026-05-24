"""
Unit tests for BlendedEngine.evaluate() — Phase 2 signal contract.

These tests lock the engine→ledger signal contract. They don't try to
reproduce the existing on_state() behavior bit-for-bit — that's Phase 3's
job. They verify that:

  1. evaluate() does not mutate self.
  2. evaluate() returns a DecisionBatch with the expected shape for each
     well-defined scenario (empty ledger, entry, chain, exit, negative exit).
  3. Per-position decisions carry counter updates even when no exit fires.
  4. Counter updates are correctly computed for each tier.

Run with:  pytest tests/test_engine_evaluate.py -v
"""
import numpy as np
import pytest

from training.nightmare_blended import BlendedEngine
from core_v2.engine_signals import (
    DecisionBatch, EntrySignal, ExitSignal, PositionDecision,
    PositionView, PositionsView,
)
from core_v2.ledger import Ledger
from core_v2.features import N_FEATURES, FEATURE_NAMES

_1M_Z_IDX = FEATURE_NAMES.index('L3_1m_z_se_15')
_1M_VELOCITY_IDX = FEATURE_NAMES.index('L2_1m_price_velocity_15')
_1H_Z_IDX = FEATURE_NAMES.index('L3_1h_z_se_12')
_1H_VELOCITY_IDX = FEATURE_NAMES.index('L2_1h_price_velocity_12')
_5M_VELOCITY_IDX = FEATURE_NAMES.index('L2_5m_price_velocity_9')
_1M_BODY_IDX = FEATURE_NAMES.index('L1_1m_body')
_1M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_1m_bar_range')
_5M_BODY_IDX = FEATURE_NAMES.index('L1_5m_body')
_5M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_5m_bar_range')
_15M_BODY_IDX = FEATURE_NAMES.index('L1_15m_body')
_15M_BAR_RANGE_IDX = FEATURE_NAMES.index('L1_15m_bar_range')


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def make_features(**overrides):
    """Build a zero feature vector with optional field overrides."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    
    # Safe defaults to prevent wick divide-by-zero
    feat[_1M_BAR_RANGE_IDX] = 1.0
    feat[_5M_BAR_RANGE_IDX] = 1.0
    feat[_15M_BAR_RANGE_IDX] = 1.0
    feat[_1M_BODY_IDX] = 1.0
    feat[_5M_BODY_IDX] = 1.0
    feat[_15M_BODY_IDX] = 1.0

    if 'velocity_1m' in overrides: feat[_1M_VELOCITY_IDX] = overrides['velocity_1m']
    if 'h1_z' in overrides: feat[_1H_Z_IDX] = overrides['h1_z']
    if 'h1_vel' in overrides: feat[_1H_VELOCITY_IDX] = overrides['h1_vel']
    if 'v5_vel' in overrides: feat[_5M_VELOCITY_IDX] = overrides['v5_vel']
    
    if 'p_center_1m' in overrides:
        import math
        pc = overrides['p_center_1m']
        z_mag = math.sqrt(-2 * math.log(pc)) if pc > 0 else 5.0
        if 'z_1m' in overrides:
            feat[_1M_Z_IDX] = -z_mag if overrides['z_1m'] < 0 else z_mag
        else:
            feat[_1M_Z_IDX] = z_mag
    elif 'z_1m' in overrides:
        feat[_1M_Z_IDX] = overrides['z_1m']
        
    if 'wick_5m' in overrides:
        w = overrides['wick_5m']
        feat[_5M_BAR_RANGE_IDX] = 1.0
        feat[_5M_BODY_IDX] = max(1.0 - w, 0.0)

    if 'wick_15m' in overrides:
        w = overrides['wick_15m']
        feat[_15M_BAR_RANGE_IDX] = 1.0
        feat[_15M_BODY_IDX] = max(1.0 - w, 0.0)

    return {'feat': feat, 'vr': overrides.get('vr_1m', 1.0)}


def state_for(feat_bundle, price: float, ts: float, positions: PositionsView = None):
    """Build the state dict evaluate() expects."""
    if isinstance(feat_bundle, dict):
        feat = feat_bundle['feat']
        vr = feat_bundle['vr']
    else:
        feat = feat_bundle
        vr = 1.0
        
    return {
        'features_79d': feat,
        'price': price,
        'timestamp': ts,
        'variance_ratio': vr,
        'positions': positions if positions is not None else PositionsView(),
    }


# One-minute boundaries: ts where ts % 60 < 5
TS_1M = 1776000000   # 1776000000 % 60 == 0 → is_1m True
TS_5S = 1776000005   # 1776000005 % 60 == 5 → is_1m False


@pytest.fixture
def engine():
    """A fresh BlendedEngine with CNNs disabled (Phase 2 is physics-only)."""
    eng = BlendedEngine()
    eng.use_cnn = False
    eng.skip_thin_market = False   # don't let Sunday gate interfere with tests
    return eng


# ═══════════════════════════════════════════════════════════════════════
# evaluate() on a flat ledger
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluateFlat:
    # Note: FADE_CALM is the default tier — it fires on every 1m boundary
    # while flat unless the classifier returns None (which only happens when
    # v5_vel and h1_vel both strongly oppose the fade direction, and that
    # condition typically triggers RIDE_AGAINST first anyway). So "flat on
    # a 1m boundary with no entry" is actually a rare state in the existing
    # engine — these tests lock the observed behavior instead.

    def test_flat_1m_empty_features_fires_default_fade_calm(self, engine):
        """Zero features → direction=long (z<=0 → long), FADE_CALM by default."""
        feat = make_features()
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M))
        assert isinstance(batch, DecisionBatch)
        assert batch.entry is not None
        assert batch.entry.tier == 'FADE_CALM'
        assert batch.entry.direction == 'long'
        # Flat ledger = no position decisions
        assert batch.position_decisions == []
        # Empty ledger still means chain_entry is None
        assert batch.chain_entry is None
        assert batch.negative_exit is None

    def test_flat_not_1m_boundary_no_entry(self, engine):
        """Even with a valid setup, no entry fires on non-1m bars."""
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_5S))
        assert batch.entry is None
        assert batch.chain_entry is None
        assert not batch.has_any

    def test_flat_1m_nmp_fires_fade_calm_short(self, engine):
        """z > 0 → fade direction = short. FADE_CALM by default."""
        feat = make_features(z_1m=2.5, vr_1m=0.5, h1_z=0.3)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M))
        assert batch.entry is not None
        assert batch.entry.direction == 'short'
        assert batch.entry.tier == 'FADE_CALM'

    def test_flat_low_price_gated(self, engine):
        """price <= 100 → entry skipped (matches on_state's price guard)."""
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 50.0, TS_1M))
        assert batch.entry is None

    def test_flat_ledger_never_produces_chain_entry(self, engine):
        """Chain entries require a primary to exist. Flat ledger → no chain."""
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M))
        assert batch.chain_entry is None


# ═══════════════════════════════════════════════════════════════════════
# evaluate() with an open primary position
# ═══════════════════════════════════════════════════════════════════════

class TestEvaluateInPosition:

    def test_primary_open_emits_one_position_decision(self, engine):
        """Every open position produces exactly one PositionDecision."""
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        feat = make_features(z_1m=-1.5)
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        assert len(batch.position_decisions) == 1
        pd = batch.position_decisions[0]
        assert pd.contract_id == 'P001'
        # No exit this bar: tier_p_center_bars would only increment if abs(z) < 0.6
        # is high. We passed default p_center=0, so counter stays at 0.
        assert pd.exit_reason is None
        assert pd.tier_p_center_bars == 0

    def test_primary_plus_chains_all_get_decisions(self, engine):
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 120, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        led.add_position('long', 24990.0, TS_1M - 60, 'FADE_CALM',
                         make_features(z_1m=-0.5), is_chain=True,
                         entry_abs_z=1.0)
        feat = make_features(z_1m=-0.5)
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        # 1 primary + 1 chain = 2 decisions
        assert len(batch.position_decisions) == 2
        ids = {pd.contract_id for pd in batch.position_decisions}
        assert ids == {'P001', 'C002'}

    def test_cascade_p_center_counter_increments(self, engine):
        """CASCADE tier: p_center > 0.60 → tier_p_center_bars increments."""
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        # High p_center triggers counter increment
        feat = make_features(z_1m=-0.5, )
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        pd = batch.position_decisions[0]
        assert pd.tier_p_center_bars == 1   # incremented from 0
        assert pd.exit_reason is None       # not enough bars yet

    def test_cascade_p_center_fires_exit_after_n_bars(self, engine):
        """CASCADE needs 3 consecutive bars of high p_center to exit."""
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        # Simulate we've already seen 2 consecutive high-p_center bars
        led.apply_position_decision(PositionDecision(
            contract_id='P001',
            tier_p_center_bars=2,
        ))
        feat = make_features(z_1m=-0.5, )   # third bar
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        pd = batch.position_decisions[0]
        assert pd.tier_p_center_bars == 3
        assert pd.exit_reason == 'cascade_center'

    def test_cascade_p_center_resets_when_condition_breaks(self, engine):
        """If p_center drops below threshold, counter resets to 0."""
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        led.apply_position_decision(PositionDecision(
            contract_id='P001',
            tier_p_center_bars=2,   # had 2 prior bars
        ))
        feat = make_features(z_1m=-1.5, )   # z escapes center, counter resets
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        assert batch.position_decisions[0].tier_p_center_bars == 0
        assert batch.position_decisions[0].exit_reason is None

    def test_not_1m_bar_no_physics_exit_but_counters_unchanged(self, engine):
        """Physics exits only evaluate on 1m boundaries."""
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        led.apply_position_decision(PositionDecision(
            contract_id='P001',
            tier_p_center_bars=2,
        ))
        feat = make_features(z_1m=-0.5, )
        # NOT a 1m boundary
        batch = engine.evaluate(state_for(feat, 25005.0, TS_5S, led.snapshot()))
        pd = batch.position_decisions[0]
        # Counter should pass through unchanged (no 1m evaluation happened)
        assert pd.tier_p_center_bars == 2
        assert pd.exit_reason is None


# ═══════════════════════════════════════════════════════════════════════
# Negative exit — opposing setup with higher conviction
# ═══════════════════════════════════════════════════════════════════════

class TestNegativeExit:

    def test_no_opposing_setup_no_negative_exit(self, engine):
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        feat = make_features(z_1m=-1.5)   # same fade direction, no opposition
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M, led.snapshot()))
        assert batch.negative_exit is None

    def test_opposing_lower_strength_no_negative_exit(self, engine):
        """Primary is CASCADE (strength 6). FADE_CALM opposing (strength 1)
        is WEAKER → no negative exit."""
        led = Ledger()
        # Primary: long CASCADE
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        # Market flips to positive z (fade direction = short) with no wick
        # → FADE_CALM short would fire at this bar
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M, led.snapshot()))
        assert batch.negative_exit is None   # FADE_CALM(1) < CASCADE(6)


# ═══════════════════════════════════════════════════════════════════════
# Chain entry
# ═══════════════════════════════════════════════════════════════════════

class TestChainEntry:

    def test_flat_no_chain_entry(self, engine):
        """Chain entries require an open primary."""
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M))
        assert batch.chain_entry is None

    def test_same_tier_as_primary_no_chain(self, engine):
        """Chain only fires if the new tier differs from primary."""
        led = Ledger()
        led.add_position('short', 25000.0, TS_1M - 60, 'FADE_CALM',
                         make_features(z_1m=1.5), entry_abs_z=1.5)
        # Same FADE_CALM setup at this bar
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M, led.snapshot()))
        # Classifier still returns FADE_CALM → same tier → no chain
        assert batch.chain_entry is None

    def test_not_1m_no_chain(self, engine):
        led = Ledger()
        led.add_position('short', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=1.5), entry_abs_z=1.5)
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_5S, led.snapshot()))
        assert batch.chain_entry is None


# ═══════════════════════════════════════════════════════════════════════
# Statelessness — the most important property
# ═══════════════════════════════════════════════════════════════════════

class TestStatelessness:

    def test_evaluate_does_not_mutate_self_in_pos(self, engine):
        """evaluate() must not set self.in_pos."""
        assert engine.in_pos is False
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5))
        feat = make_features(z_1m=-0.5)
        engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        assert engine.in_pos is False   # still flat on self

    def test_evaluate_does_not_mutate_self_trades(self, engine):
        """evaluate() must not append to self.trades."""
        initial_trades = list(engine.trades)
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        # Force an exit condition
        led.apply_position_decision(PositionDecision(
            contract_id='P001', tier_p_center_bars=2,
        ))
        feat = make_features(z_1m=-0.5, )
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        assert batch.position_decisions[0].exit_reason == 'cascade_center'
        # But the engine's own trade list is still what it was
        assert engine.trades == initial_trades

    def test_evaluate_does_not_mutate_self_chain_contracts(self, engine):
        """evaluate() must not touch self._chain_contracts."""
        initial_chains = list(engine._chain_contracts)
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        feat = make_features(z_1m=-0.5)
        engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        assert engine._chain_contracts == initial_chains

    def test_repeated_evaluate_same_inputs_same_output(self, engine):
        """Pure function: same (state, ledger_snapshot) → same DecisionBatch."""
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        feat = make_features(z_1m=-0.5, )
        st = state_for(feat, 25005.0, TS_1M, led.snapshot())

        batch1 = engine.evaluate(st)
        batch2 = engine.evaluate(st)
        batch3 = engine.evaluate(st)

        # All three calls produce equivalent decisions
        assert len(batch1.position_decisions) == len(batch2.position_decisions) == len(batch3.position_decisions)
        for pd1, pd2, pd3 in zip(batch1.position_decisions,
                                  batch2.position_decisions,
                                  batch3.position_decisions):
            assert pd1.contract_id == pd2.contract_id == pd3.contract_id
            assert pd1.tier_p_center_bars == pd2.tier_p_center_bars == pd3.tier_p_center_bars
            assert pd1.exit_reason == pd2.exit_reason == pd3.exit_reason


# ═══════════════════════════════════════════════════════════════════════
# DecisionBatch shape / convenience properties
# ═══════════════════════════════════════════════════════════════════════

class TestDecisionBatchShape:

    def test_has_any_false_for_pure_counter_update_non_1m(self, engine):
        """A batch with only counter updates on a non-1m bar is not actionable.

        We use a non-1m bar so the entry/chain classifier doesn't fire.
        Counters pass through unchanged because physics exits only evaluate
        on 1m boundaries, so has_any stays False.
        """
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        feat = make_features(z_1m=-0.5, )
        batch = engine.evaluate(state_for(feat, 25005.0, TS_5S, led.snapshot()))
        # Non-1m bar: counters unchanged, no exit, no entry, no chain
        assert len(batch.position_decisions) == 1
        assert batch.position_decisions[0].exit_reason is None
        assert batch.entry is None
        assert batch.chain_entry is None
        assert not batch.has_any

    def test_has_any_true_for_entry(self, engine):
        feat = make_features(z_1m=2.5, vr_1m=0.5)
        batch = engine.evaluate(state_for(feat, 25000.0, TS_1M))
        assert batch.entry is not None
        assert batch.has_any

    def test_has_any_true_for_exit(self, engine):
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        led.apply_position_decision(PositionDecision(
            contract_id='P001', tier_p_center_bars=2,
        ))
        feat = make_features(z_1m=-0.5, )
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        assert any(pd.exit_reason for pd in batch.position_decisions)
        assert batch.has_any

    def test_exits_accessor_extracts_exit_signals(self, engine):
        led = Ledger()
        led.add_position('long', 25000.0, TS_1M - 60, 'CASCADE',
                         make_features(z_1m=-1.5), entry_abs_z=1.5)
        led.apply_position_decision(PositionDecision(
            contract_id='P001', tier_p_center_bars=2,
        ))
        feat = make_features(z_1m=-0.5, )
        batch = engine.evaluate(state_for(feat, 25005.0, TS_1M, led.snapshot()))
        exits = batch.exits
        assert len(exits) == 1
        assert exits[0].contract_id == 'P001'
        assert exits[0].reason == 'cascade_center'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
