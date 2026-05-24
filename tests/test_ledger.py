"""
Unit tests for core_v2.ledger.Ledger.

Phase 1 of the JULES_ENGINE_DECOUPLE_ORDERS refactor. These tests lock the
ledger contract so Phases 2+ can integrate against a verified foundation.

Run with:  pytest tests/test_ledger.py -v
Or:        python -m pytest tests/test_ledger.py -v
"""
import numpy as np
import pytest

from core_v2.ledger import Ledger, Position, MAX_CHAIN_CONTRACTS
from core_v2.engine_signals import PositionsView
from core_v2.features import N_FEATURES


def _make_features(z_1m: float = 0.0) -> np.ndarray:
    """Build a feature vector with a given 1m z_se (at index 12)."""
    feat = np.zeros(N_FEATURES, dtype=np.float32)
    feat[12] = z_1m   # 1m z_se is index 12 in the canonical layout
    return feat


# ═══════════════════════════════════════════════════════════════════════
# Empty ledger
# ═══════════════════════════════════════════════════════════════════════

class TestEmptyLedger:
    def test_starts_flat(self):
        led = Ledger()
        assert led.is_flat
        assert led.primary is None
        assert led.chains == []
        assert led.closed_trades == []
        assert led.n_contracts == 0

    def test_snapshot_when_flat(self):
        led = Ledger()
        snap = led.snapshot()
        assert isinstance(snap, PositionsView)
        assert snap.is_flat
        assert snap.primary is None
        assert snap.chains == []
        assert snap.all_positions == []

    def test_get_unknown_contract(self):
        led = Ledger()
        assert led.get('P001') is None

    def test_update_bar_when_flat_is_noop(self):
        led = Ledger()
        led.update_bar(_make_features(z_1m=0.5), price=25000.0, ts=1000.0)
        assert led.is_flat   # still nothing


# ═══════════════════════════════════════════════════════════════════════
# Adding positions
# ═══════════════════════════════════════════════════════════════════════

class TestAddPosition:
    def test_add_primary_long(self):
        led = Ledger()
        pos = led.add_position(
            direction='long',
            entry_price=25000.0,
            entry_ts=1000.0,
            entry_tier='CASCADE',
            entry_features=_make_features(z_1m=-0.8),
        )
        assert isinstance(pos, Position)
        assert pos.contract_id == 'P001'
        assert pos.direction == 'long'
        assert pos.entry_price == 25000.0
        assert pos.entry_tier == 'CASCADE'
        assert not pos.is_chain
        assert pos.bars_held == 0
        assert pos.peak_pnl == 0.0
        # Oscillation tracker initialized from entry z.
        # float32 storage → use approx for the derived values.
        assert pos.z_sign == -1.0   # z_1m = -0.8 → sign is negative
        assert pos.z_peak == pytest.approx(0.8, rel=1e-5)
        assert pos.z_trough == pytest.approx(0.8, rel=1e-5)

        assert not led.is_flat
        assert led.primary is pos
        assert led.n_contracts == 1

    def test_add_primary_short(self):
        led = Ledger()
        pos = led.add_position(
            direction='short',
            entry_price=25000.0,
            entry_ts=1000.0,
            entry_tier='KILL_SHOT',
            entry_features=_make_features(z_1m=1.5),
        )
        assert pos.direction == 'short'
        assert pos.z_sign == 1.0   # z_1m > 0 → sign positive
        assert pos.z_peak == pytest.approx(1.5, rel=1e-5)

    def test_rejects_invalid_direction(self):
        led = Ledger()
        with pytest.raises(ValueError, match="direction"):
            led.add_position(
                direction='bogus',
                entry_price=25000.0,
                entry_ts=1000.0,
                entry_tier='CASCADE',
                entry_features=_make_features(),
            )

    def test_rejects_second_primary(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        with pytest.raises(ValueError, match="primary"):
            led.add_position('short', 25100.0, 1060.0, 'KILL_SHOT', _make_features())


class TestAddChain:
    def test_add_chain_same_direction(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        chain = led.add_position(
            direction='long',
            entry_price=24980.0,
            entry_ts=1060.0,
            entry_tier='FADE_CALM',
            entry_features=_make_features(z_1m=-0.5),
            is_chain=True,
        )
        assert chain.contract_id == 'C002'   # second id, 'C' prefix
        assert chain.is_chain
        assert len(led.chains) == 1
        assert led.n_contracts == 2   # primary + 1 chain

    def test_chain_opposite_direction_rejected(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        with pytest.raises(ValueError, match="direction"):
            led.add_position(
                direction='short',
                entry_price=24980.0,
                entry_ts=1060.0,
                entry_tier='KILL_SHOT',
                entry_features=_make_features(),
                is_chain=True,
            )

    def test_chain_without_primary_rejected(self):
        led = Ledger()
        with pytest.raises(ValueError, match="no primary"):
            led.add_position(
                direction='long',
                entry_price=25000.0,
                entry_ts=1000.0,
                entry_tier='FADE_CALM',
                entry_features=_make_features(),
                is_chain=True,
            )

    def test_chain_cap_enforced(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        for i in range(MAX_CHAIN_CONTRACTS):
            led.add_position(
                direction='long',
                entry_price=24980.0 + i,
                entry_ts=1060.0 + i * 60,
                entry_tier='FADE_CALM',
                entry_features=_make_features(),
                is_chain=True,
            )
        # Next chain exceeds the cap
        with pytest.raises(ValueError, match="chain cap"):
            led.add_position(
                direction='long',
                entry_price=24970.0,
                entry_ts=1300.0,
                entry_tier='FADE_CALM',
                entry_features=_make_features(),
                is_chain=True,
            )


# ═══════════════════════════════════════════════════════════════════════
# update_bar — per-bar state advances
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateBar:
    def test_bars_held_advances(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        # 2 1m bars later (120 seconds)
        led.update_bar(_make_features(), price=25005.0, ts=1120.0)
        assert led.primary.bars_held == 2

    def test_peak_pnl_tracks_mfe_long(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        # Price runs to 25020 → +20 ticks * 2 ticks/pt for MNQ... but ledger
        # math is (price - entry) / tick_size * tick_value = 20/0.25*0.50 = 40
        led.update_bar(_make_features(), price=25010.0, ts=1060.0)
        assert led.primary.peak_pnl == 20.0   # +10 points / 0.25 * 0.50
        led.update_bar(_make_features(), price=25020.0, ts=1120.0)
        assert led.primary.peak_pnl == 40.0
        # Price pulls back — peak should not drop
        led.update_bar(_make_features(), price=25015.0, ts=1180.0)
        assert led.primary.peak_pnl == 40.0

    def test_peak_pnl_tracks_mfe_short(self):
        led = Ledger()
        led.add_position('short', 25000.0, 1000.0, 'KILL_SHOT', _make_features())
        led.update_bar(_make_features(), price=24990.0, ts=1060.0)
        assert led.primary.peak_pnl == 20.0   # short profits when price drops
        led.update_bar(_make_features(), price=25005.0, ts=1120.0)
        assert led.primary.peak_pnl == 20.0   # still the high-water mark

    def test_peak_volume_monotonic(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.update_bar(_make_features(), price=25000.0, ts=1060.0, current_volume=100.0)
        assert led.primary.peak_volume == 100.0
        led.update_bar(_make_features(), price=25000.0, ts=1120.0, current_volume=50.0)
        assert led.primary.peak_volume == 100.0   # not decreasing
        led.update_bar(_make_features(), price=25000.0, ts=1180.0, current_volume=200.0)
        assert led.primary.peak_volume == 200.0

    def test_chain_advances_independently(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE',
                         _make_features(), entry_h1_z=1.8)
        # 2 bars later, add a chain
        led.update_bar(_make_features(), price=25005.0, ts=1120.0)
        led.add_position('long', 24990.0, 1120.0, 'FADE_CALM',
                         _make_features(), is_chain=True, entry_h1_z=0.5)

        # Primary has bars_held=2 at chain open time
        assert led.primary.bars_held == 2
        assert led.chains[0].bars_held == 0

        # 3 more bars (180s later)
        led.update_bar(_make_features(), price=25010.0, ts=1300.0)
        assert led.primary.bars_held == 5
        assert led.chains[0].bars_held == 3

    def test_chain_peak_pnl_independent_of_primary(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.add_position('long', 24900.0, 1000.0, 'FADE_CALM',
                         _make_features(), is_chain=True)

        # Price = 25050 → primary MFE = +100 ticks = $100
        # Chain entered at 24900, MFE = +600 ticks = $300
        led.update_bar(_make_features(), price=25050.0, ts=1060.0)
        assert led.primary.peak_pnl == 100.0
        assert led.chains[0].peak_pnl == 300.0

    def test_oscillation_zero_crossings(self):
        led = Ledger()
        # Entry at z=-0.5 (below mean), so initial z_sign = -1
        led.add_position('long', 25000.0, 1000.0, 'CASCADE',
                         _make_features(z_1m=-0.5))
        assert led.primary.z_sign == -1.0
        assert led.primary.zero_crossings == 0

        # z flips to +0.3 → one crossing
        led.update_bar(_make_features(z_1m=0.3), price=25005.0, ts=1060.0)
        assert led.primary.z_sign == 1.0
        assert led.primary.zero_crossings == 1

        # z flips back to -0.4 → second crossing
        led.update_bar(_make_features(z_1m=-0.4), price=24995.0, ts=1120.0)
        assert led.primary.z_sign == -1.0
        assert led.primary.zero_crossings == 2

        # Another update staying negative → no new crossing
        led.update_bar(_make_features(z_1m=-0.6), price=24990.0, ts=1180.0)
        assert led.primary.zero_crossings == 2


# ═══════════════════════════════════════════════════════════════════════
# remove_position — close + history
# ═══════════════════════════════════════════════════════════════════════

class TestRemovePosition:
    def test_close_primary_long_profit(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.update_bar(_make_features(), price=25010.0, ts=1120.0)

        rec = led.remove_position('P001', exit_price=25020.0, exit_ts=1180.0,
                                  reason='giveback_stop')
        assert led.is_flat
        assert led.primary is None
        assert len(led.closed_trades) == 1
        assert rec['dir'] == 'long'
        assert rec['entry_price'] == 25000.0
        assert rec['exit_price'] == 25020.0
        assert rec['pnl'] == 40.0   # +20 points / 0.25 * 0.50
        assert rec['held'] == 2
        assert rec['peak'] == 20.0   # highest MFE during the trade
        assert rec['exit_reason'] == 'giveback_stop'
        assert rec['entry_tier'] == 'CASCADE'

    def test_close_primary_short_loss(self):
        led = Ledger()
        led.add_position('short', 25000.0, 1000.0, 'KILL_SHOT', _make_features())
        rec = led.remove_position('P001', exit_price=25010.0, exit_ts=1060.0,
                                  reason='hard_stop')
        assert rec['pnl'] == -20.0   # short at 25000, exit at 25010 = -10 pts
        assert led.is_flat

    def test_close_chain_leaves_primary_intact(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.add_position('long', 24990.0, 1060.0, 'FADE_CALM',
                         _make_features(), is_chain=True)
        assert led.n_contracts == 2

        # Close the chain only
        led.remove_position('C002', exit_price=25005.0, exit_ts=1120.0,
                            reason='chain_exit')
        assert led.n_contracts == 1
        assert led.primary is not None
        assert led.primary.contract_id == 'P001'   # still there
        assert led.chains == []
        assert len(led.closed_trades) == 1

    def test_close_primary_leaves_chains_intact(self):
        """Matches nightmare_blended.py line 1176 semantics: closing primary
        does NOT flush chains. They exit independently."""
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.add_position('long', 24990.0, 1060.0, 'FADE_CALM',
                         _make_features(), is_chain=True)

        led.remove_position('P001', exit_price=25010.0, exit_ts=1120.0,
                            reason='negative_exit')
        assert led.primary is None
        assert len(led.chains) == 1   # chain still there
        # But is_flat semantics: no primary = flat. Chains without a primary
        # is an invariant violation in principle, but we allow it during the
        # brief window between primary close and chain close. The executor
        # is responsible for cleaning up.
        assert led.is_flat

    def test_unknown_contract_id_raises(self):
        led = Ledger()
        with pytest.raises(KeyError):
            led.remove_position('P999', 25000.0, 1000.0, 'bogus')


# ═══════════════════════════════════════════════════════════════════════
# Snapshots — what the engine sees
# ═══════════════════════════════════════════════════════════════════════

class TestSnapshot:
    def test_snapshot_reflects_current_state(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE',
                         _make_features(z_1m=-0.8))
        led.update_bar(_make_features(z_1m=-0.5), price=25005.0, ts=1120.0)

        snap = led.snapshot()
        assert not snap.is_flat
        assert snap.primary is not None
        assert snap.primary.contract_id == 'P001'
        assert snap.primary.entry_price == 25000.0
        assert snap.primary.bars_held == 2
        assert snap.primary.peak_pnl == 10.0

    def test_snapshot_is_immutable_view(self):
        """Mutating the snapshot must not affect the ledger."""
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())

        snap = led.snapshot()
        # PositionView is a dataclass — try to mutate
        snap.primary.peak_pnl = 999.0
        # But the ledger's underlying Position is not affected
        assert led.primary.peak_pnl == 0.0

    def test_snapshot_includes_all_chains(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.add_position('long', 24995.0, 1060.0, 'FADE_CALM',
                         _make_features(), is_chain=True)
        led.add_position('long', 24990.0, 1120.0, 'KILL_SHOT',
                         _make_features(), is_chain=True)

        snap = led.snapshot()
        assert len(snap.chains) == 2
        assert len(snap.all_positions) == 3


# ═══════════════════════════════════════════════════════════════════════
# Contract ID assignment
# ═══════════════════════════════════════════════════════════════════════

class TestContractIds:
    def test_sequential_ids(self):
        led = Ledger()
        p = led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        c1 = led.add_position('long', 24990.0, 1060.0, 'FADE_CALM',
                              _make_features(), is_chain=True)
        c2 = led.add_position('long', 24985.0, 1120.0, 'KILL_SHOT',
                              _make_features(), is_chain=True)
        assert p.contract_id == 'P001'
        assert c1.contract_id == 'C002'
        assert c2.contract_id == 'C003'

    def test_ids_not_recycled_after_close(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.remove_position('P001', 25010.0, 1060.0, 'exit')
        # Next position gets P002, not P001 again
        pos = led.add_position('long', 25010.0, 1060.0, 'CASCADE', _make_features())
        assert pos.contract_id == 'P002'

    def test_clear_resets_id_counter(self):
        led = Ledger()
        led.add_position('long', 25000.0, 1000.0, 'CASCADE', _make_features())
        led.clear()
        pos = led.add_position('long', 25010.0, 1060.0, 'CASCADE', _make_features())
        assert pos.contract_id == 'P001'   # back to start


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
