"""
Ledger — the single source of truth for position state.

One Ledger class, used by both sim (training) and live. Per-trade memory
(peak_pnl, ride counters, oscillation state, etc.) lives on the Position
dataclass. The engine reads positions through PositionView snapshots but
cannot mutate them. Only the ledger writes.

Responsibilities (single responsibility principle):
  - Hold the list of open positions.
  - Hold the history of closed trades.
  - Update per-bar runtime state (bars_held, peak_pnl, oscillation).
  - Open a position when asked (sim calls directly, live from a fill).
  - Close a position when asked (same).
  - Produce read-only snapshots for the engine.

NOT responsibilities:
  - Deciding WHEN to open or close. That's the engine's job.
  - Talking to NT8 (live-side concern — handled by OrderManager).
  - Walking bars (that's the executor's job — sim_executor.py / engine_v2.py).

Spec: docs/JULES_ENGINE_DECOUPLE_ORDERS.md
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np

from core_v2.engine_signals import PositionView, PositionsView, PositionDecision
from core_v2.features import N_FEATURES


# Chain-depth ceiling — how many parallel contracts can be stacked on one
# primary. Matches the MAX_CHAIN_CONTRACTS constant BlendedEngine uses today.
MAX_CHAIN_CONTRACTS = 3

# MNQ contract economics (from V1/training_v2)
TICK = 0.25
TICK_VALUE = 0.50      # $/tick/contract
DOLLAR_PER_POINT = 2.0  # $/pt/contract

@dataclass
class ClosedTrade:
    direction: str
    entry_price: float
    exit_price: float
    entry_ts: float
    exit_ts: float
    bars_held: int
    pnl: float
    peak_pnl: float
    entry_tier: str
    exit_reason: str
    entry_day: str
    entry_v2: np.ndarray
    entry_regime_idx: int
    cnn_filtered: bool = False
    cnn_generated: bool = False
    extras: Dict[str, Any] = field(default_factory=dict)



# ─────────────────────────────────────────────────────────────────────────
# Position — the ledger row
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """One open contract in the ledger.

    Every per-trade memory field that used to live on BlendedEngine as
    self.<field> lives here instead. Each Position carries its own copy;
    chain contracts don't share state with the primary. The chain-exit loop
    in nightmare_blended.py lines 533-554 was already manually save/restoring
    these fields to fake per-position state — this dataclass formalizes it.
    """
    # Identity (set at open, never mutated)
    contract_id: str
    direction: str              # 'long' or 'short'
    entry_price: float
    entry_ts: float
    entry_tier: str
    entry_features: np.ndarray  # feature snapshot at entry
    is_chain: bool = False

    # Runtime tracking (updated every bar by update_bar)
    bars_held: int = 0
    peak_pnl: float = 0.0

    # Entry-context snapshots (captured at open, frozen for trade lifetime)
    cnn_flipped: bool = False
    v5_aligned: bool = True
    entry_abs_z: float = 0.0
    entry_velocity: float = 0.0
    entry_h1_z: float = 0.0
    entry_vol_rel: float = 0.0

    # Per-tier exit counters (consecutive-bar confirmation)
    ride_vel_bars: int = 0
    ride_vr_bars: int = 0
    ride_rev_wick_bars: int = 0
    tier_p_center_bars: int = 0
    p_center_bars: int = 0
    z_near_zero_bars: int = 0

    # RIDE exit patience (tiered by entry_h1_z, frozen at entry time)
    ride_exit_bars: int = 0

    # Oscillation tracker (FADE tier exits)
    z_sign: float = 0.0
    zero_crossings: int = 0
    z_peak: float = 0.0
    z_trough: float = 0.0
    peak_amplitude: float = 0.0
    current_amplitude: float = 0.0

    # Sticky flags / volume tracking
    slow_flip_active: bool = False
    peak_volume: float = 0.0

    def to_view(self) -> PositionView:
        """Produce an immutable PositionView snapshot for the engine."""
        return PositionView(
            contract_id=self.contract_id,
            direction=self.direction,
            entry_price=self.entry_price,
            entry_ts=self.entry_ts,
            entry_tier=self.entry_tier,
            entry_features=self.entry_features,
            is_chain=self.is_chain,
            bars_held=self.bars_held,
            peak_pnl=self.peak_pnl,
            cnn_flipped=self.cnn_flipped,
            v5_aligned=self.v5_aligned,
            entry_abs_z=self.entry_abs_z,
            entry_velocity=self.entry_velocity,
            entry_h1_z=self.entry_h1_z,
            entry_vol_rel=self.entry_vol_rel,
            ride_vel_bars=self.ride_vel_bars,
            ride_vr_bars=self.ride_vr_bars,
            ride_rev_wick_bars=self.ride_rev_wick_bars,
            tier_p_center_bars=self.tier_p_center_bars,
            p_center_bars=self.p_center_bars,
            z_near_zero_bars=self.z_near_zero_bars,
            ride_exit_bars=self.ride_exit_bars,
            z_sign=self.z_sign,
            zero_crossings=self.zero_crossings,
            z_peak=self.z_peak,
            z_trough=self.z_trough,
            peak_amplitude=self.peak_amplitude,
            current_amplitude=self.current_amplitude,
            slow_flip_active=self.slow_flip_active,
            peak_volume=self.peak_volume,
        )


# ─────────────────────────────────────────────────────────────────────────
# Ledger — the live state machine
# ─────────────────────────────────────────────────────────────────────────

# Tick sizing for PnL math. MNQ defaults; other instruments can be set
# per-instance via Ledger(tick_size=..., tick_value=...).
DEFAULT_TICK_SIZE = 0.25
DEFAULT_TICK_VALUE = 0.50


class Ledger:
    """Single source of truth for position state.

    A ledger holds at most one primary position and up to MAX_CHAIN_CONTRACTS
    chain contracts (same direction as primary). Chains cannot exist without
    a primary. Direction is enforced at add time.

    Mutation API (only these four methods change state):
      - add_position(...)           → new primary or new chain
      - remove_position(...)        → close one contract, move to history
      - update_bar(features, ...)   → advance bars_held / peak_pnl / oscillation
      - clear()                     → forget everything (test/reset only)

    Read API (never mutates):
      - snapshot()                  → PositionsView
      - is_flat, primary, chains
      - get(contract_id)            → Position or None
      - closed_trades               → list of closed-trade records

    The contract_id is opaque — callers should not parse it. The ledger
    assigns them as 'P001', 'C002', etc. in order of creation. Sim and live
    ledgers have independent id sequences; no need to coordinate.
    """

    def __init__(self,
                 tick_size: float = DEFAULT_TICK_SIZE,
                 tick_value: float = DEFAULT_TICK_VALUE):
        self.tick_size = tick_size
        self.tick_value = tick_value

        # Active positions. primary stored separately from chains for O(1)
        # lookup on the common path. Both indexed by contract_id for removal.
        self._primary: Optional[Position] = None
        self._chains: List[Position] = []
        self._by_id: Dict[str, Position] = {}

        # Closed trades. Format matches what the training pipeline's metrics
        # code currently reads from BlendedEngine.trades:
        #   {'dir', 'entry_price', 'exit_price', 'pnl', 'held', 'peak',
        #    'entry_tier', 'exit_reason', 'cnn_flipped', 'entry_features',
        #    'exit_features'}
        self._closed_trades: List[dict] = []

        # Monotonic id counter — never recycled. On overflow wraps to 0.
        self._next_id: int = 1

    # ── Read API ──────────────────────────────────────────────────────

    @property
    def is_flat(self) -> bool:
        return self._primary is None

    @property
    def primary(self) -> Optional[Position]:
        return self._primary

    @property
    def chains(self) -> List[Position]:
        return list(self._chains)  # defensive copy — caller can't mutate

    @property
    def closed_trades(self) -> List[dict]:
        return list(self._closed_trades)  # defensive copy

    @property
    def n_contracts(self) -> int:
        """Total open contracts: 0 if flat, else 1 + len(chains)."""
        if self._primary is None:
            return 0
        return 1 + len(self._chains)

    def get(self, contract_id: str) -> Optional[Position]:
        """Look up a position by id, or None if not found."""
        return self._by_id.get(contract_id)

    def snapshot(self) -> PositionsView:
        """Produce a read-only PositionsView for the engine."""
        if self._primary is None:
            return PositionsView(primary=None, chains=[])
        return PositionsView(
            primary=self._primary.to_view(),
            chains=[c.to_view() for c in self._chains],
        )

    # ── Mutation API ──────────────────────────────────────────────────

    def add_position(self,
                     direction: str,
                     entry_price: float,
                     entry_ts: float,
                     entry_tier: str,
                     entry_features: np.ndarray,
                     is_chain: bool = False,
                     cnn_flipped: bool = False,
                     v5_aligned: bool = True,
                     entry_abs_z: float = 0.0,
                     entry_velocity: float = 0.0,
                     entry_h1_z: float = 0.0,
                     entry_vol_rel: float = 0.0,
                     ride_exit_bars: int = 0) -> Position:
        """Add a position to the ledger.

        If is_chain=False, this becomes the primary. Raises if a primary
        already exists (callers should check `is_flat` first).

        If is_chain=True, this appends to chains. Raises if there's no
        primary, if direction doesn't match primary, or if the chain cap
        is hit.

        Returns the newly-created Position (with its assigned contract_id).
        """
        if direction not in ('long', 'short'):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if entry_features is None:
            entry_features = np.zeros(N_FEATURES, dtype=np.float32)

        if is_chain:
            if self._primary is None:
                raise ValueError("cannot add chain with no primary")
            if direction != self._primary.direction:
                raise ValueError(
                    f"chain direction {direction!r} does not match "
                    f"primary direction {self._primary.direction!r}"
                )
            if len(self._chains) >= MAX_CHAIN_CONTRACTS:
                raise ValueError(
                    f"chain cap hit ({MAX_CHAIN_CONTRACTS}), cannot add more"
                )
            contract_id = self._make_id('C')
        else:
            if self._primary is not None:
                raise ValueError("cannot add primary, one already exists")
            contract_id = self._make_id('P')

        # Initialize oscillation tracker from entry features if we have them
        z_sign_init = 0.0
        if entry_features is not None and len(entry_features) > 12:
            # z_se for 1m lives at index 12 in the canonical 91-D layout.
            # This matches how BlendedEngine._open_trade computed _z_sign.
            z_1m = float(entry_features[12])
            z_sign_init = 1.0 if z_1m > 0 else -1.0

        pos = Position(
            contract_id=contract_id,
            direction=direction,
            entry_price=entry_price,
            entry_ts=entry_ts,
            entry_tier=entry_tier,
            entry_features=entry_features,
            is_chain=is_chain,
            cnn_flipped=cnn_flipped,
            v5_aligned=v5_aligned,
            entry_abs_z=entry_abs_z,
            entry_velocity=entry_velocity,
            entry_h1_z=entry_h1_z,
            entry_vol_rel=entry_vol_rel,
            ride_exit_bars=ride_exit_bars,
            z_sign=z_sign_init,
            z_peak=abs(float(entry_features[12])) if len(entry_features) > 12 else 0.0,
            z_trough=abs(float(entry_features[12])) if len(entry_features) > 12 else 0.0,
            peak_amplitude=abs(float(entry_features[12])) if len(entry_features) > 12 else 0.0,
            current_amplitude=abs(float(entry_features[12])) if len(entry_features) > 12 else 0.0,
        )

        if is_chain:
            self._chains.append(pos)
        else:
            self._primary = pos
        self._by_id[contract_id] = pos

        return pos

    def remove_position(self,
                        contract_id: str,
                        exit_price: float,
                        exit_ts: float,
                        reason: str,
                        exit_features: Optional[np.ndarray] = None) -> dict:
        """Close a position and record it in closed_trades.

        Returns the closed-trade record dict. Raises if contract_id is unknown.

        Primary semantics: removing the primary is allowed even if chains are
        still open. The chains remain in the ledger until their own exits fire.
        This matches what BlendedEngine does today (line 1176: "Chain contracts
        stay alive — they exit independently"). If you want to flush everything,
        call remove_position for each chain first, then the primary (or use
        clear() in tests).
        """
        pos = self._by_id.get(contract_id)
        if pos is None:
            raise KeyError(f"unknown contract_id: {contract_id!r}")

        # Compute final PnL in dollars (same formula as BlendedEngine)
        if pos.direction == 'long':
            pnl = (exit_price - pos.entry_price) / self.tick_size * self.tick_value
        else:
            pnl = (pos.entry_price - exit_price) / self.tick_size * self.tick_value

        record = {
            'contract_id': contract_id,
            'dir': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': exit_price,
            'entry_ts': pos.entry_ts,
            'exit_ts': exit_ts,
            'pnl': pnl,
            'held': pos.bars_held,
            'peak': pos.peak_pnl,
            'entry_tier': pos.entry_tier,
            'exit_reason': reason,
            'cnn_flipped': pos.cnn_flipped,
            'is_chain': pos.is_chain,
            'entry_features': (pos.entry_features.tolist()
                               if hasattr(pos.entry_features, 'tolist')
                               else list(pos.entry_features)),
            'exit_features': (exit_features.tolist()
                              if exit_features is not None and hasattr(exit_features, 'tolist')
                              else (list(exit_features) if exit_features is not None else [])),
        }
        self._closed_trades.append(record)

        # Remove from live maps
        del self._by_id[contract_id]
        if pos.is_chain:
            self._chains.remove(pos)
        else:
            self._primary = None

        return record

    def update_bar(self,
                   features: np.ndarray,
                   price: float,
                   ts: float,
                   current_volume: float = 0.0):
        """Advance per-bar runtime state on every open position.

        Called once per bar by the executor BEFORE the engine is consulted.
        After this call, position.bars_held / peak_pnl / oscillation tracker
        reflect the current bar, so the engine sees up-to-date state via
        snapshot().

        Mirrors the per-bar updates BlendedEngine.on_state currently does
        inline (lines 439-470 of nightmare_blended.py today).
        """
        if self._primary is None:
            return

        # z_se for 1m is feature index 12 in the canonical layout
        z = float(features[12]) if len(features) > 12 else 0.0

        # Update every open position (primary + chains) with the same market snapshot.
        for pos in [self._primary] + self._chains:
            # bars_held — elapsed 1m bars since entry (cadence-independent)
            pos.bars_held = int((ts - pos.entry_ts) // 60)

            # pnl in dollars (same TICK/TV math as engine)
            if pos.direction == 'long':
                pnl = (price - pos.entry_price) / self.tick_size * self.tick_value
            else:
                pnl = (pos.entry_price - price) / self.tick_size * self.tick_value

            # peak_pnl — monotonic MFE tracker
            if pnl > pos.peak_pnl:
                pos.peak_pnl = pnl

            # peak_volume — monotonic max volume during the trade
            if current_volume > pos.peak_volume:
                pos.peak_volume = current_volume

            # Oscillation tracker (mirrors lines 451-463 of nightmare_blended.py)
            curr_z_sign = 1.0 if z > 0 else -1.0
            if curr_z_sign != pos.z_sign and pos.z_sign != 0.0:
                pos.zero_crossings += 1
                pos.z_sign = curr_z_sign
                # New half-cycle: amplitude = distance from last extreme to zero
                pos.current_amplitude = max(pos.z_peak, pos.z_trough)
                if pos.current_amplitude > pos.peak_amplitude:
                    pos.peak_amplitude = pos.current_amplitude
                pos.z_peak = abs(z)
                pos.z_trough = abs(z)
            else:
                if pos.z_sign == 0.0:
                    pos.z_sign = curr_z_sign
                if abs(z) > pos.z_peak:
                    pos.z_peak = abs(z)
                if abs(z) < pos.z_trough:
                    pos.z_trough = abs(z)

    def apply_position_decision(self, decision: PositionDecision):
        """Apply a per-position counter update from the engine.

        The engine's evaluate() is stateless — it looks at a PositionView
        snapshot and returns a PositionDecision with the new counter values
        it computed. The executor calls this method to persist those counter
        values on the actual Position before the next bar's evaluation.

        Silently ignores decisions for unknown contract_ids (a position may
        have been closed in the same batch that produced this decision).

        The `exit_reason` field of PositionDecision is NOT handled here —
        that's the executor's responsibility. This method touches counters
        only.
        """
        pos = self._by_id.get(decision.contract_id)
        if pos is None:
            return  # position may have been closed already — harmless

        pos.ride_vel_bars = decision.ride_vel_bars
        pos.ride_vr_bars = decision.ride_vr_bars
        pos.ride_rev_wick_bars = decision.ride_rev_wick_bars
        pos.tier_p_center_bars = decision.tier_p_center_bars
        pos.p_center_bars = decision.p_center_bars
        pos.z_near_zero_bars = decision.z_near_zero_bars
        pos.slow_flip_active = decision.slow_flip_active

    def clear(self):
        """Forget everything. Tests and reset flows only."""
        self._primary = None
        self._chains = []
        self._by_id = {}
        self._closed_trades = []
        self._next_id = 1

    # ── Internal ──────────────────────────────────────────────────────

    def _make_id(self, prefix: str) -> str:
        """Assign the next contract_id. Format: 'P001', 'C002', etc."""
        cid = f"{prefix}{self._next_id:03d}"
        self._next_id += 1
        return cid
