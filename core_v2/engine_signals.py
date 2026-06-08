"""
Engine signal types — the contract between the engine and the ledger.

This module exists to separate concerns:
  - Engine         — pure signal emitter. Reads features + a read-only
                     PositionsView. Returns a DecisionBatch. Owns no state.
  - Ledger         — owns Position state (in core_v2/ledger.py). Sole writer.
                     Calls engine.evaluate(...) with a snapshot.
  - Executors      — sim_executor.py walks bars in training; engine_v2.py
                     pumps NT8 messages in live. Both consume DecisionBatch
                     and drive the ledger.

The engine returns observations. The caller decides what to do with them.
Silent state transitions are architecturally impossible: the engine cannot
mutate a Position, so it cannot close or open anything without the ledger
seeing and recording the action.

Spec: docs/JULES_ENGINE_DECOUPLE_ORDERS.md
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Position view — the read-only snapshot the engine receives
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PositionView:
    """Read-only snapshot of one open position as the engine sees it.

    The engine reads fields off this view to drive per-position exit logic
    (peak giveback, ride-velocity bars, oscillation tracker, etc.). It
    cannot write. All mutations happen on the underlying Position in the
    ledger, which is the single source of truth.
    """
    # Identity
    contract_id: str
    direction: str              # 'long' or 'short'
    entry_price: float
    entry_ts: float
    entry_tier: str
    entry_features: np.ndarray  # 91-D feature snapshot at entry
    is_chain: bool              # False for the primary, True for chain rows

    # Runtime tracking (updated each bar by the ledger)
    bars_held: int
    peak_pnl: float

    # Entry-context snapshots (captured at open, frozen for the life of the trade)
    cnn_flipped: bool
    v5_aligned: bool
    entry_abs_z: float
    entry_velocity: float
    entry_h1_z: float
    entry_vol_rel: float

    # Per-tier exit counters (N consecutive bars before triggering)
    ride_vel_bars: int
    ride_vr_bars: int
    ride_rev_wick_bars: int
    tier_p_center_bars: int
    p_center_bars: int
    z_near_zero_bars: int   # FADE phase-0 exit counter

    # RIDE exit patience — tiered by entry_h1_z, frozen at entry
    ride_exit_bars: int

    # Oscillation tracker (for FADE tier exits)
    z_sign: float
    zero_crossings: int
    z_peak: float
    z_trough: float
    peak_amplitude: float
    current_amplitude: float

    # Sticky flags / misc
    slow_flip_active: bool
    peak_volume: float


@dataclass
class PositionsView:
    """All open positions as the engine sees them.

    `primary` is the first-opened position (the one the engine_v2 wrapper
    historically tracked via self.in_pos / self.direction / self.entry_tier).
    `chains` are parallel contracts added via scale-in, always in the same
    direction as primary.

    Invariants (enforced by the ledger, not the engine):
      - If `primary is None`, then `chains == []`. No chains without a primary.
      - All chain contracts have `direction == primary.direction`.
      - `len(chains) <= MAX_CHAIN_CONTRACTS` (ledger-enforced).
    """
    primary: Optional[PositionView] = None
    chains: List[PositionView] = field(default_factory=list)

    @property
    def is_flat(self) -> bool:
        return self.primary is None

    @property
    def all_positions(self) -> List[PositionView]:
        """Primary + chains as a flat list. Empty if flat."""
        if self.primary is None:
            return []
        return [self.primary] + self.chains


# ─────────────────────────────────────────────────────────────────────────
# Decision batch — what the engine returns per tick
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class EntrySignal:
    """An entry opportunity the engine sees in the current bar."""
    tier: str
    direction: str          # 'long' or 'short'
    cnn_flipped: bool


@dataclass
class ExitSignal:
    """An exit recommendation for a specific open position.

    Used for the negative_exit channel (the primary-only "opposing setup has
    higher conviction" signal). Per-position exit reasons are bundled into
    PositionDecision instead.
    """
    contract_id: str
    reason: str             # 'hard_stop', 'giveback_stop', 'ride_velocity_exhausted', etc.


@dataclass
class PositionDecision:
    """Per-bar per-position decision bundle.

    One is emitted for every open position on every bar evaluation. Carries
    both the new counter values (so the ledger can persist them) AND an
    optional exit reason (so the executor can close the position if the
    engine decided it's done).

    The engine computes counter values as a pure function of
    (previous-counter, current-features). The ledger applies them via
    Ledger.apply_position_decision(). Exit reasons do NOT mutate the
    position — the executor processes them explicitly.

    This is the channel through which per-position state flows back to the
    ledger after the engine's stateless evaluation pass. Without it, the
    engine would need write access to Position, which would reintroduce the
    class of bugs this refactor is eliminating.
    """
    contract_id: str
    # Counter updates (engine computes, ledger persists)
    ride_vel_bars: int = 0
    ride_vr_bars: int = 0
    ride_rev_wick_bars: int = 0
    tier_p_center_bars: int = 0
    p_center_bars: int = 0
    z_near_zero_bars: int = 0
    slow_flip_active: bool = False
    # Exit signal (None if the position should keep running)
    exit_reason: Optional[str] = None


@dataclass
class DecisionBatch:
    """Everything the engine wants to happen this bar.

    The executor applies these in a fixed order:
      1. position_decisions  — apply counter updates to every open position;
                               collect exit_reason fields for closes
      2. negative_exit       — if set, overrides everything; close primary
                               (and typically all chains)
      3. chain_entry         — scale into an existing same-direction position
      4. entry               — open a fresh primary (only if flat after exits)

    Order matters because earlier actions can invalidate later ones. If a
    negative_exit fires this bar, any chain_entry or entry from the same
    batch should be ignored (the ledger is no longer in the same state the
    engine evaluated). The executor should recompute flatness between exit
    processing and entry processing.
    """
    entry: Optional[EntrySignal] = None
    chain_entry: Optional[EntrySignal] = None
    position_decisions: List[PositionDecision] = field(default_factory=list)
    negative_exit: Optional[ExitSignal] = None

    @property
    def has_any(self) -> bool:
        """True if the engine wants to do anything actionable this bar.

        Note: counter updates alone do NOT count — they happen every bar
        regardless. "Actionable" means an entry, exit, chain, or negative
        exit that changes the ledger's position shape.
        """
        has_exits = any(pd.exit_reason for pd in self.position_decisions)
        return (self.entry is not None
                or self.chain_entry is not None
                or has_exits
                or self.negative_exit is not None)

    @property
    def exits(self) -> List[ExitSignal]:
        """All per-position exit signals, extracted from position_decisions.

        Convenience accessor for executors that want to iterate exits
        without walking the full position_decisions list.
        """
        return [
            ExitSignal(contract_id=pd.contract_id, reason=pd.exit_reason)
            for pd in self.position_decisions
            if pd.exit_reason is not None
        ]
