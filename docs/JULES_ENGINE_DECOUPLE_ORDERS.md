# JULES: Decouple Order Management from the Engine

**Status:** Phase 3 complete (78 tests passing). Phase 4 ready to execute.

**Phase 3 pickup instructions (for fresh session):**
1. Read this spec for context (especially the Phase 3 section)
2. Read `docs/daily/2026-04-15.md` for what shipped in Phases 1-2
3. Run `pytest tests/ -v` to confirm 54 tests green
4. Build `core/sim_executor.py`:
   - Walk ATLAS bars via `training/sfe_ticker.py` FeatureTicker
   - Each bar: `ledger.update_bar()` → `engine.evaluate()` →
     apply `PositionDecision` counter updates → process exits/entries
   - Produce closed-trade list in same format as `BlendedEngine.trades`
5. Run one ATLAS day through both paths (old `on_state` vs new executor)
6. Compare trade tapes — document divergences (expected: silent-flip
   trades will differ because the bug is removed)
**Guiding principle:** each part does one thing well.
  - **Engine** — detects setups. Pure function of market state. Owns nothing.
  - **Ledger** — owns position state. Single source of truth.
  - **OrderManager** — owns NT8 wire handshake. Translates fills into ledger
    mutations.
  - **Sim executor** — walks bars in sim. Calls engine, applies decisions.
  - **Live engine_v2** — pumps NT8 messages through the orchestration layer.
    Glue, nothing more.

**Scope:** Strip order management out of `BlendedEngine`. The engine emits
signals; the ledger owns position state. One ledger, shared by sim and live,
lives in `core/`.

**Not in scope:** rebuilding `core/exit_engine.py` with the archived modular
exits design, 1s bar subscription, `PARTIAL_BAR` processing. Those are later
specs.

## Why

Every bug from 2026-04-15 traces back to one root cause: `BlendedEngine` does two
jobs smashed together.

1. Signal generation — "I see a CASCADE long setup," "the current trade's velocity
   is exhausted," "an opposing KILL_SHOT is forming."
2. Order management — owns `in_pos`, `direction`, `entry_price`, `_chain_contracts`,
   `trades`; calls `_close_trade` and `_open_trade` and mutates state in place.

Because the engine owns state, it can silently close and reopen a position inside
a single `on_state()` call. The live wrapper has no boundary to detect. NT8 has
no idea it happened. The silent-flip bug, the chain scale-out bug, the three-state
divergence — all downstream of this conflation.

The fix is architectural: the engine never touches position state. It just answers
a question: "given this market snapshot and these positions I'm holding, what do
you see?" The caller decides what to do with the answer.

Sim and live share one ledger (in `core/`) in this spec. Fill semantics differ
(sim fills instantly at bar close, live waits for NT8 confirmation) but the
ledger data structure and the engine→ledger contract are identical.

---

## Shape after the change

### Engine: pure signal emitter

`BlendedEngine.evaluate(state) -> DecisionBatch`

- `state` contains: features_79d, price, timestamp, **and a read-only view of
  currently open positions** (a `PositionsView` dataclass the caller constructs).
- Returns a `DecisionBatch` with all decisions the engine would recommend this bar.
- Does not mutate anything on `self`. Does not call `_close_trade`. Does not call
  `_open_trade`. Does not touch `_chain_contracts`. Does not append to `trades`.
- `_classify_full_tier` stays as-is — it's already pure.

```python
@dataclass
class PositionView:
    contract_id: str           # opaque id assigned by the caller's ledger
    direction: str             # 'long' / 'short'
    entry_price: float
    entry_ts: float
    entry_tier: str
    entry_features: np.ndarray # the 79d feature snapshot at entry
    bars_held: int
    peak_pnl: float
    # Per-position exit-state (moves OFF the engine, ONTO the position):
    cnn_flipped: bool
    v5_aligned: bool
    ride_vel_bars: int
    ride_vr_bars: int
    ride_rev_wick_bars: int
    z_sign: float
    zero_crossings: int
    z_peak: float
    z_trough: float
    peak_amplitude: float
    tier_p_center_bars: int
    p_center_bars: int
    slow_flip_active: bool
    peak_volume: float
    is_chain: bool

@dataclass
class PositionsView:
    primary: Optional[PositionView]   # the first-opened position, if any
    chains:  List[PositionView]       # parallel contracts in same direction

@dataclass
class EntrySignal:
    tier: str
    direction: str
    cnn_flipped: bool

@dataclass
class ExitSignal:
    contract_id: str   # which position this applies to
    reason: str        # 'giveback_stop', 'hard_stop', 'ride_velocity_exhausted', etc.

@dataclass
class DecisionBatch:
    entry: Optional[EntrySignal]              # open fresh primary
    chain_entry: Optional[EntrySignal]        # scale-in (same direction)
    exits: List[ExitSignal]                   # per-position exit recommendations
    negative_exit: Optional[ExitSignal]       # opposing setup w/ higher conviction
```

The engine stops owning per-position state. Every field that currently lives on
`self` and refers to "the" open trade — `bars_held`, `peak_pnl`, `_ride_vel_bars`,
`_zero_crossings`, `_z_sign`, `_z_peak`, `_z_trough`, etc. — moves into
`PositionView`. The caller's ledger is responsible for updating those fields each
bar and passing the updated view back into `evaluate()`.

### Shared ledger (`core/ledger.py`)

New file. Single source of truth for position state, used by both sim and live.

```python
@dataclass
class Position:
    contract_id: str        # opaque id the ledger assigns (e.g. 'P001', 'C002')
    direction: str          # 'long' / 'short'
    entry_price: float
    entry_ts: float
    entry_tier: str
    entry_features: np.ndarray
    bars_held: int = 0
    peak_pnl: float = 0.0
    is_chain: bool = False  # False for primary, True for chain rows
    # Per-position exit-tracking state (migrated off BlendedEngine):
    cnn_flipped: bool = False
    v5_aligned: bool = True
    ride_vel_bars: int = 0
    ride_vr_bars: int = 0
    ride_rev_wick_bars: int = 0
    z_sign: float = 0.0
    zero_crossings: int = 0
    z_peak: float = 0.0
    z_trough: float = 0.0
    peak_amplitude: float = 0.0
    tier_p_center_bars: int = 0
    p_center_bars: int = 0
    slow_flip_active: bool = False
    peak_volume: float = 0.0

class Ledger:
    def __init__(self):
        self._positions: List[Position] = []
        self._closed_trades: List[dict] = []
        self._next_id = 1

    # Query
    def snapshot(self) -> PositionsView: ...
    def is_flat(self) -> bool: ...
    def get(self, contract_id: str) -> Optional[Position]: ...
    @property
    def closed_trades(self) -> List[dict]: ...

    # Per-bar update (bars_held, peak_pnl, oscillation, etc.)
    def update_bar(self, features, price, ts): ...

    # Mutation — called by sim directly, by live from fill handlers
    def add_position(self, direction, entry_price, entry_ts, entry_tier,
                     features, is_chain=False, cnn_flipped=False) -> Position: ...
    def remove_position(self, contract_id, exit_price, exit_ts, reason,
                        exit_features) -> dict: ...   # returns closed trade record
```

The ledger is the ONLY thing that touches position state. The engine looks at it
through `snapshot()` but cannot mutate it. Sim and live both call `add_position`
and `remove_position` — the only difference is when:

- **Sim** calls them directly in the same bar an entry/exit signal fires
  (instant fill at current price).
- **Live** calls them only when a FILL message arrives from NT8 confirming the
  order actually happened. Until the fill, the position does not exist in the
  ledger. If the bridge rejects or times out the order, the position never
  materializes and nothing in the ledger changes.

### Sim executor (`core/sim_executor.py`)

Thin glue (≤100 lines). Owns the bar loop for training:

1. `ledger.update_bar(features, price, ts)` — bumps bars_held, peak_pnl, etc.
2. `decision = engine.evaluate(state_with_positions=ledger.snapshot())`.
3. Apply `decision` in a fixed order:
   - `negative_exit` / `exits[]` → `ledger.remove_position(...)` at current price.
   - `chain_entry` → `ledger.add_position(..., is_chain=True)` at current price.
   - `entry` → `ledger.add_position(...)` at current price (only if flat).
4. Return closed trades for this bar.

The training loop (`nn_v2/run.py blended`) creates a `Ledger` + `Executor`,
walks bars, reads `ledger.closed_trades` at the end. Same downstream metrics as
today.

### Live executor (`live/engine_v2.py`, rewired)

The ledger moves to `core/`, but live still needs an order handshake layer
because NT8 fills are asynchronous. `live/order_manager.py` stays as a thin
**wire adapter**:

- Still owns `OrderRecord` with the state machine I built this morning
  (PENDING → SENT → ACKED → WORKING → FILLED / REJECTED / TIMED_OUT).
- Still owns `build_entry_order`, `build_scale_in_order`, `build_scale_out_order`,
  `build_exit_order`, `mark_sent`, `check_pending_timeouts`.
- **Stops owning** `self.position` — the `PositionState` dataclass is deleted
  and replaced by queries into `core/ledger.Ledger`.
- `on_fill` stops mutating position state directly. Instead it calls
  `ledger.add_position(...)` (for OPEN fills) or `ledger.remove_position(...)`
  (for REDUCE fills). The ledger is the one writing. OrderManager just decides
  which ledger method to call based on the `OrderRecord.intent` it already tracks.

`live/engine_v2.py` on each bar:
1. `ledger.update_bar(features, price, ts)`.
2. `decision = engine.evaluate(state_with_positions=ledger.snapshot())`.
3. For each item in the decision, call the matching `OrderManager.build_*`
   method. OrderManager sends the wire message, tracks the handshake, and
   eventually calls into the ledger when the fill comes back.
4. No more `prev_in_pos`, `prev_direction`, `entered`, `exited`, `silent_flip`
   detection. Architecturally impossible — the engine has no state to flip.

---

## What changes by file

| File | Action |
|---|---|
| `core/engine_signals.py` (new) | Defines `PositionView`, `PositionsView`, `EntrySignal`, `ExitSignal`, `DecisionBatch`. Imported by engine, ledger, sim executor, and live wrapper. |
| `core/ledger.py` (new) | `Position` dataclass + `Ledger` class. Single source of truth for position state. Owns `add_position`, `remove_position`, `update_bar`, `snapshot`, `closed_trades`. Used by both sim and live. |
| `core/sim_executor.py` (new) | Thin bar-loop driver for sim. Calls `ledger.update_bar` → `engine.evaluate` → applies `DecisionBatch` to ledger. ≤100 lines. |
| `training/nightmare_blended.py` | Remove all state fields (`in_pos`, `direction`, `entry_price`, `entry_tier`, `bars_held`, `peak_pnl`, `_chain_contracts`, `trades`, and every `_ride_*` / `_z_*` / `_tier_p_center_bars` / etc. field). Remove `_close_trade`, `_open_trade`, `_flatten_all_chains`, `force_close`, `get_trade_state`, `restore_trade_state`, `reset`, `on_state`. Add `evaluate(state)`. All per-tier exit logic (`_check_exit` branches, chain-exit loop, negative-exit block) moves inside `evaluate` and reads from the passed `PositionView` instead of `self`. `_classify_full_tier` stays as-is. |
| `nn_v2/run.py` (blended path) | Replace direct `engine.on_state(state)` calls with `core.sim_executor.run(ledger, engine, bars)`. Read `ledger.closed_trades` for final metrics. Output format unchanged. |
| `live/engine_v2.py` | Each bar: `ledger.update_bar(...)`, then `engine.evaluate(...)`, then process the decision into `OrderManager.build_*` calls. Remove `prev_in_pos`/`prev_direction`/`prev_tier` snapshots, remove silent-flip detection (not possible anymore), remove phantom-chain rollback (not possible anymore — nothing happens until a fill confirms). `mark_sent` / handshake / timeout watchdog unchanged. |
| `live/order_manager.py` | **Shrinks.** Deletes the internal `PositionState` dataclass and `self.position` field. Fills no longer mutate a local position — they call into `core.ledger.Ledger` (which is injected via constructor or passed to each method). Keeps: `OrderRecord`, `OrderState`, `OrderIntent`, `mark_sent`, `check_pending_timeouts`, `build_*_order`, `on_fill`, `on_order_ack`, `on_order_status`, `on_heartbeat`, `on_position`. The `on_fill` method's job changes from "update my position" to "translate the fill into a ledger mutation." Queries like `is_flat` / `can_enter` / `can_exit` delegate to the ledger. |

Nothing gets deleted in Phase 1. Old code lives alongside new code until every
phase is verified.

---

## Decisions locked

- **Single responsibility** — one job per part (see guiding principle above).
- **Files in `core/`** — ledger shared between sim and live.
- **Drift budget for training refactor** — ±2% on `$/day` vs `safe/v740`.
- **Per-position state on the `Position` dataclass** — every per-trade memory
  field (`bars_held`, `peak_pnl`, `ride_vel_bars`, `ride_vr_bars`,
  `ride_rev_wick_bars`, `z_sign`, `zero_crossings`, `z_peak`, `z_trough`,
  `peak_amplitude`, `tier_p_center_bars`, `p_center_bars`, `slow_flip_active`,
  `peak_volume`, `cnn_flipped`, `v5_aligned`, entry context snapshots) lives on
  `Position` in `core/ledger.py`. Engine reads through `PositionView` and
  cannot write. Alternative (engine holds a dict keyed by contract_id) was
  rejected because it splits state ownership.
- **Silent-flip bug** — removed as part of this refactor. No preservation.
- **Live session is the final parity test** — not an offline equality check.
  The last phase is: run the new architecture in a live NT8 session, compare
  engine decisions to actual NT8 behavior bar-by-bar, fill-by-fill. If they
  match, we're done. Dead-code deletion happens BEFORE the live test so the
  test runs against clean code.

## Phases (approve one at a time)

### Phase 1 — Types + ledger skeleton

- Create `core/engine_signals.py` with the dataclasses (`PositionView`,
  `PositionsView`, `EntrySignal`, `ExitSignal`, `DecisionBatch`).
- Create `core/ledger.py` with `Position` + `Ledger` class. Implement
  `add_position`, `remove_position`, `update_bar`, `snapshot`, `closed_trades`,
  `is_flat`, `get`. No engine integration yet.
- Unit tests on the ledger alone: open a position, update some bars, verify
  `bars_held` / `peak_pnl` / oscillation tracking matches the values the engine
  currently computes on the same inputs. Close the position, verify the
  `closed_trades` record format matches what the training pipeline expects.
- No behavior change anywhere. `BlendedEngine` unchanged.

### Phase 2 — Add `evaluate()` alongside `on_state()`

- `BlendedEngine.evaluate(state)` implemented. Internally it reuses the existing
  `_check_exit`, `_classify_full_tier`, chain-exit, and negative-exit logic, but
  those branches get refactored to take a `PositionView` parameter instead of
  reading `self.*`. The old `on_state` stays for now and keeps mutating state.
- Unit test: for a fixed synthetic bar sequence and a fixed `PositionsView`,
  `evaluate()` returns a `DecisionBatch` whose decisions match what `on_state`
  would have done on the same inputs. Lock the signal contract.

### Phase 3 — Sim executor

- Implement `core/sim_executor.py`.
- Plug into the blended pipeline behind a runtime flag so both paths can run
  side by side for debugging without committing yet.
- At this phase we do **NOT** try to prove bit-exact trade parity with the old
  `on_state` path, because the new path fixes the silent-flip bug and the old
  path relies on it. An exact-match test would only pass if we preserved the
  bug, which we are not doing. Instead:
  - Unit test every piece of per-tier exit logic in isolation (give a known
    `PositionView` + features, assert the `DecisionBatch` shape).
  - Run one full ATLAS day through the new path, visually compare the trade
    tape against the old tape (same-ish tier distribution, same-ish $/trade,
    no pathological differences). Document any large divergences.

### Phase 4 — Switch the training loop, measure baseline drift

- `nn_v2/run.py blended` uses `core.sim_executor`. `on_state` is no longer
  called in training.
- Run full IS + OOS. Compare `$/day`, trade count, tier breakdown, win rate
  to `safe/v740`.
- **Acceptance: ±2% drift on `$/day` either way.** Tighter catches bugs;
  looser absorbs honest noise from the silent-flip removal.
- If drift exceeds threshold, stop and diagnose. Most likely causes: a
  per-position field wasn't migrated correctly, or an exit branch was
  rewritten wrong.

### Phase 5 — Switch live wrapper + delete dead code (clean break)

- `live/engine_v2.py` calls `ledger.update_bar` → `engine.evaluate` → processes
  `DecisionBatch` into `OrderManager.build_*` orders.
- `live/order_manager.py` shrinks: `PositionState` deleted, `on_fill` rewired
  to mutate the shared `Ledger`, `is_flat`/`can_enter`/`can_exit` delegate to
  the ledger.
- Remove silent-flip detection from `engine_v2.py` (not possible anymore).
- Remove phantom-chain rollback (also not possible anymore — nothing happens
  to the ledger until a fill is confirmed).
- Delete all dead code in one commit:
  - `BlendedEngine.on_state`, `_close_trade`, `_open_trade`,
    `_flatten_all_chains`, `force_close`, `get_trade_state`,
    `restore_trade_state`, `reset`, and every per-engine state field.
  - `PositionState` dataclass from `live/order_manager.py`.
  - `training/nightmare_blended.py` should drop ~200–300 lines net.
- After this phase, the codebase has one ledger, one engine, zero shadow
  state. Clean.

### Phase 6 — Live session parity test (final gate)

- Run a full US session live (~6.5 hours on a normal weekday) with the new
  architecture against NT8 SIM.
- During the session, assert bar-by-bar that:
  - `ledger.snapshot()` always matches what NT8 says via heartbeats (same
    direction, same qty, same avg price).
  - Every `DecisionBatch` item either became an NT8 order or had a logged
    reason for why not.
  - Every NT8 fill either created or closed exactly one position in the
    ledger — no unmatched fills, no phantom positions.
  - Engine `entry_price` (via `PositionView`) equals NT8 actual fill price
    for every open position.
- At session end, compare `ledger.closed_trades` against
  `reports/live/nt8_trades_*.csv` (NT8 ground truth). **They must match
  exactly**: same count, same entry/exit prices, same PnL, same timestamps.
  This is the live parity proof.
- If session passes: `JULES_ENGINE_DECOUPLE_ORDERS: complete`. Tag + safety
  branch.
- If session fails: roll back to pre-Phase-5 commit, diagnose, retry.

---

## Risks

1. **Sim baseline drift** — The current engine does same-bar close+reopen (that's
   the silent-flip bug). After the refactor, the engine emits a close signal and
   the ledger processes it, then the next bar brings a fresh entry opportunity.
   Some trades that were glued together back-to-back become separate. Baseline
   `$/day` will move. The Phase 4 acceptance gate is there specifically to catch
   this. If the drift is negative, we need a decision: keep the refactor and
   accept a lower baseline, or add a "fast re-evaluation" window on the caller
   side so that after a close the entry check runs on the next bar instead of
   waiting for the next 1m boundary.

2. **Per-position state migration** — Today the engine has one `peak_pnl`, one
   `_ride_vel_bars`, one `_z_sign`. In the new world every `PositionView` has its
   own. The chain-exit loop already does this pattern with save/restore gymnastics
   (lines 533–554 in `nightmare_blended.py`) — we're formalizing what that code
   was already doing manually. Risk: missing a field in the migration and having
   a chain exit behave like the primary by accident. Mitigation: Phase 3's exact
   parity test catches this.

3. **Checkpoint format changes** — `get_trade_state`/`restore_trade_state` are
   called by `live/engine_v2.py` on warm restart to rebuild the in-flight trade
   from `live/state/checkpoint.json`. After the refactor, the checkpoint is
   rebuilt from `OrderManager.position` + a new `PositionState` snapshot instead.
   Existing checkpoints from before this refactor will not restore into the new
   shape. Mitigation: bump a checkpoint version field, refuse to restore old
   versions, log and continue flat.

4. **Sim perf** — `evaluate()` allocates a `DecisionBatch` per bar. On 5s bars
   that's ~17k allocations per trading day, on 1m it's ~1500. Negligible for
   Python but worth benchmarking once against `safe/v740` to make sure nothing
   regresses badly.

5. **Parity contract drift** — If sim's `SimLedger.process_bar` and live's
   `engine_v2.py` disagree on how to apply a `DecisionBatch`, sim and live
   diverge silently. Mitigation: both call a shared helper
   (`apply_decision_batch(ledger, batch, fill_fn)`) where the only per-environment
   thing is the `fill_fn` callback — sim fills instantly at current price, live
   returns an OrderManager order to be sent.

---

## Success criteria

- `SILENT_FLIP` events: architecturally 0. The wrapper's silent-flip detector
  from today's fix can be removed because the condition cannot arise.
- Sim `$/day`, OOS `$/day`, trade count within 5% of safe/v740 baseline after
  Phase 4.
- Sim ↔ live parity: identical `closed_trades` list when the same bar sequence
  is fed into both paths.
- `nightmare_blended.py` shrinks by the state management and close/open logic
  (probably ~200–300 lines removed).
- One live session (Phase 5) with zero RECONCILE MISMATCH, zero ORDER TIMEOUT,
  zero engine-vs-NT8 divergence.
- Chain exits in live always show as `Sell`/`Buy to cover` in NT8 (this was
  already fixed by today's bridge patch, but the new architecture makes it
  structurally impossible to regress).

---

## What this does NOT fix

- Signal latency after an exit. Still bounded by bar cadence (5s chart, 1m
  entry-gate). This spec keeps the same gating logic — it just moves it out of
  the engine and into the signal. A separate spec can relax the gate once this
  one is landed.
- `core/exits/*.py` modular exit checkers are still orphaned. `core/exit_engine.py`
  is still missing. Restoring the archived modular exits design is a later spec.
  This spec keeps the tier-based exit logic that `nightmare_blended.py` already
  has — we're just moving it into `evaluate()`.

---

## Open questions

**All resolved.** Design locked. Pending explicit go-ahead to start Phase 1.
