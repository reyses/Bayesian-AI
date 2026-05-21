---
name: thin-wrapper-live-engine
description: "Live engine is a thin wrapper. Decisions live in engine.evaluate(); orders in OrderManager; positions in Ledger. New strategies extend by writing a new evaluate() implementation, NOT by duplicating engine_v2 / sidecars / order tracking."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

The live trading architecture (per `docs/JULES_ENGINE_DECOUPLE_ORDERS.md`)
is a thin-wrapper pattern. Each part does one job:
  - **Engine** (BlendedEngine / L5Decider / future X) -- detects setups,
    pure function of market state, owns nothing. Returns a
    `DecisionBatch` from `evaluate(state)`.
  - **Ledger** (`core/ledger.py`) -- owns position state. Shared between
    sim + live.
  - **OrderManager** (`live/order_manager.py`) -- NT8 wire handshake,
    fills -> ledger mutations.
  - **engine_v2** (`live/engine_v2.py`) -- bar-loop driver. Glue.

**Why:** Refactor 2026-04-15 explicitly decoupled signal generation from
order management to eliminate the silent-flip bug class. New strategies
must respect this decoupling.

**How to apply:** When proposing a new live capability:
  1. **First ask**: does this fit as a new `Engine.evaluate(state)`
     implementation? If yes, that's the entire build -- everything else
     (NT8 transport, pending orders, fill reconciliation, position state,
     mock-bridge) is already done in engine_v2 + OrderManager + Ledger.
  2. Only invent new infrastructure (sidecar processes, separate
     transports, parallel state stores) if the new capability literally
     cannot fit `evaluate(state) -> DecisionBatch`. That's rare.
  3. Examples of the WRONG pattern (do not repeat 2026-05-18 mistake):
     - `live/L5_sidecar.py` (414 LOC) -- duplicated NT8 transport +
       fill reconciliation; should have been an engine_v2 patch.
     - `docs/nt8/ZigzagRunnerHybrid_v1.0.0-RC.cs` (497 LOC) -- put
       strategy logic in NT8 calling back to Python; should have been
       Python-side L5Decider with NT8 as dumb pipe.
  4. Example of the RIGHT pattern (2026-05-19):
     `live/l5_decider.py` implements `evaluate(state) -> DecisionBatch`,
     `engine_v2` swaps `BlendedEngine` for `L5Decider` behind a flag.
     ~30 LOC delta in engine_v2 + new decider file. Done.

**Verify before building anything live-related:**
  - Read `docs/JULES_ENGINE_DECOUPLE_ORDERS.md`
  - Read `docs/Active/LIVE_L5_ARCHITECTURE.md` (for engine_mode flag pattern)
  - Read the file `live/engine_v2.py` 7-step startup + per-bar loop
  - Then propose changes.

Live engine is a **thin wrapper**. Don't bypass it.
