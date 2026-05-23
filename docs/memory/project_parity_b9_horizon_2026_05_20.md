---
name: parity-b9-horizon-2026-05-20
description: First L5 SIM run was an engine bug (B9 fired 12x late), not a strategy failure. Lesson - parity-check live runs vs the forward pass before interpreting them.
metadata:
  type: project
---

The first live L5 SIM run (2026-05-20) came in at −$379. A parity check
against the offline forward pass found this was an **engine bug, not the
strategy**: the validated strategy on that exact day was ≈ −$59 (a flat day).

**The bug (now fixed):** `l5_decider` fired the B9 during-trade cut at
`pos.bars_held == 5`. `core/ledger.py` computes `bars_held = (ts−entry_ts)//60`
— elapsed **minutes** — so B9 fired 5 MINUTES after entry. But the B9 model
`b9_remaining_amplitude_K5.pkl` was trained for K=5 in **5-second-bar units**
(`build_trade_trajectory_dataset.py`: `bar_ts = entry_ts + K*5` → 25 s). B9
fired 12× too late, every trade. Fixed: l5_decider now fires B9 off a
5-second-bar counter (`_bar_count − entry_bar_count >= B9_K`).

**Durable trap — units.** `bars_held` (core/ledger) is in MINUTES; the
B9/trajectory K horizons are in 5-SECOND-bar units. Anything timing a
during-trade action against a K horizon must use 5s-bars, NOT `pos.bars_held`.
core/ledger's `//60` was deliberately left as-is — the blended engine
(sim/training) still depends on `bars_held`=minutes.

**Durable lesson — methodology.** Before interpreting a live SIM run's P&L,
**parity-check it**: run the same period through the offline forward pass
(`tools/mock_week_runner.py` mock; `tools/parity_live_vs_forward.py` for price
parity) and diff the decisions. A live run can diverge wildly from the
validated strategy via engine bugs — the headline P&L is meaningless until
parity holds. The user's instinct ("if there's no parity it's back to the
drawing board") was right and caught the bug.

**Mock tool bug (also fixed):** in mock mode, fills stamped `fill_time` with
wall-clock instead of the replayed bar-time, so `entry_ts` was wall-clock and
`bars_held` went negative — B9 never fired in the mock at all. Fixed in
`mock_bridge.py` (`_last_bar_sent_ts`).

**State (2026-05-20):** `engine_v2.py` refactored to **zigzag-only** — the
blended-engine path is removed from the LIVE engine (Phase 1 of
[[zigzag-only-refactor]] / `docs/JULES_ZIGZAG_ONLY_REFACTOR.md`; verified
byte-identical mock output). BlendedEngine still exists for sim/training.
Phase 2 reframed (2026-05-21): prior-day zigzag priming is **REJECTED** — the
forward pass (`build_zigzag_pivot_dataset.py`) runs the detector **per-day**,
anchored at each day's first bar, NOT on prior-day state. The cold-start
divergence study (`tools/zigzag_coldstart_divergence.py`, 54 days / 216
samples) measured the real gap: a mid-session cold-start vs the day's-first-bar
run is a **BOUNDED TRANSIENT** — 0/216 never-resync, resync median 27 min,
median 0 mis-tiled legs, ~$103 mean mis-tiled amplitude (a $ proxy). Not
significant. Correct (optional, low-priority) fix = **same-day catch-up**:
replay today's elapsed bars through the zigzag at session start.
