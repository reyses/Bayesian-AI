# JULES — Zigzag-Only Engine Refactor

Spec / design reference. Opened 2026-05-20. Status: **Phase 1 DONE
(2026-05-20); Phase 2 reframed to low-priority same-day catch-up — see below.**

## Context (2026-05-20)
Today's parity investigation found the live L5 run on 2026-05-20 (−$379) was
**not the validated strategy**: B9 was firing its during-trade cut at 5
minutes instead of 25 seconds (`bars_held` is minutes; B9-K5 = 25 s). That
bug is **FIXED** (`live/l5_decider.py` now fires B9 off a 5-second-bar
counter; `live/mock_bridge.py` now stamps fills with bar-time). Re-run of the
mock: the validated strategy on 2026-05-20 ≈ **−$59** (a flat day) — the
−$379 was the bug.

Residual after the fix: the live run and the forward-pass mock still don't
tile legs identically (24 vs 28 trades). Root cause = **anchor mismatch**. The
forward pass (`build_zigzag_pivot_dataset.py`) runs `detect_swings` **per-day**,
anchored at each day's first bar — it does NOT carry prior-day zigzag state. A
live engine launched mid-session cold-starts its zigzag at a different,
mid-day anchor. This refactor (a) strips the engine down to the zigzag-only
path and (b) closes that anchor gap — see Phase 2.

## Goal
`engine_v2.py` currently runs two engines behind `engine_mode`: `blended`
(V1 stack) and `l5` (zigzag stack). Only L5 is being deployed. Produce a
clean **zigzag-only** live engine, and re-anchor the zigzag at the day's
first bar at startup.

## Phases (one change at a time — validate each before the next)

### Phase 1 — Blended-engine removal (pure cleanup, zero functional change)
REMOVE from `live/engine_v2.py`:
- The `engine_mode == 'blended'` branch and the `BlendedEngine` instantiation.
- The V1 `LiveFeatureEngine` **instantiation** for the blended path.
- The V1-feature warmup branch, if blended-only.
- The `--engine-mode` CLI option (or hard-default it to `l5`); `engine_mode`
  config field can stay or be dropped.

KEEP (shared by the L5 path — do NOT touch):
- `core/ledger.Ledger`, `OrderManager`, the bar-processing loop, `NT8Client`,
  `mock_bridge`, trade/ledger logging, the warmup steps (1–7).
- The `LiveFeatureEngine` **class** — `LiveFeatureEngineV2` subclasses it, so
  the base class is still needed even after the blended instantiation is gone.
- `core/sim_executor` helpers if the L5 path uses them (verify at execution).
- `core/ledger.py`'s `bars_held = //60` — NOT touched. The B9 fix already
  made `l5_decider` independent of it; the blended engine (still used by
  sim/training, just not the live engine) keeps depending on it.

RISK: moderate (engine_v2 is ~1730 LOC, the two paths are interwoven at the
warmup/feature-engine selection). Mitigated by the validation below.

VALIDATION: re-run `python tools/mock_week_runner.py --days 2026_05_20`. The
L5 trade output (`reports/live/v2_trades_2026_05_20.csv`) must be
**byte-identical** to the pre-Phase-1 mock. Blended-removal is pure dead-code
deletion for the L5 path — ANY change to an L5 trade means the removal broke
something. Diff against a snapshot taken before Phase 1.

### Phase 2 — Same-day zigzag catch-up (functional — closes the anchor gap)
STEP 1 (diagnose) — **DONE**. `build_zigzag_pivot_dataset.py` runs
`detect_swings` **per-day**: each day reads its own `{day}.parquet`, computes
ATR from that day's 1m bars, runs the detector on that day's 5s closes from
bar 0. The forward pass therefore anchors at the **day's first bar** — it does
NOT carry prior-day zigzag state. CONSEQUENCE: priming the live zigzag with
*prior-day* history (the original Phase-2 design) is **REJECTED** — it would
make the live engine diverge from the validated forward pass, not converge.

STEP 2 (quantify) — **DONE**. `tools/zigzag_coldstart_divergence.py` (run
2026-05-21, 54 days / 216 samples): a mid-day cold-start vs the day's-first-bar
run is a **BOUNDED TRANSIENT** — 0/216 never-resync, resync lag median 27 min
(mean 44, CI [38,50]), median 0 mis-tiled legs, ~$103 mean mis-tiled
leg-amplitude (a $ proxy, not a realized loss). Not a structural break.

STEP 3 (fix) — `prime_zigzag_from_history(bars_df)` on `L5Decider`: at session
start, replay **today's elapsed 5s bars** (the day's first bar → launch time)
through `_update_zigzag`. This re-creates the exact per-day anchor the forward
pass uses → byte-identical zigzag from bar 1, transient eliminated. This is
same-day catch-up, NOT prior-day priming. `engine_v2` passes the history (it
already passes 1m history to `prime_atr_from_history`).

VALIDATION: re-run the mock launched mid-session; verify the primed run tiles
legs identically to the forward pass from bar 1.

PRIORITY: **low** — the transient is bounded and small (Step 2). Worth doing
because it is cheap and makes parity exact, but it does not gate SIM deploy.

## Not in scope
- ATR-multiplier sweep / B9-K sweep — separate research task (d).
- Any change to the strategy logic (B7/B9/B10, zigzag thresholds, R-trigger).
- `core/ledger.py`.

## Artifacts
- Pre-refactor snapshots: `docs/reference/engine_v2_2026_05_20.py`,
  `docs/reference/nightmare_blended_2026_05_20.py`.
- Validation tool: `tools/mock_week_runner.py`, `tools/parity_live_vs_forward.py`.
