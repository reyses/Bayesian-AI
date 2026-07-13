# AMENDMENT to 049 — FPS CORE IS FROZEN (Moises directive)
**Doc:** 050 · **Date:** 2026-07-13 · **Author:** Claude (reviewer), directive from Moises · **Status:** BINDING — supersedes doc 049 §2 on FPS extension

## The rule
**`core_v2/FPS/forward_pass_system.py` and `core_v2/FPS/state.py` are FROZEN.**
FPS is the canonical causal engine consumed by RL training, all catalog research,
and the live path lineage. A subtle change poisoned into it corrupts everything
downstream SILENTLY. The detectors are RESEARCH consumers of FPS — they read the
stream, they do not alter it.

This REVERSES doc 049 §2 ("extending FPS (preferred)"). Extending FPS is now the
LAST resort, not the first.

## Detectors must, in this order of preference
1. **Read existing `BarState` fields only** — `ohlcv_5s`, `ohlcv_1m`, `price`,
   `v2`/`v2_vector` (any of the 185 named V2 features), `is_1m/5m/15m/1h_close`,
   `regime_2d`, `bar_idx`, `timestamp`. Everything most detectors need is already here.
2. **Carry detector-side state** (running OR high/low, prior-day levels, VWAP
   accumulation, priming flags) computed causally FROM the stream, INSIDE the
   detector class — never by touching FPS.
3. **Causal pre-compute** in a separate module the detector reads (e.g. prior-day
   OHLC from yesterday's bars) — still outside FPS.

## If a detector genuinely cannot be built without an FPS change
It must clear ALL of these before a single line is written — no exceptions:
1. **Written rationale** proving (1)-(3) above are impossible, not merely inconvenient.
2. **Additive only**: a NEW optional field or `__init__` kwarg, DEFAULT OFF, that
   cannot alter any existing output when unused. No change to existing fields,
   ordering, timestamps, or semantics.
3. **Bit-identical parity gate**: prove the yielded stream is byte-for-byte the
   same with the new option OFF, across >=3 days, via a stream hash — the exact
   discipline used for the 2026-07-12 FPS opt-in changes (parity SHA
   675e744170679490, use_5s_price + skipped_bars + vectorized joins, all default
   bit-identical). That commit (f4544208) is the ONLY sanctioned model for an FPS edit.
4. **Explicit Claude review AND Moises approval** in a comms doc BEFORE merge.
   AG may not self-approve an FPS change.

## Consequence for the doc-050 plans (AG)
In each per-test plan, the "FPS inputs" line must resolve to preference 1/2/3.
If any test claims it needs an FPS core change, that test's plan must carry the
§"cannot be built" justification separately — and I will scrutinize it hardest.
Default expectation: ZERO FPS core changes across all 24 detectors.

## Reviewer standing check (added to 049 §5)
Any AG turn diffing `core_v2/FPS/*` without a pre-approved rationale doc = automatic
REJECT, no matter what else the turn contains.

*(AG plan is now doc 051; awaiting it.)*
