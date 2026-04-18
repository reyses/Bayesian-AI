# Legacy Tier Engines — 2026-04-18 Snapshot

Frozen reference copies of the blended engine and base NMP engine from
**2026-04-18 13:25 PDT**. Preserved before the potential port of chain
support and/or further tier modifications into `training_iso/`.

## Files

| File | Source | Purpose |
|------|--------|---------|
| `blended_engine_2026_04_18.py` | `training/nightmare_blended.py` | 1597-line blended engine with 9 tiers (CASCADE, KILL_SHOT, FREIGHT_TRAIN, FADE_AGAINST, RIDE_AGAINST, MTF_EXHAUSTION, MTF_BREAKOUT, FADE_CALM, FADE_MOMENTUM), CNN integration, chain logic, tier-specific exits. This is the engine behind the (lookahead-inflated) $740/day baseline. |
| `nightmare_base_2026_04_18.py` | `training/nightmare.py` | 354-line base NMP engine — simpler single-tier version. |

## Why this exists

The 2026-04-18 session rebuilt three tiers (TREND_FOLLOWER, RIDE_AGAINST,
KILL_SHOT) in `training_iso/nightmare_iso.py` using the three-question
method (see `docs/memory/feedback_tier_three_questions.md`). The iso
engine is deliberately simpler than blended:

- No CNN overlay
- No chains (each tier = one position at a time)
- Inverse-signal exits by default, tier-specific rules when EDA justifies
- Per-tier isolation (each tier gets its own engine, no interference)

The blended engine represents the "full" design we may port chain support
from. This snapshot lets us diff against it without checking out old
commits.

## Git equivalents

These files are also preserved via:
- Tag: `legacy-blended-2026-04-18` (see `git tag` output)
- Their original paths in the repo history (commits before any port work)

Consult this reference directory when:
1. Porting chain logic into `training_iso/` (study how `_handle_chain_*`
   methods work in `blended_engine_2026_04_18.py`)
2. Comparing tier-entry rules (the blended engine's `_classify_full_tier`
   is richer than iso's current priority cascade)
3. Measuring drift — if the blended engine's tier thresholds change upstream,
   compare against this frozen copy
