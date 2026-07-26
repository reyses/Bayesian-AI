# Blackboard Discovery Architecture — spec v0 (owner-designed, 2026-07-26)

Supersedes the exit-distillation north star (exit shown edgeless 2026-07-26).
Two complementary intelligences DISCOVER causal edge together, communicating
ONLY through a shared persistent memory — never co-resident (12GB GPU reality).

## The two agents + the bus
- **MAMBA (finder)** — fast, data-scale causal sequence model. Trains on ATLAS
  history to PREDICT a forward target (return / direction / continuation),
  past-only. It is an EDGE DETECTOR, not a trader. Runs when GPU is free.
- **QWEN (interpreter)** — reads the mamba's distilled findings, writes causal
  stories + where-to-look-next. Runs when GPU is free (separately from mamba).
- **THE BANK (bus)** — the teacher_memory SQL scaffolding, extended. It is the
  ONLY channel between the two; it holds the accumulated, validated knowledge.
  Mamba's "long-term memory" = the bank, NOT its weights.

Cycle (time-sliced, async, blackboard): mamba trains -> extract findings ->
write to bank -> (mamba exits) -> qwen loads -> read findings -> interpret +
propose next search -> write to bank -> (qwen exits) -> mamba loads next round
guided by qwen's proposals. Repeat. Orchestrated by markers in the bank +
field-check GPU gating (today's overnight-loop pattern, reused).

## THE CRUX: the extraction protocol (black box -> readable findings)
A mamba emits no natural "patterns." After each run, EXTRACT structured
findings qwen can reason about — the bus is findings, never raw weights:
- feature/input attributions (what drove predictions; permutation importance)
- exemplars: the cases it nailed vs missed (high-|residual| episodes)
- prediction reliability by regime/bucket (actuary strata)
- the OOS score of THIS finding (mandatory field, see guardrail)

## Schema (extends teacher_memory)
Entry types in one bank, cross-linked:
- `finding` (mamba): target, attribution vector, exemplar ids, OOS metric +
  CI, holdout status, run hash.
- `interpretation` (qwen): references finding id(s), causal story, confidence,
  proposed next search, day-agnostic.
- existing `memo`/`retrieve`/`reflection`/ledger events carry over.

## Guardrails — baked into the schema, not our memory
- **OOS-status is a REQUIRED field on every finding.** Qwen queries can only
  promote findings whose holdout status = validated. A dev-only pattern is
  structurally un-promotable. (Today's proof it's needed: the teacher CITES
  reversion_prob, which does not discriminate — interpretation != validation.)
- **Leakage/lockbox from RIDE_EDGE_GATE_SPEC apply unchanged**: causal labels,
  no fold bleed, dev-rotation for loop looks, lockbox opened once.
- **Deflated significance / N_trials** — the recursive loop is a multiple-
  comparisons machine; every promoted edge debits the alpha ledger.

## Prerequisites (do NOT build the loop over a void)
1. **PROBE FIRST** — tiny causal model (linear / small GRU) on forward-return,
   walk-forward OOS, full ATLAS history. Does ANY learnable causal edge exist?
   YES -> build the loop to interpret+accumulate it. NO -> nothing to
   interpret, loop shelved. The probe's result is the bank's FIRST finding.
2. **FIX THE READ PATH** — gen-2 day-carry retrieval silently failed (0
   retrievals days 2-4; reflection-guard likely broke it). The blackboard
   leans on this exact retrieval; verify + fix before the loop depends on it.
3. Extraction protocol implemented + unit-tested (attributions reproduce).

## Why this is a better north star
The old tower distilled EXIT judgment — a problem we proved edgeless. This
loop DISCOVERS where edge is (entry, timing, regime, anything), keeps the
reasoning+memory strengths, respects the hardware, and its two failure modes
are both valuable: find edge (dig) or prove none exists in the causal sequence
(stop chasing). Sign is free; validated, interpreted, accumulated magnitude is
the asset.
