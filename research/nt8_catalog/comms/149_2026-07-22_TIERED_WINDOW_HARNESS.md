# DESIGN + EXECUTION — Tiered-window teacher harness (full-depth gen-0)
**Doc:** 149 · **Date:** 2026-07-22 · **Author:** Claude (Fable) · **Design:** Moises (via Telegram bridge) · **Status:** RATIFIED, RUNNING

## Supersedes doc 148's ctx-bump plan
Doc 148's 12288 rerun was stopped by its own verify-then-stop gate: packets are
20–61-frame telescopes (~2.2k tok/min accumulating; full telescope up to ~148k
tokens — beyond qwen3's 40,960 architectural max). "Raise ctx" has no endpoint;
the decision variable is WHAT HISTORY EACH DECISION SEES.

## The design (Moises, 2026-07-22, verbatim logic)
"Sub-minute is for taking immediate action; above-minute is for decisions.
Drop sub-minute after 1 minute — it's telescopic both ways; the data is kept
in the higher TFs."

Per-frame context (rebuilt each minute):
1. **ANCHOR (pinned):** frame-0 wide field — all 8 TFs incl. 1h/4h/1D + entry
   context. (The builder already emits HTF only in frame 0 — measured, not
   assumed: frames 1+ contain zero [15m]/[1h]/[4h]/[1D] lines.)
2. **HISTORY (decision context):** last **20 minutes**, `[1m]`+`[5m]` lines
   only, with the decision trail (minute j: HOLD/EXIT).
3. **NOW (action context):** current minute's full tape incl. `[5s]`/`[15s]`.

## Measured budget (tokenizer audit, `tools/audit_packet_ctx.py` + tiered calc)
anchor ~2.3k + now ~0.9k + 20×~0.28k history ≈ **~9.8k tokens, FLAT at any
episode depth** (20 or 61 min). num_ctx=12288 leaves ~2.4k headroom.
VRAM: KV ~1.95GB → 37/41 layers on the RTX 3060.

## Implementation
`pipeline/eval_native_tiered.py` — new harness, reuses eval_native_ckpt.py
machinery (loader, last-logits reader, selftest, ckpt IO, VRAM guardrail).
Packets UNTOUCHED. Per-frame `llm.reset()` + full prefill (the price of decay;
~9k tok/frame). Artifacts: `gate_state/acceptance_results_tiered.jsonl`,
`reports/acceptance_native_tiered.csv`. Readout tagged `last_logits_v2+tiered_w20`.

## Semantics note (label honesty)
Labels = "teacher judging with entry anchor + 20-min 1m/5m memory + live tape"
— a bounded-context teacher, matching how any deployable bounded model runs.
NOT comparable 1:1 with the 8192 census labels (different visible history);
census artifacts retained separately for exactly that comparison.

## Verify-then-stop (armed)
First 3 episodes must show: ALL frames served (no early break), 0 taint frames,
real p_exit at the deepest frame. Fail → kill run, re-audit. Census run's
per-frame numbers are the comparison baseline.

## Gen-1 packet design principle (Moises, 2026-07-22 — recorded for the builder)
Full log-decay retention INSIDE the frame stream: each TF keeps only the bars
the next TF up has not yet absorbed (~2 coarser-TF bars' worth): 1s until its
5s closes; 5s → ~6 bars (2×15s); 15s → ~8 (2×1m); 1m → ~10 (2×5m); 5m → ~6
(2×15m); … "Telescopic both ways — the data is kept in the higher TF."
Context becomes ~constant at ANY depth BY CONSTRUCTION (self-similar, not
accumulating). Cannot be applied at harness level to gen-0 packets (each frame
carries only the latest closed bar per TF — no bar series to decimate); lands
in the gen-1 packet builder and the student's feature-stream design.

## Gen-1 addendum: text-definition spec required (Moises, 2026-07-23 early)
The 7 residual taints (0.24%: single frames 29–183 tokens over budget) trace to
UNSPECIFIED text serialization — frame text width varies with market state
(wider numerics on volatile minutes, conditional 5m blocks), so token cost is
only statistically predictable. Owner directive: the gen-1 packet builder gets
an explicit TEXT-DEFINITION SPEC — fixed-precision/fixed-width numeric
formatting, per-TF line templates with a token budget EACH, deterministic
optional-block rules — so every frame's token count is EXACT by construction
and ctx budgets stop being estimates. Pairs with the log-decay retention
principle above.

## Gen-2 lever: memory-augmented teacher (owner, 2026-07-23 — design notes)
Long-term disk memory completing the hierarchy (tiered window = working memory):
(a) within-episode retrieval of dropped mid-history; (b) cross-episode "trader
journal" — learning WITHOUT weight updates, a third evolution channel besides
genome + student. HARD GUARDS: strictly TIME-CAUSAL (episode k sees only
journals of episodes ended before k) — walk-forward memory only; and for OOS:
**snapshot isolation** (owner-ratified): checkpoint the brain → OOS runs on a
copy-on-write branch (writes normally) → post-OOS rollback. Template-leak rule:
templates persist ONLY if derived from training-side episodes (option A); if an
OOS-derived template is ever kept, those episodes are BURNED from the held-out
pool with an alpha-ledger entry. Mirrors the student's MacroBank slow-tier
(SPEC_ARCH_LOCK_MACRO_MEMORY.md) — same architecture both sides.
**Memory-lifecycle protocol (owner+Claude synthesis, 2026-07-23):** day-agnostic
scrubbing alone is NOT leak-free (a rule distilled from OOS episodes encodes
them whether or not it names the day — the rule IS the leak on reuse). Sound
form = single-use-then-graduate: (1) OOS episodes gate ONCE on a memory branch;
(2) harvest day-agnostic templates (scrub all day-identifiable specifics);
(3) those episodes GRADUATE to the training side, never gate again (alpha-
ledger entry); (4) the lockbox replenishes from the test-base expansion. The
teacher's brain grows without contaminating any live held-out claim.
Burn granularity = the WHOLE DAY (owner-confirmed 2026-07-23): episodes within
a day are correlated (the program's unit of independence is the day — the
pseudoreplication rule), so sibling episodes can't be treated as unseen once
any of them gated. Burn the day, ledger it, replenish with fresh days.
**MEMO protocol v1 (owner directive 2026-07-23 "unload to long-term memory /
compress context / make it smarter"):** after each DECISION the model may emit
`MEMO: <=30 words` — its own salience compression. Memos ride in later frames
under YOUR MEMOS; the mechanical 1m/5m history window shrinks 20→10 min in
exchange (net context DOWN, information quality UP). v1 = within-episode only
(zero leakage). v2 = the cross-episode journal above, under the ratified
lifecycle. SEQUENCING: current exam day completes under the current protocol;
handbook-genome + MEMO ship together as the NEXT gate configuration (one
config change per gate).
**SQL memory form factor (owner directive 2026-07-23):** the teacher's memory
uses the project's own architecture — entries as source of truth + SQLite FTS5
derived index (`teacher_memory.db`: memos(episode, day, minute, tags, text) +
FTS mirror). Per-frame retrieval replaces carrying: a DETERMINISTIC query built
from the NOW frame's state (regime, giveback bucket, velocity signs) pulls
top-k memos into a RELEVANT MEMORY block — context stays flat regardless of
knowledge size; fixed query template + fixed k + rowid tie-break keeps runs
reproducible. Lifecycle (branch/burn) applies to the DB per the protocol above.

## v1 GATE FAILURE + v2 correction (same night — the gate worked)
v1's verify-then-stop FAILED: deep frames hit 23–25k REAL tokens (13–20 taint
frames/episode). Root cause: the "~9.8k plateau" estimate used chars/4; this
content is dense numerics tokenizing at **~1.65 chars/token** (census cross-
check: frame-0 9,235 chars = 5,698 tokens). Estimation error, not design error
— the offline audit tool used the real tokenizer and was correct for the
telescope; the tiered projection did not.
**v2 fix (within the ratified principle):** history keeps only the `closed-bar`
`[1m]`/`[5m]` lines — past indicator dumps dropped (derived/rolling values,
current state visible in NOW; the owner's "20 bars of 1m" means the BARS).
Re-verified with the REAL tokenizer on the worst packets (61-frame episodes):
**max 10,336 tokens < 12,288** with ~2k headroom. v1 artifacts archived as
`*_tiered_v1FAILED.*`; v2 rerun from scratch (fresh ckpt), gate re-armed.
LESSON (standing): never project token budgets by chars/4 on numeric-dense
text; always verify with the model tokenizer before burning GPU-hours.

## Prior artifacts banked
- 8192 census (156/156): frame0 clean, frame1 78% clean, frame2 100% censored.
- 12288 attempt (3 episodes, killed by gate): confirmed L-episodes overflow
  frame-2 at 12,850 tok and frame-3 exists at ~12.9k.
