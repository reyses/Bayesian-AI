# TRIAGE — external Fable red-team of the teacher-student program
**Doc:** 152 · **Date:** 2026-07-24 · **Source:** external Fable consult (owner-run, prompt 2026-07-24) · **Triage:** Claude (Fable, live session) · **Status:** ADOPTIONS RATIFY-PENDING

Meta: owner correctly critiqued the consult prompt (six bundled asks; should have
grounded on the single problem). Next consults: one problem per prompt.

## TESTED IMMEDIATELY — attack survived
**"Your CI is probably wrong (episode-iid overstates precision)"** — CORRECT
methodological attack (our own day-unit rule). Recomputed teacher−never-bail with
DAY-BLOCK bootstrap (25 days, 4k resamples): **[−18.73, −6.39], still
SIGNIFICANT** (vs episode-iid [−22.00, −3.39]). Gen-0 verdict stands, now on the
correct inference unit. All future gate CIs: day-block by default.

## ADOPTED (with actions)
1. **Handbook provenance freeze** — handbook stats must come from a frozen
   pre-period; version + hash the handbook in every run record. ACTION: handbook
   Part-B stats re-provenanced before genome-v1 gate run; hash into ckpt records.
2. **Human-loop transitivity / burned-day registry** — day-burn applies to US:
   days whose results humans have inspected are burned for FINAL claims. ACTION:
   `research/dojo_forge/gate_state/burned_days.json` registry (incl. the 25
   curriculum days + frontier-study days); final claims only on a never-touched
   forward epoch (the conveyor).
3. **Label immutability** — labels append-only; genome updates apply FORWARD
   only. Curriculum relabels are gate diagnostics, never distillation food.
4. **Sim-clock canary test** — MEMO timestamps on sim bar-close time (v2 column);
   IMMEDIATE cheap test: plant future-timestamped canary memos with unique
   tokens, assert never retrieved. ACTION: add to teacher_memory selftest.
5. **Teacher flip-rate noise floor** — measure label flip-rate under seed change
   + memo-order shuffle; gen-over-gen deltas inside the floor are illusory.
   ACTION: flip-rate probe before any gen-1-vs-gen-0 claim.
6. **Anti-freeze reward architecture** (the strongest content): distillation as
   PRIMARY objective (imitation reward = the trustworthy positive per-trade
   term, margin-weighted, churn-capped by teacher's rate) + RL as a small
   KL-leashed correction (max PnL s.t. KL≤ε) + potential-based shaping
   Φ=unrealized-MTM (Ng '99 policy-invariant — densifies gradient, no churn
   incentive). ACTION: becomes the spine of the anti-freeze session doc.
7. **"Gate first" for the student**: execute teacher labels with realistic
   fills; if net ≤0 no reward fixes it (P(label-right)≠P($), already our law).
8. **Frontier-as-instrument** (wildcard, HIGH VALUE): stratify the N=23,378
   frontier by MAE/vol/TOD/entry-type; per-stratum cut-vs-never-bail with
   day-clustered CIs. Outcome A (no stratum): exit alpha absent → teacher
   pivots to entry-veto/sizing. Outcome B: restrict teacher labeling to alpha
   strata only. Either way resolves "are we breeding a judge against noise" —
   the consult's biggest-risk framing. ACTION: research task queued.
9. **Speed**: (a) qwen3 native no-think mode + two-pass triage (escalate only
   in the uncertainty band) = our two-speed design, externally validated, with
   the concrete per-prompt no-think switch; (b) **append-only prompt layout**
   [handbook|genome|day-MEMOs once/episode|bars appended|query] so KV cache
   never busts — WARNING ADOPTED: per-minute mid-prompt retrieval kills the
   cache → memory retrieval moves to once-per-episode position. This REVISES
   the doc-149/151 per-frame retrieval design and shapes E2b.

## ADOPTED — part 2 (asks 2, 4, speed extras)
10. **Licensed-deviation upgrade** (kills the degeneration mode "fluent
    confabulation over noise / predicate-gaming"): deviations must be
    MACHINE-PARSEABLE predicates (direction, magnitude, deadline) logged at
    decision time, AUTO-GRADED by code as log-loss vs the stratum base rate
    from the burned pre-period (near-certain predicates earn ~0); the
    overridden rule's counterfactual logged; deviation edge = realized delta
    vs the rule, day-clustered. **Deviation is a BUDGET, not a right** (k per
    N decisions; expands only on ledger-proven calibration edge with CI>0).
    **Promotion valve**: a rule overridden >Y% of firings with positive edge
    auto-generates a candidate rule from the deviation pattern, tested
    walk-forward — deviations become the mutation operator, formally.
11. **MEMO redesign** (v1's free-text memo = "retrieval lottery"): store
    RESOLVED SURPRISES, not state (state is recomputable from frames).
    Schema-enforced {context-bucket tokens | prediction | outcome}, outcomes
    AUTO-ANNOTATED by code at resolution; two record types: closed
    prediction-error records + one postmortem per episode; CONTROLLED
    VOCABULARY (canonical ATR/ToD/setup bucket tokens) so FTS retrieval is
    deterministic-in-practice and memos double as verified base rates. No
    open hypotheses without deadlines. REVISES doc-149 MEMO v1 before the
    memory pilot runs.
12. **Speed extras**: batch-concurrent episode labeling (offline = no latency
    constraint; fill the GPU with parallel sequences); engine alternative
    ExLlamaV2 noted; **student-routed active labeling** once ~50k labels
    exist (teacher labels only student-uncertain/disagreement frames) — the
    endgame: stacked >10x. Queued as E3b (batching) + program-level item.

## REJECTED / DEFERRED
- Nothing rejected outright; sim-clock v2 column deferred to memory v2 (v1's
  prior-day-only granularity already blocks the same-day leak class).

## Owner asks
1. Ratify the adoptions (esp. #9's retrieval-placement revision to docs 149/151).
2. Frontier stratification (#8): priority vs memory pilot?

## Cost-model spec (owner directive, 2026-07-24 — for the student reward + gate-first fills)
- **Commission: fixed $1.00 per round-trip** (simplified from the confirmed $0.78;
  conservative).
- **Slippage/friction: sampled WITHIN the range of the NEXT 1s bar after the
  decision** (DATA/ATLAS/1s) — fill uncertainty bounded by actual immediate
  market movement, data-driven rather than assumed.
- **Determinism guard (mandatory):** the sample is seeded per-trade
  (hash(episode_id, bar_index)) so identical runs produce identical fills —
  random-in-distribution, reproducible-in-run. Uniform within [low, high];
  adverse-side fill is the pre-registered STRESS variant.
- Applies to: (a) the student's PnL/KL-correction layer; (b) doc-152 #7
  "gate-first" realistic-fills execution of teacher labels.
