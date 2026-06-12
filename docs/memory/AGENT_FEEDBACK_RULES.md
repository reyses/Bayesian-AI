# Consolidated Agent Feedback Rules


## feedback_5s_inherently_noise.md
---
name: 5s level is inherently noise â€” segmentation substrate, not prediction target
description: At 5s/30s scale, market action is dominated by stochastic micro-fluctuations; predictions should anchor at measure (15s) or coarser, not note (5s)
type: feedback
---

# 5s LEVEL IS INHERENTLY NOISE

User direction (verbatim, 2026-05-10 morning): "remember it might not be useful
to go down to 5s TF since it is inherently noise."

Confirmed empirically the same morning: when reclassifying the 9,561
NOISE-after-STEEP_LINEAR_DOWN cell at note level against the full 20-primitive
SeedPrimitiveLibrary, **53.8% remained NOISE residual** even after broadening
the template library from 13 to 20 shapes. P(fwd_up) was also uniform across
all sub-shapes (~0.31-0.45) â€” inner geometry didn't carry predictive content,
only the parent measure shape did.

## Rule

The 5-level segmentation hierarchy provides note (5s) as a SEGMENTATION
PRIMITIVE â€” it produces the chord vector that conditions Bayesian-table
lookups â€” but predictions should NOT be anchored at the note level.
Instead, anchor predictions at:
- **measure (15s)** â€” first level where simple-shape primitives carry
  signal that survives sub-shape decomposition
- **sub_motif (1m)** â€” most prediction-rich level (e.g.
  STEEP_CONCAVE_UP within STEEP_LINEAR_UP at sub_motif â†’ 68.4% UP, n=321)
- **motif (5m)** â€” strategic-context level (FLATLINE-after-rally â†’ 74.6%, n=57)
- **phrase (15m)** â€” day-shape framing (low n, wide CIs)

## Why

At 5s scale, MNQ price moves ~0.25-1.0 ticks per bar; rolling 30s std
is dominated by tick-quantization and microstructure noise rather than
trader-driven directional flow. A note's "shape" at this scale is
mostly random walk over a smooth parent context. The 50%+ NOISE-residual
rate confirms there's no consistent geometric primitive to extract.

## How to apply

- **DO** use note-level shape as a chord component when looking up
  conditional probabilities at coarser levels
- **DON'T** train classifiers, fit models, or expect direction
  predictability at 5s/30s horizons
- **DON'T** quote tight CIs from large-n cells at note level (n=9,539
  CI=0.005) as if they imply predictability â€” they reflect ABUNDANCE
  of NOISE, not predictive structure
- **DO** treat "sub-shape distribution within a NOISE bucket is uniform
  across forward-direction" as evidence that the cell is actually about
  parent context, not inner shape

## Implication for substrate use

Drop the note level from prediction tables but keep it for chord
construction. The Bayesian-table lookup at-bar should be keyed by:
    (measure_shape, sub_motif_shape, motif_shape, phrase_shape)
NOT by note_shape directly. The note is a measurement, not a forecast.

When inspecting a "huge n / tight CI" finding at note level, the answer
to "does it survive OOS?" is less interesting than the answer to "does
the same finding hold at measure or sub_motif level?" If the parent
level shows the same effect, the parent is doing the prediction. If only
the note level shows it, it's likely a microstructure artifact.

## Related memories

- `memory/project_5level_segmentation_substrate.md` â€” the substrate this
  refines; treat note level as primitive, not predictor
- `memory/feedback_quantile_selection_overfit.md` â€” same large-n /
  tight-CI / fragile-OOS pattern at a different layer



## feedback_analyze_first.md
---
name: Analyze First, Then Plan, Then Execute
description: Never iterate blindly. Always present findings and plan before making changes.
type: feedback
---

## Rule: Analyze â†’ Present â†’ Plan â†’ Execute

When working on system improvements:

1. **ANALYZE** the current results â€” read the data, run the numbers
2. **PRESENT** findings to the user â€” what's working, what's not, why
3. **PLAN** the change â€” propose what to modify and expected impact
4. **EXECUTE** only after the user agrees

Do NOT iterate through code changes without presenting findings first.
Do NOT make multiple edit â†’ run â†’ edit â†’ run cycles without stopping to analyze.

**Exception:** If the user explicitly requests "iterate until we achieve X" â€” then rapid iteration is allowed.

**Why:** Blind iteration wastes time and breaks things. Each change should be informed by data from the previous step. The RCA process (feedback_rca_process.md) requires understanding before action.

**How to apply:** After every run, stop. Read the output. Present what changed and why. Then ask what the user wants to do next â€” don't assume.



## feedback_backup_critical_files.md
---
name: Backup critical files before destructive operations
description: Always copy untracked pkl/checkpoint files before --fresh or any operation that overwrites them
type: feedback
---

Before running `--fresh` or any operation that rebuilds/overwrites checkpoint files, ALWAYS copy critical untracked files to a safe location.

**Why:** Lost the oracle brain's matching pattern_library.pkl when `--fresh` overwrote it. Had to restore from OneDrive version history. The pre_is_backup only captured the NEW files, not the old ones.

**How to apply:**
- Before `--fresh`: copy `checkpoints/pattern_library.pkl`, `clustering_scaler.pkl`, `template_tiers.pkl`, `is_brain_checkpoint.pkl`, `live_brain.pkl` to `checkpoints/backup_YYYYMMDD/`
- Before any pkl-overwriting operation: same
- Name backups with date so they're identifiable
- These files are gitignored (*.pkl, checkpoints/) so git can't recover them



## feedback_base_measurements.md
---
name: Ground features in base measurements (DOE principle)
description: User identified that features are over-abstracted. Every feature must trace to Price/Time/Volume in 1-2 transparent steps. DOE transferability principle from injection molding background.
type: feedback
---

Features must be grounded in base measurements (Price, Time, Volume) with transparent operations.
Multi-layer abstractions (PID of z-score of regression residual) are machine-specific â€” they
encode our tuning constants, not market properties.

**Why:** User comes from manufacturing engineering (injection molding, DOE, rheology). In DOE,
you measure material properties (viscosity, shear rate) not machine readings (output pressure)
because material properties TRANSFER across machines. Same principle: market properties
(velocity, std, position-in-range) transfer across instruments, timeframes, and regimes.
Machine-specific features (F_momentum with kp/ki/kd, ADX with double Wilder smoothing)
break when you change any system parameter.

**How to apply:**
- Every feature must have a one-sentence explanation: "this measures X of Y over Z window"
  where X = statistical operation, Y = Price/Time/Volume, Z = defined window
- If you can't explain it without reading the code, it doesn't belong
- Derivatives are OK at ANY order â€” so long as each step answers a nameable question
  (velocity = how fast? acceleration = is force changing? jerk = is onset smooth or sudden?)
- The PROBLEM is derivatives that compute something nobody can name. F_momentum computes
  a PID on z on regression â€” what market question does that answer? Nobody knows.
- Distribution measures are OK (std, variance â€” they measure SHAPE of base data)
- Cross-derivatives are OK (price Ã— volume â€” combines independent bases)
- When replacing a feature, keep the QUESTION it answered, replace the plumbing
- Test: "would this feature mean the same thing on ES as on MNQ?" If no, it's machine-specific
- The rule is NOT "stay close to base." The rule is "can you NAME what this measures?"



## feedback_blender_first_then_drill.md
---
name: Blender first, then drill down (research methodology preference)
description: User's preferred research approach â€” run unrestricted broad-strokes ("blender") first, observe what surfaces, then drill into surprises. Do NOT pre-narrow before seeing the unrestricted output.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
User's research methodology trademark: **"put it in a blender, then see what happens, before drilling down."**

**Why**: pre-narrowing the feature set, hypothesis list, or analysis scope before
seeing the unrestricted output anchors interpretation on what was already expected.
The interesting findings come from surprises, and surprises only show up when you
include things you didn't predict were relevant. The blender output IS the
substrate for picking what to drill into.

Stated 2026-05-03 in the v2 feature Ã— feature interaction work, when I had
proposed pruning the 23-feature set to 6 family representatives BEFORE running
the within-TF Ã— regime layer (D2). User said: "this is my trademark put in a
blender then see what happens before drilling down" â€” instructing me to run the
full unrestricted version first.

**How to apply**:

1. **Default to unrestricted runs** when the user opens a new analysis dimension.
   Don't filter the feature set, the regime axis, the TF axis, the cell count,
   the hypothesis list, etc., based on what you think is interesting.
2. **Compute the full output, then summarize the headlines.** Let the surprises
   surface from the data, not from your interpretation.
3. **Pruning / optimizing comes AFTER the blender.** Only after the user sees
   the unrestricted output do they decide what to drill into. They may explicitly
   request the optimized version as a second pass â€” that's a comparison, not a
   substitute.
4. **If you must propose a narrowed cut, mention it as Option B alongside the
   full blender as Option A.** Default to the blender when the user gives a
   directional "go" without specifying scope.
5. **The blender philosophy applies to multi-step sequences too**: when the user
   says "do the mirror" or "do all five steps", they generally mean execute the
   full sequence and report aggregate findings, not break to ask for permission
   between steps.

This is consistent with the user's general aversion to pre-loaded interpretation
("don't be a yes-man, challenge ideas, data beats intuition" â€” global persona).
The blender is how data gets to beat intuition: it shows what's there before
intuition gets to filter it.



## feedback_capture_impulse_ideas.md
---
name: Impulse ideas go to the backlog, not the build queue
description: When the user fires a new idea mid-session, capture it as a TODO with full context, return to it at the right phase. Don't context-switch the active build path on every impulse.
type: feedback
---

**Rule** (locked 2026-05-09 evening):

When the user introduces a new idea or direction during an active build,
the default is: **capture it as a TODO with full context** and return
to the current foundation work. Switching the active build on every
impulse fragments the foundation and means none of the ideas get
properly grounded.

**Why**: every idea sounds urgent in the moment, but the value of an
idea depends on what's already built. A "what if we use CRM as the
substrate for geometric primitives?" idea can't be built well until
the underlying probability table substrate is solid â€” otherwise we
build geometric primitives on top of unverified foundations and have
to redo the work.

**How to apply**:

1. When user fires a new idea, first sentence: did they ask me to
   build it now, or did they think out loud?
2. Default reading: "thinking out loud" â€” capture and continue.
3. If user wants it built now, they'll say so explicitly ("let's
   do it", "build it", or interrupt with another action).
4. The TODO entry must capture:
   - the idea ITSELF (one-line summary)
   - the CONTEXT (why it came up, what triggered it)
   - the PRECONDITIONS (what foundation needs to be solid first)
   - the COST estimate (rough; small / medium / large)
   - the TRIGGER for revisit (when in the build phase to come back)
5. Acknowledge the idea was captured so the user knows it isn't lost.

**Why this matters more than it seems**: in an auto-mode session
the impulse-to-execute is biased high (the agent wants to build).
That bias makes the agent context-switch into every new direction,
which produces a fragmented session where 5 things are half-done
and nothing is shipped. Capturing first and building second
preserves session focus.

**Counter-rule**: If the new idea is BLOCKING the current build
(e.g., reveals an error in the substrate), drop everything and
address it. The capture-first rule is for ADDITIONS, not
CORRECTIONS.

**Oshit-moment override** (user-named exception, locked 2026-05-09):

User is self-aware about chaotic-neutral / AuDHD impulse-firing
tendencies. The dual rule below is designed for that working style:

  - Default = capture (so ideas don't fragment the session)
  - Oshit pivot = allowed when I (Claude) judge it necessary, but
    pivoting MUST preserve the link between current work and the
    new direction.

**When I judge "Oshit moment"** and pivot:

| trigger                                                  | Oshit? |
|---------------------------------------------------------|:-----:|
| New idea reveals error/bias in current substrate         | YES   |
| Current work would be invalid without addressing it      | YES   |
| Current work is solving the wrong problem               | YES   |
| New idea adds value but doesn't invalidate current work  | NO    |
| User wants to chase a curiosity right now                | NO unless explicit |
| Current work is nearly done; new idea is small extension | NO (finish first) |

Examples this session:
- regime_labels_2d circular bias â†’ YES, pivoted (correct call)
- z_high V2 column was wrong metric â†’ YES, pivoted (correct call)
- "what if we use CRM for geometric primitives?" â†’ NO (captured)
- "what about psychohistory framing?" â†’ NO (added vocabulary, didn't pivot)

**The non-negotiable when we DO pivot â€” preserve the connection**:

When pivoting on an Oshit moment, the response MUST cover:

1. WHAT the current work was, where it stood (so we don't lose it)
2. WHY the new direction supersedes it (the Oshit reasoning)
3. HOW the new work connects to / replaces / completes the current
4. WHAT becomes of the in-flight artifacts (kept as research artifact?
   trashed? superseded by the new build?)
5. WHEN (if ever) we resume the original

Without #3 we death-spiral: jump to new shiny thing, lose context, the
new thing also fragments before completion, repeat. With #3 we have
a thread to pull back to even after multiple pivots.

This is the stabilizer. The user FIRES ideas; my job is to be the
connector that keeps them threaded together so we ship something.

## Examples

GOOD (capture):
- User: "what if we use CRM for geometric primitives?"
- Me: capture as TODO with context, continue current work.

GOOD (act):
- User: "you're using regime labels which we found are flawed"
- Me: stop, fix the substrate, then continue.

BAD (act on impulse):
- User: "what about psychohistory style modeling?"
- Me: spends 600 lines building a "psychohistory framework" mid-session
  while the probability table substrate is still half-built.



## feedback_challenge_harder.md
---
name: Challenge ideas harder
description: User wants pushback on proposals, less agreement, more critical analysis
type: feedback
---

Challenge the user's ideas more aggressively. Don't be complacent or agree too easily.

**Why:** User explicitly asked "how can we change your persona to challenge my ideas, be less complacent." They want a collaborator who pushes back, not a yes-man. Trading system decisions have real money consequences â€” rubber-stamping bad ideas is worse than being annoying.

**How to apply:**
- When the user proposes a change, identify at least one risk or counterargument before agreeing
- Ask "what happens when this fails?" and "what's the worst case?" before implementing
- If data doesn't support an idea, say so directly â€” don't soften it
- Propose alternative approaches when you see a better path
- Flag when a change might break something downstream
- Don't add "but your approach works too" after a pushback â€” commit to the disagreement



## feedback_challenge_ideas.md
---
name: Challenge ideas that don't make sense
description: User wants Claude to push back on proposals that have flaws, not just agree
type: feedback
---

Challenge and critique ideas the user provides if they don't make sense technically.
Don't just agree â€” flag dimensionality issues, computational concerns, or logical gaps.

**Why:** User values honest technical debate over yes-man agreement. Bad ideas that go unchallenged waste implementation time.
**How to apply:** When user proposes a technical approach, evaluate it critically before implementing. Flag specific concerns with reasoning. Still respect user's domain expertise â€” challenge the method, not the goal.



## feedback_checkpoint_every_step.md
---
name: Checkpoint every step
description: All multi-step pipelines must save intermediate results to disk after each step for crash recovery
type: feedback
---

All multi-step pipelines must save intermediate data to disk after EVERY step.

**Why:** Long-running pipelines (signal collection, L3 training data) can take hours. A crash at step 5 of 6 loses everything if steps 1-4 aren't persisted. The user lost time to this.

**How to apply:** Every step in a pipeline must:
1. Check if cached result exists on disk â†’ load and skip if so
2. Compute the result
3. Save to disk (`.npy` for arrays, `.pkl` for complex objects, `.pt` for models)
4. Print "Saved: {path}" confirmation

Use pattern: `if os.path.exists(cache_path): load; else: compute + save`

**Shard, don't monolith:** Large datasets (signals, training samples, features) must be saved as shards (per-day or per-month files in a subdirectory), NOT as one giant file. Benefits:
- Crash during collection only loses the current shard, not everything
- Resume picks up from the last completed shard
- Memory-efficient loading (can stream shards)
- Pattern: `cache_dir/signals/2025_01_02.pkl`, `cache_dir/signals/2025_01_03.pkl`, etc.

This applies to: signal collection, training data building, model training, feature extraction â€” anything that takes > 30 seconds.



## feedback_chop_edge_regime_filter.md
---
name: Chop-Edge Discovery & Regime Filter (v1.5-RC)
description: The zigzag counter-trend strategy has +$89/day on chop days and -$95/day on trend days. Forward-available rule (prior_range + range_compression) discriminates with d_OOS=+0.77/+0.78. Filter converts -$552/95-day net into +$5,000-6,000.
type: feedback
date: 2026-04-27
originSessionId: bb5b3851-d849-49aa-9f93-bcd7b0dc113f
---
## Insight

**The strategy is a CHOP SPECIALIST, not a trend follower.**

User framing (2026-04-27): *"trend days are the easy ones â€” chop is what
everyone avoids and we thrive."* This insight inverted the entire optimization
direction:

- Stop trying to make zigzag work in all regimes
- Start identifying the regimes where it DOES work and skipping the rest

## Empirical evidence

NT8 backtest of v1.0.x counter-trend over 1/2-4/24/2026 (1,678 trades, 95 days):
- Net: âˆ’$552 (looks broken)
- **Working window 1/2-2/26**: 46 days, +$4,096 (+$89/day, +$5.74/trade)
- **Bleed window 2/27-4/24**: 49 days, âˆ’$4,648 (âˆ’$95/day, âˆ’$4.82/trade)
- Inflection at exactly 2026-02-26 (= MNQ regime change from chop to strong
  bull rally)

## Forward-available regime classifier

Two features discriminate BLEED vs HARVEST days with walk-forward stability:

| Feature | d_IS | d_OOS | Source |
|---|---|---|---|
| `prior_range` | +0.576 | +0.774 | yesterday's daily H-L |
| `range_compression` | +0.475 | +0.782 | prior_range / 20d_mean_range |

Both POSITIVE: BLEED days follow LARGE-range yesterdays AND yesterdays whose
range was BIG vs the 20-day baseline. The chop edge is in the QUIET aftermath
of quiet days, not after volatile events.

## Rule

```
bleed_score = z(prior_range) + z(range_compression)
trade_today = (bleed_score <= -0.34)   // skip top-50% bleed-scored days
```

IS-calibrated (1/2-3/1/2026, N=48 days):
- MEAN_PRIOR_RANGE = 385.32, STD_PRIOR_RANGE = 219.83
- MEAN_RANGE_COMPRESSION = 1.0315, STD_RANGE_COMPRESSION = 0.5502

## Validation results

Threshold sweep on 1,678-trade ledger:
- z=âˆ’0.5: 39 days kept, +$5,021 net, $129/day on kept
- **z=âˆ’0.34 (MVP default): 50% skip, OOS-validated, $6,202 OOS lift**, 82% bleed catch
- z=+0.75: 67 days kept, +$5,214 net (biggest aggregate)
- z=0.0: AVOID â€” empirical local minimum on the threshold curve

**Strategy goes from âˆ’$552 to +$3,977-$5,214 across all reasonable thresholds.**

## Methodology lessons

1. **Start with daily features**, not multi-TF, when building day-level
   regime classifiers. Multi-TF feature extension (ATLAS 1m â†’ 5m, 1h
   variance ratios) added zero OOS lift over the simple 2-feature rule.
   Adding 3 more features dropped OOS lift from +$5,084 to +$2,094 â€” pure
   overfitting.

2. **Cohen-d walk-forward shortlist**: only keep features with sign(d_IS) ==
   sign(d_OOS) AND min(|d_IS|, |d_OOS|) >= 0.30. Two features met this bar
   with strict-forward-only data.

3. **z=0.0 is NOT the sweet spot**. The threshold curve has a literal local
   minimum at zero. Either tighten (z=-0.5, conviction) or loosen (z=+0.75,
   aggregate) â€” both beat the middle.

4. **Tier-day-classifier methodology generalizes**: same Cohen-d + IS/OOS
   walk-forward approach worked on the 79D ML pipeline (RIDE_AGAINST,
   2026-04-18) AND on the NT8 trade-export simple-feature space. Different
   domain, same statistical structure.

## Implementation

Spec: `docs/JULES_v15_chop_specialist.md`
Validation tool: `tools/v15_filter_apply.py`
Classifier tool: `tools/nt8_bleed_harvest_classifier.py`

NT8 implementation: copy v1.4-RC, replace `MaxMeanRange5dPts` with the
combined-z bleed-score logic. ~50 LOC change. Inherits all v1.4-RC risk
machinery (DRM trail, StagnationMonitor, missed-breach handler).

## Open questions

1. Does the rule generalize across MNQ contract rolls? Tested only inside
   the 06-26 contract.
2. Is the chop-edge specific to MNQ or does it apply to ES/NQ/YM?
3. R sensitivity: tonight's analysis was at R=50. Re-run at R=30 to confirm.
4. Hour-of-day mask: secondary in-sample filter shows additional +$2,194 lift
   on filter-pass days. Validate on hold-out before deploying.
5. The **bull regime drift** (+$247/day passive long over the 32-day NT8 dump
   window) is FORFEITED on filter-skip days. An optional `OnFilterSkipDay
   = AlwaysLong` mode could harvest the drift. Defer.



## feedback_chop_velocity_range_regularity.md
---
name: Multiscale chop â†’ higher-TF velocity/range regularity
description: User-identified pattern from 2026-05-03 regime-stratified TF sweep. Chop at lower TF reliably WEAKENS signed velocity at higher TF for directional regimes (UP/DOWN). Range behavior is regime-asymmetric.
type: feedback
date: 2026-05-03
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The pattern

User: "a lot of variation in one TF means that the higher TF was weaker velocity and more bar range".

**Why:** The user's reading of the multiscale-character table from `v2_features_regime_stratified_tf_sweep.py`. The pattern is real but regime-asymmetric.

## How to apply

When characterizing a TF's signal in the context of higher TFs, do NOT
assume "chop at low TF = trend at high TF". The empirical relationship is:

- **Chop at lower TF reliably WEAKENS signed velocity at the higher TF for directional regimes** (UP and DOWN). The intraday noise dilutes the directional component into the macro window.
- **Range expansion under chop is regime-dependent**:
  - Non-directional underlying (UP_*, FLAT_*) â†’ chop EXPANDS range (intraday two-way action mechanically widens bars)
  - Directional clean sell (DOWN_SMOOTH) â†’ SMOOTH version has WIDER range than CHOPPY version. Clean sells expand range further than chopped sells.

## Empirical numbers (1h cohort, IS-only, full year)

| Regime pair | 1h vel SMOOTH â†’ CHOPPY | 1h range SMOOTH â†’ CHOPPY |
|---|---|---|
| UP | 16.9 â†’ 14.7 (âˆ’13%) | 70.7 â†’ 80.1 (+13%) |
| DOWN | 20.4 â†’ 12.1 (**âˆ’41%**) | 89.6 â†’ 80.2 (âˆ’10%) |
| FLAT | 9.6 â†’ 10.6 (+10%) | 59.5 â†’ 81.0 (**+36%**) |

The strongest velocity-weakening effect: DOWN_SMOOTHâ†’DOWN_CHOPPY at 1h (âˆ’41%). The strongest range expansion: FLAT_SMOOTHâ†’FLAT_CHOPPY (+36%).

## Implications for signal design

- A high lower-TF chop signal is a hint that the higher-TF velocity will be diluted (especially for sells). Don't expect macro-trend follow-through in days that started choppy at the bar level.
- Range at the higher TF is NOT a reliable indicator of underlying regime: large 1h range can be either DOWN_SMOOTH (clean sell) or FLAT_CHOPPY (chop with no direction). Need to combine with velocity sign or direction_axis.
- The regime-conditional composite framework already established (target sign depends on modifier quantile) gets supporting evidence here: bar_range alone gives correlation with forward return ranging from -0.157 to +0.235 across regimes (range 0.39).

## Connection to other findings

- The 9-layer EDA stack already established that contextualization is real and that compositions need to be conditional. This pattern adds a specific multiscale regularity that informs which composite to choose.
- The chord finder identified 4h-TF chord cells with 100% regime purity. Combined with this regularity, it suggests a layered signal:
  1. Identify daily regime via 4h chord cells
  2. Within each regime, use the conditional rules (modifier quantile flips target sign)
  3. Calibrate trade horizon to the velocity-weakening pattern (don't over-extend into 1h hold when the lower TF is choppy in DOWN regime)

## Source

- Tool: `tools/v2_features_regime_stratified_tf_sweep.py`
- Output: `reports/findings/v2_features_regime_tf/multiscale_character.csv`
- Commit: `314e1072` (2026-05-03)
- Journal: `docs/daily/2026-05-03.md` (regime-stratified TF sweep section)



## feedback_ci_pseudoreplication_effective_n.md
---
name: feedback-ci-pseudoreplication-effective-n
description: outcome CIs must count unique market trades / days, not agent-votes or correlated samples (effective-N, block-bootstrap)
metadata:
  type: feedback
---

The unit of independence for an **outcome** confidence interval is the **unique market
trade**, not the number of measurements. Found 2026-06-08 in research_A's
`evaluate_is_mastery_gate` (`training/rl_engine/train_gpu_research_A.py`): it used
`stderr = std/sqrt(raw_N)` while up to 128 agents share one network â†’ deterministic
greedy â†’ byte-identical trades â†’ `raw_N >> effective_N`. The code even computed
`effective_N = len(set(metadata))` then ignored it. Result: CI ~âˆš(raw/eff)Ã— too
narrow â†’ the automated mastery gate passes on noise and the curriculum advances
prematurely.

**Why:** duplicating a measurement adds zero information. A diverse agent fleet helps
a CI only by producing more *distinct* trades (coverage), never by re-voting the same
one. Same-entry/different-exit trades are also NOT independent â€” they ride one realized
price path (correlated). In a historical backtest there is **one fixed path**, so the
honest independent block is ultimately the **day**.

**How to apply:** for any $/edge significance claim or gate, dedup/cluster to unique
trades (or unique entries), and prefer **block-bootstrap by day** â€” which is exactly
what the canonical $/day metric already does (4,000 resamples over days). Never divide
by a raw count of correlated/duplicated samples. This is the agentic-RL instance of
the existing metric-definitions CI rule. See [[reference_fista_gpu_cv_step_bug]] for
the other 2026-06-08 shared-math finding. Gate fix flagged but not yet applied.



## feedback_cli_script_false_orphans.md
---
name: feedback-cli-script-false-orphans
description: Standalone CLI scripts ("python path/to/x.py â€¦") do NOT show up in Python-import grep but ARE active dependencies. Always grep for the bare filename in docs/, Jules_instructions/, config/, and as `subprocess.run([â€¦])` literals before deleting.
metadata:
  type: feedback
---

**Rule**: Before deleting any `.py` file flagged as "no Python imports," verify it isn't a standalone CLI script by checking ALL of these:

1. `python <path/file.py>` or `python -m <module>` invocations in `docs/`, `Jules_instructions/`, `config/`, and tool docstrings.
2. `subprocess.run([â€¦, '<path>', â€¦])` or `[sys.executable, '<path>', â€¦]` patterns in live code.
3. Inside-function imports (`from X.Y import Z` inside `def â€¦`) â€” grep tools don't find them in cross-file dependency maps.
4. `--help` or `--example` references in README/spec docs.

**Why:** This failure mode hit four times in one cleanup session (2026-05-24):
- `training/ticker.py` â€” used by `run.py:158` (inside-function `from training.ticker import FileTicker`).
- `training/report.py` â€” used by `run.py:708,1149` (inside-function imports).
- `training/build_dataset_v2.py` â€” invoked via `python â€¦` from `docs/JULES_standalone_research_v2.md:14` and other docs; no Python importers.
- `training/train_pivot_cnn_v2.py` â€” invoked via `python â€¦`; sole "caller" was the docstring `Usage:` block.

Each restoration cost ~5 minutes. Doing the four-check audit BEFORE deletion would have cost ~2 minutes total. Net loss: ~13 minutes plus the user's confidence.

**How to apply:**
- After running a dependency-map agent, run a SECOND pass that greps the bare filename (without path) across `docs/`, `config/`, `Jules_instructions/`, and live code's `subprocess` calls.
- For any file you're about to delete, do the same grep yourself â€” don't rely solely on the agent.
- When a file has a `Usage:` block in its docstring with `python <path>` lines, that's a tell: it's a CLI script and likely has docs/config callers the import-grep missed.
- The `_v2`/`_v1` suffix is ALSO a tell: these are usually parallel CLI scripts targeting different feature schemas, not unused junk.

**Related**: see `feedback_v2_only_hard_rule.md` (the V2-only rule that made over-zealous deletion appealing in the first place) and `docs/daily/2026-05-24.md` section 9 (the build_dataset_v2 restore record).



## feedback_cnn_fragility.md
---
name: CNN Training Fragility
description: CNN model is fragile â€” seed-dependent, asymmetric loss kills it, never change two things at once
type: feedback
---

**Rules for CNN training changes:**

1. **Seed=42 is load-bearing**: model gives 0 trades with different seeds. The edge is real but fragile.
2. **Asymmetric loss kills the model**: PnL-weighted asymmetric penalty made model predict 0.5 for everything (0 trades). Symmetric magnitude-weighted BCE is the only loss that works.
3. **Never change two things at once**: removing seed AND changing loss simultaneously made debugging impossible. One variable at a time.
4. **22D features are dead**: regime drift (FM 4.82x, volume 5.51x scale shift) killed them. Features oscillated between +$24K and -$24K.
5. **3D raw features (50 layers) = 0 trades**: model can't discover DMI from raw price in 30 epochs. "The features we grounded ourselves IS the value we add."

**Why:** User called out sloppy work: "this type of stuff is the reason i cant sleep your bein to sloopy". Multiple regressions from careless simultaneous changes.
**How to apply:** When modifying CNN training, change ONE thing, run, verify, then change the next. Always keep seed=42. Never try asymmetric loss again.



## feedback_daily_hourly_review.md
---
name: Daily and Hourly Review Rule
description: Never evaluate system performance by month â€” always by day and hour. Live money can't wait for monthly convergence.
type: feedback
---

## Rule: Analyze by day and hour, not by month

Monthly aggregates hide the truth. A system that makes $5K/month but has -$1,440 days is not tradeable with a small account.

**Why:** When running live with real money, we don't have the funds to wait until the monthly average converges. Each day and each hour must be independently viable.

**How to apply:**
- Every forward pass result: show per-day breakdown with mode, not just totals
- Identify the MODE of daily PnL (the most common daily outcome)
- If the mode is negative or near zero, the system loses money on most days
- A system with 15/19 green days but mode near $0 is not the same as 19/19 green days with mode $100
- Design daily stop losses and hourly pause rules for live trading
- Target: 20-50 trades/day (quality), not 500+ (noise)



## feedback_data_validation_first.md
---
name: Data validation before modeling
description: Always validate data quality against ground truth BEFORE training any models or running analysis
type: feedback
---

Data validation must be the FIRST step, not an afterthought.

**Why:** We trained multiple CNN models (29D, trajectory, direction) and ran oscillation research on ATLAS data that contained corrupted bars (fake highs/lows from aggregation errors). Bad data produces bad labels which produce bad models. The 1m bar with a fake high of 24628 (actual was 24429) was only caught visually on a chart.

**How to apply:**
1. Before ANY model training or analysis, run `python tools/validate_data.py`
2. After any data update (NT8 export, upsampling, ATLAS merge), re-validate
3. The 1s data is ground truth â€” all higher TF bars must have OHLCV within 1s range
4. After validation fixes, delete cached features (`.npy`) and rebuild
5. Add validation as step 0 in every pipeline that touches ATLAS data

**Rule:** No model trains on unvalidated data. No analysis runs on unvalidated data. Period.



## feedback_direction_clf_alone_fails.md
---
name: feedback-direction-clf-alone-fails
description: Direction classifier alone is not a live strategy â€” entry timing is the unsolved bottleneck; tick-exact exits unmask the gap
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User pivoted to KPI-driven autonomous mode (target: $100/day NET, low MAE, high Day WR). Built a clean direction-classifier strategy through the existing `training_iso_v2/` ticker+engine pipeline. **All 72 grid configurations (symmetric and asymmetric R/R Ã— thresholds Ã— cadences) fail OOS 2026.**

**Best surviving config**: T=0.95, TP=$20/SL=$5, 15m cadence â†’ +$2.54/day NET, CI [âˆ’$4.22, +$10.34]. CI crosses zero â€” not statistically significant per CLAUDE.md mandate.

**Why it fails**: Direction accuracy 87% at the daisy oracle bars does NOT translate to TP-hit rate 87% when firing at every 15m close. Oracle bars are hindsight-selected as the START of favorable moves. Most cadence-trigger fires are mid-move or end-of-move where the remaining favorable distance is small. TP=$20 rarely hits, SL clipped on noise reversals.

**Critical engine bug discovered**: The default `HardStop`/`TakeProfit` exits in `training_iso_v2/exits.py` close at `state.price` (5s bar close) after a threshold cross. If price overshoots TP/SL intrabar, the exit fires but the close price can be much further from entry than the threshold. This INFLATES winners AND losers, producing apparently strong $/day with poor Day WR (high variance from intrabar overshoot).

**Fix shipped**:
- `training_iso_v2/exits_tick_exact.py` NEW â€” `TickExactTP` / `TickExactSL` use 5s OHLC high/low to detect intrabar threshold crossings, write the exact threshold price to `position.extras['_force_exit_price']`
- `training_iso_v2/engine.py` PATCH â€” engine `_tick` honors `_force_exit_price` if set
- `training_iso_v2/ledger.py` PATCH â€” `ClosedTrade.trough_pnl` field tracks MAE per trade

**How to apply**:
- **Never run backtests with the default HardStop/TakeProfit again** â€” always use `TickExactSL` / `TickExactTP` from `exits_tick_exact.py`. The historical numbers from the old pipeline are inflated.
- **Direction classifier â‰  live strategy.** Use it as a FILTER on existing tier strategies (FADE_CALM, CASCADE, MA_ALIGN, etc.) that have proven entry timing. The classifier vetoes signals that conflict with its direction call.
- **Don't try to "fix" the classifier by tweaking TP/SL** â€” the asymmetric R/R grid (TP=$10-40, SL=$5-10, T up to 0.95) confirmed no config survives. The bottleneck is entry timing, not exit policy.
- **The right next step is entry timing**: either a separate model that predicts "is this an oracle moment?", or a price-action trigger (breakout, pullback, sweep) that the classifier then routes.

**What's validated**:
1. Infrastructure works (ticker â†’ engine â†’ exits â†’ ledger â†’ bootstrap CI)
2. Intrabar overshoot bug is fixed
3. The earlier `2026-05-16_forward_pass.md` $50-100/day numbers held BECAUSE the daisy oracle gave entry timing for free. Without that, the classifier alone produces zero.

Connected: [[feedback-leadin-pca-rejected]], [[feedback-scenario-lstm-information-ceiling]], [[project-regret-six-layer-architecture]] (L4 selector still missing), [[user-collaboration-protocol]] (autonomous KPI iteration is a valid mode).



## feedback_dollar_lift_framing.md
---
name: $/day lift framing
description: How to frame proposed $/day improvements -- against the honest floor, not the inflated headline, and weight tail-risk reduction.
type: feedback
originSessionId: f4f2fb74-6511-49b0-a189-e2611540bf39
---
When proposing a feature that adds $X/day, the user expects the impact
framed against the HONEST FLOOR (the realistic deployment number after
caveats), not the optimistic headline.

**Why**: The deliverable's headline OOS PnL was $927/day, but after the B7
distribution-shift caveat the honest floor is $600-700/day. I described a
proposed +$200-500/day day-regime sizer as "$200-500/day, not
transformational" -- anchoring on $927. User correctly pushed back: on the
honest $600-700 floor that's +30-70% revenue, ~$50-125K/year. That IS
transformational by any reasonable definition.

**How to apply**:
1. State the HONEST FLOOR first, then compute the lift as a percentage of
   that floor -- not the inflated headline.
2. Translate $/day to $/year (Ã—250) to make the scale visible.
3. Weight tail-risk reduction separately from mean lift. Filtering 3 of 5
   negative OOS days isn't just $300/day arithmetic -- it's eliminating
   the drawdown days that cause humans to pull the plug, plus it lets the
   strategy size UP confidently on predicted-good days.
4. Don't dismiss "incremental" lifts when the strategy is real-money and
   tail-risk-sensitive. Trading systems sustain on sequence-of-returns
   risk reduction, not just mean lift.
5. Don't lead with skepticism when an improvement compounds with existing
   rails (per-leg Ã— per-hour Ã— per-day sizing -- multiplicative). Each
   layer's lift compounds with the others; the day layer isn't an add, it's
   a scalar.



## feedback_dont_pivot_on_suggestions.md
---
name: feedback-dont-pivot-on-suggestions
description: "User process rule â€” suggestions are candidates for the NEXT iteration, not immediate pivots. Only pivot on genuinely great ideas."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User explicitly stated 2026-05-17: "we have a rule for when i sugest stuff, it is not to pivot immidatly it is to think about it for next try, unless it is a very good idea".

**Why:** I was killing running experiments and rebuilding strategies the moment user threw out an idea (e.g., pivoted from running raw trend3 forward pass to hand-rolling DMI smoother the second user mentioned DMI). This wastes compute, breaks the working flow, and prevents the user's idea from being evaluated cleanly because we never finish the prior experiment for comparison.

**How to apply:**
- When the user suggests an architectural change or new approach DURING an in-flight experiment: DO NOT stop the experiment. Let it complete so we have a baseline. Note the suggestion in todos as a candidate for the NEXT iteration.
- When the user suggests something AFTER an experiment finishes: same â€” finish writing up findings before pivoting.
- Exception: a genuinely excellent idea that obsoletes the current direction. Use the critical-collaborator persona to judge â€” "is this good enough that the running experiment is wasted?" Almost always: no, finish it first.
- After noting a suggestion, ask if it's the next priority or just a backlog item.

**Concrete example (2026-05-17):**
User said "what we can do is a DMI like approach" while raw Trend3 forward pass was running. I immediately stopped, hand-rolled a DMI smoother, queued a smoothed grid. User then said "it should be integrated into the ML directly via LSTM" â€” invalidating the hand-rolled DMI work. If I had let the raw run finish first, we'd have had a baseline AND would have heard the LSTM clarification before doing the DMI rebuild.

Connected to: [[user-collaboration-protocol]] (topic-at-a-time when designing; this is the inverse â€” don't switch topics mid-execute).



## feedback_flat_pipeline_cross_param.md
---
name: flat-pipeline-cross-param
description: The FLAT hardened-leg forward pass is a hindsight-clean partition â€” its $/day is monotonic in zigzag subdivision and CANNOT be used to compare across ATR / pivot-density parameters.
metadata:
  type: feedback
---

The FLAT (no-ML) hardened-leg forward pass â€” enter every offline-zigzag leg at
its R-trigger, exit at the next pivot â€” must NOT be used to compare across the
ATR multiplier, or any parameter that changes the zigzag's pivot density.

**Why:** the offline zigzag's legs are a SEQUENTIAL PARTITION of the price path
(see [[midleg-entry-research]]: 99.8% of consecutive-leg gaps = 0). Every leg is
a genuine pivot-to-pivot swing â€” zero whipsaws by construction, because the
offline detector places pivots with knowledge of the whole day. A causal engine
flips at confirmations, many of them false, and whipsaws. As the ATR multiplier
(â†’ r_price) shrinks, the offline pass finds ever more clean swings while the
causal whipsaw cost it never pays explodes. The 2026-05-21 sweep
(`tools/atr_multiplier_sweep.py`) showed FLAT OOS $/day rising MONOTONICALLY
from âˆ’$167 (ATRÃ—10) to $3,480 (ATRÃ—1 â€” 100% winning days, PF 5.25), no interior
peak. That is the fingerprint of a metric scaling with zigzag SUBDIVISION, not
tradeability.

The X=4 FLAT baseline ($690/day IS, $454/day OOS) is ~trustworthy ONLY because
at a coarse threshold offline â‰ˆ streaming (the validated ~98% match). The whole
L5 stack is validated at a fixed X=4, where the oracle inflation is a constant
that cancels inside deltas.

**How to apply:** FLAT hardened-leg $/day is valid as (a) a fixed-X baseline and
(b) the denominator of a fixed-X delta (B7 / B9 / B10 lift). It is NOT valid for
cross-X or any cross-pivot-density comparison. To compare ATR multipliers (or
any pivot-density parameter) you must use a CAUSAL streaming forward pass that
actually pays the whipsaw cost â€” only that has a real interior optimum.
`tools/atr_multiplier_sweep.py` is fine for fixed-X use; its cross-X verdict is
annotated invalid in both the script and the report.



## feedback_high_vol_harness_failed.md
---
name: High-vol "harness as profit" angle failed
description: 2026-05-05 â€” tested two ways to convert high-vol bleed into profit (direction flip, vol-adaptive exits). Both rejected. Reveals state-exit leak as the actual mechanism.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The rule

When loser autopsy shows a "bleed zone" tied to entry volatility, the
intuitive levers (flip direction in high-vol, scale exit thresholds with
vol) DON'T work. Both fail because:

1. **Peaks are symmetric across direction in high-vol.** Both fade and
   flip directions reach similar peak amounts in Q5 vol. Flipping doesn't
   capture more.
2. **Peak distributions are fat-tailed.** Mean peak in Q5 was $144, but
   the typical (median) trade doesn't reach $144. Threshold formulas
   based on means or q_30 quantiles overshoot â€” wider TPs miss most
   trades; later giveback arming never fires.

## Evidence (2026-05-05 V2-native NMP)

Tested via re-simulation on 19,106 IS + 4,495 OOS NMP trades binned by
`L2_1m_vol_mean_15` quintile.

**Vol-flip hypothesis (rejected):**
- Q5 fade_peak $144 vs flip_peak $151 â€” 5% advantage, not real
- Counterfactual flip on Q5 only: total IS $3,419 â†’ $866 (worse)

**Vol-adaptive exit thresholds (rejected):**
- Per-bin Bayesian-derived TP/SL/gb_min/gb_keep
- Q5 thresholds: tp=$51, gb_min=$70 (vs prod tp=$26, gb_min=$41)
- IS delta vs prod: -$75.51/day (CI [-$101, -$50])
- OOS delta vs prod: -$112.41/day (CI [-$170, -$59])
- Q5 specifically: prod +$4.68/t, vol-adapt -$0.71/t (delta -$5.39/t)

## Reveal: the actual mechanism is state-exit leak

Production thresholds via re-simulation show **+$4.68/trade in Q5**.
The actual engine produced **~$0.33/trade in Q5**. The gap (~$4.35/trade)
is from state-driven exits (`ZSeReversal`, `SwingNoiseSpike`) firing in
high-vol periods and trimming profit before the re-sim policy would.

## How to apply

1. **Don't try to flip or rescale exits to harness high-vol bleed.** Both
   approaches have been tested and rejected.
2. **Two remaining levers:**
   - Filter (skip high-vol entries) â€” straightforward but doesn't HARNESS
   - Surgical state-exit modification (disable/loosen ZSeReversal in
     high-vol bins) â€” promising, recovers re-sim/engine gap
3. **Mean-based threshold derivation overshoots fat-tail distributions.**
   When the peak distribution is fat-tailed (a few big peaks pulling up
   mean), use modal/median quantiles (q_05 to q_15) for TP rather than
   q_30 default. Skews the formula toward typical, not headline peaks.

## Anti-patterns

```
# Bad: scale TP up in high-vol because "peaks are bigger"
tp_pts(vol_bin) = peak_mu(vol_bin) * tp_quantile_q30   # OVERSHOOTS

# Bad: flip direction in high-vol because "fades fail there"
if vol_at_entry > threshold:
    direction = opposite(direction)                     # peaks are symmetric

# OK: investigate state exits instead of widening primary exits
# OK: filter (skip) high-vol entries entirely if EV doesn't justify
```



## feedback_is_not_method.md
---
name: IS/IS-NOT audit method
description: Before building anything new, audit what IS built and what IS NOT built. Prevents rebuilding existing work and identifies true gaps.
type: feedback
---

Before designing or building any new feature, always run an IS/IS-NOT audit first.

**Why:** Multiple times we rebuilt things that already existed (lookback geometry, seed features, observer data) or planned features that depended on gaps we didn't know about (seed lookback prices never populated). The IS/IS-NOT method catches this upfront.

**How to apply:**
1. List every component the feature needs
2. For each: check if it EXISTS (with what data) or is MISSING (what's needed)
3. Identify the CRITICAL GAP â€” the one missing piece that blocks everything
4. Plan to fill the gap FIRST, then build on top

This applies to: new research tools, pipeline changes, observer integration, seed enrichment, any new module.



## feedback_kway_r2_saturation.md
---
name: kway-r2-saturation
description: Direction-prediction RÂ² saturates around 0.35 on daisy-chain trades; k>3 with full bins is wasteful. Stratified k=2 matches unstratified k=5 with FAR fewer parameters. Don't escalate k blindly.
metadata:
  type: feedback
---

**Rule:** For direction prediction on the daisy-chain regret-oracle trade
set, RÂ² saturates around 0.35. Don't escalate to k=4 or k=5 with full bins
expecting big gains. Stratify instead.

**Evidence (2026-05-16):**

| Method | RÂ² |
|---|---|
| k=1 single feature on signed_mfe (slope_15s_3m) | ~0.20 |
| k=2 (5 bins, all pairs) | 0.262 |
| k=3 (5 bins) | 0.307 (+0.045) |
| k=4 (3 bins) | 0.320 (+0.013) |
| k=5 (2 bins) | 0.348 (+0.028) |
| **Stratified k=2 within bar_range S3** | **0.344** |
| **Stratified k=2 within tod_minutes S5** | **0.342** |

The 4-way and 5-way interaction terms add ~0 RÂ². Signal is fully captured
by 1-way + 2-way + a stratifier.

**Stratified k=2 (2 features within a stratum) matches unstratified k=5
(5 features + all interactions)** â€” at far fewer parameters, less overfit
risk, easier to interpret. The user's "shaft from seeds" intuition
empirically verified â€” heterogeneous data masks subgroup-specific
direction signal.

**How to apply:**

For future feature-combination work on the regret oracle:
1. Start with 1D regression per feature (target = signed_mfe per [[signed-mfe-pivot]]).
2. Pair stratification next (k=2 with 5 bins).
3. **Stop escalating bin-count beyond k=3 with full bins.** Either reduce
   bin count (k=4 at 3 bins, k=5 at 2 bins) or stratify first.
4. Pick a primary feature to stratify on (`bar_range`, `tod_minutes`, or
   the regime-2d label). Run pair analysis WITHIN each stratum.
5. Accept ~0.35 RÂ² as the natural ceiling for the V2-direction feature set
   on the daisy-chain trades. Pushing past requires structurally different
   features (e.g., trajectory shape â€” Layer 3 / [[bayesian-archetypes-pending]]).

**Caveat:** This ceiling is for the LINEAR-AND-LOW-ORDER-INTERACTION model
class on these specific features. Trajectory-based models (Layer 3 of
[[regret-six-layer-architecture]]) could break through.

**Multi-comparison caution:** With ~400k+ cells tested across all the
analyses, the top cells have selection bias. OOS validation is mandatory
before any selector uses them (per MEMORY hard rule).

See also [[signed-mfe-pivot]].



## feedback_leadin_pca_rejected.md
---
name: feedback-leadin-pca-rejected
description: Lead-in PCA signatures hurt direction classifier at all lookback lengths (60/240/720 bars); V2 entry features already encode macro setup
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

Tested whether lead-in PCA signatures (centroid + direction vector in 184-D V2 space, computed over the past K 5s-bars before entry) improve a binary LONG/SHORT direction classifier on the daisy-chain regret-oracle trades. Three lookbacks (60-bar/5min, 240-bar/20min, 720-bar/60min). **All hurt.**

| Variant | Test AUC | Test Brier | Train-test gap |
|---|---|---|---|
| Baseline (V2 entry only) | **0.864** | **0.142** | 0.000 |
| + 60-bar lead-in | 0.850 | 0.152 | 0.038 |
| + 240-bar lead-in | 0.842 | 0.156 | 0.047 |
| + 720-bar lead-in | 0.849 | 0.153 | 0.037 |

**Why:** Train AUC rises (~0.864 â†’ ~0.888) while test AUC falls â€” textbook overfit. The 368 extra features (centroid 184-D + direction 184-D) carry noise the linear model latches onto.

**How to apply:**
- For direction prediction on regret-oracle-style trades, stop trying to add lead-in trajectory features as PCA-line summaries to a linear model.
- V2 entry features (L1-L3 Ã— 5s/15s/1m/5m/15m/1h/4h/1D) at the entry bar already encode the multi-TF macro setup â€” the 4h/1D-layer features carry the regime info a lead-in PCA would extract.
- PCA signatures are lossy 2-vector summaries of a KÃ—184 matrix. Unit-direction vectors going into a linear classifier are geometrically incoherent â€” the model can't use them without per-trade context.
- Next AUC lever is **non-linear** (GBM, CNN), not more features. The model class has more headroom than the feature set does.
- L3 clusters may still help for magnitude/risk/exit prediction, but NOT for direction routing.

**Doesn't mean lead-in is useless universally** â€” direct ridge regression on lead-in centroid (no entry features) gave RÂ²=0.146 on signed_mfe (better than entry-feature â†’ cluster route at RÂ²=-0.05). The failure mode is concatenation with entry features into a linear classifier, where the entry features carry the same info more cleanly.

Connected: [[feedback-kway-r2-saturation]], [[feedback-regret-research-methodology]], [[project-bayesian-archetypes-pending]], [[project-regret-six-layer-architecture]].



## feedback_live_defaults.md
---
name: Live launcher defaults
description: Default live launcher mode should be real trading (NT8 account controls sim/real). --dry-run is opt-in for observation only.
type: feedback
---

## Live Launcher Defaults

- Default mode = send orders to NT8. NT8's account setting (sim vs real) controls risk.
- `--dry-run` = observation only (no orders sent at all). This is opt-in, not the default.
- Don't suggest --dry-run when user wants to trade with sim money â€” that's NT8's job.
- The replay validation should not block sim accounts either â€” only matters for real money.



## feedback_lookahead_audit.md
---
name: Audit lookahead before trusting any baseline
description: 2026-04-16 discovery â€” higher-TF aggregation in build_dataset had 6-hour lookahead. $740/day was inflation. Audit any aggregation that uses searchsorted/index lookups.
type: feedback
---

# Always audit lookahead before trusting a baseline

**Rule:** Before reporting or using any IS/OOS baseline, trace the
feature aggregation for lookahead. Specifically check any
`np.searchsorted` or index-lookup that maps a low-TF timestamp to a
higher-TF bar â€” bars labeled at their START contain data from their
END. Subtract the TF period before lookup.

**Why:** 2026-04-17 found this exact bug in `training/build_dataset.py`.
Inner loop used:
```python
idx = np.searchsorted(tf_ts, target_ts, side='right') - 1
```
The matched TF bar started at or before `target_ts`, but its OHLCV
aggregated 5s bars forward up to its end. Every label at time T had
future data baked in for TFs â‰¥ 1m.

Fix:
```python
idx = np.searchsorted(tf_ts, target_ts - period, 'right') - 1
```

**Impact:** Baseline dropped from $740/day IS (lookahead) to -$164/day
(honest). Every analysis done between the bug introduction and the fix
is suspect. The 79D feature folder was renamed/moved during the fix:
`DATA/FEATURES_79D_1m/` â†’ `DATA/ATLAS/FEATURES_5s/`.

**How to apply:**
- Any time a baseline seems too clean (low variance across OOS days,
  high $/day, IS/OOS nearly equal), suspect lookahead before celebrating.
- When building new feature aggregation code, test with a sentinel:
  inject a huge value into the last 5s bar of a TF window and verify
  that it doesn't appear in the label at the TF start.
- When merging feature code from other projects, re-audit â€” lookahead
  bugs often travel with good-looking results.



## feedback_metaphor_origins.md
---
name: Metaphor origins â€” some physics names came from other AIs, not Moises
description: Roche limit and some quantum terminology were applied by other AI assistants to dress up standard statistical concepts. Moises's original language was sigma distances and SE bands. The metaphors stuck because they accidentally mapped well, but credit the concepts to Moises, not the names.
type: feedback
---

Not all physics metaphors originated from Moises:
- **Roche limit** = another AI's name for 2Ïƒ/3Ïƒ standard error bands
- **Quantum field** = another AI's framing of the statistical engine
- **Event horizon** = another AI's name for the 3Ïƒ boundary

Moises's actual concepts (his words):
- Standard deviation distances from regression mean
- Standard error bands in sigma units
- Where price behavior changes structurally

**Why:** The AI collaborators themed everything to quantum physics for
consistency. The names stuck because they accidentally mapped well to
the actual market dynamics. But Moises thinks in sigma distances, not
Roche limits.

**How to apply:** Use whichever language is clearest in context. Don't
assume Moises invented the physics vocabulary â€” he invented the CONCEPTS
and other AIs dressed them up. If he says "2 sigma band," don't correct
him to "Roche limit." If he uses "Roche limit," he's using the shorthand.



## feedback_mtf_review_python_parity.md
# MTF strategy review + why we run logic through Python

**Captured: 2026-04-27.** During session review of `MyCustomStrategy.cs`
(`MultiTimeframeRunner_v1.0.2`) and the user-shared backtest CSV.

## User stance (recorded verbatim, 2026-04-27)

> "i dont care about edge i care about moneys"

This is a pragmatic frame. Translated into measurable terms for this project:

- **Goal**: maximize $/day in the account.
- **Constraint**: $/day must hold across as many market regimes as possible,
  because regime changes happen and a strategy that only made money in one
  regime stops making money when the regime flips.
- **What "edge" was tracking**: the ability to make money INDEPENDENT of
  drift. Same goal as "money in account" once you require regime stability.
- **No edge required if drift is reliable**: if MNQ drifts up forever, a
  long-only strategy is fine. The risk is ONLY regime change. The honest
  question for ANY strategy is "what does it do when the drift reverses?"

## The MTF backtest CSV finding

`examples/multy timeframe.csv` covers 37 trades on MNQ JUN26 from
2026-03-19 â†’ 2026-04-24 (â‰ˆ36 calendar days). Strategy column says
`"Sample multi-timeframe"` = NT8's BUILT-IN stock sample, NOT the user's
custom `MyCustomStrategy.cs`.

| Metric | Value |
|---|---:|
| Total profit | +$3,325.70 |
| $/day | +$107.28 |
| Trades | 37 (16 wins, 43.2% WR) |
| Trade direction | **100% long, 0 shorts** |
| "Exit on session close" exits | 15/37 (40.5%) â€” passive timeout |
| Buy-and-hold MNQ over same window | **+$6,667 (+$215/day)** |
| Strategy minus buy-and-hold | **âˆ’$3,341 (âˆ’$108/day)** |

**Honest read**: the strategy makes real money in dollars-per-day terms BUT
loses to single-contract buy-and-hold by $108/day. The $107/day is partial
drift capture, not active alpha. Profit dominated by 6 outlier winners
(+$4,875 from top 6 trades; remaining 31 trades net âˆ’$1,549).

This satisfies "money in account" criterion in a single regime. It fails
"money in account ACROSS regimes" because:
- All 37 trades are long. Strategy literally cannot harvest a downtrend.
- 36-day window is too short to catch a regime change.
- Most exits are passive (session close), not skill-driven.

If MNQ enters a sideways or falling regime, this strategy is expected to
return ~zero or lose money.

## Why run logic through Python (the parity case)

**Python parity matters for the money-in-account goal**, even when "edge"
isn't the goal. Five concrete reasons:

### 1. Test on 12 months of historical data without re-running NT8

We have `DATA/ATLAS/` (Databento) covering Jan-Dec 2025 â€” 12 months of all
TFs, which spans multiple market regimes (Q1 trend, Q2 chop, Q3 squeeze,
Q4 reversal). NT8 Strategy Analyzer can run on this too, but each backtest
takes minutes-to-tens-of-minutes to load and execute. Python sim can run
the same logic on the entire 12 months in seconds-to-minutes once.

â†’ **Money question answered**: "what would this have made over 12 months,
including non-uptrend periods?" If the answer on Q3 2025 (chop) is âˆ’$X/day,
that's information that prevents real-money loss on a future chop month.

### 2. Catch fill-timing bugs that distort PnL by 10-50%

NT8 Strategy Analyzer fills "next bar open" by default with
`Calculate.OnBarClose`. Python sims often fill at "current bar close".
That's a one-tick to multi-tick per-trade slippage difference.

If Python predicts +$X but NT8 backtest gives +$Y, the gap is fill timing,
not strategy logic. Without parity, every Python prediction has unknown
real-world drift. With parity, Python is a fast NT8 oracle.

â†’ **Money question answered**: "is the Python prediction trustworthy
enough to skip the slow NT8 backtest?" Once parity is verified, yes. Saves
hours per iteration.

### 3. Identify trades that should NOT have happened

Python sim has unlimited debugging visibility. NT8 doesn't show per-bar
internal state, only the trade ledger. Python can dump:
- The exact bar where the entry signal fired
- The feature values at that bar
- Why the exit fired
- What the position state was

â†’ **Money question**: many of the strategy's losses likely come from
specific entry conditions (e.g., entering deep into a 1m extension that
immediately reverses). With Python visibility, you can identify the bad
entries and add a filter to skip them. NT8 alone can't show you this.

### 4. Cheap parameter sweeps

Python can sweep `R`, `HardStopLossPoints`, `MaxNegativeBars`, etc. across
hundreds of combinations in minutes. NT8 Strategy Analyzer Optimizer
takes hours for the same sweep. Python finds the parameter sweet spot;
NT8 confirms.

â†’ **Money question**: "what's the best parameter set?" Found 10Ã— faster
in Python.

### 5. Catch lookahead bias and bar-aggregation bugs

The 2026-04-17 lookahead audit turned a +$740/day baseline into âˆ’$164/day.
Python parity testing is the natural place to find these bugs because
you can manipulate the bar history mid-test (block out future bars
explicitly) and confirm the strategy still produces the same results.

NT8 Strategy Analyzer also avoids lookahead but it's a black box â€” you
can't poke at intermediate state to confirm.

â†’ **Money question**: "is the +$X PnL real, or am I peeking at future bars?"
Only Python parity testing answers this cleanly.

## The action plan (parity-driven validation)

To make money decisions on `MyCustomStrategy.cs`
(`MultiTimeframeRunner_v1.0.2`) we need:

1. **Run it in NT8 Strategy Analyzer** on 2026-03-19 â†’ 2026-04-24 (the same
   window as the stock-sample CSV). Export trades CSV.
2. **Build Python sim** that mirrors `MultiTimeframeRunner_v1.0.2` exactly:
   same SMA crossover entries, same DRM, same StagnationMonitor.
3. **Parity-check** Python sim vs NT8 trades CSV. Same trades fire?
   Same exits? Same PnL?
4. **Once parity holds**: run Python sim on 12 months of `DATA/ATLAS/` to
   see what the strategy does across regimes Q1-Q4 2025.
5. **Then on `DATA/ATLAS_OOS/`** for an additional independent OOS check.
6. **Then on `DATA/ATLAS_NT8/`** as the OOS-2 (NT8-feed) gate.
7. **Money decision**: $/day across all those windows. If positive in
   most regimes, it's a money-maker. If only positive in Q1+Q2 2026,
   it's a drift-capturer that fails on regime change.

## What NOT to do

- Do NOT deploy `MyCustomStrategy.cs` to live based on a 36-day rising-
  market backtest. Money math says strategy underperforms passive long
  in that window; sustainable money math is unverified.
- Do NOT trust Python predictions until parity is verified â€” see today's
  v1.5-RC episode where Python predicted +$5K and NT8 backtest gave âˆ’$1K.
- Do NOT confuse `SampleMultiTimeFrame` (NT8 stock, used in the CSV) with
  `MultiTimeframeRunner_v1.0.2` (the user's custom code). Different
  strategies. The CSV evidence is for the STOCK sample, not the custom one.

## Related memory

- `memory/feedback_oos2_designation.md` â€” two-OOS validation gate
- `memory/feedback_lookahead_audit.md` â€” lookahead history that broke
  prior baselines
- `memory/feedback_phantom_spikes.md` â€” feed-dependent fake edge case
- `memory/feedback_data_validation_first.md` â€” verify data integrity
  before trusting strategy results
- `reports/findings/2026-04-27_v15_backtest_reality.md` â€” recent example
  where Python prediction missed NT8 reality by ~$6K



## feedback_no_human_regime_terms.md
---
name: Metaphors must translate to math/statistics â€” borrow methods, ground definitions
description: Metaphors and trader/physics/biology terms can be used as descriptive shorthand IF they translate to an explicit mathematical or statistical definition; code labels and report titles stay statistical because they must stand alone
type: feedback
---

**Refined rule** (locked 2026-05-09 evening, refined twice):

The original rule was too strict. The actual rule:

> **Borrow methods and language from any discipline (physics, biology,
> trading, finance, signal processing). It IS useful â€” a good metaphor
> can communicate faster than a formula. BUT every borrowed term must
> have an explicit translation into mathematical or statistical
> terms, and that translation must be recorded.**

Without the translation, the metaphor pre-loads paradigms (e.g., "chop"
implies "trend strategies fail here"; "compression" implies "expansion
is coming"). With the translation, the metaphor is just a label for a
mathematical pattern.

## Where metaphors are OK and where they aren't

| context                           | metaphors OK? | why                                              |
|-----------------------------------|--------------:|--------------------------------------------------|
| Descriptive prose (journal, chat) | YES, with translation on first use | a metaphor lets the reader build the right picture |
| Memory project files              | YES, with explicit math definition | future sessions need to translate on encounter   |
| Code labels (variable names, axis bins) | NO â€” statistical only | labels stand alone with no surrounding context for the translation |
| Chart titles + legend entries     | NO â€” statistical only | chart consumers may not have the glossary |
| Report headlines                  | NO â€” statistical only | headlines get pasted, lose context |
| Slack-style summaries to user     | YES, with translation | quick communication, with the math noted        |

## Translation table (canonical glossary)

This list grows as terms are introduced. When using a metaphor in a doc,
either link to this table or include the translation inline.

| metaphor / borrowed term | math/stat translation                                                            |
|--------------------------|----------------------------------------------------------------------------------|
| envelope                 | M_close Â± kÂ·SE_close band region around a regression mean                        |
| 3-body system            | the three rolling-regression anchors {M_close, M_high, M_low} at a single TF      |
| anchor                   | the rolling-regression mean for a TF and column (close/high/low)                  |
| force / tension          | NOT meaningful â€” drop. The anchors don't apply force to price.                    |
| elastic                  | dropping. Use sigma-band proximity instead.                                       |
| chop                     | high variation = high SS_residual / SS_total of N-bar linear regression of close  |
| smooth                   | low variation = low SS_residual / SS_total = high RÂ²_adjusted                     |
| compression              | low sigma rank = Q1-Q2 of the rolling-60min percentile of SE_close                |
| expansion                | high sigma rank = Q4-Q5 of the same                                               |
| pivot / inflection       | bar where sign(slope_t) â‰  sign(slope_{t-Î”}) and |curvature_t| in top quantile     |
| reversion                | the new directed leg starting at a pivot (NOT mean-reversion in the OU sense)     |
| trend-follow / ride      | trade in the direction of slope at the same TF                                   |
| macro event              | contiguous 5s run where (5s_close âˆ’ M_anchor)/SE_anchor exceeds k = 3 at 1h TF    |
| crash / rally            | macro event with side='below' (price < M_anchor âˆ’ 3Ïƒ) / 'above'                   |
| Goldilocks trigger       | k = 2.0 Ïƒ band entry â€” empirical sweet-spot of frequency vs information           |
| outer wall               | k = 3.0 Ïƒ band entry â€” rare, regime-shift suspect                                 |
| primitive / precursor    | a marked event timestamp + bar-level features captured AT that timestamp          |
| state machine            | discrete-state model: {NORMAL, DIRECTIONAL, FLATTENED, CANDIDATE, ...}            |
| filter (context)         | predicate on state: bool function of features at-bar with no lookahead            |
| tier                     | strategy that fires entries with direction; combined as filter âˆ§ filter âˆ§ entry  |
| psychohistory            | predictive statistics over aggregate event populations â€” empirical conditional probability tables built from large-N events (NOT individual prediction; NOT Asimov's deterministic future) |
| chord                    | a feature vector x âˆˆ R^N at a single bar t â€” the simultaneous combination of N feature values evaluated together (the joint cell in a probability table is the chord that fires at a given bar) |
| resonance                | cross-feature or cross-TF alignment where the signs/magnitudes of multiple features agree at the same bar â€” quantified as âˆ‘_i sign(f_i) or as the joint-cell occupancy / lift over the product of marginals |
| dissonance               | the inverse of resonance â€” features disagree at the same bar; signs cancel or magnitudes anti-correlate                                              |
| harmonic                 | a feature relationship that repeats at a multiple-of-base-TF â€” e.g. an effect at 1h that mirrors at 5m at the same phase                              |
| theme                    | day-level aggregate: stats over the full session (range, net move, efficiency, total dwell time at extremes, dominant motif type)                    |
| phrase                   | (refined 2026-05-10) the 15m-CRM-defined macro segment â€” "the most stable line" the eye tracks. Typical duration 30min-3hr. The day decomposes into ~2-5 phrases. NOTE: code currently calls these "motifs" in `segment_day_motif_melody.py` â€” pending rename. |
| motif                    | (refined 2026-05-10) the 5m-CRM-defined micro segment NESTED inside a phrase. Typical duration 5-30min. Each phrase decomposes into 1-6 motifs. NOTE: code currently calls these "melodies" in the segmenter. |
| sub-motif                | (added 2026-05-10) the 1m-CRM-defined nano segment NESTED inside a motif. Typical duration 1-5min. Captures fast directional moves within a motif. |
| measure                  | (added 2026-05-10) the 15s-CRM-defined sub-nano segment NESTED inside a sub-motif. Typical duration 15s-2min. The "rhythm-unit" level of the music hierarchy. |
| chord                    | (refined 2026-05-10) the at-bar feature vector at a single 5s bar â€” the simultaneous primitives playing at that instant. Distinct from segment_chord. |
| note                     | a single primitive value at one 5s bar (one component of a chord) â€” slope_15m_at_bar, z_close_15m_at_bar, etc.                                       |
| segment                  | generic name for either a phrase OR a motif â€” used when the level isn't specified                                                                    |
| segment_chord            | the EDA aggregation of 5s chords/notes WITHIN a segment â€” distribution stats (mean, std, dominant value, mode) computed over all 5s bars inside the segment. The "fingerprint" of the segment in chord-space. |
| variation                | (added 2026-05-10) two segments with the SAME shape_class but DIFFERENT segment_chord fingerprints are variations of the same theme/motif. Each variation is a distinct cell in the Bayesian table even though the macro classification matches. EMPIRICAL example: LINEAR_DOWN phrases (n=312 IS) sub-conditioned by `slope_15m__std` quartile produce mean_ride_$ from $12 (Q1_steady) to $174 (Q4_volatile) â€” a 14Ã— spread within one shape. The variation is the unit that distinguishes failure modes within a shape. |
| oracle                   | post-hoc retrospective analysis with full event resolution known (max_z, final duration, MFE/MAE, PnL of any tier evaluated against the event); used to BUILD the Bayesian table; not available in live |
| Bayesian table           | lookup substrate keyed by primitive-bucket vector, returns conditional outcome statistics; used at-bar in live as the substitute for oracle knowledge |
| primitive chord          | the bucket-vector x(t) at a single bar (e.g. (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q)) â€” the joint key into the Bayesian table             |
| failure mode             | a (bucket, tier) cell where the tier's oracle $/trade is materially negative â€” a candidate for adding a context filter that gates the tier off in that bucket |
| Bayesian probabilistic table | (refined 2026-05-10) a HIERARCHICAL probabilistic model â€” NOT a lookup table. Per-cell posteriors (Beta-binomial for win-rate, Normal-Inverse-Gamma for $/trade, etc.) with shrinkage from cell -> parent shape -> universal. Output is a posterior DISTRIBUTION, not a point estimate. Justification: uniqueness analysis 2026-05-10 showed 80-99% unique motif compositions within directional phrase shapes â€” lookup cells would be ~1-3 events each (too thin); hierarchical priors borrow strength from parents. Probabilistic also gives risk-management outputs (tail quantiles, credible intervals) that point-estimate lookups cannot. |
| risk-management posterior | the tail of the per-cell posterior distribution used for position-sizing / stop-placement / skip decisions. E.g., 10th-percentile $/trade -$200 calls for tighter stop than 10th-percentile +$5 even when both have mean +$30. |
| primitive (shape_class)  | (locked 2026-05-10) the STRUCTURING LABEL applied to a segment via the 20-shape SeedPrimitiveLibrary. NON-NEGOTIABLE: HDBSCAN/regression/clustering on the raw chord fingerprint WITHOUT primitive labels collapses to the most basic variance axis (direction only) and loses curve-shape distinction. Empirical demonstration 2026-05-10: global HDBSCAN on 2,091 phrases produced 2 clusters (UP, DOWN) + NOISE; per-shape HDBSCAN found meaningful within-shape variations (LINEAR_DOWN C0 n=40 ride+$237). Primitives DO the structuring work that downstream clustering cannot do alone. |
| HDBSCAN within primitive | (locked 2026-05-10) the VARIATION FINDER applied AFTER primitive labels. Each shape_class is HDBSCAN'd separately on chord fingerprint features. Some shapes (LINEAR/EXPONENTIAL with ~300+ phrases) split into 2-4 natural clusters; others (LOGARITHMIC_UP/DOWN, STEP_UP/DOWN) don't sub-cluster â€” that's a FINDING (the shape IS one homogeneous bucket), not a failure. |
| Bayesian table cell      | (locked 2026-05-10) keyed on (shape_class, variation_cluster). variation_cluster is the HDBSCAN cluster id within shape, or null if the shape doesn't sub-cluster. shape-only cells (STEP, LOGARITHMIC) get one big cell each. Shape-with-variations cells get N+1 cells (N clusters + NOISE bucket). |
| 2D shape (Layer 1)       | (locked 2026-05-10) the geometric-only substrate: primitive label + within-shape HDBSCAN on segment-level scalars (slope, sigma, length, peak_z, r2adj, tod). All inputs derived from the 2D price-vs-time curve. NO at-bar 5s feature signatures used. Current 31-cell V0 substrate at 15m level lives entirely at this layer. |
| chord layer (Layer 2)    | (locked 2026-05-10) the feature-signature substrate: at-bar 5s feature vectors (slope_15m, z_close_15m, sigma_rank_15m, slope_5m, z_close_5m, sigma_rank_5m, r2adj_5m) aggregated per Layer-1 cell. Tells us WHICH features co-fire inside a shape-defined segment. Layer 2 adds dimensions BEYOND 2D shape; this is where compression-before-expansion vol-velocity signature, 1m-5m divergence, etc. become first-class cell axes. NEXT analysis step (deferred until V0 Bayesian model on Layer 1 is built and OOS-validated). |

## What labels in code may use

Statistical-only â€” math name + magnitude + sign:

```
trend:     no_trend, negative_low_trend, negative_high_trend,
                     positive_low_trend, positive_high_trend
variation: low_variation, low_mid_variation, mid_variation,
           high_mid_variation, high_variation
sigma:     low_sigma, low_mid_sigma, mid_sigma, high_mid_sigma, high_sigma
z:         zero_z, negative_near_z, negative_far_z, positive_near_z, positive_far_z
curvature: negative_curvature, no_curvature, positive_curvature
quantiles: q1, q2, q3, q4, q5 â€” pure ordinal
sign axes: UP, DOWN, FLAT â€” sign of net move with zero-threshold
```

## How to handle existing terms in the codebase

- New code: statistical labels only.
- Old code: leave as-is until next-touched, then convert.
- Journals / memory: free to use metaphors; provide translation on
  first use of each new term in that document.
- This memory file IS the canonical translation table â€” extend the
  table when new metaphors are introduced.



## feedback_nt8_dom_level5.md
# NT8 Level 5 DOM â€” available, but DON'T use it in strategy logic

**Captured: 2026-04-27.** Discussion outcome on DOM usage. Future sessions:
read this before adding any DOM feature to a strategy.

## What's available

NT8 exposes Level 5 Depth of Market (5 bid levels + 5 ask levels with size)
on the live MNQ feed. Existing infrastructure:

- **`docs/nt8/BayesianBridge.cs` v7.0.0** has `DomLevels` parameter (default 5)
  and an `OnMarketDepth` handler. **But currently only sends Level 1**:
  `bestBid`, `bestAsk`, `bestBidSize`, `bestAskSize`, throttled to 250ms
  (4 updates/sec).
- One Level-1 derived feature is computed:
  `imbalance = (bidSize - askSize) / (bidSize + askSize)`.
- Level 2-5 levels are **not consumed anywhere** â€” `OnMarketDepth` ignores
  events with `Position > 0`.

To capture full Level 5: extend `OnMarketDepth` to retain a `bidLevels[5]`
and `askLevels[5]` array indexed by `Position`. ~80 LOC in BayesianBridge.

## Why we DO NOT put DOM in strategy logic â€” the backtest problem

**NT8 Strategy Analyzer does not replay historical DOM.** This is the killer
constraint. It means:

| Backtestable | Not backtestable |
|---|---|
| Bar OHLCV (1s/1m/1h/1D) â€” `tools/atlas_nt8_rebuild.py` rebuilds from dumps | Historical DOM snapshots |
| Range / wick / z-score / variance ratio | Resting orders at any past moment |
| v1.5-RC bleed-score filter (95-day IS/OOS validated) | Bid-ask imbalance over time |
| 9-tier z>2 trigger + chains | Cumulative volume delta |
| Walk-forward Cohen-d with 3 IS/OOS splits | Aggressor-side flow |

A DOM-based strategy feature can ONLY be evaluated forward-only. You cannot
run the v1.5-RC-style 95-day IS/OOS walk-forward on it. You cannot disprove
"DOM helps" without months of forward Sim101 data.

This breaks the project's validation ladder (see `memory/MEMORY.md`):
- Validation gate 1: IS (ATLAS) â€” **broken** for DOM features (no historical data)
- Validation gate 2: OOS (ATLAS_OOS) â€” **broken** same reason
- Validation gates 3-5 still work (Phase 7 replay, Live Sim, Live Real) but
  alone are insufficient â€” without IS/OOS you cannot statistically distinguish
  signal from noise on a small forward sample.

This is the same trap as the `memory/feedback_*.md` time-of-day filter:
features that correlate with the real cause (range, liquidity, regime) but
that you cannot validate on historical data â†’ cargo-cult risk.

## Where DOM CAN help â€” three legitimate uses

### Use 1 â€” Execution-layer optimization (ENABLED, low-risk)

DOM read at the moment of order placement can improve fills WITHOUT changing
trade selection:

- **Spread/liquidity-aware order type**: market vs limit decision based on
  next 5 levels of book.
- **Slippage estimate**: pre-emptive warning if a hard SL will get filled
  beyond cap (DOM thin on the opposing side â†’ expect bad fill).
- **Fill quality monitoring**: log avg fill price vs DOM mid at execution
  time, surface persistent slippage as a flag.

**Strategy-agnostic.** Helps v1.0, v1.5-RC, the 9-tier engine, anything.
Does not require historical DOM to validate â€” improvements measure as
"better avg fill" on the live ledger.

### Use 2 â€” Live confidence gate (NOT RECOMMENDED currently)

When entry fires, DOM check downgrades weak setups:
- Strong opposing wall â†’ skip OR size down
- Aligned book pressure â†’ keep / chain

Risk: cannot tune the threshold without forward data. Same un-validated-
filter trap as time-of-day. Only acceptable in **shadow mode** (log what
the gate would have done, compare PnL pre/post for 30+ days, then promote).

### Use 3 â€” Forward-only DOM logging for research (ENABLED, slow payoff)

Capture DOM snapshots at every entry/exit during ZigzagRunner v1.0 + v1.5-RC
Sim101 runs. After 30+ days:
- ~700 trades Ã— top-5 levels Ã— bid/ask Ã— size = ~7K data points
- Post-hoc: did winners have systematically different DOM at entry vs losers?
- If a real signal emerges â†’ forward-only filter, validate on fresh hold-out.

Aligns with `memory/feedback_rca_process.md` 9-step RCA.

## Recommendation matrix

| Path | Effort | Risk | Backtestable? | When to use |
|---|---|---|---|---|
| **1: Execution layer** | ~30 LOC ZigzagRunner CSV log + ~1 day spec | Low | Validates on live fill data | **Do this when execution quality matters** |
| **2: Live gate** | ~60 LOC + threshold tuning | High | No | Only in shadow mode after Path 3 surfaces signal |
| **3: Forward research log** | ~80 LOC bridge + Python sidecar | Low | No (research only) | **Do this whenever live trading runs** |

**Default plan**: enable Paths 1 and 3 in parallel. Skip Path 2 until Path 3
research surfaces a real signal.

## Concrete next steps if/when revisited

1. Extend `BayesianBridge.cs` `OnMarketDepth` to retain `bidLevels[5]`
   and `askLevels[5]` with `(price, size)` per slot. Send full snapshot
   in DOM message.
2. Add `dom_at_entry` and `dom_at_exit` columns to ZigzagRunner CSV ledger
   (compact JSON: `{"b":[[p1,s1],...],"a":[[p1,s1],...]}`).
3. Build `tools/dom_research_eda.py`: load CSV ledger + DOM snapshots,
   per-trade compute features (imbalance, weighted-mid skew, level-5 size
   asymmetry), Cohen-d winner vs loser.
4. After 30 days of Sim101 logs, run Path 3 EDA. If any feature shows
   |d| > 0.4 OOS â€” promote to Path 2 shadow mode. Otherwise, archive.

## What NOT to do

- **Do not gate ZigzagRunner v1.0 / v1.5-RC entries on DOM features.**
  These strategies have proven IS+OOS evidence. Adding an un-validated
  filter degrades signal-to-noise.
- **Do not add DOM to the 9-tier 91D feature space.** That space was built
  on bar features and validates IS/OOS. Mixing in non-backtestable DOM
  features contaminates the validation chain.
- **Do not build "DOM imbalance > 0.X = skip trade" rules** without 30+
  days of forward shadow-mode data justifying the threshold.

## Related memory

- `memory/feedback_lookahead_audit.md` â€” historical reminder that what
  looks like signal often is data leakage
- `memory/feedback_cnn_fragility.md` â€” small samples + tunable thresholds
  = overfitting trap
- `memory/feedback_data_validation_first.md` â€” validate inputs before
  trusting outputs
- `memory/feedback_challenge_harder.md` â€” push back on "feature X must
  help" intuitions when data can't confirm



## feedback_one_question_at_a_time.md
---
name: feedback-one-question-at-a-time
description: "Ask the user one question at a time, never batch multiple questions in a single turn â€” they process sequentially, batched questions overwhelm"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0e6e30c4-bf89-4817-9ea3-0e4056c5e720
---

Present questions to the user **one at a time**, never as a numbered list of multiple questions in one response.

**Why:** User stated directly (2026-05-23): "i can only process 1 question at a time cuz thats how my brain works." Batching questions causes either incomplete answers (user picks one and forgets the others) or cognitive overload that stalls the conversation.

**How to apply:**
- When a decision-flow has N questions, ask question 1, wait for the answer, then ask question 2.
- If a topic genuinely requires N parallel decisions (e.g. greenlighting an analysis with multiple parameters), order them by gating priority and ask only the first.
- Internal reasoning / pre-commitment proposals (e.g. "I propose decision rule X â€” accept?") still count as a question. One per turn.
- Exception: yes/no confirmation paired with a single follow-up clarification (e.g. "Run it? If yes, do you want variant D included?") is borderline â€” prefer to split unless the second is trivial.
- This applies to ALL conversations on this project going forward, not just the current session.

Linked context: [[project_parity_b9_horizon_2026_05_20]], live-deploy decision flows where I previously batched 4â€“5 questions in one turn.



## feedback_oos2_designation.md
# OOS-2 designation â€” ATLAS_NT8 as second OOS gate

**Designated: 2026-04-27.** User decision to elevate ATLAS_NT8 (NT8-feed
dumps, 32 days Mar 20 â†’ Apr 26 2026) to a formal **second OOS validation
layer**, complementary to the existing Databento ATLAS_OOS.

## What changed in the validation ladder

```
OLD (5 gates):
  IS (ATLAS, Databento 2025)
  â†’ OOS (ATLAS_OOS, Databento Jan-Feb 2026)
  â†’ Phase 7 Replay â†’ Live Sim â†’ Live Real

NEW (6 gates, 2026-04-27):
  IS (ATLAS, Databento Jan-Dec 2025)
  â†’ OOS (ATLAS_OOS, Databento Jan-Feb 2026)        â€” temporal shift, same feed
  â†’ OOS-2 (ATLAS_NT8, NT8-feed Mar-Apr 2026)       â€” temporal AND feed shift
  â†’ Phase 7 Replay â†’ Live Sim â†’ Live Real
```

## Why two OOS gates instead of one

ATLAS and ATLAS_OOS are both Databento data. They cross-check temporal
stability â€” does a strategy trained on 2025 still work on Jan-Feb 2026? â€” but
they share the same data feed. They do NOT cross-check:

- **Tick-by-tick fill semantics** (NT8 fills can differ from Databento prints
  due to broker matching, contract continuity, weekend handling).
- **Contract-roll behavior** (the way each feed handles MNQ rollover affects
  daily returns near contract change).
- **Volume reporting conventions** (NT8 volume â‰  Databento volume in some
  edge cases â€” single-print vs aggregated).
- **Session-boundary cuts** (Databento splits Globex sessions; NT8 dumps are
  calendar-day. Aggregation logic that assumes one will misbehave on the other).

A strategy that wins on IS+OOS but fails on OOS-2 has a feed-dependency
that the Databento-only chain cannot detect. **OOS-2 catches it before live.**

This matters specifically because the project's history includes:
- **Phantom spikes were fake edge** (`memory/feedback_phantom_spikes.md`):
  $4,350 of "edge" was NT8-feed phantom-spike artifact, vanished on clean
  Databento data. The reverse is also possible â€” Databento-found edge that
  vanishes on NT8 fills.
- **Frozen SFE cache bug 2026-04-16** (`project_frozen_sfe_cache.md`): live
  trading drifted from training because of feed-data invariant violation.
- **Lookahead audit 2026-04-17** (`feedback_lookahead_audit.md`): a +$740/day
  baseline collapsed to -$164/day after lookahead fix. Cross-feed validation
  is exactly the kind of check that catches subtle invariant violations.

## Decision rule â€” when to use OOS-2

| Strategy state | OOS-2 role |
|---|---|
| RC awaiting promotion to release | **Must validate on OOS-2** before live |
| Tier modification with claimed lift | Claim must replicate on OOS-2 |
| New filter / regime gate | Walk-forward on IS/OOS AND OOS-2 |
| Bug fix / refactor | OOS-2 used as parity check (numbers within tolerance) |
| Pure data-pipeline change | OOS-2 sanity check only |

**Promotion rule**: if a finding holds on IS + OOS but fails on OOS-2:
- DO NOT ship to live.
- Investigate which axis breaks: feed (data-source-specific bug, slippage
  difference, contract-roll handling) or temporal regime (Mar-Apr 2026 has
  characteristics Jan-Feb didn't).
- A strategy that legitimately works should hold on both within statistical
  noise; large divergence is a red flag.

## Constraints to remember

- **32 days is small.** Stat power on small-N tier work is weak. Treat OOS-2
  as a SANITY check ("does this generalize at all?"), not as the primary
  statistical gate. The primary gate stays ATLAS_OOS until OOS-2 has 60+ days.
- **Single contract (MNQ_06-26).** Roll boundaries between contracts not yet
  represented in OOS-2.
- **Holiday-truncated days are over-represented.** Of the 32 days, several
  are Sunday-evening short sessions (~6,500 rows) and Friday-close-shortened
  (~10,000 rows). Full sessions only ~16,500 rows.
- **2026-04-26 is truncated** (3.2h instead of full session) â€” exclude from
  any tail-of-window analysis until re-dumped.

## Expansion path

To make OOS-2 a primary gate (not just sanity check):

1. Enable `BayesianHistoryDumper.cs` v2.0.0 on a chart with **180+ days of
   load history** (NT8 supports this; just set "load N days" high). Single
   chart now produces 1s/1m/1h/1D simultaneously.
2. Re-run `python tools/atlas_nt8_rebuild.py` â€” incremental, only adds new days.
3. Re-run `python training/build_dataset_v2.py --atlas DATA/ATLAS_NT8` â€” also
   incremental.
4. Once OOS-2 has 60+ days, stat power supports primary-gate use.

## Tools that need updating to know about OOS-2

Going forward, analysis tools should default to running ALL THREE gates
(IS, OOS, OOS-2) when a parity claim is made:

- `tools/v15_filter_apply.py` â€” currently only runs on the NT8 trade-export CSV
  for ZigzagRunner. Could be extended to run the v1.5-RC bleed filter on
  ATLAS_NT8 daily 1D data for a parallel cross-check.
- `tools/tier9_bleed_filter.py` â€” already supports `--atlas` flag. Run with
  three different atlases for IS / OOS / OOS-2.
- `tools/v15_calibration_drift.py` â€” has a HARDCODED path to `DATA/ATLAS/1D`.
  Should be updated to take `--atlas` arg so it can run against OOS-2.
- Any new validation tool: bake in the three-atlas convention from the start.

## Related memory

- `memory/feedback_phantom_spikes.md` â€” historical case of feed-dependent
  fake edge
- `memory/feedback_lookahead_audit.md` â€” invariants must hold across data
  sources
- `memory/feedback_data_validation_first.md` â€” data integrity before analysis
- `memory/feedback_cnn_fragility.md` â€” small-sample overfit traps (relevant
  given OOS-2's 32-day size)

## Build provenance

OOS-2 dataset was built 2026-04-27 in this session:
- Raw bars: `DATA/ATLAS_NT8/{1s,5s,...,1D}/*.parquet` (via
  `tools/atlas_nt8_rebuild.py`)
- 139D features: `DATA/ATLAS_NT8/FEATURES_5s_v2/{25 families}/*.parquet`
  (via `training/build_dataset_v2.py --atlas DATA/ATLAS_NT8`)
- Schema parity vs canonical `DATA/ATLAS/FEATURES_5s_v2/` verified
  column-by-column.
- Provenance report: `reports/findings/2026-04-27_atlas_nt8_features_built.md`.



## feedback_oracle_vs_chord_lookahead.md
---
name: Oracle labels CAN use lookahead â€” chord/at-bar primitives CANNOT
description: Two different rules for two different computation phases. Oracle = god-mode retrospective on resolved events; lookahead is required and correct. Chord/at-bar primitives = run live; lookahead is forbidden.
type: feedback
---

**Locked 2026-05-09 evening, after the V0-design discussion**:

User: 'at oracle level we can use lookahead since it is the god level
aspiration'.

Two phases of computation, two different lookahead rules:

| phase                   | what it produces                       | lookahead | reason                                                      |
|-------------------------|----------------------------------------|-----------|-------------------------------------------------------------|
| AT-BAR (live & research) | chord values used for table lookup     | NO        | runs live; future data unavailable                          |
| ORACLE (research only)   | per-event outcome labels for fitting   | YES       | answers 'what actually happened'; that REQUIRES future data |

**Why this matters**: I have been so focused on 'no lookahead anywhere'
that I started designing oracle labels with the same constraint, which
defeats their purpose. An oracle label like 'did this event resolve as
a multi-hour cascade?' or 'what was the max excursion in the 60min
following the event?' fundamentally REQUIRES seeing future data. That
is OK because:

- The label is computed offline, on completed events
- The label is used to FIT the Bayesian table
- The table itself is then queried at-bar with NO lookahead

The lookahead is in the FITTING-TIME data prep, not the LIVE-TIME
inference. Same way you can train a classifier on labeled data even
though labels were assigned with full hindsight â€” the trained model
runs forward on unlabeled inputs.

**Oracle label categories (lookahead-OK)**:

- max_z_after_event_t       peak |z| in the future window
- max_mfe_per_tier         best price reached if a tier had been firing
- duration_until_revert     bars between event start and revert-to-mean
- did_extend_60m            binary: did the run continue past 60min
- did_resolve_as_cascade    binary: duration â‰¥ X OR max_z â‰¥ k
- ride_pnl_pts              outcome of an idealized ride trade
- fade_pnl_pts              outcome of an idealized fade trade
- bars_to_max_z             time-to-peak

**At-bar primitive categories (lookahead-FORBIDDEN)**:

- slope_TF                  rolling-window slope using only data â‰¤ t
- z_close_TF                requires only TF mean/sigma at-bar
- sigma_rank_TF             rolling-percentile, only past
- r2adj_W                   linear-fit on past W bars only
- bars_since_pivot          past-only sequence position
- tf_alignment              past-only sign-agreement count

**The test**: at any inference moment, could the value be computed live?
- If yes â†’ at-bar primitive (no-lookahead rule applies)
- If no â†’ oracle label (lookahead is fine and required)

If a primitive is ambiguous (e.g., 'price_velocity_at_event_start'),
ask: does this run live? If yes, no-lookahead. If it's only computed
once per resolved event for fitting, it's an oracle label.



## feedback_outlier_day_optimizer.md
---
name: Outlier-day dominates total-PnL optimizer
description: 2026-05-04 lesson â€” total-PnL grid search hides one freak day's lottery payoff as a "+$713/day OOS uplift". Always bootstrap-CI the delta and use median-day or trimmed-mean objective.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The rule

When optimizing exit-threshold parameters on regret-replayed paths, **NEVER
maximize total summed PnL across all paths**. Single-day tail events
(macro/news days with $1k+/contract ranges) dominate the sum and produce
threshold combos that look great in IS but only "work" because the same
freak day exists in OOS too.

Use **median daily PnL** or **trimmed mean of daily PnL** as the objective
instead. They're robust to single-day outliers and reveal the real signal.

**Why:** During the 2026-05-04 V2-native pipeline run, the grid-search
optimizer (objective='total') produced an OOS result of **+$713.26/day**
across 68 days. Bootstrap 95% CI on the delta vs baseline: **[-$3.97,
+$2208.64]** â€” NOT statistically significant.

Forensics revealed:
- **97% of the uplift came from ONE day: 2026-03-20**
- That day: 2026-03-20 had a $666 range ($1,333/contract = ~2666 ticks)
- VEL_BODY_CHORD with `tp_pts=15` ($30 TP) hit TP repeatedly = lottery payoff
  of +$49,007 in one session
- The other 67 OOS days NET to ~-$500 total
- Top-1 day concentration = 97% (anything > 30% is suspect)

The same OOS-overfit pattern appeared again with the Bayesian regime-only
formulas (+$716/day OOS, 97% from same 2026-03-20). Threshold derivation
methodology didn't matter; *strategy selection* did. VEL_BODY_CHORD on 67/68
OOS days was negative; on 2026-03-20 it printed.

## How to apply

1. **Default objective**: `--objective median_day` in
   `training_v2/threshold_optimizer.py`. (Or use `threshold_bayesian.py`
   which never grid-searches.)
2. **ALWAYS bootstrap-CI the OOS delta** vs baseline before claiming uplift.
   Use 4,000 paired bootstrap resamples on per-day PnL deltas. CI [a, b]:
   if a > 0, significant; if a â‰¤ 0 â‰¤ b, not significant.
3. **Top-1 day concentration check**: if the top single OOS day contributes
   >30% of total uplift, the result is dominated by tail events â€” investigate
   that day's market action and decide whether the strategy generalizes.
4. **Verdict on VEL_BODY_CHORD (2026-05-04)**: KILLED PERMANENTLY. It's a
   lottery-day artifact, not a real strategy. Removed from
   `training_v2/run.py` default strategy list.

## Anti-pattern

```
# Bad: total summed PnL on simulated paths
optimize_cell(labels, objective='total')  # picks IS-best argmax â€” overfits

# OK: median daily PnL
optimize_cell(labels, objective='median_day')  # tail-event resistant

# Best: derive from cell distribution moments â€” no search at all
threshold_bayesian.derive_thresholds(labels, q_tp=0.30, q_sl=0.70, ttp_factor=1.5)
```



## feedback_peak_override_failed.md
---
name: Peak override on exits failed
description: Letting peak detection override exit cascade held losers too long. 5.8% WR, PF 1.01. Reverted same day.
type: feedback
---

Do NOT let peak detection override exit decisions. Sensors lag â€” by the time they flip, the trade has round-tripped.

**Why:** Tried 2026-03-19. Peak trades could suppress SL/belief_flip/regime_decay when <2 of 3 sensors opposed. Result: trades held through full MFE and back to entry. 5.8% WR with a few massive winners = lottery ticket, not a strategy.

**How to apply:** Exit cascade fires normally for all trades. Peak detection is an ENTRY signal and EXIT signal (peak_state_exit = inverted entry), but it does NOT override other exits. The code is in exit_engine.py, disabled with a comment. Don't re-enable without fundamentally different sensor logic.



## feedback_peak_physics_dead_end.md
---
name: Peak-physics exits are a dead end
description: Features don't change at the peak. Don't build exits that try to detect the peak in real time. Baseline natural exit beats every physics rule.
type: feedback
---

# Peak-physics exits are a dead end

**Rule:** Do not propose or build exit logic that tries to detect the peak
of a trade from physics signals (velocity flip, acceleration reversal,
wick on other side, p_at_center hit, etc.). Every such rule lost to the
natural exit baseline on KILL_SHOT. The peak is a statistical maximum
over noise, not a detectable feature event.

**Why:** 2026-04-17 ran `tools/killshot_peak_physics.py` on 2,043
KILL_SHOT trades with peak > $3. Measured features AT peak vs Â±3 bars:
- Velocity flips against trade: 3.3% fire rate
- Acceleration flips: 0.2%
- Wick on other side (>30% jump): 6.8%
- Largest Cohen-d across peak: 0.19 (1m_wick_ratio)

Back-test on that cohort:
| Rule | $/trade |
|---|---|
| Natural exit (baseline) | +$11.61 |
| Fixed $10 target | +$11.22 |
| 50% trail from peak | +$3.40 (worst â€” bails on every wiggle) |
| Velocity flip | +$6.83 |
| Every other physics rule | lost to baseline |

Full report: `reports/findings/2026-04-17_killshot_peak_physics.md`

**How to apply:**
- If a tier is losing and someone (incl. me) proposes "detect the peak
  and exit there," cite this finding and refuse.
- If the problem is that the tier gives back 50% of peak, the fix is
  either (a) a fixed target, (b) better entry filter to avoid the bad
  trades that never develop, or (c) tier rebuild from data. NOT
  physics peak detection.
- Exception: if a new tier is proposed with a fundamentally different
  structure (e.g., using order-flow or TF-level signals absent here),
  the finding doesn't automatically rule it out â€” but it must prove
  separability at the peak before we build exit logic around it.



## feedback_phantom_spikes.md
---
name: NT8 Phantom Spikes â€” Fake Edge Warning
description: NT8 exported data contains phantom spikes that create artificial z_se extremes. All data must come from Databento.
type: feedback
---

NT8 exported data contains phantom price spikes that don't exist in real market data.
These spikes create artificial z_se extremes â†’ easy reversion trades â†’ inflated backtest PnL.

**Evidence (2026-04-03)**:
- NT8 data: nightmare ticker = +$4,350 over 29 days
- Clean Databento data: nightmare ticker = -$2,427 over 29 days
- TradeCNN $1,609/day OOS was on NT8 data â†’ FAKE

**Why:** User discovered this during clean data rebuild. Template system found 4,287 patterns on NT8 data but only 1,381 on clean data â€” the 2,906 difference was phantom spikes.
**How to apply:** NEVER use NT8 exported data for backtesting or training. Only Databento. If someone references old baselines from NT8 data, flag them as unreliable.



## feedback_physics_engine_careful.md
---
name: PhysicsEngine changes must be surgical â€” it's the live revenue source
description: PhysicsEngine is running live in sim making money. Any change risks sinking the boat. Treat as production â€” no refactors, no experiments, only targeted fixes with clear rollback.
type: feedback
---

PhysicsEngine is the LIVE revenue source. First session: $1,495 (93 trades, PF 1.96).
It's fragile â€” $264 without the outlier. Any wrong change can turn it negative.

**Why:** The system is "hanging on a thread" â€” thin edge, lots of churn, occasional big wins.
The user explicitly said treat it like a boat that sinks with one wrong move.

**How to apply:**
- NO refactors, NO feature changes, NO "improvements" to PhysicsEngine without explicit approval
- Only surgical fixes: ORPHAN_FLATTEN bug, 0-bar SL entries, clear engineering bugs
- Every change must have a rollback plan (git revert)
- Test on OOS FIRST before deploying to live sim
- All experimental work goes to AdvanceEngine (the rebuild), NOT PhysicsEngine
- PhysicsEngine stays frozen except for bug fixes
- The grounded feature rebuild happens in AdvanceEngine â€” PhysicsEngine keeps its ugly 12 features
- When AdvanceEngine is proven better on OOS, THEN it replaces PhysicsEngine. Not before.



## feedback_probability_table_selection_bias.md
---
name: Probability table selection bias â€” averaging across condition bars overstates entry-bar win rate
description: When building a P_revert table by binning bars and measuring forward outcomes, the resulting probability averages ALL bars in the bin, but firing a strategy at the FIRST condition-bar entry gives mid-event entries, not cusp entries. Win rate at entry is ~12-25pp BELOW the table's prediction.
type: feedback
---

When a probability table predicts P_revert = 0.71 for cells like
`z_1h_low â‰¤ -3Ïƒ AND slope_15m â‰¤ -0.5`, the actual win rate of a strategy
firing at the FIRST condition-bar is ~46-48%, not 71%. Empirically verified
2026-05-10 across IS / Val / OOS in `tools/sim_strongest_cell.py`.

**Why:** The probability table measures "from this bar, what fraction of
forward windows end profitable?" The table averages across ALL bars where
the condition holds, including bars in the middle of a 30-bar crash run.
A strategy entering at the FIRST condition bar gets mid-crash entries where
the bounce hasn't happened yet. The actual bounce-eligible bars are clustered
near the END of the condition run (at the literal cusp).

**Why:** The selection-by-condition produces a stratified sample where each
condition-bar is given equal weight. A cusp run that lasts 30 bars
contributes 30 bars to the table but only 1 actual bounce event. The
probability is "1 bounce per 30 bars in this state" â€” not "if you enter at
any bar in this state, probability of profitable bounce."

**How to apply:** When using a probability lookup table built on bar-by-bar
binning:
1. NEVER trust the marginal P as the entry-strategy win rate
2. Fire on CUSP/TRANSITION events within the condition (when the underlying
   feature stopped moving in the adverse direction), not the first bar of
   the condition
3. The cusp-on-z detection (z just stopped falling) recovers ~12pp of the
   win rate gap â€” `if z_lo[t-1] <= -3Ïƒ AND z_lo[t] > z_lo[t-1]`
4. Alternative: condition on a transition that's already happened
   (e.g. only fire when slope_15m flipped from âˆ’ to + while z stays low)
5. Be skeptical of any IS-only edge from probability lookups â€” OOS regime
   shifts can erase the edge entirely (saw 58.6% IS â†’ 35.9% OOS for the
   "strongest cell")



## feedback_quantile_selection_overfit.md
---
name: Quantile-cell selection overfits massively without OOS validation
description: Discovered 2026-05-03 in Layer C1 triplet EDA â€” fine-grained quantile partitions on IS create cells that look like signal but are 75% noise. Always OOS-validate before quoting cell metrics.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
When evaluating quantile-binned feature combinations against a forward-return target,
**do not quote cell-mean / lift / WR statistics from IS as if they are real**. Run an
OOS validation pass *first*.

**Why**: Layer C1 in 2026-05-03 evaluated 270 triplets Ã— 6 regimes Ã— 27 cells (3 quantiles
per feature) = ~43k IS cells. Top |lift| cells reached **+100 ticks** above regime baseline
on IS. OOS validation (same IS-derived quantile boundaries applied to OOS data) showed
**only 25.8% of top-K cells survived**:

- 49 of 120 cells flipped sign on OOS
- 34 had |OOS lift| < 30% of |IS lift|
- The +99.9 IS-lift cell (`hurst_w_15m + price_sigma_w_5m + vol_velocity_w_15m UP_SMOOTH (1,2,1)`)
  collapsed to **âˆ’3.3** OOS â€” pure noise dressed up as signal
- True survivors had n_IS â‰¥ 200 AND a structurally sensible composition (1h structure
  anchor + amplitude carrier + reversion-context companion)

**How to apply**:

1. **Never quote a cell-mean lift from IS without OOS confirmation.** A cell that picks
   the top 1/27th of bars in a 270-triplet sweep is in the top 0.014% of all evaluated
   cells. Selection bias is enormous.
2. **Survival rule for cell-mean signals**: sign(OOS lift) == sign(IS lift) AND
   |OOS lift| â‰¥ 0.30 Ã— |IS lift| AND n_oos â‰¥ 20.
3. **Going deeper amplifies the problem.** Layer C1 had 27 cells per triplet; Layer C2
   (4-feature) would have 81. Combinatorial cell-count growth turns spurious patterns
   into "winning" cells. Stop at C1, consolidate survivors, then test orthogonal feature
   combinations rather than adding more features.
4. **Trust large-n cells over high-lift cells.** In the C1 OOS test, the survivors had
   n_IS = 88 to 371 and modest IS lift (~+65 to +120). The biggest IS lifts (98, 95, 94)
   sat on n_IS = 96 to 123 and ALL flipped sign on OOS.
5. **Regime-conditional structural composition matters.** Survivors all had pattern:
   1h-window anchor (`vol_mean_w_1h`, `hurst_w_1h`, `bar_range_1h`) + low-TF amplitude
   (`swing_noise_w_1m`, `bar_range_5m`) + reversion-context (`z_se_w_15m`,
   `reversion_prob_w_15m`). Pure velocity/sigma combinations did not survive.

This applies to any future EDA layer: chord-finder, contextualizer, multi-feature
quantile-binned analysis. Always: pre-register a survival rule, validate on
held-out data, report only the survivors.



## feedback_rca_process.md
---
name: RCA Process for System Improvement
description: Root Cause Analysis workflow â€” always follow this when improving the trading system. No shortcuts.
type: feedback
---

## RCA Process â€” MANDATORY for all system improvements

When the system underperforms, follow this process step by step. Do NOT skip steps.
Do NOT jump to "let me add a feature" or "let me train a model." Data first, always.

### Step 1: Run the system and get actual results
- Use the zero-lookahead ticker (1s data, aggregated to 1m for decisions)
- Run ONE DAY first, not the whole month
- Get real numbers with real execution

### Step 2: Analyze at the day and hour level
- Never look at monthly aggregates â€” each day must stand on its own
- What's the MODE of daily PnL? (most common outcome, not mean)
- How many winning vs losing days?

### Step 3: Pareto the exits
- Which exit category makes money? (mean_reached, max_hold_profit, lambda_flip)
- Which exit category destroys PnL? (half_cycle_loss, catastrophic_sl)
- What percentage of trades produce what percentage of profit?

### Step 4: Deep dive the losers
- Take the worst exit category (e.g., half_cycle_loss)
- List every trade: time, direction, features at entry
- What do they have in common?

### Step 5: Compare losers to winners feature by feature
- For each of the 13 features: what's the mean for losers vs winners?
- Which features are THE SAME (can't filter on these)?
- Which features are DIFFERENT (can filter on these)?
- Find the ratio â€” 4.9x separation is a signal, 1.1x is noise

### Step 6: Find the separator
- The separator is usually NOT in the bar features at entry
- It's in the CONTEXT: trend, DMI direction, volume participation
- Losers = extended in a trending market (keeps going)
- Winners = extended in a calm market (snaps back)

### Step 7: Apply the fix at the RIGHT point
- DO NOT filter at entry â€” this kills good trades too (proven: entry features identical)
- Fix at the EXIT: conditional early cut based on context at bar 2
- Check if BOTH trend AND dmi oppose â†’ early cut
- If only one opposes â†’ give it more time

### Step 8: Re-measure
- Run the same day again with the fix
- Did losers decrease? Did winners stay the same?
- If winners decreased, the fix was too aggressive â€” revert
- If losers decreased and winners held, keep the fix

### Step 9: Repeat
- After fixing one exit category, re-Pareto
- The next biggest drag becomes the target
- Continue until the mode per day is positive

## Critical Rules

**No shortcuts.** The tick-per-tick approach works because it mirrors live exactly.
If you test on pre-built bars, you get fake numbers with subtle lookahead.
Always use the 1s ticker for honest results.

**Why:** The batch SFE forward pass showed +$777/day. The honest 1s ticker showed +$48/day.
The difference was subtle lookahead in how features were computed.
The 1s ticker is the ONLY honest test. Everything else lies.

**How to apply:** Every time we modify the system:
1. Run nightmare_ticker.py on ONE day
2. Check per-exit-category PnL
3. If something regressed, find out why before running the full OOS

**No theoretical improvements.** Don't add features, models, or complexity without first
running the RCA on the current results. The data tells you what to fix â€” not intuition.



## feedback_regret_research_methodology.md
---
name: regret-research-methodology
description: The technical research method used across the regret-oracle arc â€” sediment 1D â†’ pair â†’ triplet â†’ k-way, pivot target if signal weak, stratify before escalating k, configurable defaults, lookahead audit on every selector-usable axis. Distilled from the 2026-05-14..16 sessions.
metadata:
  type: feedback
---

This is the **technical workflow** that produced the regret-oracle direction
findings. Apply the same sequence to new sub-questions in this arc.

## The escalation ladder

For ANY new "does feature X predict direction/magnitude" question:

### GLOBAL (all data)

1. **1D regression per feature** (linear + quadratic + Spearman).
   Tool pattern: `tools/regret_feature_regression.py`.
   Catches monotone and U-shape relationships. Don't skip â€” many features
   have U-shape that linear-only sees as zero.

2. **Per-feature quantile table** (5 bins, per-cell stats).
   Tool pattern: `tools/regret_feature_table.py`.
   Per cell: n, mode_$, median_$, mean_$ + 95% bootstrap CI, noise_floor_$.
   Mode-vs-noise + CI-excludes-0 are the two gates for "real edge."

3. **Pair stratification + pair regression**.
   Tools: `regret_pair_clusters.py` (5Ã—5 joint), `regret_pair_regression.py`
   (additive + interaction). Looking for: which feature Ã— feature
   combinations stratify edge most.

4. **Triplet, only if pairs surface promising joints**.
   Don't blindly escalate. 3-way interaction term adds ~0 RÂ² typically.

5. **k=4 / k=5 ONLY with reduced bins** (per [[kway-r2-saturation]] â€”
   3 bins at k=4, 2 at k=5). 4-way+ interactions add ~0.

### STRATIFIED (subset by primary feature first, then sediment within stratum)

When global signal saturates around RÂ²~0.35, switch to stratified analysis.
**Mirror the global ladder within each stratum** â€” don't skip the 1D
within-stratum step:

6. **Stratified 1D regression** (per user 2026-05-16 â€” confirmed sound).
   For each stratum Ã— each feature: 1D regression on signed_mfe.
   Surfaces **stratum-conditional features** â€” features that look weak
   globally but are strong inside specific strata (e.g., `z_1m` RÂ²=0.05
   globally but RÂ²=0.25 within RTH AM).
   Output: per (feature Ã— stratum) row with RÂ², slope, slope-per-Ïƒ,
   quadratic RÂ², Spearman Ï. Add a `conditional_lift = RÂ²_max_stratum âˆ’
   RÂ²_global` column â€” sort by this to surface the "weak globally /
   strong in one stratum" set.

7. **Stratified pair regression** (the strongest move when global signal
   is noisy). Tool: `regret_stratified.py`. Subset by one feature
   (bar_range, tod_minutes, regime_2d) THEN run pair analysis within each
   stratum. Often beats unstratified k=4/5 with simpler model.

8. **Stratified triplet, only if stratified pairs surface joints**.

**Each level within stratum mirrors a level globally.** Don't skip the
1D-within-stratum step.

## Methodological levers

When direction/magnitude signal is weak on a target, try in this order:

1. **Pivot the target.** mfe_dollars â†’ signed_mfe was decisive (see
   [[signed-mfe-pivot]]). For ANY direction work, target = signed_mfe.

2. **Stratify the data.** Heterogeneous data hides subgroup signals
   ("shaft from seeds"). Stratifying by bar_range or tod_minutes
   beats blind feature escalation.

3. **Add trajectory information.** Single-point features at entry
   collapse trade dynamics. Layer 3 (see [[bayesian-archetypes-pending]])
   uses the full N-D trajectory.

## Mandatory metrics (CLAUDE.md protocol)

Every per-cell or per-cluster $/trade stat MUST include:

- **mode** (histogram bin $2 for $/trade, $25 for $/day)
- **mean** with 95% bootstrap CI (4,000 resamples, percentile method)
- median (also useful)
- Direction stats: pct_long + Wilson 95% CI on pct_long
- Compare mode to noise floor where applicable

**Trade WR** = (âˆ‘profit/|âˆ‘loss|) âˆ’ 1 (PF-based, NOT count-based).
**Day WR** = winning_days / total_active_days (count-based).
$/day or $/trade claims WITHOUT 95% CI + significance statement are
forbidden per CLAUDE.md.

## Lookahead audit

Before stratifying or matching on any axis, ASK:

- Is this feature knowable at entry time?
- Is it knowable using only past data?
- If it requires forward MFE or end-of-day stats â†’ LOOKAHEAD.

Lookahead axes I've hit:
- `duration_bucket` (from `time_to_mfe_min` â€” forward)
- `regime_2d` (from end-of-day stats per the 2026-05-11 caveat)
- `mfe_dollars`, `signed_mfe`, `mfe_velocity` (targets, not features)
- `exit_*` state vector (mirror of entry, at exit bar)
- Centered-window oracle detection (uses future bars to find extrema)

Lookahead-clean axes:
- `tod_minutes` (wall-clock at entry)
- `bar_range`, `volume` at entry bar
- All entry-time state-vector z/dist/slope/fan features
- `full_window`, `available_fwd_min` (pure session geometry)
- All V2 features at entry bar

Direction-callable findings on LOOKAHEAD axes are descriptive only; on
CLEAN axes they're selector-usable. Distinguish in every report.

## Tool construction patterns

Every research tool:

- Takes `--input` (CSV) and `--out-dir`, `--name` for output naming
- Outputs to `reports/findings/regret_oracle/<filename>_<name>.csv`
  (per CLAUDE.md: "Tool outputs to file, not just stdout")
- Has CLI parameters for ALL thresholds (5%, K bins, min_n, etc.) â€”
  configurable from day 1 (per CLAUDE.md "no magic numbers")
- Stdout prints: header, progress (every N% for long runs), summary
  rankings (top 10-25 by relevant metric), caveats
- Saved to `tools/regret_*.py` (NOT throwaway inline code)
- Reuses helpers from sibling tools where possible (e.g., the
  bootstrap-CI and hist-mode functions)

For long runs (k>=4 cluster runs ~5-10 min):
- Run via `run_in_background=true`
- Print progress every ~10% with `flush=True`
- Or use file-based progress logging

## Reporting

After ANY meaningful research run, produce three artifacts:

1. **Findings doc** at `reports/findings/regret_oracle/<date>_<topic>.md`
2. **Daily journal entry** at `docs/daily/<date>.md`
3. **INDEX entry** in `docs/daily/INDEX.md` (prepend; one-line dense)

Per CLAUDE.md hard rules. Don't skip.

## Caveats that apply to every finding

- IS-only findings are descriptive, not actionable. 2026 OOS validation
  is mandatory before claiming edge (per MEMORY hard rule).
- Multi-comparison risk: ~400k+ cells tested across the k-way analyses.
  Some top cells will be spurious. Sign-stability OOS check required.
- No $/day claim without 95% CI + significance statement.

See also [[user-collaboration-protocol]], [[signed-mfe-pivot]],
[[kway-r2-saturation]], [[regret-six-layer-architecture]],
[[bayesian-archetypes-pending]].



## feedback_scenario_lstm_information_ceiling.md
---
name: feedback-scenario-lstm-information-ceiling
description: "LSTM multi-head scenario classifier on lead-in sequence ties or barely exceeds LR on entry features â€” V2 information ceiling, not model class, is the bottleneck"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User proposed scenario-based ML (direction Ã— speed Ã— duration Ã— trajectory buckets) with LSTM as the trunk. Built full pipeline (bucket labeler, sequence dataset, LSTM trainer, LR baseline) and validated on 2026 OOS (2,085 trades, Jan-Mar 2026, fresh oracle from daisy-chain).

**Result table (OOS):**

| Head | n_cls | Baseline | LR (entry-only) | LSTM (60-bar seq) | LSTM - LR |
|------|-------|----------|-----------------|-------------------|-----------|
| dir  | 2     | 0.501    | 0.810           | 0.827             | +0.017 |
| dur  | 4     | 0.261    | 0.330           | 0.327             | -0.003 |
| spd  | 4     | 0.292    | 0.443           | 0.400             | -0.043 |
| traj | 4     | 0.835    | 0.302           | 0.459             | (both below baseline) |

**Findings:**

1. **The V2 entry-feature information ceiling at ~83% direction accuracy holds.** LR and LSTM converge within ~2pp. Linear model with V2 features captures most of the predictable structure; sequence input adds ~1.7pp at best.

2. **Lead-in trajectory carries minimal additional signal.** Confirmed via two architecturally distinct experiments: (a) lead-in PCA centroid+direction concatenated as features (60/240/720 bar â€” all hurt, see [[feedback-leadin-pca-rejected]]); (b) raw 60-bar sequence into LSTM trunk (this experiment â€” marginal +1.7pp on direction, zero or negative on other heads).

3. **Speed/duration signal lives at the entry bar, not in lead-in.** LSTM ties LR on duration, HURTS on speed. The V2 multi-TF features at entry already encode the macro setup that would be inferrable from a 5-minute trajectory.

4. **Trajectory bucket from MAE/MFE ratio is too imbalanced** to be useful as a 4-class target â€” 84% of daisy-chain trades are MONOTONIC (MAE=$0) because the oracle picks the best extreme. Class-weighted loss forces rare-class predictions and both LR and LSTM end up BELOW the always-predict-CLEAN baseline.

**How to apply:**

- **Stop chasing the lead-in trajectory as a model input.** Two architectures (lossy PCA + raw LSTM) both confirm minimal lift.
- **Stop reaching for more complex models on this feature set.** The ceiling is in V2 feature engineering, not LR vs LSTM vs CNN. Don't escalate from LR until features change.
- **Next levers worth trying:**
  - Richer entry features (new TF combos, intra-bar microstructure, calendar/event encoding, session-relative time)
  - Different target (regret-on-skip vs direction)
  - Coarser trajectory bucket (binary CLEAN/PULLBACK)
  - GBM with isotonic calibration on entry features (60-second probe, still untested)
- **The direction classifier (`tools/regret_direction_classifier.py`) is still the L4 selector signal** â€” AUC 0.864 at 100% coverage, 88% acc at 40% coverage. The LSTM doesn't replace it.

**Doesn't mean LSTM is useless universally** â€” it might help with sequence-native problems (e.g., predicting exit timing from intra-trade trajectory, or modeling the entry decision itself as a sequence of bars where each could be a candidate). The failure mode here is "predict scenario from past 60 bars" where the past 60 bars don't carry the signal.

Connected: [[feedback-leadin-pca-rejected]], [[feedback-kway-r2-saturation]], [[project-regret-six-layer-architecture]], [[user-collaboration-protocol]].



## feedback_session_protocol.md
---
name: Session Start/End Protocol
description: On session end note time/date, on session start check time/date and resume from todo list unless fixing an error
type: feedback
---

## Session Protocol

**Session End**: User says "session end" â†’ note the current time and date.

**Session Start**: When user sends first message of new session:
1. Check current time and date
2. Unless the message is about fixing an error â†’ read the todo list
3. Resume from where we left off on the todo list
4. Don't ask "what do you want to work on?" â€” the todo list tells you

**Why:** Eliminates the cold-start problem. Every session picks up where the last one stopped. No context loss, no re-explaining.

**How to apply:** Always. Every session boundary.



## feedback_signed_mfe_pivot.md
---
name: signed-mfe-pivot
description: For direction prediction on regret-oracle trades, target MUST be signed_mfe (mfe_dollars Ã— direction sign). mfe_dollars alone hides direction signal â€” features that predict direction look invisible (RÂ²â‰ˆ0). Discovered 2026-05-16; RÂ² jumped 0.187â†’0.262 on same features.
metadata:
  type: feedback
---

**Rule:** For ANY direction prediction or direction-discrimination work on
the regret-oracle daisy-chain trades, use
`signed_mfe = mfe_dollars Ã— (+1 if LONG else âˆ’1)` as the regression/cluster
target. NOT `mfe_dollars` alone.

**Why:** `mfe_dollars` is the magnitude of forward excursion in the trade's
chosen direction. LONG trades and SHORT trades BOTH produce positive
mfe_dollars (it's the absolute excursion). Direction information is
collapsed out of the target. Features that predict direction (slope sign,
z-score sign, rail-position relative to price) appear invisible:

  - `slope_15s_3m` on `mfe_dollars`: RÂ² = 0.002 (invisible)
  - `slope_15s_3m` on `signed_mfe`: RÂ² â‰ˆ 0.20 (dominant direction predictor)

A coefficient of +$X/Ïƒ on signed_mfe means "1Ïƒ increase in feature â†’ $X
more LONG-skewed outcome." That's what direction work needs.

**How to apply:**

- For 1D, paired, triplet, k-way regression on direction: target = signed_mfe.
- For clustering by direction-archetype: target column for cluster stats = signed_mfe.
- Cells with `|mean_signed_mfe|` high AND pct_long far from 50% are
  direction-callable.
- `bar_range` and `volume` are MAGNITUDE amplifiers (symmetric across
  direction) â€” they show up strong on mfe_dollars and weak on signed_mfe
  with the same dataset. Use mfe_dollars when you specifically want to find
  magnitude-amplifier features; use signed_mfe for direction predictors.

**Evidence (2026-05-16 sleep run):**
- k=2 RÂ² jumped 0.187 (mfe_dollars) â†’ 0.262 (signed_mfe) on same features.
- Direction-callable cells (Wilson CI excludes 30/70%) at 43-59% rate.
- Per-cell accuracy 82-86% in callable cells, 93% in extreme cells.

**Where it sits in the arc:** Layer 2 (direction discrimination) of
[[regret-six-layer-architecture]]. Findings doc:
`reports/findings/regret_oracle/2026-05-16_direction_signal_kway.md`.



## feedback_sloppy_work.md
---
name: Don't Be Sloppy â€” User Loses Sleep Over Regressions
description: User explicitly called out sloppy work causing regressions â€” be extra careful with changes to production code
type: feedback
---

User quote: "this type of stuff is the reason i cant sleep your bein to sloopy"

Context: Claude removed CNN seed AND changed the loss function simultaneously, causing the model to produce 0 trades. When challenged, Claude initially pushed back ("the math shows...") instead of acknowledging the mistake.

**Why:** This is a real-money trading system. Regressions in production code directly affect the user's livelihood and sleep. The user trusts Claude with critical system components.
**How to apply:**
- Never change multiple things at once in training/model code
- When the user says something broke, believe them first, investigate second
- Don't defend bad changes with theory â€” check the actual results
- Extra caution on: loss functions, seeds, feature pipelines, live engine logic
- If unsure, run research/validation BEFORE modifying production code



## feedback_thin_wrapper_live_engine.md
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



## feedback_threshold_tuning_ceiling.md
---
name: Exit-threshold tuning has a ceiling â€” entries are the bottleneck
description: 2026-05-04 â€” adaptive exit thresholds (per regime, per tier, or per regimeÃ—tier) all give ~$28/day OOS uplift. Cell granularity is a wash. To break past, fix entries, not exits.
type: feedback
originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---
## The rule

Adaptive exit thresholds doubled baseline OOS PnL ($27 â†’ $55/day, day-WR
51%â†’57%) â€” but the uplift saturates around **+$28/day** regardless of cell
granularity (regime-only, tier-only, or tier Ã— regime). To go further, the
lever is **entries**, not exits.

## Evidence (2026-05-04 V2-native pipeline)

Same 68 OOS days, MA_ALIGN + REVERSION strategies (VEL_BODY_CHORD removed),
Bayesian-derived thresholds varying only by cell-granularity choice:

| variant | $/day | total | day-WR | 95% CI on delta vs baseline |
|---|---:|---:|---:|---|
| baseline (no thr) | $27.43 | +$1,865 | 51% | reference |
| regime-only (6 cells) | $54.82 | +$3,728 | 57% | [-$5.29, +$62.89] |
| **tier-only (3 cells)** | **$56.99** | **+$3,875** | 57% | [-$4.56, +$66.78] |
| tier Ã— regime (18 cells) | $55.57 | +$3,779 | 57% | [-$4.71, +$61.92] |

Pairwise differences between groupings: all <$2/day. The framework is
robust; the ceiling is structural.

**95% CI lower bound stays around -$5 across all three variants.** Just below
statistical significance at n=68 OOS days. The day-WR bump (51â†’57%) is
reliable â€” the dollar magnitude is too small to clear noise.

## Why this happens

The legacy V1 system had similar behavior: exit improvements helped
incrementally, but the dollar lift came from CNN flip prediction (70.6%
direction accuracy) and adding new entry types (KILL_SHOT, CASCADE,
PEAK_ExNMP). Exits trimmed losers; entries created winners.

In V2-native land, only **2 strategies** are active (MA_ALIGN + REVERSION).
Their entries cover narrow patterns:
- MA_ALIGN: 7-of-8 vwap_w alignment (~20% of 5m bars)
- REVERSION: |z_se_w|â‰¥1.8 (extreme stretches)

A lot of intermediate market behavior â€” moderate trends, breakouts,
exhaustion bars, absorption, multi-TF momentum exhaustion â€” is uncovered.

## How to apply

1. **Don't tune exits past +$28/day-uplift expectation.** It's a ceiling,
   not a starting point.
2. **The bottleneck is entries.** Three levers to break past:
   - Train CNN as filter+entry (rejects bad entries, spawns CNN-originated)
   - Add more strategies (EXHAUSTION, BREAKOUT_4H, COMPRESSION, etc.)
   - Apply contextualizer (sign-flip rules conditional on modifier quantile)
3. **Production config**: `training_v2/output/thresholds_prod.json`
   (per-tier Bayesian-derived). This is the locked exit configuration; we
   move to the entry side next.



## feedback_tier_three_questions.md
# Three Questions That Build a Tier

Discovered 2026-04-18 while rebuilding TREND_FOLLOWER (nÃ©e FREIGHT_TRAIN) from
scratch. This is the working methodology for taking any tier from "noisy idea"
to "physics-grounded rule." It replaces the old habit of brute-force Cohen-d
sweeps and CART overfitting.

## The three questions (ask in order)

### Q1 â€” Are the entries the right ones?

**What it measures:** is the direction thesis correct at entry?

**How to answer:** peak-bucket analysis. For every trade in the tier, split by
peak PnL magnitude:

| Peak bucket | What it means |
|---|---|
| peak â‰¤ $0 | Direction wrong from bar 1 |
| peak $0â€“5 | Direction barely worked |
| peak $5â€“20 | Partial fade, then overwhelmed |
| peak > $20 | Direction solidly right |

**Pass criterion:** if >80% of trades have peak > $20, the direction is right.
You don't need filters; you need better exits. If <50%, the direction is
wrong â€” consider flipping.

**On TREND_FOLLOWER (2026-04-18):** 84.8% peak > $20. Direction validated.

### Q2 â€” What signal, if persistent for X bars, says we entered wrong?

**What it measures:** can we tell early that a trade is a loser?

**How to answer:** bar-by-bar analysis of the trade path. For winners vs
losers, track `peak_pnl_by_bar_N` and find where the gap opens.

| Bar N | Winners: no +$5 yet | Losers: no +$5 yet | Gap |
|---|---:|---:|---:|
| 60 (10 min) | 20.6% | 40.9% | 20pp â† cut here |

**Pass criterion:** if the gap is >15pp at some bar N, we have a "no-progress
kill" signal: exit if `peak_pnl < $5` at bar N.

**Physics framing (not a safety):** the tier's thesis implies a timescale â€”
fades revert in minutes, trends unfold in hours. If price hasn't moved our
way in the tier's characteristic timescale, the thesis is false. Exit.

**On TREND_FOLLOWER (2026-04-18):** if peak_pnl < $5 by bar 60 (10 min),
we're in the loser distribution. Noted, not yet implemented.

### Q3 â€” What do all the peaks have in common?

**What it measures:** what feature state defines the exit (thesis complete)?

**How to answer:** for each trade, find the peak bar. Take the feature
vector AT that bar. Subtract the entry feature vector. Normalize by
per-feature entry std dev to get comparable magnitudes. Rank features by
|delta / sigma|.

The feature with the largest |d/Ïƒ| IS the exit signal.

Supporting physics come from the 2nd and 3rd rankings â€” prefer ones whose
entry condition has a natural inverse (e.g., if entry requires `vr > 1`,
the exit requiring `vr < 1` is free symmetry).

**Pass criterion:** if the top feature has |d/Ïƒ| > 5, the exit rule is
dominated by one signal â€” very clean. If top is <2, the peak is muddy
and the rule is weaker.

**On TREND_FOLLOWER (2026-04-18):**
- `1m_p_at_center` d/Ïƒ = +10.32 (THE signal)
- `1m_variance_ratio` d/Ïƒ = -1.81 (inverse of entry gate `vr > 1`)
- `1m_reversion_prob` d/Ïƒ = +1.16 (OU statistical confirmation)

Three orthogonal physics dimensions (position, regime, probability), all
firing together at peak. Exit rule: all three must fire.

## Why these three questions in this order

- Q1 answers whether the entry is worth keeping at all. If it's random
  direction, no amount of exit work helps.
- Q2 catches the losers early. Even a good-direction tier has a tail of
  trades that go wrong fast. This is the "eject if thesis clearly
  broken" rule.
- Q3 captures the winners at the right price. A good-direction tier
  held too long gives back; this rule exits at the correct moment.

Together they produce:
- An entry filter (optional, from Q1 if direction is imperfect)
- An early-exit rule (Q2, "thesis violated")
- A peak-arrival exit (Q3, "thesis complete")

## Anti-patterns to avoid (learned painfully)

- **Don't start with CART or ML** on winners vs losers. We did this; it
  overfit, and we discovered feature leakage in the corrected trades.
  The three-question method is blunt but honest.
- **Don't add more than 3 confirming features** to an exit rule. Each AND
  condition multiplicatively reduces trigger rate. 3 features that each
  fire 70% at peak â†’ ~34% combined trigger. 4 features â†’ ~24%. You
  start missing real peaks.
- **Don't propose ALL of Q1+Q2+Q3 in one code change.** Build in stages,
  run, measure, move on. Each question produces ONE rule; land that rule
  before the next one.
- **Don't skip Q1.** If the direction is wrong 50%+ of the time, Q2 and
  Q3 are about exit timing within a random signal. You'll get marginal
  lift that evaporates OOS.

## Tooling

- `tools/tier_eda.py --tier NAME` runs the Q1 segment/separator/regime-shift
  analysis on a tier. Writes markdown to `reports/findings/`.
- Ad-hoc Python for Q2 (path PnL trajectories) and Q3 (entryâ†’peak feature
  deltas). The EDA pattern is consistent: load `iso_is.pkl`, filter by
  `entry_tier`, compute, print.

## Applied to TREND_FOLLOWER â€” full chain

| Question | Answer | Rule |
|---|---|---|
| Q1: Entries right? | 84.8% peak > $20. Yes. | Keep entries. |
| Q2: Wrong signal? | If peak_pnl < $5 at bar 60, more likely loser. | (noted, deferred) |
| Q3: Peak signature? | p_center > 0.35 AND reversion > 0.80 AND vr < 1 | Primary exit rule |

Result expected to ship: TREND_FOLLOWER goes from -$4/trade to positive.
Subsequent tiers (RIDE_AGAINST, KILL_SHOT, etc.) will follow the same
three-question chain.



## feedback_timeframe_defaults.md
---
name: Stop defaulting to 15s bars
description: Use the right ATLAS timeframe for the job â€” 15s is not the default for everything. This caused the 4x mismatch bug.
type: feedback
---

## Don't default to 15s bars

ATLAS has 14 timeframes: 1s, 5s, 15s, 30s, 1m, 3m, 5m, 15m, 30m, 1h (and more).
The forward pass iterates 15s bars, but that does NOT mean everything is 15s.

**Bug caused**: Oracle stats (avg_mfe_bar, p75_mfe_bar) are computed from 1m data
in fractal_discovery_agent.py, but the system consumed them as 15s bar counts
everywhere â€” a 4x mismatch that made anchor patience expire too early, pace
run 4x too fast, envelope decay 4x too aggressive.

**Rule**: Match the timeframe to the task:
- Session-level analysis â†’ aggregate from 1h bars
- Trade overlay on price â†’ use 1m or 5m, not 15s
- Oracle/template stats â†’ computed from 1m (discovery TF)
- Forward pass iteration â†’ 15s (execution TF)
- New tools/analysis â†’ ask "what TF makes sense?" don't assume 15s



## feedback_time_assessment_calibration.md
---
name: time-assessment-calibration
description: "Time estimates in autonomous mode are systematically too pessimistic. Track actual time (start/end) and use to calibrate. When over-budget, prefer doing it RIGHT to handing back a cut-corner partial."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

In autonomous (sleep-run) mode, my time estimates are consistently too
pessimistic. I budget conservatively, cut corners "for time," and end up
with hours of unused budget at the end.

**Why:** Recorded 2026-05-19. Cut corner on L5Decider's zigzag pivot
detector during overnight build -- wrote a simpler running-extreme
state machine instead of porting the training pipeline's
`tools/build_zigzag_pivot_dataset.py` (min_bars=36 + ATR-on-pivot-series).
Result: live mock captured only 30% of OOS leg count, killing the
expected edge. User: "you always end up with extra time, improve your
time assessments."

**How to apply:**
  1. At the **start of every code-write or analysis task**, note the
     wall-clock start time (`date +%H:%M`).
  2. At the **end**, note the end time and compute actual duration.
  3. Compare to my initial estimate.
  4. Use the calibration to revise future estimates:
     - If actual is consistently <50% of estimate -> estimates are 2x too pessimistic
     - Use the surplus to do the job RIGHT instead of cutting corners
  5. **Don't cut corners on technical correctness because of a perceived
     time crunch.** A wrong-but-shipped implementation is worse than a
     right implementation that takes longer. The user can't deploy
     something they can't trust.
  6. Journal the start/end/actual/estimated for each task -- builds the
     calibration dataset over time.

**Concrete revision (2026-05-19):**
  - Build "single component end-to-end" estimate: was 30-60 min, actual
    ~15-30 min. Cut by 50%.
  - Build "research tool + run on data + chart" estimate: was 60-90 min,
    actual ~30-45 min. Cut by 50%.
  - Multi-day mock sweep: estimated 60 min wall clock, actual ~50 min.
    Fine.

**Anti-pattern to avoid:**
  - "I'll do the simpler version because I might run out of time" --
    no, do the right version. If it takes longer, deliver less but
    correct.



## feedback_v2_only_hard_rule.md
---
name: feedback-v2-only-hard-rule
description: V2 features (185D layered) are the ONLY supported feature schema. No V1, no hybrid, no custom. All new code targets V2; all V1 code is technical debt to be removed.
metadata:
  type: feedback
---

**Rule**: V2 layered features (185D, per-family parquets under `DATA/ATLAS*/FEATURES_5s_v2/{L0,L1_*,L2_*,L3_*}/`) are the only sanctioned feature schema. Do NOT write code that produces 91D V1 features, do NOT add V1â†”V2 adapters, do NOT support both with a flag.

**Why:** User declared this 2026-05-24 during the ForwardPass unification design. Rationale: maintaining two feature schemas means two lookahead-audit surfaces, two parity-validation paths, two CNN training pipelines. The `0c001c1f` baseline-invalidation commit (lookahead in V1 `build_dataset.py`) is exactly the kind of bug duplicate paths invite. Cutting V1 is the lever that prevents that class of bug.

**How to apply:**
- New code: only V2. `BarState` (from `core_v2.FPS.state`) or equivalent V2 dataclass â€” never `state['features_79d']`.
- Refactoring: when touching V1 code, propose migrating it to V2, not patching it. If V2 migration is out of scope for the task, flag it and stop â€” don't extend V1.
- Deletions: V1-only files (e.g. `core_v2/v2_to_v1_inmemory.py`, the deleted `build_v2_to_v1_compat_cache.py`) are removable as soon as their last V1 caller is migrated.
- Live engine: live and offline must share the same feature path. The proposed unification is for live to drive bars through `ForwardPass(source='live')` so SFE math is invoked from one place â€” parity by construction. See [[project_forward_pass_unification_2026_05_24]] for the migration plan.

**Known V1 holdouts as of 2026-05-24 (to be migrated):**
- `training/nightmare_blended.py` â€” engine indexes `state['features_79d']`
- `core_v2/build_dataset.py` â€” writes 91D
- `training/feature_processor.py`, `training/live_feature_engine{,_v2}.py`, `training/compute_features.py` â€” produce 91D
- `live/engine_v2.py` â€” consumes 91D
- `training/cnn_entry_direction.py`, `training/cnn_trade_manager.py` â€” trained on 91D (CNN retraining required)
- `core_v2/features.py::extract_features` â€” V1 91D constructor

**Already V2-native (good):**
- `training/pipelines/v2_native.py`
- `training/sfe_ticker.py` (yields V2 BarState)
- `core_v2/FPS/forward_pass_system.py` (V2 reader)
- `tools/research/features_v2.py`
- `DATA/ATLAS_NT8/FEATURES_5s_v2/` on disk



## feedback_zigzag_conditional_table_confounds.md
---
name: zigzag-conditional-table-confounds
description: Confounds that fake signals in conditional-probability tables built on zigzag legs, and the discipline that catches them
metadata:
  type: feedback
---

When building empirical conditional-probability tables over zigzag-leg events
("if K low-range legs, then P(next leg range)"), the zigzag's own structure
injects confounds that manufacture a fake signal. Five were caught in the
2026-05-21 table arc ([[conditional-probability-table-2026-05-21]]). The
discipline:

1. **Never measure DIRECTION on leg-anchored windows.** Zigzag legs strictly
   alternate up/down/up/down, so any directional statistic over a window
   pinned to leg boundaries is eaten by alternation PARITY â€” it alternates by
   K, not by market behavior. `trend_continuation.py` v1's chop table read
   65/37/64/36/65 â€” pure parity. Fix: measure direction LEG-DECOUPLED â€” sign
   of the regression slope of raw closes over a LONG window (90 min) so no
   single leg dominates; the zigzag only marks the event.
2. **Measure the predictor over a FIXED window**, never one that co-grows
   with the outcome. `leg_chop_survival.py` v1 measured the chop ratio over
   the whole wide leg and correlated it with the wide leg's length â€”
   tautological. Fix: measure chop over the first K tight legs only.
3. **Read the FULL curve shape, not endpoints.** `leg_age_hazard.py`'s first
   verdict compared S(age 0.5min) vs S(age 18min), saw a drop, said
   "exhaustion" â€” but the curve is a HUMP (falls to a ~5-min trough, then
   recovers). An endpoint test skips the trough. Fix: find the trough/peak,
   test both arms.
4. **OOS-confirm every cell** â€” trust a cell only where IS and OOS agree
   (cf. [[quantile-selection-overfit]]).
5. **A FLAT oracle-zigzag forward pass is a hindsight partition** â€” its
   $/day is monotonic in zigzag subdivision, not valid cross-parameter
   ([[flat-pipeline-cross-param]]). The same offline-vs-causal trap recurs:
   `atr_consensus_measure.py`'s offline +$57/leg consensus signal INVERTED
   under a causal measurement (corr +0.34 -> -0.13).

**Why:** This is a real-money system; a confounded table that looks like a
signal would get a bad gate shipped. The user's standing rule on the zigzag
work: "if there's no parity it's back to the drawing board" â€” a directional
result that just tracks K-parity is no result.

**How to apply:** Before trusting any number in a new zigzag-leg
conditional-table entry, ask: is this a directional measure on a leg-anchored
window? does the predictor co-grow with the outcome? am I reading endpoints
of a non-monotone curve? does it replicate OOS? Build the validation gate
INTO the tool â€” `trend_continuation.py` prints PASS/FAIL on the parity check;
`leg_age_hazard.py` runs a shape-aware two-arm verdict, not an endpoint test.


