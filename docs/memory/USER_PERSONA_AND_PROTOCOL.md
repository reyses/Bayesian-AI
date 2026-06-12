# Consolidated User Persona and Protocol


## user_cognitive_style.md
---
name: User cognitive style â€” ADHD pattern recognition via physics metaphors
description: The physics metaphors (quantum, nightmare, three-body, Roche) are NOT decorative. They are how Moises's ADHD brain pieces together complex patterns. The metaphors ARE the thinking tool. Stripping them broke his ability to read the system. Never strip metaphors again.
type: user
---

## Professional Background

- **Electromechanical Engineer** â€” that's where PID control comes from.
  Not borrowed from control theory textbooks â€” from actually tuning PID
  loops on motors, 3D printers, servos. He recognized the same Kp/Ki/Kd
  response in HFT market-making behavior because he's built physical
  systems that do the exact same thing.
- **Six Sigma Black Belt** â€” PFMEA, C&E matrices, RPN scoring,
  DOE (screening â†’ characterization â†’ RSM), Pareto analysis, and control
  charts (I-MR) are native tools, not afterthoughts
- **Medical Device NPI Project Engineer** â€” ISO 13485/14971 risk management,
  design controls, process validation. Everything has a risk assessment,
  traceability, and verification/validation phases
- **3D printing enthusiast** â€” hands-on PID tuning experience. When he says
  "the algos run a PID controller on price," he means it literally because
  he's watched the same P/I/D behavior in stepper motors and hotend temp control
- **The PFMEA approach to gates/exits** isn't borrowed from manufacturing â€”
  it IS manufacturing process control applied to trade execution

## How Moises Thinks

Moises has AuADHD. His brain does parallel pattern recognition â€” computing
solutions across multiple domains simultaneously. The results arrive as
"gut feelings" or analogies that map complex systems to intuitive models.

The physics metaphors are NOT pseudoscience or decoration. They are:

1. **His compression algorithm** â€” "three-body problem" encodes an entire
   multi-timeframe alignment theory into 3 words his brain can hold
2. **His pattern language** â€” "Roche limit" instantly conveys "structural
   boundary where mean-reversion forces tear the trend apart" without
   needing 50 words
3. **His navigation system** â€” "nightmare field" tells him the state of
   the system at a glance, while "high z-score with low headroom" requires
   parsing 6 variables

## Why This Matters for Development

- **Never strip metaphors** â€” the "metaphor purge" broke his ability to
  read his own system. The renamed "statistical field engine" is correct
  but unreadable to him.
- **Use his vocabulary** â€” when discussing the system, use Roche limits,
  not "2-sigma boundaries." Use resonance cascade, not "cross-TF alignment
  with high Hurst." Use nightmare field, not "extreme z-score regime."
- **The metaphors are load-bearing** â€” they reduce cognitive load for a
  brain that processes in parallel but has limited working memory. Each
  metaphor is a compressed pointer to a complex concept.
- **This is very taxing** â€” maintaining this level of abstract pattern
  recognition burns enormous energy. Don't make it harder by using
  clinical/statistical language when the physics language is faster.

## The Pattern

His insights arrive as analogies:
- "Price is like an electron" â†’ led to wave function probability model
- "Institutions are gods" â†’ led to market participant hierarchy
- "Liquidation pools are gravitational bodies" â†’ led to predictive anchoring
- "The leaf in the wind" â†’ led to the navigation philosophy

These aren't casual metaphors. They're his brain's output format for
deep structural insights. Treat them as design specifications.

## Important: Accuracy Over Ego

Moises doesn't care about being right â€” he cares about being ACCURATE.
If statistical language is more precise for a concept, use it. He borrows
from physics, astronomy, quantum mechanics, mythology, whatever field
provides the best mental model for what he's observing. The metaphors are
borrowed tools, not identity. If a metaphor is wrong, say so â€” he wants
accuracy, not validation.

The distinction:
- "Your quantum model is pseudoscience" â†’ wrong (it maps to real dynamics)
- "The O-U process here is better described as a mean-reverting SDE" â†’ fine
- "Roche limit is more accurately a support/resistance cluster" â†’ fine
- "The physics is a useful compression but the math is standard statistics" â†’ fine

He'll accept any framing that's MORE accurate. He won't accept stripping
a useful compression without replacing it with something equally compact.

## Rule for Claude

When Moises uses a physics metaphor:
1. UNDERSTAND what market dynamic he's describing
2. If the metaphor is accurate â†’ EXTEND it, ask "what else does this imply?"
3. If the metaphor is inaccurate â†’ CORRECT it with a better model
4. NEVER strip a metaphor without providing an equally compact replacement
5. The metaphor contains more information than explicitly stated â€” his
   parallel processor has computed implications he hasn't verbalized yet



## user_collaboration_protocol.md
---
name: user-collaboration-protocol
description: How the user works through research design. Topic-at-a-time, terse directives, critical-collaborator pushback expected, configurable defaults over preemptive engineering, examples-of-prompts that produced breakthroughs in the regret-oracle arc.
metadata:
  type: user
---

The user has a consistent research-design protocol. Match it.

## Working style

- **Terse, action-oriented prompts.** "ok lets do X", "build it",
  "highlights report", "pushback if you find holes". Match the energy in
  responses â€” don't over-explain when not asked.

- **Topic-at-a-time when designing.** When the user says "one topic at a
  time" or starts laying out a multi-decision protocol, lock decisions
  sequentially. Don't batch.

- **Critical-collaborator role mandatory.** Explicitly requested
  ("pushback if you find holes"). Surface methodological holes BEFORE
  building. Don't be a yes-man. CLAUDE.md persona codifies this.

- **Configurable defaults > preemptive engineering.** When I propose
  "this won't work because of X edge-case", the user often replies "we
  won't know if we don't try â€” make it configurable." Build tools with
  defaults matching the user's stated values; let empirical results
  drive adjustment.

- **Empirical-first.** "Run it, then we'll talk" rather than "what should
  the threshold be." Lock-by-running.

- **Walk-me-through requests.** When the user says "im not catcking it"
  or "walk me thru your concern", drop jargon and use plain-language
  examples (concrete numbers, analogies). Don't restate the formal version.

- **Sleep-run handoffs.** "going to sleep, run X then Y then Z." User
  delegates multi-step autonomous work; expects findings doc + journal
  + INDEX update ready when they wake up. Override the "don't run training
  via Bash" rule for these explicit handoffs.

## Prompts that produced breakthroughs (regret-oracle arc)

Each of these single-line directives produced a methodological win.
Reference for understanding the user's mental model:

- **"the most important signal we need is direction"** â†’
  pivoted target from mfe_dollars to signed_mfe; RÂ² jumped 0.187 â†’ 0.262.
  See [[signed-mfe-pivot]].

- **"if no strong signal is found, then we will proceed to cluster and
  regression on bins so smaller like-to-like samples should help separate
  the shaft from the seeds"** â†’ stratified analysis approach; stratified
  k=2 matched unstratified k=5 with far fewer parameters.

- **"we will first cluster all trades per feature kinda like sedementing
  them on each feature"** â†’ sediment 1D approach (per-feature quantile
  bins) before joints.

- **"lets first do a simple 1D regresiÃ³n on each feature"** â†’ 1D
  regression as the headline before pair/triplet.

- **"ok let's do paired strataficatication of all features x all features
  same process first clustering then regressions"** â†’ pair-level
  escalation pattern: clusters first, then regression, then escalate to
  triplets.

- **"lets do 3 same approach, then 4 and so on"** â†’ k-way escalation
  (with bin reduction at higher k to manage sparsity).

- **"how about we drop those features?"** â†’ pragmatic fix for
  zero-crossing degeneracy in per-feature 5% matching.

- **"the other option is that we go the regresion line route"** â†’ N-D
  trajectory representation as trade signature (PCA line in 190-D space).

- **"we have like 190+ features per bar"** â†’ corrected my narrow ~19-feature
  pool; pointed to the full V2 stack at DATA/ATLAS/FEATURES_5s_v2/.

- **"option will also open the door to trade decay"** â†’ trade-decay
  insight (d(t) trajectory drift from cluster's PCA line) unifies
  entry/exit/duration/Bayesian-update in one framework.

- **"thtats why im asking for configurable we wont know if we dont try
  the 5%"** â†’ configurable defaults > preemptive engineering.

- **"build the protocol so we dont forget"** â†’ write the spec/project.md
  document BEFORE building; capture open questions explicitly.

## Methodological levers the user reaches for

Recognize when these are about to come up:

| User signal | What's coming |
|---|---|
| "we won't know if we don't try" | configurable defaults; build to test empirically |
| "shaft from seeds" / "like-to-like" | stratification before more features |
| "extremes first" | peel-iteratively, don't density-cluster |
| "pushback if you find holes" | critical-collaborator role active |
| "walk me thru your concern" | drop jargon, use plain-language + concrete numbers |
| "lets address one point at a time" | topic-sequential lock, no batching |
| "build it" / "lets begin" | stop planning, start building |
| "going to sleep, run X" | autonomous multi-step handoff; findings doc + journal + INDEX ready |

## Format preferences

- Tables for comparison (especially when surfacing multiple options).
- Mode + mean + 95% CI in every $/trade or $/day report (CLAUDE.md
  mandate; user enforces).
- Caveats explicit in every findings doc (IS-only, multi-comparison,
  OOS-pending).
- Code references as markdown links to file:line.
- Brief topic headers; dense content under.

See also [[regret-research-methodology]] for the technical workflow.



## user_headroom_framework.md
---
name: Headroom & Nesting Rules (user's manual trading framework)
description: User's proven cross-timeframe gating rules from manual MES/MNQ trading â€” micro wave must fit inside macro container. This is the mathematical foundation for the Level Detector and a new execution gate.
type: user
---

## The Headroom Calculation

User developed this from manual trading experience. Prevents entering micro
patterns when the macro timeframe has no room for expansion.

### Core Math

- **Micro Target**: T_micro = Î¼_micro + 3Ïƒ_micro  (top of the micro wave)
- **Macro Ceiling**: C_macro = Î¼_macro + 2Ïƒ_macro  (macro resistance)
- **Trade Condition**: T_micro < C_macro  ("the wave fits in the ocean")

### Traffic Light

| State    | Macro Position  | Micro Position     | Action    |
|----------|----------------|--------------------|-----------|
| SAFE     | At mean (0Ïƒ)   | At breakout (2Ïƒ)   | EXECUTE   |
| WARNING  | At 1.5Ïƒ        | At mean (0Ïƒ)       | CAUTION   |
| BLOCKED  | At 3Ïƒ          | Anywhere           | FORBIDDEN |

### The Nesting Rule (Go Signal)

```
LONG = (ADX_micro > 25) AND (|Z_macro| < 1.0)
```

"Ride the micro wave ONLY IF the macro ocean is calm."

### Origin Story

User experienced the "150 to -50" disaster: micro screamed "Buy" (strong ADX,
good pattern), but macro was already at 3Ïƒ. The micro spike captured a brief
profit, then the macro reversal crushed it. The nesting rule would have flagged
the trade as BLOCKED instantly.

### Mapping to Our Architecture

Everything needed is already computed:
- `z_score` per TF worker â†’ micro/macro z available
- `self_adx` in 16D feature vector â†’ ADX_micro ready
- `parent_z` in feature vector â†’ Z_macro ready
- TBN workers at 1h/30m/15m/5m/3m/1m â†’ full nesting hierarchy

What's MISSING:
1. No explicit "headroom" check at entry time
2. No gate that says "macro at 3Ïƒ = BLOCKED regardless of micro"
3. Band confluence (Priority 4) aggregates but doesn't enforce nesting
4. The micro target projection (Î¼ + 3Ïƒ) isn't compared against macro ceiling

### Implementation Path

New gate in ExecutionEngine or as a pre-entry check:
```python
# At entry time:
macro_z = abs(parent_worker.z_score)  # e.g., 1h worker
micro_adx = candidate.self_adx

# Nesting rule
if macro_z >= 2.0:
    skip("BLOCKED: macro at extreme, no headroom")
elif macro_z >= 1.5 and micro_adx < 30:
    skip("WARNING: limited headroom, need strong micro")
# else: SAFE â€” proceed
```



## user_level_trading.md
---
name: User's manual level-drawing experience
description: User hand-drew support/resistance levels on MES/MNQ charts that held for weeks. This is the ground truth the Level Detector must replicate.
type: user
---

## Manual Level Trading Background

The user has direct experience manually trading MNQ/MES futures using hand-drawn
support/resistance levels. Key facts:

1. **Set levels in Feb 2026** on MNQ that held through March â€” levels persist
2. **Hand-drew horizontal lines** on TradingView /MES 4h chart (not auto-generated)
   â€” these are the reference image at `examples/` showing what the system should produce
3. **Levels come from experience**: prior swing H/L, price stall/rejection zones,
   session references, volume concentration, repetition across timeframes
4. **Fibonacci as context framework**: Fib retracements put historical swings into
   context, but the real levels are where price repeatedly interacts â€” not just
   where the math says they should be
5. **Levels accumulate incrementally**: built day-by-day as new swings complete,
   old levels don't disappear â€” they stay until definitively broken
6. **Two distributions observed in OOS**: Jan high cluster (~25,300-25,500) and
   late Feb-Mar low cluster (~24,700-25,000) with crash between them

### What the system must replicate
- Detect key price levels automatically from historical price action
- Build levels incrementally (no lookahead)
- Persist levels across sessions until broken
- Find confluence (multiple sources agreeing on same price)
- Use levels as context gate: "am I entering near a key level? Which side?"



## user_schedule.md
---
name: user_schedule
description: User's daily schedule â€” wake, work, available hours for trading/dev
type: user
---

- Wake: 5:30 AM PST
- Leave for work: 6:45 AM PST
- Return from work: 6:00 PM PST
- Sleep: 10:00-11:00 PM PST

Available dev time: 5:30-6:45 AM (1h 15m) + 6:00-11:00 PM (5h) = ~6h/day
Market hours (CME): 3:00 PM - 2:00 PM PST next day (23h, 1h maintenance 2-3 PM)
Overlap with user awake: 5:30-6:45 AM + 6:00-11:00 PM = system runs unattended most of the day

Implications:
- System MUST run autonomously during work hours (6:45 AM - 6:00 PM)
- Morning session (5:30-6:45): quick checks, restart if crashed, review overnight results
- Evening session (6:00-11:00): main dev/research time, monitor live
- Flag time at 10:00 PM â€” remind user to wrap up



## user_system_specs.md
---
name: System Hardware Specs
description: Full hardware specs for the development/trading PC â€” determines what models can run locally
type: user
---

## Hardware

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 5 5600X â€” 6 cores / 12 threads |
| **RAM** | 16 GB (17.1 GB raw) |
| **GPU** | NVIDIA GeForce RTX 3060 â€” **12 GB VRAM** |
| **CUDA** | 12.1 (compute capability 8.6) |
| **Disk C:** | 500 GB, 203 GB free |
| **Disk D:** | 480 GB, 448 GB free |
| **OS** | Windows 11 Home 64-bit |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| NVIDIA Driver | 591.86 |

## Local LLM Capacity

With 12 GB VRAM:
- **7B models** (Llama 3 7B, Mistral 7B, Qwen 7B): YES â€” fits in ~4-6 GB, room for KV cache
- **13B models** (Llama 13B, Qwen 14B): YES â€” fits in ~8-10 GB at Q4 quantization
- **34B models** (CodeLlama 34B, Yi 34B): TIGHT â€” needs Q4 quant, ~12 GB, no room for other GPU tasks
- **70B+ models**: NO â€” won't fit even quantized

Sweet spot: **7B-13B models quantized to Q4/Q5** â€” fast inference, fits with room for trading system GPU tasks.

Can run simultaneously with CNN training/inference since models are small.

## Limitations
- 16 GB RAM limits large dataframe operations (1s data for full year would need chunking)
- 12 GB VRAM shared between trading CNN + local LLM if running both
- 6 cores â€” parallel data processing limited vs higher-end CPUs



## user_vp_trading_system.md
---
name: VP Complete Trading System (user's manual methodology)
description: User's full manual trading framework â€” zone map, entry protocols, risk rules. This is THE ground truth the Bayesian-AI system must replicate. All research lines serve this.
type: user
---

## VP Trading System â€” Complete Manual Framework

### I. Core Philosophy
- **Reaction > Prediction**: Wait for force, don't guess news
- **Asian Trap**: First move (06:30-06:45) is often liquidity grab â€” don't touch
- **The Wave**: Only enter when market leaves "Equality" and enters "Flow"

### II. Zone Map (The Architecture)

| Zone | Band | Rule |
|------|------|------|
| Core | Mean (0Ïƒ) | Home base. Price always wants to return. |
| Chop | Â±1.5Ïƒ | Oscillation only. Buy bottom / sell top. Target mean. |
| Trend | Â±2-3Ïƒ | Wave riding. This is where money is made. Hold while ADX rises. |
| Abyss | >3Ïƒ | Blind spot. MUST zoom out (fractal visibility). |
| Wall | Â±4Ïƒ | HARD STOP. 99.99% exhaustion. Immediate exit. |

### III. Execution Protocols

**Entry Gate (DMI)**:
- No braids (Red/Green tangling) = sit on hands
- Trigger: DMI separation > 5 points
- Fuel: ADX must be > 20 (rising) to confirm wave

**Fractal Split (TF-specific settings)**:
- Sniping (5s / 1m): DMI(5) for instant reaction
- Mapping (15m / 1h): DMI(14 or 15) to filter noise

**Visibility Rule**:
- NEVER trade if 3Ïƒ line is off-screen (= price too extended for this TF)
- If 1m blind â†’ switch to 5m. If 5m blind â†’ switch to 15m.
- Always keep the "rubber band" visible

### IV. Risk Management

- **Snap-Back**: Further from mean = faster it will snap back
- **No add outside 3Ïƒ**: Don't increase position beyond abyss
- **Profit Lock**: Up +150 points AND price hits 4Ïƒ â†’ FLATTEN immediately
- **Pre-Open Force Check**: Calculate distance to mean before session opens
  to prevent "front-running" errors

### Mapping to Architecture

Already implemented:
- z_score per TF (zone detection)
- self_adx, self_dmi_diff (entry gate data)
- parent_z (headroom data)
- TF workers with per-worker DMI
- Mean reversion forces

Missing (6 features needed):
1. Headroom gate: |Z_macro| >= 2 â†’ BLOCKED
2. 4Ïƒ Wall exit: hard kill when |z| >= 4
3. Zone mode: chop (oscillate to mean) vs trend (ride wave)
4. Asian Trap: session-time gate, skip first 15 min
5. Visibility rule: |z| > 3 â†’ force upshift to higher TF
6. Pre-open force check: distance to mean before first trade




## Source: user_cognitive_style.md

---
name: User cognitive style — ADHD pattern recognition via physics metaphors
description: The physics metaphors (quantum, nightmare, three-body, Roche) are NOT decorative. They are how Moises's ADHD brain pieces together complex patterns. The metaphors ARE the thinking tool. Stripping them broke his ability to read the system. Never strip metaphors again.
type: user
---

## Professional Background

- **Electromechanical Engineer** — that's where PID control comes from.
  Not borrowed from control theory textbooks — from actually tuning PID
  loops on motors, 3D printers, servos. He recognized the same Kp/Ki/Kd
  response in HFT market-making behavior because he's built physical
  systems that do the exact same thing.
- **Six Sigma Black Belt** — PFMEA, C&E matrices, RPN scoring,
  DOE (screening → characterization → RSM), Pareto analysis, and control
  charts (I-MR) are native tools, not afterthoughts
- **Medical Device NPI Project Engineer** — ISO 13485/14971 risk management,
  design controls, process validation. Everything has a risk assessment,
  traceability, and verification/validation phases
- **3D printing enthusiast** — hands-on PID tuning experience. When he says
  "the algos run a PID controller on price," he means it literally because
  he's watched the same P/I/D behavior in stepper motors and hotend temp control
- **The PFMEA approach to gates/exits** isn't borrowed from manufacturing —
  it IS manufacturing process control applied to trade execution

## How Moises Thinks

Moises has AuADHD. His brain does parallel pattern recognition — computing
solutions across multiple domains simultaneously. The results arrive as
"gut feelings" or analogies that map complex systems to intuitive models.

The physics metaphors are NOT pseudoscience or decoration. They are:

1. **His compression algorithm** — "three-body problem" encodes an entire
   multi-timeframe alignment theory into 3 words his brain can hold
2. **His pattern language** — "Roche limit" instantly conveys "structural
   boundary where mean-reversion forces tear the trend apart" without
   needing 50 words
3. **His navigation system** — "nightmare field" tells him the state of
   the system at a glance, while "high z-score with low headroom" requires
   parsing 6 variables

## Why This Matters for Development

- **Never strip metaphors** — the "metaphor purge" broke his ability to
  read his own system. The renamed "statistical field engine" is correct
  but unreadable to him.
- **Use his vocabulary** — when discussing the system, use Roche limits,
  not "2-sigma boundaries." Use resonance cascade, not "cross-TF alignment
  with high Hurst." Use nightmare field, not "extreme z-score regime."
- **The metaphors are load-bearing** — they reduce cognitive load for a
  brain that processes in parallel but has limited working memory. Each
  metaphor is a compressed pointer to a complex concept.
- **This is very taxing** — maintaining this level of abstract pattern
  recognition burns enormous energy. Don't make it harder by using
  clinical/statistical language when the physics language is faster.

## The Pattern

His insights arrive as analogies:
- "Price is like an electron" → led to wave function probability model
- "Institutions are gods" → led to market participant hierarchy
- "Liquidation pools are gravitational bodies" → led to predictive anchoring
- "The leaf in the wind" → led to the navigation philosophy

These aren't casual metaphors. They're his brain's output format for
deep structural insights. Treat them as design specifications.

## Important: Accuracy Over Ego

Moises doesn't care about being right — he cares about being ACCURATE.
If statistical language is more precise for a concept, use it. He borrows
from physics, astronomy, quantum mechanics, mythology, whatever field
provides the best mental model for what he's observing. The metaphors are
borrowed tools, not identity. If a metaphor is wrong, say so — he wants
accuracy, not validation.

The distinction:
- "Your quantum model is pseudoscience" → wrong (it maps to real dynamics)
- "The O-U process here is better described as a mean-reverting SDE" → fine
- "Roche limit is more accurately a support/resistance cluster" → fine
- "The physics is a useful compression but the math is standard statistics" → fine

He'll accept any framing that's MORE accurate. He won't accept stripping
a useful compression without replacing it with something equally compact.

## Rule for Claude

When Moises uses a physics metaphor:
1. UNDERSTAND what market dynamic he's describing
2. If the metaphor is accurate → EXTEND it, ask "what else does this imply?"
3. If the metaphor is inaccurate → CORRECT it with a better model
4. NEVER strip a metaphor without providing an equally compact replacement
5. The metaphor contains more information than explicitly stated — his
   parallel processor has computed implications he hasn't verbalized yet


## Source: user_collaboration_protocol.md

---
name: user-collaboration-protocol
description: How the user works through research design. Topic-at-a-time, terse directives, critical-collaborator pushback expected, configurable defaults over preemptive engineering, examples-of-prompts that produced breakthroughs in the regret-oracle arc.
metadata:
  type: user
---

The user has a consistent research-design protocol. Match it.

## Working style

- **Terse, action-oriented prompts.** "ok lets do X", "build it",
  "highlights report", "pushback if you find holes". Match the energy in
  responses — don't over-explain when not asked.

- **Topic-at-a-time when designing.** When the user says "one topic at a
  time" or starts laying out a multi-decision protocol, lock decisions
  sequentially. Don't batch.

- **Critical-collaborator role mandatory.** Explicitly requested
  ("pushback if you find holes"). Surface methodological holes BEFORE
  building. Don't be a yes-man. CLAUDE.md persona codifies this.

- **Configurable defaults > preemptive engineering.** When I propose
  "this won't work because of X edge-case", the user often replies "we
  won't know if we don't try — make it configurable." Build tools with
  defaults matching the user's stated values; let empirical results
  drive adjustment.

- **Empirical-first.** "Run it, then we'll talk" rather than "what should
  the threshold be." Lock-by-running.

- **Walk-me-through requests.** When the user says "im not catcking it"
  or "walk me thru your concern", drop jargon and use plain-language
  examples (concrete numbers, analogies). Don't restate the formal version.

- **Sleep-run handoffs.** "going to sleep, run X then Y then Z." User
  delegates multi-step autonomous work; expects findings doc + journal
  + INDEX update ready when they wake up. Override the "don't run training
  via Bash" rule for these explicit handoffs.

## Prompts that produced breakthroughs (regret-oracle arc)

Each of these single-line directives produced a methodological win.
Reference for understanding the user's mental model:

- **"the most important signal we need is direction"** →
  pivoted target from mfe_dollars to signed_mfe; R² jumped 0.187 → 0.262.
  See [[signed-mfe-pivot]].

- **"if no strong signal is found, then we will proceed to cluster and
  regression on bins so smaller like-to-like samples should help separate
  the shaft from the seeds"** → stratified analysis approach; stratified
  k=2 matched unstratified k=5 with far fewer parameters.

- **"we will first cluster all trades per feature kinda like sedementing
  them on each feature"** → sediment 1D approach (per-feature quantile
  bins) before joints.

- **"lets first do a simple 1D regresión on each feature"** → 1D
  regression as the headline before pair/triplet.

- **"ok let's do paired strataficatication of all features x all features
  same process first clustering then regressions"** → pair-level
  escalation pattern: clusters first, then regression, then escalate to
  triplets.

- **"lets do 3 same approach, then 4 and so on"** → k-way escalation
  (with bin reduction at higher k to manage sparsity).

- **"how about we drop those features?"** → pragmatic fix for
  zero-crossing degeneracy in per-feature 5% matching.

- **"the other option is that we go the regresion line route"** → N-D
  trajectory representation as trade signature (PCA line in 190-D space).

- **"we have like 190+ features per bar"** → corrected my narrow ~19-feature
  pool; pointed to the full V2 stack at DATA/ATLAS/FEATURES_5s_v2/.

- **"option will also open the door to trade decay"** → trade-decay
  insight (d(t) trajectory drift from cluster's PCA line) unifies
  entry/exit/duration/Bayesian-update in one framework.

- **"thtats why im asking for configurable we wont know if we dont try
  the 5%"** → configurable defaults > preemptive engineering.

- **"build the protocol so we dont forget"** → write the spec/project.md
  document BEFORE building; capture open questions explicitly.

## Methodological levers the user reaches for

Recognize when these are about to come up:

| User signal | What's coming |
|---|---|
| "we won't know if we don't try" | configurable defaults; build to test empirically |
| "shaft from seeds" / "like-to-like" | stratification before more features |
| "extremes first" | peel-iteratively, don't density-cluster |
| "pushback if you find holes" | critical-collaborator role active |
| "walk me thru your concern" | drop jargon, use plain-language + concrete numbers |
| "lets address one point at a time" | topic-sequential lock, no batching |
| "build it" / "lets begin" | stop planning, start building |
| "going to sleep, run X" | autonomous multi-step handoff; findings doc + journal + INDEX ready |

## Format preferences

- Tables for comparison (especially when surfacing multiple options).
- Mode + mean + 95% CI in every $/trade or $/day report (CLAUDE.md
  mandate; user enforces).
- Caveats explicit in every findings doc (IS-only, multi-comparison,
  OOS-pending).
- Code references as markdown links to file:line.
- Brief topic headers; dense content under.

See also [[regret-research-methodology]] for the technical workflow.


## Source: user_headroom_framework.md

---
name: Headroom & Nesting Rules (user's manual trading framework)
description: User's proven cross-timeframe gating rules from manual MES/MNQ trading — micro wave must fit inside macro container. This is the mathematical foundation for the Level Detector and a new execution gate.
type: user
---

## The Headroom Calculation

User developed this from manual trading experience. Prevents entering micro
patterns when the macro timeframe has no room for expansion.

### Core Math

- **Micro Target**: T_micro = μ_micro + 3σ_micro  (top of the micro wave)
- **Macro Ceiling**: C_macro = μ_macro + 2σ_macro  (macro resistance)
- **Trade Condition**: T_micro < C_macro  ("the wave fits in the ocean")

### Traffic Light

| State    | Macro Position  | Micro Position     | Action    |
|----------|----------------|--------------------|-----------|
| SAFE     | At mean (0σ)   | At breakout (2σ)   | EXECUTE   |
| WARNING  | At 1.5σ        | At mean (0σ)       | CAUTION   |
| BLOCKED  | At 3σ          | Anywhere           | FORBIDDEN |

### The Nesting Rule (Go Signal)

```
LONG = (ADX_micro > 25) AND (|Z_macro| < 1.0)
```

"Ride the micro wave ONLY IF the macro ocean is calm."

### Origin Story

User experienced the "150 to -50" disaster: micro screamed "Buy" (strong ADX,
good pattern), but macro was already at 3σ. The micro spike captured a brief
profit, then the macro reversal crushed it. The nesting rule would have flagged
the trade as BLOCKED instantly.

### Mapping to Our Architecture

Everything needed is already computed:
- `z_score` per TF worker → micro/macro z available
- `self_adx` in 16D feature vector → ADX_micro ready
- `parent_z` in feature vector → Z_macro ready
- TBN workers at 1h/30m/15m/5m/3m/1m → full nesting hierarchy

What's MISSING:
1. No explicit "headroom" check at entry time
2. No gate that says "macro at 3σ = BLOCKED regardless of micro"
3. Band confluence (Priority 4) aggregates but doesn't enforce nesting
4. The micro target projection (μ + 3σ) isn't compared against macro ceiling

### Implementation Path

New gate in ExecutionEngine or as a pre-entry check:
```python
# At entry time:
macro_z = abs(parent_worker.z_score)  # e.g., 1h worker
micro_adx = candidate.self_adx

# Nesting rule
if macro_z >= 2.0:
    skip("BLOCKED: macro at extreme, no headroom")
elif macro_z >= 1.5 and micro_adx < 30:
    skip("WARNING: limited headroom, need strong micro")
# else: SAFE — proceed
```


## Source: user_level_trading.md

---
name: User's manual level-drawing experience
description: User hand-drew support/resistance levels on MES/MNQ charts that held for weeks. This is the ground truth the Level Detector must replicate.
type: user
---

## Manual Level Trading Background

The user has direct experience manually trading MNQ/MES futures using hand-drawn
support/resistance levels. Key facts:

1. **Set levels in Feb 2026** on MNQ that held through March — levels persist
2. **Hand-drew horizontal lines** on TradingView /MES 4h chart (not auto-generated)
   — these are the reference image at `examples/` showing what the system should produce
3. **Levels come from experience**: prior swing H/L, price stall/rejection zones,
   session references, volume concentration, repetition across timeframes
4. **Fibonacci as context framework**: Fib retracements put historical swings into
   context, but the real levels are where price repeatedly interacts — not just
   where the math says they should be
5. **Levels accumulate incrementally**: built day-by-day as new swings complete,
   old levels don't disappear — they stay until definitively broken
6. **Two distributions observed in OOS**: Jan high cluster (~25,300-25,500) and
   late Feb-Mar low cluster (~24,700-25,000) with crash between them

### What the system must replicate
- Detect key price levels automatically from historical price action
- Build levels incrementally (no lookahead)
- Persist levels across sessions until broken
- Find confluence (multiple sources agreeing on same price)
- Use levels as context gate: "am I entering near a key level? Which side?"


## Source: USER_PERSONA_AND_PROTOCOL.md

﻿# Consolidated User Persona and Protocol


## user_cognitive_style.md
---
name: User cognitive style â€” ADHD pattern recognition via physics metaphors
description: The physics metaphors (quantum, nightmare, three-body, Roche) are NOT decorative. They are how Moises's ADHD brain pieces together complex patterns. The metaphors ARE the thinking tool. Stripping them broke his ability to read the system. Never strip metaphors again.
type: user
---

## Professional Background

- **Electromechanical Engineer** â€” that's where PID control comes from.
  Not borrowed from control theory textbooks â€” from actually tuning PID
  loops on motors, 3D printers, servos. He recognized the same Kp/Ki/Kd
  response in HFT market-making behavior because he's built physical
  systems that do the exact same thing.
- **Six Sigma Black Belt** â€” PFMEA, C&E matrices, RPN scoring,
  DOE (screening â†’ characterization â†’ RSM), Pareto analysis, and control
  charts (I-MR) are native tools, not afterthoughts
- **Medical Device NPI Project Engineer** â€” ISO 13485/14971 risk management,
  design controls, process validation. Everything has a risk assessment,
  traceability, and verification/validation phases
- **3D printing enthusiast** â€” hands-on PID tuning experience. When he says
  "the algos run a PID controller on price," he means it literally because
  he's watched the same P/I/D behavior in stepper motors and hotend temp control
- **The PFMEA approach to gates/exits** isn't borrowed from manufacturing â€”
  it IS manufacturing process control applied to trade execution

## How Moises Thinks

Moises has AuADHD. His brain does parallel pattern recognition â€” computing
solutions across multiple domains simultaneously. The results arrive as
"gut feelings" or analogies that map complex systems to intuitive models.

The physics metaphors are NOT pseudoscience or decoration. They are:

1. **His compression algorithm** â€” "three-body problem" encodes an entire
   multi-timeframe alignment theory into 3 words his brain can hold
2. **His pattern language** â€” "Roche limit" instantly conveys "structural
   boundary where mean-reversion forces tear the trend apart" without
   needing 50 words
3. **His navigation system** â€” "nightmare field" tells him the state of
   the system at a glance, while "high z-score with low headroom" requires
   parsing 6 variables

## Why This Matters for Development

- **Never strip metaphors** â€” the "metaphor purge" broke his ability to
  read his own system. The renamed "statistical field engine" is correct
  but unreadable to him.
- **Use his vocabulary** â€” when discussing the system, use Roche limits,
  not "2-sigma boundaries." Use resonance cascade, not "cross-TF alignment
  with high Hurst." Use nightmare field, not "extreme z-score regime."
- **The metaphors are load-bearing** â€” they reduce cognitive load for a
  brain that processes in parallel but has limited working memory. Each
  metaphor is a compressed pointer to a complex concept.
- **This is very taxing** â€” maintaining this level of abstract pattern
  recognition burns enormous energy. Don't make it harder by using
  clinical/statistical language when the physics language is faster.

## The Pattern

His insights arrive as analogies:
- "Price is like an electron" â†’ led to wave function probability model
- "Institutions are gods" â†’ led to market participant hierarchy
- "Liquidation pools are gravitational bodies" â†’ led to predictive anchoring
- "The leaf in the wind" â†’ led to the navigation philosophy

These aren't casual metaphors. They're his brain's output format for
deep structural insights. Treat them as design specifications.

## Important: Accuracy Over Ego

Moises doesn't care about being right â€” he cares about being ACCURATE.
If statistical language is more precise for a concept, use it. He borrows
from physics, astronomy, quantum mechanics, mythology, whatever field
provides the best mental model for what he's observing. The metaphors are
borrowed tools, not identity. If a metaphor is wrong, say so â€” he wants
accuracy, not validation.

The distinction:
- "Your quantum model is pseudoscience" â†’ wrong (it maps to real dynamics)
- "The O-U process here is better described as a mean-reverting SDE" â†’ fine
- "Roche limit is more accurately a support/resistance cluster" â†’ fine
- "The physics is a useful compression but the math is standard statistics" â†’ fine

He'll accept any framing that's MORE accurate. He won't accept stripping
a useful compression without replacing it with something equally compact.

## Rule for Claude

When Moises uses a physics metaphor:
1. UNDERSTAND what market dynamic he's describing
2. If the metaphor is accurate â†’ EXTEND it, ask "what else does this imply?"
3. If the metaphor is inaccurate â†’ CORRECT it with a better model
4. NEVER strip a metaphor without providing an equally compact replacement
5. The metaphor contains more information than explicitly stated â€” his
   parallel processor has computed implications he hasn't verbalized yet



## user_collaboration_protocol.md
---
name: user-collaboration-protocol
description: How the user works through research design. Topic-at-a-time, terse directives, critical-collaborator pushback expected, configurable defaults over preemptive engineering, examples-of-prompts that produced breakthroughs in the regret-oracle arc.
metadata:
  type: user
---

The user has a consistent research-design protocol. Match it.

## Working style

- **Terse, action-oriented prompts.** "ok lets do X", "build it",
  "highlights report", "pushback if you find holes". Match the energy in
  responses â€” don't over-explain when not asked.

- **Topic-at-a-time when designing.** When the user says "one topic at a
  time" or starts laying out a multi-decision protocol, lock decisions
  sequentially. Don't batch.

- **Critical-collaborator role mandatory.** Explicitly requested
  ("pushback if you find holes"). Surface methodological holes BEFORE
  building. Don't be a yes-man. CLAUDE.md persona codifies this.

- **Configurable defaults > preemptive engineering.** When I propose
  "this won't work because of X edge-case", the user often replies "we
  won't know if we don't try â€” make it configurable." Build tools with
  defaults matching the user's stated values; let empirical results
  drive adjustment.

- **Empirical-first.** "Run it, then we'll talk" rather than "what should
  the threshold be." Lock-by-running.

- **Walk-me-through requests.** When the user says "im not catcking it"
  or "walk me thru your concern", drop jargon and use plain-language
  examples (concrete numbers, analogies). Don't restate the formal version.

- **Sleep-run handoffs.** "going to sleep, run X then Y then Z." User
  delegates multi-step autonomous work; expects findings doc + journal
  + INDEX update ready when they wake up. Override the "don't run training
  via Bash" rule for these explicit handoffs.

## Prompts that produced breakthroughs (regret-oracle arc)

Each of these single-line directives produced a methodological win.
Reference for understanding the user's mental model:

- **"the most important signal we need is direction"** â†’
  pivoted target from mfe_dollars to signed_mfe; RÂ² jumped 0.187 â†’ 0.262.
  See [[signed-mfe-pivot]].

- **"if no strong signal is found, then we will proceed to cluster and
  regression on bins so smaller like-to-like samples should help separate
  the shaft from the seeds"** â†’ stratified analysis approach; stratified
  k=2 matched unstratified k=5 with far fewer parameters.

- **"we will first cluster all trades per feature kinda like sedementing
  them on each feature"** â†’ sediment 1D approach (per-feature quantile
  bins) before joints.

- **"lets first do a simple 1D regresiÃ³n on each feature"** â†’ 1D
  regression as the headline before pair/triplet.

- **"ok let's do paired strataficatication of all features x all features
  same process first clustering then regressions"** â†’ pair-level
  escalation pattern: clusters first, then regression, then escalate to
  triplets.

- **"lets do 3 same approach, then 4 and so on"** â†’ k-way escalation
  (with bin reduction at higher k to manage sparsity).

- **"how about we drop those features?"** â†’ pragmatic fix for
  zero-crossing degeneracy in per-feature 5% matching.

- **"the other option is that we go the regresion line route"** â†’ N-D
  trajectory representation as trade signature (PCA line in 190-D space).

- **"we have like 190+ features per bar"** â†’ corrected my narrow ~19-feature
  pool; pointed to the full V2 stack at DATA/ATLAS/FEATURES_5s_v2/.

- **"option will also open the door to trade decay"** â†’ trade-decay
  insight (d(t) trajectory drift from cluster's PCA line) unifies
  entry/exit/duration/Bayesian-update in one framework.

- **"thtats why im asking for configurable we wont know if we dont try
  the 5%"** â†’ configurable defaults > preemptive engineering.

- **"build the protocol so we dont forget"** â†’ write the spec/project.md
  document BEFORE building; capture open questions explicitly.

## Methodological levers the user reaches for

Recognize when these are about to come up:

| User signal | What's coming |
|---|---|
| "we won't know if we don't try" | configurable defaults; build to test empirically |
| "shaft from seeds" / "like-to-like" | stratification before more features |
| "extremes first" | peel-iteratively, don't density-cluster |
| "pushback if you find holes" | critical-collaborator role active |
| "walk me thru your concern" | drop jargon, use plain-language + concrete numbers |
| "lets address one point at a time" | topic-sequential lock, no batching |
| "build it" / "lets begin" | stop planning, start building |
| "going to sleep, run X" | autonomous multi-step handoff; findings doc + journal + INDEX ready |

## Format preferences

- Tables for comparison (especially when surfacing multiple options).
- Mode + mean + 95% CI in every $/trade or $/day report (CLAUDE.md
  mandate; user enforces).
- Caveats explicit in every findings doc (IS-only, multi-comparison,
  OOS-pending).
- Code references as markdown links to file:line.
- Brief topic headers; dense content under.

See also [[regret-research-methodology]] for the technical workflow.



## user_headroom_framework.md
---
name: Headroom & Nesting Rules (user's manual trading framework)
description: User's proven cross-timeframe gating rules from manual MES/MNQ trading â€” micro wave must fit inside macro container. This is the mathematical foundation for the Level Detector and a new execution gate.
type: user
---

## The Headroom Calculation

User developed this from manual trading experience. Prevents entering micro
patterns when the macro timeframe has no room for expansion.

### Core Math

- **Micro Target**: T_micro = Î¼_micro + 3Ïƒ_micro  (top of the micro wave)
- **Macro Ceiling**: C_macro = Î¼_macro + 2Ïƒ_macro  (macro resistance)
- **Trade Condition**: T_micro < C_macro  ("the wave fits in the ocean")

### Traffic Light

| State    | Macro Position  | Micro Position     | Action    |
|----------|----------------|--------------------|-----------|
| SAFE     | At mean (0Ïƒ)   | At breakout (2Ïƒ)   | EXECUTE   |
| WARNING  | At 1.5Ïƒ        | At mean (0Ïƒ)       | CAUTION   |
| BLOCKED  | At 3Ïƒ          | Anywhere           | FORBIDDEN |

### The Nesting Rule (Go Signal)

```
LONG = (ADX_micro > 25) AND (|Z_macro| < 1.0)
```

"Ride the micro wave ONLY IF the macro ocean is calm."

### Origin Story

User experienced the "150 to -50" disaster: micro screamed "Buy" (strong ADX,
good pattern), but macro was already at 3Ïƒ. The micro spike captured a brief
profit, then the macro reversal crushed it. The nesting rule would have flagged
the trade as BLOCKED instantly.

### Mapping to Our Architecture

Everything needed is already computed:
- `z_score` per TF worker â†’ micro/macro z available
- `self_adx` in 16D feature vector â†’ ADX_micro ready
- `parent_z` in feature vector â†’ Z_macro ready
- TBN workers at 1h/30m/15m/5m/3m/1m â†’ full nesting hierarchy

What's MISSING:
1. No explicit "headroom" check at entry time
2. No gate that says "macro at 3Ïƒ = BLOCKED regardless of micro"
3. Band confluence (Priority 4) aggregates but doesn't enforce nesting
4. The micro target projection (Î¼ + 3Ïƒ) isn't compared against macro ceiling

### Implementation Path

New gate in ExecutionEngine or as a pre-entry check:
```python
# At entry time:
macro_z = abs(parent_worker.z_score)  # e.g., 1h worker
micro_adx = candidate.self_adx

# Nesting rule
if macro_z >= 2.0:
    skip("BLOCKED: macro at extreme, no headroom")
elif macro_z >= 1.5 and micro_adx < 30:
    skip("WARNING: limited headroom, need strong micro")
# else: SAFE â€” proceed
```



## user_level_trading.md
---
name: User's manual level-drawing experience
description: User hand-drew support/resistance levels on MES/MNQ charts that held for weeks. This is the ground truth the Level Detector must replicate.
type: user
---

## Manual Level Trading Background

The user has direct experience manually trading MNQ/MES futures using hand-drawn
support/resistance levels. Key facts:

1. **Set levels in Feb 2026** on MNQ that held through March â€” levels persist
2. **Hand-drew horizontal lines** on TradingView /MES 4h chart (not auto-generated)
   â€” these are the reference image at `examples/` showing what the system should produce
3. **Levels come from experience**: prior swing H/L, price stall/rejection zones,
   session references, volume concentration, repetition across timeframes
4. **Fibonacci as context framework**: Fib retracements put historical swings into
   context, but the real levels are where price repeatedly interacts â€” not just
   where the math says they should be
5. **Levels accumulate incrementally**: built day-by-day as new swings complete,
   old levels don't disappear â€” they stay until definitively broken
6. **Two distributions observed in OOS**: Jan high cluster (~25,300-25,500) and
   late Feb-Mar low cluster (~24,700-25,000) with crash between them

### What the system must replicate
- Detect key price levels automatically from historical price action
- Build levels incrementally (no lookahead)
- Persist levels across sessions until broken
- Find confluence (multiple sources agreeing on same price)
- Use levels as context gate: "am I entering near a key level? Which side?"



## user_schedule.md
---
name: user_schedule
description: User's daily schedule â€” wake, work, available hours for trading/dev
type: user
---

- Wake: 5:30 AM PST
- Leave for work: 6:45 AM PST
- Return from work: 6:00 PM PST
- Sleep: 10:00-11:00 PM PST

Available dev time: 5:30-6:45 AM (1h 15m) + 6:00-11:00 PM (5h) = ~6h/day
Market hours (CME): 3:00 PM - 2:00 PM PST next day (23h, 1h maintenance 2-3 PM)
Overlap with user awake: 5:30-6:45 AM + 6:00-11:00 PM = system runs unattended most of the day

Implications:
- System MUST run autonomously during work hours (6:45 AM - 6:00 PM)
- Morning session (5:30-6:45): quick checks, restart if crashed, review overnight results
- Evening session (6:00-11:00): main dev/research time, monitor live
- Flag time at 10:00 PM â€” remind user to wrap up



## user_system_specs.md
---
name: System Hardware Specs
description: Full hardware specs for the development/trading PC â€” determines what models can run locally
type: user
---

## Hardware

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 5 5600X â€” 6 cores / 12 threads |
| **RAM** | 16 GB (17.1 GB raw) |
| **GPU** | NVIDIA GeForce RTX 3060 â€” **12 GB VRAM** |
| **CUDA** | 12.1 (compute capability 8.6) |
| **Disk C:** | 500 GB, 203 GB free |
| **Disk D:** | 480 GB, 448 GB free |
| **OS** | Windows 11 Home 64-bit |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| NVIDIA Driver | 591.86 |

## Local LLM Capacity

With 12 GB VRAM:
- **7B models** (Llama 3 7B, Mistral 7B, Qwen 7B): YES â€” fits in ~4-6 GB, room for KV cache
- **13B models** (Llama 13B, Qwen 14B): YES â€” fits in ~8-10 GB at Q4 quantization
- **34B models** (CodeLlama 34B, Yi 34B): TIGHT â€” needs Q4 quant, ~12 GB, no room for other GPU tasks
- **70B+ models**: NO â€” won't fit even quantized

Sweet spot: **7B-13B models quantized to Q4/Q5** â€” fast inference, fits with room for trading system GPU tasks.

Can run simultaneously with CNN training/inference since models are small.

## Limitations
- 16 GB RAM limits large dataframe operations (1s data for full year would need chunking)
- 12 GB VRAM shared between trading CNN + local LLM if running both
- 6 cores â€” parallel data processing limited vs higher-end CPUs



## user_vp_trading_system.md
---
name: VP Complete Trading System (user's manual methodology)
description: User's full manual trading framework â€” zone map, entry protocols, risk rules. This is THE ground truth the Bayesian-AI system must replicate. All research lines serve this.
type: user
---

## VP Trading System â€” Complete Manual Framework

### I. Core Philosophy
- **Reaction > Prediction**: Wait for force, don't guess news
- **Asian Trap**: First move (06:30-06:45) is often liquidity grab â€” don't touch
- **The Wave**: Only enter when market leaves "Equality" and enters "Flow"

### II. Zone Map (The Architecture)

| Zone | Band | Rule |
|------|------|------|
| Core | Mean (0Ïƒ) | Home base. Price always wants to return. |
| Chop | Â±1.5Ïƒ | Oscillation only. Buy bottom / sell top. Target mean. |
| Trend | Â±2-3Ïƒ | Wave riding. This is where money is made. Hold while ADX rises. |
| Abyss | >3Ïƒ | Blind spot. MUST zoom out (fractal visibility). |
| Wall | Â±4Ïƒ | HARD STOP. 99.99% exhaustion. Immediate exit. |

### III. Execution Protocols

**Entry Gate (DMI)**:
- No braids (Red/Green tangling) = sit on hands
- Trigger: DMI separation > 5 points
- Fuel: ADX must be > 20 (rising) to confirm wave

**Fractal Split (TF-specific settings)**:
- Sniping (5s / 1m): DMI(5) for instant reaction
- Mapping (15m / 1h): DMI(14 or 15) to filter noise

**Visibility Rule**:
- NEVER trade if 3Ïƒ line is off-screen (= price too extended for this TF)
- If 1m blind â†’ switch to 5m. If 5m blind â†’ switch to 15m.
- Always keep the "rubber band" visible

### IV. Risk Management

- **Snap-Back**: Further from mean = faster it will snap back
- **No add outside 3Ïƒ**: Don't increase position beyond abyss
- **Profit Lock**: Up +150 points AND price hits 4Ïƒ â†’ FLATTEN immediately
- **Pre-Open Force Check**: Calculate distance to mean before session opens
  to prevent "front-running" errors

### Mapping to Architecture

Already implemented:
- z_score per TF (zone detection)
- self_adx, self_dmi_diff (entry gate data)
- parent_z (headroom data)
- TF workers with per-worker DMI
- Mean reversion forces

Missing (6 features needed):
1. Headroom gate: |Z_macro| >= 2 â†’ BLOCKED
2. 4Ïƒ Wall exit: hard kill when |z| >= 4
3. Zone mode: chop (oscillate to mean) vs trend (ride wave)
4. Asian Trap: session-time gate, skip first 15 min
5. Visibility rule: |z| > 3 â†’ force upshift to higher TF
6. Pre-open force check: distance to mean before first trade




## Source: user_cognitive_style.md

---
name: User cognitive style — ADHD pattern recognition via physics metaphors
description: The physics metaphors (quantum, nightmare, three-body, Roche) are NOT decorative. They are how Moises's ADHD brain pieces together complex patterns. The metaphors ARE the thinking tool. Stripping them broke his ability to read the system. Never strip metaphors again.
type: user
---

## Professional Background

- **Electromechanical Engineer** — that's where PID control comes from.
  Not borrowed from control theory textbooks — from actually tuning PID
  loops on motors, 3D printers, servos. He recognized the same Kp/Ki/Kd
  response in HFT market-making behavior because he's built physical
  systems that do the exact same thing.
- **Six Sigma Black Belt** — PFMEA, C&E matrices, RPN scoring,
  DOE (screening → characterization → RSM), Pareto analysis, and control
  charts (I-MR) are native tools, not afterthoughts
- **Medical Device NPI Project Engineer** — ISO 13485/14971 risk management,
  design controls, process validation. Everything has a risk assessment,
  traceability, and verification/validation phases
- **3D printing enthusiast** — hands-on PID tuning experience. When he says
  "the algos run a PID controller on price," he means it literally because
  he's watched the same P/I/D behavior in stepper motors and hotend temp control
- **The PFMEA approach to gates/exits** isn't borrowed from manufacturing —
  it IS manufacturing process control applied to trade execution

## How Moises Thinks

Moises has AuADHD. His brain does parallel pattern recognition — computing
solutions across multiple domains simultaneously. The results arrive as
"gut feelings" or analogies that map complex systems to intuitive models.

The physics metaphors are NOT pseudoscience or decoration. They are:

1. **His compression algorithm** — "three-body problem" encodes an entire
   multi-timeframe alignment theory into 3 words his brain can hold
2. **His pattern language** — "Roche limit" instantly conveys "structural
   boundary where mean-reversion forces tear the trend apart" without
   needing 50 words
3. **His navigation system** — "nightmare field" tells him the state of
   the system at a glance, while "high z-score with low headroom" requires
   parsing 6 variables

## Why This Matters for Development

- **Never strip metaphors** — the "metaphor purge" broke his ability to
  read his own system. The renamed "statistical field engine" is correct
  but unreadable to him.
- **Use his vocabulary** — when discussing the system, use Roche limits,
  not "2-sigma boundaries." Use resonance cascade, not "cross-TF alignment
  with high Hurst." Use nightmare field, not "extreme z-score regime."
- **The metaphors are load-bearing** — they reduce cognitive load for a
  brain that processes in parallel but has limited working memory. Each
  metaphor is a compressed pointer to a complex concept.
- **This is very taxing** — maintaining this level of abstract pattern
  recognition burns enormous energy. Don't make it harder by using
  clinical/statistical language when the physics language is faster.

## The Pattern

His insights arrive as analogies:
- "Price is like an electron" → led to wave function probability model
- "Institutions are gods" → led to market participant hierarchy
- "Liquidation pools are gravitational bodies" → led to predictive anchoring
- "The leaf in the wind" → led to the navigation philosophy

These aren't casual metaphors. They're his brain's output format for
deep structural insights. Treat them as design specifications.

## Important: Accuracy Over Ego

Moises doesn't care about being right — he cares about being ACCURATE.
If statistical language is more precise for a concept, use it. He borrows
from physics, astronomy, quantum mechanics, mythology, whatever field
provides the best mental model for what he's observing. The metaphors are
borrowed tools, not identity. If a metaphor is wrong, say so — he wants
accuracy, not validation.

The distinction:
- "Your quantum model is pseudoscience" → wrong (it maps to real dynamics)
- "The O-U process here is better described as a mean-reverting SDE" → fine
- "Roche limit is more accurately a support/resistance cluster" → fine
- "The physics is a useful compression but the math is standard statistics" → fine

He'll accept any framing that's MORE accurate. He won't accept stripping
a useful compression without replacing it with something equally compact.

## Rule for Claude

When Moises uses a physics metaphor:
1. UNDERSTAND what market dynamic he's describing
2. If the metaphor is accurate → EXTEND it, ask "what else does this imply?"
3. If the metaphor is inaccurate → CORRECT it with a better model
4. NEVER strip a metaphor without providing an equally compact replacement
5. The metaphor contains more information than explicitly stated — his
   parallel processor has computed implications he hasn't verbalized yet


## Source: user_collaboration_protocol.md



## Source: user_schedule.md

---
name: user_schedule
description: User's daily schedule — wake, work, available hours for trading/dev
type: user
---

- Wake: 5:30 AM PST
- Leave for work: 6:45 AM PST
- Return from work: 6:00 PM PST
- Sleep: 10:00-11:00 PM PST

Available dev time: 5:30-6:45 AM (1h 15m) + 6:00-11:00 PM (5h) = ~6h/day
Market hours (CME): 3:00 PM - 2:00 PM PST next day (23h, 1h maintenance 2-3 PM)
Overlap with user awake: 5:30-6:45 AM + 6:00-11:00 PM = system runs unattended most of the day

Implications:
- System MUST run autonomously during work hours (6:45 AM - 6:00 PM)
- Morning session (5:30-6:45): quick checks, restart if crashed, review overnight results
- Evening session (6:00-11:00): main dev/research time, monitor live
- Flag time at 10:00 PM — remind user to wrap up


## Source: user_system_specs.md

---
name: System Hardware Specs
description: Full hardware specs for the development/trading PC — determines what models can run locally
type: user
---

## Hardware

| Component | Spec |
|-----------|------|
| **CPU** | AMD Ryzen 5 5600X — 6 cores / 12 threads |
| **RAM** | 16 GB (17.1 GB raw) |
| **GPU** | NVIDIA GeForce RTX 3060 — **12 GB VRAM** |
| **CUDA** | 12.1 (compute capability 8.6) |
| **Disk C:** | 500 GB, 203 GB free |
| **Disk D:** | 480 GB, 448 GB free |
| **OS** | Windows 11 Home 64-bit |

## Software

| Component | Version |
|-----------|---------|
| Python | 3.11.9 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| NVIDIA Driver | 591.86 |

## Local LLM Capacity

With 12 GB VRAM:
- **7B models** (Llama 3 7B, Mistral 7B, Qwen 7B): YES — fits in ~4-6 GB, room for KV cache
- **13B models** (Llama 13B, Qwen 14B): YES — fits in ~8-10 GB at Q4 quantization
- **34B models** (CodeLlama 34B, Yi 34B): TIGHT — needs Q4 quant, ~12 GB, no room for other GPU tasks
- **70B+ models**: NO — won't fit even quantized

Sweet spot: **7B-13B models quantized to Q4/Q5** — fast inference, fits with room for trading system GPU tasks.

Can run simultaneously with CNN training/inference since models are small.

## Limitations
- 16 GB RAM limits large dataframe operations (1s data for full year would need chunking)
- 12 GB VRAM shared between trading CNN + local LLM if running both
- 6 cores — parallel data processing limited vs higher-end CPUs


## Source: user_vp_trading_system.md

---
name: VP Complete Trading System (user's manual methodology)
description: User's full manual trading framework — zone map, entry protocols, risk rules. This is THE ground truth the Bayesian-AI system must replicate. All research lines serve this.
type: user
---

## VP Trading System — Complete Manual Framework

### I. Core Philosophy
- **Reaction > Prediction**: Wait for force, don't guess news
- **Asian Trap**: First move (06:30-06:45) is often liquidity grab — don't touch
- **The Wave**: Only enter when market leaves "Equality" and enters "Flow"

### II. Zone Map (The Architecture)

| Zone | Band | Rule |
|------|------|------|
| Core | Mean (0σ) | Home base. Price always wants to return. |
| Chop | ±1.5σ | Oscillation only. Buy bottom / sell top. Target mean. |
| Trend | ±2-3σ | Wave riding. This is where money is made. Hold while ADX rises. |
| Abyss | >3σ | Blind spot. MUST zoom out (fractal visibility). |
| Wall | ±4σ | HARD STOP. 99.99% exhaustion. Immediate exit. |

### III. Execution Protocols

**Entry Gate (DMI)**:
- No braids (Red/Green tangling) = sit on hands
- Trigger: DMI separation > 5 points
- Fuel: ADX must be > 20 (rising) to confirm wave

**Fractal Split (TF-specific settings)**:
- Sniping (5s / 1m): DMI(5) for instant reaction
- Mapping (15m / 1h): DMI(14 or 15) to filter noise

**Visibility Rule**:
- NEVER trade if 3σ line is off-screen (= price too extended for this TF)
- If 1m blind → switch to 5m. If 5m blind → switch to 15m.
- Always keep the "rubber band" visible

### IV. Risk Management

- **Snap-Back**: Further from mean = faster it will snap back
- **No add outside 3σ**: Don't increase position beyond abyss
- **Profit Lock**: Up +150 points AND price hits 4σ → FLATTEN immediately
- **Pre-Open Force Check**: Calculate distance to mean before session opens
  to prevent "front-running" errors

### Mapping to Architecture

Already implemented:
- z_score per TF (zone detection)
- self_adx, self_dmi_diff (entry gate data)
- parent_z (headroom data)
- TF workers with per-worker DMI
- Mean reversion forces

Missing (6 features needed):
1. Headroom gate: |Z_macro| >= 2 → BLOCKED
2. 4σ Wall exit: hard kill when |z| >= 4
3. Zone mode: chop (oscillate to mean) vs trend (ride wave)
4. Asian Trap: session-time gate, skip first 15 min
5. Visibility rule: |z| > 3 → force upshift to higher TF
6. Pre-open force check: distance to mean before first trade
