# AG Phase 3B — Joint Bayesian evidence model (supersedes per-concept verdicts)

**User direction (Moises, 2026-07-09):** this is NOT a per-concept
if-this-then-that sweep. ALL NT8 signals are evidence terms in ONE Bayesian
likelihood framework. The deliverable is a per-bar POSTERIOR of the response,
built from all signals jointly — the confluence is the object of study.

## Architecture

1. **Event layer (per bar, all concepts at once)**: every catalog concept
   becomes a boolean/graded event flag computed causally — pivot touch, VWAP
   pullback, APZ touch, exhaustion wick, momentum exhaustion, structure
   break, round-number proximity, squeeze state, ADX regime, divergence
   flags, POC/VA position, etc. One row per bar, one column per signal.
2. **Response: PER-CONCEPT, not shared (user correction 2026-07-09).**
   Each signal is measured against ITS OWN claimed response, pre-registered
   from the article — confounding all signals to one expectation is itself a
   bias. Examples:
   - level/VWAP/APZ/pivot touches → directional first-touch ±k·σ (bounce)
   - squeeze → volatility response (range expansion within H, direction-free)
   - divergences / exhaustion wicks → turn response (swing reversal within H)
   - ORB / structure break → continuation response
   - ADX / regime flags → not an event at all: a CONDITIONING variable
   Each concept also declares its own natural horizon H. Same measurement
   hygiene everywhere (σ-scaled magnitudes, matched + phantom nulls,
   debounce, day-block CIs, both years).
3. **Per-signal likelihood tables are FIRST-CLASS**: each signal is counted
   and validated individually, against its own expectation — these tables
   are findings in their own right, not just building blocks.
4. **The joint model (the actual deliverable)**:
   - Signals enter the joint model as SEPARATE, individually-calibrated
     evidence features — each keeps its identity and its own expectation.
     Do NOT pre-merge, cluster, or discard signals as "redundant" before
     fitting: apparent co-firing can be two different measurements agreeing,
     and deduplicating by hand assumes they mean the same thing (the exact
     bias the user flagged).
   - Fit a regularized logistic on all features simultaneously; the FIT
     handles statistical dependence (that is its job) — the one thing that
     stays banned is naive multiplication of marginal likelihood ratios,
     which assumes independence that isn't there.
   - Report each signal's weight (its contribution GIVEN the others) and the
     correlation matrix, as descriptions — not as grounds for removal.
   - Regime-type concepts (ADX etc.) enter as conditioning/interaction
     terms, not as events.
   - Train 2024 → evaluate 2025 untouched.
5. **Evaluation of the posterior** (in order):
   a. **Calibration**: predicted posterior vs realized frequency, by decile
      (a Bayesian model that says 70% must be right ~70% of the time).
   b. **Tier separation**: response rate in top posterior tiers vs base +
      shuffle-label null. Signal bar: top-tier lift ≥ +10pp REAL.
   c. **Marginal-contribution ranking**: drop-one-out deltas, reported as
      "how much does the posterior lose without this signal" — a
      contribution measure, NOT a redundancy purge; low-contribution signals
      stay in the report with their individual tables intact.
   d. **ONE economics test** at the end: trade the top posterior tier only
      (agreed exits, 4t costs, day-block CI, both years). This test — not
      any AUC/calibration number — decides if anything ships.

## Estimation spec (user, 2026-07-09): quasi-binary two-part model
Rows = EVENT OCCURRENCES (not all bars). For each event row, tag:
- the response bar (when/whether the concept's pre-registered response
  happened within its horizon) → binary y
- the response MAGNITUDE where applicable (σ-normalized move), tagged at the
  resolution bar
Model = hurdle / two-part:
1. **Logistic** on y (response occurred) — per-signal tables and the joint
   combiner both live here.
2. **Magnitude regression | y=1** (σ-units) — because economics is
   EV = P(response) × E[magnitude | response] − costs; a frequent tiny
   response can lose to a rarer large one, and P alone cannot see that.
The final economics test consumes BOTH parts (EV per event, then $/day with
day-block CI).

**Magnitude is also the taxonomy tool (user, 2026-07-09).** Report each
signal's magnitude DISTRIBUTION (mode + median + tail), never just the mean,
and FLAG BIMODALITY (mixed populations). NOTE: some concepts are NATIVELY
fakeout/headfake signals per their own articles — for those, "fakeout" IS the
pre-registered response type from the start, not a reinterpretation. Later-
stage use (not this phase): read-only overlay of responses onto the labels.

## Execution requirements (user, 2026-07-09 — AG runs AUTONOMOUS)
1. **Source re-read first**: before implementing each concept, go back to the
   ORIGINAL article(s) in `research/nt8_catalog/raw_articles/` and re-read —
   the 4-pillar synthesis may have dropped nuance. Each per-signal report
   cites its source article file(s) and states the response definition AS THE
   ARTICLE CLAIMS IT (entry, fakeout-avoidance, regime, exhaustion, ...).
2. **One signal = one standalone execution** (own script, own run, own
   report). No batching that obscures the per-signal deep dive.
3. **Per-signal report** (`DOC_<ID>.md`): article citation + registered
   response; event definition (causal math); P(the article's claim happened)
   both years; magnitude distribution WITH FIGURES (mode marked); bimodality
   flag. **Stage sequencing (user, 2026-07-09): NO nulls and NO verdicts at
   this stage** — these tables are descriptive likelihoods ("claim happens
   X%, bulk magnitude Y"), nothing more. Discrimination (matched/phantom
   nulls, shuffle tests) and verdicts happen at the JOINT-model stage.
   Exception kept because it is free: where barriers are symmetric ±k·σ,
   print the 50% random-walk reference as context (arithmetic, not a null
   run).
4. **Master summary table** (`AG_cat_00_INDEX.md`), one row per signal:
   | signal | what it measures | registered response type | N events |
   P(resp) vs null | MODE of magnitude (the bulk) | tail | bimodal? |
   verdict |
5. **Document everything as you go** — the runs are unattended; a report that
   can't be reconstructed from disk didn't happen.
6. **Folder discipline — EVERYTHING lives in `research/nt8_catalog/`,
   one SELF-CONTAINED test folder per concept (VP-01 PoC = the template):**
   - `research/nt8_catalog/tests/<ID>_<name>/` containing: the deep-dive
     script, an OQ trace script (manual verification of computed
     levels/events on named days), the report `DOC_<ID>.md`, and its
     figures (relative image paths, not file:/// absolutes).
   - master table → `research/nt8_catalog/reports/AG_cat_00_INDEX.md`
   Do NOT write into `research/leg_clock/` or the top-level `reports/`.
   Update `README_synthesis_index.md` as artifacts land.

## Unchanged hard rules
Sigma-relative everything; causal/trailing extraction only (FPS discipline);
label-free; matched + phantom nulls at the per-signal layer; day-block CIs;
both-years; NT8 sealed; delta-from-OHLCV must be disclosed as a proxy.

## Why this supersedes Phase 3-as-proposed
Standalone verdicts measure each whisper alone — we already know most
whispers are <5pp alone. The user's thesis (and his manual method) is that
the EDGE lives in the joint state. The joint model measures exactly that,
while the per-signal tables + drop-one-out ranking still tell us which
concepts earn their place.
