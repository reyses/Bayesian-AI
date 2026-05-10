---
name: Metaphors must translate to math/statistics — borrow methods, ground definitions
description: Metaphors and trader/physics/biology terms can be used as descriptive shorthand IF they translate to an explicit mathematical or statistical definition; code labels and report titles stay statistical because they must stand alone
type: feedback
---

**Refined rule** (locked 2026-05-09 evening, refined twice):

The original rule was too strict. The actual rule:

> **Borrow methods and language from any discipline (physics, biology,
> trading, finance, signal processing). It IS useful — a good metaphor
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
| Code labels (variable names, axis bins) | NO — statistical only | labels stand alone with no surrounding context for the translation |
| Chart titles + legend entries     | NO — statistical only | chart consumers may not have the glossary |
| Report headlines                  | NO — statistical only | headlines get pasted, lose context |
| Slack-style summaries to user     | YES, with translation | quick communication, with the math noted        |

## Translation table (canonical glossary)

This list grows as terms are introduced. When using a metaphor in a doc,
either link to this table or include the translation inline.

| metaphor / borrowed term | math/stat translation                                                            |
|--------------------------|----------------------------------------------------------------------------------|
| envelope                 | M_close ± k·SE_close band region around a regression mean                        |
| 3-body system            | the three rolling-regression anchors {M_close, M_high, M_low} at a single TF      |
| anchor                   | the rolling-regression mean for a TF and column (close/high/low)                  |
| force / tension          | NOT meaningful — drop. The anchors don't apply force to price.                    |
| elastic                  | dropping. Use sigma-band proximity instead.                                       |
| chop                     | high variation = high SS_residual / SS_total of N-bar linear regression of close  |
| smooth                   | low variation = low SS_residual / SS_total = high R²_adjusted                     |
| compression              | low sigma rank = Q1-Q2 of the rolling-60min percentile of SE_close                |
| expansion                | high sigma rank = Q4-Q5 of the same                                               |
| pivot / inflection       | bar where sign(slope_t) ≠ sign(slope_{t-Δ}) and |curvature_t| in top quantile     |
| reversion                | the new directed leg starting at a pivot (NOT mean-reversion in the OU sense)     |
| trend-follow / ride      | trade in the direction of slope at the same TF                                   |
| macro event              | contiguous 5s run where (5s_close − M_anchor)/SE_anchor exceeds k = 3 at 1h TF    |
| crash / rally            | macro event with side='below' (price < M_anchor − 3σ) / 'above'                   |
| Goldilocks trigger       | k = 2.0 σ band entry — empirical sweet-spot of frequency vs information           |
| outer wall               | k = 3.0 σ band entry — rare, regime-shift suspect                                 |
| primitive / precursor    | a marked event timestamp + bar-level features captured AT that timestamp          |
| state machine            | discrete-state model: {NORMAL, DIRECTIONAL, FLATTENED, CANDIDATE, ...}            |
| filter (context)         | predicate on state: bool function of features at-bar with no lookahead            |
| tier                     | strategy that fires entries with direction; combined as filter ∧ filter ∧ entry  |
| psychohistory            | predictive statistics over aggregate event populations — empirical conditional probability tables built from large-N events (NOT individual prediction; NOT Asimov's deterministic future) |
| chord                    | a feature vector x ∈ R^N at a single bar t — the simultaneous combination of N feature values evaluated together (the joint cell in a probability table is the chord that fires at a given bar) |
| resonance                | cross-feature or cross-TF alignment where the signs/magnitudes of multiple features agree at the same bar — quantified as ∑_i sign(f_i) or as the joint-cell occupancy / lift over the product of marginals |
| dissonance               | the inverse of resonance — features disagree at the same bar; signs cancel or magnitudes anti-correlate                                              |
| harmonic                 | a feature relationship that repeats at a multiple-of-base-TF — e.g. an effect at 1h that mirrors at 5m at the same phase                              |
| theme                    | day-level aggregate: stats over the full session (range, net move, efficiency, total dwell time at extremes, dominant motif type)                    |
| phrase                   | (refined 2026-05-10) the 15m-CRM-defined macro segment — "the most stable line" the eye tracks. Typical duration 30min-3hr. The day decomposes into ~2-5 phrases. NOTE: code currently calls these "motifs" in `segment_day_motif_melody.py` — pending rename. |
| motif                    | (refined 2026-05-10) the 5m-CRM-defined micro segment NESTED inside a phrase. Typical duration 5-30min. Each phrase decomposes into 1-6 motifs. NOTE: code currently calls these "melodies" in the segmenter. |
| sub-motif                | (added 2026-05-10) the 1m-CRM-defined nano segment NESTED inside a motif. Typical duration 1-5min. Captures fast directional moves within a motif. |
| measure                  | (added 2026-05-10) the 15s-CRM-defined sub-nano segment NESTED inside a sub-motif. Typical duration 15s-2min. The "rhythm-unit" level of the music hierarchy. |
| chord                    | (refined 2026-05-10) the at-bar feature vector at a single 5s bar — the simultaneous primitives playing at that instant. Distinct from segment_chord. |
| note                     | a single primitive value at one 5s bar (one component of a chord) — slope_15m_at_bar, z_close_15m_at_bar, etc.                                       |
| segment                  | generic name for either a phrase OR a motif — used when the level isn't specified                                                                    |
| segment_chord            | the EDA aggregation of 5s chords/notes WITHIN a segment — distribution stats (mean, std, dominant value, mode) computed over all 5s bars inside the segment. The "fingerprint" of the segment in chord-space. |
| variation                | (added 2026-05-10) two segments with the SAME shape_class but DIFFERENT segment_chord fingerprints are variations of the same theme/motif. Each variation is a distinct cell in the Bayesian table even though the macro classification matches. EMPIRICAL example: LINEAR_DOWN phrases (n=312 IS) sub-conditioned by `slope_15m__std` quartile produce mean_ride_$ from $12 (Q1_steady) to $174 (Q4_volatile) — a 14× spread within one shape. The variation is the unit that distinguishes failure modes within a shape. |
| oracle                   | post-hoc retrospective analysis with full event resolution known (max_z, final duration, MFE/MAE, PnL of any tier evaluated against the event); used to BUILD the Bayesian table; not available in live |
| Bayesian table           | lookup substrate keyed by primitive-bucket vector, returns conditional outcome statistics; used at-bar in live as the substitute for oracle knowledge |
| primitive chord          | the bucket-vector x(t) at a single bar (e.g. (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q)) — the joint key into the Bayesian table             |
| failure mode             | a (bucket, tier) cell where the tier's oracle $/trade is materially negative — a candidate for adding a context filter that gates the tier off in that bucket |
| Bayesian probabilistic table | (refined 2026-05-10) a HIERARCHICAL probabilistic model — NOT a lookup table. Per-cell posteriors (Beta-binomial for win-rate, Normal-Inverse-Gamma for $/trade, etc.) with shrinkage from cell -> parent shape -> universal. Output is a posterior DISTRIBUTION, not a point estimate. Justification: uniqueness analysis 2026-05-10 showed 80-99% unique motif compositions within directional phrase shapes — lookup cells would be ~1-3 events each (too thin); hierarchical priors borrow strength from parents. Probabilistic also gives risk-management outputs (tail quantiles, credible intervals) that point-estimate lookups cannot. |
| risk-management posterior | the tail of the per-cell posterior distribution used for position-sizing / stop-placement / skip decisions. E.g., 10th-percentile $/trade -$200 calls for tighter stop than 10th-percentile +$5 even when both have mean +$30. |
| primitive (shape_class)  | (locked 2026-05-10) the STRUCTURING LABEL applied to a segment via the 20-shape SeedPrimitiveLibrary. NON-NEGOTIABLE: HDBSCAN/regression/clustering on the raw chord fingerprint WITHOUT primitive labels collapses to the most basic variance axis (direction only) and loses curve-shape distinction. Empirical demonstration 2026-05-10: global HDBSCAN on 2,091 phrases produced 2 clusters (UP, DOWN) + NOISE; per-shape HDBSCAN found meaningful within-shape variations (LINEAR_DOWN C0 n=40 ride+$237). Primitives DO the structuring work that downstream clustering cannot do alone. |
| HDBSCAN within primitive | (locked 2026-05-10) the VARIATION FINDER applied AFTER primitive labels. Each shape_class is HDBSCAN'd separately on chord fingerprint features. Some shapes (LINEAR/EXPONENTIAL with ~300+ phrases) split into 2-4 natural clusters; others (LOGARITHMIC_UP/DOWN, STEP_UP/DOWN) don't sub-cluster — that's a FINDING (the shape IS one homogeneous bucket), not a failure. |
| Bayesian table cell      | (locked 2026-05-10) keyed on (shape_class, variation_cluster). variation_cluster is the HDBSCAN cluster id within shape, or null if the shape doesn't sub-cluster. shape-only cells (STEP, LOGARITHMIC) get one big cell each. Shape-with-variations cells get N+1 cells (N clusters + NOISE bucket). |
| 2D shape (Layer 1)       | (locked 2026-05-10) the geometric-only substrate: primitive label + within-shape HDBSCAN on segment-level scalars (slope, sigma, length, peak_z, r2adj, tod). All inputs derived from the 2D price-vs-time curve. NO at-bar 5s feature signatures used. Current 31-cell V0 substrate at 15m level lives entirely at this layer. |
| chord layer (Layer 2)    | (locked 2026-05-10) the feature-signature substrate: at-bar 5s feature vectors (slope_15m, z_close_15m, sigma_rank_15m, slope_5m, z_close_5m, sigma_rank_5m, r2adj_5m) aggregated per Layer-1 cell. Tells us WHICH features co-fire inside a shape-defined segment. Layer 2 adds dimensions BEYOND 2D shape; this is where compression-before-expansion vol-velocity signature, 1m-5m divergence, etc. become first-class cell axes. NEXT analysis step (deferred until V0 Bayesian model on Layer 1 is built and OOS-validated). |

## What labels in code may use

Statistical-only — math name + magnitude + sign:

```
trend:     no_trend, negative_low_trend, negative_high_trend,
                     positive_low_trend, positive_high_trend
variation: low_variation, low_mid_variation, mid_variation,
           high_mid_variation, high_variation
sigma:     low_sigma, low_mid_sigma, mid_sigma, high_mid_sigma, high_sigma
z:         zero_z, negative_near_z, negative_far_z, positive_near_z, positive_far_z
curvature: negative_curvature, no_curvature, positive_curvature
quantiles: q1, q2, q3, q4, q5 — pure ordinal
sign axes: UP, DOWN, FLAT — sign of net move with zero-threshold
```

## How to handle existing terms in the codebase

- New code: statistical labels only.
- Old code: leave as-is until next-touched, then convert.
- Journals / memory: free to use metaphors; provide translation on
  first use of each new term in that document.
- This memory file IS the canonical translation table — extend the
  table when new metaphors are introduced.
