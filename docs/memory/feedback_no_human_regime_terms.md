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
| oracle                   | post-hoc retrospective analysis with full event resolution known (max_z, final duration, MFE/MAE, PnL of any tier evaluated against the event); used to BUILD the Bayesian table; not available in live |
| Bayesian table           | lookup substrate keyed by primitive-bucket vector, returns conditional outcome statistics; used at-bar in live as the substitute for oracle knowledge |
| primitive chord          | the bucket-vector x(t) at a single bar (e.g. (slope_q, curv_q, z_close_q, sigma_rank_q, r2adj_q)) — the joint key into the Bayesian table             |
| failure mode             | a (bucket, tier) cell where the tier's oracle $/trade is materially negative — a candidate for adding a context filter that gates the tier off in that bucket |

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
