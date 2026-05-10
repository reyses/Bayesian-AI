---
name: Oracle labels CAN use lookahead — chord/at-bar primitives CANNOT
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
though labels were assigned with full hindsight — the trained model
runs forward on unlabeled inputs.

**Oracle label categories (lookahead-OK)**:

- max_z_after_event_t       peak |z| in the future window
- max_mfe_per_tier         best price reached if a tier had been firing
- duration_until_revert     bars between event start and revert-to-mean
- did_extend_60m            binary: did the run continue past 60min
- did_resolve_as_cascade    binary: duration ≥ X OR max_z ≥ k
- ride_pnl_pts              outcome of an idealized ride trade
- fade_pnl_pts              outcome of an idealized fade trade
- bars_to_max_z             time-to-peak

**At-bar primitive categories (lookahead-FORBIDDEN)**:

- slope_TF                  rolling-window slope using only data ≤ t
- z_close_TF                requires only TF mean/sigma at-bar
- sigma_rank_TF             rolling-percentile, only past
- r2adj_W                   linear-fit on past W bars only
- bars_since_pivot          past-only sequence position
- tf_alignment              past-only sign-agreement count

**The test**: at any inference moment, could the value be computed live?
- If yes → at-bar primitive (no-lookahead rule applies)
- If no → oracle label (lookahead is fine and required)

If a primitive is ambiguous (e.g., 'price_velocity_at_event_start'),
ask: does this run live? If yes, no-lookahead. If it's only computed
once per resolved event for fitting, it's an oracle label.
