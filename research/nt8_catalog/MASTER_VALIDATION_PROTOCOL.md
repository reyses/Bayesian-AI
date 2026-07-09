# Master Validation Protocol (Med Device Rigor)

To meet rigorous validation standards, we cannot treat backtest scripts as black boxes. Every protocol must undergo a strict IQ/OQ/PQ lifecycle to prove traceability, operational correctness, and statistical robustness. 

This Master Protocol applies to all future raw strategy documents processed by the Bayesian-AI pipeline.

## 1. Traceability & Requirements
Every protocol must map the qualitative claims from the raw article directly to quantitative, causal mathematical definitions. No assumptions are allowed without explicit documentation.

## 2. Document Control & GDP (Good Documentation Practices)
To ensure every test is auditable and self-contained, every generated report MUST adhere to strict GDP standards:
- **Unique Identifier:** Each report must carry a formal ID (e.g., `DOC-VP-01`).
- **Dependency Tracking:** The report must explicitly reference the exact scripts executed and the exact dataset scope used.
- **Self-Contained Audit Trail:** The OQ trace logs and the PQ statistical outputs/plots must be embedded natively inside the same artifact. A reviewer should not need to run the code to verify the test vectors.
- **Strict Architecture (Test Dossiers):** All materials for a specific test must be perfectly isolated in their own dedicated test folder. Do NOT dump scripts or reports into flat, shared directories.
    - Example: `research/nt8_catalog/tests/VP-01_Volume_Profile/`
    - Inside this folder, you must place the execution script, OQ traces, the final GDP report (`report.md`), and any graphical assets. Every test is a self-contained binder.

## 3. IQ (Installation Qualification)
- Verify the dataset bounds (e.g., 2024 ATLAS dataset, 5-second resolution).
- Verify the dependent features (e.g., ensuring OHLCV columns exist and are uncorrupted).

## 4. OQ (Operational Qualification)
Before running the full sweep, the protocol must pass a unit-level inspection.
- **Trace Output:** Isolate 1-3 specific days in the dataset and output a detailed chronological trace.
- **Verification:** Manually verify that the calculated metric matches the raw price data, and that the trigger fired at the exact correct bar.
- *Acceptance Criteria:* The script's localized output exactly matches the manual visual/logical inspection of the data.

## 5. PQ (Performance Qualification) & Robust Statistics
The full statistical sweep over the dataset.
- **Pure Empirical Counting:** Null controls and base rates are discarded. Evaluate the exact events strictly against their mechanical outcomes.
- **Robust EV Math:** Compute the Empirical Win Rate. Crucially, compute the **Median Magnitude** of winners and losers to represent what happens the "bulk of the time." Do not use Arithmetic Mean, as it is vulnerable to fat-tailed black swan outliers. 
- **Statistical Significance:** Output the raw Expectation ($E[x] = P(W) \times Mag(W) - P(L) \times Mag(L)$) utilizing the Median magnitudes. Include a 95% Confidence Interval (via bootstrapping) to prove stability.
- **Graphical Descriptive Statistics:** Every report must include a plotted histogram/KDE of the Winner and Loser distributions (Magnitude in $\sigma$) to visually expose fat tails and skews.

## 6. Conclusion & The Structural Alpha Lens
*(Amended per FABLE-5 review, 2026-07-09 — approved by Moises.)*
Do not dismiss setups purely because the raw EV is negative or zero — but PQ
outputs FLAGS, never recommendations or approvals. Verdicts belong to the
discrimination (joint-model) stage.
- **INVERSION-CANDIDATE flag:** a deeply negative EV or strong opposite skew
  registers an inversion HYPOTHESIS for the discrimination stage. It is NOT a
  conclusion at PQ: across ~14 concepts x multiple setups, sign-flip picking
  on the same data manufactures inversions by chance, and an EV whose 95% CI
  spans zero (e.g., all three VP-01 setups) cannot ground one either way.
- **CLIP-CANDIDATE flag:** a clustered positive core dragged down by a fat
  left tail registers a tail-clipping HYPOTHESIS. WARNING (measured, this
  repo): every cut-the-loser overlay tested here has LOST (fixed stops
  -$31/day; session stops significant losses; 76% of clipped legs recovered).
  The flag earns its own test at the discrimination stage; it is never an
  approval basis at PQ.

## 7. Cross-dossier consistency requirements (FABLE-5, 2026-07-09)
- **Both years:** run 2024 AND 2025; report per-year tables side by side. A
  claim stable across both is a different object than a one-year artifact.
- **Event definitions from the ARTICLE's session context:** e.g., "the open"
  = the day-session open (8:30 CT), not the data file's first bar (17:00 CT
  prior-evening Globex) — the VP-01 open bug.
- **Magnitude window ends at the RESOLUTION bar:** no post-resolution
  max/min in the magnitude (the VP-01 horizon-MFE bug) — otherwise EV is
  unrealizable by any exit.
- **Sigma standard:** trailing 1m regression residual sigma (see
  research/level_hold/tools/level_hold_study.py::rolling_ols_bands), so
  magnitudes are comparable across all dossiers.
- **Free reference kept:** where barriers are symmetric +-k*sigma, print the
  50% random-walk reference (arithmetic, not a null run).

---
## User Review Required
> [!IMPORTANT]
> Does this finalized Master Validation Protocol correctly capture your methodology for evaluating the remaining catalog?
