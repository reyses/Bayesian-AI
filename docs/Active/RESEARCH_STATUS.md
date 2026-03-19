# Research Lines -- Status Summary
> Updated: 2026-03-18

## Legend
- **DONE** = completed, findings applied or archived
- **SUPERSEDED** = replaced by newer approach
- **ACTIVE** = still relevant, work ongoing
- **BLOCKED** = waiting on dependency
- **BACKLOG** = valid idea, not started

---

## Research Spec V-FF (from RESEARCH_SPEC_V_TO_FF.md)

| ID | Title | Status | Notes |
|----|-------|--------|-------|
| V | Trajectory k-NN Extrapolation | **SUPERSEDED** | Peak detection + CNN approach replaces k-NN trajectory prediction. Peak template research (174K peaks, 54D features) is the new trajectory analysis. |
| W | Partial Bar Robustness | **BACKLOG** | Still valid. Workers only update on bar close. Fix: blend completed + forming bar. In roadmap as "Partial Bar Aggregation." |
| X | Counter-Trend Scale Analysis | **SUPERSEDED** | Feb 9 deep dive + direction bias analysis covered this. Counter-trend LONGs are the problem -- entry sensor gate fixes the worst cases. |
| Y | Regime-Conditional k-NN | **SUPERSEDED** | Peak template classifier does regime-conditional entry. 3-outcome classification (reversal/plateau/continuation) replaces k-NN regime detection. |
| Z | Fractal Dimension Features (Shi 2018) | **BACKLOG** | Not started. Hurst exponent already in 16D features. Fractal dims could add discriminative power but lower priority than CNN. |
| AA | Seed Shape Direction Signal | **DONE** | Human seed analysis completed (2026-03-18). 6D geometry + direction from seeds analyzed. Findings in fake_vs_real_peaks research. Seeds validate peak detection approach. |
| BB | Stacked Multi-TF Direction Model | **SUPERSEDED** | The 192D (12 TF x 16D) context is the multi-TF stack. Peak templates use delta + slope of this vector. CNN spec replaces stacked GBM with temporal conv on raw states. |
| CC | E[PnL] Tick Prediction | **ACTIVE** | Brain has `get_expected_pnl()`. Survival stop uses it. But calibration is weak (cold start, direction-specific). Worth revisiting after CNN baseline. |
| DD | Auto-Level Detection | **BACKLOG** | Level proximity features not built. Spec exists at `docs/specs/LEVEL_DETECTOR_SPEC.md`. Could enhance peak context (is the peak at a known level?). |
| EE | Stop Loss Optimization | **ACTIVE** | SL is PF 0.86 (757 trades, net loss). Current SL is template-derived. Research needed: dynamic SL based on peak context (wider SL when approach range is large, tighter when small). |
| FF | Conviction Calibration | **SUPERSEDED** | Conviction threshold (0.48) is a rubber stamp (PFMEA RPN 160). But with peak templates + sensor gates, conviction becomes less important. Direction comes from the peak, not from voter conviction. |

## Active Specs (docs/Active/)

| Spec | Status | Notes |
|------|--------|-------|
| RESEARCH_SPEC_V_TO_FF.md | **PARTIALLY SUPERSEDED** | V, X, Y, BB, FF superseded by peak approach. W, Z, DD, EE still valid. |
| SPEC_DECISION_FUNNEL.md | **ACTIVE** | Temporal candidate narrowing (survival over bars). Still valid -- peak detection is instantaneous, funnel adds temporal confirmation. Addresses RPN 648 (score competition). |
| PATTERN_SCALE_MISMATCH_REPORT.md | **DONE** | Root cause identified (MFE scale 30x mismatch). Fixed by TF-bucketed clustering (V7.0.0) and peak detection (direction + expected MFE from peak context). |
| SPEC_PATTERN_ANCHORING.md | **SUPERSEDED** | 1m anchor refactor was Option B. Peak templates are the proper fix -- discovery anchored to peaks, not z-score crossings. |

## Research Tools Built (2026-03-18)

| Tool | Purpose | Output |
|------|---------|--------|
| `tools/peak_template_research.py` | 174K peaks, 54D features, UMAP+HDBSCAN, classifier | `reports/findings/peak_template_*` |
| `tools/peak_exit_research.py` | Peak trade trajectory analysis during trades | `reports/findings/peak_exit_*` |
| `tools/peak_exit_sensor_fusion.py` | 1s velocity + 1m volume sensor fusion | `reports/findings/sensor_fusion_*` |
| `tools/inverted_entry_exit.py` | Inverted entry signal as exit trigger | `reports/findings/inverted_entry_exit*` |
| `tools/fake_peak_inside_seeds.py` | Fake peaks within human seed trades | `reports/findings/fake_vs_real_peaks*` |
| `tools/dmi_peak_vol_drop_sequence.py` | DMI peak -> vol drop -> DMI cross sequence | `reports/findings/dmi_vol_sequence*` |
| `tools/peak_vs_seeds_dmi_volume.py` | Peak trades vs human seeds comparison | `reports/findings/peak_vs_seeds*` |
| `tools/seed_imr_dmi_volume.py` | Seed IMR chart with DMI overlay | `reports/findings/seed_imr_dmi_vol*` |
| `tools/peak_exit_imr.py` | I-MR charts for peak exit trajectories | `reports/findings/peak_exit_imr*` |
| `tools/seed_daily_chart.py` | Daily seed visualization | `reports/findings/seed_daily_chart*` |

## Missing Research (gaps identified)

| Gap | Why it matters | Priority |
|-----|---------------|----------|
| **CNN peak classifier** | Hand-crafted thresholds are brittle. CNN learns temporal patterns from raw 10-bar series. Trains on profit, not labels. | HIGH -- spec written, needs implementation |
| **Dynamic SL from peak context** | SL is static (template-derived). Peak approach range predicts volatility -- wider range = wider SL needed. | MEDIUM |
| **Level proximity at peaks** | Is the peak happening at a known support/resistance? Level-anchored peaks may have different outcome distributions. | MEDIUM |
| **Partial bar aggregation** | Slow TF workers (4h, 1h) are stale during bar formation. Blending partial bar improves exit signal freshness. | MEDIUM |
| **Peak template Phase 1 rebuild** | Current Phase 1 uses z-score crossings. Rebuild around peak detection + 54D context + outcome labels. | HIGH -- but needs CNN first |
| **Weekend recalibration** | Weekly retrain from live data. Brain persists but templates refresh. | LOW -- needs live trading first |
| **Feb 9 regime detection** | Can we detect "crash day" early and sit out? 1,679 tick range, 25 consecutive losses. ADX/volatility-based circuit breaker? | MEDIUM |
