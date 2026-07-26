# Gauge-gated exit — counterfactual on gen-0 census
148 ride episodes, 22 days. Exit rate: raw 84% -> gauge-gated 28%.

| contrast | mean capture Δ | 95% CI |
|---|---|---|
| raw teacher − never-bail | -0.017 | [-0.092, +0.063] |
| **gauge-gated − never-bail** | **-0.029** | [-0.074, +0.021] |
| gauge-gated − 5m-hold | -0.162 | [-0.287, -0.026] |

VERDICT: gauge-gating removes the value destruction but does NOT beat never-bail — the teacher has no positive exit edge even when well-timed on this curriculum. The ride edge is "ride, do not exit"; the Mamba exit-head is near-trivial. Re-scope before spend.
