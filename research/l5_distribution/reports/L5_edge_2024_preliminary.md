# PRELIMINARY L5 edge first-look (2024)

- Days with tail data: **259**  |  total tail entries: **324,479**
- Baseline snap-back return at |z|>1.8481 (fwd 5min), per-day mean: **+0.074 pts**  95% day-block CI [-0.124, +0.277]
  (context: is the raw fade entry even +EV on 2024? — NOT the L5 question)

## Per-feature separation: Spearman(L5 feature, snap-back return) at the |z| tail
| L5_1m feature | mean daily rho | 95% day-block CI | separates? |
|---|---|---|---|
| n | +0.0131 | [-0.0008, +0.0270] | no |
| min | -0.0067 | [-0.0216, +0.0080] | no |
| q1 | -0.0053 | [-0.0200, +0.0092] | no |
| median | -0.0040 | [-0.0180, +0.0104] | no |
| q3 | -0.0029 | [-0.0176, +0.0117] | no |
| max | -0.0011 | [-0.0152, +0.0135] | no |
| mean | -0.0039 | [-0.0184, +0.0101] | no |
| std | +0.0089 | [-0.0052, +0.0223] | no |
| skew | +0.0111 | [-0.0016, +0.0234] | no |
| kurtosis | -0.0031 | [-0.0161, +0.0099] | no |
| outlier_pct | -0.0031 | [-0.0160, +0.0097] | no |
| level | -0.0077 | [-0.0221, +0.0064] | no |

## Verdict (PRELIMINARY)
- **No L5_1m feature separates the forward snap-back at the |z| tail** (all CIs include 0). On this 2024 forward-return proxy, L5 adds no entry-filter edge.

### Caveats
- Forward-return proxy, NOT NMP-trade PnL (R-trigger exit). Necessary-not-sufficient.
- 1m TF only; other TFs untested here. Validate-before-bake: do NOT add to FEATURE_NAMES on this alone.
- Horizon K=60 (=5min); Z_ENTRY=1.8481; day-block bootstrap 4000 resamples.