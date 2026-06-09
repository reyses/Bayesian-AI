---
name: reference-fista-gpu-cv-step-bug
description: elasticnet_fista_cv in core_v2/math had a CV step-size bug (fixed 2026-06-08); parity test is the guard
metadata:
  type: reference
---

`core_v2/math/fista_gpu.py::elasticnet_fista_cv` had a latent bug fixed 2026-06-08.
The CV used a scalar FISTA step `1/L` that ignored the L2 penalty's Lipschitz
contribution. The smooth objective is `(1/2N)||Xz-y||² + (lam_l2/2)||z||²` with
gradient Lipschitz `L + lam_l2` (lam_l2 enters via the `z*lam_l2` gradient term,
NOT the Gram diagonal — which is why the non-CV `elasticnet_fista`, which adds
lam_l2 to the diagonal before computing L, was never affected). High-alpha columns
(large lam_l2) overshot stability → diverged to NaN → `torch.argmin` returned the
NaN column (= alpha_max) → CV always picked max regularization → all-zero fit
(alpha came out ~1000× = 1/eps too large).

DATA-DEPENDENT: only triggers when `alpha_max·(1-l1_ratio)` is large vs the Gram
norm `L`, i.e. strong-trend segments — exactly the PRISTINE class that
`research/Regression segments/stage1_speed_pass.py` (the only importer) hunts.

Fix: per-column step `1/(L + lam_l2 + 1e-6)` + `nan_to_num(mean_mse, nan=+inf)`
before argmin. Verified: synthetic ElasticNet parity vs sklearn 0.000 → 0.981
Jaccard, FISTA alpha now matches sklearn to 5 dp.

Regression guard: `research/Regression segments/test_fista_parity.py` (synthetic
ElasticNet + Group-Lasso Jaccard, plus a real-segment-data tier-flip test). Report:
`reports/findings/<date>_fista_solver_parity.md`. Stage1 artifacts produced before
the fix are stale on strong-trend days. See [[project_rl_pivot_2026_05_28]].
