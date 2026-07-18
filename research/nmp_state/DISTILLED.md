---
name: distilled-nmp_state
description: Derives lambda_hat and vr from V2 185D schema for the NMP entry criteria; parity checks pass, but the vr_proxy cross-TF shortcut fails all 5 TFs, and lambda gating was never run end-to-end.
metadata: {type: distilled, topic: nmp_state, status: live}
---
## Verdict
V1's NMP algorithm used variance ratio (vr) and stability exponent (lambda),
both dropped/hardcoded-to-0 in V2. This project (DMAIC) reconstructed
lambda_hat and vr from the V2 185D feature set + raw closes, recalibrated the
V1 z_21 thresholds onto V2's z_15, and tested whether a cheap cross-TF sigma
ratio could stand in for exact vr. Parity between the vectorized derivation
and naive/reference math passed cleanly. The vr_proxy shortcut FAILED at
every timeframe (Spearman well below the 0.8 bar) — exact vr from raw closes
is required, not the proxy. Report covers only 5 sampled 2025 days.

## Key numbers (with CIs where they exist)
- Parity: lambda_hat slope vs np.polyfit (200 random windows) max |err| = 3.89e-16 — PASS
- Parity: lambda_se vs np.polyfit cov max |err| = 1.80e-16 — PASS
- Parity: vr rolling vs brute-force (500-bar synthetic) max |err| = 1.53e-12 — PASS
- Threshold recalibration: Z* (entry) on `|z_15|` = 1.8481, matching V1's `P(|z_21|>2.0)` = 7.9972%
- Threshold recalibration: Z* (exit) on `|z_15|` = 0.4752, matching V1's `P(|z_21|<0.5)` = 27.8337%
- lambda_hat t-stat abstain band proposed [-2.0, 2.0] across k=12/21/30 on 1m and 5m (means ~0, std 0.99-1.43)
- vr_proxy vs vr_exact Spearman correlation, all FAIL (<0.8 bar): 5s=0.636, 15s=0.572, 1m=0.419, 5m=0.730, 15m=0.384
- Trigger-rate parity: V1 exact (`|z_21|>2.0 AND vr_exact<1.0`) = 7.1133%; V2 exact-vr scaled = 7.5146%; V2 proxy-vr scaled = 8.0544%

## Graveyard / never-retry (if any)
- vr_proxy (cross-TF sigma ratio, e.g. sigma_5s/sigma_slow) as a stand-in for exact vr — killed by Spearman <0.8 at all 5 tested TFs (`reports/2026-06-11_nmp_state_derivation.md` §4).

## Reusable assets
- `research/nmp_state/derive.py` — `derive_day()`: derives lambda_hat/vr_exact/vr_proxy/z_21 per day, aligned via `_last_closed_idx` (causal); constants EPS=0.1, K_SWEEP=(12,21,30), VR_WINDOWS=(10,60), PROXY_MAP (fast TF -> slow TF).
- `research/nmp_state/validate.py` — `run_validation()`: parity checks + threshold recalibration + Spearman proxy test + trigger-rate parity, writes report to `reports/`.

## Data locations
- `DATA/ATLAS/<tf>/<day>.parquet` — raw OHLCV closes, read directly by `derive_day`.
- `DATA/ATLAS/FEATURES_5s_v2` — V2 185D feature store (layers L0/L2/L3 used).

## Open threads
- lambda_hat abstain band [-2.0, 2.0] is proposed, not yet gated into a live decision path (per MEMORY.md this feeds the broader lambda-completion roadmap, but no gating run lives in this folder's reports).
- Only 5 of 2025's trading days were sampled; no OOS/full-year validation in this folder.
- Since vr_proxy failed, whether vr_exact (from raw closes) is cheap enough to compute live is not addressed here.

## Sources
- research/nmp_state/README.md
- research/nmp_state/project.md
- research/nmp_state/reports/2026-06-11_nmp_state_derivation.md
- research/nmp_state/derive.py
- research/nmp_state/validate.py

## Archive recommendation
KEEP-LIVE (reason: MEMORY.md §5 marks this as "verified" derivation layer feeding the ACTIVE lambda-completion roadmap `ROADMAP_LAMBDA_COMPLETION.md`; the vr_proxy dead-end is settled but the lambda gating work is not concluded, and downstream consumers likely still reference `derive.py`).
