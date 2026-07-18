---
name: distilled-leg_clock
description: Legs are clocked/momentum (MRL rises, transfers OOS) but untradeable via confirm-then-ride; entry-filter hunt found 10/11 dead concepts + 1 real (Footprint Imbalance).
metadata: {type: distilled, topic: leg_clock, status: concluded}
---
## Verdict
Asked: is a directional leg's remaining length memoryless or clocked, and can
elapsed-time/velocity/volume predict enough to trade it? Found: legs ARE
clocked (mean-residual-life rises with elapsed, OOS-stable 2024->2025) but the
obvious confirm-then-ride backtest LOSES money both years. Follow-on entry-
filter hunt (label signature, VP/POC, pretrend footprints, AG concept sweep)
killed almost every candidate filter; one concept (Footprint Imbalance)
cleared costs with a positive CI. A separate AG classifier's 0.87 AUC finding
was reviewed and shown circular (rediscovered the hindsight labeler's own
selection rule, not a live edge).

## Key numbers (with CIs where they exist)
- MRL (thr20 zigzag, 2024 legs 52,706/259d, 2025 legs 61,277/277d): 2024 slope 0-12min +0.38 min/min elapsed -> momentum (rising, not flat/memoryless); 2024 vs 2025 MRL mean abs diff 1.06 min (transfers OOS); duration mode~2.5/1.5min, extent median 50/57t. `reports/leg_clock_thr20.txt`
- Leg termination near a prior target: real 0.435 vs phantom-null 0.434, lift +0.001 (dead). `reports/leg_target_thr20.txt`
- Confirm-then-ride 2025 (best config C150/R80): $/day -60.0, PF 0.91; 2024 same config -18.4, PF 0.96 — LOSES both years. `reports/confirm_ride_2025.txt`, `reports/confirm_ride_2024.txt`
- Labels: 25,680 trades/576 days (44.6/day), 50/50 LONG/SHORT, duration mode~8.4/median 21.0min, extent mode~91.2/median 149.5t. `reports/label_signature.txt`
- q0.995 score tier recall of 9-13CT labels: 2.3% (46/2031); recall at q0.99 3.9%, q0.98 7.1%, q0.95 12.7%. `reports/label_recall.txt`
- NT8 deploy test (causal, real scores): tier_99.5 $/day -0.9 [CI -6.5,+5.2] NOT sig; tier_98 -4.2 [-15.2,+6.7] NOT sig; shuffled-null tiers SIGNIFICANTLY negative (sanity pass). `reports/nt8_deploy_test.txt`
- Best dev_loop variant (fade, H9-13, Z3.0, q0.995): 2025 $/day +9.7 [-3.4,+27.5] NOT sig, PF 1.45, N=249tr; 2024 +1.8 [-2.6,+6.3] NOT sig, PF 1.17, N=194tr. `reports/dev_loop_2025.txt`
- Failure autopsy (q0.995/trail20t, 2240 trades): hold<2m = -$41.62/tr, 0% win; hold 5-15m = +$53.88/tr, 80.1% win. `reports/failure_autopsy.txt`
- AG concept sweep (11 concepts, causal FPS backtest 2024-2025): 10 NOISE (e.g. VolProfile POC -$323.27/day, Slope Persistence -$308.13/day); 1 REAL — Footprint Imbalance $17.23/day net, 95% CI [$12.27, $22.99], 0.65 tr/day. `reports/AG_cat_00_INDEX.md`, `reports/AG_cat_10_Footprint_Imbalance.md`

## Graveyard / never-retry
- Confirm-then-ride at leg scale: loses -$60 to -$200/day OOS across all tested configs.
- Leg-termination-near-prior-target: lift +0.001, dead.
- 10/11 AG concept sweep: candle shapes, slope persistence, band bounce, APZ re-entry, pivot points, S/R breaks, MA crossover, VWAP pullbacks, cum-delta divergence, VolProfile POC — all NOISE.
- AG "0.87 AUC entry classifier" — FABLE-5 review found it circular: rediscovered the hindsight auto-labeler's own snap-to-extreme selection rule, not a market edge (`reports/FABLE5_REVIEW_of_AG_findings.md`).
- Pretrend footprint microstructure (body/wick, absorption, rejection): null both years (`reports/pretrend_footprint.txt`).

## Reusable assets
- `tools/leg_length_clock.py` — MRL/duration-extent distribution tool, train/test OOS split.
- `tools/label_signature.py` — label distribution characterizer (mode/median/tail).
- `tools/ag_cat_harness.py` + `tools/ag_cat_*.py` (11 concepts) — causal FPS concept-sweep harness, reusable for future dead-list checks.
- `tools/nt8_deploy_test.py`, `tools/dev_loop_2025.py` — causal deploy-test scaffolding with real-vs-shuffled-null and day-block bootstrap CI.

## Data locations
- `DATA/ATLAS/{5s,1m,5m,15m,1h}/YYYY_MM_DD.parquet` (2024=259d, 2025=277d).
- `DATA/ai_cusp_picks/ai_picks_YYYY-MM-DD_multi.json` — hindsight label source (576 days, ~37 trades/day).
- `reports/mamba_training_dumps/` — large checkpoint/CSV dump; appears to be an unrelated Mamba RL side-run parked here, not covered by this topic's own docs.

## Open threads
Footprint Imbalance (AG_cat_10) is the one cost-cleared filter found here — not yet sized or live-deploy tested.

## Sources
README.md, SPEC_leg_persistence.md, AG_TASK_label_feature_discovery.md,
reports/leg_clock_thr20.txt, reports/confirm_ride_{2024,2025}.txt,
reports/label_signature.txt, reports/label_recall.txt, reports/nt8_deploy_test.txt,
reports/dev_loop_2025.txt, reports/AG_cat_00_INDEX.md,
reports/AG_cat_10_Footprint_Imbalance.md, reports/FABLE5_REVIEW_of_AG_findings.md

## Archive recommendation
ARCHIVE (thesis answered: legs clocked but untradeable; entry-filter hunt
mostly killed, one real filter found). Flag `AG_cat_10_Footprint_Imbalance.md`
for a possible spin-out mini-project before archiving — it is the only surviving
positive-CI result in the whole topic.
