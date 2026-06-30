# Cusp/launch detector — honest day-disjoint evaluation
1580 candidate cubic turns, 85 human-accepted (5.4%), **4 labeled days**: ['2024-01-01', '2025-01-06', '2025-06-06', '2025-09-08']

## Random 80/20 split (the shipped method — LEAKY): AUC 0.630 (±0.056)  <- optimistic, same-day leakage

## Leave-one-DAY-out (honest, day-disjoint)
  held-out day | n / pos | OOS AUC
--------------------------------------------
    2024-01-01 |  226/44  | 0.466
    2025-01-06 |  466/27  | 0.490
    2025-06-06 |  248/4   | 0.737
    2025-09-08 |  640/10  | 0.693
--------------------------------------------
    POOLED OOS |         | **0.453**
  per-day mean |         | 0.597  (spread 0.466-0.737)

## Verdict: pooled OOS gap +-0.047 -> nominal **NOISE** by the signal bar

## BUT — the honest caveat that dominates everything
- Only **4 labeled days**. The unit of independence is the day (one regime/day), so
  day-disjoint validation has ~4 effective data points. The per-day AUC spread
  (0.47-0.74) is the real uncertainty — a pooled point estimate hides it.
- The random-split number is NOT trustworthy (same-day leakage); the day-disjoint number is
  honest but UNDERPOWERED. We cannot conclude the detector has durable edge from 4 days.
- Also: target='matches a human pick', NOT 'the launch paid' (MFE). A profit-targeted eval needs
  the MFE/MAE labels. ACTION: label more days (>=15-20 disjoint) before trusting any AUC here.
