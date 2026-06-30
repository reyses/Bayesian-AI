# Cusp/launch detector — honest day-disjoint evaluation (scale-ready)
1610 candidate cubic turns, 4 labeled days: ['2024-01-01', '2025-01-06', '2025-06-06', '2025-09-08']

**Leaky random-split (shipped) AUC 0.630** — same-day leakage, NOT trustworthy.

## Honest day-disjoint
### target 'target'  (n=1580, pos=85 = 5.4%, 4 days)
- **pooled OOS AUC 0.453**  day-block 95% CI [0.419, 0.697]  -> **NOISE**
- per-day OOS AUC spread: 0.466 – 0.737 (mean 0.597, k=4)
- CI includes 0.5 -> NOT significant

### target 'paid'  (n=379, pos=323 = 85.2%, 3 days)
- **pooled OOS AUC 0.654**  day-block 95% CI [0.571, 0.740]  -> **REAL**
- per-day OOS AUC spread: 0.571 – 0.740 (mean 0.648, k=3)
- CI excludes 0.5 -> signal

## Read
Day-disjoint + day-block CI is the verdict; the random number is leakage. With 4 days
this is underpowered (wide CI) — the harness is built so that when the labeler grows the day
count, the SAME run yields a conclusive CI. 'paid' (objective MFE) is the target that matters;
'target' (human-match) is a proxy. A tight CI above 0.5 on 'paid' across many days = real edge.
