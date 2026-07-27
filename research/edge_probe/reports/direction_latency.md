# Direction-latency curve — how early is leg direction detectable?
156 episodes, 25 days. OOS direction AUC (predict wrong vs right) using F-space at bar N-since-entry.

| bars since pivot (N) | OOS AUC | 95% CI | test days |
|---|---|---|---|
| 0 | 0.447 | [0.280, 0.599] | 9 |
| 1 | 0.370 | [0.170, 0.585] | 9 |
| 2 | 0.443 | [0.254, 0.619] | 9 |
| 3 | 0.399 | [0.230, 0.553] | 9 |
| 5 | 0.666 | [0.469, 0.826] | 9 |
| 8 | 0.508 | [0.312, 0.709] | 9 |
| 10 | 0.466 | [0.333, 0.626] | 9 |
| 12 | 0.516 | [0.343, 0.716] | 9 |

## HONEST VERDICT — INCONCLUSIVE (wrong instrument, not "direction is late")
Every N sits at ~0.44-0.52 with CIs spanning 0.5. That CONTRADICTS the known
direction classifier (AUC 0.864) — which means this packet probe is NOT
measuring the same thing, and cannot answer the question:
1. WRONG LABEL: it predicts trade-OUTCOME (did an ALREADY top-decile-filtered
   entry end favorable) — a hard, near-balanced target — NOT raw up/down
   direction. The 0.864 classifier predicts the latter on unfiltered bars.
2. UNDERPOWERED: 156 episodes, 9 expanding-window test days; CIs are huge.
3. CONDITIONED: packets start AT an entry the combiner already chose, so
   "direction at the pivot" is entangled with the entry filter.

DO NOT conclude "direction is late." The honest answer: this data can't tell.
The RIGHT measurement = run the EXISTING V2 direction classifier (the 0.864
one) at bars-since-pivot = 0,1,2,3,5 on the full V2 dataset + zigzag pivots.
That needs the V2 pipeline (build materialized), not the 150 packets. Harness
here is reusable once pointed at the right data+label.
