# p-stream control chart — sizing the control plane
Limits (proposal v0): ambiguous band [0.2,0.8], jitter |dp|>0.3, plus 0.5-crossing flips. Escalation-eligible = any of the three.

| source | eps | frames | median dp | p90 dp | ambig% | jitter% | flip% | ESCALATION% | exits | exits intercepted |
|---|---|---|---|---|---|---|---|---|---|---|
| gen0_tiered | 156 | 4956 | 0.066 | 0.580 | 22% | 22% | 18% | **34%** | 734 | 85% |
| gen1_oneshot_partial | 19 | 706 | 0.000 | 0.017 | 2% | 2% | 1% | **3%** | 6 | 100% |
| gen1_anchor2p | 8 | 264 | 0.000 | 0.026 | 1% | 3% | 1% | **3%** | 2 | 100% |

Reading: ESCALATION% = cost (fraction of frames the reasoning layer would be consulted). "Exits intercepted" = benefit ceiling (fraction of trigger-pulls that would have passed through the control plane first). A hybrid is attractive when intercepted% is high and escalation% is low.
