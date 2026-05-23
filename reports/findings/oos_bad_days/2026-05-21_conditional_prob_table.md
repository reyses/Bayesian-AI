# Conditional Probability Table — chop resolution (v1)

Events = zigzag legs at **ATR x4**. Diagnostic only — no trading action, so the give-up / whipsaw cost is irrelevant here. Range terciles (IS-derived): low < 28.0pt | mid | high >= 50.7pt. Every cell: n + 95% bootstrap CI, IS and OOS separate.

Events: IS 18,242 legs over 275 days; OOS 2,995 legs over 51 days.

## Base rate — unconditional P(a leg is low / mid / high)

| set | P(low) | P(mid) | P(high) |
|---|--:|--:|--:|
| IS |   33% |   33% |   33% |
| OOS |   15% |   35% |   50% |

## Q: given K consecutive LOW-range legs, what is the NEXT leg?

`P(high)` = next leg breaks out into the top tercile. `P(low)` = chop persists. Read `P(high)` against the base rate above — higher = low-range runs tend to resolve into a move; lower = chop begets chop.

| K | n(IS) | IS P(low/mid/high) | IS P(high) 95% CI | n(OOS) | OOS P(low/mid/high) | OOS P(high) 95% CI |
|--:|--:|---|---|--:|---|---|
| 1 | 5,895 |   55%/  31%/  14% | [  13%,   15%] | 443 |   33%/  41%/  26% | [  22%,   31%] |
| 2 | 3,144 |   63%/  28%/   9% | [   8%,   10%] | 139 |   41%/  40%/  19% | [  13%,   26%] |
| 3 | 1,925 |   65%/  27%/   8% | [   7%,    9%] | 55 |   40%/  42%/  18% | [   9%,   29%] |
| 4 | 1,232 |   67%/  26%/   7% | [   5%,    8%] | 21 |   43%/  48%/  10% | [   0%,   24%] |
| 5 | 811 |   71%/  24%/   6% | [   4%,    8%] | 9 |   67%/  33%/   0% | [   0%,    0%] |

## Read

- After 3 consecutive low-range legs, IS P(next is high) =    8% vs the   33% base rate — **BELOW** base. low-range runs tend to beget more chop.
- IS->OOS regime shift: OOS legs run larger (compare the base-rate rows). The conditional EFFECT replicates but absolute cell probabilities do not transfer — a v2 should use regime-relative bands.
- A cell is trustworthy only where IS and OOS agree and the CI is tight; OOS n thins fast at higher K — treat those as direction-only.
- Next: regime-relative range bands; the directional question (amplitude asymmetry, not leg sign); vol-window events.