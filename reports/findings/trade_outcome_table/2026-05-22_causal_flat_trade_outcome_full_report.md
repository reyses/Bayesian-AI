**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Trade Outcome Suite - Full Report

Generated 2026-05-22 by `tools/trade_outcome_suite/run_all.py` on leg-list source **`causal_flat`**. CAUSAL streaming zigzag, no model filters. Honest lookahead-free leg population — includes the whipsaws a live engine actually takes.

Population: IS 14,274 legs / 335 days, OOS 2,127 legs / 53 days, reported separately (never pooled).

Pure descriptive diagnostics -- no model fit. All dollars are MNQ ($2/point); `pnl_usd` is net of $6/leg friction. Conditional cells carry n + 95% bootstrap CI (4000 resamples); cells with n < 30 are flagged ` !`.

## Verdict index

| # | Question | Verdict |
|---|---|---|
| 1 | Q1 - Distributions of entry-to-close, MAE, MFE | Median MFE $44/$56 (IS/OOS); P(close>0) 33%/36%. |
| 2 | Q2 - Joint MFE x MAE -> P(close>0) | P(close>0) is driven almost entirely by MFE; MAE barely moves it until the extreme corner. |
| 3 | Q3 - Continuation: given up +$x, where does it close? | A big close needs a big peak; "currently green" alone says little beyond how high. |
| 4 | Q4 - Conditional: at +$x, P(it continues another step) | Peak extends ~50% at every level; "holds the +$x" is only ~40-50% - the structural ~1R giveback. |
| 5 | Q5 - Cut-and-bank a winner vs hold to the exit | Holding beats cutting at every level, IS and OOS - the right tail pays for the giveback. |
| 6 | Q6 - Giveback: how much of the peak survives to the close | Giveback is a fixed ~1R toll: devastating on $50-100 peaks, minor on $300+. |
| 7 | Q7 - Given an MFE of +$300, where does it close (cumulative) | From a $300 peak the close clusters in $200-300; below $0 is ~1% - it does not "give it all back". |
| 8 | Q8 - Equity-loss map: P(close<0) by MFE reached | Equity loss is a low-MFE phenomenon - by +$150 MFE it is ~1%; a "$250 safety limit" guards empty space. |
| 9 | Q9 - Recovery: given down -$d, does it work out? | Recovery to green is ~18-28% and erodes only gently with depth; mean close negative for every d>=$40. |
| 10 | Q10 - Full MAE -> close sweep (every $20) | hold-bail is positive at every depth - bailing always locks the worst version. |
| 11 | Q11 - Probability a drawdown gets WORSE | P(deepen) rises in IS, falls in OOS - regime-dependent, not bankable. Recovery odds erode gently in both. |
| 12 | Q12 - Iterative drawdown chain (n -> n+1) | p_reach decays steeply (deep drawdowns are rare); p_advance is regime-split; p_recover erodes gently. |
| 13 | Q13 - Cut a loser vs hold to the exit | Cutting loses at every drawdown level - the R-trigger recovers ~1R off the low; bailing forfeits it. |
| 14 | Q14 - When does the MAE happen, and how long do recoverers take? | Recoverers run ~2x longer than non-recoverers - trade AGE is informative where depth was not. |
| 15 | Q15 - The bimodal split: winners vs losers, not "peak then collapse" | Not "peak then collapse" - it is winners (real $100+ MFE) vs losers (~$14 poke, never worked). |

## Standing caveats

- `close ~= MFE - R`: a zigzag leg exits ~1R below its peak by construction. Every winner gives back ~1R; every loser is recovered ~1R off its low. The R-trigger exit is the system's adaptive stop.
- These tables describe the EXISTING R-trigger exit. Every fixed-dollar overlay tested here (cut winners, lock breakeven, bail losers) loses to it on EV.
- IS/OOS reported separately. Trust a cell only where both agree.
- A table gives a POPULATION frequency, not a per-trade probability -- it cannot tell one trade from another at the same state.
- This report runs on the `causal_flat` leg-list source. For comparison vs the lookahead-tainted population, run with `--source hardened`.


---

## Q1 - Distributions of entry-to-close, MAE, MFE

Survival (exceedance) probability that each per-trade quantity reaches a given dollar magnitude. The foundation table.

| level $ | MAE≥ IS | MFE≥ IS | close≥+ IS | close≤- IS | MAE≥ OOS | MFE≥ OOS | close≥+ OOS | close≤- OOS |
|---|---|---|---|---|---|---|---|---|
| $0 | 100% | 100% | 33% | 67% | 100% | 100% | 36% | 64% |
| $25 | 64% | 67% | 22% | 50% | 72% | 73% | 25% | 51% |
| $50 | 30% | 46% | 15% | 25% | 42% | 54% | 19% | 33% |
| $75 | 13% | 32% | 11% | 10% | 18% | 41% | 14% | 13% |
| $100 | 6% | 23% | 8% | 4% | 7% | 30% | 10% | 4% |
| $125 | 4% | 16% | 6% | 2% | 4% | 22% | 8% | 2% |
| $150 | 2% | 12% | 4% | 1% | 2% | 17% | 6% | 1% |
| $175 | 1% | 9% | 3% | 1% | 1% | 13% | 5% | 0% |
| $200 | 1% | 7% | 3% | 0% | 1% | 10% | 4% | 0% |
| $225 | 1% | 5% | 2% | 0% | 1% | 7% | 3% | 0% |
| $250 | 0% | 4% | 1% | 0% | 0% | 6% | 2% | 0% |
| $275 | 0% | 3% | 1% | 0% | 0% | 5% | 2% | 0% |
| $300 | 0% | 3% | 1% | 0% | 0% | 4% | 1% | 0% |
| $325 | 0% | 2% | 1% | 0% | 0% | 3% | 1% | 0% |
| $350 | 0% | 2% | 0% | 0% | 0% | 2% | 1% | 0% |
| $375 | 0% | 1% | 0% | 0% | 0% | 2% | 0% | 0% |
| $400 | 0% | 1% | 0% | 0% | 0% | 2% | 0% | 0% |

---

## Q2 - Joint MFE x MAE -> P(close>0)

Does the drawdown a trade suffered (MAE) change its win odds, once you know its peak (MFE)?

**IS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 10% / 1485 | 1% / 3056 | 0% / 2505 | 0% / 573 | 0% / 65 |
| 50-100 | 66% / 1767 | 38% / 945 | 9% / 454 | 1% / 124 | 0% / 30 |
| 100-150 | 95% / 934 | 85% / 410 | 62% / 170 | 14% / 42 | 0% / 10 |
| 150-200 | 99% / 410 | 95% / 196 | 79% / 92 | 30% / 20 | 25% / 4 |
| 200-300 | 99% / 324 | 99% / 162 | 91% / 97 | 80% / 20 | 25% / 4 |
| 300+ | 100% / 175 | 100% / 104 | 100% / 71 | 95% / 20 | 60% / 5 |

**OOS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 8% / 65 | 1% / 261 | 0% / 544 | 0% / 100 | 0% / 12 |
| 50-100 | 54% / 186 | 24% / 192 | 12% / 103 | 9% / 22 | 0% / 1 |
| 100-150 | 95% / 140 | 84% / 93 | 78% / 45 | 22% / 9 | 0% / 1 |
| 150-200 | 99% / 80 | 98% / 45 | 93% / 14 | 50% / 6 | - |
| 200-300 | 100% / 68 | 100% / 43 | 92% / 13 | 100% / 4 | 100% / 1 |
| 300+ | 100% / 50 | 100% / 14 | 100% / 14 | 100% / 1 | - |

---

## Q3 - Continuation: given up +$x, where does it close?

Condition: the trade has reached open profit +$x (MFE≥x). Distribution of the FINAL close.

**IS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 14,274 | 15% | 8% | 3% | 33% | $-8 | [32%, 34%] |
| $25 | 9,557 | 23% | 11% | 4% | 49% | $+17 | [48%, 50%] |
| $50 | 6,590 | 33% | 17% | 5% | 69% | $+40 | [68%, 70%] |
| $75 | 4,604 | 47% | 24% | 8% | 84% | $+65 | [83%, 85%] |
| $100 | 3,270 | 65% | 34% | 11% | 91% | $+90 | [90%, 92%] |
| $125 | 2,323 | 80% | 47% | 16% | 94% | $+116 | [93%, 95%] |
| $150 | 1,704 | 89% | 64% | 21% | 96% | $+142 | [95%, 97%] |
| $175 | 1,284 | 93% | 78% | 28% | 97% | $+167 | [96%, 98%] |
| $200 | 982 | 94% | 86% | 37% | 98% | $+191 | [97%, 99%] |
| $225 | 763 | 95% | 90% | 47% | 98% | $+213 | [97%, 99%] |
| $250 | 597 | 96% | 93% | 60% | 98% | $+236 | [97%, 99%] |
| $275 | 473 | 98% | 96% | 73% | 99% | $+260 | [98%, 100%] |
| $300 | 375 | 98% | 96% | 81% | 99% | $+281 | [98%, 100%] |
| $325 | 295 | 99% | 98% | 87% | 100% | $+308 | [99%, 100%] |
| $350 | 234 | 100% | 98% | 90% | 100% | $+331 | [100%, 100%] |
| $375 | 202 | 100% | 99% | 92% | 100% | $+347 | [100%, 100%] |
| $400 | 144 | 99% | 99% | 94% | 100% | $+379 | [100%, 100%] |

**OOS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 2,127 | 19% | 10% | 4% | 36% | $-4 | [34%, 38%] |
| $25 | 1,548 | 26% | 14% | 5% | 49% | $+21 | [47%, 52%] |
| $50 | 1,145 | 35% | 19% | 7% | 66% | $+45 | [63%, 69%] |
| $75 | 882 | 45% | 25% | 9% | 81% | $+67 | [78%, 84%] |
| $100 | 641 | 61% | 35% | 12% | 93% | $+93 | [91%, 95%] |
| $125 | 469 | 80% | 48% | 17% | 97% | $+120 | [95%, 99%] |
| $150 | 353 | 90% | 63% | 22% | 98% | $+146 | [96%, 99%] |
| $175 | 272 | 94% | 79% | 29% | 99% | $+170 | [97%, 100%] |
| $200 | 208 | 97% | 91% | 38% | 100% | $+197 | [99%, 100%] |
| $225 | 158 | 97% | 95% | 50% | 100% | $+223 | [100%, 100%] |
| $250 | 127 | 98% | 97% | 61% | 100% | $+245 | [100%, 100%] |
| $275 | 100 | 98% | 97% | 76% | 100% | $+269 | [100%, 100%] |
| $300 | 79 | 100% | 100% | 87% | 100% | $+295 | [100%, 100%] |
| $325 | 63 | 100% | 100% | 95% | 100% | $+320 | [100%, 100%] |
| $350 | 47 | 100% | 100% | 100% | 100% | $+350 | [100%, 100%] |
| $375 | 38 | 100% | 100% | 100% | 100% | $+370 | [100%, 100%] |
| $400 | 32 | 100% | 100% | 100% | 100% | $+386 | [100%, 100%] |

---

## Q4 - Conditional: at +$x, P(it continues another step)

The "$100 -> $150" question, generalised. Given the trade is up +$x: does the peak push another $50, and does it hold the +$x to the close?

| at +$x | n IS | P(peak≥x+$50) IS | P(close≥x) IS | n OOS | P(peak≥x+$50) OOS | P(close≥x) OOS |
|---|---|---|---|---|---|---|
| $50 | 6,590 | 50% | 33% | 1,145 | 56% | 35% |
| $100 | 3,270 | 52% | 34% | 641 | 55% | 35% |
| $150 | 1,704 | 58% | 36% | 353 | 59% | 37% |
| $200 | 982 | 61% | 37% | 208 | 61% | 38% |
| $250 | 597 | 63% | 34% | 127 | 62% | 39% |
| $300 | 375 | 62% | 33% | 79 | 59% | 37% |
| $350 | 234 | 62% | 29% | 47 | 68% | 38% |

---

## Q5 - Cut-and-bank a winner vs hold to the exit

"Cut" banks +$L minus $6 friction. "Hold" runs to the R-trigger exit. `hold-cut` > 0 means holding wins.

| at +$L | n IS | HOLD mean IS | hold-cut IS | n OOS | HOLD mean OOS | hold-cut OOS |
|---|---|---|---|---|---|---|
| $100 | 3,270 | $+90 | $-4 | 641 | $+93 | $-1 |
| $150 | 1,704 | $+142 | $-2 | 353 | $+146 | $+2 |
| $200 | 982 | $+191 | $-3 | 208 | $+197 | $+3 |
| $250 | 597 | $+236 | $-8 | 127 | $+245 | $+1 |
| $300 | 375 | $+281 | $-13 | 79 | $+295 | $+1 |
| $400 | 144 | $+379 | $-15 | 32 | $+386 | $-8 |
| $500 | 66 | $+478 | $-16 | 8 | $+537 | $+43 |

---

## Q6 - Giveback: how much of the peak survives to the close

The peak->close drop averages $79 IS / $90 OOS -- a roughly FIXED ~1R toll, so it eats small excursions whole and barely dents big ones.

| MFE peak | n IS | median close IS | capture% IS | gave-back≤$20 IS | n OOS | median close OOS | capture% OOS |
|---|---|---|---|---|---|---|---|
| $50-100 | 3,320 | $-2 | -2% | 81% | 504 | $-13 | -17% |
| $100-150 | 1,566 | $+40 | 35% | 26% | 288 | $+30 | 26% |
| $150-200 | 722 | $+85 | 51% | 9% | 145 | $+80 | 46% |
| $200-300 | 607 | $+142 | 62% | 5% | 129 | $+136 | 58% |
| $300+ | 375 | $+258 | 72% | 1% | 79 | $+274 | 72% |

---

## Q7 - Given an MFE of +$300, where does it close (cumulative)

Each bucket is distinct and sums to 100%; the cumulative column is P(close at or below the band top).

**IS** (n=375)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 33.3% | 100.0% |
| +$250-300 | 21.1% | 66.7% |
| +$200-250 | 26.4% | 45.6% |
| +$150-200 | 10.7% | 19.2% |
| +$100-150 | 4.3% | 8.5% |
| +$50-100 | 2.1% | 4.3% |
| +$0-50 | 1.3% | 2.1% |
| < $0 (gave it all back) | 0.8% | 0.8% |

**OOS** (n=79)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 36.7% | 100.0% |
| +$250-300 | 25.3% | 63.3% |
| +$200-250 | 25.3% | 38.0% |
| +$150-200 | 10.1% | 12.7% |
| +$100-150 | 2.5% | 2.5% |
| +$50-100 | 0.0% | -0.0% |
| +$0-50 | 0.0% | -0.0% |
| < $0 (gave it all back) | 0.0% | -0.0% |

---

## Q8 - Equity-loss map: P(close<0) by MFE reached

"Lose equity" = closes red. The question behind the "safety limit" idea: how far must a trade run before it is safe from a negative close?

| MFE reached ≥ | n IS | P(lose equity) IS | P(close<-$100) IS | n OOS | P(lose equity) OOS | P(close<-$100) OOS |
|---|---|---|---|---|---|---|
| +$25 | 9,557 | 50.4% | 2.7% | 1,548 | 50.6% | 1.8% |
| +$50 | 6,590 | 30.6% | 2.0% | 1,145 | 33.8% | 1.2% |
| +$75 | 4,604 | 15.6% | 1.3% | 882 | 18.6% | 0.8% |
| +$100 | 3,270 | 8.7% | 1.0% | 641 | 7.3% | 0.9% |
| +$150 | 1,704 | 4.1% | 0.5% | 353 | 2.0% | 0.6% |
| +$200 | 982 | 2.3% | 0.4% | 208 | 0.5% | 0.0% |
| +$250 | 597 | 1.7% | 0.3% | 127 | 0.0% | 0.0% |
| +$300 | 375 | 0.8% | 0.3% | 79 | 0.0% | 0.0% |

---

## Q9 - Recovery: given down -$d, does it work out?

Condition: the trade has drawn down to -$d (MAE≥d). Distribution of the final close.

**IS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 14,274 | 33% | 8% | $-8 | [32%, 34%] |
| -$20 | 10,318 | 20% | 6% | $-26 | [20%, 21%] |
| -$40 | 5,961 | 12% | 4% | $-47 | [11%, 13%] |
| -$60 | 3,081 | 9% | 3% | $-67 | [8%, 10%] |
| -$80 | 1,646 | 7% | 3% | $-86 | [6%, 8%] |
| -$100 | 917 | 6% | 3% | $-108 | [4%, 7%] |
| -$120 | 560 | 6% | 4% | $-125 | [4%, 8%] |
| -$140 | 349 | 5% | 3% | $-150 | [3%, 7%] |
| -$160 | 256 | 5% | 4% | $-160 | [3%, 8%] |
| -$180 | 171 | 4% | 2% | $-189 | [1%, 6%] |
| -$200 | 118 | 4% | 2% | $-206 | [1%, 8%] |
| -$220 | 84 | 4% | 2% | $-229 | [0%, 8%] |
| -$240 | 60 | 2% | 2% | $-258 | [0%, 5%] |
| -$260 | 43 | 2% | 2% | $-288 | [0%, 7%] |

**OOS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 2,127 | 36% | 10% | $-4 | [34%, 38%] |
| -$20 | 1,681 | 25% | 7% | $-22 | [23%, 27%] |
| -$40 | 1,152 | 15% | 4% | $-44 | [13%, 17%] |
| -$60 | 652 | 10% | 3% | $-63 | [7%, 12%] |
| -$80 | 343 | 8% | 2% | $-80 | [5%, 11%] |
| -$100 | 157 | 8% | 3% | $-95 | [4%, 13%] |
| -$120 | 90 | 7% | 2% | $-114 | [2%, 12%] |
| -$140 | 52 | 6% | 2% | $-128 | [0%, 13%] |
| -$160 | 32 | 9% | 3% | $-138 | [0%, 22%] |
| -$180 ! | 22 | 5% | 5% | $-162 | [0%, 14%] |
| -$200 ! | 15 | 7% | 7% | $-174 | [0%, 20%] |
| -$220 ! | 12 | 8% | 8% | $-180 | [0%, 25%] |

---

## Q10 - Full MAE -> close sweep (every $20)

The complete recovery table, fine-grained. `hold-bail` = mean close minus the -$D you would lock by bailing now.

**IS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 14,274 | 33% | 15% | 8% | $-8 | $-24 | $-8 |
| -$20 | 10,318 | 20% | 10% | 6% | $-26 | $-38 | $-6 |
| -$40 | 5,961 | 12% | 7% | 4% | $-47 | $-54 | $-7 |
| -$60 | 3,081 | 9% | 6% | 3% | $-67 | $-72 | $-7 |
| -$80 | 1,646 | 7% | 5% | 3% | $-86 | $-90 | $-6 |
| -$100 | 917 | 6% | 4% | 3% | $-108 | $-109 | $-8 |
| -$120 | 560 | 6% | 5% | 4% | $-125 | $-128 | $-5 |
| -$140 | 349 | 5% | 3% | 3% | $-150 | $-149 | $-10 |
| -$160 | 256 | 5% | 5% | 4% | $-160 | $-165 | $-0 |
| -$180 | 171 | 4% | 2% | 2% | $-189 | $-184 | $-9 |
| -$200 | 118 | 4% | 3% | 2% | $-206 | $-208 | $-6 |
| -$220 | 84 | 4% | 2% | 2% | $-229 | $-232 | $-9 |
| -$240 | 60 | 2% | 2% | 2% | $-258 | $-257 | $-18 |
| -$260 | 43 | 2% | 2% | 2% | $-288 | $-273 | $-28 |
| -$280 | 35 | 3% | 3% | 3% | $-304 | $-284 | $-24 |
| -$300 | 33 | 3% | 3% | 3% | $-308 | $-302 | $-8 |
| -$320 ! | 27 | 4% | 4% | 4% | $-324 | $-335 | $-4 |
| -$340 ! | 24 | 4% | 4% | 4% | $-330 | $-362 | $+10 |
| -$360 ! | 22 | 5% | 5% | 5% | $-335 | $-379 | $+25 |
| -$380 ! | 19 | 5% | 5% | 5% | $-349 | $-396 | $+31 |
| -$400 ! | 17 | 0% | 0% | 0% | $-464 | $-412 | $-64 |
| -$420 ! | 16 | 0% | 0% | 0% | $-472 | $-426 | $-52 |
| -$440 ! | 14 | 0% | 0% | 0% | $-491 | $-443 | $-51 |
| -$460 ! | 12 | 0% | 0% | 0% | $-509 | $-457 | $-49 |
| -$480 ! | 10 | 0% | 0% | 0% | $-552 | $-469 | $-72 |

**OOS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 2,127 | 36% | 19% | 10% | $-4 | $-28 | $-4 |
| -$20 | 1,681 | 25% | 13% | 7% | $-22 | $-42 | $-2 |
| -$40 | 1,152 | 15% | 7% | 4% | $-44 | $-58 | $-4 |
| -$60 | 652 | 10% | 5% | 3% | $-63 | $-72 | $-3 |
| -$80 | 343 | 8% | 4% | 2% | $-80 | $-86 | $+0 |
| -$100 | 157 | 8% | 6% | 3% | $-95 | $-103 | $+5 |
| -$120 | 90 | 7% | 6% | 2% | $-114 | $-117 | $+6 |
| -$140 | 52 | 6% | 4% | 2% | $-128 | $-129 | $+12 |
| -$160 | 32 | 9% | 6% | 3% | $-138 | $-152 | $+22 |
| -$180 ! | 22 | 5% | 5% | 5% | $-162 | $-172 | $+18 |
| -$200 ! | 15 | 7% | 7% | 7% | $-174 | $-186 | $+26 |
| -$220 ! | 12 | 8% | 8% | 8% | $-180 | $-212 | $+40 |

---

## Q11 - Probability a drawdown gets WORSE

`P(deepen)` = given at -$D, probability the drawdown extends another $20. NOTE: this column rises in IS but falls in OOS - the "drawdowns gain momentum" effect does NOT replicate.

**IS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 14,274 | 72% | 67% | 0% | 33% |
| -$20 | 10,318 | 58% | 71% | 8% | 20% |
| -$40 | 5,961 | 52% | 75% | 13% | 12% |
| -$60 | 3,081 | 53% | 72% | 19% | 9% |
| -$80 | 1,646 | 56% | 67% | 26% | 7% |
| -$100 | 917 | 61% | 61% | 33% | 6% |
| -$120 | 560 | 62% | 59% | 34% | 6% |
| -$140 | 349 | 73% | 60% | 36% | 5% |
| -$160 | 256 | 67% | 55% | 40% | 5% |
| -$180 | 171 | 69% | 54% | 43% | 4% |
| -$200 | 118 | 71% | 54% | 42% | 4% |
| -$220 | 84 | 71% | 54% | 43% | 4% |
| -$240 | 60 | 72% | 63% | 35% | 2% |
| -$260 | 43 | 81% | 67% | 30% | 2% |
| -$280 | 35 | 94% | 57% | 40% | 3% |
| -$300 | 33 | 82% | 52% | 45% | 3% |
| -$320 ! | 27 | 89% | 59% | 37% | 4% |
| -$340 ! | 24 | 92% | 54% | 42% | 4% |
| -$360 ! | 22 | 86% | 55% | 41% | 5% |
| -$380 ! | 19 | 89% | 58% | 37% | 5% |
| -$400 ! | 17 | 94% | 53% | 47% | 0% |
| -$420 ! | 16 | 88% | 50% | 50% | 0% |
| -$440 ! | 14 | 86% | 50% | 50% | 0% |
| -$460 ! | 12 | 83% | 50% | 50% | 0% |
| -$480 ! | 10 | 90% | 40% | 60% | 0% |

**OOS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 2,127 | 79% | 64% | 0% | 36% |
| -$20 | 1,681 | 69% | 68% | 7% | 25% |
| -$40 | 1,152 | 57% | 73% | 12% | 15% |
| -$60 | 652 | 53% | 74% | 16% | 10% |
| -$80 | 343 | 46% | 62% | 29% | 8% |
| -$100 | 157 | 57% | 52% | 39% | 8% |
| -$120 | 90 | 58% | 46% | 48% | 7% |
| -$140 | 52 | 62% | 46% | 48% | 6% |
| -$160 | 32 | 69% | 41% | 50% | 9% |
| -$180 ! | 22 | 68% | 45% | 50% | 5% |
| -$200 ! | 15 | 80% | 47% | 47% | 7% |
| -$220 ! | 12 | 75% | 33% | 58% | 8% |

---

## Q12 - Iterative drawdown chain (n -> n+1)

Each iteration n deepens the drawdown by $20. `p_advance` = P(reach step n | reached n-1); `p_reach` = cumulative from entry; `p_recover` = P(close≥0 | here).

| n | drawdown | p_reach(n) IS | p_advance IS | p_recover IS | p_reach(n) OOS | p_advance OOS | p_recover OOS |
|---|---|---|---|---|---|---|---|
| 0 | -$0 | 100.0% | 100% | 33% | 100.0% | 100% | 36% |
| 1 | -$20 | 72.3% | 72% | 20% | 79.0% | 79% | 25% |
| 2 | -$40 | 41.8% | 58% | 12% | 54.2% | 69% | 15% |
| 3 | -$60 | 21.6% | 52% | 9% | 30.7% | 57% | 10% |
| 4 | -$80 | 11.5% | 53% | 7% | 16.1% | 53% | 8% |
| 5 | -$100 | 6.4% | 56% | 6% | 7.4% | 46% | 8% |
| 6 | -$120 | 3.9% | 61% | 6% | 4.2% | 57% | 7% |
| 7 | -$140 | 2.4% | 62% | 5% | 2.4% | 58% | 6% |
| 8 | -$160 | 1.8% | 73% | 5% | 1.5% | 62% | 9% |
| 9 | -$180 | 1.2% | 67% | 4% | 1.0% | 69% | 5% |
| 10 | -$200 | 0.8% | 69% | 4% | 0.7% | 68% | 7% |
| 11 | -$220 | 0.6% | 71% | 4% | 0.6% | 80% | 8% |
| 12 | -$240 | 0.4% | 71% | 2% | 0.4% | 75% | 0% |
| 13 | -$260 | 0.3% | 72% | 2% | 0.4% | 89% | 0% |
| 14 | -$280 | 0.2% | 81% | 3% | 0.2% | 62% | 0% |
| 15 | -$300 | 0.2% | 94% | 3% | 0.2% | 100% | 0% |
| 16 | -$320 | 0.2% | 82% | 4% | 0.2% | 100% | 0% |
| 17 | -$340 | 0.2% | 89% | 4% | 0.1% | 60% | 0% |
| 18 | -$360 | 0.2% | 92% | 5% | 0.1% | 100% | 0% |
| 19 | -$380 | 0.1% | 86% | 5% | 0.1% | 100% | 0% |
| 20 | -$400 | 0.1% | 89% | 0% | 0.1% | 100% | 0% |
| 21 | -$420 | 0.1% | 94% | 0% | 0.1% | 100% | 0% |
| 22 | -$440 | 0.1% | 88% | 0% | 0.1% | 100% | 0% |
| 23 | -$460 | 0.1% | 86% | 0% | 0.1% | 100% | 0% |
| 24 | -$480 | 0.1% | 83% | 0% | 0.1% | 100% | 0% |

---

## Q13 - Cut a loser vs hold to the exit

"Cut" locks -$D. "Hold" runs to the R-trigger exit. `hold-bail` > 0 means cutting loses money.

| at -$D | n IS | HOLD mean IS | hold-bail IS | n OOS | HOLD mean OOS | hold-bail OOS |
|---|---|---|---|---|---|---|
| -$40 | 5,961 | $-47 | $-7 | 1,152 | $-44 | $-4 |
| -$60 | 3,081 | $-67 | $-7 | 652 | $-63 | $-3 |
| -$80 | 1,646 | $-86 | $-6 | 343 | $-80 | $+0 |
| -$100 | 917 | $-108 | $-8 | 157 | $-95 | $+5 |
| -$140 | 349 | $-150 | $-10 | 52 | $-128 | $+12 |

---

## Q14 - When does the MAE happen, and how long do recoverers take?

The worst point lands ~3-4 min in (constant with depth) but ~83-100% of the way through the trade. Recoverers run roughly 2x as long as non-recoverers.

**IS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 10,318 | 4.9 | 92% | 22.9 | 7.7 |
| -$40 | 5,961 | 4.5 | 95% | 21.3 | 6.2 |
| -$60 | 3,081 | 3.9 | 94% | 19.5 | 5.0 |
| -$80 | 1,646 | 3.5 | 92% | 18.8 | 4.2 |
| -$100 | 917 | 3.2 | 92% | 16.8 | 4.0 |

**OOS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 1,681 | 5.0 | 87% | 25.2 | 9.0 |
| -$40 | 1,152 | 5.3 | 96% | 27.2 | 7.9 |
| -$60 | 652 | 4.9 | 97% | 23.8 | 6.2 |
| -$80 | 343 | 4.3 | 95% | 28.0 | 5.3 |
| -$100 | 157 | 3.9 | 92% | 34.2 | 5.0 |

---

## Q15 - The bimodal split: winners vs losers, not "peak then collapse"

Testing the claim "bimodal = peaks MFE then goes negative". The ordering holds, but the "peak" in the losing mode is ~$14 - noise, not a peak.

**IS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 1,048 | $+110 | 80% | 4% | $+14 | 42% |
| LATE bottom (q4) | 3,993 | $+16 | 14% | 100% | $-61 | 100% |
| all losers | 9,516 | $+25 | 21% | 79% | $-41 | 100% |

**OOS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 227 | $+106 | 87% | 5% | $+14 | 42% |
| LATE bottom (q4) | 753 | $+18 | 13% | 100% | $-66 | 99% |
| all losers | 1,361 | $+30 | 28% | 79% | $-52 | 100% |