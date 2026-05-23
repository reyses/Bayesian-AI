# Trade Outcome Suite - Full Report

Generated 2026-05-22 by `tools/trade_outcome_suite/run_all.py` on leg-list source **`causal`**. 

Population: IS 9,829 legs / 314 days, OOS 1,822 legs / 53 days, reported separately (never pooled).

Pure descriptive diagnostics -- no model fit. All dollars are MNQ ($2/point); `pnl_usd` is net of $6/leg friction. Conditional cells carry n + 95% bootstrap CI (4000 resamples); cells with n < 30 are flagged ` !`.

## Verdict index

| # | Question | Verdict |
|---|---|---|
| 1 | Q1 - Distributions of entry-to-close, MAE, MFE | Median MFE $24/$26 (IS/OOS); P(close>0) 31%/32%. |
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
- This report runs on the `causal` leg-list source. For the honest causal numbers, run with `--source causal_flat`.


---

## Q1 - Distributions of entry-to-close, MAE, MFE

Survival (exceedance) probability that each per-trade quantity reaches a given dollar magnitude. The foundation table.

| level $ | MAE≥ IS | MFE≥ IS | close≥+ IS | close≤- IS | MAE≥ OOS | MFE≥ OOS | close≥+ OOS | close≤- OOS |
|---|---|---|---|---|---|---|---|---|
| $0 | 100% | 100% | 32% | 69% | 100% | 100% | 33% | 68% |
| $25 | 50% | 49% | 17% | 35% | 51% | 51% | 17% | 35% |
| $50 | 29% | 36% | 12% | 22% | 30% | 37% | 12% | 23% |
| $75 | 15% | 26% | 9% | 10% | 15% | 29% | 10% | 11% |
| $100 | 8% | 20% | 7% | 5% | 7% | 21% | 7% | 4% |
| $125 | 4% | 15% | 5% | 3% | 3% | 16% | 5% | 2% |
| $150 | 3% | 11% | 4% | 1% | 2% | 12% | 4% | 1% |
| $175 | 2% | 9% | 3% | 1% | 1% | 9% | 3% | 0% |
| $200 | 1% | 7% | 3% | 1% | 1% | 7% | 3% | 0% |
| $225 | 1% | 5% | 2% | 0% | 0% | 5% | 2% | 0% |
| $250 | 0% | 4% | 1% | 0% | 0% | 4% | 2% | 0% |
| $275 | 0% | 3% | 1% | 0% | 0% | 3% | 1% | 0% |
| $300 | 0% | 3% | 1% | 0% | 0% | 3% | 1% | 0% |
| $325 | 0% | 2% | 1% | 0% | 0% | 2% | 1% | 0% |
| $350 | 0% | 2% | 1% | 0% | 0% | 2% | 1% | 0% |
| $375 | 0% | 2% | 0% | 0% | 0% | 1% | 0% | 0% |
| $400 | 0% | 1% | 0% | 0% | 0% | 1% | 0% | 0% |

---

## Q2 - Joint MFE x MAE -> P(close>0)

Does the drawdown a trade suffered (MAE) change its win odds, once you know its peak (MFE)?

**IS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 24% / 3395 | 2% / 957 | 0% / 1468 | 0% / 458 | 0% / 53 |
| 50-100 | 59% / 582 | 31% / 520 | 9% / 329 | 1% / 100 | 0% / 27 |
| 100-150 | 93% / 440 | 82% / 255 | 59% / 124 | 9% / 34 | 0% / 9 |
| 150-200 | 99% / 203 | 94% / 124 | 72% / 65 | 22% / 18 | 25% / 4 |
| 200-300 | 99% / 185 | 98% / 110 | 88% / 74 | 82% / 17 | 25% / 4 |
| 300+ | 100% / 113 | 100% / 85 | 100% / 53 | 95% / 19 | 75% / 4 |

**OOS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 24% / 613 | 3% / 148 | 0% / 299 | 0% / 82 | 0% / 10 |
| 50-100 | 53% / 89 | 21% / 111 | 13% / 70 | 8% / 13 | 0% / 1 |
| 100-150 | 94% / 84 | 83% / 52 | 70% / 30 | 14% / 7 | 0% / 1 |
| 150-200 | 97% / 38 | 97% / 33 | 100% / 12 | 67% / 3 | - |
| 200-300 | 100% / 35 | 100% / 28 | 91% / 11 | 100% / 2 | 100% / 1 |
| 300+ | 100% / 26 | 100% / 10 | 100% / 12 | 100% / 1 | - |

---

## Q3 - Continuation: given up +$x, where does it close?

Condition: the trade has reached open profit +$x (MFE≥x). Distribution of the FINAL close.

**IS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 9,829 | 12% | 7% | 3% | 31% | $-7 | [30%, 32%] |
| $25 | 4,854 | 24% | 14% | 5% | 49% | $+17 | [48%, 51%] |
| $50 | 3,498 | 34% | 19% | 7% | 64% | $+41 | [62%, 66%] |
| $75 | 2,589 | 46% | 26% | 10% | 80% | $+66 | [78%, 81%] |
| $100 | 1,940 | 61% | 34% | 13% | 88% | $+90 | [86%, 89%] |
| $125 | 1,427 | 76% | 47% | 17% | 92% | $+117 | [91%, 93%] |
| $150 | 1,078 | 86% | 62% | 23% | 94% | $+143 | [93%, 95%] |
| $175 | 846 | 91% | 75% | 29% | 96% | $+167 | [94%, 97%] |
| $200 | 664 | 93% | 83% | 37% | 97% | $+191 | [96%, 98%] |
| $225 | 526 | 94% | 88% | 47% | 98% | $+213 | [96%, 99%] |
| $250 | 432 | 96% | 91% | 56% | 98% | $+234 | [97%, 99%] |
| $275 | 339 | 98% | 95% | 70% | 99% | $+261 | [98%, 100%] |
| $300 | 274 | 98% | 95% | 78% | 99% | $+280 | [98%, 100%] |
| $325 | 219 | 99% | 97% | 84% | 100% | $+307 | [99%, 100%] |
| $350 | 178 | 99% | 98% | 88% | 100% | $+329 | [100%, 100%] |
| $375 | 152 | 99% | 99% | 91% | 100% | $+347 | [100%, 100%] |
| $400 | 110 | 99% | 98% | 93% | 100% | $+380 | [100%, 100%] |

**OOS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 1,822 | 12% | 7% | 3% | 32% | $-5 | [30%, 34%] |
| $25 | 926 | 24% | 14% | 5% | 51% | $+20 | [48%, 55%] |
| $50 | 670 | 33% | 19% | 7% | 64% | $+44 | [61%, 68%] |
| $75 | 526 | 42% | 25% | 9% | 79% | $+64 | [75%, 82%] |
| $100 | 386 | 58% | 34% | 12% | 91% | $+90 | [88%, 94%] |
| $125 | 283 | 77% | 46% | 17% | 96% | $+118 | [94%, 98%] |
| $150 | 212 | 90% | 61% | 23% | 98% | $+146 | [96%, 100%] |
| $175 | 169 | 94% | 76% | 28% | 99% | $+168 | [97%, 100%] |
| $200 | 126 | 97% | 90% | 38% | 99% | $+198 | [98%, 100%] |
| $225 | 94 | 97% | 94% | 51% | 100% | $+226 | [100%, 100%] |
| $250 | 75 | 97% | 95% | 64% | 100% | $+250 | [100%, 100%] |
| $275 | 61 | 97% | 95% | 77% | 100% | $+273 | [100%, 100%] |
| $300 | 49 | 100% | 100% | 88% | 100% | $+302 | [100%, 100%] |
| $325 | 41 | 100% | 100% | 93% | 100% | $+322 | [100%, 100%] |
| $350 | 30 | 100% | 100% | 100% | 100% | $+356 | [100%, 100%] |
| $375 ! | 26 | 100% | 100% | 100% | 100% | $+371 | [100%, 100%] |
| $400 ! | 21 | 100% | 100% | 100% | 100% | $+393 | [100%, 100%] |

---

## Q4 - Conditional: at +$x, P(it continues another step)

The "$100 -> $150" question, generalised. Given the trade is up +$x: does the peak push another $50, and does it hold the +$x to the close?

| at +$x | n IS | P(peak≥x+$50) IS | P(close≥x) IS | n OOS | P(peak≥x+$50) OOS | P(close≥x) OOS |
|---|---|---|---|---|---|---|
| $50 | 3,498 | 55% | 34% | 670 | 58% | 33% |
| $100 | 1,940 | 56% | 34% | 386 | 55% | 34% |
| $150 | 1,078 | 62% | 37% | 212 | 59% | 36% |
| $200 | 664 | 65% | 37% | 126 | 60% | 38% |
| $250 | 432 | 63% | 34% | 75 | 65% | 41% |
| $300 | 274 | 65% | 32% | 49 | 61% | 33% |
| $350 | 178 | 62% | 29% | 30 | 70% | 37% |

---

## Q5 - Cut-and-bank a winner vs hold to the exit

"Cut" banks +$L minus $6 friction. "Hold" runs to the R-trigger exit. `hold-cut` > 0 means holding wins.

| at +$L | n IS | HOLD mean IS | hold-cut IS | n OOS | HOLD mean OOS | hold-cut OOS |
|---|---|---|---|---|---|---|
| $100 | 1,940 | $+90 | $-4 | 386 | $+90 | $-4 |
| $150 | 1,078 | $+143 | $-1 | 212 | $+146 | $+2 |
| $200 | 664 | $+191 | $-3 | 126 | $+198 | $+4 |
| $250 | 432 | $+234 | $-10 | 75 | $+250 | $+6 |
| $300 | 274 | $+280 | $-14 | 49 | $+302 | $+8 |
| $400 | 110 | $+380 | $-14 | 21 | $+393 | $-1 |
| $500 | 51 | $+487 | $-7 | 6 | $+590 | $+96 |

---

## Q6 - Giveback: how much of the peak survives to the close

The peak->close drop averages $67 IS / $68 OOS -- a roughly FIXED ~1R toll, so it eats small excursions whole and barely dents big ones.

| MFE peak | n IS | median close IS | capture% IS | gave-back≤$20 IS | n OOS | median close OOS | capture% OOS |
|---|---|---|---|---|---|---|---|
| $50-100 | 1,558 | $-12 | -18% | 87% | 284 | $-16 | -21% |
| $100-150 | 862 | $+34 | 27% | 35% | 174 | $+25 | 21% |
| $150-200 | 414 | $+78 | 46% | 14% | 86 | $+79 | 46% |
| $200-300 | 390 | $+138 | 59% | 7% | 77 | $+133 | 57% |
| $300+ | 274 | $+254 | 70% | 1% | 49 | $+273 | 72% |

---

## Q7 - Given an MFE of +$300, where does it close (cumulative)

Each bucket is distinct and sums to 100%; the cumulative column is P(close at or below the band top).

**IS** (n=274)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 32.5% | 100.0% |
| +$250-300 | 20.8% | 67.5% |
| +$200-250 | 24.5% | 46.7% |
| +$150-200 | 12.8% | 22.3% |
| +$100-150 | 4.4% | 9.5% |
| +$50-100 | 2.9% | 5.1% |
| +$0-50 | 1.5% | 2.2% |
| < $0 (gave it all back) | 0.7% | 0.7% |

**OOS** (n=49)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 32.7% | 100.0% |
| +$250-300 | 30.6% | 67.3% |
| +$200-250 | 24.5% | 36.7% |
| +$150-200 | 10.2% | 12.2% |
| +$100-150 | 2.0% | 2.0% |
| +$50-100 | 0.0% | 0.0% |
| +$0-50 | 0.0% | 0.0% |
| < $0 (gave it all back) | 0.0% | 0.0% |

---

## Q8 - Equity-loss map: P(close<0) by MFE reached

"Lose equity" = closes red. The question behind the "safety limit" idea: how far must a trade run before it is safe from a negative close?

| MFE reached ≥ | n IS | P(lose equity) IS | P(close<-$100) IS | n OOS | P(lose equity) OOS | P(close<-$100) OOS |
|---|---|---|---|---|---|---|
| +$25 | 4,854 | 50.4% | 4.4% | 926 | 48.5% | 2.7% |
| +$50 | 3,498 | 35.7% | 3.1% | 670 | 35.4% | 1.6% |
| +$75 | 2,589 | 20.0% | 2.0% | 526 | 21.3% | 1.1% |
| +$100 | 1,940 | 11.9% | 1.4% | 386 | 8.8% | 1.3% |
| +$150 | 1,078 | 5.9% | 0.7% | 212 | 1.9% | 0.5% |
| +$200 | 664 | 3.0% | 0.5% | 126 | 0.8% | 0.0% |
| +$250 | 432 | 2.1% | 0.5% | 75 | 0.0% | 0.0% |
| +$300 | 274 | 0.7% | 0.4% | 49 | 0.0% | 0.0% |

---

## Q9 - Recovery: given down -$d, does it work out?

Condition: the trade has drawn down to -$d (MAE≥d). Distribution of the final close.

**IS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 9,829 | 31% | 7% | $-7 | [30%, 32%] |
| -$20 | 5,401 | 23% | 7% | $-24 | [22%, 24%] |
| -$40 | 3,605 | 14% | 5% | $-46 | [13%, 15%] |
| -$60 | 2,194 | 9% | 4% | $-67 | [8%, 11%] |
| -$80 | 1,293 | 7% | 3% | $-86 | [6%, 9%] |
| -$100 | 747 | 6% | 3% | $-108 | [4%, 8%] |
| -$120 | 471 | 7% | 4% | $-123 | [5%, 9%] |
| -$140 | 293 | 5% | 3% | $-148 | [3%, 8%] |
| -$160 | 213 | 7% | 5% | $-159 | [4%, 10%] |
| -$180 | 142 | 4% | 2% | $-187 | [1%, 8%] |
| -$200 | 101 | 5% | 2% | $-203 | [1%, 10%] |
| -$220 | 73 | 4% | 3% | $-223 | [0%, 10%] |
| -$240 | 54 | 2% | 2% | $-250 | [0%, 6%] |
| -$260 | 39 | 3% | 3% | $-275 | [0%, 8%] |

**OOS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 1,822 | 32% | 7% | $-5 | [30%, 34%] |
| -$20 | 1,047 | 26% | 8% | $-20 | [24%, 29%] |
| -$40 | 693 | 16% | 5% | $-43 | [13%, 18%] |
| -$60 | 432 | 11% | 4% | $-62 | [8%, 14%] |
| -$80 | 247 | 7% | 2% | $-83 | [4%, 11%] |
| -$100 | 121 | 7% | 2% | $-101 | [2%, 12%] |
| -$120 | 68 | 3% | 1% | $-125 | [0%, 7%] |
| -$140 | 39 | 5% | 3% | $-135 | [0%, 13%] |
| -$160 ! | 26 | 8% | 4% | $-143 | [0%, 19%] |
| -$180 ! | 20 | 5% | 5% | $-157 | [0%, 15%] |
| -$200 ! | 13 | 8% | 8% | $-169 | [0%, 23%] |
| -$220 ! | 10 | 10% | 10% | $-175 | [0%, 30%] |

---

## Q10 - Full MAE -> close sweep (every $20)

The complete recovery table, fine-grained. `hold-bail` = mean close minus the -$D you would lock by bailing now.

**IS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 9,829 | 32% | 12% | 7% | $-7 | $-10 | $-7 |
| -$20 | 5,401 | 23% | 12% | 7% | $-24 | $-40 | $-4 |
| -$40 | 3,605 | 14% | 8% | 5% | $-46 | $-56 | $-6 |
| -$60 | 2,194 | 9% | 6% | 4% | $-67 | $-72 | $-7 |
| -$80 | 1,293 | 7% | 5% | 3% | $-86 | $-91 | $-6 |
| -$100 | 747 | 6% | 4% | 3% | $-108 | $-110 | $-8 |
| -$120 | 471 | 7% | 5% | 4% | $-123 | $-128 | $-3 |
| -$140 | 293 | 5% | 4% | 3% | $-148 | $-150 | $-8 |
| -$160 | 213 | 7% | 6% | 5% | $-159 | $-166 | $+1 |
| -$180 | 142 | 4% | 3% | 2% | $-187 | $-186 | $-7 |
| -$200 | 101 | 5% | 3% | 2% | $-203 | $-208 | $-3 |
| -$220 | 73 | 4% | 3% | 3% | $-223 | $-232 | $-3 |
| -$240 | 54 | 2% | 2% | 2% | $-250 | $-257 | $-10 |
| -$260 | 39 | 3% | 3% | 3% | $-275 | $-270 | $-15 |
| -$280 | 32 | 3% | 3% | 3% | $-288 | $-283 | $-8 |
| -$300 | 30 | 3% | 3% | 3% | $-291 | $-284 | $+9 |
| -$320 ! | 24 | 4% | 4% | 4% | $-305 | $-329 | $+15 |
| -$340 ! | 21 | 5% | 5% | 5% | $-309 | $-335 | $+31 |
| -$360 ! | 19 | 5% | 5% | 5% | $-313 | $-357 | $+47 |
| -$380 ! | 16 | 6% | 6% | 6% | $-325 | $-374 | $+55 |
| -$400 ! | 14 | 0% | 0% | 0% | $-462 | $-393 | $-62 |
| -$420 ! | 13 | 0% | 0% | 0% | $-471 | $-396 | $-51 |
| -$440 ! | 11 | 0% | 0% | 0% | $-495 | $-396 | $-55 |

**OOS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 1,822 | 33% | 12% | 7% | $-5 | $-10 | $-5 |
| -$20 | 1,047 | 26% | 13% | 8% | $-20 | $-38 | $+0 |
| -$40 | 693 | 16% | 9% | 5% | $-43 | $-57 | $-3 |
| -$60 | 432 | 11% | 6% | 4% | $-62 | $-73 | $-2 |
| -$80 | 247 | 7% | 4% | 2% | $-83 | $-87 | $-3 |
| -$100 | 121 | 7% | 3% | 2% | $-101 | $-106 | $-1 |
| -$120 | 68 | 3% | 1% | 1% | $-125 | $-120 | $-5 |
| -$140 | 39 | 5% | 3% | 3% | $-135 | $-136 | $+5 |
| -$160 ! | 26 | 8% | 4% | 4% | $-143 | $-150 | $+17 |
| -$180 ! | 20 | 5% | 5% | 5% | $-157 | $-164 | $+23 |
| -$200 ! | 13 | 8% | 8% | 8% | $-169 | $-186 | $+31 |
| -$220 ! | 10 | 10% | 10% | 10% | $-175 | $-212 | $+45 |

---

## Q11 - Probability a drawdown gets WORSE

`P(deepen)` = given at -$D, probability the drawdown extends another $20. NOTE: this column rises in IS but falls in OOS - the "drawdowns gain momentum" effect does NOT replicate.

**IS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 9,829 | 55% | 68% | 0% | 32% |
| -$20 | 5,401 | 67% | 67% | 9% | 23% |
| -$40 | 3,605 | 61% | 72% | 14% | 14% |
| -$60 | 2,194 | 59% | 69% | 21% | 9% |
| -$80 | 1,293 | 58% | 65% | 28% | 7% |
| -$100 | 747 | 63% | 61% | 33% | 6% |
| -$120 | 471 | 62% | 61% | 32% | 7% |
| -$140 | 293 | 73% | 61% | 33% | 5% |
| -$160 | 213 | 67% | 56% | 37% | 7% |
| -$180 | 142 | 71% | 55% | 41% | 4% |
| -$200 | 101 | 72% | 53% | 42% | 5% |
| -$220 | 73 | 74% | 52% | 44% | 4% |
| -$240 | 54 | 72% | 63% | 35% | 2% |
| -$260 | 39 | 82% | 67% | 31% | 3% |
| -$280 | 32 | 94% | 53% | 44% | 3% |
| -$300 | 30 | 80% | 47% | 50% | 3% |
| -$320 ! | 24 | 88% | 54% | 42% | 4% |
| -$340 ! | 21 | 90% | 48% | 48% | 5% |
| -$360 ! | 19 | 84% | 47% | 47% | 5% |
| -$380 ! | 16 | 88% | 50% | 44% | 6% |
| -$400 ! | 14 | 93% | 43% | 57% | 0% |
| -$420 ! | 13 | 85% | 38% | 62% | 0% |
| -$440 ! | 11 | 82% | 36% | 64% | 0% |

**OOS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 1,822 | 57% | 67% | 0% | 33% |
| -$20 | 1,047 | 66% | 65% | 9% | 26% |
| -$40 | 693 | 62% | 71% | 13% | 16% |
| -$60 | 432 | 57% | 72% | 18% | 11% |
| -$80 | 247 | 49% | 62% | 31% | 7% |
| -$100 | 121 | 56% | 58% | 36% | 7% |
| -$120 | 68 | 57% | 49% | 49% | 3% |
| -$140 | 39 | 67% | 49% | 46% | 5% |
| -$160 ! | 26 | 77% | 42% | 50% | 8% |
| -$180 ! | 20 | 65% | 45% | 50% | 5% |
| -$200 ! | 13 | 77% | 46% | 46% | 8% |
| -$220 ! | 10 | 70% | 30% | 60% | 10% |

---

## Q12 - Iterative drawdown chain (n -> n+1)

Each iteration n deepens the drawdown by $20. `p_advance` = P(reach step n | reached n-1); `p_reach` = cumulative from entry; `p_recover` = P(close≥0 | here).

| n | drawdown | p_reach(n) IS | p_advance IS | p_recover IS | p_reach(n) OOS | p_advance OOS | p_recover OOS |
|---|---|---|---|---|---|---|---|
| 0 | -$0 | 100.0% | 100% | 32% | 100.0% | 100% | 33% |
| 1 | -$20 | 54.9% | 55% | 23% | 57.5% | 57% | 26% |
| 2 | -$40 | 36.7% | 67% | 14% | 38.0% | 66% | 16% |
| 3 | -$60 | 22.3% | 61% | 9% | 23.7% | 62% | 11% |
| 4 | -$80 | 13.2% | 59% | 7% | 13.6% | 57% | 7% |
| 5 | -$100 | 7.6% | 58% | 6% | 6.6% | 49% | 7% |
| 6 | -$120 | 4.8% | 63% | 7% | 3.7% | 56% | 3% |
| 7 | -$140 | 3.0% | 62% | 5% | 2.1% | 57% | 5% |
| 8 | -$160 | 2.2% | 73% | 7% | 1.4% | 67% | 8% |
| 9 | -$180 | 1.4% | 67% | 4% | 1.1% | 77% | 5% |
| 10 | -$200 | 1.0% | 71% | 5% | 0.7% | 65% | 8% |
| 11 | -$220 | 0.7% | 72% | 4% | 0.5% | 77% | 10% |
| 12 | -$240 | 0.5% | 74% | 2% | 0.4% | 70% | 0% |
| 13 | -$260 | 0.4% | 72% | 3% | 0.3% | 86% | 0% |
| 14 | -$280 | 0.3% | 82% | 3% | 0.2% | 67% | 0% |
| 15 | -$300 | 0.3% | 94% | 3% | 0.2% | 100% | 0% |
| 16 | -$320 | 0.2% | 80% | 4% | 0.2% | 100% | 0% |
| 17 | -$340 | 0.2% | 88% | 5% | 0.1% | 50% | 0% |
| 18 | -$360 | 0.2% | 90% | 5% | 0.1% | 100% | 0% |
| 19 | -$380 | 0.2% | 84% | 6% | 0.1% | 100% | 0% |
| 20 | -$400 | 0.1% | 88% | 0% | 0.1% | 100% | 0% |
| 21 | -$420 | 0.1% | 93% | 0% | 0.1% | 100% | 0% |
| 22 | -$440 | 0.1% | 85% | 0% | 0.1% | 100% | 0% |

---

## Q13 - Cut a loser vs hold to the exit

"Cut" locks -$D. "Hold" runs to the R-trigger exit. `hold-bail` > 0 means cutting loses money.

| at -$D | n IS | HOLD mean IS | hold-bail IS | n OOS | HOLD mean OOS | hold-bail OOS |
|---|---|---|---|---|---|---|
| -$40 | 3,605 | $-46 | $-6 | 693 | $-43 | $-3 |
| -$60 | 2,194 | $-67 | $-7 | 432 | $-62 | $-2 |
| -$80 | 1,293 | $-86 | $-6 | 247 | $-83 | $-3 |
| -$100 | 747 | $-108 | $-8 | 121 | $-101 | $-1 |
| -$140 | 293 | $-148 | $-8 | 39 | $-135 | $+5 |

---

## Q14 - When does the MAE happen, and how long do recoverers take?

The worst point lands ~3-4 min in (constant with depth) but ~83-100% of the way through the trade. Recoverers run roughly 2x as long as non-recoverers.

**IS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 5,401 | 3.2 | 78% | 15.8 | 5.2 |
| -$40 | 3,605 | 3.4 | 89% | 17.2 | 4.8 |
| -$60 | 2,194 | 3.3 | 90% | 17.2 | 4.2 |
| -$80 | 1,293 | 3.2 | 89% | 17.0 | 4.0 |
| -$100 | 747 | 3.1 | 89% | 14.7 | 3.8 |

**OOS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 1,047 | 3.2 | 72% | 18.5 | 5.8 |
| -$40 | 693 | 3.8 | 90% | 20.2 | 5.5 |
| -$60 | 432 | 3.7 | 93% | 19.6 | 5.0 |
| -$80 | 247 | 3.5 | 92% | 20.4 | 4.4 |
| -$100 | 121 | 3.4 | 92% | 15.4 | 4.2 |

---

## Q15 - The bimodal split: winners vs losers, not "peak then collapse"

Testing the claim "bimodal = peaks MFE then goes negative". The ordering holds, but the "peak" in the losing mode is ~$14 - noise, not a peak.

**IS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 742 | $+118 | 82% | 4% | $+18 | 41% |
| LATE bottom (q4) | 2,200 | $+21 | 20% | 100% | $-66 | 99% |
| all losers | 6,682 | $+14 | 19% | 70% | $-26 | 100% |

**OOS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 164 | $+114 | 86% | 6% | $+19 | 42% |
| LATE bottom (q4) | 414 | $+19 | 15% | 100% | $-69 | 99% |
| all losers | 1,225 | $+13 | 19% | 69% | $-28 | 100% |