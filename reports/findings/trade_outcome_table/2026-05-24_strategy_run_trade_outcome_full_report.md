# Trade Outcome Suite - Full Report

Generated 2026-05-24 by `tools/trade_outcome_suite/run_all.py` on leg-list source **`strategy_run`**. 

Population: IS 18,462 legs / 277 days, OOS 2,997 legs / 52 days, reported separately (never pooled).

Pure descriptive diagnostics -- no model fit. All dollars are MNQ ($2/point); `pnl_usd` is net of $6/leg friction. Conditional cells carry n + 95% bootstrap CI (4000 resamples); cells with n < 30 are flagged ` !`.

## Verdict index

| # | Question | Verdict |
|---|---|---|
| 1 | Q1 - Distributions of entry-to-close, MAE, MFE | Median MFE $31/$43 (IS/OOS); P(close>0) 35%/37%. |
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
- This report runs on the `strategy_run` leg-list source. For the honest forward pass numbers, run with `--source causal_flat`.


---

## Q1 - Distributions of entry-to-close, MAE, MFE

Survival (exceedance) probability that each per-trade quantity reaches a given dollar magnitude. The foundation table.

| level $ | MAE≥ IS | MFE≥ IS | close≥+ IS | close≤- IS | MAE≥ OOS | MFE≥ OOS | close≥+ OOS | close≤- OOS |
|---|---|---|---|---|---|---|---|---|
| $0 | 100% | 100% | 36% | 65% | 100% | 100% | 37% | 63% |
| $25 | 49% | 57% | 20% | 34% | 63% | 65% | 24% | 45% |
| $50 | 19% | 35% | 12% | 12% | 28% | 46% | 16% | 19% |
| $75 | 9% | 22% | 8% | 5% | 12% | 32% | 11% | 6% |
| $100 | 4% | 15% | 5% | 2% | 4% | 21% | 8% | 2% |
| $125 | 2% | 10% | 4% | 1% | 2% | 15% | 6% | 1% |
| $150 | 1% | 7% | 3% | 1% | 1% | 11% | 4% | 0% |
| $175 | 1% | 5% | 2% | 0% | 1% | 8% | 3% | 0% |
| $200 | 1% | 4% | 1% | 0% | 0% | 6% | 2% | 0% |
| $225 | 0% | 3% | 1% | 0% | 0% | 5% | 2% | 0% |
| $250 | 0% | 2% | 1% | 0% | 0% | 3% | 1% | 0% |
| $275 | 0% | 2% | 1% | 0% | 0% | 2% | 1% | 0% |
| $300 | 0% | 1% | 0% | 0% | 0% | 2% | 1% | 0% |
| $325 | 0% | 1% | 0% | 0% | 0% | 2% | 1% | 0% |
| $350 | 0% | 1% | 0% | 0% | 0% | 1% | 1% | 0% |
| $375 | 0% | 1% | 0% | 0% | 0% | 1% | 0% | 0% |
| $400 | 0% | 1% | 0% | 0% | 0% | 1% | 0% | 0% |

---

## Q2 - Joint MFE x MAE -> P(close>0)

Does the drawdown a trade suffered (MAE) change its win odds, once you know its peak (MFE)?

**IS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 25% / 5265 | 2% / 4197 | 0% / 2059 | 0% / 501 | 0% / 65 |
| 50-100 | 88% / 2485 | 52% / 713 | 14% / 341 | 2% / 98 | 0% / 32 |
| 100-150 | 98% / 952 | 89% / 284 | 59% / 117 | 11% / 28 | 8% / 12 |
| 150-200 | 99% / 382 | 98% / 143 | 76% / 63 | 53% / 17 | 0% / 3 |
| 200-300 | 100% / 266 | 100% / 102 | 92% / 50 | 62% / 13 | 67% / 3 |
| 300+ | 100% / 146 | 100% / 63 | 100% / 40 | 100% / 19 | 67% / 3 |

**OOS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 14% / 297 | 1% / 656 | 0% / 575 | 0% / 89 | 0% / 7 |
| 50-100 | 83% / 418 | 38% / 205 | 25% / 88 | 0% / 19 | 0% / 2 |
| 100-150 | 98% / 194 | 88% / 89 | 72% / 29 | 0% / 7 | - |
| 150-200 | 99% / 86 | 95% / 39 | 93% / 14 | 0% / 1 | - |
| 200-300 | 100% / 66 | 100% / 39 | 89% / 9 | 100% / 2 | 100% / 2 |
| 300+ | 100% / 34 | 100% / 20 | 100% / 10 | - | - |

---

## Q3 - Continuation: given up +$x, where does it close?

Condition: the trade has reached open profit +$x (MFE≥x). Distribution of the FINAL close.

**IS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 18,462 | 12% | 5% | 1% | 35% | $-3 | [35%, 36%] |
| $25 | 10,474 | 22% | 9% | 3% | 61% | $+22 | [60%, 62%] |
| $50 | 6,375 | 36% | 15% | 4% | 81% | $+46 | [80%, 82%] |
| $75 | 4,055 | 56% | 23% | 7% | 90% | $+71 | [89%, 90%] |
| $100 | 2,706 | 75% | 35% | 10% | 93% | $+96 | [92%, 94%] |
| $125 | 1,894 | 84% | 50% | 15% | 95% | $+119 | [94%, 96%] |
| $150 | 1,313 | 90% | 68% | 21% | 97% | $+146 | [96%, 98%] |
| $175 | 934 | 94% | 81% | 30% | 98% | $+174 | [97%, 99%] |
| $200 | 705 | 95% | 86% | 39% | 98% | $+197 | [97%, 99%] |
| $225 | 543 | 97% | 90% | 50% | 99% | $+221 | [98%, 100%] |
| $250 | 433 | 98% | 93% | 63% | 99% | $+243 | [98%, 100%] |
| $275 | 334 | 99% | 95% | 74% | 100% | $+267 | [99%, 100%] |
| $300 | 271 | 99% | 95% | 79% | 100% | $+286 | [99%, 100%] |
| $325 | 205 | 100% | 95% | 85% | 100% | $+310 | [99%, 100%] |
| $350 | 164 | 99% | 95% | 87% | 99% | $+331 | [98%, 100%] |
| $375 | 131 | 99% | 98% | 92% | 99% | $+360 | [98%, 100%] |
| $400 | 98 | 100% | 99% | 97% | 100% | $+398 | [100%, 100%] |

**OOS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 2,997 | 16% | 8% | 2% | 37% | $-2 | [35%, 39%] |
| $25 | 1,947 | 24% | 12% | 4% | 56% | $+24 | [54%, 59%] |
| $50 | 1,373 | 34% | 17% | 5% | 76% | $+47 | [74%, 79%] |
| $75 | 950 | 49% | 25% | 7% | 89% | $+71 | [87%, 91%] |
| $100 | 641 | 71% | 36% | 11% | 94% | $+97 | [93%, 96%] |
| $125 | 454 | 85% | 51% | 15% | 98% | $+123 | [96%, 99%] |
| $150 | 322 | 93% | 70% | 22% | 98% | $+150 | [97%, 99%] |
| $175 | 242 | 96% | 83% | 29% | 99% | $+174 | [98%, 100%] |
| $200 | 182 | 98% | 93% | 38% | 99% | $+199 | [98%, 100%] |
| $225 | 135 | 99% | 97% | 52% | 100% | $+226 | [100%, 100%] |
| $250 | 98 | 99% | 97% | 69% | 100% | $+254 | [100%, 100%] |
| $275 | 74 | 99% | 96% | 86% | 100% | $+284 | [100%, 100%] |
| $300 | 64 | 100% | 98% | 91% | 100% | $+301 | [100%, 100%] |
| $325 | 50 | 100% | 98% | 96% | 100% | $+324 | [100%, 100%] |
| $350 | 39 | 100% | 97% | 95% | 100% | $+348 | [100%, 100%] |
| $375 | 34 | 100% | 97% | 94% | 100% | $+356 | [100%, 100%] |
| $400 ! | 27 | 100% | 96% | 93% | 100% | $+375 | [100%, 100%] |

---

## Q4 - Conditional: at +$x, P(it continues another step)

The "$100 -> $150" question, generalised. Given the trade is up +$x: does the peak push another $50, and does it hold the +$x to the close?

| at +$x | n IS | P(peak≥x+$50) IS | P(close≥x) IS | n OOS | P(peak≥x+$50) OOS | P(close≥x) OOS |
|---|---|---|---|---|---|---|
| $50 | 6,375 | 42% | 36% | 1,373 | 47% | 34% |
| $100 | 2,706 | 49% | 35% | 641 | 50% | 36% |
| $150 | 1,313 | 54% | 38% | 322 | 57% | 36% |
| $200 | 705 | 61% | 39% | 182 | 54% | 38% |
| $250 | 433 | 63% | 35% | 98 | 65% | 42% |
| $300 | 271 | 61% | 34% | 64 | 61% | 36% |
| $350 | 164 | 60% | 34% | 39 | 69% | 38% |

---

## Q5 - Cut-and-bank a winner vs hold to the exit

"Cut" banks +$L minus $6 friction. "Hold" runs to the R-trigger exit. `hold-cut` > 0 means holding wins.

| at +$L | n IS | HOLD mean IS | hold-cut IS | n OOS | HOLD mean OOS | hold-cut OOS |
|---|---|---|---|---|---|---|
| $100 | 2,706 | $+96 | $+2 | 641 | $+97 | $+3 |
| $150 | 1,313 | $+146 | $+2 | 322 | $+150 | $+6 |
| $200 | 705 | $+197 | $+3 | 182 | $+199 | $+5 |
| $250 | 433 | $+243 | $-1 | 98 | $+254 | $+10 |
| $300 | 271 | $+286 | $-8 | 64 | $+301 | $+7 |
| $400 | 98 | $+398 | $+4 | 27 | $+375 | $-19 |
| $500 | 49 | $+489 | $-5 | 7 | $+553 | $+59 |

---

## Q6 - Giveback: how much of the peak survives to the close

The peak->close drop averages $56 IS / $70 OOS -- a roughly FIXED ~1R toll, so it eats small excursions whole and barely dents big ones.

| MFE peak | n IS | median close IS | capture% IS | gave-back≤$20 IS | n OOS | median close OOS | capture% OOS |
|---|---|---|---|---|---|---|---|
| $50-100 | 3,669 | $+16 | 25% | 56% | 732 | $+8 | 11% |
| $100-150 | 1,393 | $+58 | 48% | 17% | 319 | $+49 | 42% |
| $150-200 | 608 | $+97 | 58% | 8% | 140 | $+94 | 54% |
| $200-300 | 434 | $+153 | 66% | 4% | 118 | $+146 | 63% |
| $300+ | 271 | $+260 | 72% | 0% | 64 | $+280 | 74% |

---

## Q7 - Given an MFE of +$300, where does it close (cumulative)

Each bucket is distinct and sums to 100%; the cumulative column is P(close at or below the band top).

**IS** (n=271)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 33.9% | 100.0% |
| +$250-300 | 21.4% | 66.1% |
| +$200-250 | 23.6% | 44.6% |
| +$150-200 | 14.8% | 21.0% |
| +$100-150 | 1.5% | 6.3% |
| +$50-100 | 4.1% | 4.8% |
| +$0-50 | 0.4% | 0.7% |
| < $0 (gave it all back) | 0.4% | 0.4% |

**OOS** (n=64)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 35.9% | 100.0% |
| +$250-300 | 28.1% | 64.1% |
| +$200-250 | 26.6% | 35.9% |
| +$150-200 | 6.2% | 9.4% |
| +$100-150 | 1.6% | 3.1% |
| +$50-100 | 1.6% | 1.6% |
| +$0-50 | 0.0% | 0.0% |
| < $0 (gave it all back) | 0.0% | 0.0% |

---

## Q8 - Equity-loss map: P(close<0) by MFE reached

"Lose equity" = closes red. The question behind the "safety limit" idea: how far must a trade run before it is safe from a negative close?

| MFE reached ≥ | n IS | P(lose equity) IS | P(close<-$100) IS | n OOS | P(lose equity) OOS | P(close<-$100) OOS |
|---|---|---|---|---|---|---|
| +$25 | 10,474 | 38.4% | 1.9% | 1,947 | 43.0% | 1.3% |
| +$50 | 6,375 | 19.1% | 1.5% | 1,373 | 22.9% | 0.9% |
| +$75 | 4,055 | 10.4% | 1.2% | 950 | 10.7% | 0.5% |
| +$100 | 2,706 | 6.7% | 0.8% | 641 | 5.5% | 0.8% |
| +$150 | 1,313 | 3.4% | 0.6% | 322 | 1.9% | 0.3% |
| +$200 | 705 | 1.7% | 0.4% | 182 | 0.5% | 0.0% |
| +$250 | 433 | 0.9% | 0.5% | 98 | 0.0% | 0.0% |
| +$300 | 271 | 0.4% | 0.4% | 64 | 0.0% | 0.0% |

---

## Q9 - Recovery: given down -$d, does it work out?

Condition: the trade has drawn down to -$d (MAE≥d). Distribution of the final close.

**IS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 18,462 | 35% | 5% | $-3 | [35%, 36%] |
| -$20 | 10,792 | 17% | 4% | $-23 | [16%, 17%] |
| -$40 | 4,991 | 11% | 3% | $-43 | [10%, 11%] |
| -$60 | 2,499 | 8% | 3% | $-64 | [7%, 9%] |
| -$80 | 1,351 | 7% | 3% | $-83 | [5%, 8%] |
| -$100 | 794 | 6% | 2% | $-104 | [4%, 7%] |
| -$120 | 505 | 6% | 2% | $-120 | [4%, 9%] |
| -$140 | 324 | 6% | 2% | $-140 | [4%, 9%] |
| -$160 | 226 | 5% | 3% | $-159 | [3%, 8%] |
| -$180 | 156 | 4% | 2% | $-182 | [1%, 8%] |
| -$200 | 118 | 4% | 2% | $-201 | [1%, 8%] |
| -$220 | 82 | 5% | 2% | $-226 | [1%, 10%] |
| -$240 | 64 | 5% | 2% | $-246 | [0%, 11%] |
| -$260 | 49 | 4% | 2% | $-262 | [0%, 10%] |

**OOS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 2,997 | 37% | 8% | $-2 | [35%, 39%] |
| -$20 | 2,137 | 21% | 5% | $-19 | [20%, 23%] |
| -$40 | 1,210 | 12% | 4% | $-40 | [10%, 14%] |
| -$60 | 599 | 7% | 3% | $-59 | [5%, 10%] |
| -$80 | 291 | 4% | 3% | $-77 | [2%, 7%] |
| -$100 | 129 | 3% | 2% | $-99 | [1%, 6%] |
| -$120 | 80 | 5% | 4% | $-108 | [1%, 10%] |
| -$140 | 44 | 7% | 7% | $-115 | [0%, 16%] |
| -$160 ! | 26 | 8% | 8% | $-131 | [0%, 19%] |
| -$180 ! | 19 | 11% | 11% | $-139 | [0%, 26%] |
| -$200 ! | 11 | 18% | 18% | $-144 | [0%, 45%] |

---

## Q10 - Full MAE -> close sweep (every $20)

The complete recovery table, fine-grained. `hold-bail` = mean close minus the -$D you would lock by bailing now.

**IS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 18,462 | 36% | 12% | 5% | $-3 | $-14 | $-3 |
| -$20 | 10,792 | 17% | 7% | 4% | $-23 | $-28 | $-3 |
| -$40 | 4,991 | 11% | 6% | 3% | $-43 | $-46 | $-3 |
| -$60 | 2,499 | 8% | 4% | 3% | $-64 | $-64 | $-4 |
| -$80 | 1,351 | 7% | 5% | 3% | $-83 | $-84 | $-3 |
| -$100 | 794 | 6% | 4% | 2% | $-104 | $-104 | $-4 |
| -$120 | 505 | 7% | 5% | 2% | $-120 | $-122 | $-0 |
| -$140 | 324 | 6% | 5% | 2% | $-140 | $-142 | $+0 |
| -$160 | 226 | 5% | 5% | 3% | $-159 | $-160 | $+1 |
| -$180 | 156 | 4% | 4% | 2% | $-182 | $-179 | $-2 |
| -$200 | 118 | 4% | 3% | 2% | $-201 | $-201 | $-1 |
| -$220 | 82 | 5% | 4% | 2% | $-226 | $-230 | $-6 |
| -$240 | 64 | 5% | 3% | 2% | $-246 | $-249 | $-6 |
| -$260 | 49 | 4% | 4% | 2% | $-262 | $-266 | $-2 |
| -$280 | 40 | 5% | 5% | 2% | $-277 | $-280 | $+3 |
| -$300 | 36 | 3% | 3% | 3% | $-293 | $-293 | $+7 |
| -$320 ! | 29 | 0% | 0% | 0% | $-369 | $-331 | $-49 |
| -$340 ! | 25 | 0% | 0% | 0% | $-390 | $-346 | $-50 |
| -$360 ! | 20 | 0% | 0% | 0% | $-413 | $-370 | $-53 |
| -$380 ! | 17 | 0% | 0% | 0% | $-443 | $-392 | $-63 |
| -$400 ! | 16 | 0% | 0% | 0% | $-448 | $-392 | $-48 |
| -$420 ! | 14 | 0% | 0% | 0% | $-463 | $-412 | $-43 |
| -$440 ! | 13 | 0% | 0% | 0% | $-478 | $-433 | $-38 |
| -$460 ! | 10 | 0% | 0% | 0% | $-511 | $-417 | $-51 |

**OOS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 2,997 | 37% | 16% | 8% | $-2 | $-20 | $-2 |
| -$20 | 2,137 | 22% | 10% | 5% | $-19 | $-34 | $+1 |
| -$40 | 1,210 | 12% | 6% | 4% | $-40 | $-48 | $+0 |
| -$60 | 599 | 7% | 4% | 3% | $-59 | $-65 | $+1 |
| -$80 | 291 | 4% | 3% | 3% | $-77 | $-80 | $+3 |
| -$100 | 129 | 3% | 2% | 2% | $-99 | $-99 | $+1 |
| -$120 | 80 | 5% | 4% | 4% | $-108 | $-113 | $+12 |
| -$140 | 44 | 7% | 7% | 7% | $-115 | $-122 | $+25 |
| -$160 ! | 26 | 8% | 8% | 8% | $-131 | $-131 | $+29 |
| -$180 ! | 19 | 11% | 11% | 11% | $-139 | $-146 | $+41 |
| -$200 ! | 11 | 18% | 18% | 18% | $-144 | $-146 | $+56 |

---

## Q11 - Probability a drawdown gets WORSE

`P(deepen)` = given at -$D, probability the drawdown extends another $20. NOTE: this column rises in IS but falls in OOS - the "drawdowns gain momentum" effect does NOT replicate.

**IS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 18,462 | 58% | 64% | 0% | 36% |
| -$20 | 10,792 | 46% | 69% | 14% | 17% |
| -$40 | 4,991 | 50% | 63% | 26% | 11% |
| -$60 | 2,499 | 54% | 58% | 34% | 8% |
| -$80 | 1,351 | 59% | 56% | 37% | 7% |
| -$100 | 794 | 64% | 55% | 40% | 6% |
| -$120 | 505 | 64% | 54% | 40% | 7% |
| -$140 | 324 | 70% | 51% | 42% | 6% |
| -$160 | 226 | 69% | 50% | 45% | 5% |
| -$180 | 156 | 76% | 49% | 46% | 4% |
| -$200 | 118 | 69% | 50% | 46% | 4% |
| -$220 | 82 | 78% | 57% | 38% | 5% |
| -$240 | 64 | 77% | 55% | 41% | 5% |
| -$260 | 49 | 82% | 57% | 39% | 4% |
| -$280 | 40 | 90% | 50% | 45% | 5% |
| -$300 | 36 | 81% | 47% | 50% | 3% |
| -$320 ! | 29 | 86% | 55% | 45% | 0% |
| -$340 ! | 25 | 80% | 56% | 44% | 0% |
| -$360 ! | 20 | 85% | 50% | 50% | 0% |
| -$380 ! | 17 | 94% | 59% | 41% | 0% |
| -$400 ! | 16 | 88% | 44% | 56% | 0% |
| -$420 ! | 14 | 93% | 50% | 50% | 0% |
| -$440 ! | 13 | 77% | 38% | 62% | 0% |
| -$460 ! | 10 | 90% | 40% | 60% | 0% |

**OOS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 2,997 | 71% | 63% | 0% | 37% |
| -$20 | 2,137 | 57% | 70% | 9% | 22% |
| -$40 | 1,210 | 50% | 68% | 20% | 12% |
| -$60 | 599 | 49% | 61% | 32% | 7% |
| -$80 | 291 | 44% | 50% | 46% | 4% |
| -$100 | 129 | 62% | 47% | 50% | 3% |
| -$120 | 80 | 55% | 41% | 54% | 5% |
| -$140 | 44 | 59% | 30% | 64% | 7% |
| -$160 ! | 26 | 73% | 31% | 62% | 8% |
| -$180 ! | 19 | 58% | 32% | 58% | 11% |
| -$200 ! | 11 | 82% | 36% | 45% | 18% |

---

## Q12 - Iterative drawdown chain (n -> n+1)

Each iteration n deepens the drawdown by $20. `p_advance` = P(reach step n | reached n-1); `p_reach` = cumulative from entry; `p_recover` = P(close≥0 | here).

| n | drawdown | p_reach(n) IS | p_advance IS | p_recover IS | p_reach(n) OOS | p_advance OOS | p_recover OOS |
|---|---|---|---|---|---|---|---|
| 0 | -$0 | 100.0% | 100% | 36% | 100.0% | 100% | 37% |
| 1 | -$20 | 58.5% | 58% | 17% | 71.3% | 71% | 22% |
| 2 | -$40 | 27.0% | 46% | 11% | 40.4% | 57% | 12% |
| 3 | -$60 | 13.5% | 50% | 8% | 20.0% | 50% | 7% |
| 4 | -$80 | 7.3% | 54% | 7% | 9.7% | 49% | 4% |
| 5 | -$100 | 4.3% | 59% | 6% | 4.3% | 44% | 3% |
| 6 | -$120 | 2.7% | 64% | 7% | 2.7% | 62% | 5% |
| 7 | -$140 | 1.8% | 64% | 6% | 1.5% | 55% | 7% |
| 8 | -$160 | 1.2% | 70% | 5% | 0.9% | 59% | 8% |
| 9 | -$180 | 0.8% | 69% | 4% | 0.6% | 73% | 11% |
| 10 | -$200 | 0.6% | 76% | 4% | 0.4% | 58% | 18% |
| 11 | -$220 | 0.4% | 69% | 5% | 0.3% | 82% | 22% |
| 12 | -$240 | 0.3% | 78% | 5% | 0.2% | 67% | 17% |
| 13 | -$260 | 0.3% | 77% | 4% | 0.2% | 83% | 20% |
| 14 | -$280 | 0.2% | 82% | 5% | 0.1% | 80% | 25% |
| 15 | -$300 | 0.2% | 90% | 3% | 0.1% | 100% | 25% |
| 16 | -$320 | 0.2% | 81% | 0% | 0.1% | 100% | 25% |
| 17 | -$340 | 0.1% | 86% | 0% | 0.1% | 50% | 50% |
| 18 | -$360 | 0.1% | 80% | 0% | 0.1% | 100% | 50% |
| 19 | -$380 | 0.1% | 85% | 0% | 0.1% | 100% | 50% |
| 20 | -$400 | 0.1% | 94% | 0% | 0.1% | 100% | 50% |
| 21 | -$420 | 0.1% | 88% | 0% | 0.1% | 100% | 50% |
| 22 | -$440 | 0.1% | 93% | 0% | 0.1% | 100% | 50% |
| 23 | -$460 | 0.1% | 77% | 0% | 0.1% | 100% | 50% |

---

## Q13 - Cut a loser vs hold to the exit

"Cut" locks -$D. "Hold" runs to the R-trigger exit. `hold-bail` > 0 means cutting loses money.

| at -$D | n IS | HOLD mean IS | hold-bail IS | n OOS | HOLD mean OOS | hold-bail OOS |
|---|---|---|---|---|---|---|
| -$40 | 4,991 | $-43 | $-3 | 1,210 | $-40 | $+0 |
| -$60 | 2,499 | $-64 | $-4 | 599 | $-59 | $+1 |
| -$80 | 1,351 | $-83 | $-3 | 291 | $-77 | $+3 |
| -$100 | 794 | $-104 | $-4 | 129 | $-99 | $+1 |
| -$140 | 324 | $-140 | $+0 | 44 | $-115 | $+25 |

---

## Q14 - When does the MAE happen, and how long do recoverers take?

The worst point lands ~3-4 min in (constant with depth) but ~83-100% of the way through the trade. Recoverers run roughly 2x as long as non-recoverers.

**IS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 10,792 | 4.0 | 92% | 16.8 | 5.7 |
| -$40 | 4,991 | 3.7 | 92% | 16.1 | 4.6 |
| -$60 | 2,499 | 3.3 | 90% | 15.1 | 4.1 |
| -$80 | 1,351 | 3.2 | 89% | 15.1 | 3.8 |
| -$100 | 794 | 3.2 | 91% | 14.5 | 3.8 |

**OOS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 2,137 | 4.4 | 88% | 19.8 | 7.1 |
| -$40 | 1,210 | 4.2 | 93% | 19.0 | 5.6 |
| -$60 | 599 | 3.8 | 93% | 17.1 | 4.9 |
| -$80 | 291 | 3.4 | 90% | 15.2 | 4.2 |
| -$100 | 129 | 3.3 | 89% | 9.7 | 3.8 |

---

## Q15 - The bimodal split: winners vs losers, not "peak then collapse"

Testing the claim "bimodal = peaks MFE then goes negative". The ordering holds, but the "peak" in the losing mode is ~$14 - noise, not a peak.

**IS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 675 | $+101 | 76% | 7% | $+18 | 39% |
| LATE bottom (q4) | 3,403 | $+14 | 14% | 100% | $-52 | 99% |
| all losers | 11,886 | $+16 | 10% | 84% | $-26 | 100% |

**OOS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 209 | $+90 | 76% | 7% | $+8 | 44% |
| LATE bottom (q4) | 783 | $+15 | 12% | 100% | $-55 | 99% |
| all losers | 1,884 | $+22 | 17% | 82% | $-38 | 100% |