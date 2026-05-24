**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**

# Trade Outcome Suite - Full Report

Generated 2026-05-22 by `tools/trade_outcome_suite/run_all.py`. Population: every hardened zigzag-leg trade -- IS 17,767 legs / 275 days, OOS 2,936 legs / 51 days, reported separately (never pooled).

Pure descriptive diagnostics -- no model fit, no leakage, no production code touched. All dollars are MNQ ($2/point); `pnl_usd` is net of $6/leg friction. Conditional cells carry n + 95% bootstrap CI (4000 resamples); cells with n < 30 are flagged ` !`.

## Verdict index

| # | Question | Verdict |
|---|---|---|
| 1 | Q1 - Distributions of entry-to-close, MAE, MFE | Median MFE $40/$51 (IS/OOS); P(close>0) 43%/41%. |
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


---

## Q1 - Distributions of entry-to-close, MAE, MFE

Survival (exceedance) probability that each per-trade quantity reaches a given dollar magnitude. The foundation table.

| level $ | MAE≥ IS | MFE≥ IS | close≥+ IS | close≤- IS | MAE≥ OOS | MFE≥ OOS | close≥+ OOS | close≤- OOS |
|---|---|---|---|---|---|---|---|---|
| $0 | 100% | 100% | 43% | 57% | 100% | 100% | 42% | 59% |
| $25 | 38% | 66% | 26% | 28% | 59% | 72% | 29% | 41% |
| $50 | 10% | 42% | 17% | 7% | 20% | 51% | 20% | 15% |
| $75 | 3% | 27% | 11% | 2% | 6% | 36% | 14% | 4% |
| $100 | 1% | 19% | 8% | 1% | 1% | 25% | 9% | 1% |
| $125 | 1% | 13% | 5% | 0% | 1% | 18% | 7% | 0% |
| $150 | 0% | 9% | 4% | 0% | 0% | 13% | 5% | 0% |
| $175 | 0% | 7% | 3% | 0% | 0% | 10% | 4% | 0% |
| $200 | 0% | 5% | 2% | 0% | 0% | 7% | 3% | 0% |
| $225 | 0% | 4% | 2% | 0% | 0% | 5% | 2% | 0% |
| $250 | 0% | 3% | 1% | 0% | 0% | 4% | 2% | 0% |
| $275 | 0% | 2% | 1% | 0% | 0% | 3% | 1% | 0% |
| $300 | 0% | 2% | 1% | 0% | 0% | 2% | 1% | 0% |
| $325 | 0% | 2% | 1% | 0% | 0% | 2% | 1% | 0% |
| $350 | 0% | 1% | 1% | 0% | 0% | 1% | 1% | 0% |
| $375 | 0% | 1% | 0% | 0% | 0% | 1% | 1% | 0% |
| $400 | 0% | 1% | 0% | 0% | 0% | 1% | 1% | 0% |

---

## Q2 - Joint MFE x MAE -> P(close>0)

Does the drawdown a trade suffered (MAE) change its win odds, once you know its peak (MFE)?

**IS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 18% / 6097 | 2% / 3291 | 1% / 865 | 1% / 112 | 0% / 11 |
| 50-100 | 87% / 2902 | 67% / 881 | 32% / 271 | 6% / 36 | 0% / 8 |
| 100-150 | 98% / 1066 | 95% / 402 | 75% / 150 | 44% / 27 | 0% / 1 |
| 150-200 | 100% / 438 | 97% / 228 | 87% / 70 | 88% / 16 | 0% / 2 |
| 200-300 | 100% / 292 | 99% / 164 | 98% / 66 | 75% / 20 | 100% / 1 |
| 300+ | 100% / 132 | 100% / 117 | 99% / 72 | 96% / 25 | 100% / 4 |

**OOS** (cell = P(close>0) / n)

| MFE \ MAE | 0-25 | 25-50 | 50-100 | 100-200 | 200+ |
|---|---|---|---|---|---|
| 0-50 | 8% / 374 | 1% / 677 | 0% / 369 | 0% / 23 | 0% / 3 |
| 50-100 | 75% / 423 | 45% / 240 | 38% / 80 | 0% / 4 | - |
| 100-150 | 97% / 215 | 94% / 109 | 84% / 31 | 67% / 6 | - |
| 150-200 | 100% / 97 | 100% / 52 | 100% / 29 | - | - |
| 200-300 | 100% / 60 | 98% / 47 | 100% / 23 | 100% / 2 | 100% / 1 |
| 300+ | 100% / 30 | 100% / 28 | 100% / 12 | 0% / 1 | - |

---

## Q3 - Continuation: given up +$x, where does it close?

Condition: the trade has reached open profit +$x (MFE≥x). Distribution of the FINAL close.

**IS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 17,767 | 17% | 8% | 2% | 43% | $+11 | [42%, 43%] |
| $25 | 11,712 | 25% | 12% | 4% | 65% | $+33 | [64%, 65%] |
| $50 | 7,391 | 40% | 19% | 6% | 86% | $+59 | [86%, 87%] |
| $75 | 4,820 | 62% | 29% | 9% | 94% | $+86 | [94%, 95%] |
| $100 | 3,293 | 83% | 42% | 13% | 96% | $+113 | [96%, 97%] |
| $125 | 2,319 | 92% | 60% | 19% | 98% | $+141 | [97%, 98%] |
| $150 | 1,647 | 95% | 79% | 26% | 98% | $+169 | [97%, 99%] |
| $175 | 1,205 | 97% | 90% | 36% | 99% | $+198 | [98%, 100%] |
| $200 | 893 | 98% | 94% | 48% | 99% | $+226 | [98%, 100%] |
| $225 | 673 | 98% | 96% | 64% | 99% | $+256 | [98%, 100%] |
| $250 | 528 | 99% | 97% | 79% | 100% | $+284 | [99%, 100%] |
| $275 | 427 | 99% | 98% | 88% | 100% | $+307 | [99%, 100%] |
| $300 | 350 | 99% | 98% | 90% | 99% | $+328 | [99%, 100%] |
| $325 | 282 | 99% | 98% | 94% | 99% | $+355 | [98%, 100%] |
| $350 | 224 | 99% | 99% | 97% | 100% | $+386 | [99%, 100%] |
| $375 | 183 | 99% | 99% | 97% | 99% | $+413 | [98%, 100%] |
| $400 | 143 | 99% | 99% | 99% | 99% | $+448 | [98%, 100%] |

**OOS**

| up +$x | n | close≥$50 | close≥$100 | close≥$200 | close>0 | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|---|---|
| $0 | 2,936 | 20% | 9% | 3% | 41% | $+8 | [40%, 43%] |
| $25 | 2,110 | 28% | 13% | 4% | 58% | $+30 | [56%, 60%] |
| $50 | 1,490 | 39% | 18% | 5% | 79% | $+54 | [77%, 81%] |
| $75 | 1,045 | 56% | 26% | 7% | 93% | $+80 | [91%, 94%] |
| $100 | 743 | 77% | 37% | 10% | 97% | $+105 | [96%, 98%] |
| $125 | 530 | 92% | 51% | 14% | 99% | $+131 | [98%, 100%] |
| $150 | 382 | 98% | 70% | 20% | 99% | $+157 | [99%, 100%] |
| $175 | 280 | 99% | 84% | 27% | 99% | $+182 | [98%, 100%] |
| $200 | 204 | 99% | 94% | 37% | 99% | $+210 | [98%, 100%] |
| $225 | 152 | 99% | 97% | 50% | 99% | $+238 | [98%, 100%] |
| $250 | 106 | 99% | 97% | 71% | 99% | $+275 | [97%, 100%] |
| $275 | 79 | 99% | 99% | 90% | 99% | $+311 | [96%, 100%] |
| $300 | 71 | 99% | 99% | 94% | 99% | $+323 | [96%, 100%] |
| $325 | 55 | 98% | 98% | 98% | 98% | $+351 | [95%, 100%] |
| $350 | 44 | 98% | 98% | 98% | 98% | $+380 | [93%, 100%] |
| $375 | 37 | 97% | 97% | 97% | 97% | $+400 | [92%, 100%] |
| $400 ! | 29 | 97% | 97% | 97% | 97% | $+429 | [90%, 100%] |

---

## Q4 - Conditional: at +$x, P(it continues another step)

The "$100 -> $150" question, generalised. Given the trade is up +$x: does the peak push another $50, and does it hold the +$x to the close?

| at +$x | n IS | P(peak≥x+$50) IS | P(close≥x) IS | n OOS | P(peak≥x+$50) OOS | P(close≥x) OOS |
|---|---|---|---|---|---|---|
| $50 | 7,391 | 45% | 40% | 1,490 | 50% | 39% |
| $100 | 3,293 | 50% | 42% | 743 | 51% | 37% |
| $150 | 1,647 | 54% | 44% | 382 | 53% | 37% |
| $200 | 893 | 59% | 48% | 204 | 52% | 37% |
| $250 | 528 | 66% | 49% | 106 | 67% | 45% |
| $300 | 350 | 64% | 44% | 71 | 62% | 42% |
| $350 | 224 | 64% | 44% | 44 | 66% | 50% |

---

## Q5 - Cut-and-bank a winner vs hold to the exit

"Cut" banks +$L minus $6 friction. "Hold" runs to the R-trigger exit. `hold-cut` > 0 means holding wins.

| at +$L | n IS | HOLD mean IS | hold-cut IS | n OOS | HOLD mean OOS | hold-cut OOS |
|---|---|---|---|---|---|---|
| $100 | 3,293 | $+113 | $+19 | 743 | $+105 | $+11 |
| $150 | 1,647 | $+169 | $+25 | 382 | $+157 | $+13 |
| $200 | 893 | $+226 | $+32 | 204 | $+210 | $+16 |
| $250 | 528 | $+284 | $+40 | 106 | $+275 | $+31 |
| $300 | 350 | $+328 | $+34 | 71 | $+323 | $+29 |
| $400 | 143 | $+448 | $+54 | 29 | $+429 | $+35 |
| $500 | 65 | $+587 | $+93 | 12 | $+513 | $+19 |

---

## Q6 - Giveback: how much of the peak survives to the close

The peak->close drop averages $53 IS / $68 OOS -- a roughly FIXED ~1R toll, so it eats small excursions whole and barely dents big ones.

| MFE peak | n IS | median close IS | capture% IS | gave-back≤$20 IS | n OOS | median close OOS | capture% OOS |
|---|---|---|---|---|---|---|---|
| $50-100 | 4,098 | $+19 | 28% | 53% | 747 | $+8 | 10% |
| $100-150 | 1,646 | $+64 | 54% | 10% | 361 | $+52 | 43% |
| $150-200 | 754 | $+109 | 64% | 5% | 178 | $+97 | 56% |
| $200-300 | 543 | $+162 | 71% | 2% | 133 | $+151 | 65% |
| $300+ | 350 | $+284 | 78% | 1% | 71 | $+279 | 77% |

---

## Q7 - Given an MFE of +$300, where does it close (cumulative)

Each bucket is distinct and sums to 100%; the cumulative column is P(close at or below the band top).

**IS** (n=350)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 43.7% | 100.0% |
| +$250-300 | 29.1% | 56.3% |
| +$200-250 | 16.9% | 27.1% |
| +$150-200 | 6.6% | 10.3% |
| +$100-150 | 1.7% | 3.7% |
| +$50-100 | 0.6% | 2.0% |
| +$0-50 | 0.9% | 1.4% |
| < $0 (gave it all back) | 0.6% | 0.6% |

**OOS** (n=71)

| close lands in | P(bucket) | cum P(close≤top) |
|---|---|---|
| ≥ +$300 (ran further) | 42.3% | 100.0% |
| +$250-300 | 25.4% | 57.7% |
| +$200-250 | 26.8% | 32.4% |
| +$150-200 | 4.2% | 5.6% |
| +$100-150 | 0.0% | 1.4% |
| +$50-100 | 0.0% | 1.4% |
| +$0-50 | 0.0% | 1.4% |
| < $0 (gave it all back) | 1.4% | 1.4% |

---

## Q8 - Equity-loss map: P(close<0) by MFE reached

"Lose equity" = closes red. The question behind the "safety limit" idea: how far must a trade run before it is safe from a negative close?

| MFE reached ≥ | n IS | P(lose equity) IS | P(close<-$100) IS | n OOS | P(lose equity) OOS | P(close<-$100) OOS |
|---|---|---|---|---|---|---|
| +$25 | 11,712 | 34.8% | 0.6% | 2,110 | 41.8% | 0.3% |
| +$50 | 7,391 | 13.2% | 0.4% | 1,490 | 20.5% | 0.1% |
| +$75 | 4,820 | 5.7% | 0.4% | 1,045 | 7.2% | 0.2% |
| +$100 | 3,293 | 3.6% | 0.3% | 743 | 2.7% | 0.1% |
| +$150 | 1,647 | 1.8% | 0.2% | 382 | 0.5% | 0.3% |
| +$200 | 893 | 1.0% | 0.3% | 204 | 1.0% | 0.5% |
| +$250 | 528 | 0.4% | 0.0% | 106 | 0.9% | 0.9% |
| +$300 | 350 | 0.6% | 0.0% | 71 | 1.4% | 1.4% |

---

## Q9 - Recovery: given down -$d, does it work out?

Condition: the trade has drawn down to -$d (MAE≥d). Distribution of the final close.

**IS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 17,767 | 43% | 8% | $+11 | [42%, 43%] |
| -$20 | 9,004 | 31% | 8% | $+1 | [30%, 32%] |
| -$40 | 2,934 | 28% | 10% | $-8 | [27%, 30%] |
| -$60 | 1,070 | 28% | 11% | $-15 | [25%, 30%] |
| -$80 | 478 | 29% | 14% | $-17 | [25%, 33%] |
| -$100 | 263 | 28% | 14% | $-24 | [23%, 33%] |
| -$120 | 150 | 24% | 13% | $-32 | [17%, 31%] |
| -$140 | 99 | 22% | 10% | $-53 | [14%, 31%] |
| -$160 | 64 | 20% | 9% | $-60 | [11%, 31%] |
| -$180 | 40 | 18% | 10% | $-56 | [8%, 30%] |
| -$200 ! | 27 | 19% | 7% | $-50 | [4%, 33%] |
| -$220 ! | 19 | 26% | 11% | $-14 | [11%, 47%] |
| -$240 ! | 11 | 27% | 9% | $+25 | [0%, 55%] |

**OOS**

| down -$d | n | P(close>0) | P(close≥+$100) | mean close | P(close>0) 95% CI |
|---|---|---|---|---|---|
| -$0 | 2,936 | 41% | 9% | $+8 | [40%, 43%] |
| -$20 | 2,017 | 30% | 8% | $-6 | [28%, 32%] |
| -$40 | 945 | 22% | 7% | $-22 | [20%, 25%] |
| -$60 | 380 | 21% | 7% | $-34 | [17%, 24%] |
| -$80 | 128 | 19% | 7% | $-43 | [12%, 26%] |
| -$100 | 40 | 18% | 2% | $-72 | [8%, 30%] |
| -$120 ! | 26 | 19% | 4% | $-73 | [4%, 35%] |
| -$140 ! | 17 | 24% | 6% | $-65 | [6%, 47%] |

---

## Q10 - Full MAE -> close sweep (every $20)

The complete recovery table, fine-grained. `hold-bail` = mean close minus the -$D you would lock by bailing now.

**IS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 17,767 | 43% | 17% | 8% | $+11 | $-7 | $+11 |
| -$20 | 9,004 | 31% | 16% | 8% | $+1 | $-26 | $+21 |
| -$40 | 2,934 | 29% | 16% | 10% | $-8 | $-46 | $+32 |
| -$60 | 1,070 | 28% | 17% | 11% | $-15 | $-62 | $+45 |
| -$80 | 478 | 30% | 21% | 14% | $-17 | $-78 | $+63 |
| -$100 | 263 | 28% | 22% | 14% | $-24 | $-98 | $+76 |
| -$120 | 150 | 25% | 21% | 13% | $-32 | $-102 | $+88 |
| -$140 | 99 | 23% | 18% | 10% | $-53 | $-130 | $+87 |
| -$160 | 64 | 20% | 17% | 9% | $-60 | $-136 | $+100 |
| -$180 | 40 | 18% | 18% | 10% | $-56 | $-143 | $+124 |
| -$200 ! | 27 | 19% | 19% | 7% | $-50 | $-144 | $+150 |
| -$220 ! | 19 | 26% | 26% | 11% | $-14 | $-144 | $+206 |
| -$240 ! | 11 | 27% | 27% | 9% | $+25 | $-250 | $+265 |

**OOS**

| down -$D | n | P(≥0) recover | P(≥+$50) | P(≥+$100) | mean close | median close | hold-bail |
|---|---|---|---|---|---|---|---|
| -$0 | 2,936 | 42% | 20% | 9% | $+8 | $-13 | $+8 |
| -$20 | 2,017 | 30% | 16% | 8% | $-6 | $-31 | $+14 |
| -$40 | 945 | 22% | 13% | 7% | $-22 | $-48 | $+18 |
| -$60 | 380 | 21% | 12% | 7% | $-34 | $-67 | $+26 |
| -$80 | 128 | 19% | 11% | 7% | $-43 | $-84 | $+37 |
| -$100 | 40 | 18% | 8% | 2% | $-72 | $-91 | $+28 |
| -$120 ! | 26 | 19% | 12% | 4% | $-73 | $-104 | $+47 |
| -$140 ! | 17 | 24% | 12% | 6% | $-65 | $-64 | $+75 |

---

## Q11 - Probability a drawdown gets WORSE

`P(deepen)` = given at -$D, probability the drawdown extends another $20. NOTE: this column rises in IS but falls in OOS - the "drawdowns gain momentum" effect does NOT replicate.

**IS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 17,767 | 51% | 57% | 0% | 43% |
| -$20 | 9,004 | 33% | 57% | 12% | 31% |
| -$40 | 2,934 | 36% | 56% | 15% | 29% |
| -$60 | 1,070 | 45% | 52% | 20% | 28% |
| -$80 | 478 | 55% | 49% | 22% | 30% |
| -$100 | 263 | 57% | 47% | 25% | 28% |
| -$120 | 150 | 66% | 47% | 28% | 25% |
| -$140 | 99 | 65% | 46% | 30% | 23% |
| -$160 | 64 | 62% | 38% | 42% | 20% |
| -$180 | 40 | 68% | 38% | 45% | 18% |
| -$200 ! | 27 | 70% | 41% | 41% | 19% |
| -$220 ! | 19 | 58% | 42% | 32% | 26% |
| -$240 ! | 11 | 73% | 55% | 18% | 27% |

**OOS**

| at -$D | n | P(deepen +$20) | P(close<-$D worse) | P(stuck -D..0) | P(recover ≥0) |
|---|---|---|---|---|---|
| -$0 | 2,936 | 69% | 58% | 0% | 42% |
| -$20 | 2,017 | 47% | 62% | 9% | 30% |
| -$40 | 945 | 40% | 64% | 13% | 22% |
| -$60 | 380 | 34% | 62% | 18% | 21% |
| -$80 | 128 | 31% | 56% | 25% | 19% |
| -$100 | 40 | 65% | 45% | 38% | 18% |
| -$120 ! | 26 | 65% | 38% | 42% | 19% |
| -$140 ! | 17 | 53% | 18% | 59% | 24% |

---

## Q12 - Iterative drawdown chain (n -> n+1)

Each iteration n deepens the drawdown by $20. `p_advance` = P(reach step n | reached n-1); `p_reach` = cumulative from entry; `p_recover` = P(close≥0 | here).

| n | drawdown | p_reach(n) IS | p_advance IS | p_recover IS | p_reach(n) OOS | p_advance OOS | p_recover OOS |
|---|---|---|---|---|---|---|---|
| 0 | -$0 | 100.0% | 100% | 43% | 100.0% | 100% | 42% |
| 1 | -$20 | 50.7% | 51% | 31% | 68.7% | 69% | 30% |
| 2 | -$40 | 16.5% | 33% | 29% | 32.2% | 47% | 22% |
| 3 | -$60 | 6.0% | 36% | 28% | 12.9% | 40% | 21% |
| 4 | -$80 | 2.7% | 45% | 30% | 4.4% | 34% | 19% |
| 5 | -$100 | 1.5% | 55% | 28% | 1.4% | 31% | 18% |
| 6 | -$120 | 0.8% | 57% | 25% | 0.9% | 65% | 19% |
| 7 | -$140 | 0.6% | 66% | 23% | 0.6% | 65% | 24% |
| 8 | -$160 | 0.4% | 65% | 20% | 0.3% | 53% | 33% |
| 9 | -$180 | 0.2% | 62% | 18% | 0.2% | 56% | 20% |
| 10 | -$200 | 0.2% | 68% | 19% | 0.1% | 80% | 25% |
| 11 | -$220 | 0.1% | 70% | 26% | 0.1% | 75% | 0% |
| 12 | -$240 | 0.1% | 58% | 27% | 0.1% | 100% | 0% |

---

## Q13 - Cut a loser vs hold to the exit

"Cut" locks -$D. "Hold" runs to the R-trigger exit. `hold-bail` > 0 means cutting loses money.

| at -$D | n IS | HOLD mean IS | hold-bail IS | n OOS | HOLD mean OOS | hold-bail OOS |
|---|---|---|---|---|---|---|
| -$40 | 2,934 | $-8 | $+32 | 945 | $-22 | $+18 |
| -$60 | 1,070 | $-15 | $+45 | 380 | $-34 | $+26 |
| -$80 | 478 | $-17 | $+63 | 128 | $-43 | $+37 |
| -$100 | 263 | $-24 | $+76 | 40 | $-72 | $+28 |
| -$140 | 99 | $-53 | $+87 | 17 | $-65 | $+75 |

---

## Q14 - When does the MAE happen, and how long do recoverers take?

The worst point lands ~3-4 min in (constant with depth) but ~83-100% of the way through the trade. Recoverers run roughly 2x as long as non-recoverers.

**IS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 9,004 | 3.3 | 57% | 11.8 | 6.6 |
| -$40 | 2,934 | 3.5 | 88% | 11.6 | 6.3 |
| -$60 | 1,070 | 3.4 | 84% | 11.2 | 6.1 |
| -$80 | 478 | 3.3 | 82% | 12.5 | 5.7 |
| -$100 | 263 | 3.1 | 80% | 12.4 | 5.8 |

**OOS**

| drew down ≥ | n | t->bottom (min) | bottom @ %dur | RECOVER dur (min) | NO-REC dur (min) |
|---|---|---|---|---|---|
| -$20 | 2,017 | 4.0 | 80% | 15.2 | 8.2 |
| -$40 | 945 | 4.4 | 100% | 14.8 | 7.4 |
| -$60 | 380 | 4.2 | 100% | 14.2 | 7.4 |
| -$80 | 128 | 3.7 | 100% | 16.2 | 7.2 |
| -$100 | 40 | 3.4 | 42% | 16.0 | 6.4 |

---

## Q15 - The bimodal split: winners vs losers, not "peak then collapse"

Testing the claim "bimodal = peaks MFE then goes negative". The ordering holds, but the "peak" in the losing mode is ~$14 - noise, not a peak.

**IS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 976 | $+106 | 80% | 3% | $+25 | 35% |
| LATE bottom (q4) | 1,523 | $+14 | 12% | 99% | $-56 | 99% |
| all losers | 10,097 | $+20 | 10% | 63% | $-24 | 100% |

**OOS** (drew down ≥$40)

| MAE-bottom mode | n | median MFE | MFE≥$50 | peak-before-trough | median close | loss rate |
|---|---|---|---|---|---|---|
| EARLY bottom (q1) | 294 | $+94 | 75% | 5% | $+15 | 42% |
| LATE bottom (q4) | 547 | $+14 | 8% | 99% | $-58 | 100% |
| all losers | 1,708 | $+26 | 18% | 70% | $-36 | 100% |