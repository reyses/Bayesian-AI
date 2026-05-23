# Leg-Chop Survival v2 — does EARLY chop forecast a long wide leg?

Wide legs = zigzag at ATR x8; underlying chop = nested zigzag legs at ATR x1. **Fixed early window**: chop is measured over only the first K tight legs of the wide leg — `early_chop_ratio` = path of those K legs / their net displacement (~1 clean, >>1 choppy). This is decoupled from the wide leg's eventual length (the v1 confound). Outcome = total tight-leg count C; "runs long" = C >= K+5.
IS 7,457 wide legs / 275 days; OOS 1,136 / 51 days.

## Window K = 3 tight legs

**IS** — 4,330 wide legs reached >=3 tight legs.  corr(early chop ratio, total length C) = +0.220
**OOS** — 779 wide legs reached >=3 tight legs.  corr(early chop ratio, total length C) = +0.121

| set | early-chop band | n | median C | P(C>=8) | 95% CI |
|---|---|--:|--:|--:|---|
| IS | clean | 1,443 | 5 | 27% | [25%, 30%] |
| IS | medium | 1,443 | 5 | 33% | [30%, 35%] |
| IS | choppy | 1,444 | 7 | 48% | [45%, 50%] |
| OOS | clean | 254 | 5 | 33% | [27%, 39%] |
| OOS | medium | 262 | 5 | 38% | [32%, 44%] |
| OOS | choppy | 263 | 9 | 56% | [50%, 62%] |

## Window K = 5 tight legs

**IS** — 2,819 wide legs reached >=5 tight legs.  corr(early chop ratio, total length C) = +0.251
**OOS** — 546 wide legs reached >=5 tight legs.  corr(early chop ratio, total length C) = +0.095

| set | early-chop band | n | median C | P(C>=10) | 95% CI |
|---|---|--:|--:|--:|---|
| IS | clean | 940 | 7 | 31% | [28%, 33%] |
| IS | medium | 939 | 9 | 36% | [34%, 40%] |
| IS | choppy | 940 | 11 | 54% | [51%, 57%] |
| OOS | clean | 180 | 7 | 37% | [30%, 44%] |
| OOS | medium | 191 | 9 | 43% | [37%, 51%] |
| OOS | choppy | 175 | 13 | 67% | [61%, 74%] |

## Read

- `early_chop_ratio` is measured over a FIXED count of tight legs, so it is NOT mechanically tied to the wide leg's eventual length — unlike the confounded v1.
- If `P(C>=K+N)` and the corr are flat / ~0 across the early-chop bands, **early chop does not forecast leg length** — the chop content carries no continuation information.
- If `choppy` early shows a higher P than `clean`, an early-choppy wide leg genuinely tends to run longer (and vice versa). Trust it only where IS and OOS agree.