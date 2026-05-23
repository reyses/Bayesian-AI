# Leg-age exhaustion hazard — is the move expended? (table entry 4)

Given a leg has already run **M minutes**, P(it continues another X min before the pivot). Object = hardened FLAT legs (R-trigger entry → next-pivot exit); age = seconds since entry — the clock a live trade has. `S(X|m) = P(D >= m+X | D >= m)`. Leg-internal, so no zigzag-alternation parity artifact.

**IS duration** (n=17,758): median 9.7 min, mean 17.4, p25 5.2, p75 19.2, p90 37.6, max 670.
**OOS duration** (n=2,930): median 10.9 min, mean 19.1, p25 5.8, p75 21.9, p90 41.4, max 233.

## P(leg continues >= +2 min) by current age M

| age M (min) | n(IS) | IS P(continues) [CI] | n(OOS) | OOS P(continues) [CI] |
|--:|--:|---|--:|---|
| 0.5 | 17,748 | 96% [96%, 97%] | 2,925 | 96% [95%, 96%] |
| 1 | 17,690 | 94% [93%, 94%] | 2,913 | 94% [93%, 95%] |
| 2 | 17,390 | 87% [87%, 88%] | 2,864 | 88% [87%, 90%] |
| 3 | 16,580 | 83% [83%, 84%] | 2,730 | 86% [84%, 87%] |
| 5 | 13,788 | 82% [82%, 83%] | 2,342 | 86% [85%, 87%] |
| 8 | 10,376 | 84% [83%, 84%] | 1,854 | 85% [83%, 87%] |
| 12 | 7,358 | 86% [85%, 87%] | 1,355 | 88% [86%, 90%] |
| 18 | 4,822 | 87% [86%, 88%] | 922 | 90% [88%, 92%] |

## P(leg continues >= +5 min) by current age M

| age M (min) | n(IS) | IS P(continues) [CI] | n(OOS) | OOS P(continues) [CI] |
|--:|--:|---|--:|---|
| 0.5 | 17,748 | 74% [73%, 74%] | 2,925 | 77% [75%, 78%] |
| 1 | 17,690 | 71% [70%, 71%] | 2,913 | 74% [72%, 76%] |
| 2 | 17,390 | 65% [65%, 66%] | 2,864 | 70% [69%, 72%] |
| 3 | 16,580 | 63% [62%, 63%] | 2,730 | 68% [66%, 70%] |
| 5 | 13,788 | 63% [62%, 64%] | 2,342 | 67% [65%, 69%] |
| 8 | 10,376 | 65% [65%, 66%] | 1,854 | 69% [67%, 71%] |
| 12 | 7,358 | 70% [69%, 71%] | 1,355 | 72% [69%, 74%] |
| 18 | 4,822 | 74% [72%, 75%] | 922 | 75% [72%, 78%] |

## Hazard and median remaining life by age

Hazard h(M) = P(leg ends within the next 1 min | alive at M). Median remaining = median(D − M) over legs alive at M.

| age M (min) | IS h(M) | IS med. remaining (min) | OOS h(M) | OOS med. remaining (min) |
|--:|--:|--:|--:|--:|
| 0.5 | 1% | 9.2 | 1% | 10.4 |
| 1 | 2% | 8.8 | 2% | 9.9 |
| 2 | 5% | 7.9 | 5% | 9.1 |
| 3 | 8% | 7.5 | 7% | 8.8 |
| 5 | 9% | 7.8 | 8% | 9.2 |
| 8 | 9% | 8.8 | 7% | 9.8 |
| 12 | 8% | 10.2 | 6% | 11.3 |
| 18 | 7% | 12.6 | 5% | 12.5 |

## Exhaustion verdict

Reference horizon +2 min. IS hazard peaks at **5 min** (9%/min); IS continuation troughs at **5 min** — call ~5 min the danger window. The shape is read with two ARMS off that trough, not an endpoint fresh-vs-aged test (which would skip the trough).

- **IS early arm** — S(+2m | 0.5m) − S(+2m | 5m) = +14pp [+13, +15]  SIGNIFICANT.
- **IS late arm** — S(+2m | 18m) − S(+2m | 5m) = +5pp [+4, +6]  SIGNIFICANT.
- **OOS early arm** — S(+2m | 0.5m) − S(+2m | 5m) = +10pp [+8, +11]  SIGNIFICANT.
- **OOS late arm** — S(+2m | 18m) − S(+2m | 5m) = +4pp [+1, +6]  SIGNIFICANT.

- **IS above-floor** — S(+2m | 3m) − S(+2m | 18m) = -4pp [-5, -3]  SIGNIFICANT (positive = exhaustion persists above the 3-min floor).
- **OOS above-floor** — S(+2m | 3m) − S(+2m | 18m) = -4pp [-7, -2]  SIGNIFICANT (positive = exhaustion persists above the 3-min floor).

**HUMP-SHAPED HAZARD — NOT monotone exhaustion.** Continuation is high for a fresh leg, troughs at the ~5-min danger window, then RECOVERS for aged legs — both arms significant in IS and OOS. An endpoint-only "fresh vs aged" test would skip the trough and mislabel this "exhaustion". **min_bars caveat**: the zigzag enforces a 3-min minimum leg, so a hardened leg rarely dies below ~3 min — the steep early arm is partly that construction floor, not market momentum. The clean, non-structural signal is the danger window itself and the recovery past it: a leg that clears ~5 min becomes MORE persistent, not less. A pure "old leg = cut it" time-stop is unsupported; only a danger-window-aware check is backed by the data.

## Read

- Compare S(X|m) DOWN each table column. Falling = exhaustion; flat = memoryless; rising = momentum; falls-then-rises = a HUMP (a danger window). The hazard h(M) is the same story per-bin.
- Trust a row only where IS and OOS agree.
- min_bars caveat: the zigzag enforces a 3-min minimum leg (min_bars=36 5s-bars), so a hardened leg rarely dies below ~3 min — the low early hazard / steep early ramp is partly this construction floor, NOT pure market momentum. Read the above-floor test for the non-structural picture.
- Selection caveat: legs still alive at a high age skew toward low-volatility overnight legs — a real part of the conditional law, but it means a "still alive at 18 min" leg is not the same animal as a fresh RTH leg.