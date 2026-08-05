# How wide must a stop be to hold a real leg?

56,985 entries into 14,246 legs of >= 15pt, 112 days. Entry-rule independent: this is the tape, not a policy.

## entering 0s after the leg starts (n=30,082)

| stop | legs survived | mean kept | median kept | E[pt] |
|---|---|---|---|---|
| −5 | 96% | 26.2 | 22.5 | +24.14 |
| −10 | 99% | 26.4 | 22.5 | +25.30 |
| −15 | 100% | 26.5 | 22.5 | +25.52 |
| −20 | 100% | 26.5 | 22.5 | +25.56 |
| −25 | 100% | 26.5 | 22.5 | +25.60 |
| −30 | 100% | 26.5 | 22.5 | +25.61 |
| −40 | 100% | 26.5 | 22.5 | +25.62 |
| −50 | 100% | 26.5 | 22.5 | +25.62 |

## entering 30s after the leg starts (n=15,105)

| stop | legs survived | mean kept | median kept | E[pt] |
|---|---|---|---|---|
| −5 | 87% | 15.4 | 13.0 | +11.75 |
| −10 | 100% | 15.1 | 12.8 | +14.19 |
| −15 | 100% | 15.1 | 12.8 | +14.24 |
| −20 | 100% | 15.1 | 12.8 | +14.25 |
| −25 | 100% | 15.1 | 12.8 | +14.25 |
| −30 | 100% | 15.1 | 12.8 | +14.25 |
| −40 | 100% | 15.1 | 12.8 | +14.25 |
| −50 | 100% | 15.1 | 12.8 | +14.25 |

## entering 60s after the leg starts (n=8,385)

| stop | legs survived | mean kept | median kept | E[pt] |
|---|---|---|---|---|
| −5 | 89% | 14.0 | 11.5 | +10.92 |
| −10 | 100% | 13.9 | 11.5 | +12.98 |
| −15 | 100% | 13.9 | 11.5 | +13.00 |
| −20 | 100% | 13.9 | 11.5 | +13.00 |
| −25 | 100% | 13.9 | 11.5 | +13.01 |
| −30 | 100% | 13.9 | 11.5 | +13.01 |
| −40 | 100% | 13.9 | 11.5 | +13.01 |
| −50 | 100% | 13.9 | 11.5 | +13.01 |

## entering 120s after the leg starts (n=3,413)

| stop | legs survived | mean kept | median kept | E[pt] |
|---|---|---|---|---|
| −5 | 90% | 13.2 | 11.0 | +10.41 |
| −10 | 100% | 13.1 | 10.8 | +12.19 |
| −15 | 100% | 13.1 | 10.8 | +12.21 |
| −20 | 100% | 13.1 | 10.8 | +12.21 |
| −25 | 100% | 13.1 | 10.8 | +12.21 |
| −30 | 100% | 13.1 | 10.8 | +12.21 |
| −40 | 100% | 13.1 | 10.8 | +12.21 |
| −50 | 100% | 13.1 | 10.8 | +12.21 |

## VERDICT — and it reverses the single-day claim

MAE **inside** a real leg (>= 15pt), entering 30s after it starts, 15,105
entries across 112 days:

| p50 | p75 | p90 | p95 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|
| 2.25 | 3.75 | 5.25 | 6.50 | 8.25 | 11.22 | 19.25 |

**Only 0.21% of real legs shake out more than 10pt. None exceed 25pt.**
Even for big legs (>= 40pt, n=2,205) the p95 heat is 7.50pt and just 1.0%
exceed 10pt.

So a −10 stop already holds essentially every real leg, and the claim I made
from 2024_09_16 — "−10 is structurally incompatible with holding through a
25pt shakeout" — is **false at scale**.

### Why the single day looked different

Those 8 "late" entries on 2024_09_16 were not inside the leg. They were in
the oscillation BEFORE it started. The 24-27pt of heat they took was the
chop at the top, not the descent's own shakeout. Entering during the noise
and calling the noise a shakeout is a category error, and I made it.

### The actual finding

The binding constraint is **not stop width — it is whether you are in a leg
at all**. In a leg, −10 is ample. Outside one, no sane stop helps, because
there is nothing to hold.

That relocates the whole problem back to the same place everything else
landed this week: identifying that a leg has STARTED. Which is the one thing
the owner does well and the machine does not.

### Caveat that limits this table

Legs here are selected because they completed (>= 15pt). The `E[pt]` column
is therefore conditioned on the leg existing and is NOT a tradeable
expectancy — it answers "how much room does a real leg need", not "does
widening a stop make money".
