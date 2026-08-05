# Scoring the owner's directional theses — v0, and it is NOT trustworthy yet

The measurement the teacher-student program rests on. Built with the
discipline announced beforehand: theses extracted and frozen in pass 1,
outcomes joined only in pass 2, horizons fixed in advance, unscoreable
statements counted rather than dropped.

## The raw v0 numbers

35 theses extracted from 443 day-tagged owner messages; 24 scoreable
(8 fell too near the day's end, 1 unresolved at 60m).

| horizon | n | hit rate | day-clustered 95% CI |
|---|---|---|---|
| 5 min | 24 | 33.3% | [16.7%, 42.9%] — **excludes 50%** |
| 15 min | 24 | 41.7% | [16.7%, 57.1%] |
| 60 min | 23 | 56.5% | [33.3%, 71.4%] |

Taken literally that says his 5-minute calls are significantly WORSE than a
coin flip, and drift toward chance as the horizon lengthens.

## Why I am not reporting that as a finding

**1. The extractor is crude and demonstrably wrong.** A regex for direction
words plus a regex for prediction words. Auditing its own output, at least
3 of 35 are not predictions at all:

- "This turn is botched but we will play as if, hopefully it went down" — a
  hope about the past, not a forecast.
- "If this were real I would go down to 5s level to see what's happening" —
  *go down to the 5s chart*, a navigation instruction.
- "Naw let's just note it, the 1m looks smooched..." — no forecast present.

That is a ~9% visible contamination rate, and only for cases obvious enough
for me to catch by eye.

**2. The horizon almost certainly does not match his intent.** Most of his
statements are about the NEXT BAR or the next few seconds — "next bar will
go down", "keep watching next 1m bar". Scoring a next-bar call over 5 minutes
measures something he never claimed.

**3. N = 24.** After a day spent demolishing my own N=1 and N=15 results, a
24-sample verdict on the program's central question is not a verdict.

**4. Sim-clock mapping is approximate.** Each thesis inherits the bar of the
nearest preceding dojo event; if the tape sat frozen while we talked, that
bar can be stale relative to what he was actually looking at.

## What v1 needs

- LLM extraction, not regex: pull the CLAIM, the DIRECTION, and the HORIZON
  he actually stated, and discard anything that is an instruction, a
  question, or a comment about the past.
- Score each thesis at ITS OWN stated horizon (next bar, next 60s, next
  hour), not a fixed grid.
- Sim clock taken from the state file at that instant, not the nearest event.
- Then, and only then, a hit rate worth acting on.

**Status: instrument built, calibration failed, number withheld.**
