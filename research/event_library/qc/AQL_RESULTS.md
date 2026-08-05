# AQL sampling inspection — 2-year event corpus

ISO 2859-1 style, general inspection level II, AQL 1.0%.
Sample 100/batch drawn at random across 2024-01 to 2026-03.
**Accept <= 10 defects, reject >= 11.**

| detector | lot | inspected | defects | verdict |
|---|---|---|---|---|
| fakeout_poke | 153,029 | 100 | **0** | ACCEPT |
| leg_descent | 58,480 | 100 | **0** | ACCEPT |
| ultra_chop | 18,601 | 100 | **0** | ACCEPT |
| stall | 41,180 | 100 | **0** | ACCEPT |

**400 events inspected across four detectors, zero defects.**

Every inspector worked clean-room — reimplementing the detector and the
outcome scan from the WRITTEN spec rather than importing the code — then
replayed the raw 1s/5s bars. Three of the four also hand-traced at least one
event bar-by-bar with no script.

## Findings that were NOT defects, but were worth surfacing

1. **ultra_chop de-dup**: the forward-looking refractory rule is real and
   observable, but it is a SAMPLING rule — it changes which events enter the
   set, never a fired event's own features or outcome. Not a per-event
   defect; a live implementation must self-suppress for 60s instead.
2. **stall give_frac skew** (mean 3.43, 87/100 rows > 1.0, max 15.875): hand
   traced the extreme (2026_03_05, mfe 8.0pt, give_frac 15.875 = 127pt
   giveback) against raw bars — a real continuous 127pt decline. It is a
   property of measuring 10-minute giveback against a minimum-size 8pt leg,
   not a bug.
3. **fakeout_poke coverage gap**: this batch drew 60 BREAKOUT / 40 RETURN /
   0 STUCK, with resolve lag topping out at 25s against a 60s boundary. The
   STUCK branch and the RETURN-after-60s path are real code paths that went
   untested. A targeted batch is required before claiming coverage of them.

## Standing limit

0/100 per detector is evidence of a clean sample, not proof of a clean lot.
Batches 1-4 remain undrawn for each detector; the full AQL plan is 500 per
detector.
