# EVENT LIBRARY v0 — owner-named tape states as causal detectors + cohort tables

Owner architecture: *identify specific events, read the fuzzy events.* Each named tape state gets (a) a strictly causal detector and (b) its own cohort outcome table. This is the substrate for an event-classification + table-lookup ML target, not price prediction.

Corpus: `DATA/ATLAS/{1s,5s,1m}` day files, 603 day files in the 3-timeframe intersection, of which **540 carry RTH tape** (the remaining 64 are Sunday-evening / holiday files whose only bars are the 18:00-19:00 prior-evening session).

**Live-day guard**: `2024_09_16` is the pocket-dojo live-sim day and is EXCLUDED from every table below. It appears only in `reports/anchor_fire.md`, where each detector is fired against the owner's calibration anchors.

**Two race definitions are reported for every continuation question.** The natural race is distance-ASYMMETRIC (a new low sits 1 tick away, breaking the stair sits 10-15pt away), so its headline percentage is mostly geometry. Every such event therefore also carries a distance-SYMMETRIC race (+-10pt from the event close, 10pt = the POCKET_CARD floor stop) whose null is 50% by construction. **Read the symmetric race to answer 'is this event informative?'**

**Causality is demonstrated, not asserted.** `tools/causality_audit.py` replays every detector on days truncated at 11:00 / 13:00 / 14:30 ET and requires every event stamped at or before the cut to reappear identically. Current result over 40 sampled days: **0 missing, 0 extra, 0 field mismatches across all six detectors** (`reports/causality_audit.md`). That test caught one real lookahead — DEFENDED_POKE_AT_SHELF was reading a 3-bar poke minimum at a stamp that could fire on bar +1 — which is fixed; the fix moved its crack rate from 28% to 38%, so the leak was materially optimistic.

## Headline verdicts

| event | N | detector prevalence | headline | sharp or fuzzy? |
|---|---|---|---|---|
| 1. ULTRA_CHOP | 18601 | 34.4/day on 529 days | escape UP 50.9% [50%,52%]; +15min drift median -0.25pt | **FUZZY** — escape direction is a coin flip and the post-escape drift median is ~0 |
| 2. LEG_DESCENT | 58480 | 108.3 defended pushes/day | sym CONT N>=2 49.7% vs N=1 49.6%, delta +0.1% [-0.7%,+0.9%] | **FUZZY** — stair depth N carries no information: the symmetric continuation rate is flat at 50% for every N |
| 3. FAKEOUT_POKE | 78731 | 145.8 snap-backs/day | never clears the level 33.4% vs 9.5% for sticking pokes, delta +23.9% | **SHARP on the level question, FUZZY on direction** — the snap-back cuts p(level clears) by ~24pp vs a poke that sticks, while the symmetric +-10pt direction race moves ~1pp (formally significant at n=153k, operationally nil) |
| 4. STALL | 461 | 461/41180 peak candidates = 1.1% | NEW_EXTREME 85.6% vs control 9.5%; sym CONT 50.7% | **FUZZY** — the 85% new-extreme rate is positional mechanics (the stall is defined as not having given back); the symmetric race is 50/50 |
| 5. DEFENDED_POKE_AT_SHELF | 1583 | 1585 events on 444 days | CRACK flushV 39.1% vs other 37.2%, delta +1.9% [-4.9%,+9.0%] | **FUZZY across day-class** — crack rate is ~37% on any high-dwell shelf and flushV days are not distinguishable |
| 6. FLUSH_V_DAY | 136 | 136/540 scored days = 25.2% | peak reclaim 86.8% vs control 73.8%, delta +13.0% [+5.6%,+20.1%] | **SHARP as a day-class label** — flushV days reclaim the recovery peak far more often than matched control days |

## Calibration-day anchor check (2024_09_16 — EXCLUDED from all tables)

Detection timestamps in `reports/anchor_fire.md`. Owner anchors: ULTRA_CHOP 10:23:50-10:24:31; LEG_DESCENT the 09:56-10:24 stair 19697 -> 19633; FLUSH_V_DAY the open flush.

| event | fires on the calibration day? |
|---|---|
| 1. ULTRA_CHOP | **NO — 37 fires that day, 0 inside the anchor window.** Not a threshold miss: see the anchor-honesty table below. |
| 2. LEG_DESCENT | YES — 85 defended pushes, 53 with chain_n>=2, **11 inside the 09:56-10:24 anchor**, chain descent up to 156pt. |
| 3. FAKEOUT_POKE | YES — 264 armed pokes, 148 RETURN events. |
| 4. STALL | NO — 63 peak candidates, 0 stalls. No owner anchor was given for STALL; the day trended hard and STALL is a ~1%-of-candidates big-leg event, so 0 on one day is within expectation. |
| 5. DEFENDED_POKE_AT_SHELF | YES — 5 events (09:58, 10:30, 14:23, 14:55, 15:28), all flushV class, all HOLD. |
| 6. FLUSH_V_DAY | YES — confirmed 09:50, flush 110.2pt, recovery 85%. (The owner quotes -173.5pt; the imported detector measures from the 09:30 open, not the overnight high.) |

---

## 1. ULTRA_CHOP

### Definition (reproducible)

1s closes, RTH 09:30-15:30 ET, rolling **60s** window (`CHOP_WIN_S=60`) which must contain >= 40 1s bars. Fires when BOTH:

- `flips >= 30` — direction flips of the non-zero 1s close-to-close moves inside the window (corpus RTH p75 ~= 30, i.e. top-quartile flip density);
- `box <= 0.60 x ambient` where `box` = window high-low and `ambient` = median of the last 60 NON-overlapping 1-minute boxes, read through the previous minute only.

Chop box = the firing window's `[low, high]`. **Escape** = first close beyond a box edge by `0.50 x box` (scale-free buffer). One event per episode: the next fire is blocked until the escape, and never within 60s.

### Why 'small net range' is RELATIVE, not absolute (anchor honesty)

The owner's anchor (2024_09_16 10:23:50-10:24:31, '~24 flips / 42s in 13.25pt') **does not fire this detector, and no useful absolute threshold makes it fire.** Measured on 1s closes, the 60s windows ending inside that anchor carry:

| quantity | anchor window | that day's RTH p50 | p90 |
|---|---|---|---|
| flips / 60s | 27-33 | 27 | 32 |
| box (pt) | 15.50-24.00 | 11.00 | 21.00 |

Flip density is genuinely elevated (p75-p90). The **box is ABOVE median** — the 60s window swallows the 11.25pt one-second flush at 10:24:11, so the anchor is an impulse-with-churn, not a tight box. An absolute threshold loose enough to fire there (`box <= 24pt`) fires on ~40% of all RTH bars and measures nothing. Absolute point thresholds are also era-broken: MNQ traded 16k in 2024 and 28k in 2026. Hence the ambient-relative box test. See `reports/anchor_fire.md`.

### Prevalence

- 18601 events on 529 of 540 trading days = 34.45/day
- per year:
  - 2024: 8645 events / 258 day files = 33.51/day
  - 2025: 7717 events / 277 day files = 27.86/day
  - 2026: 2239 events / 68 day files = 32.93/day

### TABLE — escape statistics

- escaped within 30min: 100.0% [100.0%, 100.0%] (n=18601)
- time-to-escape: median +49.00s [q25 +24.00, q75 +90.00] (n=18601)
- escape direction UP: 50.9% [50.1%, 51.6%] (n=18601) (null 50%; CI excludes 50%)

Signed displacement AFTER the escape, in the escape direction (positive = the break kept going), vs the unconditional |move| from 6468 random RTH anchors (12/day):

| horizon | signed move in escape dir | median 95% CI | control \|move\| |
|---|---|---|---|
| +5min | median -0.25pt [q25 -10.75, q75 +10.75] (n=18601) | [-0.50, +0.00] | median +10.00pt [q25 +4.25, q75 +19.75] (n=6468) |
| +15min | median -0.25pt [q25 -18.25, q75 +18.25] (n=18601) | [-0.50, +0.25] | median +16.75pt [q25 +7.25, q75 +33.50] (n=6468) |
| +30min | median -0.25pt [q25 -26.00, q75 +26.00] (n=18540) | [-0.75, +0.50] | median +23.75pt [q25 +10.50, q75 +47.00] (n=6468) |

Escape direction is 50.9% up [50.1%, 51.6%]. On n=18601 the CI does clear 50%, but the effect is a 0.9pp lean — statistically detectable, operationally nothing, and it is the corpus's own upward drift rather than a property of chop. Post-escape drift medians are a fraction of a point against unconditional |moves| of 16.8pt at 15min: **the break carries no persistent direction.**

### Causality self-audit

- Every input is a TRAILING window ending at the firing bar: flips, box, and the ambient scale (which reads only minutes strictly before the firing bar's own minute, via `prev_slot = slot - 1`).
- The escape and all magnitudes are computed in `outcomes.py`, strictly after the stamp.
- The episode de-dup guard uses the forward escape time. That is a SAMPLING decision (which candidate bars become rows), not a feature — it cannot leak into an event's own outcome, but it does mean the row set is not reproducible bar-by-bar in live without buffering. **This is the one place in the library where a live implementation must differ**: live must fire on the first qualifying bar and self-suppress for 60s.
- RTH mask is bounded on both sides, so prior-evening bars (mod >= 1080) can never be selected.

**Verdict:** **FUZZY** — escape direction is a coin flip and the post-escape drift median is ~0.

---

## 2. LEG_DESCENT (stair-down)

### Definition (reproducible)

5s closes, repo-canonical 8.0pt close zigzag (= `research/reversal_gauge` REVERSAL_PT).

A **push** opens when a swing HIGH confirms (price falls 8.0pt off the running max). Inside the push the running low `L` is tracked on bar LOWS. A **defense** confirms at the first RTH bar whose CLOSE is >= `L + 2.0pt` within 30s of the bar that set `L` — this covers both owner phrasings at once: a long lower wick (same-bar low, recovering close) and a fast multi-bar V-up. One defense per push (the first).

**chain_n** = consecutive defended pushes whose high does not exceed the previous push's high by more than 2.0pt (a push clearing the prior high by <= 2pt is a poke, not a new high — same tolerance as FAKEOUT_POKE). `chain_n >= 2` is the owner's '>= 2 lower-high pushes'; `chain_n == 1` is the structurally matched CONTROL (a defended push with no lower-high predecessor).

Stamp = the defense bar. Outcomes over 30min.

### Prevalence

- 58480 defended pushes on 539 days = 108.3/day; `chain_n>=2` = 32895 (56.2% of pushes)
- chain length distribution: N=1:25585, N=2:14441, N=3:8112, N=4:4547, N=5:2486, N=6:3309  (N=6 bucket is 6+)

### TABLE — continuation after the Nth stair step

ASYMMETRIC race (owner's literal question): NEW_LOW = a low 1 tick below the step low; STAIR_BREAK = a high 1 tick above the step high (which sits >= 8pt away).

| chain_N | N | NEW_LOW | STAIR_BREAK | NEITHER |
|---|---|---|---|---|
| 1 | 25585 | 17523 (68.5%) | 8059 (31.5%) | 3 (0.0%) |
| 2 | 14441 | 10017 (69.4%) | 4421 (30.6%) | 3 (0.0%) |
| 3 | 8112 | 5612 (69.2%) | 2500 (30.8%) | 0 (0.0%) |
| 4 | 4547 | 3111 (68.4%) | 1436 (31.6%) | 0 (0.0%) |
| 5 | 5795 | 4077 (70.4%) | 1717 (29.6%) | 1 (0.0%) |

SYMMETRIC race (+-10pt from the defense close; null = 50%):

| chain_N | N | CONT | AGAINST | NEITHER |
|---|---|---|---|---|
| 1 | 25584 | 12677 (49.6%) | 12876 (50.3%) | 31 (0.1%) |
| 2 | 14441 | 7196 (49.8%) | 7227 (50.0%) | 18 (0.1%) |
| 3 | 8112 | 3973 (49.0%) | 4128 (50.9%) | 11 (0.1%) |
| 4 | 4547 | 2218 (48.8%) | 2320 (51.0%) | 9 (0.2%) |
| 5 | 5795 | 2952 (50.9%) | 2836 (48.9%) | 7 (0.1%) |

- **symmetric continuation, N>=2 vs N=1**: N>=2 49.7% (n=32850) vs N=1 49.6% (n=25553) -> delta +0.1% 95% CI [-0.7%, +0.9%] -> not significant (CI includes 0)

### TABLE — defense-hold rate (defended low survives 5min un-undercut)

- N=1: 15.2% [14.8%, 15.6%] (n=25585)
- N>=2: 15.1% [14.7%, 15.5%] (n=32895)
- **defense hold, N>=2 vs N=1**: N>=2 15.1% (n=32895) vs N=1 15.2% (n=25585) -> delta -0.1% 95% CI [-0.7%, +0.5%] -> not significant (CI includes 0)

### TABLE — stair depth distribution

- step depth (step high -> step low): median +13.00pt [q25 +10.75, q75 +16.75] (n=58480)
- cumulative chain descent at step N>=2: median +30.00pt [q25 +19.00, q75 +48.25] (n=32895)
- defense size (close - low): median +3.50pt [q25 +2.50, q75 +5.00] (n=58480)
- defense lag: median +0.00s [q25 +0.00, q75 +5.00] (n=58480)

### Causality self-audit

- The step high is a zigzag pivot CONFIRMED before the push began; the step low is a running min over bars <= the stamp; the defense is a close at the stamp bar. Nothing is back-dated to the pivot bar.
- `chain_n` uses only previously CLOSED pushes (a push is closed by its own confirmed low pivot, or by the next confirmed high pivot).
- Known asymmetry, not a leak: the asymmetric race is ~69% NEW_LOW at every N because the two triggers sit at very different distances. The symmetric race is the interpretable one.

**Verdict:** **FUZZY** — stair depth N carries no information: the symmetric continuation rate is flat at 50% for every N. The chain is real and easy to detect (58k instances), but conditioning on it moves nothing.

---

## 3. FAKEOUT_POKE

### Definition (reproducible)

5s closes, same 8.0pt zigzag. During an active leg, when the leg's running extreme first clears a REMEMBERED same-direction leg extreme (confirmed pivot, aged <= 90min) by `0 < over <= 2.0pt`, a poke ARMS. It then resolves as exactly one of:

| kind | resolution | meaning |
|---|---|---|
| **RETURN** | a close back inside the level within 60s | **the owner's fakeout poke** |
| BREAKOUT | clears the level by > 2.0pt first | the level actually broke |
| STUCK | still outside after 60s without clearing | hung on the level |

Stamp = the resolution bar. Outcomes over 45min. Everything is close-based, matching the close-based zigzag that defines the reference extremes.

### Prevalence

- 153029 armed pokes on 539 days; RETURN 78731 (51.4%), BREAKOUT 74232 (48.5%), STUCK 66 (0.0%)
- RETURN (the event) = 145.8/day
- poke depth beyond the level: median +1.00pt [q25 +0.50, q75 +1.50] (n=78731)
- reference-level age: median +16.08min [q25 +5.00, q75 +41.17] (n=78731)

### TABLE — resume vs reverse from the poke

'Never exceeds the prior extreme' is reported two ways. UNBOUNDED ('ever, within 45min') is nearly vacuous: price wanders 2pt past any level given 45 minutes. BOUNDED (`exceed_ref_first`: clears the level by > 2.0pt BEFORE a 10pt adverse move) is the load-bearing number.

| cohort | N | clears level, unbounded | clears level, BOUNDED | **never clears (bounded)** |
|---|---|---|---|---|
| RETURN | 78731 | 91.8% | 66.6% [66.3%, 67.0%] | **33.4%** |
| STUCK | 66 | 95.5% | 87.9% [77.9%, 93.7%] | **12.1%** |
| BREAKOUT | 74232 | 98.7% | 90.5% [90.3%, 90.7%] | **9.5%** |

- **clears the level (bounded), RETURN vs BREAKOUT**: RETURN 66.6% (n=78731) vs BREAKOUT 90.5% (n=74232) -> delta -23.9% 95% CI [-24.3%, -23.5%] -> SIGNIFICANT

Asymmetric RESUME/REVERSE race (RESUME = clears the poke extreme by 0.5pt, REVERSE = 10pt adverse, whichever first):

| kind | N | RESUME | REVERSE | NEITHER |
|---|---|---|---|---|
| BREAKOUT | 74232 | 60335 (81.3%) | 13897 (18.7%) | 0 (0.0%) |
| RETURN | 78731 | 54673 (69.4%) | 24051 (30.5%) | 7 (0.0%) |
| STUCK | 66 | 58 (87.9%) | 8 (12.1%) | 0 (0.0%) |

SYMMETRIC race (+-10pt from the resolution close, in leg direction; null = 50%):

| kind | N | CONT | AGAINST | NEITHER |
|---|---|---|---|---|
| BREAKOUT | 74232 | 36572 (49.3%) | 37622 (50.7%) | 38 (0.1%) |
| RETURN | 78731 | 39554 (50.2%) | 39116 (49.7%) | 61 (0.1%) |
| STUCK | 66 | 34 (51.5%) | 29 (43.9%) | 3 (4.5%) |

- **symmetric continuation, RETURN vs BREAKOUT**: RETURN 50.3% (n=78670) vs BREAKOUT 49.3% (n=74194) -> delta +1.0% 95% CI [+0.5%, +1.5%] -> SIGNIFICANT
- RETURN symmetric continuation vs the 50% null: 50.3% [49.9%, 50.6%] (n=78670) — coin flip

### On the '~78.5% never exceed the prior extreme' reference

That figure could not be reproduced and I could not find a re-poke library that produces it. The only 78.5% in the repo is `research/dojo_forge/reports/oscillation_harvest.md` — P(a sigma-band traverse COMPLETES) at K>=5 prior traverses, over 54,911 fade attempts. That is a different measurement (band-to-band traverse completion), not level re-poke survival. The adjacent level-memory claim in `human_dojo/POCKET_CARD.md` is '+10 survives 98.5% of re-pokes', which is about STOP survival distance, not exceedance.

This library's comparable number: **33.4% of snap-back pokes never clear the level** (n=78731) before a 10pt adverse move, and 8.2% never clear it at all inside 45min. Neither lands near 78.5%; the 78.5% reference is a different event.

### Causality self-audit

- The reference extreme is a zigzag pivot confirmed strictly earlier; arming uses the running extreme at the current bar; resolution is a condition on the current bar's close or the elapsed 60s. No forward bar is read by the detector.
- Arming is blocked on bars where a pivot confirms (`ev is None` guard), so a leg reversal cannot masquerade as a poke.
- An armed poke is resolved BEFORE the leg-reversal bookkeeping in the same bar, so a snap-back that coincides with a leg turn is still recorded as RETURN rather than dropped.

**Verdict:** **SHARP on the level question, FUZZY on direction** — the snap-back cuts p(level clears) by ~24pp vs a poke that sticks, while the symmetric +-10pt direction race moves ~1pp (formally significant at n=153k, operationally nil).

---

## 4. STALL

### Definition (reproducible)

5s closes, same zigzag. A stall CANDIDATE opens at every new running leg extreme with leg MFE >= 8pt in RTH (de-duplicated: a new candidate only once the extreme has advanced 25% of MFE past the last one opened). A candidate is:

- **VOID** if price extends > 25% of MFE beyond it (the leg was still RUNNING, not stalling — generalises `four_phase_cohort`'s implicit assumption that the peak is the peak);
- **FAILED** if giveback exceeds 30% of MFE (= `four_phase_cohort` STALL_GIVE) before its mark;
- **STALL** if it survives to its mark at peak + 10min (= `four_phase_cohort` STALL_MIN).

Stamp = the 10-minute mark. FAILED candidates are emitted too, at the SAME relative moment, as the matched control. This generalises the four-phase stall off flush days: no flush, no V, no shape gate — any leg peak on any day.

### Prevalence

- 41180 peak candidates on 539 days; **461 STALL (1.12%)**, 40719 control
- leg MFE at the peak — STALL: median +38.50pt [q25 +29.00, q75 +49.75] (n=461)
- leg MFE at the peak — control: median +15.00pt [q25 +10.50, q75 +22.75] (n=40719)
- Selection effect, stated up front: a 30%-of-MFE tolerance is only survivable for big legs (an 8pt zigzag reversal alone exceeds 30% of any MFE below ~27pt), so STALL is structurally a big-leg event. That is a property of the owner's definition, not a bug.

### TABLE — what follows a stall

ASYMMETRIC race from the mark, 60min: NEW_EXTREME = 0.5pt beyond the stalled peak; GIVEBACK_50 = 50% of MFE given back.

| stalled | N | NEW_EXTREME | GIVEBACK_50 | NEITHER | NO_DATA |
|---|---|---|---|---|---|
| False | 40719 | 3861 (9.5%) | 36844 (90.5%) | 2 (0.0%) | 12 (0.0%) |
| True | 461 | 393 (85.2%) | 66 (14.3%) | 1 (0.2%) | 1 (0.2%) |

- **p(new extreme first), STALL vs control**: STALL 85.6% (n=459) vs control 9.5% (n=40705) -> delta +76.1% 95% CI [+72.7%, +79.2%] -> SIGNIFICANT

Giveback bucket at the mark (monotone read; the stall bucket is the 0-30% row by definition):

| bucket | N | NEW_EXTREME | GIVEBACK_50 | NEITHER | NO_DATA |
|---|---|---|---|---|---|
| <=30% (STALL) | 461 | 393 (85.2%) | 66 (14.3%) | 1 (0.2%) | 1 (0.2%) |
| 30-50% | 1005 | 658 (65.5%) | 345 (34.3%) | 1 (0.1%) | 1 (0.1%) |
| 50-100% | 4674 | 1459 (31.2%) | 3208 (68.6%) | 1 (0.0%) | 6 (0.1%) |
| >100% | 35040 | 1744 (5.0%) | 33291 (95.0%) | 0 (0.0%) | 5 (0.0%) |

SYMMETRIC race (+-10pt from the mark close, in leg direction; null = 50%):

| stalled | N | CONT | AGAINST | NEITHER |
|---|---|---|---|---|
| False | 40707 | 20361 (50.0%) | 20314 (49.9%) | 32 (0.1%) |
| True | 460 | 231 (50.2%) | 225 (48.9%) | 4 (0.9%) |

- **symmetric continuation, STALL vs control**: STALL 50.7% (n=456) vs control 50.1% (n=40675) -> delta +0.6% 95% CI [-4.1%, +5.2%] -> not significant (CI includes 0)
- STALL symmetric continuation vs the 50% null: 50.7% [46.1%, 55.2%] (n=456) — coin flip

- time to resolution, STALL: median +0.25min [q25 +0.08, q75 +2.00] (n=459)
- time to resolution, control: median +0.08min [q25 +0.08, q75 +0.08] (n=40705)
- net move in leg direction at +60min, STALL: median +2.62pt [q25 -18.25, q75 +25.38] (n=418)
- net move in leg direction at +60min, control: median -0.25pt [q25 -41.50, q75 +40.75] (n=36937)

### Causality self-audit

- The candidate's peak, MFE and running giveback are all computed from bars <= the current bar; the stamp is the 10-minute mark, at which the stall is fully observed.
- `p['dir']` is pinned at candidate open so a leg reversal cannot flip the sign of the giveback measurement mid-candidate (this was a real bug in the first implementation).
- The first implementation used a single candidate slot, which let a failed candidate block the next 10 minutes of tape and deleted most real stalls (0 stalls / 20 candidates on the calibration day). Candidates are now a pending LIST. Overlapping candidates within one leg are correlated — rows are NOT independent; day-level clustering should be assumed in any downstream fit.

**Verdict:** **FUZZY** — the 85% new-extreme rate is positional mechanics (the stall is defined as not having given back); the symmetric race is 50/50.

---

## 5. DEFENDED_POKE_AT_SHELF

### Definition (reproducible)

1m bars — deliberately the same bar size and the same window constants as `research/dojo_forge/tools/vshape_retest_cohort.py`, so the flushV sub-cohort is directly comparable to its published number.

- **shelf** = mode of the prior 120 1m closes in 2pt bins (STRICTLY prior bars; >= 100 bars required), and the mode bin must hold >= 8% of them (2.4x uniform for a 60pt spread) — a genuine high-dwell level, not just an argmax;
- **approach** = a high >= shelf + 10pt in the prior 30min (price must come back TO the shelf);
- **trigger** = a bar with low <= shelf + 5pt;
- **poke** = min low over 3 bars from the trigger;
- **defended** = a high >= poke + 5pt within 5 bars. Stamp = that bar.
- **outcome**, 90min: CRACK if low <= poke - 5pt before high >= poke + 15pt; HOLD on the reverse.
- `day_class` is causal: flushV only if the flush confirmation ts (imported from the reversal_gauge builder) is <= the stamp.

### Prevalence

- 1585 events on 444 of 540 trading days; 1583 decided (CRACK/HOLD), 2 unresolved
- shelf dwell fraction: median +9.17% [q25 +8.33, q75 +10.83] (n=1585)
- defense bounce: median +11.50pt [q25 +7.75, q75 +17.75] (n=1585)

### TABLE — crack vs hold BY day-class

| day_class | N | CRACK | HOLD |
|---|---|---|---|
| flushV | 225 | 88 (39.1%) | 137 (60.9%) |
| other | 1358 | 505 (37.2%) | 853 (62.8%) |

- **p(CRACK), flushV vs other**: flushV 39.1% (n=225) vs other 37.2% (n=1358) -> delta +1.9% 95% CI [-4.9%, +9.0%] -> not significant (CI includes 0)

### Reproducing the vshape 1.4% (or explaining the gap)

`vshape_retest_cohort.py` reported **CRACK 1/72 = 1.4% [0%, 7%]**. Restricting THIS detector to the nearest matching sub-cohort — flushV day-class, FIRST event of the day, trigger between 10:00 and 12:30 — gives:

- **CRACK 30.8% [18.6%, 46.4%] (n=39)**

Why the generalised number is higher: vshape's shelf is a specific construct (modal close inside the lower 45% of the flush range, computed over 09:30-10:05, i.e. the flush-consolidation dwell) tested at the FIRST retest after the V-recovery peak, with the outcome window additionally truncated at 12:30. This detector's shelf is any 2-hour dwell mode anywhere in the session, so it samples ordinary intraday shelves that carry no V-floor memory. **The 1.4% is a property of the V-floor shelf specifically, not of defended pokes at shelves in general** — which is the useful finding: generalising the event destroys the edge.

### Causality self-audit

- The dwell histogram reads `close[i-120:i]` — strictly prior bars, never including the trigger bar.
- The poke extreme and the defense high are read over bars AT AND AFTER the trigger, and the stamp is the defense bar itself, so no condition uses a bar later than the stamp.
- `day_class` compares the stamp ts against the imported causal flush confirmation ts; it is never applied retroactively to earlier events on the same day.
- The trigger scan is bounded to RTH on both sides; the 2h lookback may reach into the same file's pre-open and prior-evening bars, which is legitimate same-contract tape (ATLAS day files are per-day outrights, so there is no roll seam inside a file).

**Verdict:** **FUZZY across day-class** — crack rate is ~37% on any high-dwell shelf and flushV days are not distinguishable.

---

## 6. FLUSH_V_DAY

### Definition (reproducible)

**Imported, not reimplemented**: `_flush_confirm_ts` from `research/reversal_gauge/builders/extract_freeze_events.py` — the FIXED detector, whose AUDIT FIX comment documents the prior-evening bug (unbounded `mod >= X` matched 18:00 bars first, mislabelling flushV on 167/600 days and killing the window-closed guard on 992 events). Importing rather than copying is deliberate: this detector must not be able to drift from its audited version.

- flush: 09:30 open minus min low over [09:30, 09:50) >= 60pt;
- recovery: a high after the min-low bar reaching low + 60% of the flush at or before 10:20;
- confirmation ts = max(recovery bar, first bar at/after 09:50) — the day is only KNOWABLY flushV once both have printed.

**Control**: every non-flushV day contributes one row at the 10:20 recovery deadline with the identical V-low / V-peak construction, so the day-class comparison is same-construction rather than free-floating.

### Prevalence

- 540 scored days of 540 trading days (days whose 09:30-09:50 window or 10:20 anchor is missing are dropped); **flushV = 136 (25.2%)**
- flush size, flushV days: median +92.50pt [q25 +75.00, q75 +119.25] (n=136)
- recovery fraction at confirm, flushV days: median +84.16% [q25 +65.48, q75 +125.99] (n=136)

### TABLE — what the day does after the class is knowable

| metric | flushV (n=136) | control (n=404) | delta 95% CI | sig |
|---|---|---|---|---|
| V-low broken later | 70.6% [62.4%,77.6%] | 63.6% [58.8%,68.2%] | [-2.3%, +15.8%] | not significant (CI includes 0) |
| V-peak reclaimed later | 86.8% [80.0%,91.5%] | 73.8% [69.3%,77.8%] | [+5.6%, +20.1%] | SIGNIFICANT |

First event after confirmation:

| cls | N | PEAK_RECLAIM | LOW_BREAK | NEITHER |
|---|---|---|---|---|
| control | 404 | 200 (49.5%) | 203 (50.2%) | 1 (0.2%) |
| flushV | 136 | 99 (72.8%) | 37 (27.2%) | 0 (0.0%) |

- RTH close position in [V-low, V-peak], flushV: median +0.81 [q25 -0.91, q75 +1.79] (n=136)
- RTH close position in [V-low, V-peak], control: median +0.72 [q25 -0.47, q75 +1.36] (n=404)

### Causality self-audit

- Detector imported verbatim from the audited source; confirmation ts is the later of the recovery print and the flush-window close, so the label is never available before both facts exist.
- V-peak is recomputed as the max high from the flush-low bar THROUGH the confirmation bar only — not the day's eventual peak.
- The control anchor uses a BOTH-SIDES-BOUNDED mask `[10:20, 11:20)`; an unbounded `mod >= 10:20` would have selected prior-evening bars, which is exactly the audited bug class.
- Outcome scans stop at 16:00 ET and never reach the file's evening session.

**Verdict:** **SHARP as a day-class label** — flushV days reclaim the recovery peak far more often than matched control days. Note this is a DAY-CLASS label, and the sharpness is partly definitional: a flushV day has, by construction, already recovered 60% of its flush by 10:20, so 'peak reclaimed later' is measured against a peak that momentum just produced. It is still the one event here whose conditioning moves a day-scale outcome well outside its control CI.

---

## What this says for the teacher-student target

- **Detection is easy; discrimination is not.** All six states are cheaply and causally detectable at scale (18k-153k instances). Five of six move the direction-outcome distribution by roughly nothing once the distance asymmetry is removed. This is the same wall the program has hit repeatedly (oscillator-vs-runaway stuck ~0.57 AUC; 'every mechanical decider loses').
- **The one non-trivial conditional found here is level-flavoured, not direction-flavoured**: FAKEOUT_POKE's snap-back cuts p(the level clears) from ~91% to ~67%. Its own +-10pt DIRECTION race moves ~1pp over the same split — the information is about the LEVEL's fate, not about which way price goes, and that is exactly what a table-lookup layer can serve and a direction classifier cannot.
- **Generalising a sharp event destroys it.** DEFENDED_POKE_AT_SHELF reproduces nothing like the V-floor's 1.4% crack rate once the shelf is any dwell mode. If a cohort table is sharp, the sharpness lives in the SPECIFICITY of the construct, so the event vocabulary must keep day-shape context attached rather than abstracting it away.
- **Every table here is a base-rate table, and most base rates are ~50%.** A student trained to classify these events and look up the table would output the base rate. The value of v0 is negative information: it prices out five candidate features before anyone spends GPU on them.

