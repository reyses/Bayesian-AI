# Inspector qualification — can a cheaper model inspect the corpus?

Owner asked (2026-08-04) whether haiku can run the AQL inspection, and how
capable it is. Answered with a KNOWN-ANSWER sample rather than an opinion.

## The test

100 real `fakeout_poke` events, **12 deliberately corrupted** in 4 types
(price field shifted 3.25pt / depth made inconsistent / kind label flipped /
timestamp shifted 45s). Answer key withheld from the inspector, which was not
told that any rows were corrupt or how many.

## Haiku result

| metric | value |
|---|---|
| flagged | 47 of 100 |
| true positives | 8 |
| **recall** | **67%** (8 of 12) |
| **precision** | **17%** (39 false positives) |
| missed | 3 timestamp_shifted, 1 kind_label_flipped |

Caught all 3 price-field corruptions and all 3 depth inconsistencies —
the arithmetic checks. Missed 3 of 3 timestamp shifts, which need a
structural judgement about whether the event still holds.

## The corroborating failure on live data

On the un-seeded batch the same model reported **16 defects in
`exceed_ref_first`**, claiming the recorded label was wrong. Verified by hand
against `outcomes.py` line 162: the definition tests **closes**
(`d.close`), and haiku tested **bar highs**. Its flagship example
(2024_05_29, ts 1717008975) resolves to `exceed_ref_first = False` exactly as
recorded — first close >2pt past ref at +1140s, first 10pt adverse close at
+40s. **All 16 were false alarms from misreading the specification.**

## Verdict

**Not qualified as a sole inspector for this job.** 17% precision means five
of every six alarms are noise, and a 16-defect false report against a
load-bearing table would have sent the program chasing a bug that does not
exist — precisely the failure mode QC exists to prevent.

**Qualified as a first-pass screen.** It reliably catches arithmetic
inconsistencies (6 of 6 here) and is ~10x cheaper. Use it to triage, and
have a stronger inspector adjudicate everything it flags. Never let it
accept or reject a lot.

## Standing use

`seeded_fakeout_b0.csv` + `seeded_ANSWER_KEY.json` are now a permanent
proficiency test: any new inspector or model upgrade runs it first, and the
recall/precision go on the record before that inspector's verdicts are
trusted.

## Retest with a tightened brief — and the brief was partly MINE to blame

| brief | flagged | recall | precision | false alarms |
|---|---|---|---|---|
| v1 loose | 47 | 67% (8/12) | 17% | 39 |
| v2 tight (close-only rule stated, checks numbered, "don't flag when unsure") | 42 | **83% (10/12)** | 24% | 32 |

Recall rose sharply, precision barely moved — and the reason is in MY spec,
not the model. 36 of 42 flags were on `kind`, where only 3 defects were
planted.

**The check-5 rule I wrote does not match the code.** I briefed from the
docstring — "RETURN = returns inside within 60s" (`POKE_RETURN_S = 60`,
line 164) — but the actual branch (lines 232-236) is:

```
if   over > POKE_MAX_PT:                      kind = "BREAKOUT"
elif dd * (c - poke_ref) < 0:                 kind = "RETURN"   # NO time bound
elif ts[i] - ts[poke_arm_i] > POKE_RETURN_S:  kind = "STUCK"
```

RETURN is priority-checked **before** the 60s test and carries no bound of
its own; the 60s only gates STUCK. A return at +90s is still labelled RETURN.
The sonnet inspector caught this independently and logged it as
`SPEC_VIOLATION_RETURN_LATE` (0 occurrences in its batch, so not a defect
there — but the docstring is wrong).

So the fair verdict on the inspector changes:

- A tight brief lifts haiku's recall from 67% to 83% — briefing quality
  matters more than the model on this task.
- Most of its "false alarms" were faithful executions of an instruction that
  was itself wrong. Blaming the inspector for that would have been the easy
  and incorrect conclusion.
- Precision on the checks I specified CORRECTLY (arithmetic, checks 2/3/4)
  remains its strength: 6/6 planted arithmetic defects caught in v1.

**Repo defect recorded**: `detectors.py:164` docstring and lines 232-236
disagree about whether RETURN is time-bounded. The code is authoritative;
the comment should be fixed or the bound implemented — a decision for the
owner, since it changes the RETURN/STUCK split.
