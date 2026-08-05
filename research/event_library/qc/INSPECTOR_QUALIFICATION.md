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
