# False-profit vs catastrophic tail — holding WRONG (underwater≥20pt) trades
49,637 wrong trades (combiner entries that went ≥20pt adverse), full ATLAS. Never-bail to EOD. Points ($2/pt).

- Recovery rate: **72.2%** (35,814/49,637)
- **False profit when it recovers** (max favorable past entry): mean +111.4pt ($+223), median +66.8pt
- **Non-recovery tail** (never recovers): mean terminal -116.0pt ($-232), median -77.8pt, worst -2055.2pt ($-4110)

## Never-bail EOD outcome distribution (the skew the mean hides)
- mean -26.0pt ($-52) | median -25.5pt
- **left tail**: p25 -90.5 | p5 -262.5 | p1 -468.9 | worst -2055.2pt ($-4110)
- catastrophe rate: 37.7% lose >50pt, 22.7% lose >100pt, 8.7% lose >200pt

## Hold-for-recovery vs cut-at-threshold
- cut at 20pt: every wrong trade = -20pt ($-40), NO tail
- never-bail mean: -26.0pt — worse than cutting, but carries a p1 of -469pt and worst -2055pt.

Read: the false profits are small + frequent; the non-recovery tail is deep + rare = negative skew. Even where never-bail wins on the MEAN, the catastrophic tail (p1/worst) is the real exposure — "hold for recovery" harvests pennies in front of the steamroller. Confirms the owner: the hold premise for wrong trades is catastrophic.
