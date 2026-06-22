# GA-Kalman OOS failure anatomy (where did it bleed?)

## OOS (H2-24 + 25-26)  (N=4833 trades, 465 days)
- net total $4,672  | $/day +10.0  | $/trade +0.97
- winners 1729 (sum $264,562) | losers 3104 (sum $-259,890)
- **big stop-outs** (≤$-90): 1938 trades, sum **$-203,825** (40% of trades, -4363% of net)
- **scratch/chop losers** (>-90 & ≤0): 1166 trades, sum $-56,065
- **exit giveback on winners**: $248,124 left on table vs MFE peak (winners kept 52% of peak)
- **chop entries** (MFE<10pt, never developed): 855 = 18% of trades, net $-86,646
- direction: LONG net $3,498 | SHORT net $1,174

- worst entry-hours (NY approx): 11h $-4,944(388t), 08h $-4,556(365t), 10h $-4,386(553t), 13h $-2,571(279t)
- losing-day concentration: worst 10 days sum $-13,517 of $-78,604 total loss (17%)

## IS (H1-24, reference)  (N=630 trades, 128 days)
- net total $7,406  | $/day +57.9  | $/trade +11.76
- winners 252 (sum $37,782) | losers 378 (sum $-30,375)
- **big stop-outs** (≤$-90): 221 trades, sum **$-23,042** (35% of trades, -311% of net)
- **scratch/chop losers** (>-90 & ≤0): 157 trades, sum $-7,333
- **exit giveback on winners**: $29,011 left on table vs MFE peak (winners kept 57% of peak)
- **chop entries** (MFE<10pt, never developed): 101 = 16% of trades, net $-9,896
- direction: LONG net $4,874 | SHORT net $2,532

## Read
- chop entries (no move) = 18% of OOS trades → ENTRY-side leak if high.
- exit giveback = $248,124 → EXIT-side opportunity (risk-reduction, not new edge).
- big stop-outs cost $-203,825 → STOP-side if dominant.
Pick the lever from the LARGEST concrete leak above — pre-commit + OOS + CI before building.