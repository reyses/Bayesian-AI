# ADVERSARIAL AUDIT — Bayesian Table Actuary v0

Auditor: independent recomputation from `research/event_library/events/*.parquet`.
Nothing below reads `build_tables.py` output as truth; every number was recounted
with separately written code (scratchpad scripts `aud_cov.py`, `aud_recompute.py`,
`aud_boot.py`, `aud_fdr_circ.py`, `aud_prior_ties.py`, `aud_actuary.py`,
`aud_actuary2.py`, `aud_semantics.py`, `aud_stall_uc.py`, `aud_dominated.py`,
`aud_final.py`).

**Verdict: the arithmetic is clean. The statistics are clean-to-conservative.
The SEMANTICS are broken.** Three of the four headline claims are restatements
of barrier geometry the outcome definition itself imposes, and the flagship NULL
claim is false.

---

## Scorecard

| # | Item | Verdict |
|---|------|---------|
| a | Recompute headline cells | **CLEAN** |
| b | Coverage / silent drops | **CLEAN** (one dead line: `EXCLUDE_DAYS` is a no-op) |
| c | Day-clustered bootstrap | **CLEAN** implementation; **FLAW** in what it is attached to |
| d | Benjamini-Hochberg + p-value floor | **CLEAN** |
| e | Shrinkage prior PRIOR_STRENGTH=20 | **FLAW** — decorative; zero effect on any decision |
| f | Circularity of the global-rate prior | **CLEAN** (self-consistent, mildly conservative) |
| g | `actuary.py` lookup / backoff / pooling | **FLAW** (4 distinct defects) |
| h | The NULL claims | **FLAW — the ultra_chop null is FALSE** |
| — | Semantics of the headline effects (unasked, decisive) | **FLAW** |

---

## a. Cell arithmetic — CLEAN

Rebuilt all six tables from the raw parquets with independently written bucketing.
Merged on the dimension keys; every cell matched:

| table | cells | max abs diff `n`/`hits`/`days` | max abs diff `raw`/`post`/`lo`/`hi` |
|---|---|---|---|
| fakeout_poke / exceed_ref_first | 198 | 0 / 0 / 0 | ≤ 5.0e-5 (rounding to 4dp) |
| fakeout_poke / sym_race | 70 | 0 / 0 / 0 | ≤ 5.0e-5 |
| leg_descent / race | 15 | 0 / 0 / 0 | ≤ 5.0e-5 |
| stall / race | 10 | 0 / 0 / 0 | ≤ 4.4e-5 |
| ultra_chop / escape_dir | 15 | 0 / 0 / 0 | ≤ 4.8e-5 |
| defended_poke_shelf / outcome | 10 | 0 / 0 / 0 | ≤ 4.9e-5 |

Global rates reproduce exactly: 0.782218 / 0.497683 / 0.689808 / 0.103303 /
0.508521 / 0.374132.

Three strongest fakeout cells, recounted from raw events:

```
RETURN dn 1-2   30m+  0930  n=475 (claim 475)  hits=267  raw=0.5621  post=0.5710 (claim 0.5710)
RETURN up 0.5-1 5-30m 0930  n=428 (claim 428)  hits=247  raw=0.5771  post=0.5863 (claim 0.5863)
RETURN dn <=0.5 5-30m 0930  n=414 (claim 414)  hits=242  raw=0.5845  post=0.5937 (claim 0.5937)
```

Bucketing edge cases: no off-by-one that changes a rate. One **labelling** off-by-one:
`pd.cut` is right-closed, so minute-of-day 600 (10:00 ET) lands in the bucket
*labelled* `0930`, 630 (10:30) in `1000`, 720 (12:00) in `1030`, 840 (14:00) in
`1200`. Actual bucket spans are `0930`=09:30–10:00, `1000`=10:01–10:30,
`1030`=10:31–12:00, `1200`=12:01–14:00, `1400`=14:01–close. 2,013 of 153,029
fakeout events (1.3%) sit on those boundary minutes. Given the repo's verified
10:00 ET volatility peak, putting the 10:00 minute in the `0930` bucket is a
semantic error, but the rate impact is ≤0.4pp on one bucket. `1400` for stall
formally spans to 18:00 ET; only 13 of 9,534 rows are ≥16:00, so no practical
contamination.

## b. Coverage — CLEAN

For every one of the six tables, `sum(table.n)` equals the number of source rows
surviving `outcome.notna()` **exactly**, and the number of NaN values in every
context dimension is **zero**:

```
fakeout_poke  153,029 rows -> 153,029 covered (198 cells)   0 dropped
leg_descent    58,480      ->  58,480        (15)           0 dropped
stall          41,180      ->  41,180        (10)           0 dropped
ultra_chop     18,601      ->  18,601        (15)           0 dropped
defended_shelf  1,585      ->   1,585        (10)           0 dropped
```

The `dropna=True` in `groupby` is therefore currently inert — but it is a live
landmine: `depth_b` tops out at 1e6, `age_b` at 1e9, `ratio_b` at **0.61** and
`chain_b` at **99**. Observed maxima are 81, 5440, 0.600 and 19. `ratio_b` has
1.7% headroom; any rebuild on a wider corpus silently deletes rows again with no
warning. There is no coverage assertion in the builder.

`EXCLUDE_DAYS = {'2024_09_16'}` removes **0 rows from all five event files** —
that day is not in the corpus (`2024_01_02` … `2026_03_19/20`). The report line
"Live sim day excluded." is vacuous. Either the sim day has a different key or
the exclusion never applied.

Minor semantics: `stall.race` contains 13 `NO_DATA` and 3 `NEITHER` rows that are
counted as *misses* for `NEW_EXTREME`; `fakeout_poke.sym_race` has 102 `NEITHER`
counted as misses. Below noise, but "no data" is being scored as a negative.

## c. Day-clustered bootstrap — implementation CLEAN, attachment FLAWED

It genuinely resamples **days**: `g.groupby('day')` → sample day indices with
replacement → ratio of summed hits to summed counts. Correct cluster bootstrap of
a ratio estimator.

Reran at BOOT=20,000 with an independent seed. Max disagreement vs shipped bounds:

| table | max abs Δ day_lo | max abs Δ day_hi | cells over 0.02 |
|---|---|---|---|
| fakeout / exceed_ref_first | 0.0111 | 0.0028 | 0 |
| fakeout / sym_race | 0.0108 | 0.0052 | 0 |
| leg_descent | 0.0007 | 0.0005 | 0 |
| stall | 0.0003 | 0.0005 | 0 |
| ultra_chop | 0.0051 | **0.0227** | 1 (n=45 cell, Monte-Carlo noise) |
| defended_shelf | 0.0011 | 0.0026 | 0 |

The three strongest cells agree to ≤0.0021 on both bounds. **No disagreement >0.02
on any cell that matters.**

FLAW — **the interval and the point estimate are different estimators.** `day_lo`/
`day_hi` bracket the **raw** rate; the table reports **`post`** (shrunk) as the
point. 286 cells have an interval; in 1 of them `post` falls outside its own
reported CI:

```
BREAKOUT dn 15+ 30m+ 1400   n=15  days=9  raw=1.0000  post=0.8756  day_CI=[1.0000,1.0000]
```

That same cell is the concrete **degenerate-bootstrap** failure: all 9 days are
100%, so every bootstrap draw equals 1.0, the interval has **zero width**, the
empirical p collapses to the floor 0.00025 — the *smallest possible p in the
table* — and the cell is flagged ACTIONABLE on n=15 with the same nominal
confidence as an n=9,133 cell. A zero-variance cluster bootstrap is not evidence
of certainty; it is evidence of too few clusters. Six more actionable cells sit at
n=24–47 (all `BREAKOUT … 15+`).

## d. Benjamini-Hochberg — CLEAN

Implementation `passed = flatnonzero(pv <= q*i/m); reject order[:passed[-1]+1]` is
textbook step-up BH. Verified against three hand-built cases:

- `p=[.001,.008,.039,.041,.042,.60]`, m=6, q=.05 → crit `[.00833,.01667,.025,.0333,.04167,.05]`;
  largest passing index = 2 → reject 2. My reimplementation: `[T,T,F,F,F,F]`. Correct.
- `p=[.01,.02,.03,.04,.05]`, m=5 → all reject (p₅=.05≤.05). Correct.
- Unsorted `p=[.04,.005,.9,.5,.02]` → reject indices 1 and 4 only. Correct
  (confirms the step-up rule takes everything up to the largest passing rank, not
  the first failure).

Re-running BH over the shipped `p` column reproduces the shipped `actionable`
column **exactly** in all six tables (158 / 2 / 6 / 8 / 0 / 2).

The 1/BOOT = 2.5e-4 floor does **not** distort the ranking. It is inert because BH
is a step-up procedure: the fakeout cutoff lands at rank 158 (crit = 0.0399),
1,600× above the floor. There is no tie straddling the cutoff in any table
(`p[k-1]=0.025`, `p[k]=0.0405` for fakeout; checked all six). It is *fragile*
though: crit(1) for the fakeout table is 0.05/198 = 2.525e-4, only 1% above the
floor. Drop BOOT to 3,900 or add 3 cells and the rank-1 comparison inverts. The
floor also renders `p` uninformative for the "strongest cells" ranking — 148 of
198 cells share the identical value 0.00025.

Two conservative-direction quibbles, both safe: (i) 20 cells with n<15 are given
p=1.0 but still counted in m, inflating the BH denominator by 11%; (ii) cells are
disjoint partitions all compared to a shared `glob`, which is negative dependence,
where BH strictly needs PRDS. The joint re-test in (f) shows this is not producing
false positives.

Doc/code mismatch: the module docstring says *"A cell is only ACTIONABLE if its
95% CI excludes the global rate"* — that is not the implemented rule. The report
says it survives *"BOTH a day-clustered bootstrap AND Benjamini-Hochberg"* — that
is one criterion, not two (the BH p *is* the bootstrap). The docstring's stated
rule would give **8 actionable cells for sym_race instead of 2** and 155 vs 158 for
exceed_ref_first. Three documents, three different rules.

## e. Shrinkage prior — FLAW: it is decorative

**PRIOR_STRENGTH has zero effect on the `actionable` flag.** `actionable =
BH(p_emp) & (n ≥ 15)`, and `p_emp` comes from the day bootstrap of the **raw**
rate, which never touches the prior. Re-ran the flag at PRIOR_STRENGTH ∈
{5, 20, 50, 200}:

```
                              PS=5   PS=20  PS=50  PS=200
fakeout / exceed_ref_first     158    158    158    158
fakeout / sym_race               2      2      2      2
leg_descent                      6      6      6      6
stall                            8      8      8      8
ultra_chop                       0      0      0      0
defended_poke_shelf              2      2      2      2
```

**Zero cells flip at any prior strength.** The docstring's claim that shrinkage
means "a thin cell is pulled toward the base rate rather than shouting from 3
samples" is false as a description of the decision: the decision is made on the
un-shrunk rate. The thin `n=15, raw=1.0` cell in (c) shouts as loudly as it
possibly can.

What the prior *does* do is attenuate the **reported** effect size, most for the
cells the reader is least able to judge. Mean attenuation of `|lift|` from raw to
post at PS=20:

```
fakeout / exceed_ref_first:  all cells 16.4%   thin (n<50) 60.2%
fakeout / sym_race:          all 14.9%         thin 80.8%
ultra_chop:                  all 22.7%         thin 65.9%
defended_poke_shelf:         all 26.2%         thin 62.4%
```

So the table is tested on one number and reports another, understating the very
cells whose significance is most fragile. (Direction is safe — `post` is a convex
combination of `raw` and `glob`, so no sign can flip.) If PRIOR_STRENGTH were
raised to 50 to genuinely protect thin cells, it would still flip nothing; only a
CI-based rule (which the docstring claims and the code does not implement) is
prior-sensitive.

## f. Circularity of the global-rate prior — CLEAN, mildly conservative

The concern is real in form: `glob` is computed from the same rows as the cell,
and the bootstrap resamples days *within the cell* while holding `glob` fixed at
its full-sample value with zero uncertainty.

I ran the correct test — resample **days from the whole event corpus**, recompute
the cell rate and the global rate on the *same* resample, and take the CI/p on the
difference `cell − glob`:

| table | shipped actionable | joint-test actionable | lost | gained | mean CI width ratio (joint/shipped) |
|---|---|---|---|---|---|
| fakeout / exceed_ref_first | 158 | 158 | 0 | 0 | 1.000 |
| fakeout / sym_race | 2 | 2 | 0 | 0 | 0.994 |
| leg_descent | 6 | **8** | 0 | +2 | 0.963 |
| stall | 8 | 8 | 0 | 0 | 0.915 |
| ultra_chop | 0 | 0 | 0 | 0 | 0.964 |
| defended_poke_shelf | 2 | **3** | 0 | +1 | 0.955 |

The self-consistent test is **narrower**, not wider — cell and global co-move
under day resampling, so the difference has lower variance than the cell alone.
The shipped procedure therefore **understates power by 0–9%**, not uncertainty. It
loses no discovery; it misses 3 (2 in leg_descent, 1 in defended_poke_shelf).
Effect size: negligible for the 153k-row table, ~9% CI inflation for stall.

## g. `actuary.py` — FLAW ×4

**g1. Pooled Beta recomputation is exact.** Forced three pooled answers and
recounted from raw events: `n` matched exactly (10,202 / 10,679 / 29,785) and the
posterior matched to <1e-6 (0.6498 / 0.6719 / 0.9035). CLEAN.

**g2. The backoff drops valid dimensions before the offending one.** `BACKOFF_ORDER`
is a fixed list; when a lookup misses, it pops `age_b` first regardless of *which*
dimension caused the miss. The repo's own demo case:

```
lookup(kind='BREAKOUT', dir_s='up', depth_b='1-2', age_b='<5m', clock_b='1200')
  -> 91% [91%,91%] n=39409 — not separated (dropped age_b, clock_b, depth_b)
```

`depth_b='1-2'` is the impossible value (see the semantics section: BREAKOUT ⇒
depth ∈ {2-5, 5-15, 15+}, zero rows). The correct backoff drops `depth_b` alone
and lands on n=2,424 with post **0.9213**. Instead it discards two perfectly valid
conditions and answers from a pool 16× larger. Same defect in demo case 3
(`clock_b='2359'` → drops `age_b` *and* `clock_b`).

Worse, the drop order is not even calibrated: it drops `age_b` (marginal rate
spread **0.0141**) before `dir_s` (spread **0.0161**) — but `dir_s` is a coin flip
on this outcome (up 0.790 / dn 0.774) while `depth_b` (spread 0.2735) is the whole
signal and is dropped *third*.

**g3. Pooled answers report an iid Beta interval and a claim FDR never tested.**
The cell branch returns the day-clustered interval; the pooled branch returns
`stats.beta.ppf` on summed counts — iid, no clustering. On the demo pooled cell
`RETURN/dn/<=0.5` (n=10,202) the actuary reports [0.6406, 0.6591] (width 0.0185)
where the day-clustered interval is [0.6389, 0.6606] (width 0.0217) — **1.18×
too narrow**, and the two are surfaced through the same `Answer.lo/.hi` field
with no marking.

Hold-one-cell-out replay (178 cells, n≥15: delete the cell, ask the actuary what
it would answer):
- mean |backoff − truth| = **0.0166**, max **0.0712**
- 3 sign contradictions (backoff on the opposite side of global from the true
  cell) — all small `BREAKOUT 15+` cells, none flagged actionable
- **3 cases where the pooled answer is flagged ACTIONABLE while the true cell is
  not**: `BREAKOUT dn 15+ 30m+ 1030` (true post 0.8414, NOT actionable → pooled
  0.8752 ACTIONABLE), `BREAKOUT up 15+ 5-30m 1200` (0.8259 → 0.8771), `BREAKOUT up
  15+ 30m+ 0930` (0.8435 → 0.8808). The `all()` guard is satisfied because the
  surviving constituents happen to be actionable; the pooled aggregate itself was
  never an FDR test unit.
- worst mispricing where both are actionable: **`RETURN dn 1-2 30m+ 0930` — the
  #1 headline cell.** True post 0.5710; if that cell were absent the actuary
  returns **0.6422** (pool over `age_b`, n=1,398), still labelled ACTIONABLE. A
  7.1pp error on the strongest claim in the report.

The `all()` guard does hold in the specific sense the code claims: I searched every
one-dimension pooling of all six tables and found **0** groups that are
all-actionable with mixed sign. That specific failure mode is not present today.

**g4. `days` on a pooled answer is nonsense.** `int(sel['days'].sum())` sums
per-cell distinct-day counts, double-counting every session. Demo case 3 reports
`days=4640`; the corpus has **539 distinct days**. Demo case 2 reports
`days=10625` — 19.7× the truth. Anything downstream that sizes confidence by
`days` is off by an order of magnitude.

**g5. `basis='BASE'` is unreachable dead code.** The loop's terminal state is
`use={}`, which selects the entire table (len ≥ 1) and returns POOLED. The
docstring's promise "If nothing survives, return the global rate labelled BASE"
never fires: `lookup(event, question, kind='NOPE')` returns
`78% n=153029 — not separated`, basis **POOLED**. (Its n-weighted post is 0.78249
vs the true global 0.78222 — the n-weighted mean of shrunk posts is not the global
rate.)

## h. The NULL claims — the ultra_chop null is FALSE

**Recomputed:** P(escape_dir==1) = 9,459/18,601 = **0.50852**. Day-clustered 95%
CI on the *global* rate = **[0.5011, 0.5158]**, which **excludes 0.500**. Every
cell is tested against 0.5085, not against 0.5, so "0 of 15 actionable" means "no
cell differs from the corpus's own up-bias" — it does not establish chance. The
day-level overdispersion test is negative (χ²=562.3, df=528, p=0.146), so the
clustering is honest here.

**The null is an artifact of the two dimensions chosen.** `clock_b × ratio_b` are
the two weakest fields available. Using `mid_px`, `box_lo`, `box_hi` — all stamped
causally at detection, all already in the parquet — the direction is strongly
predictable:

```
P(escape UP) by quintile of (mid_px - box_lo)/(box_hi - box_lo), global 0.5085
  q0  pos<=0.286   n=3786  days=520  P(up)=0.3637  day-CI=[0.3484,0.3793]  lift -0.1448  excludes global
  q1  <=0.450      n=3687  days=511  P(up)=0.4486  day-CI=[0.4336,0.4638]  lift -0.0599  excludes global
  q2  <=0.600      n=3866  days=519  P(up)=0.5026  day-CI=[0.4868,0.5186]  lift -0.0059  —
  q3  <=0.750      n=3623  days=521  P(up)=0.5783  day-CI=[0.5623,0.5938]  lift +0.0697  excludes global
  q4  <=1.000      n=3639  days=520  P(up)=0.6568  day-CI=[0.6407,0.6727]  lift +0.1483  excludes global
```

Spread **0.293**, AUC **0.620** (0.613 using the signed barrier distance
`(box_hi+esc_buf-mid) − (mid-box_lo+esc_buf)`). Four of five quintiles have
day-clustered intervals excluding the global rate, with n≈3,700 and days≈520 each.
For scale: the largest |lift| in the *entire* shipped fakeout table is 0.2112 on
n=475.

A blind sweep of alternative bucketings confirms this is the only real one
(day-clustered p, Bonferroni-adjusted across bins within each sweep):

```
30-min clock            9 bins   min p 0.0032  adj 0.029
ambient-ratio quintile  5 bins   min p 0.1608  adj 0.804
flips quartile          2 bins   min p 0.0500  adj 0.100
box_pt quartile         4 bins   min p 0.0257  adj 0.103
escape_lag quartile     4 bins   min p 0.0462  adj 0.185
30min x ratio-quintile 43 bins   min p 0.0145  adj 0.624
mid-in-box quartile     4 bins   min p 0.0001  adj 0.001   <-- q0: n=4883, P(up)=0.3799
```

**Conclusion: "direction carries nothing, level/structure does" is not supported.**
Direction carries a great deal; the table simply did not condition on the field
that carries it. The correct statement is: *escape direction is not predictable
from time-of-day or box tightness, but is strongly predictable from where price
sits in the box.*

The `fakeout / sym_race` null (2 of 70) I did **not** overturn — the two survivors
(BREAKOUT up 2-5 1400 at −0.0297, RETURN up 1-2 1030 at +0.0265) are tiny lifts on
n≈6,000, and the marginal `dir_s` spread on that outcome is negligible. That null
stands as reported.

---

## The unasked question: are the positive findings real?

All four positive headlines are dominated by barrier geometry that the outcome
definition itself creates, and in three of four the shipped context dimensions are
strictly weaker than a causal field already sitting in the parquet.

### 1. `kind` and `depth_b` are the same variable

The detector sets `kind = BREAKOUT` iff `poke_depth > POKE_MAX_PT = 2.0`. Crosstab:

```
depth_b   <=0.5  0.5-1    1-2    2-5   5-15   15+
BREAKOUT      0      0      0  54259  19090   883
RETURN    21348  20213  37170      0      0     0
STUCK         0      2     64      0      0     0
```

Zero overlap. `kind × depth_b` is not a 2-D grid, it is one 6-level variable plus a
66-row `STUCK` label. The "198 cells" are ~180 real combinations plus 18 STUCK
artefacts, and the redundancy inflates the BH denominator.

### 2. The BREAKOUT/RETURN gap is the 10-point kill barrier, not "clearing the level"

`exceed_ref_first` = clears ref by >2pt **before** a 10pt adverse move from the
emission close. Splitting the bounded outcome from the unbounded one:

```
kind        n       P(ever clears)  P(clears first)  lost to the 10pt barrier
BREAKOUT  74232        0.9868          0.9050              0.0818
RETURN    78731        0.9182          0.6664              0.2518
```

A BREAKOUT is *by construction* already >2pt past the reference at emission — the
trigger is behind it and the killer is 10pt away. A RETURN is *by construction*
back inside — the trigger is >2pt ahead and the killer is 10pt away. The 24pp gap
is the geometry of where the emission close sits between two barriers, imposed by
the event definition. It is not a statement about pokes.

### 3. The "09:30 pokes are weak" headline is the volatility clock

```
clock   n       P(ever clears)  P(clears first)  lost to barrier   median resolve_s
0930  16427        0.9525          0.7453           0.2072              10
1000  16581        0.9536          0.7615           0.1920              15
1030  45284        0.9527          0.7846           0.1681              15
1200  44056        0.9504          0.7981           0.1523              20
1400  30681        0.9496          0.7867           0.1628              20
```

"Ever clears" is **flat to 3 decimal places across the session** (0.9496–0.9536).
The entire clock gradient lives in the barrier-race term, which tracks tape speed
(median resolve 10s at the open vs 20s midday) and REVERSE share (0.298 → 0.227).
The 158 ACTIONABLE cells are re-deriving the already-verified 9:30/10:00 ET
volatility peak. That is a true fact; it is not new information and it is not a
property of pokes.

### 4. The shipped dimensions are dominated by causal fields already in the parquet

Rate spread across the shipped grid vs across quintiles of an unused, causally
stamped field (all confirmed emitted at the event bar, no lookahead):

| event | shipped grid spread | best unused field | its spread |
|---|---|---|---|
| stall / NEW_EXTREME | 0.0936 (dir × clock) | `give_frac` quintiles | **0.3409** |
| leg_descent / NEW_LOW | 0.0513 (chain × clock) | `defense_pt` quintiles | **0.3103** |
| defended_shelf / CRACK | 0.3096 (day_class × clock) | `bounce_pt` quartiles | **0.4279** |
| ultra_chop / escape_dir | 0 actionable | mid-in-box quintiles | **0.2931** |

For stall the mechanism is explicit. NEW_EXTREME needs a fixed +0.5pt past the
peak; GIVEBACK_50 needs 50% of MFE. At emission, distance-to-new-extreme is
`give_frac/100·mfe + 0.5` and distance-to-giveback is `(50−give_frac)/100·mfe`.
Their ratio is near-deterministic for the outcome:

```
NEW_EXTREME rate by decile of that ratio (n≈4,118 each):
  0.392, 0.193, 0.121, 0.105, 0.074, 0.058, 0.047, 0.029, 0.011, 0.003
```

A 120× range, versus the 2.8× range across the shipped 10 cells. Standardising the
clock gradient on that ratio shrinks it from raw 0.0606→0.1270 (2.10×) to
0.0722→0.1172 (1.62×): roughly 40% of the celebrated `up/1200 vs dn/1000` split is
barrier geometry, and the rest is conditioned on the wrong covariate. The table's
context dimensions are the weakest fields available.

---

## Recommendations (report only, nothing changed)

1. **Do not act on `exceed_ref_first` cells as a poke-quality signal.** Report the
   unbounded/bounded decomposition alongside every cell so the barrier-race
   component is visible. Consider a volatility-normalised adverse barrier
   (e.g. k·ATR instead of a fixed 10pt) before re-running.
2. **Retract the ultra_chop null.** Add mid-in-box position as a dimension and
   rerun; it is the strongest single effect anywhere in this corpus.
3. **Drop `kind` or `depth_b`** — they are the same variable. Drop or demote
   `dir_s` and `age_b` (marginal spreads 0.016 and 0.014); they inflate m from
   ~30 to 198 and cost real power under BH.
4. **Add the dominating causal fields**: `give_frac`/`mfe_pt` for stall,
   `defense_pt` for leg_descent, `bounce_pt` for defended_poke_shelf.
5. **Fix the actuary**: targeted backoff (drop the dimension that caused the miss),
   day-clustered interval on pooled answers, `days = nunique` not `sum`, and either
   implement the BASE branch or delete it.
6. **Gate degenerate cells**: require ≥ some minimum number of *discordant* days,
   not just n≥15 and days≥5, so a zero-variance bootstrap cannot produce the
   smallest p in the table.
7. **Reconcile the three stated actionability rules** (docstring / report / code)
   and add a coverage assertion so `dropna=True` can never silently delete rows.
