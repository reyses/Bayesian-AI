# Mid-Leg Entry / Missed-Signal Late-Join — Research Verdict

**2026-05-20 · autonomous sleep-run · 51 sealed OOS days · 2,926 legs**

## TL;DR — DO NOT BUILD

Mid-leg entry (late-joining an R-trigger the live engine missed) has **no
addressable population**. A 1-contract greedy engine already catches
**2,922 of 2,926** OOS legs; it misses **4** in 51 days — and all 4 are
losers. Hardened zigzag legs are a *sequential partition* of the price path
(leg N exits exactly where leg N+1 is born), so the engine is essentially
never "busy and missing a parallel signal."

| Exp | Test | Headline | Verdict |
|-----|------|----------|---------|
| E1 | Fork 1: B9-gated late join, **unconstrained** | +$303/day @ K=5, CI [+160,+457] | +EV — but the population does not exist |
| E2 | Fork 2: B1-B6 pivot-structure augmentation | −$76/day @ K=5, CI [−158,−2] | **rejected** (significantly negative) |
| E3 | **Position-constrained** 1-contract sim | incremental −$1/day, 4 late-joins / 51 days | **moot** |
| E4 | Leg overlap structure | 99.9% engine utilisation; 99.8% of gaps = 0 | explains E3 |

The "genuine lost signals" the user sees on the NT8 chart are **not**
busy-missed legs. They are **cold-start transients** (the engine's zigzag is
not primed at startup → it is blind until ~20-min warmup) or **deliberate B7
skips**. The fix is **zigzag-state priming**, not late-join.

---

## Background

The user observed lost signals on the live NT8 chart (`examples/example1.png`,
~9 swing legs over 5 h) and proposed: when the engine is flat and a missed
R-trigger's leg is still running, late-join it and capture whatever P&L is
left. Two forks were specified:

- **Fork 1** — gate the late-join with B9 (the during-trade remaining-amplitude
  regressor). A late-joiner at bar K captures
  `remaining_pnl_usd = exit_pnl_usd − pnl_usd_so_far`, which is *exactly* B9's
  training target — so B9 is in-distribution for this decision.
- **Fork 2** — augment with the B1-B6 pivot-structure models ("those carry
  the past state since zigzag is dumb fire").

Plus "other options." Five experiments were run, all on the 51-day sealed
OOS trajectory dataset, all metric-compliant (mode + mean + 95% bootstrap CI,
4000 resamples; significance = CI excludes 0; $6/leg friction primary).

---

## E1 — Fork 1: B9-gated late join (unconstrained)

`tools/forward_pass_midleg_entry.py` → `2026-05-20_midleg_fork1_b9.txt`

Every OOS leg treated as a late-join candidate at each K horizon. Principled
gate: join iff B9 predicted remaining > friction ($6). $6 friction.

| K (lateness) | unconditional join | **B9-gated** | 95% CI | verdict |
|---|---|---|---|---|
| 5 (25 s) | +$98/day | **+$303/day** | [+160, +457] | **significant** |
| 10 (50 s) | −$8/day | **+$232/day** | [+86, +388] | **significant** |
| 30 (150 s) | −$170/day | +$78/day | [−30, +202] | not sig |
| 60 (300 s) | −$231/day | +$50/day | [−40, +154] | not sig |
| 120 (600 s) | −$231/day | −$19/day | [−80, +46] | not sig |

**Reading.** The B9 gate is essential — unconditional late-join loses money
at every horizon (the legs are 58–70% losers). Gated, B9 sorts correctly:
at K=5 joined legs average $+16 remaining vs $−1 skipped. But the signal
decays hard with lateness; by 150 s late it is noise. Robust to friction
(K=5: $0→+$478, $6→+$303, $12→+$128).

**Critical nuance — the +$303 is not free money.** The baseline engine
*already* enters every leg at the R-trigger (K=0). E1's late-join at K=5
enters the *same* legs 25 s later — strictly worse than on-time entry. The
+$303 is the value of **B9-gating** (declining bad legs), measured on late
entries. It re-confirms that B9-at-K5 has discriminative power; it is not a
new revenue stream. (That discriminative power is already deployed — as the
B9 K=5 cut in the live L5 stack, which sizes the leg you are *in*.)

---

## E2 — Fork 2: B1-B6 pivot-structure augmentation

`tools/midleg_b1to6_augmented.py` → `2026-05-20_midleg_fork2_b1to6.txt`

Model A = GBM on B9's 190-feature set (reproduces B9). Model B = A's features
+ 18 B1-B6 prediction features. Both trained on full IS trajectory, evaluated
on sealed OOS. Note: B1-B6 are trained on *all* 1m bars (B2 on pivot rows),
so they are genuinely in-distribution at a mid-leg bar — the user's "they
carry state" framing is conceptually correct.

| K | A (B9) | B (augmented) | delta(B−A) | 95% CI | verdict |
|---|---|---|---|---|---|
| 5 | +$303 | +$227 | **−$76/day** | [−158, −2] | **significantly HURTS** |
| 10 | +$232 | +$153 | −$79/day | [−164, +8] | not sig (point negative) |
| 30 | +$78 | +$32 | −$46/day | [−116, +21] | not sig |
| 60 | +$50 | −$44 | −$94/day | [−191, +2] | not sig |
| 120 | −$19 | −$13 | +$6/day | [−58, +68] | not sig |

**Verdict: Fork 2 rejected.** B1-B6 augmentation is neutral-to-harmful at
every horizon. OOS prediction Pearson barely moves (K=5: 0.163 → 0.123, it
*drops*). The mechanism is the **lead-in-PCA failure pattern** (MEMORY.md):
the B9 GBM, trained on 190 raw V2 features, already extracts whatever
pivot-structure signal exists; the 18 B1-B6 predictions are deterministic
functions of those same features and add only columns to overfit to. The
user's intuition was sound — B1-B6 *do* encode leg state — but operationally
the raw features already carry that state and the GBM already uses it.

**Structure gates (E4a) also fail.** Standalone b1/b3/b5 gates lose money or
are non-significant. Adding a structure filter to the B9 gate *removes
profitable joins*: `B9 & b5-early` at K=5 = +$182/day vs B9-alone +$303
(drops 477 joins). `B9 & struct` collapses to ~20 joins. Structure filtering
is destructive.

**Pullback filter (E4b)** gives only a non-significant nudge (K=10,
mae ≥ 4 pt: +$256 vs +$232 — CIs overlap heavily). Not worth the complexity.

---

## E3 — Position-constrained 1-contract sim (the operational truth)

`tools/midleg_constrained_sim.py` → `2026-05-20_midleg_constrained_sim.txt`

E1/E2 are unconstrained — they treat every leg as a candidate. A real
1-contract engine can only late-join while flat, and late-joining occupies
the engine. Greedy sim, 51 days:

- **Baseline** (greedy 1-contract, no late-join): +$122/day net of $6/leg
  friction; **2,922 of 2,926 legs taken** (57.3/day). Gross ≈ +$466/day —
  matches the known FLAT-1c reference (~$454/day): **the sim is calibrated.**
- **Late-join** (B9-gated, any `max_k`): total +$122/day, **incremental
  −$1/day, CI [−2, +1], not significant.** Only **4 late-joins in 51 days.**
  Identical for max_k ∈ {120, 30, 10}.

The unconstrained E1 ceiling (+$303/day) does **not** survive the position
constraint. There are only 4 missable legs.

---

## E4 — Why: leg overlap structure (the mechanism)

`tools/midleg_overlap_analysis.py` → `2026-05-20_midleg_overlap.txt`

- Consecutive-leg gap (`entry[N+1] − exit[N]`): **99.8% are exactly 0**;
  0.1% negative (overlap), 0.1% positive. Max gap 20 s, min −15 s.
- Legs overlapping any earlier leg: **4 / 2,926 (0.14%)**.
- Greedy 1-contract engine **utilisation: 99.9%** — the engine is essentially
  always in a trade.
- Leg duration: median 655 s, mean 1146 s (~19 min), max 13970 s.
- The 4 genuinely-missable legs: `2026_04_21` leg 1593 (−$110), `2026_04_23`
  leg 1709 (−$32), `2026_04_24` leg 1778 ($0), `2026_05_07` leg 2367 (−$46)
  — **all losers or breakeven.** Catching them would *lose* money.

Hardened zigzag legs are a near-perfect sequential partition of the price
path: leg N exits at a pivot, leg N+1 is born from that pivot, its R-trigger
fires moments later. A 1-contract engine riding leg N to its exit is free and
ready before leg N+1 triggers. **There is no "busy and missed a parallel
leg" population to act on.**

---

## What the "lost signals" actually are

The engine misses ~7 first-of-session legs per day to the ~20-minute (240-bar)
warmup, when the feature engine and zigzag state are still cold. These are
missed because the engine is **not computing yet** — not because it is busy.
Late-join cannot help: the engine cannot evaluate B9 (or anything) during
warmup, and by the time it can, those legs are long over.

The other "lost signal" source is **deliberate B7 skips** — those are the
filter working as designed, not losses. Late-joining a B7-skipped leg means
overriding a validated filter (+$129/day IS lift); if B7 over-skips, that is
a B7-retune question, not a late-join feature.

Neither source is the "busy, missed a parallel leg" scenario the mid-leg-entry
idea was built for.

## Why this won't change with contract scaling

The killer is the **sequential leg structure**, not the 1-contract limit.
The zigzag produces one leg at a time; even with N contracts there is rarely
a second leg running in parallel. Scaling contracts will *not* create a
mid-leg population. (Adding to the leg you are *already in* — pyramiding — is
a different action, and B9's continuous sizing already covers it.)

---

## Recommendation

1. **Mid-leg entry — SHELVED.** No addressable population. This is structural
   (sequential legs), not a tuning problem; it will not improve with more
   data or contract scaling.
2. **Cold-start lost signals — build zigzag-state priming.** At engine
   startup (and on reconnect), replay `detect_swings` over recent prior-day
   history so the current leg / pivot state is known from bar one, and
   pre-warm the V2 feature engine from history. This eliminates the
   ~20-min warmup blind window — the actual cause of the lost signals the
   user is seeing. Small, well-scoped live-engine change. **Awaiting user
   approval** (it touches the live engine).
3. **SIM deployment — unaffected.** Mid-leg entry was a candidate *add*; it
   is rejected; the L5 stack (B7+B9+B10) goes to SIM unchanged. SIM data will
   confirm the cold-start hypothesis (if the engine misses the first N legs
   each session, that is warmup blindness — priming fixes it).

---

## Artifacts

Tools (all in `tools/`, registered in `research/TOOLS_INDEX.md`):
- `forward_pass_midleg_entry.py` — E1 B9-gated unconstrained late-join
- `midleg_b1to6_augmented.py` — E2 B1-B6 augmentation + structure gates + pullback
- `midleg_constrained_sim.py` — E3 position-constrained 1-contract sim
- `midleg_overlap_analysis.py` — E4 leg overlap-structure analysis

Outputs: `2026-05-20_midleg_{fork1_b9,fork2_b1to6,constrained_sim,overlap}.txt`
Project frame: `research/midleg_entry/project.md`
