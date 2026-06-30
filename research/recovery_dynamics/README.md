# research/recovery_dynamics

**Question (Moises):** what does holding a *wrong* trade back to breakeven actually cost? Not
"does it recover" (trend-conditioned illusion) but **how long, how much, and how many other
trades you forwent** while waiting — the opportunity cost of not realizing you were wrong in time.

## Layout
- `tools/opportunity_cost.py` — THE exercise. Samples long/short entries on a real day, keeps the
  ones that go underwater (wrong trades), holds each to breakeven, and measures dead-hold time,
  drawdown depth, and tradeable swings foregone in that window. Mode-first histograms + worked example.
- `reports/opportunity_cost.md` — output.

## Run
```bash
python research/recovery_dynamics/tools/opportunity_cost.py --day 2024_02_20
```

## Status / findings
**Multi-year (536 days, 2024+2025, ~21k random seeded wrong trades)** — `recovery_2024_2025.md`:
- **Return-to-zero = one full oscillation** of price around the entry level (Moises). The period
  distribution is heavily right-skewed: **MODE ~5 min** (most wrong trades bounce fast) with a long
  tail to hours. ~2 trades foregone (mode & median) per wrong trade, depth ~12–14pt ($25–28).
- **The clock is NOT a fixed constant** — it shifts by regime/year: median oscillation **2024 = 27 min
  vs 2025 = 15 min** (2025 reverts faster but kills more: never-recovered **11% vs 7%**). The MODE is
  stable (~5 min); the median/tail are not. So a fixed "cut at N min" rule is unsafe — the cut clock
  must be **adaptive to the current period**, which is itself a readable environment variable.
- **The genuinely-wrong tail = 7–11% never recover same-day** — the population where cutting actually
  matters. The other ~90% bounce, mostly fast; blanket cutting churns those.

(First single-day pass — `opportunity_cost.md` reports — agreed: mode 2 foregone / 6 min, 41 never.)

## Feature attribution — what describes a DEATH event (`event_features.md`)
- Target: wrong trade that does NOT return to breakeven within **60 min** of going underwater
  (fixed horizon — a first version used "never same-day" and time_of_day faked AUC 0.642 via EOD
  **censoring**; the fix dropped it to the honest number).
- **OOS (train 2024 → test 2025): AUC 0.572, gap +0.072 → CONDITIONAL** (weak, not a green light).
- **VOL describes death, weakly; TREND/ADVERSE ~zero.** Counterintuitive sign: HIGHER vol → LESS death
  (the killer is the LOW-vol slow grind that won't snap back in the window; high vol whipsaws back).
  My "trends-against-you kills it" hypothesis was WRONG OOS.
- **Implication:** at ENTRY, death is barely describable. The real signal is likely DURING the trade —
  elapsed-underwater-time vs the current period (the recovery clock as a live feature). Next test.

## Better mechanism — TRADES UNLOCKED by cutting in time (`cut_in_time.md`)
Period-to-recovery is self-trapping (a 12h bleed scores as "a long period"). New unit = TRADES:
if a wrong trade hasn't bounced within H min, cut it, and count tradeable swings (good+bad) from the
cut to where it would have recovered = trades you UNLOCK. Opportunity baseline ~6 swings/hour.

| cut at | bounced in time (no cut) | still stuck (cut helps) | trades unlocked (stuck): median / mean / 90th |
|---|---|---|---|
| 15 min | 49% | 51% | 4 / 18 / 58 |
| 30 min | 63% | 37% | 6 / 23 / 72 |
| **60 min** | **74%** | **26%** | **11 / 29 / 84** |

- The cut-timing tradeoff, in trades: cut **early** → you cut ~half (many were about to bounce → churn);
  cut **late (60m)** → only the genuinely-stuck **26%** get cut, and each frees a **median 11 / mean 29**
  trades (90th pct **84** — the multi-hour bleeders, now measured as forgone TRADES, not "a long time").
- Heavy-tailed: most stuck trades free few; the tail frees dozens. Cutting in time is tail protection.
- "Unlocked" = capacity freed (good+bad swings), not guaranteed profit. Preliminary; thresholds tunable.

## The kicker — second half of the oscillation (`oscillation_kicker.md`)
A full oscillation is away→back→away(other side)→back; the first return to zero is the MIDDLE. For
adverse-first ("wrong") trades that come back to zero (91%): the SECOND leg is **KICKER 47% + JACKPOT
5% = 52% favorable**, median **18pt ($35)**, with **A2/A1 = 1.13** (symmetric, mildly favorable).
Cutting at breakeven discards the paying half. BUT 49% STALL (coin-flip) and **9% never come back =
adverse runaway/death** — the tail that ruins "hold for the kicker."

## SYNTHESIS — the whole arc collapses to ONE question
Upside of holding = kicker (~52%, 18pt, symmetric). Downside = adverse runaway (~9%, →∞). Cut-in-time,
the death study, and the kicker all reduce to a single discrimination: **is this an OSCILLATOR (hold →
catch the kicker) or a RUNAWAY (cut now)?** Unlike direction (91% noise), this is symmetric and
regime-driven → the one prediction with a real chance. That is the next build.

## Cleanest period measurement — anchor every bar (`anchor_period.md`)
Pure measurement (no positions): anchor at each bar's price, time the FIRST RETURN through that
level = one oscillation period, sampled everywhere (~644k anchors). Empirical OU first-return-time.
- **mode ~2m, median 5m, mean 20m**; heavy right-skew; **no-return (trend) share ~7.2%**.
- Bulk resolved into sensible buckets (share): 2-3m **25%** (mode), 3-5m 21% (→46% revert <5m),
  5-8m 13%, 8-15m 12% (→**71% revert <15m**), 15-30m 9%, 30-60m 6%, 1-2h 4%, 2-6h 4%, **TREND 7.2%**.
  Smooth monotonic decay; **2024≈2025 every bucket to <0.3%** — a structural market constant, not a regime.
  (cached: `artifacts/anchor_period_cache_v2.npz`; `--fresh` to recompute.)
- **Amplitude inside each period bucket (the 2nd dimension)** — peak excursion per anchor:
  - **Period & amplitude are tightly coupled**: 2-3m≈2-3pt, 8-15m≈8-12pt, 1-2h≈33-45pt, 2-6h≈53-71pt.
  - **Super-diffusive**: amp/√period RISES (1.4→3.4 in 2024) — longer oscillations swing WIDER than a
    random walk (√t) would give → momentum inside the longer cycles, not mean-reversion damping.
  - **DECOMPOSITION**: period (timing) is the market CONSTANT; amplitude (size) is the REGIME — 2025 is
    **30-50% wider amplitude at every period** vs 2024, while the period buckets are identical.
    "A fixed clock with a volume knob." This EXPLAINS the earlier bogus "period non-stationarity": a
    fixed-pt drawdown gate let 2025's bigger amplitude leak into the conditioned period. Period≠amplitude.
- **STABLE across years: 5m vs 5m, 7.2% vs 7.3%.** This CORRECTS the earlier `period_evolution`/
  `recovery_2024_2025` claim that the clock was non-stationary (27m vs 15m) — that divergence was an
  ARTIFACT of conditioning on a ≥5pt drawdown (regime-sensitive selection). Unconditional period is stable.
- **Nuance:** period is a FUNCTION of amplitude scale — raw returns ~5m (stable, micro-wiggle);
  amplitude-gated (≥5pt excursion) ~15–27m (regime-sensitive). Trend fraction ~7% either way.

## Amplitude regime over time — the volume knob (`amplitude_evolution.md`)
Sliding-window (21d) per-day amplitude scale across 518 days. Period is the fixed clock; this is how
WIDE the oscillations are.
- Typical 10-min swing swings **~4×: 5.5pt (calm, Jun-2024) → ~20pt (Apr-2025)**, with clear vol
  events; the two scale measures (ref-amp & amp/√period) track perfectly (robust).
- **Slow & persistent** (week-scale regimes, smooth transitions) → estimable CAUSALLY with low lag.
- **Trend share is comparatively stable (~6-8%)** — trend FREQUENCY ≈ constant; AMPLITUDE is the regime.
- Complete decomposition: **fixed period clock · stable ~7% trend rate · slow 4×-breathing amplitude knob.**
- This curve = the live "normal envelope" setter. START of a framework — the causal trailing estimator
  of this knob + validation that envelope-breach predicts the runaway is the unbuilt rest.

## Detrended oscillation — the zigzag persists while climbing (`oscillation_detrended.md`)
Gap fixed: the fixed anchor misses oscillation that rides a drifting mean (a trend zigzags up around
a MOVING center). Measure crossings of a 30-min rolling mean instead. **price = drift + oscillation.**
- **No "no-return" anymore** — 66k cycles, every one crosses the mean. The 7% "trend" was a
  fixed-reference artifact; oscillation is EVERYWHERE, inside trends included.
- **The clock keeps ticking in trends**: median period chop 3m vs trend 5m (same order; fast <3m wiggle
  dominates both). But trend cycles are BIGGER (amp 3→11pt) and skew longer (31% are 15-60m vs 10%).
  Caveat: part of the bigger amplitude is regime-coupling (trends ride the wide-amplitude regime), not
  proven causation.
- **Corrected model**: a "trend" is drift in the mean, NOT absence of the cycle. The normal envelope
  lives around the MOVING mean, not a fixed level. Decomposition depends on the chosen trend scale (W_MA).

## Price returns to the anchor WHILE TRENDING (`anchor_drift_conditional.md`)
Moises' point: because we anchor EVERY bar, the within-trend zigzag already shows as returns to recent
anchors — no detrending needed. Verified by conditioning the every-bar return on trailing drift:
- return rate barely drops across the whole trend spectrum: **calmest 93.7% → most-trending 91.1%**
  (drift up to 48 pt/min), and **median period is 5 min at EVERY drift level** (invariant to drift).
- So the 7% no-return = the trend's **abandoned trailing levels (footprint)**, not "trending time with no
  oscillation." The fixed every-bar anchor already contains the within-trend oscillation.
- **Period is invariant to drift** — a deeper constant than cross-year stability: the clock doesn't notice
  the trend. (The `oscillation_detrended.py` MA version gave a consistent answer but was an unneeded param.)

## Cubic / inflection hypothesis — NOT supported as tested (`cubic_inflection.md`)
Hypothesis (Moises): the 7% no-return anchors = trend-start convex/concave inflections; a cubic
regression lights up there, strongest in large-period zones. Test (31k sampled anchors, centered
±20m cubic): **no-return |d3| 0.61 vs return 0.69 (0.88×, WEAKER)**; cubic strength PEAKS at 15-30m
(window-resonant) then falls toward trends. Inflections ubiquitous (82-85%). Hypothesis not supported
by this instrument. Two reasons (one is a premise fix):
- a clean trend is LOCALLY LINEAR (a line, not an S) in a 40m window → small cubic term; |d3| actually
  detects a SWING that fits the window (peak at ~15-30m), not a launch.
- **the 7% no-return is ALL abandoned levels, not "trend starts"** — a 100→110 run abandons 101..109
  too; the launch is a small SUBSET, so testing all no-return washes the launch signal out (mostly the
  linear middle). Right refinement: isolate the FIRST abandoned level of each run (the launch) and test
  curvature THERE. (Data pushed back on the intuition — reported straight, not spun.)

## Caveats
- "Foregone" = swings that *existed*, not guaranteed wins (capturing them needs correct direction).
- One day, one threshold set (`MIN_ADVERSE_PTS`, `SWING_PTS`). Sensitivity + multi-day = TODO.
- Next: condition recovery on the *causal environment read* (regime / λ / z / vol) to predict WHICH
  wrong trades enter the tail vs bounce — the deducible pattern.
