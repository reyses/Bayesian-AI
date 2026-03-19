# Gate Cascade — C&E Matrix + PFMEA
> Generated 2026-03-15 from IS run (9,253 traded / 661,513 candidates evaluated)
> Commit: b2b5343

---

## 1. GATE CASCADE ORDER

Sequential — first failure stops evaluation. Competition happens mid-cascade.

```
SIGNAL DETECTED (661,513 candidates on 90,850 bars)
  │
  ├─ Gate 0a: Pattern Quality (no pattern/noise/approach zone)     → 121,657 blocked (18.4%)
  ├─ Gate 0b: Hurst < 0.50 (choppy market)                        →  53,898 blocked  (8.2%)*
  ├─ Gate 0c: Momentum Override (reversion dominates momentum)     →  11,747 blocked  (1.8%)*
  ├─ Gate 0d: Tunnel Prob < 40% (low reversion probability)        →  34,703 blocked  (5.2%)*
  ├─ Gate 0.5: Depth Filter (depth<3 or blacklist)                 →   1,166 blocked  (0.2%)
  ├─ Gate 1: Template Match (cluster dist > 3.0)                   →   3,625 blocked  (0.5%)
  ├─ Gate 2: Brain Reject (unprofitable pattern)                   →       0 blocked  (0.0%) ← DEAD
  ├─ Gate 2.5: TF Confluence (DMI alignment)                       →       ? blocked
  │
  ├─ ★ SCORE COMPETITION (best candidate per bar wins)             → 465,924 losers  (70.4%)
  │     └─ NOT TRACKED — appears as "Unaccounted"
  │
  ├─ Gate 3: Low Conviction (belief < 0.48)                        →  44,279 blocked  (6.7%)
  ├─ Gate 4: Momentum Misalign (F_mom sign ≠ direction)            →  15,320 blocked  (2.3%)
  ├─ Gate 4.5: FDMI Fakeout (State A block)                        →       ? blocked
  │
  └─ ✓ TRADED                                                     →   9,253 passed   (1.4%)

  * Gates 0b-0d are sub-categories of Gate 0a total (121,657)
```

### Gate Ordering Issue

Gates 3 and 4 run AFTER the score competition. This means:
- A high-conviction candidate can lose the score competition to a closer-distance candidate
- The winner then fails conviction → no trade
- The original high-conviction candidate was discarded without being checked

This ordering means conviction and momentum are applied to the WINNER of a distance-based
competition, not to the best-conviction candidate. The competition optimizes for template
proximity, not trade quality.

---

## 2. CAUSE & EFFECT MATRIX

**Y (Output)** = Trade taken / Trade PnL
**Scoring**: Contribution to gate pass/fail 1-10

### 2a. Gate Input Variables

| X (Input Variable) | Gate | C&E Score | Effect on Y | Mechanism |
|---------------------|------|-----------|-------------|-----------|
| **Pattern type** (BAND_REVERSAL / MOMENTUM_BREAK) | 0a | **9** | Determines regime compatibility rules | Approach zone checks differ by pattern type |
| **Z-score (micro)** | 0a | **9** | < 0.5σ = noise, > 3σ = nightmare zone | Directly gates signal quality |
| **Hurst exponent** | 0b | **8** | < 0.50 = choppy/mean-reverting = bad for trend | 53,898 signals blocked (9.7% of FN misses) |
| **Tunnel probability** | 0d | **7** | < 40% = low confidence in reversion | 34,703 blocked — second largest physics gate |
| **F_momentum vs F_reversion** | 0c | **7** | Reversion dominates = ranging market | 11,747 blocked |
| **Cluster distance** | 1 | **6** | How well signal matches known template | > 3.0 = poor match. Only 0.5% blocked — very permissive |
| **Depth** | 0.5 | **5** | Fractal hierarchy level | < 3 blocked. Blacklist for known bad depths |
| **Brain profitability** | 2 | **0** | Pattern historically profitable? | **DEAD GATE — 0 signals blocked ever** |
| **Conviction** | 3 | **4** | Path confidence across TF workers | 0.48 threshold = rubber stamp. Median traded = 0.563 |
| **F_momentum sign** | 4 | **5** | Momentum direction vs trade direction | 2.3% blocked. When it fires, prevents wrong-side entries |
| **Score (competition)** | comp | **10** | Which candidate wins on multi-signal bars | 70.4% of all signals eliminated here — THE dominant filter |
| **ADX (macro)** | 0a | **6** | Trend confirmation for MOMENTUM_BREAK in approach zone | Sub-gate: ADX < threshold blocks weak momentum |

### 2b. Critical X's (Pareto)

Top 3 inputs explaining ~90% of signal filtering:

1. **Score Competition** (70.4%) — The score tiebreaker IS the filter. Everything else is secondary.
   On bars with multiple candidates (86.3% of trading bars), only 1 wins. The rest vanish
   without being counted. This is by design (can't take 7 trades simultaneously), but the
   competition metric (distance + tier + depth) optimizes for template proximity, not profitability.

2. **Pattern Quality gates** (18.4%) — Z-score zones, Hurst, momentum override, tunnel prob.
   These are physics-based filters that remove structurally unsound signals. They're the
   second line of defense but catch mostly genuine noise.

3. **Low Conviction** (6.7%) — At 0.48 threshold, barely filters anything. Conviction correlates
   with PnL ($7.43 at 0.48-0.55 vs $10.47 at 0.65-0.75) but the threshold is too low to
   exploit this relationship.

---

## 3. SIGNAL ACCOUNTING GAP

### The 70.4% Problem

465,924 candidate signals are evaluated but not attributed to any gate. They appear as
"⚠ Unaccounted" in the report. These are candidates that:

1. **Passed all quality gates** (0-2.5)
2. **Competed on a multi-signal bar**
3. **Lost the score competition** to a better-scoring candidate
4. **Were discarded without tracking**

This means:
- We don't know if competition losers had higher conviction than winners
- We don't know if losers would have been profitable
- We can't analyze whether the scoring function (distance + tier + depth) produces
  better outcomes than conviction-based selection

### Fix Required

Add `gate_stats['competition_loser']` counter in `_gate_check` or the competition logic.
Track the losing candidates' conviction, distance, and template_id so we can compare
competition winners vs losers in post-run analysis.

---

## 4. PFMEA — GATE FAILURES

**Scoring Scale:**
- **Severity (S)**: 1-10. Impact on trade quality if failure occurs.
- **Occurrence (O)**: 1-10. How often this failure mode happens.
- **Detection (D)**: 1-10. How hard it is to detect. (10 = undetectable)
- **RPN** = S × O × D.

| # | Gate | Failure Mode | Effect | S | O | D | **RPN** | Current Control | Recommended Action |
|---|------|-------------|--------|---|---|---|---------|-----------------|-------------------|
| 1 | **Competition** | Distance-based winner has lower conviction than loser | Takes worse trade when better candidate exists on same bar. 70% of all signals lost here without quality check. | **8** | **9** | **9** | **648** | Score = depth + dist + tier_adj | Track competition losers. Test conviction-weighted scoring. |
| 2 | **Gate 3 (position)** | Conviction check AFTER competition, not before | High-conviction candidate loses to closer-distance candidate, winner fails conviction → no trade taken. Both lose. | **7** | **7** | **8** | **392** | Sequential cascade | Move conviction check before competition, or include conviction in score |
| 3 | **Gate 2 (brain)** | Brain reject is dead (0 signals blocked) | No historical profitability filter. Every template is treated as profitable regardless of track record. | **6** | **10** | **5** | **300** | `should_fire()` always True | Investigate why brain never rejects. Check min_prob threshold. |
| 4 | **Gate 3 (threshold)** | 0.48 conviction threshold is a rubber stamp | Only blocks 6.7% of signals. Median traded conviction is 0.563. Pass rate ~93%. | **5** | **8** | **4** | **160** | MIN_CONVICTION = 0.48 | Raise to 0.55 (cuts 43% lowest-conviction trades, avg PnL improves 11%) |
| 5 | **Gate 0b (Hurst)** | Hurst 0.50 threshold too aggressive | 53,898 signals blocked. Hurst at 0.48-0.52 is ambiguous — some are trending. Could be blocking profitable signals. | **5** | **7** | **6** | **210** | Hard cutoff at 0.50 | Analyze FN: what % of Hurst-blocked signals were real moves? Consider softening to 0.45. |
| 6 | **Gate 4 (position)** | Momentum misalign AFTER direction cascade | Direction picked by cascade, then momentum check vetoes it. Momentum should INFORM direction, not override it. Wastes the cascade computation. | **4** | **6** | **5** | **120** | Post-direction veto | Include F_momentum sign as a direction cascade voter (priority weight) |
| 7 | **Gate 0d (tunnel)** | Tunnel prob < 40% too aggressive | 34,703 blocked. Tunnel probability is model-dependent — if model is miscalibrated, this blocks good signals. | **4** | **6** | **7** | **168** | Hard cutoff at 40% | Cross-validate: do blocked tunnel signals have lower oracle MFE? |
| 8 | **Gate 1 (threshold)** | Template distance 3.0 too permissive | Only 0.5% blocked. Almost any feature vector matches some template. No quality discrimination. | **3** | **2** | **4** | **24** | gate1_dist = 3.0 | Acceptable — template match is a soft filter by design |
| 9 | **Gate 0a (regime)** | Regime-pattern compatibility too strict | May block valid BAND_REVERSAL in approach zone that would have worked | **3** | **4** | **6** | **72** | Static regime-pattern map | Validate with FN analysis |
| 10 | **Gate 2.5 (confluence)** | TF confluence gate not tracked | Cannot assess impact — no counter in gate_stats | **3** | **5** | **9** | **135** | Unknown fire rate | Add counter and include in report |
| 11 | **Gate 4.5 (FDMI)** | FDMI fakeout gate not tracked | Cannot assess impact — no counter in report | **3** | **5** | **9** | **135** | Unknown fire rate | Add counter and include in report |
| 12 | **ALL (cascade order)** | Late gates waste computation on doomed candidates | Gates 3-4 run on competition winner only. If winner fails, the entire bar produces no trade even if other candidates would pass. | **6** | **5** | **7** | **210** | Sequential cascade | Consider parallel evaluation of top-N candidates |
| 13 | **Competition (scoring)** | Score function doesn't include conviction | `score = depth + dist + tier_adj`. Conviction, momentum alignment, and template WR are not in the score. Competition optimizes proximity, not quality. | **7** | **9** | **6** | **378** | Static additive score | Add conviction and template_wr to score function |
| 14 | **Gate 0c (threshold)** | Momentum override blocks during volatile moves | 11,747 blocked. Volatile markets have high reversion force — this can block entries at exactly the moments with highest MFE. | **5** | **5** | **7** | **175** | Ratio check: F_mom < F_rev * override_ratio | Analyze: are momentum-blocked signals correlated with high oracle MFE? |
| 15 | **Unaccounted tracking** | 70.4% of signals have no gate attribution | Cannot diagnose why signals were rejected. Gate funnel analysis is incomplete and misleading. | **4** | **10** | **8** | **320** | None | **FIX**: Count competition losers explicitly. Critical for gate tuning. |

---

## 5. PFMEA PRIORITY RANKING (by RPN)

| Rank | RPN | Gate | Failure Mode | Status |
|------|-----|------|-------------|--------|
| 1 | **648** | Competition | Distance-based winner ignores conviction quality | OPEN — needs competition loser tracking |
| 2 | **392** | Gate 3 position | Conviction after competition = lost high-conviction candidates | OPEN — restructure cascade |
| 3 | **378** | Competition scoring | Score doesn't include conviction/WR/momentum | OPEN — redesign score function |
| 4 | **320** | Unaccounted | 70.4% signals not attributed to any gate | OPEN — **FIX: add competition_loser counter** |
| 5 | **300** | Gate 2 (brain) | Brain reject is dead (never fires) | OPEN — investigate should_fire() |
| 6 | **210** | Gate 0b (Hurst) | 0.50 threshold may be too aggressive | OPEN — FN analysis needed |
| 7 | **210** | Cascade order | Late gates waste computation, miss candidates | OPEN — consider top-N evaluation |
| 8 | **175** | Gate 0c | Momentum override blocks volatile moves | OPEN — FN correlation analysis |
| 9 | **168** | Gate 0d (tunnel) | Tunnel prob 40% threshold not validated | OPEN — cross-validation needed |
| 10 | **160** | Gate 3 threshold | 0.48 conviction = rubber stamp | OPEN — raise to 0.55 |
| 11 | **135** | Gate 2.5 | TF confluence not tracked | OPEN — add counter |
| 12 | **135** | Gate 4.5 | FDMI fakeout not tracked | OPEN — add counter |
| 13 | **120** | Gate 4 position | Momentum veto after direction cascade | OPEN — integrate into cascade |
| 14 | **72** | Gate 0a | Regime-pattern compat too strict | LOW — validate with FN |
| 15 | **24** | Gate 1 | Template distance too permissive | ACCEPTABLE |

---

## 6. KEY FINDINGS

### Finding 1: Score Competition is the Real Gate (RPN 648)
70.4% of all evaluated signals are eliminated by the score competition, not by any
quality gate. The score function (`depth + distance + tier_adj`) optimizes for
template proximity and depth, not for trade quality (conviction, momentum alignment,
template WR). A candidate with 0.85 conviction and dist=2.8 loses to a candidate
with 0.50 conviction and dist=1.5. This is the single highest-impact improvement
opportunity in the entire system.

**Action**: Include conviction and template historical WR in the score function.
Track competition losers to quantify how often the "better trade" loses.

### Finding 2: Conviction Check Position is Wrong (RPN 392)
Gate 3 (conviction) runs AFTER the score competition. Sequence:
1. 7 candidates pass gates 0-2.5
2. Best-score candidate wins competition
3. Winner checked for conviction → fails (0.45)
4. Bar produces NO trade

Meanwhile, candidate #3 had conviction 0.78 but lost on distance. If conviction
were in the score or checked before competition, that trade would have been taken.

**Action**: Either move conviction before competition, or (better) include it in
the score so high-conviction candidates win competitions more often.

### Finding 3: Brain Reject is Dead Code (RPN 300)
Gate 2 (brain reject) has blocked exactly 0 signals across 661,513 evaluations.
`brain.should_fire()` always returns True. Either:
- The min_prob threshold is set to 0 (everything passes)
- The brain table never accumulates negative profitability
- The brain is initialized with optimistic priors that never get overridden

This gate was designed to prevent trading patterns with negative historical PnL.
Its complete inaction means every template is treated as equally profitable.

**Action**: Investigate `should_fire()` logic and the `min_prob` / `min_conf`
thresholds. The brain DOES have E[PnL] data (visible in reports) — the gate
just doesn't use it.

### Finding 4: The 70% Blind Spot (RPN 320)
The report shows "⚠ Unaccounted: 465,924 (70.4%)" which is misleading.
These are competition losers, not untracked signals. But because they're not
labeled as such, any gate funnel analysis is incomplete. You can't tune gates
if you can't see where 70% of signals go.

**Action**: Add explicit `competition_loser` counter. Log the loser's conviction,
distance, and template_id for post-run analysis.

### Finding 5: Conviction Threshold is Too Low (RPN 160)
At 0.48, conviction blocks 6.7% of post-competition signals. Conviction buckets show:
- 0.48-0.55: $7.43/trade (753 trades)
- 0.65-0.75: $10.47/trade (249 trades)

Raising to 0.55 would remove 753 trades at $7.43/trade avg. These are the weakest
trades by conviction. Impact: fewer trades but higher avg PnL and fewer losses.

**Risk**: Conviction is dominated by slow TF weights (4h=5.0, 1s=0.1). A 4h worker
disagreeing kills conviction regardless of fast TF signals. Raising the threshold
amplifies this slow-TF dominance.

---

## 7. RECOMMENDED FIXES (Priority Order)

### Immediate (this session)
1. **Track competition losers** — add `gate_stats['competition_loser']` counter
   and log loser conviction/dist/tid for analysis

### Short-term (next session)
2. **Investigate brain.should_fire()** — why does it never reject?
3. **Include conviction in score** — `score = depth + dist + tier_adj - conviction_bonus`
4. **Track gates 2.5 and 4.5** — add to skip reason report

### Medium-term (research)
5. **FN analysis on Hurst/tunnel/momentum gates** — are they blocking profitable signals?
6. **Top-N competition** — evaluate top 3 candidates, not just winner
7. **Conviction threshold sweep** — run at 0.50, 0.55, 0.60 and compare WR/PnL

---

## 8. AUDIT UPDATE (2026-03-18) -- Quality Filter + Peak Detection Changes

### Hard Quality Filter on Templates
- **Before**: 436 templates in valid_tids, exception_tids soft override allowed noise templates
- **After**: Hard filter at checkpoint load: >=10 members, >=55% WR, <=10 tick sigma
- **Result**: 176 templates pass, 260 rejected (60% of library was noise)
- **Impact**: Gate 1 (template match) now matches against 176 centroids instead of 436
- Gate 0a (Pattern Quality) block rate should decrease (quality checked at load, not gate)

### Peak Detection Entry + Cooldown
- Peak detection (template_id=-100) fires when P_center jumps + F_momentum collapses
- NEW: 6-bar cooldown after peak_state_exit fires -- prevents re-entry stutter
- Stutter was the #1 source of 1-bar trades (1,239 in OOS before fix)

### Updated Gate Cascade Items

| Orig RPN | New RPN | Gate | Change | Reason |
|----------|---------|------|--------|--------|
| 300 | **300** | Gate 2 (brain) | UNCHANGED | should_fire() restored but still needs investigation |
| 648 | **648** | Competition | UNCHANGED | Score function still distance-based, no conviction |

### New PFMEA Items (Gate Cascade)

| # | Gate | Failure Mode | Effect | S | O | D | **RPN** | Status |
|---|------|-------------|--------|---|---|---|---------|--------|
| 16 | **Quality filter** | Threshold too strict (WR>=55%) | Rejects templates that are profitable in specific conditions (e.g., 50% WR but high $/trade) | 4 | 3 | 5 | **60** | MONITOR -- compare trade count + $/trade before/after |
| 17 | **Peak cooldown** | Cooldown too long (6 bars) | Misses valid re-entry opportunity after genuine double-peak | 3 | 3 | 5 | **45** | LOW -- 6 bars = 1.5 min at 15s, reasonable |
| 18 | **Peak detection** | Fires on fake peaks (volume still flowing) | Enters against trend on a pause, not a reversal | 6 | 5 | 6 | **180** | OPEN -- sensor fusion at entry would filter (see peak template spec) |
