# Counter-Trend Scalp & Too-Early Exit Investigation
> Date: 2026-03-07 | Data: OOS Jan-Mar 2026 (1,924 trades)

## The Pattern

The system behaves like a nervous human trader:
1. Enters correct direction (58.4%)
2. Gets spooked by micro-pullbacks, exits too early (<20% captured)
3. Sees the pullback and enters counter-trend (scalp)
4. Profits on the micro-move but misses the macro trend

## Key Findings

### Trade Classification (OOS)
- Correct direction: 1,123 (58.4%) -> $15,821
- Counter-trend scalps: 574 (29.8%) -> $10,314 (profitable!)
- Genuinely wrong: 199 (10.3%) -> -$4,561
- Noise: 28 (1.5%) -> -$109

### Scalp Clustering (analyze_scalp_timing.py)
- 52.8% of scalps follow another scalp (burst pattern)
- Median 16 min between scalps, 158 within 5 min of each other
- MFE is real: 68.7 ticks avg, 224 trades with 64+ tick MFE
- F_momentum much stronger on scalps (-8690 vs -5235 correct)
- Wave maturity identical between scalps and correct trades

### Hourly Overlap (analyze_scalp_vs_early_exit.py)
- **Spearman r=0.716, p<0.001** between scalp count and too-early exit count by hour
- 10/24 hours flagged as "BOTH HIGH" (scalp rate >30% AND early exit rate >15%)
- Worst hours: 0, 22, 23 (overnight), 9, 15 (session transitions)
- NOT a sequential pattern (only 5 direct wave-flip pairs in 5min window)
- It's a REGIME problem: same market conditions cause both symptoms

### Worker Signal Analysis (analyze_scalps.py)
- 4h worker agrees with oracle 57.4% on SHORT scalps fighting LONG oracle
- Workers SEE the trend but get overruled by template bias
- Template #21 (long_bias=0.03 = 97% SHORT) produces most scalps
- Template bias is TF-agnostic: same 97% SHORT applied to 4h and 1s workers

### Oracle Label Distribution
- 72% of scalps fight MEGA_LONG moves
- 76% of too-early exits are on MEGA_SHORT moves
- These are big trending periods where we should hold with conviction

## Root Cause 1: Template Bias Dominance (Brain Aggregation)

### How direction is computed per worker
```
dir_prob = 0.5 * _logistic_prob(template_bias) + 0.5 * _phys_dir(velocity, accel)
```

### The problem
Template biases are **TF-agnostic**. Template #21 has long_bias=0.03 (97% SHORT).
This same bias gets applied to the 4h worker AND the 1s worker.

Per-worker math for template #21:
- `_logistic_prob(0.03)` -> P(LONG) ~ 0.03
- Physics says LONG (velocity up): `_phys_dir` -> P(LONG) ~ 0.75
- Blended: `0.5 * 0.03 + 0.5 * 0.75 = 0.39` -> still SHORT
- Even with correct physics, template bias wins

### Evidence
- 4h worker agrees with oracle (LONG) 57.4% on SHORT scalps
- Workers SEE the trend but `dir_prob` stays SHORT because template dominates
- Templates #21, #54, #55 produce most scalps — all heavily SHORT-biased
- `get_belief()` aggregation (weighted geometric mean) amplifies this:
  if 8/10 workers say SHORT, the geometric mean is heavily SHORT

### Mitigation applied
Band confluence entry blend in `get_belief()`: up to 40% influence when
multi-TF SE bands agree on direction. This provides a structural trend
signal that bypasses per-worker template bias.

### What would really fix it
Template biases should scale by TF: slow TFs should weight physics MORE
than template bias (trends dominate at 4h), fast TFs can weight template
MORE (patterns are more relevant at 1m). Something like:
```
phys_weight = 0.3 + 0.5 * (tf_seconds / 14400)  # 0.3 at 1s, 0.8 at 4h
dir_prob = (1 - phys_weight) * template + phys_weight * physics
```

## Root Cause 2: Timescale Mismatch in Signal Aggregation

### Entry Side
Workers update only on TF bar close:
- 4h worker: frozen for up to 4 hours between updates
- 1m worker: updates every minute
- Band confluence uses TF_WEIGHTS (4h=5.0 down to 1s=0.1)

Slow TFs dominate entry direction (good), but template biases override
physics signals even at slow TFs (bad). Band confluence entry blend added
(up to 40% influence) to counter this.

### Exit Side (the bigger problem)
Band confluence exit logic:
- at_resistance (z >= +1) -> TIGHTEN trail
- at_support AND direction_aligned -> WIDEN trail
- band direction flips against you -> URGENT exit

Problem: In a trend, fast TFs (1m, 5m) hit their local resistance bands
constantly. Each time, trail tightens. The 4h band still says "trending up"
but it only has weight in the aggregation, not a VETO on fast-TF tighten.

Result: correct-direction trades exit too early because fast TF bands
keep triggering tighten, even though slow TFs confirm the trend.

### Proposed Fix: Partial Bar Aggregation with Maturity Weighting

Current: worker belief = last completed bar only (stale but reliable)
Proposed: blend completed bar with forming bar, weighted by completion %

```
effective_signal = completed_bar * (1 - maturity) + partial_bar * maturity
where maturity = time_into_bar / bar_duration
```

- 4h bar, 5 min in: 2% partial weight (noise suppressed)
- 4h bar, 2h in: 50% blend (partial bar earns influence)
- 4h bar, 3h50m in: 96% partial (nearly complete, reliable)

This solves both problems:
1. Slow TFs gradually update instead of sudden jumps at bar close
2. Early-bar noise gets multiplied by near-zero maturity weight
3. Exit decisions see continuous trend state, not stale snapshots

### Scope
Full architecture change: worker tick loop, quantum engine partial states,
band context interpolation. Jules-sized task.

## Analysis Tools Created
- `tools/analyze_scalps.py` — worker signal analysis on scalps
- `tools/analyze_scalp_timing.py` — temporal clustering, MFE, physics
- `tools/analyze_scalp_vs_early_exit.py` — too-early/scalp overlap
- `tools/analyze_wrong_dir.py` — wrong-direction filter simulations

## Action Items
1. Run `--fresh --forward-pass` with band confluence entry blend (current code)
2. Compare scalp count and direction accuracy vs baseline
3. If entry side improves but exit side still bleeds -> write Jules spec for
   partial bar aggregation
4. Pattern relevance branch (ROADMAP.md) is prerequisite for production
