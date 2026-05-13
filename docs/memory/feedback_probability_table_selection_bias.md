---
name: Probability table selection bias — averaging across condition bars overstates entry-bar win rate
description: When building a P_revert table by binning bars and measuring forward outcomes, the resulting probability averages ALL bars in the bin, but firing a strategy at the FIRST condition-bar entry gives mid-event entries, not cusp entries. Win rate at entry is ~12-25pp BELOW the table's prediction.
type: feedback
---

When a probability table predicts P_revert = 0.71 for cells like
`z_1h_low ≤ -3σ AND slope_15m ≤ -0.5`, the actual win rate of a strategy
firing at the FIRST condition-bar is ~46-48%, not 71%. Empirically verified
2026-05-10 across IS / Val / OOS in `tools/sim_strongest_cell.py`.

**Why:** The probability table measures "from this bar, what fraction of
forward windows end profitable?" The table averages across ALL bars where
the condition holds, including bars in the middle of a 30-bar crash run.
A strategy entering at the FIRST condition bar gets mid-crash entries where
the bounce hasn't happened yet. The actual bounce-eligible bars are clustered
near the END of the condition run (at the literal cusp).

**Why:** The selection-by-condition produces a stratified sample where each
condition-bar is given equal weight. A cusp run that lasts 30 bars
contributes 30 bars to the table but only 1 actual bounce event. The
probability is "1 bounce per 30 bars in this state" — not "if you enter at
any bar in this state, probability of profitable bounce."

**How to apply:** When using a probability lookup table built on bar-by-bar
binning:
1. NEVER trust the marginal P as the entry-strategy win rate
2. Fire on CUSP/TRANSITION events within the condition (when the underlying
   feature stopped moving in the adverse direction), not the first bar of
   the condition
3. The cusp-on-z detection (z just stopped falling) recovers ~12pp of the
   win rate gap — `if z_lo[t-1] <= -3σ AND z_lo[t] > z_lo[t-1]`
4. Alternative: condition on a transition that's already happened
   (e.g. only fire when slope_15m flipped from − to + while z stays low)
5. Be skeptical of any IS-only edge from probability lookups — OOS regime
   shifts can erase the edge entirely (saw 58.6% IS → 35.9% OOS for the
   "strongest cell")
