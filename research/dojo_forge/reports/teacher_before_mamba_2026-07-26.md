# Teacher-before-Mamba: is qwen ready to distill? (2026-07-26)

Owner directive: "run the qwen before the mamba." Ran the teacher's exit
policy through the pre-registered ride-edge gate metric (DEV-holdout, gen-0
census, 148 ride episodes / 22 days). CPU scoring of an already-completed run.

## ROBUST finding (interpretation-independent)
**The teacher's exits do NOT beat the never-bail constant.** Q2 fancy-constant
check: teacher − never-bail = −0.017 capture, 95% CI [−0.092, +0.063]. Its 84%
exit rate adds NO state-dependent value over just riding to the end.
Corroborated in reasoning mode: gen-1 exits net **−319 pts** vs never-bail,
gen-2 (gauge-in-frames) **−207 pts**. Every version's exits DESTROY capture.

## Gate metric (directional; scorer needs spec-intent review)
teacher capture − 5m-hold capture = −0.150, CI [−0.257, −0.030] — teacher's
premature exits capture LESS of the ride than a dumb 5-minute hold. (Even
never-bail only ties 5m-hold on peak-capture here: −0.134 [−0.261, +0.007] —
this specific capture-ratio lens differs from doc-107's PnL lens; flagged for
review.)

## VERDICT: the Mamba is premature
There is currently **no state-dependent exit edge to distill.** Distilling the
logit teacher now clones a constant (best case) or value-destroying early
exits (worse). "Sign is free; the asset is state-dependent magnitude" — the
magnitude isn't there yet.

## The gen-2 signal that points the way
Gauge-in-frames worked directionally: gen-1's 4 exits ALL fired at gauge state
ALIVE-0 (healthy — the gauge would have vetoed every one, incl. the
catastrophic minute-1/-4 exits); gen-2's single exit fired at FADED (warning).
Seeing the instrument made the teacher exit less and later.

## Recommended next teacher step (before ANY mamba spend)
Gen-3 with MECHANICAL gauge-gated exit PERMISSION: the runner blocks EXIT
unless the gauge is ARMED (terminal-phase warning). This removes the
early-exit disease by construction — the teacher can only express exits where
the physics licenses them — then re-score the gate. Iterate the teacher until
it has a real, state-dependent ride-exit edge. THEN distill. The 3 mamba
ratifications stay parked behind this gate.
