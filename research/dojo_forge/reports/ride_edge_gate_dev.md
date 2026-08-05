# Ride-Edge Gate — DEV-HOLDOUT diagnostic (logit teacher, gen-0 census)
148 ride episodes (peak>=10.0pts), 22 days. Teacher exit rate on rides: 84%.
NOT the lockbox gate — that is a one-shot owner-triggered run. This says where the teacher stands.

## Q0 power (rough)
- 22 days available; day-block CI half-widths below are the empirical power. If the gate-metric CI half-width > a few ticks of capture, underpowered.

## THE GATE METRIC: teacher capture − 5m-hold capture (per-day mean)
- **-0.150**, 95% day-block CI [-0.257, -0.030] — **WORSE than 5m-hold**

## Q2 fancy-constant check: teacher − never-bail
- **-0.017**, 95% CI [-0.092, +0.063] — **teacher ≈ never-bail constant — nothing state-dependent to distill (Q2 HARD-FAIL risk)**

## context: never-bail − 5m-hold
- -0.134, 95% CI [-0.261, +0.007] (the moat: does riding beat holding-5m on these ride days)

## Verdict for teacher-before-mamba
If the gate metric ties/loses AND Q2 shows teacher≈never-bail, the LOGIT teacher is a fancy constant: distilling it yields never-bail, not a state-dependent ride edge. The teacher needs a distillable exit-JUDGMENT channel (reasoning/gauge-conditioned) before the Mamba is justified. Sign is free; the asset is state-dependent magnitude.
