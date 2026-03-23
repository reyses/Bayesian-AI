---
name: Ground features in base measurements (DOE principle)
description: User identified that features are over-abstracted. Every feature must trace to Price/Time/Volume in 1-2 transparent steps. DOE transferability principle from injection molding background.
type: feedback
---

Features must be grounded in base measurements (Price, Time, Volume) with transparent operations.
Multi-layer abstractions (PID of z-score of regression residual) are machine-specific — they
encode our tuning constants, not market properties.

**Why:** User comes from manufacturing engineering (injection molding, DOE, rheology). In DOE,
you measure material properties (viscosity, shear rate) not machine readings (output pressure)
because material properties TRANSFER across machines. Same principle: market properties
(velocity, std, position-in-range) transfer across instruments, timeframes, and regimes.
Machine-specific features (F_momentum with kp/ki/kd, ADX with double Wilder smoothing)
break when you change any system parameter.

**How to apply:**
- Every feature must have a one-sentence explanation: "this measures X of Y over Z window"
  where X = statistical operation, Y = Price/Time/Volume, Z = defined window
- If you can't explain it without reading the code, it doesn't belong
- Derivatives are OK when grounded (1 transparent step: velocity = dP/dt)
- Distribution measures are OK (std, variance — they measure SHAPE of base data)
- Cross-derivatives are OK (price × volume — combines independent bases)
- When replacing a feature, keep the QUESTION it answered, replace the plumbing
- Test: "would this feature mean the same thing on ES as on MNQ?" If no, it's machine-specific
