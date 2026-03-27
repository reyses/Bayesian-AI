---
name: CNN Training Fragility
description: CNN model is fragile — seed-dependent, asymmetric loss kills it, never change two things at once
type: feedback
---

**Rules for CNN training changes:**

1. **Seed=42 is load-bearing**: model gives 0 trades with different seeds. The edge is real but fragile.
2. **Asymmetric loss kills the model**: PnL-weighted asymmetric penalty made model predict 0.5 for everything (0 trades). Symmetric magnitude-weighted BCE is the only loss that works.
3. **Never change two things at once**: removing seed AND changing loss simultaneously made debugging impossible. One variable at a time.
4. **22D features are dead**: regime drift (FM 4.82x, volume 5.51x scale shift) killed them. Features oscillated between +$24K and -$24K.
5. **3D raw features (50 layers) = 0 trades**: model can't discover DMI from raw price in 30 epochs. "The features we grounded ourselves IS the value we add."

**Why:** User called out sloppy work: "this type of stuff is the reason i cant sleep your bein to sloopy". Multiple regressions from careless simultaneous changes.
**How to apply:** When modifying CNN training, change ONE thing, run, verify, then change the next. Always keep seed=42. Never try asymmetric loss again.
