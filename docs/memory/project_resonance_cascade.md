---
name: Resonance Cascade Hypothesis
description: Multi-TF peak agreement detects crashes/rallies. Each TF pair (child+parent) validates peaks as real or fake. Full cascade (5/5 pairs agree) = crash/rally.
type: project
---

Every TF has a parent that validates its peaks:
- 15s -> 1m, 1m -> 5m, 5m -> 15m, 15m -> 1h, 1h -> 4h

When ALL pairs agree on direction = resonance cascade = extreme trend.
When <3 pairs agree = chop. Feb 9 was full 5/5 SHORT cascade.

A "trend" is NOT a separate concept. It's:
1. The decay of peaks in one direction over time
2. The distance between two macro-scale peaks
3. Visible through rolling MFE per direction at micro scale

**Research needed**: `tools/resonance_cascade_research.py`
- Run peak detection on ALL TFs simultaneously
- Count pair agreement per bar
- Correlate with next N bars' direction
- If 5/5 agreement predicts 90%+ accuracy = cascade detector works

**Why:** This would turn Feb 9 from -$1K to +$5K by riding the crash.
Same peak detection code, just applied at multiple scales simultaneously.

**How to apply:**
- Cat brain reads macro peak state (1h/4h) for regime direction
- When cascade detected, ALL micro peaks forced to cascade direction
- Peak still provides timing (bounce entry), cascade provides direction
- Ref: Half-Life resonance cascade (Dr. Freeman), Nightmare Protocol fractal
