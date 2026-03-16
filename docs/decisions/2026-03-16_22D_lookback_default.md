# Decision: 22D Lookback Features Are Default (16D Deprecated)

**Date:** 2026-03-16
**Status:** Final
**Decision by:** Moises

## What Changed

The 6D lookback geometry (slope, curvature, efficiency, range, end_position,
monotonicity) from the 10-bar price path before entry is now always computed.
The `--lookback` flag is a no-op. Total feature vector is 22D (16D canonical + 6D).

## Why

1. **16D was blind to approach context.** It only captured the state AT the
   pattern event — not HOW price arrived there. Two identical z-score readings
   have different outcomes depending on whether price crept up slowly (range) or
   plunged in fast (momentum). The 6D geometry captures this.

2. **Lookback geometry is the compressed t-10m funnel.** The decision tree
   research (2026-03-16) showed that 10-bar price path features (range, sigma)
   predict regime starts at 85% accuracy. The 6D geometry is the lightweight
   version of this signal, embedded in every template.

3. **No computational cost.** The 6D extraction is a numpy operation on 10 close
   prices — microseconds per candidate. No GPU, no extra data loading.

4. **One-way door for templates.** Once templates are built with 22D centroids
   and the scaler is fitted to 22D, running with 16D causes dimension mismatches.
   Making 22D the default prevents accidental mismatches.

## What's Deprecated

- `--lookback` flag: still accepted, now a no-op
- 16D-only templates: cannot be used with current scaler/centroids
- Any checkpoint built before this change requires `--fresh` to rebuild

## CLI Impact

| Before | After |
|--------|-------|
| `python training/trainer.py --fresh --lookback` | `python training/trainer.py --fresh` |
| `python training/trainer.py --forward-pass --lookback` | `python training/trainer.py` |
| `python training/trainer.py` (16D) | N/A — no longer possible |

## Default Run Behavior (post-change)

```
python training/trainer.py
```
- If checkpoints exist: skip Phase 1-3, load templates/library, fresh brain → IS → OOS (~20 min)
- If no checkpoints: full Phase 1-3 → IS → OOS (~1 hour)
- 22D features always on
- `--fresh` to wipe and rebuild everything
