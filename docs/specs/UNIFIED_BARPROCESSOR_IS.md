# Unified BarProcessor for IS — Eliminate Lookahead

**Priority**: Next session #1
**Date**: 2026-03-18

## Problem

IS forward pass uses `pattern_map` pre-computed from ALL bars.
At bar 100, the system knows patterns from bar 50,000.
This is lookahead — inflates IS results.

## Solution

Use BarProcessor for IS, same as OOS and live. Delete the 2,000-line
inline forward pass. Peak detection replaces pattern_map.

## Architecture

```
IS:   ParquetFeeder(ATLAS)     → BarProcessor.process_bar() → oracle labels (parallel)
OOS:  ParquetFeeder(ATLAS_OOS) → BarProcessor.process_bar()
Live: NT8Feeder                → BarProcessor.process_bar()
```

Oracle is an OBSERVER — reads trade log, labels retroactively. Never decides.

## What BarProcessor Already Has

- Peak detection entry (forced_template_id=-100)
- Compressed state candidate building
- Full exit cascade with sensor fusion
- TBN tick_all on every bar
- Trade recording via brain
- Self-tuning exits

## What Needs to Change

1. **Delete**: IS inline forward pass (~lines 1540-1710 in trainer.py)
2. **Keep**: Daily file iteration, TBN warmup, progress reporting
3. **Add**: Oracle post-labeling (read trade log after forward pass, add labels)
4. **Move**: Oracle audit from inline to post-processing step

## The Edge IS Has

IS templates match IS data because they were built from the same data.
This is legitimate — it's the training set advantage. OOS templates
DON'T match as well → lower WR. That's the honest IS/OOS gap.

But the entry DECISION must be the same code. The template match
quality differs (IS=high, OOS=lower), not the decision logic.

## Verification

1. Run IS with BarProcessor → compare to previous IS results
2. If IS PnL drops significantly → the lookahead was inflating results
3. OOS should stay the same (already uses BarProcessor)
4. The HONEST IS number is the one without lookahead
