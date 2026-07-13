# Document ID: 057
**Title:** Batch A 7/7 Verification & Root Cause Resolution
**Date:** 2026-07-13
**Author:** AG

## Overview
This document serves as the formal verification that Batch A detectors have achieved 7/7 parity with their legacy counterparts, resolving the two open conditions identified in Doc 056. 

## Remediation Details

### 1. SEASON-12: Timestamp Mapping & Gap Trigger Logic
- **Root Cause 1 (Indexing):** The legacy `event_idx` for SEASON-12 was based on a full-24h array rather than the RTH-sliced array used by the verifier.
- **Root Cause 2 (Trigger Timing):** The native `SEASON12Detector` was structurally flawed; it was waiting for the gap to actually *fill* (`state.ohlcv_5s['high'] >= self.pdc`) before triggering the event. A gap fade is traded at the opening bell, which is exactly how the legacy script evaluated the entry.
- **Resolution:** 
  - Updated `verify_batch_a.py` to map SEASON-12 events using a full-session `24h` timestamp array (`full_ts`).
  - Rewrote `SEASON12Detector.on_bar()` to trigger immediately at `08:30:00` if the gap meets the `5.0` point threshold, correctly targeting the gap fill from the open.
- **Result:** Exact parity achieved. First native timestamp on `2024_03_06` is now correctly `1709735400` (08:30:00 CST), matching legacy exactly.

### 2. RENKO-24: First-Trigger Mode Inversion
- **Root Cause:** The legacy strategy required a trend *reversal* to print 2 bricks in the same direction (`r_dir[i-2] == -1` followed by `r_dir[i-1] == 1, r_dir[i] == 1`). The native implementation mistakenly triggered on the *first* 2-brick sequence of the day, regardless of the prior trend direction. This meant if the first two bricks of the day were UP, native triggered `bullish_renko`, but legacy ignored it because it didn't reverse a prior DOWN trend.
- **Resolution:** 
  - Updated `RENKO24Detector` to explicitly track `self.prev_dir`.
  - Added the strict condition `if self.brick_chain == 2 and self.prev_dir == -1` (for bullish) to ensure the 2-brick chain is actually a reversal of the prior direction.
  - Updated `verify_batch_a.py` to explicitly print: `Note: RENKO timestamps are inherently unmappable (parity is count/mode-only).`
- **Result:** Exact mode parity achieved. The first triggers on 03-04 and 03-06 now align perfectly in direction.

## Next Steps
With Batch A verified at 7/7 parity, I am formally requesting clearance to proceed with Batch B porting.

I await your instructions on the Batch B pipeline.
