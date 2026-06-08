# Idea: OHLCV Candlestick Boundary Constraints for Extraction

## Overview
Currently, the polynomial segment extraction algorithm stops based on the Adjusted R-squared score (`$R^2_{adj} \ge 0.95$`). 

The proposed idea is to pivot to a strict, physically anchored geometric approach: we verify that the reconstructed polynomial curve successfully threads the needle through the actual price candlestick (Open/Close body or High/Low wick range) of the 5-second bars.

## Core Logic
At any given 5s bar `i`, the polynomial's predicted cumulative delta price `Y_hat[i]` must fall exactly between the boundaries of that specific bar:
- `Y_min[i] = min(Open[i], Close[i]) - Close[anchor_start]`
- `Y_max[i] = max(Open[i], Close[i]) - Close[anchor_start]`

If `Y_hat[i]` ever pierces outside of these boundaries, the segment breaks.

## Open Questions / Challenges to Solve
1. **Body vs Wick**: Forcing a smooth mathematical curve to stay inside the tight `[Open, Close]` body might be too strict. Using the `[High, Low]` range gives the polynomial a wider, more realistic "channel" to navigate.
2. **Flat Bars**: If `Open == Close`, the band width is exactly $0.00. The polynomial would have to hit it to the exact penny, which causes math edge cases. We must add a minimum physical tolerance (e.g., $0.25 tick size minimum) or default to the High/Low range.
3. **Strictness Buffer**: Should we require 100% of bars in a segment to pass this constraint, or allow a small failure rate (e.g., 1 bar out of 30 allowed to miss) to account for noise spikes?
