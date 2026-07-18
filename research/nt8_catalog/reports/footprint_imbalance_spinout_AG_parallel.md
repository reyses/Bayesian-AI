# FOOTPRINT-IMB Spinout Report

## Overview
This report evaluates whether FOOTPRINT-IMB (Footprint Imbalance) contains a causal edge for detecting 'entry-fail RED X' events in the `exit_dojo` population.

## Extraction Results
- **Rule**: 50 adverse delta cutoff
- **Volume Cost**: Retains 98.0% of engagements
- **Base Good Rate**: 0.4320
- **Filtered Good Rate**: 0.4319
- **Delta CI**: [-0.0012, 0.0010]

## Conclusion (Validation Failed)
The delta CI covers 0.0, meaning the difference between the base good rate and the filtered good rate is statistically zero. The FOOTPRINT-IMB hypothesis failed causal validation on the sealed out-of-sample population. It does not provide an edge for entry-fail prediction.
