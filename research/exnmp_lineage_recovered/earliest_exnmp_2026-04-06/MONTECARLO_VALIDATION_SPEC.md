# Monte Carlo Strategy Validation Spec
> Date: 2026-04-05
> Status: PARKED — design captured, implement after ExNMP pipeline stable

## Problem
We validate strategies on IS data (277 real days). But IS is finite — a strategy
that looks good on 277 days could be curve-fit. We need to know: does this strategy
have a real edge, or did it just get lucky on this specific data?

## Solution: Monte Carlo with Synthetic Price Paths

Generate thousands of synthetic MNQ-like price paths with KNOWN parameters.
Run each ExNMP strategy against each path. The distribution of outcomes tells us
if the strategy has a real edge vs random noise.

### Synthetic Path Types
```
Trend:     price drifts ±$X/bar for N bars, noise σ
Chop:      price oscillates ±$Y around flat, noise σ
Spike:     flat → sudden ±$Z in K bars → partial revert
Breakout:  range compression → drift ±$X
Random:    pure random walk, calibrated to MNQ volatility
```

### Parameters (calibrated from real MNQ data)
- Tick: 0.25, TV: $0.50
- 1m bar noise: from cord length analysis ($2.00 median at 5s, $4.50 at 1m)
- Trend drift: from real trending day measurements
- Chop amplitude: from real choppy day measurements
- Spike magnitude: from real spike measurements

### Per Strategy Test
```
Strategy #47 × 10,000 synthetic paths:
  1. Generate 79D features for synthetic path (SFE on synthetic prices)
  2. Run strategy entry/exit logic
  3. Record PnL per path

Output per strategy:
  - Mean PnL by regime (trend/chop/spike/random)
  - WR by regime
  - Max drawdown distribution
  - Sharpe ratio
  - P-value vs random paths (does it beat pure noise?)
```

### Null Hypothesis Test
If a strategy can't beat random paths at p < 0.05, it has no edge.
This is the real validation — not IS/OOS split, but strategy vs random.

### Architecture
```
tools/montecarlo.py:
  - SyntheticPathGenerator: generates price paths by regime type
  - StrategyTester: runs one ExNMP against one path
  - MonteCarloRunner: orchestrates N paths × M strategies
  - ReportGenerator: per-strategy report card + regime breakdown

Input:  list of ExNMP strategies (from corrected trades / tree leaves)
Output: reports/findings/montecarlo_YYYYMMDD.md
```

### Connection to ExNMP Pipeline
```
NMP → regret → corrected trades (ExNMP instances)
  → test each ExNMP on IS (real data report card)
  → test each ExNMP on Monte Carlo (synthetic validation)
  → cluster validated strategies
  → tree organizes clusters
```

### Parked Because
1. Need ExNMP pipeline stable first (corrected trades + tree)
2. Synthetic path generator needs calibration from real MNQ data
3. SFE on synthetic prices needs to work (or simplified feature computation)
4. Big compute: 10K paths × N strategies × SFE per path

### Prerequisites
- Cord length analysis results (from 2026-04-03 session)
- Real regime measurements for calibration
- Simplified 79D computation for synthetic paths (skip full SFE?)
