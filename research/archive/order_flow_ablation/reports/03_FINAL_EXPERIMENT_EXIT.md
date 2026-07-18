# Final Experiment Exit: Order Flow Ablation

## The Core Hypothesis
The driving thesis behind this experiment was that **True Delta** (tick-level order flow, capturing the exact physics of limit order absorption and exhaustion) would provide a massive, orthogonal, and causally valid edge over our existing 416D baseline and free OHLCV wicks. If proven, this would justify purchasing expensive high-fidelity tick data (e.g., Databento).

## The Strict Evaluation Thresholds
Per the project's statistical standards for new features:
- **`+0.05` AUC Lift:** Conditionally approved.
- **`+0.10` AUC Lift:** Real signal.

## The Results
The 3-Layer Ablation gauntlet was run on ~1.5 million rows (with overlapping contract roll-weeks strictly excluded to prevent data leaks):

| Feature Layer | AUC | Lift |
| :--- | :--- | :--- |
| **L1 (Baseline 416D)** | 0.6265 | — |
| **L2 (+ OHLCV Wicks)** | 0.6247 | -0.0018 |
| **L3 (+ True Delta)** | 0.6326 | **+0.0079** |

*Note: While L3 technically passed the Fourier phase-randomized null (0.6273), the absolute magnitude of the lift is the ultimate gate.*

## Final Verdict: DEAD
The `+0.0079` AUC lift provided by True Delta falls an entire order of magnitude short of the `+0.05` conditional approval threshold. 

A lift this microscopic, even if technically surviving a Fourier null, is insufficient to justify the architectural complexity, latency budget, or financial cost of integrating high-fidelity tick data into the production pipeline. 

### Key Takeaways
1. **Tick Data Purchase: REJECTED.** Do not buy the tick data. 
2. **Order Flow Alpha:** The limit order book physics (Absorption/Exhaustion), while mathematically real, is completely absorbed or overpowered by the broader macro/structural features already present in our 416D baseline.
3. **Focus Shift:** We are officially abandoning the pursuit of tick-level micro-structure alpha. All future efforts will refocus on the macro baseline and Mamba RL reward design using the existing 416D state representation.
