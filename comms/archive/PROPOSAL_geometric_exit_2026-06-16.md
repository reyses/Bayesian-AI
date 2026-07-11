# PROPOSAL → Gemini — Geometric Rollover Exit (2026-06-16)

> **Context:** The Genetic Algorithm (GA) successfully tuned the Kalman CA filter on H1 2024. It locked in a velocity entry threshold of `0.066 pts/sec`. However, the GA optimized the exit by defaulting to a massive `79.4-point trailing stop`, which is highly capital intensive and causes us to surrender a median of 80+ points of open profit during chopped trades. We need an intelligent, causal geometric exit to lock in the ~71-point MFEs earlier.

## PRIMARY TASK: Engineer a Geometric Rollover Exit

Instead of using a static trailing drawdown, we need to mathematically detect the "rollover geometry" at the peak of the wave. 

### 1. Test Candidate Exits (`research/test_geometric_exits.py`)
Using the 5,463 trade paths from `reports/findings/trade_paths.parquet`, evaluate the following causal exit triggers to see which best captures the right side of the peak *before* the 79-point trail would fire:

*   **Kalman Acceleration Flip:** Exit exactly when the Kalman acceleration vector drops below zero (a leading indicator of velocity decay).
*   **Kinematic Velocity Decay:** Exit when the Kalman velocity drops by X% (e.g., 50%) from its maximum achieved velocity during the current trade.
*   **Dual-Timeline Pinch:** Exit when a faster, leading timeline (e.g., a 2m cubic) hooks backward while the macro Kalman is still ascending, signaling structural exhaustion.

### 2. Evaluation & Validation (`reports/findings/geometric_exit_results.md`)
Compare the candidates against the 79.4-point trailing stop baseline.
*   **Key Metric:** Which exit captures the highest percentage of the peak MFE on average?
*   **Cost Metric:** Which exit reduces the Maximum Adverse Excursion (MAE) and open risk the most?
*   **Deliverable:** A markdown report detailing the metrics, plus a `matplotlib.animation` GIF overlaying the winning geometric exit trigger on an actual "Big Winner" trade path to visually confirm it fires exactly at structural rollover.

### 3. Final Strategy Integration (`research/orange_kalman_strategy.py`)
Once the winning geometric exit is identified:
*   Integrate it into a final strategy script.
*   Combine it with the GA's confirmed `0.066 pts/sec` entry gate.
*   Run the final OOS validation across the 2025-2026 dataset to verify that the geometric exit stops the PnL degradation seen in the prior OOS run.

## DISCIPLINES (MANDATORY)
- **Causal Only:** No hindsight. Exits can only trigger based on past/present data.
- **Reporting:** Update `docs/daily/INDEX.md` and follow all standard disciplines defined in `comms/CONTEXT_FOR_GEMINI.md`.

Please review this proposal, confirm if the exit candidates are sound, and proceed with execution.
