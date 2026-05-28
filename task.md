# Dynamic Optimization & Velocity Constraints

- `[x]` Implement Absolute Floor on Pity Pass in `train_historical.py` (`max(0.0, avg_oos + 0.10)`).
- `[x]` Implement Capital Velocity Constraint in IS loop (`net_pnl >= 200 * len(this_week)`).
- `[x]` Implement Capital Velocity Constraint in OOS loop (`oos_net_pnl >= 200 * len(next_week)`).
- `[x]` Implement Fail-Fast LR Step-Down (`lr *= 0.5` upon successful OOS segment graduation when in smoke test mode).
- `[x]` Verify changes via local single-agent smoke test launch.

### Future State Architecture
- `[x]` **Phase 1 Curriculum:** `--agent-type EXIT_NMP` trains the actor-critic to manage positions using NMP entries as a baseline.
- `[x]` **Phase 2 Curriculum:** `--agent-type ENTRY_NMP` trains the agent to select optimal entries while relying on baseline mathematical exits.
- `[ ]` **Phase 3 Curriculum:** `--agent-type YOLO` for high volatility exploitation.
- `[ ]` **2-Sigma Overfit Protection:** Implement an early-stopping / rollback mechanism in `train_historical.py`. If the OOS `Metric (n)` drops > 2 standard deviations from the rolling mean of previous epochs, automatically scrap the current brain and roll back to the `best_model.pth` from before the degradation started.
- `[ ]` **LR High-Velocity Lock:** The Auto-Decay logic is completely overridden. The Learning Rate is hard-locked at `x30` (`0.003`) to maintain sufficient velocity to glide over local minima and prevent the network from memorizing the In-Sample dataset.
- `[ ]` **Systematic Sanity Checks:** Run the `nmp_sanity_check.py` on the latest `best_model.pth` (or latest epoch checkpoint) out-of-sample every 4 segments, or upon direct user request, to continuously evaluate whether the Exit logic is actually learning to cut drawdown.
