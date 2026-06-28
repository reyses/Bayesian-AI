# Beta Exit Walkthrough

This walkthrough demonstrates the asymmetric exit policy, tracking the causal path from observations to reward allocation.

## Process vs Outcome
The policy does not maximize raw trade PnL. It maximizes **decision quality**:
- **Red Trade (Wrong direction):** Fast cuts are rewarded. Hanging on yields negative penalties.
- **Green Trade (Right direction):** Capture of the oracle half-cycle move is rewarded. Early exit yields penalties.

## The Conditional Funnel
The scorecard breaks the decision space into three additive components, trained simultaneously but evaluated sequentially for diagnostics:

1. **Selectivity:** Did the agent take the trade? (Gated by Vol-Normalized MFE/MAE quality).
2. **Direction:** Was the agent right?
3. **Exit:** Did the agent cut fast (on wrong) or capture the move (on right)?

### Example: The Asymmetric Exit
Consider a trade where the agent predicted UP, but the actual oracle half-cycle is DOWN.
- Direction Reward: -1.0
- Exit Reward: Maxes at +0.5 if cut immediately (speed penalty ~0, MAE penalty ~0).

Because the rewards are **additive**, the small positive exit reward provides a gradient to the exit head even though the trade was a loser! The agent learns to cut losers because *the exit head is rewarded for it*, independent of the direction head's failure.

## Diagnostics
*The conditional funnel will be tracked across training epochs and emitted as a diagnostic report.*
