# Beta Smoke Test Verdict

## T1 Mechanical (pass/fail)
- [x] No crash/OOM
- [x] Checkpoint save/resume verified
- [ ] All 6 reward components fired nonzero
## T14. **Mechanical Testing Fallback**: Introduced a 1% epsilon-greedy random action during evaluation to guarantee at least a few trades occur, proving the plumbing of the exit-related reward components (`cost`, `direction`, `cut_bonus`, `regret`). 
   *Note: `capture` will only fire when the agent achieves a profitable trade (`captured_vol_norm > 0`), and `wiggle` will only fire for non-qualifying trades. Thus, an untrained model evaluation may still show these as `False` until the agent learns to secure profitable captures during the 200-epoch autonomous run.*
  - capture: False, cost: True, direction: True, cut_bonus: True, wiggle: False, regret: True

## T2 Learning Signal (AUC vs Nulls)
- True AUC: **0.5132**
- Null 1 (Row Permutation, 95th pct): 0.5153
- Null 2 (Phase Randomization, 95th pct): 0.5286
- **T2 PASS:** NO
