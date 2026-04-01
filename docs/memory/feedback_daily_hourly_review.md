---
name: Daily and Hourly Review Rule
description: Never evaluate system performance by month — always by day and hour. Live money can't wait for monthly convergence.
type: feedback
---

## Rule: Analyze by day and hour, not by month

Monthly aggregates hide the truth. A system that makes $5K/month but has -$1,440 days is not tradeable with a small account.

**Why:** When running live with real money, we don't have the funds to wait until the monthly average converges. Each day and each hour must be independently viable.

**How to apply:**
- Every forward pass result: show per-day breakdown with mode, not just totals
- Identify the MODE of daily PnL (the most common daily outcome)
- If the mode is negative or near zero, the system loses money on most days
- A system with 15/19 green days but mode near $0 is not the same as 19/19 green days with mode $100
- Design daily stop losses and hourly pause rules for live trading
- Target: 20-50 trades/day (quality), not 500+ (noise)
