---
name: Peak override on exits failed
description: Letting peak detection override exit cascade held losers too long. 5.8% WR, PF 1.01. Reverted same day.
type: feedback
---

Do NOT let peak detection override exit decisions. Sensors lag — by the time they flip, the trade has round-tripped.

**Why:** Tried 2026-03-19. Peak trades could suppress SL/belief_flip/regime_decay when <2 of 3 sensors opposed. Result: trades held through full MFE and back to entry. 5.8% WR with a few massive winners = lottery ticket, not a strategy.

**How to apply:** Exit cascade fires normally for all trades. Peak detection is an ENTRY signal and EXIT signal (peak_state_exit = inverted entry), but it does NOT override other exits. The code is in exit_engine.py, disabled with a comment. Don't re-enable without fundamentally different sensor logic.
