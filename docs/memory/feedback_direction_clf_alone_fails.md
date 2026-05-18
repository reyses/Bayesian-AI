---
name: feedback-direction-clf-alone-fails
description: Direction classifier alone is not a live strategy — entry timing is the unsolved bottleneck; tick-exact exits unmask the gap
metadata: 
  node_type: memory
  type: feedback
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

User pivoted to KPI-driven autonomous mode (target: $100/day NET, low MAE, high Day WR). Built a clean direction-classifier strategy through the existing `training_iso_v2/` ticker+engine pipeline. **All 72 grid configurations (symmetric and asymmetric R/R × thresholds × cadences) fail OOS 2026.**

**Best surviving config**: T=0.95, TP=$20/SL=$5, 15m cadence → +$2.54/day NET, CI [−$4.22, +$10.34]. CI crosses zero — not statistically significant per CLAUDE.md mandate.

**Why it fails**: Direction accuracy 87% at the daisy oracle bars does NOT translate to TP-hit rate 87% when firing at every 15m close. Oracle bars are hindsight-selected as the START of favorable moves. Most cadence-trigger fires are mid-move or end-of-move where the remaining favorable distance is small. TP=$20 rarely hits, SL clipped on noise reversals.

**Critical engine bug discovered**: The default `HardStop`/`TakeProfit` exits in `training_iso_v2/exits.py` close at `state.price` (5s bar close) after a threshold cross. If price overshoots TP/SL intrabar, the exit fires but the close price can be much further from entry than the threshold. This INFLATES winners AND losers, producing apparently strong $/day with poor Day WR (high variance from intrabar overshoot).

**Fix shipped**:
- `training_iso_v2/exits_tick_exact.py` NEW — `TickExactTP` / `TickExactSL` use 5s OHLC high/low to detect intrabar threshold crossings, write the exact threshold price to `position.extras['_force_exit_price']`
- `training_iso_v2/engine.py` PATCH — engine `_tick` honors `_force_exit_price` if set
- `training_iso_v2/ledger.py` PATCH — `ClosedTrade.trough_pnl` field tracks MAE per trade

**How to apply**:
- **Never run backtests with the default HardStop/TakeProfit again** — always use `TickExactSL` / `TickExactTP` from `exits_tick_exact.py`. The historical numbers from the old pipeline are inflated.
- **Direction classifier ≠ live strategy.** Use it as a FILTER on existing tier strategies (FADE_CALM, CASCADE, MA_ALIGN, etc.) that have proven entry timing. The classifier vetoes signals that conflict with its direction call.
- **Don't try to "fix" the classifier by tweaking TP/SL** — the asymmetric R/R grid (TP=$10-40, SL=$5-10, T up to 0.95) confirmed no config survives. The bottleneck is entry timing, not exit policy.
- **The right next step is entry timing**: either a separate model that predicts "is this an oracle moment?", or a price-action trigger (breakout, pullback, sweep) that the classifier then routes.

**What's validated**:
1. Infrastructure works (ticker → engine → exits → ledger → bootstrap CI)
2. Intrabar overshoot bug is fixed
3. The earlier `2026-05-16_forward_pass.md` $50-100/day numbers held BECAUSE the daisy oracle gave entry timing for free. Without that, the classifier alone produces zero.

Connected: [[feedback-leadin-pca-rejected]], [[feedback-scenario-lstm-information-ceiling]], [[project-regret-six-layer-architecture]] (L4 selector still missing), [[user-collaboration-protocol]] (autonomous KPI iteration is a valid mode).
