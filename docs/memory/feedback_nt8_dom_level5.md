# NT8 Level 5 DOM — available, but DON'T use it in strategy logic

**Captured: 2026-04-27.** Discussion outcome on DOM usage. Future sessions:
read this before adding any DOM feature to a strategy.

## What's available

NT8 exposes Level 5 Depth of Market (5 bid levels + 5 ask levels with size)
on the live MNQ feed. Existing infrastructure:

- **`docs/nt8/BayesianBridge.cs` v7.0.0** has `DomLevels` parameter (default 5)
  and an `OnMarketDepth` handler. **But currently only sends Level 1**:
  `bestBid`, `bestAsk`, `bestBidSize`, `bestAskSize`, throttled to 250ms
  (4 updates/sec).
- One Level-1 derived feature is computed:
  `imbalance = (bidSize - askSize) / (bidSize + askSize)`.
- Level 2-5 levels are **not consumed anywhere** — `OnMarketDepth` ignores
  events with `Position > 0`.

To capture full Level 5: extend `OnMarketDepth` to retain a `bidLevels[5]`
and `askLevels[5]` array indexed by `Position`. ~80 LOC in BayesianBridge.

## Why we DO NOT put DOM in strategy logic — the backtest problem

**NT8 Strategy Analyzer does not replay historical DOM.** This is the killer
constraint. It means:

| Backtestable | Not backtestable |
|---|---|
| Bar OHLCV (1s/1m/1h/1D) — `tools/atlas_nt8_rebuild.py` rebuilds from dumps | Historical DOM snapshots |
| Range / wick / z-score / variance ratio | Resting orders at any past moment |
| v1.5-RC bleed-score filter (95-day IS/OOS validated) | Bid-ask imbalance over time |
| 9-tier z>2 trigger + chains | Cumulative volume delta |
| Walk-forward Cohen-d with 3 IS/OOS splits | Aggressor-side flow |

A DOM-based strategy feature can ONLY be evaluated forward-only. You cannot
run the v1.5-RC-style 95-day IS/OOS walk-forward on it. You cannot disprove
"DOM helps" without months of forward Sim101 data.

This breaks the project's validation ladder (see `memory/MEMORY.md`):
- Validation gate 1: IS (ATLAS) — **broken** for DOM features (no historical data)
- Validation gate 2: OOS (ATLAS_OOS) — **broken** same reason
- Validation gates 3-5 still work (Phase 7 replay, Live Sim, Live Real) but
  alone are insufficient — without IS/OOS you cannot statistically distinguish
  signal from noise on a small forward sample.

This is the same trap as the `memory/feedback_*.md` time-of-day filter:
features that correlate with the real cause (range, liquidity, regime) but
that you cannot validate on historical data → cargo-cult risk.

## Where DOM CAN help — three legitimate uses

### Use 1 — Execution-layer optimization (ENABLED, low-risk)

DOM read at the moment of order placement can improve fills WITHOUT changing
trade selection:

- **Spread/liquidity-aware order type**: market vs limit decision based on
  next 5 levels of book.
- **Slippage estimate**: pre-emptive warning if a hard SL will get filled
  beyond cap (DOM thin on the opposing side → expect bad fill).
- **Fill quality monitoring**: log avg fill price vs DOM mid at execution
  time, surface persistent slippage as a flag.

**Strategy-agnostic.** Helps v1.0, v1.5-RC, the 9-tier engine, anything.
Does not require historical DOM to validate — improvements measure as
"better avg fill" on the live ledger.

### Use 2 — Live confidence gate (NOT RECOMMENDED currently)

When entry fires, DOM check downgrades weak setups:
- Strong opposing wall → skip OR size down
- Aligned book pressure → keep / chain

Risk: cannot tune the threshold without forward data. Same un-validated-
filter trap as time-of-day. Only acceptable in **shadow mode** (log what
the gate would have done, compare PnL pre/post for 30+ days, then promote).

### Use 3 — Forward-only DOM logging for research (ENABLED, slow payoff)

Capture DOM snapshots at every entry/exit during ZigzagRunner v1.0 + v1.5-RC
Sim101 runs. After 30+ days:
- ~700 trades × top-5 levels × bid/ask × size = ~7K data points
- Post-hoc: did winners have systematically different DOM at entry vs losers?
- If a real signal emerges → forward-only filter, validate on fresh hold-out.

Aligns with `memory/feedback_rca_process.md` 9-step RCA.

## Recommendation matrix

| Path | Effort | Risk | Backtestable? | When to use |
|---|---|---|---|---|
| **1: Execution layer** | ~30 LOC ZigzagRunner CSV log + ~1 day spec | Low | Validates on live fill data | **Do this when execution quality matters** |
| **2: Live gate** | ~60 LOC + threshold tuning | High | No | Only in shadow mode after Path 3 surfaces signal |
| **3: Forward research log** | ~80 LOC bridge + Python sidecar | Low | No (research only) | **Do this whenever live trading runs** |

**Default plan**: enable Paths 1 and 3 in parallel. Skip Path 2 until Path 3
research surfaces a real signal.

## Concrete next steps if/when revisited

1. Extend `BayesianBridge.cs` `OnMarketDepth` to retain `bidLevels[5]`
   and `askLevels[5]` with `(price, size)` per slot. Send full snapshot
   in DOM message.
2. Add `dom_at_entry` and `dom_at_exit` columns to ZigzagRunner CSV ledger
   (compact JSON: `{"b":[[p1,s1],...],"a":[[p1,s1],...]}`).
3. Build `tools/dom_research_eda.py`: load CSV ledger + DOM snapshots,
   per-trade compute features (imbalance, weighted-mid skew, level-5 size
   asymmetry), Cohen-d winner vs loser.
4. After 30 days of Sim101 logs, run Path 3 EDA. If any feature shows
   |d| > 0.4 OOS — promote to Path 2 shadow mode. Otherwise, archive.

## What NOT to do

- **Do not gate ZigzagRunner v1.0 / v1.5-RC entries on DOM features.**
  These strategies have proven IS+OOS evidence. Adding an un-validated
  filter degrades signal-to-noise.
- **Do not add DOM to the 9-tier 91D feature space.** That space was built
  on bar features and validates IS/OOS. Mixing in non-backtestable DOM
  features contaminates the validation chain.
- **Do not build "DOM imbalance > 0.X = skip trade" rules** without 30+
  days of forward shadow-mode data justifying the threshold.

## Related memory

- `memory/feedback_lookahead_audit.md` — historical reminder that what
  looks like signal often is data leakage
- `memory/feedback_cnn_fragility.md` — small samples + tunable thresholds
  = overfitting trap
- `memory/feedback_data_validation_first.md` — validate inputs before
  trusting outputs
- `memory/feedback_challenge_harder.md` — push back on "feature X must
  help" intuitions when data can't confirm
