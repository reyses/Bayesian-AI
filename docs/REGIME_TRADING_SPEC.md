# Regime Trading Framework — The Anchor

> **Status**: Design spec — not yet implemented
> **Date**: 2026-03-13
> **Problem**: Current system is a scalper ($1.67/trade avg). The market offers $22/trade with $10 risk.

## The Core Insight

The system captures micro-movements (scalps) when the market produces regime-level moves
that are 10-15x more profitable with the same or less risk. A single 95-point MNQ move
(~$47.50 per contract) occurs multiple times daily, yet we average $1.67/trade.

## Three-Component Architecture

### 1. DMI Regime Detection (Entry Trigger)

**What**: DI+/DI- crossovers on 1m bars identify regime transitions.
- Bullish: prev DI+ < DI- AND curr DI+ > DI- (momentum shift to buyers)
- Bearish: prev DI+ > DI- AND curr DI+ < DI- (momentum shift to sellers)

**Why DMI**: It's the only indicator that directly measures directional *commitment*
(not just price level or momentum magnitude). A crossover means the dominant force changed.

**Parameters**: Wilder 14-period (standard), computed via `compute_adx_dmi_cpu()`.

**ADX filter**: Optionally require ADX > threshold at crossover time. Higher ADX = stronger
regime, fewer false signals, larger moves. To be tuned via `dmi_crossover_validation.py`.

**Exit signal**: Reverse crossover (DI crosses back), OR MR UCL break (I-MR regime end),
OR SL hit.

### 2. 192D State Vector (Noise Filter)

**What**: 12 TFs x 16 features = 192-dimensional MarketState snapshot at entry time.

**Purpose**: Not every DMI crossover is a real regime change. Many are noise in choppy
ranges. The 192D vector captures multi-TF context to filter:
- Is the crossover aligned with higher-TF trend? (features: parent_z, tf_alignment)
- Is volatility expanding or contracting? (features: log1p_vol, entropy)
- Are multiple TFs showing the same directional shift? (feature: dmi_diff across TFs)
- Is the market in a trending regime vs mean-reverting? (features: hurst, adx)

**How**: At each DMI crossover, build the (12, 16) state matrix (same as `build_stacked_matrices()`
in `tools/research/data.py`). Feed to a classifier trained on historical crossover outcomes:
- Label 1: crossover led to MFE > $10 within 10 min (real regime)
- Label 0: crossover led to SL hit or MFE < $5 (noise)

**Classifier**: Start simple (logistic regression on flattened 192D). Upgrade to CNN/attention
on (12, 16) matrix if logistic underfits.

### 3. Equity-Based Position Sizing

**Model**: `risk_per_trade = max($10, equity * risk_fraction)`

| Equity | Risk (10% frac) | Contracts | SL (ticks) |
|--------|-----------------|-----------|------------|
| $0     | $10             | 1         | 20         |
| $500   | $50             | 5         | 20         |
| $1,000 | $100            | 10        | 20         |
| $5,000 | $500            | 50        | 20         |
| $10,000| $1,000          | 100       | 20         |

**Anti-fragile property**: Drawdowns reduce equity -> reduce position size -> limit further
damage. Winners grow equity -> grow positions -> compound gains.

**SL is always 20 ticks (5 points, $10 per contract)**. Only the NUMBER of contracts changes.

**Risk fraction**: 10% is the starting point. Tunable via `equity_risk_simulator.py --sensitivity`.
Lower fraction = slower growth, smaller drawdowns. Higher = faster growth, bigger drawdowns.

## Risk Budget (empirical from IS data)

From `l2_risk_budget.py` analysis of July 2025 (representative month):

| Metric | Value |
|--------|-------|
| L2 segments ($30+) | 9,870 per month |
| Win rate ($10 SL) | 94.9% |
| Avg reward (survivors) | $22.00 |
| Avg MAE before MFE | $1.25 |
| R:R | 1:2.2 |
| EV per trade | $20.35 |
| Median time to MFE | 2.4 min |
| Zero-MAE trades | 76% (no drawdown at all) |

## Trade Lifecycle

```
1. WAIT    — Monitor 1m DMI continuously. No position.
2. SIGNAL  — DI+/DI- crossover detected on 1m bar close.
3. FILTER  — Build 192D state vector. Classifier says GO/NO-GO.
4. SIZE    — Compute contracts: max(1, floor(risk_budget / $10))
5. ENTER   — Market order in crossover direction. SL = 20 ticks.
6. MANAGE  — Hold until: (a) reverse crossover, (b) MR UCL break, (c) SL hit
7. EXIT    — Close position. Update equity. Back to WAIT.
```

## Validation Plan

### Phase 1: DMI Crossover Accuracy (tool: `dmi_crossover_validation.py`)
- Run on full IS (12 months) and OOS (2 months)
- Measure: win rate, avg MFE, false signal rate, ADX influence
- **Pass criteria**: >85% win rate with $10 SL, avg MFE > $15

### Phase 2: 192D Noise Filter
- Label all crossovers from Phase 1 as regime (MFE>$10) or noise (MFE<$5)
- Train logistic regression on flattened 192D vectors
- **Pass criteria**: filter removes >50% of false signals without removing >10% of real signals

### Phase 3: Equity Simulation (tool: `equity_risk_simulator.py`)
- Run with labeled regime crossovers (after 192D filter)
- Compare flat vs dynamic sizing
- **Pass criteria**: dynamic > 5x flat final equity, max DD < 20% of peak

### Phase 4: Live Dry-Run
- Integrate into `live_engine.py` as alternate strategy mode
- Run `--dry-run` for 1 week, compare to current scalping system
- **Pass criteria**: fewer trades, higher avg PnL, similar or better total

## Key Difference from Current System

| Aspect | Current (Scalper) | New (Regime) |
|--------|-------------------|--------------|
| Entry trigger | Pattern template match | DMI crossover |
| Entry filter | Gate cascade (8 gates) | 192D classifier |
| SL | Variable (oracle-derived) | Fixed 20 ticks ($10) |
| TP | Envelope decay / giveback | Reverse crossover / MR UCL |
| Avg PnL/trade | $1.67 | $20+ (projected) |
| Trade frequency | ~30/day | ~5-10/day |
| Position sizing | 1 contract | Equity-scaled |
| Oracle | Micro-pattern labeling | Regime transition labeling |

## Files

| File | Purpose |
|------|---------|
| `tools/dmi_crossover_validation.py` | Phase 1: validate DMI crossover accuracy |
| `tools/equity_risk_simulator.py` | Phase 3: simulate equity growth with dynamic sizing |
| `tools/l2_risk_budget.py` | Risk budget analysis (MAE/MFE of $30+ segments) |
| `tools/imr_golden_path.py` | I-MR + golden path visualization |
| `core/physics_utils.py` | DMI computation (Wilder 14-period) |
| `core/market_state.py` | DMI fields on MarketState (dmi_plus, dmi_minus, di_plus_prev, di_minus_prev) |

## Open Questions

1. **ADX threshold at crossover**: What minimum ADX gives best signal quality?
2. **Reverse crossover exit vs trailing**: Hold until DI re-crosses, or trail with MR UCL?
3. **Multiple contracts partial exit**: Scale out (e.g., close half at +10 ticks, let rest run)?
4. **Reentry after exit**: If regime continues after exit, re-enter on next crossover?
5. **Session boundaries**: How to handle crossovers near session open/close?
