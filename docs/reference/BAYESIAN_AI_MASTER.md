# Bayesian-AI Trading System — Master Architecture Document
**Version:** 2.0  
**Last Updated:** March 16, 2026  
**Author:** Moises (system designer) + Claude/Gemini (implementation partners)  
**Platform:** NinjaTrader 8  
**Instrument:** MNQ / NQ (Micro/Full Nasdaq-100 E-mini Futures)  
**Status:** Live simulation validated (67.3% WR, 2.29 PF, 55 trades/6hrs)

---

## 1. SYSTEM OVERVIEW

Bayesian-AI is a multi-component automated trading system for NQ/MNQ futures. It combines physics-inspired market modeling, fractal pattern discovery, multi-timeframe consensus voting, Bayesian probability learning, and adaptive execution. The system executes via NinjaTrader 8.

### Core Thesis

The market behaves like a quantum particle trapped between competing gravitational wells (the Three-Body Problem). Price orbits between real liquidation pool levels where stop orders cluster. The system identifies these structural levels, detects repeating patterns at those levels, learns which patterns produce profitable outcomes, and executes trades when probability thresholds are met.

### Performance (Most Recent Validated Run)

| Metric | Value |
|--------|-------|
| Net P&L | +$1,496 |
| Win Rate | 67.3% |
| Profit Factor | 2.29 |
| Total Trades | 55 (over ~6 hours) |
| Gross Wins | +$2,652 |
| Gross Losses | -$1,156 |
| Instrument | MNQ (1 contract) |
| Equity Curve | Clean upward slope, controlled drawdowns |

---

## 2. ARCHITECTURE (6-Layer Pipeline)

The system processes data through six sequential layers. Each layer transforms its input and passes results downstream.

```
Market Data (OHLCV + Tick)
        │
        ▼
┌─ LAYER 1: PHYSICS ENGINE (Three-Body Quantum Field Model) ─┐
│  Input:  Raw price bars across multiple timeframes           │
│  Output: ThreeBodyQuantumState (16D+ vector per bar)         │
│  Computes: z-score, forces, wave function, coherence         │
│  Mode: Level-anchored (preferred) or statistical fallback    │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ LAYER 2: PATTERN DISCOVERY (Fractal Discovery Agent) ──────┐
│  Input:  ThreeBodyQuantumState time series                   │
│  Output: PatternEvent list                                   │
│  Detects: Roche Limit breaks, Structural Drives,             │
│           Tunnel Events, Coherence Collapses                 │
│  Scope: Multi-timeframe fractal hierarchy (1D → 15s)         │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ LAYER 3: CLUSTERING (K-Means + DOE Optimization) ──────────┐
│  Input:  PatternEvent feature vectors                        │
│  Output: Reusable Pattern Templates with optimized TP/SL    │
│  Method: K-Means Euclidean distance → nearest centroid       │
│  DOE: Screening (200 params) → Characterization → RSM       │
│  Creates: 16×12 geometric template library                   │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ LAYER 4: CONSENSUS (Timeframe Belief Network) ─────────────┐
│  Input:  Current state across 8 timeframes                   │
│  Output: Directional conviction score                        │
│  Workers: 1D, 4h, 1h, 30m, 15m, 5m, 1m, 15s               │
│  Rule: Majority agreement + conviction > 0.48 required       │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ LAYER 5: VALIDATION (Bayesian Brain) ───────────────────────┐
│  Input:  Matched template ID                                 │
│  Output: P(win) for this template in current conditions      │
│  Map: TemplateID → {wins, losses, total}                     │
│  Threshold: Adaptive (starts at 80% prob + 30% confidence)   │
│  Learning: Laplace-smoothed Bayesian updates after each trade│
│  Persistence: Save/load probability tables (.pkl)            │
└──────────────────────────────────────────────────────────────┘
        │
        ▼
┌─ LAYER 6: EXECUTION (WaveRider) ────────────────────────────┐
│  Input:  Approved trade signal with template parameters      │
│  Output: Filled orders via NinjaTrader 8                     │
│  Features: Adaptive trailing stops (10/20/30 ticks by PnL), │
│            Structure break detection, Regret analysis,       │
│            Auto-calibrating trail distances every 10 trades  │
└──────────────────────────────────────────────────────────────┘
```

### Decision Flow (Per Bar)

```python
# Pseudocode for the main loop
quantum_state = physics_engine.compute_state(bar_data)
physics_event = discovery_agent.detect_event(quantum_state)

if not physics_event:
    return  # No pattern detected

template = clustering.match_template(physics_event)
conviction = belief_network.vote(quantum_state)

if conviction < 0.48:
    return  # No timeframe consensus

win_prob = bayesian_brain.get_probability(template.id)

if win_prob < adaptive_threshold:
    return  # Insufficient historical probability

wave_rider.execute(template)  # Place order via NinjaTrader 8
```

---

## 3. COMPONENT DETAILS

### 3.1 Three-Body Quantum Field Engine (Layer 1)

The market is modeled as a particle (current price) trapped between three gravitational bodies:

**Body 1 — Center Attractor (Fair Value / Ground State):**
Pulls price toward equilibrium. Force follows Hooke's Law: F = -k(x - x₀). In statistical mode, this is the linear regression line. In level-anchored mode, this is the nearest major liquidation level.

**Body 2 — Upper Singularity (Resistance / Excited State):**
Repulsive force at +2σ (or upper liquidation level). Models resistance where sellers overwhelm buyers. Event horizon at +3σ represents point of no return (breakout).

**Body 3 — Lower Singularity (Support / Excited State):**
Repulsive force at -2σ (or lower liquidation level). Models support where buyers overwhelm sellers. Event horizon at -3σ represents capitulation (breakdown).

#### ThreeBodyQuantumState Fields

The state vector produced per bar contains (16D+ dimensions):

- **Attractor positions:** center_position, upper_singularity, lower_singularity, event_horizon_upper, event_horizon_lower
- **Particle state:** particle_position (current price), particle_velocity (momentum $/sec), z_score (normalized distance from center in σ units)
- **Force fields:** F_reversion (center pull, tidal force = z²/9), F_upper_repulsion (1/r³), F_lower_repulsion (1/r³), F_momentum (kinetic energy)
- **Wave function:** amplitude coefficients a₀ (center), a₁ (upper), a₂ (lower), satisfying |a₀|² + |a₁|² + |a₂|² = 1
- **Probabilities:** P_at_center, P_near_upper, P_near_lower
- **Decoherence:** pattern_maturity (0 = superposition, 1 = collapsed), momentum_strength
- **Measurement:** observation_made (L8 confirmation), collapse_velocity (L9 cascade)
- **Outcome:** dominant_attractor ('CENTER'|'UPPER'|'LOWER'), tunnel_probability (revert to center), escape_probability (break through singularity)

#### Two Operating Modes

**Statistical Mode (Fallback):** Center = rolling regression line, boundaries = ±2σ bands. Levels follow price (less predictive). Used when no manual levels are loaded.

**Level-Anchored Mode (Preferred):** Center/Upper/Lower = real horizontal price levels where liquidation pools exist (stop order clusters). Levels are fixed structural features. Price orbits them. This mode was a critical breakthrough insight — the three bodies should anchor to real market structure, not statistical abstractions.

#### Level Management

Levels are stored in `price_levels.json`:

```json
{
  "levels": [
    {
      "price": 25087.25,
      "strength": 1.0,
      "timeframe": "intraday",
      "source": "manual",
      "touch_count": 4,
      "break_count": 0,
      "notes": "Clear resistance, multiple rejections"
    }
  ]
}
```

The LiquidationLevelManager loads levels, finds nearest above/below current price, tracks touches and breaks, and deactivates levels that break too many times.

### 3.2 Fractal Discovery Agent (Layer 2)

Scans the ThreeBodyQuantumState time series for "physics events" — moments where the quantum state transitions meaningfully:

- **Roche Limit Break:** Price escapes ±3σ (or breaches event horizon). Signals potential cascade/breakout.
- **Structural Drive:** Momentum + force alignment across timeframes. Signals directional commitment.
- **Tunnel Event:** Reversion probability spikes. Price near singularity but forces favor snap-back to center.
- **Coherence Collapse:** Wave function collapses from superposition to definite state (pattern matures from ambiguous to clear).

The agent operates across a fractal timeframe hierarchy from 1D down to 15s, detecting events at each scale. Rule-based triggers (e.g., if z > 3.0, if force alignment > threshold).

**Planned Enhancement:** CNN Pattern Builder to replace or augment this agent. The CNN would learn visual chart patterns (flags, wedges, H&S, triangles) directly from OHLCV data, discovering patterns the physics rules don't encode.

### 3.3 K-Means Clustering + DOE (Layer 3)

Groups similar PatternEvents into reusable templates using K-Means clustering on the feature vectors. Each template becomes a centroid that represents a class of market setup.

**Template Library:** 16-dimensional feature vectors across 12 timeframe slices = 16×12 geometric templates.

**DOE (Design of Experiments) Optimization:** For each template, DOE optimizes TP (take-profit) and SL (stop-loss) parameters using a three-phase approach:
1. **Screening:** Fractional factorial on ~200 parameters at 2 levels (low/high). ~1000 runs. Eliminates ~180 insignificant parameters.
2. **Characterization:** Full factorial on ~20 surviving parameters at 3 levels. ~500 runs. Identifies critical interactions.
3. **Response Surface Methodology (RSM):** Central composite design on ~15 critical parameters at 5 levels. ~300 runs. Finds optimal configuration with confidence intervals.

**Critical Note:** DOE must use Days 1→N-1 to optimize for Day N (walk-forward). A prior bug used look-ahead data which produced catastrophically overfit results (14.5% WR on 7,806 trades, -$60K).

### 3.4 Timeframe Belief Network (Layer 4)

Eight independent worker agents, each analyzing a single timeframe:

| Worker | Timeframe | Role |
|--------|-----------|------|
| 1 | 1 Day | Macro trend direction |
| 2 | 4 Hour | Swing structure |
| 3 | 1 Hour | Intraday trend |
| 4 | 30 Min | Session structure |
| 5 | 15 Min | Setup formation |
| 6 | 5 Min | Entry timing |
| 7 | 1 Min | Micro confirmation |
| 8 | 15 Sec | Trigger precision |

Each worker votes independently on direction (LONG/SHORT/NEUTRAL). Trade requires majority agreement plus conviction score > 0.48. This provides redundancy and interpretability — you can see which timeframes agree and which dissent.

### 3.5 Bayesian Brain (Layer 5)

Probability lookup table mapping template IDs to historical outcomes:

```python
probability_table = {
    template_id: {
        'wins': int,
        'losses': int,
        'total': int
    }
}
```

**Learning Loop:**
```python
# After trade completes
outcome = TradeOutcome(template_id=matched_template, result='WIN', pnl=40)
brain.update(outcome)  # Bayesian update with Laplace smoothing

# Before next trade with same template
if brain.should_fire(template_id):  # prob > threshold AND confidence > min_samples
    execute_trade()
```

**Thresholds:** 80% win probability + 30% confidence (minimum sample size). These are adaptive — the system tightens/loosens based on recent performance.

**Persistence:** Probability tables saved/loaded as pickle files. The brain accumulates knowledge across sessions.

### 3.6 WaveRider Execution Module (Layer 6)

Manages position lifecycle with adaptive exits:

**Adaptive Trailing Stops:**
- Tight (10 ticks): When profit < $50
- Medium (20 ticks): When profit $50–$150
- Wide (30 ticks): When profit > $150

Trail distances auto-calibrate every 10 trades based on regret analysis (did we exit too early or too late?).

**Regret Analysis:** After each trade, WaveRider tracks:
- Price at exit vs. price at peak favorable excursion
- Categorizes exits: Optimal (>80% of move captured), Partial (20-80%), Early (<20%), Reversed (gave back >92%)
- Recommends trail adjustments: WIDEN if closing too early, TIGHTEN if closing too late

**NinjaTrader 8 Integration:** Places orders through NinjaTrader's API. Supports market, limit, and OCO bracket orders.

---

## 4. STATE VECTOR SYSTEM (Phase 1 Core)

The original Phase 1 implementation used a simpler 9-layer StateVector before the Three-Body model was developed:

**Layers 1–4 (Static Context — computed once per session at 6:45 AM):**
- L1: 90-day macro bias (bull/bear/neutral)
- L2: 30-day regime (trending/ranging/volatile)
- L3: 1-week swing structure (higher_highs/lower_lows/etc.)
- L4: Daily zone (at_killzone/mid_range/extended)

**Layers 5–6 (Fluid — CPU updates during session):**
- L5: 4-hour trend direction
- L6: 1-hour market structure

**Layers 7–9 (Real-time — designed for CUDA acceleration):**
- L7: 15-min pattern detection (flags, wedges, compression)
- L8: 5-min volume/structure confirmation
- L9: 1-second velocity cascade detection (10+ point moves in <0.5s)

**Hashing:** The StateVector is immutable and hashable (excludes timestamp/price metadata). Same market conditions produce the same hash, enabling HashMap lookups in the Bayesian Brain.

The ThreeBodyQuantumState replaced this as the primary state representation but the layered concept (static vs. fluid vs. real-time) persists in the architecture.

---

## 5. NIGHTMARE PROTOCOL (Level Classification Framework)

A hierarchical system for classifying market levels by structural importance. Developed by the user for manual chart analysis, later integrated into the AI system.

### Level Hierarchy

| Level Type | Visual | Origin | Physics | AI Treatment |
|-----------|--------|--------|---------|-------------|
| TITANS | Solid lines (green/red) | Monthly/Weekly Open/High/Low | HORMIGÓN (concrete) | Hard stop. Price will bounce. Don't trade breakout directly. |
| ARCHITECTS | Dashed lines (green/red) | Daily/Session High/Low, 4H levels | ELÁSTICO (elastic) | Trap zone. Wait for false break and re-entry. |
| EXPLORERS | Dotted lines | VWAP, 1H, 15M levels | PAPEL (paper) | Noise. Break easily. Only useful as entry triggers. |

### Visual Color System

- **Solid Green/Red:** Concrete limits (TITANS). Don't move until new patterns form.
- **Dashed Green/Red:** Recent 4H levels (ARCHITECTS).
- **Dotted (any color):** 1H levels (EXPLORERS).
- **Yellow:** Entry point (buy or sell trigger).
- **Cyan:** Suggested target/expected limit.

### Session Phases

- **Asia (Overnight):** Accumulation phase. Price builds range.
- **Europe (Pre-Market):** Manipulation phase. False moves to trap traders.
- **America (RTH Open):** Resolution phase. True directional move.

### Key Pattern Rules

- **God Candle:** 1H body > 70 pts + wick < 10 pts = liquidation mode. Do NOT trade against.
- **Stop Hunt Signal:** 15M wick > 20 pts at key level = failed stop hunt. Entry signal.
- **Liquidity Spring:** Price breaks Hard Limit but reclaims within 3 candles (V-shape reversal). Do not short the first breakdown.
- **Megaphone Expansion:** Consecutive support tests with expanding wicks (5pt → 10pt → 30pt). Standard stops will fail. Trade the extremes.

---

## 6. CNN INTEGRATION (Experimental Branch)

### Purpose

Add visual pattern recognition to complement/augment the physics-based Fractal Discovery Agent. The CNN learns patterns directly from OHLCV chart data that rule-based physics might miss.

### Architecture: Three-Path CNN Pattern Builder

```
Market Data (OHLCV bars)
        │
   ┌────┴────┬────────────┐
   ▼         ▼            ▼
PHYSICS   CNN PATH    REFERENCE
 PATH     (New)        LEVELS
(Keep)       │         (Shared)
   │         ▼            │
   │   GeometricSignature │
   │         │            │
   ▼         ▼            │
Fractal   CNN Pattern ◄──┘
Discovery  Detector
   │         │
   └────┬────┘
        ▼
  Combined Pattern Signal
        │
        ▼
  Belief Network → Bayesian Brain → WaveRider
```

### CNN Inputs (Three Paths)

1. **OHLCV Visual Shape:** 60-bar window of normalized price bars. 1D CNN with kernel sizes 3/5/9 learns local, mid-range, and longer patterns.
2. **Distance to Liquidation Levels:** Scalar features encoding proximity to nearest support/resistance.
3. **Physics Confirmation State:** ThreeBodyQuantumState features (z-score, forces, coherence).

### Multi-Timeframe Design

Four parallel CNN encoders (1H, 15M, 5M, 1M), each processing a GeometricSignature for its timeframe. A cross-attention layer learns which timeframe to trust in different market conditions. Single forward pass produces unified pattern classification + direction prediction.

### GeometricSignature

Encodes each bar's relationship to reference levels as a 7-dimensional feature vector per bar (60 bars = 60×7 input):
- Normalized OHLCV (4 features)
- Distance to nearest resistance
- Distance to nearest support
- Wick penetration depth at levels

### Integration Options (Ranked by Risk)

1. **Validation Layer (Safest):** Add CNN after Bayesian filter. CNN confirms/rejects signals. Expected impact: WR 67% → 70-72%, fewer but higher-quality trades.
2. **Augment Template Matching:** Replace Euclidean distance with CNN-learned embeddings for template similarity scoring.
3. **Replace Discovery Agent (Boldest):** CNN replaces physics-based event detection entirely.

### Implementation

- Developed on `experimental-cnn` git branch (isolated from production)
- Feature flags toggle CNN on/off for A/B comparison
- Training data sourced from `oracle_trade_log.csv` (historical trade outcomes)

---

## 7. CRITICAL BUGS (Historical — Fixed)

These bugs caused a catastrophic failure (14.5% WR, -$60K over 7,806 trades) before being identified and corrected:

1. **Look-Ahead Bias in DOE:** DOE optimization used future data to set TP/SL parameters. Fix: Walk-forward design (optimize on Days 1→N-1, test on Day N).
2. **Inverted TP/SL:** Take-profit and stop-loss values were swapped in some templates. Fix: Explicit direction-aware parameter assignment with debug verification.
3. **Direction Logic Error:** SHORT signals fired when z > 0 should have been LONG (and vice versa in some edge cases). Fix: Corrected conditional logic with unit tests.
4. **State Space Explosion in Brain:** 5,877 unique states with almost no repeat observations meant the Bayesian Brain never accumulated enough data to learn. Fix: Coarser state binning to increase observation density per state.

---

## 8. LIQUIDATION LEVEL ANCHORING (Key Innovation)

### The Insight

Standard quantitative approaches compute levels statistically from price data (e.g., rolling regression ± standard deviations). These levels follow price and have limited predictive power.

The breakthrough: The three bodies in the physics engine should anchor to real horizontal price levels where stop-order clusters (liquidation pools) exist. These levels are fixed structural features of the market. Price orbits them. They don't follow price — price reacts to them.

### Why It Matters

The same pattern means different things depending on level context:

- **V-reversal at a liquidation level:** Bounce off real stop-order pool. High-confidence signal.
- **V-reversal in empty space:** Random noise. Ignore.
- **Step function at a level:** Breakthrough/cascade as stops trigger. Momentum signal.
- **Oscillation between levels:** Price trapped between two pools. Range-bound.

### Implementation Path

1. **Manual level identification:** User marks 5–10 active levels from daily chart, stored in `price_levels.json`.
2. **Distance features:** For each bar, compute distance to nearest level above/below.
3. **Physics anchoring:** Replace statistical regression mean with nearest level as equilibrium point.
4. **CNN context:** Feed distance-to-level as input alongside OHLCV shape.
5. **Validation:** Compare statistical vs. level-anchored performance.
6. **Auto-detection (future):** Algorithmic level detection via peak clustering (DBSCAN or similar).

---

## 9. DEVELOPMENT ENVIRONMENT & FILE STRUCTURE

### Tools

- **Trading Platform:** NinjaTrader 8 (execution)
- **Language:** Python
- **GPU:** CUDA-capable (for K-Means clustering, future CNN training)
- **IDE:** VS Code with Claude integration
- **Version Control:** Git with feature branches

### Repository Structure

```
bayesian-ai/
├── core/
│   ├── state_vector.py              # 9-layer immutable state (Phase 1)
│   ├── three_body_state.py          # ThreeBodyQuantumState (Phase 2)
│   ├── bayesian_brain.py            # Probability table + Bayesian updates
│   ├── layer_engine.py              # Static/fluid context computation
│   ├── quantum_field_engine.py      # Three-Body physics (stat + level-anchored)
│   ├── fractal_discovery_agent.py   # Physics event detection
│   ├── belief_network.py            # 8-worker timeframe voting
│   └── wave_rider.py               # Adaptive execution + regret analysis
│
├── clustering/
│   ├── kmeans_templates.py          # Pattern template clustering
│   └── doe_optimizer.py             # DOE screening → characterization → RSM
│
├── config/
│   ├── symbols.py                   # Asset profiles (MNQ $0.50/tick, MES $1.25/tick, NQ $5/tick)
│   └── experimental_config.py       # Feature flags for CNN toggle
│
├── checkpoints/
│   ├── price_levels.json            # Manual liquidation levels
│   └── probability_table.pkl        # Bayesian Brain saved state
│
├── cuda/
│   ├── pattern_detector.py          # L7 CUDA kernel (15-min patterns)
│   ├── confirmation.py              # L8 CUDA kernel (5-min validation)
│   └── velocity_gate.py             # L9 CUDA kernel (1-sec cascade)
│
├── experimental-cnn/                # Feature branch
│   ├── core/
│   │   ├── geometric_signature.py   # Bar → reference level relationships
│   │   ├── cnn_pattern_validator.py # Option 4: validation layer
│   │   ├── cnn_pattern_detector.py  # Option 1: discovery replacement
│   │   └── cnn_models.py           # Neural network architectures
│   ├── training/
│   │   ├── cnn_trainer.py
│   │   ├── cnn_data_pipeline.py
│   │   └── cnn_evaluator.py
│   └── config/
│       └── experimental_config.py   # CNN feature flags
│
├── training/
│   └── orchestrator.py              # Training loop (1000 iterations)
│
├── DATA/ATLAS/                      # Historical market data
├── oracle_trade_log.csv             # Trade outcome history
│
└── tests/
    ├── test_phase1.py               # Core logic validation
    └── test_phase2.py               # CUDA validation
```

---

## 10. TRAINING & DATA

### Training Loop

```python
for iteration in range(1000):
    for tick in historical_data:
        state = engine.compute_current_state(tick)
        if brain.should_fire(state):
            outcome = simulate_trade(tick)
            brain.update(outcome)
```

### Data Source

Historical NQ/MNQ data (1-year minimum). Previously sourced from Databento ($125 free credits). Stored in DATA/ATLAS directory.

### Walk-Forward Validation

Train on Days 1→N-1, test on Day N. Slide forward. This prevents look-ahead bias that caused the catastrophic failure.

---

## 11. RISK PARAMETERS (For Live/Sim Deployment)

| Parameter | MNQ Value | NQ Value |
|-----------|-----------|----------|
| Tick Value | $0.50 | $5.00 |
| Position Size | 1 contract | 1 contract |
| Stop Loss | 20 ticks ($10) | 20 ticks ($100) |
| Take Profit | 40 ticks ($20) | 40 ticks ($200) |
| Risk:Reward | 2:1 minimum | 2:1 minimum |
| Daily Max Loss | $200 | $200 |
| Max Drawdown (Eval) | ~$2,000 | ~$2,000 |

### Scaling Plan

Start on MNQ (micro) to validate edge with minimal risk. Same probability table works for both instruments (price patterns are identical, only dollar risk differs). Scale to NQ once edge is statistically proven over 100+ trades.

---

## 12. OPEN DEVELOPMENT ITEMS

### In Progress

- **Liquidation level encoding:** Articulating the implicit rules for identifying and weighting levels across a fractal timeframe hierarchy so they can be encoded into the AI.
- **CNN Pattern Builder:** Experimental branch. Three-path architecture (OHLCV shape + level distance + physics state). Multi-timeframe transformer with cross-attention.

### On the Horizon

- **Level Detection CNN:** Automated identification of liquidation pools from chart data (replacing manual marking).
- **CNN integration into Fractal Discovery Agent:** Taking three inputs: OHLCV visual shape, distance to liquidation levels, physics confirmation state.
- **Live deployment:** Moving from simulation to live NinjaTrader execution.
- **Trail stop optimization:** Using regret analysis data to converge on optimal adaptive trail parameters per template type.

### Design Principles

- New features go on experimental git branches (never touch production).
- CNN development is not the bottleneck if the underlying system already performs well — fixing bugs and stabilizing the working system takes priority over adding complexity.
- The system should work WITH the user's natural cognitive style (visual, level-based, intuitive) rather than imposing foreign methodologies.

---

## 13. KEY LEARNINGS & PRINCIPLES

1. **Liquidation levels > statistical levels.** The Three-Body Engine gains its predictive power when anchored to real market structure where stop orders cluster, not to regression lines that follow price.

2. **DOE must be walk-forward.** Any optimization that uses future data will produce spectacular backtest results and catastrophic live performance.

3. **State space must be manageable.** The Bayesian Brain needs enough repeat observations per state to learn. Coarser binning beats fine-grained uniqueness.

4. **The combination is the edge.** No single layer is sufficient. WHERE (liquidation levels) + ACTIVE (physics confirmation) + HOW (pattern shape) together create the signal.

5. **Stabilize before adding complexity.** A working 67.3% WR system should be preserved and understood before CNN or other enhancements are layered on top.

6. **Experimental branches protect production.** Feature flags allow A/B comparison without risking the working system.

---

*End of document. Any AI resuming work on this system should start by reading this document, then examining the actual codebase for current implementation state.*
