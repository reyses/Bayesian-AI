# Bayesian-AI System Architecture

> Technical reference for the full pipeline: data → physics → discovery → clustering → forward pass → execution

---

## Pipeline Overview

```
MARKET DATA (ATLAS parquet files, 14 timeframes)
  │
  ▼
Phase 2: DISCOVERY ─────────── fractal_discovery_agent.py
  │  Top-down scan: 1D → 4h → 1h → ... → 1s
  │  Detect ROCHE_SNAP + STRUCTURAL_DRIVE patterns
  │  Oracle classifies each: MEGA / SCALP / NOISE
  │  Output: manifest (List[PatternEvent])
  │
  ▼
Phase 2.5: CLUSTERING ──────── fractal_clustering.py
  │  I-MR on PC1 → coarse geometry segments
  │  DBSCAN on full 16D within each segment → templates
  │  OLS + Logistic regression per template
  │  DNA maps (per-TF centroids + bounds)
  │  Output: HypervolumeTree + List[PatternTemplate]
  │
  ▼
Phase 3: VALIDATION ─────────── orchestrator.py
  │  Analytical exit derivation (TP/SL/trail from MFE/MAE)
  │  Consistency check (temporal halves, PnL CV)
  │  Register to pattern_library
  │  Output: pattern_library.pkl, template_tiers.pkl
  │
  ▼
Phase 4: FORWARD PASS (IS) ─── orchestrator.py + belief_network + wave_rider
  │  10-worker belief network (1h → 1s)
  │  15m DNA anchor matching → multi-TF verification
  │  WaveRider position management (two-phase stops)
  │  Output: oracle_trade_log.csv, phase4_report.txt
  │
  ▼
Phase 5: STRATEGY SELECTION ── orchestrator.py
  │  Rank templates by PnL/Sharpe/WR → assign tiers
  │  Compute depth weights
  │  Output: template_tiers.pkl, depth_weights.json
  │
  ▼
Phase 6: OOS VALIDATION ────── orchestrator.py (same as Phase 4, frozen library)
  │  DATA/ATLAS_OOS (Nov 2025 - Feb 2026)
  │  Output: reports/oos/
```

---

## 1. Physics Engine

**File**: `core/quantum_field_engine.py`
**Class**: `QuantumFieldEngine`

Computes a `ThreeBodyQuantumState` per bar via GPU-accelerated batch processing.

### Per-Bar Computation

| Step | Output | Method |
|------|--------|--------|
| Linear regression (21-bar window) | center, slope, sigma | Convolution-based O(N) |
| Z-score | `z = (price - center) / sigma` | — |
| Three-body forces | gravity, repulsion, momentum, net | Inverse-cubic + reversion |
| Wave function | P0, P1, P2 (3-state quantum probs) | Softmax on energy levels |
| Entropy / Coherence | `entropy / ln(3)` | Shannon entropy |
| ADX / DMI | trend strength, directional bias | Wilder smoothing |
| Hurst exponent | persistence measure | R/S analysis (numba) |
| PID control | oscillation correction term | P=0.5z, I=0.1∫z, D=0.2∂z |
| OU tunneling | first-passage probability | Analytical erfi-based |

### ThreeBodyQuantumState (key fields)

```
z_score, particle_velocity, momentum_strength
coherence, entropy
lagrange_zone: L1_STABLE | CHAOS | L2_ROCHE | L3_ROCHE
adx_strength, hurst_exponent, dmi_plus, dmi_minus
term_pid, oscillation_coherence
tunnel_probability, barrier_height
cascade_detected, structure_confirmed
```

### Pattern Detection Triggers

- **ROCHE_SNAP**: `|z| > 2.0 AND |velocity| > 0.5` — extreme reversion zone
- **STRUCTURAL_DRIVE**: `|momentum| > 5.0 AND coherence < 0.3` — trend breakout

---

## 2. 16D Feature Vector

**Function**: `FractalClusteringEngine.extract_features(pattern)`

| Idx | Name | Source | Live? |
|-----|------|--------|-------|
| 0 | abs(z) | state.z_score | Yes |
| 1 | log1p(v) | state.particle_velocity | Yes |
| 2 | log1p(m) | state.momentum_strength | Yes |
| 3 | coherence | state.coherence | Yes |
| 4 | tf_scale | log2(timeframe_seconds) | Yes |
| 5 | depth | fractal depth (0-4) | No* |
| 6 | parent_ctx | 1.0 if parent is ROCHE | No* |
| 7 | adx | state.adx_strength × 0.01 | Yes |
| 8 | hurst | state.hurst_exponent | Yes |
| 9 | dmi_diff | (dmi+ − dmi−) × 0.01 | Yes |
| 10 | parent_z | parent chain z-score | No* |
| 11 | parent_dmi | parent chain DMI diff | No* |
| 12 | root_is_roche | 1.0 if root = ROCHE | No* |
| 13 | tf_alignment | self × root DMI sign | No* |
| 14 | pid | state.term_pid | Yes |
| 15 | osc_coh | state.oscillation_coherence | Yes |

*Dims [5,6,10,11,12,13] are zeroed in live `state_to_features()` — no parent chain in real-time.

**DNA_LIVE_DIMS** = `[0,1,2,3,4,7,8,9,14,15]` — the 10 dimensions comparable between clustering and live.

---

## 3. Pattern Discovery

**File**: `training/fractal_discovery_agent.py`
**Class**: `FractalDiscoveryAgent`

### Discovery Flow

```
scan_atlas_topdown()
  │
  ├─ Context TFs (1D, 4h, 1h, 30m): enrich parent chain, never block
  ├─ Signal TFs (15m → 1s): detect patterns
  │
  For each TF level:
    ├─ Level 0: _batch_scan_full() — load all files, GPU batch
    ├─ Level 1+: _batch_scan_windowed() — filter to parent windows
    │
    For each bar with cascade_detected OR structure_confirmed:
      ├─ _consult_oracle() — lookahead N bars
      │   ├─ MFE = max favorable excursion
      │   ├─ MAE = max adverse excursion
      │   ├─ Classify: MEGA_LONG/SHORT, SCALP_LONG/SHORT, NOISE
      │   └─ Structural integrity: 16D distance over lookahead
      └─ Create PatternEvent (oracle_marker + oracle_meta + parent_chain)
```

### PatternEvent (key fields)

```
pattern_type: ROCHE_SNAP | STRUCTURAL_DRIVE
timeframe: str (e.g. '15m')
depth: int (0 = top TF, 4 = deepest)
state: ThreeBodyQuantumState
parent_chain: [{tf, z, features_16d, ...}, ...]
oracle_marker: int (MEGA_LONG=2, SCALP_LONG=1, NOISE=0, SCALP_SHORT=-1, MEGA_SHORT=-2)
oracle_meta: {mfe, mae, mfe_bar, structural_integrity}
```

---

## 4. Clustering Pipeline

**File**: `training/fractal_clustering.py`
**Class**: `FractalClusteringEngine`

### Current Pipeline: I-MR → DBSCAN

```
fit_hypervolume_tree(patterns)
  │
  ├─ Build hypervolume matrices (depth × 16D per pattern)
  ├─ Extract 16D features → StandardScaler
  │
  ├─ _imr_geometric_split(feat_scaled):
  │   │
  │   ├─ PHASE 1: I-MR on PC1
  │   │   PCA → project onto PC1 → sort
  │   │   Moving Range between consecutive projections
  │   │   UCL = IMR_D4 (2.0) × mean(MR)
  │   │   Boundaries where MR > UCL
  │   │   Fallback: PC1 median if < 2 segments
  │   │
  │   └─ PHASE 2: DBSCAN on full 16D per I-MR segment
  │       GPU distance matrix (torch.cdist on CUDA)
  │       Auto-tune eps from median k-distance (k=5)
  │       min_samples = max(3, min_group_size // 10)
  │       Noise → nearest cluster reassignment
  │       Small clusters → merge into nearest
  │
  ├─ Returns: labels, lineage (cluster_id → segment_id, sub_id)
  │
  ├─ Build templates per cluster:
  │   ├─ PatternTemplate (centroid, cell bounds, member patterns)
  │   ├─ _aggregate_oracle_intelligence() — stats + OLS/logistic regression
  │   └─ _build_dna_maps() — per-TF centroids + bounds on DNA_LIVE_DIMS
  │
  └─ _analytical_exits() — TP/SL/trail from MFE/MAE distributions
```

### Template Naming

Templates inherit lineage from clustering: `{segment_letter}:{sub_cluster_id}`
- e.g. `A:0`, `A:1`, `B:0`, `C:2`
- Segment letter = I-MR segment (geometry group)
- Sub ID = DBSCAN cluster within that segment

### Key Constants

```python
MIN_GROUP_SIZE = 30        # Minimum patterns per template
IMR_D4 = 2.0              # UCL factor for Moving Range
IMR_MIN_SEGMENTS = 2      # Force at least 2 I-MR segments
```

### PatternTemplate (key fields)

```
template_id: int           # hash(node_id) mod 1000000
centroid: np.ndarray       # 16D centroid
patterns: List[PatternEvent]

# Oracle stats
stats_win_rate, stats_mega_rate, stats_expectancy
long_bias, short_bias, direction
mean_mfe_ticks, mean_mae_ticks, p75_mfe_ticks, p25_mae_ticks
avg_mfe_bar, p75_mfe_bar

# Regression (all 16D)
mfe_coeff: List[float]     # OLS: predicted MFE = coeff @ scaled_features + intercept
mfe_intercept: float
dir_coeff: List[float]     # Logistic: P(LONG) = sigmoid(coeff @ scaled_features + intercept)
dir_intercept: float

# DNA (per-TF ancestry matching)
dna_centroids: {tf_label: 16D array}
dna_bounds_min: {tf_label: 10D array}  # DNA_LIVE_DIMS only
dna_bounds_max: {tf_label: 10D array}

# Exits
best_params: {tp, sl, trail, max_hold_seconds, ...}
```

---

## 5. Belief Network

**File**: `training/timeframe_belief_network.py`
**Class**: `TimeframeBeliefNetwork`

10 parallel workers (1h, 30m, 15m, 5m, 3m, 1m, 30s, 15s, 5s, 1s), each monitoring one timeframe.

### Worker Tasks

| Task | Frequency | What |
|------|-----------|------|
| Aggregation | Once/day | Pre-compute quantum states for all TF bars |
| Analysis | Per TF-bar change | Feature extraction → template match → belief update |

### Three-Path Template Matching

**Path 1 — DNA Anchor (15m worker only, ANCHOR_TF=900s)**
```
Extract feat_s[DNA_LIVE_DIMS] → 10D
For each template's 15m DNA cell:
  If feat_live inside [bounds_min, bounds_max]:
    Cell containment match → set _active_template_id
    Tiebreak by distance to centroid
If no match → _active_template_id = None → no signal
```

**Path 2 — DNA Verification (all other workers)**
```
Get active_tid from 15m anchor
Find DNA cell for active_tid at this worker's TF
If inside cell:
  dna_agreement = 1.0 - (center_distance / max_distance)
If outside:
  dna_agreement = 0.0
If no DNA for this TF:
  dna_agreement = 0.5 (neutral)
```

**Path 3 — Legacy Fallback (no DNA data)**
```
Nearest centroid: dists = ||centroids_scaled - feat_s||
Leaf (15s): top-K=3 parallel match, average outputs
Non-leaf: single nearest centroid
```

### Conviction Pipeline

```
conviction = |dir_prob - 0.5| × 2.0           # Base: how far from 50/50
  × (0.5 + 0.5 × dna_agreement)               # DNA quality scaling [0.5x..1.0x]
  × price_aware_factor                          # Trade side + P&L agreement
  × conviction_scale                            # EOD-learned calibration
  × regret_discount                             # Poor exit zone penalty
```

### Belief Aggregation (get_belief)

```
For each active worker:
  path_long  += weight × log(dir_prob)
  path_short += weight × log(1 - dir_prob)
  weight = tf_weight × (0.5 + 0.5 × dna_agreement)

path_conviction = weighted geometric mean of P(direction)
direction = 'long' if path_long > path_short else 'short'
```

### Physics Blend

```python
phys_dir = sigmoid(-z_raw × sensitivity)  # Mean-reversion oscillator
dir_prob = 0.5 × phys_dir + 0.5 × regression_dir_prob
```

Higher-TF workers have stronger z-sensitivity (more statistical power).

### Key Outputs

**WorkerBelief**: `dir_prob, pred_mfe, template_id, conviction, wave_maturity, dna_agreement`

**BeliefState**: `direction, conviction, predicted_mfe, active_levels, wave_maturity, tf_beliefs`

---

## 6. Position Management (Wave Rider)

**File**: `training/wave_rider.py`
**Class**: `WaveRider`

### Two-Phase Stop Logic

```
Phase 1 (INITIAL):
  Wide hard stop = SL from template params
  Hold until profit_ticks >= trail_activation_ticks
  trail_activation = max(3, profit_target × 0.15)

Phase 2 (TRAILING):
  Trail from high_water_mark
  Distance adapts to wave_maturity:
    < 0.3 (early):  trail × 1.5  — let trade develop
    0.3-0.7 (mid):  trail × 1.0
    > 0.7 (late):   trail × 0.5  — protect gains
```

### Adaptive Trail (Belief Network Signals)

| Signal | Condition | Factor |
|--------|-----------|--------|
| Tighten | wave_maturity > 0.85 OR time-exhausted | × 0.92 (floor: 60% of original) |
| Widen | confident + aligned + wave < 0.30 | × 1.30 (cap: 3× original) |
| Breakeven | profit >= original trail distance | Lock stop at entry ± 1 tick |

### Exit Reasons

| Exit | Trigger |
|------|---------|
| Stop Loss | Phase 1 hard stop hit |
| Trail Stop | Phase 2 trailing stop hit |
| Profit Target | PT reached (may extend in runner mode) |
| Structure Break | 16D vector leaves hypervolume cell bounds (CST) |
| Physics Decay | Z-score drift cascade > 1.5 threshold |
| Loss Watchdog | DMI inverse + underwater ≥ 8 ticks + ≥ 5 workers disagree |
| Runner Mode | At PT + conviction > 0.6 → extend PT 1.5×, tighten trail 0.6× |

### Regret Analysis (Post-Trade)

5-minute delayed review after exit:
```
exit_efficiency = actual_pnl / potential_max_pnl
  ≥ 90%: optimal
  peak after exit: closed_too_early
  gave back > 20%: closed_too_late
  pnl < 0: wrong_direction
```

Feeds back into worker calibration (conviction_scale, regret discount).

---

## 7. Spectral Gates

**File**: `training/orchestrator_worker.py`

Applied during forward pass trade simulation:

| Gate | Logic | Purpose |
|------|-------|---------|
| Fourier | Block TP before half-cycle completes | Prevent premature profit-taking |
| Laplace | Exit if kinetic damping > 0.8 | Energy exhaustion detection |
| TP | Hit take_profit price | Standard profit target |
| SL | Hit stop_loss price | Standard stop loss |
| Time | Exceeded max_hold_seconds | Prevent stale trades |

---

## 8. DNA Tree

**File**: `training/fractal_dna_tree.py`
**Class**: `FractalDNATree`

Hierarchical clustering of patterns by timeframe ancestry. Builds LONG and SHORT trees separately.

### Structure

```
Root (LONG)
  └─ 1h: cluster 0, 1, 2
       └─ 30m: cluster 0, 1
            └─ 15m: cluster 0, 1, 2
                 └─ 5m: cluster 0
                      └─ 15s: leaf patterns
```

Each node stores: centroid, member_count, oracle stats (win_rate, mean_mfe, expectancy).

### Matching

Top-down traversal from 1h → 15s:
- At each TF level: find nearest child centroid
- Accumulate path → `PatternDNA(path=['1h:3', '15m:7', '5m:2', '15s:9'])`
- Confidence = `1.0 / (1.0 + avg_distance)`

---

## 9. Bayesian Brain

**File**: `core/bayesian_brain.py`
**Class**: `BayesianBrain`

HashMap-based probability engine for empirical learning.

### Lookup

```python
table[state_key] = {'wins': X, 'losses': Y, 'total': X+Y}
```

### Decision

```python
probability = (wins + 1) / (total + 11)    # Beta(1,10) pessimistic prior
confidence  = min(total / 100, 1.0)         # Full trust at 100 trades
should_fire = (probability >= 0.80) AND (confidence >= 0.30)
```

---

## 10. CLI Reference

| Flag | Effect |
|------|--------|
| `--fresh` | Wipe all checkpoints, full pipeline |
| `--train-only` | Phases 2-3 only (no forward pass) |
| `--forward-pass` | IS → Strategy → OOS auto-chain |
| `--forward-pass --skip-oos` | IS → Strategy only |
| `--oos` | Standalone OOS rerun |
| `--data DATA/ATLAS_1DAY` | Single-day fast validation (~3s) |
| `--forward-data PATH` | Custom data for forward pass |
| `--min-tier N` | Filter templates below tier N |
| `--strategy-report` | Phase 5 only |

---

## 11. File Map

| File | Role |
|------|------|
| `core/quantum_field_engine.py` | ThreeBodyQuantumState per bar (GPU physics) |
| `core/bayesian_brain.py` | HashMap probability table + decision gate |
| `training/fractal_discovery_agent.py` | Top-down pattern discovery + oracle classification |
| `training/fractal_clustering.py` | I-MR → DBSCAN clustering, templates, DNA maps |
| `training/fractal_dna_tree.py` | Hierarchical TF context tree |
| `training/orchestrator.py` | Main pipeline, forward pass, pattern library |
| `training/orchestrator_worker.py` | Spectral gates, analytical exits, trade simulation |
| `training/timeframe_belief_network.py` | 10-worker multi-TF consensus + DNA matching |
| `training/wave_rider.py` | Position management, adaptive trails, regret analysis |
| `training/pipeline_checkpoint.py` | Phase checkpointing + resume |

---

## 12. Data Flow Diagram

```
                    ATLAS (14 TFs, parquet)
                           │
              ┌────────────┤
              ▼            ▼
      QuantumFieldEngine   FractalDiscoveryAgent
      (physics per bar)    (pattern detection)
              │                    │
              │            PatternEvent list
              │                    │
              │            FractalClusteringEngine
              │            ├─ I-MR on PC1
              │            ├─ DBSCAN on 16D
              │            ├─ OLS/Logistic regression
              │            └─ DNA maps
              │                    │
              │            PatternTemplate list
              │                    │
              │            pattern_library (dict)
              │                    │
              ▼                    ▼
         ┌─────────────────────────────┐
         │   TimeframeBeliefNetwork    │
         │   10 workers (1h → 1s)     │
         │   15m anchor → DNA verify   │
         │   Physics blend + playbook  │
         │   → BeliefState            │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │     WaveRider        │
         │  Two-phase stops     │
         │  Adaptive trail      │
         │  Spectral gates      │
         │  Regret analysis     │
         └──────────┬───────────┘
                    │
                    ▼
              Trade Logs
         (oracle_trade_log.csv)
```

---

## 13. Checkpoint Files

| File | Phase | Contents |
|------|-------|----------|
| `checkpoints/discovery_manifest.pkl` | 2 | All PatternEvent objects |
| `checkpoints/hypervolume_tree.pkl` | 2.5 | HypervolumeTree (nodes + templates) |
| `checkpoints/pattern_library.pkl` | 3 | Dict: tid → {centroid, params, stats, regression, DNA} |
| `checkpoints/template_tiers.pkl` | 5 | Dict: tid → tier (1-4) |
| `checkpoints/depth_weights.json` | 5 | Per-depth PnL weights |
| `checkpoints/fractal_dna_tree.pkl` | 3 | FractalDNATree (hierarchical TF clusters) |
| `checkpoints/template_occurrences.parquet` | 2.5 | Pattern-to-template mapping audit |
