# Feature Spec v2 — Layered Architecture

> Supersedes the flat 91D structure in `core/features.py`.
> Drafted 2026-04-23. Source of today's session findings + user architecture input.

## Purpose

Replace the current 91D feature vector with a **layered architecture** where:

1. Every feature has **one clear physical meaning**
2. Every window is **explicit in the feature name** (no hidden parameters)
3. No **duplicate signal** across different-named features
4. **True primitives** (1-bar Δ) exist alongside smoothed variants
5. **Missing coverage** (kurtosis, ATR, efficiency, price-vol correlation) is added
6. The name tells you the **layer**, the **TF**, and the **window** — reading any feature name alone answers "what does this measure and over what horizon"

The motivation is CNN training quality: the current 91D mixes layers, hides windows, and duplicates signal. The model can't find clean boundaries.

---

## Design principles

### Principle 1 — **Naming convention**

Format: `<L#>_<TF>_<name>_<window>`

- `L2_1m_velocity_1b` — layer 2, 1m TF, 1-bar velocity, window = 1 bar
- `L4_15m_beta_12` — layer 4, 15m TF, OLS slope, window = 12 bars
- `L5_1h_z_se_60` — layer 5, 1h TF, z-score residual, window = 60 bars
- `L6_5m_kurt_50` — layer 6, 5m TF, kurtosis, window = 50 bars

Benefits: impossible to mis-read. Window choice is always visible. Refactoring a window = explicit rename.

### Principle 2 — **Windows scaled by estimator type, anchored on "3× higher TF"**

**Base rule**: For TF at rank `k` in the hierarchy `[15s, 1m, 5m, 15m, 1h, 1D]`, the **base window** N_base = 3 × period_of_TF_{k+1}.

| TF | Next higher | N_base |
|---|---|---:|
| 15s | 1m | 12 bars (3 × 60s / 15s) |
| 1m | 5m | 15 bars |
| 5m | 15m | 9 bars |
| 15m | 1h | 12 bars |
| 1h | 4h | 12 bars |
| 1D | 1W | 5 bars (3 × 5D / 1D) |

**Scaling by estimator stability requirements** (decision 2026-04-23):

Different statistical objects need different sample sizes to converge. A uniform N=12 across every feature produces stable OLS slopes but meaningless 4th moments. The rule is tiered:

| Layer / feature class | Multiplier on N_base | Rationale |
|---|---:|---|
| L4 `beta_N`, `gamma_N` (OLS slope, OLS curvature) | **× 1** | OLS converges fast; N≥10 is sufficient |
| L5 `z_se_N`, `z_high_N`, `z_low_N`, `SE_N` | **× 1** | Same OLS fit; already validated by 1h sweep (65-67% within ±1σ at N_base = 12) |
| L5 `p_center_N`, `reversion_prob_N` | **× 1** | Derived from same OLS |
| L6 `sigma_N`, `vr_short_long` | **× 1** | 2nd moment is stable at N≈10-15 |
| L6 `kurt_N`, `L7 skew_N` | **× 4** | 4th/3rd moments need ~50 samples for stable estimate |
| L6 `atr_N`, `cv_N` | **× 1** | Simple averages, OK at N_base |
| L7 `hurst_N`, `efficiency_N` | **× 8** | Hurst needs N≥100 to distinguish trend from mean-revert |
| L8 `vol_rel_N`, `vol_delta_N` | **× 2** | Volume is bursty; needs a bit more smoothing than price |

**Concrete window table per TF** (defaults):

| TF | N_base | L4/L5/L6 scale | L6 kurt, L7 skew | L7 hurst |
|---|---:|---:|---:|---:|
| 15s | 12 | 12 | 48 | 96 |
| 1m | 15 | 15 | 60 | 120 |
| 5m | 9 | 9 | 36 | 72 |
| 15m | 12 | 12 | 48 | 96 |
| 1h | 12 | 12 | 48 | 96 |
| 1D | 5 | 5 | 20 | 40 |

Each feature takes a window parameter; the DEFAULT is this table, but ANY window can be requested per-feature via rename (e.g., `L5_1h_z_se_60` is a legal override).

### Principle 3 — **One window per computation, labeled in the name**

If a feature needs two windows (like variance_ratio), the name carries both: `L6_15m_vr_5_60`.

### Principle 4 — **Layer purity**

No feature crosses layers. L2 primitives are exactly 1-bar. L4 smoothed kinematics are OLS-derived only. Etc. If you want a smoothed version of an L2 primitive, that's a **new, named feature** (e.g., `L2_smoothed_velocity_ma_10`), not a redefinition of velocity.

### Principle 5 — **Primitives of each feature class are MANDATORY (2026-04-23 decision)**

User decision: *"primitives will always be better since they carry the purest signal."*

**Clarified scope (later in the same session)**: "primitive" means **the purest form of each feature class**, not the raw L1 values.

- For velocity: 1-bar Δ is the primitive; smoothed/OLS forms are non-primitive variants
- For dispersion: minimum-window σ is the primitive; rolling-window σ is a variant
- For volume change: 1-bar vol Δ is the primitive; smoothed vol trend is a variant

**Every feature class that has a primitive form MUST include it alongside any smoothed/aggregated variants** in the feature vector.

Rationale: the CNN sees both primitive and smoothed. Pure primitive = current-moment truth. Smoothed = denoised context. Both carry information; the model decides which matters per decision. Hiding the primitive behind a smoother removes the model's ability to learn when raw signal beats context.

Implication: **never drop a primitive even if it seems noisy.** Noise IS information — volatility, shocks, regime shifts show up first in primitives.

### Principle 7 — **Feature vector stops at L2 + approved L3 exceptions (2026-04-23 decision)**

User insight: *"I think we are entering Response Surface territory"* — recognized that most L3 candidates (z-scores, variance ratios, velocity-normalized-by-σ, VWAP offsets) are trivial algebraic compositions of L1/L2 features that a CNN's first layer can learn from labels.

**Default rule**: L3+ interactions are the model's job, not the feature builder's. Hand-crafting L3 features:
1. **Bakes in anchors** — `price_zscore_N` assumes equal-weighted mean is the right baseline, not VWAP or trend-adjusted. We don't know which.
2. **Collapses alternatives** — shipping one composition blocks the CNN from learning a different composition.
3. **Adds feature creep** — each L3 feature multiplies by windows × domains with no signal guarantee.

**Exceptions (user decision 2026-04-23)**: some compositions are too complex / multi-scale / physically-meaningful for a single conv layer to learn reliably on ~300 days of data. Those go in the L3 approved list. Additional user guidance: *"CNN can drop them if needed"* — the model's own weight updates will downweight features that don't carry signal during training, so including something "just in case" has small cost (extra input dim) and bounded risk.

**Approved L3 exceptions** (closed list; additions require explicit justification):
- `L3_{TF}_z_se_{N}` — core band z-score (foundational trading hypothesis)
- `L3_{TF}_z_high_{N}`, `L3_{TF}_z_low_{N}` — band extensions using proper high/low OLS fits (corrected from v1)
- `L3_{TF}_SE_high_{N}`, `L3_{TF}_SE_low_{N}` — wick dispersion for chop detection; genuinely orthogonal to close-dispersion
- `L3_{TF}_hurst_{N}` — chop/trend regime detector; multi-scale R/S analysis
- `L3_{TF}_reversion_prob_{N}` — OU first-passage probability from z_se
- `L3_{TF}_swing_noise_{N}` — max pullback over 30-bar window using highs/lows; used by exit-engine giveback logic

No other L3+ features. Ratios, variance ratios, vol z-scores, VWAP offsets, DMI, wick_ratio, vol_rel, dir_vol, p_center: all model's job.

### Principle 6 — **No feature-space normalization (2026-04-23 decision)**

User decision: *"I don't want to normalize cuz it will distort the rates of change."*

Raw primitives (L1) are passed to the CNN **without** z-score / min-max / rolling normalization.

**Why**:
- Rolling normalization (e.g., `(x − rolling_μ) / rolling_σ`) directly distorts computed velocities — conflates "real Δ" with "recent-volatility context"
- Static global normalization is just a constant rescaling; the CNN can do the same thing internally via batch/layer norm with no information loss

**Consequence**:
- L1 raw values stay non-stationary (MNQ 15k→21k range across 2025)
- Stationarity is provided at L2+ via derivatives (Δ) and statistical moments (σ, var), which are stationary by construction regardless of absolute level
- **Magnitude imbalance between features is handled INSIDE the CNN** (batch norm / layer norm on hidden layers), not by preprocessing the feature vector
- Feature builder stores numbers exactly as computed — no transforms on save

This is a principled commitment: the feature vector is the raw signal; the model is responsible for shaping it internally.

---

## Layer architecture

**Layer number = whether smoothing / rolling window is applied** (decision 2026-04-23, revised).

- **L0** = raw source data (OHLCV not in vector; `L0_time_of_day` IS in vector as timestamp exposure)
- **L1** = window-free operators (1-bar Δ, spatial Δ — instant truth, no smoothing)
- **L2** = rolling-window operators (σ, mean, MA-smoothed derivatives, VWAP — smoothed context)
- **L3** = **exceptions only** — band z-scores, wick dispersions, Hurst, reversion_prob, swing_noise — see Principle 7
- **L4+** = does not exist in the feature vector (compositions are the CNN's job)

This replaces the earlier "layer by category" structure (bar shape, residual space, dispersion) which was arbitrary groupings. The new distinction — **instant vs smoothed, and approved exceptions for foundational signals** — is mechanically unambiguous and aligns with the physical meaning of each feature class.

**Final feature count**: L0 (1: time_of_day) + L1 (36) + L2 (54) + L3 (48) = **185 features total**. The model learns all other higher-order compositions internally.

### **L0 — Source data**

User decision 2026-04-23: *"at the end of the day we are working with rates of change not prices the prices does not matter."*

L0 is the **substrate** from which features are computed. **Market data primitives (OHLCV) are NOT in the feature vector** — they're source data for L1/L2/L3 derivations.

**Timestamp is the one exception** (user decision 2026-04-23): *"time of day context should be exposed to CNN, but as part of the L0 — in other words the timestamp."* Exposed directly as `L0_time_of_day`, which is the timestamp modulo 86400 (seconds per day) normalized to [0, 1]. This is a direct timestamp representation, not a derived-feature category — conceptually the same thing as the Unix timestamp but in a form the CNN can actually use (raw Unix ts is monotonically increasing, useless as a feature).

| Primitive | In feature vector? | Notes |
|---|---|---|
| open, high, low, close | **No** | Absolute price non-stationary; L1/L2/L3 derivations carry the stationary signal |
| volume | **No** (as raw) | L1_vol_velocity_1b, L2_vol_mean_N etc. expose volume dynamics |
| timestamp | **Yes**, as `L0_time_of_day` | `ts % 86400 / 86400` — minimum transform needed because raw Unix ts is monotonically increasing |

**Count**: 1 global feature (`L0_time_of_day`). Not per-TF.

**Rationale**: raw prices and volumes are non-stationary — we rely on L1/L2/L3 derivations for stationary signal. But time-of-day context IS cyclic and stationary naturally (same 0-1 range every day). The CNN needs it to distinguish "NY open regime" from "overnight regime" — without it, a 60-bar window at 3am UTC looks structurally identical to one at 14:30 UTC. Explicitly NOT adding `day_of_week` or `session_phase` buckets — the timestamp's modular form is the minimum sufficient exposure; bucketed variants bake in assumptions we don't need yet.

### **L1 — Window-free operators** (instant, no smoothing)

User decision 2026-04-23: *"L2 should be a derivate that requires a rolling window since it is smoothing the signal"* — implying L1 is reserved for window-free operators (no smoothing applied).

Every L1 feature can be computed from a small fixed number of consecutive bars (1-3), with NO rolling window parameter. These are the "instant truth" features — what's happening right now without any denoising.

| Group | Feature | Formula | Bars used |
|---|---|---|---:|
| Temporal Δ | `L1_{TF}_price_velocity_1b` | close[t] − close[t−1] | 2 |
| | `L1_{TF}_vol_velocity_1b` | volume[t] − volume[t−1] | 2 |
| Temporal ΔΔ | `L1_{TF}_price_accel_1b` | price_velocity_1b[t] − price_velocity_1b[t−1] | 3 |
| | `L1_{TF}_vol_accel_1b` | vol_velocity_1b[t] − vol_velocity_1b[t−1] | 3 |
| Spatial Δ | `L1_{TF}_bar_range` | high[t] − low[t] | 1 |
| | `L1_{TF}_body` | close[t] − open[t] | 1 |

**Count**: 6 per TF × 6 TFs = **36 L1 features**.

*(Note: acceleration belongs at L1 under this numbering because it uses no rolling window — only a fixed 3-bar reach. Earlier draft had it at L3 under operator-order semantics. New semantics supersede.)*

### **L2 — Rolling-window operators** (smoothed, window N)

Every L2 feature has a rolling window parameter N. The window is what smooths the signal. Per Principle 2, default N = N_base from the tiered table (12 for most TFs).

**Design rule (user decision 2026-04-23)**: *"simpler data has higher precedence."* L2 smoothed derivatives use arithmetic moving averages of L1 primitives, NOT OLS regression. This drops explicit `beta_N` / `gamma_N` names and replaces them with direct MA variants of the L1 primitives they smooth.

**Also dropped**: `variance_N` features. `variance = sigma²` carries identical information at higher representational complexity. Sigma wins (linear, primary).

| Group | Feature | Formula |
|---|---|---|
| Smoothed derivatives (MA of L1 primitives) | `L2_{TF}_price_velocity_{N}` | (close[t] − close[t−N]) / N |
| | `L2_{TF}_price_accel_{N}` | (price_velocity_1b[t] − price_velocity_1b[t−N]) / N |
| | `L2_{TF}_vol_velocity_{N}` | (volume[t] − volume[t−N]) / N |
| | `L2_{TF}_vol_accel_{N}` | (vol_velocity_1b[t] − vol_velocity_1b[t−N]) / N |
| Statistical 1st moments (equal-weighted) | `L2_{TF}_price_mean_{N}` | mean(close, N) |
| | `L2_{TF}_price_sigma_{N}` | std(close, N) |
| | `L2_{TF}_vol_mean_{N}` | mean(volume, N) |
| | `L2_{TF}_vol_sigma_{N}` | std(volume, N) |
| Volume-weighted 1st moment (cross-domain) | `L2_{TF}_vwap_{N}` | Σ(close_i × vol_i) / Σ(vol_i) over last N bars |

**Count**: 9 per TF × 6 TFs = **54 L2 features**.

Running total: L0 (substrate, not in vector) + L1 (36) + L2 (54) = **90 market-data features** defined so far.

**Window N**: default = N_base from Principle 2 tiered table (12 bars for most TFs). Single window per feature. Multi-window comparisons (σ_short / σ_long etc.) are NOT stored — they're compositions the CNN learns (Principle 7).

### **L3 — Approved exceptions only**

Per Principle 7, L3+ does not exist in the general feature vector — compositions are the CNN's job. The exceptions are features that are foundational to our trading hypothesis OR too complex for a single conv layer to learn reliably.

Complex supporting machinery (OLS, R/S analysis, OU integration) computes internally for these features only; intermediates (RM_N, SE_N, β_N, γ_N, R/S statistics, OU α/σ parameters) are NOT stored separately — only the final derived feature is exposed.

**Band exceptions** (SE bands — core trading signal):

| Feature | Formula | Role |
|---|---|---|
| `L3_{TF}_z_se_{N}` | (close[t] − OLS_RM_N) / OLS_SE_N | Core band z-score: how far is close from OLS-fitted mean, in SE units |
| `L3_{TF}_z_high_{N}` | (high[t] − OLS_RM_N) / OLS_SE_N | Upper-bar extension in σ units (wick-above-band detection) |
| `L3_{TF}_z_low_{N}` | (low[t] − OLS_RM_N) / OLS_SE_N | Lower-bar extension in σ units |

Where:
- `OLS_RM_N` = OLS fitted value at endpoint of last N bars of closes
- `OLS_SE_N` = standard deviation of OLS residuals over that same window

**Dispersion exceptions** (new with v2 deep-dive — wick dispersion captures chop that close-dispersion can't):

| Feature | Formula | Role |
|---|---|---|
| `L3_{TF}_SE_high_{N}` | OLS residual std on high[] over N bars | Up-wick dispersion — measures stop-hunt / liquidity-probe regimes that leave closes tight |
| `L3_{TF}_SE_low_{N}` | OLS residual std on low[] over N bars | Down-wick dispersion — mirror of SE_high |

**Regime/probability exceptions** (too complex for single-layer CNN to learn):

| Feature | Formula | Role | Window |
|---|---|---|---:|
| `L3_{TF}_hurst_{N}` | Hurst exponent via R/S analysis | Chop vs trend regime: H<0.5 = mean-reverting, H≈0.5 = random walk, H>0.5 = trending | × 8 multiplier (needs N≥100-ish for stable estimate) |
| `L3_{TF}_reversion_prob_{N}` | OU first-passage probability | Probability of returning to mean within horizon; computed via analytical formula using z_se | — (uses z_se) |
| `L3_{TF}_swing_noise_{N}` | max(max_drawdown, max_drawup) / tick over 30-bar window using highs/lows | How much pullback is normal right now in ticks — used by exit-engine giveback logic; genuinely orthogonal to close-only primitives | 30 fixed |

These are kept because:
- **Wick dispersion**: no close-only primitive can reveal stop-hunt regimes. SE_close (from L2 sigma or from OLS) ≠ SE_high / SE_low when wicks diverge from closes
- **Multi-scale / multi-step**: Hurst and swing_noise require nonlinear path-tracking (R/S across scales, running max/min over a window) that single conv layers cannot easily learn
- **Foundation of existing tier logic**: reversion_prob and hurst are empirically tested signals; dropping them loses known-useful info
- User guidance: CNN's weight updates during training will downweight them naturally if they don't carry signal — small cost, bounded risk

**Statistical correction from v1**: z_high and z_low now use their OWN OLS fits (`(high − RM_high) / SE_high`, `(low − RM_low) / SE_low`) instead of sharing the close-fit sigma. v1's semantics were statistically inconsistent; v2 fixes this without changing feature count.

**Count**: 8 per TF × 6 TFs = **48 L3 features**.

Running total: L0 (1 global) + L1 (36) + L2 (54) + L3 (48) = **185 features total**.

---

## Mapping v1 (91D) → v2 (108D)

Most v1 features fall into one of three categories under v2: (a) directly map to L1/L2 (same signal, cleaner name/formula), (b) get RECOMPUTED (v1 had "velocity" that was really a windowed OLS slope; v2 computes both a true 1-bar primitive at L1 AND a simple MA at L2), or (c) get **DROPPED** because they were L3+ compositions (z-scores, ratios, reversion probabilities) that the CNN will learn internally under Principle 7.

| v1 name | v2 equivalent | Status |
|---|---|---|
| `{TF}_bar_range` | `L1_{TF}_bar_range` | RENAME (same formula) |
| `{TF}_velocity` | `L1_{TF}_price_velocity_1b` + `L2_{TF}_price_velocity_N` | RECOMPUTE — v1 was OLS slope; v2 exposes both the 1-bar primitive AND a simple MA |
| `{TF}_acceleration` | `L1_{TF}_price_accel_1b` + `L2_{TF}_price_accel_N` | RECOMPUTE — same pattern as velocity |
| `{TF}_vol_rel` | *DROPPED* | Was close/vol_mean_N — trivial composition (close + L2_vol_mean_N ratio) |
| `{TF}_dir_vol` | *DROPPED* | sign(β)×vol_rel — composition of L2 features |
| `{TF}_z_se`, `{TF}_z_high`, `{TF}_z_low` | `L3_{TF}_z_se_N`, `L3_{TF}_z_high_N`, `L3_{TF}_z_low_N` | KEEP (approved L3 exception — core trading signal) |
| `{TF}_p_at_center` | *DROPPED* | 3-class probability of z-range; composition |
| `{TF}_reversion_prob` | `L3_{TF}_reversion_prob_N` | KEEP (approved L3 exception — OU math too complex for single-layer CNN) |
| `{TF}_variance_ratio` | *DROPPED* | σ²_short/σ²_long; CNN composes from two L2 sigmas |
| `{TF}_hurst` | `L3_{TF}_hurst_N` | KEEP (approved L3 exception — multi-scale R/S statistics not single-layer learnable) |
| `{TF}_wick_ratio` | *DROPPED* | Composition of L1 (bar_range, body) |
| `{TF}_dmi_diff`, `{TF}_dmi_gap` | *DROPPED* | DMI is a derived indicator family; CNN learns directional structure from L1/L2 |
| `time_of_day` | `L0_time_of_day` | RENAME (kept as direct L0 timestamp exposure, same formula) |

**New in v2**: complete L1 + L2 kinematic and dispersion stack for both price AND volume (parallel domains), plus VWAP (cross-domain volume-weighted mean), plus L3 SE-band exceptions, plus the L0 time-of-day exposure. 1 L0 + 36 L1 + 54 L2 + 18 L3 = **109 features total**.

## Feature count

| Layer | Per TF | × 6 TFs | Contents |
|---|---:|---:|---|
| L0 | — | — | `L0_time_of_day` — 1 global feature, not per-TF |
| L1 | 6 | 48 | price_velocity_1b, price_accel_1b, vol_velocity_1b, vol_accel_1b, bar_range, body |
| L2 | 9 | 72 | velocity_N, accel_N, mean_N, sigma_N (price + volume) + vwap_N |
| L3 | 8 | 64 | z_se_N, z_high_N, z_low_N, SE_high_N, SE_low_N, hurst_N, reversion_prob_N, swing_noise_N |
| **Total** | **23 per TF** | **184 + 1 global = 185** | Full feature vector |

TFs (8): 5s, 15s, 1m, 5m, 15m, 1h, 4h, 1D. Only `L0_time_of_day` lives outside the per-TF structure (single global scalar).

---

## Storage layout

**Decoupled by layer-family** (decision 2026-04-23). Each (layer, TF) pair gets its own parquet per day. Motivation:

- Enables rebuilding one layer in isolation (e.g., iterate L3 band computation without re-exporting L1/L2)
- Enables feature-ablation by not loading a file (drop L3 by skipping the L3_{TF} parquets)
- Keeps layer-level versioning clean (one compute generation per layer-family file)
- Windows are stable under the "3× higher TF" rule (Principle 2) — no expected per-feature retuning, so per-feature decoupling would be overkill

**Directory structure:**

```
DATA/ATLAS/FEATURES_5s/
├── L0/
│   └── YYYY_MM_DD.parquet           # [time_of_day] — 1 column per day
├── L1_15s/
│   └── YYYY_MM_DD.parquet           # [6 L1 features at 15s]
├── L1_1m/ ... L1_1D/                # same, other TFs
├── L2_15s/
│   └── YYYY_MM_DD.parquet           # [9 L2 features at 15s]
├── L2_1m/ ... L2_1D/
├── L3_15s/
│   └── YYYY_MM_DD.parquet           # [5 L3 features at 15s]
└── L3_1m/ ... L3_1D/
```

**File count**: 1 (L0) + 6×3 (L1/L2/L3 per TF) = 19 files per day × ~350 days = ~6,600 files. NTFS-manageable.

**Load pattern**: training scripts call a loader that joins all layer-family parquets on timestamp for requested days + TFs + layers. Shared bar alignment (5s-native across all TFs after resampling) makes the join trivial.

**Schema guard**: each layer-family parquet carries a `schema_version` in its metadata. Loader checks that all family files for a given day share the same version — catches drift if one family gets re-computed without others.

## Implementation path

1. **Rewrite SFE** (`core/statistical_field_engine.py`) to compute L0 + L1 + L2 + L3 as four distinct methods, each returning a DataFrame for its layer-family:
   - `L0`: `time_of_day` (ts % 86400 / 86400)
   - `L1` per TF: 1-bar Δ operations (price/vol velocity_1b, accel_1b), spatial Δ (bar_range, body)
   - `L2` per TF: rolling-window MA (velocity_N, accel_N, mean_N, sigma_N for price/vol), VWAP
   - `L3` per TF: OLS-based SE bands (z_se, z_high, z_low) + Hurst + reversion_prob
   - **Remove**: p_center, DMI, variance_ratio, wick_ratio, vol_rel, dir_vol, and all other v1 L3+ derivations
2. **Update builder** (`training/build_dataset.py`) to write per-layer-family parquets
3. **Write loader** (`core/features.py::load_features`) that joins requested (layers × TFs × days) into a single training-ready DataFrame
4. **Regenerate** `DATA/ATLAS/FEATURES_5s/{L0,L1_*,L2_*,L3_*}/*.parquet` from scratch (breaking rebuild — baseline is -$164/day, nothing worth preserving)
5. **Retrain** pivot-direction CNN on v2 features
6. **Compare**: v1-CNN OOS AUC vs v2-CNN OOS AUC

Expected lift: AUC 0.63 → 0.65-0.70 range. The v2 rebuild tests two hypotheses simultaneously:
- Whether the CNN can learn most L3+ compositions internally better than we can pre-craft them (Principle 7)
- Whether clean primitives + VWAP + explicit SE bands + Hurst + reversion_prob + no normalization yields cleaner gradient flow than the mixed-layer v1 did

---

## Open questions

1. ~~**Context layer**~~ — **RESOLVED 2026-04-23**: `L0_time_of_day` is in the vector as direct timestamp exposure. No separate context layer; no `day_of_week`, no `session_phase` bucketing. See L0 section.
2. **CNN capacity check** — with most L3+ compositions left to the model (only SE bands exposed), does the current CNN architecture have enough depth/capacity to learn compositions like `variance_ratio`, `velocity / sigma`, `close − vwap` from L1/L2 inputs? Worth a quick architectural sanity test before committing to the full breaking rebuild.
