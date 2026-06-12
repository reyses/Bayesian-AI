# NMP → V2 Feature Space — Canonical Mapping

**Purpose:** definitive term-by-term mapping of the Nightmare Protocol (Master
Equation, `docs/archive/legacy/nightmare protocol.pdf`) and its implemented V1
form onto the current V2 185D feature schema (`core_v2/features.py`,
`core_v2/statistical_field_engine.py`). This is the foundation doc for the
λ-completion program: what exists, what's renamed, what's semantically changed,
what's missing, and the reconstruction recipe for every gap.

**Language note:** physics metaphors below appear only as provenance (the PDF's
names). Canonical vocabulary is statistical: regression mean, residual σ,
standardized residual z, stability exponent λ, variance ratio.

**Verified sources:** V2 schema read directly from `core_v2/features.py`
(FEATURE_NAMES_V2, N_BASE) and `core_v2/statistical_field_engine.py` (kernels,
constants). V1/NMP implementation recovered from
`docs/reference/legacy_tiers/nightmare_base_2026_04_18.py`,
`docs/reference/nightmare_blended_2026_05_20.py`, and git
(`9aede781~1:core/statistical_field_engine.py`, `22c61c02:core/quantum_field_engine.py`).

---

## 0. V2 ground truth (for reference)

- 8 TFs × 25 features + L0_time_of_day = 201 columns; the 185D figure refers to
  the active subset after price_mean/vwap exclusions in some consumers.
- Per-TF: L1 (8 bar primitives) · L2 (9 rolling stats, window `N_BASE[tf]`) ·
  L3 (8 approved statistical exceptions).
- `N_BASE`: 5s:9 · 15s:12 · 1m:15 · 5m:9 · 15m:12 · 1h:12 · 4h:18 · 1D:5.
- Constants: `N_HURST_MULT=8` (Hurst window = N_BASE×8), `SWING_NOISE_WINDOW=30`,
  `OU_BOUNDARY=3.0`. z_se uses OLS residual std **ddof=2**; price_sigma_w is
  rolling std **ddof=1**.
- Causality: V2 build uses `_last_closed_idx` (load-bearing) — at a 5s anchor,
  each TF's features come from that TF's **last closed bar**.

---

## 1. Master-Equation terms → V2

| # | PDF term (provenance) | Math | V1 as actually shipped | V2 feature | Status | Reconstruction (if missing) |
|---|---|---|---|---|---|---|
| 1 | μ(t) "center of mass" | trailing OLS mean of close | 21-bar OLS endpoint (`rp=21`), per TF | not materialized; embedded inside `z_se`; nearest stored = `L2_{tf}_price_mean_{N}` (rolling mean ≠ OLS endpoint) | **IMPLICIT** | decision logic needs z, not μ; if μ itself needed, recompute 21-bar OLS from raw closes at decision layer |
| 2 | θ reversion speed | OU pull coefficient | **never implemented** (metaphor only) | `reversion_prob` is the analytic stand-in | NEVER EXISTED | OU fit on residual series if ever wanted; not required by any trigger |
| 3 | σ(v,τ) fractal vol | σ_base·(v_micro/v_macro)^H | **dropped before NMP existed** (only in Feb-2026 QFE, git `22c61c02`); NMP used raw 21-bar residual σ | residual σ embedded in `z_se` denom; `SE_high`/`SE_low` (own-fit residual stds); `price_sigma_w` (plain rolling) | PARTIAL | fractal scaling has no counterpart; `hurst` survives standalone. Do not resurrect without evidence |
| 4 | **Z_fit** | (X−μ)/σ | `z` = (close−center)/σ_resid, 21-bar fit | **`L3_{tf}_z_se_{N}`** — same formula, ddof=2 | **EXISTS** ⚠ window changed (21 → N_BASE; 1m: 15) | — |
| 5 | **λ stability exponent** | |δZ(t)| ≈ e^{λt}; λ<0 stable / λ>0 chaotic | **NEVER COMPUTED.** Feb-2026 had a 1-bar Δ\|z\| proxy; V1 SFE hardcoded `lyapunov_exponent=0.0`; no trigger ever consumed it | **MISSING** — closest semantic relatives: `hurst`, `reversion_prob` | THE GAP | **λ̂ = trailing OLS slope of log(\|z_se\|+ε) over k bars** (k≈12–30, calibrate). Derivable from the stored `z_se` series — **no feature-store rebuild needed**. Segment corpus = the empirical/labeled version |
| 6 | (λ's operational stand-in) **variance_ratio** | std(close last 10)/std(close last 60), per TF | `vr` — THE actual stability gate in the NMP entry (`vr<1` = contracting = "stable") | **MISSING — dropped from V2 entirely** | THE GAP | (a) exact: recompute std10/std60 from raw closes at decision layer; (b) stored-feature proxy: cross-TF σ ratio, e.g. `L2_1m_price_sigma_15 / L2_15m_price_sigma_12` (unit-free; closer to a true Lo-MacKinlay VR than V1's same-TF ratio); (c) materialize as L3 column at next rebuild |
| 7 | F_PID (Kp, Ki, Kd) | P∝e, I∝∫e, D∝de/dt | **never implemented** (descriptive) | P ≅ `z_se`; I ≅ rolling sum of z (derivable); D ≅ Δz_se per bar (derivable) | NEVER EXISTED | derive as diagnostics only if a model wants them |
| 8 | J(λ) jumps | jump diffusion | never implemented | nearest: `bar_range` / `swing_noise` extremes | NEVER EXISTED | — |
| 9 | Bands μ±kσ (k=2 act, k=3 boundary) | Roche/event-horizon | `±2σ`/`±3σ` stored in MarketState; triggers consumed z directly | implicit: z thresholds ≡ band distance in σ units; `z_high`/`z_low` + `SE_high`/`SE_low` for high/low excursions | **EXISTS (implicit)** | — |
| 10 | Hurst H | R/S exponent | 30-bar R/S, default 0.5 warmup | `L3_{tf}_hurst_{N}` — same R/S kernel, window **N_BASE×8** (~100 samples) | **EXISTS** ⚠ window changed | — |
| 11 | (derived) reversion_prob | OU first passage P(hit μ before ±3σ) = 1−erfi(\|z\|/√2)/erfi(3/√2) | identical analytic form | `L3_{tf}_reversion_prob_{N}` — identical, `OU_BOUNDARY=3.0` | **EXISTS** | — |

## 2. Secondary V1 state (tier-ladder inputs) → V2

| V1 variable | V1 definition | V2 | Status / recipe |
|---|---|---|---|
| `velocity` | 1-bar close delta per TF | `price_velocity_1b` (identical); `price_velocity_w` (proper windowed secant) | **EXISTS** |
| `acceleration` | Δvelocity | `price_accel_1b` / `price_accel_w` | **EXISTS** |
| `p_at_center` | softmax over E0=−z²/2, E1=−(z−2)²/2, E2=−(z+2)²/2 → p0 | none stored | **DERIVE** deterministically from `z_se` (exact formula above) |
| `vol_rel` | vol[-1]/mean(vol[-30:]) | none stored (only `vol_mean_w`, `vol_sigma_w`) | **DERIVE**: current bar volume / `vol_mean_w` (window差 caveat: N_BASE vs 30), or recompute from raw bars |
| `wick_ratio` | 1−\|close−open\|/range | replaced by one-sided `upper_wick`/`lower_wick` + `body` | **DERIVE**: `1 − |body|/bar_range` |
| `bar_range` | (high−low)/**tick** | `bar_range` in **points** | **EXISTS ⚠ UNITS**: V1 thresholds ÷4 to convert (tick=0.25) |
| `dmi_diff`/`dmi_gap`/`dir_vol` | DI+−DI− Wilder-14 + helpers | **dropped** | recompute from raw OHLC at decision layer if a tier needs it (only MTF_BREAKOUT did) |
| `swing_noise` | max(drawdown,drawup)/tick over 30 bars | `swing_noise` — same kernel | **EXISTS** ⚠ warmup: V1 default 35.0, V2 NaN |
| `z_high`/`z_low` | (high−center)/σ, (low−center)/σ — **shared the close fit** | `z_high`/`z_low` — **own OLS fits** (rm_high/se_high, rm_low/se_low) | **EXISTS ⚠ SEMANTIC CHANGE** — V2 is statistically corrected; values differ from V1 |

## 3. Trigger translation

**BASE NMP as implemented** (decisions at 1m closes):
```
V1: |z_1m| > 2.0  AND  vr_1m < 1.0          → fade z   (one position)
V2: |L3_1m_z_se_15| > Z*  AND  vr̂_1m < VR*  → fade z
exits: |z_1m| < 0.5 ("mean_reached") · vr_1m > 1.0 ("regime_flip") · EOD
```
⚠ **Thresholds do NOT transfer.** V1's z was a 21-bar fit; V2's 1m z_se is a
15-bar fit — different residual distribution ⇒ `Z*=2.0`, `VR*=1.0` must be
**recalibrated on V2 features**, not copied. Same for the 0.5 exit.

**λ-complete trigger (the program's goal — the never-built branch):**
```
|z| > Z*  AND  λ̂ < 0           → fade   (stable: deviation decaying)
            λ̂ > 0 (persistent) → ride   (the missing branch)
            λ̂ uncertain / regime hazard high → stand aside
```

**Timing semantics:** V1 base NMP evaluated only at 1m boundaries
(`ts % 60 < 5`). V2 anchors at 5s with last-closed-bar features. To replicate
NMP timing exactly, act on the 5s anchor immediately following a 1m close;
acting every 5s with stale-up-to-55s 1m features is a (defensible) behavioral
change — flag it in any comparison.

## 4. Gaps — build decision matrix

| Missing term | (A) Derive at decision layer (now) | (B) Stored-feature proxy (ML on existing parquets) | (C) Materialize at next feature-store rebuild |
|---|---|---|---|
| **λ̂** | OLS slope of log\|z_se\| over k bars — from stored z_se series | same as (A) — purely derived | optional `L3_{tf}_lambda_{k}` |
| **vr** | exact std10/std60 from raw closes | cross-TF σ ratio (`price_sigma_w` fast/slow) | `L3_{tf}_vr_10_60` |
| p_at_center | exact transform of z_se | same | unnecessary (deterministic) |
| vol_rel | raw-bar recompute | vol / `vol_mean_w` approx | optional |
| dmi family | Wilder-14 recompute | none | only if a ride-tier needs it |

**Recommendation:** (A)+(B) now — both λ̂ and vr are available **today** without
touching the schema. (C) is batched for a future deliberate rebuild only;
**do not change the V2 schema while the VM year-run is mid-flight** on the
current 201-column layout.

## 5. Standing traps (carry into every downstream doc)

1. **`z_se` is a residual-σ z, not a standard-error z** — the name lies (V1
   comment acknowledged it; V2 kept name + semantics).
2. **V2 `z_high`/`z_low` ≠ V1's** (own fits vs shared close fit) — don't port V1
   thresholds onto them.
3. **Window drift:** regression 21→N_BASE (1m:15), Hurst 30→N_BASE×8,
   vol_rel 30→N_BASE. Every V1-calibrated threshold needs re-fitting.
4. **Units:** V1 bar_range/swing thresholds are in ticks; V2 bar_range is points.
5. **Legacy NMP backtest numbers are tainted** (TF-alignment lookahead in the
   offline V1 feature files; baseline-740 invalidated, commit `0c001c1f`). The
   V1 **engines** were causal; the feature files weren't. Use shapes, never numbers.
6. λ was hardcoded 0.0 in the NMP era and `variance_ratio` was its de-facto
   replacement; vr has **no V2 counterpart** — any "NMP on V2" claim that doesn't
   reconstruct vr (or λ̂) is not running the NMP trigger.
7. **Hurst verified causal in V2** (kernel uses strictly trailing
   `prices[i−window+1 .. i]`, sub-scales end at i, `_last_closed_idx` alignment,
   no warmup backfill) — but it is a **LAGGING descriptor**: window = N_BASE×8
   (1m → 2 hours). It describes the past regime; in-sample it impersonates a
   predictor because regimes are autocorrelated; at regime turns it is late by
   construction. Use as slow prior, never as the regime gate. Also: class doc
   promises hurst warmup → NaN but `compute_l3` applies **no mask** — the kernel
   emits 0.5 + partial-window estimates from bar 8 on (tiny-sample noise that
   NaN-droppers won't remove). Fix at next rebuild only (value-changing). And
   verify the per-day builder feeds enough history for 1h/4h/1D hurst to ever
   exit partial-window territory.
8. **SEGMENT FIREWALL (mandatory).** Segment-regression quantities (membership,
   tier, betas, boundaries, remaining life) leak the future via three channels:
   boundary placement (membership ⇒ survival to b), full-segment fit estimation,
   and existence-conditioning (segments exist only where fits succeeded). They
   are therefore guaranteed-significant and non-deployable as inputs.
   RULES: (1) segment quantities appear ONLY on the label side (Y) of any model
   — never as features; (2) value is established as oracle-skill (hindsight
   ceiling) vs nowcast-skill (causal recovery) — the ratio is the deployable
   fraction; (3) evaluation is boundary-conditional with lead-time > 0, not
   all-bars-averaged; (4) must beat constant-hazard / age-only / carry-forward
   baselines; (5) final gate = action-level risk-adjusted delta (day-block CI,
   net of cost), not feature significance. Kill-points: if the ORACLE doesn't
   pay, stop before building any nowcaster; if the nowcast recovers ~nothing,
   λ is unobservable to us and sizing/participation is the rational fallback.

---
*Created 2026-06-11 (session: NMP drawdown → λ-completion program). Companion
docs: `SEGMENT_OPTIMIZATION.md` (segment corpus = empirical λ ground truth),
`PLAN_regime_cloud_phases.md`.*
