# THESIS — The Continuous F-Space (sliding windows from 1s, not segregated timeframes)
*Captured 2026-06-17. Synthesis of the 2026-06-16/17 design discussion.*

## One-line
**Timeframes are a HUMAN attention/retention crutch, not a property of the market.** There is ONE
continuous price signal. The right *machine* representation reads it with **sliding windows from the
1s base** — LOCATION (where price is) + DISTRIBUTION (structure: touch/dwell/wick) + KINEMATICS
(formation: how this became that) — NOT clock-aligned, context-resetting OHLC bars treated as 8
segregated, ordinal, collinear half-measurements. **A model has no attention limit; it must not
inherit the human's chunking.**

## 1. Why the TF-segregated F-space is wrong (by design, not by mistake)
1. **Human crutch.** A trader can't hold the tick stream in working memory, so they CHUNK it into
   bars/TFs and switch between them — a limited-attention compression. A model has no such limit;
   forcing TF chunking imposes a constraint the machine doesn't have.
2. **A "1h price" is a slice, not segregated data** — the one stream sampled at an arbitrary,
   clock-phase-aligned boundary, summarized lossily (OHLCV). Shift the grid 30min → different bars of
   the *identical* stream → a phase artifact baked into the feature.
3. **Variable-type error.** TF is encoded ORDINAL/categorical (levels 5s..1D) when scale is
   CARDINAL/continuous (window length in seconds; 3600 = 2×1800). Binning a continuous quantity →
   no interpolation (can't find the 2.5h sweet spot), false equal-spacing (1m→5m = 240s vs 1h→4h =
   10,800s), destroyed statistical power, threshold artifacts. **"Use a higher TF" really meant
   "use a bigger sample window."**
4. **Collinearity.** 8 TFs are collinear slices of one signal; composing/regressing them as an
   orthogonal basis (esp. with polynomial interaction expansion) → multicollinearity. Symptoms we
   measured: SMEP betas exploded to |β|≈471, coin-flip beta signs, ~25% IS→OOS cell survival,
   inconclusive DOE/segments. First cut (one day): cond# **3.6e29 (TF-mix) vs 5.1e16 (1s)**;
   beta-stability **0.20 vs 0.96**.
5. **Tiled bars reset context + go stale + are phase-locked.** At each clock boundary the bar discards
   the prior window and starts cold; between closes the feature is stale; the boundary is arbitrary.

## 2. The reframe — one signal, three reads, all from 1s (mirrors how a human reads a chart)
- **LOCATION (where)** — position in the current context = sliding-window z / position-in-distribution.
  Always-current, phase-free.
- **STRUCTURE (what)** — touch vs dwell, wick shape = the **L5 within-window DISTRIBUTION** (density,
  quantiles, skew, outlier_pct). OHLC "high=X" can't tell a touch from acceptance; the distribution can.
- **FORMATION (how this became that)** — the path/sequence that built the structure = **KINEMATICS over
  the window** (velocity, acceleration, inflection = the Kalman/R-curve states). The "lower TF" you drop
  to is just finer resolution INSIDE the window; the ordered 1s stream IS the formation. *(This is why
  the orange/blue/Kalman work is the formation-layer, not a detour.)*
- The distribution is **order-agnostic**; the kinematics / ordered-1s supply the **sequence** it can't.

## 3. The crux + the honest counter
**Sliding window** (continuous context, always-current, phase-free) **vs tiled bar** (resets context,
stale, phase-locked — BUT ~independent samples). Genuine tradeoff: sliding = continuous but
**autocorrelated** (tiny effective-N); tiled = discontinuous but **independent** (a statistical virtue).
So "explains better" MUST be measured **effective-N-honest** — the independent unit is not the second.

## 4. Test program (falsifiable; the REAL stage-1 tier process, not a Ridge proxy)
- **Test 1 — base sufficiency:** does the 1s-ONLY F-space explain the 1s price via the real tier process
  (R-curve forward expansion + error-band + PRISTINE/RECOVERED/CHAOS)? **Circularity control:** some 1s
  features are price-derived (velocity≈price-delta) → must beat a trivial AR(lagged-price) baseline, and
  the explanation must come from structural features (z/hurst/reversion). GATE: if it can't explain its
  own resolution, stop.
- **Test 2 — window-from-1s vs TF-bar at matched horizon:** expand the sliding window to mirror each
  higher-TF horizon (matched in WALL-CLOCK, per-horizon to isolate, common 1s target). Similar / better /
  worse? Sliding(continuous) vs tiled(independent) is the contest; effective-N-honest, OOS.
- **Test 3 — generalization:** full scale-space response surface, **IS vs OOS overlay**; a peak counts
  ONLY if it reproduces across OOS sub-periods (2025H1 / 2025H2 / 2026).
- **Orthogonalization:** **Gram-Schmidt by increasing window** (scale-preserving + orthogonal), NOT
  PCA-across-scales (which destroys the horizon axis the surface needs).

## 5. Standing caveats (necessary-not-sufficient — bias leashed)
- Circularity (price-derived features fit price trivially).
- Autocorrelation / effective-N (sliding windows look artificially rich).
- **Coordinated levels are REAL** (session reopen, daily close — the market itself resets; keep a few
  explicit levels alongside the continuous process).
- The **vol-ceiling and base-rate kills STILL stand**. A cleaner ruler removes the INSTRUMENT confound;
  it does not manufacture signal.
- Path-featurization (keep order without exploding dimensionality) is unsolved engineering.
- **The prize:** settle "was it our collinear/tiled ruler, or no signal?" — only an orthogonal,
  full-resolution, sliding ruler + the OOS overlay answers it. If still inconclusive OOS with a clean
  ruler → no signal; accept it.

## 6. Current evidence + status
- **First cut** (collinearity/stability PROXY, NOT the tier process): TF-mix vastly more collinear +
  unstable than 1s (3.6e29 / 0.20 vs 5.1e16 / 0.96) → **instrument-collinearity thesis CONFIRMED**. BUT
  both R² negative (no signal yet; crude forward-return target, and 1s still collinear at 5e16).
- **Real tier-process tests (Test 1/2) PENDING** — Gemini hit its usage limit before completing.
- Genuine 1s features computed for 2025_02_20 → `DATA/ATLAS/FEATURES_1s_TRUE/`. Scripts:
  `research/segment_1s_test.py` (proxy + 1s feature-gen via SFE `compute_L1/L2/L3` with explicit N).

## 7. Next action
Wire the archived **stage-1 tier pipeline** (`archive/research/Regression segments/`) to the genuine-1s
(sliding, GS-by-window orthogonalized) features and run **Test 1** (self-explanation, circularity-
controlled), then **Test 2** (sliding vs tiled, matched horizon, effective-N-honest, OOS). A Ridge proxy
does NOT answer either test.
