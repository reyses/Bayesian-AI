# Waveform Screening Methodology

> Reference doc for the I-MR factor screening process.
> Tool: `tools/waveform_screening.py`
> Created: 2026-02-27

---

## Genesis: How This System Was Originally Built

The entire system grew from one simple idea, applied in three steps:

### 1. I-MR Chart on 15-Minute Price

Started with a single I-MR control chart on 15-minute close prices.
- **I chart**: z_score = (close - center) / sigma, where center is the
  21-bar rolling linear regression and sigma is the residual std dev.
- **MR chart**: bar-to-bar change in z_score (moving range).
- This is the "read the chart" step — where is price relative to fair value?

### 2. Make It Fractal (One Per Timeframe)

Replicated the I-MR chart across ALL timeframes — one control chart per TF.
Each timeframe gets its own z-score, its own center line, its own sigma.
The 15-minute chart sees the mid-frequency picture; the 1-hour sees the
macro trend; the 15-second sees the micro noise.

Stacking these gives the **(depth × 16) hypervolume matrix** — the fractal
fingerprint of a market moment across the entire timescale hierarchy.

### 3. Regression on the Full Matrix → ~70% Explained

Ran regressions on the full hypervolume matrix to identify WHICH dimensions
of the matrix help explain variation in outcome (MFE). Correlated every
feature at every depth against MFE. Found that the fractal I-MR features
could explain approximately **70% of the outcome variation**.

This proved the core thesis: the I-MR control chart, applied fractally
across timeframes, contains enough information to predict trade outcomes.

### 4. Segment by Matching Patterns → Repeat at Pattern Level

Clustered the patterns into groups with similar geometry (I-MR segmentation
on DMI differential + DBSCAN). Then **repeated the regression within each
segment** — not just globally, but per-pattern-group.

This is critical: the global regression says "on average, feature X
matters." The segmented regression says "for THIS specific pattern type,
feature X matters THIS much." Different patterns have different drivers.

### 5. Drill Down → Find Temporal Special Cause

After identifying which features matter per segment, drilled into the
residual variation to find **temporal special causes** — patterns in
time that the features alone didn't capture:
- Which market sessions (Asia, Europe, US RTH) produce better outcomes?
- Which UTC hours are consistently good or bad?
- Does day-of-week matter?

This is the SPC discipline: first explain the common-cause variation
(features), then hunt for special-cause variation (time, regime shifts).

```
15m I-MR chart
     ↓
fractal (one per TF) → hypervolume matrix
     ↓
regression on full matrix → 70% explained
     ↓
segment by pattern match → repeat regression per segment
     ↓
drill for temporal special cause → session/hour filters
     ↓
classify: KEEP / SPLIT / DROP → screening gates
```

---

## Purpose

Determine which of the 16F × 12D hypervolume dimensions actually explain
outcome (MFE/MAE) variation. Cause-and-effect screening BEFORE building
or tuning the regression model. This is the analytical layer that sits
between oracle labeling (raw truth) and the forward pass (live trading).

**Oracle labels the data. Waveform screening finds what matters in it.**

---

## The Thought Process (Step by Step)

### Step 1: Gather the Raw Material

**Question**: "What do we have to work with?"

Load all patterns from trained checkpoints. Each pattern carries:
- A **(depth × 16) hypervolume matrix** — the 16D feature vector at every
  fractal depth level (macro → micro). This is the **I chart** (Individual
  measurements).
- **Oracle labels**: MFE (max favorable excursion), MAE (max adverse
  excursion), marker (±2/±1/0).

Temporal windowing: skip the first N days (warmup, cold-start indicators),
then collect a fixed window of data. Prevents overfitting to warmup noise.

```
templates.pkl → extract patterns → build (depth × 16) matrices
                                 → collect oracle MFE/MAE per pattern
```

**File**: `waveform_screening.py` lines 57-145 (`load_templates`, `extract_matrices`)

---

### Step 2: Pad to Fixed Depth

**Question**: "How do we compare patterns with different cascade depths?"

Patterns have variable depth (0-12). Pad all matrices to (12, 16) with
zeros so they can be stacked into a single numpy array (n, 12, 16).

**File**: `waveform_screening.py` lines 148-157 (`pad_to_fixed_depth`)

---

### Step 3: Build the Moving Range (MR Chart)

**Question**: "How do features CHANGE across depth levels?"

The I chart captures feature values at each depth. The MR chart captures
**transitions** — depth-to-depth differences. This is critical because
it reveals regime changes in the fractal hierarchy.

For each pattern and each of the 16 features:
- **MR values**: `diff(padded, axis=1)` → (n, 11, 16) — 11 transitions per feature
- **UCL flags**: where `|MR| > 3.267 × mean|MR|` (D4 constant for n=2 subgroup).
  UCL violations = out-of-control transitions = regime breaks.
- **Column summaries** (per feature, across 12 depths):
  - `slope`: linear trend across depths (is the feature growing or shrinking?)
  - `mr_bar`: mean |MR| (average movement between depths)
  - `n_breaks`: count of UCL violations (how many regime changes?)

Total MR features: 176 MR + 176 UCL + 48 summaries = **448 columns**.

```
I chart:  feature value at each depth (what IS the state)
MR chart: depth-to-depth change       (how the state TRANSITIONS)
UCL:      flag for "out of control"    (where regime BREAKS)
```

**File**: `waveform_screening.py` lines 160-248 (`compute_moving_range`)

---

### Step 4: Flatten and Combine

**Question**: "How do we get everything into one feature matrix?"

- Flatten I values: (n, 12, 16) → (n, 192) with column names like `d3__self_adx`
- Combine: I (192) + MR (448) = **Z matrix (640 columns)**

```
Z = [I₁₉₂ | MR₄₄₈] = 640 features per pattern
```

**File**: `waveform_screening.py` lines 251-263, 415-426

---

### Step 5: Screen Every Factor Against Outcome

**Question**: "Which features actually predict MFE?"

Correlate each of the 640 columns against oracle MFE. Sort by |correlation|.
This is the cause-and-effect screening — no assumptions about which features
matter. Let the data speak.

Dead columns (zero variance) get correlation = 0.

Output: sorted list of (column_name, correlation, |correlation|).

```
for each of 640 columns:
    corr = pearson(column, MFE)
sort by |corr| descending
→ "d8__self_pid has r=+0.23 with MFE"
→ "MR_d6>d7__coherence has r=-0.18 with MFE"
```

**File**: `waveform_screening.py` lines 266-289 (`screen_factors`)

---

### Step 6: Stepwise Regression (OLS)

**Question**: "How much variance do the top factors explain, together?"

Add the top-K correlated factors one at a time into an OLS regression.
Track adj-R² at each step. This reveals:
- Diminishing returns (when adding more factors stops helping)
- Multicollinearity (if a new factor adds nothing, it's redundant)
- The practical ceiling of linear prediction

```
Step 1: d8__self_pid           R²=0.052  (explains 5.2%)
Step 2: + MR_d6>d7__coherence  R²=0.087  (+3.5%)
Step 3: + d5__self_dmi_diff    R²=0.103  (+1.6%)
...
Step 20: plateau at adj-R² ≈ 0.15
```

**File**: `waveform_screening.py` lines 292-329 (`regression_r2`)

---

### Step 7: Report by Depth and Feature

**Question**: "Which timeframe depths and which features matter most?"

Two cross-tabulations:
1. **By depth**: average |corr| across all features at each depth level.
   Reveals: "d8 (micro) matters more than d0 (macro)" or vice versa.
2. **By feature**: average |corr| across all depths for each feature.
   Reveals: "self_pid matters more than coherence" etc.

**File**: `waveform_screening.py` lines 332-383 (`print_screening_report`)

---

### Step 8: Directional Split

**Question**: "Does direction change the picture?"

Split all patterns by DMI sign:
- **LONG** (DMI_diff ≥ 0): patterns where directional momentum is upward
- **SHORT** (DMI_diff < 0): patterns where directional momentum is downward

Compute win rate (MFE > MAE), mean MFE for each side separately.
This reveals if one direction is systematically better.

**File**: `waveform_screening.py` lines 479-507

---

### Step 9: Segmented Screening (Template × Direction)

**Question**: "Which template + direction combos have real signal?"

For each **(template_id × direction)** segment with ≥15 patterns:

1. **Screen factors within segment** — what predicts MFE for THIS specific
   template in THIS direction?
2. **Top-5 regression** — segment-level adj-R² (can we predict MFE here?)
3. **Process capability**:
   - `Cpk = mean / (3 × std)` — process centered vs own spread
   - `Ppk = mean / (3 × global_std)` — process centered vs overall spread
4. **Probability metrics**:
   - `win_rate` — P(MFE > MAE) — actual profitability
   - `p_positive` — P(MFE > 0) — any favorable movement
   - `SNR = mean / std` — consistency (signal-to-noise ratio)
5. **Good vs Bad split** — split at median MFE, compute Cohen's d effect
   size for each feature. "What distinguishes winners from losers in this
   segment?"
6. **MR entry/exit signals** — MR factors with |corr| > 0.20. Positive
   correlation = entry signal, negative = exit signal.
7. **Dominant context feature** — most common feature in top-5 factors.

```
Segment L_746456 (LONG, template 746456):
  n=402, WR=66.2%, SNR=0.82, R²=0.31
  Dominant: self_pid
  Good vs Bad: d7__self_pid higher in winners (d=+0.45)
  Entry: self_pid @ d6>d7 (r=+0.28)
  Exit: coherence @ d8>d9 (r=-0.22)
```

**File**: `waveform_screening.py` lines 509-627

---

### Step 10: Model Fission (KEEP / SPLIT / DROP)

**Question**: "Which segments should we trade, refine, or discard?"

Classify each segment by win rate + consistency:

| Class | Criteria | Action |
|-------|----------|--------|
| **KEEP** | WR ≥ 65% AND SNR ≥ 0.5 | Trade as-is — strong signal, consistent |
| **SPLIT** | WR ≥ 50% | Signal exists but noisy — needs finer cuts |
| **DROP** | WR < 50% | Net noise — block from trading |

For KEEP segments: report entry/exit signals, good-vs-bad differentiator.
For SPLIT segments: report which feature to split on next.
For DROP segments: just list them (noise generators to remove).

**File**: `waveform_screening.py` lines 646-734

---

### Step 11: Export Gate Config

**Question**: "How do we wire this into the forward pass?"

Save fission results to `checkpoints/screening_gates.json`:
```json
{
  "fission_map": {
    "746456_long": "KEEP",
    "773089_long": "KEEP",
    "924823_long": "SPLIT",
    ...
  },
  "good_hours_utc": [0, 5, 17, 18, 19, 20],
  "default_class": "DROP"
}
```

The orchestrator loads this at forward pass init and applies as Gate 3.5:
- KEEP/SPLIT → pass through
- DROP → block from trading

**File**: `waveform_screening.py` lines 736-760

---

### Step 12: What-If Impact Analysis

**Question**: "How much does fission improve results?"

Compare:
- ALL segments: baseline WR and MFE
- KEEP only: how much does dropping noise improve WR?
- KEEP + SPLIT: intermediate (before refining splits)

```
CURRENT (all): 30 segs, 8,464 patterns, WR: 44.4%
KEEP ONLY:      3 segs,   402 patterns, WR: 66.2%  (+21.8%)
KEEP + SPLIT:  12 segs, 1,875 patterns, WR: 56.5%  (+12.1%)
```

**File**: `waveform_screening.py` lines 762-796

---

### Step 13: PID Drill-Down (I-MR × Direction)

**Question**: "How does the PID control signal behave across the hierarchy?"

Deep analysis of feature[14] (self_pid) specifically:
1. **PID I-chart**: mean value at each depth, LONG vs SHORT, correlation with MFE
2. **PID MR**: depth-to-depth gradients, flag key transitions (|corr| > 0.15)
3. **PID UCL breaks**: % of patterns with control limit violations per transition,
   conditional win rate when breaks occur
4. **PID by fission class**: how PID profiles differ for KEEP vs SPLIT vs DROP
5. **PID × Direction confirmation**: does PID sign agreeing with DMI direction
   improve win rate?

**File**: `waveform_screening.py` lines 804-951

---

### Step 14: Temporal Special Cause Analysis

**Question**: "Does time-of-day, day-of-week, or session matter?"

Analyze patterns by:
- **Market session**: Asia (22-08 UTC), Europe (08-14), US RTH (14-21), US Close (21-22)
- **Hour of day**: WR per UTC hour, find "good hours"
- **Day of week**: Monday-Friday WR comparison

This subsumes manual skip-Europe / skip-session-opens rules with data-driven
temporal gates.

**File**: `waveform_screening.py` lines 953+

---

## Summary: The Flow

```
1. GATHER     templates.pkl → (depth × 16) matrices + oracle MFE/MAE
2. PAD        variable depth → fixed (12, 16)
3. MR         depth-to-depth differences + UCL flags + summaries
4. FLATTEN    I(192) + MR(448) = Z(640) feature matrix
5. SCREEN     correlate each of 640 features vs MFE → rank by |r|
6. REGRESS    stepwise OLS on top-K → adj-R² curve
7. REPORT     importance by depth, by feature
8. SPLIT      LONG vs SHORT by DMI sign
9. SEGMENT    template × direction → per-segment screening + R² + Cpk
10. FISSION   KEEP (WR≥65%, SNR≥0.5) / SPLIT (WR≥50%) / DROP (<50%)
11. EXPORT    screening_gates.json → orchestrator Gate 3.5
12. WHAT-IF   impact analysis: all vs KEEP vs KEEP+SPLIT
13. PID       drill-down on PID control signal across hierarchy
14. TEMPORAL  session, hour, day-of-week special cause analysis
```

## Key SPC Concepts Used

| Concept | What It Is | How We Use It |
|---------|-----------|---------------|
| **I chart** | Individual measurements | Feature values at each depth |
| **MR chart** | Moving Range (consecutive differences) | Depth-to-depth feature transitions |
| **UCL** | Upper Control Limit (D4 × MR_bar) | Flag regime breaks |
| **Cpk** | Process capability (centered) | Segment consistency vs own spread |
| **Ppk** | Process performance (overall) | Segment consistency vs global spread |
| **Special cause** | Assignable variation | Temporal patterns, PID breaks |

## CLI

```bash
# Standard run (30d warmup, 7d collection window)
python tools/waveform_screening.py --warmup 30 --window 7 --top 30

# Full dataset (no windowing)
python tools/waveform_screening.py --all --top 15

# Quick validation (post-retrain)
python tools/waveform_screening.py --warmup 30 --window 0 --top 15
```

Output: console + `checkpoints/screening_gates.json`
