# ONSET MAMBA — production training spec v1

**Owner go, 2026-08-04.** The student's job is **EYES, not judgment**: see a
named event forming before it confirms, so the fast lane can pre-stage an
order. It never predicts direction, price, or table outcomes — those reduce
to a closed-form barrier ratio (`docs/daily/2026-08-04.md` §3), and a network
that learns `d_down/(d_up+d_down)` is an expensive ruler.

---

## 1. Target

For each RTH second `t` and each event type `k`, predict

    y_k(t) = 1  if an event of type k CONFIRMS in (t, t+H]

`H = 10s` primary; `H ∈ {5, 30}` as auxiliary heads (multi-horizon training
regularises and gives the fast lane a sense of urgency).

Event types (v1 — three that survived the ablation):
| head | event | measured GBM AUC @H=10s |
|---|---|---|
| 0 | `fakeout_poke` | 0.769 |
| 1 | `leg_descent` | 0.868 |
| 2 | `ultra_chop` | 0.830 |

`stall` (0.659) and `defended_poke_shelf` (0.627, and 97% regime) are
EXCLUDED from v1. Adding them later requires re-passing the ablation.

## 2. Labels

`research/event_library/events/*.parquet` — confirmation timestamps, already
causally stamped and truncation-audited (`event_library/tools/causality_audit.py`,
0 mismatches). 273k events total; 539 trading days. **Day `2024_09_16` is
excluded everywhere** (live sim day, hindsight-contaminated).

## 3. Inputs

Sequence of **1s bars**, window **300s** ending at `t` (300 steps).
Per step, causal only:
`open, high, low, close, volume` normalised as
- returns: `close_i − close_{i-1}` in ticks
- bar shape: `(high−max(o,c))`, `(min(o,c)−low)`, `(close−open)` in ticks
- `log1p(volume)` z-scored per day on a TRAILING window only
- clock: `sin/cos` of seconds-since-09:30

Price is never fed as a level — only differences. Rationale: MNQ ran
16k→28k across the corpus; absolute price is a date stamp and the model will
memorise the era (this bug class already bit the flushV detector).

## 4. Architecture

Mamba (SSM), 4 layers, d_model 128, d_state 16, ~1.2M params. 3 event heads
× 3 horizons = 9 logits. BCE with per-head `pos_weight` from the training
base rate. Reuse the compiled inference path from
`research/mamba_zigzag_baseline/` (measured 431 bars/s compiled).

## 5. Split — TEMPORAL, not random

| split | days |
|---|---|
| train | 2024-01-02 → 2024-12-31 |
| val | 2025-01-01 → 2025-06-30 |
| test | 2025-07-01 → 2026-03-19 (**touched once, at the end**) |

Day-blocked CV was enough to measure the probe; it is NOT enough to ship —
it cannot see regime drift. The test window is opened once.

## 6. Training distribution vs evaluation distribution

**Train on the NATURAL distribution**: every RTH second is a sample, class
imbalance handled by `pos_weight`. **Evaluate on the MATCHED design**
(`builders/build_onset_matched.py`): negatives are the same event rewound
300s on the same day. Tonight's lesson — quiet-stretch negatives inflated
AUC to 0.9965 by letting the model answer "is the tape active?".

## 7. Pre-registered success criteria (set BEFORE training)

The baseline to beat is not zero, it is the **HistGradientBoosting model we
already have**: fakeout 0.769 / descent 0.868 / chop 0.830 (matched design,
day-blocked CV).

| outcome | rule | action |
|---|---|---|
| SHIP | test AUC ≥ baseline + 0.02 on ≥2 of 3 heads, and ≥ baseline − 0.01 on the third | integrate into the fast lane |
| KEEP GBM | within ±0.02 of baseline | **use the GBM** — it is cheaper, interpretable, already built |
| KILL | below baseline − 0.02 on any head | do not deploy; write the null up |

A sequence model that cannot beat 22 hand-made features by 2 points has not
earned its latency budget or its complexity.

## 8. Latency budget

Fast lane ≤200ms (owner-measured; order round-trip ~800ms, so protective
levels must rest at the broker — see `two-lane-latency-architecture` memory).
Inference must land **≤50ms p99** on one 300-step window, batch 1, GPU.
Exceeding it fails the spec regardless of AUC.

## 9. Kill conditions during training

- Val AUC on the **matched** design diverges upward from train while the
  natural-distribution loss stalls → label leakage, stop and audit.
- Any head where shuffling the labels within a day does NOT collapse AUC to
  ~0.5 → the model is reading a day-level artifact.
- Test-window AUC drops >0.05 below val → regime overfit; ship the GBM.

## 10. What this model is NOT allowed to do

It emits **onset probability per event type**. It does not size, does not
choose direction, does not set stops. Those come from the owner's protocol
and the Bayesian tables (`research/bayes_tables/`), which are a RISK
instrument. Any future head predicting direction or a table outcome must
first pass the driftless-barrier control in
`research/bayes_tables/reports/tables_v0.md`.

## 11. Build order

1. `builders/build_onset_sequences.py` — windows + multi-head labels to
   a memmapped array (604 days × ~23k RTH seconds; store as int16 ticks).
2. `pipeline/train_onset_mamba.py` — training loop, val on matched design.
3. `tools/eval_onset.py` — the pre-registered table above, one shot on test.
4. Adversarial audit BEFORE the test window is opened.

---

## BUILD STATUS (2026-08-04 overnight)

| step | state |
|---|---|
| 1. sequence dataset | **DONE** — `builders/build_onset_sequences.py`, 545 days, **11,175,332 sampleable seconds**, 111 MB (per-day arrays; windows sliced in the loader — materialising them would have cost ~68 GB) |
| 2. training script | **WRITTEN, NOT RUN** — `pipeline/train_onset_mamba.py`. Project rule: the assistant does not launch training. |
| 3. eval on sealed test | pending step 2 |
| 4. adversarial audit | pending step 2 |

Verified without training: model constructs at **0.47M params**, forward pass
correct, **inference 1.17 ms at batch 1 on the RTX 3060** — 43x inside the
50 ms budget, so latency is not the constraint. Temporal split function
spot-checked (2024→train, 2025-H1→val, later→test).

**To run:**
```
python research/event_onset/pipeline/train_onset_mamba.py --epochs 3
```
`--stride 5` by default: adjacent 300s windows share 299 bars, so stride 1
is ~300x redundant compute for almost no extra information.

Each epoch prints matched-design AUC per head against the pre-registered
baseline (fakeout 0.769 / leg_descent 0.868 / ultra_chop 0.830).
