# ARCH LOCK — Macro Memory & Causal Data Architecture (Mamba RL)

**Scope:** This document locks the **system architecture** — the substrate that feeds learning. It does **not** specify **policy mechanics** (reward shaping, inaction cure, regret). Those are deferred to a second portion. Rationale: the architecture defines *what the model sees and when*; the policy defines *what it's paid to do*. They are separable, and everything below is policy-agnostic.

**Project spine:** lookahead leakage is the recurring kill-shot. Every decision here is subordinate to causal correctness.

---

## 1. Killed: VRAM-Watchdog Dynamic TBPTT
The original plan (accumulate gradient tape → trim at ~90% VRAM → detach) is **rejected**.
- `detach()` severs cross-window gradient → the multi-week horizon is **never trained** anyway. The headline benefit was unfunded.
- VRAM-triggered truncation = **nondeterministic** TBPTT length → kills reproducibility.
- `.backward()` spikes VRAM *during* the trim → OOM risk at the worst moment.
- ~1000× fewer optimizer steps; averaged-gradient clip destroys magnitude.

**Replacement principle:** model the macro **at macro resolution**, as explicit input — do not reconstruct it through recurrence.

## 2. Two-Tier Memory (fast / slow)
- **Short-term (fast):** live 30×5s intraday window + optional carried **detached** recurrent state. Volatile, high-res, resets each session.
- **Long-term (slow):** persisted macro-structure **bank** — levels + regime templates, each with metadata: `as-of timestamp, age, touch_count, strength, distance-to-price`. Persisted to **parquet** (extends existing warmup work). Read at inference; survives restarts; crosses session boundaries.

The real win the watchdog was groping toward is **durable macro memory that outlives the process**, not VRAM relief.

## 3. Bank Is a Feature Input — Never Trained Through
- Gradient flows through the **macro sub-encoder + fusion layer** (learns to *read* the bank). Bank **contents are detached constants** at every step.
- Bank is derived from **market structure only** (exogenous) — not the agent's trades → no policy-dependent feedback loop. Same object in train and live.

## 4. Consolidation Trigger = Structural Events, NOT VRAM
What gets written to long-term memory is decided by **market salience** (4H/1D closes, confirmed pivots), not GPU occupancy. VRAM-aware logic is legitimate **only** for training throughput (sequence packing) — a separate, deferred concern.

---

## 5. Leakage Invariants (load-bearing — the spine)
1. **One builder, two modes.** Single function builds the bank: replay/batch (training) + incremental (live). **Assert byte-identical** on the same window → this *is* the train/serve-parity guard.
2. **Point-in-time reads.** At bar `t`, bank holds only structure confirmable **strictly before t**. Every entry carries an as-of timestamp.
3. **Confirmation-bar entry.** A pivot/level enters the bank at its **confirmation** bar (`as-of = confirmation`), **lagged by the detection forward-window** — never at the pivot bar. (Live, this lag is unavoidable anyway.)
4. **Epoch reset.** Bank is **reset and reconstructed each epoch**. **Assert:** bank state at bar `t` is identical across epochs. Any difference = leakage alarm.
5. **Backward read across all boundaries (incl. OOS).** Reading prior-zone bars to *populate* the bank is **not training on them**. Blanking the bank at a boundary = cold-start artifact that *underestimates* live. Read-back is causal by construction → safe and required.

---

## 6. Data Zones (temporal, gradient-disjoint, ordered)
Pure-Gymnasium-first ⇒ **three zones** (no Grip-B pretrain, no cubic labels, no Junction-1):

```
WARM-UP (bank fills, 0 grad)  →  RL TRAIN (2024–2025)  →  OOS (2026, held last)
```

- **Warm-up:** ~6mo, zero gradient, bank populates causally. Cheap (macro resolution: ~180 daily / ~1k 4H bars, not 140k 5s bars).
- **Cold-start is causally clean** — a thin bank is just less past data, strictly past-only. Cost is representation quality only, fixed for free by **depth-gating** any future pretrain loss.

### Boundary buffer — one rule, every junction
Buffer width = **producing-side label/reward horizon**. Buffer bars are **read-but-never-graded** (no gradient, no score) — **bank still reads through**.
- **RL → OOS:** clip OOS *scoring* start by the **RL reward horizon** (≤ 1 session, given intraday flat-at-close).
- **Calendar note (June 2026):** trailing OOS clip is **risk-free** — H1 data runs past the buffer, leaving ~Mar–Jun as scored OOS. (Sitting in January this clip would leave ~zero OOS.)

## 7. Session Model — Intraday, 22:00→22:00 UTC, Flat at Close
- **Episode = one session.** RL reward horizon ≤ one session (hard cap 24h).
- **Cross-day RL gradient is dead by construction** → confirms detach-across-days.
- **Bank crosses 22:00** (carries confirmed levels into next session); intraday state + gradient reset each session. Causal rebuild runs at **session granularity**.

---

## 8. DEFERRED — Policy Portion (second)
Not decided here; carry forward:
- Reward design (asymmetric per-step mark-to-market + underwater carry).
- Inaction cure (opportunity-cost on flat-in-regime, entropy/exploration bonus).
- **MFE-as-regret:** offline **diagnostic** (forward MFE OK) vs **training signal** (must be causal, realized-only). The reward is the *new* leak surface once labels are gone.
- Whether to add Grip-B **auxiliary predictive pretraining** / build label data at all (parked as Phase-4 lever). If pursued, the **cubic plateau provenance** question must be resolved first: does the plateau-detection fit window **straddle the inflection** (leaked at birth) or strictly **trail t-1** (clean)? "Not refitted" proves *stability*, not *causal construction*.

## 9. OPEN NUMBERS — needed to finalize sizing
1. **RL reward horizon** exact width (full-session vs shorter MFE/episode window) → sizes the RL→OOS buffer precisely.
2. **Macro sub-encoder spec:** tiny-Mamba vs MLP; input schema (which levels/features, downsample cadence, lookback depth); fusion point in `MambaRLTradingNetwork.forward`.

---
*Status: architecture locked. Policy mechanics + the two open numbers move to the next portion.*
