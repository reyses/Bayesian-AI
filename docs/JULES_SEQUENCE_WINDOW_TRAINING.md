# Sequence-Window Training for Mamba RL (two-pass TBPTT)

**Date**: 2026-07-06 · **Status**: approved by user ("let's try the sequence improvement")
**Goal**: replace the per-bar differentiable loop (65–90 bars/s) with an
act-eager / learn-parallel window trainer targeting ~300+ bars/s (<12h for
200 epochs × 5 days).

## Why this is safe (and what actually changes)

Today's trainer already acts with frozen weights inside each 500-bar TBPTT
window (updates happen only at boundaries). The two-pass design keeps that
structure exactly:

1. **Acting pass** (per window, bar-by-bar, `torch.no_grad()`): sample
   actions from the current policy, step the env, record per-bar packed
   observations (686 floats), actions, rewards, `is_flat`, `turn_imminent`,
   session-reset points, and the final bootstrap value V(s_W).
2. **Learning pass** (once per window): a single differentiable
   sequence forward over the recorded [W, 686] inputs reproduces logits /
   values for all bars; vectorized A2C losses (log_prob of the TAKEN action,
   entropy, TD(0) critic with in-window shifted values, BCE hazard aux);
   mean → backward → clip → optimizer step. Hidden states carry to the next
   window detached (same as today).

On-policy: yes — actions are sampled from the same θ the window's gradients
update, identical to current TBPTT. NOT the off-policy "predict a window
without env interaction" variant.

**Numerics change (accepted)**: the parallel associative scan reorders float
ops → same drift class as `--compile` (~1e-3 bf16). Bitwise parity vs the
per-bar trainer is impossible by construction; gates below.

**Semantics change (forced, deliberate)**: restores the conv1d 3-bar
receptive field that the L=1 refactor silently dropped. The sequence conv is
the true causal conv; the acting pass must match, so PureMambaBlock gains a
carried conv state. This is the "conv-state decision" resolved as "fix it".
Old checkpoints trained without conv memory are not comparable.

## Components

- `pipeline/mamba_rl_network.py`
  - `PureMambaBlock.step(x_t, h, conv_state)` — L=1 path with carried
    (d_conv−1)-bar conv state ring. Returns (y_t, h, conv_state).
  - `PureMambaBlock.forward_sequence(x, h0, conv_state0)` — differentiable
    window forward: causal conv over [conv_state0 ‖ x] (slice off context),
    silu, x_proj/dt, **associative SSM scan with initial state h0**
    (log-depth, pure torch — mamba-ssm's Mamba1 kernel lacks an
    initial-state arg, and its step kernels are inference-only), gate,
    out_proj. Returns (y_seq, hW, conv_stateW).
  - `MambaRLTradingNetwork.forward_sequence(...)` — full-trunk window
    forward (macro encoder, norm, embedding, blocks, heads) for [1, W, ·].
  - Back-compat: existing `forward` keeps working for eval tools; the L=1
    path now threads conv_state through `hidden_states` (per-layer tuple
    grows to (h, conv_state)); existing boundary-detach code already
    handles tuples.
- `pipeline/mamba_scan.py` — `associative_ssm_scan(A, B, h0)` solving
  h_t = A_t·h_{t−1} + B_t via Blelloch-style log-depth composition.
  Differentiable, exact recurrence (modulo float reassociation).
- `pipeline/train_mamba_rl_seq.py` — NEW trainer (the current
  `train_mamba_rl.py` stays untouched as fallback/reference). Same CLI +
  instrumentation flags. Window = `--tbptt_window` (500). Session resets
  split the scan into chunks with zeroed initial state mid-window (mirrors
  today's `hidden_states = None` mid-window without forcing an update).
  Boundary bootstrap V(s_W) recorded in the acting pass with pre-update θ
  (matches today). Entropy/reward bookkeeping preserved.

## Loss equivalence map (per bar t in window, vs current trainer)

| term | current | sequence |
|---|---|---|
| policy logits | per-bar fwd, θ_k | seq fwd, θ_k (same inputs/hidden) |
| log_prob | of sampled action | of RECORDED action (same estimator) |
| next_value | V(s_{t+1}) same θ (dedup'd by deferral) | shifted in-window values; last bar: recorded V(s_W) |
| entropy, BCE(pos_weight 10.40, w_aux 0.20) | per-bar | vectorized, identical formula |
| window reduction | mean of per-bar losses | mean over W (identical) |
| update | per 500 bars, clip 1.0, Adam 1e-4 | identical |

## Acceptance gates (ordered)

1. **Unit — scan exactness**: associative scan vs python loop recurrence,
   fp32 random tensors: max|Δh| < 1e-5. (Bit-exactness not expected —
   reassociation.)
2. **Unit — forward equivalence**: `forward_sequence` over W random bars ≡
   W chained `step` calls (same states), fp32: max|Δlogits| < 1e-4;
   bf16 autocast: < 3e-2 (document actual).
3. **Training-track check**: fixed seed, 1 day, per-bar trainer (conv-state
   build) vs seq trainer: same actions until first numeric flip; report
   flip step + loss deltas over the identical-action prefix (compile-gate
   style; bitwise NOT expected).
4. **Speed**: bars/s on quiet box (nvidia-smi checked), interleaved A/B vs
   current trainer. Target ≥ 300 bars/s eager.
5. Reward curves over ≥ 3 epochs qualitatively consistent (no divergence /
   NaN; entropy decay similar).

## Rollback

`train_mamba_rl.py` (deferred bootstrap, 65.8 bars/s eager / 90.5 compiled)
remains the working trainer. The seq trainer is additive; deleting it and
the two network additions restores status quo. Conv-state addition to the
step path is OFF for the old trainer unless it opts in (state threading is
backward compatible: `hidden_states=None` init unchanged).

## Phases

1. Scan module + unit test (gate 1). ✅ when committed
2. PureMambaBlock step/forward_sequence + network forward_sequence + unit
   test (gate 2).
3. Seq trainer (gate 3 + 4).
4. Report + journal + A/B artifacts; user decides adoption for long runs.
