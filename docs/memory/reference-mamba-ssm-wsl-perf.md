---
name: reference-mamba-ssm-wsl-perf
description: "Mamba RL runtime facts — mamba-ssm step() autograd trap, WSL wheel matrix, CUPTI dead, cudagraph-vs-carried-state crash, measured baselines (2026-07-05)"
metadata: 
  node_type: memory
  type: reference
  originSessionId: 580dfdb1-f5da-48a4-b423-63d89c91bbd5
---

Measured 2026-07-05 on RTX 3060 / WSL2 / torch 2.6.0+cu124 (report:
`research/mamba_zigzag_baseline/reports/perf/PERF_REVIEW_2026-07-05.md`):

- **mamba-ssm `Mamba.step()` is an autograd TRAP**: backward runs but input
  grads are silently None (`selective_state_update` has no grad edge). Never
  put it in a training forward. The fused PARALLEL scan IS differentiable and
  ~250× the python scan at L=500 (0.454 vs 113.8 ms/block).
- **WSL wheel matrix**: mamba_ssm 2.2.4 + causal_conv1d 1.5.0.post8, tag
  `+cu12torch2.6cxx11abiFALSE-cp312` (torch pip wheels are cxx11abi FALSE).
  causal-conv1d 1.6.x is ABI-broken vs torch 2.6. mamba_ssm 2.2.4 import
  breaks vs transformers 5.x (GreedySearchDecoderOnlyOutput removed) — stub
  it; don't downgrade transformers.
- **CUPTI emits no device events under this WSL driver** — torch.profiler
  CUDA-time tables are empty even for pure matmul. Use host tables +
  sync-bracketed wall timers (`tools/perf_step_breakdown.py`).
- **torch.compile reduce-overhead (CUDA graphs) crashes with carried RNN/SSM
  hidden state** re-fed across calls ("output overwritten by subsequent
  run"), especially with a second no_grad forward per step. Default mode
  runs but ≈ eager for this loop. Windows skips compile (`sys.platform`
  check) — WSL-only bugs hide from Windows smoke tests.
- Per-bar A2C loop baselines: eager ~53-61 bars/s (±10% run variance,
  OneDrive-hosted repo); cost split = TBPTT-500 backward 32%, double forward
  38%, env CPU 16%, syncs 10%. Same-seed reruns are BITWISE deterministic —
  parity gates can demand exact equality.
- 12h/200-epoch needs ~380 bars/s (200 epochs × 5 days ≈ 16.6M bars).
  **next_value-reuse (deferred bootstrap) REINSTATED 2026-07-06, +29%**:
  clean-box ABAB n=4 = two-forward 50.9 vs deferred 65.8 bars/s
  (non-overlapping, p≈0.014); parity BITWISE. It was first rejected on an
  A/B that ran against a CONCURRENT user training run — contamination
  false-negative. **LESSONS (hard-won)**: (1) check `nvidia-smi` for GPU
  co-tenants BEFORE any benchmark — a negative A/B under contention is a
  false-negative machine; (2) sync-bracketed breakdowns attribute WORK, not
  critical path; only interleaved A/B on a quiet box counts; (3) contention
  skews timing only — bitwise/parity results survive it. Deferred HEAD ≈
  70h per 200-epoch run; only remaining <12h lever = sequence-window
  training on the fused parallel scan (>10×, math change, needs Moises'
  approval). Related: [[organize-research-folders]]
