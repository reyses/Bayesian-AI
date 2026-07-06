# Mamba RL Training Stack — Runtime Review (RTX 3060, WSL2)

**Date**: 2026-07-05 · **Scope**: runtime only — no reward logic, architecture semantics, or training math touched.
**Files**: `pipeline/{train_mamba_rl,mamba_env,mamba_rl_network}.py`, `core_v2/features.py`, `core_v2/FPS/forward_pass_system.py`.

## TL;DR

- Clean eager baseline: **53.15 bars/s**. The shipped `torch.compile(reduce-overhead)` config **crashes on WSL at step ~2** (CUDA-graph output aliasing with the carried hidden state) — the de-facto production path was already eager; prior smoke tests passed because Windows skips compile.
- Three of the four mandated fixes were already implemented or moot; the measured bottlenecks were elsewhere (per-step syncs, env CPU string-lookup tax, kernel-launch overhead).
- All applied changes are **bitwise loss-identical** (not just tol 1e-4) over a 1300-step fixed-seed run spanning an optimizer step.
- **The <12h / 200-epoch target is NOT reachable within the "no training-math change" constraint** (needs ~380 bars/s ≈ 7×). Two math-boundary options that get there are listed at the end; both need user approval.

## Profiler results (task 1)

CUPTI does not emit device events on this WSL2 driver — even a pure-matmul
probe shows no CUDA-time columns (**WSL blocker**: `torch.profiler` CUDA
tables unavailable; also `nvidia-smi`-level CUPTI permissions cannot be
granted from inside WSL). Host-side tables + sync-bracketed wall-clock
attribution were used instead; the workload is host-launch-bound, so CPU
tables ARE the relevant view.

**Top ops by CPU time, 500 profiled steps** (`reports/perf/baseline_nocompile/top_ops_cpu.txt`):

| op | self CPU | total CPU | calls | note |
|---|---|---|---|---|
| cudaLaunchKernel | 5.03s (33.7%) | 5.16s | **252,497** | ~505 launches/bar |
| aten::linear | 0.17s | 3.87s | 28,000 | 56/bar (2 fwd + bwd) |
| aten::to / _to_copy | 0.43s | 2.81s | 64,530 | bf16 autocast casts |
| aten::copy_ | 0.65s | 2.43s | 82,495 | |
| MulBackward0 (engine) | 0.19s | 1.27s | 9,000 | scan backward |
| aten::mul | 0.55s | 1.26s | 36,003 | |
| SliceBackward0 | 0.10s | 1.10s | 12,000 | scan backward |
| aten::mm | 0.50s | 0.90s | 18,500 | |
| aten::conv1d | 0.02s | 0.73s | 4,000 | 8/bar (2 blocks × 2 fwd ×2) |
| cudaMemcpyAsync | 0.68s | 0.68s | 30,000 | 60/bar (see fix 3) |

**Wall-clock attribution per bar** (sync-bracketed, 600 steps, eager — `reports/perf/step_breakdown.txt`):

| component | ms/bar | share |
|---|---|---|
| TBPTT window backward+opt (amortized) | 5.28 | 32% |
| forward #2 (no_grad next_value) | 3.17 | 19% |
| forward #1 (action) | 3.14 | 19% |
| env.step CPU (2.4ms = assemble_v2_grid tax) | 2.66 | 16% |
| loss seam + entropy.item() sync | 1.59 | 10% |
| action.item() (inherent: env needs the action) | 0.50 | 3% |
| pack + single H2D | 0.12 | 0.7% |

## Hypothesis verdicts (task 2)

| # | hypothesis | verdict | evidence |
|---|---|---|---|
| a | Python `for t in range(L)` scan + per-bar full-window recompute dominates (O(L²)) | **STALE — already fixed before this session.** The loop feeds L=1/bar with carried SSM state (commit 97bf1dd4), so the scan body runs once per bar. The *residual* cost is kernel-launch overhead (~505 launches/bar) and the backward through the 500-step unrolled graph — 32%+38% of wall time. NOTE: the L=1 refactor silently dropped the conv1d's 3-bar temporal receptive field (no carried conv state); pre-existing, not touched — flagging as a semantics question for the user. | breakdown table; profiler launch counts |
| b | 10 separate H2D transfers per env step | **STALE — already fixed** (pinned pack buffer, single H2D per state, commit 0bb3b49d era). Packing+H2D is now 0.12 ms/bar (0.7%). The 60 memcpys/bar that remained were per-step *scalar* `torch.tensor(...)` creations and `.item()` D2H syncs — fixed now (fix 3). | breakdown; memcpy count |
| c | Per-step `.item()` GPU syncs | **CONFIRMED** — `entropy.item()` every bar (+ `float(step_loss)` in dump mode). Fixed: entropies buffered on-GPU, flushed once per TBPTT window. `action.item()` is inherent (the env consumes the action on CPU). loss seam bucket: 1.59 ms/bar → sub-ms. | breakdown; fix-3 commit |
| d | Stale "Triton unavailable on Windows" hardcode; WSL supports mamba-ssm | **HALF-TRUE, and the half that matters is worse than stale — it's a trap.** WSL2 runs the fused kernels fine (installed `mamba_ssm 2.2.4` + `causal_conv1d 1.5.0.post8`, cu12/torch2.6/cxx11abiFALSE wheels). BUT `Mamba.step()` is inference-only: backward *runs* and the input grad is **None** — `selective_state_update` records no autograd edge, so the TBPTT recurrence would train **silently wrong**. It's also only 1.4× at L=1 (0.613 vs 0.850 ms/block; both launch-bound). `MAMBA_AVAILABLE=False` must stay for training; the comment now states the real reason. | `reports/perf/mamba_ssm_probe.txt` |

## Fix-by-fix results (task 3)

Noise floor first: same-seed repeat of the baseline is **bitwise identical**
over 1300 steps (losses, actions, rewards) — the parity gates below are exact,
not approximate.

| change | commit | parity vs baseline (1300 steps, seed 42) | bars/s |
|---|---|---|---|
| baseline (eager; compile config crashes on WSL) | 16dea332 | — | **53.15** |
| instrumentation (inert w/o flags) | 0175b4f1 | bitwise | = |
| **[3] sync/H2D-scalar removal** (entropy deferral, `torch.full`, pos_weight hoist, VRAM-watchdog cache) | e3d8f3a5 | **bitwise PASS** | 60.85 |
| **env/data hot path** (`assemble_v2_grid` index cache ×2 call sites, per-bar grid row cache, per-day tz caches, lazy `v2_dict`) | 32b10810 | **bitwise PASS** | 57.47† |
| **[1] mamba-ssm** — REJECTED for the grad path (silent gradient severing); wheels installed, probe committed | (docs) | n/a (no runtime change) | = |
| **[2] nan_to_num removal** — ALREADY DONE pre-session (0bb3b49d/97bf1dd4); verified absent from train + eval paths | — | — | — |
| **[4] torch.compile repair** (default mode, no cudagraphs, **opt-in**) | 88e125f4 + opt-in flip | loss Δ **1.5e-3 → FAILS 1e-4 gate** (actions/rewards still identical over 1300 steps) | 63.37 |

† run-to-run variance on this box is ±10% (WSL + OneDrive-hosted repo); see
the controlled sweep at the bottom for the final numbers.

**Fix [4] resolution (per the "any divergence = revert and report" rule)**:
default-mode compile RUNS (crash fixed) and measured +~20% in the same-sweep
comparison, but per-step loss drifts 1.5e-3 vs eager from step 0 — inductor
refuses/reorders bf16 reductions; inherent to ANY compile of this loop, not
a defect of the change. Sampled actions and rewards remained identical over
the full 1300-step window (drift never crossed a decision boundary there),
but the 1e-4 criterion is failed, so compile is now **opt-in** (`--compile`),
eager is the default. Reverting to the OLD config was not an option — it
crashes ("accessing tensor output of CUDAGraphs that has been overwritten by
a subsequent run": carried hidden_states/value are cudagraph outputs, and
the second no_grad next_value invocation overwrites them).

**Env fix component evidence** (sync-bracketed breakdown, before → after):
env.step CPU **2.66 → 0.34 ms/bar** (enqueue 1.24 → 0.07, getobs 1.12 →
0.18, iter 0.10 → 0.03). `reports/perf/step_breakdown_postfix.txt`.

## Acceptance vs target

- 200-epoch, 5-day run at the final eager rate: **~29–34 h** — target <12h
  **NOT met** within "runtime only".
- The remaining 70% of wall time is (i) backward through the 500-step
  unrolled python-scan graph and (ii) two full forwards per bar at batch=1
  — both are *structural*, not overhead.

## Two math-boundary options that reach the target (need approval)

1. **Reuse forward #1(t+1) as forward #2(t)** — `next_value(s_{t+1})` is
   recomputed with grad at the next step anyway; values are numerically
   identical except at the 1-in-500 window-boundary step (post-update
   weights). Halves model compute → est. ~1.6×. Small, auditable change.
2. **Sequence-window training via the fused parallel scan** — observations
   don't depend on actions (only `ledger_state` does); a windowed
   forward with the differentiable `selective_scan_fn` is **250×** faster on
   the mamba trunk (0.454 vs 113.8 ms per 500 bars, measured). This is the
   restructure the whitepaper's GPU-batched direction implies. Est. total
   >10×; changes gradient/bootstrap semantics (off-policy `ledger_state`
   within a window) — a real training-math change, not a tweak.

## WSL install/tooling notes

- `mamba-ssm` pip sdist tries to compile CUDA ext; use the prebuilt wheels
  (`+cu12torch2.6cxx11abiFALSE`, py312). `causal-conv1d 1.6.2.post1` was
  ABI-broken vs torch 2.6 (undefined `c10_cuda_check_implementation`);
  1.5.0.post8 matches.
- `mamba_ssm 2.2.4` import breaks against `transformers 5.12.1`
  (`GreedySearchDecoderOnlyOutput` removed) — probe stubs it; not needed for
  kernels. Don't downgrade transformers just for this.
- CUPTI dead under this WSL driver → no CUDA-time profiler tables, no
  nsight; host-side + cuda-event timing only.
- `torch._inductor.config.compile_threads = 1` in train_mamba_rl.py header
  serializes inductor compilation (minutes of warmup); Windows-compat
  leftover, safe to lift under WSL (not changed this session — header
  config, one line, flag for next touch).

## Controlled final speed sweep (serial, back-to-back, same day, seed 42, 2000 bars)

| state | bars/s |
|---|---|
| baseline eager (pre-session, separate run) | 53.15 |
| all fixes, eager (sweep run 1 / run 2) | 49.38 / 52.88 |
| all fixes + compile default (`--compile`) | 63.37 |

**Honest read on totals** (per the repo's own CI discipline): single-run
bars/s swings ±10% on this box (eager measurements across the session:
49.4, 53.2, 57.5, 60.85). The env-fix and sync-fix gains are proven at the
COMPONENT level (env 2.66→0.34 ms/bar; per-step syncs/memcpys removed per
profiler counts) but the end-to-end delta is within single-run noise — do
not quote "X% faster overall" from these runs without an N≥10 A/B. The
compile win (+~20% same-sweep) is the one lever with a visible total effect,
and it costs a 1.5e-3 loss drift.

## Bottom line

Eager end-to-end: **~50–61 bars/s** → a 200-epoch × 5-day run is **~29–34 h**,
not <12 h. The remaining wall time is structural: TBPTT-500 backward
(~37%), two full batch-1 forwards per bar (~44%). No runtime-only change
moves those; the two math-boundary options above (next_value reuse ~1.6×,
sequence-window training on the fused parallel scan >10×) are the paths to
the target and both need explicit approval since they alter training
semantics (boundary-step bootstrap freshness / within-window ledger_state
off-policy drift respectively).
