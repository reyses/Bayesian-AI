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

- Target <12h **NOT met** within "runtime only": clean-box HEAD (with
  deferred bootstrap) is 65.8 bars/s vs ~380 needed → ~70 h for
  200 epochs × 5 days (see corrected Bottom line; the original "~29–34 h"
  figure here was an arithmetic error).
- The remaining wall time is backward through the 500-step unrolled
  python-scan graph + one batch-1 forward per bar — *structural*, not
  overhead.

## ⚠ Measurement contamination notice (added 2026-07-06)

The user confirmed ANOTHER training run (with checkpoint saving) shared the
GPU during the 2026-07-05 measurements. Consequences:

- **All parity/bitwise gates stand** — contention changes timing, never
  computed values. The crash diagnosis, autograd verdicts, and ABI findings
  also stand.
- **All 07-05 bars/s figures are contaminated** (explains the 49–63 spread
  on identical code). Clean-box (verified idle via nvidia-smi) re-measurement
  below supersedes them.
- The first deferred-bootstrap rejection was a **contamination
  false-negative** (see next section).
- The user's concurrent run was NOT corrupted by this session: every
  measurement run used `--no-checkpoint` (no checkpoint read/write) and
  broke before the plot/save block; all transient file states carried
  bitwise-identical training math.

## Paths to the target — UPDATED 2026-07-06 (clean box)

1. **Reuse forward #1(t+1) as forward #2(t) — WORKS. REINSTATED (7e8c4620).**
   Implemented bit-exactly (deferred loss formation; explicit re-forward
   kept only at window-close/episode-end so no optimizer step ever lands
   behind a bootstrap): parity gate BITWISE PASS over 1299 steps, re-verified
   on the reinstated file. Clean-box interleaved ABAB (n=4 each,
   `ab_deferred_quiet.txt`): two-forward 50.90 vs deferred **65.81 bars/s —
   +29%**, non-overlapping distributions (min B 57.5 > max A 54.0, rank-sum
   p≈0.014). The earlier same-day rejection (Δ −1.85 in
   `ab_deferred_forward.txt`) was measured against a concurrent GPU tenant —
   false negative — and its "the no_grad forward was free behind the
   action.item() sync" mechanism story is **retracted**.
2. **Sequence-window training via the fused parallel scan** — still the only
   lever that reaches <12h (deferral gets ~70h, see below). Observations
   don't depend on actions (only `ledger_state` does); a windowed forward
   with the differentiable `selective_scan_fn` is **250×** faster on the
   mamba trunk (0.454 vs 113.8 ms per 500 bars, measured). Est. total >10×;
   changes gradient/bootstrap semantics (off-policy `ledger_state` within a
   window) — a real training-math change requiring explicit approval.

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
  serialized inductor compilation (minutes of warmup); Windows-compat
  leftover — now gated to `os.name == 'nt'` (commit d5c671ec).

## Controlled final speed sweep (2026-07-05 — CONTAMINATED, superseded)

| state | bars/s |
|---|---|
| baseline eager (pre-session, separate run) | 53.15 |
| all fixes, eager (sweep run 1 / run 2) | 49.38 / 52.88 |
| all fixes + compile default (`--compile`) | 63.37 |

All 07-05 totals above were measured while a concurrent training run shared
the GPU (user-confirmed) — treat them as unreliable. The env-fix and
sync-fix gains remain proven at the COMPONENT level (env 2.66→0.34 ms/bar;
per-step syncs/memcpys removed per profiler counts). The compile figure
(+~20%, at a 1.5e-3 loss-drift cost) predates the deferred-bootstrap
reinstatement and hasn't been re-measured on a quiet box or on top of the
deferral.

## Bottom line (corrected 2026-07-06 — clean box, concurrent run absent)

An epoch = 5 days × ~16.5k bars ≈ 83k bars → 200 epochs ≈ 16.6M bars.
(The original "~29–34 h" line was an arithmetic error; retracted.)

| config | bars/s (clean box, n=4) | 200-epoch × 5-day estimate |
|---|---|---|
| two-forward eager | 50.9 | ~90 h |
| deferred bootstrap (per-bar trainer HEAD) | 62.9–65.8 | ~70 h |
| deferred + `--compile` (warm cache) | 90.5 | ~51 h |
| **sequence-window trainer (`train_mamba_rl_seq.py`)** | **268–280** | **~17 h** |
| needed for target | ~380 | <12 h |

**Sequence-window trainer landed 2026-07-06** (user-approved;
`docs/JULES_SEQUENCE_WINDOW_TRAINING.md`, commits 22e00238 + d71c70c4):
two-pass on-policy design — act eager under no_grad, learn via ONE
differentiable associative-scan forward per 500-bar window. Gates:
scan/forward equivalence 1e-7 fp32; self-determinism BITWISE over 2300
bars; 268–280 bars/s sustained. Deliberate semantic change: conv1d 3-bar
receptive field restored (the "conv-state decision" resolved; old
checkpoints not comparable). Remaining bottleneck = the acting pass's
bar-by-bar python loop (~2.7 ms/bar); candidate next lever:
torch.compile on forward_step (~1.4× would cross the <12h line).

`--compile` stacking measured 2026-07-06 (`ab_compile_deferred.txt`): eager
62.94 vs compile 90.49 warm-cache mean, **+44%**, non-overlapping. First
run after a cache invalidation reads low (~55) because the backward graph
compiles at the first TBPTT boundary inside the timed window — one-time
cost. Compile stays opt-in: ~1.5e-3 bf16 loss drift vs eager (fails the
1e-4 gate; actions were identical over the 1300-step check on the
two-forward build — drift not yet re-characterized on the deferred build).

Remaining wall time is structural: backward through the 500-step unrolled
python-scan graph + one batch-1 forward per bar. The sequence-window
restructure (>10×, training-math change, needs approval) remains the only
identified path across the remaining ~4×.
