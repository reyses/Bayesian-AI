# research/mamba_zigzag_baseline — Mamba sequential trade-management engine

The sequential lane of the nt8_catalog program: 46 static detectors + the
409-dim snapshot all fail the ±2m turn bar → "turns live in paths, not
snapshots" → a state-space model (Mamba) holding per-bar carried state is THE
remaining lane for exits/turns. Entry signal comes from the calibrated
40-stream combiner (see `research/nt8_catalog/`).

## Layout
- `THESIS_reward_design.md` — reward axioms (§6 regret-on-knowable, §7
  freeze/overcut tension). Read before touching the reward.
- `pipeline/`
  - `mamba_rl_network.py` — the network (Mamba trunk, entry/exit/critic heads;
    `forward_step` = acting, `forward_sequence` = learning windows).
  - `mamba_env.py` — env: FPS bar stream + ledger + scorecard reward
    (BetaRewardPolicy). Labels TEACH the reward (hindsight inputs at exit) —
    they NEVER enter the observation.
  - `reward_env.py` — RewardConfig + BetaRewardPolicy (+ synthetic tests).
  - `phit_feed.py` — causal knowability feed: calibrated decile-9/0 combiner
    fires (120s window) = c_t for regret/selectivity.
  - `warmstart_supervised.py` — supervised pre-train (entry head = active-label
    direction; exit head = label-ends-within-3min). Labels-teach is
    user-authorized (2026-07-17).
  - `train_mamba_rl_seq.py` — seq-window trainer (acting pass no_grad bar-by-bar
    + windowed A2C learning pass). Flags: `--smoke_metrics`, `--init_from`,
    `--compile_act` (torch.compile acting path; parity-gate first),
    `--no_autocast` (fp32; for parity), `--loss-dump` (per-bar npz).
- `tools/`
  - `run_speed_gates.sh` — the 3 compile gates (fp32 parity 1e-6 / bitwise
    self-determinism / bf16 perf A/B). Run from repo root in WSL, GPU free.
  - `compile_parity_check.py` — npz comparator behind the gates.
- `reports/` — `antifreeze_smoke.{md,log}` (3-arm smoke), `cold_arm/`,
  `warm_arm/` (per-arm trade CSVs), parity npz artifacts.
- Checkpoint: `checkpoints/mamba_warmstart.pth` (repo root, gitignored, 1.26MB;
  rebuild via `warmstart_supervised.py`).

## Running (user runs heavy training; smoke arms are ~9 min each)
```
# from repo root, WSL venv:
python pipeline/train_mamba_rl_seq.py \
  --days 2024_02_20,2024_05_15,2024_08_13,2024_11_12 \
  --num_episodes 2 --no-checkpoint --smoke_metrics --seed 0 \
  [--init_from checkpoints/mamba_warmstart.pth] [--compile_act]
```

## State (2026-07-17 night — see comms/096 + docs/daily)
- Anti-freeze smoke, 3 arms (COLD / WARM / FIXED): NO freeze in any arm;
  overtrading is the live failure mode; warm-start = direction knowledge, not
  rate discipline.
- Reward-wiring audit found the scorecard was mostly dead wire (raw $ stacked
  100:1, capture structurally 0, cut bonus flat-paid, wiggle dead) — repaired
  in `mamba_env.py` (5 fixes, scorecard-only reward). ARM FIXED = the
  candidate policy.
- Speed: eager ~250 bars/s; `--compile_act` + bounded oracle scan added;
  gates in `tools/run_speed_gates.sh`.
