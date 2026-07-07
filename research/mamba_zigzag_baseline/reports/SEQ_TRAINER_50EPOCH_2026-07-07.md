# Seq-Window Trainer — 50-Epoch Run Verdict (2026-07-07)

Autonomous overnight run per user request ("make sure it runs, trades, run the
50 epochs"). Trainer: `train_mamba_rl_seq.py` (two-pass sequence-window,
day-preload), 5-day set `2024_02_20…02_26`, seed 42, 50 epochs.

## Headline

Runs clean, trades properly (after a bug fix), and **learns** — but converges
to a **non-profitable** policy and the back half **regresses into degenerate
over-holding**. Best checkpoint is **epoch 25**, not the final.

## Bug found & fixed first (the user's suspicion was correct)

The exit head samples Bernoulli {0=hold, 1=exit}; the raw int was passed to
`env.step`, but the env closes a LONG only on action 2/3 and a SHORT only on
1/3. So a long receiving exit-intent `1` no-oped and stayed open until the
15:55 guard rail → **1 trade/day, 16,459-bar hold**. Fix: map exit-intent →
SCRATCH (3), keep the policy action {0,1} for the loss. Commit in the seq
trainer; the per-bar trainer has the same latent bug (left as-is, flagged).

## Run health

- 3.9 h wall (14,060 s), 285 bars/s sustained, 4.01M bars, **no crash, no NaN**.
- Checkpoints every epoch (`mamba_rl_seq_checkpoint_ep{N}.pth`, 3.79 MB each).
- Recovered from a launch failure first: the initial `nohup` sent stdout to a
  dead pipe with no file → the process blocked mid-epoch once the pipe buffer
  filled (22 min, zero checkpoints). Relaunched via a `setsid` launcher that
  redirects internally (`tools/launch_seq_train.sh`).

## Learning curve (per-10-epoch windowed mean reward)

| window | mean reward | character |
|---|---|---|
| ep0-9 | -9,767 | untrained → fee-bleed fix (ep0 alone -54,187) |
| ep10-19 | -3,772 | direction learning |
| ep20-29 | **-3,402** | **peak** (best single: ep25 = -2,558) |
| ep30-39 | -4,191 | regression |
| ep40-49 | -3,832 | noisy, over-holding drift |

Peak window ep19-26 mean -3,186; final-10 mean -3,832.

## Behavior evolution (the real story)

| epoch | trades/5d | avg hold | avg $/trade |
|---|---|---|---|
| 0 | 9,630 | 7 bars | -$4.89 (pure fees) |
| 29 | 52 | 1,583 bars | -$2.51 ± $32 |
| 47 | 19 | 4,336 bars | -$38 ± $79 |
| 48 | 9 | 9,156 bars (12.7 h) | -$81 ± $133 |
| 49 | 15 | 5,493 bars | -$29 ± $101 |

The agent **overshot from over-trading to over-HOLDING**: it fixed the fee
bleed by epoch ~5, found decent moderate-frequency trading around epoch 25,
then in the back half drifted to a few very long-held positions with large
swings — approaching the original stuck-open pathology, but now as a *learned*
policy exploiting a long-hold local optimum.

## Diagnosis

The regression correlates with the **entropy-coefficient schedule**: it decays
from 0.01 → 0.001 over the first 50% of epochs (floor reached at **epoch 25**),
exactly where the peak is. After exploration dies, the policy exploits toward
the degenerate long-hold optimum. So the back-half regression is likely an
exploration-decay artifact, not a fundamental limit — but the *level* it peaked
at is still non-profitable (best -2,558/5d on the shaped reward).

## Verdict

- ✅ Mechanically sound: runs, trades, checkpoints, converges, no NaN.
- ✅ Learns: 20× reward improvement to the epoch-25 peak.
- ❌ Not profitable: best -2,558/5d; converged policy loses on direction.
- ⚠️ Back-half regression to over-holding (entropy floor too early).

## Recommended next steps (for the user)

1. **Evaluate the epoch-25 checkpoint OOS** — the reward is a *shaped* composite
   (hourly penalties, regret, capture), not literal P&L. Real $/day on
   ATLAS_NT8 is the honest metric. Use `mamba_rl_seq_checkpoint_ep25.pth`.
2. **Fix the entropy schedule** — floor at 0.001 by epoch 25 killed the back
   half. Try a higher floor (e.g. 0.005) or decay over the full 50 epochs, and
   re-run; the peak may extend rather than regress.
3. **Add a hold-duration penalty or exit-hazard prior** — the over-holding
   drift suggests the reward under-penalizes long holds once exploration stops.
4. **Fix the same exit-action bug in `train_mamba_rl.py`** (per-bar trainer)
   before it's ever used for a real run.
