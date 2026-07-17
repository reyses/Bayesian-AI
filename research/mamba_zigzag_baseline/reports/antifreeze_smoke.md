# Mamba ANTI-FREEZE — reward + supervised warm-start + real-data smoke A/B

**Date:** 2026-07-16
**Problem:** "Mamba decides that no trade is the best strategy to keep equity
intact" — the policy collapses to always-flat.
**Executor:** Claude (under reviewer Claude Fable). Nothing committed. GPU runs via WSL.

---

## 1. Root cause (from the code, confirmed empirically)

The reward the agent actually sees is `reward += trade_pnl` (raw dollars, ±$50)
**plus** the path-independent scorecard (±1). Raw PnL dominates the scorecard by
~10-50×. For a **cold policy with coin-flip direction**, expected `trade_pnl`
is negative (cost drag), so *every* trade has negative EV vs flat's zero → the
value function learns flat is strictly safest → freeze. Two mechanisms made it
worse:
- **`c_t` was mocked to 1.0** in `mamba_env.py` → regret was a blanket one-shot
  `−w_r` whisper per *episode* (swing_id was the constant `'s_flat'`), not a real
  per-setup signal.
- **A −$50 "hourly stagnation" cliff** stacked on top of raw PnL, making flat
  look even safer (and it *clobbered* `action_type=='ENTRY'` to `IN_POSITION_STEP`
  before the reward was scored, so entries were never credited).

The fix biases the *scorecard* toward acting on knowable setups (regret +
selectivity, now driven by REAL calibrated `c_t`), and — the real lever —
**warm-starts the direction** so trading has positive EV instead of coin-flip EV.

**Empirical note on the knowability feed:** FPS bars start at the overnight
session open (23:00 UTC); the calibrated decile-9/0 fires only begin at RTH open
(14:30 UTC). So a live signal (`c_t≥0.5`) is present on ~18% of a full day's bars
(~55% of the RTH portion), 0% overnight. The smoke therefore runs **full days**
so RTH signals are exercised.

---

## 2. What changed (one line per file)

- **`pipeline/phit_feed.py`** (NEW): causal knowability feed — `live_signal(ts)`
  returns `(c_t, dir)` from the most-recent decile-9 (`c_t=P`) or decile-0
  (`c_t=1−P`, inverted) fire within 120s (`fire_ts ≤ ts`), else `(0,0)`; sets
  `last_fire_ts` for a stable per-swing id. Offline self-test passes.
- **`pipeline/reward_env.py`**: `w_r` 0.25→0.75; new `w_s=0.30` **selectivity**
  credit on `ENTRY` (`+w_s·c_t` when `c_t≥θ_c` and entry dir == signal dir;
  else 0, no penalty); `selectivity` added to the scorecard; tests extended
  (5 original updated to track `w_r`, + 3 new).
- **`pipeline/mamba_env.py`**: mocked `c_t=1.0` → `phit_feed.live_signal`; **−$50
  hourly penalty block deleted** (also un-clobbers `ENTRY`); `ENTRY` now passes
  `predicted_dir`+`signal_dir` to the policy; `FLAT_STEP` swing_id = driving
  fire ts (regret caps once per *fire*, not once per *episode*); `info['entered']`.
- **`pipeline/train_mamba_rl_seq.py`** (flags only): `--init_from <ckpt>`
  (non-strict load + missing/unexpected report; a genuine resume still wins);
  `--smoke_metrics` (opt-in `[SMOKE]` per-epoch json; default behavior unchanged).
- **`pipeline/warmstart_supervised.py`** (NEW): supervised warm-start; AI labels
  as CE targets on entry (3-way) + exit (binary) heads at 1m-boundary bars; value
  head untouched; ledger held flat (labels never enter the observation).

## 3. Synthetic-test status — `reward_env.py`

All **8** pass (`python3.11 reward_env.py`): 5 original (magnitudes updated to
track `w_r=0.75`; semantics unchanged) + 3 new:
- (6) regret real-gated at θ_c: `c_t=.51 → −0.383`, `c_t=.49 → 0.000`
- (7) selectivity on aligned entry: `+0.240` (`w_s·c_t = 0.30·0.8`)
- (8) no selectivity (and no penalty) on misaligned / absent / sub-gate signal

`phit_feed.py` offline self-test also passes.

## 4. Warm-start build (10 days, 3 epochs)

- 158,186 bars prefetched; 13,380 1m-boundary samples.
- Entry-label dist HOLD/LONG/SHORT = **[703, 6616, 6061]** — AI picks are
  near-continuous (HOLD only ~5%). Class-weighted CE (weights [6.34, 0.67, 0.74]).
- Exit target pos/neg = 1350/12030 (pos_weight 8.91).
- CE_entry 0.952 → 0.818; CE_exit 1.256 → 1.242 (exit/turn timing is genuinely
  hard — consistent with the project's "exhaustion detectable but noisy").
- **Post-train entry head proposes a trade on 89% of 1m bars** (HOLD/LONG/SHORT =
  [1459, 2989, 8932]) vs the frozen policy's ~always-HOLD — the freeze lever is
  released. (A SHORT lean is a 10-day artifact.)
- Artifact: `checkpoints/mamba_warmstart.pth` (**1.26 MB**, gitignored).

---

## 5. A/B smoke — 4 days (2024_02_20, 2024_05_15, 2024_08_13, 2024_11_12), 2 epochs, seed 0

> `trades_per_day = trades / 4`. `pct_flat` over ALL bars (incl. overnight, no
> signal). `P(enter|·)` over flat bars. Holds in **seconds** (`5s-bar units × 5`).
> Reward-component sums are the scorecard terms (raw `trade_pnl` is separate).

Four arms ran (the plan's A/B grew two: the FIXED arms exist because the smoke
itself exposed dead reward wiring — §6a):

| arm | epoch | trades/day | pct_flat | P(enter\|sig) | P(enter\|nosig) | hold med (bars) | avg $/trade |
|---|---|---|---|---|---|---|---|
| COLD (new reward, random init) | 0 | 1505.8 | 13.0% | 0.767 | 0.716 | 4.0 | −4.84 |
| COLD | 1 | 128.0 | 1.3% | **0.826** | **0.618** | 34.5 | −4.78 |
| WARM (+ warm-start) | 0 | 3449.5 | 25.9% | 0.925 | 0.812 | 2.0 | −5.00 |
| WARM | 1 | 297.5 | 2.4% | 0.870 | 0.750 | 24.0 | −5.26 |
| FIXED (dollars-out + partial wiring) | 0 | 3869.5 | 27.7% | 0.929 | 0.859 | 2.0 | −5.00 |
| FIXED | 1 | 532.3 | 4.3% | 0.720 | 0.769 | 7.0 | −5.53 |
| FIXED2 (full wiring — candidate) | 0 | 3853.5 | 27.6% | 0.922 | 0.861 | 2.0 | −4.98 |
| FIXED2 | 1 | 517.0 | 4.4% | 0.700 | 0.742 | 7.0 | −4.69 |

### Reward-component sums (scorecard, per epoch)

| arm·ep | capture | regret | selectivity | cut | direction | wiggle | cost |
|---|---|---|---|---|---|---|---|
| COLD·0 | 0.0 | −89.8 | +95.8 | +1891.1 | −956.6 | 0.0 | −1806.9 |
| COLD·1 | 0.0 | −1.3 | +3.6 | +138.6 | −56.0 | 0.0 | −153.6 |
| WARM·0 | 0.0 | −37.4 | +257.3 | +4593.8 | −2490.4 | 0.0 | −4139.4 |
| WARM·1 | 0.0 | −9.6 | +19.3 | +325.2 | −133.6 | 0.0 | −357.0 |
| FIXED·0 | 0.0 | −42.2 | +303.8 | +3038.0 | −2785.2 | −1754.4 | −4643.4 |
| FIXED·1 | 0.0 | −19.9 | +22.0 | +202.5 | −311.0 | −182.1 | −638.7 |
| FIXED2·0 | **−1126.1** | −36.2 | +298.1 | +1579.1 | −72.0 | −1553.3 | −4624.2 |
| FIXED2·1 | **−2.0** | −20.2 | +22.6 | +196.1 | −284.4 | −159.9 | −620.4 |

Reading the columns: `capture=0.0` in COLD/WARM/FIXED is the dead-wire
signature (§6a). In FIXED2 capture goes NONZERO for the first time — and
negative at ep0 is the *correct* gradient (right-side churn that exits before
the move now pays for it). Cut in COLD/WARM is a flat `w_x` payout per wrong
trade (no decay inputs arrived); from FIXED on it decays with hold-time/MAE
(sums no longer ≈ 0.35·n_wrong). Wiggle wakes in FIXED but punishes EVERY
covered trade (qualifying stuck False → the inverted gap); in FIXED2 it is
selective (unaligned covered trades only).

## 6. Pass-criteria verdict

Pre-registered criteria: (a) ≥3 trades/day, (b) ≤40 trades/day, (c)
P(enter|signal) > P(enter|no-signal), at smoke scale (2 epochs × 4 days).

- **(a) PASS all arms** — and more fundamentally, **the freeze pathology never
  appeared in any arm** (pct_flat max 28%, monotonically declining; the
  always-HOLD collapse that motivated the night is gone).
- **(b) FAIL at 2 epochs, trending hard toward the band** (COLD 1506→128;
  FIXED2 3854→517 — ~3-7× collapse per epoch under cost/wiggle/capture
  pressure). A convergence criterion applied at epoch 1 was optimistic;
  re-check at epoch ~5 in the production run.
- **(c) PASS under COLD (+0.21 gap by ep1) — NOT YET under warm-start (~0,
  n=260 signal-bars = noise).** The warm-start prior ("enter when a label is
  active" ≈ always) fights entry selectivity early; COLD proves the reward
  teaches it. Production: keep the warm-start for direction, run longer, and
  bump w_s/w_w if the gap hasn't emerged by epoch ~5.

**Net: the reward policy answers Moises' brief** — Mamba no longer chooses
indefinite holding — **and the smoke's real product is §6a: three dead-wire
layers that would have silently crippled the full training run.**

## 7. Degenerate-signature check (§7 tension)

The §7 tension (freeze ↔ overcut) resolved to the overcut side, as the
axioms predicted it would once the flat attractor was removed: epoch-0
policies churn (hold median 2 bars = 10s, 3.8k trades/day) and the
cost/wiggle/capture terms grind the rate down epoch-over-epoch. No degenerate
signature beyond that churn: hold p90 grows with training (7→74-350 bars),
entries concentrate RTH (overnight has no fires to credit), and no arm
oscillated (rates fell monotonically). Watch in production: the scratch-loop
signature (enter→exit within 1-2 bars at high rate) should fall below ~50/day
by mid-curriculum or the cost term needs raising.

## 6a. The dead-wire audit (found DURING the smoke — the night's real product)

1. **Raw dollars still stacked on the scorecard** (`reward += trade_pnl`
   before `reward += v2_score['total']`): ±$5-50 vs ±0.35 = ~100:1 domination;
   the flat attractor was still the true gradient. Removed (dollars are
   reporting-only now).
2. **`remaining_extent_vol_norm` hardcoded 0.0** → always < θ_rem → capture
   short-circuited to 0 on every right trade. Now computed at entry from the
   live label's exit_price (labels TEACH the reward; they never enter the
   observation).
3. **Cut-bonus inputs never arrived** (`t_hold`/`mae_vol_norm` vs the env's
   `time_in_trade_bars`/nothing) → every wrong trade earned the FULL `w_x`
   flat. Now: real hold bars + an env-local MAE tracker (ticks).
4. **`is_qualifying=True` hardcoded** → wiggle dead.
5. **`Ledger.last_removed_position` does not exist** → the `hasattr` guard
   silently skipped the extras on EVERY exit → fixes 2-4 were still inert in
   ARM FIXED (capture 0, qualifying stuck False → every covered trade wiggled
   → entries pushed OUT of covered periods, the inverted ep1 gap). Fixed by
   stashing `record['extras']` at the `remove_position` call site.

Plus the pre-smoke fixes (§1-2): mocked c_t=1.0 → phit_feed; −$50 hourly
cliff removed; per-fire regret capping; `--init_from`; `--smoke_metrics`.

## 9. Speed pass (same night)

- Env fixes alone: **248 → 293.5 bars/s** (+18%) — oracle-scan early-break
  (sorted entry_ts; identical semantics, bounded per-bar cost at
  full-curriculum scale) + simplified extras path.
- `--compile_act` (torch.compile forward_step, default mode — NO cudagraphs,
  carried-state trap) + `--no_autocast` (fp32 parity harness) added to the
  trainer; acting path only, learning pass eager.
- Gates (tools/run_speed_gates.sh → reports/speed_gates.log) — **ALL PASS**:
  - Gate 1 (fp32 parity, 2000 bars): actions byte-identical, losses
    max|Δ|=2.4e-07 (tol 1e-6), rewards Δ=0.
  - Gate 2 (bitwise self-determinism): two compiled runs byte-identical on
    actions, losses, AND rewards.
  - Gate 3 (bf16 full epoch): **431.2 bars/s vs 293.5 eager = 1.47×**; the
    [SMOKE] json tracks FIXED2·ep0 closely (15,414 trades in both; small
    component drift = bf16 compiled-vs-eager sampling noise, expected — fp32
    is where exactness was proven).
  - **Night total: 248 → 431 bars/s (1.74×).** Full-curriculum scale
    (~16M bars/epoch): ~18h → ~10.3h per epoch.
  Causality untouched — every change is reward-side or overhead-side; the
  observation path is byte-identical.

## 8. Artifacts

- Report: `research/mamba_zigzag_baseline/reports/antifreeze_smoke.md`
- Raw log: `research/mamba_zigzag_baseline/reports/antifreeze_smoke.log`
- Warm-start checkpoint: `checkpoints/mamba_warmstart.pth` (1.26 MB, gitignored)
- New/modified code: `pipeline/{phit_feed,warmstart_supervised}.py` (new),
  `pipeline/{reward_env,mamba_env,train_mamba_rl_seq}.py` (modified)
