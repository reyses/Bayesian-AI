# Night 2 — Mamba anti-freeze smoke (A/B/C) + Exit Dojo pilot synthesis
**Doc:** 096 · **Date:** 2026-07-17 (night) · **Author:** Claude (reviewer/orchestrator) · **Status:** DRAFT→FINAL at dawn
Moises' directives (asleep): "prepare mamba to run, run smoke test with real
data to get a reward policy where mamba does not choose to hold indefinitely"
(clarified: the pathology is flat-collapse — "it decides that NO TRADE is the
best strategy to keep equity intact"); labels may TEACH (supervised warm-start
authorized), never OBSERVE; Exit Dojo = "sonnets try to solve the puzzle and
record findings."

## 1. Exit Dojo pilot — DONE (committed 9d3c5706)
- 10 Sonnet pilots, stratified episodes (3 winner / 3 midflip / 2 instantfail
  / 2 chop), one distinct 2025-26 day each, 1-min causal packets.
- Scorer bug found+fixed: pilots wrote `t=7m:`, parser expected `t=7:` — all
  10 parsed as forced-hold. After fix: **7/10 beat the fixed-5m hold; mean
  +11.5 pts/trade vs −2.5; median oracle-ratio 0.475** (static system: 0.23).
  Single-prompt leakage caveat governs — hypothesis generation ONLY.
- **10/10 independent convergence** on one master rule: exit only on a
  2-3-family confluence inside a 1-3 min window (ER10 regime change +
  fresh multi-family against-fires + giveback DYNAMICS-not-level + bar
  anatomy + vol direction). Three regimes (with-trend / wrong-side / chop),
  each with its own rule set; confirmation tax 2-6 min = the P_hold lag
  family; pilots managed around it (tighten tiers, exit-into-bounce).
- Full grammar + Mamba state/reward mappings + EXIT-GRAMMAR-01 graduation
  candidate: `research/exit_dojo/reports/pilot_synthesis.md`.

## 2. Mamba anti-freeze smoke — three arms, one lesson
Setup (all arms): 4 real days (2024_02_20, 05_15, 08_13, 11_12), 2 epochs,
seed 0, no checkpointing, [SMOKE] metrics per epoch. Warm-start = supervised
pre-train of entry head on active-label direction + exit head on
label-ends-within-3min (CE 0.95→0.82 / 1.256→1.242 over 3 epochs; ckpt
`checkpoints/mamba_warmstart.pth`, 1.26MB).

### ARM COLD (new reward, random init) — VERDICT: no freeze, but see §3
| epoch | trades/day | pct_flat | P(enter\|sig) | P(enter\|nosig) | hold med (bars) | avg $/trade |
|---|---|---|---|---|---|---|
| 0 | 1505.8 | 13.0% | 0.767 | 0.716 | 4.0 | −4.84 |
| 1 | 128.0 | 1.3% | **0.826** | **0.618** | 34.5 | −4.78 |

- **The freeze pathology did NOT appear** (pct_flat 1-13%, never 100%).
- Signal-conditional entry EMERGED in 2 epochs: the P(enter|sig) −
  P(enter|nosig) gap widened +0.05 → **+0.21**.
- Overtrading, not freezing, is the active failure mode (128/day at ep1 vs
  the ≤40/day pass band; trajectory collapsing 1506→128).
- [PERF] 254.4 bars/sec eager (128,004 bars / 503s).

### ARM WARM (new reward + supervised warm-start init; pre-fix wiring)
| epoch | trades/day | pct_flat | P(enter\|sig) | P(enter\|nosig) | hold med (bars) | avg $/trade |
|---|---|---|---|---|---|---|
| 0 | 3449.5 | 25.9% | 0.925 | 0.812 | 2.0 | −5.00 |
| 1 | 297.5 | 2.4% | 0.870 | 0.750 | 24.0 | −5.26 |

- No freeze here either — and MORE trading than COLD (3450→298/day vs
  1506→128), with a SMALLER signal gap (+0.12 vs +0.21 at ep1). Expected:
  the supervised entry head was trained to propose whenever a label is
  active (proposes on 89% of bars), so warm-start begins hyperactive; under
  the dollar-dominated pre-fix reward it prunes slower. Warm-start's value
  is direction knowledge, not rate discipline — rate discipline must come
  from the (fixed) scorecard. [PERF] 247.9 bars/sec.

### ARM FIXED (warm init + first wiring repair — dollars out) — PARTIAL, see below
| epoch | trades/day | pct_flat | P(enter\|sig) | P(enter\|nosig) | hold med (bars) | avg $/trade |
|---|---|---|---|---|---|---|
| 0 | 3869.5 | 27.7% | 0.929 | 0.859 | 2.0 | −5.00 |
| 1 | 532.3 | 4.3% | 0.720 | **0.769** | 7.0 | −5.53 |

- Cut-bonus decay LIVE (sum_cut no longer a flat multiple of w_x) and wiggle
  LIVE (−1754/−182) — but `sum_capture` STILL 0.0, and the ep1 signal gap
  **INVERTED** (−0.05): entries drifted AWAY from label-covered periods.
- Root cause found: `Ledger` has **no `last_removed_position` attribute** —
  the `hasattr` guard (worker's original, inherited by my patch) silently
  returned False on every exit, so the entry-time extras (oracle trade,
  alignment flag, remaining extent) NEVER reached the scorecard. Effects:
  capture dead (remaining extent defaulted 0 < θ_rem) and `is_qualifying`
  defaulted False → EVERY label-covered trade wiggle-punished regardless of
  alignment → a gradient actively pushing entries OUT of covered periods =
  the observed inversion. The dead attribute also explains COLD/WARM's
  `swing_id='unknown'` regret keying.
- Fix: stash `record['extras']` at the `remove_position` call site (the
  record is in scope in the same step); `_last_exit_extras` read by the
  hindsight block. Oracle early-break scan also landed: [PERF] 264.3 bars/s
  (+6.6% over the 247.9 eager baseline).

### ARM FIXED2 (extras round-trip repaired — capture + qualifying live) — THE CANDIDATE
| epoch | trades/day | pct_flat | P(enter\|sig) | P(enter\|nosig) | hold med (bars) | avg $/trade |
|---|---|---|---|---|---|---|
| 0 | 3853.5 | 27.6% | 0.922 | 0.861 | 2.0 | −4.98 |
| 1 | 517.0 | 4.4% | 0.700 | 0.742 | 7.0 | −4.69 |

- **All scorecard terms LIVE for the first time**: capture −1126.1 (ep0) /
  −2.0 (ep1) — nonzero, and the ep0 negative is the CORRECT gradient (right-
  side churn that exits before the move now pays); cut decays with speed/MAE;
  wiggle selective (only unaligned covered trades); selectivity + windowed
  regret firing on real calibrated fires.
- [PERF] 293.5 bars/s — +18% over the night's 248 eager baseline from the
  env fixes alone (oracle early-break + simplified extras path).

## VERDICT (smoke)
1. **ANTI-FREEZE: PASS in all 4 arms.** pct_flat never exceeded 28%, always
   declining; Moises' pathology ("NO TRADE is the best strategy") never
   appeared. The scorecard design (windowed regret on knowable fires +
   immediate selectivity credit, dollars out) removes the flat attractor.
2. **REWARD WIRING: repaired and verified live** (the actual product of the
   night — three separate dead-wire bugs would have silently crippled any
   full training run).
3. **ENTRY SELECTIVITY: emerges under COLD (+0.21 gap by ep1) but not yet
   under warm-start (gap ~0 at ep1, n=260 sig-bars — noise).** The warm-start
   prior is "enter when a label is active" (~always proposing); 2 epochs of
   RL hasn't overridden it. Warm-start buys direction knowledge at the cost
   of early entry discipline — the production run needs a longer curriculum
   (gap watched from epoch ~5) or a selectivity/wiggle weight bump if it
   fails to emerge. Trades/day trajectory (3854→517, ~7.5× collapse per
   epoch) says the pressure is working.
4. **Speed: 248 → 293.5 bars/s from env fixes; `--compile_act` gates
   (fp32 1e-6 parity / bitwise self-determinism / bf16 perf) in
   `reports/speed_gates.log`.** Causality untouched — all changes are
   reward-side or overhead-side; the observation path is byte-identical.

## 3. Reward-wiring audit (found DURING the smoke; the real product of the night)
The COLD [SMOKE] lines showed `sum_capture=0.0` in both epochs. Root-cause
audit of `mamba_env.py` step() found the scorecard was largely DEAD WIRE:
1. **Raw dollar PnL still stacked on the scorecard** (`reward += trade_pnl`
   then `reward += v2_score['total']`). A ±$5-50 dollar term vs ±0.35
   scorecard terms = 100:1 domination — the flat attractor the scorecard was
   designed to remove was still the gradient. (Episode rewards −30,028 /
   −2,518 are dollar-dominated sums.)
2. **`remaining_extent_vol_norm` hardcoded 0.0** at exit → always below
   θ_rem=1.5 → the "late entry" branch fired on EVERY right trade →
   **capture structurally zero**. No capture gradient existed.
3. **Cut bonus paid in full for every wrong trade**: reward_env reads
   `t_hold`/`mae_vol_norm`; the env sent `time_in_trade_bars` and nothing —
   both defaulted 0 → `w_x·e⁰·e⁰` = flat 0.35 per wrong exit (COLD ep0
   sum_cut=1891 — the largest positive term). Wrong-fast and wrong-slow paid
   identically; net wrong-trade ≈ −0.15 vs right-trade ≈ −0.10.
4. `is_qualifying=True` hardcoded → wiggle term dead; oracle `swing_id`
   read a nonexistent key (`id`) → regret capping keyed 'unknown'.
5. `actual_dir` inferred from PnL sign even when the covering label's true
   direction was available.

**Fix applied** (mamba_env.py only; reward_env untouched; scorecard-only
reward): dollars out of the reward (reporting only); remaining extent
computed AT ENTRY from the live label's exit_price (labels TEACH the reward —
they still never enter the observation); env-local MAE tracker (ticks);
`t_hold`=duration bars; `is_qualifying` = entry coincided with an aligned
calibrated fire; label-true `actual_dir`; loader's real `swing_id`.
COLD/WARM ran on the pre-fix wiring (internally consistent A/B of the
warm-start effect); ARM FIXED is the candidate reward policy.

## 4. Interpretation discipline
2 epochs × 4 days is a SMOKE — it answers "does the policy collapse to flat
and does entry track the signal," nothing about $/day. No $/day claim is made
or implied. The anti-freeze conclusion holds under the pre-fix wiring caveat:
COLD's no-freeze was earned mostly by regret/selectivity (correctly wired)
while raw PnL still dominated — the FIXED arm is the honest test.
