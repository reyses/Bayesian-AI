# Mamba anti-freeze — evidence dossier + framing for the 2026-07-23 design session

**Owner's brief:** "the goal is to stop the decay to no action"
(`docs/daily/2026-07-22.md:91`). This document compiles the measured evidence,
ranks falsifiable mechanism hypotheses, and lays out candidate levers WITHOUT
picking a winner. Every claim is cited file:line. Nothing here is committed.

**Sources read:** `reports/antifreeze_smoke.md`, `reports/antifreeze_smoke.log`,
`THESIS_reward_design.md`, `pipeline/mamba_env.py`, `pipeline/reward_env.py`,
`docs/daily/INDEX.md:6`, `docs/daily/2026-07-22.md`,
`docs/memory/reference-mamba-ssm-wsl-perf.md`, `reports/runs/epoch_0_trades.csv`.

---

## A) The problem, stated precisely

**"Decay to no action" mechanically = policy collapse to always-flat.** The
trade/no-trade head learns HOLD (action 0) on every bar, so the ledger never
opens a position (`mamba_env.py:386` gates entry on `action in [1,2]`). Origin
diagnosis: for a cold, coin-flip-direction policy every trade has negative
expected `trade_pnl` (cost drag) vs flat's exact zero, so the value function
learns flat is strictly safest → freeze (`antifreeze_smoke.md:10-17`). It was
amplified by two now-removed bugs: raw dollars stacked on the ±0.35 scorecard at
~100:1 (`antifreeze_smoke.md:161-163`; dollars pulled OUT of reward at
`mamba_env.py:368-372`) and a −$50 hourly-stagnation cliff that also clobbered
ENTRY before it was scored (`antifreeze_smoke.md:22-23`; deleted
`mamba_env.py:424-432`).

**The mirror failure — overtrading — is what actually shows up in every run.**
"NO FREEZE in any arm; overtrading is the live failure mode"
(`INDEX.md:6`; `antifreeze_smoke.md:130-132`). The 3-arm smoke (4 days × 2
epochs) hit pct_flat 13–28%, never the always-HOLD collapse
(`antifreeze_smoke.md:92-101`). **Tonight's fresh single-day epoch-0 smoke
(2024-01-02) is the same pathology, sharper:** 2533 trades in one day, reward
−963.42 (`2026-07-22.md:26-27`), pct_flat ≈ 0.216, P(enter|sig) 0.85 vs
P(enter|nosig) 0.74. Recomputed from `reports/runs/epoch_0_trades.csv`:
median hold 3 bars (15 s), 41.3% of trades ≤ 2 bars (≤ 10 s = scratch-loop
churn), 144 winners / 2376 losers, PF-based Trade WR **−0.931** (gross win
$950 vs gross loss $13,844 — near-total loss), mean −$5.09/trade.

**Which arms/inits/lengths show which face:** freeze is the COLD-policy /
early-training / dollars-dominant regime (now defused). Overtrade is the
epoch-0 face of every current arm; rate then collapses 3–7× per epoch under
cost/wiggle/capture pressure (`antifreeze_smoke.md:135-137`). **The two are the
same gradient at different points on one curve — see §C.1.** The selectivity
gap is weak everywhere: tonight 0.11 (0.85 vs 0.74); warm-start ≈ 0 at n≈260
signal-bars (noise); only COLD reached +0.21 by ep1 (`antifreeze_smoke.md:138-141`).

---

## B) Evidence table (every freeze/overtrade number, with source)

| Metric | Value | Source |
|---|---|---|
| Owner brief "stop decay to no action" | — | `2026-07-22.md:91` |
| Freeze root cause (neg-EV vs flat=0) | qualitative | `antifreeze_smoke.md:10-17` |
| Raw-$ vs scorecard domination | ~100:1 | `antifreeze_smoke.md:161-163` |
| Freeze never appeared (max pct_flat) | 28% | `antifreeze_smoke.md:130-132` |
| COLD ep0 → ep1 trades/day | 1506 → 128 | `antifreeze_smoke.md:94-95` |
| WARM ep0 → ep1 trades/day | 3450 → 298 | `antifreeze_smoke.md:96-97` |
| FIXED2 ep0 → ep1 trades/day | 3854 → 517 | `antifreeze_smoke.md:100-101` |
| Per-epoch rate collapse | 3–7× | `antifreeze_smoke.md:135-137` |
| Selectivity gap: COLD ep1 / WARM ep1 | +0.21 / ~0 (noise) | `antifreeze_smoke.md:138-141` |
| FIXED2 ep0 capture (first nonzero) | −1126.1 | `antifreeze_smoke.md:113` |
| FIXED2 ep0 cost / wiggle | −4624 / −1553 | `antifreeze_smoke.md:113` |
| **Tonight** reward / trades | −963.42 / 2533 | `2026-07-22.md:26-27` |
| Tonight pct_flat / P(enter\|sig) / P(enter\|nosig) | 0.216 / 0.85 / 0.74 | task brief + `[SMOKE]` stdout* |
| Tonight cost term (= 2533 × −0.30) | −759.9 | derived, `reward_env.py:84-85` |
| Tonight wiggle term (⇒ 65% non-qual) | −246 | derived, `reward_env.py:97-99` |
| Tonight median hold / % ≤2 bars | 3 bars / 41.3% | `runs/epoch_0_trades.csv` (recomputed) |
| Tonight PF-Trade-WR / mean $/trade | −0.931 / −$5.09 | `runs/epoch_0_trades.csv` (recomputed) |
| Reward weights w_c/w_x/w_r/w_s/w_d/w_w/w_cost | 1.0/.35/.75/.30/.20/.15/1.0 | `reward_env.py:9-18` |
| cost_ticks / sigma_ticks(fixed) / tau / theta_rem / theta_c | 3.0 / 10.0 / 5.0 / 1.5 / 0.5 | `reward_env.py:21-26`, `mamba_env.py:94` |

\*The full per-epoch `[SMOKE]` json for tonight's single-day run was **not
persisted to a report file** — only reward and trade count reached the journal
(`2026-07-22.md:26-27`). pct_flat / P(enter|·) come from the task brief quoting
run stdout; treat as un-archived until re-run with `--smoke_metrics` tee'd to a
log. Recommend fixing the tee before the production run.

---

## C) Mechanism hypotheses (ranked, falsifiable)

**C.1 — Reward has no positive per-trade term, so gradient descent's fixed
point IS flat (STRONGEST).** Tonight's −963 decomposes to cost −759.9 (an
EXACT fixed −0.30 per exit: `−w_cost·cost_ticks/sigma_ticks = −1.0·3/10`,
`reward_env.py:84-85`, sigma_ticks pinned 10.0 `mamba_env.py:94`) plus wiggle
−246 (⇒ ~65% of entries non-qualifying, `reward_env.py:97-99`). **Both terms
are monotone-increasing penalties in trade count with NO counterbalancing
per-trade reward** — capture is structurally ≈ 0 (`sum_capture` 0.0 in
COLD/WARM/FIXED, `antifreeze_smoke.md:107-112`; NEGATIVE once wired,
`:113`). So the only gradient the policy can follow is *trade less*. The
smoke's "monotonic decline 3854→517" (`:135-137`) is therefore the trajectory
**toward** freeze, not a healthy settle — extrapolated with capture ≈ 0 it
converges on ~0 trades = the always-flat collapse the night set out to kill.
*The "freeze is gone" verdict (`:130-132`) is only established at 2 epochs.*
**For:** exact arithmetic match (759.9), capture ≈ 0, monotone collapse.
**Against:** COLD grew a real selectivity gap (+0.21) before collapsing, so
*some* structure is learned on the way down. **Cheap experiment:** run FIXED2
to epoch ~8–10 on the 4-day smoke; plot trades/day. Flat asymptote at ≪ 3/day
confirms; a floor in the 3–40 band refutes.

**C.2 — Capture can't pay, so the exit "edge" never earns.** Capture is gated
three ways: right direction AND remaining_extent ≥ θ_rem(1.5) AND entry aligned
(`reward_env.py:107-115`); late entries score 0 (`:112-113`). The thesis calls
the exit head "the edge... where profitability is manufactured"
(`THESIS:33-37`) — but if capture is ~0 in practice, the one term that rewards
*good* trading is silent, leaving only penalties (⇒ C.1). **For:** capture 0 /
negative across all arms. **Against:** capture went nonzero (−1126) once extras
round-tripped (`antifreeze_smoke.md:113`) — the wiring is now live, just
punishing churn that exits before the move. **Cheap experiment:** log capture
conditional on {right, remaining≥θ_rem}; if that subset is also ≤ 0, the
denominator/θ_rem calibration is wrong, not the policy.

**C.3 — Entropy collapse / no exploration schedule.** THESIS §7 explicitly
pairs the wiggle penalty with "a decaying entropy/exploration bonus so it
samples trades early instead of defaulting to safe-flat" (`THESIS:52`) — **no
such bonus is wired** in `reward_env.py` or `mamba_env.py`. Without it, once
C.1 pressure starts shrinking the action distribution there is nothing holding
exploration open. **For:** the design doc predicts exactly this gap.
**Against:** ep0 is wildly over-exploring (2533 trades), so collapse is
gradient-driven, not premature entropy starvation — this is a *late*-training
risk, not tonight's cause. **Cheap experiment:** log action-entropy per epoch;
a monotone decline tracking the trade-rate collapse implicates it.

**C.4 — Warm-start imprints "enter whenever a label is active ≈ always."** The
supervised warm-start proposes a trade on 89% of 1m bars (`antifreeze_smoke.md:77`);
its prior fights entry *selectivity* early, and WARM's selectivity gap stays ~0
while COLD's reaches +0.21 (`:138-141`). Imprinting the *rate* (not just
direction) means the policy must first *unlearn* over-entry — and the only
lever it has to do so is C.1's "trade less," which overshoots toward flat.
**For:** WARM over-trades ep0 harder than COLD (3450 vs 1506). **Against:**
COLD (no warm-start) *also* collapses, so warm-start is an amplifier, not the
root. **Cheap experiment:** the COLD-vs-WARM arms already isolate this; extend
both to epoch ~8 and compare asymptotic trades/day.

**C.5 — TBPTT-500 horizon truncates exit credit.** Holds are in seconds/5s-bars;
median 3 bars tonight but p90 grows to 74–350 bars with training
(`antifreeze_smoke.md:155`). A ride that pays off beyond the 500-step TBPTT
window (`reference-mamba-ssm-wsl-perf.md:26`) gets its entry decision credited
without the downstream capture — biasing toward short churn. **For:** consistent
with 41% ≤2-bar churn. **Against:** 500 5s-bars = ~42 min ≫ median hold, so
most trades fit the window; only rare long rides truncate. **Cheap experiment:**
histogram hold-length vs 500; if ≪ 1% exceed it, deprioritize.

---

## D) Candidate levers (trade-offs only — NO recommendation)

1. **Opportunity-cost on flat-in-regime (asymmetric anti-inaction).** Regret
   already fires on missing a knowable fire (`reward_env.py:69-74`, gated on
   calibrated c_t per `THESIS:47`). *Lever:* raise w_r / widen the window so
   sitting out a readable setup costs more than a bad trade. *Trade-off:* over-
   tuned, it re-creates a gambler that enters junk to dodge regret — the exact
   pathology c_t-gating exists to prevent. Fights C.1 directly but risks C.2's
   silent-capture leaving *any* entry positive-EV vs regret.

2. **Entropy / exploration-bonus schedule.** The missing THESIS §7 term
   (`THESIS:52`). *Trade-off:* holds the action distribution open against C.1/C.3
   collapse, but a floor that's too high just re-funds the ep0 over-trade; it
   treats the *symptom* (distribution narrowing) not the *cause* (no positive
   per-trade reward).

3. **Reward-term rebalancing.** Lower w_cost or make cost *proportional* to
   trade quality instead of a flat −0.30 tax (`reward_env.py:84-85`); raise w_c
   / lower θ_rem so capture can actually pay (C.2). *Trade-off:* the flat cost
   tax is the honest per-trade friction (spread+commission+slip, cost_ticks 3.0);
   softening it to avoid freeze re-introduces the exact "trading looks free" bias
   that lets churn survive. One-change-at-a-time discipline applies.

4. **Curriculum with entry-forced episodes.** THESIS §8 already prescribes
   selectivity → direction → exit ordering (`THESIS:56-57`). *Lever:* seed early
   epochs with forced entries so the exit head gets gradient before the entry
   head is allowed to go flat. *Trade-off:* off-policy forcing biases the value
   function; must be annealed out cleanly or it imprints (cf. C.4).

5. **Teacher-distillation KL anchor on the exit head (sidesteps RL freeze).**
   The qwen3:14b teacher now emits **real p_exit soft-labels**
   (`2026-07-22.md:37-38`); the north-star path is soft-label distillation into
   the student (CLAUDE.md; `PRODUCTION_RUN_SPEC.md §8`). *Lever:* add a KL term
   anchoring the student's exit head to teacher p_exit, so the exit policy is
   *supervised toward a non-degenerate target* and cannot collapse to
   never-act — RL then only *refines* around it. *Trade-off:* the teacher's
   own effectiveness is unvalidated (morning agenda item #1, `2026-07-22.md:88-89`);
   anchoring to a weak teacher imports its errors, and the ctx-taint work
   (`2026-07-22.md:70-86`) means soft-labels aren't fully trustworthy yet.
   **But it is the only lever that removes the freeze attractor by construction
   rather than by tuning against it.**

6. **Asymmetric action costs.** Make HOLD-in-regime cost > HOLD-in-chop, or
   make the cut/reversal cheaper than a fresh entry. *Trade-off:* more magic
   knobs (each must become a named `RewardConfig` field per project rule); risks
   compounding with levers 1/3 in ways the one-change discipline can't isolate.

---

## E) Open questions for the owner (decision-shaped, max 5)

1. **Is the target a trade-RATE band (3–40/day) or a decision-QUALITY
   asymptote?** If capture stays ≈ 0 (C.2), pinning a rate band just picks
   where on the freeze curve to stop — it doesn't make trading pay. Which do we
   gate on?

2. **RL-refines-teacher, or RL-from-scratch?** Lever 5 (KL-anchor exit head to
   qwen p_exit) removes the freeze attractor structurally but couples us to an
   unvalidated teacher (agenda #1). Do we block anti-freeze on teacher
   effectiveness first, or run them in parallel?

3. **Do we extend the smoke to epoch ~8–10 before deciding anything?** The
   "no freeze" verdict is a 2-epoch snapshot; C.1 predicts the collapse
   continues to flat. One 4-day run settles whether freeze is truly gone or
   just slow. Cheapest possible discriminator — run it first?

4. **Flat cost tax vs quality-scaled cost:** keep the honest −0.30/trade
   friction (and add a positive per-trade reward to balance it), or fold trade
   quality into the cost term itself? These pull the same knob in opposite ways.

5. **How many levers at once?** Baseline discipline says one change per run.
   Anti-freeze plausibly needs entropy + capture-repair + regret balance
   *together*. Do we accept a multi-change run (and lose clean attribution), or
   serialize and spend the epochs?
