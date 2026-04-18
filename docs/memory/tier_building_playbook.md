# Tier Building Playbook — Consolidated Methodology

**Last updated: 2026-04-18.** This is the working methodology for building,
fixing, and iterating trading tiers in the Bayesian-AI isolated pipeline.
Supersedes `feedback_tier_three_questions.md` (now a subsection here).

The playbook is organized by the order you use it: data integrity first,
then EDA, then entries, then exits, then the anti-patterns that burn time.

---

## 1. Data Integrity Checklist (do this before anything)

Before running any EDA, verify data is honest. **Most "edge" in backtests
is bugs.** The 2026-04-17 lookahead fix alone turned +$740/day into
-$164/day. Every analysis below assumes these checks pass:

- **Lookahead bias**: `searchsorted(timestamps, target_ts, side='right') - 1`
  must subtract the period before lookup for higher-TF aggregation.
  Anything that peeks at in-progress bars contaminates features.
- **Clean price data**: phantom spikes in NT8 data cost $4,350 of fake
  edge on one analysis (2026-04-03). Use Databento for IS, NT8 only for
  OOS-2 live parity. Validate: max price jump, contract continuity, zero-volume gaps.
- **Feature parity**: training features must match live-engine features
  cell-by-cell (target: 100% parity). On 2026-04-14 we moved from 7.5%
  parity to 100% by switching to `LiveFeatureEngine`. Check
  `training/live_feature_engine.py`.
- **SFE cache**: cache key must include `(valid_idx, latest_bar_ts)` not
  just `valid_idx` (2026-04-16 frozen features bug — collision after
  5000-bar trim left stale state since mid-Feb).
- **Regret LOOKAHEAD cap**: `training_iso/regret.py` defaults to 6h
  counterfactual — use 10min-before-entry / 30min-after-exit bounds.

When in doubt, run `tools/validate_data.py` before touching a tier.

---

## 2. The Three-Question Method (core tier-fixing loop)

Given an iso trade pickle (`training_iso/output/trades/iso_is.pkl`) tagged
with `entry_tier`, ask three questions in order. Each produces at most one
rule. Measure between rules — don't stack blindly.

### Q1 — Are the entries right?

**Method:** peak-bucket analysis. Split tier trades by `peak` (the max
PnL reached during the trade, in $):
- `peak <= 0`: direction wrong from bar 1
- `0 < peak <= 5`: barely worked
- `5 < peak <= 20`: partial success
- `peak > 20`: direction solidly right

**Decision tree:**
- **>80% peak > $20**: direction is correct. Move to Q3.
- **<50%**: direction is inverted. **Flip direction** in the tier's
  fire function (`return 'short' if z > 0 else 'long'` becomes the
  opposite). Rerun single-tier isolated pipeline. THEN move to Q3.
- **50-80%**: mixed. Don't flip yet. Move to Q3; reconsider later.

**Observed examples:**
| Tier | % peak > $20 | Decision |
|---|---:|---|
| TREND_FOLLOWER | 84.8% | Solid — kept direction |
| RIDE_AGAINST (ride dir) | 66.5% | Mixed; later flip still helped |
| KILL_SHOT | 62.4% | Mixed — kept |
| MTF_BREAKOUT | 66.9% | Mixed |
| NMP_FADE | 52.7% | Noise (catch-all) |

### Q2 — Hold-time cliff (natural timescale)

**Method:** bucket trades by held minutes, report WR and $/trade per
bucket. Look for the boundary where buckets flip sign.

**Observed natural timescales:**
| Tier | Cliff | Interpretation |
|---|---:|---|
| TREND_FOLLOWER | 60 min | Fade thesis timescale |
| RIDE_AGAINST | 15 min | 1h-vel reversal plays out fast or not at all |
| KILL_SHOT | 30 min | Wick-rejection intermediate |
| MTF_BREAKOUT | (inverted) | Trend tier — 300+min is BEST bucket |
| NMP_FADE | N/A | No tier-specific timeout helps |

**Rule derivation:** set `max_hold_min = <cliff>`. Exit trades past that
threshold as `<tier>_timeout`.

**Don't apply universally.** Catch-all tiers (NMP_FADE) and trend tiers
(MTF_BREAKOUT) are harmed by timeouts.

### Q3 — Peak signature (the universal exit rule)

**Method:** for each trade, find the bar where `peak_pnl` is maximum.
Extract the feature vector at that bar. Compute `delta = feat_peak -
feat_entry` per feature. Normalize by entry std dev. Rank by `|d/σ|`.

**The universal result** (confirmed on 4 tiers so far):
| Feature | d/σ range | Meaning |
|---|---:|---|
| `1m_p_at_center` | +9.9 to +12.8 | Price at regression mean |
| `1m_reversion_prob` | +0.8 to +1.2 | OU model expects reversion |
| `1m_variance_ratio` | -0.5 to -1.8 | Chaotic regime settled |

**Universal rule of three:**
```
Exit when:
  1m_p_at_center      > 0.35
  AND 1m_reversion_prob > 0.80
  AND 1m_variance_ratio < 1.0
```
**Plus amplitude gate** (ALWAYS needed): `peak_pnl >= $10`. Without it
the rule fires on 90%+ of trades at tiny peaks, giving back everything.

**When to skip the peak rule** (negative findings):
- **NMP_FADE** (catch-all, mean winner peak $69): early peak-exit
  shortens winners more than saving losers. Inverse-only is best.
- **MTF_BREAKOUT** (trend tier): 254 "never worked" trades bleed to
  inverse at -$82/trade, eating the $21K peak wins. Also best with
  inverse-only.

**Rule of thumb:** if winners' mean peak is under ~$80, peak rule
probably hurts. Tiers with mean winner peak > $130 benefit (CASCADE
+$225, NMP_RIDE +$X, RIDE_AGAINST pre-flip +$213, TREND_FOLLOWER after
flip).

---

## 3. Phantom Entry (the fizzle protector)

**The idea (2026-04-06 original, 2026-04-18 reimplemented).** When tier
conditions fire, don't enter — create a PENDING signal and watch. Enter
only if price moves `N` ticks in our direction within `M` minutes.
Cancel otherwise.

**Why it works**: sets with the conditions that fire split into
- **Will confirm quickly**: setup is working, enter at slightly worse
  price, keep most of the edge
- **Won't confirm**: fizzle — skip entirely, skip the resulting loss

The 2-tick/3-tick/4-tick filter CUTS LOSERS MORE than it cuts winners
because losers typically don't move favorably at all — the "confirmation"
is exactly the thing losers fail to do.

**Observed optimum on RIDE_AGAINST**: 4-tick confirm / 2-minute window.
Sweep showed 2→3→4 ticks progressively improved; 5+ eroded winners.

**Implementation**: `self._pending` slot on the engine. In `on_state`,
before entry check, update pending. If confirmed → enter at CURRENT
price. If expired → clear. See `nightmare_iso.py`.

---

## 4. Entry Relaxation Principle (the counter-intuitive one)

**Once phantom is in place, relax other entry filters. Phantom does the
fizzle work — feature-based filters become redundant or over-restrictive.**

Observed on RIDE_AGAINST:
- Remove `5m_bar_range > 55`: +$498 (was tuned for ride direction)
- Remove `h1_against_fade` tier-overlap guard: +$1,444
- Lower `h1_vel > 3` to `> 2`: +$498
- Tighten phantom `> 3 → 4 ticks`: +$52

**Principle:** with phantom catching the market-level confirmation,
feature filters that attempt to PREDICT the same thing (will this trade
work?) become noise. Keep only filters that define WHAT tier this is
(the physics setup), not WHETHER it will work.

**Watch out for over-filtering signatures:**
- Peak-rule $/trade per-exit is high (+$20+) but total tier is flat/neg
- Trade count is low relative to tier's natural universe
- A sweep of any filter threshold shows negative gradient (looser = better)

---

## 5. Direction Flip (the 38%-WR tell)

**If a tier's WR is significantly below 50%, flip the direction first.**
The prior that the 91D feature set predicts direction at entry with >58%
is BROKEN (2026-04-17 CART experiment, OOS acc 46.5% = worse than majority
baseline). Most of what looks like "direction signal" is noise.

BUT — persistent <45% WR across thousands of trades IS a signal: the
tier's direction RULE is systematically wrong. Flip.

**Observed:**
| Tier | Pre-flip WR | Post-flip WR | Status |
|---|---:|---:|---|
| RIDE_AGAINST | 38% | 65% | Flipped 2026-04-18 |
| FREIGHT_TRAIN | 20% | (nuked) | 5 trades, no stat power |

**Not all low-WR tiers want flipping** — sometimes they just have bad
exits (winners give back). Check peak-bucket first: if >80% had peak
> $20, direction's fine, fix exits.

---

## 6. Chain Positions (multi-entry per tier)

**Feature (2026-04-18):** each tier engine can hold up to `max_chains`
concurrent positions. When tier fires again in SAME direction while
already in position and under cap → open chain.

**Observed effect at chains=4** (full engine, no tier-specific tuning):
- NMP_RIDE: 284 → 967 trades (+$26,808)
- NMP_FADE: 6,070 → 18,218 (+$11,960)
- CASCADE: 112 → 199 (+$4,306)
- MTF_BREAKOUT: 866 → 1,611 (+$4,607)
- **Engine total: $21,090 → $73,760 (+$43,448 = +$157/day)**

**Chain-hurt tiers:** TREND_FOLLOWER (-$3,813), MTF_EXHAUSTION (-$3,679).
Peak-based tiers that work on "first entry at extreme" — later chains
enter at worse prices.

**For tier FIXING**, run `chains=1` (isolated single-position) so per-tier
WR measurements are honest. Chains are a separate multiplier to apply at
the end.

---

## 7. Five EDA Questions Beyond the Three

From journal accumulation; useful for deeper dives.

### 7a. Peak-reacher vs non-reacher separators

For tiers where peak rule fires on 60-70% of trades (rest go to timeout/
inverse at a loss), split by "reached $10 peak" vs "didn't" and Cohen-d
entry features. The separators are candidate ENTRY FILTERS. (Note: post-
2026-04-18, phantom entry may subsume this; try phantom first.)

### 7b. Higher-TF state at entry (2026-04-11 finding)

Universal pattern across tiers: **winners enter when higher TFs are calm
or aligned; losers enter when higher TFs are racing against them.** Check
`5m_velocity`, `1h_velocity`, `1h_z` in lookback window pre-entry.

### 7c. Resonance Cascade (2026-04-06 finding)

Multi-TF alignment amplifies edge:
- 1m z extreme = "enter"
- 5m/15m wick rejection = "confirmed"
- 1h z aligned = "amplified"

Base KILL_SHOT: 96% WR $16/tr. + 1h alignment |hz|>1.5: 97% WR **$24/tr**.
Each TF layer adds confidence AND $/trade.

### 7d. Chop is universal loser signal (2026-04-14 finding)

Top Cohen-d across 89K trades:
- `15m_wick_ratio` d = -0.27 (high wick = chop = loss)
- `1h_wick_ratio` d = -0.26
- Per-tier chop-d: RIDE_AGAINST -0.43, FADE_CALM -0.36, FADE_AGAINST -0.42

**High wick_ratio at higher TFs = caution.** Consider as a filter.

### 7e. Gravity well (2026-04-14 finding)

`z_high` / `z_low` are not just historical extremes — they're **attractor
wells** that pull price toward the regression mean. Deeper historical
extreme = stronger gravity.
- Long near 1h floor = trading WITH gravity (good)
- Long near 1h ceiling = trading AGAINST gravity (bad)

**Feature to watch**: `15m_z_high` / `15m_z_low` — winners enter with
deeper highs in their direction (d ≈ +0.08, weak but present).

---

## 8. Exit Physics Beyond the Peak Rule

### 8a. Breakeven lifespan (2026-04-06)

For kill-shot / fade trades with peak > $5:
- 43-59% of trades NEVER break even — they're permanent moves
- Peak takes 5-7 hours on those
- Lifespan (time at profit before reversion) median 50-85 min

**Implication:** trailing stops at modest giveback preserve most $.

### 8b. Trailing stop optimization (2026-04-06)

| Giveback | $/trade | WR | Peak Capture |
|---|---:|---:|---:|
| 10% | $19.6 | 98% | 74% |
| 25% | $20.6 | 98% | 62% |
| 70% | $25.5 | 93% | 35% |

Sweet spot: 10-25% giveback. User stripped trail mechanic 2026-04-18
in favor of peak rule + timeout — but a soft trail might be worth
re-adding for tiers without a clean peak signature.

### 8c. Winner MAE (2026-04-16)

- **97% of winners DIP NEGATIVE FIRST** before becoming winners
- Median dip: -$8.50
- 90% of winners recover from dips ≤ -$17
- Hard stops at -$50 kill 25% of winners; at -$150 kill 0.2%

**Rule:** don't stop winners early. Patience is an edge. No hard stop
above -$150 unless physics says exit.

### 8d. 5m alignment exit patience (2026-04-09)

Split trades by 5m velocity at entry:
- 5m WITH fade: 549 trades, **85% WR**, $35.5/trade
- 5m AGAINST fade: 3,801 trades, 63% WR, $14.5/trade

Use as EXIT PATIENCE (not entry filter):
- Aligned → hold longer (high confidence reversion)
- Opposed → exit faster (lower confidence)

### 8e. Thesis-validity (today)

Exit when the entry condition DECAYS below its threshold (not when the
opposite extreme fires). Examples:
- RIDE_AGAINST: h1_vel flipped sign → thesis gone (sign flip, not wait for |h1_vel|>3 on other side)
- TREND_FOLLOWER: |1m_vel| drops below entry threshold → momentum dead
- KILL_SHOT: 5m_wick decays → wick rejection gone

Note: direction-of-thesis matters. For fades (RIDE_AGAINST post-flip),
h1_vel decay is FAVORABLE, not unfavorable — thesis-validity must be
defined against pos direction, not entry conditions blindly.

---

## 9. Anti-Patterns (burned time in these)

### 9a. Don't trust $740/day baselines

The nn_v2 blended $740/day baseline (2026-04-09) was **lookahead-inflated**.
Post-fix 2026-04-17: same pipeline → -$164/day. Any "historical baseline"
before 2026-04-17 should be considered contaminated unless the feature
file timestamp is post-lookahead-fix.

### 9b. Don't start with CART on winners vs losers

We tried (2026-04-17). CART found `1m_reversion_prob > 0.866` = 86%
oracle capture, looked clean... Turns out `regret.py::correct_trades`
replaces `entry_79d` with approach-buffer features (99.7% of trades).
The CART learned FEATURE LEAKAGE, not signal. After joining to actual
iso entry features, OOS accuracy dropped to 46.5% (worse than majority).

**Use the three-question method (blunt but honest).** Only CART after
you've verified data isn't polluted.

### 9c. Direction at entry is RANDOM on 91D

Hard-proven on 2026-04-17 regime-discovery work. Day-stratified CART
can't beat majority baseline. Don't chase "counter-flip" tiers without
a physics reason (like a mechanically-flipped direction rule).

### 9d. Don't use time-of-day filters

Lazy. Hour-of-day correlates with the real physics cause (liquidity,
session chop), but the filter doesn't generalize. Find the FEATURE that
spikes in those hours and filter on THAT. The 2026-04-17 user note:
"time filter admits we don't understand the physics."

### 9e. Peak rule isn't universal

Applied blindly it hurts catch-all tiers (NMP_FADE, MTF_BREAKOUT).
Rule: if winner mean peak < ~$80, peak rule probably costs more than
it saves. Stick to inverse-signal.

### 9e-bis. Phantom entry isn't universal either (2026-04-18 discovery)

Phantom + short timeout (≤15m) = safe. RIDE_AGAINST had 15m timeout,
phantom was a huge win. MTF_BREAKOUT had no timeout, phantom + inverse
exit helped (+$1,020).

Phantom + long timeout (60m+) = dangerous. TREND_FOLLOWER has 60m
timeout. Adding phantom went +$495 → **-$1,814** (-$2,309 swing).
When phantom-confirmed trades fail, the 60m timeout lets losses bleed
for the full window: 80 trades at **-$166/tr** = -$13K disaster.

**Pattern:** phantom entry trades off worse entry price for fizzle
protection. Works when timeout is short enough to cap the downside
quickly. Fails when slow exit lets the worse-entry-price compound.

**Rule:** if tier's natural timescale is > 15-20 minutes, DON'T add
phantom unless you also shorten the timeout.

### 9f. Timeout isn't universal

KILL_SHOT had bimodal holds pre-peak-rule (0-10m wins, 300+min wins,
60-300m losers). A timeout between 30-300m would cut winners.
Always do Q2 (hold-time bucket) before imposing a timeout.

### 9g. Don't skip negative findings

When a rule DOESN'T help, that's valuable information. Record it in
the journal with the data that proves it. Future sessions will ask
"why don't we do X?" and the journal answers.

---

## 10. Tools Reference

Living tools, updated 2026-04-18:

- **`tools/tier_eda.py --tier NAME`** — parameterized Q1/Q3 analysis with
  segment breakdown, feature separators, regime shift. Writes markdown
  to `reports/findings/tier_eda_<TIER>_<ts>.md`.
- **`tools/daily_hourly_pnl.py`** — mode/WR/hourly breakdown for revenue
  framing. Target metric: $/active-hour, not $/day total.
- **`tools/path_features_eda.py`** — 12-bar 5s path metrics (efficiency,
  R², reversal count) on pre-entry window. Found nothing stronger than
  `1m_bar_range` already in 91D (deprecated but retained).
- **`tools/slope_eda.py`** — OLS β+γ on 30-bar windows per TF. The β
  (velocity of regression mean) was d=-0.31 moderate signal for
  TREND_FOLLOWER. Infra available via `IsoEngine._slope_1m(ts)` but not
  yet consumed.
- **Ad-hoc Python** for Q2 (path pnl trajectories) and Q3 (entry→peak
  feature deltas). Pattern is consistent: load `iso_is.pkl`, filter by
  `entry_tier`, compute, print.

---

## 11. Current Tier Scorecard (2026-04-18 end of day)

With all current rules, chains=1:

| Tier | N | WR | $/trade | $/day | Status |
|---|---:|---:|---:|---:|---|
| NMP_RIDE | 299 | 51% | +$40.01 | +$43 | ✓ Winner, keep |
| FADE_AGAINST | 326 | 47% | +$23.19 | +$27 | ✓ Winner, keep |
| MTF_EXHAUSTION | 125 | 47% | +$23.12 | +$10 | ✓ Winner, small sample |
| CASCADE | 113 | 55% | +$42.84 | +$17 | ✓ Winner, small sample |
| NMP_FADE | 8,174 | 55% | +$1.36 | +$40 | ✓ Workhorse |
| RIDE_AGAINST | 4,204 | 65% | +$0.70 | +$11 | ✓ Fixed today |
| TREND_FOLLOWER | 780 | 68% | +$0.82 | +$2 | ✓ Fixed earlier |
| KILL_SHOT | 519 | 67% | -$1.36 | -$3 | ~ Break-even |
| MTF_BREAKOUT | 943 | 40% | -$1.18 | -$4 | ⚠ Marginal (trend tier) |
| **Engine total** | **14,683** | — | — | **+$143** | — |

Pre-session baseline: $76/day. Session lift: **+$67/day**.

Target: $60/active-hour = ~$300/day. Still work to do — likely via
applying phantom entry pattern to other tiers + chain multiplier.
