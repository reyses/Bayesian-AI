# Three Questions That Build a Tier

Discovered 2026-04-18 while rebuilding TREND_FOLLOWER (née FREIGHT_TRAIN) from
scratch. This is the working methodology for taking any tier from "noisy idea"
to "physics-grounded rule." It replaces the old habit of brute-force Cohen-d
sweeps and CART overfitting.

## The three questions (ask in order)

### Q1 — Are the entries the right ones?

**What it measures:** is the direction thesis correct at entry?

**How to answer:** peak-bucket analysis. For every trade in the tier, split by
peak PnL magnitude:

| Peak bucket | What it means |
|---|---|
| peak ≤ $0 | Direction wrong from bar 1 |
| peak $0–5 | Direction barely worked |
| peak $5–20 | Partial fade, then overwhelmed |
| peak > $20 | Direction solidly right |

**Pass criterion:** if >80% of trades have peak > $20, the direction is right.
You don't need filters; you need better exits. If <50%, the direction is
wrong — consider flipping.

**On TREND_FOLLOWER (2026-04-18):** 84.8% peak > $20. Direction validated.

### Q2 — What signal, if persistent for X bars, says we entered wrong?

**What it measures:** can we tell early that a trade is a loser?

**How to answer:** bar-by-bar analysis of the trade path. For winners vs
losers, track `peak_pnl_by_bar_N` and find where the gap opens.

| Bar N | Winners: no +$5 yet | Losers: no +$5 yet | Gap |
|---|---:|---:|---:|
| 60 (10 min) | 20.6% | 40.9% | 20pp ← cut here |

**Pass criterion:** if the gap is >15pp at some bar N, we have a "no-progress
kill" signal: exit if `peak_pnl < $5` at bar N.

**Physics framing (not a safety):** the tier's thesis implies a timescale —
fades revert in minutes, trends unfold in hours. If price hasn't moved our
way in the tier's characteristic timescale, the thesis is false. Exit.

**On TREND_FOLLOWER (2026-04-18):** if peak_pnl < $5 by bar 60 (10 min),
we're in the loser distribution. Noted, not yet implemented.

### Q3 — What do all the peaks have in common?

**What it measures:** what feature state defines the exit (thesis complete)?

**How to answer:** for each trade, find the peak bar. Take the feature
vector AT that bar. Subtract the entry feature vector. Normalize by
per-feature entry std dev to get comparable magnitudes. Rank features by
|delta / sigma|.

The feature with the largest |d/σ| IS the exit signal.

Supporting physics come from the 2nd and 3rd rankings — prefer ones whose
entry condition has a natural inverse (e.g., if entry requires `vr > 1`,
the exit requiring `vr < 1` is free symmetry).

**Pass criterion:** if the top feature has |d/σ| > 5, the exit rule is
dominated by one signal — very clean. If top is <2, the peak is muddy
and the rule is weaker.

**On TREND_FOLLOWER (2026-04-18):**
- `1m_p_at_center` d/σ = +10.32 (THE signal)
- `1m_variance_ratio` d/σ = -1.81 (inverse of entry gate `vr > 1`)
- `1m_reversion_prob` d/σ = +1.16 (OU statistical confirmation)

Three orthogonal physics dimensions (position, regime, probability), all
firing together at peak. Exit rule: all three must fire.

## Why these three questions in this order

- Q1 answers whether the entry is worth keeping at all. If it's random
  direction, no amount of exit work helps.
- Q2 catches the losers early. Even a good-direction tier has a tail of
  trades that go wrong fast. This is the "eject if thesis clearly
  broken" rule.
- Q3 captures the winners at the right price. A good-direction tier
  held too long gives back; this rule exits at the correct moment.

Together they produce:
- An entry filter (optional, from Q1 if direction is imperfect)
- An early-exit rule (Q2, "thesis violated")
- A peak-arrival exit (Q3, "thesis complete")

## Anti-patterns to avoid (learned painfully)

- **Don't start with CART or ML** on winners vs losers. We did this; it
  overfit, and we discovered feature leakage in the corrected trades.
  The three-question method is blunt but honest.
- **Don't add more than 3 confirming features** to an exit rule. Each AND
  condition multiplicatively reduces trigger rate. 3 features that each
  fire 70% at peak → ~34% combined trigger. 4 features → ~24%. You
  start missing real peaks.
- **Don't propose ALL of Q1+Q2+Q3 in one code change.** Build in stages,
  run, measure, move on. Each question produces ONE rule; land that rule
  before the next one.
- **Don't skip Q1.** If the direction is wrong 50%+ of the time, Q2 and
  Q3 are about exit timing within a random signal. You'll get marginal
  lift that evaporates OOS.

## Tooling

- `tools/tier_eda.py --tier NAME` runs the Q1 segment/separator/regime-shift
  analysis on a tier. Writes markdown to `reports/findings/`.
- Ad-hoc Python for Q2 (path PnL trajectories) and Q3 (entry→peak feature
  deltas). The EDA pattern is consistent: load `iso_is.pkl`, filter by
  `entry_tier`, compute, print.

## Applied to TREND_FOLLOWER — full chain

| Question | Answer | Rule |
|---|---|---|
| Q1: Entries right? | 84.8% peak > $20. Yes. | Keep entries. |
| Q2: Wrong signal? | If peak_pnl < $5 at bar 60, more likely loser. | (noted, deferred) |
| Q3: Peak signature? | p_center > 0.35 AND reversion > 0.80 AND vr < 1 | Primary exit rule |

Result expected to ship: TREND_FOLLOWER goes from -$4/trade to positive.
Subsequent tiers (RIDE_AGAINST, KILL_SHOT, etc.) will follow the same
three-question chain.
