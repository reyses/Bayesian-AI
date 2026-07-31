# ±N BRACKET STRATEGY — VERDICT: STRUCTURAL LOSER AT EVERY WIDTH TESTED

**Date**: 2026-07-30 · **Tool**: `research/dojo_forge/tools/bracket_test.py`
**Data**: 60 days, 5s bars, `DATA/ATLAS/5s` · **Friction**: 0.89pt/round-trip

## The strategy under test (owner's definition, verbatim mechanics)

- stop loss at **−N** points → **CLOSE and FLIP** to the opposite direction
- take profit at **+N** points → **CLOSE and REENTER** the same direction
- both re-armed at ±N around each new entry; **always in the market**

## Result

| width | n | stop% | net $ | $/trade | 95% CI | significant | Trade WR (PF-based) | median hold |
|---|---|---|---|---|---|---|---|---|
| ±5  | 161,624 | 48.2% | −229,771 | **−1.42** | [−1.47, −1.37] | **YES** | −0.250 | 10s |
| ±10 | 49,603 | 50.2% | −92,593 | **−1.87** | [−2.04, −1.69] | **YES** | −0.171 | 30s |
| ±15 | 22,803 | 50.8% | −52,079 | **−2.28** | [−2.67, −1.90] | **YES** | −0.141 | 60s |
| ±20 | 12,879 | 51.4% | −37,045 | **−2.88** | [−3.58, −2.17] | **YES** | −0.134 | 105s |
| ±30 | 5,755 | 51.9% | −23,264 | **−4.04** | [−5.61, −2.58] | **YES** | −0.126 | 235s |
| ±50 | 2,021 | 50.6% | −5,897 | −2.92 | [−7.47, +1.34] | no | −0.057 | 715s |

**Every width loses. All but ±50 are statistically significant losers** (±50's CI
includes zero only because n=2,021 is the smallest sample, not because it works).

## Why — the mechanism is exact, not hand-waving

**Stop fraction sits at 48–52% across every width.** The flip is a coin.

For a symmetric ±N bracket with stop-fraction `p`, expected points per trade is:

```
EV = −p·(N + f) + (1−p)·(N − f)  =  N·(1 − 2p) − f          (f = 0.89pt friction)
```

Checked against the ±30 row: p=0.519, N=30 → `30·(1−1.038) − 0.89 = −2.03pt = −$4.06/trade`.
Measured: **−$4.04**. The model reproduces the data.

Two consequences fall straight out of that formula:

1. **There is no width that fixes this.** The `N·(1−2p)` term is *negative* whenever
   p > 0.5, and it scales **with N** — so widening the bracket *amplifies* the
   adverse skew rather than escaping friction. That is exactly what the table
   shows: $/trade gets monotonically **worse** from −1.42 (±5) to −4.04 (±30).
2. **Narrowing doesn't fix it either** — it just trades a smaller per-trade loss
   for far more trades. ±5 has the best $/trade of the losers but the worst
   *total* (−$229,771 over 60 days) because it pays friction 161,624 times.

**Why p > 0.5**: on a stop you FLIP, i.e. you bet on *continuation* after an
adverse move. The data says continuation-after-an-adverse-break is slightly
*worse* than a coin — the whipsaw/mean-reversion signature. The flip is
systematically on the wrong side of that.

## Correction to earlier advice

During the live session I told the owner the ±10 bracket looked "narrower than
bar-to-bar noise" and that a **wider bracket** was worth considering. **That was
wrong**, and the data above is what corrects it: widening makes per-trade results
*worse*, not better. The problem was never the width — it is that the flip has no
edge, and any symmetric bracket on a ~coin-flip trigger is a friction pump.
The live sample that prompted the suggestion (4 consecutive stop-flips) was n=4:
a real observation, but far too small to support the inference I drew from it.

## What this does NOT say

- It does not condemn stop-and-reverse as a *discretionary* tool applied at a
  chosen moment. This tests the **mechanical, always-in-market, re-armed-forever**
  version. A human arming one stop-reverse at a spot they've read is a different
  claim, untested here.
- ~~It does not test asymmetric brackets... asymmetry is the only lever with a
  chance~~ — **SUPERSEDED, see addendum below.** The owner corrected this same
  day: the +N leg reenters rather than exits, so it is a ratchet, not a
  take-profit, and the "payoff ratio" reasoning does not apply. Optional
  stopping then kills asymmetry outright — gross EV is identically zero for
  every (S,T) pair.
- 60 days is one regime slice. The stop-fraction stability (48–52% across six
  widths and three orders of magnitude of sample size) argues it generalizes,
  but that is an argument, not a measurement.

## Method notes

- Runs on **5s bars, not 1m**: with a ±N bracket both sides can be touched inside
  one 1m bar, and the 1m aggregate cannot say which came first — that ambiguity
  would silently decide trades. 5s mostly removes it.
- Residual same-5s-bar ties (2,387 across all widths) are resolved
  **adverse-first** (conservative) and **counted**, never silently resolved.
- Metrics follow the project standard: PF-based Trade WR (not count-based),
  4,000-resample bootstrap CI, N always reported.

---

# ADDENDUM (same day) — asymmetry is dead too, and the +2 breakeven stop tested

## Owner correction #1: the +N "target" is not a take-profit at all

Owner: *"asymetric is the same cuz its +10 and continue, so the only thing it
does it reduces the amount of continue."* **Correct, and it invalidates the
"payoff ratio" framing used above.** A target hit closes and **reenters the
same direction** — market exposure is unchanged across the event. The target
is therefore not an exit; it is a **stop-ratchet that costs friction per click**.
Widening the target just ratchets less often.

## Asymmetry cannot work either — optional stopping

For a driftless walk, `P(hit −S before +T) = T/(S+T)`, so gross EV is:

```
EV = −S·[T/(S+T)] + T·[S/(S+T)] = 0     for EVERY (S,T) pair
```

Verified numerically across (10,10), (10,20), (10,30), (10,50), (5,50), (20,10):
**gross EV is identically zero in all of them.** Net = −friction, always.
So the "asymmetry is the one untested lever with a chance" suggestion in the
main verdict above was **also wrong** — no bracket *geometry* can create edge
on a martingale. Second correction from the owner on this thread; both stood.

## Owner proposal: the +2 breakeven stop

*"probsably the best option is the +2 stop, this works to make sure we dont
lose money."* Mechanic: once a trade is far enough in profit, move the stop to
entry+2 so the worst case becomes a small win instead of a loss.

Tested (`research/dojo_forge/tools/breakeven_test.py`): 60 days, 12,496 trades,
independent sampled entries both directions, initial stop −10, horizon 30min.
**Strict A/B — identical entries, identical initial stop, identical horizon;
the only difference is whether the breakeven move is armed**, so the delta is
attributable and gets a paired bootstrap CI.

| config | n | $/trade | 95% CI | % losing | Δ vs control | Δ CI | sig |
|---|---|---|---|---|---|---|---|
| CONTROL fixed −10 | 12,496 | −2.714 | [−3.49,−1.88] | **78.2%** | — | — | — |
| BE +2 armed at +5 | 12,496 | **−2.096** | [−2.52,−1.64] | **33.4%** | +0.619 | [−0.08,+1.28] | **no** |
| BE +2 armed at +10 | 12,496 | −2.525 | [−3.09,−1.92] | 50.1% | +0.190 | [−0.40,+0.73] | no |
| BE +2 armed at +20 | 12,496 | −2.683 | [−3.41,−1.97] | 65.9% | +0.031 | [−0.34,+0.39] | no |

**The owner's claim is correct on its own terms**: losing trades drop from
**78.2% → 33.4%**. That is the single largest effect anything tested today has
produced, and it was predicted from intuition before measurement.

**But it does not make money.** Best variant is still **−$2.10/trade**, and the
+$0.62 improvement has CI **[−0.08, +1.28]** — includes zero, **not significant**.

**Mechanism of the offset** (outcome mix): trades surviving to the horizon
collapse **24.1% → 5.0%** when the breakeven stop is armed at +5. The rule
converts big-winner potential into +2 crumbs (+$2.22 each). Saved losers and
truncated winners roughly cancel — which is exactly what optional stopping
predicts: **no stopping rule can manufacture edge from a martingale.**

## The caveat that matters most

This tests management on **random entries**. Management rules cannot create
edge where none exists — so a null here is unsurprising and is *not* evidence
against the owner's discretionary process. But it cuts the other way too:
**on entries that genuinely carry edge, truncating winners costs MORE, not
less.** A breakeven stop is a seatbelt, not an engine. If the entry read is
where the edge lives (the whole premise of the dojo program), then capping
those trades at +2 is precisely the wrong place to economise.

**Untested and worth more than any bracket geometry**: the same A/B run on
*owner-selected* entries from the dojo corpus rather than random ones — that
is the only version of this question whose answer could change a decision.

---

# ADDENDUM 2 — I scored the safety rule on the wrong objective

Owner: *"no it is a safety need, if you look in the useage was 'im unsure that
this will play out' arm +2 to protect current equity."*

**The correction**: the +2 stop was proposed as a **conditional risk control**
— armed when the owner's *confidence drops mid-trade* — not as a universal
alpha rule. Addendum 1 judged it by **EV**, which is the wrong objective
function for a safety device. Worse, that framing inverted the reading: for a
seatbelt, **EV-neutrality is the ideal outcome**, because it means the
protection is *free*. I reported "not significant" as a failure when it was
the best possible news for the actual purpose.

## Re-scored on risk (same 12,496 trades, same strict A/B)

| config | %losers | std $ | p05 | p01 | CVaR5 | worst | max DD |
|---|---|---|---|---|---|---|---|
| CONTROL fixed −10 | 78.2% | **46.44** | −21.78 | −21.78 | −21.78 | −21.78 | **34,692** |
| **BE +2 armed at +5** | **33.4%** | **25.19** | −21.78 | −21.78 | −21.78 | −21.78 | **26,283** |
| BE +2 armed at +10 | 50.1% | 33.68 | −21.78 | −21.78 | −21.78 | −21.78 | 32,416 |
| BE +2 armed at +20 | 65.9% | 41.48 | −21.78 | −21.78 | −21.78 | −21.78 | 34,712 |

**Verdict for its actual purpose: the rule works.**
- volatility **−46%** ($46.44 → $25.19)
- max drawdown **−24%** ($34,692 → $26,283)
- losing trades **−57%** (78.2% → 33.4%)
- EV cost: **statistically zero** (+$0.62, CI [−0.08, +1.28])

Half the volatility and a quarter less drawdown, bought for no measurable
expectancy. That is a good risk trade by any standard.

## The one place it does NOT help — stated plainly

**Tail risk is unchanged**: p05, p01, CVaR5 and worst-case are all **−$21.78 in
every arm**, identical. The breakeven stop can never protect a trade that goes
straight against you, because it never gets armed. **The −10 initial stop is
what caps catastrophe; the +2 smooths the curve.** Two different jobs — keep
both, and don't let the +2 create false confidence about the cliff.

## Practical implication

Halving volatility at zero EV cost **roughly doubles Sharpe on any entry that
carries genuine edge** (Sharpe = E/σ; E unchanged, σ halved → 2×). Since the
premise of the dojo program is that the *entry read* is where the edge lives,
this rule is worth **more** on the owner's real entries than it scored here on
random ones. The random-entry test understates it, and understates it in a
knowable direction.

## Methodological lesson (for me)

Ask what objective a proposal is optimising **before** choosing the metric. A
risk control scored on EV will always look like a null; an alpha rule scored on
variance will always look like a win. Getting the objective wrong made me
report a working tool as a failure — the owner had to correct the framing twice
before the right measurement got run.

---

# ADDENDUM 3 — owner: "if it goes straight against it should trigger immediately a stop"

Addendum 2 found the tail **unchanged** by the +2 breakeven rule (worst case
−$21.78 in every arm) because the +2 never arms on a trade that goes straight
against you. Owner's response went at exactly that gap: a trade that never
goes your way shouldn't be getting −10 of rope in the first place.

**Tested**: initial-stop sweep, 60 days, 12,496 trades per config, BE +2 armed
at +5, 30min horizon.

| init stop | BE? | $/trade | 95% CI | %losers | std $ | CVaR5 | worst $ | max DD |
|---|---|---|---|---|---|---|---|---|
| −2 | no | −2.129 | [−2.47,−1.75] | 95.2% | 20.73 | −5.78 | −5.78 | 27,636 |
| **−2** | **yes** | **−1.837** | **[−2.08,−1.56]** | 71.0% | **14.84** | **−5.78** | **−5.78** | **23,625** |
| −3 | yes | −1.900 | [−2.18,−1.58] | 62.3% | 17.25 | −7.78 | −7.78 | 24,196 |
| −5 | yes | −2.110 | [−2.44,−1.75] | 50.2% | 20.11 | −11.78 | −11.78 | 26,749 |
| −10 | yes | −2.096 | [−2.52,−1.64] | 33.4% | 25.19 | −21.78 | −21.78 | 26,283 |
| −10 | no | −2.714 | [−3.49,−1.88] | 78.2% | 46.44 | −21.78 | −21.78 | 34,692 |

**The owner's instinct is correct and the effect is large.** `−2 stop + BE+2`
is best on *every* risk metric tested **and** carries the best EV point
estimate of all eight configurations:

- worst case **−$5.78 vs −$21.78** — the tail Addendum 2 called unfixable, cut **73%**
- volatility **14.84 vs 46.44** vs the original control — **−68%**
- max drawdown **23,625 vs 34,692** — **−32%**
- EV: CIs overlap the control's throughout → **statistically unchanged**, and
  certainly not worse

**The price**: losing trades jump **33.4% → 71.0%**. Same money, radically
different to sit through — many small cuts instead of a few large ones. That is
a psychological cost, not a statistical one, but it is real and it is the main
argument against.

## What this test structurally CANNOT answer

On **random** entries a tight stop costs nothing, because there is no edge to
be shaken out of — gross EV is zero by construction, so premature exits are
free. On entries that carry **genuine edge**, a −2 stop may cut good trades
before the edge resolves, and this test is **blind to that by design**.

This also **tempers the Sharpe claim in Addendum 2**: "halving σ at zero EV cost
doubles Sharpe" holds *only if* EV is genuinely unchanged on edged entries —
which the random-entry harness cannot verify. The risk *benefit* here is
measured and solid; the EV *cost* on real entries is unmeasured. Do not treat
the net as established.

**The decisive experiment remains the same one named in Addendum 1**: replay
these stop rules over **owner-selected dojo-corpus entries** rather than random
ones. Until then this says how to shape risk, not whether it pays.

---

# ADDENDUM 4 — adaptive-by-chop is a null; the +2 protects GIVEBACK, not entry-loss

## Owner proposal: "when we see that we are in a chop we arm an adaptive stop loss"

Structurally the strongest idea in this thread, and it passes the owner's own
earlier test: chop is measured from bars **strictly before** entry, so it is
**structure-conditioned**, not outcome-conditioned.

**Prerequisite tested** (`research/dojo_forge/tools/adaptive_stop_test.py`): an
adaptive rule can only beat a fixed one if the **optimal stop width genuinely
differs by regime**. Binned 12,496 entries by causal Kaufman Efficiency Ratio
(5min window; ER→0 = chop, ER→1 = trend) and priced stops −2/−5/−10 in each bin.

**Result: null.**

| ER bin | ER range | n | mean (−2 minus −10) $ | 95% CI | sig |
|---|---|---|---|---|---|
| 0 (deepest chop) | 0.000–0.039 | 2,500 | −0.132 | [−1.03, +0.70] | no |
| 1 | 0.039–0.082 | 2,502 | +0.602 | [−0.07, +1.23] | no |
| 2 | 0.082–0.131 | 2,498 | +0.432 | [−0.33, +1.18] | no |
| 3 | 0.131–0.200 | 2,500 | +0.177 | [−0.73, +1.00] | no |
| 4 (strongest trend) | 0.201–0.558 | 2,496 | +0.213 | [−0.62, +0.99] | no |

**Spearman corr(ER, stop-width advantage) = −0.0006** across all 12,496 trades.
Every bin's CI includes zero. Optimal stop width does **not** depend on regime;
an adaptive rule has nothing to exploit.

### Tooling honesty note
The first version of `adaptive_stop_test.py` printed *"best stop CHANGES across
regime bins → adaptation has something real to exploit."* **That headline was
wrong** — it picked a winner per bin from point estimates whose CIs overlapped
almost completely, i.e. it chased noise. Caught only by running the **paired
per-entry interaction test** above. Lesson: an auto-verdict that compares point
estimates without testing the difference will manufacture findings.

## Owner clarification: "the +2 is when we are unsure, or it has developed enough, to only protect catastrophic"

This corrects what "catastrophic" means, and it invalidates Addendum 2's
framing of the tail. The catastrophe is **not** loss-from-entry — the initial
stop owns that. It is **giveback**: a developed winner collapsing back.

**Measured** (n=8,344 trades that developed to MFE ≥ 5pt, initial stop −10):

| | control | BE+2 @ +5 | change |
|---|---|---|---|
| mean giveback | $48.30 | **$21.04** | **−56%** |
| p95 giveback | $107.78 | **$57.78** | **−46%** |
| worst giveback | $384.28 | $303.78 | −21% |
| **% of peak profit kept** | **12.3%** | **26.7%** | **+117%** |

**The owner's model is correct.** The +2 halves mean giveback, cuts p95
giveback nearly in half, and **more than doubles the fraction of peak profit
actually retained**. Addendum 2's "the tail is unchanged, the +2 doesn't
protect catastrophe" was measuring loss-from-entry — the wrong catastrophe for
the stated usage.

Both of the owner's trigger conditions now check out, on **different** metrics:

- **"unsure"** → curve smoothing: volatility −46%, drawdown −24% (Addendum 2)
- **"developed enough"** → giveback protection: −56% mean, +117% profit kept (here)

That is a coherent two-condition rule, and each condition is validated by the
metric appropriate to it. Third time in this thread that the owner's framing
was right and my chosen metric was wrong — see
`feedback-score-the-stated-objective`.

---

# ADDENDUM 5 — the trail rule: retention CONFIRMED, EV claim WITHDRAWN

Owner: *"on every favorable trade we should try to catch at least the
commission and then a trailing stop at the 10% of the MFE of the trade."*

Mechanics: phase 1 — once profit ≥ arm, stop → entry+2 (covers the 0.89pt
friction). Phase 2 — thereafter trail at `trail_frac` of MFE behind the peak,
`stop = entry + MFE·(1 − trail_frac)`, ratcheting forward only.

## First run (60d, init stop −10, arm +5)

| config | $/trade | 95% CI | %losers | std $ | mean giveback | **% of peak kept** |
|---|---|---|---|---|---|---|
| control −10 | −2.714 | [−3.49,−1.88] | 78.2% | 46.44 | $48.30 | 12.4% |
| BE+2 only | −2.096 | [−2.52,−1.64] | 33.4% | 25.19 | $20.81 | 27.1% |
| **BE+2 → trail 10%** | **−0.227** | **[−0.51,+0.05]** | 33.4% | **15.79** | **$3.15** | **77.0%** |
| BE+2 → trail 25% | −0.909 | [−1.17,−0.64] | 33.4% | 15.46 | $5.55 | 63.2% |
| BE+2 → trail 50% | −1.678 | [−1.97,−1.39] | 33.4% | 16.41 | $11.54 | 42.0% |

On its face the best result in this entire document — EV within noise of zero
and 6× the profit retention. **So it got stress-tested before being believed.**

## Stress test 1 — parameter boundary

| trail % | 5% | 10% | 15% | 25% | 40% |
|---|---|---|---|---|---|
| $/trade | **+0.078** | −0.227 | −0.481 | −0.909 | −1.356 |

**Monotonic: tighter is always better, with the optimum at the tightest value
tested.** An optimum sitting on the edge of the swept range is a standard
artifact signature, not an edge.

## Stress test 2 — IS/OOS split (30 days each, trail 10%)

| split | n | $/trade | 95% CI | % peak kept |
|---|---|---|---|---|
| IS | 6,196 | −0.158 | [−0.55,+0.22] | 77.0% |
| OOS | 6,300 | −0.294 | [−0.67,+0.10] | 77.0% |

Consistent across periods — so **not** period-overfit. But this check is blind
to a systematic simulation bias, which is what stress test 3 found.

## Stress test 3 — THE BUG: same-bar fill bias

The sim set the trail from bar *i*'s high but only checked the stop from bar
*i+1* — **granting a free pass on the pullback inside the peak bar**. OHLC
cannot resolve intrabar order, so both bounds must be reported:

| trail % | optimistic (as first reported) | conservative | bias | conservative CI |
|---|---|---|---|---|
| 5% | +0.078 | −0.151 | 0.229 | [−0.43,+0.12] |
| **10%** | **−0.227** | **−0.580** | 0.354 | **[−0.85,−0.31] SIGNIFICANT LOSS** |
| 25% | −0.909 | −1.811 | 0.902 | [−2.06,−1.56] significant loss |
| 40% | −1.356 | −2.788 | 1.432 | [−3.03,−2.54] significant loss |

(Bias grows with trail width because a wider trail keeps trades alive longer,
so there are more bars on which the free pass can be collected.)

## Verdict — split

**WITHDRAWN: the EV claim.** At the owner's stated 10%, honest fill pricing
turns −$0.23 into **−$0.58 with a CI excluding zero — a significant loser**.
The headline figure should never have been led with; only the optimistic bound
supported it.

**CONFIRMED: profit retention, which is what the owner actually proposed.**
**12.4% → 77.0% of peak kept**, mean giveback **−93%**. This is *mechanical* —
the trail exits near the peak by construction — so unlike the EV number it is
not an artifact of the fill assumption. Volatility also drops (46.44 → 15.79).

**So the rule is a strong profit-retention mechanism and is not established as
an edge.** Those are different claims and only the first one survives.

## Lesson

The result that looks best is the one to attack hardest. Three checks —
boundary, IS/OOS, fill realism — and the third one, the boring
implementation-detail check, is what actually broke it. IS/OOS passed and was
*uninformative*, because a systematic sim bias contaminates both splits equally.

---

# ADDENDUM 6 — THE MEASUREMENT ERROR: I scored the rule on trades it never governed

Owner: *"how can we lose if we are arming once it is fully developed... if we
are +50pt and we protect ourselves with +5pt it does not make sense to lose
money, that's only if we are in a losing trade. If you are measuring against
losing trades then you're measuring incorrectly."*

**Correct, and it invalidates every $/trade figure in Addenda 1–5.**

The rule is *conditional*: it arms only once a trade reaches +N. Trades that
never get there keep the initial stop and are **never touched by the rule**. My
averages pooled both populations, so I was reporting the seatbelt's performance
over crashes it was not buckled for.

## Decomposition (60d, stop −10, arm +5, trail 10%, conservative fills)

| population | n | share | mean $ | min $ | max $ | any loss? |
|---|---|---|---|---|---|---|
| **ARMED (developed)** | 8,325 | 66.6% | **+10.02** | **+7.22** | +143.57 | **NO — zero in 8,325** |
| never armed | 4,171 | 33.4% | −21.73 | −21.78 | +1.22 | yes |
| ALL *(as previously reported)* | 12,496 | 100% | −0.58 | −21.78 | +143.57 | — |

**Not one armed trade lost money.** Minimum outcome +$7.22; CI [+9.92, +10.11].
This is guaranteed by construction — arming moves the stop to entry+2, so the
floor is +2pt − 0.89pt friction = **+$2.22**. The owner's logic was airtight and
the data confirms it exactly.

## The reframing: it was never a management problem, it is an ENTRY problem

If armed trades cannot lose, total P&L depends on one number only — **what
fraction of entries develop far enough to arm**. Break-even arm rate:

```
hurdle = −unarmed_avg / (armed_avg − unarmed_avg)
```

| stop | arm | armed avg | unarmed avg | actual arm% (random) | **hurdle** | margin |
|---|---|---|---|---|---|---|
| −5 | +5 | +10.01 | −11.78 | 49.8% | 54.1% | −4.3pp |
| −10 | +5 | +10.01 | −21.73 | 66.6% | 68.4% | −1.8pp |
| −10 | +3 | +6.34 | −21.77 | 76.5% | 77.4% | −0.9pp |
| −15 | +5 | +10.05 | −31.24 | 75.0% | 75.6% | −0.6pp |
| **−15** | **+3** | **+6.38** | **−31.60** | **83.0%** | **83.2%** | **−0.2pp** |

**At −15 / +3, a coin flip lands 0.2 percentage points short of break-even.**
The owner's entry read has to supply only that much development-rate skill.

This is the first concrete, falsifiable target produced in this whole thread:
not "find an edge", but **beat random by ~1pp on "does this reach +3 before
−15"**. That is a directly measurable property of the dojo corpus entries.

## Honest flag on the absolute numbers

Every configuration lands *slightly* negative — consistent with friction — but
the absolute EV is better than a pure martingale plus friction should allow
(≈ −$1.78/trade). That gap is unexplained and most likely residual optimism in
the simulator that the same-bar fix did not fully remove. **Treat the hurdle
percentages as indicative, not precise.**

What is *not* assumption-dependent, and stands regardless: **armed trades cannot
lose** — that is arithmetic from the stop placement, not an estimate. So the
decomposition and the "it's an entry problem, not a management problem"
conclusion both hold even if the exact hurdle numbers shift.

## Standing correction to Addenda 1–5

Every "$/trade" figure in the preceding addenda pools armed and unarmed trades
and therefore **understates any conditional rule**. They remain valid as
descriptions of *unconditional always-on* application; they are the wrong
statistic for a rule the owner applies selectively. This is the fourth time in
one session the owner corrected the framing rather than the arithmetic — see
`feedback-score-the-stated-objective`, now extended: score the rule on the
**population the rule actually governs**.

---

# ADDENDUM 7 — FAKEOUTS ARE MEASURABLY SEPARABLE (the actionable result)

Owner spec: *random entry, stop $20, arm at +10 → protect +2, trail 10% of MFE,
warning at 80%, if it touches 80% N times move trailing to 70%* — then:
*"the warning is the mechanical arm trigger to brace for exit"* and *"the
concept is specifically to target fake outs — measure how much do fakeouts dip."*

## Implementation notes (two of my own bugs, both caught here)

1. **Reading ambiguity, flagged not guessed**: with a trail retaining 90% of
   MFE, a "touch at 80% of MFE" is already *past* the stop — unreachable. The
   self-consistent reading is that the percentages are **retention levels**:
   trail sits at 10% of MFE (loose), warning at 80% of MFE (above it, so
   reachable), tighten to 70% after N touches.
2. **Touch counting was level-triggered, not edge-triggered** — a single
   sustained pullback racked up dozens of "touches", which is why N=40 still
   fired constantly. Fixed to count fresh crossings. With the fix there is a
   genuine **interior optimum at N=3** (armed mean $16.42 vs $15.76 with no
   warning) rather than a boundary artifact.

## Is the warning a "brace for exit" signal? — NO

| metric | value |
|---|---|
| additional MFE gained **after** the warning | **+95.7% mean, +32.4% median** |
| trades making NO new high afterwards | 30.6% |
| fraction of trade life remaining at warning | 55.5% |

The warning fires **mid-move**, not at the end. 69% of trades go on to new
highs. *(My script printed an auto-generated line claiming the warning "does
mark the move is done" — written before the numbers existed. It was wrong.
Second pre-written interpretation string to mislead in this session; stop
writing them.)*

## How deep do fakeouts dip? — the measurement that explains everything

Fakeout = a retracement off the running peak that is **later exceeded**.
Real reversal = the retracement that **ends** the move.
n = 23,970 fakeouts, 5,809 real reversals. Depth as % retracement of running MFE:

| percentile | **FAKEOUT** | **REAL REVERSAL** |
|---|---|---|
| 10% | 6.4% | 39.3% |
| 25% | 10.9% | 63.6% |
| **50%** | **19.8%** | **76.7%** |
| 75% | 35.7% | 84.2% |
| 90% | 56.5% | 89.4% |
| 95% | 67.9% | 91.8% |
| mean | 25.8% | 70.6% |

**The two populations genuinely separate — medians are 4× apart.** Fakeouts are
shallow; real reversals are deep. The owner's concept is confirmed by the data.

## The diagnosis

**The 80% warning level = a 20% retracement = exactly the median fakeout depth.**
**49.9% of fakeouts dip at least that far and then resume.** The trigger sits on
the precise point where fakeout-vs-real is a coin flip — the single least
informative location on the curve. That fully explains why tightening there
truncates winners and why the "brace for exit" reading failed.

## The fix

| survive this share of fakeouts | trail must allow retracement deeper than | i.e. keep ≤ |
|---|---|---|
| 50% | 19.8% | 80.2% of MFE |
| 75% | 35.7% | 64.3% of MFE |
| **90%** | **56.5%** | **43.5% of MFE** |
| 95% | 67.9% | 32.1% of MFE |

A trail at **~56% retracement survives 90% of fakeouts** while still exiting
well before the **76.7% median real-reversal** depth — roughly a **20-point
separation band** to operate in. That band is the actual exploitable structure
found in this entire thread.

Consistent from the other direction: **N=3 edge-triggered touches** helps
because requiring repeats filters shallow fakeouts — the same insight expressed
as a counter instead of a depth.

---

# ADDENDUM 8 — the multi-touch logic: real effect, too weak to call a top

Owner specified the mechanism precisely: *"if it's a fakeout it will touch and
resume; if we are in the top it will multi-touch and then crash, since no new
MFE is being made the likelihood to return to around the same value is high."*

**This exposed another implementation error of mine.** Addendum 7 counted
touches *cumulatively across the whole trade*, pooling touches that were
followed by new highs (fakeouts, by the owner's own definition) with those that
were not. The owner's logic requires the counter to **RESET on every new MFE** —
a touch that resumes *proves* it was a fakeout and carries no topping
information. Rebuilt accordingly.

## Decisive test: does P(new high) fall as touches accumulate at the SAME peak?

| touches at same peak | n | P(new high after) | P(move is done) |
|---|---|---|---|
| 1 | 3,526 | **76.8%** | 23.2% |
| 2 | 3,433 | 65.6% | 34.4% |
| 3 | 2,307 | 67.2% | 32.8% |
| 4 | 1,697 | 67.6% | 32.4% |
| 5 | 1,303 | 69.1% | 30.9% |
| 6+ | 5,267 | 64.5% | 35.5% |
| **baseline** | 17,533 | 68.2% | 31.8% |

**The effect is real.** 1 touch → 2+ touches drops P(new high) from **76.8% to
66.0%** — a **−10.8pp** move, 95% CI **[9.1, 12.3]pp**, clearly significant.
The owner's intuition that repeat-touching-without-new-MFE carries information
is **confirmed**.

## But it cannot support the conclusion drawn from it

1. **It plateaus immediately.** 2 touches ≈ 3 ≈ 4 ≈ 5 ≈ 6+ ≈ 66%. Everything
   the signal has to say is said by the *second* touch; "multi" beyond that adds
   nothing. The counter-sweep (N=3 optimum in Addendum 7) was reading noise in
   that flat region.
2. **The direction of the majority never flips.** Even at **6+ touches, the move
   still resumes 64.5% of the time.** Calling a top on multi-touch would be
   wrong roughly **2 times in 3**. It is a continuation-*dampener*, not a
   reversal signal.

## Synthesis — depth beats count

| discriminator | separation achieved |
|---|---|
| **dip DEPTH** (Addendum 7) | fakeout median 19.8% vs real reversal 76.7% — **4× apart** |
| touch COUNT (here) | 76.8% → 66.0% — **10.8pp, then flat** |

**Depth is the informative variable; count is a weak proxy for it.** A top
detector should be built on *how deep the retracement goes*, not *how many
times price taps a level*. That also re-explains Addendum 7's core finding from
a third angle: the 80%-of-MFE level (a 20% retracement) sits at the median
fakeout depth, so no amount of counting touches *at that level* can rescue it —
the level itself is in the wrong place.

---

# ADDENDUM 9 — TIME is the free top discriminator (the answer)

Owner's two objections, both correct:

1. *"how deep the dip goes loses money by design"* — **right, and decisive.**
   Depth diagnoses well (4× median separation, Addendum 7) but you must **pay
   the retracement to learn it**. Waiting for a 56% dip to confirm a real
   reversal means having already surrendered 56% of MFE. Depth is a *post-mortem*
   variable, not a decision variable.
2. *"if you're resetting the counter it's obvious that it will top out"* — fair.
   Resetting on each new MFE guarantees the **terminal** cluster is a
   no-new-high, so part of Addendum 8's decline is mechanical. (The measured
   effect came out *weak* — 66% still resume — which is itself evidence it was
   not purely tautological, but the objection stands as a caution.)
3. *"around 20% is the allowance, the breathing room we should give"* — matches
   the fakeout median (19.8%) exactly. Kept as the breathing room here.

**So the requirement is a discriminator that costs nothing to observe.** Time
qualifies: stalling below the peak surrenders no giveback.

## P(a new MFE is still to come | time since the last peak)

armed trades, 20% breathing room, n = 567,417 observations

| time since peak | n | P(new high) | after touching 20% | P(new high) |
|---|---|---|---|---|
| 0–1 min | 180,628 | **71.6%** | 108,674 | 64.4% |
| 1–2 min | 93,053 | 58.7% | 83,174 | 57.5% |
| 2–3 min | 61,275 | 51.6% | 58,487 | 51.3% |
| 3–4 min | 45,213 | 46.6% | 44,022 | 46.5% |
| 4–5 min | 34,574 | 42.9% | 34,103 | 42.8% |
| 5–6 min | 27,440 | 40.2% | 27,233 | 40.1% |
| 6–7 min | 22,440 | 37.7% | 22,385 | 37.7% |
| 7–8 min | 18,579 | 35.6% | 18,555 | 35.6% |
| **8+ min** | 84,215 | **26.3%** | 84,201 | 26.3% |
| baseline | 567,417 | 52.8% | | |

**Monotonic, no plateau, 45pp spread, and it crosses 50% at ~3 minutes.**

## Discriminator comparison

| variable | separation | cost to observe |
|---|---|---|
| dip **depth** | 4× median (19.8% vs 76.7%) | **you pay the depth** |
| touch **count** | 10.8pp then flat; still 2:1 wrong | cheap but weak |
| **TIME since peak** | **45pp, monotonic, crosses 50%** | **free** |

## The rule this yields

Keep the **20% breathing room** (owner's figure, = median fakeout depth) as the
trail. Use **time without a new MFE** as the top signal: **past ~3 minutes the
odds flip against continuation**, and waiting for that costs zero giveback.

**Notable**: past ~2 minutes, whether price touched the 20% level adds almost
nothing (57.5% vs 58.7%) — **time dominates the touch entirely**. The touch only
carries information inside the first minute (64.4% vs 71.6%). The owner's
instinct to watch the 20% level was right; it is the **clock at that level**
that matters, not the count of taps.

---

# ADDENDUM 10 — THE RATCHET WORKS (first significant positive result)

Owner framed the design correctly: *"this is the ratchet trailing stop — when
should we stop letting it breathe, and how hard should we ratchet"*, plus
*"after 3 minutes have passed we should probably move trailing stop to 50% at
least."*

Two implementation errors of mine had to be fixed before the mechanism was even
**observable**.

## Error 1 — a pure %-trail is narrower than one bar at small MFE

At MFE = 10pt, a 20%-of-MFE trail leaves **2pt** of room — narrower than a
typical 5s bar range (1–3pt). Armed trades died in ~3 bars. Instrumentation
showed `stale-bars = 0`: **the time condition could never fire**, which is
exactly why 13 different ratchet configs returned byte-identical results.
Fixed with an absolute floor: `room = max(5pt, pct × MFE)`.

## Error 2 — the watch line was the stop line

More fundamental: I had the observation level *at* the stop. So "price touches
the 20% level" and "price stops out" were **the same event** — the multi-touch
pattern the owner describes was unobservable by construction. Only **0.5%** of
armed trades ever stalled 3 minutes.

**The architecture the owner's design actually requires:**

| element | level | role |
|---|---|---|
| **stop** | ~50% retracement | real protection, loose — lets it breathe |
| **watch line** | ~20% retracement | observation only — never exits |
| **ratchet** | tighten to 10–20% | fires when the watch line stalls / multi-touches |

Watch and stop are **different objects**. Collapsing them into one is what made
every earlier version come out flat.

## Results — corrected architecture (60d, ARMED trades, n=6,198)

| config | ARMED $ | 95% CI | % peak kept | % fired |
|---|---|---|---|---|
| loose 50% trail, no ratchet | 16.91 | [16.45, 17.41] | 46.9% | — |
| **3min stall → ratchet to 10%** | **17.62** | [17.17, 18.10] | 50.2% | 9.6% |
| 3min stall → ratchet to 20% | 17.40 | [16.98, 17.85] | 49.4% | 9.6% |
| **2 touches → ratchet to 20%** | **17.42** | [17.12, 17.74] | **55.4%** | 44.8% |
| 3 touches → ratchet to 20% | 17.45 | [17.13, 17.80] | 53.9% | 33.6% |

**Paired test against the no-ratchet baseline** (same entries, only the ratchet
differs):

| variant | delta | 95% CI | verdict |
|---|---|---|---|
| 3min stall → tighten to 10% | **+$0.713** | [+0.48, +0.93] | **SIGNIFICANT** |
| 2 touches → tighten to 20% | **+$0.507** | [+0.13, +0.85] | **SIGNIFICANT** |

**This is the first statistically significant positive result in the entire
thread.** Retention also improves 46.9% → 55.4%.

## On the owner's "move to 50%"

The 50% figure lands naturally as the **stop** (loose protection), not the
ratchet target. The coherent scheme: **breathe at 50%, watch at 20%, ratchet to
10–20% once the watch line stalls or is multi-touched.**

## Standing caveats

Still random entries; absolute EV retains the unresolved simulator optimism
noted in Addendum 6. What is new and solid here is the **relative, paired**
improvement — same entries, same stop, only the ratchet differing — which is
robust to a bias that affects both arms equally. The decisive next step is
unchanged: run this on **owner-selected dojo-corpus entries**.

---

# ADDENDUM 11 — the owner's literal sequence: the CONDITIONAL GATE is what works

Owner, after rejecting the previous numbers: *"if we enter a trade, and we are
in the right direction, and we arm a warning at 80% and we have a retracement
to that level, and we have not developed a new MFE in 3 minutes, then we ratchet
up to 50%."*

**Decisive reading correction: there is NO continuous trail.** The stop sits at
entry+2 until the warning **and** the timer both fire, then it jumps to 50% of
MFE. "Ratchet up to 50%" means up *from +2* — a large step up — not a tightening
from 80%. Every previous addendum ran a continuous trail that was choking trades
before the mechanism could engage.

## Auditable trace (real trade, 2026_03_20)

```
SHORT 24401.75 -> MFE 51.75pt, result +9.48pt
  bar 74  ARMED (MFE 10.25) -> stop to entry+2
  bar 87  touched 80% line
  bar134  3min stalled -> RATCHET stop to 50% of MFE = 24391.38
  bar183+ new MFE ... 51.75pt          <-- stop never re-stepped
  bar306  STOPPED at 24391.38
```

The trace is what found the real defect: the ratchet fired **once at a 20pt-era
peak** and never re-stepped as MFE grew to 51.75pt. Aggregate metrics had hidden
this completely.

## Does the warning+timer machinery earn its keep? (60d, ARMED trades, paired)

| variant | ARMED $ | % peak kept | vs do-nothing |
|---|---|---|---|
| +2 floor only (no trail) | 16.74 | 32.1% | — |
| **warning+timer gated → 50%** | **17.37** | 36.5% | **+$0.63, CI [+0.02,+1.22] SIGNIFICANT** |
| always trail at 50% of MFE | 16.91 | 46.9% | +$0.17, CI [−0.72,+1.03] **ns** |

**The owner's gated design is the only variant that significantly beats doing
nothing.** Trailing continuously at the *same* 50% level does **not** — it gets
shaken out by fakeouts. **The conditional gate is the value**, exactly as
designed: wait for the warning *and* the stall before tightening, so ordinary
fakeouts don't trigger it.

**A real tension the numbers expose**: always-trailing *keeps* more of peak
(46.9% vs 36.5%) yet *earns less*. Higher retention of a smaller peak. The gated
version lets trades run further and nets more dollars — retention and P&L are
not the same objective, and optimising the first can cost the second.

## Honest limits

- **gated vs always-50** is +$0.46 with CI [−1.09, +0.13] — so the claim
  supported is *gated beats nothing*, **not** *gated beats always-trailing*.
- One test in this sequence was **botched**: a "one-shot vs persistent ratchet"
  comparison had both arms accidentally persistent and returned a +0.00 delta.
  It proved nothing. The bar-by-bar **trace**, not the aggregate sweep, is what
  actually located the defect — worth remembering when a sweep returns
  suspiciously identical numbers.
- Still random entries; absolute EV retains the optimism flagged in Addendum 6.
  The paired deltas are the robust part.

---

# ADDENDUM 12 — THE FULL SPEC, component-isolated (definitive)

Owner's complete state machine, stated precisely:

> enter, stop −10 · MFE ≥ +10 → protection triggers, stop to +2 · **frozen until
> MFE > 20** · then every new MFE ratchets the stop to **10% of MFE** (at 50pt
> MFE the stop sits at +5) · it fakes out to **80% of MFE** and **stalls 3
> minutes** with no new MFE → ratchet the stop to **50% of MFE** (25pt) · if it
> continues, the mechanism **re-arms**.

Traced end-to-end on real data — a 176pt-MFE short returned **+166.11pt**. The
machine works as designed.

## Component isolation (60d) then high-power confirmation (200d, 18,766 armed)

| component | delta | 95% CI | verdict |
|---|---|---|---|
| 10% trail after 20pt | **+0.00** | [−0.20, +0.18] | **contributes nothing** |
| warn/stall ratchet → 50% | **+0.284** | **[+0.022, +0.545]** | **SIGNIFICANT** |

The **10% trail is inert** — it sits so far below price that it almost never
binds before something else exits the trade. **The warn/stall ratchet is the
part that works.**

## The finding that matters more than the headline

| population | n | ratchet delta |
|---|---|---|
| small movers (MFE < 30pt) | 14,814 | **+$1.21** |
| **big movers (MFE ≥ 30pt)** | **3,952** | **−$3.17** |

**The ratchet helps small trades and badly hurts big ones.** Its net positive
survives only because small trades are 79% of a random-entry sample. Big movers
stall *and then continue* — ratcheting to 50% cuts them at half their peak.

**Direct implication for this program**: if the owner's entry read
preferentially produces big movers — which is the entire purpose of selecting
entries — the ratchet may be **net negative on his actual trades** while
testing positive on random ones. The random-entry harness is biased *toward*
the ratchet here, in a knowable direction. This is the sharpest example yet of
why the outstanding "replay on owner-selected dojo entries" experiment is the
one that matters.

**Obvious next test**: scale the ratchet level with MFE — tighten to 50% on
small trades but only ~25% on large ones, preserving room for runners.

---

# ADDENDUM 13 — "how can it be negative" — resolving the opportunity-cost confusion

Owner's two challenges: *"how the hell can it be negative, by definition it
can't"* and *"the only way it is negative is when the +2 never arms, which is
out of scope."*

**Both fair questions; both answered by the same distinction.**

1. **The +2 guarantee holds completely.** The comparison population is
   armed-only (n=18,766) — unarmed trades were already excluded. Every trade is
   profitable in **both** arms: floor +$2.22, **zero losses out of 18,766** in
   either variant. Nothing in the data contradicts "once armed, we cannot lose
   money." That claim is confirmed, again.
2. **The −$3.17 is not a trade P&L — it is a paired difference between two
   profits on the same trade.** Concrete worst case (2025_12_11, short @
   25662.50, ultimate move 223pt):
   - WITHOUT ratchet: rides, exits **+217.7pt** (+$435)
   - WITH ratchet: 80% fakeout at MFE 86.5 → 3min stall → stop to +43; second
     fakeout at MFE 99 → stall → stop to +49.5; **dip tags it, out at +48.6pt**
     (+$97). The move then runs another 120+ points without the position.
   - Both outcomes are profits. The ratchet **cost −169pt (−$338)** by
     selecting the smaller one.

**Mechanism, precisely**: the ratchet only ever *raises* the stop, and a raised
stop exits *earlier* on a dip that would have recovered. On big movers, "80%
fakeout + 3-minute stall" occurs **mid-move** often enough that the raised stop
catches the next dip and cuts the runner. So:

- small movers (79% of sample): the cut usually banks the right side → **+$1.21**
- big movers: the cut reliably trades a large profit for a small one → **−$3.17**

**Summary of what is and is not claimed**: *"armed trades never lose"* —
confirmed, 18,766/18,766 positive, floor +$2.22. *"a tighter stop can never
make results worse"* — rejected; on runners it systematically picks the smaller
of two profits. Risk guarantee and opportunity cost are different objects, and
the ratchet's cost lives entirely in the second.

---

# ADDENDUM 14 — honest fills kill the exit sweep; fakeout count is a CONTINUATION signal

## Correction first: the T×N sweep had phantom fills

The first sweep of "stall → exit at N% of MFE" showed N=80% dominating
(+$3.75 overall, +$10 on runners). **That was a fill-realism bug**: at
stall-confirmation, price can already be *below* the N% line, and the sim
placed the stop on the profit side of the market and booked the exit AT it —
phantom profit, maximised at the tightest N. Caught before reporting because
+$10-on-runners failed the smell test.

**Honest version** (stop capped at market; if the N% line is already above
price it degenerates to a market exit): **every T×N config is non-significant
and every one still bleeds on runners.** Best: `T=3min → market exit`,
+$0.63 [−0.00,+1.23] ns. The deliberate-exit-on-stall idea, priced honestly,
does not clear noise on random entries.

## The owner's three questions (pure event measurement, no fill assumptions)

**Q3 — how many fakeouts before the actual pivot?** *(the headline)*

| population | mean fakeouts | 0 | 1 | 2 | **3+** |
|---|---|---|---|---|---|
| all armed (n=6,198) | 1.93 | 37.6% | 19.4% | 12.4% | 30.6% |
| **runners MFE≥30 (n=1,506)** | **4.68** | 3.5% | 5.4% | 10.4% | **80.7%** |

Seeing 3+ fakeouts moves P(runner) from the 24% base rate to **~64%**.
**Fakeout count is a CONTINUATION signal, not a reversal signal** — faking out
repeatedly is what runners *do*. This is the structural reason every
tighten-on-fakeout variant in Addenda 10–13 bled on big movers.

**Q2 — how much new MFE defines a new state?** **None — no threshold exists.**
P(another high) by increment size: 0.25–1pt **89.9%**, 1–2.5pt 89.7%,
2.5–5pt 89.4%, 5pt+ 88.8% (n=47,169 events). **Even a one-tick new high resets
the state at full strength.** Increment size only scales the *magnitude* of
what follows (median additional MFE 9.25pt → 16.75pt).

**Q1 — what happens when new MFE is found after a warning?** **Nothing
special — resumption ≈ fresh.** P(another high): 88.8% resumed vs 89.9% fresh;
median additional MFE 10.5pt vs 11.0pt. Once a new high prints, the stall
history carries essentially no information. The owner's re-arm design (full
reset on new MFE) is exactly correct.

## Where this leaves the design

The **floor architecture stands**: −10 → +2 at MFE 10 (frozen to 20) → 10%
trail; guaranteed minimum verified at every tier. The **fakeout/stall
machinery reads backwards**: repeated fakeouts are evidence you are holding a
runner — the last thing to do on the third fakeout is tighten. On random
entries no mechanical exit-on-stall variant clears noise once fills are honest;
the remaining live question is unchanged — the same machine replayed on
**owner-selected dojo entries**.
