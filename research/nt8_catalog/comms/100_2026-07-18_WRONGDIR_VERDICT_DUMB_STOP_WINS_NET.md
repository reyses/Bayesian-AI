# Wrong-Direction Dojo verdict — dumb stop WINS net; the LLM's edge is real but mispriced
**Doc:** 100 · **Date:** 2026-07-18 · **Author:** Claude (reviewer) · **Status:** FINAL
199/200 episodes finished (1 timeout), 198 scored (nonce-chain audit PASS on all
scored). Economic truth (WRONG = terminal ≤ −4pts, GOOD ≥ +4pts, goods split
50 dipped / 50 clean). Pre-registered bar: beat the dumb adverse-drawdown-stop
ROC AND its best-X net ticks.

## 1. Headline — PRE-REGISTERED VERDICT: **FAIL**
| metric | blind agents | dumb stop (best X=24t) | never-bail |
|---|---|---|---|
| catch (bail \| loser) | **95%** | (per ROC) | 0% |
| false-bail overall | 32% | — | 0% |
| false-bail on CLEAN goods | **10%** | ~0% (rarely hit) | 0% |
| false-bail on DIPPED goods | **54%** | ~100% (definitionally) | 0% |
| precision | 75% | — | — |
| **net ticks/episode** | **+7.5** | **+17.7** | 0 |

The agents beat doing nothing (+7.5 vs 0) but **lose to the dumb 24-tick stop
on net (+7.5 vs +17.7)** — the pre-registered kill criterion. The lane
"LLM-discretionary bail beats a stop" is CLOSED, consistent with the exit
dojo (098) and the whole stop-overlay graveyard.

## 2. Why — the dipped-good knife is catastrophically priced
The agents' discrimination is REAL: 10% false-bail on clean winners vs 54% on
dipped ones proves they can read "still healthy" when the path is clean. But
a knifed dipped winner is the most expensive mistake on the board (single
episodes at −454t, −630t, −795t forgone), and at 54% false-bail the knife
erases the catch gains. The dumb stop bails EVERY dipped good by definition —
yet still nets more, because its losses are capped early and uniformly while
the agents' late catches (median bail minute later than X=24's trigger) leak
adverse drift before acting.

## 3. What survives — the candidate cut-conditions (for the sealed harness)
Bail-reason grammar from 126 binding EXITs: adverse extreme 88%,
against-fires 63%, giveback/retrace 61%, ACCELERATING loss 38%, ER10 22%.
Same confluence family as the exit-dojo grammar — the vocabulary is stable
across both blind experiments. Status: HYPOTHESIS (graduation firewall).

## 4. Downstream (fills PRODUCTION_RUN_SPEC §6)
Per the spec's pre-declared branch: **the cut-head target IS the dumb stop**
— bail at the ROC-optimal X≈24 ticks (6pts) as the baseline behavior — and
the network's learnable margin is exactly ONE skill: **not-bailing the dipped
goods** (the 54%→10% gap says the information exists in the path). Reward
shaping: penalize bail-on-dipped-good by forgone ticks; credit early catch by
damage-avoided. Do NOT ask the net to out-detect the stop; ask it to VETO the
stop on recoverable dips.

## 5. Bookkeeping
Artifacts: reports/wrongdir/{scorecard,synthesis}.md, gate_state/ (198
audited transcripts), selection_table.md, fleet_run.log. The 1 unfinished
episode (timeout) excluded; 198 distinct days. Fleet ran with the scoped
gate-only allowlist throughout.
