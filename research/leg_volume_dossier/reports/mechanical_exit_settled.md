# The causal mechanical exit — SETTLED (2026-07-26)

Composed all validated instruments (gauge ARMED = faded-vigor + stacked
anomalies, conviction fade, give-back confirmation, catastrophic floor) into a
layered LegExitEngine and backtested vs never-bail (148 ride eps, 22 days,
day-block CI).

## Result: the LEANEST version wins
- Full layered engine (floor 50 + terminal-confirmed layer): **−7.9 pts/ep**
  vs never-bail, CI [−16.1, −0.5] — LOSES, 84% exit rate (too trigger-happy).
- Floor-ONLY (= trail50, ride + wide disaster stop): **−6.0 pts/ep**, the
  least-bad of everything tested.
- Adding instrument-driven exits made it WORSE (−8 vs −6). On this data, MORE
  exit intelligence = WORSE outcome.

## The settled mechanical exit
**Ride (never-bail) + a wide ~50pt catastrophic give-back floor. Nothing else.**
The floor's job is disaster insurance, not timing: worst single episode
never-bail −132 pts → engine −82 pts (+50pt tail cap) for ~6pt/ep average cost.
Whether that insurance is worth its premium is a risk-tolerance call, not a
capture call.

## What this closes and what it frees
CLOSED: the exit-timing question. Exhaustively tested (binary, trailing,
scale-out, gauge-gated, layered-engine) — never-bail + floor dominates. No
instrument improves exit TRIGGERING.
FREED: the instruments (gauge, conviction, anomalies, wrong-direction) are not
exit triggers — their validated home is ATTENTION / ENTRY / SIZING. That is
where the next research goes.
Caveat: 22 days, dev; lockbox re-test before production commit.
