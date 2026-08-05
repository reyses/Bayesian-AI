# PROTECT PROTOCOL — canonical (owner, 2026-08-02)

The owner's profit-protection sequence, stated after it was mis-executed twice.
This document supersedes all earlier paraphrases (including the 2026-08-01
"80% warn = live exit" note — the 80 line is a DECISION point, not an exit).

## Naming convention (PROPOSED — awaiting owner sign-off)
| term | meaning |
|---|---|
| **MFE peak** | running maximum favourable excursion of the open trade; every new high RESETS the whole protection structure to the new peak |
| **expected region** | the target zone of the current leg — the opposite oscillation region, coinciding with previously established levels |
| **ARM** | protection activates only when price is NEARING / inside the expected region's approach zone — never at entry |
| **the 80 line** | warning at 20% giveback of MFE peak (profit ≤ 0.80 × MFE). SIM: **stop the tape**, owner decides. Not an exit. |
| **the 70 line** | hard limit at an additional 10% giveback (profit ≤ 0.70 × MFE), armed only BY the owner's decision at the 80 halt. Hitting it = EXIT, capturing 70% of MFE. |

## The sequence
1. Oscillation identified; regions fixed (entry region A, expected region B).
2. Enter at A toward B. Fail-safe hard stop sized OUTSIDE the fakeout
   distribution (separate, standing rule).
3. Track MFE continuously while the leg runs. **No protection armed yet.**
4. Price NEARS region B → **ARM the 80 line** on the current MFE peak.
5. 80 line touched → **STOP THE TAPE**. Decision point (this decision is
   itself corpus data). Options: exit now · extend to the 70 line · other.
6. If extended: the 70 line is the hard limit — touch = exit, 70% of MFE kept.
7. **Any new MFE peak resets everything**: lines recompute off the new peak,
   structure returns to armed-warning state (assumed: an extension also
   resets — CONFIRM).
8. On exit at region B: evaluate the REVERSAL — enter the opposite direction
   toward the next region (≈ the previous trade's entry zone). The ping-pong
   continues.

## Why the arming rule is load-bearing (measured 2026-08-02)
Every prior implementation armed protection at entry (peak > 2pt). The
cushion-curve measurement shows why that misrepresents the design: P(loss)
under the ratchet is 36–44% at a 2–5pt cushion but **1–7% at 10–20pt and
~1–3% at ≥20pt (mean +19 to +21pt)**. Arming only NEAR the expected region
means, by construction, the cushion at arm-time is a near-full traverse
(≈15–40pt) — the protocol lives in the armor zone of its own curve. The
pooled "P(loss|armed) ≈ 27%" finding does NOT apply to this protocol; it
described entry-armed ratchets.

## Open items
- Owner sign-off on the naming table.
- Confirm: new MFE while EXTENDED → full reset to warning state?
- Definition of "nearing" for ARM: proposed = inside the region's measured
  density band (the `region` construct, 68% half-width). CONFIRM.
- `protect` as a first-class pocket_dojo command (state machine
  IDLE→ARMED→WARNED→EXTENDED, auto-reset on new MFE) — queued for AFTER the
  current dojo session; no hot-patching tools under a live operator. Until
  then the operator executes it manually via `warnstop` recomputation.
