# Causality audit — truncation replay

Each of 40 randomly sampled days was replayed with the tape cut at 11:00, 13:00, 14:30 ET. Every event stamped at or before the cut must appear, identically, in the truncated run. Outcome fields are excluded (they are forward-looking by design).

| detector | events compared | field mismatches | MISSING in truncated run | EXTRA in truncated run |
|---|---|---|---|---|
| ultra_chop | 1993 | 0 | 0 | 0 |
| leg_descent | 6510 | 0 | 0 | 0 |
| fakeout_poke | 17066 | 0 | 0 | 0 |
| stall | 4302 | 0 | 0 | 0 |
| defended_poke_shelf | 164 | 0 | 0 | 0 |
| flush_v_day | 102 | 0 | 0 | 0 |

`MISSING` = the detector needed a FUTURE bar to emit an event the full run produced -> lookahead. `EXTRA` = the truncated run emitted an event the full run suppressed -> a forward-looking SAMPLING rule (de-dup / refractory), which does not leak into any event's own features or outcome but does mean the live row set differs near a cut.

