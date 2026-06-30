# RUNG 1 — regime labeling vs manual labels | model=gemma4:latest | split=OOS n=30

      date | truth dir/var | pred dir/var | dir var
------------------------------------------------------------
2026-01-28 | FLAT/CHOPPY | <no-json>
2026-01-29 | DOWN/SMOOTH | DOWN/CHOPPY | Y   n
2026-02-23 | FLAT/CHOPPY | DOWN/SMOOTH | n   n
2026-01-23 | FLAT/CHOPPY | DOWN/SMOOTH | n   n
2026-02-02 |   UP/SMOOTH | <no-json>
2026-02-26 | FLAT/CHOPPY | DOWN/CHOPPY | n   Y
2026-01-06 | FLAT/CHOPPY |   UP/SMOOTH | n   n
2026-03-08 | DOWN/SMOOTH | DOWN/CHOPPY | Y   n
2026-02-06 |   UP/SMOOTH | <no-json>
2026-03-20 | DOWN/SMOOTH | DOWN/CHOPPY | Y   n
2026-03-04 |   UP/CHOPPY | <no-json>
2026-01-30 | FLAT/CHOPPY | <no-json>
2026-02-03 | DOWN/SMOOTH | <no-json>
2026-02-16 | DOWN/CHOPPY | DOWN/SMOOTH | Y   n
2026-02-05 | DOWN/CHOPPY | DOWN/CHOPPY | Y   Y
2026-03-03 | FLAT/CHOPPY | <no-json>
2026-03-19 | FLAT/CHOPPY | <no-json>
2026-03-11 | FLAT/CHOPPY | <no-json>
2026-02-17 | FLAT/CHOPPY |   UP/SMOOTH | n   n
2026-01-02 | FLAT/CHOPPY | DOWN/CHOPPY | n   Y
2026-03-15 |   UP/SMOOTH |   UP/SMOOTH | Y   Y
2026-02-25 |   UP/SMOOTH |   UP/SMOOTH | Y   Y
2025-12-19 | FLAT/SMOOTH | <no-json>
2026-02-13 | FLAT/CHOPPY | FLAT/SMOOTH | Y   n
2026-01-11 | DOWN/SMOOTH | <no-json>
2026-01-01 |   UP/SMOOTH |   UP/SMOOTH | Y   Y
2026-03-02 | FLAT/CHOPPY | <no-json>
2026-02-19 | FLAT/SMOOTH | DOWN/SMOOTH | n   Y
2026-01-09 |   UP/SMOOTH | <no-json>
2026-02-15 |   UP/SMOOTH | <no-json>

DIRECTION accuracy: 9/30 = 30%   (3-class chance ~33%)
VARIATION accuracy: 7/30 = 23%   (2-class chance ~50%)
JOINT (both right): 4/30 = 13%
JSON-format failures: 14/30 = 47%
latency/call: mean 4.4s
