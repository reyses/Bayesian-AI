"""
ADX signal behaviour around LABEL TRANSITIONS (Moises' question, 2026-07-15).

For every ADX signal (doc-074 setting, continuous windows): minutes since the active
label's start, agreement with CURRENT label, agreement with PREVIOUS and NEXT label.
Result 2026-07-15 (576 days, 1,359 in-label signals):
  mins-since-turn   N    agree-CURRENT   agree-OLD
    0-1             9      0.78           0.22
    1-2            80      0.69           0.31
    2-5           621      0.62           0.38
    5-10          318      0.60           0.40
    10-20         179      0.45           0.55
    20-60         138      0.43           0.57
  LATE (>10min) signals: 0.45 vs current, 0.55 vs NEXT label (median 16.4 min before
  the next turn). EARLY (<=5min): 0.63 current / 0.37 next.
=> ADX is a TURN-CONFIRMER whose edge decays with pivot age and INVERTS past ~10 min:
   late ADX signals weakly pre-announce the NEXT turn. The pooled 0.58 hides a 0.62-0.78
   early population and a fade-worthy late population.
CAVEAT: "time since transition" here is ORACLE-clocked (hindsight). A live rule needs a
CAUSAL turn-clock (streaming zigzag pivot age) — the ADX x pivot-age interaction is the
combiner feature to build.
Code: the two inline blocks committed with comms/078 (this file = the record + rerun spec;
reuse signals_for_files from adx_label_overlap.py).
"""
