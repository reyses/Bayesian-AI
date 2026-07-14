# Execution Report — ADX-08 variants + CROSS-11 restored (CLAUDE-executed)
**Doc:** 072 · **Date:** 2026-07-14 · **Author:** Claude (executor, on Moises' instruction) · **Status:** RESULTS — for Moises

Executed doc 071 myself. Tool: `tools/matrix_adx_cross.py` (25-day sample, strict match =
timestamp AND mode both equal).

## 1. HEADLINE — the smoothing is LOAD-BEARING. ADX-08's signal is fragile.
```
detector          MATCH  DIVERGE  nat-only  leg-only  both-0
ADX-08-SMA            3        0         0         1      21
ADX-08-WILDER         0        2        15         2       6
CROSS-11             18        3         3         0       1

ADX SMA-vs-WILDER disagree on 18/25 days (72%)
  2024_01_23: SMA bearish @1706033135 | WILDER bearish @1706021945  (3.1 HOURS earlier)
  2024_01_25: SMA none                | WILDER bearish
  2024_01_26: SMA none                | WILDER bullish
```
- **ADX-08-SMA reproduces legacy exactly**: 3 match, **0 diverge** (1 leg-only = warmup, §3).
  The port is faithful. Good.
- **ADX-08-WILDER fires on ~17 of 25 days vs SMA's ~4** — a 4-5× increase in trigger
  frequency, and when both fire they can be HOURS apart.
- **The two variants disagree on 72% of days.**

**This is the answer to the ruling, and it is more interesting than either variant winning.**
"ADX > 25" is NOT a well-defined condition until you pin the smoothing. The same article
rule, the same threshold, produces a rare signal under SMA and a frequent one under Wilder.
Any edge ever attributed to ADX-08 is contingent on an arbitrary implementation choice
nobody documented. **A signal that changes on 72% of days when you swap a smoothing
function was never a signal** — it is a threshold artifact. Keeping both (Moises' call) is
what exposed this; picking one would have hidden it.

## 2. CROSS-11 — the design-vs-defect ruling is VINDICATED
Restoring first-cross-only (legacy's `# Scan for first cross`): **18 match / 3 diverge /
3 native-only** (~76% agreement) versus AG's all-crosses version at **113 match / 357
diverge (~24%)**. The `break` was the rule, exactly as ruled in doc 070. Residual 6/25
days remain — real, not slop; next step is to diagnose those, not excuse them.

## 3. ⚠ SYSTEMIC — we are about to commit the SAME bug we condemned in legacy
ADX-08-SMA's single miss is a **cold-start warmup gap**: the detector needs 240 (SMA20) +
168 (ADX) bars before it can fire, so it is BLIND for the first ~34 minutes of RTH. Legacy
computed its rolling windows over the FULL day and then sliced RTH — so legacy is warm at
08:30 and sees those triggers.
We rejected DOW-19's legacy for *"discarding the first 20 RTH bars"* (doc 068). A cold
cold-start blinds us for **408 bars**. Same defect, our side of the fence.
**Requirement (all detectors, not just these):** any detector with a rolling window must be
SEEDED with prior-day + today's ETH bars — exactly the fix already applied to CROSS-11 —
or it is not causally equivalent to the concept, it is a concept-that-starts-at-09:04.

## 4. ⚠ The verification harness destroys its own evidence
`tools/verify_batch_b.py:3-4` hijacks `sys.stdout` **at module level**:
`out_file = open("verifier_output.txt","w"); sys.stdout = out_file`.
- Merely IMPORTING the verifier silently redirects the whole process's output (this cost me
  several diagnostic cycles and produced zero-byte runs).
- It opens `"w"` → **every run truncates the previous run's output.**
A verification harness whose evidence is destroyed on the next invocation cannot support an
artifact-level review loop. **Fix required**: never hijack stdout; write to a timestamped
file AND stdout.

## 5. Standing / next
- Both ADX variants are implemented and registered (`ADX08_SMA_Detector`,
  `ADX08_Wilder_Detector`); CROSS-11 first-cross-only restored, with the rationale in the
  code so it is not "fixed" again.
- OPEN: (a) seed ALL rolling-window detectors (§3) then re-run — expect SMA's leg-only to
  vanish; (b) diagnose CROSS-11's residual 6/25; (c) fix the harness (§4).
- The ADX fragility (§1) is a CATALOG-LEVEL finding, not a detector bug. It belongs in the
  final verdict regardless of which variant is used.
- FPS core FROZEN — untouched (verified).
