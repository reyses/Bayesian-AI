# CORRECTION — my "72% fragility" finding was contaminated by MY OWN cold-start bug
**Doc:** 073 · **Date:** 2026-07-14 · **Author:** Claude (executor) · **Status:** FINAL
**Corrects:** doc 072 §1. Same error class I have been rejecting AG for. Owning it.

## 1. Both bugs FIXED (doc 072 §3, §4)
- **Harness**: removed `verify_batch_b.py`'s module-level `sys.stdout = open(...,"w")`.
  Importing it no longer hijacks stdout; runs no longer truncate prior evidence.
- **Cold start**: root cause was the detectors gating on RTH **before** updating buffers —
  every overnight bar was DISCARDED, so the windows needed 408 RTH bars (~34 min) to warm.
  Per Moises: indicators now update on EVERY bar (overnight included), never reset, and the
  detector instances **persist across days/months** — the rolling window streams unbroken.
  Only the TRIGGER is RTH-gated. **There is no cold start anywhere, ever.**

## 2. THE FIX CHANGED THE ANSWER — retract doc 072 §1
Same 25 days, both variants now fully warm:
```
                  BEFORE (cold-start bug)          AFTER (continuous window)
detector          MATCH DIV nat-only leg-only      MATCH DIV nat-only leg-only
ADX-08-SMA            3   0        0        1          3   0        0        1
ADX-08-WILDER         0   2       15        2          0   0        0        4
SMA-vs-WILDER disagree   18/25 = 72%                        3/25 = 12%
```
**Wilder went from firing on 17 of 25 days to firing on ZERO.** Its cold-started RMA was
producing inflated ADX values that spuriously crossed 25. My "72% disagreement → the
smoothing is load-bearing → ADX-08 is a threshold artifact" headline was **measuring my own
bug**, not the indicator. **Retracted.**

## 3. The HONEST finding, from warm windows
- **ADX-08-SMA**: faithful to legacy (3 match, **0 diverge**). It fires ~3-4 times per 25 days.
- **ADX-08-WILDER**: fires **0 times in 25 days** — Wilder-smoothed ADX on 5s bars essentially
  never exceeds 25.
- So the smoothing choice is still decisive, but in the OPPOSITE direction and for a different
  reason: it is not "the signal jumps around", it is **"Wilder gives you no strategy at all."**
  Under SMA, ADX-08 exists; under canonical Wilder, ADX-08 does not fire.
- That is a real and useful result for the keep-both ruling — but it is NOT the fragility
  story I told in doc 072, and the difference between the two stories is a bug I introduced.

## 4. Still open (do not let these ride)
- **ADX-08-SMA's 1 leg-only day did NOT go away** with the continuous window. So that miss is
  NOT a warmup artifact — cause unknown, now cleanly isolated. Diagnose it.
- **CROSS-11 unchanged** (18 match / 3 diverge / 3 native-only) — its residual 6/25 still needs
  root-causing.
- The cold-start fix must be applied to **every** rolling-window detector (ATR/DOW/FIB/Batch-A),
  not just ADX. Any detector that gates before updating has the same defect.

## 5. The lesson, applied to myself
I rejected AG three times for declaring results from runs it had not scrutinised (docs 060,
066, 070). I then published a headline finding computed on a detector I had written that
threw away every overnight bar. The rule is the rule: **a result from unverified plumbing is
not a result.** The only reason this was caught within the hour is that Moises ordered the
plumbing fixed — the fix, not the analysis, is what falsified the claim.
