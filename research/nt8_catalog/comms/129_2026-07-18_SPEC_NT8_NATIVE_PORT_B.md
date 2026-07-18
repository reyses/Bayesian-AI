# SPEC — Interim NT8 NATIVE PORT (Architecture B, Moises-confirmed) — doc 129
**Doc:** 129 · **Date:** 2026-07-18 · **Author:** Claude Fable · **Status:** SPEC v1
Moises confirmed B (docs 127 + direct go). Reviewer pushback is on the record
(126); executed now WITH the safety harness the risk profile demands.
The system being ported (docs 107/125): entry P + R-trigger exit + sizing.
NO cut logic, NO turn detection, NO entry filter beyond P.

## Phased plan — parity harness is the spine
**P0 — GOLDEN VECTORS (no-regret; drone dispatched):** the python reference
decider emits per-1m-bar golden records over 20 reference days (2024+2025 mix):
timestamp, top-K stream fire states, P value, entry decision, R-trigger state,
zigzag pivots. File: research/nt8_port/golden/*.parquet + a generator tool.
Every later C# component validates against these vectors bar-by-bar.
**P1 — Entry port (reduced):** the top-K combiner streams by |coef| (K chosen
so cumulative |coef| ≥ 80%) re-fit as a compact 2024-sealed model; C# port of
those K generators + the logistic; thresholds QUANTILE-MATCHED on 2024 (the
NMP9 lesson: never transplant raw thresholds). Parity bar: ≥99% decision
agreement + P within tolerance on golden days, else no advance.
**P2 — R-trigger native:** adapt the RELEASED ZigzagRunner v1.0 (.cs exists,
proven live) — same ATR(14)x4 / 5s-close / min_bars=36 constants; parity vs
golden pivots.
**P3 — SIM parity:** NT8 SIM vs python sim, same days; report under the
multi-gap-assumption rule (0/30/60/100% of the known ~$680/day gap).
**P4 — live SIM (intervention rules ON) → micro live.** Versioning + deploy
gate per revision; nothing enters Strategies/ without per-revision approval.
Sizing: v1 = fixed size (B9 continuous sizing deferred to v2 — model hosting
in NinjaScript is its own project; do not couple it to the port's critical path).

## Standing rules
Each phase = its own reviewer gate with parity evidence. C# files live in
docs/nt8/ as versioned -RC files per the versioning policy. python3.11 for
all reference tooling. research/nt8_port/ = the project folder (WoW layout).
