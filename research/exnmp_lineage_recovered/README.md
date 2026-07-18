# ExNMP lineage — earliest version, recovered from git history (2026-07-17)

Moises asked for the earliest version of the **ExNMP** (extended NMP). Found and
recovered from git history. The ExNMP was born 2026-04-05/06 as a set of
**tiered-exit strategies layered on one NMP engine** — the direct ancestor of
the V1 tier ladder (MTFBRK / MTFEXH / FREIGHT / RIDEAGN / FADECALM) that was
ported into the dossier league this session, and of the ride/fade split.

## Timeline
- **2026-04-04** — base **NMP** engine appears: `nn_v2/nightmare.py` ("NMP engine
  reads 79D from SFE contract") + `nn_v2/tree.py` ("fractures NMP into strategy
  branches" — proto-tiers). Recovered in `base_nmp_2026-04-04/`.
- **2026-04-05** — the term **"ExNMP" first appears**, in
  `docs/Active/MONTECARLO_VALIDATION_SPEC.md` (commit 95800daf): "implement
  after ExNMP pipeline stable."
- **2026-04-06** — the **earliest ExNMP code**: the **Trio** — three tiered-exit
  strategies on one NMP, run by `tools/exnmp_trio_test.py`. Recovered in
  `earliest_exnmp_2026-04-06/`:
  1. **Killshot** — `nightmare_killshot.py` (KillShotEngine; p_at_center exit).
  2. **Wick-Overshoot** — `nightmare_wick_overshoot.py` (wick entry + overshoot
     exit / opposite-extreme momentum).
  3. **Cascade** — `nightmare_cascade.py` (multi-TF resonance: 1m z-extreme +
     5m/15m wick rejection + 1h z-alignment; p_center exit). Header calls it
     "Third ExNMP".
  Unified same day into **`nightmare_blended.py`** (BlendedEngine — "one NMP,
  tiered exits: cascade / killshot / base+overshoot").
- **2026-04-08** — grows to **9 ExNMP tiers + FADE/RIDE/SKIP** (commit 06d14190) —
  this is the tier ladder whose descendants (MTFBRK/FREIGHT/RIDEAGN/FADECALM…)
  were ported verbatim into the dossier league in doc 085 this session.

## What ExNMP was (one line)
One NMP entry (regression-band z-extreme + wick rejection), split into
NAMED tiers by multi-TF alignment strength, each tier with its own exit
(killshot / overshoot / cascade / base) — the transparent, tiered version of
the master equation `|Z|>Z* ∧ λ<0 → fade; λ>0 → ride`.

## Lineage to today
`nightmare.py` (NMP) → the Trio (killshot/overshoot/cascade) → `nightmare_blended.py`
(BlendedEngine, tiered) → 9-tier ExNMP ladder → **V1 tier ladder ported to the
dossier league (doc 085)** + **NMP-LAMBDA λ-completion (doc 084: pure-fade 0.26 →
+λ̂ 0.54)**. Note `nightmare_blended.py` still lives today as a compat SHIM
(`training/nightmare_blended.py` → frozen snapshot `docs/reference/
nightmare_blended_2026_05_20.py`); this folder holds its ORIGINAL 2026-04-06 form.

## Provenance
Recovered via `git show <commit>:<path>`. base NMP ← 18905620/9beecd77 (04-04);
trio ← ce7f8c10 (killshot) / 5e09508a (overshoot) / 5d8aa877 (cascade + trio
runner) (04-06); blended ← 65b257b5 (04-06); spec ← 95800daf (04-05).
Superseded by the ported tier ladder + NMP-LAMBDA; kept as the documented origin.
