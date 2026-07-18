# NMP9 — the ORIGINAL 2026-04-08 nine-tier ExNMP ladder, ported plain

**Doc 101 · 2026-07-18 · Opus drone (ladder discipline; reviewer = Fable).**
Verbatim port of `nightmare_blended_9tier.py::_classify_full_tier` (git `06d14190`,
recovered at `research/exnmp_lineage_recovered/nine_tier_2026-04-08/`) into the dossier
signal league as 9 plain-named streams, evaluated on **direction agreement with the AI
labels** (train 2024 / test 2025+26, day-block bootstrap CIs, baseline 0.50). This is the
ORIGINAL waterfall, NOT the later 2026-04-18 re-derivation the `NMPT-*` streams port.

> **Metric caveat (project discipline):** every AUC / base below is `P(fire direction ==
> AI-label direction)`, a *direction-agreement* score — NOT $/day and NOT an economic
> edge. base>0.50 = the tier's default direction tends to match the label ("aligned");
> base<0.50 = anti-aligned (the direction is systematically *wrong*, i.e. an invertible
> signal). AUC>0.50 = the shared logistic (pivot-age, leg-agreement, value, tod) can rank
> reliability *within* the tier.

Code: `research/nt8_catalog/tools/dossier_signal_pipeline.py` (append-only NMP9 block) +
runner `research/nt8_catalog/tools/nmp9_league.py`. Raw rows (reviewer reproduction):
`reports/signal_rows_NMP9*.parquet`. Machine table: `tools/nmp9_results.json`.

---

## 1. League table — the 9 NMP9 streams

| # | stream | tier (orig) | N (all) | N test | base (agree) | base 95% CI | OOS AUC | terciles (low / mid / high, mean agree) | fires/day |
|---|--------|-------------|--------:|-------:|-------------:|-------------|--------:|------------------------------------------|----------:|
| 1 | NMP9-CASCADE | CASCADE | 71 | — | 0.17 (raw) | — | — *(N<200, too thin)* | — | 1.16 |
| 2 | NMP9-KILLSHOT | KILL_SHOT | 329 | 174 | **0.172** | [0.124, 0.227] | 0.635 | 0.10 / 0.16 / 0.26 | 1.36 |
| 3 | NMP9-FREIGHT | FREIGHT_TRAIN | 1478 | 1060 | **0.854** | [0.830, 0.874] | 0.638 | 0.79 / 0.84 / 0.94 | 3.87 |
| 4 | NMP9-FADEAGAINST | FADE_AGAINST | 1146 | 396 | **0.758** | [0.715, 0.803] | 0.547 | 0.76 / 0.67 / 0.85 | 3.22 |
| 5 | NMP9-RIDEAGAINST | RIDE_AGAINST | 4015 | 1992 | **0.789** | [0.768, 0.808] | 0.641 | 0.69 / 0.79 / 0.88 | 7.45 |
| 6 | NMP9-RIDEMOM † | RIDE_MOMENTUM | 1146 | 691 | **0.810** | [0.779, 0.840] | 0.636 | 0.73 / 0.81 / 0.89 | 2.59 |
| 7 | NMP9-RIDECALM † | RIDE_CALM | 1885 | 897 | **0.781** | [0.752, 0.809] | 0.603 | 0.74 / 0.75 / 0.86 | 3.82 |
| 8 | NMP9-FADEMOM | FADE_MOMENTUM | 530 | 321 | **0.206** | [0.159, 0.251] | 0.634 | 0.08 / 0.28 / 0.25 | 1.69 |
| 9 | NMP9-FADECALM | FADE_CALM | 836 | 460 | **0.289** | [0.248, 0.330] | 0.561 | 0.23 / 0.30 / 0.34 | 2.18 |

† **λ̂-completed (head seat).** RIDEMOM / RIDECALM's "does the market want to RIDE?" gate
is the exact **NMP-LAMBDA** derivation already in the pipeline (trailing OLS slope, k=21,
of `log(|z_se|+0.1)`): λ̂>0 → RIDE (flip the fade direction); λ̂≤0 (or undefined) → fall
through to FADEMOM/FADECALM. **No SKIP branch** — the original CNN head had a SKIP class;
λ̂ gives no abstain signal, so that tier is *omitted* (documented, doc 101).

**The alignment split is razor-clean and reproduces the known league structure** (MEMORY §5:
ride family aligned, fade family anti-aligned): the RIDE / against-follow family (FREIGHT
0.85, RIDEAGAINST 0.79, RIDEMOM 0.81, RIDECALM 0.78, FADEAGAINST 0.76) sits at
**0.76–0.85 agreement**; the pure-fade family (KILLSHOT 0.17, FADEMOM 0.21, FADECALM 0.29,
CASCADE 0.17-raw) sits at **0.17–0.29** — strongly *anti*-aligned (i.e. the naive NMP fade
is ~75–83% wrong on direction, exactly the doc-084 pure-fade result of 0.26).

---

## 2. NMP9 vs NMPT — the 6 tiers with counterparts

Same evaluation path (`eval_from_parquet`, identical train/test logistic) on the existing
`signal_rows_NMPT*.parquet`. NMPT ports the LATER 04-18 re-derivation (reordered priority,
added MTFEXH/MTFBRK, extra FREIGHT/AGAINST gates, **no `|z|>2` entry gate**).

| tier | NMP9 AUC | NMP9 base | NMP9 fires/day(te) | NMPT AUC | NMPT base | NMPT fires/day(te) | note |
|------|---------:|----------:|-------------------:|---------:|----------:|-------------------:|------|
| CASCADE | — (thin) | 0.17 raw | 1.2 | 0.514 | 0.429 | 2.1 | NMP9 rarer & more anti-aligned (entry-gated) |
| KILLSHOT | 0.635 | **0.172** | 1.4 | 0.552 | **0.399** | 6.0 | NMP9 far rarer, ~2× more anti-aligned |
| FREIGHT | 0.638 | **0.854** | 4.6 | 0.582 | 0.753 | 15.7 | NMP9 simpler def but MORE aligned & cleaner |
| FADEAGAINST | 0.547 | **0.758** | 2.7 | 0.638 | **0.412** | 2.6 | **direction sign differs — see below** |
| RIDEAGAINST | 0.641 | **0.789** | 7.1 | 0.656 | 0.609 | 39.3 | both aligned; NMP9 tighter (entry-gated) |
| FADECALM | 0.561 | 0.289 | 2.3 | 0.676 | 0.421 | 39.9 | different populations; NMPT huge-N default |

**No-counterpart (the 3 recovered tiers):** NMP9-FADEMOM, NMP9-RIDEMOM, NMP9-RIDECALM.
RIDEMOM/RIDECALM were **definitionally lost** in the V1 port (they required the CNN head,
never reachable CNN-free); FADEMOM was the **genuinely droppable-by-accident** tier —
reachable CNN-free yet absorbed into FADECALM (NINE_TIER_EXTRACTION.md).

### The FADEAGAINST reconciliation (a real port-divergence, flagged)
NMP9-FADEAGAINST is **aligned (0.758)**; NMPT-FADEAGN is **anti-aligned (0.412)** — because
they trade *opposite directions*. The original `FADE_AGAINST` (line 476-478) **flips to
follow the 1h z** (`short if h1_z>0 else long`) — it *rides the dominant 1h trend*. The
04-18 re-derivation (`_nmp_tier_events`, `res = (direction, 'FADEAGN', …)`) **kept the fade
direction**. So the two "FADE_AGAINST" streams are direction-inverted ports of the same
condition. The ORIGINAL is the aligned one — the NMPT re-derivation mis-ported the sign.
This is the single most consequential reconciliation between the 79D-era file and the port.

---

## 3. The three questions

**(a) Does FADEMOM separate from FADECALM? (was the V1 absorption a real loss?) — YES, marginally.**
- FADEMOM base **0.206** [0.159, 0.251], AUC **0.634**, N=530.
- FADECALM base **0.289** [0.248, 0.330], AUC **0.561**, N=836.
The 95% agreement CIs **barely touch** (overlap ≈ 0.0024, [0.248, 0.251]) — a near-significant
separation. FADEMOM is materially **more anti-aligned** (0.206 vs 0.289 → 79% wrong vs 71%
wrong on direction) **and more rankable** (AUC 0.634 vs 0.561). i.e. the momentum-fade sub-
population is a *cleaner, more strongly invertible* signal than the calm default. **The V1
absorption of FADEMOM into FADECALM WAS a real loss** — it blended a sharper, higher-AUC,
more-invertible sub-tier into the muddier catch-all.

**(b) Do the λ̂-completed RIDE tiers land in the aligned family? — YES, squarely.**
RIDEMOM **0.810** and RIDECALM **0.781** sit right inside the ride family band (0.76–0.85),
far above 0.50. The λ̂ head does exactly what doc-084 predicted: it **flips an anti-aligned
fade (~0.20 agree) into an aligned ride (~0.80 agree)**. This completes the 9-tier ladder
for the first time — the two tiers that were *definitionally lost* without the CNN are now
reconstituted with λ̂ and land aligned.
- **Honest nuance on λ̂'s selectivity (clean within-bucket test):** RIDEMOM/FADEMOM share
  the `|vel|≥50` bucket, split only by λ̂ sign; RIDECALM/FADECALM share `|vel|<50`.
  - Momentum bucket: λ̂>0 (RIDEMOM, flipped) → 0.810; λ̂≤0 (FADEMOM) is 79% anti-aligned, so
    *if it too were flipped* it would be 0.794. **λ̂ selectivity edge = +0.016** — negligible.
    At high velocity the fade is ~uniformly wrong regardless of λ̂; "flip everything" ≈ λ̂-gate.
  - Calm bucket: λ̂>0 (RIDECALM, flipped) → 0.781; FADECALM-if-flipped → 0.711.
    **λ̂ selectivity edge = +0.070** — real. At low velocity λ̂ sign genuinely separates
    snap-back from continuation.
  - **Takeaway:** most of the RIDE-tier alignment comes from the *flip* (fade is anti-aligned,
    so inverting it wins); λ̂'s *selective* value is concentrated in the **calm** regime and
    is small in the momentum regime. λ̂ is a useful head, but it is not carrying tiers 6-7 on
    its own — the anti-aligned base fade is.

**(c) Does adding the 9 lift the combiner? — marginally, +0.004.**
| combiner pool | streams | pooled OOS AUC |
|---|--:|--:|
| documented baseline (doc 100, stale) | 38 | 0.689 |
| **current on-disk pool (pre-NMP9)** | 46 | **0.672** |
| **+ NMP9 (9 streams)** | 55 | **0.676** |

The honest same-pool delta is **0.672 → 0.676 (+0.004)**. (The documented 0.689 predates the
8 turn/prop streams added since, which dragged the honest pool to 0.672 — it is not a valid
comparison anchor; the +0.004 is measured on the identical 46-stream pool ± NMP9.) The lift is
carried entirely by the **ride family** identity weights — NMP9RIDEAGAINST **+0.084**,
NMP9RIDECALM +0.051, NMP9FADEAGAINST +0.047, NMP9RIDEMOM +0.046, NMP9FREIGHT +0.037 — while
the anti-aligned fade tiers take negative weight (FADEMOM −0.040, FADECALM −0.036, KILLSHOT
−0.034), correctly *down*-weighting P(right) when they fire. **Interpretation:** the ride
tiers add a small amount of real signal, but they **overlap heavily** with ride-family streams
already in the pool (NMPT-RIDEAGN, NMPT-FREIGHT, NMP-LAMBDA all encode the same "ride the
dominant trend / λ̂>0" structure), so the marginal contribution is small. NMP9 is worth
keeping as the *clean, entry-gated, correctly-signed* member of that family, not as a new
independent edge.

---

## 4. Unit / threshold reconciliations (79D-era file → 5s-stream port)

1. **Base z:** original read `feat[10]` = 1m `z_se` (SFE standard-error z). Port uses `_z21`
   (21-bar OLS endpoint z on 1m closes) as the stand-in — the same substitution the NMPT
   port made (map recipe A). ROCHE=2.0 entry threshold transfers.
2. **Velocity units = TICKS.** Original `feat[13]` velocity and constants
   VELOCITY_THRESHOLD=50 / FREIGHT=100 are ticks-based (NINE_TIER_EXTRACTION.md). Port
   velocity = `diff(closes)/TICK` (ticks). Verified consistent — thresholds used verbatim.
3. **Wick thresholds verbatim:** WICK_5M_MIN=**0.83**, WICK_15M_MIN=**0.77** (read from the
   recovered file, not guessed); wick = `1 − |c−o|/range` per `_tf_state`.
4. **1h-against threshold** H1_AGAINST_Z_MIN=**1.5** applies to BOTH h1_z (FADE_AGAINST) and
   h1_vel (RIDE_AGAINST), verbatim from the file (lines 471, 481). The NMPT port instead used
   `h1_vel<-3.0` for its RIDEAGN — another divergence; NMP9 uses the original 1.5.
5. **λ̂ head uses `z_se`, not `z21`.** The base/entry z is `z21`; the λ̂ head is derived from
   the canonical `L3_1m_z_se_15` store (NMP-LAMBDA machinery). Two different z's by design —
   `z21` for the fade base (matches the ladder), `z_se`-λ̂ for the ride/fade head decision.
   If the z_se store is missing for a day, λ̂ is undefined → RIDE never fires → those fires
   fall through to FADEMOM/FADECALM (graceful, documented).
6. **Entry gate restored:** the original gated entries at `|z|>2.0 ∧ vr<1.0` *before*
   classifying (NMPT dropped this). NMP9 restores it — this is why every NMP9 tier fires far
   rarer and cleaner than its NMPT counterpart (e.g. RIDEAGAINST 7/day vs 39/day).
7. **Edge-trigger adaptation:** the legacy fired continuously via position occupancy (a
   trade-management artifact). NMP9 emits on the `(tier, direction)` edge at 1m boundaries —
   identical adaptation and rationale as the NMPT port. Entry-gate misses reset the edge.
8. **FREIGHT simplified to the 04-08 original** (`|vel|≥100` only; no `vr<0.85`, no
   `vel*acc>0`) and priority order = original (CASCADE > KILLSHOT > FREIGHT > FADEAGAINST >
   RIDEAGAINST > head > FADEMOM > FADECALM), NOT the NMPT reordering.

---

## 5. What the reviewer can reproduce
- Any stream's numbers: `eval_from_parquet('reports/signal_rows_NMP9RIDEAGAINST.parquet')`
  reproduces AUC 0.641 / base 0.789 / terciles from the saved rows (rows saved BEFORE gating).
- The whole run: `python3.11 research/nt8_catalog/tools/nmp9_league.py` (streams 604 days,
  576 labeled; ~2.5 min CPU). Combiner delta: `python3.11 .../combiner_preview.py` (globs all
  `signal_rows_*.parquet`).
- **Committed: nothing** (drone discipline). Files touched: `dossier_signal_pipeline.py`
  (append-only NMP9 block), new `tools/nmp9_league.py`, this report, the 9
  `signal_rows_NMP9*.parquet`, and `combiner_preview.md` (regenerated to the 55-stream state).
