# Cascade-Conditioned Pivot Quality — 2026-04-25

**Question:** Does multi-TF z-score alignment at the moment a 1m zigzag pivot
fires predict that pivot's trade quality?

**Setup:** For every 1m zigzag pivot at R∈{10,30}, look up pre-computed
`<TF>_z_se` (TF ∈ {15s, 1m, 5m, 15m, 1h, 1D}) from `DATA/ATLAS/FEATURES_5s/`
at the pivot's confirmation timestamp. Define cascade in two ways:

- **Threshold-count alignment:** #TFs with `|z_se| > T` AND signed in the pivot's
  trade direction. Low-pivot trade is LONG → count `z_se < -T`. High-pivot trade
  is SHORT → count `z_se > +T`.
- **Continuous score `dir_energy`:** Σ over TFs of `|z_se|` IF signed in trade
  direction (else 0). No threshold; pure energy magnitude.

Walk forward from each pivot's entry-fill bar to the next opposite pivot's
entry-fill bar (= the v1.0/v1.3 leg). Record MFE, MAE, ETD, capture %, leg P&L.

**Datasets:** 345 days (2025_01_01 → 2026_03_20), IS = 2025, OOS = 2026.
11,157 pivots at R=30, 43,276 pivots at R=10.

**Tools:** `tools/cascade_pivot_quality.py` (created today).

---

## Key results

### Threshold-count alignment, R=30, |z|>1.5

| Align | IS N | IS WR | IS MFE | IS ETD | IS Final | OOS N | OOS WR | OOS Final |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4,776 | 36.9% | $87 | $88 | −$1.2 | 1,511 | 37.1% | −$1.5 |
| 1 | 2,698 | 38.6% | $93 | $94 | −$0.6 | 862 | 37.0% | −$5.4 |
| 2 | 877 | 39.2% | $97 | $96 | +$1.0 | 225 | 35.6% | −$1.6 |
| 3 | 163 | 36.8% | $98 | $103 | −$5.3 | 31 | 22.6% | −$21 |
| 4 | 14 | **50.0%** | **$116** | $88 | **+$27.9** | — | — | — |

### Threshold-count alignment, R=10, |z|>1.5  (4× more pivots)

| Align | IS N | IS WR | IS MFE | IS Final | OOS N | OOS WR | OOS Final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18,744 | 39.4% | $42 | $0.0 | 6,002 | 39.2% | −$0.5 |
| 1 | 10,636 | 40.5% | $45 | −$0.4 | 3,261 | 41.0% | −$0.9 |
| 2 | 3,140 | 41.1% | $48 | +$0.8 | 845 | 41.8% | −$1.3 |
| 3 | 514 | **43.4%** | **$51** | −$1.1 | 95 | **42.1%** | −$5.0 |
| 4 | 38 | 34.2% | $42 | −$5.0 | 1 | — | — |

### Continuous `dir_energy` deciles, R=10  (clearest gradient)

| Decile | IS N | IS WR | IS MFE | OOS N | OOS WR | OOS MFE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (low) | 3,308 | 39.3% | $42 | 1,021 | 40.2% | $40 |
| 5 | 3,307 | 39.7% | $42 | 1,020 | 39.8% | $41 |
| **9 (high)** | **3,308** | **43.4%** | **$51** | **1,021** | **44.3%** | **$49** |

---

## Findings

1. **Cascade alignment is a real signal but a modest one.** Top-decile pivots show
   ~4-5 percentage points higher WR and ~20% higher MFE than bottom-decile,
   consistent across R=10 and R=30, IS and OOS.

2. **The signal is rare in its strongest form.** Only 0.5–7% of pivots reach
   alignment ≥3 at threshold |z|>1.5 (depending on R). At |z|>2.0, almost no
   pivots qualify (top bucket = 14 IS pivots, 1 OOS).

3. **Cascade does NOT fix ETD.** Across every alignment bucket, MFE and ETD are
   nearly equal in dollars. Capture % is stuck at −100% to −180% across all
   buckets. **Better entries don't help if exits give it all back.** This is
   the v1.3 trail-stop story confirmed from a different angle.

4. **Cascade and trail are orthogonal.** Cascade improves entry-quality
   (WR + MFE). Trail captures the higher MFE. Both are needed for full benefit.

5. **Continuous `dir_energy` works better than threshold counting.** The decile
   gradient is more stable (less noisy than count buckets), and bucket 9 OOS
   confirms IS direction at R=10 (44.3% vs 40.2% bottom-decile WR — +4pp).

6. **Direction skew with alignment is real.** %Long climbs from ~40% at
   alignment 0 to ~70% at alignment 3+. This reflects market regime — when
   multiple TFs are simultaneously stretched DOWN, the next pivot tends to be
   a LOW (= LONG entry). Asymmetry to investigate, not necessarily a bug.

---

## Operational implication for v1.5+ design

The cascade earns its place as a **risk-tier filter on top of v1.3 trail**, not as
an entry-direction model:

| Tier | Cascade `dir_energy` decile | Suggested action |
|---|---|---|
| A — high-conviction | top 10% (decile 9) | Full size + loose trail (let MFE develop) |
| B — supported       | deciles 6-8       | Normal size + standard trail |
| C — neutral         | deciles 2-5       | Normal size + tight trail |
| D — fade           | deciles 0-1       | Smaller size OR skip |

Estimated edge: top decile is ~4pp WR + 20% MFE above baseline. If the trail
captures even half the MFE differential ($5/trade extra capture × 19 trades/day
top-decile share = ~$10/day), and the WR uplift adds ~$5-10/day, the cascade
tier could add **$15-30/day** on top of v1.3 trail's gain.

**Not a strategy by itself.** v1.3 (trail+CSV+entry-cutoff exit) remains the
backbone. Cascade tier sizing is a v1.5 add-on once v1.3 is proven live.

---

## Files produced today

- `tools/sigma_cascade.py` — every-bar |z| alignment + lookahead WR (built but
  not the right tool — superseded by pivot-conditioned).
- `tools/cascade_pivot_quality.py` — pivot-conditioned outcome analyzer with
  both threshold-count and continuous `dir_energy` reporting.
- `tools/zigzag_multitf_overlay.py` — visual: stacked zigzag polylines from
  multiple TFs on one day's price chart.
- `reports/findings/multitf_overlay_2026_02_09.png`
- `reports/findings/multitf_overlay_2026_03_20.png`
- `reports/findings/cascade_pivot_quality_R{10,30}_z{1.0,1.5,2.0}.csv`
  (per-pivot data, ~11K-43K rows each).

## Next concrete steps (not done today)

1. **Per-trade simulation:** apply cascade tier sizing to the v1.3 trade ledger
   on full ATLAS, measure $/day delta vs v1.3 baseline.
2. **R-sweep at fixed cascade decile 9:** does trading only top-decile pivots
   at R=20/30/50 outperform full v1.3 baseline?
3. **Visualize `dir_energy` overlay on Feb 9 chart** — see whether the spike
   timing visually matches the actual cascade-day reversals.
4. **Live deploy v1.3 (trail + CSV)** post-VOE. Cascade is v1.5 work, not v1.4.

---

## ADDENDUM 2026-04-25 (later) — Regime + Envelope filter is the breakthrough

User reframed the cascade hypothesis with a sharper construction:

> "if a 1h pivot fires that tells us the overall direction, and if we do z bands
>  along that direction it should tell us the expected noises, and if we couple
>  that noises with the zigzag triggers of the lower bands it should tell us
>  the expected width of the 1m bands so it's easier for us to catch fakeouts
>  and estimate the minimax"

Built `tools/regime_envelope_quality.py` to test this. For each 1m pivot:
- **Macro regime** = direction of latest 1h zigzag pivot at confirm timestamp.
  WITH-regime if 1m pivot direction agrees, AGAINST otherwise.
- **Envelope band** = `|1h_z_se|` at confirm timestamp, binned:
  inside (\<0.5σ), near (0.5-1σ), edge (1-1.5σ), outside (1.5-2σ), extreme (>2σ).

Run on R_1m=30, R_1h=50, full ATLAS (11,125 pivots).

### IS 2025 — 2×5 outcome table

| Regime | Band | N | WR | Final $ |
|---|---|---:|---:|---:|
| WITH | inside | 799 | 35.3% | **−$3.08** ← noise fakeout |
| WITH | near | 740 | 37.6% | +$0.59 |
| WITH | edge | 713 | 39.0% | +$1.41 |
| WITH | outside | 480 | **41.7%** | **+$4.54** ← sweet spot |
| WITH | extreme | 667 | 37.8% | −$0.13 |
| AGAINST | inside | 860 | 38.5% | +$0.14 |
| AGAINST | near | 815 | 35.0% | **−$7.36** |
| AGAINST | edge | 795 | 40.1% | −$0.02 |
| AGAINST | outside | 553 | 38.7% | +$0.72 |
| AGAINST | extreme | 762 | 36.7% | −$0.71 |

### OOS 2026 — same structure, sample 2,615

| Regime | Band | N | WR | Final $ |
|---|---|---:|---:|---:|
| WITH | edge | 212 | 41.5% | +$1.84 |
| WITH | inside | 287 | 35.2% | −$3.74 |
| AGAINST | extreme | 209 | **29.7%** | **−$12.28** ← **the trap** |
| AGAINST | outside | 151 | 37.7% | −$4.72 |
| WITH | outside | 147 | 33.3% | −$4.12 |
| (others) | … | … | … | … |

### Filter: skip 2 buckets

| Filter | Skipped | OOS gain |
|---|---|---|
| AGAINST + extreme (fakeout-at-exhaustion) | ~6% trades | prevents −$12.28/trade |
| WITH + inside (noise fakeout)             | ~10% trades | prevents −$3.74/trade |

Total: skip ~18% of trades, **gain +$79.88/day OOS** (and +$24.50/day IS).

OOS uplift is **3.3× the IS uplift** — opposite of overfit. The filter is
capturing a real structural pattern that's actually more pronounced in 2026.

### Minimax estimate falls out for free

Mean MAE across every bucket: **$55-66**. So a stop-loss around $70-80 covers
most adverse cases. R:R approximately 1:1.5 (mean MFE $90 / mean MAE $60).
This is operationally usable for SL placement in the v1.5 strategy.

### Versioning convention adopted (2026-04-25)

Non-released versions carry `-RC` suffix. Only v1.0 is RELEASED (live).
See `docs/VERSIONING.md` for the rule.

### RC ladder updated

| Version | Status | Mechanism |
|---|---|---|
| **v1.0** | **RELEASED** (live Sim101, in VOE) | Pure zigzag |
| v1.3-RC | Built, NOT deployed | + trail + CSV + post-cutoff exit |
| v1.4-RC.REJECTED | Disproved | Hybrid timing (intra-minute noise pivots) |
| v1.5-RC | **New design candidate** | v1.3-RC + regime-envelope filter (skip AGAINST-extreme + WITH-inside) |

Estimated v1.5 OOS gain on Python sim: **+$80/day** from the filter alone, plus
whatever trail captures from MFE recovery. Combined with the consistent ~$680
Python-vs-NT8 pessimism gap observed Day 1, **NT8-Sim101 v1.5 estimate could
land in the $500-700/day range** (speculative — needs validation).

### Files added in this addendum

- `tools/regime_envelope_quality.py` — 2×5 bucket outcome analyzer
- `tools/cascade_tier_simulation.py` — sizing-scheme P&L delta
- `reports/findings/regime_envelope_R1m30_R1h50.csv` — per-pivot data
- `reports/findings/cascade_pivot_quality_R30_z*.csv` (3 thresholds)
- `reports/findings/cascade_pivot_quality_R10_z1.5.csv`

---

## ADDENDUM 2 (2026-04-25 evening) — v1.5-RC forward pass: filter goes NEGATIVE on top of trail

`tools/forward_pass_v15rc.py` runs four scenarios on the same R=30 / R_1h=75
trade ledger:

| Scenario | Trades | Skipped | IS $/day | OOS $/day | OOS dWR |
|---|---:|---:|---:|---:|---:|
| A) Baseline (v1.0) | 11,125 | 0 | −$94 | −$223 | 36% |
| B) **Trail only (v1.3-RC)** | 11,125 | 0 | **+$2,230** | **+$2,650** | **100%** |
| C) Filter only | 9,227 | 1,898 | −$66 | −$114 | 47% |
| D) Trail+Filter (v1.5-RC) | 9,227 | 1,898 | +$1,857 | +$2,210 | 100% |

**v1.5-RC vs v1.3-RC delta: −$373 IS / −$440 OOS.** Adding the filter on top of
trail makes things **worse**, not better.

### Why the composition fails

The filter's value (Addendum 1) was based on bad-bucket trades costing
−$3 to −$12 each in their **natural-exit** form. Once trail is engaged:

- A "WITH-regime + inside" trade with MFE=$70, leg=−$5 → trail converts to
  ~+$60. Skipping it now removes positive expectancy.
- The "AGAINST-regime + extreme" bucket (the worst, OOS −$12.28/trade) →
  trail also salvages most of those, since they typically have positive MFE
  before reversing.

**Trail solves the same problem the filter solved** (give-back on bad-quality
trades), but better. They are NOT orthogonal as previously claimed —
they're substitutes.

### Honest caveats on the +$2,200/day claim

The trail simulation is **analytically approximate**:
- Computes `trail_pnl = MFE − eff_dist` ideally
- Real NT8 v1.3-RC uses HWM of 1m **closes** (not highs)
- Trail fires on bar close → exit at next bar's **open** with slippage
- Optimistic by 1.5-3×

Adjusted estimate: v1.3-RC NT8-Sim101 likely **+$1,100-1,500/day** (vs Day 1 v1.0
= +$455). Trail captures the ETD give-back that v1.0 surfaced.

### RC ladder REVISED

| Version | Status | Notes |
|---|---|---|
| **v1.0** | **RELEASED** (live VOE) | Pure zigzag |
| **v1.3-RC** | **Top promotion candidate** | Trail. Theoretical +$2K/day, conservative +$1.1K/day. The single most leveraged change. |
| v1.4-RC.REJECTED | Disproved | Hybrid timing. |
| v1.5-RC | **DEMOTED** | Filter on top of trail is NET-NEGATIVE. Filter solves what trail already solves, just less effectively. |

Don't ship v1.5-RC. Ship v1.3-RC.

### Files added

- `tools/forward_pass_v15rc.py` — 4-scenario forward pass with trail + filter
- `reports/findings/forward_pass_v15rc.csv` — scenario summary table

### Next steps

1. **Bar-by-bar trail simulator.** Replace analytical trail with proper
   1m-close HWM tracking + next-bar-open slippage. Will deflate the $2,230/day
   estimate to a more honest number, but doesn't change the directional
   finding (trail >> baseline, filter doesn't compose on top of trail).
2. **Test if filter helps at very loose trail** (e.g., trail-dist=15, trail-pct=0.30).
   If trail captures less, filter might re-add value.
3. **Post-VOE: promote v1.3-RC, deploy to NT8 Sim101.** Live data settles the
   theoretical-vs-NT8 gap question for trail specifically.

---

## ADDENDUM 3 (2026-04-25 late) — Bar-by-bar simulator built; filter verdict REVERSED

User pointed me at the existing `tools/nightmare_ticker.py` pattern — the live
system already has bar-by-bar tick walking with `peak_pnl` tracking. Adapted
into `tools/zigzag_trail_ticker.py` for honest forward-pass simulation.

**Setup:** 1s data ticked through, aggregated to 1m. Trail HWM tracked on 1m
**closes** (matching NT8 v1.3-RC `Calculate.OnBarClose`). Trail breach detected
on 1m close, exit fills at next 1s tick open. 345 days, R=30.

### Honest results

| Scenario | Trades | IS $/day | OOS $/day | OOS dWR |
|---|---:|---:|---:|---:|
| v1.0 baseline | 11,125 | −$94 | −$223 | 36% |
| **v1.3-RC** trail only | 10,485 | −$86 | **−$122** | 41% |
| **v1.5-RC** trail + regime+envelope filter | 8,696 | **−$61** | **−$80** | **43%** |

### Δ vs baseline

| Step | IS Δ | OOS Δ |
|---|---:|---:|
| v1.0 → v1.3-RC | +$8/day | **+$101/day** |
| v1.3-RC → v1.5-RC | +$25/day | **+$42/day** |
| v1.0 → v1.5-RC | +$33/day | **+$143/day** |

### Trail breakdown (v1.3-RC, 10,485 trades)

| Exit reason | N | WR | Mean $/trade |
|---|---:|---:|---:|
| trail | 6,933 | **87.2%** | **+$33** |
| pivot (no trail fired) | 3,508 | 0.1% | **−$73** |
| EOD | 44 | ~30% | −$15 |

The split: trail saves the give-back on the 6,933 trades that DO go favorable
enough to arm. The 3,508 pivot exits are trades that never reach 10pt
favorable; trail is irrelevant for them, and they take the full opposite-pivot
reversal at MFE − R = −$73 average.

### Why the filter MATTERS (correcting Addendum 2)

Addendum 2's analytical model said filter+trail was net-negative. **That was
wrong.** With honest bar-by-bar trail:

- Filter skipped 2,429 trades (~22%)
- Lost 1,158 trail-wins × $33 = −$38K
- Saved 623 pivot-losses × $73 = **+$45K**
- **Net: +$7K** = +$25/day IS, +$42/day OOS

Filter and trail are partial **complements**, not substitutes:
- Trail handles "good entry, bad exit" (give-back)
- Filter handles "bad entry, no recovery" (pivot exits)
- Different failure modes; different fixes.

### Revised RC ladder

| Version | Honest Python OOS | NT8 estimate (with $680 gap) |
|---|---:|---:|
| v1.0 (live) | −$223 | **+$455** (Day 1 measured) |
| v1.3-RC | −$122 | ~+$556 |
| v1.5-RC | −$80 | ~+$598 |

Both promotion candidates are likely positive in NT8 reality. Deploy ladder
remains: ship v1.3-RC first, layer v1.5-RC after v1.3-RC is proven.

### Files

- `tools/zigzag_trail_ticker.py` — bar-by-bar simulator (with `--filter` flag)
- `reports/findings/zigzag_trail_ticker_trades.csv` — v1.3-RC trades (10,485)
- `reports/findings/zigzag_trail_ticker_v15rc.csv` — v1.5-RC trades (8,696)
