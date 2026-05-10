# PRIORITY SHIFT — Trends Preferred (2026-05-10)

## Decision

The system's primary edge is in **trend-following / ride-direction** signals,
not in fade-mean-reversion. All FADE tiers retuned today buy modest OOS
uplift ($658 across 3 tiers). All RIDE tiers in the current pipeline are
structurally OOS-broken. The framework needs to pivot.

## Evidence

```
TREND/RIDE EDGES (validated, OOS-stable):

1. MA-alignment (2026-05-01)
   7-of-8 TF vwap_w alignment → 70.5% directional accuracy
   20% of 5m bars qualify, +17.6% lift over baseline
   DETERMINISTIC rule, walk-forward stable

2. HL RM cat-event harvest (2026-05-09/10)
   96% of |max_z|>=6 events are CRASHES
   Pre-positioning SHORT in Tue UTC 1 zone: 40% P_cat in next 60m
   Per-cell EV $185-$260 per event when materializes
   Direction is the harvest, not the fade

3. Compression-bounce (2026-05-10)
   L2_15m_vol_sigma_12 below -3σ from native RM
   P_up_OOS = 64% (vs 50% baseline)
   Mean OOS return +$6.82 per event, n=95 OOS
   The bounce IS the trend post-compression

4. Chord-shape table (2026-05-09)
   STEEP_CONCAVE_UP within STEEP_LINEAR_UP at sub_motif → 68% UP
   FLATLINE-after-rally at motif → 74.6% UP
   These are trend continuation cells, not fades

FADE EDGES (validated, small):

1. FADE_CALM + retune: +$280 OOS uplift
2. FADE_MOMENTUM + retune: +$128 OOS uplift
3. NMP_FADE_RAW + retune: +$250 OOS uplift
Total fade uplift: +$658 OOS

THE TREND EDGES, if properly wired, dwarf the fade edges.
```

## What changes

```
BEFORE                                  AFTER
═══════                                  ═════
9 ExNMP tiers, fade-dominant            Trend-dominant pipeline
FADE_CALM, FADE_MOMENTUM,              MA_ALIGN_TREND (NEW, primary)
FADE_AGAINST, NMP_FADE_RAW (4)         CAT_HARVEST_RIDE (NEW, harvest)
all firing on z-extremes                COMPRESSION_BOUNCE_LONG (NEW, OOS-validated)
                                        STRONG_TREND_RIDE (NEW, multi-TF aligned)
RIDE_CALM, RIDE_MOMENTUM,
RIDE_AGAINST, NMP_RIDE_RAW (4)         Fades become SECONDARY:
all structurally OOS-broken              FADE_CALM (retuned, $280 OOS)
                                        FADE_MOMENTUM (retuned, $128 OOS)
KILL_SHOT, CASCADE,                     NMP_FADE_RAW (retuned, $250 OOS)
FREIGHT_TRAIN (3)                       
fire at wick + cascade events          Wick tiers stay (different framework):
                                        KILL_SHOT, CASCADE, FREIGHT_TRAIN

Total: 12 NMP+wick tiers,              Total: 4-7 trend + 3 fade + 3 wick tiers
mostly losing on OOS                   each independently OOS-validated
```

## The four trend tiers to build

### TIER 1: MA_ALIGN_TREND  (highest confidence — already validated)

```
SIGNAL: 7-of-8 TFs (5s, 15s, 1m, 5m, 15m, 30m, 1h, 4h) of vwap_w aligned
DIRECTION: LONG if 7+ TFs show price > vwap, SHORT if 7+ show price < vwap
EVALUATED AT: every 5m bar close (matches the 2026-05-01 finding)
EXPECTED: 70.5% direction accuracy on ~20% of 5m bars

EXISTING IMPLEMENTATION HINT: tools/v2_composite_ma_alignment.py 
                                already exists from 2026-05-01 research
```

### TIER 2: CAT_HARVEST_RIDE  (validated 96% crash bias)

```
SIGNAL: 10 min before known cat-harvest window opens (Tue/Wed/Thu UTC 1-2)
DIRECTION: SHORT (96% crash bias) — pre-position
SIZE: scaled by TOD risk multiplier (already 0.1-0.3 in cat zones)
EXIT: BayesConditionalExit at peak (or post-window time stop)
EXPECTED: ~50-100 trades/year, mean +$185-$260 per cat event when materializes

REUSE: cat_harvest_signal() in training_iso_v2/filters/bayes_filters.py
```

### TIER 3: COMPRESSION_BOUNCE_LONG  (validated single cell)

```
SIGNAL: L2_15m_vol_sigma_12 < (native_rm - 3sigma) for >= 60s
DIRECTION: LONG always (bounce-after-compression bias)
EXPECTED: ~400 events/year (from full 90-feature scan), 
          mean +$3 IS / +$7 OOS, P_up_OOS = 0.64, n_oos = 95

REUSE: CompressionBounce class in training_iso_v2/filters/bayes_filters.py
```

### TIER 4: STRONG_TREND_RIDE  (multi-TF velocity confirmation)

```
SIGNAL: |L2_1h_price_velocity_12| >= threshold AND
        L3_15m_hurst_12 >= 0.60 (trending) AND
        L2_15m_price_velocity_12 sign matches 1h velocity sign
DIRECTION: WITH the velocity sign (long if positive, short if negative)
EXPECTED: Trend-aligned entries on strong-trend bars
          Per 2026-05-04 prior work: deserves separate test
```

## Implementation order (one tier at a time per CLAUDE.md)

```
RANK   TIER                    REUSE_EXISTING_CODE      VALIDATION_NEED
1.     MA_ALIGN_TREND          v2_composite_ma_align    re-validate OOS
2.     COMPRESSION_BOUNCE_LONG bayes_filters.CompressionBounce  IS+OOS sim
3.     CAT_HARVEST_RIDE        bayes_filters.cat_harvest_signal IS+OOS sim
4.     STRONG_TREND_RIDE       (new build)              full validation
```

## What stays / disappears

```
STAYS (retuned):
  FADE_CALM      +$280 OOS uplift, deploy as secondary
  FADE_MOMENTUM  +$128 OOS uplift, deploy as secondary
  NMP_FADE_RAW   +$250 OOS uplift, deploy as secondary

STAYS (different framework — wick/cascade-based):
  KILL_SHOT, CASCADE, FREIGHT_TRAIN — separate research track

DISAPPEARS (structurally OOS-broken):
  RIDE_CALM     n=77, no positive-on-both-splits band
  RIDE_MOMENTUM n=73, same
  RIDE_AGAINST  n=845, OOS -$1,981
  NMP_RIDE_RAW  n=281, no optimal band
  FADE_AGAINST  E1 hurt OOS (-$60), no veto cells

REPLACED:
  4 broken RIDE tiers → MA_ALIGN_TREND + STRONG_TREND_RIDE
  Hopes pinned on fade harvest → CAT_HARVEST_RIDE + COMPRESSION_BOUNCE_LONG
```

## The expected impact

```
CURRENT BASELINE (pre-retune):
  Fade tiers (4):     ~$684 + $2502 + $1398 + $5584 = $10,168 IS
                      ~$261 + $163 + $145 + $90      = $659 OOS  ★ structural
  Ride tiers (4):     all OOS negative, ~-$3,000 OOS aggregate

POST-RETUNE FADE (today's work):
  Fade tiers (3 deployable): +$658 OOS uplift over baseline

POST-TREND-PIVOT (this proposal):
  MA_ALIGN_TREND:      ?? — 70.5% acc on 20% bars implies LARGE edge
  COMPRESSION_BOUNCE:  +$650/year OOS estimate (95 OOS events × $6.82)
  CAT_HARVEST:         +$10-30k/year estimate (per earlier harvest analysis)
  STRONG_TREND_RIDE:   unknown, need build

  REMOVING BROKEN RIDE_*:  removes ~$3,000 OOS structural drag
```

## Concrete next steps

1. **Don't deploy the today's retune yet.** It's wired but secondary.
2. **Build MA_ALIGN_TREND first** — highest confidence, already-researched
3. **Re-validate on 2026 OOS only** (vs the 2025 IS where it was found)
4. **If MA_ALIGN_TREND validates, deploy it BEFORE the fade retunes**
5. **Then add COMPRESSION_BOUNCE + CAT_HARVEST in rank order**
6. **DELETE or DISABLE the 4 broken RIDE tiers from the engine roster**

## Risk

- **The fade retunes today still represent VALIDATED edge.** Don't lose them by deprioritizing the work.
- **MA_ALIGN_TREND was found IS-only in May 2026** — need fresh OOS validation since 2026 data extends past that.
- **CAT_HARVEST's 96% crash bias was on FULL data** — needs OOS-only confirmation before sizing positions on it.
- **The pivot adds 4 new tiers** — implementation + validation work ~2-3 sessions.
