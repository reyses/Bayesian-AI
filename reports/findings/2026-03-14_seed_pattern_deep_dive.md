# Seed Pattern Deep Dive — Jan 5-7, 2025 (255 human seeds)
> Generated 2026-03-14

## Market Narrative (1h)
Textbook accumulation -> distribution -> crash cycle over 48 hours.
- Phase 1: Overnight grind up (Sun 23:00 - Mon 06:00, +68t)
- Phase 2: London impulse (Mon 07:00 - Mon 13:00, +802t)
- Phase 3: US shakeout + blow-off top (Mon 14:00 - Mon 16:00)
- Phase 4: Waterfall crash + V-bounce (Mon 16:00 - Mon 21:00, -841t then +485t)
- Phase 5: Overnight distribution (Mon 21:00 - Tue 06:00, -283t)
- Phase 6: Balance zone / coiling (Tue 07:00 - Tue 13:00, range 21720-21780)
- Phase 7: Capitulation cascade (Tue 13:00 - Tue 20:00, -1373t T15 = biggest seed)
- Phase 8: Dead cat bounce (Tue 20:00 - Tue 23:00, +133t)

## Shape Distribution

| Shape | 1h (18) | 15m (39) | 5m (198) | Cross-TF |
|-------|---------|----------|----------|----------|
| V_REVERSAL | 11% | **56%** | **34%** | Dominant at all scales |
| IMPULSE | 28% | 21% | 30% | Fast directional bursts |
| RAMP | 22% | 8% | 13% | Steady grinds during trends |
| COMPRESSION | 17% | 3% | 15% | Noise zones |
| FAKEOUT | 6% | 3% | 2% | Rare but important |
| EXHAUSTION | 11% | 0% | 1% | Peak then giveback |
| TREND_CONT | 6% | 10% | 0% | Prior move extends |

## Key Findings

### 1. V_REVERSAL is the dominant pattern (56% at 15m, 34% at 5m)
Human marker naturally identifies inflection points where price reverses from lookback trend.
V_REVERSAL -> V_REVERSAL is the most common transition (50% probability).
Clean V_REV signature: late MFE timing + low MAE (<30t) + eff>0.8 + 5-9 bars.

### 2. Session character matters enormously
- ASIA: many small clean trades, compression dominant, 60% have MAE<10t
- EUROPE: sweet spot, balanced IMPULSE + V_REV, best R:R
- US: biggest moves (172t avg at 5m, 489t avg at 15m) but messiest (MAE/MFE=1.28)
- US cash open (14:00 UTC) is the most volatile single hour

### 3. Quality scales with duration
- 15m late-MFE seeds (MFE in last 30% of duration): 72% efficient, 74% are V_REVERSAL
- 5m seeds >20min: MFE/MAE = 1.91 vs 1.40 for <10min seeds
- 1h seeds: best R:R at 3-5 hour duration (London session ramps)

### 4. Cross-TF fractal nesting confirmed
- 1h V-reversal contains 2-3 15m V-reversals
- Each 15m V-reversal contains 3-5 5m V-reversals
- Same shape appears at every scale = fractal self-similarity

### 5. Direction alternation = 97% at 5m, 77% at 15m, 65% at 1h
- Higher TFs show multi-seed trend streaks (the LONG run T1-T4 at 1h)
- At 5m, non-alternating streaks (3% of transitions) are the highest quality:
  2.25x higher MFE, much lower MAE. These mark genuine trend moments.

### 6. Whipsaw zones identified
- 1h T11-T13: 3 consecutive 60min flips (121/163/128t) = balance zone
- 5m T38-T43: overnight chop, avg eff=0.38 = noise
- These represent the "coiling before breakout" pattern preceding T15's crash

### 7. Fakeouts are rare (3%) but concentrated
- All in US session
- Typical: strong initial MFE then >60% retracement in first 30% of duration
- None ended in net loss = entry timing was good, hold timing was the issue

## Expected Shape Primitives for Builder

| Primitive | Source TFs | Shape | Confidence |
|-----------|-----------|-------|------------|
| Clean V-Reversal | 15m, 1h | V_REV, late MFE, eff>0.8 | HIGH |
| Fast Impulse | 5m, 15m | IMPULSE, 2-3 bars, high mono | HIGH |
| Overnight Ramp | 1h, 15m | RAMP, Asia, low volume | MEDIUM |
| Waterfall Crash | 1h, 15m | Monotonic sell cascade | HIGH |
| Balance Zone Chop | 5m, 15m | COMPRESSION, low eff | NOISE |
| US Open Shakeout | 1h | FAKEOUT then continuation | MEDIUM |

## Human Marking Signature
- Identifies inflection points (mean-reversion eye), not trend continuations
- Cleanest marks at 15m (efficiency 0.78 avg)
- Catches all major turning points (T15 crash, T3 London impulse, T6 blow-off)
- Over-marks at 5m in chop zones (97% alternation)
- Weakness: whipsaw zones (eff<0.15) get marked alongside quality patterns
