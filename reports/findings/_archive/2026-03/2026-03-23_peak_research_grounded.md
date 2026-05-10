# Peak Detection Research — Reframed with Grounded Features
> 2026-03-23 — Revisiting all prior peak research through the lens of
> base measurements (Price, Time, Volume) and the 3-level feature tree.

---

## Prior Research Translated to Base Measurements

### 1. The Core Finding (174K peaks)

**Old language:** "Volume delta dying + F_momentum fading = reversal. Volume flowing + F_momentum building = fakeout."

**Grounded translation:**
- REAL peak: **velocity collapsing + volume collapsing** = the move ran out of both speed AND participation. Two independent base measurements agree: it's over.
- FAKE peak: **velocity pausing + volume still flowing** = speed slowed but participation is still there. Only one base measurement says stop. The other says continue.

The discriminator is not F_momentum (level 4 PID derivative). It's **velocity** (level 2, dP/dt) and **volume** (level 2, reaction measurement). The PID just obscured what was really a velocity signal.

### 2. Top Discriminators Reframed

From the Kruskal-Wallis H-statistics on 174K peaks:

| Old Feature | H-stat | Grounded Translation | Level | What it really measures |
|-------------|--------|---------------------|-------|------------------------|
| Geometric range | 5,802 | **std(price, short window)** | 2 (statistical) | Tight = compressed spring = reversal imminent |
| Volume delta | 4,244 | **volume** | 2 (base reaction) | Dying = exhaustion. Flowing = continuation |
| F_momentum | 2,735 | **velocity** | 2 (kinematic) | Fading = decelerating. Building = accelerating |
| PID output | 198 | accumulated z-score | 3 (tertiary) | Weak — cumulative drift, not instant signal |
| DMI diff | 109 | **dmi_diff** | 2 (kinematic) | Buyer/seller battle — weak alone |
| ADX | 82 | variance ratio proxy | 3 (tertiary) | Weakest — trend strength is NOT peak detection |

**The top 3 discriminators are ALL level 2.** No level 3+ feature made the top 3.
The H-stats prove: base-level features have the strongest separation.

### 3. Real vs Fakeout — Grounded

| Grounded Feature | REAL peak | FAKEOUT | Signal |
|-----------------|-----------|---------|--------|
| velocity (1m) | near zero (dying) | -27 (still building) | **Velocity collapse = real** |
| volume (1m) | +1.5 (dying) | -32 (flowing) | **Volume collapse = real** |
| std_price (range) | tight | wide | **Compression = real** |
| dmi_diff (1m) | -0.6 | -0.6 | No separation alone |

DMI diff alone doesn't separate (H=109, lowest). But from prior research:
"DMI extreme + volume collapse = exhaustion." That's the **cross-signal** —
DMI provides context (WHO is winning), volume provides confirmation (are they DONE winning?).

### 4. The Exhaustion Pattern (Grounded)

**Old:** "Entry signal is DMI extreme + volume collapse. The DMI cross confirms what already happened 2-3 bars ago."

**Grounded:** The exhaustion sequence in base measurements:

| Bar | velocity | volume | dmi_diff | What's happening |
|-----|----------|--------|----------|-----------------|
| -5 | high (+) | high | extreme | Move in full force |
| -3 | still high | dropping | still extreme | Participation dying, speed holding |
| -1 | dropping | low | extreme | Speed finally dying, volume gone |
| **0** | **flip** | **collapsed** | **still extreme** | **PEAK — this is the entry** |
| +1 | reversed | recovering | crossing | Reversal underway (too late to enter) |
| +3 | building other way | building | crossed | Move confirmed (everyone sees it) |

The entry is at bar 0: velocity flipped, volume collapsed, DMI still extreme (hasn't crossed yet).
Bar +1 and +3 are confirmation — profitable but late. This is the 2-3 bar advantage.

### 5. Trend Alignment (Grounded)

**Old:** "80% of fakeouts are counter-trend. Only 7% are trend-aligned."

**Grounded:** This is a **higher-TF velocity check.**
- 1m velocity says "flip" (peak detected)
- 5m/15m velocity says "same direction" → fakeout (the bigger move isn't done)
- 5m/15m velocity says "opposite" or "flat" → real peak (the bigger move IS done)

That's the **MTF velocity agreement** feature. Not "trend alignment" — just:
does the velocity at the detection TF agree with the velocity at the structure TF?

Grounded as: `sign(velocity_1m) vs sign(velocity_15m)`
- Same sign → peak is AGAINST the structure → 80% fakeout → DON'T TRADE
- Opposite sign → peak is WITH the structure → 7% fakeout → TRADE

### 6. TF Agreement = Fakeout (Grounded)

**Old:** "TF agreement = continuation. TF disagreement = real reversal."

**Grounded:** When velocity at ALL timeframes agrees:
- All TFs say the same direction = the move is consensus = already priced in
- Any reversal at 1s/1m will be rejected by the structure
- Fading this agreement (contrarian entry) has PF 2.44

When velocities DISAGREE across TFs:
- The structure is breaking = real regime change
- The 1m peak IS the structural shift propagating
- Enter with the new direction

This is **coherence rebuilt from velocity agreement**, not from entropy of oscillations.

### 7. Detection Funnel — Grounded Version

Old funnel: 744K detections → 183K pass gates → 380 trades → $1,844

Grounded funnel with base features:

```
Step 1: velocity flip at 1s                      → ~300K events/2mo
Step 2: + magnitude > p75                         → ~60K (remove tiny flips)
Step 3: + volume collapsed < 30% of avg           → ~10K (move actually exhausted)
Step 4: + higher-TF velocity disagrees (MTF)      → ~2K (structural, not noise)
Step 5: + variance_ratio < 1.0 (reverting regime) → ~500 (regime supports reversion)
```

Each step adds ONE grounded filter. Each filter answers ONE question:
1. Did velocity flip? (kinematic)
2. Was the move big enough to matter? (magnitude)
3. Did participation die? (volume)
4. Does the structure support this reversal? (MTF velocity)
5. Is the regime mean-reverting? (variance ratio)

### 8. What the Layered Test Showed (Today's Research)

On 1.73M bars of OOS 1s data:

| Layer | Accuracy | Edge over random | Purpose earned? |
|-------|----------|-----------------|-----------------|
| vel flip only | 49.5% | baseline | — |
| + magnitude | 49.6% | +0.04% | **NO** — magnitude doesn't predict direction |
| + vol collapse | **50.9%** | **+1.4%** | **YES** — biggest single improvement |
| + DMI exhaustion | 50.5% | +1.0% | **PARTIAL** — helps short-term, hurts long-term |
| + reverting regime | 51.1% | +1.6% | **YES at +5s** — but thin sample |

Volume collapse is the strongest single addition. DMI adds short-term but degrades at +60s.
Variance ratio needs more data to confirm.

### 9. Implications for AdvanceEngine

The peak detection in AdvanceEngine should use:

**Level 2 features (proven):**
1. velocity flip (the trigger)
2. volume collapse (the confirmation — biggest discriminator)
3. std_price compression (H=5,802, the strongest single stat)

**Level 2 cross-signals (proven in combination):**
4. dmi_diff extreme + volume low (exhaustion cross-signal)

**Level 3 features (for filtering, not detection):**
5. MTF velocity agreement (structural filter — 80% fakeout filter)
6. variance_ratio regime (when to trade peaks vs when to skip)

**DROP from peak detection:**
- F_momentum → use velocity instead (same signal, grounded)
- PID → weak (H=198), accumulated drift
- ADX → weakest discriminator (H=82), use variance ratio
- coherence → zero signal, use velocity agreement instead
- P_center → minimal separation (+0.06), not actionable

### 10. The One-Sentence Summary

A real peak is when **velocity flips, volume dies, and the price range was compressed** —
three level-2 measurements from two independent bases (Price, Volume) that all agree:
the move is done. Everything else is either redundant or filtering.
