# Roadmap note — overfit-decay = standard eval; add ZIGZAG + CURVE-REGRESSION dossiers
**Doc:** 075 · **Date:** 2026-07-15 · **Author:** Claude (recording Moises' direction) · **Status:** STANDING

Recording three directions from Moises (2026-07-15). NOT a task this turn — the active
work item remains ADX-08.

## 1. OVERFIT-DECAY becomes the STANDARD shelf-life test (every detector)
For each detector/concept: deliberately OVERFIT its params on an IS window (hard, no
regularization), walk forward, and measure the DECAY CURVE — time until performance
falls below 70% of the in-sample peak. That time-to-70% = the shelf life / retune
cadence, MEASURED not guessed. Multiple rolling anchors -> a distribution, not one number.
NT8 (ATLAS_NT8) stays a sealed one-shot gate; the decay study lives inside Databento IS.
This is a GENERAL evaluation applied to ADX first, then the rest.

## 2. ZIGZAG is added as a dossier/detector
The zigzag leg (ATR(14)x4, min_bars=36, 5s-close pivots — `training/strategies/zigzag.py`)
IS the "displacement -> hold n -> return" object the trend concepts approximate. It joins
the FPS-native detector roster and gets the same treatment (parity port + overfit-decay).
KNOWN SCAR: causal streaming zigzag loses to whipsaws (offline +$454 = hindsight); the
persistence/outlasts-the-ambient-clock filter is the candidate whipsaw filter.

## 3. CURVE (cubic) REGRESSION is added as a dossier/detector
The curve/cubic-regression turn detector (cusp_marker "Cubic N=20" overlay lineage; also
the orange/pink Kalman curve work). Trigger rule = curve inflection / turn. Exact spec TBD
when we reach it (pin down cubic-overlay vs regression-segments before porting).

## Active item (unchanged)
ADX-08: apply the §1 overfit-decay treatment now. Metric decision pending (captured
points/day vs trend-hit-rate). FPS core FROZEN. Batch A verified 7/7; Batch B Sub-1
in progress.
