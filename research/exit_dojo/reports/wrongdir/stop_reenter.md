# Stop + Re-entry sim (task 103) -- Moises' "oops, re-enter" mechanism

Mechanism: adverse-drawdown stop at X ticks (bail when favorable-signed drift <= -X, measured from the ORIGINAL entry -- the doc-100 rule), then RE-ENTER the same direction when drift recovers to >= bail_level + M ticks; cap re-entries at B. Friction = **2.4t/leg** (0.6pt MNQ round trip) on the original AND every re-entry leg. net vs never-bail extends score_wrongdir.net_ticks_vs_neverbail (drift in points x4 = ticks); the original round trip cancels against never-bail's one, so only re-entry legs pay incremental friction.

> **1m-granularity caveat**: the drift series is per-minute. Intrabar stop/trigger crossings are invisible -- a real stop would fire earlier and often deeper, and a real re-entry trigger could fire and reverse within a minute. ALL numbers below are 1m-resolution estimates and are OPTIMISTIC about clean fills.

BAND=4pts (WRONG=terminal<=-4, GOOD>=+4); DIP=4pts. Tuning = 2024 train split (disjoint from the 2025-26 test); test = the 198 doc-100 scored episodes.

## 1. Full 2024 tuning grid (X x M x B), sorted by mean net ticks/ep
Population: 200 balanced one-per-day episodes (100 wrong / 100 good), same select_wrongdir machinery.

| X (t) | M (t) | B | mean net | median | mode | #re-entered | #churn(>=2 bails) |
|---|---|---|---|---|---|---|---|
| 48 | 4 | 1 | +17.3 | +0.0 | +2.0 | 88 | 65 |  <-- WINNER
| 48 | 16 | 1 | +17.0 | +0.0 | +2.0 | 83 | 59 |
| 24 | 16 | 2 | +16.4 | +0.0 | +2.0 | 100 | 67 |
| 24 | 16 | 1 | +16.4 | +0.0 | +2.0 | 100 | 67 |
| 48 | 8 | 1 | +16.3 | +0.0 | +2.0 | 85 | 63 |
| 32 | 16 | 1 | +16.1 | +0.0 | +2.0 | 94 | 66 |
| 24 | 4 | 1 | +16.1 | +0.0 | +2.0 | 110 | 80 |
| 24 | 8 | 1 | +15.8 | +0.0 | +2.0 | 105 | 76 |
| 32 | 4 | 1 | +15.6 | +0.0 | +2.0 | 101 | 73 |
| 32 | 8 | 1 | +15.1 | +0.0 | +2.0 | 97 | 71 |
| 16 | 16 | 2 | +14.8 | +0.0 | +2.0 | 110 | 78 |
| 32 | 16 | 2 | +14.7 | +0.0 | +2.0 | 94 | 66 |
| 8 | 16 | 2 | +14.5 | +0.0 | +2.0 | 121 | 85 |
| 16 | 8 | 1 | +14.4 | +0.0 | +2.0 | 114 | 85 |
| 16 | 16 | 1 | +14.1 | +0.0 | +2.0 | 110 | 78 |
| 16 | 4 | 1 | +13.6 | +0.0 | +2.0 | 118 | 89 |
| 8 | 8 | 1 | +13.6 | +0.0 | +2.0 | 123 | 91 |
| 48 | 16 | 2 | +13.6 | +0.0 | +2.0 | 83 | 59 |
| 48 | 8 | 2 | +13.3 | +0.0 | +2.0 | 85 | 63 |
| 8 | 4 | 1 | +12.8 | +0.0 | +2.0 | 127 | 95 |
| 8 | 16 | 1 | +12.7 | +0.0 | +2.0 | 121 | 85 |
| 48 | 4 | 2 | +12.4 | +0.0 | +2.0 | 88 | 65 |
| 32 | 4 | 2 | +12.4 | +0.0 | +2.0 | 101 | 73 |
| 24 | 4 | 2 | +11.7 | +0.0 | +2.0 | 110 | 80 |
| 8 | 8 | 2 | +11.4 | +0.0 | +2.0 | 123 | 91 |
| 32 | 8 | 2 | +11.2 | +0.0 | +2.0 | 97 | 71 |
| 16 | 8 | 2 | +11.2 | +0.0 | +2.0 | 114 | 85 |
| 24 | 8 | 2 | +10.7 | +0.0 | +2.0 | 105 | 76 |
| 16 | 4 | 2 | +9.8 | +0.0 | +2.0 | 118 | 89 |
| 8 | 4 | 2 | +8.8 | +0.0 | +2.0 | 127 | 95 |

### Plain-stop grid on the SAME 2024 population (friction cancels -> = doc-100 convention)
| X (t) | mean net | median |
|---|---|---|
| 48 | +6.8 | +0.0 |  <-- best-X
| 24 | +6.5 | +0.0 |
| 32 | +5.4 | +0.0 |
| 16 | +0.5 | +0.0 |
| 8 | -1.1 | +0.0 |

**FROZEN**: re-entry **X=48, M=4, B=1** (2024 mean +17.3 t/ep); plain-stop best-X **X=48** (2024 mean +6.8 t/ep).

## 2. FROZEN evaluation on the 198 test episodes (single shot)

| policy | mean net (ticks/ep) | median | mode |
|---|---|---|---|
| never-bail (reference) | +0.0 | +0.0 | +0.0 |
| blind agents (doc-100) | +7.5 | +0.0 | +2.0 |
| plain-stop best-X (X=48, frozen 2024) | +17.0 | +0.0 | +2.0 |
| plain-stop same-X (X=48) | +17.0 | +0.0 | +2.0 |
| **stop+re-entry (X=48,M=4,B=1)** | **+11.0** | +0.0 | +2.0 |

Reference: doc-100 plain-stop best-X on the 198 = +17.7 @ X=24 (wider grid); on this grid the 198 plain-stop best-X = +17.7 @ X=24 (report-only; NOT the frozen bar).

### Day-block bootstrap deltas (198 distinct days; 4000 resamples; * = CI excludes 0)
- **re-entry - plain-stop best-X (X=48, frozen)** = -6.0 [95% CI -16.8,+6.3]  <- PRE-REGISTERED BAR
- re-entry - plain-stop same-X (X=48)           = -6.0 [95% CI -16.8,+6.3]  (isolates the re-entry add-on at fixed X)
- re-entry - never-bail                            = +11.0 [95% CI -12.2,+34.5]

### PRE-REGISTERED VERDICT: **FAIL**
Bar: stop+re-entry retained ONLY if test net > plain-stop best-X AND the delta CI excludes 0. Delta = -6.0 t/ep, CI [-16.8,+6.3]. CI includes 0 (or delta <= 0) -> FAIL: re-entry does NOT beat the plain stop.

## 3. Per-class breakdown -- the dipped-good knife, before vs after re-entry
| class | N | plain-stop (X=48) mean | re-entry mean | delta | re-entry mode |
|---|---|---|---|---|---|
| WRONG | 100 | +109.3 | +74.3 | -35.0 | +2.0 |
| GOOD-dipped | 48 | -157.8 | -109.4 | +48.4 | +2.0 |
| GOOD-clean | 50 | +0.0 | +0.0 | +0.0 | +2.0 |

**The knife**: dipped-goods are the trades a plain stop bails at the dip then watches run without them. Plain-stop (X=48) nets -157.8 t/ep on the 48 dipped-goods; re-entry nets -109.4 t/ep (29/48 re-entered). Re-entry recovers the knifed run (turns the irreversible cut into an M-tick+friction give-up).

## 4. Chop-churn cost (episodes with >= 2 bails under B=2)
75/198 episodes bail >= 2x at (X=48,M=4,B=2) (60 wrong / 15 good). These are the whipsaw payers: each extra bail-and-re-enter cycle gives up its confirmation margin + friction.
- churn-episode net (B=2): mean -68.5 | median -47.8 | mode -190.0 t/ep
- total give-up (sum of re-entry margins crossed) on churn eps: mean +87.7 t/ep

**Give-up quantified** (all re-entries, frozen B=1): 96/198 test eps re-entered, 96 re-entry events. Margin crossed on re-entry: mean +50.5 | median +35.0 | mode +22.0 ticks (>= M=4 by design); each also pays 2.4t friction. This is the "slightly worse position" toll Moises described.

## 5. Distribution (mode-first) -- stop+re-entry net ticks/ep on the 198
- mode **+2.0** | median +0.0 | mean +11.0 [95% CI -12.2,+34.5] ticks/ep (N=198).
```
  [  -600,  -596)    2 #
  [  -588,  -584)    1 #
  [  -400,  -396)    1 #
  [  -328,  -324)    1 #
  [  -324,  -320)    1 #
  [  -316,  -312)    1 #
  [  -256,  -252)    1 #
  [  -240,  -236)    1 #
  [  -216,  -212)    1 #
  [  -212,  -208)    1 #
  [  -208,  -204)    1 #
  [  -172,  -168)    1 #
  [  -168,  -164)    4 ##
  [  -148,  -144)    1 #
  [  -144,  -140)    1 #
  [  -140,  -136)    1 #
  [  -132,  -128)    2 #
  [  -128,  -124)    1 #
  [  -120,  -116)    1 #
  [  -116,  -112)    1 #
  [  -112,  -108)    2 #
  [  -108,  -104)    2 #
  [  -100,   -96)    1 #
  [   -92,   -88)    1 #
  [   -84,   -80)    2 #
  [   -80,   -76)    1 #
  [   -76,   -72)    1 #
  [   -64,   -60)    2 #
  [   -60,   -56)    1 #
  [   -56,   -52)    1 #
  [   -52,   -48)    1 #
  [   -48,   -44)    5 ##
  [   -44,   -40)    4 ##
  [   -36,   -32)    1 #
  [   -32,   -28)    2 #
  [   -28,   -24)    3 ##
  [   -24,   -20)    4 ##
  [   -20,   -16)    2 #
  [   -16,   -12)    2 #
  [   -12,    -8)    1 #
  [    -8,    -4)    1 #
  [    -4,    +0)    1 #
  [    +0,    +4)   80 ########################################
  [    +4,    +8)    1 #
  [    +8,   +12)    1 #
  [   +12,   +16)    1 #
  [   +20,   +24)    1 #
  [   +28,   +32)    1 #
  [   +32,   +36)    1 #
  [   +40,   +44)    1 #
  [   +44,   +48)    1 #
  [   +48,   +52)    3 ##
  [   +52,   +56)    1 #
  [   +56,   +60)    2 #
  [   +64,   +68)    1 #
  [   +76,   +80)    1 #
  [   +84,   +88)    1 #
  [  +100,  +104)    2 #
  [  +104,  +108)    1 #
  [  +124,  +128)    1 #
  [  +144,  +148)    1 #
  [  +152,  +156)    1 #
  [  +156,  +160)    1 #
  [  +168,  +172)    1 #
  [  +176,  +180)    1 #
  [  +184,  +188)    2 #
  [  +200,  +204)    1 #
  [  +216,  +220)    2 #
  [  +228,  +232)    1 #
  [  +232,  +236)    1 #
  [  +244,  +248)    1 #
  [  +248,  +252)    1 #
  [  +252,  +256)    1 #
  [  +264,  +268)    1 #
  [  +272,  +276)    1 #
  [  +280,  +284)    1 #
  [  +304,  +308)    1 #
  [  +316,  +320)    1 #
  [  +328,  +332)    2 #
  [  +332,  +336)    1 #
  [  +340,  +344)    1 #
  [  +344,  +348)    1 #
  [  +348,  +352)    1 #
  [  +368,  +372)    1 #
  [  +440,  +444)    1 #
  [  +504,  +508)    1 #
  [  +540,  +544)    1 #
  [  +728,  +732)    1 #
```

_1m-granularity path sim on the sealed doc-100 test set; a dojo/path number is a hypothesis, not a live result -- any retained rule still graduates through the sealed harness (graduation firewall)._