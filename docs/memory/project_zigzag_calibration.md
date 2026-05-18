---
name: project-zigzag-calibration
description: "Python zigzag pipeline calibration — ATR×4 dynamic R, 1m ATR source, 5s pivot detection. NT8 strategy must match these to produce comparable trades."
metadata: 
  node_type: memory
  type: project
  originSessionId: e1202bdb-1787-4774-b48b-b312af212043
---

The Python zigzag pipeline that produced the `composite_forward_pass_hardened.csv` baseline ($475/day flat, 87% Day WR) uses these EXACT calibration parameters. The NT8 strategy must match them or NT8 fills will not be comparable to Python sim.

Source of truth: `tools/build_zigzag_pivot_dataset.py` + `tools/_viz/auto_swing_marker.py:detect_swings`

**Calibration**:
- ATR period: **14**
- ATR source: **1m bars** (NOT the pivot-detection series)
- ATR multiplier: **4.0**  ← "we made it dynamic" — was static, now ATR×4
- Pivot detection bars: **5s closes**
- min_reversal_ticks = `max(4, round(atr_pts / tick_size * 4))`  → in points = `max(1.0, atr_pts × 4)`
- min_bars between pivots: **36** (= 3 minutes at 5s)
- Tick size: 0.25, dollar/point: $2 (MNQ)

**Why:** Why: User visual-calibrated this on 14-month MNQ data; ATR×4 produces zigzags that match real swings rather than noise. Static R values (e.g. R=30) produce different pivot counts and trade timestamps; not comparable.

**How to apply:** When recommending NT8 strategy params for parity testing against Python sim, ALWAYS:
1. Verify the NT8 strategy can compute ATR on a 1m series and detect pivots on a 5s series independently
2. Set ATR period 14, multiplier 4.0, MinR floor 1.0pt (= 4 ticks)
3. Set pivot TF to 5s if the strategy supports it
4. If the NT8 strategy lacks separate ATR/pivot timeframes (single-series ATR), there is a structural mismatch — flag it, propose a fix (use ZigZagATR.cs indicator as input), do NOT recommend static R or single-TF dynamic R as "raw"

**Current NT8 state**:
- `docs/nt8/ZigZagATR.cs` indicator has correct architecture (AtrTfMinutes=1, AtrPeriod=14, AtrMult=4.0, ZigZagTfSeconds=5, MinBars=36)
- `docs/nt8/ZigzagRunner_v1.0.cs` (deployed as `ZigzagRunner.cs`) uses STATIC R=30 → does not match Python
- `docs/nt8/ZigzagRunner_v1.0.8-RC.cs` has UseDynamicR but defaults wrong (AtrLookbackBars=60, AtrMultiplier=5.0, MinRPoints=5.0) AND ATR is computed on the pivot series (not a separate 1m series) → does not match Python

**The gap**: no strategy currently matches Python. Either patch v1.0.8-RC to add a separate 1m ATR series and fix defaults, OR build a new strategy that consumes ZigZagATR.cs indicator as the pivot source. The latter is cleaner and is the foundation for the hybrid build (see [[user_collaboration_protocol]]).

Related: [[feedback_no_human_regime_terms]] (translate metaphors), [[project_v2_iso_pipeline]] (V2 pipeline calibration).
