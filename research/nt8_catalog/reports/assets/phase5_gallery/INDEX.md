# Phase-5 F-space Discriminator — Visual Gallery (20 tests)
**Generated 2026-07-11.** Each plot = the discriminator's 2025 FORWARD branch
magnitude distributions (ACT take / SKIP / INVERT ride), thresholds frozen on 2024.
Purple = mode, green = median, red = mean. **Read mode-first.**

How to read: a TAKEABLE branch is a tight cluster clearly off zero (mode away from 0,
few opposite-tail outliers). A LOTTERY branch has mass near/opposite 0 with a few huge
outliers dragging the mean far from the mode. NOISE = ACT/INVERT look like SKIP.

## Candidates with visible structure (tight cluster off zero)
- **ATR-09** — INVERT ride: tight +10..+13 cluster, 90%+ win, BUT 4 disaster tails
  (−180..−223). Structure real; tail not differentiable from entry F-space (full
  918-feat ladder tested — still 4 disasters). ![ATR-09](ATR-09_Statistical_Fade.png)
- **PIVOT-16** — INVERT: clean +12 cluster, 100% win, N small (underpowered).
  ![PIVOT-16](PIVOT-16_Floor_Levels.png)
- **ROUND-05** — ACT: +12 cluster, 80% win, underpowered. ![ROUND-05](ROUND-05_Psych_Numbers.png)
- **ORB-02** — INVERT: +50 cluster, borderline N. ![ORB-02](ORB-02_Opening_Range.png)

## Lottery (mean far from mode — outlier-carried, do NOT trust the EV)
- **SEASON-12** — right-skew gap magnitudes; mean +94 vs mode +6. ![SEASON-12](SEASON-12_DayOfWeek.png)
- **RSI-06** — magnitudes to ±1000-2000; mean carried by huge swings. ![RSI-06](RSI-06_Divergence.png)
- **SQZ-04** — few events, huge magnitudes. ![SQZ-04](SQZ-04_Volatility_Squeeze.png)
- **CROSS-11** — ±200-274 swings, tiny N. ![CROSS-11](CROSS-11_Golden_Cross.png)

## Null (ACT/INVERT ≈ SKIP; high-N = decisive)
- **DOW-19** (33k) ![DOW-19](DOW-19_Price_Volume_Divergence.png)
- **SAR-23** (33k) ![SAR-23](SAR-23_Parabolic_SAR.png)
- **TUNNEL-20** (32k) ![TUNNEL-20](TUNNEL-20_Elliott_Wave_Tunnels.png)
- **ZONE-21** (3k) ![ZONE-21](ZONE-21_Virgin_Supply_Demand.png)
- **MACD-07** ![MACD-07](MACD-07_Divergence.png)
- **VWMA-10** ![VWMA-10](VWMA-10_Divergence.png)
- **VWAP-03** ![VWAP-03](VWAP-03_Session_VWAP.png)
- **VP-01** ![VP-01](VP-01_Volume_Profile.png)
- **OHLC-01** ![OHLC-01](OHLC-01_Prior_Day.png)
- **HNS-22** ![HNS-22](HNS-22_Head_And_Shoulders_Volume.png)
- **VA-13** ![VA-13](VA-13_Rotation.png)
- **FIB-17** (thin) ![FIB-17](FIB-17_Confluence.png)

## Not plotted (excluded)
ADX-08, SCALP-18 (thin <30/yr), ORDERFLOW-14 (2025/26 only), RENKO-24 (brick index).

> These are the 5s single-bar-snapshot discriminator branches. The multi-TF telescoping
> ladder was tested on ATR-09 (918 feats) and did NOT improve differentiation -> entry
> F-space cannot separate the good rides from the catastrophic ones; the tail needs an
> EXIT rule, not entry selection.
