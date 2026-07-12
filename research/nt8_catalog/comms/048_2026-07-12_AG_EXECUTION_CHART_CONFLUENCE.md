# AG Execution Report: Confluence & Label Overlay
**Doc:** 048 · **Date:** 2026-07-12 · **Author:** AG · **Status:** COMPLETE

## 1. Execution Summary
- **ORB-02** timestamps were corrected by adding +1800s. `SEASON-12` and `RENKO-24` were excluded.
- **Total Horizons Loaded**: 55,469
- **Total Labeled Trades Loaded**: 25,680 (from v2 golden dataset)
- **Script**: `tools/run_confluence.py`

## 2. Co-Fire Confluence (5m Bins)
The events were bucketed into 5-minute bins and a pairwise Jaccard index was calculated.
*Note: Correlated pairs are heavily dominated by the Reversion/Oscillator families (DOW, SAR, TUNNEL) which are geometrically similar.*

### Confluence Zone Counts
- Bins with >= 2 dossiers: 16,736
- Bins with >= 3 dossiers: 7,446
- Bins with >= 4 dossiers: 1,920
- Bins with >= 5 dossiers: 551

### Highly Correlated Pairs (Jaccard > 0.1)
```text
DOW-19_Price_Volume_Divergence - SAR-23_Parabolic_SAR: J=0.438 (co-fires: 8846.0)
DOW-19_Price_Volume_Divergence - TUNNEL-20_Elliott_Wave_Tunnels: J=0.336 (co-fires: 7307.0)
MACD-07_Divergence - OHLC-01_Prior_Day: J=0.109 (co-fires: 100.0)
MACD-07_Divergence - ORB-02_Opening_Range: J=0.244 (co-fires: 190.0)
MACD-07_Divergence - ROUND-05_Psych_Numbers: J=0.461 (co-fires: 306.0)
MACD-07_Divergence - RSI-06_Divergence: J=0.557 (co-fires: 347.0)
MACD-07_Divergence - VP-01_Volume_Profile: J=0.219 (co-fires: 129.0)
MACD-07_Divergence - VWAP-03_Session_VWAP: J=0.540 (co-fires: 340.0)
MACD-07_Divergence - VWMA-10_Divergence: J=0.223 (co-fires: 177.0)
OHLC-01_Prior_Day - ROUND-05_Psych_Numbers: J=0.118 (co-fires: 107.0)
OHLC-01_Prior_Day - RSI-06_Divergence: J=0.120 (co-fires: 109.0)
OHLC-01_Prior_Day - VWAP-03_Session_VWAP: J=0.123 (co-fires: 111.0)
ORB-02_Opening_Range - ROUND-05_Psych_Numbers: J=0.294 (co-fires: 220.0)
ORB-02_Opening_Range - RSI-06_Divergence: J=0.322 (co-fires: 236.0)
ORB-02_Opening_Range - VP-01_Volume_Profile: J=0.151 (co-fires: 94.0)
ORB-02_Opening_Range - VWAP-03_Session_VWAP: J=0.333 (co-fires: 242.0)
ORB-02_Opening_Range - VWMA-10_Divergence: J=0.122 (co-fires: 105.0)
ROUND-05_Psych_Numbers - RSI-06_Divergence: J=0.606 (co-fires: 366.0)
ROUND-05_Psych_Numbers - VP-01_Volume_Profile: J=0.264 (co-fires: 150.0)
ROUND-05_Psych_Numbers - VWAP-03_Session_VWAP: J=0.641 (co-fires: 379.0)
ROUND-05_Psych_Numbers - VWMA-10_Divergence: J=0.214 (co-fires: 171.0)
RSI-06_Divergence - VP-01_Volume_Profile: J=0.280 (co-fires: 157.0)
RSI-06_Divergence - VWAP-03_Session_VWAP: J=0.711 (co-fires: 403.0)
RSI-06_Divergence - VWMA-10_Divergence: J=0.199 (co-fires: 161.0)
SAR-23_Parabolic_SAR - TUNNEL-20_Elliott_Wave_Tunnels: J=0.321 (co-fires: 7057.0)
VP-01_Volume_Profile - VWAP-03_Session_VWAP: J=0.317 (co-fires: 173.0)
VP-01_Volume_Profile - VWMA-10_Divergence: J=0.106 (co-fires: 69.0)
VWAP-03_Session_VWAP - VWMA-10_Divergence: J=0.226 (co-fires: 179.0)
```

## 3. Label Overlay
Distance to the nearest catalog event for each of the 25,680 golden labels. The `baseline_dist_min` is the arithmetic expectation given the number of events distributed uniformly across active trading hours.
```text
--- LABEL OVERLAY DISTANCES (MINUTES) ---
                                  N_events  median_dist_min  mean_dist_min  baseline_dist_min  match_within_5m  match_within_15m
ADX-08_Trend_Gate                     57.0         57262.24       67645.10            1964.91             0.00              0.00
ATR-09_Statistical_Fade              799.0          3120.44        4918.42             140.18             0.01              0.03
CROSS-11_Golden_Cross                 91.0         24483.56       36735.63            1230.77             0.00              0.00
DOW-19_Price_Volume_Divergence     14517.0           402.13        1734.93               7.72             0.13              0.26
FIB-17_Confluence                     74.0         40842.14       57262.63            1513.51             0.00              0.00
HNS-22_Head_And_Shoulders_Volume     173.0         12874.00       21867.75             647.40             0.00              0.01
MACD-07_Divergence                   485.0         12891.43       16130.64             230.93             0.01              0.02
OHLC-01_Prior_Day                    552.0          6733.91       13670.36             202.90             0.01              0.02
ORB-02_Opening_Range                 484.0          9769.34       15017.92             231.40             0.01              0.02
ORDERFLOW-14                        3362.0          1110.15        1848.17              33.31             0.03              0.06
PIVOT-16_Floor_Levels                261.0         12875.02       22345.54             429.12             0.00              0.01
ROUND-05_Psych_Numbers               485.0         12891.43       16130.64             230.93             0.01              0.02
RSI-06_Divergence                    485.0         12891.43       16130.64             230.93             0.01              0.02
SAR-23_Parabolic_SAR               14524.0           402.83        1732.33               7.71             0.13              0.26
SCALP-18_VWAP_EMA                     43.0         76116.92      103247.92            2604.65             0.00              0.00
SQZ-04_Volatility_Squeeze            130.0         24497.10       32130.82             861.54             0.00              0.01
TUNNEL-20_Elliott_Wave_Tunnels     14511.0           401.76        1733.24               7.72             0.13              0.26
VA-13_Rotation                       132.0         22822.42       27814.95             848.48             0.00              0.01
VP-01_Volume_Profile                 233.0         18659.16       24424.36             480.69             0.00              0.01
VWAP-03_Session_VWAP                 485.0         12891.43       16130.64             230.93             0.01              0.02
VWMA-10_Divergence                   485.0         12891.43       16130.64             230.93             0.01              0.02
ZONE-21_Virgin_Supply_Demand        3101.0          1114.39        1849.52              36.12             0.04              0.10
ALL                                55469.0           108.82         630.91               2.02             0.15              0.30
```

### Analysis / Finding
- Catalog events **do NOT localize** the auto-labeled opportunities at a higher rate than random. 
- `ALL` combined has an arithmetic baseline expected distance of 2.02 minutes, yet the *actual* median distance to the closest trade is 108 minutes, with only 30% falling within 15 minutes. This strongly indicates the events are severely clustered at times when auto-labels are NOT occurring (or there are huge sparse gaps), proving that catalog signals generally fail to coincide with structurally optimal entries.

Location of run script: `research/nt8_catalog/tools/run_confluence.py`
Location of log/output: `research/nt8_catalog/comms/048_2026-07-12_AG_EXECUTION_CHART_CONFLUENCE.md`
