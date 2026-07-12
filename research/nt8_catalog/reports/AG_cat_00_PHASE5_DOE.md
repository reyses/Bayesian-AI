# Phase-5 DOE — Entry Discriminator across all proposals (ALIGNMENT-FIXED)

Leakage-free: entry anchor (PhE) only, RTH-aligned V2 5s features, thresholds frozen on 2024, evaluated on 2025, day-block bootstrap (4000). VALID = branch N>=30 & >=20 days & day-block CI excludes 0 & |mode|>=2pts. INV EV = mirror approx (-magnitude).

| Dossier | train/test base feats | ACT branch | INVERT branch |
|---|---|---|---|
| ADX-08_Trend_Gate | SKIP: thin (tr 27, te 30) |
| ATR-09_Statistical_Fad | tr432/te367 b0.11 f13 | N=31/28d WR0.03 EV-10.8 CI[-13.9,-7.1] m-10 ns | N=74/60d WR0.96 EV+10.3 CI[+2.0,+16.7] m+10 VALID |
| CROSS-11_Golden_Cross | tr60/te31 b0.38 f5 | N=4/4d WR0.25 EV-13.2 CI[-207.4,+185.1] m-274 under | N=5/5d WR0.60 EV-27.8 CI[-132.2,+39.7] m-230 under |
| DOW-19_Price_Volume_Di | tr17506/te15518 b0.60 f17 | N=1571/222d WR0.61 EV-0.3 CI[-1.0,+0.3] m+2 ns | N=3459/227d WR0.45 EV-0.5 CI[-0.8,-0.1] m-4 ns |
| FIB-17_Confluence | tr32/te42 b0.03 f5 | N=10/10d WR0.00 EV-19.3 CI[-27.4,-12.9] m-13 under | N=7/7d WR1.00 EV+11.4 CI[+10.8,+12.0] m+12 under |
| HNS-22_Head_And_Should | tr86/te87 b0.47 f5 | N=8/8d WR0.62 EV+2.7 CI[-8.3,+13.8] m+4 under | N=19/18d WR0.68 EV+3.3 CI[+0.2,+6.4] m+5 under |
| MACD-07_Divergence | tr258/te227 b0.59 f5 | N=2/2d WR0.50 EV+71.2 CI[-36.5,+179.0] m-36 under | N=185/185d WR0.41 EV+1.8 CI[-6.4,+10.6] m-2 ns |
| OHLC-01_Prior_Day | tr294/te258 b0.66 f9 | N=15/15d WR0.67 EV-5.4 CI[-9.3,-1.4] m-12 under | N=120/97d WR0.39 EV+1.7 CI[-1.3,+4.5] m+10 ns |
| ORB-02_Opening_Range | tr258/te226 b0.32 f2 | N=46/46d WR0.35 EV+1.2 CI[-32.8,+38.1] m-82 ns | N=29/29d WR0.72 EV+32.6 CI[-1.3,+58.5] m+50 under |
| ORDERFLOW-14 | SKIP: not both years ([np.str_('2025'), np.str_('2026')]) |
| PIVOT-16_Floor_Levels | tr131/te130 b0.14 f6 | N=38/38d WR0.03 EV-10.9 CI[-13.0,-7.5] m-11 ns | N=11/11d WR1.00 EV+19.5 CI[+13.9,+28.5] m+12 under |
| RENKO-24_Time_Filterin | SKIP: foreign index space (brick) |
| ROUND-05_Psych_Numbers | tr258/te227 b0.76 f5 | N=20/20d WR0.80 EV+18.7 CI[+11.5,+27.3] m+12 under | N=61/61d WR0.10 EV-34.8 CI[-41.8,-28.4] m-38 ns |
| RSI-06_Divergence | tr258/te227 b0.48 f5 | N=39/39d WR0.44 EV-5.1 CI[-67.4,+55.7] m-26 ns | N=33/33d WR0.76 EV+97.2 CI[+0.8,+235.6] m+25 VALID |
| SAR-23_Parabolic_SAR | tr17808/te15603 b0.49 f15 | N=3156/227d WR0.49 EV-0.4 CI[-0.8,+0.1] m+6 ns | N=2224/225d WR0.49 EV-0.1 CI[-0.8,+0.5] m+4 ns |
| SCALP-18_VWAP_EMA | SKIP: thin (tr 24, te 19) |
| SEASON-12_DayOfWeek | tr248/te219 b0.59 f10 | N=50/50d WR0.68 EV+94.1 CI[+64.7,+125.8] m+6 VALID | N=39/39d WR0.38 EV-116.8 CI[-162.2,-79.7] m-91 ns |
| SQZ-04_Volatility_Sque | tr70/te60 b0.54 f5 | N=2/2d WR0.00 EV-175.4 CI[-196.5,-154.2] m-196 under | N=12/12d WR0.50 EV-5.3 CI[-97.1,+92.2] m-331 under |
| TUNNEL-20_Elliott_Wave | tr16946/te14718 b0.49 f16 | N=4654/227d WR0.49 EV-0.4 CI[-0.7,-0.0] m-4 ns | N=706/202d WR0.50 EV-0.2 CI[-0.8,+0.4] m+4 ns |
| VA-13_Rotation | tr82/te50 b0.11 f4 | N=17/17d WR0.06 EV-16.8 CI[-31.6,-7.2] m-10 under | N=2/2d WR1.00 EV+1.1 CI[+1.0,+1.2] m+1 under |
| VP-01_Volume_Profile | tr135/te98 b0.27 f1 | N=15/15d WR0.27 EV-1.8 CI[-11.9,+10.6] m-12 under | N=18/18d WR0.72 EV+3.5 CI[-6.3,+11.2] m+14 under |
| VWAP-03_Session_VWAP | tr258/te227 b0.61 f4 | N=13/13d WR0.92 EV+10.6 CI[-1.3,+20.6] m+6 under | N=3/3d WR0.33 EV+6.5 CI[-1.2,+21.8] m-1 under |
| VWMA-10_Divergence | tr258/te227 b0.45 f6 | N=40/40d WR0.47 EV+10.3 CI[-2.6,+24.8] m-12 ns | N=32/32d WR0.59 EV+11.7 CI[-11.6,+34.0] m-134 ns |
| ZONE-21_Virgin_Supply_ | tr1404/te1708 b0.46 f7 | N=587/170d WR0.48 EV-0.3 CI[-1.0,+0.3] m-4 ns | N=296/156d WR0.53 EV+0.4 CI[-1.0,+1.8] m+6 ns |

**VALID branches: 3** ATR-09_Statistical_Fade, RSI-06_Divergence, SEASON-12_DayOfWeek