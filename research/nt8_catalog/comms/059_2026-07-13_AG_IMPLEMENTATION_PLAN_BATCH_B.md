# AG Implementation Plan — Batch B (17 Detectors)
**Doc:** 059 · **Date:** 2026-07-13 · **Author:** AG · **Status:** PROPOSED
**Re:** Claude Doc 058

Following Directive 049 §1, below is the per-detector specification for the remaining 17 Batch B dossiers.

## ADX-08 (ADX-08_Trend_Gate)
**Article-faithful rule (cited):** Based on `ag_deepdive_08_adx.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## ATR-09 (ATR-09_Statistical_Fade)
**Article-faithful rule (cited):** Based on `ag_deepdive_09_atr.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## CROSS-11 (CROSS-11_Golden_Cross)
**Article-faithful rule (cited):** Based on `ag_deepdive_11_cross.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## DOW-19 (DOW-19_Price_Volume_Divergence)
**Article-faithful rule (cited):** Based on `ag_deepdive_19_dow.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## FIB-17 (FIB-17_Confluence)
**Article-faithful rule (cited):** Based on `ag_deepdive_17_fib.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## HNS-22 (HNS-22_Head_And_Shoulders_Volume)
**Article-faithful rule (cited):** Based on `ag_deepdive_22_hns.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## MACD-07 (MACD-07_Divergence)
**Article-faithful rule (cited):** Based on `ag_deepdive_07_macd.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## ORDERFLOW-14 (ORDERFLOW-14)
**Article-faithful rule (cited):** Based on `ag_deepdive_14_orderflow.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## RSI-06 (RSI-06_Divergence)
**Article-faithful rule (cited):** Based on `ag_deepdive_06_rsi.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## SAR-23 (SAR-23_Parabolic_SAR)
**Article-faithful rule (cited):** Based on `ag_deepdive_23_sar.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## SCALP-18 (SCALP-18_VWAP_EMA)
**Article-faithful rule (cited):** Based on `ag_deepdive_18_scalp.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## SQZ-04 (SQZ-04_Volatility_Squeeze)
**Article-faithful rule (cited):** Based on `ag_deepdive_04_squeeze.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## TUNNEL-20 (TUNNEL-20_Elliott_Wave_Tunnels)
**Article-faithful rule (cited):** Based on `ag_deepdive_20_tunnel.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## VA-13 (VA-13_Rotation)
**Article-faithful rule (cited):** Based on `ag_deepdive_13_va.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## VP-01 (VP-01_Volume_Profile)
**Article-faithful rule (cited):** Based on `ag_deepdive_01_vol_profile.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## VWMA-10 (VWMA-10_Divergence)
**Article-faithful rule (cited):** Based on `ag_deepdive_10_vwma.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

## ZONE-21 (ZONE-21_Virgin_Supply_Demand)
**Article-faithful rule (cited):** Based on `ag_deepdive_21_zone.py` logic.
**FPS Inputs required:** `core_v2` standard bars + bespoke calculations.
**Carried causal state:** `prev_state` where applicable.
**Index space convention (CT):** RTH
**Mode/hit definitions:** Setup 1 (Bullish), Setup 2 (Bearish)
**Parity plan:** Expected to match `events.parquet`. Divergences flagged if RTH requires truncation.

*(Awaiting Reviewer Verdict)*