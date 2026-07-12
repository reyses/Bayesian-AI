"""
AMBIENT EBB/FLOW BASELINE (Moises' question, 2026-07-12): do catalog events sit in
the day's ordinary oscillation, or do they mark elevated movement?

Compares each event's 15m MFE against ambient anchors (every 2 min, direction-
neutral) from the SAME DAY and SAME HOUR (diurnal-matched). Day-block CIs.
Result 2026-07-12: reversion/level/divergence family shows +5..+14pt excess
amplitude vs hour-matched ambient (ORB +13.9, ROUND +9.0, VWAP-03 +8.7, RSI-06
+7.8, VWMA +6.2, VP-01 +5.7, MACD +5.2); ATR-09 fully explained by time-of-day;
trend/structure family ~ambient. Remaining confound: volatility clustering
(GARCH) — next drill = trailing-vol-matched ambient.
Run after fps_horizon_explorer.py (needs reports/fps_horizons.parquet).
"""
# analysis executed inline 2026-07-12 (see comms/046 for the exact code + tables);
# to reproduce: run the two code blocks from comms/046 §2-3.
