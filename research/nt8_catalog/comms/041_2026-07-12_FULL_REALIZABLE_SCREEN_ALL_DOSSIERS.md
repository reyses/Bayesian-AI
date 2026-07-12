# Full Realizable-Trade Screen — every dossier, both directions (final tiers)
**Doc:** 041 · **Date:** 2026-07-12 · **Author:** Claude (executor) · **Status:** FINAL
Full tables: `reports/AG_cat_00_REALIZABLE.md`. Method: PnL from stored MFE/MAE,
worst-case ordering (stop dominates), grid T/S ∈ {10/10,10/20,15/15,20/20}, day-block
CIs, FINDING = CI_lo>0 in BOTH years. ~192 tests → expect 1-2 false findings; trust
config-robust ones. Envelope check run on all 12 findings (censoring at old stops —
the artifact I caught mid-screen; most windows extend well beyond, two don't).

## TIER 1 — TRADABLE CANDIDATES (>2pts/event, config-robust, envelope OK)
| Dossier | trade | 2024 | 2025 | note |
|---|---|---|---|---|
| **ROUND-05** | breach continuation ±20 | +5.43 [+3.95,+6.90] | +6.61 [+4.76,+8.37] | 4/4 configs |
| **PIVOT-16 flip** (doc 039) | short S1-touch/long R1-touch, TP+11, stop 20 | +8.07 [+6.00,+10.04] | +9.05 [+6.80,+11.20] | stop-robust |
| **PIVOT-16 stated** | bounce w/ T20/S20 | +2.75 [+1.67,+4.03] | +2.31 [+1.08,+3.64] | 4/4 — see §3 |
| VWMA-10 | stated T20/S20 | +2.09 [+1.01,+3.18] | +3.00 [+1.67,+4.32] | 3/4 configs |
| ATR-09 | stated fade T20/S20 | +2.04 [+1.52,+2.64] | +1.47 [+0.91,+2.09] | borderline friction |

## TIER 2 — SIGNIFICANT BUT SUB-FRICTION (real drift, unpayable per-trade)
OHLC-01 (+1.2/+1.5), DOW-19 flip (+1.25/+1.32), SAR-23 (+0.6/+1.3),
TUNNEL-20 (+0.6/+1.3), ZONE-21 (+0.8/+1.2), ORDERFLOW-14 (+0.23!). Huge N makes
tiny drift significant; 2-5 ticks/event dies to commissions+slippage. NOT standalone
trades — but a consistent post-event drift worth knowing about.

## TIER 3 — ENVELOPE-SUSPECT (measurement truncation at the tested stop)
HNS-22 flip (+2.8/+3.9 but max adverse 19.8 ≈ S=20 → censored), VP-01 (+4.6/+4.5,
max adverse 27.5 vs S=20 → marginal). Need re-measurement with wider windows before
believing.

## NOTHING both-year: ADX-08, CROSS-11, FIB-17, MACD-07, ORB-02, RENKO-24, RSI-06,
SCALP-18, SEASON-12 (confirmed doc 040), SQZ-04, VA-13, VWAP-03.

## §3 The structural discovery: PIERCE-THEN-BOUNCE
PIVOT-16 profits BOTH ways at different geometries — impossible at the same T/S
(algebra: stated+flip ≤ 0), possible only if the path OSCILLATES: the level touch
first pierces ~10-15 pts (the flip's +11 take-profit), THEN bounces 20+ (the stated
side's T20 with a stop wide enough to survive the pierce). The level is not support
OR continuation — it's a two-phase magnet: pierce, then revert. ROUND-05's
continuation and VWMA/ATR's stated wins are consistent with the same anatomy.
This is the most structural thing the catalog has produced: a repeatable price
ANATOMY around watched levels, exploitable at two different horizons.

## Next (ordered)
1. Path-accurate backtest (offline forward pass) of ROUND-05 ±20 and PIVOT-16 flip —
   the two clean Tier-1s — with real bar-sequence stop/target ordering (kills the
   worst-case approximation both directions).
2. Pierce-then-bounce anatomy study on PIVOT/ROUND: time-to-pierce, pierce depth
   distribution, bounce onset — one combined dossier.
3. Re-measure HNS-22/VP-01 with wider windows (Tier-3 resolution).
