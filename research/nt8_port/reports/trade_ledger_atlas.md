# Trade ledger v0.4 (rebuilt from frozen golden vectors, R-trigger LIVE)
6192 trades, 520 days. CSV: reports/trade_ledger_v04.csv (ATLAS-bar-aligned).

- RIDE-ONLY (R-trigger/session, NO stop): net $16,260 | day-WR 53% | $31/day
- WITH 50pt stop: net $2,208 | day-WR 48%
- Trade WR (PF-based, ride-only): +0.07
- exit reasons: {'R_TRIGGER': 4903, 'CAT_STOP': 857, 'SESSION_CLOSE': 432}
- **t = pvt + n**: n(bars from pivot to entry) median 8, mean 15.4, p10-p90 [3, 31]

R-trigger firing (vs v0.2 backtest where it fired 0x) is the key check.
Every row carries pivot_bar / entry_bar / exit_bar as ATLAS indices +
timestamps for bar-by-bar review (the GBM input).
