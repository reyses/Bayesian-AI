# Trade ledger v0.4 (rebuilt from frozen golden vectors, R-trigger LIVE)
197 trades, 13 days. CSV: reports/trade_ledger_v04.csv (ATLAS-bar-aligned).

- Net: $-1,234 | Day-WR: 31% (4/13)
- Trade WR (PF-based): -0.10
- exit reasons: {'CAT_STOP': 98, 'R_TRIGGER': 87, 'SESSION_CLOSE': 12}
- **t = pvt + n**: n(bars from pivot to entry) median 10, mean 23.0, p10-p90 [3, 62]

R-trigger firing (vs v0.2 backtest where it fired 0x) is the key check.
Every row carries pivot_bar / entry_bar / exit_bar as ATLAS indices +
timestamps for bar-by-bar review (the GBM input).
