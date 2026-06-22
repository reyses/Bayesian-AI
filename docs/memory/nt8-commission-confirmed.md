---
name: nt8-commission-confirmed
description: NT8 MNQ commission CONFIRMED at $0.78 ROUND-TRIP from real trade fills (2026-06-16). Use this, not the old ~$2.5/trade assumption.
metadata:
  type: reference
---
NT8 MNQ **commission = $0.78 ROUND-TRIP** (confirmed via actual trade fills, 2026-06-16).
- Slippage is SEPARATE and still ASSUMED (~0.25pt/side = $0.50/side = ~$1.00 RT at MNQ $2/pt).
- Realistic RT cost = **$0.78 comm + ~$1.00 slip = ~$1.78 RT** (commission-only floor = $0.78).
- Prior backtests used ~$2.0-2.5 RT -> OVERSTATED cost by ~$0.7-1.7/trade. Re-run with $0.78 comm.
- Slippage is now the only un-confirmed cost component.
