---
name: tunnel probability = live slippage model
description: tunnel_prob serves dual purpose — entry scoring (P reach TP) AND live fill probability (slippage through order book barrier)
type: project
---

Tunnel probability (`reversion_probability` in current code) maps to live trade slippage:
- Original: P(price reaches target band without hitting SL) via O-U Monte Carlo
- Live extension: P(fill at expected price) given order book depth + volatility

**Why:** At high sigma, order book thins (HFT algos pull liquidity), spreads widen.
The "barrier" the order must tunnel through becomes thicker. Low tunnel_prob =
expect slippage, reduce size or skip.

**How to apply:**
- Factor tunnel_prob into position sizing (Kelly × tunnel_prob)
- Log expected vs actual fill price in live trade logger
- Calibrate tunnel_prob → slippage model from live fill data over time
- At 1σ (PID trance): thick book, clean fills, tunnel_prob high
- At 3σ+ (cascade): thin book, slippage likely, tunnel_prob low
