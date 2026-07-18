# APEX — gemma4 causal trade run (ZERO lookahead) | 2024_02_20
model=gemma4:latest | warmup=20 | decision bars=90 | fill=next-bar-open

## Result
- Realized PnL: **$+0.00** (1 contract, MNQ $2.00/pt)
- Trades: 0  | PF-based Trade WR: +0.000  (0=breakeven, +1=PF 2)
- gross wins $0 / gross losses $0

## Behavior (the real findings)
- action counts: {'LONG': 0, 'SHORT': 0, 'CLOSE': 0, 'HOLD': 90, '_BAD': 0}
- BAD/unparseable outputs (forced HOLD): 0/90 = 0%
- latency/bar: mean 0.99s  -> a 390-bar RTH day ~ 6 min of inference
- wall time: 1.5 min for 90 bars

## Limitations observed
- (latency vs a 5s/1m live cadence; non-determinism even at temp=0; format failures; churn) — see action counts + BAD rate above; full decision log in causal_decisions_*.jsonl
