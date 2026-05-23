# training_zigzag

Causal forward-pass pipeline for the **L5 / zigzag engine** — parallel to
`training/` (the blended baseline-740 pipeline), but for the zigzag system.

It does **not** re-implement the engine. It imports and drives the *live*
zigzag engine (`live/l5_decider.py` + `core/ledger.py`) bar-by-bar over
historical data, with the streaming pivot detector — a genuine causal forward
pass, no lookahead.

Design: [`docs/JULES_TRAINING_ZIGZAG.md`](../docs/JULES_TRAINING_ZIGZAG.md).

## Why this exists

The `trade_outcome_suite` was built on the **hardened legs**
(`oos_hardened_legs_full.csv`), whose pivot sequence comes from offline
`is_pivot` labels — a hindsight-clean partition with **zero whipsaw**. A live
streaming engine takes whipsaws (false pivots acted on before revision), so
every table in the suite is optimistic. This pipeline produces the **causal**
leg list to re-base the suite on.

## Run

```
python training_zigzag/forward_zigzag.py --oos          # OOS window (2026)
python training_zigzag/forward_zigzag.py --oos --limit 3   # smoke test, 3 days
python training_zigzag/forward_zigzag.py --with-oos     # IS + OOS
```

Output (hardened-legs schema → consumed unchanged by `excursions.py`):
```
reports/findings/trade_outcome_table/causal_zigzag_legs_{IS,OOS}.csv
reports/findings/trade_outcome_table/causal_zigzag_forward_pass.txt
```

## Layout

| File | Role |
|---|---|
| `forward_zigzag.py` | Causal harness — per day: fresh `L5Decider` (`pivot_source='stream'`), prime ATR from prior 1m, stream the day's 5s bars through `engine.evaluate()`, drive a `Ledger`, emit legs. |

## Verification (before trusting the numbers)

1. **Parity check** — pick one day also run via `engine_v2 --mock`; trade
   count + entry/exit prices must match within zero-slip tolerance.
2. **Sanity** — `entry_ts < exit_ts` every leg; `pnl_pts` sign matches
   `leg_dir`; per-day leg counts plausible.
3. **Headline** — causal $/day vs the hardened forward pass; causal
   P(close>0) / loss-rate / MAE-MFE vs the suite's hardened numbers. The
   delta is the lookahead inflation.

## Notes / open items

- **IS window**: the V2 feature engine reads `DATA/ATLAS_NT8`. If the 2025 IS
  days are not present there, run `--oos` only (the sealed OOS is the honest
  number anyway) or pass `--atlas-root` pointing at the IS V2 data root.
- This harness is a faithful *proxy* for the live engine; the absolute
  reference is `engine_v2 --mock`. It exists because 326 separate async
  `--mock` sessions (full 7-step startup each) is impractical.
- `training/` (blended baseline) is untouched.
