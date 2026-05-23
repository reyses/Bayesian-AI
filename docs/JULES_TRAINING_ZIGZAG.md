# JULES_TRAINING_ZIGZAG — Causal zigzag forward-pass pipeline

## Context — why this exists

The `trade_outcome_suite` (15-question probability tables) was built on the
**hardened legs** (`is/oos_hardened_legs.csv`). Those legs derive their pivot
sequence from `is_pivot==1` flags in the offline truth dataset
(`zigzag_pivot_dataset_*_atr4.parquet`) — i.e. the **offline zigzag**, which
sees the whole day. The R-trigger hardening fixes entry/exit *price* lookahead,
but the *set of legs* is still the hindsight-clean offline partition: every leg
is a genuine swing, **zero whipsaw by construction**. A live streaming engine
takes whipsaws (false pivots acted on before revision) — exactly the extra
losers — so every table in the suite is optimistic by that population.

`training/forward_blended.py` is a genuine causal forward pass (bar-by-bar,
`on_state`, no lookahead) — but it runs the **BlendedEngine** (the 9-tier
baseline-740 system), a different strategy from the zigzag legs the suite
analyses.

**Goal:** a causal, bar-by-bar forward pass of the **L5 / zigzag engine**, so
the suite can be re-based on lookahead-free legs and we can measure how much
the offline-pivot selection inflated the tables.

## Design decision — lean folder, reuse the live engine

The user asked to copy `training/` → `training_zigzag/` and purge the blended
engine. **Rejected as literal instruction**, for two reasons:
1. `training/` is ~40 files, the bulk blended/CNN-specific (`nightmare_blended`,
   six `cnn_*.py`, `build_dataset*`, the 7-phase trainer). Copy-then-purge
   leaves orphaned files importing the deleted engine.
2. The zigzag engine is **not in `training/`** — it lives in `live/`
   (`engine_v2.py`, `l5_decider.py`) + `core/ledger.py`, and has a validated
   mock. Re-implementing it inside `training_zigzag/` would drift from the
   live engine (the project already has scar tissue from streaming-detector
   mismatches and the B9-horizon bug).

**Adopted:** `training_zigzag/` is a **lean, purpose-built** folder that
*imports and drives the existing live zigzag engine* — no re-implementation,
nothing blended copied in, nothing to purge. The original `training/` is left
**untouched** (proven blended baseline — must not break it).

## Architecture

```
training_zigzag/
  __init__.py
  forward_zigzag.py   causal bar-by-bar forward-pass harness
  README.md
```

`forward_zigzag.py` mirrors `training/forward_blended.py`'s per-day pattern,
but drives `L5Decider` instead of `BlendedEngine`. It is the lean equivalent
of `engine_v2._step7_trade()` with the NT8/async/bridge layer stripped out and
zero-slip immediate fills (the `mock_bridge` fill convention).

### Per-day loop (causal — no lookahead)

For each day (IS = `DATA/ATLAS_NT8/` 2025 / OOS = 2026; 5s + aggregated TFs):

1. **Instantiate fresh per day**: `L5Decider(ctx)` with `pivot_source='stream'`
   (the causal streaming zigzag — `l5_decider._update_zigzag`), a
   `LiveFeatureEngineV2`, and a `core.ledger.Ledger`.
2. **Warmup**: `LiveFeatureEngineV2.load_history()` for the prior days +
   `L5Decider.prime_atr_from_history()` (5-day rolling ATR median — locks
   `min_rev_ticks`, matches production).
3. **Stream bars**: for each 5s bar of the day —
   - `lfe.on_bar(bar)` (aggregates to 1m/5m/…);
   - build `eval_state` (`features_79d`, `price/high/low/volume`, `timestamp`,
     `positions = ledger.snapshot()`, `v2_getter = lfe.get_v2_vector`);
   - `batch = engine.evaluate(eval_state)`;
   - apply `batch.position_decisions` via `ledger.apply_position_decision`;
   - process `batch.exits` (`ledger.remove_position`, zero-slip at bar close)
     and `batch.entry` / `chain_entry` (`ledger.add_position`) — mirroring
     `engine_v2._step7_trade` lines ~1070-1174, allowing same-bar flips.
4. **Day end**: force-close any open position at the last bar; harvest
   `ledger.closed_trades`.
5. **Emit legs**: convert `closed_trades` → rows in the **hardened-legs schema**
   so the suite consumes them unchanged:
   `day, entry_ts, leg_dir, entry_price, exit_ts, exit_price, pnl_pts,
   pnl_usd, r_price, atr_pts`.

### Output

```
reports/findings/trade_outcome_table/causal_zigzag_legs_IS.csv
reports/findings/trade_outcome_table/causal_zigzag_legs_OOS.csv
reports/findings/trade_outcome_table/causal_zigzag_forward_pass.txt   (summary)
```

## Suite integration

`tools/trade_outcome_suite/excursions.py` gets a **source switch**:
- `SRC['hardened']` → the old offline legs (kept for the comparison).
- `SRC['causal']`  → the new `causal_zigzag_legs_{IS,OOS}.csv`.

`excursions.py` already reconstructs MAE/MFE from the 5s path given
`entry_ts`/`exit_ts` — it works on causal legs **unchanged**. `run_all.py`
gets a `--source {hardened,causal}` flag (default `causal` once validated).
A final comparison run prints hardened-vs-causal side by side → quantifies the
lookahead inflation.

## Phases

- **Phase 1** — build `training_zigzag/` + `forward_zigzag.py`; verify it runs
  one day and emits sane legs (entry/exit ordered, pnl signs correct).
- **Phase 2** — full IS+OOS run (**user runs** — heavy, ~hundreds of days).
- **Phase 3** — `excursions.py` source switch + rerun the 15-question suite on
  causal legs + hardened-vs-causal comparison report.

## Verification

- **Parity check**: pick one day already run via `engine_v2 --mock`; run
  `forward_zigzag.py` on it; trade count + entry/exit prices must match the
  mock within zero-slip tolerance. (The mock was previously validated
  decision-identical to `forward_pass_full_stack` — this is the same bar.)
- Sanity: `entry_ts < exit_ts` every leg; `pnl_pts` sign matches `leg_dir` vs
  price delta; per-day leg counts plausible (causal should be near hardened or
  modestly higher — whipsaws add legs / replace clean ones).
- Headline: causal $/day vs the hardened forward-pass $/day; causal P(close>0),
  loss rate, MAE/MFE distributions vs the suite's hardened numbers.

## Files

- **New**: `training_zigzag/{__init__.py, forward_zigzag.py, README.md}`,
  `docs/JULES_TRAINING_ZIGZAG.md` (this file).
- **Modified**: `tools/trade_outcome_suite/excursions.py` (source switch),
  `run_all.py` (`--source` flag) — Phase 3.
- **Untouched**: `training/` (proven blended baseline).

## Key risks

- The harness must faithfully replicate `engine_v2._step7_trade` fill / exit /
  entry / same-bar-flip handling. Mitigation: zero-slip immediate fills (the
  `mock_bridge` convention); the Phase-1 parity check against an
  `engine_v2 --mock` day catches divergence.
- ATR priming + feature warmup must match production — reuse
  `prime_atr_from_history` and `LiveFeatureEngineV2.load_history` verbatim,
  do not re-derive.
- This is a faithful *proxy* for the live engine; the absolute reference is
  `engine_v2 --mock`. The harness exists because running 326 separate async
  `--mock` sessions (full 7-step startup each) is impractical — the harness is
  the lean batch equivalent.
