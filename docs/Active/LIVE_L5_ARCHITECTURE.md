# Live L5 Architecture -- thin wrapper, V2 features, 1-contract phase-1

Last updated: 2026-05-19

## Guiding principle (carried from `docs/JULES_ENGINE_DECOUPLE_ORDERS.md`)

> Each part does one thing well.
>   - **Engine** -- detects setups. Pure function of market state. Owns nothing.
>   - **Ledger** -- owns position state. Single source of truth.
>   - **OrderManager** -- owns NT8 wire handshake. Translates fills to ledger mutations.
>   - **Sim executor** -- walks bars in sim. Calls engine, applies decisions.
>   - **Live engine_v2** -- pumps NT8 messages through the orchestration layer. Glue, nothing more.

L5 fits this pattern. It is a *new engine* (`L5Decider`), not a new wrapper.

## Component map

```
                       ATLAS_NT8 (warmup)
                              |
                              v
   NT8  --- BAR (5s) --->  LiveFeatureEngineV2  --->  V2 vector (185D)
                              |                           |
                              v                           |
                          V1 vector (91D)                 |
                              |                           |
                              +-----.       .-------------+
                                    v       v
                              eval_state{ price, ts, v1, v2, positions }
                                    |
                                    v
                      L5Decider.evaluate(state)  ---> DecisionBatch
                                    |
                                    v
                          engine_v2 bar loop (UNCHANGED)
                                    |
                                    v
                              OrderManager  ---> NT8 PLACE_ORDER / CLOSE_POSITION
                                    |
                                    v
                          NT8 FILL  ---> Ledger mutation
```

The dashed boundary is the only thing `L5Decider` sees on the input side: a
`state` dict with features and a read-only `positions` snapshot. It returns
a `DecisionBatch` and nothing else. **No state mutation inside L5Decider.**

## What L5Decider actually contains

```python
class L5Decider:
    def __init__(self, ctx: L5Context):
        self._ctx = ctx       # B7/B9/B10 models, thresholds, zigzag params
        self._zz = ZigzagState()    # running extreme + R-trigger detection
        self._b10_day = None  # cached B10 day-mode (computed on first eval of day)

    def evaluate(self, state) -> DecisionBatch:
        # 1. Update zigzag state from this bar
        # 2. If first bar of new day: compute B10 day-mode -> set thresholds
        # 3. For each open position: check K=5 -> fire B9 query if due
        #    Returns PositionDecision with exit_reason='b9_cut' if triggered
        # 4. If R-trigger fires this bar: load V2 vector, run B7
        #    Returns EntrySignal if pred_R >= threshold
        # 5. Compose DecisionBatch
        return batch
```

The only side effect is updating `self._zz` and `self._b10_day`. Both are
opaque to the rest of the system (the live engine never reads them).

## Phase 1 deployment configuration (locked 2026-05-19, autonomous build)

Constraint: **1 contract per position, no scaling, no chains.**

This collapses three of B7/B9/B10's surfaces to filters and one of B10's
surfaces to a parameter-modulator:

| Model | Original surface | Phase-1 1c behaviour |
|---|---|---|
| B7 | size in {0, 1, 2, 3} | binary: take if pred_R >= 1.0, else skip |
| B9 | size factor in {0, 0.5, 1.0, 1.5} | binary: CUT if pred < -50, else HOLD |
| B10 | day mult in {0.7, 1.0, 1.3} | risk regime: cap-day -> tighten B7+B9 thresholds |

**Cautious mode on B10 cap days** (NOT skip-day -- user pushback locked
this in 2026-05-18):
  - B7 skip threshold: 1.0 -> 1.5  (only take stronger entries)
  - B9 cut threshold:  -50 -> -25 (cut weaker losers faster)

Boost wing of B10 is structurally dead at 1c (cannot size up); cap days
are reinterpreted as risk-regime markers, not size cuts.

## V2 features in live -- on-demand, not streaming

`LiveFeatureEngineV2` subclasses `LiveFeatureEngine`. It inherits all bar
ingestion, per-TF aggregation, dedupe -- nothing new there. The single
addition is `get_v2_vector(ts) -> np.ndarray` which:

1. For each TF in `TF_ORDER`: runs `core_v2.StatisticalFieldEngine`'s
   `compute_L0/L1/L2/L3` on the cached per-TF bar DataFrame.
2. Step-fills each TF's per-bar values to the 5s anchor using
   `np.searchsorted(tf_ts, anchor_ts - period, side='right') - 1`
   (the lookahead-safe pattern from `build_dataset_v2.py:83`).
3. Returns a single 1xN row matching the column ordering of
   `core_v2.features.FEATURE_NAMES`.

V2 is computed only at three moments per day:
  - Session start (B10 day-mode -- but B10 uses cross-day, not V2)
  - R-trigger fire (B7 entry decision)
  - Each open-position bar at `bars_held == 5` (B9 K=5 cut decision)

The per-bar feature path stays V1 91D for the ledger / position counters.
This keeps the existing exit-logic untouched and avoids re-validating a
parallel streaming pipeline against batch V2.

## Engine-mode flag (zero-blast-radius rollout)

`LiveConfig.engine_mode` switches between:
  - `'blended'` -- existing BlendedEngine + V1 LiveFeatureEngine (default,
                  current production behaviour, unchanged)
  - `'l5'`      -- L5Decider + V1 + V2 LiveFeatureEngineV2

Default remains `'blended'`. To run SIM L5: `--engine-mode l5` on the
launcher CLI.

No code path other than `__init__` checks this flag. Everything downstream
(bar loop, fill handling, ledger, OrderManager) uses the same interfaces.

## Decision flow per bar (when engine_mode='l5')

```
on BAR:
  1. lfe_v1.on_bar(bar)        -> 91D vector  (ledger counters, exits)
  2. lfe_v2._append_bar(bar)   -> (caches per-TF bars; no V2 computed yet)
  3. ledger.update_bar(v1_vec, price, ts)
  4. state = build_state(v1=v1_vec, v2_getter=lfe_v2.get_v2_vector,
                          price=price, ts=ts, positions=ledger.snapshot())
  5. batch = l5_decider.evaluate(state)
       (inside evaluate, v2 is fetched ONLY when needed via the lazy getter)
  6. engine_v2 applies batch -> OrderManager.build_*_order -> NT8 send
```

The lazy `v2_getter` means we only pay the V2 compute cost on bars that
need it -- a handful per day. Streaming V2 was never the right pattern;
this is.

**Note on B9 K=5 clock skew:** Live `_PosTraj.entry_bar_count` snapshots when the ledger first SEES the position (post-FILL), so B9 fires ~1 bar later than the backtest's entry-bar anchor. Magnitude: one 5s bar. This is expected and a future 1-bar B9 timing discrepancy should not be chased as a bug.

## Parity test contract

`tools/test_live_v2_parity.py` replays a recorded day through
`LiveFeatureEngineV2` and compares column-by-column to the day's
`DATA/ATLAS/FEATURES_5s_v2/{L0,L1_*,L2_*,L3_*}/YYYY_MM_DD.parquet`.

Pass criterion: every column's max abs diff < 1e-9 on non-NaN cells,
NaN mask identical. Anything else is a bug.

## Status

  - [x] Architecture doc written (this file)
  - [ ] LiveFeatureEngineV2 implemented
  - [ ] Parity test passing
  - [ ] L5Decider implemented
  - [ ] engine_v2 wired with engine_mode flag
  - [ ] 1-contract OOS forward pass run
  - [ ] preflight_check.py
  - [ ] SIM deploy ready

See `docs/daily/2026-05-19.md` for daily build progress.
