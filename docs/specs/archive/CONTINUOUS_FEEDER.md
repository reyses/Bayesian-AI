# Continuous Data Feeder Spec

**Priority**: #1 next session
**Goal**: IS→OOS runs as one continuous stream, same engine, checkpoint between

## Architecture

```
ParquetFeeder(ATLAS)          ParquetFeeder(ATLAS_OOS)
       |                              |
       v                              v
  TradingEngine.feed_bar()  →  TradingEngine.feed_bar()
       |                              |
   (IS phase)                    (OOS phase)
       |                              |
  save checkpoint              save live_brain.pkl
  (brain + TBN)                (frozen for live)
```

## Flow

1. Initialize engine (fresh brain, TBN, exits, gates)
2. Feed ATLAS bars (IS) — engine trades + learns
3. At IS→OOS boundary: save `checkpoints/is_brain.pkl`
4. Continue feeding ATLAS_OOS bars (OOS) — SAME engine, brain keeps learning
5. At OOS end: save `checkpoints/live_brain.pkl`
6. Re-run OOS validation: load `live_brain.pkl` (frozen), feed ATLAS_OOS again
   Brain doesn't learn in this phase — pure validation

## What Changes

### New: `core/data_feeder.py`
```python
class BarFeeder:
    def iter_bars(self) -> Iterator[dict]:
        """Yields {timestamp, open, high, low, close, volume}"""
        raise NotImplementedError

class ParquetFeeder(BarFeeder):
    def __init__(self, atlas_root: str, tf: str = '15s'):
        self.files = sorted(Path(atlas_root, tf).glob('*.parquet'))

    def iter_bars(self):
        for f in self.files:
            df = pd.read_parquet(f)
            for _, row in df.iterrows():
                yield row.to_dict()
```

### Modified: `training/trainer.py`
- `run_forward_pass()` accepts a `BarFeeder` instead of `data_source` string
- Remove IS vs OOS branching — same loop for both
- Add `--continuous` flag: IS feeder → checkpoint → OOS feeder → live_brain

### Not modified
- `core/execution_engine.py` — already unified
- `core/exit_engine.py` — already shared
- `core/bar_processor.py` — already has peak detection
- `live/live_engine.py` — will use NT8Feeder (future)

## Checkpoint Format

```python
checkpoint = {
    'brain': brain.save_state(),
    'tbn_worker_states': {tf: worker.get_state() for tf, worker in tbn.workers},
    'bar_index': last_bar_index,
    'running_pnl': pnl,
    'timestamp': last_timestamp,
}
```

## CLI

```bash
# Full continuous run (IS → OOS → save live_brain)
python training/trainer.py --continuous

# OOS validation only (load live_brain, no learning)
python training/trainer.py --oos --frozen-brain

# Live (uses live_brain.pkl)
python -m live.launcher
```

## Verification

1. Run `--continuous`: IS PnL should match standalone IS
2. OOS should show balanced direction (brain bias fixed)
3. Peak trades should appear in both IS and OOS
4. `--oos --frozen-brain` should reproduce same OOS results
5. January direction split should be ~50/50 not 99% SHORT
