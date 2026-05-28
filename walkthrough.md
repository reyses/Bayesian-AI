# V2 Historical Physics Engine Integration

The simulated curriculum is now running on the actual data physics using the V2 Forward Pass System! We have completely replaced the random mock generators with the true historical features and empirical exit rules.

## 1. V2 Feature Grid Pipeline (`network.py`)
The `MasterNetwork` no longer expects an arbitrary 13-feature array. We have successfully re-wired it to ingest the exact **185-feature** array provided by `core_v2.FPS`:
- We stripped out the 1 `L0` feature.
- We reshaped the remaining 184 features into an `8x23` grid (8 Timeframes x 23 features).
- The `8x23` grid is processed organically by `Conv2D` across time, meaning the CNN treats the 8 timeframes exactly like cross-correlating RGB image channels.
- The `L0` base feature is concatenated directly onto the flattened CNN output right before entering the LSTM.

> [!TIP]
> Per your instruction, **the Regime feature embedding was intentionally dropped** and is entirely isolated from the data pipeline, preventing any taint or lookahead bias during the convolution.

## 2. HDF5 Substrate Update (`hdf5_shadow_queue.py`)
To handle the new V2 data ingestion smoothly, the HDF5 disk storage has been updated from a single `states` dataset to directly storing `v2_grids` `[Batch, 8, 60, 23]` and `l0_features` `[Batch, 60, 1]` independently. 

## 3. Structural Physics Integration (`train_historical.py`)
Instead of fixed `trade_pnl` simulation ticks, the environment now perfectly mirrors actual live operations using `core_v2.ledger.Ledger`:
- The curriculum partitions your Parquet directories into chronological 5-day chunks (weeks). 
- Trades execute instantly when the RL agent issues a `Buy` or `Sell` action by opening a `Ledger.Position`.
- The environment continues streaming `BarState` arrays into the ledger via `MultiDayForwardPassSystem`.
- At each tick, the open trade evaluates `core_v2.exits.default_exit_suite()`. When a rule like `MFEArmedGiveback` triggers, it locks in the structural exit and calculates the realized PnL. This final PnL serves as the pure reward scalar sent down to the `HDF5ShadowQueue` for backpropagation!

---
**Status:** The data pipeline, PyTorch ONNX exports, and training loop are fully synthesized. You can run `python train_historical.py --smoke-test` to test the true physics curriculum pipeline!
