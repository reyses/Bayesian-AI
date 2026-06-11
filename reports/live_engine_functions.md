# Live Engine Functions Report

The `live.engine_v2` acts as a lightweight production wrapper responsible for physics, execution, and state management, without housing complex feature generation logic internally. 

Its core functions revolve around a **7-Step Startup Pipeline** and an **Execution Loop**:

### The 7-Step Startup Pipeline
1. **CHECK**: Validates if the local ATLAS_NT8 historical dataset is up-to-date. If not, it requests a dump of missing days from NT8.
2. **BUILD**: Triggers feature extraction for any newly dumped historical days (offline).
3. **WARMUP**: Loads the last 5 days of ATLAS_NT8 + ATLAS_LIVE delta into memory to seed any moving aggregators.
4. **SYNC**: Establishes the TCP connection to NT8 and receives any historical bars that occurred between the last checkpoint and now.
5. **CATCH-UP**: Runs the engine over the received sync bars to bring internal state (e.g. ZigZag pivot detector) up to the current wall-clock time.
6. **VERIFY**: Validates that system latency is < 1s and the data feed is synchronized before arming trading.
7. **TRADE**: Enters the main live event loop.

### The Live Execution Loop (Step 7)
Once armed, the engine's sole purpose is to serve as a high-performance router and state tracker:
* **Market Data Routing**: Receives 5-second tick bars and routes them to the L5Decider logic.
* **Order Execution**: Sends entry and exit commands (e.g., `BAY_CLOSE`) to NT8 via the `OrderManager`.
* **Ledger & PnL Tracking**: Receives FILL events, updates the internal `Ledger`, tracks peak favorable/adverse excursions, and records realized/unrealized PnL.
* **Telemetry**: Periodically dumps state and telemetry to `v2_ledger_YYYY_MM_DD.csv` for dashboarding.

By offloading feature generation to the offline `ForwardPassSystem` and decision-making to the `L5Decider` (and the RL engine), the live wrapper focuses strictly on ensuring zero-slip execution, network syncing, and robust ledger tracking.
