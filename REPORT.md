# Bayesian-AI: Math Application & Databento Integration Report

## 1. Mathematical Application

The Bayesian-AI system applies mathematics primarily through its **Bayesian Probability Engine** and **High-Frequency State Analysis**.

### A. Bayesian Probability (`core/bayesian_brain.py`)

The core decision-making logic relies on Bayesian inference to estimate the probability of a "WIN" outcome given a specific market state.

1.  **State Representation**:
    The market is discretized into a unique `StateVector` composed of 9 hierarchical layers (L1-L9).
    $$ S = \{L1, L2, L3, L4, L5, L6, L7, L8, L9\} $$
    This vector serves as the key for the probability lookup table.

2.  **Probability Estimation**:
    The system calculates the win probability $P(Win|S)$ using **Laplace Smoothing** to handle small sample sizes and avoid zero-probability issues.

    $$ P(Win|S) = \frac{Wins + 1}{Total\_Trials + 2} $$

    *   **Prior**: With 0 data, $P(Win|S) = \frac{0+1}{0+2} = 0.5$ (Neutral 50%).
    *   **Updates**: As trades execute, the counts ($Wins$, $Total$) are updated in the `probability_table.pkl`.

3.  **Confidence Metric**:
    A confidence score determines if the sample size is sufficient to trust the probability estimate.

    $$ C(S) = \min\left(\frac{Total\_Trials}{30}, 1.0\right) $$

    *   Full confidence (1.0) is achieved at **30 samples**.
    *   A trade is only executed if $P(Win|S) \ge 0.80$ and $C(S) \ge 0.30$.

### B. High-Frequency Math (`cuda/velocity_gate.py`)

The **L9 Velocity Gate** prevents execution during adverse high-volatility events (cascades) using a sliding window algorithm.

1.  **Cascade Detection**:
    The system analyzes the last **50 ticks** to detect rapid price displacements.

    $$ \Delta P = | \max(P_{window}) - \min(P_{window}) | $$
    $$ \Delta t = t_{end} - t_{start} $$

    A cascade is flagged if:
    $$ \Delta P \ge 10.0 \text{ points} \quad \text{AND} \quad \Delta t \le 0.5 \text{ seconds} $$

2.  **Optimization**:
    To maintain $O(1)$ performance on high-frequency data streams, the system creates a localized slice (buffer) of the last **200 ticks** before passing data to the CPU/GPU, ensuring processing time does not scale with total history size.

## 2. Databento File Usage

The system is designed to ingest and process high-fidelity market data from **Databento** (`.dbn` files).

### A. Ingestion Pipeline (`training/databento_loader.py`)

1.  **Loading**:
    The system uses the `databento` Python library to read compressed Databento files (`.dbn.zst`).
    ```python
    data = db.DBNStore.from_file(filepath)
    df = data.to_df()
    ```

2.  **Normalization**:
    Raw Databento columns are mapped to the internal schema:
    *   `ts_event` / `ts_recv` $\rightarrow$ `timestamp` (converted to UNIX float seconds)
    *   `price` / `close` $\rightarrow$ `price`
    *   `size` / `volume` $\rightarrow$ `volume`
    *   `action` / `side` $\rightarrow$ `type`

    *Update*: The loader was enhanced to preserve OHLC columns (`open`, `high`, `low`) when processing OHLCV data, ensuring compatibility with Layer 1-4 static context generation.

### B. Data Flow

1.  **Training Mode**:
    *   **Input**: Historical `.dbn` files or `.parquet` archives in `DATA/RAW`.
    *   **Process**: `TrainingOrchestrator` iterates through ticks, simulating the `BayesianEngine` response to build the probability table.
    *   **Output**: A trained `probability_table.pkl`.

2.  **Verification**:
    *   Scripts like `scripts/setup_test_data.py` convert raw Databento trades into standardized Parquet files (`trades.parquet`, `ohlcv-1s.parquet`) to facilitate rapid integration testing without re-parsing large `.dbn` files repeatedly.
