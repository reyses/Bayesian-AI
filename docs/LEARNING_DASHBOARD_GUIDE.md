# Learning Dashboard Guide 🧠

The **Learning Dashboard** (`notebooks/learning_dashboard.ipynb`) is the primary interface for validating the system environment and performing rapid learning simulations before executing full training cycles.

## 🚀 Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure `jupyter` and `ipywidgets` are installed.*

2. **Generate/Reset Notebook**
   If the notebook is missing or you want to reset it to the latest version:
   ```bash
   python scripts/generate_learning_dashboard.py
   ```

3. **Launch Notebook**
   ```bash
   jupyter notebook notebooks/learning_dashboard.ipynb
   ```

## 📘 Workflow Sections

### 1. Preflight Checks ✈️
Verifies the operational readiness of the system.
*   **Operational Mode**: Must be set to `LEARNING` in `config/settings.py`.
*   **Environment**: Checks Python version and path resolution.
*   **CUDA Audit**: Runs the 3-stage hardened verification (Handshake -> Injection -> Handoff) to ensure the GPU is ready for heavy lifting. Logs details to `CUDA_Debug.log`.

### 2. Data Setup & Verification 📊
*   Loads the entire dataset from `DATA/RAW` into memory.
*   Displays the date range and a sample candlestick chart.
*   *Critical Check*: Ensure your data covers the expected periods.

### 3. Quick Learn: 3 Discrete Day Simulation 🎲
This is the "Quick Learning" phase.
*   **Action**: Randomly selects 3 distinct days from the available dataset.
*   **Process**: Runs a "simulation" (1 training iteration) for each day individually using the `TrainingOrchestrator`.
*   **Goal**: Rapidly verify that the logic holds up across different market conditions (e.g., trend day vs. chop day) without waiting for a full historical run.
*   **Output**: Displays Win Rate and PnL for each of the 3 sampled days.

### 4. Full Learning Cycle 🚀
Once confident in the Quick Learn results:
*   **Action**: Triggers the `TrainingOrchestrator` on the **entire** dataset.
*   **Parameters**: Default 10 iterations (can be adjusted in the cell code).
*   **Output**: Saves the final trained model to `models/probability_table.pkl`.

### 5. Result Analysis 📈
*   Inspects the generated probability tables.
*   Compares the "Quick Learn" temporary model vs the "Main Model".

## ⚠️ Notes

- **Operational Mode**: The system enforces `LEARNING` mode to prevent accidental live execution commands during training.
- **Resource Usage**: The "Quick Learn" step loads the full dataset into memory to perform slicing. Ensure you have sufficient RAM (or modify the loader to lazy-load if dataset exceeds 10GB).
- **Output Directory**: Temporary files from Quick Learn are saved to `debug_outputs/quick_learn/` to avoid overwriting your production model.
