# Dashboard Guide 🧠

The **Bayesian-AI Dashboard** (`notebooks/dashboard.ipynb`) is the primary consolidated interface for verifying the system environment, performing rapid learning simulations, and executing full training cycles.

## 🚀 Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure `jupyter` and `ipywidgets` are installed.*

2. **Generate/Reset Notebook**
   If the notebook is missing or you want to reset it to the latest version:
   ```bash
   python scripts/generate_dashboard.py
   ```

3. **Launch Notebook**
   ```bash
   jupyter notebook notebooks/dashboard.ipynb
   ```

## 📘 Workflow Sections

### 1. Preflight & Environment Checks ✈️
Verifies the operational readiness of the system.
*   **Operational Mode**: Must be set to `LEARNING` in `config/settings.py`.
*   **Environment**: Checks Python version and path resolution.
*   **CUDA Audit**: Runs the 3-stage hardened verification (Handshake -> Injection -> Handoff) to ensure the GPU is ready for heavy lifting. Logs details to `CUDA_Debug.log`.

### 2. Data Pipeline Test 📊
*   Validates loading of a single data file from `DATA/RAW`.
*   Ensures that data can be read and processed correctly before attempting larger operations.

### 3. Core Component Tests ⚙️
*   Verifies initialization of critical modules: `StateVector`, `BayesianBrain`, and `LayerEngine`.
*   Checks if CUDA acceleration is active for the Layer Engine.

### 4. Quick Learn: 3 Discrete Day Simulation 🎲
This is the "Quick Learning" phase.
*   **Action**: Randomly selects 3 distinct files/days from the available dataset.
*   **Process**: Runs a "simulation" (1 training iteration) for each file individually using the `TrainingOrchestrator`.
*   **Goal**: Rapidly verify that the logic holds up across different market conditions without waiting for a full historical run.
*   **Output**: Displays Win Rate and PnL for each of the 3 sampled files.

### 5. Mini Training Run (5 Iterations) 🏃‍♂️
*   Runs a short, interactive 5-iteration training session on the sample data loaded in Section 2.
*   Useful for a quick end-to-end test of the training loop logic.

### 6. Full Learning Cycle (Production Run) 🚀
Once confident in the Quick Learn and Mini Run results:
*   **Action**: Triggers the `TrainingOrchestrator` on the **entire** dataset.
*   **Parameters**: Default 50 iterations (can be adjusted in the cell code).
*   **Output**: Saves the final trained model to `models/production_learning/probability_table.pkl` and visualizes PnL/Confidence in real-time.

### 7. Result Analysis & Visualization 📈
*   Inspects the learned probability tables.
*   Compares the "Quick Learn", "Mini Run", and "Production Model" results.
*   Integrates with the Visualization Module to plot training results.

## ⚠️ Notes

- **Operational Mode**: The system enforces `LEARNING` mode to prevent accidental live execution commands during training.
- **Resource Usage**: The "Quick Learn" step loads data into memory. Ensure you have sufficient RAM.
- **CUDA**: The Full Learning Cycle enforces GPU usage by default. Ensure your environment is CUDA-ready.
