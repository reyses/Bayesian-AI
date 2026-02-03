# Debug Dashboard Notebook 🐞

Interactive Jupyter notebook for rapid validation, troubleshooting, and debugging of the Bayesian-AI system.

## 🚀 Quick Start

1. **Install Requirements**
   ```bash
   pip install -r requirements_notebook.txt
   ```

2. **Launch Notebook**
   ```bash
   jupyter notebook debug_dashboard.ipynb
   ```

3. **Run All Cells**
   - Click "Run All" to perform a full system health check.
   - Use interactive buttons for Mini Training and Utilities.

## 📘 Sections Overview

| Section | Purpose |
|---------|---------|
| **1. Environment Check** | Verifies Python, CUDA, and Data presence. |
| **2. Data Pipeline Test** | Loads sample data and visualizes price action to ensure loading logic works. |
| **3. Core Component Tests** | Instantiates `StateVector`, `BayesianBrain`, and `LayerEngine` to verify imports and basic functionality. |
| **4. Mini Training Run** | Interactive button to run `orchestrator.py` in a subprocess with live logs. |
| **5. Probability Table Analysis** | Visualizes learned patterns (Win Rate vs Sample Size) from `probability_table.pkl`. |
| **6. Performance Profiling** | Benchmarks hashing and velocity detection speeds. |
| **7. DOE Simulation Preview** | Generates a sample grid of parameters for optimization (Preview only). |
| **8. Quick Fixes** | Utilities to clear `__pycache__` and clean up. |

## 🛠️ Common Workflows

- **After Code Changes:** Run **Section 3** to verify components still load and pass basic logic.
- **Data Issues:** Run **Section 2** to inspect the raw data format and loading.
- **Training validation:** Run **Section 4** to see if the training loop executes without crashing, then **Section 5** to analyze results.
- **Performance Tuning:** Run **Section 6** to measure overhead.

## ⚠️ Notes

- **Output Directory:** Temporary files are saved to `debug_outputs/`.
- **Parallel Execution:** The notebook runs `orchestrator.py` as a subprocess to avoid blocking the kernel, but avoid running multiple training instances simultaneously.
- **CUDA:** If CUDA is unavailable, the system (and notebook) will fall back to CPU.
