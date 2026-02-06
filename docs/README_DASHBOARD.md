# Live Training Dashboard

A real-time visual interface for monitoring the Bayesian-AI training process.

## Features
- **4-Panel Layout:**
  - **Live Market Chart:** Visualizes recent 15-minute candles.
  - **Training Metrics:** Real-time iteration count, ETA, and elapsed time.
  - **Cumulative P&L:** Line chart of profit/loss over trades.
  - **Performance Stats:** Win rate, total trades, and learned states.
- **Auto-Update:** Polls `training/training_progress.json` every second.
- **Dark Theme:** Optimized for long monitoring sessions.

## Usage

1. **Start Training:**
   Run the training orchestrator (which generates the progress data).
   ```bash
   python training/orchestrator.py --data-dir DATA/RAW --iterations 100
   ```

2. **Launch Dashboard:**
   In a separate terminal:
   ```bash
   python visualization/live_training_dashboard.py
   ```

## Dependencies
- `tkinter` (System installed)
- `matplotlib`
- `pandas`
- `numpy`

Install python deps:
```bash
pip install -r requirements_dashboard.txt
```

## Data Source
Reads from `training/training_progress.json`. This file is updated by `training/orchestrator.py` at the end of each iteration.
