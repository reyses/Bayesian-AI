# Progress Tracking Integration Guide

## Files Created

1. **orchestrator_enhanced.py** - Enhanced orchestrator with tqdm progress bars
2. **test_progress_display.py** - Demo script showing progress display in action

## Quick Test

Run the demo to see how it looks:

```bash
cd /path/to/your/repo
python test_progress_display.py
```

Expected output:
```
[PHASE 0: EXPLORATION] 100%|████████████| 50/50 [00:02<00:00]
Trades:  234 | States:  78 | P&L: $  1,234.56 | WR:  58.2%

[PHASE 1: ADAPTIVE] 100%|██████████████| 50/50 [00:02<00:00]
Trades:  345 | States: 124 | P&L: $  2,456.78 | WR:  64.1%
```

## Integration Steps

### Option 1: Replace your orchestrator.py

```bash
# Backup original
cp training/orchestrator.py training/orchestrator.py.backup

# Replace with enhanced version
cp orchestrator_enhanced.py training/orchestrator.py
```

### Option 2: Add to existing orchestrator.py

Add this to the top of your `training/orchestrator.py`:

```python
from tqdm import tqdm

class TrainingProgressBar:
    """Manages nested progress bars for phases and iterations"""
    
    def __init__(self, total_iterations: int, phase_name: str = "Training"):
        self.total_iterations = total_iterations
        self.phase_name = phase_name
        
        # Main progress bar
        self.pbar_main = tqdm(
            total=total_iterations,
            desc=f"[{phase_name}]",
            position=0,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )
        
        # Metrics display
        self.pbar_metrics = tqdm(
            total=0,
            bar_format='{desc}',
            position=1,
            leave=False
        )
    
    def update(self, n=1, **metrics):
        """Update progress and metrics"""
        self.pbar_main.update(n)
        
        # Format metrics
        trades = metrics.get('trades', 0)
        states = metrics.get('states', 0)
        pnl = metrics.get('pnl', 0.0)
        wr = metrics.get('win_rate', 0.0)
        
        metrics_str = f"Trades: {trades:>6} | States: {states:>5} | P&L: ${pnl:>8,.2f} | WR: {wr:>5.1%}"
        self.pbar_metrics.set_description_str(metrics_str)
    
    def close(self):
        self.pbar_main.close()
        self.pbar_metrics.close()
```

Then in your `run_training()` method:

```python
def run_training(self, iterations=1000, params=None, on_progress=None):
    # Initialize progress
    progress = TrainingProgressBar(
        total_iterations=iterations,
        phase_name=f"{self.mode} MODE"
    )
    
    try:
        for iteration in range(iterations):
            # Your existing training logic
            # ...
            
            # Update progress
            progress.update(
                n=1,
                trades=len(self.engine.trades),
                states=len(self.engine.prob_table.table),
                pnl=self.engine.daily_pnl,
                win_rate=self._get_win_rate()
            )
    finally:
        progress.close()
```

## For Notebook Integration

The progress bars work in Jupyter notebooks too! In your dashboard.ipynb:

```python
from training.orchestrator import TrainingOrchestrator

# Progress will display in notebook output
orch = TrainingOrchestrator(...)
orch.run_training(iterations=100)
```

## Suppressing Progress (for background jobs)

If running in background or logging to file:

```python
# Disable progress bars
import os
os.environ['TQDM_DISABLE'] = '1'

# Or use quiet mode
progress = TrainingProgressBar(..., disable=True)
```

## Customization

### Change colors (requires colorama):
```python
from tqdm import tqdm
from colorama import Fore, Style

pbar = tqdm(..., bar_format=f'{Fore.GREEN}{{l_bar}}{{bar}}{Style.RESET_ALL}')
```

### Add more metrics:
```python
# In TrainingProgressBar.update()
conf = metrics.get('confidence', 0.0)
phase = metrics.get('phase', 'N/A')

metrics_str = f"Phase: {phase} | Trades: {trades} | Conf: {conf:.1%} | ..."
```

### Nested progress (day/iteration):
```python
# Outer loop (days)
pbar_days = tqdm(total=5, desc="Days", position=0)

for day in range(5):
    # Inner loop (iterations)
    pbar_iter = tqdm(total=100, desc=f"Day {day+1}", position=1)
    
    for iteration in range(100):
        # Training...
        pbar_iter.update(1)
    
    pbar_iter.close()
    pbar_days.update(1)
```

## Commit Instructions

Since I can't commit directly:

```bash
# 1. Copy files to your repo
cp orchestrator_enhanced.py /path/to/Bayesian-AI/training/orchestrator.py
cp test_progress_display.py /path/to/Bayesian-AI/tests/

# 2. Test
cd /path/to/Bayesian-AI
python tests/test_progress_display.py

# 3. Commit
git add training/orchestrator.py tests/test_progress_display.py
git commit -m "feat: Add tqdm progress tracking to training orchestrator

- Real-time progress bars showing iteration progress
- Live metrics: trades, states, P&L, win rate
- Phase transition indicators
- Nested progress for multi-day training
- Works in both CLI and Jupyter notebooks"

git push
```

## Troubleshooting

**Progress bars not showing:**
- Install tqdm: `pip install tqdm`
- Check if running in IDE that doesn't support ANSI (use `disable=True`)

**Progress bars overlapping:**
- Use `position` parameter to stack them vertically
- Close bars explicitly with `.close()`

**Notebook display issues:**
- Restart kernel if bars get stuck
- Use `tqdm.notebook.tqdm` for better notebook support:
  ```python
  from tqdm.notebook import tqdm
  ```

**Too much output:**
- Increase update frequency: `if iteration % 10 == 0: progress.update()`
- Use miniters parameter: `tqdm(..., miniters=10)`
