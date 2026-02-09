# CONSOLIDATION APPROVED - EXECUTE NOW

**Date:** February 9, 2026
**Status:** USER APPROVED - PROCEED WITH ALL PHASES
**Estimated Time:** 4.0 hours
**Goal:** Functional single-entry-point training system with live dashboard

---

## USER'S DIRECTIVE

**"Option 1, let's consolidate, let's hit a functional product"**

Stop designing future features. Build what's been planned. Make it work.

---

## EXECUTE THESE 4 PHASES

### ✅ PHASE 1: Archive Old Code (15 minutes)

Move these files to `archive/`:
- `core/exploration_mode.py`
- `core/unconstrained_explorer.py`
- `core/engine_core.py`
- `core/fractal_three_body.py`
- `core/resonance_cascade.py`
- `training/cuda_backtest.py`
- `training/run_optimizer.py`
- All files in `scripts/` folder
- 18 files total

**Do NOT delete** - move to archive for reference.

---

### ✅ PHASE 2: Create New Components (60 minutes)

**File 1:** `training/pattern_analyzer.py`
```python
"""
Analyze learned states and identify strongest patterns
"""

def get_strongest_patterns(brain, top_n=20):
    """Top N states by win rate with confidence > 80%"""
    pass

def analyze_by_context(brain):
    """Breakdown performance by context (killzone, pattern, etc.)"""
    pass

def generate_pattern_report(brain, day_results):
    """Complete pattern analysis"""
    pass
```

**File 2:** `training/progress_reporter.py`
```python
"""
Real-time terminal output with metrics tracking
"""

class ProgressReporter:
    def update(self, day_result):
        """Print day summary"""
        pass
    
    def print_trade_summary(self, trade):
        """Print trade result with duration and efficiency"""
        pass
```

**File 3:** `execution/batch_regret_analyzer.py`
```python
"""
Batch multi-timeframe regret analysis
Runs at END OF DAY, not during trading
"""

class BatchRegretAnalyzer:
    def batch_analyze_day(self, all_trades_today, full_day_data_30s):
        """
        Analyze all trades with 2m context
        Find patterns: "70% exited early in trends"
        Generate parameter adjustments
        """
        pass
```

---

### ✅ PHASE 3: Consolidate Orchestrator (90 minutes)

**Merge `walk_forward_trainer.py` INTO `orchestrator.py`:**

**orchestrator.py structure:**
```python
"""
SINGLE ENTRY POINT - All training features integrated
"""

from visualization.live_training_dashboard import LiveDashboard
from training.pattern_analyzer import PatternAnalyzer
from training.progress_reporter import ProgressReporter
from execution.batch_regret_analyzer import BatchRegretAnalyzer
from training.doe_parameter_generator import DOEParameterGenerator
from core.context_detector import ContextDetector

class BayesianTrainingOrchestrator:
    def __init__(self, config):
        # Initialize all components
        self.brain = QuantumBayesianBrain()
        self.engine = QuantumFieldEngine()
        self.context_detector = ContextDetector()
        self.param_generator = DOEParameterGenerator()
        self.pattern_analyzer = PatternAnalyzer()
        self.progress_reporter = ProgressReporter()
        self.regret_analyzer = BatchRegretAnalyzer()
        self.dashboard = None
        
        # Tracking
        self.todays_trades = []
    
    def train(self, data, config):
        """Master training loop"""
        
        # Launch dashboard in background thread
        self.launch_dashboard()
        
        # Split data into days
        days = self.split_into_trading_days(data)
        
        # Train day by day
        for day_idx, (date, day_data) in enumerate(days):
            print(f"\n{'='*60}")
            print(f"DAY {day_idx + 1}: {date}")
            print(f"{'='*60}")
            
            # Run DOE optimization on this day
            day_result = self.optimize_day(day_idx, date, day_data)
            
            # Batch regret analysis (end of day)
            regret_report = self.regret_analyzer.batch_analyze_day(
                self.todays_trades,
                day_data
            )
            
            # Update dashboard
            self.dashboard.update(day_result, regret_report)
            
            # Save checkpoint
            self.save_checkpoint(day_idx, date, day_result)
            
            # Print summary
            self.print_day_summary(day_result, regret_report)
        
        # Final summary
        self.print_final_summary()
    
    def optimize_day(self, day_idx, date, day_data):
        """Run 1000 parameter iterations on single day"""
        
        self.todays_trades = []  # Reset
        best_sharpe = -999
        best_params = None
        
        for iteration in range(1000):
            # Generate params
            params = self.param_generator.generate(iteration, day_idx)
            
            # Simulate trading (FAST - no regret analysis)
            trades = self.simulate_trading_day(day_data, params)
            
            # Log trades
            self.todays_trades.extend(trades)
            
            # Track best
            sharpe = self.calculate_sharpe(trades)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = params
        
        return {
            'day': day_idx,
            'date': date,
            'best_params': best_params,
            'best_sharpe': best_sharpe,
            'total_trades': len(self.todays_trades)
        }
    
    def simulate_trading_day(self, day_data, params):
        """Fast simulation - no regret overhead"""
        trades = []
        
        for bar in day_data:
            state = self.engine.compute_state(bar)
            
            if self.brain.should_fire(state, params):
                trade = self.execute_trade(bar, state, params)
                trades.append(trade)
        
        return trades
    
    def launch_dashboard(self):
        """Launch dashboard in background thread"""
        import threading
        
        def run_dashboard():
            self.dashboard = LiveDashboard()
            self.dashboard.launch()
        
        thread = threading.Thread(target=run_dashboard, daemon=True)
        thread.start()

def main():
    """Single entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--max-days', type=int, default=None)
    args = parser.parse_args()
    
    # Load data
    data = pd.read_parquet(args.data)
    
    # Run training
    orchestrator = BayesianTrainingOrchestrator(args)
    orchestrator.train(data, args)

if __name__ == "__main__":
    main()
```

**DELETE (after merge):**
- `training/walk_forward_trainer.py` → Move to `archive/walk_forward_trainer_merged.py`

---

### ✅ PHASE 4: Testing (30 minutes)

**Smoke Test:**
```bash
python training/orchestrator.py \
    --data DATA/glbx-mdp3-20251230-20260129.ohlcv-1s.parquet \
    --max-days 2 \
    --iterations 100
```

**Verify:**
- [ ] Dashboard window opens
- [ ] Training runs without errors
- [ ] Progress bars display
- [ ] Charts update after each day
- [ ] Checkpoints save
- [ ] Terminal output readable
- [ ] Regret analysis prints at end of day

---

## SUCCESS CRITERIA

When complete, user can run:

```bash
python training/orchestrator.py --data DATA/file.parquet
```

**And see:**
1. Dashboard opens (charts appear)
2. Terminal: "Training on 22 days..."
3. Progress bars updating
4. Charts updating per day (P&L, win rate, Sharpe)
5. End of day: Regret analysis report
6. Checkpoints saving to `checkpoints/`
7. Final summary when complete

---

## WHAT TO INCLUDE (From All Discussions)

**Duration Tracking:**
- Average trade duration (overall, wins, losses)
- Duration by pattern type
- Display in terminal and dashboard

**Batch Regret Analysis:**
- Execute trades fast (no overhead)
- At end of day: Resample to 2m, analyze all trades
- Find patterns: "70% early exits in trends"
- Adjust parameters for next day
- Display aggregate report

**Dashboard:**
- Use existing `live_training_dashboard.py` (tkinter)
- 2x2 grid: P&L, Win Rate, Sharpe, Current Metrics
- Update after each day
- Add regret efficiency panel

---

## WHAT TO SKIP (Future Phases)

**DO NOT BUILD YET:**
- Wave Rider v2.0 (position flipping) - Phase 3+
- Multi-timeframe decision hierarchy (1m/1s) - Phase 3+
- CUDA optimization - Phase 4+
- Any feature marked "Future" in documents

**BUILD ONLY:** The consolidation plan as approved.

---

## REFERENCE DOCUMENTS

All in `/mnt/user-data/outputs/`:
- `CONSOLIDATION_AUDIT_REPORT.md` - Your audit (approved)
- `CONSOLIDATION_ADDENDUM.md` - Duration + regret additions
- `MULTI_TIMEFRAME_REGRET.md` - Batch regret specification
- `WAVE_RIDER_V2_CONCEPT.md` - Future phase (DON'T BUILD YET)

---

## TIME BUDGET

| Phase | Time | Status |
|-------|------|--------|
| Phase 1: Archive | 15 min | Approved |
| Phase 2: New Components | 60 min | Approved |
| Phase 3: Consolidate | 90 min | Approved |
| Phase 4: Testing | 30 min | Approved |
| **TOTAL** | **195 min** | **~3.25 hours** |

---

## START NOW

Execute Phase 1 immediately. Report progress after each phase.

User wants a **functional product today**, not more design documents.

**GO BUILD IT.**
