"""
Adaptive Learning Training Orchestrator
Integrates all components for end-to-end learning
Includes Phase 0: Unconstrained Exploration
"""
from core.quantum_field_engine import QuantumFieldEngine
from core.three_body_state import ThreeBodyQuantumState
from core.bayesian_brain import QuantumBayesianBrain, TradeOutcome
from core.adaptive_confidence import AdaptiveConfidenceManager
from core.fractal_three_body import FractalMarketState, FractalTradingLogic
from core.resonance_cascade import ResonanceCascadeDetector
from core.exploration_mode import UnconstrainedExplorer, ExplorationConfig
from config.symbols import MNQ, calculate_pnl
import pandas as pd
import argparse
import sys
import os

def train_complete_system(historical_data_path: str, max_iterations: int = 1000, output_dir: str = 'models/', phase0_enabled: bool = True):
    """
    Complete training pipeline with all 5 layers
    
    PHASE 0: Unconstrained exploration (500 trades, no rules)
    PHASE 1-4: Adaptive confidence (0% -> 80% threshold)
    """
    
    # Initialize components
    field_engine = QuantumFieldEngine(regression_period=21)
    brain = QuantumBayesianBrain()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("[LOADING] Historical data...")
    df_raw = load_data(historical_data_path)
    
    # Resample to required timeframes
    # Ensure index is DatetimeIndex
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        if 'timestamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
            df_raw.set_index('timestamp', inplace=True)
        else:
             # Try converting index
             df_raw.index = pd.to_datetime(df_raw.index)

    df_15m = df_raw.resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    df_15s = df_raw.resample('15s').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    })
    
    # Ensure timezone awareness matches
    if df_15m.index.tz is None and df_15s.index.tz is not None:
        df_15m.index = df_15m.index.tz_localize(df_15s.index.tz)
    if df_15m.index.tz is not None and df_15s.index.tz is None:
        df_15s.index = df_15s.index.tz_localize(df_15m.index.tz)

    print(f"[DATA] Loaded {len(df_15m)} 15min bars, {len(df_15s)} 15sec bars")
    
    # ---------------------------------------------------------
    # PHASE 0: UNCONSTRAINED EXPLORATION
    # ---------------------------------------------------------
    if phase0_enabled:
        explorer = UnconstrainedExplorer(
            ExplorationConfig(
                max_trades=500,
                min_unique_states=50,
                fire_probability=1.0,      # Fire on everything
                ignore_all_gates=True,     # Ignore L8/L9
                allow_chaos_zone=True,     # Trade anywhere
                learn_from_failures=True
            )
        )
        
        print("\n" + "="*70)
        print("PHASE 0: UNCONSTRAINED EXPLORATION")
        print("="*70)
        print("Trading without constraints to discover patterns...")
        print()
        
        trades_phase0 = []
        
        # Iterate for Phase 0
        # Re-using the logic from the main loop but simplified for exploration
        # We need a fresh iterator or reset logic if we want to run Phase 1 on same data
        # For simplicity, we run Phase 0 on the *same* dataset as Phase 1, 
        # but typically exploration happens on a subset or we reset.
        # Here we will iterate the dataset for Phase 0.
        
        for i in range(100, len(df_15m) - 20):
            if explorer.is_complete():
                break
                
            current_macro_time = df_15m.index[i]
            macro_start_time = current_macro_time
            macro_end_time = current_macro_time + pd.Timedelta(minutes=15)
            
            macro_window = df_15m.loc[:current_macro_time].iloc[-100:]
            micro_window = df_15s.loc[macro_start_time:macro_end_time]
            
            if len(micro_window) == 0:
                continue
            
            current_price = macro_window['close'].iloc[-1]
            current_volume = macro_window['volume'].iloc[-1]
            
            if len(micro_window) >= 2:
                tick_velocity = (micro_window['close'].iloc[-1] - micro_window['close'].iloc[-2]) / 15.0
            else:
                tick_velocity = 0.0
            
            quantum_state = field_engine.calculate_three_body_state(
                macro_window, micro_window, current_price, current_volume, tick_velocity
            )
            
            # UNCONSTRAINED DECISION
            decision = explorer.should_fire(quantum_state)
            
            if decision['should_fire']:
                entry_price = current_price
                
                # EXPLORATION: Try direction based on z-score or random if neutral
                if quantum_state.z_score > 0:
                    side = 'short'
                    target = quantum_state.center_position
                    stop = quantum_state.event_horizon_upper
                elif quantum_state.z_score < 0:
                    side = 'long'
                    target = quantum_state.center_position
                    stop = quantum_state.event_horizon_lower
                else:
                    # Random in chaos zone
                    side = 'long' if (i % 2 == 0) else 'short' # Deterministic random for reproducibility
                    target = current_price * 1.001 if side == 'long' else current_price * 0.999
                    stop = current_price * 0.999 if side == 'long' else current_price * 1.001

                future_start = macro_end_time
                future_end = future_start + pd.Timedelta(minutes=15*20)
                future = df_15m.loc[future_start:future_end]
                
                if len(future) == 0:
                    continue
                
                if side == 'short':
                    hit_target = future['low'].min() <= target
                    hit_stop = future['high'].max() >= stop
                    exit_price = target if hit_target and not hit_stop else stop
                else:
                    hit_target = future['high'].max() >= target
                    hit_stop = future['low'].min() <= stop
                    exit_price = target if hit_target and not hit_stop else stop
                
                result = 'WIN' if (hit_target and not hit_stop) else 'LOSS'
                pnl = calculate_pnl(MNQ, entry_price, exit_price, side)
                
                outcome = TradeOutcome(
                    state=quantum_state,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                    result=result,
                    timestamp=quantum_state.timestamp,
                    exit_reason='target' if result == 'WIN' else 'stop'
                )
                
                brain.update(outcome)
                explorer.record_trade(outcome)
                trades_phase0.append(outcome)
                
        # Report Phase 0
        print(explorer.get_completion_report())
        if len(trades_phase0) > 0:
            phase0_winrate = sum(1 for t in trades_phase0 if t.result == 'WIN') / len(trades_phase0)
            phase0_pnl = sum(t.pnl for t in trades_phase0)
            print(f"Phase 0 Results:")
            print(f"  Win Rate: {phase0_winrate:.2%}")
            print(f"  Total P&L: ${phase0_pnl:.2f}")
            print(f"  States Learned: {len(brain.table)}")
        print()

    # ---------------------------------------------------------
    # PHASE 1-4: ADAPTIVE CONFIDENCE LEARNING
    # ---------------------------------------------------------
    print("\n" + "="*70)
    print("PHASE 1-4: ADAPTIVE CONFIDENCE LEARNING")
    print("="*70)
    print("Starting with pre-populated probability table from exploration...")
    print()
    
    confidence_mgr = AdaptiveConfidenceManager(brain)
    fractal_analyzer = FractalMarketState() # Initialized but not used in Phase 0 yet
    resonance_detector = ResonanceCascadeDetector() # Initialized but not used in Phase 0 yet
    
    trades_executed = []
    
    # Iterate for main training (Phase 1-4)
    # We iterate the SAME dataset again, but now applying the learned probabilities
    # In a real scenario, we might use a subsequent period.
    # Here we are simulating "learning on the job".
    
    for i in range(100, len(df_15m) - 20):
        current_macro_time = df_15m.index[i]
        macro_start_time = current_macro_time
        macro_end_time = current_macro_time + pd.Timedelta(minutes=15)
        
        macro_window = df_15m.loc[:current_macro_time].iloc[-100:]
        micro_window = df_15s.loc[macro_start_time:macro_end_time]
        
        if len(micro_window) == 0:
            continue
        
        current_price = macro_window['close'].iloc[-1]
        current_volume = macro_window['volume'].iloc[-1]
        
        if len(micro_window) >= 2:
            tick_velocity = (micro_window['close'].iloc[-1] - micro_window['close'].iloc[-2]) / 15.0
        else:
            tick_velocity = 0.0
        
        quantum_state = field_engine.calculate_three_body_state(
            macro_window, micro_window, current_price, current_volume, tick_velocity
        )
        
        decision = confidence_mgr.should_fire(quantum_state)
        
        if decision['should_fire']:
            entry_price = current_price
            target = quantum_state.center_position
            
            future_start = macro_end_time
            future_end = future_start + pd.Timedelta(minutes=15*20)
            future = df_15m.loc[future_start:future_end]
            
            if len(future) == 0:
                continue
            
            if quantum_state.z_score > 0:
                side = 'short'
                hit_target = future['low'].min() <= target
                hit_stop = future['high'].max() >= quantum_state.event_horizon_upper
                exit_price = target if hit_target and not hit_stop else quantum_state.event_horizon_upper
            else:
                side = 'long'
                hit_target = future['high'].max() >= target
                hit_stop = future['low'].min() <= quantum_state.event_horizon_lower
                exit_price = target if hit_target and not hit_stop else quantum_state.event_horizon_lower
            
            result = 'WIN' if (hit_target and not hit_stop) else 'LOSS'
            pnl = calculate_pnl(MNQ, entry_price, exit_price, side)
            
            outcome = TradeOutcome(
                state=quantum_state,
                entry_price=entry_price,
                exit_price=exit_price,
                pnl=pnl,
                result=result,
                timestamp=quantum_state.timestamp,
                exit_reason='target' if result == 'WIN' else 'stop'
            )
            
            brain.update(outcome)
            confidence_mgr.record_trade(outcome)
            trades_executed.append(outcome)
            
            if len(trades_executed) % 50 == 0:
                print(confidence_mgr.generate_progress_report())
    
    # Save trained model
    model_path = os.path.join(output_dir, 'quantum_probability_table.pkl')
    brain.save(model_path)
    
    # Final statistics
    if len(trades_executed) > 0:
        win_rate = sum(1 for t in trades_executed if t.result == 'WIN') / len(trades_executed)
        total_pnl = sum(t.pnl for t in trades_executed)
    else:
        win_rate = 0.0
        total_pnl = 0.0
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Total Trades: {len(trades_executed)}")
    print(f"Win Rate: {win_rate:.2%}")
    print(f"Total P&L: ${total_pnl:.2f}")
    print(f"Unique States: {len(brain.table)}")
    print(f"Elite States (80%+): {len(brain.get_all_states_above_threshold(0.80))}")
    
    return brain, confidence_mgr


def load_data(path: str) -> pd.DataFrame:
    """Load data from .dbn.zst or .parquet files"""
    if path.endswith('.dbn.zst') or path.endswith('.dbn'):
        return load_databento_data(path)
    elif path.endswith('.parquet'):
        return pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")

def load_databento_data(path: str) -> pd.DataFrame:
    """Load .dbn.zst files from Databento"""
    import databento as db
    
    # Read DBN file
    data = db.DBNStore.from_file(path)
    df = data.to_df()
    
    # Convert to OHLCV format
    # Check if 'ts_event' exists, otherwise use index if it's already datetime
    if 'ts_event' in df.columns:
        df = df.rename(columns={'ts_event': 'timestamp'})
        df = df.set_index('timestamp')
    
    # Ensure 'close' exists, if not maybe it is 'price'
    if 'close' not in df.columns and 'price' in df.columns:
        df = df.rename(columns={'price': 'close'})
        
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian-AI Training Orchestrator")
    parser.add_argument("--data-file", type=str, required=True, help="Path to data file (.dbn.zst or .parquet)")
    parser.add_argument("--output", type=str, default="models/", help="Output directory for the model")
    parser.add_argument("--iterations", type=int, default=1000, help="Max iterations (not fully utilized in this version)")
    parser.add_argument("--skip-phase0", action="store_true", help="Skip Phase 0 exploration")
    
    args = parser.parse_args()
    
    try:
        train_complete_system(args.data_file, args.iterations, args.output, not args.skip_phase0)
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
