"""
Bayesian-AI - Main Execution Engine
File: bayesian_ai/engine_core.py
"""
from core.bayesian_brain import BayesianBrain, TradeOutcome, QuantumBayesianBrain
from execution.wave_rider import WaveRider
from core.layer_engine import LayerEngine
from core.data_aggregator import DataAggregator
from core.logger import setup_logger
from core.three_body_state import ThreeBodyQuantumState
from core.quantum_field_engine import QuantumFieldEngine
from core.unconstrained_explorer import UnconstrainedExplorer
import time
import os

class BayesianEngine:
    def __init__(self, asset, use_gpu=True, verbose=False, log_path=None, mode="LEGACY"):
        self.asset = asset
        self.verbose = verbose
        self.logger = None
        self.mode = mode
        self.last_price = None

        if self.verbose:
            self.logger = setup_logger("BayesianEngine", log_path, console=False)
            if self.logger:
                self.logger.info(f"BayesianEngine Initialized. GPU={use_gpu}, Asset={asset}, Mode={mode}")

        self.wave_rider = WaveRider(asset)
        # Increase buffer for Phase 0 to ensure enough history for Quantum Field (needs ~315m history)
        # 50000 ticks @ 1s = ~13 hours. Should be plenty.
        self.aggregator = DataAggregator(max_ticks=50000)
        self.daily_pnl = 0.0
        self.MAX_DAILY_LOSS = -200.0
        self.MIN_PROB = 0.80
        self.MIN_CONF = 0.30

        # Mode-specific initialization
        try:
            if self.mode == "PHASE0":
                self.prob_table = QuantumBayesianBrain()
                self.quantum_engine = QuantumFieldEngine()
                self.explorer = UnconstrainedExplorer()
                # Note: fluid_engine might not be needed for Phase 0, but keeping it initialized doesn't hurt
                # providing it doesn't consume resources if not used.
                # However, LayerEngine does static context init which might be useful or redundant.
                # Let's keep it initialized for now as it doesn't conflict.
                self.fluid_engine = LayerEngine(use_gpu=use_gpu, logger=self.logger)
            else:
                self.prob_table = BayesianBrain()
                # Pass logger to LayerEngine for deep inspection
                self.fluid_engine = LayerEngine(use_gpu=use_gpu, logger=self.logger)
        except Exception as e:
            if self.logger:
                self.logger.critical(f"Failed to initialize engine components: {e}")
            raise type(e)(str(e)) from e

    def initialize_session(self, historical_data, user_kill_zones):
        """Initialize static context (L1-L4)"""
        if self.logger:
            self.logger.info("Initializing Static Context...")
        self.fluid_engine.initialize_static_context(historical_data, user_kill_zones)

    def on_tick(self, tick_data: dict):
        if self.mode != "PHASE0" and self.daily_pnl <= self.MAX_DAILY_LOSS:
            if self.logger:
                self.logger.warning("Max Daily Loss Reached. Skipping tick.")
            return

        # Calculate velocity
        current_price = tick_data['price']
        tick_velocity = 0.0
        if self.last_price is not None:
            tick_velocity = current_price - self.last_price
        self.last_price = current_price

        # Add tick to aggregator
        self.aggregator.add_tick(tick_data)

        # Get full data context
        current_data = self.aggregator.get_current_data()

        if self.mode == "PHASE0":
            # PHASE 0: Unconstrained Exploration (Quantum)
            if current_data['bars_15m'] is None or current_data['bars_15s'] is None:
                return # Not enough data yet

            # Compute Quantum State
            current_state = self.quantum_engine.calculate_three_body_state(
                current_data['bars_15m'],
                current_data['bars_15s'],
                current_price,
                tick_data.get('volume', 0),
                tick_velocity
            )

            # Skip invalid states
            if current_state.center_position == 0.0 and current_state.z_score == 0.0:
                return

            if self.logger:
                self.logger.debug(f"Tick: {current_price} | Quantum State Z={current_state.z_score:.2f}")

            # Exit Management
            if self.wave_rider.position:
                # Capture position snapshot before update_trail might clear it
                position_snapshot = self.wave_rider.position
                decision = self.wave_rider.update_trail(current_price, current_state)

                if decision['should_exit']:
                    if self.logger:
                        self.logger.info(f"Exit Triggered: {decision}")
                    self._close_position(current_price, decision, position_snapshot)
                return

            # Entry Logic (Unconstrained)
            decision = self.explorer.should_fire(current_state)
            if decision['should_fire']:
                # Randomize direction for true exploration (Momentum vs Reversion)
                import random
                side = 'short' if random.random() > 0.5 else 'long'

                if self.logger:
                    self.logger.info(f"PHASE0 FIRE: {side.upper()} at {current_price}. Reason: {decision['reason']}")

                self.wave_rider.open_position(current_price, side, current_state)

        else:
            # LEGACY: 9-Layer Hierarchy
            current_state = self.fluid_engine.compute_current_state(current_data)

            # Log detailed state every tick if verbose (High Detail)
            if self.logger:
                self.logger.debug(f"Tick: {current_price} | State: {current_state}")

            # Exit Management
            if self.wave_rider.position:
                # Capture position snapshot before update_trail might clear it
                position_snapshot = self.wave_rider.position
                decision = self.wave_rider.update_trail(current_price, current_state)

                if decision['should_exit']:
                    if self.logger:
                        self.logger.info(f"Exit Triggered: {decision}")
                    self._close_position(current_price, decision, position_snapshot)
                return

            # Entry Logic
            prob = self.prob_table.get_probability(current_state)
            conf = self.prob_table.get_confidence(current_state)

            if self.logger and current_state.L9_cascade:
                 self.logger.debug(f"L9 Cascade. Prob={prob:.2f}, Conf={conf:.2f}")

            if current_state.L9_cascade and prob >= self.MIN_PROB and conf >= self.MIN_CONF:
                if self.logger:
                    self.logger.info(f"Opening SHORT at {current_price}. Prob={prob}, Conf={conf}")
                self.wave_rider.open_position(current_price, 'short', current_state)

    def _close_position(self, price, info, position_snapshot=None):
        # Use snapshot if provided (as update_trail clears position), else current
        pos = position_snapshot if position_snapshot else self.wave_rider.position

        if pos is None:
            if self.logger:
                self.logger.error("Attempted to close position but no position found.")
            return

        outcome = TradeOutcome(
            state=pos.entry_layer_state,
            entry_price=pos.entry_price,
            exit_price=price,
            pnl=info['pnl'],
            result='WIN' if info['pnl'] > 0 else 'LOSS',
            timestamp=time.time(),
            exit_reason=info['exit_reason']
        )
        self.prob_table.update(outcome)
        self.daily_pnl += info['pnl']

        if self.mode == "PHASE0":
            self.explorer.record_trade(outcome)

        if self.logger:
            self.logger.info(f"Position Closed. PnL: {info['pnl']:.2f}, Reason: {info['exit_reason']}")

        self.wave_rider.position = None

if __name__ == "__main__":
    print("Bayesian AI Engine v2.0 - Standalone Mode")
    print("Initializing...")

    # Simple check for resources
    if os.path.exists("probability_table.pkl"):
        print("[OK] Probability table found.")
    else:
        print("[WARNING] Probability table NOT found.")

    if os.path.exists("config"):
        print("[OK] Config directory found.")
    else:
        print("[WARNING] Config directory NOT found.")

    print("Engine ready. (Live feed connection pending...)")
    # Keep alive for testing
    try:
        # Just sleep for a bit then exit to confirm it runs without crashing
        print("Running diagnostics... (Press Ctrl+C to stop)")
        time.sleep(2)
        print("Diagnostics complete.")
    except KeyboardInterrupt:
        print("Exiting.")
