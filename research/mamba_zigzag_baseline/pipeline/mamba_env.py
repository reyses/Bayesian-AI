import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import numpy as np
import torch
from collections import deque
import logging
import datetime
import pytz

from core_v2.FPS.forward_pass_system import MultiDayForwardPassSystem
from core_v2.ledger import Ledger
from core_v2.features import assemble_v2_grid
from core_v2.exits import default_exit_suite

logger = logging.getLogger(__name__)

class MambaRLTradingEnv:
    """
    OpenAI Gym-style wrapper for the V2 Forward Pass System.
    Yields 2D V2 Grids + Ledger State for the MambaRLTradingNetwork.
    """
    def __init__(self, atlas_root, features_root, labels_csv, days, target_pnl_per_trade=100.0, seq_len=100):
        self.fps = MultiDayForwardPassSystem(
            atlas_root=atlas_root,
            features_root=features_root,
            labels_csv=labels_csv,
            days=days
        )
        self.target_pnl_per_trade = target_pnl_per_trade
        self.ledger = Ledger()
        self.seq_len = seq_len
        self.central_tz = pytz.timezone('US/Central')
        
        # We need a small queue to build the sequence window
        self.state_queue = deque(maxlen=self.seq_len)
        self.l0_queue = deque(maxlen=self.seq_len)
        self.iterator = None
        self.current_bar = None
        
        self.realized_pnl = 0.0
        self.last_hour_ts = None
        self.last_hour_equity = 0.0
        
        self.exit_suite = default_exit_suite()

    def update_curriculum_state(self, epoch, total_epochs):
        """
        Triggered by Curriculum Orchestration.
        Applies Phased Friction by tracking synthetic limits dynamically based on epoch.
        """
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.phased_friction = (epoch / max(1, total_epochs - 1))
        # Logic to scale synthetic exchange limits can be wired into self.ledger here.

    def reset(self):
        self.iterator = iter(self.fps)
        self.ledger.clear()
        self.state_queue.clear()
        self.l0_queue.clear()
        
        self.realized_pnl = 0.0
        self.last_hour_ts = None
        self.last_hour_equity = 0.0
        
        # Warmup: we need `seq_len` bars of valid features
        try:
            while len(self.state_queue) < self.seq_len:
                bar_state = next(self.iterator)
                if bar_state.v2_vector is not None and len(bar_state.v2_vector) >= 185:
                    self.state_queue.append(bar_state.v2_vector)
                    self.l0_queue.append(bar_state.v2_vector[0:1])
                self.current_bar = bar_state
        except StopIteration:
            raise ValueError(f"Dataset too small to warmup {self.seq_len} bars.")
            
        return self._get_observation()

    def _get_observation(self):
        """
        Returns:
            v2_grid: [8, 100, 23]
            l0_feature: [100, 1]
            ledger_state: [100, 4]
        """
        raw_matrix = np.array(self.state_queue, dtype=np.float32) # [100, 393]
        grid = assemble_v2_grid(raw_matrix) # [100, 8, 23]
        grid = grid.transpose((1, 0, 2)) # [8, 100, 23]
        
        l0_feature = np.array(self.l0_queue, dtype=np.float32) # [100, 1]
        
        # Build Ledger State history
        # We need a [seq_len, 4] vector representing the state over the sequence
        # For simplicity, we can just broadcast the *current* ledger state across the seq_len steps
        # or build a strict historical tracker. Given Mamba processes causality, broadcasting the 
        # current state is acceptable for 'goal conditioning' at timestep t.
        pos_code = 0.0
        if not self.ledger.is_flat:
            pos_code = 1.0 if self.ledger.primary.direction == 'long' else -1.0
            
        # The rolling target is per-trade as specified by the user
        current_pnl = self.ledger.primary.peak_pnl if not self.ledger.is_flat else 0.0
        distance_to_target = self.target_pnl_per_trade - current_pnl
        
        state_vec = np.array([pos_code, current_pnl, self.target_pnl_per_trade, distance_to_target], dtype=np.float32)
        ledger_state = np.tile(state_vec, (self.seq_len, 1)) # [seq_len, 4]
        
        return grid, l0_feature, ledger_state

    def step(self, action: int, expected_outcome: float):
        """
        action: 0=HOLD, 1=LONG, 2=SHORT, 3=SCRATCH
        expected_outcome: Predictor scalar for PnL
        """
        reward = 0.0
        done = False
        info = {}
        
        # 1. Update ledger state for the current bar (monitors peak PnL, draws, etc)
        if not self.ledger.is_flat:
            self.ledger.update_bar(self.current_bar.v2_vector, self.current_bar.price, self.current_bar.timestamp)
            
            exit_reason = None
            
            # Check Action-Based Exits
            if exit_reason is None:
                # Add time-based guard rail (5 minutes before maintenance and weekends)
                dt = datetime.datetime.fromtimestamp(self.current_bar.timestamp, tz=datetime.timezone.utc)
                ct = dt.astimezone(self.central_tz)
                if ct.hour == 15 and ct.minute >= 55:
                    exit_reason = "TIME_GUARD_RAIL"
                elif action == 3: # SCRATCH
                    exit_reason = "RL_SCRATCH"
                elif action == 1 and self.ledger.primary.direction == 'short':
                    exit_reason = "RL_REVERSAL_LONG"
                elif action == 2 and self.ledger.primary.direction == 'long':
                    exit_reason = "RL_REVERSAL_SHORT"

            # Execute Exit
            if exit_reason:
                pos = self.ledger.primary
                record = self.ledger.remove_position(pos.contract_id, self.current_bar.price, self.current_bar.timestamp, exit_reason)
                trade_pnl = record['pnl']
                
                # Reward for closing the trade
                reward += trade_pnl
                self.realized_pnl += trade_pnl
                
                # Store the expected vs actual diff for auxiliary tracking
                exp_pnl = record['extras'].get('expected_outcome', 0.0)
                info['expected_pnl'] = exp_pnl
                info['actual_pnl'] = trade_pnl
                info['duration'] = (self.current_bar.timestamp - pos.entry_ts) / 5.0
                info['prediction_error'] = abs(trade_pnl - exp_pnl)
                info['trade_closed'] = True
                info['entry_ts'] = pos.entry_ts
                info['exit_ts'] = self.current_bar.timestamp
                info['direction'] = pos.direction
        
        # 3. Process Entries
        if self.ledger.is_flat and action in [1, 2]:
            # Block new entries if we are within the maintenance guard rail
            dt = datetime.datetime.fromtimestamp(self.current_bar.timestamp, tz=datetime.timezone.utc)
            ct = dt.astimezone(self.central_tz)
            if not (ct.hour == 15 and ct.minute >= 55):
                direction = 'long' if action == 1 else 'short'
                self.ledger.add_position(
                    direction=direction,
                    entry_price=self.current_bar.price,
                    entry_ts=self.current_bar.timestamp,
                    entry_tier="RL_MAMBA",
                    entry_features=self.current_bar.v2_vector,
                    restore_extras={'expected_outcome': expected_outcome}
                )
                # No $5 penalty anymore! We want the agent to trade freely.
        
        # 4. Compute Equity and check for hourly stagnation
        unrealized_pnl = 0.0
        if not self.ledger.is_flat:
            pos = self.ledger.primary
            if pos.direction == 'long':
                unrealized_pnl = ((self.current_bar.price - pos.entry_price) / self.ledger.tick_size * self.ledger.tick_value) - self.ledger.round_trip_fee
            else:
                unrealized_pnl = ((pos.entry_price - self.current_bar.price) / self.ledger.tick_size * self.ledger.tick_value) - self.ledger.round_trip_fee
                
        total_equity = self.realized_pnl + unrealized_pnl
        
        if self.last_hour_ts is None:
            self.last_hour_ts = self.current_bar.timestamp
            self.last_hour_equity = total_equity
        elif self.current_bar.timestamp - self.last_hour_ts >= 3600: # 1 hour passed
            if total_equity < self.last_hour_equity + 10.0:
                reward -= 50.0 # Massive penalty for failing to grow equity by $10 over the hour
                info['hourly_penalty'] = True
                
            self.last_hour_ts = self.current_bar.timestamp
            self.last_hour_equity = total_equity

        # 5. Advance Time
        try:
            bar_state = next(self.iterator)
            while bar_state.v2_vector is None or len(bar_state.v2_vector) < 185:
                bar_state = next(self.iterator)
                
            self.state_queue.append(bar_state.v2_vector)
            self.l0_queue.append(bar_state.v2_vector[0:1])
            self.current_bar = bar_state
        except StopIteration:
            done = True
            
        next_state = self._get_observation() if not done else None
        
        return next_state, reward, done, info
