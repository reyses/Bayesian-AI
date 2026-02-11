"""
Bayesian-AI - Layer Computation Engine (CUDA-Enhanced)
Extracts 9-layer state from raw tick/OHLC data with GPU acceleration
"""
import pandas as pd
import numpy as np
from core.state_vector import StateVector
from typing import Dict, List, Optional
import logging

from cuda_modules.pattern_detector import get_pattern_detector
from cuda_modules.confirmation import get_confirmation_engine
from cuda_modules.velocity_gate import get_velocity_gate

class LayerEngine:
    """
    Computes all 9 layers from market data
    Separates STATIC (L1-L4) from FLUID (L5-L9) computation
    Uses CUDA acceleration for L7-L9 when available
    """
    # Threshold for L2 regime detection (random walk expansion ~2.2x)
    L2_TRENDING_THRESHOLD = 3.0

    def __init__(self, use_gpu=True, logger=None):
        # Buffers for different timeframes
        self.daily_data = None
        self.weekly_data = None
        self.monthly_data = None
        
        # Static context (computed once per session)
        self.static_context = None
        self.daily_low_5d = None
        self.daily_high_5d = None
        
        # User-provided kill zones
        self.kill_zones = []
        
        # Logging
        self.logger = logger

        # Initialize helpers
        self.pattern_detector = None
        self.confirmation_engine = None
        self.velocity_gate = None

        # Attempt to initialize CUDA modules. If they fail (e.g. no GPU), catch RuntimeError.
        if use_gpu:
            try:
                self.pattern_detector = get_pattern_detector(use_gpu=True)
            except RuntimeError:
                print("[LAYER ENGINE] PatternDetector not available (No GPU). CPU fallback disabled.")
                self.pattern_detector = None

            try:
                self.confirmation_engine = get_confirmation_engine(use_gpu=True)
            except RuntimeError:
                 self.confirmation_engine = None

            try:
                self.velocity_gate = get_velocity_gate(use_gpu=True)
            except RuntimeError:
                 self.velocity_gate = None

        # Check effective GPU usage (if any component uses GPU)
        self.use_gpu = (
            (self.pattern_detector and self.pattern_detector.use_gpu) or
            (self.confirmation_engine and self.confirmation_engine.use_gpu) or
            (self.velocity_gate and self.velocity_gate.use_gpu)
        )

        if self.use_gpu:
            print("[LAYER ENGINE] CUDA acceleration ENABLED")
        else:
            print("[LAYER ENGINE] CUDA acceleration DISABLED (GPU modules unavailable)")

    def _log(self, msg):
        if self.logger:
            self.logger.debug(msg)
    
    def initialize_static_context(self, historical_data: pd.DataFrame, kill_zones: List[float]):
        """
        Compute Layers 1-4 once at session start
        Args:
            historical_data: DataFrame with OHLC data (90+ days)
            kill_zones: User's pre-marked support/resistance levels
        """
        self.kill_zones = kill_zones
        
        # Resample to different timeframes
        self.daily_data = historical_data.resample('1D').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        self.weekly_data = historical_data.resample('1W').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        self.monthly_data = historical_data.resample('1ME').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        
        # Pre-compute static stats
        self.daily_low_5d = self.daily_data.tail(5)['low'].min()
        self.daily_high_5d = self.daily_data.tail(5)['high'].max()

        # Compute static layers
        self.static_context = {
            'L1': self._compute_L1_90d(),
            'L2': self._compute_L2_30d(),
            'L3': self._compute_L3_1wk(),
            'L4': self._compute_L4_daily()
        }
        
        self._log(f"[LAYER ENGINE] Static context initialized: {self.static_context}")
        print(f"[LAYER ENGINE] Static context initialized:")
        print(f"  L1 (90d): {self.static_context['L1']}")
        print(f"  L2 (30d): {self.static_context['L2']}")
        print(f"  L3 (1wk): {self.static_context['L3']}")
        print(f"  L4 (daily): {self.static_context['L4']}")
    
    def _compute_L1_90d(self) -> str:
        """Layer 1: 90-day bias (bull/bear/range)"""
        if len(self.daily_data) < 90:
            return 'range'
        
        last_90 = self.daily_data.tail(90)
        first_close = last_90.iloc[0]['close']
        last_close = last_90.iloc[-1]['close']
        pct_change = (last_close - first_close) / first_close
        
        if pct_change > 0.05:
            return 'bull'
        elif pct_change < -0.05:
            return 'bear'
        else:
            return 'range'
    
    def _compute_L2_30d(self) -> str:
        """Layer 2: 30-day regime (trending/chopping)"""
        if len(self.daily_data) < 30:
            return 'chopping'
        
        last_30 = self.daily_data.tail(30)
        avg_range = (last_30['high'] - last_30['low']).mean()
        recent_range = (last_30.tail(5)['high'].max() - last_30.tail(5)['low'].min())
        
        # AUDIT FIX: Increased threshold to prevent false positives in random walk.
        # Original 1.5 was too low (random walk ~2.2x).
        if recent_range > avg_range * self.L2_TRENDING_THRESHOLD:
            return 'trending'
        else:
            return 'chopping'
    
    def _compute_L3_1wk(self) -> str:
        """Layer 3: 1-week swing structure"""
        if len(self.weekly_data) < 4:
            return 'sideways'
        
        last_4_weeks = self.weekly_data.tail(4)
        highs = last_4_weeks['high'].values
        lows = last_4_weeks['low'].values
        
        if highs[-1] > highs[-2] and highs[-2] > highs[-3]:
            return 'higher_highs'
        if lows[-1] < lows[-2] and lows[-2] < lows[-3]:
            return 'lower_lows'
        
        return 'sideways'
    
    def _compute_L4_daily(self) -> str:
        """
        Layer 4: Daily zone (at support/resistance/killzone/mid)
        NOTE: This static method returns a placeholder.
        The actual L4 state depends on current price and is computed dynamically
        in compute_current_state().
        """
        return 'mid_range'
    
    def _check_kill_zone(self, price: float, tolerance: float = 5.0) -> bool:
        """Check if price is within tolerance of any kill zone"""
        for zone in self.kill_zones:
            if abs(price - zone) <= tolerance:
                self._log(f"Price {price} inside killzone {zone} (tol={tolerance})")
                return True
        return False
    
    def compute_current_state(self, current_data: Dict) -> StateVector:
        """
        Compute full 9-layer state from current market snapshot
        """
        if self.static_context is None:
            raise ValueError("Must call initialize_static_context() first")
        
        # Extract static layers
        L1_bias = self.static_context['L1']
        L2_regime = self.static_context['L2']
        L3_swing = self.static_context['L3']
        
        # L4: Check current price vs kill zones
        price = current_data['price']
        if self._check_kill_zone(price):
            L4_zone = 'at_killzone'
        else:
            # Use pre-computed static stats
            if abs(price - self.daily_low_5d) < abs(price - self.daily_high_5d):
                L4_zone = 'at_support'
            elif abs(price - self.daily_high_5d) < abs(price - self.daily_low_5d):
                L4_zone = 'at_resistance'
            else:
                L4_zone = 'mid_range'
        
        # L5: 4-hour trend
        L5_trend = self._compute_L5_4hr(current_data.get('bars_4hr'))
        
        # L6: 1-hour structure
        L6_structure = self._compute_L6_1hr(current_data.get('bars_1hr'))
        
        # L7: 15-min pattern
        L7_pattern, L7_maturity = self._compute_L7_15m(current_data.get('bars_15m'))
        
        # L8: 5-min confirmation
        L8_confirm = self._compute_L8_5m(current_data.get('bars_5m'), L7_pattern)
        
        # L9: 1-sec velocity cascade
        L9_cascade = self._compute_L9_1s(current_data.get('ticks'))
        
        state = StateVector(
            L1_bias=L1_bias,
            L2_regime=L2_regime,
            L3_swing=L3_swing,
            L4_zone=L4_zone,
            L5_trend=L5_trend,
            L6_structure=L6_structure,
            L7_pattern=L7_pattern,
            L8_confirm=L8_confirm,
            L9_cascade=L9_cascade,
            timestamp=current_data['timestamp'],
            price=price
        )

        # Detailed logging if enabled
        if self.logger and self.logger.isEnabledFor(logging.DEBUG):
             self._log(f"Computed State: {state}")

        return state
    
    def _compute_L5_4hr(self, bars: Optional[pd.DataFrame]) -> str:
        """Layer 5: 4-hour trend"""
        if bars is None or len(bars) < 3:
            return 'flat'
        
        closes = bars['close'].values
        res = 'flat'
        if closes[-1] > closes[-2] > closes[-3]:
            res = 'up'
        elif closes[-1] < closes[-2] < closes[-3]:
            res = 'down'

        if self.logger and res != 'flat':
             self._log(f"L5 Trend: {res} (Closes: {closes[-3:]})")
        return res
    
    def _compute_L6_1hr(self, bars: Optional[pd.DataFrame]) -> str:
        """Layer 6: 1-hour structure"""
        if bars is None or len(bars) < 5:
            return 'neutral'
        
        closes = bars['close'].values
        bullish = sum(1 for i in range(len(closes)-1) if closes[i+1] > closes[i])
        bearish = sum(1 for i in range(len(closes)-1) if closes[i+1] < closes[i])
        
        res = 'neutral'
        if bullish > bearish * 1.5:
            res = 'bullish'
        elif bearish > bullish * 1.5:
            res = 'bearish'

        if self.logger and res != 'neutral':
             self._log(f"L6 Structure: {res} (Bull: {bullish}, Bear: {bearish})")
        return res
    
    def _compute_L7_15m(self, bars: Optional[pd.DataFrame]) -> tuple:
        """Layer 7: 15-min pattern detection"""
        if bars is None or len(bars) < 10:
            return ('none', 0.0)
        
        if self.pattern_detector is None:
             return ('none', 0.0)

        res, maturity = self.pattern_detector.detect(bars, window_size=20)
        if self.logger and res != 'none':
            self._log(f"L7 Pattern: {res} (Maturity: {maturity})")
        return res, maturity
    
    def _compute_L8_5m(self, bars: Optional[pd.DataFrame], L7_pattern: str) -> bool:
        """Layer 8: 5-min confirmation"""
        if bars is None or len(bars) < 3:
            return False
        
        if L7_pattern == 'none':
            return False
        
        if self.confirmation_engine is None:
             return False

        confirmed = self.confirmation_engine.confirm(bars, L7_pattern != 'none')
        if self.logger and confirmed:
            self._log(f"L8 Confirmed for pattern {L7_pattern}")
        return confirmed
    
    def _compute_L9_1s(self, ticks) -> bool:
        """Layer 9: 1-sec velocity cascade detector"""
        if ticks is None or len(ticks) < 50:
            return False
        
        if self.velocity_gate is None:
             return False

        # Pass directly to VelocityGate which handles DataFrame/Array/List
        cascade = self.velocity_gate.detect_cascade(ticks)
        if self.logger and cascade:
            self._log("L9 CASCADE DETECTED")
        return cascade
