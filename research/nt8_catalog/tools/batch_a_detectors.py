import numpy as np
import pandas as pd

class Detector:
    def on_bar(self, state) -> tuple[int, str]:
        """Process one 5s BarState and return (setup_id, mode) if triggered, else (0, '')."""
        return 0, ''

class ORB02Detector(Detector):
    def __init__(self):
        self.or_high = -np.inf
        self.or_low = np.inf
        self.or_set = False
        self.triggered = False

    def on_bar(self, state) -> tuple[int, str]:
        if self.triggered:
            return 0, ''
            
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if pd.Timestamp('08:30').time() <= t < pd.Timestamp('09:00').time():
            self.or_high = max(self.or_high, state.ohlcv_5s['high'])
            self.or_low = min(self.or_low, state.ohlcv_5s['low'])
            
        elif t >= pd.Timestamp('09:00').time() and t <= pd.Timestamp('15:15').time():
            self.or_set = True
            
        if self.or_set and not self.triggered:
            if state.price > self.or_high:
                self.triggered = True
                return 1, 'bullish_runner'
            elif state.price < self.or_low:
                self.triggered = True
                return 2, 'bearish_runner'
                
        return 0, ''

class SEASON12Detector(Detector):
    def __init__(self, pdc: float):
        self.pdc = pdc
        self.gap_measured = False
        self.mode = ''
        self.setup_id = 0
        self.triggered = False
        self.day_of_week = None

    def on_bar(self, state) -> tuple[int, str]:
        if self.triggered:
            return 0, ''
            
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if self.day_of_week is None:
            self.day_of_week = dt.dayofweek + 1
            
        if not self.gap_measured and t >= pd.Timestamp('08:30').time():
            gap = state.price - self.pdc
            if abs(gap) >= 5.0:
                self.gap_measured = True
                self.setup_id = self.day_of_week
                self.mode = 'gap_down' if gap < 0 else 'gap_up'
            else:
                self.triggered = True
                return 0, ''
                
        if self.gap_measured and not self.triggered:
            if self.mode == 'gap_down' and state.ohlcv_5s['high'] >= self.pdc:
                self.triggered = True
                return self.setup_id, self.mode
            elif self.mode == 'gap_up' and state.ohlcv_5s['low'] <= self.pdc:
                self.triggered = True
                return self.setup_id, self.mode
                
        return 0, ''

class RENKO24Detector(Detector):
    def __init__(self):
        self.brick_size = 2.0
        self.prev_brick_close = None
        self.curr_dir = 0
        self.brick_chain = 0
        
    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        close = state.ohlcv_5s['close']
        
        if self.prev_brick_close is None:
            self.prev_brick_close = close
            return 0, ''
            
        diff = close - self.prev_brick_close
        if diff >= self.brick_size:
            bricks = int(diff // self.brick_size)
            self.prev_brick_close += bricks * self.brick_size
            if self.curr_dir == 1:
                self.brick_chain += bricks
            else:
                self.curr_dir = 1
                self.brick_chain = bricks
        elif diff <= -self.brick_size:
            bricks = int(-diff // self.brick_size)
            self.prev_brick_close -= bricks * self.brick_size
            if self.curr_dir == -1:
                self.brick_chain += bricks
            else:
                self.curr_dir = -1
                self.brick_chain = bricks
        else:
            return 0, ''
            
        if self.brick_chain == 2:
            return (1, 'bullish_renko') if self.curr_dir == 1 else (2, 'bearish_renko')
            
        return 0, ''

class VWAP03Detector(Detector):
    def __init__(self):
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.prices = []
        self.primed_bull = False
        self.primed_bear = False
        self.z_prev = 0.0
        
    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''

        close = state.ohlcv_5s['close']
        vol = state.ohlcv_5s['volume']
        
        self.cum_pv += close * vol
        self.cum_vol += vol
        if self.cum_vol == 0:
            vwap = close
        else:
            vwap = self.cum_pv / self.cum_vol
            
        self.prices.append(close)
        if len(self.prices) > 20:
            self.prices.pop(0)
            
        if len(self.prices) < 20:
            return 0, ''
            
        vwap_std = max(0.25, np.std(self.prices, ddof=1))
        z_curr = (close - vwap) / vwap_std
        
        res = (0, '')
        
        if z_curr > 2.0:
            self.primed_bear = True
        elif self.primed_bear and z_curr < self.z_prev and z_curr > 0:
            res = (1, 'bearish_bounce')
            self.primed_bear = False
        elif z_curr <= 0:
            self.primed_bear = False
            
        if z_curr < -2.0:
            self.primed_bull = True
        elif self.primed_bull and z_curr > self.z_prev and z_curr < 0:
            res = (2, 'bullish_bounce')
            self.primed_bull = False
        elif z_curr >= 0:
            self.primed_bull = False
            
        self.z_prev = z_curr
        return res

class OHLC01Detector(Detector):
    def __init__(self, pdh: float, pdl: float, pdc: float):
        self.pdh = pdh
        self.pdl = pdl
        self.pdc = pdc
        self.setup_id = 0
        self.mode = ''
        self.triggered = False
        self.open_price = None

    def on_bar(self, state) -> tuple[int, str]:
        if self.triggered:
            return 0, ''
            
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if self.open_price is None:
            self.open_price = state.price
            if self.open_price < self.pdh:
                self.setup_id = 1
                self.mode = 'bearish_bounce'
            elif self.open_price > self.pdl:
                self.setup_id = 2
                self.mode = 'bullish_bounce'
            elif abs(self.open_price - self.pdc) > 2.5:
                self.setup_id = 3
                self.mode = 'bullish_bounce' if self.open_price < self.pdc else 'bearish_bounce'
            else:
                self.triggered = True
                return 0, ''
                
        if self.setup_id == 1 and state.price >= self.pdh:
            self.triggered = True
            return 1, self.mode
        if self.setup_id == 2 and state.price <= self.pdl:
            self.triggered = True
            return 2, self.mode
        if self.setup_id == 3:
            if self.mode == 'bullish_bounce' and state.price >= self.pdc:
                self.triggered = True
                return 3, self.mode
            elif self.mode == 'bearish_bounce' and state.price <= self.pdc:
                self.triggered = True
                return 3, self.mode
                
        return 0, ''

class PIVOT16Detector(Detector):
    def __init__(self, pdh: float, pdl: float, pdc: float):
        self.pp = (pdh + pdl + pdc) / 3.0
        self.s1 = (2 * self.pp) - pdh
        self.r1 = (2 * self.pp) - pdl
        self.setup_id = 0
        self.mode = ''
        self.triggered = False
        self.open_price = None

    def on_bar(self, state) -> tuple[int, str]:
        if self.triggered:
            return 0, ''
            
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if self.open_price is None:
            self.open_price = state.price
            if self.open_price > self.s1:
                self.setup_id = 1
                self.mode = 'bullish_bounce'
            elif self.open_price < self.r1:
                self.setup_id = 2
                self.mode = 'bearish_bounce'
            else:
                self.triggered = True
                return 0, ''
                
        if self.setup_id == 1 and state.price <= self.s1:
            self.triggered = True
            return 1, self.mode
        if self.setup_id == 2 and state.price >= self.r1:
            self.triggered = True
            return 2, self.mode
            
        return 0, ''

class ROUND05Detector(Detector):
    def __init__(self):
        self.primed_bullish = {}
        self.primed_bearish = {}

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        price = state.price
        base = int(price / 50) * 50
        levels = [base - 50, base, base + 50]
        
        triggered_mode = ''
        triggered_setup = 0
        
        for L in levels:
            if L not in self.primed_bullish:
                self.primed_bullish[L] = False
                self.primed_bearish[L] = False
                
            if price >= L and self.primed_bullish[L]:
                self.primed_bullish[L] = False
                if not triggered_mode:
                    triggered_setup = 1
                    triggered_mode = 'bullish_continuation'
                    
            if price <= L and self.primed_bearish[L]:
                self.primed_bearish[L] = False
                if not triggered_mode:
                    triggered_setup = 2
                    triggered_mode = 'bearish_continuation'
                
            if price < L - 5:
                self.primed_bullish[L] = True
            elif price >= L:
                self.primed_bullish[L] = False
                
            if price > L + 5:
                self.primed_bearish[L] = True
            elif price <= L:
                self.primed_bearish[L] = False
                
        if triggered_mode:
            return triggered_setup, triggered_mode
            
        return 0, ''
