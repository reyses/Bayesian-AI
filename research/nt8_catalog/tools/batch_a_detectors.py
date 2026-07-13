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
            self.or_high = max(self.or_high, state.ohlcv_5s['close'])
            self.or_low = min(self.or_low, state.ohlcv_5s['close'])
            
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
            
        if t >= pd.Timestamp('08:30').time():
            self.triggered = True
            gap = state.price - self.pdc
            if abs(gap) >= 5.0:
                setup_id = self.day_of_week
                mode = 'gap_down' if gap < 0 else 'gap_up'
                return setup_id, mode
            else:
                return 0, ''
                
        return 0, ''

class RENKO24Detector(Detector):
    def __init__(self):
        self.brick_size = 2.0
        self.prev_brick_close = None
        self.curr_dir = 0
        self.prev_dir = 0
        self.brick_chain = 0
        
    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        close = state.ohlcv_5s['close']
        
        if self.prev_brick_close is None:
            self.prev_brick_close = np.floor(close / self.brick_size) * self.brick_size
            return 0, ''
            
        triggered_setup = 0
        triggered_mode = ''
        
        while True:
            if self.curr_dir == 0:
                if close >= self.prev_brick_close + self.brick_size:
                    self.curr_dir = 1
                    self.prev_dir = 0
                    self.prev_brick_close += self.brick_size
                    self.brick_chain = 1
                elif close <= self.prev_brick_close - self.brick_size:
                    self.curr_dir = -1
                    self.prev_dir = 0
                    self.prev_brick_close -= self.brick_size
                    self.brick_chain = 1
                else:
                    break
            elif self.curr_dir == 1:
                if close >= self.prev_brick_close + self.brick_size:
                    self.curr_dir = 1
                    self.prev_brick_close += self.brick_size
                    self.brick_chain += 1
                    if self.brick_chain == 2 and self.prev_dir == -1:
                        triggered_setup, triggered_mode = 1, 'bullish_renko'
                elif close <= self.prev_brick_close - 2 * self.brick_size:
                    self.prev_dir = self.curr_dir
                    self.curr_dir = -1
                    self.prev_brick_close -= 2 * self.brick_size
                    self.brick_chain = 1
                else:
                    break
            elif self.curr_dir == -1:
                if close <= self.prev_brick_close - self.brick_size:
                    self.curr_dir = -1
                    self.prev_brick_close -= self.brick_size
                    self.brick_chain += 1
                    if self.brick_chain == 2 and self.prev_dir == 1:
                        triggered_setup, triggered_mode = 2, 'bearish_renko'
                elif close >= self.prev_brick_close + 2 * self.brick_size:
                    self.prev_dir = self.curr_dir
                    self.curr_dir = 1
                    self.prev_brick_close += 2 * self.brick_size
                    self.brick_chain = 1
                else:
                    break
                    
        if triggered_setup != 0:
            return triggered_setup, triggered_mode
            
        return 0, ''

class VWAP03Detector(Detector):
    def __init__(self):
        self.cum_pv = 0.0
        self.cum_vol = 0.0
        self.prices = []
        self.primed_bull = False
        self.primed_bear = False
        self.z_prev = 0.0
        self.triggered = False
        
    def on_bar(self, state) -> tuple[int, str]:
        if self.triggered:
            return 0, ''
            
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
        if res[0] != 0:
            self.triggered = True
        return res

class OHLC01Detector(Detector):
    def __init__(self, pdh: float, pdl: float, pdc: float):
        self.pdh = pdh
        self.pdl = pdl
        self.pdc = pdc
        self.triggered_setups = set()
        self.open_price = None

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if self.open_price is None:
            self.open_price = state.price
            self.primed_1 = self.open_price < self.pdh
            self.primed_2 = self.open_price > self.pdl
            self.primed_3 = abs(self.open_price - self.pdc) > 2.5
                
        if self.primed_1 and 1 not in self.triggered_setups and state.price >= self.pdh:
            self.triggered_setups.add(1)
            return 1, 'bearish_bounce'
            
        if self.primed_2 and 2 not in self.triggered_setups and state.price <= self.pdl:
            self.triggered_setups.add(2)
            return 2, 'bullish_bounce'
            
        if self.primed_3 and 3 not in self.triggered_setups:
            if self.open_price < self.pdc and state.price >= self.pdc:
                self.triggered_setups.add(3)
                return 3, 'bullish_bounce'
            elif self.open_price > self.pdc and state.price <= self.pdc:
                self.triggered_setups.add(3)
                return 3, 'bearish_bounce'
                
        return 0, ''

class PIVOT16Detector(Detector):
    def __init__(self, pdh: float, pdl: float, pdc: float):
        self.pp = (pdh + pdl + pdc) / 3.0
        self.s1 = (2 * self.pp) - pdh
        self.r1 = (2 * self.pp) - pdl
        self.triggered_setups = set()
        self.open_price = None

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if self.open_price is None:
            self.open_price = state.price
            self.primed_1 = self.open_price > self.s1
            self.primed_2 = self.open_price < self.r1
                
        if self.primed_1 and 1 not in self.triggered_setups and state.price <= self.s1:
            self.triggered_setups.add(1)
            return 1, 'bullish_bounce'
            
        if self.primed_2 and 2 not in self.triggered_setups and state.price >= self.r1:
            self.triggered_setups.add(2)
            return 2, 'bearish_bounce'
            
        return 0, ''

class ROUND05Detector(Detector):
    def __init__(self):
        self.primed_bullish = {}
        self.primed_bearish = {}
        self.triggered = False

    def on_bar(self, state) -> tuple[int, str]:
        if self.triggered:
            return 0, ''
            
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
            self.triggered = True
            return triggered_setup, triggered_mode
            
        return 0, ''
