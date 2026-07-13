import numpy as np
import pandas as pd

class Detector:
    def on_bar(self, state) -> tuple[int, str]:
        """Process one 5s BarState and return (setup_id, mode) if triggered, else (0, '')."""
        return 0, ''

class ADX08Detector(Detector):
    def __init__(self):
        self.prices = []
        self.highs = []
        self.lows = []
        self.dms_plus = []
        self.dms_minus = []
        self.trs = []
        self.dxs = []
        
        self.triggered_bull = False
        self.triggered_bear = False

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        high = state.ohlcv_5s['high']
        low = state.ohlcv_5s['low']
        close = state.ohlcv_5s['close']
        
        self.prices.append(close)
        self.highs.append(high)
        self.lows.append(low)
        
        if len(self.prices) > 240:
            self.prices.pop(0)
            self.highs.pop(0)
            self.lows.pop(0)
            
        if len(self.prices) < 2:
            return 0, ''
            
        upMove = high - self.highs[-2]
        downMove = self.lows[-2] - low
        
        dm_plus = upMove if (upMove > downMove and upMove > 0) else 0.0
        dm_minus = downMove if (downMove > upMove and downMove > 0) else 0.0
        
        tr1 = high - low
        tr2 = abs(high - self.prices[-2])
        tr3 = abs(low - self.prices[-2])
        tr = max(tr1, tr2, tr3)
        
        self.dms_plus.append(dm_plus)
        self.dms_minus.append(dm_minus)
        self.trs.append(tr)
        
        if len(self.dms_plus) > 168:
            self.dms_plus.pop(0)
            self.dms_minus.pop(0)
            self.trs.pop(0)
            
        # compute DX
        if len(self.dms_plus) == 168:
            tr_sum = np.mean(self.trs)
            if tr_sum == 0: tr_sum = 1e-10
            di_plus = 100 * (np.mean(self.dms_plus) / tr_sum)
            di_minus = 100 * (np.mean(self.dms_minus) / tr_sum)
            dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10))
        else:
            dx = 0.0
            
        self.dxs.append(dx)
        if len(self.dxs) > 168:
            self.dxs.pop(0)
            
        if len(self.prices) < 240 or len(self.dxs) < 168:
            return 0, ''
            
        adx_proxy = np.mean(self.dxs)
        sma20 = np.mean(self.prices)
        prev_sma20 = np.mean(self.prices[:-1])
        prev_close = self.prices[-2]
        
        cross_above = (prev_close <= prev_sma20) and (close > sma20)
        cross_below = (prev_close >= prev_sma20) and (close < sma20)
        
        if not self.triggered_bull and adx_proxy > 25.0 and cross_above:
            self.triggered_bull = True
            return 1, 'bullish_runner'
            
        if not self.triggered_bear and adx_proxy > 25.0 and cross_below:
            self.triggered_bear = True
            return 2, 'bearish_runner'
            
        return 0, ''

class ATR09Detector(Detector):
    def __init__(self, daily_atr: float):
        self.daily_atr = daily_atr
        self.running_high = -np.inf
        self.running_low = np.inf
        self.thresholds = [0.5, 0.75, 1.0]
        self.triggered = {x: False for x in self.thresholds}

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        price = state.price
        self.running_high = max(self.running_high, price)
        self.running_low = min(self.running_low, price)
        
        current_range = self.running_high - self.running_low
        
        for x in self.thresholds:
            if not self.triggered[x] and current_range >= x * self.daily_atr:
                self.triggered[x] = True
                
                if price >= self.running_high - 0.25:
                    return int(x * 100), 'bearish_fade'
                elif price <= self.running_low + 0.25:
                    return int(x * 100) + 1, 'bullish_fade'
                    
        return 0, ''

class CROSS11Detector(Detector):
    def __init__(self, prefill_closes=None):
        self.prices = []
        if prefill_closes is not None:
            self.prices.extend(prefill_closes[-2400:])
        self.cross_state = 0 # 1 if > , -1 if <
        self.triggered_bull = False
        self.triggered_bear = False

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        close = state.price
        self.prices.append(close)
        if len(self.prices) > 2400:
            self.prices.pop(0)
            
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if len(self.prices) < 2400:
            return 0, ''
            
        sma50 = np.mean(self.prices[-600:])
        sma200 = np.mean(self.prices)
        
        prev_sma50 = np.mean(self.prices[-601:-1])
        prev_sma200 = np.mean(self.prices[:-1])
        
        cross_up = (prev_sma50 <= prev_sma200) and (sma50 > sma200)
        cross_down = (prev_sma50 >= prev_sma200) and (sma50 < sma200)
        
        if cross_up and not self.triggered_bull:
            self.triggered_bull = True
            return 1, 'bullish_runner'
            
        if cross_down and not self.triggered_bear:
            self.triggered_bear = True
            return 2, 'bearish_runner'
            
        return 0, ''

class DOW19Detector(Detector):
    def __init__(self):
        self.prices = []
        self.volumes = []
        self.cooldown = 0
        self.in_trade = False

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        close = state.price
        vol = state.ohlcv_5s['volume']
        
        self.prices.append(close)
        self.volumes.append(vol)
        
        if len(self.prices) > 20:
            self.prices.pop(0)
            self.volumes.pop(0)
            
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if len(self.prices) < 20:
            return 0, ''
            
        if self.cooldown > 0:
            self.cooldown -= 1
            return 0, ''
            
        # 10-bar rolling max/min of previous close (close.shift(1))
        # self.prices[-1] is the current close, so self.prices[-11:-1] are the previous 10 closes
        high_10 = np.max(self.prices[-11:-1])
        low_10 = np.min(self.prices[-11:-1])
        
        vol_sma20 = np.mean(self.volumes)
        
        if close > high_10 and vol < vol_sma20:
            self.cooldown = 60
            return 1, 'bearish_trap'
            
        if close < low_10 and vol < vol_sma20:
            self.cooldown = 60
            return 2, 'bullish_trap'
            
        return 0, ''

class FIB17Detector(Detector):
    def __init__(self, fib_50: float, fib_618: float, adx_val: float, trend: str):
        self.fib_50 = fib_50
        self.fib_618 = fib_618
        self.adx_val = adx_val
        self.trend = trend
        self.in_trade = False
        self.triggered_setups = set()

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()
        
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''
            
        if self.in_trade:
            return 0, ''
            
        if self.adx_val <= 25.0:
            return 0, ''
            
        price = state.price
        
        if self.trend == 'UP':
            # Price drops into fib zone (bounce up)
            if price <= self.fib_50 and price >= self.fib_618:
                if 1 not in self.triggered_setups:
                    self.triggered_setups.add(1)
                    self.in_trade = True
                    return 1, 'bullish_bounce'
        else:
            # Price rallies into fib zone (bounce down)
            if price >= self.fib_50 and price <= self.fib_618:
                if 2 not in self.triggered_setups:
                    self.triggered_setups.add(2)
                    self.in_trade = True
                    return 2, 'bearish_bounce'
                    
        return 0, ''
