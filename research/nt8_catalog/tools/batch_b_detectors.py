import numpy as np
import pandas as pd

class Detector:
    def on_bar(self, state) -> tuple[int, str]:
        """Process one 5s BarState and return (setup_id, mode) if triggered, else (0, '')."""
        return 0, ''

class ADX08_SMA_Detector(Detector):
    """ADX-08, LEGACY smoothing (doc 071 variant A).

    Reproduces ag_deepdive_08_adx.py:56-60 exactly: +DI/-DI/DX/ADX are smoothed with a
    SIMPLE moving average (the legacy's own comment: "# Use SMA approximation for speed").
    This is the comparability baseline against the audited event population.
    168-bar DMI ADX + 240-bar SMA20; trigger = close crossing SMA20 while ADX > 25;
    one bullish + one bearish trigger per day (:104).
    """
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
        self._day = None          # session-day tracker: resets the once-per-day flags only

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()

        # CONTINUOUS ROLLING WINDOW (Moises 2026-07-14): indicators are updated on EVERY
        # bar — overnight/ETH included — and NEVER reset, so the window streams unbroken
        # through days and months. There is no cold start, ever.
        # The previous code returned BEFORE updating on non-RTH bars, which threw the
        # overnight bars away and left the detector blind for the first ~34 min of RTH
        # (240-bar SMA + 168-bar ADX = 408 bars) — the same defect we condemned in legacy
        # DOW-19 for discarding 20 bars. Only the TRIGGER is gated to RTH.
        if state.day != self._day:
            self._day = state.day
            self.triggered_bull = False
            self.triggered_bear = False

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

        # Indicators are now warm from the continuous stream; only ENTRIES are RTH-gated.
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
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

class ADX08_Wilder_Detector(Detector):
    """ADX-08, CANONICAL Wilder smoothing (doc 071 variant B).

    Identical rule to ADX08_SMA_Detector (168-bar DMI, 240-bar SMA20 cross, ADX>25,
    one bull + one bear per day) EXCEPT +DM/-DM/TR/DX are smoothed with Wilder's RMA
    (alpha = 1/N) instead of a simple mean. This is the ADX the article actually means;
    the legacy concedes its SMA is an approximation "for speed" (:56).
    Sibling of the SMA variant per Moises' ruling (doc 071) — NOT a replacement.
    """
    N_ADX = 168
    N_SMA = 240

    def __init__(self):
        self.prices = []
        self.prev_high = None
        self.prev_low = None
        self.prev_close = None
        # Wilder RMA running state
        self.rma_dm_plus = None
        self.rma_dm_minus = None
        self.rma_tr = None
        self.rma_dx = None
        self.warm = 0
        self.triggered_bull = False
        self.triggered_bear = False
        self._day = None

    @staticmethod
    def _rma(prev, x, n):
        return x if prev is None else prev + (x - prev) / n

    def on_bar(self, state) -> tuple[int, str]:
        ts_s = state.ohlcv_5s['timestamp']
        dt = pd.to_datetime(ts_s, unit='s', utc=True).tz_convert('America/Chicago')
        t = dt.time()

        # CONTINUOUS ROLLING WINDOW (see SMA variant): RMA state streams unbroken through
        # days and months, updated on every bar incl. overnight. No cold start. Only the
        # TRIGGER is RTH-gated.
        if state.day != self._day:
            self._day = state.day
            self.triggered_bull = False
            self.triggered_bear = False

        high = state.ohlcv_5s['high']
        low = state.ohlcv_5s['low']
        close = state.ohlcv_5s['close']

        adx = None
        if self.prev_high is not None:
            up_move = high - self.prev_high
            down_move = self.prev_low - low
            dm_plus = up_move if (up_move > down_move and up_move > 0) else 0.0
            dm_minus = down_move if (down_move > up_move and down_move > 0) else 0.0
            tr = max(high - low, abs(high - self.prev_close), abs(low - self.prev_close))

            self.rma_dm_plus = self._rma(self.rma_dm_plus, dm_plus, self.N_ADX)
            self.rma_dm_minus = self._rma(self.rma_dm_minus, dm_minus, self.N_ADX)
            self.rma_tr = self._rma(self.rma_tr, tr, self.N_ADX)

            tr_s = self.rma_tr if self.rma_tr else 1e-10
            di_plus = 100.0 * (self.rma_dm_plus / tr_s)
            di_minus = 100.0 * (self.rma_dm_minus / tr_s)
            dx = 100.0 * (abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10))
            self.rma_dx = self._rma(self.rma_dx, dx, self.N_ADX)
            self.warm += 1
            if self.warm >= self.N_ADX:      # same warmup depth as the SMA variant
                adx = self.rma_dx

        self.prices.append(close)
        if len(self.prices) > self.N_SMA:
            self.prices.pop(0)

        self.prev_high, self.prev_low, self.prev_close = high, low, close

        if adx is None or len(self.prices) < self.N_SMA:
            return 0, ''

        # Warm from the continuous stream; only ENTRIES are RTH-gated.
        if t < pd.Timestamp('08:30').time() or t > pd.Timestamp('15:15').time():
            return 0, ''

        sma20 = float(np.mean(self.prices))
        prev_sma20 = float(np.mean(self.prices[:-1]))
        prev_close = self.prices[-2]

        cross_above = (prev_close <= prev_sma20) and (close > sma20)
        cross_below = (prev_close >= prev_sma20) and (close < sma20)

        if not self.triggered_bull and adx > 25.0 and cross_above:
            self.triggered_bull = True
            return 1, 'bullish_runner'
        if not self.triggered_bear and adx > 25.0 and cross_below:
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
    """CROSS-11: FIRST cross of the session only.

    The legacy is not buggy here — its own comment is `# Scan for first cross` and it
    `break`s on the first cross in EITHER direction (ag_deepdive_11_cross.py:75-86).
    One setup per day IS the rule (doc 070/071). Emitting every cross would be a
    different strategy ("trade every cross"), not a bug fix — so we stop after the
    first trigger, whichever way it goes.
    600-bar (50-min) / 2400-bar (200-min) SMAs; buffer seeded with prior-day + today's
    ETH closes so the 2400-bar SMA is warm at the RTH open, as legacy's concat produces.
    """
    def __init__(self, prefill_closes=None):
        self.prices = []
        if prefill_closes is not None:
            self.prices.extend(prefill_closes[-2400:])
        self.cross_state = 0 # 1 if > , -1 if <
        self.done = False   # first-cross-only: one event per session, either direction

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

        if self.done:                      # legacy `break` — first cross ends the scan
            return 0, ''

        if cross_up:
            self.done = True
            return 1, 'bullish_runner'

        if cross_down:
            self.done = True
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
