"""Peak Detector — grounded peak detection from base measurements.

Detects peaks using ONLY level-2 grounded features:
  1. velocity flip (dP/dt sign changed)
  2. volume collapse (participation dying)
  3. range compression (std of recent price changes is low)

No SFE, no PhysicsEngine, no templates, no brain.
Takes raw price/volume data, returns peak events.

This is the infant AdvanceEngine — step 1 of the rebuild.

Usage:
    detector = PeakDetector()
    for bar in bars:
        peak = detector.on_bar(close, high, low, volume, timestamp)
        if peak.detected:
            print(f"PEAK at {peak.price}, direction={peak.direction}")
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


TICK = 0.25

# Detection parameters (from OOS research: 1.73M bars)
VEL_WINDOW = 5             # bars for velocity computation
VOL_AVG_WINDOW = 60         # bars for volume average
VOL_COLLAPSE_THRESHOLD = 0.3  # volume must drop below 30% of avg
MAG_WINDOW = 20             # bars for prior move magnitude
MAG_PCTILE_WINDOW = 3600    # 1h of history for magnitude ranking
MAG_MIN_PCTILE = 0.75       # prior move must exceed p75
STD_WINDOW = 20             # bars for range compression measurement


@dataclass
class PeakEvent:
    """Result of peak detection on one bar."""
    detected: bool = False
    price: float = 0.0
    timestamp: float = 0.0
    direction: str = ''        # direction of the NEW move (post-peak)

    # Grounded measurements at detection time
    velocity: float = 0.0      # current velocity (level 2)
    volume_ratio: float = 0.0  # current_vol / avg_vol (level 2)
    magnitude: float = 0.0     # prior move in ticks (level 2)
    mag_pctile: float = 0.0    # magnitude percentile (level 2)
    std_price: float = 0.0     # recent price volatility (level 2)
    reason: str = ''           # why detected or why skipped


class PeakDetector:
    """Grounded peak detection from Price + Time + Volume.

    Maintains rolling buffers of base measurements.
    On each bar, checks if conditions match a peak event.
    """

    def __init__(
        self,
        vel_window: int = VEL_WINDOW,
        vol_avg_window: int = VOL_AVG_WINDOW,
        vol_collapse: float = VOL_COLLAPSE_THRESHOLD,
        mag_window: int = MAG_WINDOW,
        mag_pctile_window: int = MAG_PCTILE_WINDOW,
        mag_min_pctile: float = MAG_MIN_PCTILE,
        std_window: int = STD_WINDOW,
    ):
        self._vel_w = vel_window
        self._vol_avg_w = vol_avg_window
        self._vol_collapse = vol_collapse
        self._mag_w = mag_window
        self._mag_pct_w = mag_pctile_window
        self._mag_min_pct = mag_min_pctile
        self._std_w = std_window

        # Rolling buffers
        self._closes = deque(maxlen=max(mag_pctile_window, 5000))
        self._volumes = deque(maxlen=max(vol_avg_window + 10, 500))
        self._price_changes = deque(maxlen=max(mag_pctile_window, 5000))
        self._magnitudes = deque(maxlen=mag_pctile_window)

        # Velocity tracking
        self._vel_buffer = deque(maxlen=vel_window)
        self._prev_vel_sign = 0
        self._bar_count = 0

        # Stats
        self.stats = {
            'bars': 0,
            'peaks_detected': 0,
            'skipped_warmup': 0,
            'skipped_no_flip': 0,
            'skipped_magnitude': 0,
            'skipped_volume': 0,
        }

    def on_bar(self, close: float, high: float, low: float,
               volume: float, timestamp: float) -> PeakEvent:
        """Process one bar. Returns PeakEvent.

        Args:
            close: bar close price
            high: bar high price
            low: bar low price
            volume: bar volume
            timestamp: bar timestamp (unix seconds)
        """
        self._bar_count += 1
        self.stats['bars'] += 1

        # Update buffers
        if self._closes:
            dp = (close - self._closes[-1]) / TICK
        else:
            dp = 0.0

        self._closes.append(close)
        self._volumes.append(volume)
        self._price_changes.append(dp)
        self._vel_buffer.append(dp)

        # Compute magnitude (absolute move over mag_window)
        if len(self._closes) >= self._mag_w:
            mag = abs(close - self._closes[-self._mag_w]) / TICK
        else:
            mag = 0.0
        self._magnitudes.append(mag)

        # Warmup check
        min_bars = max(self._vel_w, self._vol_avg_w, self._mag_w, self._std_w) + 1
        if self._bar_count < min_bars:
            self.stats['skipped_warmup'] += 1
            return PeakEvent(reason=f'warmup {self._bar_count}/{min_bars}')

        # ── Level 2: Velocity ──
        velocity = np.mean(list(self._vel_buffer))
        vel_sign = 1 if velocity > 0 else (-1 if velocity < 0 else 0)

        # Check for velocity flip
        flipped = (vel_sign != 0 and self._prev_vel_sign != 0
                   and vel_sign != self._prev_vel_sign)
        self._prev_vel_sign = vel_sign

        if not flipped:
            self.stats['skipped_no_flip'] += 1
            return PeakEvent(
                velocity=velocity,
                reason=f'no_flip vel={velocity:+.2f}',
            )

        # ── Level 2: Magnitude ──
        # Causal percentile: how big is this move vs recent history?
        if len(self._magnitudes) >= 100:
            recent_mags = list(self._magnitudes)
            mag_pctile = sum(1 for m in recent_mags[:-1] if m < mag) / max(len(recent_mags) - 1, 1)
        else:
            mag_pctile = 0.5  # default during thin history

        if mag_pctile < self._mag_min_pct:
            self.stats['skipped_magnitude'] += 1
            return PeakEvent(
                velocity=velocity,
                magnitude=mag,
                mag_pctile=mag_pctile,
                reason=f'small_move mag={mag:.1f}t p={mag_pctile:.0%}<{self._mag_min_pct:.0%}',
            )

        # ── Level 2: Volume collapse ──
        vol_list = list(self._volumes)
        vol_avg = np.mean(vol_list[-self._vol_avg_w:]) if len(vol_list) >= self._vol_avg_w else 1.0
        vol_ratio = volume / vol_avg if vol_avg > 0 else 1.0

        if vol_ratio > self._vol_collapse:
            self.stats['skipped_volume'] += 1
            return PeakEvent(
                velocity=velocity,
                magnitude=mag,
                mag_pctile=mag_pctile,
                volume_ratio=vol_ratio,
                reason=f'vol_active ratio={vol_ratio:.2f}>{self._vol_collapse}',
            )

        # ── Level 2: Range compression (informational, not gating) ──
        if len(self._price_changes) >= self._std_w:
            recent_dp = list(self._price_changes)[-self._std_w:]
            std_price = np.std(recent_dp, ddof=1)
        else:
            std_price = 0.0

        # ── PEAK DETECTED ──
        direction = 'LONG' if vel_sign > 0 else 'SHORT'
        self.stats['peaks_detected'] += 1

        return PeakEvent(
            detected=True,
            price=close,
            timestamp=timestamp,
            direction=direction,
            velocity=velocity,
            volume_ratio=vol_ratio,
            magnitude=mag,
            mag_pctile=mag_pctile,
            std_price=std_price,
            reason=f'PEAK {direction} vel={velocity:+.2f} mag={mag:.0f}t vol_r={vol_ratio:.2f} std={std_price:.2f}',
        )

    def report(self) -> str:
        """Return stats summary."""
        s = self.stats
        det = s['peaks_detected']
        total = s['bars']
        pct = det / total * 100 if total > 0 else 0
        return (
            f'PeakDetector: {det:,} peaks in {total:,} bars ({pct:.2f}%)\n'
            f'  Skipped: warmup={s["skipped_warmup"]} '
            f'no_flip={s["skipped_no_flip"]} '
            f'magnitude={s["skipped_magnitude"]} '
            f'volume={s["skipped_volume"]}'
        )
