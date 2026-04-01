"""
Probabilistic Trading Engine — probability-driven entry/exit decisions.

Replaces the deterministic gate cascade with:
  - CNN trajectory prediction: P(long) at 10 horizons
  - Brain cascade calibration: CNN → IS → OOS → Live
  - Template-based exit management: MFE/MAE/giveback from nearest template
  - Probability collapse exits: direction uncertainty → close trade

No gates, no binary direction, no brain overrides. Just probability.

Usage:
    engine = ProbabilisticTradingEngine(cnn_model, brain_cascade, templates, config)
    for bar in bars:
        result = engine.process_bar(bar_index, price, high, low, timestamp, features_22d)
        if result.trade_closed:
            trades.append(result.trade_closed)
"""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List

TICK = 0.25
TICK_VALUE = 0.50  # MNQ: $0.50 per tick


@dataclass
class ProbConfig:
    """All probabilistic engine tunable parameters."""
    entry_threshold: float = 0.65       # min calibrated P(direction) to enter
    hold_threshold: float = 0.45        # P below this = probability collapse exit
    horizon_entry: list = field(default_factory=lambda: [0, 1, 2])  # horizons for entry (n+1 to n+3)
    min_hold_bars: int = 2              # minimum bars before soft exits fire
    template_match_dist: float = 10.0   # loose match for exit params (not gating)
    default_sl_ticks: float = 40.0      # SL when no template matches
    default_tp_ticks: float = 80.0      # TP when no template matches
    trail_activation_pct: float = 0.6   # activate trail at 60% of expected MFE
    trail_pct: float = 0.35             # trail distance as fraction of peak MFE
    max_hold_bars: int = 480            # 2 hours at 15s (absolute backstop)
    lookback: int = 10                  # bars of history for CNN input


@dataclass
class Position:
    """Active trade state."""
    direction: str           # 'LONG' or 'SHORT'
    entry_price: float
    entry_bar: int
    entry_timestamp: float
    entry_p: float           # calibrated P(direction) at entry
    entry_trajectory: Optional[np.ndarray] = None  # P(long) at 10 horizons at entry
    template_id: Optional[int] = None
    sl_ticks: float = 40.0
    tp_ticks: float = 80.0
    expected_peak_bar: float = 0.5  # fraction of max_hold
    giveback_pct: float = 0.35
    peak_mfe_ticks: float = 0.0     # running max favorable excursion
    trail_active: bool = False
    trail_price: float = 0.0
    # Trajectory tracking: detect peak→trend→reversal lifecycle
    trajectory_history: list = field(default_factory=list)  # list of (bar, trajectory) tuples
    phase: str = 'peak'     # 'peak' → 'trend' → 'reversal_imminent'


@dataclass
class BarResult:
    """Output of process_bar."""
    action: str = 'HOLD'       # 'ENTER_LONG', 'ENTER_SHORT', 'EXIT', 'HOLD'
    exit_reason: str = ''
    p_direction: float = 0.5   # calibrated P(long) this bar
    trajectory: Optional[np.ndarray] = None  # 10 P(long) values
    trade_closed: Optional[dict] = None      # completed trade record


class ProbabilisticTradingEngine:
    """Probability-driven trading engine.

    CNN predicts trajectory → brain calibrates → enter on conviction →
    exit on probability collapse or trail.
    """

    def __init__(self, cnn_model, brain_cascade, template_library: dict,
                 scaler=None, centroids_scaled=None, valid_tids=None,
                 config: Optional[ProbConfig] = None, device: str = 'cuda'):
        self.cnn = cnn_model
        self.cascade = brain_cascade
        self.templates = template_library
        self.scaler = scaler
        self.centroids_scaled = centroids_scaled
        self.valid_tids = valid_tids or []
        self.config = config or ProbConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # State
        self.position: Optional[Position] = None
        self._feature_buffer: List[np.ndarray] = []  # rolling 22D feature history
        self._bar_count = 0

        # Stats
        self.total_trades = 0
        self.total_pnl = 0.0
        self.wins = 0

    def process_bar(self, bar_index: int, price: float, high: float, low: float,
                    timestamp: float, features_22d: np.ndarray) -> BarResult:
        """Process one bar. Returns action and optional trade record.

        Args:
            features_22d: (22,) array of current bar's features
                          (13D base + 4D wave + 2D prob + 3D level)
        """
        # Update feature buffer
        self._feature_buffer.append(features_22d.copy())
        if len(self._feature_buffer) > self.config.lookback:
            self._feature_buffer = self._feature_buffer[-self.config.lookback:]
        self._bar_count += 1

        # Need full lookback before trading
        if len(self._feature_buffer) < self.config.lookback:
            return BarResult(action='HOLD', p_direction=0.5)

        # Run CNN trajectory prediction
        trajectory = self._predict_trajectory()  # (10,) P(long) per horizon
        p_entry_horizons = trajectory[self.config.horizon_entry]
        p_long_avg = float(np.mean(p_entry_horizons))

        # If in position → manage trade
        if self.position is not None:
            return self._manage_position(bar_index, price, high, low, timestamp,
                                         trajectory, p_long_avg)

        # If flat → evaluate entry
        return self._evaluate_entry(bar_index, price, timestamp,
                                     trajectory, p_long_avg, features_22d)

    def _predict_trajectory(self) -> np.ndarray:
        """Run CNN on feature buffer, return 10 P(long) values."""
        window = np.array(self._feature_buffer[-self.config.lookback:], dtype=np.float32)
        with torch.no_grad():
            x = torch.FloatTensor(window).unsqueeze(0).to(self.device)
            raw = self.cnn(x)[0].cpu().numpy()  # (230,) = 10 × 23D

        # Extract P(long) from each horizon (last value of each 23D block)
        n_feat = 22
        p_longs = np.array([raw[h * (n_feat + 1) + n_feat] for h in range(10)])
        return p_longs

    def _evaluate_entry(self, bar_index, price, timestamp,
                        trajectory, p_long_avg, features_22d) -> BarResult:
        """Decide whether to enter a trade."""
        # Determine direction
        if p_long_avg > 0.5:
            direction = 'LONG'
            p_direction = p_long_avg
        else:
            direction = 'SHORT'
            p_direction = 1.0 - p_long_avg

        # Match nearest template for exit params
        template_id, exit_params = self._match_template(features_22d)

        # Calibrate through brain cascade
        p_calibrated = self.cascade.calibrate(template_id, direction, p_direction)

        # Entry decision
        if p_calibrated >= self.config.entry_threshold:
            self.position = Position(
                direction=direction,
                entry_price=price,
                entry_bar=bar_index,
                entry_timestamp=timestamp,
                entry_p=p_calibrated,
                entry_trajectory=trajectory.copy(),
                template_id=template_id,
                sl_ticks=exit_params.get('sl_ticks', self.config.default_sl_ticks),
                tp_ticks=exit_params.get('tp_ticks', self.config.default_tp_ticks),
                expected_peak_bar=exit_params.get('peak_bar', 0.5),
                giveback_pct=exit_params.get('giveback_pct', 0.35),
                trajectory_history=[(bar_index, trajectory.copy())],
                phase='peak',
            )
            action = f'ENTER_{direction}'
            return BarResult(action=action, p_direction=p_calibrated,
                             trajectory=trajectory)

        return BarResult(action='HOLD', p_direction=p_calibrated, trajectory=trajectory)

    def _manage_position(self, bar_index, price, high, low, timestamp,
                         trajectory, p_long_avg) -> BarResult:
        """Manage open position using peak→trend→reversal lifecycle.

        Phase transitions:
          PEAK:    just entered, first few bars. SL active, wait for trend.
          TREND:   sustained P(direction) across horizons. Trail loosely.
          REVERSAL_IMMINENT: near horizons flipped before far horizons.
                            Tighten trail aggressively → exit.
        """
        pos = self.position
        bars_held = bar_index - pos.entry_bar

        # PnL calculation
        if pos.direction == 'LONG':
            pnl_ticks = (price - pos.entry_price) / TICK
            mfe_ticks = (high - pos.entry_price) / TICK
            mae_ticks = (pos.entry_price - low) / TICK
            p_direction = p_long_avg
        else:
            pnl_ticks = (pos.entry_price - price) / TICK
            mfe_ticks = (pos.entry_price - low) / TICK
            mae_ticks = (high - pos.entry_price) / TICK
            p_direction = 1.0 - p_long_avg

        pos.peak_mfe_ticks = max(pos.peak_mfe_ticks, mfe_ticks)

        # Calibrate
        p_calibrated = self.cascade.calibrate(pos.template_id, pos.direction, p_direction)

        # Store trajectory for history
        pos.trajectory_history.append((bar_index, trajectory.copy()))

        # Orient trajectory to trade direction (P of OUR direction)
        if pos.direction == 'LONG':
            dir_traj = trajectory
        else:
            dir_traj = 1.0 - trajectory

        # Near horizons (n+1 to n+3) and far horizons (n+7 to n+10)
        p_near = float(np.mean(dir_traj[:3]))    # first 3 horizons
        p_far = float(np.mean(dir_traj[6:]))     # last 4 horizons

        # === Hard exits (always active) ===

        # 1. Stop loss
        if mae_ticks >= pos.sl_ticks:
            return self._close_position(price, timestamp, bar_index,
                                         'stop_loss', pnl_ticks, trajectory, p_calibrated)

        # 2. Max hold backstop
        if bars_held >= self.config.max_hold_bars:
            return self._close_position(price, timestamp, bar_index,
                                         'max_hold', pnl_ticks, trajectory, p_calibrated)

        # === Phase transitions ===

        if pos.phase == 'peak':
            # PEAK phase: just entered. Wait for trend confirmation or early exit.
            if bars_held >= self.config.min_hold_bars:
                if p_near > 0.6 and p_far > 0.55:
                    # Trend confirmed: sustained conviction across horizons
                    pos.phase = 'trend'
                elif p_near < self.config.hold_threshold:
                    # Peak failed: near horizons lost conviction immediately
                    return self._close_position(price, timestamp, bar_index,
                                                 'peak_failed', pnl_ticks,
                                                 trajectory, p_calibrated)

        elif pos.phase == 'trend':
            # TREND phase: hold with loose trail. Watch for near/far disagreement.

            # Activate trail once MFE reaches threshold
            trail_activation = pos.tp_ticks * self.config.trail_activation_pct
            if pos.peak_mfe_ticks >= trail_activation:
                pos.trail_active = True

            # Trail stop (loose during trend)
            if pos.trail_active:
                trail_distance = pos.peak_mfe_ticks * self.config.trail_pct
                retracement = pos.peak_mfe_ticks - pnl_ticks
                if retracement >= trail_distance:
                    return self._close_position(price, timestamp, bar_index,
                                                 'trail_stop', pnl_ticks,
                                                 trajectory, p_calibrated)

            # Detect reversal: near horizons weakening while far still strong
            # "The first sensors see the peak, far sensors think trend continues"
            NEAR_FAR_DISAGREE = 0.12  # min gap between far and near P
            if p_far - p_near > NEAR_FAR_DISAGREE and p_near < 0.55:
                pos.phase = 'reversal_imminent'

            # Also transition if overall conviction drops
            if p_near < self.config.hold_threshold:
                pos.phase = 'reversal_imminent'

        elif pos.phase == 'reversal_imminent':
            # REVERSAL phase: near horizons have flipped. Exit aggressively.

            # Immediate exit if near horizons crossed 0.5 (direction flipped)
            if p_near < 0.5:
                return self._close_position(price, timestamp, bar_index,
                                             'reversal_detected', pnl_ticks,
                                             trajectory, p_calibrated)

            # Tight trail: protect whatever profit remains
            if pos.peak_mfe_ticks > 3:
                tight_trail = max(3.0, pos.peak_mfe_ticks * 0.15)  # 15% of peak or 3 ticks min
                retracement = pos.peak_mfe_ticks - pnl_ticks
                if retracement >= tight_trail:
                    return self._close_position(price, timestamp, bar_index,
                                                 'reversal_trail', pnl_ticks,
                                                 trajectory, p_calibrated)

            # If near recovers → back to trend (false alarm)
            if p_near > 0.6 and p_far > 0.55:
                pos.phase = 'trend'

        # === Giveback (any phase after min_hold) ===
        if bars_held >= self.config.min_hold_bars:
            if pos.peak_mfe_ticks > 5 and pnl_ticks < pos.peak_mfe_ticks * (1 - pos.giveback_pct):
                return self._close_position(price, timestamp, bar_index,
                                             'giveback', pnl_ticks,
                                             trajectory, p_calibrated)

        return BarResult(action='HOLD', p_direction=p_calibrated, trajectory=trajectory)

    def _close_position(self, price, timestamp, bar_index, reason,
                        pnl_ticks, trajectory, p_calibrated) -> BarResult:
        """Close position and record trade."""
        pos = self.position
        pnl_dollars = pnl_ticks * TICK_VALUE
        bars_held = bar_index - pos.entry_bar

        trade = {
            'direction': pos.direction,
            'entry_price': pos.entry_price,
            'exit_price': price,
            'entry_bar': pos.entry_bar,
            'exit_bar': bar_index,
            'bars_held': bars_held,
            'entry_timestamp': pos.entry_timestamp,
            'exit_timestamp': timestamp,
            'pnl_ticks': pnl_ticks,
            'pnl': pnl_dollars,
            'result': 'WIN' if pnl_dollars > 0 else 'LOSS',
            'exit_reason': reason,
            'entry_p': pos.entry_p,
            'exit_p': p_calibrated,
            'peak_mfe_ticks': pos.peak_mfe_ticks,
            'template_id': pos.template_id,
            'phase_at_exit': pos.phase,
            # Seed data: entry trajectory + trajectory evolution for clustering
            'entry_trajectory': pos.entry_trajectory.tolist() if pos.entry_trajectory is not None else None,
            'trajectory_history_len': len(pos.trajectory_history),
        }

        # Update brain cascade
        self.cascade.update(pos.template_id, pos.direction, pnl_dollars)

        # Stats
        self.total_trades += 1
        self.total_pnl += pnl_dollars
        if pnl_dollars > 0:
            self.wins += 1

        # Clear position
        self.position = None

        return BarResult(action='EXIT', exit_reason=reason,
                         p_direction=p_calibrated, trajectory=trajectory,
                         trade_closed=trade)

    def _match_template(self, features_22d) -> tuple:
        """Match features to nearest template for exit params.

        Returns (template_id, exit_params_dict). Uses only the first 16D
        of features for template matching (templates are built on 16D/23D base).
        """
        default_params = {
            'sl_ticks': self.config.default_sl_ticks,
            'tp_ticks': self.config.default_tp_ticks,
            'peak_bar': 0.5,
            'giveback_pct': 0.35,
        }

        if self.scaler is None or self.centroids_scaled is None or len(self.valid_tids) == 0:
            return None, default_params

        # Use available features for matching (pad if needed)
        feat = features_22d[:16].reshape(1, -1)  # base 16D for template match
        expected_dim = getattr(self.scaler, 'n_features_in_', feat.shape[-1])
        if feat.shape[-1] < expected_dim:
            pad = np.zeros((1, expected_dim - feat.shape[-1]))
            feat = np.concatenate([feat, pad], axis=-1)

        feat_scaled = self.scaler.transform(feat)
        dists = np.linalg.norm(self.centroids_scaled - feat_scaled, axis=1)
        nearest_idx = int(np.argmin(dists))
        dist = float(dists[nearest_idx])

        if dist > self.config.template_match_dist:
            return None, default_params

        tid = self.valid_tids[nearest_idx]
        lib_entry = self.templates.get(tid, {})

        exit_params = {
            'sl_ticks': lib_entry.get('p25_mae_ticks', self.config.default_sl_ticks),
            'tp_ticks': lib_entry.get('p75_mfe_ticks', self.config.default_tp_ticks),
            'peak_bar': lib_entry.get('expected_peak_bar', 0.5),
            'giveback_pct': lib_entry.get('giveback_pct', 0.35),
        }
        return tid, exit_params

    @property
    def in_position(self) -> bool:
        return self.position is not None

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    def reset(self):
        """Reset for new session."""
        self.position = None
        self._feature_buffer = []
        self._bar_count = 0
