"""
Shared Bar Processor — single per-bar decision loop.

Replaces duplicated bar-processing logic across:
  - training/trainer.py (OOS compressed path)
  - live/history_replay.py (_replay_day)
  - live/live_engine.py (_check_entry / _check_exit)

Owns: candidate building, feature extraction, EE gate cascade,
      exit evaluation, trade recording.
Does NOT own: data loading, day preparation, equity tracking,
              GUI, NT8 orders, oracle audit, reporting.

Callers provide fully initialized engines via engine_factory.py.
Context-specific side effects are injected via BarProcessorHooks.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.bayesian_brain import MarketBayesianBrain, record_trade
from core.execution_engine import ActionType, Candidate, ExecutionEngine, TradeAction
from core.exit_engine import ExitEngine
from core.feature_extraction import extract_feature_vector
from core.timeframe_belief_network import TimeframeBeliefNetwork


# TF seconds for feature extraction
_TF_SECS = {
    '1s': 1, '5s': 5, '15s': 15, '30s': 30, '1m': 60,
    '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600,
}


@dataclass
class BarResult:
    """What happened on this bar."""
    action: TradeAction
    trade_completed: Optional[dict] = None   # filled when a trade closes
    candidates_built: int = 0


@dataclass
class BarProcessorHooks:
    """Optional callbacks for context-specific behavior.

    on_entry:       (action: TradeAction, bar_index: int) -> bool|None
                    Return False to veto the entry. None/True = proceed.
    on_exit:        (trade_dict: dict, outcome: TradeOutcome) -> None
    modify_pnl:     (pnl_dollars: float) -> float   (e.g. slippage injection)
    pre_exit_eval:  (price: float, bar_index: int) -> dict
                    Returns extra kwargs for exec_engine.on_bar() during exit
                    (sub_bar_highs, sub_bar_lows, net_force, noise_ticks, etc.)
    """
    on_entry: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    modify_pnl: Optional[Callable] = None
    pre_exit_eval: Optional[Callable] = None


class BarProcessor:
    """Shared per-bar processing core.

    Usage:
        processor = BarProcessor(exec_engine, tbn, exit_engine, brain,
                                 pattern_library, hooks=hooks)
        for bar in bars:
            result = processor.process_bar(bar_index, price, high, low, ts, state)
            if result.trade_completed:
                trades.append(result.trade_completed)
    """

    def __init__(
        self,
        exec_engine: ExecutionEngine,
        belief_network: TimeframeBeliefNetwork,
        exit_engine: ExitEngine,
        brain: MarketBayesianBrain,
        pattern_library: dict,
        anchor_tf: str = '15s',
        anchor_depth: int = 8,
        tick_size: float = 0.25,
        point_value: float = 2.0,
        hooks: Optional[BarProcessorHooks] = None,
    ):
        self.exec_engine = exec_engine
        self.belief_network = belief_network
        self.exit_engine = exit_engine
        self.brain = brain
        self.pattern_library = pattern_library

        self._anchor_tf = anchor_tf
        self._anchor_depth = anchor_depth
        self._tf_seconds = _TF_SECS.get(anchor_tf, 15)
        self._tick_size = tick_size
        self._point_value = point_value
        self._tick_value = tick_size * point_value

        self._hooks = hooks or BarProcessorHooks()
        self._current_entry: Optional[dict] = None

    # ── Feature Extraction (single source of truth) ──────────────────

    def _build_features(self, state) -> np.ndarray:
        """Build 16D feature vector from MarketState. No parent chain."""
        return np.array([extract_feature_vector(
            z_score=getattr(state, 'z_score', 0.0),
            velocity=getattr(state, 'velocity', 0.0),
            momentum=getattr(state, 'momentum_strength',
                             getattr(state, 'momentum', 0.0)),
            entropy_normalized=getattr(state, 'entropy_normalized', 0.0),
            tf_seconds=self._tf_seconds,
            depth=float(self._anchor_depth),
            parent_is_band_reversal=0.0,
            adx=getattr(state, 'adx_strength', 0.0) / 100.0,
            hurst=getattr(state, 'hurst_exponent', 0.5),
            dmi_diff=(getattr(state, 'dmi_plus', 0.0)
                      - getattr(state, 'dmi_minus', 0.0)) / 100.0,
            parent_z=0.0,
            parent_dmi_diff=0.0,
            root_is_roche=0.0,
            tf_alignment=0.0,
            pid=getattr(state, 'term_pid', 0.0),
            osc_coherence=getattr(state, 'oscillation_entropy_normalized', 0.0),
        )])

    # ── Candidate Building ───────────────────────────────────────────

    def _build_candidates(self, state, timestamp: float,
                          yolo: bool = False) -> list:
        """Build Candidate list from MarketState (compressed path)."""
        candidates = []
        _pt = getattr(state, 'pattern_type', '')
        if not _pt or _pt == 'NONE':
            return candidates

        _cascade = getattr(state, 'cascade_detected', False)
        _struct = getattr(state, 'structure_confirmed', False)

        if _cascade or _struct or yolo:
            _z = getattr(state, 'z_score', 0.0)
            _feat = self._build_features(state)
            candidates.append(Candidate(
                state=state,
                depth=self._anchor_depth,
                timeframe=self._anchor_tf,
                timestamp=timestamp,
                pattern_type=_pt,
                z_score=_z,
                features=_feat,
            ))
        return candidates

    # ── Main Bar Processing ──────────────────────────────────────────

    def process_bar(
        self,
        bar_index: int,
        price: float,
        bar_high: float,
        bar_low: float,
        timestamp: float,
        state,
        *,
        exit_state=None,
        pp_dir_override: str = None,
        yolo: bool = False,
    ) -> BarResult:
        """Process one bar. Returns BarResult with action and optional trade.

        Args:
            state:      Current bar's MarketState — used for BOTH entry candidates
                        (pattern_type, z_score, features) AND exit evaluation
                        (net_force, noise_ticks). Matches inline OOS behavior.
            exit_state: Optional override for exit evaluation. Usually omitted
                        (defaults to `state`). Only useful if caller needs
                        different state for exits (e.g. live 1s sub-bar exits).

        Callers iterate bars and call this once per bar. The processor handles:
        TBN tick, candidate building, EE entry/exit, trade recording.
        """
        # 1. Tick TBN workers
        self.belief_network.tick_all(bar_index)

        # 2. If in position → exit evaluation (uses PREVIOUS bar state)
        if self.exec_engine.in_position:
            _es = exit_state if exit_state is not None else state
            return self._process_exit(
                bar_index, price, bar_high, bar_low, timestamp, _es)

        # 3. If flat → entry evaluation (uses CURRENT bar state)
        candidates = self._build_candidates(state, timestamp, yolo=yolo)
        if not candidates:
            return BarResult(
                action=TradeAction(type=ActionType.HOLD),
                candidates_built=0,
            )

        # Get exit signal (None when flat, but EE expects it)
        action = self.exec_engine.on_bar(
            price=price,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_index=bar_index,
            candidates=candidates,
            pp_dir_override=pp_dir_override,
        )

        if action.type == ActionType.ENTER:
            return self._handle_entry(action, bar_index, price, timestamp,
                                      len(candidates))

        return BarResult(
            action=action,
            candidates_built=len(candidates),
        )

    def _process_exit(self, bar_index, price, bar_high, bar_low,
                      timestamp, state) -> BarResult:
        """Evaluate exit for open position."""
        # Gather TBN signals
        _exit_sig = self.belief_network.get_exit_signal(
            side=self.exec_engine.active_side,
            entry_price=self.exec_engine.entry_price,
        )
        _band_ctx = self.belief_network.get_band_confluence()
        _net_force = float(getattr(state, 'net_force',
                                   getattr(state, 'F_net', 0.0)))
        _noise = float(getattr(state, 'swing_noise_ticks', 0.0))

        # Extra exit kwargs from caller (sub-bar wicks, etc.)
        _extra = {}
        if self._hooks.pre_exit_eval:
            _extra = self._hooks.pre_exit_eval(price, bar_index) or {}

        action = self.exec_engine.on_bar(
            price=price,
            bar_high=bar_high,
            bar_low=bar_low,
            bar_index=bar_index,
            net_force=_net_force,
            band_context=_band_ctx,
            exit_signal=_exit_sig,
            noise_ticks=_noise,
            **_extra,
        )

        if action.type == ActionType.EXIT and self._current_entry is not None:
            return self._handle_exit(action, bar_index, price, timestamp)

        return BarResult(action=action)

    def _handle_entry(self, action: TradeAction, bar_index: int,
                      price: float, timestamp: float,
                      n_candidates: int) -> BarResult:
        """Handle ENTER action from ExecutionEngine."""
        tid = action.template_id
        lib_entry = self.pattern_library.get(tid, {})

        # Hook: on_entry (can veto)
        if self._hooks.on_entry:
            allow = self._hooks.on_entry(action, bar_index)
            if allow is False:
                return BarResult(
                    action=TradeAction(type=ActionType.HOLD),
                    candidates_built=n_candidates,
                )

        # Open position via EE (pass network_tp for TP fallback — FIX #2)
        self.exec_engine.position_opened(
            side=action.side,
            price=action.price,
            bar_index=bar_index,
            template_id=tid,
            lib_entry=lib_entry,
            sl_ticks=action.sl_ticks,
            tp_ticks=action.tp_ticks,
            network_tp=getattr(action, 'network_tp', None),
            max_hold_bars=getattr(action, 'max_hold_bars', 960),
        )

        # Track entry state
        self._current_entry = {
            'side': action.side,
            'entry_price': price,
            'entry_bar': bar_index,
            'entry_ts': timestamp,
            'tid': tid,
            'dir_source': getattr(action, 'dir_source', 'unknown'),
        }

        # Start TBN trade tracking (FIX #4: full params matching inline OOS)
        _avg_mfe_bar = lib_entry.get('avg_mfe_bar', 0.0)
        _p75_mfe_bar = lib_entry.get('p75_mfe_bar', 0.0)
        _p75_mfe_ticks = lib_entry.get('p75_mfe_ticks', 0.0)
        _max_hold = getattr(action, 'max_hold_bars', 960)
        self.belief_network.start_trade_tracking(
            side=action.side,
            entry_bar=bar_index,
            pattern_horizon_bars=_max_hold,
            target_mfe_ticks=_p75_mfe_ticks,
            resolve_bars=_avg_mfe_bar,
            entry_price=price,
        )
        # Per-template exit timescale (inline OOS lines 1364-1366)
        if _avg_mfe_bar > 0:
            self.belief_network.set_active_trade_timescale(_avg_mfe_bar, _p75_mfe_bar)

        return BarResult(
            action=action,
            candidates_built=n_candidates,
        )

    def _handle_exit(self, action: TradeAction, bar_index: int,
                     price: float, timestamp: float) -> BarResult:
        """Handle EXIT action — record trade, clean up."""
        entry = self._current_entry
        pnl_ticks = getattr(action, 'pnl_ticks', 0)
        pnl_dollars = pnl_ticks * self._tick_size * self._point_value

        # Hook: modify PnL (slippage)
        if self._hooks.modify_pnl:
            pnl_dollars = self._hooks.modify_pnl(pnl_dollars)

        bars_held = bar_index - entry['entry_bar']
        exit_reason = getattr(action, 'exit_reason', 'unknown')

        # Record trade in brain — use actual fill price
        _fill_price = getattr(action, 'price', price)
        outcome = record_trade(
            self.brain,
            tid=entry['tid'],
            entry_price=entry['entry_price'],
            exit_price=_fill_price,
            pnl=pnl_dollars,
            side=entry['side'],
            exit_reason=exit_reason,
            timestamp=timestamp,
            entry_time=entry['entry_ts'],
            exit_time=timestamp,
            tick_value=self._tick_value,
            hold_bars=bars_held,
        )

        # Build trade dict
        trade = {
            **entry,
            'exit_price': _fill_price,
            'exit_bar': bar_index,
            'exit_ts': timestamp,
            'pnl': pnl_dollars,
            'pnl_ticks': pnl_ticks,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
        }

        # Self-tune exit engine (FIX 3: matches inline OOS self-tuning)
        _pos = self.exec_engine.pos_state
        if _pos is not None:
            if entry['side'] == 'long':
                _trade_mfe = (_pos.peak_favorable - entry['entry_price']) / self._tick_size
            else:
                _trade_mfe = (entry['entry_price'] - _pos.peak_favorable) / self._tick_size
            _cap = pnl_ticks / _trade_mfe if _trade_mfe > 0 else 0.0
            self.exit_engine.record_trade_outcome(_trade_mfe, pnl_ticks, _cap)

        # Stop TBN trade tracking
        self.belief_network.stop_trade_tracking()

        # Clean up position state
        self.exec_engine.position_closed()
        self._current_entry = None

        # Hook: on_exit
        if self._hooks.on_exit:
            self._hooks.on_exit(trade, outcome)

        return BarResult(
            action=action,
            trade_completed=trade,
        )

    # ── End-of-Day Force Close ───────────────────────────────────────

    def force_close(self, price: float, timestamp: float,
                    bar_index: int) -> Optional[dict]:
        """Force-close open position at end of day. Returns trade dict or None."""
        if not self.exec_engine.in_position or self._current_entry is None:
            return None

        entry = self._current_entry
        side = entry['side']

        # Compute PnL
        if side == 'long':
            pnl_ticks = (price - entry['entry_price']) / self._tick_size
        else:
            pnl_ticks = (entry['entry_price'] - price) / self._tick_size
        pnl_dollars = pnl_ticks * self._tick_value

        if self._hooks.modify_pnl:
            pnl_dollars = self._hooks.modify_pnl(pnl_dollars)

        bars_held = bar_index - entry['entry_bar']

        outcome = record_trade(
            self.brain,
            tid=entry['tid'],
            entry_price=entry['entry_price'],
            exit_price=price,
            pnl=pnl_dollars,
            side=side,
            exit_reason='eod_flatten',
            timestamp=timestamp,
            entry_time=entry['entry_ts'],
            exit_time=timestamp,
            tick_value=self._tick_value,
            hold_bars=bars_held,
        )

        trade = {
            **entry,
            'exit_price': price,
            'exit_bar': bar_index,
            'exit_ts': timestamp,
            'pnl': pnl_dollars,
            'pnl_ticks': pnl_ticks,
            'exit_reason': 'eod_flatten',
            'bars_held': bars_held,
        }

        self.belief_network.stop_trade_tracking()
        self.exec_engine.position_closed()
        self._current_entry = None

        if self._hooks.on_exit:
            self._hooks.on_exit(trade, outcome)

        return trade

    @property
    def in_position(self) -> bool:
        return self.exec_engine.in_position
