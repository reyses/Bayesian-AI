"""
Shared Bar Processor  -- single per-bar decision loop.

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

from dataclasses import dataclass
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
    on_entry: Optional[Callable] = None       # (action, bar_index) -> bool|None
    on_exit: Optional[Callable] = None        # (trade_dict, outcome) -> None
    on_bar: Optional[Callable] = None         # (bar_index, price, state, result) -> None
    modify_pnl: Optional[Callable] = None     # (pnl_dollars) -> float
    pre_exit_eval: Optional[Callable] = None  # (price, bar_index) -> dict


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
        use_cat: bool = False,
        **kwargs,
    ):
        self.exec_engine = exec_engine
        self._cfg = getattr(exec_engine, 'config', None)  # TradingConfig
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

        # Cat brain: rolling delta regime classifier (Schrodinger's quantum cat)
        self._use_cat = use_cat
        self._cat = None
        if use_cat:
            from core.cat_brain import CatBrain
            self._cat = CatBrain(window=200)

        # Monkey brain: counterfactual engine (phantom trades for every decision)
        self._use_monkey = kwargs.get('use_monkey', False)
        self._monkey = None
        if self._use_monkey:
            from core.counterfactual_engine import CounterfactualEngine
            self._monkey = CounterfactualEngine(tick_size=tick_size)

        # Peak detection entry: reversal signal from P_center + F_momentum
        self._peak_detection_enabled = True
        self._prev_P_center = 0.0
        self._prev_F_momentum = 0.0

        # 10-bar buildup buffer: tracks P_center deltas and F_momentum deltas
        # over a 10-bar window (10 min at 1m). A real reversal builds over
        # multiple bars. A fake peak is a single-bar spike.
        from collections import deque
        self._pc_delta_buffer = deque(maxlen=10)   # P_center bar-over-bar changes
        self._fm_delta_buffer = deque(maxlen=10)   # |F_momentum| bar-over-bar changes

        # Observational counters for peak detection validation
        self.peak_stats = {
            'peak_detected': 0,         # instantaneous peak signal fired
            'blocked_cooldown': 0,       # blocked by post-exit cooldown
            'blocked_no_buildup': 0,     # blocked by buildup/exhaustion filter
            'blocked_1m_sensor': 0,      # blocked by 1m sensor opposition
            'blocked_fake_peak': 0,      # blocked by volume+momentum fake filter
            'blocked_cat': 0,            # blocked by cat brain regime assessment
            'peak_entered': 0,           # made it through all gates -> candidate created
        }

    # ── Feature Extraction (single source of truth) ──────────────────

    def _build_features(self, state) -> np.ndarray:
        """Build feature vector from MarketState. Pads to scaler dims if needed."""
        feat = extract_feature_vector(
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
        )
        # Pad to match scaler dimensions (22D when --lookback, 16D otherwise)
        _expected = getattr(self.exec_engine.scaler, 'n_features_in_', len(feat))
        if len(feat) < _expected:
            feat = feat + [0.0] * (_expected - len(feat))
        return np.array([feat])

    # ── Candidate Building ───────────────────────────────────────────

    def _build_candidates(self, state, timestamp: float,
                          yolo: bool = False, bar_index: int = 0) -> list:
        """Build Candidate list from MarketState (compressed path).

        Two sources:
        1. Pattern detection (original): Roche breaks, structural drives
        2. Peak detection (new): P_center jump + F_momentum collapse = reversal entry
        """
        candidates = []
        _pt = getattr(state, 'pattern_type', '')
        _z = getattr(state, 'z_score', 0.0)

        # Source 1: Pattern detection (existing)
        if _pt and _pt != 'NONE':
            _cascade = getattr(state, 'cascade_detected', False)
            _struct = getattr(state, 'structure_confirmed', False)

            if _cascade or _struct or yolo:
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

        # Feed cat brain on every bar (regardless of peak detection)
        if self._cat is not None:
            self._cat.update(state)

        # Source 2: Peak detection (reversal entry)
        # Detects when a move just peaked: P_center rising + F_momentum collapsing.
        # Research: 83% detection rate, 85% precision, 0% false alarms.
        # The reversal IS the entry for the opposite direction.
        _in_cooldown = (hasattr(self, '_peak_cooldown_until')
                        and bar_index < self._peak_cooldown_until)
        if not candidates and self._peak_detection_enabled:
            if _in_cooldown:
                self._detect_peak_reversal(state)  # update buffers
                self.peak_stats['blocked_cooldown'] += 1
            else:
                self._last_bar_ts = timestamp  # for skip logs
                _peak_entry = self._detect_peak_reversal(state)
                if _peak_entry:
                    self.peak_stats['peak_detected'] += 1
                    # Gate 1: 1m sensor confirmation (vol + fm)
                    _sensor_ok = self._1m_confirms_peak(state, bar_ts=timestamp)
                    if not _sensor_ok:
                        pass  # already logged + counted in _1m_confirms_peak
                    else:
                        _feat = self._build_features(state)
                        _fm = getattr(state, 'F_momentum', 0.0)
                        _peak_long = _fm < 0
                        _dir = 'LONG' if _peak_long else 'SHORT'

                        # Gate 2: Cat brain regime check (if enabled)
                        _cat_ok = True
                        _cat_reason = ''
                        if self._cat is not None:
                            _cat_ok, _cat_reason = self._cat.should_enter_peak(_dir)
                            if not _cat_ok:
                                self.peak_stats['blocked_cat'] += 1
                                self._log_peak_skip(timestamp, f'cat: {_cat_reason}')
                                # Monkey: spawn phantom for blocked entry
                                if self._monkey is not None:
                                    self._monkey.on_skip(bar_index, price, _dir, f'cat_{_cat_reason}')

                        if _cat_ok:
                            self._log_peak_accept(timestamp, _dir, state)
                            candidates.append(Candidate(
                                state=state,
                                depth=self._anchor_depth,
                                timeframe=self._anchor_tf,
                                timestamp=timestamp,
                                pattern_type='PEAK_REVERSAL',
                                z_score=_z,
                                features=_feat,
                                forced_template_id=-100,
                            ))
                            self.peak_stats['peak_entered'] += 1
                            # Monkey: spawn phantoms with alt exit thresholds
                            if self._monkey is not None:
                                self._monkey.on_entry(bar_index, price, _dir)
        else:
            # Always update peak state even when not checking (same fix as IS path)
            self._detect_peak_reversal(state)  # updates _prev_P_center/_prev_F_momentum

        return candidates

    def _1m_confirms_peak(self, state, bar_ts: float = 0.0) -> bool:
        """Check if 1m sensors + approach context confirm peak is real.

        Two-layer gate:
        Layer 1 (sensor): 1m volume, DMI, F_momentum must not oppose the
            proposed direction. Blocks LONG-into-crash / SHORT-into-rally.
            Research (Feb 9): 183 LONG trades into selloff = -$8,593.
        Layer 2 (context): top-3 classifier features from peak template
            research (174K peaks, H-stats 2700-5800):
            - Volume at peak: exhausted (real) vs flowing (fake)
            - F_momentum at peak: decaying (real) vs building (fake)

        Returns True if peak is confirmed, False to block entry.
        """
        import numpy as np

        # Determine peak direction from F_momentum sign
        _fm = getattr(state, 'F_momentum', 0.0)
        _peak_long = _fm < 0  # old move was down -> reversal is up -> LONG

        # ── Layer 1: 1m sensor — OBSERVATIONAL ONLY (no blocking) ──
        # Research (2026-03-21): parent TF agreement is CONTRARIAN.
        # When 1m agrees with 15s peak direction, the move is MORE exhausted.
        # Validated peaks (parent agrees) have WORSE WR than unvalidated.
        # The 1m gate was costing $95K in IS by blocking profitable entries.
        # Signal moved to EXIT side: cascade fade (4+ TFs agree against trade = exit).
        # Kept as observational for the trade log / skip log.
        pass  # 1m entry gate REMOVED — all peaks pass through

        # ── Layer 2: peak context quality (research-backed thresholds) ──
        _peak_vol = abs(getattr(state, 'volume_delta', 0.0))
        _peak_fm_abs = abs(_fm)
        _log_vol = np.log1p(_peak_vol)
        _log_fm = np.log1p(_peak_fm_abs)

        # Fake peak: observational flag (no blocking action).
        # High vol+fm at peak = strong institutional move = good ENTRY but signals
        # the move hasn't peaked yet for EXIT timing.
        _FAKE_VOLUME_THRESHOLD = self._cfg.peak_fake_vol_threshold if self._cfg else 2.5
        _FAKE_FM_THRESHOLD = self._cfg.peak_fake_fm_threshold if self._cfg else 3.0
        _peak_vol = abs(getattr(state, 'volume_delta', 0.0))
        _peak_fm_abs = abs(_fm)
        _log_vol = np.log1p(_peak_vol)
        _log_fm = np.log1p(_peak_fm_abs)
        _is_fake_peak = (_log_vol > _FAKE_VOLUME_THRESHOLD and _log_fm > _FAKE_FM_THRESHOLD)
        if _is_fake_peak:
            self.peak_stats['fake_peak_flagged'] = self.peak_stats.get('fake_peak_flagged', 0) + 1
        self._last_fake_peak_flag = _is_fake_peak  # accessible by exit engine

        # ── Layer 3: ADX regime check (chop filter) ──
        # Primary: NT8 ADX (injected by live engine from bridge).
        # Fallback: computed adx_strength from MarketState.
        _1m_adx = getattr(self, '_nt8_1m_adx', 0.0)
        if _1m_adx < 0.01:
            _1m_adx = getattr(_ms, 'adx_strength', 0.0)
        _ADX_CHOP_THRESHOLD = self._cfg.peak_adx_chop_threshold if self._cfg else 15.0
        if _1m_adx < _ADX_CHOP_THRESHOLD:
            self.peak_stats['blocked_adx_chop'] = self.peak_stats.get('blocked_adx_chop', 0) + 1
            self._log_peak_skip(bar_ts, f"adx_chop: 1m ADX={_1m_adx:.1f} < {_ADX_CHOP_THRESHOLD} (choppy market)")
            return False

        return True

    def _detect_peak_reversal(self, state) -> bool:
        """Detect if the current bar shows a peak reversal with buildup.

        Two checks:
        1. Instantaneous: P_center increased + F_momentum collapsed (same as before)
        2. Buildup: the reversal has been building over multiple bars (decision funnel)

        The 10-bar buffer tracks P_center and F_momentum deltas. A real reversal
        shows consistent buildup (P_center rising for 3+ of last 10 bars AND
        F_momentum decaying for 3+ bars). A fake peak is a single-bar spike.

        Returns True if peak reversal detected with sufficient buildup.
        """
        _pc = getattr(state, 'P_at_center', 0.0)
        _fm = abs(getattr(state, 'F_momentum', 0.0))
        _coherence = getattr(state, 'oscillation_entropy_normalized', 0.0)

        # Compare with previous bar
        _prev_pc = getattr(self, '_prev_P_center', _pc)
        _prev_fm = getattr(self, '_prev_F_momentum', _fm)

        # Compute deltas and update buffers
        _pc_delta = (_pc - _prev_pc) / max(abs(_prev_pc), 1e-6) if _prev_pc > 0.01 else 0.0
        _fm_delta = (_fm - _prev_fm) / max(abs(_prev_fm), 1e-6) if _prev_fm > 0.5 else 0.0
        self._pc_delta_buffer.append(_pc_delta)
        self._fm_delta_buffer.append(_fm_delta)

        # Update for next bar
        self._prev_P_center = _pc
        self._prev_F_momentum = _fm

        # ── Check 1: Instantaneous signal (same as before) ──
        _pc_up = _pc_delta > 0.05       # P_center rose >5%
        _fm_down = _fm_delta < -0.10    # |F_momentum| fell >10%

        if not ((_pc_up or _fm_down) and _coherence > 0.55):
            return False  # no instantaneous signal

        # ── Check 2: Buildup OR Exhaustion (decision funnel, simplified) ──
        # Two valid reversal patterns in the 10-bar window:
        #
        # BUILDUP: momentum was growing, now suddenly collapses at the peak.
        #   F_momentum was INCREASING (positive deltas) then drops on this bar.
        #   The move built to a climax and broke.
        #
        # EXHAUSTION: momentum has been dying gradually over several bars.
        #   F_momentum DECREASING (negative deltas) for 3+ bars.
        #   P_center shifting as the center of gravity moves.
        #   The move ran out of energy.
        #
        # FAKE PEAK: neither pattern. Single-bar noise spike with no history.
        #   0-1 bars of consistent direction in the buffer.
        #
        if len(self._pc_delta_buffer) >= 3:
            _fm_decaying_bars = sum(1 for d in self._fm_delta_buffer if d < -0.01)
            _fm_building_bars = sum(1 for d in self._fm_delta_buffer if d > 0.01)
            _pc_shifting_bars = sum(1 for d in self._pc_delta_buffer if abs(d) > 0.01)

            # Exhaustion: momentum fading for 3+ bars
            _is_exhaustion = _fm_decaying_bars >= 3

            # Buildup: momentum was building (3+ bars growing), now collapsed
            _is_buildup = _fm_building_bars >= 3 and _fm_down

            # Either pattern is valid. Neither = fake spike.
            if not (_is_exhaustion or _is_buildup):
                # Check P_center shifting as fallback (center moving = regime change)
                if _pc_shifting_bars < 3:
                    self.peak_stats['blocked_no_buildup'] += 1
                    self._log_peak_skip(getattr(self, '_last_bar_ts', 0.0), f"no_buildup: fm_decay={_fm_decaying_bars} "
                                        f"fm_build={_fm_building_bars} pc_shift={_pc_shifting_bars} "
                                        f"(need 3+ in any pattern)")
                    return False  # no pattern in buffer, likely noise

        return True

    def _log_peak_skip(self, bar_ts: float, reason: str):
        """Log peak skip to terminal (throttled) + CSV (every skip)."""
        from datetime import datetime, timezone
        _bar_time = '??:??:??'
        if bar_ts > 1e9:
            _bar_time = datetime.fromtimestamp(bar_ts, tz=timezone.utc).strftime('%H:%M:%S')
        elif hasattr(self, '_last_bar_ts') and self._last_bar_ts > 1e9:
            bar_ts = self._last_bar_ts
            _bar_time = datetime.fromtimestamp(bar_ts, tz=timezone.utc).strftime('%H:%M:%S')
        _py = datetime.now().strftime('%H:%M:%S')

        # Terminal: throttled 1/second
        import time as _t
        _now = _t.monotonic()
        if not hasattr(self, '_last_skip_log') or _now - self._last_skip_log > 1.0:
            print(f"{_py} [{_bar_time}] [PEAK SKIP] {reason}", flush=True)
            self._last_skip_log = _now

        # CSV: every skip, append to file
        if not hasattr(self, '_skip_log_file'):
            import os
            os.makedirs('reports/live', exist_ok=True)
            _date = datetime.now().strftime('%Y%m%d')
            self._skip_log_path = f'reports/live/peak_skips_{_date}.csv'
            _exists = os.path.exists(self._skip_log_path)
            self._skip_log_file = open(self._skip_log_path, 'a', newline='')
            import csv
            self._skip_writer = csv.writer(self._skip_log_file)
            if not _exists:
                self._skip_writer.writerow(['py_time', 'nt8_time', 'bar_ts', 'reason'])
        self._skip_writer.writerow([_py, _bar_time, bar_ts, reason])
        self._skip_log_file.flush()

    def _log_peak_accept(self, bar_ts: float, direction: str, state):
        """Log accepted peak to terminal + CSV."""
        from datetime import datetime, timezone
        _nt8 = datetime.fromtimestamp(bar_ts, tz=timezone.utc).strftime('%H:%M:%S') if bar_ts > 0 else '??:??:??'
        _py = datetime.now().strftime('%H:%M:%S')
        _vol = abs(getattr(state, 'volume_delta', 0.0))
        _fm = abs(getattr(state, 'F_momentum', 0.0))
        _z = getattr(state, 'z_score', 0.0)

        print(f"{_py} [{_nt8}] [PEAK ENTRY] {direction} vol={_vol:.0f} fm={_fm:.1f} z={_z:.2f}", flush=True)

        # Same CSV file as skips — reason column distinguishes
        if not hasattr(self, '_skip_log_file'):
            import os, csv
            os.makedirs('reports/live', exist_ok=True)
            _date = datetime.now().strftime('%Y%m%d')
            self._skip_log_path = f'reports/live/peak_skips_{_date}.csv'
            _exists = os.path.exists(self._skip_log_path)
            self._skip_log_file = open(self._skip_log_path, 'a', newline='')
            self._skip_writer = csv.writer(self._skip_log_file)
            if not _exists:
                self._skip_writer.writerow(['py_time', 'nt8_time', 'bar_ts', 'reason'])
        self._skip_writer.writerow([_py, _nt8, bar_ts,
                                    f"ACCEPTED: {direction} vol={_vol:.0f} fm={_fm:.1f} z={_z:.2f}"])
        self._skip_log_file.flush()

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
        exit_only: bool = False,
    ) -> BarResult:
        """Process one bar. Returns BarResult with action and optional trade.

        Args:
            state:      Current bar's MarketState  -- used for BOTH entry candidates
                        (pattern_type, z_score, features) AND exit evaluation
                        (net_force, noise_ticks). Matches inline OOS behavior.
            exit_state: Optional override for exit evaluation. Usually omitted
                        (defaults to `state`). Only useful if caller needs
                        different state for exits (e.g. live 1s sub-bar exits).
            exit_only:  If True, only evaluate exits (skip TBN tick + entry).
                        Used for sub-bar (1s) exit checks between anchor bars.
                        SL and trail need 1s resolution to protect capital.

        Callers iterate bars and call this once per bar. The processor handles:
        TBN tick, candidate building, EE entry/exit, trade recording.
        """
        if not exit_only:
            # 1. Tick TBN workers (only on anchor bars, not sub-bar ticks)
            self.belief_network.tick_all(bar_index)

            # 1b. Update monkey phantoms (every anchor bar)
            if self._monkey is not None:
                self._monkey.on_bar(bar_index, price, bar_high, bar_low)

        # 2. If in position -> exit evaluation
        if self.exec_engine.in_position:
            _es = exit_state if exit_state is not None else state
            return self._process_exit(
                bar_index, price, bar_high, bar_low, timestamp, _es)

        # 3. If flat and not exit_only -> entry evaluation
        if exit_only:
            return BarResult(action=TradeAction(type=ActionType.HOLD))

        candidates = self._build_candidates(state, timestamp, yolo=yolo,
                                             bar_index=bar_index)
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
        # Gather TBN signals (routed to discovery TF for TF-aware exits)
        _disc_tf = 300.0  # fallback 5m
        _pos = self.exec_engine.pos_state
        if _pos is not None:
            _disc_tf = getattr(_pos, 'discovery_tf_seconds', 300.0)
        _exit_sig = self.belief_network.get_exit_signal(
            side=self.exec_engine.active_side,
            entry_price=self.exec_engine.entry_price,
            discovery_tf_seconds=_disc_tf,
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

        # Open position via EE (pass network_tp for TP fallback  -- FIX #2)
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
        """Handle EXIT action  -- record trade, clean up."""
        entry = self._current_entry
        pnl_ticks = getattr(action, 'pnl_ticks', 0)
        pnl_dollars = pnl_ticks * self._tick_size * self._point_value

        # Hook: modify PnL (slippage)
        if self._hooks.modify_pnl:
            pnl_dollars = self._hooks.modify_pnl(pnl_dollars)

        bars_held = bar_index - entry['entry_bar']
        exit_reason = getattr(action, 'exit_reason', 'unknown')

        # Record trade in brain  -- use actual fill price
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

        # Set peak cooldown to prevent re-entry on decaying peak
        self._peak_cooldown_until = bar_index + self.exit_engine.peak_state.cooldown_until - bar_index
        if hasattr(self.exit_engine.peak_state, 'cooldown_until'):
            self._peak_cooldown_until = self.exit_engine.peak_state.cooldown_until

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
