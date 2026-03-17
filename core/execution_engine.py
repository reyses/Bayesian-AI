"""
Unified Execution Engine
========================
Single decision module for IS, OOS, and live trading.
Owns: direction cascade, entry gating, exit evaluation, position lifecycle.
Does NOT own: data I/O, oracle labeling, order submission, reporting.

Usage:
    eng = ExecutionEngine(brain, belief_network, exit_engine, pattern_library, ...)

    # Each bar:
    action = eng.on_bar(price, bar_high, bar_low, bar_index, candidates, ...)

    # action.type in: HOLD, ENTER, EXIT
    # Caller handles execution (sim record / NT8 order / etc.)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any

from core.exit_engine import ExitEngine, ExitAction, PositionState


# ── Action types returned to caller ──────────────────────────────────────

class ActionType(Enum):
    HOLD = 'hold'
    ENTER = 'enter'
    EXIT = 'exit'


@dataclass
class TradeAction:
    """What the engine wants the caller to do."""
    type: ActionType
    side: str = ''                # 'long' or 'short'
    price: float = 0.0
    template_id: Any = None
    dir_source: str = ''          # which cascade level decided direction
    p_long: float = 0.5
    # Exit-specific
    exit_reason: str = ''
    exit_result: Any = None       # ExitResult from exit_engine
    exit_signal: dict = field(default_factory=dict)
    pnl_ticks: float = 0.0
    pnl_dollars: float = 0.0
    bars_held: int = 0
    # Entry-specific
    sl_ticks: float = 0.0
    tp_ticks: float = 0.0
    trail_ticks: float = 0.0
    trail_activation_ticks: float = 0.0
    network_tp: Optional[float] = None
    lib_entry: dict = field(default_factory=dict)
    depth: int = 0
    dist: float = 0.0
    score: float = 999.0
    gate_label: str = ''
    # Diagnostic fields for caller bookkeeping
    conviction: float = 0.0
    belief_state: Any = None
    band_context: Any = None
    long_bias: float = 0.0
    short_bias: float = 0.0
    parent_tf: str = ''
    max_hold_bars: int = 0
    live_features_scaled: Optional[np.ndarray] = None
    # Per-candidate gate tracking (for decision matrix)
    candidate_gates: dict = field(default_factory=dict)
    # Score competition details: id(raw) -> {score, tid, dist, depth}
    candidate_scores: dict = field(default_factory=dict)
    # Worker bypass info
    is_bypass: bool = False
    bypass_candidate: Any = None
    bypass_dist: float = 999.0
    # Raw event reference
    raw_event: Any = None


@dataclass
class Candidate:
    """Unified candidate representation for both training and live."""
    state: Any                    # MarketState
    depth: int
    timeframe: Any
    timestamp: float
    pattern_type: str = ''
    z_score: float = 0.0
    features: Optional[np.ndarray] = None  # 16D feature vector
    raw_event: Any = None         # original PatternEvent (for oracle access)
    forced_template_id: Optional[int] = None  # bypass template match (PEAK_REVERSAL)


# ── Internal result from gate pre-check ──────────────────────────────────

@dataclass
class _GateResult:
    """Internal: result of gates 0-2 for a single candidate."""
    passed: bool
    gate_label: str = ''
    tid: Any = None
    dist: float = 999.0
    score: float = 999.0
    lib_entry: dict = field(default_factory=dict)
    feat_scaled: Optional[np.ndarray] = None
    cand: Any = None              # Candidate ref
    depth: int = 0


class ExecutionEngine:
    """
    Stateful per-bar decision engine.

    Execution flow:
      1. Pattern Quality + Depth + Template Match + Brain filter ALL candidates
      2. Score competition picks the BEST passing candidate
      3. Direction cascade + conviction check for winner ONLY
      4. If winner rejected, try worker-bypass path
      5. Return ENTER or HOLD

    Caller responsibilities:
    - Feed bars in order via on_bar()
    - Execute returned TradeActions (open/close positions)
    - Call position_opened() after executing an ENTER action
    - Call position_closed() after executing an EXIT action
    """

    def __init__(
        self,
        brain,
        belief_network,
        exit_engine: ExitEngine,
        pattern_library: dict,
        scaler,
        centroids_scaled: np.ndarray,
        valid_tids: list,
        tick_size: float = 0.25,
        point_value: float = 2.0,
        mode: str = 'is',
        # Scoring
        tier_score_adj: dict = None,
        depth_score_adj: dict = None,
        template_tier_map: dict = None,
        exception_tids: set = None,
        # Thresholds (deprecated — use config)
        bias_threshold: float = None,
        dmi_threshold: float = None,
        worker_bypass_conviction: float = None,
        # Depth filters
        depth_blacklist: set = None,
        depth_filter_out: set = None,
        depth_only: int = None,
        # Feature extractor
        feature_extractor=None,
        # Gate looseness (0=default, 1=relaxed, 2=open, 3=wide, 4=yolo)
        looseness: int = 0,
        config=None,
    ):
        # Config — single source of truth for all thresholds
        if config is None:
            from core.trading_config import TradingConfig
            config = TradingConfig()
        self.config = config

        self.brain = brain
        self.belief_network = belief_network
        self.exit_engine = exit_engine
        # Wire brain into exit engine for Bayesian ePnL exits
        if exit_engine is not None and brain is not None:
            exit_engine.set_brain(brain)
        self.pattern_library = pattern_library
        self.scaler = scaler
        self.centroids_scaled = centroids_scaled
        self.valid_tids = valid_tids
        self.tick_size = tick_size
        self.point_value = point_value
        self.mode = mode
        self.looseness = looseness

        self.tier_score_adj = tier_score_adj or {}
        self.depth_score_adj = depth_score_adj or {}
        self.template_tier_map = template_tier_map or {}
        self.exception_tids = exception_tids or set()

        # Thresholds from config (constructor args override for backward compat)
        self.bias_threshold = bias_threshold if bias_threshold is not None else config.bias_threshold
        self.dmi_threshold = dmi_threshold if dmi_threshold is not None else config.dmi_threshold
        self.gate1_dist = config.gate1_dist
        self.worker_bypass_conviction = (worker_bypass_conviction if worker_bypass_conviction is not None
                                          else config.worker_bypass_conviction)
        self._ADX_TREND_CONFIRMATION = config.adx_trend_confirmation
        self._HURST_TREND_CONFIRMATION = config.hurst_trend_confirmation

        self.depth_blacklist = depth_blacklist if depth_blacklist is not None else {0, 1, 2}
        self.depth_filter_out = depth_filter_out or set()
        self.depth_only = depth_only

        self.feature_extractor = feature_extractor

        # Gate thresholds from config (gate_thresholds.json can still override)
        self.hurst_min = config.hurst_min
        self.reversion_prob_min = config.reversion_prob_min
        self.momentum_override_ratio = config.momentum_override_ratio
        self._min_trade_depth = 3
        self._brain_min_prob = config.brain_min_prob
        self._load_gate_thresholds()
        self._apply_looseness()

        # Fractal DMI (dual-TF trend gating)
        from core.fractal_dmi import FractalDMI
        self.fractal_dmi = FractalDMI(config=config)

        # Quality-based scoring (loaded from quality_weights.json)
        self._quality_weights = None
        self._load_quality_weights()

        # Competition tracking
        self.bars_with_competition = 0   # 2+ candidates passed gates
        self.bars_single_candidate = 0   # exactly 1 candidate passed
        self.bars_no_candidate = 0       # 0 candidates passed
        self.tier_changed_winner = 0     # tier preference flipped the winner

        # Live ATR (set by caller for live/replay mode)
        self._live_atr_ticks: float = 0.0

        # Position state
        self.pos_state: Optional[PositionState] = None
        self.active_side: Optional[str] = None
        self.active_tid = None
        self.entry_price: float = 0.0
        self.entry_bar: int = 0


        # Gate stats
        self.gate_stats = {
            'gate0_skip': 0, 'gate0_noise': 0,
            'gate0_r3_struct': 0, 'gate0_r3_snap': 0,
            'gate0_r4_nightmare': 0, 'gate0_r4_struct': 0,
            'gate0_hurst': 0, 'gate0_momentum': 0, 'gate0_tunnel': 0,
            'gate0_regime': 0, 'gate0_session': 0,
            'gate0_5_skip': 0,
            'gate1_skip': 0, 'gate2_skip': 0,
            'gate2_5_tf_disagree': 0,
            'gate3_skip': 0,
            'gate4_momentum_align': 0,
            'physics_qg_skip': 0,
            'fdmi_fakeout_block': 0,
            'traded': 0, 'bypass_traded': 0,
            'total_candidates': 0,
            'competition_loser': 0,
        }

    # ── Regime Classification (improvement A) ─────────────────────────────

    # Pattern × regime compatibility matrix
    # True = allow, False = block, 'reduce' = allow with lower score
    _REGIME_PATTERN_COMPAT = {
        'strong_trend':  {'MOMENTUM_BREAK': True,  'BAND_REVERSAL': False},
        'developing':    {'MOMENTUM_BREAK': True,  'BAND_REVERSAL': 'reduce'},
        'exhausting':    {'MOMENTUM_BREAK': False, 'BAND_REVERSAL': True},
        'range':         {'MOMENTUM_BREAK': False, 'BAND_REVERSAL': True},
        'chop':          {'MOMENTUM_BREAK': False, 'BAND_REVERSAL': False},
    }

    def _classify_regime(self, state) -> str:
        """Classify current market regime from ADX, ADX slope, and Hurst.

        Returns: 'strong_trend', 'developing', 'exhausting', 'range', or 'chop'.
        """
        _cfg = self.config
        adx = getattr(state, 'adx_strength', 0.0)
        adx_prev = getattr(state, 'adx_prev', adx)
        hurst = getattr(state, 'hurst_exponent', 0.5)
        adx_slope = adx - adx_prev

        if adx >= _cfg.regime_strong_adx and adx_slope >= 0 and hurst > 0.55:
            return 'strong_trend'
        elif adx >= _cfg.regime_developing_adx and adx_slope > 0:
            return 'developing'
        elif adx >= _cfg.regime_strong_adx and adx_slope < _cfg.regime_exhaust_slope:
            return 'exhausting'
        elif adx < _cfg.regime_range_adx and hurst < 0.45:
            return 'range'
        else:
            return 'chop'

    # Looseness levels:
    #   0 = Default (current production thresholds)
    #   1 = Relaxed: softer physics (hurst 0.35, tunnel 0.25, mom 0.5)
    #   2 = Open: + wider template match (7.0), + depth min 2
    #   3 = Wide: + disable physics gates, + bypass conviction 0.50
    #   4 = YOLO: + all depths, + dist 12.0, + no brain reject
    _LOOSENESS_PRESETS = {
        1: {'hurst_min': 0.35, 'reversion_prob_min': 0.25, 'momentum_override_ratio': 0.5},
        2: {'hurst_min': 0.35, 'reversion_prob_min': 0.25, 'momentum_override_ratio': 0.5,
            'gate1_dist': 7.0, 'min_trade_depth': 2},
        3: {'hurst_min': 0.0, 'reversion_prob_min': 0.0, 'momentum_override_ratio': 0.0,
            'gate1_dist': 7.0, 'min_trade_depth': 2,
            'worker_bypass_conviction': 0.50},
        4: {'hurst_min': 0.0, 'reversion_prob_min': 0.0, 'momentum_override_ratio': 0.0,
            'gate1_dist': 12.0, 'min_trade_depth': 1,
            'worker_bypass_conviction': 0.40, 'brain_min_prob': 0.0,
            'depth_blacklist': set()},
    }

    def _apply_looseness(self):
        """Override gate thresholds based on looseness level."""
        if self.looseness <= 0:
            return
        preset = self._LOOSENESS_PRESETS.get(self.looseness)
        if not preset:
            # Clamp to max level
            preset = self._LOOSENESS_PRESETS[max(self._LOOSENESS_PRESETS)]
        for key, val in preset.items():
            if hasattr(self, key):
                setattr(self, key, val)
        # Special fields not directly attributes
        self._min_trade_depth = preset.get('min_trade_depth', 3)
        self._brain_min_prob = preset.get('brain_min_prob', 0.05)
        if 'depth_blacklist' in preset:
            self.depth_blacklist = preset['depth_blacklist']
        print(f"  [ExecutionEngine] Looseness={self.looseness}: "
              f"hurst>{self.hurst_min} tunnel>{self.reversion_prob_min} "
              f"mom_ratio>{self.momentum_override_ratio} dist<{self.gate1_dist} "
              f"min_depth={self._min_trade_depth}")

    def _load_gate_thresholds(self):
        """Load oracle-computed gate thresholds from checkpoints/gate_thresholds.json."""
        import json, os
        for path in ('checkpoints/gate_thresholds.json',
                      'checkpoints/snowflake/gate_thresholds.json'):
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        gt = json.load(f)
                    if 'hurst_min' in gt:
                        self.hurst_min = float(gt['hurst_min'])
                    if 'tunnel_prob_min' in gt:  # backward compat key name
                        self.reversion_prob_min = float(gt['tunnel_prob_min'])
                    if 'momentum_override_ratio' in gt:
                        self.momentum_override_ratio = float(gt['momentum_override_ratio'])
                    print(f"  [ExecutionEngine] Gate thresholds from {path}: "
                          f"hurst>{self.hurst_min} reversion_prob>{self.reversion_prob_min} "
                          f"mom_ratio>{self.momentum_override_ratio}")
                    return
                except Exception as e:
                    print(f"  [ExecutionEngine] WARN: failed to load {path}: {e}")

    def _load_quality_weights(self):
        """Load physics-based quality weights from checkpoints/quality_weights.json.

        When available, score competition uses predicted signal quality
        instead of structural score (depth + dist + tier).
        """
        import json, os
        path = 'checkpoints/quality_weights.json'
        if not os.path.exists(path):
            return
        try:
            with open(path, 'r') as f:
                qw = json.load(f)
            self._quality_weights = qw
            n_feat = len(qw.get('features', []))
            r2 = qw.get('r2', 0)
            print(f"  [ExecutionEngine] Quality weights loaded: "
                  f"{n_feat} features, R2={r2:.3f}")
        except Exception as e:
            print(f"  [ExecutionEngine] WARN: failed to load {path}: {e}")

    def _compute_quality_score(self, state, cand=None) -> float:
        """Predict signal quality (0-10) from physics state using loaded weights.

        Falls back to 0.0 if weights not loaded or feature missing.
        """
        qw = self._quality_weights
        if qw is None:
            return 0.0

        weights = qw['weights']
        means = qw['scaler_mean']
        stds = qw['scaler_std']

        score = qw.get('intercept', 0.0)
        for feat in qw['features']:
            # Get raw value (some features come from candidate, not state)
            if feat == 'micro_z' and cand is not None:
                raw = abs(cand.z_score)
            elif feat == 'depth' and cand is not None:
                raw = float(cand.depth)
            elif feat == 'macro_z' and cand is not None:
                chain = (getattr(cand.raw_event, 'parent_chain', [])
                         if cand.raw_event else [])
                raw = abs(chain[-1]['z']) if chain else 0.0
            elif feat == 'mom_rev_ratio':
                f_mom = abs(float(getattr(state, 'F_momentum', 0.0)))
                f_rev = abs(float(getattr(state, 'mean_reversion_force', 0.0)))
                raw = f_mom / f_rev if f_rev > 0 else 0.0
            elif feat == 'band_speed':
                vel = abs(float(getattr(state, 'velocity', 0.0)))
                sig = float(getattr(state, 'regression_sigma', 0.0))
                raw = vel / sig if sig > 0 else 0.0
            else:
                raw = self._get_physics_value(state, feat)
            # Standardize
            m = means.get(feat, 0.0)
            s = stds.get(feat, 1.0)
            if s < 1e-12:
                s = 1.0
            scaled = (raw - m) / s
            score += weights.get(feat, 0.0) * scaled

        return score

    @staticmethod
    def _get_physics_value(state, feat_name: str) -> float:
        """Extract a physics feature value from MarketState."""
        _MAP = {
            'hurst': 'hurst_exponent',
            'tunnel_prob': 'reversion_probability',
            'F_momentum': 'F_momentum',
            'F_reversion': 'mean_reversion_force',
            'velocity': 'velocity',
            'sigma': 'regression_sigma',
            'micro_z': None,  # from candidate, not state
            'macro_z': None,
            'depth': None,
        }
        attr = _MAP.get(feat_name, feat_name)
        if attr is None:
            return 0.0
        return float(getattr(state, attr, 0.0))

    @property
    def in_position(self) -> bool:
        return self.pos_state is not None

    # ── MAIN ENTRY POINT ─────────────────────────────────────────────────

    def on_bar(
        self,
        price: float,
        bar_high: float,
        bar_low: float,
        bar_index: int,
        candidates: List[Candidate] = None,
        net_force: float = 0.0,
        sub_bar_highs: list = None,
        sub_bar_lows: list = None,
        band_context: dict = None,
        exit_signal: dict = None,
        pp_dir_override: str = None,
        noise_ticks: float = 0.0,
    ) -> TradeAction:
        """Process one bar. Returns a TradeAction."""
        if self.in_position:
            return self._check_exit(
                price, bar_high, bar_low, bar_index,
                net_force, sub_bar_highs, sub_bar_lows,
                band_context, exit_signal,
                noise_ticks=noise_ticks,
            )

        if candidates:
            return self._check_entry(
                price, bar_index, candidates,
                pp_dir_override=pp_dir_override,
            )

        return TradeAction(type=ActionType.HOLD)

    # ── POSITION LIFECYCLE ────────────────────────────────────────────────

    def position_opened(self, side: str, price: float, bar_index: int,
                        template_id, lib_entry: dict,
                        sl_ticks: float = 0, tp_ticks: float = 0,
                        network_tp: float = None,
                        max_hold_bars: int = 960):
        """Caller tells engine a position was opened."""
        _sl = float(sl_ticks) if sl_ticks > 0 else self.config.sl_default_ticks
        _tp = float(tp_ticks) if tp_ticks > 0 else (float(network_tp) if network_tp and network_tp > 0 else self.config.tp_default_ticks * 0.8)
        self.pos_state = self.exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=bar_index,
            template_id=template_id,
            sl_ticks=_sl, tp_ticks=_tp,
            max_hold_bars=max_hold_bars,
            lib_entry=lib_entry,
        )
        self.active_side = side
        self.active_tid = template_id
        self.entry_price = price
        self.entry_bar = bar_index

    def position_closed(self):
        """Caller tells engine a position was closed."""
        self.pos_state = None
        self.active_side = None
        self.active_tid = None
        self.entry_price = 0.0
        self.entry_bar = 0

    # ── EXIT EVALUATION ──────────────────────────────────────────────────

    def _check_exit(self, price, bar_high, bar_low, bar_index,
                    net_force, sub_bar_highs, sub_bar_lows,
                    band_context=None, exit_signal=None,
                    noise_ticks: float = 0.0) -> TradeAction:
        """Evaluate exit conditions via ExitEngine."""
        if band_context is None and hasattr(self.belief_network, 'get_band_confluence'):
            band_context = self.belief_network.get_band_confluence()

        if exit_signal is None and hasattr(self.belief_network, 'get_exit_signal'):
            exit_signal = self.belief_network.get_exit_signal(
                self.active_side, self.entry_price)

        result = self.exit_engine.evaluate(
            pos=self.pos_state,
            bar_high=bar_high, bar_low=bar_low, bar_close=price,
            current_bar_index=bar_index,
            band_context=band_context,
            net_force=net_force,
            exit_signal=exit_signal,
            sub_bar_highs=sub_bar_highs,
            sub_bar_lows=sub_bar_lows,
            noise_ticks=noise_ticks,
            belief_network=self.belief_network,
        )

        if result.action != ExitAction.HOLD:
            pnl_ticks = result.pnl_ticks
            pnl_dollars = pnl_ticks * self.tick_size * self.point_value
            return TradeAction(
                type=ActionType.EXIT,
                side=self.active_side,
                price=result.exit_price,
                template_id=self.active_tid,
                exit_reason=result.action.value,
                exit_result=result,
                exit_signal=exit_signal or {},
                pnl_ticks=pnl_ticks,
                pnl_dollars=pnl_dollars,
                bars_held=result.bars_held,
            )

        return TradeAction(type=ActionType.HOLD)

    # ── ENTRY EVALUATION (matches orchestrator flow exactly) ─────────────

    def _check_entry(self, price, bar_index, candidates,
                     pp_dir_override=None) -> TradeAction:
        """
        Two-phase entry evaluation:
          Phase 1: Pattern/Depth/Template/Brain for ALL candidates → score competition
          Phase 2: Direction + conviction for winner ONLY
          Fallback: Worker bypass if no winner
        """
        candidate_gates = {}    # id(raw) -> gate_label
        candidate_scores = {}   # id(raw) -> {score, tid, dist, depth}
        gate_passers = {}       # id(raw) -> _GateResult
        bypass_candidate = None
        bypass_dist = 999.0

        # ── Phase 1: Gate cascade + score competition ─────────
        for cand in candidates:
            self.gate_stats['total_candidates'] += 1
            raw = cand.raw_event
            raw_id = id(raw) if raw is not None else id(cand)

            gr = self._gate_check(cand)

            if gr.passed:
                gate_passers[raw_id] = gr
            else:
                candidate_gates[raw_id] = gr.gate_label
                # Track best Template Match reject for worker bypass
                if gr.gate_label == 'gate1' and gr.dist < bypass_dist:
                    bypass_dist = gr.dist
                    bypass_candidate = cand

        # Pick best scorer + track competition
        n_passers = len(gate_passers)
        if n_passers == 0:
            self.bars_no_candidate += 1
        elif n_passers == 1:
            self.bars_single_candidate += 1
        else:
            self.bars_with_competition += 1

        best_raw_id = None
        best_score = 999.0
        best_gr = None
        for raw_id, gr in gate_passers.items():
            if gr.score < best_score:
                best_score = gr.score
                best_raw_id = raw_id
                best_gr = gr

        # Track if tier preference changed the winner
        if n_passers >= 2 and self.tier_score_adj and best_gr is not None:
            # Recompute winner without tier adjustment
            _best_no_tier = min(gate_passers.values(),
                                key=lambda g: g.score - self.tier_score_adj.get(
                                    self.template_tier_map.get(g.tid, 3), 0.0))
            if _best_no_tier.tid != best_gr.tid:
                self.tier_changed_winner += 1

        # Mark score losers + record scores for all passers
        for raw_id, gr in gate_passers.items():
            candidate_scores[raw_id] = {
                'score': gr.score, 'tid': gr.tid,
                'dist': gr.dist, 'depth': gr.depth,
            }
            if raw_id != best_raw_id:
                candidate_gates[raw_id] = 'score_loser'
                self.gate_stats['competition_loser'] += 1

        # ── Phase 2: Direction + conviction for winner ─────────
        if best_gr is not None:
            action = self._finalize_entry(
                best_gr, price, bar_index,
                pp_dir_override=pp_dir_override,
            )

            if action.type == ActionType.ENTER:
                action.candidate_gates = candidate_gates
                action.candidate_scores = candidate_scores
                return action
            else:
                # Conviction/momentum rejected the winner
                raw_id = id(best_gr.cand.raw_event) if best_gr.cand.raw_event else id(best_gr.cand)
                candidate_gates[raw_id] = action.gate_label

        # ── Fallback: Worker bypass ───────────────────────────
        if bypass_candidate is not None:
            bypass_action = self._check_worker_bypass(
                bypass_candidate, bypass_dist, price, bar_index)
            if bypass_action.type == ActionType.ENTER:
                bypass_action.candidate_gates = candidate_gates
                bypass_action.candidate_scores = candidate_scores
                return bypass_action

        # Nothing fired
        action = TradeAction(type=ActionType.HOLD)
        action.candidate_gates = candidate_gates
        action.candidate_scores = candidate_scores
        return action

    def _gate_check(self, cand: Candidate) -> _GateResult:
        """Gates 0-2: headroom, physics, depth, distance, brain.
        Returns _GateResult with pass/fail and intermediate values."""
        state = cand.state
        fail = lambda gate: _GateResult(passed=False, gate_label=gate, cand=cand)

        # ── Feature extraction ────────────────────────────────
        if cand.features is not None:
            features = cand.features
        elif self.feature_extractor is not None and cand.raw_event is not None:
            features = np.array([self.feature_extractor(cand.raw_event)])
        else:
            self.gate_stats['gate0_skip'] += 1
            return fail('gate0_no_features')

        # ── Cluster match ─────────────────────────────────────
        _feat_2d = features.reshape(1, -1) if features.ndim == 1 else features
        _expected_dim = getattr(self.scaler, 'n_features_in_', _feat_2d.shape[-1])
        if _feat_2d.shape[-1] < _expected_dim:
            _pad = np.zeros((_feat_2d.shape[0], _expected_dim - _feat_2d.shape[-1]))
            _feat_2d = np.concatenate([_feat_2d, _pad], axis=-1)
        # Check for forced template (PEAK_REVERSAL bypasses distance check)
        _forced_tid = getattr(cand, 'forced_template_id', None)
        if _forced_tid is not None and _forced_tid in self.pattern_library:
            tid = _forced_tid
            dist = 0.0  # forced match — no distance penalty
            lib_entry = self.pattern_library[tid]
        else:
            feat_scaled = self.scaler.transform(_feat_2d)
            dists = np.linalg.norm(self.centroids_scaled - feat_scaled, axis=1)
            nearest_idx = int(np.argmin(dists))
            dist = float(dists[nearest_idx])
            tid = self.valid_tids[nearest_idx]
            lib_entry = self.pattern_library.get(tid, {})

        # ── Data quality override ─────────────────────────────
        _data_override = False
        if self.exception_tids and cand.pattern_type:
            if dist < self.gate1_dist and tid in self.exception_tids:
                _data_override = True

        # ── Pattern Quality: headroom & physics rules ─────────
        micro_z = abs(cand.z_score)
        micro_pattern = cand.pattern_type
        should_skip = False
        skip_label = 'gate0'

        _cfg = self.config

        # ── Improvement A: Regime-aware pattern compatibility ──
        if not _data_override and micro_pattern:
            _regime = self._classify_regime(state)
            _compat = self._REGIME_PATTERN_COMPAT.get(_regime, {}).get(micro_pattern)
            if _compat is False:
                should_skip = True
                skip_label = 'gate0_regime'
            # 'reduce' = allow but pattern is fighting the regime (handled by scoring later)

        # ── Improvement F: Time-of-day session filter ──
        if not should_skip and _cfg.session_filter_enabled and cand.timestamp > 0:
            try:
                from datetime import datetime, timezone
                from zoneinfo import ZoneInfo
                _et = datetime.fromtimestamp(cand.timestamp, tz=timezone.utc).astimezone(
                    ZoneInfo('US/Eastern'))
                _hour_et = _et.hour + _et.minute / 60.0
                _is_overnight = _hour_et < 9.5 or _hour_et >= 16.0
                if _is_overnight and micro_z < _cfg.overnight_z_min:
                    should_skip = True
                    skip_label = 'gate0_session'
            except Exception:
                pass  # timestamp parse failure — don't block

        _is_peak_reversal = (micro_pattern == 'PEAK_REVERSAL')

        if not _data_override and not should_skip:
            if not micro_pattern:
                should_skip = True
                skip_label = 'gate0'
            elif _is_peak_reversal:
                pass  # peak reversal bypasses z-score gates (signal is state-based, not z-based)
            elif micro_z < _cfg.noise_z_threshold:
                should_skip = True
                skip_label = 'gate0_noise'
            elif _cfg.noise_z_threshold <= micro_z < _cfg.approach_z_threshold:
                if micro_pattern == 'MOMENTUM_BREAK':
                    adx = getattr(state, 'adx_strength', 0)
                    hurst = getattr(state, 'hurst_exponent', 0)
                    if (adx < self._ADX_TREND_CONFIRMATION or
                            hurst < self._HURST_TREND_CONFIRMATION):
                        should_skip = True
                        skip_label = 'gate0_r3_struct'
                elif micro_pattern == 'BAND_REVERSAL':
                    should_skip = True
                    skip_label = 'gate0_r3_snap'
            elif micro_z >= _cfg.approach_z_threshold:
                chain = (getattr(cand.raw_event, 'parent_chain', [])
                         if cand.raw_event else [])
                root_entry = chain[-1] if chain else None
                macro_z = abs(root_entry['z']) if root_entry else 0.0
                headroom = macro_z < _cfg.headroom_z_max
                if micro_pattern == 'BAND_REVERSAL':
                    if not headroom and micro_z > _cfg.nightmare_z:
                        should_skip = True
                        skip_label = 'gate0_r4_nightmare'
                elif micro_pattern == 'MOMENTUM_BREAK':
                    if not headroom:
                        should_skip = True
                        skip_label = 'gate0_r4_struct'

        # Rule 5: Physics safety
        if not should_skip and not _data_override:
            _st = state
            if self.hurst_min > 0 and getattr(_st, 'hurst_exponent', 1.0) < self.hurst_min:
                should_skip = True
                skip_label = 'gate0_hurst'
            # Rule 5b: low momentum filter — skip when reversion dominates
            # (choppy/ranging, no follow-through). We WANT high momentum.
            elif (abs(getattr(_st, 'F_momentum', 0.0)) <
                  abs(getattr(_st, 'mean_reversion_force', 0.0)) * self.momentum_override_ratio
                  and abs(getattr(_st, 'mean_reversion_force', 0.0)) > 0):
                should_skip = True
                skip_label = 'gate0_momentum'
            elif self.reversion_prob_min > 0 and getattr(_st, 'reversion_probability', 1.0) < self.reversion_prob_min:
                should_skip = True
                skip_label = 'gate0_tunnel'

        if should_skip:
            _counter = skip_label if skip_label in self.gate_stats else 'gate0_skip'
            self.gate_stats[_counter] += 1
            r = fail(skip_label)
            r.dist = dist
            return r

        # ── Depth Filter: min depth & blacklist ───────────────
        _cand_depth = cand.depth

        if self.depth_only is not None and _cand_depth != self.depth_only:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        if _cand_depth < self._min_trade_depth:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        if _cand_depth in self.depth_blacklist:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        if _cand_depth in self.depth_filter_out:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        # ── Template Match: cluster distance ──────────────────
        if dist >= self.gate1_dist:
            self.gate_stats['gate1_skip'] += 1
            r = fail('gate1')
            r.dist = dist
            return r

        # ── Brain Reject: profitability check ─────────────────
        if not self.brain.should_fire(tid, min_prob=self._brain_min_prob, min_conf=0.0):
            self.gate_stats['gate2_skip'] += 1
            r = fail('gate2')
            r.dist = dist
            r.tid = tid
            return r

        # ── Gate 2.5: Multi-TF Confluence (improvement C) ─────
        if _cfg.tf_confluence_enabled:
            _align = self.belief_network.get_dmi_alignment()
            _total = max(1, _align['total_tfs'])
            _tf_agree = _align['aligned_tfs'] / _total
            if _total >= 3 and _tf_agree < _cfg.tf_confluence_min:
                self.gate_stats['gate2_5_tf_disagree'] += 1
                r = fail('gate2_5_tf_disagree')
                r.dist = dist
                r.tid = tid
                return r

        # ── Score computation ─────────────────────────────────
        # Quality-based scoring: use physics-predicted signal quality when
        # weights are available. Negative because lower score = wins competition.
        if self._quality_weights is not None:
            quality = self._compute_quality_score(state, cand=cand)
            score = -quality  # higher quality → lower (better) score
        else:
            tier_adj = self.tier_score_adj.get(
                self.template_tier_map.get(tid, 3), 0.0)
            depth_adj = self.depth_score_adj.get(_cand_depth, 0.0)
            score = _cand_depth + dist + tier_adj + depth_adj

        return _GateResult(
            passed=True, tid=tid, dist=dist, score=score,
            lib_entry=lib_entry,
            feat_scaled=feat_scaled[0] if feat_scaled.ndim > 1 else feat_scaled,
            cand=cand, depth=_cand_depth,
        )

    def _finalize_entry(self, gr: _GateResult, price: float, bar_index: int,
                        pp_dir_override=None) -> TradeAction:
        """Phase 2: Direction cascade + conviction + sizing for the winning candidate."""
        cand = gr.cand
        tid = gr.tid
        lib_entry = gr.lib_entry
        feat_scaled = gr.feat_scaled

        # ── Direction cascade ─────────────────────────────────
        side, p_long, dir_source = self._direction_cascade(
            cand, tid, lib_entry, pp_dir_override, feat_scaled)

        if side is None:
            return TradeAction(type=ActionType.HOLD, gate_label='no_direction',
                               raw_event=cand.raw_event)

        # ── Low Conviction: belief strength check ─────────────
        _belief = self.belief_network.get_belief()
        network_tp = None
        _band = None

        if _belief is not None:
            if not getattr(_belief, 'is_confident', True):
                self.gate_stats['gate3_skip'] += 1
                return TradeAction(
                    type=ActionType.HOLD, gate_label='gate3',
                    dist=gr.dist, conviction=_belief.conviction,
                    template_id=tid, raw_event=cand.raw_event,
                    belief_state=_belief,
                )
            # Path direction override
            if hasattr(_belief, 'direction') and _belief.direction != side:
                side = _belief.direction
            # Network TP from predicted MFE
            if hasattr(_belief, 'predicted_mfe') and _belief.predicted_mfe > self.config.significance_threshold:
                network_tp = max(self.config.tp_min_ticks, int(round(_belief.predicted_mfe)))

        # ── Momentum Misalign: F_mom vs trade direction ──────
        # F_momentum = velocity * volume / sigma.  When its sign disagrees
        # with the trade direction, WR drops from 88% to ~45%.  Skip.
        _F_mom = getattr(cand.state, 'F_momentum', 0.0)
        _mom_sign = 1 if _F_mom > 0 else (-1 if _F_mom < 0 else 0)
        _side_sign = 1 if side == 'long' else -1
        if _mom_sign != 0 and _mom_sign != _side_sign:
            self.gate_stats['gate4_momentum_align'] += 1
            return TradeAction(
                type=ActionType.HOLD, gate_label='gate4_momentum_align',
                dist=gr.dist, template_id=tid, raw_event=cand.raw_event,
                belief_state=_belief if '_belief' in dir() else None,
            )

        # ── Fractal DMI: State A fakeout filter ──────────────
        _fdmi = self.fractal_dmi.evaluate(self.belief_network)
        if _fdmi.state_a_block:
            self.gate_stats['fdmi_fakeout_block'] += 1
            return TradeAction(
                type=ActionType.HOLD, gate_label='fdmi_fakeout',
                dist=gr.dist, template_id=tid, raw_event=cand.raw_event,
                belief_state=_belief if '_belief' in dir() else None,
            )

        # ── Exit sizing ───────────────────────────────────────
        sl_ticks, tp_ticks, trail_ticks, trail_act_ticks = self._compute_sizing(
            lib_entry, network_tp, feat_scaled, trade_side=side, template_id=tid,
            state=cand.state)

        # ── Max hold bars from parent TF ──────────────────────
        chain = (getattr(cand.raw_event, 'parent_chain', None)
                 if cand.raw_event else None) or []
        parent_tf = chain[0].get('tf', '4h') if chain else str(getattr(cand, 'timeframe', '4h'))
        from training.fractal_discovery_agent import TIMEFRAME_SECONDS
        parent_tf_sec = TIMEFRAME_SECONDS.get(parent_tf, 14400)
        max_hold_bars = max(self.config.max_hold_min_bars,
                            (parent_tf_sec * self.config.max_hold_parent_bars) // 15)

        self.gate_stats['traded'] += 1

        return TradeAction(
            type=ActionType.ENTER,
            side=side,
            price=price,
            template_id=tid,
            dir_source=dir_source,
            p_long=p_long,
            sl_ticks=sl_ticks,
            tp_ticks=tp_ticks,
            trail_ticks=trail_ticks,
            trail_activation_ticks=trail_act_ticks,
            network_tp=network_tp,
            lib_entry=lib_entry,
            depth=gr.depth,
            dist=gr.dist,
            score=gr.score,
            conviction=_belief.conviction if _belief else 0.0,
            belief_state=_belief,
            band_context=_band,
            long_bias=lib_entry.get('long_bias', 0.0),
            short_bias=lib_entry.get('short_bias', 0.0),
            parent_tf=parent_tf,
            max_hold_bars=max_hold_bars,
            live_features_scaled=feat_scaled,
            raw_event=cand.raw_event,
        )

    # ── WORKER BYPASS ─────────────────────────────────────────────────────

    def _check_worker_bypass(self, bypass_cand: Candidate,
                             bypass_dist: float,
                             price: float, bar_index: int) -> TradeAction:
        """Worker bypass: high-conviction belief fires without template match."""
        belief = self.belief_network.get_belief()
        if belief is None or belief.conviction < self.worker_bypass_conviction:
            return TradeAction(type=ActionType.HOLD)

        # Physics Quality: depth <= 3 and z < 0
        depth = bypass_cand.depth
        z = getattr(bypass_cand, 'z_score',
                    getattr(bypass_cand.state, 'z_score', 0.0))
        if depth > 3 or z >= 0:
            self.gate_stats['physics_qg_skip'] += 1
            return TradeAction(type=ActionType.HOLD, gate_label='physics_qg')

        _cfg = self.config
        side = belief.direction
        _st = bypass_cand.state
        _sigma = getattr(_st, 'regression_sigma', 0.0)
        sl_ticks = (max(_cfg.sl_min_ticks, int(round(_sigma / self.tick_size * _cfg.timescale_tighten_mult)))
                    if _sigma > 0 else _cfg.sl_default_ticks * 0.4)
        tp_ticks = (max(_cfg.tp_min_ticks * 2, int(round(belief.predicted_mfe)))
                    if belief.predicted_mfe > _cfg.significance_threshold else _cfg.sl_default_ticks)

        chain = (getattr(bypass_cand.raw_event, 'parent_chain', None)
                 if bypass_cand.raw_event else None) or []
        parent_tf = (chain[0].get('tf', '4h') if chain
                     else str(getattr(bypass_cand, 'timeframe', '4h')))
        from training.fractal_discovery_agent import TIMEFRAME_SECONDS
        parent_tf_sec = TIMEFRAME_SECONDS.get(parent_tf, 14400)
        max_hold_bars = max(_cfg.max_hold_min_bars,
                            (parent_tf_sec * _cfg.max_hold_parent_bars) // 15)

        self.gate_stats['bypass_traded'] += 1

        return TradeAction(
            type=ActionType.ENTER,
            side=side,
            price=price,
            template_id=-1,
            dir_source='worker_bypass',
            conviction=belief.conviction,
            belief_state=belief,
            sl_ticks=float(sl_ticks),
            tp_ticks=float(tp_ticks),
            trail_ticks=float(_cfg.trail_default_ticks * 0.6),
            trail_activation_ticks=0.0,
            depth=depth,
            dist=bypass_dist,
            parent_tf=parent_tf,
            max_hold_bars=max_hold_bars,
            is_bypass=True,
            bypass_candidate=bypass_cand,
            bypass_dist=bypass_dist,
            raw_event=bypass_cand.raw_event,
            lib_entry={
                'p25_mae_ticks': 0, 'mean_mae_ticks': 0,
                'regression_sigma_ticks': 0,
                'p75_mfe_ticks': tp_ticks,
            },
        )

    # ── EXIT SIZING ───────────────────────────────────────────────────────

    def _compute_sizing(self, lib_entry: dict, network_tp: float,
                        live_scaled: np.ndarray = None,
                        trade_side: str = None,
                        template_id: int = None,
                        state=None) -> Tuple[float, float, float, float]:
        """Compute SL, TP, trail, trail activation from template stats.

        Oracle stats (MAE/MFE) are computed in the discovery TF (e.g. 30m bars).
        Execution happens on 15s bars. We scale by sqrt(tf_ratio) because price
        excursion scales with sqrt(time) under diffusion (volatility scaling).
        """
        _cfg = self.config
        _EXEC_TF_SEC = 15.0  # execution timeframe
        _disc_tf = lib_entry.get('discovery_tf_seconds', _EXEC_TF_SEC)
        _tf_ratio = _disc_tf / _EXEC_TF_SEC
        # TF scaling: discovery stats → execution timeframe.
        # discovery_tf > exec_tf: sqrt scaling (diffusion model, MAE ~ sqrt(T))
        # discovery_tf < exec_tf: linear scaling (1s oracle covers 15× more bars,
        #   captures moves we can't replicate on coarser execution bars)
        _tf_scale = _tf_ratio ** 0.5 if _tf_ratio >= 1.0 else 1.0 / _tf_ratio

        _reg_sigma = lib_entry.get('regression_sigma_ticks', 0.0) / _tf_scale
        _mean_mae = lib_entry.get('mean_mae_ticks', 0.0) / _tf_scale
        _p75_mfe = lib_entry.get('p75_mfe_ticks', 0.0) / _tf_scale
        _p25_mae = lib_entry.get('p25_mae_ticks', 0.0) / _tf_scale
        _p95_mae = lib_entry.get('p95_mae_ticks', 0.0) / _tf_scale
        _mae_std = lib_entry.get('mae_std_ticks', 0.0) / _tf_scale
        params = lib_entry.get('params', {})

        # Phase 1: initial hard stop — tolerance interval from MAE distribution
        # SL = p95 MAE × multiplier (last-resort: only ~5% of trades should naturally exceed)
        # sl_tolerance_mult tunes this: >1 = wider (more room), <1 = tighter
        if _p95_mae > _cfg.significance_threshold:
            sl_ticks = max(_cfg.sl_min_ticks, int(round(_p95_mae * _cfg.sl_tolerance_mult)))
        elif _mean_mae > _cfg.significance_threshold and _mae_std > 0:
            # Fallback: mean + k*σ (k=5 ≈ 99.99994%)
            sl_ticks = max(_cfg.sl_min_ticks, int(round((_mean_mae + _cfg.sl_tolerance_k * _mae_std) * _cfg.sl_tolerance_mult)))
        elif _mean_mae > _cfg.significance_threshold:
            sl_ticks = max(_cfg.sl_min_ticks, int(round(_mean_mae * _cfg.sl_mean_mae_mult)))
        else:
            sl_ticks = params.get('stop_loss_ticks', _cfg.sl_default_ticks)
        # Hard cap: prevents runaway SL from unscaled/extreme TF stats
        sl_ticks = min(sl_ticks, _cfg.sl_max_ticks)

        # MFE-proportional cap: SL should not exceed sl_mfe_ratio × expected profit
        # Small expected MFE → tight SL (breakeven protection)
        # Large expected MFE → wider SL (room to develop)
        if _p75_mfe > _cfg.significance_threshold:
            _mfe_cap = max(_cfg.sl_min_ticks, int(round(_p75_mfe * _cfg.sl_mfe_ratio)))
            sl_ticks = min(sl_ticks, _mfe_cap)

        # Phase 2: trailing stop distance
        if _reg_sigma > _cfg.significance_threshold:
            trail_ticks = max(_cfg.trail_min_ticks, int(round(_reg_sigma * _cfg.trail_sigma_mult)))
        elif _mean_mae > _cfg.significance_threshold:
            trail_ticks = max(_cfg.trail_min_ticks, int(round(_mean_mae * _cfg.trail_mae_mult)))
        else:
            trail_ticks = params.get('trailing_stop_ticks', _cfg.trail_default_ticks)

        # Trail activation
        trail_act_ticks = (max(_cfg.trail_min_ticks, int(round(_p25_mae * _cfg.trail_activation_mae_mult)))
                           if _p25_mae > _cfg.significance_threshold else 0)

        # TP: template p75_mfe as anchor, OLS adjusts within sanity bounds
        # Brain expected PnL offsets the anchor with actual realized performance
        _anchor_ticks = (_p75_mfe if _p75_mfe > _cfg.significance_threshold
                         else params.get('take_profit_ticks', _cfg.tp_default_ticks))

        if network_tp is not None:
            tp_ticks = network_tp
        else:
            _mfe_coeff = lib_entry.get('mfe_coeff')
            if _mfe_coeff is not None and live_scaled is not None:
                _pred_mfe_pts = (np.dot(live_scaled, np.array(_mfe_coeff))
                                 + lib_entry.get('mfe_intercept', 0.0))
                _pred_mfe_ticks = max(0.0, float(_pred_mfe_pts) / self.tick_size)
                # Sanity gate: OLS must be within bounds of anchor
                if (_pred_mfe_ticks > _cfg.significance_threshold
                        and _anchor_ticks * _cfg.ols_lower_pct <= _pred_mfe_ticks <= _anchor_ticks * _cfg.ols_upper_pct):
                    tp_ticks = max(_cfg.tp_min_ticks, int(round(_pred_mfe_ticks)))
                else:
                    tp_ticks = max(_cfg.tp_min_ticks, int(round(_anchor_ticks)))
            else:
                tp_ticks = max(_cfg.tp_min_ticks, int(round(_anchor_ticks)))

        # Brain offset: adjust anchor with actual realized performance
        if self.brain is not None and trade_side is not None:
            _tid = template_id
            _side = trade_side.lower()
            _exp_pnl = self.brain.get_expected_pnl(_tid, _side) if _tid is not None else None
            if _exp_pnl is not None:
                _exp_ticks = _exp_pnl / (self.tick_size * self.point_value)  # $ → ticks
                # Blend: anchor stays base, brain nudges within cap
                _adj = np.clip(_exp_ticks,
                               -_anchor_ticks * _cfg.brain_tp_adjust_pct,
                               _anchor_ticks * _cfg.brain_tp_adjust_pct)
                tp_ticks = max(_cfg.tp_min_ticks, int(round(tp_ticks + _adj)))

        # ── Improvement B: Volatility-normalized ATR sizing by regime ──
        _atr = self._live_atr_ticks
        if _atr <= 0 and state is not None:
            _atr = getattr(state, 'swing_noise_ticks', 0.0)

        if _cfg.vol_sizing_enabled and _atr > 0:
            _regime = self._classify_regime(state) if state is not None else 'chop'
            if _regime == 'strong_trend':
                _sl_mult, _tp_mult = _cfg.vol_sl_strong_trend, _cfg.vol_tp_strong_trend
            elif _regime == 'developing':
                _sl_mult, _tp_mult = _cfg.vol_sl_developing, _cfg.vol_tp_developing
            elif _regime in ('range', 'exhausting'):
                _sl_mult, _tp_mult = _cfg.vol_sl_range, _cfg.vol_tp_range
            else:
                _sl_mult, _tp_mult = _cfg.vol_sl_default, _cfg.vol_tp_default
            sl_ticks = max(sl_ticks, max(_cfg.sl_min_ticks, int(round(_atr * _sl_mult))))
            tp_ticks = max(tp_ticks, max(_cfg.tp_min_ticks, int(round(_atr * _tp_mult))))

        return float(sl_ticks), float(tp_ticks), float(trail_ticks), float(trail_act_ticks)

    def set_live_atr(self, atr_ticks: float):
        """Set ATR in ticks for live/replay mode SL/TP floor enforcement."""
        self._live_atr_ticks = atr_ticks

    # ── DIRECTION CASCADE ────────────────────────────────────────────────

    def _direction_cascade(self, cand: Candidate, tid, lib_entry: dict,
                           pp_dir_override=None,
                           live_scaled=None) -> Tuple[Optional[str], float, str]:
        """
        Unified direction cascade matching orchestrator priority order.

        Priority order:
          -1   Ping-pong live bias (caller provides override)
          -0.5 Live brain dir_bias (live/replay only, min 5 trades)
           0.3 Live momentum (velocity+accel, live/replay only)
           0.5 Signed MFE regression (learned)
           1   Per-cluster logistic regression
           1.5 Brain direction-specific win rate
           2   Template aggregate bias
           3   Multi-TF band confluence
           4   DMI (trend-following)
           5   Velocity fallback
        """
        state = cand.state
        _cfg = self.config
        _BIAS_THRESH = self.bias_threshold

        # ── Priority -1: Ping-pong / live direction override ──
        if pp_dir_override is not None:
            return pp_dir_override, 0.65, 'pp_override'

        # ── Priority -0.5: Brain dir_bias (H0/H1 counterfactual learned) ──
        _learned_bias = self.get_live_dir_bias(tid)
        if _learned_bias is not None:
            _p = 0.60 if _learned_bias == 'long' else 0.40
            return _learned_bias, _p, 'brain_bias'

        # ── Priority 0.3: Live momentum (velocity+accel, live/replay only) ──
        if self.mode in ('live', 'replay'):
            _vel = float(getattr(state, 'velocity', 0.0))
            _fnet = float(getattr(state, 'F_net', 0.0))
            _mom = _vel + _cfg.momentum_accel_weight * _fnet
            if abs(_mom) > _cfg.momentum_trigger:
                s = 'long' if _mom > 0 else 'short'
                _p = 0.5 + min(_cfg.momentum_conviction_cap,
                               abs(_mom) * _cfg.momentum_conviction_coeff)
                return s, _p if s == 'long' else 1.0 - _p, 'live_momentum'

        # ── Improvement G: Confidence-weighted direction voting ──
        # All direction sources contribute weighted votes instead of first-match waterfall.
        # Each vote: (side, confidence 0-1, weight, source_name)
        # Weighted sum determines direction. Minimum threshold to enter.
        _votes_long = 0.0    # weighted long score
        _votes_short = 0.0   # weighted short score
        _sources = []

        # Vote 1: Signed MFE regression (highest-trained signal)
        _smfe_coeff = lib_entry.get('signed_mfe_coeff')
        if _smfe_coeff is not None:
            _depth = cand.depth
            _dmi = (getattr(state, 'dmi_plus', 0.0)
                    - getattr(state, 'dmi_minus', 0.0))
            _pred = float(
                np.dot(np.array([[_depth, _dmi]]), np.array(_smfe_coeff)).item()
                + lib_entry.get('signed_mfe_intercept', 0.0)
            )
            if abs(_pred) > _cfg.momentum_trigger:
                _conf = min(1.0, abs(_pred) * _cfg.signed_mfe_conviction_coeff * 2)
                _w = _cfg.dir_smfe_weight
                if _pred > 0:
                    _votes_long += _w * _conf
                else:
                    _votes_short += _w * _conf
                _sources.append('smfe')

        # Vote 2: Per-cluster logistic regression
        _dir_coeff = lib_entry.get('dir_coeff')
        if _dir_coeff is not None and live_scaled is not None:
            _dir_logit = (np.dot(live_scaled, np.array(_dir_coeff))
                          + lib_entry.get('dir_intercept', 0.0))
            _dir_prob = 1.0 / (1.0 + np.exp(-float(_dir_logit)))
            _conf = abs(_dir_prob - 0.5) * 2.0  # 0-1
            _w = _cfg.dir_logistic_weight
            if _dir_prob > 0.5:
                _votes_long += _w * _conf
            else:
                _votes_short += _w * _conf
            _sources.append('logistic')

        # Vote 3: Brain direction-specific win rate
        _dir_long = self.brain.get_dir_probability(tid, 'LONG')
        _dir_short = self.brain.get_dir_probability(tid, 'SHORT')
        if _dir_long is not None and _dir_short is not None:
            _diff = _dir_long - _dir_short
            if abs(_diff) > _cfg.brain_winrate_margin:
                _conf = min(1.0, abs(_diff) * 2.0)
                _w = _cfg.dir_brain_weight
                if _diff > 0:
                    _votes_long += _w * _conf
                else:
                    _votes_short += _w * _conf
                _sources.append('brain_dir')

        # Vote 4: Template aggregate bias
        long_bias = lib_entry.get('long_bias', 0.0)
        short_bias = lib_entry.get('short_bias', 0.0)
        if long_bias + short_bias >= _cfg.template_bias_min_sum:
            _diff = long_bias - short_bias
            _conf = min(1.0, abs(_diff) * 2.0)
            _w = _cfg.dir_template_weight
            if _diff > 0:
                _votes_long += _w * _conf
            else:
                _votes_short += _w * _conf
            _sources.append('template')

        # Vote 5: Fractal DMI ignition / reversion (strongest dynamic signal)
        _fdmi = self.fractal_dmi.evaluate(self.belief_network)
        if _fdmi.state_b_long or _fdmi.state_d_reversion_long:
            _w = _cfg.dir_fdmi_weight
            _votes_long += _w * 0.8
            _sources.append('fdmi_L')
        elif _fdmi.state_b_short or _fdmi.state_d_reversion_short:
            _w = _cfg.dir_fdmi_weight
            _votes_short += _w * 0.8
            _sources.append('fdmi_S')

        # Vote 6: Multi-TF band confluence
        _band = self.belief_network.get_band_confluence()
        if _band is not None and _band.get('direction') is not None:
            _conf = _band['strength']
            _w = _cfg.dir_band_weight
            if _band['direction'] == 'long':
                _votes_long += _w * _conf
            else:
                _votes_short += _w * _conf
            _sources.append('band')

        # Vote 7: Multi-TF DMI trend
        _dmi_trend = self.belief_network.get_dmi_trend(min_strength=20.0)
        if _dmi_trend is not None:
            _conf = min(1.0, _dmi_trend['strength'] / 40.0)  # normalize
            _w = _cfg.dir_dmi_weight
            if _dmi_trend['direction'] == 'long':
                _votes_long += _w * _conf
            else:
                _votes_short += _w * _conf
            _sources.append('dmi')

        # Vote 8: Velocity fallback (weakest, always available)
        _vel = float(getattr(state, 'velocity', 0.0))
        if abs(_vel) > 0.01:
            _conf = min(1.0, abs(_vel) * 0.5)
            _w = _cfg.dir_velocity_weight
            if _vel > 0:
                _votes_long += _w * _conf
            else:
                _votes_short += _w * _conf
            _sources.append('vel')

        # ── Aggregate votes ────────────────────────────────────
        _total_votes = _votes_long + _votes_short
        if _total_votes < 1e-6:
            # No signals at all — velocity zero, no model, no FDMI
            return 'long', 0.50, 'no_signal'

        _net_score = abs(_votes_long - _votes_short)

        # Minimum vote threshold — too close = skip
        if _cfg.dir_voting_enabled and _net_score < _cfg.dir_min_vote_score:
            return None, 0.50, 'insufficient_votes'

        if _votes_long >= _votes_short:
            _p_long = 0.5 + 0.5 * (_votes_long - _votes_short) / max(1.0, _total_votes)
            _p_long = min(0.90, max(0.51, _p_long))
            _src = '+'.join(_sources[:3]) if _sources else 'vote'
            return 'long', _p_long, f'vote_L({_src})'
        else:
            _p_long = 0.5 - 0.5 * (_votes_short - _votes_long) / max(1.0, _total_votes)
            _p_long = max(0.10, min(0.49, _p_long))
            _src = '+'.join(_sources[:3]) if _sources else 'vote'
            return 'short', 1.0 - _p_long, f'vote_S({_src})'

    # ── LIVE DIRECTION LEARNING (delegated to brain) ─────────────────────

    def learn_direction(self, tid, side: str, pnl: float, hold_bars: int = 0):
        """Delegate to brain.direction_learn() — shared H0/H1 engine."""
        self.brain.direction_learn(tid, side, pnl)
        if hold_bars > 0:
            self.brain.record_hold_bars(tid, side, hold_bars)

    def get_live_dir_bias(self, tid) -> Optional[str]:
        """Check if brain's direction bias has a strong preference.

        Uses actual trade counts (long_n + short_n), not PnL-weighted sums.
        """
        bias = self.brain.get_dir_bias(tid)
        if not bias:
            return None
        _cfg = self.config
        total_trades = bias.get('long_n', 0) + bias.get('short_n', 0)
        if total_trades < _cfg.live_bias_min_trades:
            return None
        long_wr = bias['long_w'] / max(1, bias['long_w'] + bias['long_l'])
        short_wr = bias['short_w'] / max(1, bias['short_w'] + bias['short_l'])
        if long_wr > _cfg.live_bias_winrate_min and long_wr > short_wr + _cfg.live_bias_margin:
            return 'long'
        if short_wr > _cfg.live_bias_winrate_min and short_wr > long_wr + _cfg.live_bias_margin:
            return 'short'
        return None

    # ── UTILITIES ────────────────────────────────────────────────────────

    def reset_gate_stats(self):
        for k in self.gate_stats:
            self.gate_stats[k] = 0

    def get_skip_counts(self) -> dict:
        """Gate skip counts in the format the report expects."""
        return {
            'skip_headroom': (self.gate_stats.get('gate0_skip', 0) +
                              self.gate_stats.get('gate0_noise', 0) +
                              self.gate_stats.get('gate0_r3_struct', 0) +
                              self.gate_stats.get('gate0_r3_snap', 0) +
                              self.gate_stats.get('gate0_r4_nightmare', 0) +
                              self.gate_stats.get('gate0_r4_struct', 0) +
                              self.gate_stats.get('gate0_hurst', 0) +
                              self.gate_stats.get('gate0_momentum', 0) +
                              self.gate_stats.get('gate0_tunnel', 0)),
            'skip_regime': self.gate_stats.get('gate0_regime', 0),
            'skip_session': self.gate_stats.get('gate0_session', 0),
            'skip_depth': self.gate_stats.get('gate0_5_skip', 0),
            'skip_dist': self.gate_stats.get('gate1_skip', 0),
            'skip_brain': self.gate_stats.get('gate2_skip', 0),
            'skip_tf_disagree': self.gate_stats.get('gate2_5_tf_disagree', 0),
            'skip_conviction': self.gate_stats.get('gate3_skip', 0),
            'skip_momentum_align': self.gate_stats.get('gate4_momentum_align', 0),
            'skip_physics_qg': self.gate_stats.get('physics_qg_skip', 0),
            'skip_fdmi_fakeout': self.gate_stats.get('fdmi_fakeout_block', 0),
            'skip_competition': self.gate_stats.get('competition_loser', 0),
            'n_signals_seen': self.gate_stats.get('total_candidates', 0),
            'n_traded': self.gate_stats.get('traded', 0),
            'n_bypass_traded': self.gate_stats.get('bypass_traded', 0),
        }
