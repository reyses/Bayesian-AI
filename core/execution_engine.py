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

    Execution flow (matches orchestrator exactly):
      1. Gates 0-2 filter ALL candidates
      2. Score competition picks the BEST passing candidate
      3. Direction cascade + Gate 3 (conviction) run for winner ONLY
      4. If winner rejected by Gate 3, try worker-bypass path
      5. Return ENTER or HOLD

    Caller responsibilities:
    - Feed bars in order via on_bar()
    - Execute returned TradeActions (open/close positions)
    - Call position_opened() after executing an ENTER action
    - Call position_closed() after executing an EXIT action
    """

    _ADX_TREND_CONFIRMATION = 25.0
    _HURST_TREND_CONFIRMATION = 0.6

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
        # Thresholds
        bias_threshold: float = 0.55,
        dmi_threshold: float = 0.0,
        worker_bypass_conviction: float = 0.65,
        # Depth filters
        depth_blacklist: set = None,
        depth_filter_out: set = None,
        depth_only: int = None,
        # Feature extractor
        feature_extractor=None,
    ):
        self.brain = brain
        self.belief_network = belief_network
        self.exit_engine = exit_engine
        self.pattern_library = pattern_library
        self.scaler = scaler
        self.centroids_scaled = centroids_scaled
        self.valid_tids = valid_tids
        self.tick_size = tick_size
        self.point_value = point_value
        self.mode = mode

        self.tier_score_adj = tier_score_adj or {}
        self.depth_score_adj = depth_score_adj or {}
        self.template_tier_map = template_tier_map or {}
        self.exception_tids = exception_tids or set()

        self.bias_threshold = bias_threshold
        self.dmi_threshold = dmi_threshold
        self.gate1_dist = 4.5
        self.worker_bypass_conviction = worker_bypass_conviction

        self.depth_blacklist = depth_blacklist if depth_blacklist is not None else {0, 1, 2}
        self.depth_filter_out = depth_filter_out or set()
        self.depth_only = depth_only

        self.feature_extractor = feature_extractor

        # Oracle-computed gate thresholds (loaded from gate_thresholds.json)
        self.hurst_min = 0.5            # default fallback
        self.tunnel_prob_min = 0.40     # default fallback
        self.momentum_override_ratio = 1.0  # block when mom < rev (ratio < 1.0)
        self._load_gate_thresholds()

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
            'gate0_5_skip': 0,
            'gate1_skip': 0, 'gate2_skip': 0,
            'gate3_skip': 0,
            'gate4_momentum_align': 0,
            'physics_qg_skip': 0,
            'traded': 0, 'bypass_traded': 0,
            'total_candidates': 0,
        }

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
                    if 'tunnel_prob_min' in gt:
                        self.tunnel_prob_min = float(gt['tunnel_prob_min'])
                    if 'momentum_override_ratio' in gt:
                        self.momentum_override_ratio = float(gt['momentum_override_ratio'])
                    print(f"  [ExecutionEngine] Gate thresholds from {path}: "
                          f"hurst>{self.hurst_min} tunnel>{self.tunnel_prob_min} "
                          f"mom_ratio>{self.momentum_override_ratio}")
                    return
                except Exception as e:
                    print(f"  [ExecutionEngine] WARN: failed to load {path}: {e}")

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
        oracle_marker_fn=None,   # callable(raw_event) -> int
        pp_dir_override: str = None,
    ) -> TradeAction:
        """
        Process one bar. Returns a TradeAction.

        oracle_marker_fn: callable that takes a raw PatternEvent and returns
        the effective oracle marker (int). IS mode only. This keeps oracle
        logic in the caller while letting the engine use it for direction.
        """
        if self.in_position:
            return self._check_exit(
                price, bar_high, bar_low, bar_index,
                net_force, sub_bar_highs, sub_bar_lows,
                band_context, exit_signal,
            )

        if candidates:
            return self._check_entry(
                price, bar_index, candidates,
                oracle_marker_fn=oracle_marker_fn,
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
        _ee_lib = {
            'p25_mae': lib_entry.get('p25_mae_ticks', lib_entry.get('p25_mae', 0)),
            'mean_mae': lib_entry.get('mean_mae_ticks', lib_entry.get('mean_mae', 0)),
            'regression_sigma': lib_entry.get('regression_sigma_ticks',
                                              lib_entry.get('regression_sigma', 0)),
            'p75_mfe': lib_entry.get('p75_mfe_ticks', lib_entry.get('p75_mfe', 0)),
            'max_hold_bars': max_hold_bars,
        }
        self.pos_state = self.exit_engine.open_position(
            side=side, entry_price=price, entry_bar_index=bar_index,
            template_id=template_id, lib_entry=_ee_lib,
            network_tp=network_tp,
        )
        if sl_ticks > 0:
            self.pos_state.sl_ticks = float(sl_ticks)
        if tp_ticks > 0:
            self.pos_state.tp_ticks = float(tp_ticks)
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
                    band_context=None, exit_signal=None) -> TradeAction:
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
                     oracle_marker_fn=None, pp_dir_override=None) -> TradeAction:
        """
        Two-phase entry evaluation matching orchestrator:
          Phase 1: Gates 0-2 for ALL candidates → score competition
          Phase 2: Direction + Gate 3 for winner ONLY
          Fallback: Worker bypass if no winner
        """
        candidate_gates = {}    # id(raw) -> gate_label
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
                # Track best Gate 1 reject for worker bypass
                if gr.gate_label == 'gate1' and gr.dist < bypass_dist:
                    bypass_dist = gr.dist
                    bypass_candidate = cand

        # Pick best scorer
        best_raw_id = None
        best_score = 999.0
        best_gr = None
        for raw_id, gr in gate_passers.items():
            if gr.score < best_score:
                best_score = gr.score
                best_raw_id = raw_id
                best_gr = gr

        # Mark score losers
        for raw_id in gate_passers:
            if raw_id != best_raw_id:
                candidate_gates[raw_id] = 'score_loser'

        # ── Phase 2: Direction + Gate 3 for winner ────────────
        if best_gr is not None:
            oracle_marker = None
            if oracle_marker_fn is not None and best_gr.cand.raw_event is not None:
                oracle_marker = oracle_marker_fn(best_gr.cand.raw_event)

            action = self._finalize_entry(
                best_gr, price, bar_index,
                oracle_marker=oracle_marker,
                pp_dir_override=pp_dir_override,
            )

            if action.type == ActionType.ENTER:
                action.candidate_gates = candidate_gates
                return action
            else:
                # Gate 3 rejected the winner
                raw_id = id(best_gr.cand.raw_event) if best_gr.cand.raw_event else id(best_gr.cand)
                candidate_gates[raw_id] = action.gate_label

        # ── Fallback: Worker bypass ───────────────────────────
        if bypass_candidate is not None:
            bypass_action = self._check_worker_bypass(
                bypass_candidate, bypass_dist, price, bar_index)
            if bypass_action.type == ActionType.ENTER:
                bypass_action.candidate_gates = candidate_gates
                return bypass_action

        # Nothing fired
        action = TradeAction(type=ActionType.HOLD)
        action.candidate_gates = candidate_gates
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
        feat_scaled = self.scaler.transform(
            features.reshape(1, -1) if features.ndim == 1 else features)
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

        # ── Gate 0: Headroom & Physics ────────────────────────
        micro_z = abs(cand.z_score)
        micro_pattern = cand.pattern_type
        should_skip = False
        skip_label = 'gate0'

        if not _data_override:
            if not micro_pattern:
                should_skip = True
                skip_label = 'gate0'
            elif micro_z < 0.5:
                should_skip = True
                skip_label = 'gate0_noise'
            elif 0.5 <= micro_z < 2.0:
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
            elif micro_z >= 2.0:
                chain = (getattr(cand.raw_event, 'parent_chain', [])
                         if cand.raw_event else [])
                root_entry = chain[-1] if chain else None
                macro_z = abs(root_entry['z']) if root_entry else 0.0
                headroom = macro_z < 3.0
                if micro_pattern == 'BAND_REVERSAL':
                    if not headroom and micro_z > 3.0:
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
            elif self.tunnel_prob_min > 0 and getattr(_st, 'reversion_probability', 1.0) < self.tunnel_prob_min:
                should_skip = True
                skip_label = 'gate0_tunnel'

        if should_skip:
            _counter = skip_label if skip_label in self.gate_stats else 'gate0_skip'
            self.gate_stats[_counter] += 1
            r = fail(skip_label)
            r.dist = dist
            return r

        # ── Gate 0.5: Depth filter ────────────────────────────
        _cand_depth = cand.depth

        if self.depth_only is not None and _cand_depth != self.depth_only:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        _MIN_TRADE_DEPTH = 3
        if _cand_depth < _MIN_TRADE_DEPTH:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        if _cand_depth in self.depth_blacklist:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        if _cand_depth in self.depth_filter_out:
            self.gate_stats['gate0_5_skip'] += 1
            return fail('gate0_5')

        # ── Gate 1: Cluster distance ──────────────────────────
        if dist >= self.gate1_dist:
            self.gate_stats['gate1_skip'] += 1
            r = fail('gate1')
            r.dist = dist
            return r

        # ── Gate 2: Brain profitability ───────────────────────
        if not self.brain.should_fire(tid, min_prob=0.05, min_conf=0.0):
            self.gate_stats['gate2_skip'] += 1
            r = fail('gate2')
            r.dist = dist
            r.tid = tid
            return r

        # ── Score computation ─────────────────────────────────
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
                        oracle_marker=None,
                        pp_dir_override=None) -> TradeAction:
        """Phase 2: Direction cascade + Gate 3 + sizing for the winning candidate."""
        cand = gr.cand
        tid = gr.tid
        lib_entry = gr.lib_entry
        feat_scaled = gr.feat_scaled

        # ── Direction cascade ─────────────────────────────────
        side, p_long, dir_source = self._direction_cascade(
            cand, tid, lib_entry, oracle_marker, pp_dir_override, feat_scaled)

        if side is None:
            return TradeAction(type=ActionType.HOLD, gate_label='no_direction',
                               raw_event=cand.raw_event)

        # ── Gate 3: Belief conviction ─────────────────────────
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
            if hasattr(_belief, 'predicted_mfe') and _belief.predicted_mfe > 2.0:
                network_tp = max(4, int(round(_belief.predicted_mfe)))

        # ── Gate 4: Momentum alignment ───────────────────────
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

        # ── Exit sizing ───────────────────────────────────────
        sl_ticks, tp_ticks, trail_ticks, trail_act_ticks = self._compute_sizing(
            lib_entry, network_tp, feat_scaled)

        # ── Max hold bars from parent TF ──────────────────────
        chain = (getattr(cand.raw_event, 'parent_chain', None)
                 if cand.raw_event else None) or []
        _HOLD_PARENT_BARS = 5
        parent_tf = chain[0].get('tf', '4h') if chain else str(getattr(cand, 'timeframe', '4h'))
        from training.fractal_discovery_agent import TIMEFRAME_SECONDS
        parent_tf_sec = TIMEFRAME_SECONDS.get(parent_tf, 14400)
        max_hold_bars = max(20, (parent_tf_sec * _HOLD_PARENT_BARS) // 15)

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
        """Gate 1 override: high-conviction belief fires without template match."""
        belief = self.belief_network.get_belief()
        if belief is None or belief.conviction < self.worker_bypass_conviction:
            return TradeAction(type=ActionType.HOLD)

        # Physics quality gate: depth <= 3 and z < 0
        depth = bypass_cand.depth
        z = getattr(bypass_cand, 'z_score',
                    getattr(bypass_cand.state, 'z_score', 0.0))
        if depth > 3 or z >= 0:
            self.gate_stats['physics_qg_skip'] += 1
            return TradeAction(type=ActionType.HOLD, gate_label='physics_qg')

        side = belief.direction
        _st = bypass_cand.state
        _sigma = getattr(_st, 'regression_sigma', 0.0)
        sl_ticks = (max(4, int(round(_sigma / self.tick_size * 1.5)))
                    if _sigma > 0 else 8)
        tp_ticks = (max(8, int(round(belief.predicted_mfe)))
                    if belief.predicted_mfe > 2.0 else 20)

        chain = (getattr(bypass_cand.raw_event, 'parent_chain', None)
                 if bypass_cand.raw_event else None) or []
        parent_tf = (chain[0].get('tf', '4h') if chain
                     else str(getattr(bypass_cand, 'timeframe', '4h')))
        from training.fractal_discovery_agent import TIMEFRAME_SECONDS
        parent_tf_sec = TIMEFRAME_SECONDS.get(parent_tf, 14400)
        max_hold_bars = max(20, (parent_tf_sec * 5) // 15)

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
            trail_ticks=6.0,
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
                        live_scaled: np.ndarray = None) -> Tuple[float, float, float, float]:
        """Compute SL, TP, trail, trail activation from template stats."""
        _reg_sigma = lib_entry.get('regression_sigma_ticks', 0.0)
        _mean_mae = lib_entry.get('mean_mae_ticks', 0.0)
        _p75_mfe = lib_entry.get('p75_mfe_ticks', 0.0)
        _p25_mae = lib_entry.get('p25_mae_ticks', 0.0)
        params = lib_entry.get('params', {})

        # Phase 1: initial hard stop
        if _p25_mae > 2.0:
            sl_ticks = max(4, int(round(_p25_mae * 3.0)))
        elif _mean_mae > 2.0:
            sl_ticks = max(4, int(round(_mean_mae * 2.0)))
        else:
            sl_ticks = params.get('stop_loss_ticks', 20)

        # Phase 2: trailing stop distance
        if _reg_sigma > 2.0:
            trail_ticks = max(2, int(round(_reg_sigma * 1.1)))
        elif _mean_mae > 2.0:
            trail_ticks = max(2, int(round(_mean_mae * 1.1)))
        else:
            trail_ticks = params.get('trailing_stop_ticks', 10)

        # Trail activation
        trail_act_ticks = (max(2, int(round(_p25_mae * 0.3)))
                           if _p25_mae > 2.0 else 0)

        # TP: network > OLS regression > p75 > DOE param
        if network_tp is not None:
            tp_ticks = network_tp
        else:
            _mfe_coeff = lib_entry.get('mfe_coeff')
            if _mfe_coeff is not None and live_scaled is not None:
                _pred_mfe_pts = (np.dot(live_scaled, np.array(_mfe_coeff))
                                 + lib_entry.get('mfe_intercept', 0.0))
                _pred_mfe_ticks = max(0.0, float(_pred_mfe_pts) / self.tick_size)
                if _pred_mfe_ticks > 2.0:
                    tp_ticks = max(4, int(round(_pred_mfe_ticks)))
                elif _p75_mfe > 2.0:
                    tp_ticks = max(4, int(round(_p75_mfe)))
                else:
                    tp_ticks = params.get('take_profit_ticks', 50)
            elif _p75_mfe > 2.0:
                tp_ticks = max(4, int(round(_p75_mfe)))
            else:
                tp_ticks = params.get('take_profit_ticks', 50)

        return float(sl_ticks), float(tp_ticks), float(trail_ticks), float(trail_act_ticks)

    # ── DIRECTION CASCADE ────────────────────────────────────────────────

    def _direction_cascade(self, cand: Candidate, tid, lib_entry: dict,
                           oracle_marker=None, pp_dir_override=None,
                           live_scaled=None) -> Tuple[Optional[str], float, str]:
        """
        Unified direction cascade matching orchestrator priority order.

        Priority order:
          -1  Ping-pong live bias (caller provides override)
           0  Oracle marker (IS only)
         0.5  Signed MFE regression (learned)
           1  Per-cluster logistic regression
         1.5  Brain direction-specific win rate
           2  Template aggregate bias
           3  Multi-TF band confluence
           4  DMI (trend-following)
           5  Velocity fallback
        """
        state = cand.state
        _BIAS_THRESH = self.bias_threshold

        # ── Priority -1: Ping-pong / live direction override ──
        if pp_dir_override is not None:
            return pp_dir_override, 0.65, 'pp_override'

        # ── Priority 0: Oracle marker (IS mode only) ──────────
        if self.mode == 'is' and oracle_marker is not None:
            if oracle_marker > 0:
                return 'long', 0.7, 'oracle'
            elif oracle_marker < 0:
                return 'short', 0.3, 'oracle'

        # ── Priority 0.5: Signed MFE regression ──────────────
        _smfe_coeff = lib_entry.get('signed_mfe_coeff')
        if _smfe_coeff is not None:
            _depth = cand.depth
            _dmi = (getattr(state, 'dmi_plus', 0.0)
                    - getattr(state, 'dmi_minus', 0.0))
            _pred = float(
                np.dot(np.array([[_depth, _dmi]]), np.array(_smfe_coeff))
                + lib_entry.get('signed_mfe_intercept', 0.0)
            )
            if abs(_pred) > 0.5:
                side = 'long' if _pred > 0 else 'short'
                _p = 0.5 + min(0.3, abs(_pred) * 0.1)
                return side, _p if side == 'long' else 1 - _p, 'signed_mfe'

        # ── Priority 1: Per-cluster logistic regression ───────
        _dir_coeff = lib_entry.get('dir_coeff')
        if _dir_coeff is not None and live_scaled is not None:
            _dir_logit = (np.dot(live_scaled, np.array(_dir_coeff))
                          + lib_entry.get('dir_intercept', 0.0))
            _dir_prob = 1.0 / (1.0 + np.exp(-float(_dir_logit)))
            if _dir_prob > _BIAS_THRESH:
                return 'long', _dir_prob, 'logistic'
            elif _dir_prob < (1.0 - _BIAS_THRESH):
                return 'short', _dir_prob, 'logistic'

        # ── Priority 1.5: Brain direction-specific win rate ───
        _dir_long = self.brain.get_dir_probability(tid, 'LONG')
        _dir_short = self.brain.get_dir_probability(tid, 'SHORT')
        if _dir_long is not None and _dir_short is not None:
            if _dir_long > _dir_short + 0.10:
                return 'long', _dir_long, 'brain_dir'
            elif _dir_short > _dir_long + 0.10:
                return 'short', 1.0 - _dir_short, 'brain_dir'

        # ── Priority 2: Template aggregate bias ───────────────
        long_bias = lib_entry.get('long_bias', 0.0)
        short_bias = lib_entry.get('short_bias', 0.0)
        if long_bias >= _BIAS_THRESH:
            return 'long', long_bias, 'template_bias'
        elif short_bias >= _BIAS_THRESH:
            return 'short', 1.0 - short_bias, 'template_bias'
        elif long_bias + short_bias >= 0.10:
            s = 'long' if long_bias >= short_bias else 'short'
            return s, long_bias if s == 'long' else 1.0 - short_bias, 'template_bias'

        # ── Priority 3: Multi-TF band confluence ─────────────
        if hasattr(self.belief_network, 'get_band_confluence'):
            _band = self.belief_network.get_band_confluence()
            if _band is not None and _band.get('direction') is not None:
                return _band['direction'], 0.55, 'band_confluence'

        # ── Priority 4: DMI (trend-following) ─────────────────
        _dmi_diff = (getattr(state, 'dmi_plus', 0.0)
                     - getattr(state, 'dmi_minus', 0.0))
        if abs(_dmi_diff) >= self.dmi_threshold and _dmi_diff != 0:
            s = 'long' if _dmi_diff > 0 else 'short'
            return s, 0.55 if s == 'long' else 0.45, 'dmi'

        # ── Priority 5: Velocity fallback ─────────────────────
        _vel = float(getattr(state, 'velocity', 0.0))
        s = 'long' if _vel >= 0 else 'short'
        return s, 0.52 if s == 'long' else 0.48, 'velocity'

    # ── LIVE DIRECTION LEARNING (delegated to brain) ─────────────────────

    def learn_direction(self, tid, side: str, pnl: float):
        """Delegate to brain.direction_learn() — shared H0/H1 engine."""
        self.brain.direction_learn(tid, side, pnl)

    def get_live_dir_bias(self, tid) -> Optional[str]:
        """Check if brain's direction bias has a strong preference."""
        bias = self.brain.get_dir_bias(tid)
        if not bias:
            return None
        total = sum(bias.values())
        if total < 5:
            return None
        long_wr = bias['long_w'] / max(1, bias['long_w'] + bias['long_l'])
        short_wr = bias['short_w'] / max(1, bias['short_w'] + bias['short_l'])
        if long_wr > 0.60 and long_wr > short_wr + 0.15:
            return 'long'
        if short_wr > 0.60 and short_wr > long_wr + 0.15:
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
            'skip_dist': self.gate_stats.get('gate1_skip', 0),
            'skip_brain': self.gate_stats.get('gate2_skip', 0),
            'skip_conviction': self.gate_stats.get('gate3_skip', 0),
            'skip_physics_qg': self.gate_stats.get('physics_qg_skip', 0),
            'n_signals_seen': self.gate_stats.get('total_candidates', 0),
        }
