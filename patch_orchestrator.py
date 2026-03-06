import re

with open('training/orchestrator.py', 'r') as f:
    content = f.read()

# Add import before the loop
import_str = """
            from core.exit_engine import ExitEngine, ExitAction
            _exit_engine = ExitEngine(
                mode='training',
                wave_rider=self.wave_rider,
                tick_size=self.asset.tick_size,
                tick_value=self.asset.point_value,
            )

            # Reset PID analyzer for the day
"""
content = re.sub(r'# Reset PID analyzer for the day', import_str, content)

# Modify position open
open_str = """
                        self.wave_rider.open_position(
                            entry_price=price,
                            side=side,
                            state=best_candidate.state,
                            stop_distance_ticks=_sl_ticks,
                            profit_target_ticks=_tp_ticks,
                            trailing_stop_ticks=_trail_ticks,
                            trail_activation_ticks=_trail_act_ticks,
                            template_id=best_tid
                        )
                        _pos_state = _exit_engine.open_position(
                            side=side,
                            entry_price=price,
                            entry_bar_index=_bar_i,
                            template_id=best_tid,
                            lib_entry=lib_entry,
                        )
"""
content = re.sub(r'self\.wave_rider\.open_position\([\s\S]*?template_id=best_tid\n\s*\)', open_str, content)

# Modify bypass position open
bypass_open_str = """
                            self.wave_rider.open_position(
                                entry_price=price,
                                side=side,
                                state=_bypass_candidate.state,
                                stop_distance_ticks=_bp_sl_ticks,
                                profit_target_ticks=_bp_tp_ticks,
                                template_id=-1,
                            )
                            _pos_state = _exit_engine.open_position(
                                side=side,
                                entry_price=price,
                                entry_bar_index=_bar_i,
                                template_id=-1,
                                lib_entry={'atr': 20.0},
                            )
"""
content = re.sub(r'self\.wave_rider\.open_position\([\s\S]*?template_id=-1,\n\s*\)', bypass_open_str, content)

# Modify evaluate
eval_str = """
                        _band_ctx = None
                        if hasattr(belief_network, 'get_band_confluence'):
                            _band_ctx = belief_network.get_band_confluence()

                        _net_force = getattr(_states_map.get(_bar_i, {}), 'net_force', 0.0)

                        _sub_highs = None
                        _sub_lows = None
                        if _has_1s:
                            _s0 = np.searchsorted(_1s_ts, ts_raw, side='left')
                            _s1 = np.searchsorted(_1s_ts, ts_raw + 15, side='left')
                            _sub_highs = _1s_highs[_s0:_s1].tolist() if _s0 < _s1 else None
                            _sub_lows = _1s_lows[_s0:_s1].tolist() if _s0 < _s1 else None

                        _exit_result = _exit_engine.evaluate(
                            pos=_pos_state,
                            bar_high=getattr(row, 'high', price),
                            bar_low=getattr(row, 'low', price),
                            bar_close=price,
                            current_bar_index=_bar_i,
                            band_context=_band_ctx,
                            net_force=_net_force,
                            worker_beliefs=None,
                            sub_bar_highs=_sub_highs,
                            sub_bar_lows=_sub_lows,
                        )

                        res = {'should_exit': False}
                        if _exit_result.action != ExitAction.HOLD:
                            res['should_exit'] = True
                            res['exit_price'] = _exit_result.exit_price
                            res['exit_reason'] = _exit_result.reason
                            res['pnl'] = _exit_result.pnl_ticks * self.asset.tick_value
                            res['adjustment_reason'] = _exit_result.band_action
"""
content = re.sub(r'# Get exit signal from belief network every bar[\s\S]*?if res\[\'should_exit\'\]:', eval_str + '\n                        if res[\'should_exit\']:', content)

with open('training/orchestrator.py', 'w') as f:
    f.write(content)
