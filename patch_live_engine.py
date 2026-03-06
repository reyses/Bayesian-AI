import re

with open('live/live_engine.py', 'r') as f:
    content = f.read()

# Add init exit engine
init_str = """
        self._belief_network: Optional[TimeframeBeliefNetwork] = None

        from core.exit_engine import ExitEngine
        self._exit_engine = ExitEngine(
            mode='live',
            wave_rider=self._wave_rider,
            tick_size=config.tick_size if hasattr(config, 'tick_size') else 0.25,
            tick_value=config.point_value if hasattr(config, 'point_value') else 0.50,
        )
"""
content = content.replace("self._belief_network: Optional[TimeframeBeliefNetwork] = None", init_str)

# Modify entry inside _check_entry
open_str = """
        self._wave_rider.open_position(
            entry_price=price, side=side,
            state=best_candidate.state,
            stop_distance_ticks=sl_ticks,
            profit_target_ticks=tp_ticks,
            trailing_stop_ticks=trail_ticks,
            trail_activation_ticks=trail_act,
            template_id=best_tid,
        )

        self._pos_state = self._exit_engine.open_position(
            side=side,
            entry_price=price,
            entry_bar_index=self._bar_i,
            template_id=best_tid,
            lib_entry=lib_entry,
        )
"""
content = re.sub(r'self\._wave_rider\.open_position\([\s\S]*?template_id=best_tid,\n\s*\)', open_str, content)

# Also apply it for manual entries and other places
open_str2 = """
        self._wave_rider.open_position(
            entry_price=price, side=side,
            state=state,
            stop_distance_ticks=sl_ticks,
            profit_target_ticks=tp_ticks,
            trailing_stop_ticks=trail_ticks,
            trail_activation_ticks=trail_act,
            template_id='MANUAL',
        )

        self._pos_state = self._exit_engine.open_position(
            side=side,
            entry_price=price,
            entry_bar_index=self._bar_i,
            template_id=-1,
            lib_entry={'atr': self._live_atr_ticks if self._live_atr_ticks > 0 else 8.0},
        )
"""
content = re.sub(r'self\._wave_rider\.open_position\([\s\S]*?template_id=\'MANUAL\',\n\s*\)', open_str2, content)

# Check exit modifications
exit_str = """
        from core.exit_engine import ExitAction
        _band_ctx = None
        if hasattr(self._belief_network, 'get_band_confluence'):
            _band_ctx = self._belief_network.get_band_confluence()

        _last_st = self._last_states[-1]['state'] if self._last_states else None
        _net_force = getattr(_last_st, 'net_force', 0.0) if _last_st else 0.0

        if not hasattr(self, '_pos_state'):
            return

        _exit_result = self._exit_engine.evaluate(
            pos=self._pos_state,
            bar_high=price,
            bar_low=price,
            bar_close=price,
            current_bar_index=self._bar_i,
            band_context=_band_ctx,
            net_force=_net_force,
            worker_beliefs=self._belief_network.get_worker_snapshot() if self._belief_network else None,
            sub_bar_highs=None,
            sub_bar_lows=None,
        )

        # Feed acceleration + envelope tuning to wave_rider for half-life envelope logging
"""
content = re.sub(r'# Feed acceleration \+ envelope tuning to wave_rider for half-life envelope', exit_str, content)

# Adjust result variable
res_str = """
        res = {'should_exit': False}
        if _exit_result.action != ExitAction.HOLD:
            res['should_exit'] = True
            res['exit_reason'] = _exit_result.reason

        if res.get('should_exit', False):
"""
content = re.sub(r'result = self\._wave_rider\.update_trail\([\s\S]*?if result\.get\(\'should_exit\', False\):', res_str, content)

with open('live/live_engine.py', 'w') as f:
    f.write(content)
