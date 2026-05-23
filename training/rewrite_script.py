import os
import re

filepath = 'training/nightmare_blended.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace _1M_OFFSET + _Z
content = content.replace('feat[_1M_OFFSET + _Z]', 'feat[_1M_Z_IDX]')
# Replace _1M_OFFSET + _VR (we will compute vr instead of reading from feat)
content = content.replace('feat[_1M_OFFSET + _VR]', 'vr')
# Replace _1M_OFFSET + 4
content = content.replace('feat[_1M_OFFSET + 4]', 'feat[_1M_ACCEL_IDX]')

# Replace missing helper indices with local variables computed from v1_compat
content = content.replace('feat[_5M_WICK_IDX]', 'wick_5m')
content = content.replace('feat[_15M_WICK_IDX]', 'wick_15m')
content = content.replace('feat[_1M_WICK_IDX]', 'wick_1m')
content = content.replace('feat[_1M_P_CENTER_IDX]', 'p_center')
content = content.replace('feat[_1M_DMI_IDX]', 'dmi')
content = content.replace('feat[_1M_VOL_REL_IDX]', 'vol_rel')

# Ensure v1_compat is imported
if 'from core_v2 import v1_compat' not in content:
    content = content.replace('from typing import Dict, List', 'from typing import Dict, List\nfrom core_v2 import v1_compat')

# Add history buffers
init_sig = 'self._last_price = 0.0'
if 'self._close_history_1m = []' not in content:
    content = content.replace(init_sig, init_sig + '\n        self._close_history_1m = []\n        self._volume_history_1m = []')

# Inject v1_compat computation in on_state
on_state_start = '''        # Read 1m state
        z = feat[_1M_Z_IDX]
        vr = vr'''
on_state_new = '''        # Read 1m state
        z = feat[_1M_Z_IDX]
        
        # Maintain history for V1 concepts
        if is_1m and state.get('bar_data'):
            bar = state['bar_data']
            self._close_history_1m.append(bar['close'])
            self._volume_history_1m.append(bar['volume'])
            if len(self._close_history_1m) > 60:
                self._close_history_1m.pop(0)
            if len(self._volume_history_1m) > 30:
                self._volume_history_1m.pop(0)
                
        # Derive V1 concepts
        if self._close_history_1m:
            vr = v1_compat.variance_ratio_from_history(np.array(self._close_history_1m))
            vol_rel = v1_compat.vol_rel_from_history(self._volume_history_1m[-1], np.array(self._volume_history_1m), 30)
        else:
            vr = 1.0
            vol_rel = 1.0
            
        p_center = v1_compat.p_at_center_from_z(z)
        dmi = np.sign(feat[_1M_VELOCITY_IDX]) * 5.0
        wick_5m = v1_compat.wick_ratio_from_v2(feat[_5M_BODY_IDX], feat[_5M_BAR_RANGE_IDX])
        wick_15m = v1_compat.wick_ratio_from_v2(feat[_15M_BODY_IDX], feat[_15M_BAR_RANGE_IDX])
        wick_1m = v1_compat.wick_ratio_from_v2(feat[_1M_BODY_IDX], feat[_1M_BAR_RANGE_IDX])'''
content = content.replace(on_state_start, on_state_new)

# _classify_full_tier needs these variables passed to it since it uses them!
# Find definition:
sig1 = 'def _classify_full_tier(self, feat, z):'
new_sig1 = 'def _classify_full_tier(self, feat, z, vr, wick_5m, wick_15m, dmi):'
content = content.replace(sig1, new_sig1)

# Find calls to _classify_full_tier
call1 = 'direction_new, tier_new, flipped_new = self._classify_full_tier(feat, z)'
new_call1 = 'direction_new, tier_new, flipped_new = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)'
content = content.replace(call1, new_call1)
call2 = 'direction, tier, cnn_flipped = self._classify_full_tier(feat, z)'
new_call2 = 'direction, tier, cnn_flipped = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)'
content = content.replace(call2, new_call2)
call3 = '_, tier, cnn_flipped = self._classify_full_tier(feat, z)'
new_call3 = '_, tier, cnn_flipped = self._classify_full_tier(feat, z, vr, wick_5m, wick_15m, dmi)'
content = content.replace(call3, new_call3)

# Update evaluate
eval_start = '''        is_1m = (int(ts) % 60) < 5
        z = feat[_1M_Z_IDX]
        vr = vr'''
eval_new = '''        is_1m = (int(ts) % 60) < 5
        z = feat[_1M_Z_IDX]
        
        # In evaluate (stateless), history should be provided via state.
        # Fallback to 1.0 if not provided (for simple tests).
        vr = state.get('variance_ratio', 1.0)
        wick_5m = v1_compat.wick_ratio_from_v2(feat[_5M_BODY_IDX], feat[_5M_BAR_RANGE_IDX])
        wick_15m = v1_compat.wick_ratio_from_v2(feat[_15M_BODY_IDX], feat[_15M_BAR_RANGE_IDX])
        dmi = np.sign(feat[_1M_VELOCITY_IDX]) * 5.0'''
content = content.replace(eval_start, eval_new)

# _check_exit also needs p_center, vol_rel, wick_1m? It's passed `z, vr` currently.
# Wait, _check_exit does: p_center = feat[_1M_P_CENTER_IDX] which is replaced by p_center. But `p_center` is not defined in `_check_exit`.
# Let's modify _check_exit to accept them.
sig_exit = 'def _check_exit(self, feat, z, vr, pnl):'
new_sig_exit = 'def _check_exit(self, feat, z, vr, pnl, p_center, vol_rel, wick_1m):'
content = content.replace(sig_exit, new_sig_exit)

# Update calls to _check_exit
call_exit1 = 'exit_reason = self._check_exit(feat, z, vr, pnl)'
new_call_exit1 = 'exit_reason = self._check_exit(feat, z, vr, pnl, p_center, vol_rel, wick_1m)'
content = content.replace(call_exit1, new_call_exit1)
call_exit2 = 'exit_reason = self._check_exit(feat, z, vr, cc_pnl)'
new_call_exit2 = 'exit_reason = self._check_exit(feat, z, vr, cc_pnl, p_center, vol_rel, wick_1m)'
content = content.replace(call_exit2, new_call_exit2)

# Update _evaluate_position_exit
sig_eval_exit = 'def _evaluate_position_exit(self, pos, feat, z, vr, price, is_1m):'
new_sig_eval_exit = 'def _evaluate_position_exit(self, pos, feat, z, vr, price, is_1m, p_center, vol_rel, wick_1m):'
content = content.replace(sig_eval_exit, new_sig_eval_exit)

call_eval_exit = '''            new_counters, exit_reason = self._evaluate_position_exit(
                pos, feat, z, vr, price, is_1m
            )'''
new_call_eval_exit = '''            # Compute stateless variables
            p_center = v1_compat.p_at_center_from_z(z)
            vol_rel = state.get('vol_rel', 1.0)
            wick_1m = v1_compat.wick_ratio_from_v2(feat[_1M_BODY_IDX], feat[_1M_BAR_RANGE_IDX])
            new_counters, exit_reason = self._evaluate_position_exit(
                pos, feat, z, vr, price, is_1m, p_center, vol_rel, wick_1m
            )'''
content = content.replace(call_eval_exit, new_call_eval_exit)

# There is also entry_1m['vr'] which needs vr:
content = content.replace("'vr': feat[_1M_OFFSET + _VR]", "'vr': vr")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("done")
