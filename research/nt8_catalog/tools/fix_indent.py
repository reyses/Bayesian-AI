import os

injected_code_template = """
        # --- INJECTED MFE/MAE ---
        try:
            _mode_str = str(mode).lower() if 'mode' in locals() else ''
            _setup_val = setup if 'setup' in locals() else 0
            _is_bullish = ('bull' in _mode_str or 'long' in _mode_str or 'buy' in _mode_str or _setup_val == 1)
            _dir = 1 if _is_bullish else -1
            _exit_price_approx = p0 + _dir * magnitude
            _exit_idx = -1
            for _i, _p in enumerate(path):
                if (_dir == 1 and _p >= _exit_price_approx - 0.0001) or (_dir == -1 and _p <= _exit_price_approx + 0.0001):
                    _exit_idx = _i
                    break
            if _exit_idx == -1: _exit_idx = len(path) - 1
            _sub_path = path[:_exit_idx+1]
            if len(_sub_path) > 0:
                if _dir == 1:
                    mfe = max(0.0, np.max(_sub_path) - p0)
                    mae = max(0.0, p0 - np.min(_sub_path))
                else:
                    mfe = max(0.0, p0 - np.min(_sub_path))
                    mae = max(0.0, np.max(_sub_path) - p0)
            else:
                mfe, mae = 0.0, 0.0
        except Exception:
            mfe, mae = 0.0, 0.0
            
        try:
            _idx_var = event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)
            _sigma_val = sigmas[_idx_var] if 'sigmas' in locals() else 1.0
            if np.isnan(_sigma_val) or _sigma_val <= 0: _sigma_val = 1.0
            magnitude_sigma = magnitude / _sigma_val
            mfe_sigma = mfe / _sigma_val
            mae_sigma = mae / _sigma_val
        except Exception:
            magnitude_sigma, mfe_sigma, mae_sigma = magnitude, mfe, mae
        # ------------------------
"""

for script in ['tests/OHLC-01_Prior_Day/ag_deepdive_01_ohlc.py', 'tests/PIVOT-16_Floor_Levels/ag_deepdive_16_pivots.py']:
    with open(script, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    out_lines = []
    in_block = False
    
    for i, line in enumerate(lines):
        if '# --- INJECTED MFE/MAE ---' in line:
            in_block = True
            indent = line[:len(line) - len(line.lstrip())]
            injected = "\n".join([indent + l[8:] if l.startswith("        ") else indent + l for l in injected_code_template.split('\n') if l.strip()])
            out_lines.append(injected + "\n")
            continue
            
        if in_block and '# ------------------------' in line:
            in_block = False
            continue
            
        if in_block:
            continue
            
        out_lines.append(line)
        
    with open(script, 'w', encoding='utf-8') as f:
        f.writelines(out_lines)
