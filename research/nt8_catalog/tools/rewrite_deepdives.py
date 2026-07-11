import os
import glob
import re

def rewrite_scripts():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    scripts = glob.glob(os.path.join(base_dir, '**', 'ag_deepdive_*.py'), recursive=True)
    
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

    for script in scripts:
        if 'archive' in script.lower(): continue
        
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'INJECTED MFE/MAE' in content:
            continue
            
        # We need to find the line `results.append({` or `events_found.append({` or `events.append({`
        # BUT only if it is the ONE that writes `magnitude`!
        # Because some scripts have multiple loops (like event detection loop then evaluation loop).
        # We look for the dictionary that contains `'magnitude': magnitude`
        
        # Regex to find the dictionary block
        # We will split by lines and look for the dict that has 'magnitude': magnitude
        lines = content.split('\n')
        out_lines = []
        i = 0
        in_dict_block = False
        dict_start_idx = -1
        indent = ""
        
        while i < len(lines):
            line = lines[i]
            
            if not in_dict_block and ('results.append({' in line or 'events.append({' in line or 'events_found.append({' in line or 'return {' in line):
                in_dict_block = True
                dict_start_idx = len(out_lines)
                indent = line[:len(line) - len(line.lstrip())]
                out_lines.append(line)
                i += 1
                continue
                
            if in_dict_block:
                if '}' in line and not '{' in line:
                    in_dict_block = False
                
                if "'magnitude': magnitude" in line:
                    # We found the target dictionary!
                    # We must inject our code BEFORE the start of this dictionary block
                    # dict_start_idx is where the append({ or return { started
                    
                    injected = "\n".join([indent + l[8:] if l.startswith("        ") else indent + l for l in injected_code_template.split('\n')])
                    out_lines.insert(dict_start_idx, injected)
                    
                    # Update dict_start_idx offset since we inserted lines
                    # The current line is modified to add new keys
                    if line.rstrip().endswith(','):
                        new_line = line + f"\n{indent}    'mfe': mfe,\n{indent}    'mae': mae,\n{indent}    'magnitude_sigma': magnitude_sigma,\n{indent}    'mfe_sigma': mfe_sigma,\n{indent}    'mae_sigma': mae_sigma"
                    else:
                        new_line = line + f",\n{indent}    'mfe': mfe,\n{indent}    'mae': mae,\n{indent}    'magnitude_sigma': magnitude_sigma,\n{indent}    'mfe_sigma': mfe_sigma,\n{indent}    'mae_sigma': mae_sigma"
                    
                    out_lines.append(new_line)
                    in_dict_block = False # We are done with this block
                else:
                    out_lines.append(line)
            else:
                out_lines.append(line)
                
            i += 1
            
        with open(script, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines))
        print(f"Patched {os.path.basename(script)}")

if __name__ == '__main__':
    rewrite_scripts()
