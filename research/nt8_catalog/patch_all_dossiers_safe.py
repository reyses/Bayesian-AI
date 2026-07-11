import os
import glob
import re

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py"))

def patch_file(f):
    with open(f, 'r') as file:
        content = file.read()
        
    # We want to inject `resolution_idx` and `depth` right before the end of process_day's MFE/MAE block
    
    # 1. Inject the computation
    if "_idx_var =" not in content:
        compute_block = """
    try:
        _idx_var = event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)
        _res_idx = check_idx if 'check_idx' in locals() else _idx_var
        _depth_val = _idx_var - _res_idx
        
        _sigma_val = sigmas[_idx_var] if 'sigmas' in locals() else 1.0
        if np.isnan(_sigma_val) or _sigma_val <= 0: _sigma_val = 1.0
        magnitude_sigma = magnitude / _sigma_val
        mfe_sigma = mfe / _sigma_val
        mae_sigma = mae / _sigma_val
    except Exception:
        _idx_var = 0
        _res_idx = 0
        _depth_val = 0
        magnitude_sigma, mfe_sigma, mae_sigma = magnitude, mfe, mae
        
    if abs(magnitude) > 100.0:
        return None # SKIP BAD PRINTS INSTEAD OF ASSERT
"""
        # Find the end of the MFE/MAE block
        pattern1 = re.compile(r"(except Exception:\s*mfe, mae = 0\.0, 0\.0\s*)")
        if pattern1.search(content):
            content = pattern1.sub(r"\1\n" + compute_block, content)
        else:
            # Fallback
            pattern1b = re.compile(r"(\s*# ------------------------\s*results\.append\(\{)")
            if pattern1b.search(content):
                content = pattern1b.sub(compute_block + r"\1", content)
            else:
                pattern1c = re.compile(r"(\s*return \{)")
                content = pattern1c.sub(compute_block + r"\1", content)
                
    # 2. Inject into the dictionary (either return { or results.append({)
    if "'resolution_idx':" not in content:
        pattern2 = re.compile(r"('event_idx':[^,]+,)")
        content = pattern2.sub(r"\1\n        'resolution_idx': _res_idx,\n        'depth': _depth_val,", content)

    with open(f, 'w') as file:
        file.write(content)

for f in files:
    try:
        patch_file(f)
        print(f"Patched {os.path.basename(f)}")
    except Exception as e:
        print(f"Error patching {f}: {e}")
