import os
import glob
import re

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py"))

def repair(f):
    with open(f, 'r') as file:
        content = file.read()
        
    # Remove all injected 'resolution_idx' and 'depth' lines that are broken
    # For example: "        'resolution_idx': _res_idx,\n        'depth': _depth_val, "
    content = re.sub(r"\s*'resolution_idx': _res_idx,\n\s*'depth': _depth_val,", "", content)
    
    # Also fix the except block indentation error:
    #                 _res_idx = check_idx if 'check_idx' in locals() else _idx_var
    #                 _depth_val = _idx_var - _res_idx
    # which replaced `try: ... except Exception:` incorrectly or caused indentation issues
    
    # Let's just find and remove them all, and re-insert properly
    # Actually, the previous try block injection also broke:
    # "    except Exception:\n        _idx_var = 0\n        _res_idx = 0..."
    
    with open(f, 'w') as file:
        file.write(content)

for f in files:
    repair(f)
