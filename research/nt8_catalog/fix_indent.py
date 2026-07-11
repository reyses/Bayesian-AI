import os
import glob
import re

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py"))

for f in files:
    with open(f, 'r') as file:
        content = file.read()
        
    content = content.replace(
        "        _res_idx = check_idx if 'check_idx' in locals() else _idx_var\n        _depth_val = _idx_var - _res_idx",
        "                _res_idx = check_idx if 'check_idx' in locals() else _idx_var\n                _depth_val = _idx_var - _res_idx"
    )
    
    with open(f, 'w') as file:
        file.write(content)

print("Indentation fixed.")
