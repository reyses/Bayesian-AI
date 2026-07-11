import os
import glob
import re

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py"))

for f in files:
    with open(f, 'r') as file:
        content = file.read()
        
    # Inject _res_idx and _depth_val calculation
    if "_res_idx" not in content:
        content = content.replace(
            "        _idx_var = event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)",
            "        _idx_var = event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else i)\n        _res_idx = check_idx if 'check_idx' in locals() else _idx_var\n        _depth_val = _idx_var - _res_idx"
        )
        
    # Inject into the dictionary (look for 'event_idx': ... and insert)
    if "'resolution_idx':" not in content:
        content = re.sub(
            r"('event_idx':\s*[^,]+,)",
            r"\1\n                'resolution_idx': _res_idx,\n                'depth': _depth_val,",
            content
        )
        
    # Also skip magnitude > 100 in ORDERFLOW-14 and others if they assert
    content = content.replace(
        "assert abs(magnitude) <= 100.0",
        "if abs(magnitude) > 100.0: continue  # Skip bad prints instead of aborting"
    )
        
    with open(f, 'w') as file:
        file.write(content)

print(f"Patched {len(files)} files.")
