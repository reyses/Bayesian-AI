import os
import glob
import re

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py"))

for f in files:
    with open(f, 'r') as file:
        content = file.read()
        
    # We want to replace the dictionary passed to results.append({ ... }) or return { ... }
    # Since the dictionary keys are hardcoded, let's find `'mfe': mfe,` and add the new keys there.
    
    if "'resolution_idx':" not in content:
        # Some scripts use 'mfe': mfe,
        pattern = re.compile(r"('mfe': mfe,)")
        
        replacement = r"\1\n        'resolution_idx': _exit_idx if '_exit_idx' in locals() else -1,\n        'depth': (_exit_idx if '_exit_idx' in locals() else -1) - (event_idx if 'event_idx' in locals() else (e_idx if 'e_idx' in locals() else 0)),"
        
        new_content = pattern.sub(replacement, content)
        
        # If it didn't match, maybe the file returns differently. Let's check.
        if new_content == content:
            print(f"Warning: Could not patch {f}")
        else:
            with open(f, 'w') as file:
                file.write(new_content)
            print(f"Patched {os.path.basename(f)}")

print("Done patching.")
