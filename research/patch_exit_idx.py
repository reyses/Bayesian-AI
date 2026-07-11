import os
import glob
import re

tests_dir = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(tests_dir, "**", "ag_deepdive_*.py"), recursive=True)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    modified = False
    
    # Check if event_idx is there, but exit_idx is missing
    if "'event_idx': i," in content and "'exit_idx':" not in content:
        # We need to add exit_idx tracking.
        # Find the path iteration block:
        # e.g., for p in path:
        # e.g., for p, a in zip(path, adx_path):
        
        # Step 1: Initialize exit_idx = -1
        if "hit_target = False\n" in content and "exit_idx = -1\n" not in content:
            content = content.replace("hit_target = False\n", "hit_target = False\n            exit_idx = -1\n")
            modified = True
            
        # Step 2: Replace for p ... loops with enumerate
        # Match 'for p in path:' -> 'for step_idx, p in enumerate(path):'
        content = re.sub(r'for\s+p\s+in\s+path:', 
                         r'for step_idx, p in enumerate(path):', 
                         content)
        
        # Match 'for p, \w+ in zip(path, \w+):' -> 'for step_idx, (p, \w+) in enumerate(zip(path, \w+)):'
        content = re.sub(r'for\s+p,\s+(\w+)\s+in\s+zip\(path,\s+(\w+)\):', 
                         r'for step_idx, (p, \1) in enumerate(zip(path, \2)):', 
                         content)
        
        # Step 3: Insert exit_idx inside loop before break
        # We find 'hit_target = magnitude > 0\n                        break'
        # and insert 'exit_idx = i + 1 + step_idx'
        content = re.sub(r'(hit_target\s*=\s*magnitude\s*>\s*0\n\s*)(break)', 
                         r'\1exit_idx = i + 1 + step_idx\n\g<1>\2', 
                         content)
                         
        content = re.sub(r'(hit_target\s*=\s*magnitude\s*>\s*0\n\s*)(continue)', 
                         r'\1exit_idx = i + 1 + step_idx\n\g<1>\2', 
                         content)

        # Step 4: Insert exit_idx at the end of path if no break happened
        # We find 'if magnitude == 0.0:\n                    magnitude = path\[-1\] - p0\n                    hit_target = magnitude > 0'
        # and insert 'exit_idx = i + len(path)'
        content = re.sub(r'(if magnitude == 0\.0:\s*\n\s*magnitude = path\[-1\][^\n]+\n\s*hit_target = [^\n]+)',
                         r'\1\n                    exit_idx = i + len(path)',
                         content)
                         
        content = re.sub(r'(if magnitude == 0\.0:\s*\n\s*magnitude = p0 - path\[-1\][^\n]+\n\s*hit_target = [^\n]+)',
                         r'\1\n                    exit_idx = i + len(path)',
                         content)
                         
        # Step 5: add 'exit_idx': exit_idx, to events.append
        content = content.replace("'event_idx': i,", "'event_idx': i,\n                'exit_idx': exit_idx,")
        
        if modified:
            with open(file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Patched {os.path.basename(file)}")

print("Done patching.")
