import os
import glob
import re

tests_dir = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = glob.glob(os.path.join(tests_dir, "**", "ag_deepdive_*.py"), recursive=True)

for file in files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "'exit_idx'" in content:
        continue
        
    lines = content.split('\n')
    new_lines = []
    
    for idx, line in enumerate(lines):
        # Insert exit_idx initialization
        if "hit_target = False" in line:
            new_lines.append(line)
            # Match indentation
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(indent + "exit_idx = -1")
            continue
            
        # Detect loops over path
        if "for p in path:" in line:
            line = line.replace("for p in path:", "for step_idx, p in enumerate(path):")
        elif "for p, " in line and "zip(path," in line:
            # e.g. for p, a in zip(path, adx_path):
            line = re.sub(r'for p,\s*(\w+)\s*in\s*zip\(path,\s*(.*?)\):', r'for step_idx, (p, \1) in enumerate(zip(path, \2)):', line)
            
        # Detect hit_target assignments inside a break condition
        if "hit_target = " in line and ("break" in lines[min(idx+1, len(lines)-1)] or "continue" in lines[min(idx+1, len(lines)-1)]):
            indent = line[:len(line) - len(line.lstrip())]
            
            # determine offset name
            offset = "event_idx" if "event_idx =" in content else "i"
            match = re.search(r'(event_idx[a-zA-Z0-9_]*)\s*=', content)
            if match and "i" not in offset:
                offset = match.group(1)
                
            new_lines.append(indent + f"exit_idx = {offset} + 1 + step_idx")
            new_lines.append(line)
            continue
            
        if "hit_target = " in line and "magnitude" in line and "==" in lines[max(idx-1, 0)] and "0.0" in lines[max(idx-1, 0)]:
            indent = line[:len(line) - len(line.lstrip())]
            offset = "event_idx" if "event_idx =" in content else "i"
            new_lines.append(indent + f"exit_idx = {offset} + len(path)")
            new_lines.append(line)
            continue

        if "events.append(" in line:
            if "'event_idx'" in line:
                line = line.replace("'event_idx'", "'exit_idx': exit_idx, 'event_idx'")
                
        new_lines.append(line)
        
    with open(file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    print(f"Patched {os.path.basename(file)}")
