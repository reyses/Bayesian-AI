import os
import re

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find the indent of events.append({ or results.append({
    match = re.search(r'\n( +)(events\.append\({|results\.append\({)', content)
    if not match: return
    target_indent = match.group(1)
    
    # We want to replace any indent before # --- ROUND 2 DEPTH FIX --- and all subsequent lines until the append
    
    lines = content.split('\n')
    out = []
    in_block = False
    for line in lines:
        if "# --- ROUND 2 DEPTH FIX ---" in line:
            in_block = True
            out.append(target_indent + "# --- ROUND 2 DEPTH FIX ---")
            continue
            
        if in_block:
            if "events.append({" in line or "results.append({" in line:
                in_block = False
                out.append(line) # append line keeps its indent
            else:
                if line.strip() != "":
                    out.append(target_indent + line.lstrip())
                else:
                    out.append(line)
        else:
            out.append(line)
            
    with open(filepath, 'w') as f:
        f.write('\n'.join(out))

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("ag_deepdive_") and file.endswith(".py"):
                fix_file(os.path.join(root, file))
    print("Fixed round 5.")
