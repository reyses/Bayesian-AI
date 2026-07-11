import glob
import os

for script in glob.glob('tests/*/ag_deepdive_*.py'):
    with open(script, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    injected_idx = -1
    mag_idx = -1
    
    for i, line in enumerate(lines):
        if '# --- INJECTED MFE/MAE ---' in line:
            injected_idx = i
        if "'magnitude': magnitude" in line:
            mag_idx = i
            
    if injected_idx != -1 and mag_idx != -1 and (mag_idx - injected_idx) > 50:
        print(f"Fixing {os.path.basename(script)} (injected at {injected_idx}, mag at {mag_idx})")
        
        # Extract block
        block = []
        in_block = False
        new_lines = []
        for line in lines:
            if '# --- INJECTED MFE/MAE ---' in line:
                in_block = True
                block.append(line)
                continue
            if in_block and '# ------------------------' in line:
                in_block = False
                block.append(line)
                continue
            if in_block:
                block.append(line)
                continue
            new_lines.append(line)
            
        # Re-insert block right before the dictionary containing 'magnitude': magnitude
        final_lines = []
        i = 0
        while i < len(new_lines):
            line = new_lines[i]
            if "'magnitude': magnitude" in line:
                # Find start of dict
                j = len(final_lines) - 1
                while j >= 0 and 'results.append({' not in final_lines[j] and 'events_found.append({' not in final_lines[j] and 'events.append({' not in final_lines[j]:
                    j -= 1
                if j == -1:
                    j = len(final_lines) - 1
                
                indent = final_lines[j][:len(final_lines[j]) - len(final_lines[j].lstrip())]
                adjusted_block = []
                for bl in block:
                    adjusted_block.append(indent + bl.lstrip() if bl.strip() else '\n')
                    
                final_lines = final_lines[:j] + adjusted_block + final_lines[j:]
                final_lines.append(line)
            else:
                final_lines.append(line)
            i += 1
            
        with open(script, 'w', encoding='utf-8') as f:
            f.writelines(final_lines)
            
