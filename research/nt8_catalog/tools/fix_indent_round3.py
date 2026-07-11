import os

def fix_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if "# --- ROUND 2 DEPTH FIX ---" in line:
            # Found the block
            block = []
            while i < len(lines) and "events.append({" not in lines[i] and "results.append({" not in lines[i]:
                block.append(lines[i])
                i += 1
                
            if i < len(lines):
                append_line = lines[i]
                target_indent = len(append_line) - len(append_line.lstrip())
                
                # Re-indent the block
                for bline in block:
                    if bline.strip() == "":
                        out.append("\n")
                    else:
                        out.append(" " * target_indent + bline.lstrip())
                out.append(append_line)
            else:
                out.extend(block)
        else:
            out.append(line)
        i += 1
        
    with open(filepath, 'w') as f:
        f.writelines(out)
        
if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("ag_deepdive_") and file.endswith(".py"):
                fix_file(os.path.join(root, file))
    print("Fixed.")
