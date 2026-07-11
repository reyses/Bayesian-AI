import os
import glob
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # The issue is that "        _trigger_depth = 0.0" is injected inside a block that expects 12 spaces in some scripts.
    # We will just replace 8 spaces with 12 spaces for the whole injector block when it's before `            events.append({` (12 spaces)
    # Let's just fix it by standardizing indentation based on the trailing `events.append({` or `results.append({`
    
    # Let's find the block:
    pattern = r'( +)(# --- ROUND 2 DEPTH FIX ---.*?_trigger_depth = abs\(p0 - open_price\)\n)(\s*)(events\.append\({|results\.append\({)'
    
    def replacer(match):
        injector_indent = match.group(1) # currently 8 spaces
        injector_content = match.group(2)
        append_indent = match.group(3)   # this is usually \n        or \n            
        append_statement = match.group(4)
        
        # Determine the target indent from append_indent
        target_indent = append_indent.replace('\n', '')
        
        # Re-indent the injector content
        lines = injector_content.strip().split('\n')
        reindented_lines = []
        for line in lines:
            if line.startswith('        '):
                reindented_lines.append(target_indent + line[8:])
            else:
                reindented_lines.append(target_indent + line.lstrip())
                
        return "\n" + "\n".join(reindented_lines) + "\n" + target_indent + append_statement

    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    # Also fix the orderflow syntax error if it's there
    if "14_orderflow" in filepath:
        # Check if the bar_filter is mis-indented
        # Actually bar_filter might be correctly indented or not. Let's just run it to see.
        pass
        
    with open(filepath, 'w') as f:
        f.write(new_content)

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("ag_deepdive_") and file.endswith(".py"):
                process_file(os.path.join(root, file))
    print("Fixed indentation.")
