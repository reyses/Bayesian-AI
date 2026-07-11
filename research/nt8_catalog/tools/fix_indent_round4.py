import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # The block that causes the problem:
    bad_block = """
        # --- ROUND 2 DEPTH FIX ---
        _trigger_depth = 0.0
        if 'div' in locals() and div is not None: _trigger_depth = abs(div)
        elif 'adx_val' in locals() and adx_val is not None: _trigger_depth = float(adx_val)
        elif 'z' in locals() and z is not None: _trigger_depth = abs(z)
        elif 'z_val' in locals() and z_val is not None: _trigger_depth = abs(z_val)
        elif 'z_score' in locals() and z_score is not None: _trigger_depth = abs(z_score)
        elif 'distance' in locals() and distance is not None: _trigger_depth = abs(distance)
        elif 'gap' in locals() and gap is not None: _trigger_depth = abs(gap)
        elif 'p0' in locals() and 'open_price' in locals(): _trigger_depth = abs(p0 - open_price)
"""
    # Some lines might have extra indentation. Let's just strip everything matching # --- ROUND 2 DEPTH FIX --- down to the end of that block, and re-inject it cleanly.

    def clean_and_inject(match):
        pre_spaces = match.group(1)
        if pre_spaces.endswith("\n"): pre_spaces = pre_spaces.replace("\n", "")
        # The append line determines the correct indent
        append_line = match.group(2)
        indent_len = len(append_line) - len(append_line.lstrip())
        indent_str = " " * indent_len
        
        fixed_block = f"""
{indent_str}# --- ROUND 2 DEPTH FIX ---
{indent_str}_trigger_depth = 0.0
{indent_str}if 'div' in locals() and div is not None: _trigger_depth = abs(div)
{indent_str}elif 'adx_val' in locals() and adx_val is not None: _trigger_depth = float(adx_val)
{indent_str}elif 'z' in locals() and z is not None: _trigger_depth = abs(z)
{indent_str}elif 'z_val' in locals() and z_val is not None: _trigger_depth = abs(z_val)
{indent_str}elif 'z_score' in locals() and z_score is not None: _trigger_depth = abs(z_score)
{indent_str}elif 'distance' in locals() and distance is not None: _trigger_depth = abs(distance)
{indent_str}elif 'gap' in locals() and gap is not None: _trigger_depth = abs(gap)
{indent_str}elif 'p0' in locals() and 'open_price' in locals(): _trigger_depth = abs(p0 - open_price)
{append_line}"""
        return fixed_block

    pattern = r'(\s*)# --- ROUND 2 DEPTH FIX ---.*?(?:_trigger_depth = abs\(p0 - open_price\)\n|p0 - open_price\)\s*\n)\s*(events\.append\({|results\.append\({)'
    new_content = re.sub(pattern, clean_and_inject, content, flags=re.DOTALL)
    
    with open(filepath, 'w') as f:
        f.write(new_content)

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.startswith("ag_deepdive_") and file.endswith(".py"):
                process_file(os.path.join(root, file))
    print("Fixed.")
