import os
import glob

def strip_old():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tests'))
    scripts = glob.glob(os.path.join(base_dir, '**', 'ag_deepdive_*.py'), recursive=True)
    
    for script in scripts:
        with open(script, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if 'INJECTED MFE/MAE CALCULATION' not in content:
            continue
            
        lines = content.split('\n')
        out_lines = []
        in_old_block = False
        
        for line in lines:
            if '# --- INJECTED MFE/MAE CALCULATION ---' in line:
                in_old_block = True
                continue
            if in_old_block and '# ------------------------------------' in line:
                in_old_block = False
                continue
            if in_old_block:
                continue
                
            # Also strip the old 'mfe' and 'mae' keys
            if line.strip().startswith("'mfe': mfe"): continue
            if line.strip().startswith("'mae': mae"): continue
            
            # For ADX-08, 'magnitude': magnitude, might have had a trailing comma
            # We'll just append
            out_lines.append(line)
            
        with open(script, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines))
        print(f"Stripped {os.path.basename(script)}")

if __name__ == '__main__':
    strip_old()
