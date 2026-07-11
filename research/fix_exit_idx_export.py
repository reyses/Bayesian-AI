import os
import glob
import re

def fix_all_scripts():
    base_dir = r"c:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
    scripts = glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py"))
    
    for script_path in scripts:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # We look for the dictionary returned or appended.
        # It usually looks like:
        # return {
        #    'year': ...,
        #    'day': ...,
        #    'setup': ...,
        #    'event_idx': event_idx,
        #    'hit': hit,
        #    'magnitude': magnitude
        # }
        
        # Or:
        # events.append({ ... })
        
        # Let's find `'event_idx':` and add `'exit_idx': exit_idx,` right after it.
        if "'exit_idx'" in content:
            # Already patched or has it
            pass
            
        # Pattern: `'event_idx': [a-zA-Z0-9_]+,`
        new_content = re.sub(r"('event_idx':\s*[a-zA-Z0-9_]+,)", r"\1\n                'exit_idx': exit_idx,", content)
        
        # But wait, is exit_idx always defined? 
        # Yes, I previously initialized `exit_idx = -1` in all scripts (hopefully).
        # Let's check if exit_idx = -1 exists.
        if 'exit_idx = -1' not in new_content:
            # If not initialized, we initialize it right after event_idx
            new_content = re.sub(r"(event_idx\s*=\s*.*?)\n", r"\1\n    exit_idx = -1\n", new_content, count=1)
            
        if new_content != content:
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"Patched {os.path.basename(script_path)}")

if __name__ == '__main__':
    fix_all_scripts()
