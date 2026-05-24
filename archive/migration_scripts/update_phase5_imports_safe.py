import os
import glob
import re

files = glob.glob('**/*.py', recursive=True) + glob.glob('**/*.md', recursive=True)
for f in files:
    if not os.path.isfile(f): continue
    
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
    except UnicodeDecodeError:
        try:
            with open(f, 'r', encoding='utf-16') as file:
                content = file.read()
        except:
            continue
            
    orig = content
    content = content.replace('tools.', 'tools.')
        
    if content != orig:
        with open(f, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Updated {f}")

print("Phase 5 import rewrites completed.")
