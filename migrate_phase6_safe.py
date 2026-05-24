import os
import glob
import re

header = "**DEPRECATED — LOOKAHEAD ARTIFACT, DO NOT QUOTE**\n\n"

def prepend_header(filepath):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    if header.strip() not in content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(header + content)

for f in glob.glob('reports/findings/trade_outcome_table/**/*', recursive=True):
    if os.path.isfile(f) and f.endswith(('.txt', '.csv', '.md')):
        prepend_header(f)

for f in glob.glob('reports/**/*', recursive=True):
    if os.path.isfile(f) and f.endswith(('.txt', '.csv', '.md')):
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            if '454' in content or '$454' in content:
                prepend_header(f)

for f in glob.glob('reports/**/*', recursive=True):
    if os.path.isfile(f) and ('forward_pass_1contract' in f or 'forward_pass_full_stack' in f):
        prepend_header(f)

for f in glob.glob('**/*.py', recursive=True):
    if os.path.isfile(f):
        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()
        except UnicodeDecodeError:
            try:
                with open(f, 'r', encoding='utf-16') as file:
                    content = file.read()
            except:
                continue
                
        new_content = re.sub(r'\b[Cc]ausal\b', lambda m: 'Forward pass' if m.group(0).istitle() else 'forward pass', content)
        
        if new_content != content:
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)

print("Phase 6 lookahead purge scripts completed.")
