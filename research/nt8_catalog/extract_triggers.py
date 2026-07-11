import glob, os, re

tests_dir = 'c:/Users/reyse/OneDrive/Desktop/Bayesian-AI/research/nt8_catalog/tests'
dirs = sorted(glob.glob(os.path.join(tests_dir, '*')))

res = []
for d in dirs:
    if not os.path.isdir(d): continue
    name = os.path.basename(d)
    scripts = glob.glob(os.path.join(d, 'ag_deepdive_*.py'))
    if not scripts: continue
    script = scripts[0]
    with open(script, 'r') as f:
        text = f.read()
    
    # Try to find what triggers the setup
    trigger_lines = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'setup_triggered =' in line or 'mode =' in line:
            # get the preceding if/elif condition
            for j in range(i, max(-1, i-5), -1):
                if 'if ' in lines[j] and 'triggered_' not in lines[j]:
                    trigger_lines.append(lines[j].strip())
                    break
    
    res.append(f'{name}: {list(set(trigger_lines))}')

print('\n'.join(res))
