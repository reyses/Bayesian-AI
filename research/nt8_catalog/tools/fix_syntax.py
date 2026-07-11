import glob

for script in glob.glob('tests/*/ag_deepdive_*.py'):
    with open(script, 'r', encoding='utf-8') as f:
        content = f.read()
        
    if "'mae_sigma': mae_sigma\n" in content and "'mfe': mfe" in content:
        # Search for the block where the syntax error is
        # Specifically, when mae_sigma does not have a comma, and the next line has 'mfe': mfe
        lines = content.split('\n')
        out_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.strip() == "'mae_sigma': mae_sigma":
                # Check next line
                if i + 1 < len(lines) and lines[i+1].strip() == "'mfe': mfe,":
                    # Skip the next two lines!
                    out_lines.append(line)
                    i += 3
                    continue
            out_lines.append(line)
            i += 1
            
        with open(script, 'w', encoding='utf-8') as f:
            f.write('\n'.join(out_lines))
            
