with open('tests/FIB-17_Confluence/ag_deepdive_17_fib.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

out_lines = []
injected_block = []
in_injected = False

for line in lines:
    if '# --- INJECTED MFE/MAE ---' in line:
        in_injected = True
        injected_block.append(line)
        continue
    if in_injected and '# ------------------------' in line:
        in_injected = False
        injected_block.append(line)
        continue
    if in_injected:
        injected_block.append(line)
        continue
    out_lines.append(line)

# Now find the correct dict to insert injected_block
final_lines = []
i = 0
while i < len(out_lines):
    line = out_lines[i]
    if "'magnitude': magnitude," in line:
        # Step back to find 'results.append({'
        j = len(final_lines) - 1
        while j >= 0 and 'results.append({' not in final_lines[j]:
            j -= 1
        
        # Insert injected block before j
        indent = final_lines[j][:len(final_lines[j]) - len(final_lines[j].lstrip())]
        adjusted_block = []
        for bl in injected_block:
            adjusted_block.append(indent + bl[24:] if bl.startswith('                        ') else bl)
            
        final_lines = final_lines[:j] + adjusted_block + final_lines[j:]
        final_lines.append(line)
    else:
        final_lines.append(line)
    i += 1
    
with open('tests/FIB-17_Confluence/ag_deepdive_17_fib.py', 'w', encoding='utf-8') as f:
    f.writelines(final_lines)
