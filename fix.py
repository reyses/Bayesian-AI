with open('training/orchestrator.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if line.strip() == 'print("\\n':
        continue
    if line.strip() == 'Learning direction corrections from oracle...")':
        new_lines.append('            print("\\n  Learning direction corrections from oracle...")\n')
        continue
    new_lines.append(line)

with open('training/orchestrator.py', 'w') as f:
    f.writelines(new_lines)
