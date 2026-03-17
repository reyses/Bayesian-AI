with open('training/orchestrator_worker.py', 'r') as f:
    lines = f.readlines()
# simple fixes
with open('training/orchestrator_worker.py', 'w') as f:
    for line in lines:
        if "typing.List" in line:
            continue
        f.write(line)
