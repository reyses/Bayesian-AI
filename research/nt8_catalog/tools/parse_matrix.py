import os

with open('verifier_output.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

days = {}
current_day = None
current_det = None

for line in lines:
    line = line.strip()
    if line.startswith('--- Verifying'):
        current_day = line.split()[2]
        days[current_day] = {}
        current_det = None
    elif current_day and line.endswith(':'):
        det = line[:-1]
        if det in ['ADX-08', 'ATR-09', 'CROSS-11', 'DOW-19', 'FIB-17']:
            current_det = det
            days[current_day][current_det] = {'native': 0, 'legacy': 0, 'first_n': None, 'first_l': None}
    elif current_day and current_det:
        if line.startswith('Native triggers:'):
            days[current_day][current_det]['native'] = int(line.split(':')[1].strip())
        elif line.startswith('Legacy triggers:'):
            days[current_day][current_det]['legacy'] = int(line.split(':')[1].strip())
        elif line.startswith('First native:'):
            days[current_day][current_det]['first_n'] = line.split('First native:')[1].strip()
        elif line.startswith('First legacy:'):
            days[current_day][current_det]['first_l'] = line.split('First legacy:')[1].strip()

# Aggregate
matrix = {
    'ADX-08': {'match': 0, 'diverge': 0},
    'ATR-09': {'match': 0, 'diverge': 0},
    'CROSS-11': {'match': 0, 'diverge': 0},
    'DOW-19': {'match': 0, 'diverge': 0},
    'FIB-17': {'match': 0, 'diverge': 0},
}

for day, dets in days.items():
    for det, data in dets.items():
        n = data['native']
        l = data['legacy']
        fn = data['first_n']
        fl = data['first_l']
        
        # Determine match
        is_match = (n == l)
        
        if det == 'CROSS-11':
            # For cross 11, legacy only reports 1. If native triggers >= 1 and legacy triggers >= 1 and they match on the first, it's considered matched for the bug exception (legacy 'break' bug)
            if fn and fl and fn == fl:
                is_match = True
            elif n == 0 and l == 0:
                is_match = True
            else:
                is_match = False
                
        if is_match:
            matrix[det]['match'] += 1
        else:
            matrix[det]['diverge'] += 1
            if det == 'ADX-08' and matrix[det]['diverge'] <= 5:
                print(f"ADX-08 Mismatch on {day}: Native={n}, Legacy={l} | {fn} vs {fl}")

print(f"| Detector | Days Matched | Days Diverged | Divergence Reason (if applicable) |")
print(f"|---|---|---|---|")
for det, stats in matrix.items():
    reason = ""
    if det == 'ATR-09':
        reason = "Legacy used `close` instead of `high`/`low` for daily extremes."
    elif det == 'DOW-19':
        reason = "Legacy skipped the first 20 bars of the RTH session."
    elif det == 'FIB-17':
        reason = "Legacy lacked an RTH filter and used `close` for daily extremes."
    elif det == 'CROSS-11' and stats['diverge'] > 0:
        reason = "Legacy loops stopped after the very first cross, missing subsequent ones."
    
    print(f"| {det} | {stats['match']} | {stats['diverge']} | {reason} |")
