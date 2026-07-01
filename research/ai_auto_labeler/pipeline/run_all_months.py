import os
import subprocess

months = ['2024_01', '2024_02', '2024_03', '2024_04', '2024_05', '2024_06', '2024_07', '2024_08', 
          '2024_09', '2024_10', '2024_11', '2024_12', '2025_01', '2025_02', '2025_03', '2025_04', 
          '2025_05', '2025_06', '2025_07', '2025_08', '2025_09', '2025_10', '2025_11', '2025_12', 
          '2026_01', '2026_02', '2026_03']

script_path = r"C:\Users\reyse\.gemini\antigravity\brain\0b405af3-d525-4c87-b71d-cb77ea225a55\scratch\ai_labeler.py"

total_trades = 0
for m in months:
    print(f"\nProcessing {m}...")
    result = subprocess.run(['python', script_path, '--month', m, '--prominence', '7.0'], capture_output=True, text=True)
    out = result.stdout
    print(out.strip())
    
    # parse "Generated X trades"
    for line in out.split('\n'):
        if "Generated" in line and "trades" in line:
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                total_trades += int(parts[1])

print(f"\n--- LABELING COMPLETE ---")
print(f"Total optimal trades generated across all months: {total_trades}")
