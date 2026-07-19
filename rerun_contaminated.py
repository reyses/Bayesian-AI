import os
import json
import subprocess
import shutil

DOJO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'research', 'dojo_forge'))
GEN0_DIR = os.path.join(DOJO_ROOT, 'gate_state', 'gen0')
HARNESS = os.path.join(DOJO_ROOT, 'pipeline', 'forge_harness.py')

def main():
    if not os.path.exists('contaminated.json'):
        print("No contaminated.json found.")
        return

    with open('contaminated.json', 'r') as f:
        contaminated = json.load(f)

    if not contaminated:
        print("No episodes were contaminated!")
        return

    print(f"Renaming {len(contaminated)} tainted episodes...")
    for eid in contaminated:
        st_path = os.path.join(GEN0_DIR, f"{eid}.state.json")
        tr_path = os.path.join(GEN0_DIR, f"{eid}.transcript.jsonl")
        
        if os.path.exists(st_path):
            os.rename(st_path, st_path + '.tainted')
        if os.path.exists(tr_path):
            os.rename(tr_path, tr_path + '.tainted')

    print("Re-running contaminated episodes with forge_harness.py...")
    cmd = [
        "python3", HARNESS,
        "--run-id", "gen0",
        "--fallback-url", "http://172.25.112.1:11435/api/chat",
        "--episodes"
    ] + contaminated
    
    subprocess.run(cmd, check=True)
    print("Re-run complete.")

if __name__ == '__main__':
    main()
