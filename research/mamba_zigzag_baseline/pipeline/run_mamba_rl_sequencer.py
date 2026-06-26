import os
import sys
import subprocess
import time

def run_sequencer():
    # GUI spawn disabled
    # print("Spawning Telemetry GUI Viewer...", flush=True)
    # subprocess.Popen(
    #     [sys.executable, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'core_v2', 'telemetry', 'gui.py'))], 
    #     creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
    # )
    # time.sleep(2)
    
    num_episodes = 25
    start_episode = 1
    
    # Check what the highest checkpoint is
    for ep in range(num_episodes - 1, -1, -1):
        if os.path.exists(f"mamba_rl_checkpoint_ep{ep}.pth"):
            start_episode = ep + 1
            print(f"Found checkpoint for episode {ep}. Resuming from episode {start_episode}.", flush=True)
            break
            
    print(f"Starting sequencer from episode {start_episode} to {num_episodes-1}...", flush=True)
    
    for episode in range(start_episode, num_episodes):
        print(f"\n[{time.strftime('%H:%M:%S')}] Sequencer launching worker for Episode {episode}...", flush=True)
        
        result = subprocess.run([sys.executable, "-u", "train_mamba_rl.py", "--episode", str(episode), "--num_episodes", str(num_episodes)])
        
        if result.returncode != 0:
            print(f"Worker for episode {episode} failed with exit code {result.returncode}. Stopping sequencer.", flush=True)
            break
            
    print("\nSequencer finished!", flush=True)

if __name__ == "__main__":
    run_sequencer()
