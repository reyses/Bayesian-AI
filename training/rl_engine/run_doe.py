import subprocess
import time
import os

def launch_doe():
    print("================================================================")
    print("      PW-CRL: 8-AGENT DESIGN OF EXPERIMENTS (DOE) RUNNER        ")
    print("================================================================")
    print("[INFO] Launching 2 concurrent trials across 4 agent architectures.")
    print("[INFO] Trial A: Learning Rate = 3e-3")
    print("[INFO] Trial B: Learning Rate = 4e-3")
    print("[INFO] Agent Types: NMP, ENTRY_NMP, EXIT_NMP, YOLO")
    
    agent_types = ['NMP', 'ENTRY_NMP', 'EXIT_NMP', 'YOLO']
    trials = [
        {'id': 'TRIAL_A', 'lr': '3e-3'},
        {'id': 'TRIAL_B', 'lr': '4e-3'}
    ]
    
    processes = []
    
    script_path = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/train_historical.py"
    
    for trial in trials:
        for agent in agent_types:
            run_id = f"{trial['id']}_{agent}"
            print(f"[LAUNCH] Starting {run_id} (LR: {trial['lr']})")
            
            # Subprocess launch
            # We redirect stdout to logs so the terminal isn't overwhelmed by 8 simultaneous loops
            log_file = open(f"C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/training/rl_engine/logs_{run_id}.txt", "w")
            
            p = subprocess.Popen(
                ["python", script_path, "--agent-type", agent, "--lr", trial['lr'], "--run-id", run_id, "--max-epochs", "3", "--smoke-test"],
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            processes.append((p, run_id, log_file))
            
            # Stagger launches slightly to prevent HDF5 / File IO collisions on boot
            time.sleep(2)
            
    print(f"\n[INFO] All 8 Agents successfully launched in the background.")
    print(f"[INFO] Monitoring process execution...")
    
    try:
        while True:
            all_done = True
            for p, run_id, _ in processes:
                if p.poll() is None: # Still running
                    all_done = False
                    break
            
            if all_done:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[WARN] Kill signal received. Terminating all 8 agents...")
        for p, _, log_file in processes:
            p.terminate()
            log_file.close()
        return

    print("\n[SUCCESS] 8-Agent DOE Parameter Sweep Completed Successfully!")
    for _, _, log_file in processes:
        log_file.close()

if __name__ == "__main__":
    launch_doe()
