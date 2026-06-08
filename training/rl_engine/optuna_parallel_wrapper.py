import subprocess
import time
import psutil
import os
import sys

MAX_WORKERS = 2
RAM_REQUIRED_MB = 5500  # Give it a safe 5.5GB threshold

def check_resources():
    available_mb = psutil.virtual_memory().available / (1024 * 1024)
    print(f"[WRAPPER] System RAM available: {available_mb:.0f} MB")
    return available_mb > RAM_REQUIRED_MB

def main():
    print("===========================================")
    print(" [OPTUNA PARALLEL WRAPPER ACTIVATED]")
    print(f" Target Workers: {MAX_WORKERS}")
    print("===========================================")
    
    workers = []
    
    script_path = os.path.join(os.path.dirname(__file__), 'optimize_hyperparams.py')
    
    while True:
        # Clean up finished/dead workers
        workers = [w for w in workers if w.poll() is None]
        
        if len(workers) < MAX_WORKERS:
            if check_resources():
                print(f"[WRAPPER] Spawning Worker {len(workers) + 1}...")
                p = subprocess.Popen([sys.executable, script_path])
                workers.append(p)
                print("[WRAPPER] Waiting 10 seconds for worker to launch...")
                time.sleep(10)
            else:
                print(f"[WRAPPER] Insufficient RAM. Operating with {len(workers)} active workers.")
                time.sleep(30)
        else:
            time.sleep(30)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[WRAPPER] Shutting down all workers...")
        for w in workers:
            w.terminate()
