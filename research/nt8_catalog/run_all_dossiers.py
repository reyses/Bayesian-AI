import os
import glob
import subprocess
import time

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = sorted(glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py")))

start_time = time.time()
print(f"Running {len(files)} dossiers sequentially...")

for idx, f in enumerate(files):
    print(f"\n[{idx+1}/{len(files)}] Running {os.path.basename(f)}...")
    t0 = time.time()
    try:
        res = subprocess.run(["python", f], cwd=os.path.dirname(f), capture_output=True, text=True, check=True)
        print(f"Success in {time.time()-t0:.1f}s.")
    except subprocess.CalledProcessError as e:
        print(f"FAILED in {time.time()-t0:.1f}s!")
        print("Output:", e.output[-500:])
        print("Error:", e.stderr[-500:])

print(f"\nAll done in {time.time()-start_time:.1f}s.")
