import os
import glob
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

base_dir = r"C:\Users\reyse\OneDrive\Desktop\Bayesian-AI\research\nt8_catalog\tests"
files = sorted(glob.glob(os.path.join(base_dir, "*", "ag_deepdive_*.py")))

def run_script(f):
    t0 = time.time()
    try:
        # Just run it, capture output. We'll print when done.
        res = subprocess.run(["python", f], cwd=os.path.dirname(f), capture_output=True, text=True, check=True)
        return (f, True, time.time()-t0, res.stdout[-300:])
    except subprocess.CalledProcessError as e:
        return (f, False, time.time()-t0, e.stderr[-300:] + "\n" + e.output[-300:])

if __name__ == '__main__':
    start_time = time.time()
    print(f"Running {len(files)} dossiers in parallel (max_workers=6)...")
    
    with ProcessPoolExecutor(max_workers=6) as executor:
        results = executor.map(run_script, files)
        
        for f, success, duration, out in results:
            name = os.path.basename(os.path.dirname(f))
            if success:
                print(f"[{name}] SUCCESS in {duration:.1f}s")
            else:
                print(f"[{name}] FAILED in {duration:.1f}s:\n{out}")
                
    print(f"\nAll done in {time.time()-start_time:.1f}s.")
