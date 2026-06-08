import os
import time
import requests
import zipfile
import subprocess
import shutil
import argparse
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, required=True, help="IP address of the Mothership (e.g., http://192.168.1.100:5050)")
    args = parser.parse_args()
    
    host = args.host
    if not host.startswith("http"):
        host = f"http://{host}"
    if ":" not in host.replace("http://", ""):
        host = f"{host}:5050"
        
    print(f"==================================================")
    print(f"   DRONE WORKER INITIALIZED")
    print(f"   Target Mothership: {host}")
    print(f"==================================================\n")
    
    while True:
        try:
            print("[DRONE] Pinging mothership for a job...")
            resp = requests.get(f"{host}/get_job", timeout=10)
            data = resp.json()
            
            day = data.get("day")
            if not day:
                print(f"[DRONE] No jobs available right now. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
                
            print(f"\n[DRONE] >>> RECEIVED JOB FOR {day} <<<")
            
            # Step 1: Download Data
            temp_dir = f"drone_temp_{day}"
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            zip_path = os.path.join(temp_dir, f"{day}_data.zip")
            print(f"[DRONE] Downloading 5s data for {day}...")
            
            with requests.get(f"{host}/download/{day}", stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(zip_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
            print(f"[DRONE] Download complete. Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)
                
            os.remove(zip_path)
            
            # Reconstruct the ATLAS root path inside the temp folder
            atlas_root = os.path.abspath(os.path.join(temp_dir, "DATA", "ATLAS")).replace('\\', '/')
            
            # Step 2: Run Stage 1
            print(f"\n[DRONE] Executing Stage 1 (GroupLasso + ElasticNet)...")
            subprocess.run([sys.executable, "research/Regression segments/stage1_speed_pass.py", "--day", day, "--atlas_root", atlas_root], check=True)
            
            # Step 3: Run Stage 2
            print(f"\n[DRONE] Executing Stage 2 (GPU Batched OLS)...")
            subprocess.run([sys.executable, "research/Regression segments/stage2_parallel_chaos.py", "--day", day, "--atlas_root", atlas_root], check=True)
            
            # Step 4: Upload JSON back to Mothership
            final_json = f"artifacts/stage2_segments_{day}.json"
            if not os.path.exists(final_json):
                print(f"[DRONE] ERROR: Expected {final_json} was not generated!")
                time.sleep(10)
                continue
                
            print(f"\n[DRONE] Uploading results to Mothership...")
            with open(final_json, 'rb') as f:
                files = {'file': (f"stage2_segments_{day}.json", f, 'application/json')}
                upload_resp = requests.post(f"{host}/submit/{day}", files=files, timeout=30)
                
            if upload_resp.status_code == 200:
                print(f"[DRONE] Upload Successful! Cleaning up...")
                shutil.rmtree(temp_dir)
                # Keep local artifacts just in case, or delete them
            else:
                print(f"[DRONE] Upload Failed: {upload_resp.text}")
                
            print(f"[DRONE] Job {day} complete.\n")
            
        except requests.exceptions.RequestException as e:
            print(f"[DRONE] Network error connecting to Mothership: {e}")
            time.sleep(10)
        except subprocess.CalledProcessError as e:
            print(f"[DRONE] Execution crashed during Stage 1/2: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"[DRONE] Unexpected error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
