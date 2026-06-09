import subprocess
import hashlib
import os

print("Calculating remote hashes...")
result = subprocess.run(
    'gcloud compute ssh bayesian-ai-node --zone=us-west2-a --command="cd /home/reyse && find FEATURES_5s_v2 5s -type f -exec md5sum {} +"', 
    capture_output=True, text=True, shell=True
)

if result.returncode != 0:
    print("Failed to get remote hashes:", result.stderr)
    exit(1)

remote_hashes = {}
for line in result.stdout.strip().split('\n'):
    line = line.strip()
    if not line: continue
    parts = line.split('  ', 1) # md5sum uses two spaces between hash and file path
    if len(parts) == 2:
        md5, path = parts
        remote_hashes[path.strip()] = md5.strip()

print(f"Got {len(remote_hashes)} remote hashes.")
print("Calculating local hashes and comparing...")

local_base = os.path.join("DATA", "ATLAS")
mismatches = []
missing_local = []
checked = 0

for remote_path, remote_md5 in remote_hashes.items():
    if remote_path.startswith("./"):
        remote_path = remote_path[2:]
        
    local_path = os.path.join(local_base, remote_path.replace("/", os.sep))
    if not os.path.exists(local_path):
        missing_local.append(local_path)
        continue
    
    # Calculate local MD5
    md5_hash = hashlib.md5()
    with open(local_path, "rb") as f:
        for byte_block in iter(lambda: f.read(8192), b""):
            md5_hash.update(byte_block)
    local_md5 = md5_hash.hexdigest()
    
    if local_md5 != remote_md5:
        mismatches.append(remote_path)
    
    checked += 1
    if checked % 500 == 0:
        print(f"Checked {checked} / {len(remote_hashes)} files...")

print(f"\nValidation Complete. Checked {checked} files.")
if mismatches:
    print(f"Found {len(mismatches)} mismatches:")
    for m in mismatches[:10]: print(f" - {m}")
if missing_local:
    print(f"Found {len(missing_local)} files on remote that are missing locally.")
    for m in missing_local[:10]: print(f" - {m}")
if not mismatches and not missing_local:
    print("SUCCESS: All remote files exactly match local files!")
