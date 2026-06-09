import subprocess
print("Moving files to DATA/ATLAS on the VM...")
cmd = "gcloud compute ssh bayesian-ai-node --zone=us-west2-a --command=\"mkdir -p ~/DATA/ATLAS && mv ~/FEATURES_5s_v2 ~/DATA/ATLAS/ && mv ~/5s ~/DATA/ATLAS/\""
result = subprocess.run(cmd, shell=True)
if result.returncode == 0:
    print("Done! Files successfully moved.")
else:
    print("Failed to move files.")
