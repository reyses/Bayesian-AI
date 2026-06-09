import subprocess
print("Creating Bayesian-AI directory on VM...")
subprocess.run('gcloud compute ssh bayesian-ai-node --zone=us-west2-a --command="mkdir -p ~/Bayesian-AI"', shell=True)

print("Copying artifacts directory to VM...")
result = subprocess.run('gcloud compute scp --recurse artifacts bayesian-ai-node:/home/reyse/Bayesian-AI/ --zone=us-west2-a', shell=True)

if result.returncode == 0:
    print("Successfully copied artifacts to VM.")
else:
    print("Failed to copy artifacts.")
