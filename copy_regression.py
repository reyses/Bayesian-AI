import subprocess

print("Creating research directory structure on VM...")
subprocess.run('gcloud compute ssh bayesian-ai-node --zone=us-west2-a --command="mkdir -p /home/reyse/Bayesian-AI/research"', shell=True)

print("Copying Regression segments directory to VM...")
# Use explicit forward slashes on the destination to be safe.
cmd = 'gcloud compute scp --recurse "research\\Regression segments" "bayesian-ai-node:/home/reyse/Bayesian-AI/research/" --zone=us-west2-a'
result = subprocess.run(cmd, shell=True)

if result.returncode == 0:
    print("Successfully copied Regression segments to VM.")
else:
    print("Failed to copy Regression segments.")
