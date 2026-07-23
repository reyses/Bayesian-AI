import subprocess

VM = "bayesian-ai-node"
ZONE = "us-west2-a"

print("Fetching last 100 lines of run_year.log from VM...")
r = subprocess.run(
    f'gcloud compute ssh {VM} --zone={ZONE} --command="tail -n 100 /home/reyse/run_year.log"',
    shell=True, capture_output=True, text=True
)

if r.returncode == 0:
    print(r.stdout)
else:
    print("Failed to fetch log:", r.stderr)
