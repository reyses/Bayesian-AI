"""
DOJO FORGE - Run Generation
Executes a generation of dojo forge using forge_harness.py.
"""
import os
import sys
import json
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_FORGE_ROOT = os.path.abspath(os.path.join(HERE, '..'))

def run_chunk(episodes, run_id, run_dir, url=None, model_blob=None):
    cmd = [sys.executable, os.path.join(DOJO_FORGE_ROOT, 'pipeline', 'forge_harness.py')]
    cmd.extend(['--run-id', run_id])
    if url:
        cmd.extend(['--fallback-url', url])
    if model_blob:
        cmd.extend(['--model-blob', model_blob])
    cmd.extend(['--episodes'] + episodes)
    
    print(f"Running chunk of {len(episodes)} episodes...")
    env = os.environ.copy()
    env['DOJO_RUN_DIR'] = run_dir
    subprocess.run(cmd, check=True, env=env)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selection', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--fallback-url', default=None)
    ap.add_argument('--model-blob', default=None)
    ap.add_argument('--workers', type=int, default=1)
    args = ap.parse_args()

    with open(args.selection, 'r') as f:
        data = json.load(f)
    
    run_dir = os.path.dirname(os.path.abspath(args.selection))

    episodes = [ep['eid'] for ep in data['episodes']]
    print(f"Loaded {len(episodes)} from {args.selection}")
    
    if args.workers == 1:
        run_chunk(episodes, args.run_id, run_dir, args.fallback_url, args.model_blob)
    else:
        # Shard episodes
        chunk_size = (len(episodes) + args.workers - 1) // args.workers
        chunks = [episodes[i:i + chunk_size] for i in range(0, len(episodes), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=args.workers) as p:
            for chunk in chunks:
                p.submit(run_chunk, chunk, args.run_id, run_dir, args.fallback_url, args.model_blob)

if __name__ == '__main__':
    main()
