"""
DOJO FORGE - Score Generation
Computes the Win Rate (WR) and calibration metrics for a generation.
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOJO_FORGE_ROOT = os.path.abspath(os.path.join(HERE, '..'))
GATE_STATE_DIR = os.path.join(DOJO_FORGE_ROOT, 'gate_state')

def load_truth(reports_dir, eid):
    truth_path = os.path.join(reports_dir, 'truth', f'{eid}.json')
    with open(truth_path, 'r') as f:
        return json.load(f)

def score_episode(run_id, eid, truth):
    state_path = os.path.join(GATE_STATE_DIR, run_id, f'{eid}.state.json')
    if not os.path.exists(state_path):
        return None
    
    with open(state_path, 'r') as f:
        state = json.load(f)
        
    exit_frame = state.get('exit_frame')
    drift_path = truth['per_minute_forward_drift']
    
    if exit_frame is not None:
        if exit_frame < len(drift_path):
            terminal = drift_path[exit_frame]
        else:
            terminal = drift_path[-1]
    else:
        terminal = drift_path[-1]
        
    is_win = terminal >= 4.0
    is_loss = terminal <= -4.0
    
    return {
        'eid': eid,
        'type': truth['type'],
        'exit_frame': exit_frame,
        'terminal_drift': terminal,
        'is_win': is_win,
        'is_loss': is_loss
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--selection', required=True)
    args = ap.parse_args()

    reports_dir = os.path.dirname(os.path.abspath(args.selection))

    with open(args.selection, 'r') as f:
        data = json.load(f)

    results = []
    missing = 0
    
    for ep in data['episodes']:
        eid = ep['eid']
        truth = load_truth(reports_dir, eid)
        res = score_episode(args.run_id, eid, truth)
        if res is None:
            missing += 1
        else:
            results.append(res)
            
    if not results:
        print("No completed episodes found.")
        return

    df = pd.DataFrame(results)
    
    n_wins = df['is_win'].sum()
    n_losses = df['is_loss'].sum()
    n_total = len(df)
    
    wr = n_wins / n_total if n_total > 0 else 0
    
    print(f"\n--- SCORECARD FOR {args.run_id} ---")
    print(f"Total Episodes Scored: {n_total} (Missing: {missing})")
    print(f"Wins (>= 4pt): {n_wins}")
    print(f"Losses (<=-4pt): {n_losses}")
    print(f"Win Rate (WR): {wr:.2%}\n")
    
    for t, group in df.groupby('type'):
        tw = group['is_win'].sum()
        tl = group['is_loss'].sum()
        twr = tw / len(group) if len(group) > 0 else 0
        print(f"[{t.upper()}] WR: {twr:.2%} ({tw}/{len(group)})")

if __name__ == '__main__':
    main()
