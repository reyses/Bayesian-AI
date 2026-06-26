import os
import torch
import numpy as np
import datetime
import pytz
import argparse
import time

from mamba_env import MambaRLTradingEnv
from mamba_rl_network import MambaRLTradingNetwork
from epoch_summary import plot_epoch_summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=str, default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26", help="Comma-separated list of days to run")
    parser.add_argument('--checkpoint', type=str, default="mamba_rl_checkpoint_ep19.pth", help="Path to model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, "FEATURES_5s_v2")
    labels_csv = os.path.join(atlas_root, "regime_labels_2d.csv")
    days = [d.strip() for d in args.days.split(',')]

    env = MambaRLTradingEnv(atlas_root, features_root, labels_csv, days)
    policy_net = MambaRLTradingNetwork().to(device)

    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(f"Loaded checkpoint {checkpoint_path}")
    else:
        print(f"Error: Could not find {checkpoint_path}")
        return

    policy_net.eval()
    state = env.reset()
    done = False
    
    epoch_trades = []
    
    step_count = 0
    start_time = time.time()
    while not done:
        with torch.no_grad():
            s_g = torch.tensor(state[0]).unsqueeze(0).to(device)
            s_l0 = torch.tensor(state[1]).unsqueeze(0).to(device)
            s_ls = torch.tensor(state[2]).unsqueeze(0).to(device)
            
            logits, exp_val = policy_net(s_g, s_l0, s_ls)
            
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()
            exp_val = exp_val.item()

        next_state, reward, done, info = env.step(action, exp_val)
        
        if info.get('trade_closed', False):
            epoch_trades.append({
                'pnl': info['actual_pnl'],
                'duration': info['duration'],
                'exit_ts': info.get('exit_ts', env.current_bar.timestamp)
            })
            
        state = next_state
        step_count += 1
        if step_count % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"Processed {step_count} bars in {elapsed:.2f}s, {len(epoch_trades)} trades so far...")
        
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Collected {len(epoch_trades)} trades in {total_time:.2f} seconds.")
    
    if len(epoch_trades) > 0:
        plot_epoch_summary(19, epoch_trades, "C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55")
        
if __name__ == "__main__":
    main()
