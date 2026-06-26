import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pytz

from research.mamba_zigzag_baseline.pipeline.mamba_env import MambaRLTradingEnv
from research.mamba_zigzag_baseline.pipeline.mamba_rl_network import MambaRLTradingNetwork

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, "FEATURES_5s_v2")
    labels_csv = os.path.join(atlas_root, "regime_labels_2d.csv")
    days = ["2024_02_20", "2024_02_21", "2024_02_22", "2024_02_23", "2024_02_26"]

    env = MambaRLTradingEnv(atlas_root, features_root, labels_csv, days)
    policy_net = MambaRLTradingNetwork().to(device)

    checkpoint_path = "mamba_rl_checkpoint_ep19.pth"
    if os.path.exists(checkpoint_path):
        policy_net.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint {checkpoint_path}")
    else:
        print(f"Error: Could not find {checkpoint_path}")
        return

    policy_net.eval()
    state = env.reset()
    done = False
    
    trades = []
    
    # We will need the current_bar timestamp right before we step
    while not done:
        # Get start time of the bar to record when the trade was entered
        bar_ts = env.current_bar.timestamp
        dt = datetime.datetime.fromtimestamp(bar_ts, tz=datetime.timezone.utc)
        ct = dt.astimezone(pytz.timezone('US/Central'))
        tod = ct.hour + ct.minute / 60.0 + ct.second / 3600.0

        s_g = torch.tensor(state[0]).unsqueeze(0).to(device)
        s_l0 = torch.tensor(state[1]).unsqueeze(0).to(device)
        s_ledg = torch.tensor(state[2]).unsqueeze(0).to(device)

        with torch.no_grad():
            q_vals, expected_outcome = policy_net(s_g, s_l0, s_ledg)
            action = q_vals.argmax(1).item()
            exp_val = expected_outcome.item()

        next_state, reward, done, info = env.step(action, exp_val)
        
        if info.get('trade_closed', False):
            # We record the time of day when it was ENTERED or EXITED? The user wants time of day. 
            # We will use the time of day of the EXIT, or we can use the duration. 
            # Let's record both! But we plot based on EXIT time of day.
            trades.append({
                'pnl': info['actual_pnl'],
                'tod': tod,
                'duration': info['duration']
            })
            
        state = next_state
        
    print(f"Collected {len(trades)} trades.")
    
    if len(trades) == 0:
        print("No trades executed.")
        return
        
    tods = [t['tod'] for t in trades]
    pnls = [t['pnl'] for t in trades]
    colors = ['green' if p >= 0 else 'red' for p in pnls]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(tods, pnls, c=colors, alpha=0.6, edgecolors='none')
    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    
    plt.title('Trade PnL vs Time of Day (Central Time)')
    plt.xlabel('Time of Day (Hours)')
    plt.ylabel('Net PnL ($)')
    plt.xlim(0, 24)
    
    # Add vertical lines for market open and close
    plt.axvline(17, color='blue', linestyle=':', label='Market Reopen (17:00 CT)')
    plt.axvline(16, color='blue', linestyle=':', label='Market Halt (16:00 CT)')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/epoch_19_tod_scatter.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved scatter plot to {out_path}")

if __name__ == "__main__":
    main()
