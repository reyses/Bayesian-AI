import os
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from mamba_rl_network import MambaRLTradingNetwork
from mamba_env import MambaRLTradingEnv

def build_brain_scanner():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading scanner on {device}...")

    # Load 1 day of textbook setups from Feb 20th for the scanner
    days = ["2024_02_20"] 

    atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    features_root = os.path.join(atlas_root, "FEATURES_5s_v2")
    labels_csv = os.path.join(atlas_root, "regime_labels_2d.csv")

    try:
        env = MambaRLTradingEnv(
            atlas_root=atlas_root,
            features_root=features_root,
            labels_csv=labels_csv,
            days=days,
            target_pnl_per_trade=10.0,
            seq_len=30
        )
    except Exception as e:
        print(f"Environment load failed. You might need to change the test day: {e}")
        return

    net = MambaRLTradingNetwork().to(device)
    
    checkpoint_path = "mamba_rl_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("_orig_mod.", "")] = v
        net.load_state_dict(new_state_dict)
        print("Successfully injected trained weights into scanner.")
    else:
        print("Warning: No checkpoint found, scanning with random weights!")

    net.eval()

    # --- Setup Forward Hooks ---
    spatial_attention_maps = []
    
    def conv2_hook(module, input, output):
        # output is [Batch, 64, Seq, Features]
        # We average across the 64 filters to get a heatmap of structural attention
        heatmap = output.mean(dim=1).squeeze(0).detach().cpu().numpy() # [Seq, Features]
        spatial_attention_maps.append(heatmap)

    hook_handle = net.conv2.register_forward_hook(conv2_hook)

    # --- Scanner Loop ---
    state = env.reset()
    done = False
    
    q_values_history = []
    expected_outcomes = []
    actions = []
    
    print("Executing full brain scan on day 1...")
    step_count = 0
    found_entry = False
    bars_since_entry = 0
    
    while not done and step_count < 20000: # Scan up to 20,000 bars
        grid_np, l0_np, ledger_np = state
        
        # FAST FORWARD: Skip PyTorch inference for the first 10,000 bars to save time
        if step_count < 10000:
            action = 0 # HOLD
            exp_out = 0.0
            # Keep history aligned (we'll just append dummy zeroes)
            q_values_history.append(np.zeros(3))
            expected_outcomes.append(0.0)
            actions.append(action)
            spatial_attention_maps.append(np.zeros((30, 19)))
            
            if step_count % 2000 == 0:
                print(f"Fast forwarding... {step_count}/10000 bars")
        else:
            g_t = torch.tensor(grid_np).unsqueeze(0).to(device)
            l0_t = torch.tensor(l0_np).unsqueeze(0).to(device)
            ledg_t = torch.tensor(ledger_np).unsqueeze(0).to(device)

            with torch.no_grad():
                q_vals, exp_out_tensor = net(g_t, l0_t, ledg_t)
                action = q_vals.argmax(dim=1).item()
                exp_out = exp_out_tensor.item()
                
            q_values_history.append(q_vals.squeeze(0).cpu().numpy())
            expected_outcomes.append(exp_out)
            actions.append(action)

            if step_count % 100 == 0:
                print(f"Scanning PyTorch... {step_count} bars processed")

        if action in [1, 2] and not found_entry:
            found_entry = True
            bars_since_entry = 0
            print(f"Trade Triggered at bar {step_count}! Scanning 50 more bars...")
            
        if found_entry:
            bars_since_entry += 1
            if bars_since_entry >= 50:
                print("Collected 50 post-entry bars. Halting scan.")
                break

        next_state, reward, done, info = env.step(action, exp_out)
        state = next_state
        step_count += 1

    hook_handle.remove()
    print("Scan complete. Generating Interpretability Artifacts...")

    # --- Generate Heatmap GIF ---
    # Find the first entry action (1=Long, 2=Short)
    entry_idx = -1
    for i, a in enumerate(actions):
        if a in [1, 2]:
            entry_idx = i
            break
            
    if entry_idx == -1:
        print("Agent took no trades. Defaulting to first 50 steps for the GIF.")
        entry_idx = 50
        
    start_idx = max(0, entry_idx - 50)
    end_idx = min(len(actions), entry_idx + 50)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    
    print("Generating Mamba Spatial Attention Scan (GIF)...")
    def update(frame_idx):
        ax.clear()
        heatmap = spatial_attention_maps[frame_idx]
        
        # The heatmap is [Seq(30), Features(19)]
        # We transpose it to match the ZigZag visualization [Features, Seq]
        cax = ax.imshow(heatmap.T, aspect='auto', cmap='magma', origin='lower')
        
        action_name = ["HOLD", "LONG", "SHORT", "SCRATCH"][actions[frame_idx]]
        exp_val = expected_outcomes[frame_idx]
        
        color = 'white'
        if actions[frame_idx] == 1: color = 'lime'
        elif actions[frame_idx] == 2: color = 'red'
        
        ax.set_title(f"Brain Scan (Step {frame_idx}) | Action: {action_name} | Expected PnL: ${exp_val:.2f}", color=color)
        ax.set_ylabel("Spatial Features (CNN)")
        ax.set_xlabel("Time (Mamba Sequence)")
        
    anim = FuncAnimation(fig, update, frames=range(start_idx, end_idx), interval=200)
    anim.save('C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/mamba_spatial_attention.gif', writer='pillow')
    plt.close()
    print("Saved spatial attention GIF.")

    # --- Generate Trajectory Plot ---
    q_vals = np.array([list(q) for q in q_values_history])
    exp = np.array([float(e) for e in expected_outcomes])
    
    plt.figure(figsize=(12, 6))
    plt.plot(exp, label="Expected PnL Outcome", color='purple')
    plt.plot(q_vals[:, 1], label="Q-Value (Long)", color='green', alpha=0.5)
    plt.plot(q_vals[:, 2], label="Q-Value (Short)", color='red', alpha=0.5)
    
    for i, a in enumerate(actions):
        if a == 1:
            plt.axvline(x=i, color='green', linestyle='--', alpha=0.5)
        elif a == 2:
            plt.axvline(x=i, color='red', linestyle='--', alpha=0.5)
            
    plt.title("Brain Scanner: Mamba-RL Internal State Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("C:/Users/reyse/.gemini/antigravity/brain/0b405af3-d525-4c87-b71d-cb77ea225a55/mamba_q_trajectories.png", dpi=150)
    plt.close()
    
    print("Saved Q-Trajectory plot.")

    # --- Print Decision Log ---
    print("\n--- DECISION LOG (10 Bars Before to 10 Bars After Entry) ---")
    print(f"{'Bar Offset':<12} | {'HOLD':<8} | {'LONG':<8} | {'SHORT':<8} | {'ACTION':<8} | {'EXPECTED PNL':<12}")
    print("-" * 75)
    
    log_start = max(0, entry_idx - 10)
    log_end = min(len(actions), entry_idx + 10)
    
    for i in range(log_start, log_end):
        q = q_vals[i]
        # Softmax to get probabilities
        exp_q = np.exp(q - np.max(q))
        probs = exp_q / exp_q.sum()
        
        act_name = ["HOLD", "LONG", "SHORT", "SCRATCH"][actions[i]]
        offset = i - entry_idx
        
        marker = ">> ENTRY >>" if offset == 0 else ""
        
        print(f"T{offset:<10+} | {probs[0]:.4f}   | {probs[1]:.4f}   | {probs[2]:.4f}   | {act_name:<8} | ${exp[i]:<10.2f} {marker}")
        
    print("-" * 75)
    print("Scan complete.")

if __name__ == "__main__":
    build_brain_scanner()
