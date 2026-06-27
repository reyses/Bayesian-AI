from mamba_rl_network import MambaRLTradingNetwork
from mamba_env import MambaRLTradingEnv
from core_v2.telemetry.reporter import TelemetryReporter
import logging
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import os
import sys
import time
import datetime
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def e_exit_preflight_ram(required_gb=16):
    """E-Exit RAM Pre-flight check (cgroup aware if on Linux)"""
    # Use psutil as baseline
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    
    # Cgroup override if present
    if os.path.exists('/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                limit = int(f.read().strip())
            with open('/sys/fs/cgroup/memory/memory.usage_in_bytes', 'r') as f:
                usage = int(f.read().strip())
            if limit < 1e15: # Not unlimited
                cgroup_avail = (limit - usage) / (1024**3)
                available_gb = min(available_gb, cgroup_avail)
        except Exception:
            pass

    if available_gb < required_gb:
        logger.error(f"[E-EXIT] RAM Pre-flight failed. Required {required_gb}GB, Available {available_gb:.2f}GB")
        sys.exit(88)

def e_exit_vram_check(pct_limit=0.15, absolute_floor_mb=4000):
    """E-Exit VRAM Per-Step Watchdog"""
    if not torch.cuda.is_available():
        return False
    reserved = torch.cuda.memory_reserved(0)
    total = torch.cuda.get_device_properties(0).total_memory
    headroom = total - reserved
    floor_bytes = absolute_floor_mb * 1024 * 1024
    pct_bytes = total * pct_limit
    required_headroom = max(floor_bytes, pct_bytes)
    if headroom < required_headroom:
        logger.error(f"[E-EXIT] VRAM Headroom critical! Required: {required_headroom/1024**2:.0f}MB, Available: {headroom/1024**2:.0f}MB")
        return True
    return False

def per_step_loss_seam(reward, value, next_value, log_prob, entropy, current_entropy_coef, gamma, device):
    """
    PLUGGABLE REWARD SEAM.
    The real policy logic and reward shaping will slot in here later.
    Do NOT entangle this with the TBPTT control flow loop.
    """
    reward_tensor = torch.tensor([[reward]], device=device, dtype=torch.float32)
    td_target = reward_tensor + (gamma * next_value)
    advantage = td_target - value.detach()
    
    critic_loss = F.smooth_l1_loss(value, td_target)
    actor_loss = -(log_prob * advantage).mean() - (current_entropy_coef * entropy)
    return actor_loss + critic_loss

def train_mamba_rl():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--days', type=str, default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26")
    parser.add_argument('--tbptt_window', type=int, default=500, help="N parameter for Fixed-Window TBPTT")
    args = parser.parse_args()

    e_exit_preflight_ram(required_gb=8) # Pre-flight before allocating PyTorch/Env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device} | TBPTT Window: {args.tbptt_window}")

    if os.name == 'posix' and 'microsoft' in os.uname().release.lower():
        atlas_root = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    else:
        atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    
    features_root = os.path.join(atlas_root, "FEATURES_5s_v2")
    labels_csv = os.path.join(atlas_root, "regime_labels_2d.csv")
    days = [d.strip() for d in args.days.split(',')]

    env = MambaRLTradingEnv(
        atlas_root=atlas_root,
        features_root=features_root,
        labels_csv=labels_csv,
        days=days,
        target_pnl_per_trade=10.0,
        seq_len=30
    )

    model = MambaRLTradingNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    if os.path.exists("mamba_rl_checkpoint.pth"):
        checkpoint = torch.load("mamba_rl_checkpoint.pth")
        if 'model' in checkpoint and 'optimizer' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            # Fallback for old state_dict-only checkpoints
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
            model.load_state_dict(new_state_dict)
    reporter = TelemetryReporter("Mamba_RL_PPO")
    from epoch_summary import plot_epoch_summary, plot_learning_curve

    total_epochs = args.num_episodes
    base_entropy = 0.05
    gamma = 0.99
    global_step = 0
    history_rewards, history_mean_pnls, history_mean_entropies = [], [], []
    
    training_start_time = time.time()

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        if hasattr(env, 'update_curriculum_state'):
            env.update_curriculum_state(epoch, total_epochs)
            
        state = env.reset()
        
        progress = epoch / total_epochs
        if progress < 0.25:
            current_entropy_coef = base_entropy
        else:
            decay_factor = 1.0 - ((progress - 0.25) / 0.75)
            current_entropy_coef = base_entropy * max(0.001, decay_factor)

        done = False
        episode_reward = 0.0
        step_count = 0
        epoch_trades, epoch_step_entropies = [], []
        
        # TBPTT State Setup
        hidden_states = None
        window_loss = 0.0
        window_steps = 0
        
        optimizer.zero_grad()

        while not done:
            if e_exit_vram_check():
                logger.error("[E-EXIT] Triggered mid-epoch. Failsafe activated. Landing sequence initiated.")
                torch.save(model.state_dict(), f"mamba_rl_e_exit_failsafe_ep{epoch}.pth")
                torch.cuda.empty_cache()
                sys.exit(88)

            v2_grid, l0_feature, ledger_state, macro_tensor, time_of_day = state
            
            v2_grid_t = torch.nan_to_num(torch.tensor(v2_grid, dtype=torch.float32).unsqueeze(0).to(device), 0)
            l0_feature_t = torch.nan_to_num(torch.tensor(l0_feature, dtype=torch.float32).unsqueeze(0).to(device), 0)
            ledger_state_t = torch.nan_to_num(torch.tensor(ledger_state, dtype=torch.float32).unsqueeze(0).to(device), 0)
            macro_tensor_t = torch.nan_to_num(torch.tensor(macro_tensor, dtype=torch.float32).unsqueeze(0).to(device), 0)
            time_of_day_t = torch.nan_to_num(torch.tensor(time_of_day, dtype=torch.float32).unsqueeze(0).to(device), 0)

            # Forward pass explicitly tracks hidden_states
            policy_logits, value, hidden_states = model(v2_grid_t, l0_feature_t, ledger_state_t, macro_tensor_t, time_of_day_t, hidden_states)
            
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            next_state, reward, done, info = env.step(action.item(), 0.0)
            
            # Session Boundary Reset (Decoupled from 'done')
            if info.get('session_reset', False):
                hidden_states = None
                
            episode_reward += reward
            
            if info.get('trade_closed', False):
                epoch_trades.append({
                    'pnl': info['actual_pnl'], 'duration': info['duration'], 
                    'entry_ts': info['entry_ts'], 'exit_ts': info['exit_ts'], 'direction': info['direction']
                })
            
            if not done and next_state is not None:
                with torch.no_grad():
                    n_v2_t = torch.nan_to_num(torch.tensor(next_state[0], dtype=torch.float32).unsqueeze(0).to(device), 0)
                    n_l0_t = torch.nan_to_num(torch.tensor(next_state[1], dtype=torch.float32).unsqueeze(0).to(device), 0)
                    n_ledg_t = torch.nan_to_num(torch.tensor(next_state[2], dtype=torch.float32).unsqueeze(0).to(device), 0)
                    n_macro_t = torch.nan_to_num(torch.tensor(next_state[3], dtype=torch.float32).unsqueeze(0).to(device), 0)
                    n_tod_t = torch.nan_to_num(torch.tensor(next_state[4], dtype=torch.float32).unsqueeze(0).to(device), 0)
                    _, next_value, _ = model(n_v2_t, n_l0_t, n_ledg_t, n_macro_t, n_tod_t, hidden_states)
            else:
                next_value = torch.tensor([[0.0]], device=device)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            epoch_step_entropies.append(entropy.item())
            
            # Use pluggable seam for loss
            step_loss = per_step_loss_seam(reward, value, next_value, log_prob, entropy, current_entropy_coef, gamma, device)
            window_loss += step_loss
            window_steps += 1
            
            # --- FIXED-WINDOW TBPTT LOGIC ---
            # Also detaching at 22:00 UTC cross-day logic if env signals 'end_of_day' (mocked via done for now)
            if window_steps >= args.tbptt_window or done:
                window_loss = window_loss / window_steps
                window_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                # Detach hidden states strictly at window boundaries!
                if hidden_states is not None:
                    hidden_states = [h.detach() if h is not None else None for h in hidden_states]
                    
                window_loss = 0.0
                window_steps = 0
            
            state = next_state
            step_count += 1
            global_step += 1
            
            if step_count % 100 == 0:
                reporter.update(global_step, total_epochs * 80000,
                                f"Ep {epoch}/{total_epochs} | Rwd: {episode_reward:.2f}")

            # Memory cleanup
            del v2_grid_t, l0_feature_t, ledger_state_t, macro_tensor_t, time_of_day_t
            del policy_logits, value, probs, dist, action
            del log_prob, entropy, step_loss
            if not done and next_state is not None:
                del n_v2_t, n_l0_t, n_ledg_t, n_macro_t, n_tod_t, next_value

        epoch_end_time = time.time()
        print(f"Epoch {epoch} | Reward: {episode_reward:.2f} | Duration: {epoch_end_time - epoch_start_time:.2f}s")
        
        history_rewards.append(episode_reward)
        history_mean_pnls.append(np.mean([t['pnl'] for t in epoch_trades]) if epoch_trades else 0.0)
        history_mean_entropies.append(np.mean(epoch_step_entropies) if epoch_step_entropies else 0.0)
        
        plot_epoch_summary(epoch, epoch_trades)
        plot_learning_curve(history_rewards, history_mean_pnls, history_mean_entropies)
        
        checkpoint_data = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint_data, "mamba_rl_checkpoint.pth")
        torch.save(checkpoint_data, f"mamba_rl_checkpoint_ep{epoch}.pth")
        
        try:
            print(f"TELEGRAM_TRIGGER: epoch_{epoch}_summary.png and mamba_learning_curve.png are ready!")
        except Exception:
            pass
            
    print(f"Training fully complete! Total Duration: {time.time() - training_start_time:.2f}s")

if __name__ == "__main__":
    train_mamba_rl()
