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
try:
    import torch._inductor.config
    torch._inductor.config.layout_optimization = False
except ImportError:
    pass

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

def a2c_loss_seam(reward, value, next_value, log_prob, entropy, current_entropy_coef, gamma, bce_loss, w_aux, device):
    reward_tensor = torch.tensor([[reward]], device=device, dtype=torch.float32)
    td_target = reward_tensor + (gamma * next_value)
    advantage = td_target - value.detach()
    critic_loss = 0.5 * F.mse_loss(value, td_target)
    actor_loss = -(log_prob * advantage).mean() - (current_entropy_coef * entropy)
    total_loss = actor_loss + critic_loss
    if bce_loss is not None:
        total_loss += w_aux * bce_loss
    return total_loss
def train_mamba_rl():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--days', type=str, default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26")
    parser.add_argument('--tbptt_window', type=int, default=500, help="N parameter for Fixed-Window TBPTT")
    args = parser.parse_args()

    e_exit_preflight_ram(required_gb=1) # Pre-flight before allocating PyTorch/Env

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
        checkpoint = torch.load("mamba_rl_checkpoint.pth", map_location=device, weights_only=False)
        if 'model' in checkpoint and 'optimizer' in checkpoint:
            try:
                missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
                if missing or unexpected:
                    logger.info(f"Checkpoint loaded with strict=False. Missing: {missing}, Unexpected: {unexpected}")
                optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                logger.warning(f"Failed to load checkpoint due to size mismatch: {e}")
        else:
            # Fallback for old state_dict-only checkpoints
            new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
            try:
                missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
                if missing or unexpected:
                    logger.info(f"Old checkpoint loaded with strict=False. Missing: {missing}, Unexpected: {unexpected}")
            except Exception as e:
                logger.warning(f"Failed to load old checkpoint due to size mismatch: {e}")
                
    if sys.platform != "win32" and device.type == 'cuda':
        try:
            model = torch.compile(model)
            logger.info("torch.compile applied successfully.")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
            
    reporter = TelemetryReporter("Mamba_RL_PPO")
    from epoch_summary import plot_epoch_summary, plot_learning_curve

    total_epochs = args.num_episodes
    base_entropy = 0.01
    gamma = 0.99
    global_step = 0
    history_rewards, history_mean_pnls, history_mean_entropies = [], [], []
    
    # Preallocate pinned memory buffers (O(1) H2D transfer per step)
    pinned_buffer = torch.zeros((1, 1, 686), dtype=torch.float32, device='cpu').pin_memory()
    gpu_buffer = torch.zeros((1, 1, 686), dtype=torch.float32, device=device)
    
    next_pinned_buffer = torch.zeros((1, 1, 686), dtype=torch.float32, device='cpu').pin_memory()
    next_gpu_buffer = torch.zeros((1, 1, 686), dtype=torch.float32, device=device)
    
    training_start_time = time.time()

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        
        if hasattr(env, 'update_curriculum_state'):
            env.update_curriculum_state(epoch, total_epochs)
            
        state = env.reset()
        
        progress = epoch / total_epochs
        if progress < 0.50:
            decay_factor = 1.0 - (progress / 0.50)
            current_entropy_coef = 0.001 + (base_entropy - 0.001) * decay_factor
        else:
            current_entropy_coef = 0.001
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
            
            # 1. Extract only the last timestep (L=1)
            v2_last = v2_grid[:, -1, :].reshape(-1) # 416
            l0_last = l0_feature[-1, :] # 1
            ledger_last = ledger_state[-1, :] # 4
            macro_last = macro_tensor[-1, :] # 260
            tod_last = time_of_day[-1, :] # 4
            
            # 2. Pack and transfer O(1)
            packed_np = np.concatenate([v2_last, l0_last, ledger_last, macro_last, tod_last])
            pinned_buffer[0, 0].copy_(torch.from_numpy(packed_np))
            gpu_buffer.copy_(pinned_buffer, non_blocking=True)
            
            # 3. Unpack on GPU
            v2_grid_t = gpu_buffer[:, :, :416].view(1, 1, 8, 52).permute(0, 2, 1, 3) # [1, 8, 1, 52]
            l0_feature_t = gpu_buffer[:, :, 416:417]
            ledger_state_t = gpu_buffer[:, :, 417:421]
            macro_tensor_t = gpu_buffer[:, :, 421:682]
            time_of_day_t = gpu_buffer[:, :, 682:686]
            
            # Forward pass explicitly tracks hidden_states with autocast
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                entry_logits, exit_logits, value, hidden_states = model(v2_grid_t, l0_feature_t, ledger_state_t, macro_tensor_t, time_of_day_t, hidden_states)
            
            # ledger_state is [seq_len, 4]
            is_flat = ledger_state[-1, 0] == 0.0
            if is_flat:
                probs = torch.softmax(entry_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            else:
                exit_prob = torch.sigmoid(exit_logits.squeeze(-1))
                dist = torch.distributions.Bernoulli(probs=exit_prob)
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
                    n_v2_last = next_state[0][:, -1, :].reshape(-1)
                    n_l0_last = next_state[1][-1, :]
                    n_ledger_last = next_state[2][-1, :]
                    n_macro_last = next_state[3][-1, :]
                    n_tod_last = next_state[4][-1, :]
                    
                    n_packed = np.concatenate([n_v2_last, n_l0_last, n_ledger_last, n_macro_last, n_tod_last])
                    next_pinned_buffer[0, 0].copy_(torch.from_numpy(n_packed))
                    next_gpu_buffer.copy_(next_pinned_buffer, non_blocking=True)
                    
                    n_v2_t = next_gpu_buffer[:, :, :416].view(1, 1, 8, 52).permute(0, 2, 1, 3)
                    n_l0_t = next_gpu_buffer[:, :, 416:417]
                    n_ledg_t = next_gpu_buffer[:, :, 417:421]
                    n_macro_t = next_gpu_buffer[:, :, 421:682]
                    n_tod_t = next_gpu_buffer[:, :, 682:686]
                    
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                        _, _, next_value, _ = model(n_v2_t, n_l0_t, n_ledg_t, n_macro_t, n_tod_t, hidden_states)
            else:
                next_value = torch.tensor([[0.0]], device=device)

            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            epoch_step_entropies.append(entropy.item())
            
            # Use pluggable seam for loss
            bce_loss = None
            if not is_flat:
                turn_target = info.get('turn_imminent', 0.0)
                target_t = torch.tensor([1.0 if turn_target else 0.0], device=device, dtype=torch.float32)
                pos_weight = torch.tensor([10.40], device=device, dtype=torch.float32) # Derived from turn bar vs non-turn bar ratio
                bce_loss = F.binary_cross_entropy_with_logits(exit_logits.squeeze(-1), target_t, pos_weight=pos_weight)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                step_loss = a2c_loss_seam(reward, value, next_value, log_prob, entropy, current_entropy_coef, gamma, bce_loss, 0.20, device)
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
                # Detach hidden states for next TBPTT window
                detached_states = []
                for h in hidden_states:
                    if h is None:
                        detached_states.append(None)
                    elif isinstance(h, tuple):
                        detached_states.append((h[0].detach(), h[1].detach()))
                    else:
                        detached_states.append(h.detach())
                hidden_states = detached_states
                    
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
            del entry_logits, exit_logits, value, dist, action
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
    import os
    if os.name == 'nt':
        print("Detected Windows! Auto-respawning in WSL GPU environment...")
        import subprocess
        import sys
        try:
            # Use wsl to run the .venv_wsl python environment
            subprocess.run(["wsl", ".venv_wsl/bin/python", sys.argv[0]] + sys.argv[1:], check=True)
            sys.exit(0)
        except Exception as e:
            print("Failed to auto-respawn in WSL:", e)
            print("Falling back to Windows CPU training...")
    
    train_mamba_rl()
