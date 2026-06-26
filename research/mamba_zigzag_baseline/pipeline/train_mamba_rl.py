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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_mamba_rl():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--days', type=str, default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26", help="Comma-separated list of days to run")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

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
    if os.path.exists("mamba_rl_checkpoint.pth"):
        print("Found existing checkpoint! Loading weights to resume training...")
        state_dict = torch.load("mamba_rl_checkpoint.pth")
        new_state_dict = {}
        for k, v in state_dict.items():
            k_clean = k.replace("_orig_mod.", "")
            new_state_dict[k_clean] = v
        model.load_state_dict(new_state_dict)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    reporter = TelemetryReporter("Mamba_RL_PPO")
    from epoch_summary import plot_epoch_summary

    total_epochs = args.num_episodes
    base_entropy = 0.05
    gamma = 0.99
    
    global_step = 0

    for epoch in range(total_epochs):
        # Curriculum Orchestration: Trigger Phased Friction
        if hasattr(env, 'update_curriculum_state'):
            env.update_curriculum_state(epoch, total_epochs)
            
        state = env.reset()
        
        # Entropy Lock: Flat schedule for first 25%
        progress = epoch / total_epochs
        if progress < 0.25:
            current_entropy_coef = base_entropy
        else:
            decay_factor = 1.0 - ((progress - 0.25) / 0.75)
            current_entropy_coef = base_entropy * max(0.001, decay_factor)

        done = False
        episode_reward = 0.0
        step_count = 0
        epoch_trades = []
        
        while not done:
            # State Unpacking constraint: Expects (v2_grid, l0_feature, ledger_state)
            v2_grid, l0_feature, ledger_state = state
            
            # Add batch dimension and move to device
            v2_grid_t = torch.tensor(v2_grid, dtype=torch.float32).unsqueeze(0).to(device)
            l0_feature_t = torch.tensor(l0_feature, dtype=torch.float32).unsqueeze(0).to(device)
            ledger_state_t = torch.tensor(ledger_state, dtype=torch.float32).unsqueeze(0).to(device)

            # Forward pass through Unblurred Mamba Network
            policy_logits, value = model(v2_grid_t, l0_feature_t, ledger_state_t)
            
            probs = torch.softmax(policy_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            
            # 0.0 passed for Expected Outcome since PPO Critic handles value estimation
            next_state, reward, done, info = env.step(action.item(), 0.0)
            episode_reward += reward
            
            if info.get('trade_closed', False):
                epoch_trades.append({
                    'pnl': info['actual_pnl'], 
                    'duration': info['duration'], 
                    'entry_ts': info['entry_ts'],
                    'exit_ts': info['exit_ts'],
                    'direction': info['direction']
                })
            
            # Fetch next value for Advantage calculation
            if not done and next_state is not None:
                with torch.no_grad():
                    next_v2, next_l0, next_ledger = next_state
                    n_v2_t = torch.tensor(next_v2, dtype=torch.float32).unsqueeze(0).to(device)
                    n_l0_t = torch.tensor(next_l0, dtype=torch.float32).unsqueeze(0).to(device)
                    n_ledg_t = torch.tensor(next_ledger, dtype=torch.float32).unsqueeze(0).to(device)
                    _, next_value = model(n_v2_t, n_l0_t, n_ledg_t)
            else:
                next_value = torch.tensor([[0.0]], device=device)

            # Advantage Calculation (TD Error)
            reward_tensor = torch.tensor([[reward]], device=device, dtype=torch.float32)
            td_target = reward_tensor + (gamma * next_value)
            advantage = td_target - value.detach()
            
            # Loss derivation
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            
            # Actor-Critic Loss formulation
            critic_loss = F.smooth_l1_loss(value, td_target)
            actor_loss = -(log_prob * advantage).mean() - (current_entropy_coef * entropy)
            total_loss = actor_loss + critic_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            
            state = next_state
            step_count += 1
            global_step += 1
            
            if step_count % 100 == 0:
                reporter.update(global_step, total_epochs * 80000,
                                f"Ep {epoch}/{total_epochs} | Rwd: {episode_reward:.2f} | Loss: {total_loss.item():.4f}")

            # VRAM Memory Management (Critical for WSL2 Mamba)
            del v2_grid_t, l0_feature_t, ledger_state_t
            del policy_logits, value, probs, dist, action
            del reward_tensor, td_target, advantage, log_prob, entropy
            del critic_loss, actor_loss, total_loss
            if not done and next_state is not None:
                del n_v2_t, n_l0_t, n_ledg_t, next_value
            torch.cuda.empty_cache()

        print(f"Epoch {epoch} | Reward: {episode_reward:.2f} | Entropy Coef: {current_entropy_coef:.4f}")
        logger.info(f"=== Epoch {epoch} Complete! Reward: {episode_reward:.2f} ===")
        
        plot_epoch_summary(epoch, epoch_trades)
        torch.save(model.state_dict(), "mamba_rl_checkpoint.pth")
        torch.save(model.state_dict(), f"mamba_rl_checkpoint_ep{epoch}.pth")

if __name__ == "__main__":
    train_mamba_rl()
