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

# All .pth artifacts go to the sanctioned repo-root checkpoints/ (gitignored),
# resolved file-relative so no cwd ever gets checkpoint droppings.
_CKPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'checkpoints'))
os.makedirs(_CKPT_DIR, exist_ok=True)
try:
    import torch._inductor.config
    torch._inductor.config.layout_optimization = False
    if os.name == 'nt':
        # Windows-only workaround; under WSL it needlessly serializes inductor
        # compilation (minutes of extra --compile warmup)
        torch._inductor.config.compile_threads = 1
    torch._inductor.config.fx_graph_cache = True
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

_TOTAL_VRAM_BYTES = None

def e_exit_vram_check(pct_limit=0.15, absolute_floor_mb=4000):
    """E-Exit VRAM Per-Step Watchdog"""
    global _TOTAL_VRAM_BYTES
    if not torch.cuda.is_available():
        return False
    reserved = torch.cuda.memory_reserved(0)
    if _TOTAL_VRAM_BYTES is None:
        _TOTAL_VRAM_BYTES = torch.cuda.get_device_properties(0).total_memory
    total = _TOTAL_VRAM_BYTES
    headroom = total - reserved
    floor_bytes = absolute_floor_mb * 1024 * 1024
    pct_bytes = total * pct_limit
    required_headroom = max(floor_bytes, pct_bytes)
    if headroom < required_headroom:
        logger.error(f"[E-EXIT] VRAM Headroom critical! Required: {required_headroom/1024**2:.0f}MB, Available: {headroom/1024**2:.0f}MB")
        return True
    return False

def a2c_loss_seam(reward, value, next_value, log_prob, entropy, current_entropy_coef, gamma, bce_loss, w_aux, device):
    # torch.full embeds the scalar in the kernel launch (no pageable H2D memcpy).
    # Must be a FRESH tensor each step: it lives in the TBPTT graph until backward.
    reward_tensor = torch.full((1, 1), reward, device=device, dtype=torch.float32)
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
    # --- Perf instrumentation (runtime-only; all inert unless explicitly passed) ---
    parser.add_argument('--seed', type=int, default=None, help='Fix torch/numpy RNG (reproducible parity runs)')
    parser.add_argument('--max-steps', type=int, default=0, help='Stop after N env steps (0 = full run); skips plots/saves')
    parser.add_argument('--no-checkpoint', action='store_true', help='Skip checkpoint load/save (fresh deterministic init)')
    parser.add_argument('--compile', action='store_true',
                        help='Opt-in torch.compile (default mode). OFF by default: measured '
                             'loss drift vs eager is ~1.5e-3 (bf16 refusion), over the 1e-4 '
                             'parity gate; speed win ~+20%% (49-53 -> 63 bars/s same-sweep).')
    parser.add_argument('--profile-dir', type=str, default='', help='Write torch.profiler op tables to this dir')
    parser.add_argument('--profile-steps', type=int, default=500, help='Steps inside the profiler window')
    parser.add_argument('--loss-dump', type=str, default='', help='Write per-step loss/action/reward .npz for parity checks')
    parser.add_argument('--perf-warmup', type=int, default=300, help='Steps excluded from bars/sec timing (compile warmup)')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

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
    start_epoch = 0

    _ckpt_main = os.path.join(_CKPT_DIR, "mamba_rl_checkpoint.pth")
    if not args.no_checkpoint and os.path.exists(_ckpt_main):
        checkpoint = torch.load(_ckpt_main, map_location=device, weights_only=False)
        if 'model' in checkpoint and 'optimizer' in checkpoint:
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
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
                
    if args.compile and sys.platform != "win32" and device.type == 'cuda':
        try:
            # Default mode (inductor fusion, NO CUDA graphs). reduce-overhead crashes
            # here: cudagraph output tensors (hidden_states, value) are carried across
            # steps and get overwritten when the same graph re-runs for the no_grad
            # next_value forward ("accessing tensor output of CUDAGraphs that has been
            # overwritten by a subsequent run").
            model = torch.compile(model)
            logger.info("torch.compile applied (default mode; cudagraphs unsafe with carried hidden state).")
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

    # Loop-invariant GPU constant (read-only in the graph, safe to hoist)
    pos_weight_gpu = torch.tensor([10.40], device=device, dtype=torch.float32)  # Derived from turn bar vs non-turn bar ratio

    training_start_time = time.time()

    # --- Perf instrumentation state (inert without flags) ---
    perf_t0 = None
    perf_step0 = 0
    prof = None
    prof_start_step = None
    parity_losses, parity_actions, parity_rewards = [], [], []
    stop_training = False

    for epoch in range(start_epoch, total_epochs):
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
        entropy_buf = []  # detached GPU scalars; synced once per TBPTT window, not per step
        
        # TBPTT State Setup
        hidden_states = None
        window_loss = 0.0
        window_steps = 0
        # One-step-deferred loss pieces: bar t's bootstrap V(s_{t+1}) is exactly the
        # NEXT iteration's critic output (same weights/hidden/observation), so the
        # dedicated no_grad re-forward is redundant except at window close / episode
        # end. Bit-exact: no optimizer step or hidden detach ever lands between
        # stash and consume (deferral is skipped on closing steps).
        pending_loss = None
        
        optimizer.zero_grad()

        while not done:
            if e_exit_vram_check():
                logger.error("[E-EXIT] Triggered mid-epoch. Failsafe activated. Landing sequence initiated.")
                torch.save(model.state_dict(),
                           os.path.join(_CKPT_DIR, f"mamba_rl_e_exit_failsafe_ep{epoch}.pth"))
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
                if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                    torch.compiler.cudagraph_mark_step_begin()
                entry_logits, exit_logits, value, hidden_states = model(v2_grid_t, l0_feature_t, ledger_state_t, macro_tensor_t, time_of_day_t, hidden_states)

            # Complete the previous bar's deferred loss: this forward's critic output
            # IS its bootstrap value (detached, as the no_grad re-forward's was).
            if pending_loss is not None:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    step_loss = a2c_loss_seam(pending_loss['reward'], pending_loss['value'],
                                              value.detach(), pending_loss['log_prob'],
                                              pending_loss['entropy'], current_entropy_coef,
                                              gamma, pending_loss['bce_loss'], 0.20, device)
                window_loss += step_loss
                window_steps += 1
                pending_loss = None
                if args.loss_dump:
                    parity_losses.append(float(step_loss.detach()))

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
            
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().mean()
            # Defer the GPU->CPU sync: buffer detached entropies, flush at TBPTT boundary
            entropy_buf.append(entropy.detach())

            # Use pluggable seam for loss
            bce_loss = None
            if not is_flat:
                turn_target = info.get('turn_imminent', 0.0)
                # Fresh tensor per step (lives in the TBPTT graph); torch.full avoids the H2D memcpy
                target_t = torch.full((1,), 1.0 if turn_target else 0.0, device=device, dtype=torch.float32)
                bce_loss = F.binary_cross_entropy_with_logits(exit_logits.squeeze(-1), target_t, pos_weight=pos_weight_gpu)

            # This bar's loss needs V(s_{t+1}). Normally that is the NEXT iteration's
            # forward (deferred path — saves the duplicate no_grad forward). It must be
            # formed NOW only when (a) the window closes on this loss, because the
            # optimizer step below would otherwise change the weights behind the
            # bootstrap, or (b) the episode ends, where the bootstrap is zero.
            will_close_window = (window_steps == args.tbptt_window - 1)
            if done or next_state is None or will_close_window:
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

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == 'cuda')):
                    step_loss = a2c_loss_seam(reward, value, next_value, log_prob, entropy, current_entropy_coef, gamma, bce_loss, 0.20, device)
                window_loss += step_loss
                window_steps += 1
                if args.loss_dump:
                    parity_losses.append(float(step_loss.detach()))
            else:
                pending_loss = {'reward': reward, 'value': value, 'log_prob': log_prob,
                                'entropy': entropy, 'bce_loss': bce_loss}

            # --- FIXED-WINDOW TBPTT LOGIC ---
            # Also detaching at 22:00 UTC cross-day logic if env signals 'end_of_day' (mocked via done for now)
            if window_steps >= args.tbptt_window or done:
                window_loss = window_loss / window_steps
                window_loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                # Flush deferred entropies (one sync per window instead of per step)
                if entropy_buf:
                    epoch_step_entropies.extend(torch.stack(entropy_buf).cpu().tolist())
                    entropy_buf.clear()
                
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
            
            if args.loss_dump:
                # losses are appended at their two formation sites (deferred-consume
                # and eager); actions/rewards stay per-iteration and index-aligned
                parity_actions.append(int(action))
                parity_rewards.append(float(reward))

            state = next_state
            step_count += 1
            global_step += 1

            # bars/sec timer starts after warmup (excludes compile/caching)
            if perf_t0 is None and global_step >= args.perf_warmup:
                perf_t0 = time.time()
                perf_step0 = global_step

            # torch.profiler window: [perf_warmup, perf_warmup + profile_steps)
            if args.profile_dir:
                if prof is None and prof_start_step is None and global_step >= args.perf_warmup:
                    prof = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA])
                    prof.__enter__()
                    prof_start_step = global_step
                elif prof is not None and (global_step - prof_start_step) >= args.profile_steps:
                    prof.__exit__(None, None, None)
                    os.makedirs(args.profile_dir, exist_ok=True)
                    ka = prof.key_averages()
                    with open(os.path.join(args.profile_dir, 'top_ops_cuda.txt'), 'w') as f:
                        f.write(ka.table(sort_by='cuda_time_total', row_limit=15))
                    with open(os.path.join(args.profile_dir, 'top_ops_cpu.txt'), 'w') as f:
                        f.write(ka.table(sort_by='cpu_time_total', row_limit=15))
                    print(f"[PERF] Profiler tables written to {args.profile_dir}")
                    prof = None

            if args.max_steps and global_step >= args.max_steps:
                stop_training = True
                done = True

            if step_count % 100 == 0:
                reporter.update(global_step, total_epochs * 80000,
                                f"Ep {epoch}/{total_epochs} | Rwd: {episode_reward:.2f}")

            # Memory cleanup (step_loss / n_* / next_value exist only on the eager
            # loss path now; python rebinding handles them, pending_loss holds the
            # deferred refs deliberately)
            del v2_grid_t, l0_feature_t, ledger_state_t, macro_tensor_t, time_of_day_t
            del entry_logits, exit_logits, value, dist, action
            del log_prob, entropy

        if entropy_buf:
            epoch_step_entropies.extend(torch.stack(entropy_buf).cpu().tolist())
            entropy_buf.clear()

        epoch_end_time = time.time()
        print(f"Epoch {epoch} | Reward: {episode_reward:.2f} | Duration: {epoch_end_time - epoch_start_time:.2f}s")

        if stop_training:
            break

        history_rewards.append(episode_reward)
        history_mean_pnls.append(np.mean([t['pnl'] for t in epoch_trades]) if epoch_trades else 0.0)
        history_mean_entropies.append(np.mean(epoch_step_entropies) if epoch_step_entropies else 0.0)

        plot_epoch_summary(epoch, epoch_trades)
        plot_learning_curve(history_rewards, history_mean_pnls, history_mean_entropies)

        if not args.no_checkpoint:
            checkpoint_data = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            torch.save(checkpoint_data, os.path.join(_CKPT_DIR, "mamba_rl_checkpoint.pth"))
            torch.save(checkpoint_data,
                       os.path.join(_CKPT_DIR, f"mamba_rl_checkpoint_ep{epoch}.pth"))

        try:
            print(f"TELEGRAM_TRIGGER: epoch_{epoch}_summary.png and mamba_learning_curve.png are ready!")
        except Exception:
            pass

    if perf_t0 is not None:
        perf_elapsed = time.time() - perf_t0
        perf_bars = global_step - perf_step0
        if perf_elapsed > 0 and perf_bars > 0:
            print(f"[PERF] bars/sec = {perf_bars / perf_elapsed:.2f} "
                  f"({perf_bars} bars in {perf_elapsed:.1f}s, warmup {args.perf_warmup} excluded)")

    if args.loss_dump:
        os.makedirs(os.path.dirname(args.loss_dump) or '.', exist_ok=True)
        np.savez(args.loss_dump,
                 losses=np.array(parity_losses, dtype=np.float64),
                 actions=np.array(parity_actions, dtype=np.int64),
                 rewards=np.array(parity_rewards, dtype=np.float64))
        print(f"[PERF] Parity dump ({len(parity_losses)} steps) written to {args.loss_dump}")

    print(f"Training fully complete! Total Duration: {time.time() - training_start_time:.2f}s")

if __name__ == "__main__":
    import os
    if os.name == 'nt':
        print("Detected Windows! Auto-respawning in WSL GPU environment...")
        import subprocess
        import sys
        try:
            # venv moved to WSL ext4 2026-07-16 (perf: avoid /mnt/c); absolute path — wsl -e does no ~ expansion
            subprocess.run(["wsl", "/home/reyses/venvs/bayesian-ai/bin/python", sys.argv[0]] + sys.argv[1:], check=True)
            sys.exit(0)
        except Exception as e:
            print("Failed to auto-respawn in WSL:", e)
            print("Falling back to Windows CPU training...")
    
    train_mamba_rl()
