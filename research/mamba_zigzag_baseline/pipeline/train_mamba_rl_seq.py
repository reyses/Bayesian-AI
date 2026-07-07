"""Two-pass sequence-window trainer (docs/JULES_SEQUENCE_WINDOW_TRAINING.md).

Per TBPTT window of W bars:
  1. ACTING pass (no_grad, bar-by-bar): sample actions from the current
     policy via forward_step (carried h + conv state), step the env, record
     packed observations / actions / rewards / masks, and the boundary
     bootstrap V(s_W) with pre-update weights.
  2. LEARNING pass (one differentiable forward_sequence over the window,
     chunked at session resets): vectorized A2C losses identical in form to
     train_mamba_rl.py's a2c_loss_seam, mean over the window, one backward,
     clip 1.0, Adam step. States carry to the next window detached.

On-policy, same update cadence as the per-bar trainer. Numeric drift vs the
per-bar trainer is bf16-refusion class (~2e-3, see seq_equivalence_test.txt);
semantic change: the conv1d 3-bar receptive field is restored (deliberate,
user-approved). Checkpoints use SEPARATE filenames (mamba_rl_seq_*.pth) so
this trainer can never clobber the per-bar trainer's checkpoints.
"""
from mamba_rl_network import MambaRLTradingNetwork
from mamba_env import MambaRLTradingEnv
from core_v2.telemetry.reporter import TelemetryReporter
import logging
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import os
import sys
import time
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from train_mamba_rl import e_exit_preflight_ram, e_exit_vram_check  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OBS_DIM = 686  # 416 v2 + 1 l0 + 4 ledger + 261 macro + 4 tod
GAMMA = 0.99
W_AUX = 0.20
POS_WEIGHT = 10.40  # turn-bar vs non-turn-bar ratio (matches per-bar trainer)
CHECKPOINT = "mamba_rl_seq_checkpoint.pth"


def pack_state(state):
    v2_grid, l0, ledger, macro, tod = state
    return np.concatenate([v2_grid[:, -1, :].reshape(-1), l0[-1, :],
                           ledger[-1, :], macro[-1, :], tod[-1, :]])


def unpack_window(obs, device):
    """obs: [W, 686] gpu tensor -> the 5 model inputs with batch dim 1."""
    W = obs.shape[0]
    o = obs.unsqueeze(0)  # [1, W, 686]
    v2 = o[:, :, :416].reshape(1, W, 8, 52).permute(0, 2, 1, 3)
    return (v2, o[:, :, 416:417], o[:, :, 417:421],
            o[:, :, 421:682], o[:, :, 682:686])


def detach_states(states):
    if states is None:
        return None
    return [(h.detach(), cs.detach()) for (h, cs) in states]


def window_losses(model, obs_win, actions, rewards, is_flat, turn_imm,
                  reset_idx, bootstrap_value, states_in, entropy_coef, device):
    """Learning pass: differentiable forward over the window (chunked at
    session resets) + vectorized per-bar A2C losses. Returns
    (window_loss_mean, final_states, per_bar_losses.detach, entropies.detach)."""
    W = obs_win.shape[0]
    # Chunk boundaries: session resets zero the hidden state mid-window,
    # mirroring hidden_states=None in the per-bar trainer.
    bounds = [0] + sorted(i for i in reset_idx if 0 < i < W) + [W]
    ent_out, exi_out, val_out = [], [], []
    states = states_in
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a > bounds[0]:
            states = None  # reset at chunk start
        v2, l0, ledg, macro, tod = unpack_window(obs_win[a:b], device)
        e, x, v, states = model.forward_sequence(v2, l0, ledg, macro, tod, states)
        ent_out.append(e); exi_out.append(x); val_out.append(v)
    entry_logits = torch.cat(ent_out, dim=1)[0]   # [W, 3]
    exit_logits = torch.cat(exi_out, dim=1)[0, :, 0]  # [W]
    values = torch.cat(val_out, dim=1)[0, :, 0]   # [W]

    # log_prob + entropy of the TAKEN action, head selected by is_flat
    logp_entry = torch.log_softmax(entry_logits.float(), dim=-1)
    logp_cat = logp_entry.gather(1, actions.clamp(max=2).unsqueeze(1))[:, 0]
    ent_cat = -(logp_entry.exp() * logp_entry).sum(-1)

    exit_f = exit_logits.float()
    act_bin = actions.clamp(max=1).float()
    logp_bern = -F.binary_cross_entropy_with_logits(exit_f, act_bin, reduction='none')
    p = torch.sigmoid(exit_f)
    ent_bern = F.binary_cross_entropy_with_logits(exit_f, p, reduction='none')

    log_prob = torch.where(is_flat, logp_cat, logp_bern)
    entropy = torch.where(is_flat, ent_cat, ent_bern)

    # TD(0): next value is the in-window shift; the last bar uses the
    # recorded pre-update bootstrap. Bars entering a reset bootstrap on the
    # post-reset value (values[r] is computed with freshly zeroed state,
    # matching the per-bar trainer's post-reset no_grad forward).
    next_values = torch.cat([values[1:], bootstrap_value.reshape(1)]).detach()
    vals_f = values.float()
    td_target = rewards + GAMMA * next_values.float()
    advantage = (td_target - vals_f).detach()
    critic_loss = 0.5 * (vals_f - td_target) ** 2
    actor_loss = -(log_prob * advantage) - entropy_coef * entropy

    bce = F.binary_cross_entropy_with_logits(
        exit_f, turn_imm,
        pos_weight=torch.tensor([POS_WEIGHT], device=device), reduction='none')
    per_bar = actor_loss + critic_loss + torch.where(
        is_flat, torch.zeros_like(bce), W_AUX * bce)

    return per_bar.mean(), states, per_bar.detach(), entropy.detach()


def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--days', type=str,
                        default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26")
    parser.add_argument('--tbptt_window', type=int, default=500)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=0)
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--loss-dump', type=str, default='')
    parser.add_argument('--perf-warmup', type=int, default=300)
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    e_exit_preflight_ram(required_gb=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Seq-window trainer | device {device} | W={args.tbptt_window}")

    if os.name == 'posix' and 'microsoft' in os.uname().release.lower():
        atlas_root = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    else:
        atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"

    env = MambaRLTradingEnv(
        atlas_root=atlas_root,
        features_root=os.path.join(atlas_root, "FEATURES_5s_v2"),
        labels_csv=os.path.join(atlas_root, "regime_labels_2d.csv"),
        days=[d.strip() for d in args.days.split(',')],
        target_pnl_per_trade=10.0,
        seq_len=30)

    model = MambaRLTradingNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    if not args.no_checkpoint and os.path.exists(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck.get('epoch', -1) + 1
        logger.info(f"Resumed {CHECKPOINT} at epoch {start_epoch}")

    reporter = TelemetryReporter("Mamba_RL_SEQ")
    from epoch_summary import plot_epoch_summary, plot_learning_curve

    total_epochs = args.num_episodes
    base_entropy = 0.01
    global_step = 0
    history_rewards, history_mean_pnls, history_mean_entropies = [], [], []

    W_max = args.tbptt_window
    pinned = torch.zeros(OBS_DIM, dtype=torch.float32).pin_memory()
    obs_win = torch.zeros(W_max, OBS_DIM, dtype=torch.float32, device=device)
    boot_pinned = torch.zeros(OBS_DIM, dtype=torch.float32).pin_memory()

    perf_t0, perf_step0 = None, 0
    parity_losses, parity_actions, parity_rewards = [], [], []
    stop_training = False
    t_start = time.time()

    for epoch in range(start_epoch, total_epochs):
        t_ep = time.time()
        if hasattr(env, 'update_curriculum_state'):
            env.update_curriculum_state(epoch, total_epochs)
        state = env.reset()

        progress = epoch / total_epochs
        if progress < 0.50:
            entropy_coef = 0.001 + (base_entropy - 0.001) * (1.0 - progress / 0.50)
        else:
            entropy_coef = 0.001

        done = False
        episode_reward = 0.0
        epoch_trades, epoch_entropies = [], []
        states_carry = None  # canonical (h, conv) carry, from the LEARNING pass

        while not done:
            if e_exit_vram_check():
                logger.error("[E-EXIT] VRAM failsafe (window boundary).")
                torch.save(model.state_dict(), f"mamba_rl_seq_e_exit_ep{epoch}.pth")
                sys.exit(88)

            # ── ACTING PASS ──
            acts, rews, flats, turns, resets = [], [], [], [], []
            act_states = states_carry
            w = 0
            with torch.no_grad():
                while w < W_max and not done:
                    packed = pack_state(state)
                    pinned.copy_(torch.from_numpy(packed))
                    obs_win[w].copy_(pinned, non_blocking=True)

                    v2, l0, ledg, macro, tod = unpack_window(obs_win[w:w + 1], device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                        e_l, x_l, _, act_states = model.forward_step(
                            v2, l0, ledg, macro, tod, act_states)

                    is_flat = bool(state[2][-1, 0] == 0.0)
                    if is_flat:
                        dist = torch.distributions.Categorical(
                            probs=torch.softmax(e_l, dim=-1))
                    else:
                        dist = torch.distributions.Bernoulli(
                            probs=torch.sigmoid(x_l.squeeze(-1)))
                    action = int(dist.sample().item())

                    next_state, reward, done, info = env.step(action, 0.0)

                    acts.append(action)
                    rews.append(reward)
                    flats.append(is_flat)
                    turns.append(1.0 if info.get('turn_imminent', 0.0) else 0.0)
                    episode_reward += reward
                    if info.get('trade_closed', False):
                        epoch_trades.append({
                            'pnl': info['actual_pnl'], 'duration': info['duration'],
                            'entry_ts': info['entry_ts'], 'exit_ts': info['exit_ts'],
                            'direction': info['direction']})
                    if info.get('session_reset', False):
                        act_states = None
                        resets.append(w + 1)  # state reset applies FROM the next bar

                    state = next_state
                    w += 1
                    global_step += 1
                    if perf_t0 is None and global_step >= args.perf_warmup:
                        perf_t0, perf_step0 = time.time(), global_step
                    if args.max_steps and global_step >= args.max_steps:
                        stop_training = True
                        done = True

                # Boundary bootstrap V(s_W) with pre-update weights
                if not done and state is not None:
                    boot_pinned.copy_(torch.from_numpy(pack_state(state)))
                    boot_obs = boot_pinned.to(device, non_blocking=True).unsqueeze(0)
                    v2, l0, ledg, macro, tod = unpack_window(boot_obs, device)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=(device.type == 'cuda')):
                        _, _, boot_v, _ = model.forward_step(
                            v2, l0, ledg, macro, tod, act_states)
                    bootstrap = boot_v.reshape(()).float()
                else:
                    bootstrap = torch.zeros((), device=device)

            # ── LEARNING PASS ──
            actions_t = torch.tensor(acts, dtype=torch.long, device=device)
            rewards_t = torch.tensor(rews, dtype=torch.float32, device=device)
            flats_t = torch.tensor(flats, dtype=torch.bool, device=device)
            turns_t = torch.tensor(turns, dtype=torch.float32, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == 'cuda')):
                loss, states_new, per_bar, ents = window_losses(
                    model, obs_win[:w], actions_t, rewards_t, flats_t, turns_t,
                    resets, bootstrap, states_carry, entropy_coef, device)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            # A reset on the window's LAST bar applies to the next window's
            # first bar (chunking above only handles resets strictly inside).
            states_carry = None if (resets and resets[-1] >= w) else detach_states(states_new)

            epoch_entropies.extend(ents.float().cpu().tolist())
            if args.loss_dump:
                parity_losses.extend(per_bar.float().cpu().tolist())
                parity_actions.extend(acts)
                parity_rewards.extend(rews)

            reporter.update(global_step, total_epochs * 80000,
                            f"Ep {epoch}/{total_epochs} | Rwd: {episode_reward:.2f}")

        print(f"Epoch {epoch} | Reward: {episode_reward:.2f} | "
              f"Duration: {time.time() - t_ep:.2f}s")
        if stop_training:
            break

        history_rewards.append(episode_reward)
        history_mean_pnls.append(np.mean([t['pnl'] for t in epoch_trades]) if epoch_trades else 0.0)
        history_mean_entropies.append(np.mean(epoch_entropies) if epoch_entropies else 0.0)
        plot_epoch_summary(epoch, epoch_trades)
        plot_learning_curve(history_rewards, history_mean_pnls, history_mean_entropies)

        if not args.no_checkpoint:
            ck = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                  'epoch': epoch}
            torch.save(ck, CHECKPOINT)
            torch.save(ck, f"mamba_rl_seq_checkpoint_ep{epoch}.pth")

    if perf_t0 is not None:
        el = time.time() - perf_t0
        n = global_step - perf_step0
        if el > 0 and n > 0:
            print(f"[PERF] bars/sec = {n / el:.2f} ({n} bars in {el:.1f}s, "
                  f"warmup {args.perf_warmup} excluded)")
    if args.loss_dump:
        os.makedirs(os.path.dirname(args.loss_dump) or '.', exist_ok=True)
        np.savez(args.loss_dump,
                 losses=np.array(parity_losses), actions=np.array(parity_actions),
                 rewards=np.array(parity_rewards))
        print(f"[PERF] Parity dump ({len(parity_losses)} bars) -> {args.loss_dump}")
    print(f"Training complete. Total {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    if os.name == 'nt':
        print("Detected Windows! Auto-respawning in WSL GPU environment...")
        import subprocess
        try:
            subprocess.run(["wsl", ".venv_wsl/bin/python", sys.argv[0]] + sys.argv[1:],
                           check=True)
            sys.exit(0)
        except Exception as e:
            print("Failed to auto-respawn in WSL:", e)
    train()
