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
import json
import psutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from train_mamba_rl import e_exit_preflight_ram, e_exit_vram_check  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GAMMA = 0.99
W_AUX = 0.20
POS_WEIGHT = 10.40  # turn-bar vs non-turn-bar ratio (matches per-bar trainer)
# All .pth artifacts go to the sanctioned repo-root checkpoints/ (gitignored),
# resolved file-relative so no cwd ever gets checkpoint droppings.
_CKPT_DIR = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'checkpoints'))
os.makedirs(_CKPT_DIR, exist_ok=True)
CHECKPOINT = os.path.join(_CKPT_DIR, "mamba_rl_seq_checkpoint.pth")


def prefetch_day_tensors(env, device):
    """Materialize the action-independent observation columns for the whole
    multi-day stream as per-type GPU tensors, sliced by bar index thereafter.
    Uses env.compute_bar_obs — the same code path the legacy observation
    builder uses, so values are bit-identical. Only the 4-float ledger vector
    (action-dependent) is written per bar during the acting pass."""
    v2_rows, l0s, macros, tods, tss = [], [], [], [], []
    for bar in iter(env.fps):
        if bar.v2_vector is None:
            continue
        grid_row, l0, macro, tod = env.compute_bar_obs(bar)
        v2_rows.append(grid_row)
        l0s.append(l0)
        macros.append(macro.astype(np.float32))
        tods.append(tod)
        tss.append(bar.timestamp)
    n = len(v2_rows)
    logger.info(f"Prefetched {n} bars into per-type day tensors")
    return (torch.from_numpy(np.stack(v2_rows)).to(device),          # [N, 8, 52]
            torch.from_numpy(np.stack(l0s)).to(device),              # [N, 1]
            torch.from_numpy(np.stack(macros)).to(device),           # [N, 261]
            torch.from_numpy(np.stack(tods)).to(device),             # [N, 4]
            np.array(tss))                                           # [N] timestamps


def detach_states(states):
    if states is None:
        return None
    return [(h.detach(), cs.detach()) for (h, cs) in states]


def slice_inputs(day, s, e):
    """Model inputs for absolute bar range [s, e) from the per-type day
    tensors. day = (v2 [N,8,52], l0 [N,1], macro [N,261], tod [N,4],
    ledger [N,4])."""
    v2_day, l0_day, macro_day, tod_day, ledger_day = day
    return (v2_day[s:e].unsqueeze(0).permute(0, 2, 1, 3),  # [1, 8, L, 52]
            l0_day[s:e].unsqueeze(0),
            ledger_day[s:e].unsqueeze(0),
            macro_day[s:e].unsqueeze(0),
            tod_day[s:e].unsqueeze(0))


def window_losses(model, day, win_start, W, actions, rewards, is_flat, turn_imm,
                  reset_idx, bootstrap_value, states_in, entropy_coef, device):
    """Learning pass: differentiable forward over the window (chunked at
    session resets) + vectorized per-bar A2C losses. Returns
    (window_loss_mean, final_states, per_bar_losses.detach, entropies.detach)."""
    # Chunk boundaries: session resets zero the hidden state mid-window,
    # mirroring hidden_states=None in the per-bar trainer.
    bounds = [0] + sorted(i for i in reset_idx if 0 < i < W) + [W]
    ent_out, exi_out, val_out = [], [], []
    states = states_in
    for a, b in zip(bounds[:-1], bounds[1:]):
        if a > bounds[0]:
            states = None  # reset at chunk start
        v2, l0, ledg, macro, tod = slice_inputs(day, win_start + a, win_start + b)
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
    parser.add_argument('--init_from', type=str, default='',
                        help='Warm-start checkpoint (state_dict or {"model":...}); '
                             'loaded non-strict. Overridden by a resumed --no-checkpoint '
                             'CHECKPOINT if present.')
    parser.add_argument('--smoke_metrics', action='store_true',
                        help='Emit a per-epoch [SMOKE] json line (trades/day, %%flat, '
                             'P(enter|signal), hold dist, reward-component sums). '
                             'Off by default: default training behavior is unchanged.')
    parser.add_argument('--compile_act', action='store_true',
                        help='torch.compile the acting forward_step (default mode; NOT '
                             'reduce-overhead — cudagraphs crash on carried (h, conv) '
                             'state). Learning pass stays eager. Parity-gate before '
                             'trusting any run that uses this.')
    parser.add_argument('--no_autocast', action='store_true',
                        help='Disable bf16 autocast (fp32 everywhere). For the compile '
                             'parity harness: bf16 fusion reassociation masks real '
                             'graph bugs at the 1e-7 gate.')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    e_exit_preflight_ram(required_gb=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Seq-window trainer | device {device} | W={args.tbptt_window}")

    # Repo-root-relative ATLAS (native Linux, 2026-07-22). The old WSL/Windows
    # os.uname heuristic hardcoded dead /mnt/c and C:/Users/reyse/OneDrive paths;
    # --atlas-root overrides. pipeline/ -> mamba_zigzag_baseline/ -> research/ -> repo.
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    atlas_root = getattr(args, "atlas_root", None) or os.path.join(_repo_root, "DATA", "ATLAS")

    env = MambaRLTradingEnv(
        atlas_root=atlas_root,
        features_root=os.path.join(atlas_root, "FEATURES_5s_v2"),
        labels_csv=os.path.join(atlas_root, "regime_labels_2d.csv"),
        days=[d.strip() for d in args.days.split(',')],
        target_pnl_per_trade=10.0,
        seq_len=30,
        build_observation=False)  # obs come from the prefetched day tensors

    model = MambaRLTradingNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    if args.init_from:
        if not os.path.exists(args.init_from):
            raise FileNotFoundError(f"--init_from checkpoint not found: {args.init_from}")
        sd = torch.load(args.init_from, map_location=device, weights_only=False)
        if isinstance(sd, dict) and 'model' in sd:
            sd = sd['model']
        res = model.load_state_dict(sd, strict=False)
        logger.info(f"[INIT_FROM] loaded {args.init_from} | "
                    f"missing={list(res.missing_keys)} | unexpected={list(res.unexpected_keys)}")
    if not args.no_checkpoint and os.path.exists(CHECKPOINT):
        ck = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_epoch = ck.get('epoch', -1) + 1
        logger.info(f"Resumed {CHECKPOINT} at epoch {start_epoch}")

    act_step = model.forward_step
    if args.compile_act:
        # Acting-only compile: fuses the per-bar no_grad path (launch overhead
        # dominates at batch 1). Two graph specializations expected (states
        # None at session resets vs carried tuples) — both compile once.
        act_step = torch.compile(model.forward_step, mode='default')
        logger.info("[COMPILE] acting forward_step compiled (default mode, learning pass eager)")

    autocast_on = (device.type == 'cuda' and not args.no_autocast)

    reporter = TelemetryReporter("Mamba_RL_SEQ")
    from epoch_summary import plot_epoch_summary, plot_learning_curve

    total_epochs = args.num_episodes
    base_entropy = 0.01
    global_step = 0
    history_rewards, history_mean_pnls, history_mean_entropies = [], [], []

    W_max = args.tbptt_window
    v2_day, l0_day, macro_day, tod_day, ts_day = prefetch_day_tensors(env, device)
    ledger_day = torch.zeros(v2_day.shape[0], 4, dtype=torch.float32, device=device)
    day = (v2_day, l0_day, macro_day, tod_day, ledger_day)
    ledger_pin = torch.zeros(4, dtype=torch.float32).pin_memory()

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
        # Smoke A/B accumulators (only touched when --smoke_metrics).
        sm = dict(n_bars=0, n_flat=0, n_entry=0,
                  n_flat_sig=0, n_entry_sig=0, n_flat_nosig=0, n_entry_nosig=0,
                  sum_capture=0.0, sum_regret=0.0, sum_selectivity=0.0,
                  sum_cut=0.0, sum_direction=0.0, sum_wiggle=0.0, sum_cost=0.0)
        states_carry = None  # canonical (h, conv) carry, from the LEARNING pass
        # env.reset() consumed the first seq_len valid bars as warmup, so the
        # current bar is prefetch index seq_len-1 (guarded per bar below).
        obs_idx = env.seq_len - 1

        while not done:
            if e_exit_vram_check():
                logger.error("[E-EXIT] VRAM failsafe (window boundary).")
                torch.save(model.state_dict(),
                           os.path.join(_CKPT_DIR, f"mamba_rl_seq_e_exit_ep{epoch}.pth"))
                sys.exit(88)

            # ── ACTING PASS ──
            acts, rews, flats, turns, resets = [], [], [], [], []
            act_states = states_carry
            win_start = obs_idx
            w = 0
            with torch.no_grad():
                while w < W_max and not done:
                    t = obs_idx
                    # Anti-scramble guard: prefetched stream must track the env
                    assert ts_day[t] == env.current_bar.timestamp, \
                        f"prefetch misalignment at idx {t}: ts_day={float(ts_day[t])} env={env.current_bar.timestamp}"
                    # The 4 action-dependent floats are the only per-bar write
                    ledger_pin.copy_(torch.from_numpy(env.ledger_state_vec()))
                    ledger_day[t].copy_(ledger_pin, non_blocking=True)

                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=autocast_on):
                        e_l, x_l, _, act_states = act_step(
                            v2_day[t].view(1, 8, 1, 52), l0_day[t].view(1, 1, 1),
                            ledger_day[t].view(1, 1, 4), macro_day[t].view(1, 1, 261),
                            tod_day[t].view(1, 1, 4), act_states)

                    is_flat = env.ledger.is_flat
                    if is_flat:
                        dist = torch.distributions.Categorical(
                            probs=torch.softmax(e_l, dim=-1))
                        action = int(dist.sample().item())
                        env_action = action  # 0=HOLD,1=LONG,2=SHORT map directly
                    else:
                        dist = torch.distributions.Bernoulli(
                            probs=torch.sigmoid(x_l.squeeze(-1)))
                        action = int(dist.sample().item())  # 0=hold,1=exit (recorded for learning)
                        # BUGFIX: the env exits a LONG only on action 2/3 and a
                        # SHORT only on action 1/3; the raw Bernoulli 1 no-ops a
                        # long, leaving it open until the 15:55 guard rail (the
                        # "1 trade, 16459-bar hold" pathology). SCRATCH (3) exits
                        # any direction. Record the policy action {0,1} for the
                        # learning pass; send the env its exit code.
                        env_action = 3 if action == 1 else 0

                    next_state, reward, done, info = env.step(env_action, 0.0)

                    acts.append(action)
                    rews.append(reward)
                    flats.append(is_flat)
                    turns.append(1.0 if info.get('turn_imminent', 0.0) else 0.0)
                    episode_reward += reward

                    if args.smoke_metrics:
                        # is_flat is the PRE-step state; 'entered' is set by the
                        # env only on an actual (post guard-rail) entry.
                        ct = info.get('c_t', 0.0)
                        entered = info.get('entered', False)
                        sm['n_bars'] += 1
                        if entered:
                            sm['n_entry'] += 1
                        if is_flat:
                            sm['n_flat'] += 1
                            if ct >= 0.5:
                                sm['n_flat_sig'] += 1
                                if entered:
                                    sm['n_entry_sig'] += 1
                            else:
                                sm['n_flat_nosig'] += 1
                                if entered:
                                    sm['n_entry_nosig'] += 1
                        sm['sum_capture'] += info.get('reward_capture', 0.0)
                        sm['sum_regret'] += info.get('reward_regret', 0.0)
                        sm['sum_selectivity'] += info.get('reward_selectivity', 0.0)
                        sm['sum_cut'] += info.get('reward_cut_bonus', 0.0)
                        sm['sum_direction'] += info.get('reward_direction', 0.0)
                        sm['sum_wiggle'] += info.get('reward_wiggle', 0.0)
                        sm['sum_cost'] += info.get('reward_cost', 0.0)
                    if info.get('trade_closed', False):
                        epoch_trades.append({
                            'pnl': info['actual_pnl'], 'duration': info['duration'],
                            'entry_ts': info['entry_ts'], 'exit_ts': info['exit_ts'],
                            'direction': info['direction']})
                    if info.get('session_reset', False):
                        act_states = None
                        resets.append(w + 1)  # state reset applies FROM the next bar

                    obs_idx += 1
                    w += 1
                    global_step += 1
                    if perf_t0 is None and global_step >= args.perf_warmup:
                        perf_t0, perf_step0 = time.time(), global_step
                    if args.max_steps and global_step >= args.max_steps:
                        stop_training = True
                        done = True

                # Boundary bootstrap V(s_W) with pre-update weights
                if not done:
                    t = obs_idx
                    assert ts_day[t] == env.current_bar.timestamp, \
                        f"prefetch misalignment at boundary idx {t}"
                    # Post-step ledger; the next window's first bar rewrites the
                    # same value (no env.step in between)
                    ledger_pin.copy_(torch.from_numpy(env.ledger_state_vec()))
                    ledger_day[t].copy_(ledger_pin, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=autocast_on):
                        _, _, boot_v, _ = act_step(
                            v2_day[t].view(1, 8, 1, 52), l0_day[t].view(1, 1, 1),
                            ledger_day[t].view(1, 1, 4), macro_day[t].view(1, 1, 261),
                            tod_day[t].view(1, 1, 4), act_states)
                    bootstrap = boot_v.reshape(()).float()
                else:
                    bootstrap = torch.zeros((), device=device)

            # ── LEARNING PASS ──
            actions_t = torch.tensor(acts, dtype=torch.long, device=device)
            rewards_t = torch.tensor(rews, dtype=torch.float32, device=device)
            flats_t = torch.tensor(flats, dtype=torch.bool, device=device)
            turns_t = torch.tensor(turns, dtype=torch.float32, device=device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=autocast_on):
                loss, states_new, per_bar, ents = window_losses(
                    model, day, win_start, w, actions_t, rewards_t, flats_t, turns_t,
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

        if args.smoke_metrics:
            n_days = len([d for d in args.days.split(',') if d.strip()])
            dur = [t['duration'] for t in epoch_trades]
            metrics = {
                'epoch': epoch,
                'n_bars': sm['n_bars'],
                'n_days': n_days,
                'n_trades': len(epoch_trades),
                'trades_per_day': (len(epoch_trades) / n_days) if n_days else 0.0,
                'pct_flat': (sm['n_flat'] / sm['n_bars']) if sm['n_bars'] else 0.0,
                'n_flat_sig': sm['n_flat_sig'],
                'n_flat_nosig': sm['n_flat_nosig'],
                'P_enter_given_sig': (sm['n_entry_sig'] / sm['n_flat_sig']) if sm['n_flat_sig'] else 0.0,
                'P_enter_given_nosig': (sm['n_entry_nosig'] / sm['n_flat_nosig']) if sm['n_flat_nosig'] else 0.0,
                'hold_median': float(np.median(dur)) if dur else 0.0,
                'hold_p90': float(np.percentile(dur, 90)) if dur else 0.0,
                'sum_capture': round(sm['sum_capture'], 3),
                'sum_regret': round(sm['sum_regret'], 3),
                'sum_selectivity': round(sm['sum_selectivity'], 3),
                'sum_cut': round(sm['sum_cut'], 3),
                'sum_direction': round(sm['sum_direction'], 3),
                'sum_wiggle': round(sm['sum_wiggle'], 3),
                'sum_cost': round(sm['sum_cost'], 3),
                'episode_reward': round(episode_reward, 2),
            }
            print("[SMOKE] " + json.dumps(metrics))

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
            torch.save(ck, os.path.join(_CKPT_DIR, f"mamba_rl_seq_checkpoint_ep{epoch}.pth"))

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
            script_path = sys.argv[0].replace('\\', '/')
            # venv moved to WSL ext4 2026-07-16 (perf: avoid /mnt/c); absolute path — wsl -e does no ~ expansion
            subprocess.run(["wsl", "/home/reyses/venvs/bayesian-ai/bin/python", script_path] + sys.argv[1:],
                           check=True)
            sys.exit(0)
        except Exception as e:
            print("Failed to auto-respawn in WSL:", e)
    train()
