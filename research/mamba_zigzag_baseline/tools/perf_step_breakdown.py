"""Per-step wall-time breakdown of the Mamba RL training loop.

CUPTI does not capture device events on this WSL2 driver, so torch.profiler
cannot give CUDA-time tables. This tool instead brackets each component of
the training step with torch.cuda.synchronize() and wall clocks, attributing
milliseconds per bar to:

  env_iter        - FPS next(iterator) inside env.step (data feed)
  env_enqueue     - _enqueue_bar_state (assemble_v2_grid on 1 bar, tz math)
  env_getobs      - _get_observation (assemble_v2_grid on full window)
  env_rest        - remaining env.step logic (ledger, reward policy)
  pack_h2d        - numpy concatenate + pinned copy + H2D
  forward_action  - model forward #1 (with grad)
  sample_sync     - dist construction + sample + action.item() sync
  forward_value   - model forward #2 (no_grad next_value)
  loss_entropy    - loss seam + entropy.item() sync
  backward_opt    - TBPTT window backward + optimizer step (amortized)

Usage (from repo root, inside WSL venv):
  .venv_wsl/bin/python research/mamba_zigzag_baseline/tools/perf_step_breakdown.py \
      --days 2024_02_20 --steps 600 [--compile] [--out report.txt]

Runtime-only measurement; no training math is altered (same ops, same order,
just timed). Results are written to --out (default: reports/perf/step_breakdown.txt).
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
_PIPE = os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'pipeline')
sys.path.insert(0, _REPO)
sys.path.insert(0, _PIPE)

from mamba_rl_network import MambaRLTradingNetwork  # noqa: E402
from mamba_env import MambaRLTradingEnv  # noqa: E402
from train_mamba_rl import a2c_loss_seam  # noqa: E402


class Timer:
    def __init__(self, sync):
        self.sync = sync
        self.acc = defaultdict(float)
        self.n = defaultdict(int)
        self._t = None

    def start(self):
        if self.sync:
            torch.cuda.synchronize()
        self._t = time.perf_counter()

    def stop(self, key):
        if self.sync:
            torch.cuda.synchronize()
        dt = time.perf_counter() - self._t
        self.acc[key] += dt
        self.n[key] += 1
        self._t = time.perf_counter()
        return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=str, default='2024_02_20')
    ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--warmup', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tbptt_window', type=int, default=500)
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--out', type=str,
                    default=os.path.join(_REPO, 'research', 'mamba_zigzag_baseline',
                                         'reports', 'perf', 'step_breakdown.txt'))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')

    if os.name == 'posix' and 'microsoft' in os.uname().release.lower():
        atlas_root = '/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS'
    else:
        atlas_root = 'C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS'

    env = MambaRLTradingEnv(
        atlas_root=atlas_root,
        features_root=os.path.join(atlas_root, 'FEATURES_5s_v2'),
        labels_csv=os.path.join(atlas_root, 'regime_labels_2d.csv'),
        days=[d.strip() for d in args.days.split(',')],
        target_pnl_per_trade=10.0,
        seq_len=30,
    )

    model = MambaRLTradingNetwork().to(device)
    if args.compile:
        model = torch.compile(model, mode='reduce-overhead')
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- monkeypatch env internals with timers (no logic change) ---
    t_env = defaultdict(float)
    n_env = defaultdict(int)

    orig_enqueue = env._enqueue_bar_state
    orig_getobs = env._get_observation

    def timed_enqueue(bar_state):
        t0 = time.perf_counter()
        r = orig_enqueue(bar_state)
        t_env['env_enqueue'] += time.perf_counter() - t0
        n_env['env_enqueue'] += 1
        return r

    def timed_getobs():
        t0 = time.perf_counter()
        r = orig_getobs()
        t_env['env_getobs'] += time.perf_counter() - t0
        n_env['env_getobs'] += 1
        return r

    env._enqueue_bar_state = timed_enqueue
    env._get_observation = timed_getobs

    class TimedIter:
        def __init__(self, it):
            self.it = it

        def __iter__(self):
            return self

        def __next__(self):
            t0 = time.perf_counter()
            r = next(self.it)
            t_env['env_iter'] += time.perf_counter() - t0
            n_env['env_iter'] += 1
            return r

    state = env.reset()
    env.iterator = TimedIter(env.iterator)

    pinned = torch.zeros((1, 1, 686), dtype=torch.float32).pin_memory()
    gpu_buf = torch.zeros((1, 1, 686), dtype=torch.float32, device=device)
    n_pinned = torch.zeros((1, 1, 686), dtype=torch.float32).pin_memory()
    n_gpu_buf = torch.zeros((1, 1, 686), dtype=torch.float32, device=device)

    tm = Timer(sync=True)
    hidden_states = None
    window_loss = 0.0
    window_steps = 0
    gamma = 0.99
    entropy_coef = 0.01
    optimizer.zero_grad()

    total_t0 = None
    done = False
    step = 0

    def pack(s, pin, gbuf):
        v2, l0, ledg, macro, tod = s
        packed = np.concatenate([v2[:, -1, :].reshape(-1), l0[-1, :], ledg[-1, :],
                                 macro[-1, :], tod[-1, :]])
        pin[0, 0].copy_(torch.from_numpy(packed))
        gbuf.copy_(pin, non_blocking=True)
        return (gbuf[:, :, :416].view(1, 1, 8, 52).permute(0, 2, 1, 3),
                gbuf[:, :, 416:417], gbuf[:, :, 417:421],
                gbuf[:, :, 421:682], gbuf[:, :, 682:686])

    while not done and step < args.steps + args.warmup:
        if step == args.warmup:
            for k in list(t_env):
                t_env[k] = 0.0
                n_env[k] = 0
            tm.acc.clear()
            tm.n.clear()
            total_t0 = time.perf_counter()

        tm.start()
        v2_t, l0_t, ledg_t, macro_t, tod_t = pack(state, pinned, gpu_buf)
        tm.stop('pack_h2d')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            if hasattr(torch.compiler, 'cudagraph_mark_step_begin'):
                torch.compiler.cudagraph_mark_step_begin()
            entry_logits, exit_logits, value, hidden_states = model(
                v2_t, l0_t, ledg_t, macro_t, tod_t, hidden_states)
        tm.stop('forward_action')

        is_flat = state[2][-1, 0] == 0.0
        if is_flat:
            dist = torch.distributions.Categorical(probs=torch.softmax(entry_logits, dim=-1))
        else:
            dist = torch.distributions.Bernoulli(probs=torch.sigmoid(exit_logits.squeeze(-1)))
        action = dist.sample()
        a = action.item()
        tm.stop('sample_sync')

        t0 = time.perf_counter()
        next_state, reward, done, info = env.step(a, 0.0)
        t_env['env_step_total'] += time.perf_counter() - t0
        n_env['env_step_total'] += 1
        tm.start()

        if info.get('session_reset', False):
            hidden_states = None

        if not done and next_state is not None:
            with torch.no_grad():
                nv2, nl0, nledg, nmacro, ntod = pack(next_state, n_pinned, n_gpu_buf)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, _, next_value, _ = model(nv2, nl0, nledg, nmacro, ntod, hidden_states)
        else:
            next_value = torch.tensor([[0.0]], device=device)
        tm.stop('forward_value')

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        _ = entropy.item()
        bce = None
        if not is_flat:
            target_t = torch.tensor([1.0 if info.get('turn_imminent', 0.0) else 0.0],
                                    device=device)
            pw = torch.tensor([10.40], device=device)
            bce = F.binary_cross_entropy_with_logits(exit_logits.squeeze(-1), target_t,
                                                     pos_weight=pw)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            step_loss = a2c_loss_seam(reward, value, next_value, log_prob, entropy,
                                      entropy_coef, gamma, bce, 0.20, device)
        window_loss = window_loss + step_loss
        window_steps += 1
        tm.stop('loss_entropy')

        if window_steps >= args.tbptt_window or done:
            window_loss = window_loss / window_steps
            window_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            hs = []
            for h in hidden_states:
                if h is None:
                    hs.append(None)
                elif isinstance(h, tuple):
                    hs.append((h[0].detach(), h[1].detach()))
                else:
                    hs.append(h.detach())
            hidden_states = hs
            window_loss = 0.0
            window_steps = 0
        tm.stop('backward_opt')

        state = next_state
        step += 1

    total_wall = time.perf_counter() - total_t0
    measured = step - args.warmup

    lines = []
    lines.append(f'Per-step wall-time breakdown ({measured} steps measured, '
                 f'warmup {args.warmup}, compile={args.compile}, sync-bracketed)')
    lines.append(f'TOTAL: {total_wall:.2f}s wall -> {measured / total_wall:.2f} bars/sec '
                 f'({1000 * total_wall / measured:.1f} ms/bar)')
    lines.append('')
    lines.append(f'{"component":<18}{"total_s":>10}{"ms/bar":>10}{"share":>8}{"calls":>8}')
    rows = []
    env_sub = 0.0
    for k, v in t_env.items():
        if k != 'env_step_total':
            env_sub += v
    for k, v in sorted(tm.acc.items(), key=lambda kv: -kv[1]):
        rows.append((k, v, tm.n[k]))
    est = t_env.get('env_step_total', 0.0)
    inner = (t_env.get('env_iter', 0.0) + t_env.get('env_enqueue', 0.0)
             + t_env.get('env_getobs', 0.0))
    rows.append(('env_step_total', est, n_env.get('env_step_total', 0)))
    rows.append(('  env_iter', t_env.get('env_iter', 0.0), n_env.get('env_iter', 0)))
    rows.append(('  env_enqueue', t_env.get('env_enqueue', 0.0), n_env.get('env_enqueue', 0)))
    rows.append(('  env_getobs', t_env.get('env_getobs', 0.0), n_env.get('env_getobs', 0)))
    rows.append(('  env_rest', max(est - inner, 0.0), n_env.get('env_step_total', 0)))
    for k, v, n in sorted(rows, key=lambda r: -r[1]):
        lines.append(f'{k:<18}{v:>10.2f}{1000 * v / measured:>10.2f}'
                     f'{100 * v / total_wall:>7.1f}%{n:>8}')
    out = '\n'.join(lines)
    print(out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(out + '\n')
    print(f'\nWritten to {args.out}')


if __name__ == '__main__':
    main()
