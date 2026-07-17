"""warmstart_supervised.py -- supervised warm-start of the Mamba RL trunk+heads.

Labels TEACH, never OBSERVE. The model consumes its NORMAL causal observations
(v2 grid, l0, ledger[=flat], macro, tod); the AI cusp-pick labels appear ONLY as
cross-entropy targets on the entry (3-way) and exit (binary) heads. The value
head is left untouched. This breaks the always-flat collapse by teaching the
entry head to PROPOSE long/short when a profitable setup is active, so at RL
time the policy has a non-degenerate starting distribution.

Targets (evaluated at 1m-boundary bars, ts % 60 == 0):
  entry head : LONG(1)/SHORT(2) if an AI label is ACTIVE at the bar, else HOLD(0)
  exit head  : 1 if the active label ENDS within the next 3 minutes, else 0

The ledger channel is held FLAT (zeros) throughout -- encoding the label's
position into the ledger would leak the label into the observation, which the
"labels only as targets" rule forbids. Consequence: the entry head (the freeze
lever) trains in-distribution (flat == the RL-time flat state); the exit head
trains on flat-ledger inputs, a documented mild covariate shift vs its RL-time
in-position inputs (the market features it keys on are ledger-independent).

Run (WSL GPU):
  ~/venvs/bayesian-ai/bin/python -u warmstart_supervised.py \
      --days 2024_01_09,... --epochs 3 --out <repo>/checkpoints/mamba_warmstart.pth
"""
import os
import sys
import time
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from mamba_rl_network import MambaRLTradingNetwork
from mamba_env import MambaRLTradingEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DAY_GAP_S = 300           # ts gap that marks a session boundary (reset Mamba state)
EXIT_HORIZON_S = 180      # "ends within the next 3 minutes"
MINUTE_S = 60             # 1m-boundary sampling
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def prefetch_day_tensors(env, device):
    """Materialize the action-independent observation columns as per-type GPU
    tensors. Mirrors train_mamba_rl_seq.prefetch_day_tensors (same compute_bar_obs
    path, so values are bit-identical)."""
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
    logger.info(f"Prefetched {len(v2_rows)} bars into per-type day tensors")
    return (torch.from_numpy(np.stack(v2_rows)).to(device),
            torch.from_numpy(np.stack(l0s)).to(device),
            torch.from_numpy(np.stack(macros)).to(device),
            torch.from_numpy(np.stack(tods)).to(device),
            np.array(tss, dtype=np.int64))


def slice_inputs(day, s, e):
    """Model inputs for absolute bar range [s, e). day = (v2, l0, macro, tod, ledger)."""
    v2_day, l0_day, macro_day, tod_day, ledger_day = day
    return (v2_day[s:e].unsqueeze(0).permute(0, 2, 1, 3),  # [1, 8, L, 52]
            l0_day[s:e].unsqueeze(0),
            ledger_day[s:e].unsqueeze(0),
            macro_day[s:e].unsqueeze(0),
            tod_day[s:e].unsqueeze(0))


def build_labels(ts_arr, ai_picks):
    """Vectorized supervised targets from the AI cusp picks.
    Returns (entry_tgt[int64 N], exit_tgt[float32 N])."""
    n = ts_arr.shape[0]
    entry_tgt = np.zeros(n, dtype=np.int64)   # 0 = HOLD
    exit_tgt = np.zeros(n, dtype=np.float32)  # 0 = hold
    for p in ai_picks:
        e_ts = float(p['entry_ts'])
        x_ts = float(p['exit_ts'])
        d = 1 if str(p['direction']).upper() == 'LONG' else 2
        lo = int(np.searchsorted(ts_arr, e_ts, side='left'))
        hi = int(np.searchsorted(ts_arr, x_ts, side='right'))
        if hi > lo:
            entry_tgt[lo:hi] = d
            elo = int(np.searchsorted(ts_arr, x_ts - EXIT_HORIZON_S, side='left'))
            exit_tgt[max(elo, lo):hi] = 1.0
    return entry_tgt, exit_tgt


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--tbptt_window', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=str,
                        default=os.path.join(_REPO_ROOT, 'checkpoints', 'mamba_warmstart.pth'))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.name == 'posix' and 'microsoft' in os.uname().release.lower():
        atlas_root = "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    else:
        atlas_root = "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"

    days = [d.strip() for d in args.days.split(',') if d.strip()]
    logger.info(f"Warm-start | device {device} | {len(days)} days | W={args.tbptt_window}")

    env = MambaRLTradingEnv(
        atlas_root=atlas_root,
        features_root=os.path.join(atlas_root, "FEATURES_5s_v2"),
        labels_csv=os.path.join(atlas_root, "regime_labels_2d.csv"),
        days=days,
        target_pnl_per_trade=10.0,
        seq_len=30,
        build_observation=False)

    model = MambaRLTradingNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    v2_day, l0_day, macro_day, tod_day, ts_day = prefetch_day_tensors(env, device)
    n = v2_day.shape[0]
    # Ledger held FLAT (zeros) -- labels must not enter the observation.
    ledger_day = torch.zeros(n, 4, dtype=torch.float32, device=device)
    day = (v2_day, l0_day, macro_day, tod_day, ledger_day)

    # env.ai_picks is the full sorted pick list (env is never stepped here, so it
    # is not pruned). Labels appear ONLY here, as targets.
    entry_tgt_np, exit_tgt_np = build_labels(ts_day, env.ai_picks)
    minute_mask_np = (ts_day % MINUTE_S == 0)

    entry_tgt = torch.from_numpy(entry_tgt_np).to(device)
    exit_tgt = torch.from_numpy(exit_tgt_np).to(device)
    minute_mask = torch.from_numpy(minute_mask_np).to(device)

    # Class weights on the SAMPLED (1m-boundary) label distribution -- otherwise
    # HOLD / no-exit dominate and the warm-start just relearns "always flat".
    me = entry_tgt_np[minute_mask_np]
    counts = np.bincount(me, minlength=3).astype(np.float64)
    entry_w = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1.0)),
                           dtype=torch.float32, device=device)
    mx = exit_tgt_np[minute_mask_np]
    npos = float(mx.sum())
    nneg = float(len(mx) - npos)
    exit_pos_w = torch.tensor([nneg / max(1.0, npos)], dtype=torch.float32, device=device)
    logger.info(f"Sampled 1m bars: {int(minute_mask_np.sum())} | "
                f"entry HOLD/LONG/SHORT={counts.astype(int).tolist()} "
                f"weights={entry_w.tolist()} | exit pos/neg={int(npos)}/{int(nneg)} "
                f"pos_weight={exit_pos_w.item():.2f}")

    # Per-day segments (fresh Mamba state per session).
    boundaries = [0] + [int(i) for i in np.where(np.diff(ts_day) > DAY_GAP_S)[0] + 1] + [n]
    logger.info(f"Day segments: {len(boundaries) - 1}")

    W = args.tbptt_window
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        ep_loss_e, ep_loss_x, ep_batches, ep_bars = 0.0, 0.0, 0, 0
        for di in range(len(boundaries) - 1):
            s0, e0 = boundaries[di], boundaries[di + 1]
            states = None
            for ws in range(s0, e0, W):
                we = min(ws + W, e0)
                m = minute_mask[ws:we]
                v2, l0, ledg, macro, tod = slice_inputs(day, ws, we)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    e_l, x_l, _, states = model.forward_sequence(v2, l0, ledg, macro, tod, states)
                states = [(h.detach(), cs.detach()) for (h, cs) in states]
                if not bool(m.any()):
                    continue
                el = e_l[0].float()[m]          # [k, 3]
                xl = x_l[0, :, 0].float()[m]     # [k]
                et = entry_tgt[ws:we][m]
                xt = exit_tgt[ws:we][m]
                ce_e = F.cross_entropy(el, et, weight=entry_w)
                ce_x = F.binary_cross_entropy_with_logits(xl, xt, pos_weight=exit_pos_w)
                loss = ce_e + ce_x
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss_e += float(ce_e.item())
                ep_loss_x += float(ce_x.item())
                ep_batches += 1
                ep_bars += int(m.sum().item())
        b = max(ep_batches, 1)
        logger.info(f"[WARMSTART] epoch {epoch} | CE_entry={ep_loss_e / b:.4f} "
                    f"CE_exit={ep_loss_x / b:.4f} | batches={ep_batches} 1m-bars={ep_bars} "
                    f"| {time.time() - t0:.1f}s")

    # Diagnostic: entry-head argmax distribution on 1m bars post-warmstart.
    model.eval()
    with torch.no_grad():
        preds = []
        states = None
        for di in range(len(boundaries) - 1):
            s0, e0 = boundaries[di], boundaries[di + 1]
            states = None
            for ws in range(s0, e0, W):
                we = min(ws + W, e0)
                v2, l0, ledg, macro, tod = slice_inputs(day, ws, we)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == 'cuda')):
                    e_l, _, _, states = model.forward_sequence(v2, l0, ledg, macro, tod, states)
                states = [(h.detach(), cs.detach()) for (h, cs) in states]
                m = minute_mask_np[ws:we]
                if m.any():
                    preds.append(e_l[0].float().argmax(-1).cpu().numpy()[m])
        if preds:
            allp = np.concatenate(preds)
            dist = np.bincount(allp, minlength=3)
            logger.info(f"[WARMSTART] post-train entry argmax on 1m bars "
                        f"HOLD/LONG/SHORT={dist.tolist()} "
                        f"(non-HOLD frac={1.0 - dist[0] / max(1, dist.sum()):.3f})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({'model': model.state_dict(), 'warmstart': True,
                'days': args.days, 'epochs': args.epochs}, args.out)
    sz = os.path.getsize(args.out) / 1e6
    logger.info(f"[WARMSTART] saved {args.out} ({sz:.2f} MB)")


if __name__ == "__main__":
    if os.name == 'nt':
        print("Detected Windows! Auto-respawning in WSL GPU environment...")
        import subprocess
        try:
            script_path = sys.argv[0].replace('\\', '/')
            subprocess.run(["wsl", "/home/reyses/venvs/bayesian-ai/bin/python", script_path]
                           + sys.argv[1:], check=True)
            sys.exit(0)
        except Exception as e:
            print("Failed to auto-respawn in WSL:", e)
    train()
