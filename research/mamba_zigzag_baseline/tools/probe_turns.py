"""Turn-detection probes (perception-first curriculum gate).

Probe A (--probe model): does the trained checkpoint's exit head actually
rank turn bars? Rolls the policy (sampled, seeded) exactly like the acting
pass, collects the exit logit every bar, scores AUC vs the oracle
turn_imminent label. Per-day AUCs (the day spread is the honest dispersion),
overall + in-position-only (the training distribution of the BCE loss).

Probe B (--probe signal): the user's question — ignore the model; do the V2
features contain signal at the hindsight turn points AT ALL? Leave-one-day-out
logistic + small MLP on the 682 action-independent obs dims, plus
shuffled-label nulls. House signal bar: AUC-0.5 >= 0.10 real, 0.05-0.10
conditional, < 0.05 noise.

Labels (both probes): turn_imminent = within 125s (25 bars) BEFORE any oracle
pick's exit_ts — same definition the trainer's BCE used. Labels are
HINDSIGHT (offline-labeled golden trades); features are causal. Label-side
hindsight is allowed (segment firewall); conclusions are about signal
existence, not a tradeable causal system.

Usage (WSL venv, repo root):
  python .../probe_turns.py --probe model --ckpt mamba_rl_seq_checkpoint_ep25.pth \
      --days 2024_02_20,...,2024_02_27
  python .../probe_turns.py --probe signal --days 2024_02_20,...,2024_02_27
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'pipeline'))

from mamba_rl_network import MambaRLTradingNetwork  # noqa: E402
from mamba_env import MambaRLTradingEnv  # noqa: E402
from train_mamba_rl_seq import prefetch_day_tensors  # noqa: E402

TURN_WINDOW_S = 125  # h_bars_tolerance(25) * 5s — matches env.step
REPORT_DIR = os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'reports')

lines = []


def log(s):
    print(s)
    lines.append(s)


def auc_score(scores, labels):
    """Rank-based AUC (Mann-Whitney). scores float, labels 0/1."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float('nan'), n1, n0
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    u = ranks[labels == 1].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0)), n1, n0


def atlas_root():
    if os.name == 'posix' and 'microsoft' in os.uname().release.lower():
        return "/mnt/c/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"
    return "C:/Users/reyse/OneDrive/Desktop/Bayesian-AI/DATA/ATLAS"


def build_env(days):
    root = atlas_root()
    return MambaRLTradingEnv(
        atlas_root=root,
        features_root=os.path.join(root, "FEATURES_5s_v2"),
        labels_csv=os.path.join(root, "regime_labels_2d.csv"),
        days=days, target_pnl_per_trade=10.0, seq_len=30,
        build_observation=False)


def turn_labels_for_ts(ts_arr, days, window_s=TURN_WINDOW_S):
    """turn_imminent per bar straight from the picks JSONs (no env stepping)."""
    exits = []
    root = atlas_root()
    for day in days:
        p = os.path.join(root, '..', 'ai_cusp_picks',
                         f"ai_picks_{day.replace('_', '-')}_multi.json")
        if os.path.exists(p):
            with open(p) as f:
                data = json.load(f)
            exits.extend(t['exit_ts'] for t in data.get('trades', []))
    exits = np.array(sorted(exits), dtype=np.float64)
    labels = np.zeros(len(ts_arr), dtype=np.int64)
    if len(exits):
        # bar is a turn-lead-in iff some exit in [ts, ts+125]
        idx = np.searchsorted(exits, ts_arr)  # first exit >= ts
        valid = idx < len(exits)
        labels[valid] = (exits[idx[valid]] - ts_arr[valid] <= window_s).astype(np.int64)
    return labels


def collect_prices(env):
    """Close price + ts per valid bar, same filtering as prefetch_day_tensors."""
    prices, tss = [], []
    for bar in iter(env.fps):
        if bar.v2_vector is None:
            continue
        prices.append(bar.price)
        tss.append(bar.timestamp)
    return np.array(prices, dtype=np.float64), np.array(tss)


def cubic_features(prices, windows=(60, 180, 300)):
    """The user's manual method, mechanized: per bar, fit a cubic to the
    trailing W closes (x in [-1,1], t at x=1) on standardized y, and emit
    turn-geometry features: coefficients c0..c3, end slope (c1+2c2+3c3),
    end curvature (2c2+6c3), fit residual RMS, and bars since the curvature
    last flipped sign (inflection recency, log1p-compressed).
    Vectorized via sliding windows + precomputed pseudo-inverse."""
    n = len(prices)
    feats = []
    for W in windows:
        x = np.linspace(-1.0, 1.0, W)
        X = np.stack([np.ones(W), x, x ** 2, x ** 3], axis=1)   # [W, 4]
        P = np.linalg.pinv(X)                                    # [4, W]
        sw = np.lib.stride_tricks.sliding_window_view(prices, W)  # [n-W+1, W]
        mu = sw.mean(axis=1, keepdims=True)
        sd = sw.std(axis=1, keepdims=True) + 1e-9
        yn = (sw - mu) / sd
        C = yn @ P.T                                             # [n-W+1, 4]
        fit = C @ X.T                                            # reconstruction
        resid = np.sqrt(((yn - fit) ** 2).mean(axis=1))
        d1 = C[:, 1] + 2 * C[:, 2] + 3 * C[:, 3]
        d2 = 2 * C[:, 2] + 6 * C[:, 3]
        # inflection recency: bars since sign(d2) changed
        flip = np.r_[True, np.sign(d2[1:]) != np.sign(d2[:-1])]
        idx = np.arange(len(d2))
        last_flip = np.maximum.accumulate(np.where(flip, idx, 0))
        recency = np.log1p(idx - last_flip)
        block = np.column_stack([C, d1, d2, resid, recency])     # [n-W+1, 8]
        pad = np.zeros((W - 1, block.shape[1]))
        feats.append(np.vstack([pad, block]))
    return np.hstack(feats).astype(np.float32)                   # [n, 8*len(windows)]


def probe_cubic(args, device):
    """User proposal: detect-first with cubic-regression geometry features.
    Grid: features {cubic-only, v2-only, v2+cubic} x label window {125s, 300s,
    900s}, LODO across days, logistic head, shuffled null per label window."""
    days = [d.strip() for d in args.days.split(',')]
    Xv2, Xcu, day_of, ts_all = [], [], [], []
    for di, day in enumerate(days):
        env = build_env([day])
        v2d, l0d, macd, todd, tsd = prefetch_day_tensors(env, device)
        env2 = build_env([day])
        prices, ts_p = collect_prices(env2)
        assert len(ts_p) == len(tsd) and ts_p[0] == tsd[0], "price/feature misalignment"
        Xv2.append(torch.cat([v2d.reshape(v2d.shape[0], -1), l0d, macd, todd], dim=1))
        Xcu.append(torch.from_numpy(cubic_features(prices)).to(device))
        day_of.append(torch.full((len(ts_p),), di, device=device))
        ts_all.append(tsd)
        log(f"[C] {day}: {len(ts_p)} bars prepared")
    Xv2 = torch.cat(Xv2)
    Xcu = torch.cat(Xcu)
    D = torch.cat(day_of)

    def lodo(X, Y, tag):
        aucs = []
        for di, day in enumerate(days):
            tr, te = D != di, D == di
            mu = X[tr].mean(0, keepdim=True)
            sd = X[tr].std(0, keepdim=True).clamp_min(1e-6)
            torch.manual_seed(0)
            net = torch.nn.Linear(X.shape[1], 1).to(device)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            ytr = Y[tr]
            pw = ((ytr == 0).sum() / ytr.sum().clamp_min(1)).reshape(1)
            Xtr = (X[tr] - mu) / sd
            for _ in range(300):
                opt.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    net(Xtr).squeeze(-1), ytr, pos_weight=pw)
                loss.backward()
                opt.step()
            with torch.no_grad():
                s = net((X[te] - mu) / sd).squeeze(-1).cpu().numpy()
            a, n1, _ = auc_score(s, Y[te].cpu().numpy())
            aucs.append(a)
        m = float(np.mean(aucs))
        log(f"[C:{tag}] LODO AUC mean {m:.4f} (min {np.min(aucs):.4f}, "
            f"max {np.max(aucs):.4f})")
        return m

    for window_s, wtag in [(125, '125s'), (300, '5min'), (900, '15min')]:
        Y = torch.cat([
            torch.from_numpy(turn_labels_for_ts(ts_all[di], [days[di]], window_s)).to(device)
            for di in range(len(days))]).float()
        log(f"[C] --- label: turn within {wtag} (pos rate {100*float(Y.mean()):.1f}%) ---")
        m_cu = lodo(Xcu, Y, f'{wtag}:cubic-only')
        m_v2 = lodo(Xv2, Y, f'{wtag}:v2-only')
        m_bo = lodo(torch.cat([Xv2, Xcu], dim=1), Y, f'{wtag}:v2+cubic')
        # shuffled null on the combined set (worst-case leak check)
        rng = np.random.default_rng(0)
        Yn = Y.clone()
        tr = D != 0
        perm = torch.from_numpy(rng.permutation(int(tr.sum()))).to(device)
        Yn[tr] = Y[tr][perm]
        mu = Xcu[tr].mean(0, keepdim=True); sd = Xcu[tr].std(0, keepdim=True).clamp_min(1e-6)
        torch.manual_seed(0)
        net = torch.nn.Linear(Xcu.shape[1], 1).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        ytr = Yn[tr]; pw = ((ytr == 0).sum() / ytr.sum().clamp_min(1)).reshape(1)
        Xtr = (Xcu[tr] - mu) / sd
        for _ in range(300):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(
                net(Xtr).squeeze(-1), ytr, pos_weight=pw).backward()
            opt.step()
        with torch.no_grad():
            s = net((Xcu[D == 0] - mu) / sd).squeeze(-1).cpu().numpy()
        a, _, _ = auc_score(s, Y[D == 0].cpu().numpy())
        log(f"[C:{wtag}:null] shuffled-label cubic AUC = {a:.4f}")


def probe_model(args, device):
    days = [d.strip() for d in args.days.split(',')]
    model = MambaRLTradingNetwork().to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'] if 'model' in ck else ck)
    model.eval()
    log(f"[A] checkpoint {args.ckpt} (epoch {ck.get('epoch', '?')})")

    per_day = {}
    for day in days:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        env = build_env([day])
        v2d, l0d, macd, todd, tsd = prefetch_day_tensors(env, device)
        ledger_day = torch.zeros(v2d.shape[0], 4, dtype=torch.float32, device=device)
        env.reset()
        t = env.seq_len - 1
        states = None
        scores, labels, inpos = [], [], []
        done = False
        with torch.no_grad():
            while not done:
                assert tsd[t] == env.current_bar.timestamp
                ledger_day[t].copy_(torch.from_numpy(env.ledger_state_vec()))
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    e_l, x_l, _, states = model.forward_step(
                        v2d[t].view(1, 8, 1, 52), l0d[t].view(1, 1, 1),
                        ledger_day[t].view(1, 1, 4), macd[t].view(1, 1, 261),
                        todd[t].view(1, 1, 4), states)
                is_flat = env.ledger.is_flat
                if is_flat:
                    a = int(torch.distributions.Categorical(
                        probs=torch.softmax(e_l, dim=-1)).sample().item())
                    env_action = a
                else:
                    a = int(torch.distributions.Bernoulli(
                        probs=torch.sigmoid(x_l.squeeze(-1))).sample().item())
                    env_action = 3 if a == 1 else 0
                _, _, done, info = env.step(env_action, 0.0)
                scores.append(float(x_l.float().squeeze()))
                labels.append(1 if info.get('turn_imminent', 0.0) else 0)
                inpos.append(not is_flat)
                if info.get('session_reset', False):
                    states = None
                t += 1
        s, l, ip = np.array(scores), np.array(labels), np.array(inpos)
        a_all, n1, n0 = auc_score(s, l)
        a_ip, n1i, n0i = auc_score(s[ip], l[ip]) if ip.sum() > 100 else (float('nan'), int(ip.sum()), 0)
        per_day[day] = (a_all, n1, n0, a_ip, n1i)
        log(f"[A] {day}: AUC(all bars) = {a_all:.4f}  (pos {n1}/{n1+n0})"
            f" | AUC(in-position, train dist) = {a_ip:.4f} (n={n1i + n0i if not np.isnan(a_ip) else int(ip.sum())})")

    aucs = [v[0] for v in per_day.values() if not np.isnan(v[0])]
    log(f"[A] mean AUC across {len(aucs)} days: {np.mean(aucs):.4f} "
        f"(min {np.min(aucs):.4f}, max {np.max(aucs):.4f})")
    gap = np.mean(aucs) - 0.5
    verdict = 'REAL' if gap >= 0.10 else ('CONDITIONAL' if gap >= 0.05 else 'NOISE')
    log(f"[A] AUC-0.5 gap = {gap:+.4f} -> {verdict} (house bar: >=0.10 real, <0.05 noise)")


def probe_signal(args, device):
    days = [d.strip() for d in args.days.split(',')]
    # Features: 682 action-independent dims per bar, per day
    feats, labs, day_of = [], [], []
    for di, day in enumerate(days):
        env = build_env([day])
        v2d, l0d, macd, todd, tsd = prefetch_day_tensors(env, device)
        x = torch.cat([v2d.reshape(v2d.shape[0], -1), l0d, macd, todd], dim=1)
        y = turn_labels_for_ts(tsd, [day])
        feats.append(x)
        labs.append(torch.from_numpy(y).to(device))
        day_of.append(torch.full((x.shape[0],), di, device=device))
        log(f"[B] {day}: {x.shape[0]} bars, {int(y.sum())} turn bars ({100*y.mean():.1f}%)")
    X = torch.cat(feats)
    Y = torch.cat(labs).float()
    D = torch.cat(day_of)

    def fit_eval(train_mask, test_mask, y_train, hidden=0, epochs=300, seed=0):
        torch.manual_seed(seed)
        mu = X[train_mask].mean(0, keepdim=True)
        sd = X[train_mask].std(0, keepdim=True).clamp_min(1e-6)
        Xtr = (X[train_mask] - mu) / sd
        Xte = (X[test_mask] - mu) / sd
        if hidden:
            net = torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], hidden), torch.nn.SiLU(),
                torch.nn.Linear(hidden, 1)).to(device)
        else:
            net = torch.nn.Linear(X.shape[1], 1).to(device)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        pw = ((y_train == 0).sum() / y_train.sum().clamp_min(1)).reshape(1)
        for _ in range(epochs):
            opt.zero_grad()
            out = net(Xtr).squeeze(-1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                out, y_train, pos_weight=pw)
            loss.backward()
            opt.step()
        with torch.no_grad():
            s = net(Xte).squeeze(-1).cpu().numpy()
        return s

    for tag, hidden in [('logistic', 0), ('mlp64', 64)]:
        aucs = []
        for di, day in enumerate(days):
            tr, te = D != di, D == di
            s = fit_eval(tr, te, Y[tr], hidden=hidden)
            a, n1, n0 = auc_score(s, Y[te].cpu().numpy())
            aucs.append(a)
            log(f"[B:{tag}] holdout {day}: AUC = {a:.4f} (pos {n1})")
        gap = np.mean(aucs) - 0.5
        verdict = 'REAL' if gap >= 0.10 else ('CONDITIONAL' if gap >= 0.05 else 'NOISE')
        log(f"[B:{tag}] mean LODO AUC = {np.mean(aucs):.4f} "
            f"(min {np.min(aucs):.4f}) | gap {gap:+.4f} -> {verdict}")

    # Shuffled-label null (logistic, 2 shuffles, first day held out)
    for k in range(2):
        rng = np.random.default_rng(k)
        Yn = Y.clone()
        tr = D != 0
        perm = torch.from_numpy(rng.permutation(int(tr.sum()))).to(device)
        Yn[tr] = Y[tr][perm]
        s = fit_eval(tr, D == 0, Yn[tr], hidden=0, seed=k)
        a, _, _ = auc_score(s, Y[D == 0].cpu().numpy())
        log(f"[B:null{k}] shuffled-label AUC on holdout = {a:.4f} (should be ~0.5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', choices=['model', 'signal', 'cubic'], required=True)
    ap.add_argument('--days', type=str,
                    default="2024_02_20,2024_02_21,2024_02_22,2024_02_23,2024_02_26,2024_02_27")
    ap.add_argument('--ckpt', type=str, default='mamba_rl_seq_checkpoint_ep25.pth')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', type=str, default='')
    args = ap.parse_args()
    device = torch.device('cuda')

    if args.probe == 'model':
        probe_model(args, device)
    elif args.probe == 'cubic':
        probe_cubic(args, device)
    else:
        probe_signal(args, device)

    out = args.out or os.path.join(REPORT_DIR, f'probe_turns_{args.probe}.txt')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {out}')


if __name__ == '__main__':
    main()
