"""Gates 1+2 of docs/JULES_SEQUENCE_WINDOW_TRAINING.md.

1. associative_ssm_scan == sequential python recurrence (fp32, random, with
   and without initial state), tol 1e-5.
2. MambaRLTradingNetwork.forward_sequence over a W-bar window == W chained
   forward_step calls with carried states, fp32 tol 1e-4 (bf16 documented).
3. Gradient sanity: backward through forward_sequence produces finite,
   nonzero grads on all parameters.

Writes results to reports/perf/seq_equivalence_test.txt. Exit 1 on failure.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'pipeline'))

from mamba_scan import associative_ssm_scan  # noqa: E402
from mamba_rl_network import MambaRLTradingNetwork  # noqa: E402

OUT = os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'reports', 'perf',
                   'seq_equivalence_test.txt')
lines = []
failed = False


def log(s):
    print(s)
    lines.append(s)


def check(name, delta, tol):
    global failed
    ok = delta < tol
    log(f'{"[PASS]" if ok else "[FAIL]"} {name}: max|delta| = {delta:.3e} (tol {tol:g})')
    if not ok:
        failed = True


def main():
    torch.manual_seed(0)
    device = torch.device('cuda')

    # ── Gate 1: scan vs sequential loop ──
    B, L, D, N = 2, 517, 64, 16  # non-power-of-2 L on purpose
    A = torch.rand(B, L, D, N, device=device) * 0.98 + 0.01   # (0,1) like exp(dt*A)
    b = torch.randn(B, L, D, N, device=device)
    h0 = torch.randn(B, D, N, device=device)

    for name, init in [('scan_no_h0', None), ('scan_with_h0', h0)]:
        h = torch.zeros(B, D, N, device=device) if init is None else init.clone()
        ref = []
        for t in range(L):
            h = A[:, t] * h + b[:, t]
            ref.append(h)
        ref = torch.stack(ref, dim=1)
        got = associative_ssm_scan(A, b, init)
        check(name, (ref - got).abs().max().item(), 1e-5)

    # ── Gate 2: forward_sequence vs chained forward_step (fp32) ──
    torch.manual_seed(1)
    model = MambaRLTradingNetwork().to(device).float().eval()
    W = 133
    v2 = torch.randn(1, 8, W, 52, device=device)
    l0 = torch.randn(1, W, 1, device=device)
    ledg = torch.randn(1, W, 4, device=device)
    macro = torch.randn(1, W, 261, device=device)
    tod = torch.randn(1, W, 4, device=device)

    with torch.no_grad():
        e_seq, x_seq, v_seq, _ = model.forward_sequence(v2, l0, ledg, macro, tod)
        states = None
        e_st, x_st, v_st = [], [], []
        for t in range(W):
            e, xx, v, states = model.forward_step(
                v2[:, :, t:t + 1], l0[:, t:t + 1], ledg[:, t:t + 1],
                macro[:, t:t + 1], tod[:, t:t + 1], states)
            e_st.append(e); x_st.append(xx); v_st.append(v)
        e_st = torch.stack(e_st, dim=1)
        x_st = torch.stack(x_st, dim=1)
        v_st = torch.stack(v_st, dim=1)

    check('fwd_seq_vs_step_entry_fp32', (e_seq - e_st).abs().max().item(), 1e-4)
    check('fwd_seq_vs_step_exit_fp32', (x_seq - x_st).abs().max().item(), 1e-4)
    check('fwd_seq_vs_step_value_fp32', (v_seq - v_st).abs().max().item(), 1e-4)

    # ── Gate 2b: same under bf16 autocast (documented, looser tol) ──
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        e_seq_b, _, v_seq_b, _ = model.forward_sequence(v2, l0, ledg, macro, tod)
        states = None
        e_st_b, v_st_b = [], []
        for t in range(W):
            e, xx, v, states = model.forward_step(
                v2[:, :, t:t + 1], l0[:, t:t + 1], ledg[:, t:t + 1],
                macro[:, t:t + 1], tod[:, t:t + 1], states)
            e_st_b.append(e); v_st_b.append(v)
        e_st_b = torch.stack(e_st_b, dim=1)
        v_st_b = torch.stack(v_st_b, dim=1)
    log(f'[INFO] bf16 autocast entry-logit drift seq-vs-step: '
        f'{(e_seq_b.float() - e_st_b.float()).abs().max().item():.3e} '
        f'(documented, not gated)')

    # ── Gate 3: gradient sanity through forward_sequence ──
    model.train()
    e_seq, x_seq, v_seq, _ = model.forward_sequence(v2, l0, ledg, macro, tod)
    loss = e_seq.square().mean() + x_seq.square().mean() + v_seq.square().mean()
    loss.backward()
    bad = [n for n, p in model.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    if bad:
        log(f'[FAIL] grad sanity: no/invalid/zero grads for: {bad}')
        globals()['failed'] = True
    else:
        log('[PASS] grad sanity: all parameters have finite nonzero grads')

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {OUT}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
