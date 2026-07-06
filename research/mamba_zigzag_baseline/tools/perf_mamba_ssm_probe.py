"""Probe: is the mamba-ssm fused path usable for the Mamba RL TRAINING loop?

Checks, with evidence:
  1. mamba_ssm + causal_conv1d import and run on this WSL2 + RTX 3060 box.
  2. The fused per-bar recurrence (Mamba.step / selective_state_update) is
     INFERENCE-ONLY: its output does not carry grad, so it cannot replace the
     differentiable PureMambaBlock scan inside the TBPTT window.
  3. The fused PARALLEL scan (full-sequence forward) IS differentiable, and
     its speed vs the pure-PyTorch python scan at L=1 (what training does
     per bar) and at L=500 (what a sequence-restructure would use).

Writes results to reports/perf/mamba_ssm_probe.txt.
"""
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'pipeline'))

OUT = os.path.join(_REPO, 'research', 'mamba_zigzag_baseline', 'reports', 'perf',
                   'mamba_ssm_probe.txt')

lines = []


def log(s):
    print(s)
    lines.append(s)


def bench(fn, n=200, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000  # ms


def main():
    device = torch.device('cuda')
    d_model, d_state, d_conv, expand = 128, 16, 4, 2

    # --- 1. import probe ---
    # transformers >= 5 removed the decode-output classes mamba_ssm.utils.generation
    # imports at module load. Stub them: the fused KERNELS under probe don't use them.
    try:
        import transformers.generation as _tg
        for _name in ('GreedySearchDecoderOnlyOutput', 'SampleDecoderOnlyOutput',
                      'TextStreamer'):
            if not hasattr(_tg, _name):
                setattr(_tg, _name, type(_name, (), {}))
                log(f'[1] NOTE: stubbed transformers.generation.{_name} '
                    f'(removed in transformers 5.x; unused by the kernels)')
    except ImportError:
        pass
    try:
        from mamba_ssm import Mamba
        log('[1] mamba_ssm import: OK')
    except Exception as e:
        log(f'[1] mamba_ssm import FAILED: {type(e).__name__}: {e}')
        _flush()
        return
    try:
        import causal_conv1d  # noqa: F401
        log('[1] causal_conv1d import: OK')
    except Exception as e:
        log(f'[1] causal_conv1d import FAILED: {type(e).__name__}: {e}')

    m = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).to(device)

    # --- 2. autograd probe on step() ---
    x1 = torch.randn(1, 1, d_model, device=device, requires_grad=True)
    conv_state = torch.zeros(1, m.d_inner, m.d_conv, device=device)
    ssm_state = torch.zeros(1, m.d_inner, m.d_state, device=device)
    out, cs, ss = m.step(x1, conv_state.clone(), ssm_state.clone())
    log(f'[2] Mamba.step() output requires_grad = {out.requires_grad} '
        f'(input requires_grad = {x1.requires_grad})')
    if out.requires_grad:
        try:
            out.sum().backward()
            g = x1.grad
            log(f'[2] step() backward ran; input grad norm = '
                f'{float(g.norm()) if g is not None else None}')
        except Exception as e:
            log(f'[2] step() backward RAISED: {type(e).__name__}: {str(e)[:200]}')
    else:
        log('[2] => VERDICT: fused step API severs autograd; cannot replace the '
            'differentiable per-bar scan inside the TBPTT window.')

    # --- 3. parallel-scan autograd + speed ---
    L = 500
    xs = torch.randn(1, L, d_model, device=device, requires_grad=True)
    y = m(xs)
    log(f'[3] Mamba parallel forward (L={L}) output requires_grad = {y.requires_grad}')
    if y.requires_grad:
        y.sum().backward()
        log(f'[3] parallel backward OK; input grad norm = {float(xs.grad.norm()):.4f}')

    from mamba_rl_network import PureMambaBlock
    p = PureMambaBlock(d_model=d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand).to(device)

    x_l1 = torch.randn(1, 1, d_model, device=device)
    h0 = torch.zeros(1, p.d_inner, d_state, device=device)
    cs0 = torch.zeros(1, m.d_inner, m.d_conv, device=device)
    ss0 = torch.zeros(1, m.d_inner, m.d_state, device=device)

    with torch.no_grad():
        t_pure_l1 = bench(lambda: p(x_l1, h0))
        t_step = bench(lambda: m.step(x_l1, cs0, ss0))
        x_l500 = torch.randn(1, L, d_model, device=device)
        t_pure_l500 = bench(lambda: p(x_l500), n=20, warmup=3)
        t_par_l500 = bench(lambda: m(x_l500), n=20, warmup=3)

    log('')
    log('[3] Speed (no_grad, ms per call, one block):')
    log(f'    PureMambaBlock L=1 python scan : {t_pure_l1:8.3f} ms')
    log(f'    Mamba.step() fused L=1        : {t_step:8.3f} ms')
    log(f'    PureMambaBlock L=500 scan     : {t_pure_l500:8.3f} ms')
    log(f'    Mamba parallel L=500 fused    : {t_par_l500:8.3f} ms')
    _flush()


def _flush():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nWritten to {OUT}')


if __name__ == '__main__':
    main()
