"""Differentiable associative scan for diagonal SSM recurrences.

Solves h_t = A_t * h_{t-1} + b_t (elementwise A, as in Mamba's selective
scan) for a whole window in O(log L) tensor sweeps, with support for an
initial state h_0 — the two things the fused mamba-ssm path can't give us
for TRAINING (its Mamba1 kernel has no initial-state argument and its
step kernels are inference-only; see reports/perf/mamba_ssm_probe.txt).

Exactness: identical recurrence to the sequential python loop in exact
arithmetic; float reassociation gives ~1e-6 fp32 / ~1e-3 bf16 drift (same
class as torch.compile refusion). Verified by tools/test_seq_equivalence.py.
"""
import torch


def associative_ssm_scan(A: torch.Tensor, b: torch.Tensor,
                         h0: torch.Tensor = None) -> torch.Tensor:
    """Inclusive scan of h_t = A_t * h_{t-1} + b_t along dim 1.

    Args:
        A:  [B, L, D, N] elementwise transition factors (Mamba: exp(dt*A) in (0,1))
        b:  [B, L, D, N] input terms (Mamba: dt * x ⊗ B_t)
        h0: [B, D, N] optional initial state (folded into step 0)

    Returns:
        h: [B, L, D, N] where h[:, t] is the state AFTER absorbing bar t.

    Uses Hillis–Steele doubling with the affine-composition monoid
    (A2, b2) ∘ (A1, b1) = (A2*A1, A2*b1 + b2). O(L log L) work — trivial at
    Mamba sizes (L=500, D=256, N=16 ≈ 2M elements) and fully differentiable.
    """
    _, L, _, _ = A.shape
    if h0 is not None:
        b = torch.cat([(b[:, :1] + A[:, :1] * h0.unsqueeze(1)), b[:, 1:]], dim=1)

    acc_A, acc_b = A, b
    offset = 1
    while offset < L:
        # Compose element t with the accumulated element at t-offset.
        # Elements t < offset compose with the identity (A=1, b=0).
        pad_A = torch.ones_like(acc_A[:, :offset])
        pad_b = torch.zeros_like(acc_b[:, :offset])
        shifted_A = torch.cat([pad_A, acc_A[:, :-offset]], dim=1)
        shifted_b = torch.cat([pad_b, acc_b[:, :-offset]], dim=1)
        acc_b = acc_A * shifted_b + acc_b
        acc_A = acc_A * shifted_A
        offset *= 2
    return acc_b
