"""GPTQ, GPTQ-intrinsic LoRA, OLrC and Bid-Up.

Reference implementations of the four layer-wise routines this work builds on,
written against a Hessian ``H = X^T X`` rather than the calibration matrix
itself, which is how open-source PTQ codebases keep peak memory down.

  gptq                 Frantar et al. (2022), Alg. 1 of Zhang & Saab.
  gptq_intrinsic_lora  Zhang & Saab, Alg. 3: run GPTQ for N steps on the
                       augmented Hessian [[H, H V_r], [V_r^T H, V_r^T H V_r]]
                       so the trailing r full-precision rows absorb the
                       rounding error as they go.
  olrc                 Zhang & Saab, Alg. 2: the GSVD-optimal rank-r
                       compensation for a fixed Q.
  bid_up               Zhang & Saab, Alg. 4: fixed-grid coordinate descent on Q.

``block_size`` reproduces GPTQ's lazy batch updates. It is a compute-to-memory
optimisation only: the arithmetic is identical to block_size = 1.
"""

from __future__ import annotations

import torch

from .linalg import _psd_power, cholesky_tri_factor, safe_eigh, stable_tri_factor
from .quant import Grid, make_grid, quantize

__all__ = [
    "gptq",
    "gptq_intrinsic_lora",
    "gilora_factor",
    "gilora_with_factor",
    "olrc",
    "bid_up",
    "layer_error",
]


def layer_error(H: torch.Tensor, W: torch.Tensor, What: torch.Tensor) -> float:
    """``||X W - X What||_F^2`` evaluated through ``H = X^T X``."""
    D = (W - What).to(H.dtype)
    return torch.einsum("ij,ik,kj->", D, H, D).item()


def _tri_factor(H: torch.Tensor, damp: float, robust: bool) -> torch.Tensor:
    if not robust:
        return cholesky_tri_factor(H, damp)
    return stable_tri_factor(H, damp)


def _error_diffusion(
    Wfull: torch.Tensor,
    Psi: torch.Tensor,
    grid: Grid,
    n_quant: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared GPTQ inner loop.

    Quantises the first ``n_quant`` rows of ``Wfull`` (shape ``(n_total, N')``)
    against ``Psi``, the lower-triangular factor of the damped inverse Hessian,
    and returns ``(Q, tail)`` where ``tail`` is the untouched trailing
    ``n_total - n_quant`` rows *after* they have absorbed the diffused error.

    For plain GPTQ ``n_quant == n_total`` and ``tail`` is empty. For
    GPTQ-intrinsic LoRA ``n_total = N + r`` and ``tail`` is the error
    absorption factor ``R``.
    """
    n_total, ncol = Wfull.shape
    Wcur = Wfull.clone()
    Q = torch.zeros(n_quant, ncol, device=Wfull.device, dtype=Wfull.dtype)

    for start in range(0, n_quant, block_size):
        stop = min(start + block_size, n_quant)
        # Error accumulated inside this block, to be applied to everything after
        # it in one matmul (GPTQ's lazy batch update).
        Eblk = torch.zeros(stop - start, ncol, device=Wfull.device, dtype=Wfull.dtype)

        for t in range(start, stop):
            w_t = Wcur[t]
            q_t = quantize(w_t, grid)
            Q[t] = q_t
            # (q_t - w_t) / Psi_tt is the scalar multiplying column t of Psi.
            err = (q_t - w_t) / Psi[t, t]
            Eblk[t - start] = err
            if t + 1 < stop:
                Wcur[t + 1 : stop] += torch.outer(Psi[t + 1 : stop, t], err)

        if stop < n_total:
            Wcur[stop:] += Psi[stop:, start:stop] @ Eblk

    tail = Wcur[n_quant:]
    return Q, tail


def gptq(
    H: torch.Tensor,
    W: torch.Tensor,
    bits: int,
    beta: float = 1.0,
    damp_frac: float = 0.01,
    robust: bool = True,
    block_size: int = 128,
) -> tuple[torch.Tensor, Grid]:
    """Quantise ``W`` (shape ``(N, N')``) with GPTQ.

    ``robust=False`` selects the reference Cholesky-of-the-inverse route
    (plain "GPTQ"); ``robust=True`` selects the eigendecomposition-plus-QR
    route ("GPTQ*").
    """
    N = W.shape[0]
    damp = damp_frac * H.diagonal().mean()
    Psi = _tri_factor(H, damp, robust)
    grid = make_grid(W, bits, beta)
    Q, _ = _error_diffusion(W, Psi, grid, n_quant=N, block_size=block_size)
    return Q, grid


def gptq_intrinsic_lora(
    H: torch.Tensor,
    W: torch.Tensor,
    bits: int,
    rank: int,
    beta: float = 1.0,
    damp_frac: float = 0.01,
    block_size: int = 128,
    L_bits: int | None = None,
    eigvecs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Grid]:
    """GPTQ-intrinsic LoRA. Returns ``(Q, L, R, grid)`` with ``W ~ Q + L @ R``.

    ``L = V_r``, the top-``rank`` eigenvectors of ``H`` (equivalently the top
    right singular vectors of ``X``). Pass ``eigvecs`` to reuse a cached
    eigendecomposition -- this routine is called once per budget point per
    layer, and the eigendecomposition does not depend on the budget.

    ``L_bits`` quantises the feature-extraction factor *before* forming the
    augmented Hessian, so the pass absorbs that rounding error too.
    """
    N, ncol = W.shape
    if rank <= 0:
        Q, grid = gptq(H, W, bits, beta, damp_frac, robust=True, block_size=block_size)
        empty_L = torch.zeros(N, 0, device=W.device, dtype=W.dtype)
        empty_R = torch.zeros(0, ncol, device=W.device, dtype=W.dtype)
        return Q, empty_L, empty_R, grid

    L, Psi = gilora_factor(H, rank, damp_frac, eigvecs, L_bits)
    Q, R, grid = gilora_with_factor(W, L, Psi, bits, beta, block_size)
    return Q, L, R, grid


def gilora_factor(
    H: torch.Tensor,
    rank: int,
    damp_frac: float = 0.01,
    eigvecs: torch.Tensor | None = None,
    L_bits: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(L, Psi)`` pair GPTQ-intrinsic LoRA needs, hoisted out of the loop.

    Neither depends on the bit-width -- only on ``H`` and ``rank`` -- so a
    profile that sweeps bits at fixed rank computes this once instead of once
    per bit-width. That is the difference between minutes and hours over a full
    model, because the factorisation is an eigendecomposition of an
    ``(N+r) x (N+r)`` matrix while the quantisation pass itself is cheap.
    """
    N = H.shape[0]
    if rank <= 0:
        damp = damp_frac * H.diagonal().mean()
        return (
            torch.zeros(N, 0, device=H.device, dtype=H.dtype),
            stable_tri_factor(H, damp),
        )
    if eigvecs is None:
        _, V = safe_eigh(H)
        eigvecs = V.flip(-1)
    L = eigvecs[:, :rank].contiguous()
    if L_bits is not None:
        L = quantize(L, make_grid(L, L_bits, beta=1.0))
    HL = H @ L
    Haug = torch.empty(N + rank, N + rank, device=H.device, dtype=H.dtype)
    Haug[:N, :N] = H
    Haug[:N, N:] = HL
    Haug[N:, :N] = HL.transpose(0, 1)
    Haug[N:, N:] = L.transpose(0, 1) @ HL
    Haug = 0.5 * (Haug + Haug.transpose(0, 1))
    damp = damp_frac * Haug.diagonal().mean()
    return L, stable_tri_factor(Haug, damp)


def gilora_with_factor(
    W: torch.Tensor,
    L: torch.Tensor,
    Psi: torch.Tensor,
    bits: int,
    beta: float = 1.0,
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, Grid]:
    """Run the quantisation pass against a factorisation from `gilora_factor`."""
    N, ncol = W.shape
    rank = L.shape[1]
    Waug = torch.zeros(N + rank, ncol, device=W.device, dtype=W.dtype)
    Waug[:N] = W
    grid = make_grid(W, bits, beta)
    Q, R = _error_diffusion(Waug, Psi, grid, n_quant=N, block_size=block_size)
    return Q, R, grid


def olrc(
    H: torch.Tensor,
    W: torch.Tensor,
    Q: torch.Tensor,
    rank: int,
    damp_frac: float = 0.01,
    oversample: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal rank-``rank`` compensation for a fixed ``Q`` (GSVD closed form).

    Solves ``min_{L,R} ||X(W - Q) - X L R||_F^2`` as
    ``L R = H^{-1/2} T_r( H^{1/2} (W - Q) )``, with the same damping as GPTQ so
    that alternating OLrC and Bid-Up act on one common regularised objective.

    Uses `torch.svd_lowrank` at rank ``rank + oversample``: a full SVD is not
    only wasted work but is exactly the call that fails to converge on this
    cluster's older silicon.
    """
    if rank <= 0:
        N, ncol = W.shape
        return (
            torch.zeros(N, 0, device=W.device, dtype=W.dtype),
            torch.zeros(0, ncol, device=W.device, dtype=W.dtype),
        )
    n = H.shape[0]
    damp = damp_frac * H.diagonal().mean()
    Hd = H + damp * torch.eye(n, device=H.device, dtype=H.dtype)
    Hh = _psd_power(Hd, 0.5)
    Hinv_h = _psd_power(Hd, -0.5)

    M = Hh @ (W - Q).to(H.dtype)
    q = min(rank + oversample, min(M.shape))
    U, S, Vh = torch.svd_lowrank(M, q=q, niter=4)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:, :rank]
    L = Hinv_h @ (U * S.unsqueeze(0))
    R = Vh.transpose(0, 1)
    return L.to(W.dtype), R.to(W.dtype)


def bid_up(
    H: torch.Tensor,
    Wt: torch.Tensor,
    Q0: torch.Tensor,
    grid: Grid,
    damp_frac: float = 0.01,
    sweeps: int = 1,
) -> torch.Tensor:
    """Fixed-grid coordinate refinement of ``Q`` against residual ``Wt``.

    Each coordinate is set to the exact minimiser over the *fixed* alphabet
    given every other coordinate, so the layer-wise error is non-increasing by
    construction. Updates are in place: when row ``i`` is visited, rows below
    already hold their new values and rows above still hold the old ones.
    """
    n = H.shape[0]
    damp = damp_frac * H.diagonal().mean()
    Hd = H + damp * torch.eye(n, device=H.device, dtype=H.dtype)
    d = Hd.diagonal().clamp_min(torch.finfo(Hd.dtype).tiny)
    C = Hd - torch.diag(Hd.diagonal())

    Q = Q0.clone()
    HW = Hd @ Wt.to(Hd.dtype)  # (N, N'), row i is <H_i, wtilde>
    CQ = C @ Q.to(Hd.dtype)  # refreshed incrementally below
    for _ in range(sweeps):
        for i in range(n):
            target = (HW[i] - CQ[i]) / d[i]
            qi_new = quantize(target.to(Q.dtype), grid)
            delta = (qi_new - Q[i]).to(Hd.dtype)
            if torch.any(delta != 0):
                CQ += torch.outer(C[:, i], delta)
                Q[i] = qi_new
    return Q
