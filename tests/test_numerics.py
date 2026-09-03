"""Correctness of the layer-wise routines, against closed forms where they exist."""

import numpy as np
import pytest
import torch

from qlr.gptq import bid_up, gptq, gptq_intrinsic_lora, layer_error, olrc
from qlr.linalg import cholesky_tri_factor, safe_eigh, stable_tri_factor
from qlr.quant import make_grid, quantize


@pytest.fixture(scope="module")
def problem():
    torch.manual_seed(0)
    m, N, Np = 384, 64, 32
    U = torch.linalg.qr(torch.randn(m, N, dtype=torch.float64))[0]
    V = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64))[0]
    sv = torch.cat([torch.logspace(2, 1, 8, dtype=torch.float64),
                    torch.full((N - 8,), 0.3, dtype=torch.float64)])
    X = U @ torch.diag(sv) @ V.T
    W = torch.randn(N, Np, dtype=torch.float64) * 0.05
    return X, X.T @ X, W, sv


def test_tri_factors_agree_and_are_lower_triangular(problem):
    _, H, _, _ = problem
    damp = 0.01 * H.diagonal().mean()
    P1 = cholesky_tri_factor(H, damp)
    P2 = stable_tri_factor(H, damp)
    target = torch.linalg.inv(H + damp * torch.eye(H.shape[0], dtype=H.dtype))
    for P in (P1, P2):
        assert torch.allclose(P, P.tril())
        assert torch.allclose(P @ P.T, target, atol=1e-10)
    assert torch.allclose(P1, P2, atol=1e-10)


def test_gptq_beats_round_to_nearest(problem):
    _, H, W, _ = problem
    for bits in (2, 3, 4):
        grid = make_grid(W, bits)
        rtn = layer_error(H, W, quantize(W, grid))
        Q, _ = gptq(H, W, bits)
        assert layer_error(H, W, Q) < rtn


@pytest.mark.parametrize("block_size", [1, 7, 128])
def test_lazy_batching_is_exact(problem, block_size):
    _, H, W, _ = problem
    ref, _ = gptq(H, W, 3, block_size=1)
    got, _ = gptq(H, W, 3, block_size=block_size)
    assert torch.equal(ref, got)
    Qr, _, Rr, _ = gptq_intrinsic_lora(H, W, 3, 8, block_size=1)
    Qb, _, Rb, _ = gptq_intrinsic_lora(H, W, 3, 8, block_size=block_size)
    assert torch.equal(Qr, Qb)
    assert torch.allclose(Rr, Rb, atol=1e-12)


def test_olrc_attains_the_gsvd_optimum(problem):
    _, H, W, _ = problem
    from qlr.linalg import _psd_power

    Q, _ = gptq(H, W, 3)
    damp = 0.01 * H.diagonal().mean()
    Hd = H + damp * torch.eye(H.shape[0], dtype=H.dtype)
    s = torch.linalg.svdvals(_psd_power(Hd, 0.5) @ (W - Q))
    for r in (4, 8, 16):
        L, R = olrc(H, W, Q, r)
        D = W - Q - L @ R
        got = torch.einsum("ij,ik,kj->", D, Hd, D).item()
        want = float((s[r:] ** 2).sum())
        assert got == pytest.approx(want, rel=1e-3)


def test_gilora_respects_its_upper_bound(problem):
    """Corollary 3.9 of Zhang & Saab, in the infinite-alphabet regime."""
    _, H, W, sv = problem
    N = W.shape[0]
    bits = 8  # enough that clipping never binds
    for r in (4, 8, 16):
        Q, L, R, grid = gptq_intrinsic_lora(H, W, bits, r)
        lam = 0.01 * torch.cat([H.diagonal(), (L.T @ H @ L).diagonal()]).mean()
        lhs = (
            layer_error(H, W, Q + L @ R)
            + lam * ((W - Q) ** 2).sum()
            + lam * ((L @ R) ** 2).sum()
        )
        rhs = (grid.scale**2).sum() / 4 * ((sv[r:] ** 2).sum() + (N + r) * lam)
        assert float(lhs) <= float(rhs)


def test_gilora_wins_when_the_spectrum_drops(problem):
    """The bound replaces ||X||_F^2 by the rank-r tail, so a sharp drop after r
    is exactly the regime where the intrinsic construction should win."""
    _, H, W, _ = problem
    Q, _ = gptq(H, W, 3)
    L, R = olrc(H, W, Q, 8)
    baseline = layer_error(H, W, Q + L @ R)
    Q2, L2, R2, _ = gptq_intrinsic_lora(H, W, 3, 8)
    assert layer_error(H, W, Q2 + L2 @ R2) < baseline


def test_bid_up_never_increases_the_error(problem):
    _, H, W, _ = problem
    damp = 0.01 * H.diagonal().mean()
    Hd = H + damp * torch.eye(H.shape[0], dtype=H.dtype)
    Q, L, R, grid = gptq_intrinsic_lora(H, W, 3, 8)

    def err(Q, L, R):
        D = W - Q - L @ R
        return torch.einsum("ij,ik,kj->", D, Hd, D).item()

    prev = err(Q, L, R)
    for _ in range(3):
        L, R = olrc(H, W, Q, 8)
        after_olrc = err(Q, L, R)
        assert after_olrc <= prev + 1e-9
        Q = bid_up(H, W - L @ R, Q, grid)
        after_bidup = err(Q, L, R)
        assert after_bidup <= after_olrc + 1e-9
        prev = after_bidup


def test_safe_eigh_survives_a_singular_matrix():
    A = torch.zeros(8, 8, dtype=torch.float64)
    A[0, 0] = 1.0
    w, V = safe_eigh(A)
    assert torch.isfinite(w).all() and torch.isfinite(V).all()
    assert w.max().item() == pytest.approx(1.0)


def test_rank_zero_gilora_reduces_to_gptq(problem):
    _, H, W, _ = problem
    Q, _ = gptq(H, W, 3, robust=True)
    Q0, L0, R0, _ = gptq_intrinsic_lora(H, W, 3, 0)
    assert torch.equal(Q, Q0)
    assert L0.numel() == 0 and R0.numel() == 0


def test_clipping_search_improves_the_weighted_error(problem):
    """search_beta minimises the calibration-weighted error column by column,
    which is exact because the layer-wise objective separates over columns."""
    from qlr.quant import search_beta

    X, H, W, _ = problem
    W = W.clone()
    W[0, ::5] *= 10.0  # outliers: the situation clipping exists for
    for bits in (2, 3, 4):
        beta = search_beta(H, W, bits)
        assert beta.shape == (W.shape[1],)
        e_plain = layer_error(H, W, quantize(W, make_grid(W, bits, 1.0)))
        e_clip = layer_error(H, W, quantize(W, make_grid(W, bits, beta)))
        assert e_clip <= e_plain
        # the choice must be per column, not a single global value
        assert beta.unique().numel() > 1


def test_clipping_search_is_columnwise_optimal(problem):
    """No candidate beats the chosen beta on any single column."""
    from qlr.quant import search_beta

    _, H, W, _ = problem
    cands = (1.0, 0.9, 0.8, 0.7, 0.6)
    beta = search_beta(H, W, 3, candidates=cands)
    chosen_err = None
    errs = {}
    for c in cands:
        D = W - quantize(W, make_grid(W, 3, c))
        errs[c] = ((H @ D) * D).sum(dim=0)
    stacked = torch.stack([errs[c] for c in cands])
    best = stacked.min(dim=0).values
    chosen_err = torch.stack(
        [errs[float(b)][j] for j, b in enumerate(beta.tolist())]
    )
    assert torch.allclose(chosen_err, best)
