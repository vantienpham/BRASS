"""The rotation must be exactly orthogonal, and must leave the spectrum alone."""

import pytest
import torch

from qlr.rotate import Rotation, largest_valid_block


@pytest.mark.parametrize(
    "n,requested,expected",
    [(1024, 1024, 1024), (3072, 1024, 1024), (3072, 4096, 1024), (2048, 128, 128),
     (12288, 4096, 4096), (6144, 8192, 2048), (1024, 1, 1)],
)
def test_block_selection(n, requested, expected):
    """The block must be a power of two dividing n, as large as requested allows."""
    assert largest_valid_block(n, requested) == expected


@pytest.mark.parametrize("n,block", [(256, 256), (384, 128), (512, 64), (768, 256)])
def test_rotation_is_orthogonal_and_matches_dense(n, block):
    torch.manual_seed(0)
    rot = Rotation(n, block, seed=1, dtype=torch.float64)
    theta = rot.left(torch.eye(n, dtype=torch.float64))
    assert torch.allclose(theta.T @ theta, torch.eye(n, dtype=torch.float64), atol=1e-10)

    M = torch.randn(n, 17, dtype=torch.float64)
    K = torch.randn(13, n, dtype=torch.float64)
    assert torch.allclose(rot.left_T(M), theta.T @ M, atol=1e-10)
    assert torch.allclose(rot.left(M), theta @ M, atol=1e-10)
    assert torch.allclose(rot.right(K), K @ theta, atol=1e-10)

    X = torch.randn(4 * n, n, dtype=torch.float64)
    H = X.T @ X
    assert torch.allclose(rot.conjugate_hessian(H), theta.T @ H @ theta, atol=1e-6)


@pytest.mark.parametrize("n,block", [(256, 256), (384, 128), (512, 64)])
def test_spectrum_is_invariant(n, block):
    """Orthogonal conjugation cannot change eigenvalues, so the spectral tail --
    and therefore the entire rank factor h(r) -- is untouched by rotation.
    Rotation acts only on the weight-range statistic S."""
    torch.manual_seed(1)
    rot = Rotation(n, block, seed=2, dtype=torch.float64)
    X = torch.randn(3 * n, n, dtype=torch.float64) @ torch.diag(
        torch.logspace(1, -2, n, dtype=torch.float64)
    )
    H = X.T @ X
    e0 = torch.linalg.eigvalsh(H)
    e1 = torch.linalg.eigvalsh(rot.conjugate_hessian(H))
    assert torch.allclose(e0, e1, rtol=1e-8, atol=1e-8 * float(e0.max()))


def test_rotation_preserves_the_layer_function():
    """X W is unchanged: the rotation is a change of basis, not an approximation."""
    torch.manual_seed(2)
    n, p, m = 384, 40, 200
    rot = Rotation(n, 128, seed=3, dtype=torch.float64)
    X = torch.randn(m, n, dtype=torch.float64)
    W = torch.randn(n, p, dtype=torch.float64)
    assert torch.allclose(rot.right(X) @ rot.left_T(W), X @ W, atol=1e-9)


def test_rotation_shrinks_weight_ranges_with_outliers():
    """The point of incoherence processing: outliers stop setting the grid."""
    torch.manual_seed(3)
    n, p = 512, 64
    W = torch.randn(n, p, dtype=torch.float64) * 0.05
    W[0, ::5] *= 20.0
    rot = Rotation(n, 512, seed=4, dtype=torch.float64)
    Wr = rot.left_T(W)
    rng = lambda A: ((A.max(0).values - A.min(0).values) ** 2).sum()
    assert rng(Wr) < 0.75 * rng(W)


def test_compression_in_the_rotated_basis_is_measured_consistently():
    """Compressing in the rotated basis and mapping back must give exactly the
    error the rotated-basis Hessian reports, or the whole pipeline is comparing
    numbers from two different problems."""
    from qlr.gptq import gptq_intrinsic_lora, layer_error

    torch.manual_seed(4)
    torch.set_default_dtype(torch.float64)
    m, n, p = 512, 256, 64
    X = torch.randn(m, n, dtype=torch.float64) @ torch.diag(
        torch.logspace(1, -1, n, dtype=torch.float64)
    )
    H = X.T @ X
    W = torch.randn(n, p, dtype=torch.float64) * 0.05
    W[0, ::5] *= 12.0

    rot = Rotation(n, 256, seed=5, dtype=torch.float64)
    H_rot = rot.conjugate_hessian(H)
    W_rot = rot.left_T(W)

    Q, L, R, _ = gptq_intrinsic_lora(H_rot, W_rot, bits=3, rank=8)
    What_rot = Q + L @ R
    What = rot.left(What_rot)          # back to the original basis

    err_rotated = layer_error(H_rot, W_rot, What_rot)
    err_original = layer_error(H, W, What)
    assert err_original == pytest.approx(err_rotated, rel=1e-9)
