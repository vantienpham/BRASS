"""Randomized block-Hadamard rotations, for incoherence preprocessing.

Weight-only incoherence processing of the QuaRot / QuIP family applies an
orthogonal ``Theta`` to a layer's input features and the inverse to its weights:

    X -> X Theta,    W -> Theta^T W,

which leaves ``X W`` exactly unchanged while spreading weight outliers across
coordinates, so the per-channel quantization grid no longer has to span them.
In terms of the quantities this work allocates against, the substitution is

    H -> Theta^T H Theta,   W -> Theta^T W.

**Two consequences are worth being explicit about, because they are not
symmetric.** Conjugation by an orthogonal matrix preserves eigenvalues, so the
calibration spectrum -- and hence the spectral tail ``T(r)``, the damping
``lambda = tau * tr(H)/N``, and the whole rank factor ``h(r)`` -- is
*invariant* under rotation. What rotation does change is the weight-range
statistic ``S``, since ``Theta^T W`` has far smaller per-column max-minus-min
than ``W`` when ``W`` carries outliers. Rotation therefore acts on the
bit-distortion factor and leaves the rank-distortion factor alone.

``block`` controls how much rotation is applied: a block-diagonal ``Theta`` made
of ``N / block`` independent ``block x block`` randomized Hadamard matrices.
``block = 1`` is the identity and ``block = N`` is a full rotation, so sweeping
it interpolates between no preprocessing and the strongest available.

Blocks are applied by reshaping rather than by forming the dense ``N x N``
matrix, which for a 12288-wide layer would be 600 MB on its own.
"""

from __future__ import annotations

import torch

__all__ = ["largest_valid_block", "Rotation"]


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def largest_valid_block(n: int, requested: int) -> int:
    """Largest power of two that is at most ``requested`` and divides ``n``.

    Transformer widths are not all powers of two (3072, 6144, 12288), so a
    requested block is reduced until it tiles the layer exactly. Returns 1 when
    no non-trivial block works, which makes the rotation the identity.
    """
    # Start from the largest power of two not exceeding the request, then halve
    # until it also divides n. Starting from the request itself would step
    # through non-powers of two (3072 -> 1536 -> ...) and never reach 1024.
    cap = min(requested, n)
    b = 1
    while b * 2 <= cap:
        b *= 2
    while b > 1 and n % b != 0:
        b //= 2
    return max(b, 1)


def _hadamard(b: int, device, dtype) -> torch.Tensor:
    """Normalized Sylvester Hadamard matrix of size ``b``, a power of two."""
    h = torch.ones(1, 1, device=device, dtype=dtype)
    while h.shape[0] < b:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / (b**0.5)


class Rotation:
    """A block-diagonal randomized Hadamard rotation of dimension ``n``.

    ``Theta = blockdiag(D_1 H, ..., D_k H)`` with ``H`` a normalized Hadamard
    matrix and ``D_i`` independent random sign diagonals. Orthogonal by
    construction, and its own transpose is applied by reversing the two factors.
    """

    def __init__(self, n: int, block: int, seed: int = 0, device=None, dtype=torch.float32):
        self.n = n
        self.block = largest_valid_block(n, block)
        self.k = n // self.block
        self.identity = self.block <= 1
        device = device or torch.device("cpu")
        self.dtype = dtype
        if self.identity:
            self.H = None
            self.signs = None
            return
        self.H = _hadamard(self.block, device, dtype)
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + n)
        signs = (torch.randint(0, 2, (self.k, self.block), generator=g) * 2 - 1)
        self.signs = signs.to(device=device, dtype=dtype)

    def to(self, device, dtype=None) -> "Rotation":
        if not self.identity:
            self.H = self.H.to(device=device, dtype=dtype or self.dtype)
            self.signs = self.signs.to(device=device, dtype=dtype or self.dtype)
        return self

    # Theta^T M, with M of shape (n, p).
    def left_T(self, M: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return M
        p = M.shape[1]
        # (Theta^T M)_i = H^T D_i M_i = H^T (signs_i * M_i), H symmetric.
        Mb = M.reshape(self.k, self.block, p)
        Mb = self.signs.unsqueeze(-1) * Mb
        return torch.bmm(self.H.t().expand(self.k, -1, -1), Mb).reshape(self.n, p)

    # M Theta, with M of shape (p, n).
    def right(self, M: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return M
        p = M.shape[0]
        # (M Theta)_i = M_i (D_i H) = (M_i * signs_i) H -- the sign diagonal
        # multiplies on the same side it sits in Theta, so it must come first.
        Mb = M.reshape(p, self.k, self.block).transpose(0, 1)  # (k, p, block)
        Mb = Mb * self.signs.unsqueeze(1)
        Mb = torch.bmm(Mb, self.H.expand(self.k, -1, -1))
        return Mb.transpose(0, 1).reshape(p, self.n)

    # Theta M, with M of shape (n, p) -- used to map a compressed weight back.
    def left(self, M: torch.Tensor) -> torch.Tensor:
        if self.identity:
            return M
        p = M.shape[1]
        Mb = M.reshape(self.k, self.block, p)
        Mb = torch.bmm(self.H.expand(self.k, -1, -1), Mb)
        return (self.signs.unsqueeze(-1) * Mb).reshape(self.n, p)

    def conjugate_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """``Theta^T H Theta``, symmetrised against accumulation round-off."""
        if self.identity:
            return H
        # left_T gives Theta^T H, right then post-multiplies by Theta. Both act
        # on the n-sized axis, which for a Hessian is both of them.
        R = self.right(self.left_T(H))
        return 0.5 * (R + R.t())
