"""Asymmetric per-channel uniform quantisation.

Follows the grid used by Zhang & Saab (Sec. 1, "Notation"): for a channel
(column) ``w`` and bit-width ``b``,

    s = beta * (max(w) - min(w)) / (2^b - 1),
    z = round( -min(w) / (max(w) - min(w)) * (2^b - 1) ),
    Q(x) = s * ( clip( round(x/s) + z ; 0, 2^b - 1 ) - z ).

``beta`` shrinks the min-max range, trading representable range against grid
spacing. It matters enormously at low bit-width: with per-channel min-max
grids, a single outlier weight in a column stretches the range and inflates the
step for all the others. Zhang & Saab set it by hand (1.0 at 4 bit, 0.9 at
3 bit for Qwen3; 0.6-0.7 for their vision transformers).

`search_beta` instead selects it per column by direct minimisation of the
calibration-weighted error, which needs no hand tuning, adapts to each layer
and bit-width, and -- because the layer-wise objective separates over columns
-- is exact rather than a heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["Grid", "make_grid", "quantize", "grid_step", "search_beta"]


@dataclass
class Grid:
    """A per-channel affine grid. ``scale`` and ``zero`` are one per column."""

    scale: torch.Tensor  # (N',)
    zero: torch.Tensor  # (N',)
    bits: int

    @property
    def qmax(self) -> int:
        return 2**self.bits - 1

    def to(self, *args, **kwargs) -> "Grid":
        return Grid(self.scale.to(*args, **kwargs), self.zero.to(*args, **kwargs), self.bits)


def make_grid(W: torch.Tensor, bits: int, beta=1.0) -> Grid:
    """Build the per-column asymmetric grid for ``W`` of shape ``(N, N')``.

    ``beta`` is a scalar or a per-column tensor of shape ``(N',)``.
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    wmin = W.min(dim=0).values
    wmax = W.max(dim=0).values
    rng = (wmax - wmin).clamp_min(torch.finfo(W.dtype).tiny)
    qmax = 2**bits - 1
    if not torch.is_tensor(beta):
        beta = torch.as_tensor(beta, device=W.device, dtype=W.dtype)
    scale = beta.to(W.device, W.dtype) * rng / qmax
    # Zero point placed on the *unscaled* min-max range, as in the paper, so
    # that beta < 1 shrinks the step without moving the grid's origin.
    zero = torch.round(-wmin / rng * qmax)
    return Grid(scale=scale, zero=zero, bits=bits)


def search_beta(
    H: torch.Tensor,
    W: torch.Tensor,
    bits: int,
    candidates=(1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5),
) -> torch.Tensor:
    """Per-column clipping factor minimising the calibration-weighted error.

    The layer-wise objective separates over columns,

        ||X(W - Q)||_F^2 = sum_j (W_j - Q_j)^T H (W_j - Q_j),

    so choosing ``beta`` independently per column is an exact minimisation of
    that objective over the candidate set, not a heuristic decomposition. The
    proxy used is round-to-nearest: it captures the range-versus-step trade-off
    that ``beta`` controls, and costs one quadratic form per candidate rather
    than a full quantisation pass.

    Returns a ``(N',)`` tensor to be passed straight to `make_grid`.
    """
    best = torch.full((W.shape[1],), float("inf"), device=W.device, dtype=torch.float32)
    chosen = torch.ones(W.shape[1], device=W.device, dtype=W.dtype)
    for beta in candidates:
        D = (W - quantize(W, make_grid(W, bits, beta))).to(H.dtype)
        # diag((W-Q)^T H (W-Q)), one entry per column, without forming the matrix
        err = ((H @ D) * D).sum(dim=0).to(torch.float32)
        better = err < best
        best = torch.where(better, err, best)
        chosen = torch.where(better, torch.as_tensor(beta, device=W.device, dtype=W.dtype), chosen)
    return chosen


def quantize(V: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Round-to-nearest onto ``grid``. ``V`` is ``(N, N')`` or ``(N',)``."""
    s, z, qmax = grid.scale, grid.zero, grid.qmax
    q = torch.round(V / s + z).clamp_(0, qmax)
    return s * (q - z)


def grid_step(grid: Grid) -> torch.Tensor:
    """Per-column grid spacing ``delta``, i.e. the scale."""
    return grid.scale
