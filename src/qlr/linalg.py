"""Numerically robust linear algebra for calibration Hessians.

Every routine here exists because a naive call failed on this cluster. See
`cluster.local.md`, "Old GPUs also lie about linear algebra":

  * cuSOLVER SVD can fail to converge on ill-conditioned input *after* torch's
    own internal fallback, raising `LinAlgError`.
  * `cholesky_ex` can return ``info == 0`` next to a factor containing NaN --
    the status code reports the driver's opinion, not a usable factor.

The rule followed throughout: **verify the output, not the status code.**
"""

from __future__ import annotations

import warnings

import torch

__all__ = [
    "safe_eigh",
    "safe_svdvals",
    "inv_sqrt_psd",
    "sqrt_psd",
    "stable_tri_factor",
    "cholesky_tri_factor",
]


def _finite(*mats: torch.Tensor) -> bool:
    return all(torch.isfinite(m).all().item() for m in mats)


def safe_eigh(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric eigendecomposition ``A = V diag(w) V^T``, ascending ``w``.

    Tries the native device in the tensor's own dtype, then float64 on device,
    then float64 on CPU (LAPACK uses a different algorithm and routinely
    succeeds where cuSOLVER does not). Results are checked for finiteness
    rather than trusted.
    """
    A = 0.5 * (A + A.transpose(-1, -2))  # kill asymmetry from accumulation order
    attempts = [
        (A.device, A.dtype),
        (A.device, torch.float64),
        (torch.device("cpu"), torch.float64),
    ]
    last: Exception | None = None
    for device, dtype in attempts:
        try:
            w, V = torch.linalg.eigh(A.to(device=device, dtype=dtype))
        except (torch.linalg.LinAlgError, RuntimeError) as exc:  # pragma: no cover
            last = exc
            continue
        if _finite(w, V):
            return w.to(A.device, A.dtype), V.to(A.device, A.dtype)
        last = RuntimeError("eigh returned non-finite output")
    raise RuntimeError(f"safe_eigh exhausted every fallback: {last}")


def safe_svdvals(A: torch.Tensor) -> torch.Tensor:
    """Singular values, descending, with the same fallback ladder as `safe_eigh`."""
    attempts = [
        (A.device, A.dtype),
        (A.device, torch.float64),
        (torch.device("cpu"), torch.float64),
    ]
    last: Exception | None = None
    for device, dtype in attempts:
        try:
            s = torch.linalg.svdvals(A.to(device=device, dtype=dtype))
        except (torch.linalg.LinAlgError, RuntimeError) as exc:  # pragma: no cover
            last = exc
            continue
        if _finite(s):
            return s.to(A.device, A.dtype)
        last = RuntimeError("svdvals returned non-finite output")
    raise RuntimeError(f"safe_svdvals exhausted every fallback: {last}")


def _psd_power(A: torch.Tensor, power: float, floor: float = 0.0) -> torch.Tensor:
    w, V = safe_eigh(A)
    w = w.clamp_min(floor if floor > 0 else 0.0)
    if power < 0:
        # A negative power of a zero eigenvalue is +inf; clamp to the largest
        # eigenvalue scaled by the float epsilon of the working dtype.
        eps = torch.finfo(w.dtype).eps
        w = w.clamp_min(w.max() * eps * w.numel())
    return (V * w.pow(power).unsqueeze(-2)) @ V.transpose(-1, -2)


def sqrt_psd(A: torch.Tensor) -> torch.Tensor:
    """Principal square root of a positive semidefinite matrix."""
    return _psd_power(A, 0.5)


def inv_sqrt_psd(A: torch.Tensor) -> torch.Tensor:
    """Inverse square root of a positive definite matrix."""
    return _psd_power(A, -0.5)


def cholesky_tri_factor(H: torch.Tensor, damp: float) -> torch.Tensor:
    """The GPTQ triangular factor via the reference implementation's route.

    Reproduces `ist2022gptq`: Cholesky of ``H + damp*I``, Cholesky inverse,
    then a second Cholesky of the inverse. Returns lower-triangular ``Psi``
    with ``Psi @ Psi.T == (H + damp*I)^-1``.

    Raises `torch.linalg.LinAlgError` when the matrix is too ill-conditioned --
    that failure is the phenomenon `stable_tri_factor` exists to avoid, and the
    paper's GPTQ-vs-GPTQ* comparison needs it to be reachable.
    """
    n = H.shape[-1]
    Hd = H + damp * torch.eye(n, device=H.device, dtype=H.dtype)
    Phi = torch.linalg.cholesky(Hd)
    Hinv = torch.cholesky_inverse(Phi)
    Hinv = 0.5 * (Hinv + Hinv.transpose(-1, -2))
    Psi = torch.linalg.cholesky(Hinv)
    if not _finite(Psi):
        raise torch.linalg.LinAlgError("Cholesky factor contains non-finite entries")
    return Psi


def stable_tri_factor(H: torch.Tensor, damp: float) -> torch.Tensor:
    """Lower-triangular ``Psi`` with ``Psi @ Psi.T == (H + damp*I)^-1``, robustly.

    The eigendecomposition-plus-QR route of Zhang & Saab (App. C): form
    ``(H + damp I)^{-1/2}`` by eigendecomposition, take ``QR`` of it, and
    return ``G^T`` where ``G`` is the (positive-diagonal) upper-triangular
    factor. Then ``G^T G = (H + damp I)^{-1}`` as required.

    This is the "GPTQ*" factorisation. It matters here because the augmented
    Hessian of GPTQ-intrinsic LoRA is *singular* before damping -- its extra
    columns ``X V_r`` lie in the span of ``X`` -- so a direct Cholesky of the
    damped inverse can fail at the damping level the quantisation objective
    actually wants.
    """
    n = H.shape[-1]
    Hd = H + damp * torch.eye(n, device=H.device, dtype=H.dtype)
    Hinv_sqrt = _psd_power(Hd, -0.5)
    # QR in float64 -- the sign convention below needs a trustworthy diagonal.
    Q, G = torch.linalg.qr(Hinv_sqrt.to(torch.float64))
    # Enforce the positive-diagonal convention so G^T is genuinely the Cholesky
    # factor of the inverse rather than a sign-flipped relative of it.
    sign = torch.sign(torch.diagonal(G, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    G = G * sign.unsqueeze(-1)
    Psi = G.transpose(-1, -2).to(H.dtype)
    if not _finite(Psi):  # pragma: no cover
        raise RuntimeError("stable_tri_factor produced non-finite output")
    return Psi
