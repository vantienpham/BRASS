"""Accumulate calibration Hessians ``H = X^T X`` for the linear layers of a block.

``X`` has one row per token, so for a linear layer seeing ``(batch, seq, N)``
activations the contribution is the Gram matrix of the flattened ``(batch*seq,
N)`` view. Accumulation is in float32 even when the model runs in float16: the
Gram matrix of 2048*128 tokens overflows and loses ordering in half precision,
and every downstream eigenvalue inherits that error.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["HessianAccumulator", "collect_block_hessians"]


class HessianAccumulator:
    """Forward hook that maintains ``H = sum_t x_t x_t^T`` for one linear layer."""

    def __init__(self, layer: nn.Linear, device: torch.device):
        self.n_in = layer.in_features
        self.H = torch.zeros(self.n_in, self.n_in, device=device, dtype=torch.float32)
        self.n_samples = 0

    def __call__(self, module, inputs, output) -> None:
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).to(torch.float32)
        self.H.addmm_(x.t(), x)
        self.n_samples += x.shape[0]

    def finalize(self) -> torch.Tensor:
        """Symmetrised Hessian. Not normalised by sample count -- the GPTQ
        update, the damping fraction and the error bound are all invariant to a
        positive rescaling of ``H``, and keeping the raw sum makes the
        eigenvalues directly comparable to ``||X||_F^2``."""
        H = 0.5 * (self.H + self.H.t())
        # A feature that is identically zero across the calibration set leaves a
        # zero row and column. Leave it: damping handles it, and zeroing it out
        # here would silently change the layer's effective dimension.
        return H


def collect_block_hessians(
    block: nn.Module,
    layers: dict[str, nn.Linear],
    hidden: list[torch.Tensor],
    kwargs: list[dict],
    device: torch.device,
    keep_outputs: bool = True,
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """Run the calibration batches through ``block``; return per-layer ``H`` and
    the block's outputs.

    The outputs are returned from this same pass on purpose. They are the
    *full-precision* outputs, and the calibration protocol requires the next
    block to see exactly those -- capturing them here rather than after
    compression removes any chance of accidentally propagating quantised
    activations, and costs nothing extra.
    """
    accs = {name: HessianAccumulator(mod, device) for name, mod in layers.items()}
    handles = [mod.register_forward_hook(accs[name]) for name, mod in layers.items()]
    outputs: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for h, kw in zip(hidden, kwargs):
                y = block(h, **kw)
                if keep_outputs:
                    outputs.append((y[0] if isinstance(y, tuple) else y).detach())
    finally:
        for handle in handles:
            handle.remove()
    return {name: acc.finalize() for name, acc in accs.items()}, outputs
