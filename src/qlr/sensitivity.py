"""Layer sensitivity weights for the allocation objective.

The layer-wise objective ``||X_l W_l - X_l What_l||_F^2`` is what every method
in this line of work minimises, but it is not what anyone cares about. Two
layers with equal layer-wise error can do very different damage to the network
output, so the allocator minimises ``sum_l omega_l D_l``, where ``omega_l``
converts layer-``l`` output distortion into end-to-end loss.

Writing ``L`` for the language-modelling loss and ``Y_l`` for layer ``l``'s
output, a perturbation ``Delta`` gives

    L(Y_l + Delta) = L(Y_l) + <g, Delta> + (1/2) Delta^T Hess Delta + O(|Delta|^3).

The quadratic term is the one that transfers layer-wise error to the network,
and the linear term is pure noise for this purpose -- it is zero-mean over
random ``Delta`` but has large variance, and with few draws it dominates and
can even drive the estimate negative.

**Antithetic sampling removes it exactly.** Evaluating at both ``+Delta`` and
``-Delta``,

    L(Y+Delta) + L(Y-Delta) - 2 L(Y) = Delta^T Hess Delta + O(|Delta|^4),

so the linear term cancels identically rather than merely in expectation. The
estimator below is that central second difference, normalised by
``||Delta||_F^2``. It costs two forward passes per draw instead of one and is
worth every bit of it: with the one-sided estimator, layers routinely returned
zero or negative sensitivity at small draw counts.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

__all__ = ["measure_sensitivity"]


@torch.no_grad()
def measure_sensitivity(
    model: nn.Module,
    layers: dict[str, nn.Linear],
    tokens: torch.Tensor,
    device: torch.device,
    n_draws: int = 3,
    rel_scale: float = 0.02,
    seed: int = 0,
    floor_frac: float = 1e-3,
    progress: bool = True,
) -> dict[str, float]:
    """Estimate ``omega_l = E[Delta^T Hess Delta] / ||Delta||_F^2`` per layer.

    ``rel_scale`` sets ``||Delta||_F`` as a fraction of the layer's own output
    norm, so the injected distortion is comparable across layers of different
    width. It must be small enough to stay in the quadratic regime; the ratio
    is then scale-free, which `scripts/check_sensitivity.py` verifies by
    sweeping it.

    ``floor_frac`` clamps the result to that fraction of the median across
    layers. A genuinely zero weight would tell the allocator to spend no bits
    at all on a layer, which no measurement at this precision can justify.
    """
    model.eval()
    base = _loss(model, tokens, device)
    omegas: dict[str, float] = {}

    for name, mod in tqdm(layers.items(), disable=None if progress else True, desc="sensitivity"):
        ratios = []
        for d in range(n_draws):
            state = {"sq": 0.0, "sign": +1.0, "call": 0}

            def hook(module, inputs, output, _s=state, _d=d):
                y = output
                g = torch.Generator(device=y.device).manual_seed(
                    (seed * 1_000_003 + _d * 1009 + _s["call"]) % (2**31)
                )
                _s["call"] += 1
                noise = torch.randn(y.shape, generator=g, device=y.device, dtype=y.dtype)
                scale = rel_scale * y.norm() / noise.norm().clamp_min(1e-12)
                delta = noise * scale
                if _s["sign"] > 0:
                    _s["sq"] += float(delta.float().pow(2).sum())
                return y + _s["sign"] * delta

            handle = mod.register_forward_hook(hook)
            try:
                # The generator is re-seeded per call from a counter, so the
                # minus pass must replay exactly the same noise sequence: reset
                # the counter, flip the sign, and the draws line up.
                state["sign"] = +1.0
                state["call"] = 0
                l_plus = _loss(model, tokens, device)
                state["sign"] = -1.0
                state["call"] = 0
                l_minus = _loss(model, tokens, device)
            finally:
                handle.remove()
            if state["sq"] > 0:
                ratios.append((l_plus + l_minus - 2.0 * base) / state["sq"])
        omegas[name] = float(np.median(ratios)) if ratios else 0.0

    vals = np.array(list(omegas.values()))
    positive = vals[vals > 0]
    if positive.size == 0:
        raise RuntimeError("every sensitivity estimate was non-positive; check rel_scale")
    floor = floor_frac * float(np.median(positive))
    n_floored = int((vals < floor).sum())
    if n_floored:
        print(f"[sensitivity] floored {n_floored}/{len(vals)} layer(s) at {floor:.3e}")
    return {k: max(v, floor) for k, v in omegas.items()}


@torch.no_grad()
def _loss(model: nn.Module, tokens: torch.Tensor, device: torch.device) -> float:
    total = 0.0
    for i in range(tokens.shape[0]):
        batch = tokens[i : i + 1].to(device)
        total += float(model(batch, labels=batch).loss)
    return total / tokens.shape[0]
