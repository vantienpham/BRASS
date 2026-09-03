"""Apply a per-layer allocation to a model, block by block.

The pipeline for one layer, given its budget ``(b, r)``:

    1. GPTQ-intrinsic LoRA on the augmented Hessian  ->  Q, L, R
    2. optionally ``n_loops`` of (OLrC, Bid-Up) refinement
    3. write ``Q + L R`` back into the layer's weight

Step 3 materialises the sum rather than keeping the factors separate. That is
the right choice for *measuring* what a budget buys -- it is numerically
identical to running the factored form at inference -- and it keeps evaluation
independent of any particular kernel. The rate accounting in `alloc.py` still
charges for ``L`` and ``R`` separately, which is what a deployed model stores.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from .gptq import bid_up, gptq, gptq_intrinsic_lora, layer_error, olrc
from .hessian import collect_block_hessians
from .linalg import safe_eigh
from .models import BlockRunner, ModelBundle, linear_layers
from .quant import make_grid, quantize

__all__ = ["LayerBudget", "compress_model", "CompressionReport"]


@dataclass
class LayerBudget:
    bits: int
    rank: int
    beta: object = 1.0  # scalar, or a per-column tensor from quant.search_beta
    method: str = "gilora"  # gilora | gptq | gptq_olrc | rtn
    n_loops: int = 0
    L_bits: int | None = None  # quantize the feature-extraction factor
    R_bits: int | None = None  # quantize the error-absorption factor


@dataclass
class CompressionReport:
    per_layer: dict
    wall_seconds: float

    def save(self, path) -> None:
        with open(path, "w") as fh:
            json.dump({"per_layer": self.per_layer, "wall_seconds": self.wall_seconds}, fh, indent=2)


def compress_layer(
    H: torch.Tensor,
    W: torch.Tensor,
    budget: LayerBudget,
    damp_frac: float = 0.01,
    eigvecs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Return the reconstructed weight and a record of what it cost."""
    err0 = layer_error(H, W, torch.zeros_like(W))
    b, r = budget.bits, budget.rank

    if budget.method == "rtn":
        grid = make_grid(W, b, budget.beta)
        What = quantize(W, grid)
    elif budget.method == "gptq":
        Q, grid = gptq(H, W, b, budget.beta, damp_frac, robust=True)
        What = Q
    elif budget.method == "gptq_olrc":
        Q, grid = gptq(H, W, b, budget.beta, damp_frac, robust=True)
        L, R = olrc(H, W, Q, r, damp_frac)
        What = Q + L @ R
    elif budget.method == "gilora":
        # L is quantized *before* the augmented Hessian is formed, so the
        # quantization pass absorbs its rounding error; R is produced in full
        # precision and rounded afterwards. This is the scheme of Zhang & Saab.
        Q, L, R, grid = gptq_intrinsic_lora(
            H, W, b, r, budget.beta, damp_frac, eigvecs=eigvecs, L_bits=budget.L_bits
        )
        for _ in range(budget.n_loops):
            L, R = olrc(H, W, Q, r, damp_frac)
            Q = bid_up(H, W - L @ R, Q, grid, damp_frac)
        if budget.n_loops:
            L, R = olrc(H, W, Q, r, damp_frac)
        if budget.R_bits is not None and r > 0:
            # Per-row scaling on R: the r scale factors fold into L, so only r
            # extra full-precision numbers are stored, which the rate model
            # neglects as second order.
            R = _quantize_rowwise(R, budget.R_bits)
        What = Q + L @ R if r > 0 else Q
    else:
        raise ValueError(f"unknown method {budget.method!r}")

    err = layer_error(H, W, What)
    return What, {
        "bits": b,
        "rank": r,
        "method": budget.method,
        "n_loops": budget.n_loops,
        "beta": (float(budget.beta.median()) if hasattr(budget.beta, "median")
                 else budget.beta),
        "rel_error": err / err0 if err0 > 0 else 0.0,
        "abs_error": err,
    }


def _quantize_rowwise(R: torch.Tensor, bits: int) -> torch.Tensor:
    """Quantize ``R`` (shape ``(r, N')``) with one asymmetric grid per row."""
    Rt = R.t().contiguous()          # (N', r): make_grid quantizes per column
    return quantize(Rt, make_grid(Rt, bits)).t().contiguous()


def compress_model(
    bundle: ModelBundle,
    calib_tokens: torch.Tensor,
    budgets: dict[str, LayerBudget],
    device: torch.device,
    damp_frac: float = 0.01,
    hessian_cache: dict[str, torch.Tensor] | None = None,
    work_dtype: torch.dtype = torch.float32,
    progress: bool = True,
) -> CompressionReport:
    """Compress every linear layer inside every transformer block.

    ``budgets`` is keyed by ``"<block_index>.<layer_name>"``. A layer with no
    entry is left in full precision -- that is how the embedding, the head and
    any deliberately excluded layer stay untouched.
    """
    t0 = time.time()
    runner = BlockRunner(bundle, device)
    hidden, kwargs = runner.capture_inputs(calib_tokens)
    record: dict = {}

    for bi, block in enumerate(tqdm(runner.blocks, disable=not progress, desc="blocks")):
        block.to(device)
        layers = linear_layers(block)
        # One pass gives both the Hessians and the full-precision outputs. The
        # outputs must be taken now: after the loop below the block's weights
        # are compressed, and forwarding through them would feed quantisation
        # error into every later block's calibration, which is not the protocol.
        Hs, next_hidden = collect_block_hessians(block, layers, hidden, kwargs, device)

        for name, mod in layers.items():
            key = f"{bi}.{name}"
            if key not in budgets:
                continue
            H = Hs[name].to(work_dtype)
            W = mod.weight.data.t().contiguous().to(work_dtype)  # (in, out)
            eig = None
            if budgets[key].method == "gilora" and budgets[key].rank > 0:
                _, V = safe_eigh(H)
                eig = V.flip(-1)
            What, rec = compress_layer(H, W, budgets[key], damp_frac, eigvecs=eig)
            mod.weight.data.copy_(What.t().to(mod.weight.dtype))
            record[key] = rec
            del H, W, What, eig

        if hessian_cache is not None:
            for name, H in Hs.items():
                hessian_cache[f"{bi}.{name}"] = H.cpu()
        del Hs

        hidden = next_hidden
        block.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return CompressionReport(per_layer=record, wall_seconds=time.time() - t0)
