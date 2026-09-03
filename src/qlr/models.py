"""Model loading and the layer-by-layer traversal every routine here shares.

Compression is done one transformer block at a time. The block's inputs are
captured once, the seven (or however many) linear layers inside it get their
Hessians accumulated from those inputs, the block is compressed, and its
*full-precision* outputs become the next block's inputs.

Propagating full-precision rather than quantised activations is deliberate: it
is the protocol of Zhang & Saab (following Qronos), and it is what makes the
calibration Hessians independent of the compression budget -- so one collection
pass serves an entire sweep of budget points.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

__all__ = ["load_model", "block_list", "linear_layers", "BlockRunner", "ModelBundle"]


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: object
    name: str
    seqlen: int


def load_model(name: str, dtype=torch.float16, seqlen: int | None = None) -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval()
    model.config.use_cache = False
    if seqlen is None:
        seqlen = min(getattr(model.config, "max_position_embeddings", 2048), 2048)
    return ModelBundle(model=model, tokenizer=tok, name=name, seqlen=seqlen)


def block_list(model: nn.Module) -> nn.ModuleList:
    """The transformer blocks, for the architectures used in this work."""
    for path in ("model.layers", "model.decoder.layers", "transformer.h", "layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, (nn.ModuleList, list)):
            return obj
    raise ValueError(f"could not locate transformer blocks in {type(model).__name__}")


def linear_layers(block: nn.Module, skip: tuple[str, ...] = ()) -> dict[str, nn.Linear]:
    """Named `nn.Linear` submodules of one block, in forward order."""
    out = {}
    for name, mod in block.named_modules():
        if isinstance(mod, nn.Linear) and not any(s in name for s in skip):
            out[name] = mod
    return out


class Catcher(nn.Module):
    """Wraps block 0 to capture its inputs, then aborts the forward pass.

    Everything after the embedding -- position ids, attention masks, rotary
    caches -- is architecture-specific and changes between transformers
    releases, so the kwargs are captured verbatim and replayed rather than
    reconstructed.
    """

    def __init__(self, block: nn.Module, store: list):
        super().__init__()
        self.block = block
        self.store = store

    def forward(self, hidden_states, **kwargs):
        self.store.append((hidden_states.detach(), kwargs))
        raise _StopForward


class _StopForward(Exception):
    pass


class BlockRunner:
    """Iterate transformer blocks, holding one block on the GPU at a time."""

    def __init__(self, bundle: ModelBundle, device: torch.device):
        self.bundle = bundle
        self.device = device
        self.blocks = block_list(bundle.model)

    def capture_inputs(self, token_batches: torch.Tensor) -> tuple[list, list]:
        """Run the embedding and stop at block 0, returning its inputs."""
        model = self.bundle.model
        store: list = []
        self.blocks[0] = Catcher(self.blocks[0], store)
        # Only the pre-block modules need to be on the GPU for this.
        _move_preamble(model, self.device)
        for i in range(token_batches.shape[0]):
            try:
                model(token_batches[i : i + 1].to(self.device))
            except _StopForward:
                pass
        self.blocks[0] = self.blocks[0].block
        _move_preamble(model, torch.device("cpu"))
        torch.cuda.empty_cache() if self.device.type == "cuda" else None
        hidden = [h for h, _ in store]
        kwargs = [k for _, k in store]
        return hidden, kwargs

    @staticmethod
    def forward_block(block: nn.Module, hidden: list, kwargs: list) -> list:
        out = []
        with torch.no_grad():
            for h, kw in zip(hidden, kwargs):
                y = block(h, **kw)
                out.append((y[0] if isinstance(y, tuple) else y).detach())
        return out


def _move_preamble(model: nn.Module, device: torch.device) -> None:
    """Move embeddings and rotary caches, but not the blocks themselves."""
    blocks = block_list(model)
    inner = getattr(model, "model", model)
    for attr in ("embed_tokens", "rotary_emb", "embed_positions", "wte", "wpe"):
        mod = getattr(inner, attr, None)
        if isinstance(mod, nn.Module):
            mod.to(device)
    # Buffers registered directly on the parent (e.g. causal masks).
    for name, buf in list(inner.named_buffers(recurse=False)):
        setattr(inner, name, buf.to(device))
    del blocks
