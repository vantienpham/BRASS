"""Calibration and evaluation data.

The calibration protocol follows Zhang & Saab (Sec. 6): 128 sequences of 2048
tokens drawn once from the WikiText-2 *training* split, reused for every linear
layer in the network. Evaluation uses the entire WikiText-2 test split, split
into non-overlapping windows of the same length.

Everything here must work with ``HF_DATASETS_OFFLINE=1``: compute nodes on this
cluster have no outbound network, so a cache miss has to fail immediately
rather than hang. Prefetch on the login node with ``scripts/prefetch.py``.
"""

from __future__ import annotations

import torch

__all__ = ["get_calibration", "get_eval_tokens"]

_WIKITEXT = ("wikitext", "wikitext-2-raw-v1")


def _load_split(split: str):
    from datasets import load_dataset

    return load_dataset(*_WIKITEXT, split=split)


def get_calibration(
    tokenizer,
    n_samples: int = 128,
    seqlen: int = 2048,
    seed: int = 0,
    dataset: str = "wikitext2",
) -> torch.Tensor:
    """Return ``(n_samples, seqlen)`` token ids sampled once, for all layers."""
    if dataset == "wikitext2":
        ds = _load_split("train")
        text = "\n\n".join(ds["text"])
    elif dataset == "c4":
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        text = "\n\n".join(ds[: 20 * n_samples]["text"])
    else:
        raise ValueError(f"unknown calibration dataset {dataset!r}")

    enc = tokenizer(text, return_tensors="pt").input_ids[0]
    if enc.numel() < n_samples * seqlen:
        raise ValueError(
            f"calibration corpus has {enc.numel()} tokens, need {n_samples * seqlen}"
        )
    g = torch.Generator().manual_seed(seed)
    hi = enc.numel() - seqlen - 1
    starts = torch.randint(0, hi, (n_samples,), generator=g)
    return torch.stack([enc[s : s + seqlen] for s in starts])


def get_eval_tokens(tokenizer, seqlen: int = 2048, split: str = "test") -> torch.Tensor:
    """Whole WikiText-2 split as ``(n_windows, seqlen)`` non-overlapping windows."""
    ds = _load_split(split)
    enc = tokenizer("\n\n".join(ds["text"]), return_tensors="pt").input_ids[0]
    n = enc.numel() // seqlen
    if n == 0:
        raise ValueError(f"{split} split shorter than one {seqlen}-token window")
    return enc[: n * seqlen].view(n, seqlen)
