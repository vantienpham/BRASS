"""Evaluation: WikiText-2 perplexity and zero-shot multiple-choice accuracy.

Progress bars pass ``disable=None`` rather than ``False``: tqdm then suppresses
itself whenever stdout is not a terminal. Under Slurm it always is not, and a
bar redrawn with carriage returns puts thousands of frames -- and the result
line that follows them -- on a single unusable log line.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm

__all__ = ["perplexity", "zero_shot"]


@torch.no_grad()
def perplexity(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    batch_size: int = 1,
    progress: bool = True,
) -> float:
    """Token-level perplexity over non-overlapping windows.

    The standard protocol: each ``(n, seqlen)`` window is scored with the model's
    own next-token loss, and the windows are averaged with equal weight (they
    all contain ``seqlen - 1`` predicted tokens).
    """
    model.eval()
    n = tokens.shape[0]
    nlls = []
    it = range(0, n, batch_size)
    for i in tqdm(it, disable=None if progress else True, desc="ppl", leave=False):
        batch = tokens[i : i + batch_size].to(device)
        out = model(batch, labels=batch)
        # HF returns the mean over predicted tokens of the batch.
        nlls.append(out.loss.float().item() * batch.shape[0])
    return float(torch.exp(torch.tensor(sum(nlls) / n)))


@torch.no_grad()
def zero_shot(
    model: nn.Module,
    tokenizer,
    task: str,
    device: torch.device,
    limit: int | None = None,
    progress: bool = True,
) -> dict:
    """Length-normalised log-likelihood accuracy on a multiple-choice task.

    Implemented directly rather than through lm-eval-harness so it runs under
    ``HF_DATASETS_OFFLINE=1`` with only the raw dataset cached, and so the
    scoring rule is visible in the paper rather than inherited.

    Every task is reduced to the same shape: a document is a list of
    ``(context, continuation)`` pairs plus the index of the correct one. Most
    tasks vary the continuation against a fixed context; Winogrande varies the
    context against a fixed continuation, which this representation handles
    without a special case.
    """
    docs = _load_task(task, limit)
    correct = correct_norm = 0
    for doc in tqdm(docs, disable=None if progress else True, desc=task, leave=False):
        scores, norms = [], []
        for ctx, cont in doc["pairs"]:
            ll, ntok = _loglikelihood(model, tokenizer, ctx, cont, device)
            scores.append(ll)
            norms.append(ll / max(ntok, 1))
        correct += int(max(range(len(scores)), key=scores.__getitem__) == doc["label"])
        correct_norm += int(max(range(len(norms)), key=norms.__getitem__) == doc["label"])
    n = len(docs)
    return {"task": task, "n": n, "acc": correct / n, "acc_norm": correct_norm / n}


def _loglikelihood(model, tokenizer, context: str, continuation: str, device):
    ctx = tokenizer(context, return_tensors="pt").input_ids
    full = tokenizer(context + continuation, return_tensors="pt").input_ids
    n_ctx = ctx.shape[1]
    n_cont = full.shape[1] - n_ctx
    if n_cont <= 0:  # continuation merged into the last context token
        return float("-inf"), 1
    logits = model(full.to(device)).logits.float()
    logprobs = torch.log_softmax(logits[0, n_ctx - 1 : -1], dim=-1)
    target = full[0, n_ctx:].to(device)
    return float(logprobs.gather(-1, target.unsqueeze(-1)).sum()), n_cont


def _load_task(task: str, limit: int | None):
    """Return documents as ``{"pairs": [(context, continuation), ...], "label": int}``."""
    from datasets import load_dataset

    if task in ("arc_easy", "arc_challenge"):
        cfg = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        ds = load_dataset("allenai/ai2_arc", cfg, split="test")
        docs = [
            {
                "pairs": [
                    (f"Question: {d['question']}\nAnswer:", " " + t)
                    for t in d["choices"]["text"]
                ],
                "label": d["choices"]["label"].index(d["answerKey"]),
            }
            for d in ds
            if d["answerKey"] in d["choices"]["label"]
        ]
    elif task == "piqa":
        # ybisk/piqa is a script dataset, which `datasets` >= 4 refuses to run;
        # baber/piqa is the same data as parquet.
        ds = load_dataset("baber/piqa", split="validation")
        docs = [
            {
                "pairs": [
                    (f"Question: {d['goal']}\nAnswer:", " " + d["sol1"]),
                    (f"Question: {d['goal']}\nAnswer:", " " + d["sol2"]),
                ],
                "label": int(d["label"]),
            }
            for d in ds
        ]
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        docs = [
            {
                "pairs": [
                    (d["activity_label"] + ": " + d["ctx"], " " + e) for e in d["endings"]
                ],
                "label": int(d["label"]),
            }
            for d in ds
            if d["label"] != ""
        ]
    elif task == "winogrande":
        # The blank splits the sentence; each option yields a different context
        # scored against the *same* continuation, so the comparison is fair.
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        docs = []
        for d in ds:
            i = d["sentence"].index("_")
            prefix, suffix = d["sentence"][:i], d["sentence"][i + 1 :]
            docs.append(
                {
                    "pairs": [
                        (prefix + d["option1"], suffix),
                        (prefix + d["option2"], suffix),
                    ],
                    "label": int(d["answer"]) - 1,
                }
            )
    else:
        raise ValueError(f"unknown task {task!r}")
    return docs[:limit] if limit else docs
