#!/usr/bin/env python
"""Peak GPU memory and per-phase cost, for the runtime table.

Wall-clock for each phase is already recorded by the jobs themselves; what is
not is peak device memory, because the pipeline holds one transformer block at
a time and the true peak is set by the widest layer's Hessian and its
eigendecomposition rather than by the model.

This measures that directly: for the widest cached layer it times and peaks the
three operations the pipeline actually performs -- the eigendecomposition, the
augmented triangular factorization GPTQ-intrinsic LoRA needs, and the
quantization pass -- and reports the model's own resident footprint alongside.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.gptq import gilora_factor, gilora_with_factor  # noqa: E402
from qlr.linalg import safe_eigh  # noqa: E402
from qlr.models import block_list, linear_layers, load_model  # noqa: E402
from qlr.runlog import RunDir  # noqa: E402


def peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hessian-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--bits", type=int, default=3)
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    run.config(vars(args))
    run.env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdir = pathlib.Path(args.hessian_dir)

    bundle = load_model(args.model)
    modules = {}
    for bi, block in enumerate(block_list(bundle.model)):
        for name, mod in linear_layers(block).items():
            modules[f"{bi}.{name}"] = mod
    n_params = sum(p.numel() for p in bundle.model.parameters())

    # The widest layer sets the peak: its Hessian is N x N.
    widest = max(modules, key=lambda k: modules[k].in_features)
    N = modules[widest].in_features
    print(f"{args.model}: {len(modules)} layers, widest input {N} ({widest})", flush=True)

    out = {"model": args.model, "n_layers": len(modules), "widest_in": N,
           "model_params_M": n_params / 1e6,
           "model_fp16_MB": n_params * 2 / 2**20}

    blob = torch.load(hdir / f"{widest}.pt", map_location="cpu", weights_only=True)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    H = blob["H"].to(device=device, dtype=torch.float32)
    W = modules[widest].weight.data.t().contiguous().to(device=device, dtype=torch.float32)
    out["hessian_MB"] = H.numel() * 4 / 2**20

    for label, fn in [
        ("eigendecomposition", lambda: safe_eigh(H.to(torch.float64))),
        ("augmented factorization", lambda: gilora_factor(H, args.rank)),
    ]:
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        t0 = time.time()
        res = fn()
        torch.cuda.synchronize() if device.type == "cuda" else None
        out[label] = {"seconds": time.time() - t0, "peak_MB": peak_mb()}
        print(f"  {label:<24}{out[label]['seconds']:7.2f}s  peak {out[label]['peak_MB']:8.1f} MB",
              flush=True)
        del res

    L, Psi = gilora_factor(H, args.rank)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    t0 = time.time()
    gilora_with_factor(W, L, Psi, args.bits)
    torch.cuda.synchronize() if device.type == "cuda" else None
    out["quantization pass"] = {"seconds": time.time() - t0, "peak_MB": peak_mb()}
    print(f"  {'quantization pass':<24}{out['quantization pass']['seconds']:7.2f}s  "
          f"peak {out['quantization pass']['peak_MB']:8.1f} MB", flush=True)

    run.metrics(out)
    with open(run.artifact("cost.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
