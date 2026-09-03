#!/usr/bin/env python
"""Experiment 2: measure the true per-layer distortion surface D_l(b, r).

The allocator's theory rests on a surrogate -- the Corollary-3.9 upper bound.
An upper bound is not automatically a good *ranking* of configurations, and
there is a specific reason to worry here: the bound's rank dependence is the
spectral tail ``sum_{i>r} sigma_i^2``, which is worst-case, while the algorithm
actually fits the residual it sees. If the bound systematically undervalues
rank, an allocator driven by it will underspend on rank.

So measure the real thing. For each profiled layer this sweeps ``(b, r)`` and
records ``||X W - X What||_F^2`` exactly, through the cached Hessian. No
evaluation and no forward passes are involved, so the cost is the quantisation
passes alone.

The sweep is rank-outer / bits-inner because the augmented factorisation
depends on ``r`` but not on ``b``: hoisting it (see `gilora_factor`) cuts the
number of eigendecompositions by the number of bit-widths.

The output doubles as the distortion table for the *measured* allocator, which
is the honest upper baseline for how well any surrogate could do.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.gptq import (  # noqa: E402
    bid_up, gilora_factor, gilora_with_factor, layer_error, olrc,
)
from qlr.models import block_list, linear_layers, load_model  # noqa: E402
from qlr.rotate import Rotation  # noqa: E402
from qlr.runlog import RunDir  # noqa: E402


def select_layers(keys: list[str], every: int, kinds: list[str] | None) -> list[str]:
    """Stratified subsample: keep every ``every``-th block, all layer kinds.

    Layer *kind* (q/k/v/o/gate/up/down) is the axis along which the difficulty
    coefficient varies most, and block index the axis along which it varies
    smoothly, so this keeps the informative variation at a fraction of the cost.
    """
    out = []
    for k in keys:
        bi = int(k.split(".")[0])
        kind = k.split(".")[-1]
        if bi % every:
            continue
        if kinds and not any(c in kind for c in kinds):
            continue
        out.append(k)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--hessian-dir", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--bits", type=int, nargs="+", default=[2, 3, 4, 5, 6])
    ap.add_argument("--ranks", type=int, nargs="+", default=[0, 8, 16, 32, 64, 128])
    ap.add_argument("--every", type=int, default=1, help="profile every k-th block")
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--n-loops", type=int, default=0)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate-block", type=int, default=1,
                    help="must match the value used to build the Hessian cache")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    run.config(vars(args))
    run.env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hdir = pathlib.Path(args.hessian_dir)

    bundle = load_model(args.model, seqlen=args.seqlen)
    modules = {}
    for bi, block in enumerate(block_list(bundle.model)):
        for name, mod in linear_layers(block).items():
            modules[f"{bi}.{name}"] = mod

    keys = select_layers(sorted(modules, key=lambda k: (int(k.split(".")[0]), k)),
                         args.every, args.kinds)
    print(f"profiling {len(keys)}/{len(modules)} layers "
          f"over {len(args.bits)} bits x {len(args.ranks)} ranks", flush=True)

    out_path = run.artifact("profile.json")
    table: dict = {}
    if args.resume and out_path.exists():
        table = json.load(open(out_path))
        print(f"resuming with {len(table)} layer(s) already profiled", flush=True)

    t0 = time.time()
    for i, key in enumerate(keys):
        if key in table:
            continue
        blob = torch.load(hdir / f"{key}.pt", map_location="cpu", weights_only=True)
        H = blob["H"].to(device=device, dtype=torch.float32)
        evecs = blob["evecs"].to(device=device, dtype=torch.float32)
        W = modules[key].weight.data.t().contiguous().to(device=device, dtype=torch.float32)
        if args.rotate_block > 1:
            # The cached H is already in the rotated basis; put W in it too.
            W = Rotation(W.shape[0], args.rotate_block, seed=args.seed,
                         device=device, dtype=torch.float32).left_T(W)
        e0 = layer_error(H, W, torch.zeros_like(W))

        entry = {"N": int(W.shape[0]), "Nout": int(W.shape[1]), "e0": e0, "D": {}}
        for r in args.ranks:
            r_eff = min(r, W.shape[0])
            L, Psi = gilora_factor(H, r_eff, args.damp_frac, evecs)
            for b in args.bits:
                Q, R, grid = gilora_with_factor(W, L, Psi, b, args.beta)
                What = Q + L @ R if r_eff > 0 else Q
                d = layer_error(H, W, What)
                rec = {"gilora": d}
                if args.n_loops:
                    Qr, Lr, Rr = Q, L, R
                    for _ in range(args.n_loops):
                        Lr, Rr = olrc(H, W, Qr, r_eff, args.damp_frac)
                        Qr = bid_up(H, W - Lr @ Rr, Qr, grid, args.damp_frac)
                    Lr, Rr = olrc(H, W, Qr, r_eff, args.damp_frac)
                    rec["refined"] = layer_error(H, W, Qr + Lr @ Rr)
                entry["D"][f"{b},{r}"] = rec
                del Q, R, What
            del L, Psi
            if device.type == "cuda":
                torch.cuda.empty_cache()

        table[key] = entry
        del H, evecs, W, blob
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with open(out_path, "w") as fh:
            json.dump(table, fh)
        el = time.time() - t0
        print(f"[{i+1}/{len(keys)}] {key:34s} e0={e0:.4e}  "
              f"{el:.0f}s elapsed, ~{el/(i+1)*(len(keys)-i-1):.0f}s left", flush=True)

    run.metrics({"n_layers_profiled": len(table), "bits": args.bits, "ranks": args.ranks})
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
