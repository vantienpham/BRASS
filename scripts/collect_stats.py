#!/usr/bin/env python
"""Pass 1: everything the allocator needs, collected once per model.

Walks the model block by block, accumulates each linear layer's calibration
Hessian, and records:

  * the Hessian eigenvalues (descending) -- the rank-distortion curve,
  * ``S = sum_j range(W_.j)^2``           -- the bit-distortion curve,
  * ``omega``, the sensitivity weight     -- optional, one extra sweep,
  * the Hessians themselves, cached to disk.

Because the protocol propagates full-precision activations, none of this
depends on the compression budget. One run of this script therefore serves an
entire sweep of budget points, and `allocate.py` needs no GPU at all.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.data import get_calibration  # noqa: E402
from qlr.hessian import collect_block_hessians  # noqa: E402
from qlr.linalg import safe_eigh  # noqa: E402
from qlr.models import BlockRunner, linear_layers, load_model  # noqa: E402
from qlr.quant import search_beta  # noqa: E402
from qlr.rotate import Rotation  # noqa: E402
from qlr.runlog import RunDir  # noqa: E402
from qlr.sensitivity import measure_sensitivity  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-dataset", default="wikitext2")
    ap.add_argument("--save-hessians", action="store_true", help="cache H per layer to disk")
    ap.add_argument("--sensitivity", action="store_true", help="measure omega_l")
    ap.add_argument("--sens-seqs", type=int, default=8)
    ap.add_argument("--sens-draws", type=int, default=3)
    ap.add_argument("--sens-scale", type=float, default=0.02)
    ap.add_argument("--max-blocks", type=int, default=None,
                    help="stop after this many blocks (smoke tests only)")
    ap.add_argument("--bit-choices", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8],
                    help="bit-widths to run the per-column clipping search for")
    ap.add_argument("--rotate-block", type=int, default=1,
                    help="block-Hadamard incoherence preprocessing; 1 disables it, "
                         "larger blocks rotate harder. Applied to H and W before any "
                         "statistic is taken, so the whole pipeline stays consistent.")
    args = ap.parse_args()

    run = RunDir(args.run_dir)
    run.config(vars(args))
    run.env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bundle = load_model(args.model, seqlen=args.seqlen)
    calib = get_calibration(
        bundle.tokenizer, args.n_calib, args.seqlen, args.seed, args.calib_dataset
    )
    print(f"model={args.model}  calib={tuple(calib.shape)}  device={device}", flush=True)

    runner = BlockRunner(bundle, device)
    hidden, kwargs = runner.capture_inputs(calib)

    stats: dict[str, dict] = {}
    hdir = pathlib.Path(run.artifact("hessians"))
    if args.save_hessians:
        hdir.mkdir(exist_ok=True)

    for bi, block in enumerate(runner.blocks):
        if args.max_blocks is not None and bi >= args.max_blocks:
            break
        block.to(device)
        layers = linear_layers(block)
        Hs, next_hidden = collect_block_hessians(block, layers, hidden, kwargs, device)

        for name, mod in layers.items():
            key = f"{bi}.{name}"
            H = Hs[name].to(torch.float32)
            W = mod.weight.data.t().to(torch.float32)  # (in, out)
            if args.rotate_block > 1:
                # X -> X Theta, W -> Theta^T W leaves X W exactly unchanged, so
                # this is a change of basis rather than an approximation. Every
                # statistic below is then taken in the rotated basis, which is
                # also the basis the compressor will work in.
                rot = Rotation(W.shape[0], args.rotate_block, seed=args.seed,
                               device=H.device, dtype=torch.float32)
                H = rot.conjugate_hessian(H)
                W = rot.left_T(W)
            evals64, evecs64 = safe_eigh(H.to(torch.float64))
            # Descending, so eigs[i] pairs with evecs[:, i] and L = V_r is the
            # leading block. clamp_min removes the tiny negative eigenvalues that
            # a PSD Gram matrix picks up from accumulation round-off.
            evals = evals64.flip(0).clamp_min(0).to(torch.float32)
            evecs = evecs64.flip(1).to(torch.float32)
            rng = W.max(dim=0).values - W.min(dim=0).values
            # Per-column clipping factors, one set per bit-width. These make the
            # bit-distortion factor S(b) = sum_j beta_j(b)^2 rho_j^2 bit-dependent,
            # and are stored so the compressor uses exactly the grid the
            # allocator priced.
            betas, S_bits = {}, {}
            for b in args.bit_choices:
                beta_b = search_beta(H, W, b)
                betas[b] = beta_b.cpu()
                S_bits[b] = float(((beta_b * rng) ** 2).sum())
            stats[key] = {
                "N": int(W.shape[0]),
                "Nout": int(W.shape[1]),
                "S": float((rng**2).sum()),      # beta = 1, used by the theory
                "S_bits": S_bits,                # measured, used by the allocator
                "eigs": evals.cpu(),
                "trH": float(H.diagonal().sum()),
                "wfro2": float((W**2).sum()),    # for the network lower bound
                "winf": float(W.abs().max()),
                "n_tokens": args.n_calib * args.seqlen,
            }
            if args.save_hessians:
                # Both H and its eigendecomposition. Caching the eigenvectors is
                # what makes a budget sweep cheap: they are identical at every
                # budget point, and re-deriving them costs seconds per layer.
                torch.save(
                    {"H": H.cpu(), "eigs": evals.cpu(), "evecs": evecs.cpu(),
                     "betas": betas},
                    hdir / f"{key}.pt",
                )

        print(
            f"block {bi:3d}/{len(runner.blocks)}  layers={len(layers)}  "
            f"eig1/eigN={stats[f'{bi}.{next(iter(layers))}']['eigs'][0]:.3e}",
            flush=True,
        )
        hidden = next_hidden
        block.to("cpu")
        del Hs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.sensitivity:
        print("measuring sensitivity weights ...", flush=True)
        bundle.model.to(device)
        # Only the layers actually collected above -- with --max-blocks the
        # model still has blocks that have no stats entry to attach omega to.
        named = {}
        for bi, block in enumerate(runner.blocks):
            for name, mod in linear_layers(block).items():
                key = f"{bi}.{name}"
                if key in stats:
                    named[key] = mod
        sens_tokens = calib[: args.sens_seqs]
        omegas = measure_sensitivity(
            bundle.model, named, sens_tokens, device,
            n_draws=args.sens_draws, rel_scale=args.sens_scale, seed=args.seed,
        )
        for key, w in omegas.items():
            stats[key]["omega"] = w
        bundle.model.to("cpu")

    torch.save(stats, run.artifact("stats.pt"))
    run.metrics(
        {
            "n_layers": len(stats),
            "rotate_block": args.rotate_block,
            "hessians_cached": bool(args.save_hessians),
            "sensitivity_measured": bool(args.sensitivity),
        }
    )
    print(f"wrote {run.artifact('stats.pt')}  ({len(stats)} layers)")


if __name__ == "__main__":
    main()
