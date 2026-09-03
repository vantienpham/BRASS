#!/usr/bin/env python
"""Pass 2: solve the budget allocation. CPU only, seconds.

Reads ``stats.pt`` from `collect_stats.py` and writes a budgets file that
`run_compress.py` consumes. Because the Hessians do not depend on the budget,
an entire Pareto sweep is produced from one collection pass.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.alloc import (  # noqa: E402
    LayerStats,
    allocate_lagrangian,
    allocate_uniform,
    coding_gain,
    continuous_waterfill,
    effective_bits,
)


def load_layers(stats_path, damp_frac: float, use_omega: bool) -> list[LayerStats]:
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    layers = []
    for key, s in stats.items():
        eigs = np.asarray(s["eigs"], dtype=np.float64)
        # The damping GPTQ actually applies is a fraction of mean(diag(H)),
        # and mean(diag(H)) = tr(H)/N = sum(eigs)/N. Deriving it here keeps the
        # allocator's model and the compressor's behaviour in step.
        damp = damp_frac * float(eigs.sum()) / len(eigs)
        layers.append(
            LayerStats(
                name=key,
                N=int(s["N"]),
                Nout=int(s["Nout"]),
                S=float(s["S"]),
                S_bits={int(k): float(v) for k, v in (s.get("S_bits") or {}).items()} or None,
                eigs=eigs,
                damp=damp,
                omega=float(s.get("omega", 1.0)) if use_omega else 1.0,
            )
        )
    return layers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True, help="budgets json to write")
    ap.add_argument("--target-bits", type=float, help="effective bits/weight budget")
    ap.add_argument("--match-uniform", nargs=2, type=int, metavar=("BITS", "RANK"),
                    help="budget = the rate of this uniform (b, r) configuration")
    ap.add_argument("--mode", default="lagrangian",
                    choices=["lagrangian", "uniform", "waterfill"])
    ap.add_argument("--uniform-bits", type=int, default=3)
    ap.add_argument("--uniform-rank", type=int, default=32)
    ap.add_argument("--bit-choices", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8])
    ap.add_argument("--rank-choices", type=int, nargs="+", default=None)
    ap.add_argument("--lr-bits", type=float, default=16.0)
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--no-omega", action="store_true", help="ablate sensitivity weights")
    ap.add_argument("--method", default="gilora")
    ap.add_argument("--n-loops", type=int, default=0)
    args = ap.parse_args()

    layers = load_layers(args.stats, args.damp_frac, use_omega=not args.no_omega)
    # beta scales every delta_j, hence S, by beta^2. When collect_stats.py has
    # already searched per-column clipping factors, S_bits carries the measured
    # values and this scalar must not be applied on top of them.
    if all(st.S_bits is None for st in layers):
        for st in layers:
            st.S *= args.beta**2

    if args.match_uniform:
        ub, ur = args.match_uniform
        ref = allocate_uniform(layers, ub, ur, args.lr_bits)
        target = ref.bits_per_weight
        print(f"matching uniform b={ub} r={ur}: {target:.4f} bits/weight")
    elif args.target_bits:
        target = args.target_bits
    else:
        ap.error("give --target-bits or --match-uniform")

    if args.mode == "uniform":
        alloc = allocate_uniform(layers, args.uniform_bits, args.uniform_rank, args.lr_bits)
    elif args.mode == "lagrangian":
        alloc = allocate_lagrangian(
            layers, target, tuple(args.bit_choices), args.rank_choices, args.lr_bits
        )
    else:
        # Continuous water-filling, rounded onto the admissible integer grid.
        # Reported for the theory section; the Lagrangian sweep is what runs.
        nu = 1e-12
        bits, ranks = continuous_waterfill(layers, nu, args.lr_bits)
        alloc = allocate_lagrangian(
            layers, target, tuple(args.bit_choices), args.rank_choices, args.lr_bits
        )
        print(f"waterfill continuous ranks: median={np.median(ranks):.0f} max={ranks.max()}")

    gain = coding_gain(layers, alloc.ranks)
    payload = {
        "target_bits_per_weight": target,
        "achieved_bits_per_weight": alloc.bits_per_weight,
        "total_rate_bits": alloc.total_rate,
        "surrogate_distortion": alloc.total_distortion,
        "nu": alloc.nu,
        "gap_bound": alloc.gap_bound,
        "coding_gain": gain,
        "mode": args.mode,
        "used_omega": not args.no_omega,
        "budgets": {
            st.name: {
                "bits": int(b),
                "rank": int(r),
                "beta": args.beta,
                "method": args.method,
                "n_loops": args.n_loops,
                "effective_bits": float(effective_bits(st, int(b), int(r), args.lr_bits)),
            }
            for st, b, r in zip(layers, alloc.bits, alloc.ranks)
        },
    }
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)

    b, r = alloc.bits, alloc.ranks
    print(f"mode={args.mode}  achieved {alloc.bits_per_weight:.4f} bits/weight "
          f"(target {target:.4f})")
    print(f"  bits : min={b.min()} med={int(np.median(b))} max={b.max()}  "
          f"histogram={dict(zip(*[x.tolist() for x in np.unique(b, return_counts=True)]))}")
    print(f"  ranks: min={r.min()} med={int(np.median(r))} max={r.max()}")
    print(f"  surrogate distortion = {alloc.total_distortion:.6e}  (gap <= {alloc.gap_bound:.3e})")
    print(f"  predicted coding gain = {gain['gain_ratio']:.3f}x "
          f"= {gain['bits_saved']:.3f} bits/weight  (log-spread sd {gain['log_spread_std']:.2f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
