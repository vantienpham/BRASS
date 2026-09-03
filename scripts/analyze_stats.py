#!/usr/bin/env python
"""Experiment 1: how heterogeneous are the layers, and what does that buy?

Reads ``stats.pt`` and reports, without running any quantisation:

  * the spread of the per-layer difficulty coefficient
    ``a_l = omega_l S_l h_l(r) / (4 N_l N'_l)``, whose logarithm is what the
    allocator exploits;
  * the predicted coding gain -- the parameter-weighted AM/GM ratio of ``a_l``
    -- and the bit-width saving it corresponds to;
  * the spectral tail profile per layer type, which is what decides whether a
    layer wants rank at all;
  * ``r*``, the largest rank that still reduces the bound.

Every number here is available before a single weight is quantised, which is
the practical point: the allocation is free relative to the compression.
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
    LayerStats, allocate_lagrangian, allocate_uniform, coding_gain, network_lower_bound,
)


def layer_kind(name: str) -> str:
    tail = name.split(".")[-1]
    return tail.replace("_proj", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default=None, help="write a json summary here")
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--lr-bits", type=float, default=16.0)
    ap.add_argument("--ranks", type=int, nargs="+", default=[0, 8, 16, 32, 64])
    ap.add_argument("--no-omega", action="store_true")
    ap.add_argument("--rho", type=float, default=1.0,
                    help="Frobenius budget on the low-rank part, for the lower bound")
    args = ap.parse_args()

    raw = torch.load(args.stats, map_location="cpu", weights_only=False)
    layers, wfro2 = [], {}
    for key, s in raw.items():
        e = np.asarray(s["eigs"], dtype=np.float64)
        layers.append(
            LayerStats(
                name=key, N=int(s["N"]), Nout=int(s["Nout"]), S=float(s["S"]),
                S_bits={int(k): float(v) for k, v in (s.get("S_bits") or {}).items()} or None,
                eigs=e, damp=args.damp_frac * float(e.sum()) / len(e),
                omega=1.0 if args.no_omega else float(s.get("omega", 1.0)),
            )
        )
        wfro2[key] = float(s.get("wfro2", 1.0))
    print(f"{len(layers)} layers, {sum(l.n_params for l in layers)/1e6:.1f}M compressed params\n")

    summary: dict = {"n_layers": len(layers)}

    # --- difficulty spread and predicted gain --------------------------------
    print("=== predicted coding gain (before any quantisation) ===")
    print(f"{'rank':>6} {'AM/GM':>10} {'bits saved':>12} {'log2 a spread (sd)':>20}")
    summary["coding_gain"] = {}
    for r in args.ranks:
        ranks = np.array([min(r, l.N, l.Nout) for l in layers])
        cg = coding_gain(layers, ranks)
        summary["coding_gain"][r] = cg
        print(f"{r:>6} {cg['gain_ratio']:>10.3f} {cg['bits_saved']:>12.3f} "
              f"{cg['log_spread_std']:>20.2f}")

    # --- where the heterogeneity lives ---------------------------------------
    print("\n=== per-layer-type difficulty a_l (rank 0), log2 scale ===")
    a = np.array([l.omega * l.S * float(l.h(0)) / (4 * l.n_params) for l in layers])
    kinds = np.array([layer_kind(l.name) for l in layers])
    print(f"{'kind':>10} {'n':>4} {'median log2 a':>15} {'min':>9} {'max':>9}")
    summary["by_kind"] = {}
    for k in sorted(set(kinds)):
        v = np.log2(a[kinds == k])
        summary["by_kind"][k] = {"n": int(v.size), "median": float(np.median(v)),
                                 "min": float(v.min()), "max": float(v.max())}
        print(f"{k:>10} {v.size:>4} {np.median(v):>15.2f} {v.min():>9.2f} {v.max():>9.2f}")
    la = np.log2(a)
    print(f"\n  overall log2 a: range {la.max() - la.min():.2f} bits, "
          f"sd {la.std():.2f}, IQR {np.subtract(*np.percentile(la, [75, 25])):.2f}")
    summary["log2_a"] = {"range": float(la.max() - la.min()), "sd": float(la.std()),
                         "min": float(la.min()), "max": float(la.max()),
                         "iqr": float(np.subtract(*np.percentile(la, [75, 25])))}
    # Per-layer values, so figures can show the actual distribution rather than
    # a summary that a handful of outliers dominates.
    summary["per_layer"] = [
        {"name": l.name, "block": int(l.name.split(".")[0]), "kind": layer_kind(l.name),
         "log2_a": float(v), "omega": float(l.omega), "N": l.N, "Nout": l.Nout}
        for l, v in zip(layers, la)
    ]

    # --- spectral structure: does the layer want rank at all? ----------------
    print("\n=== spectral tail: fraction of ||X||_F^2 left after rank r ===")
    hdr = "  ".join(f"r={r}" for r in args.ranks if r > 0)
    print(f"{'kind':>10}  {hdr}   {'r*':>6}")
    summary["tail"] = {}
    for k in sorted(set(kinds)):
        sel = [l for l in layers if layer_kind(l.name) == k]
        fr = []
        for r in args.ranks:
            if r == 0:
                continue
            fr.append(np.median([float(l.tail[min(r, len(l.tail) - 1)] / l.tail[0]) for l in sel]))
        rstar = np.median([int((np.asarray(l.eigs) > l.damp).sum()) for l in sel])
        summary["tail"][k] = {"fractions": fr, "r_star_median": float(rstar)}
        print(f"{k:>10}  " + "  ".join(f"{f:.4f}" for f in fr) + f"   {rstar:>6.0f}")

    # --- what the allocator actually chooses ---------------------------------
    print("\n=== allocation at budgets matched to uniform configurations ===")
    summary["allocations"] = {}
    for ub, ur in [(2, 32), (3, 16), (3, 32), (3, 64), (4, 16), (4, 32)]:
        u = allocate_uniform(layers, ub, ur, args.lr_bits)
        al = allocate_lagrangian(layers, u.bits_per_weight, lr_bits=args.lr_bits,
                                 rank_choices=[0, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
        bh = {int(k): int(v) for k, v in zip(*np.unique(al.bits, return_counts=True))}
        rq = np.percentile(al.ranks, [0, 25, 50, 75, 100]).astype(int).tolist()
        ratio = u.total_distortion / al.total_distortion
        summary["allocations"][f"b{ub}_r{ur}"] = {
            "bits_per_weight": u.bits_per_weight, "bit_histogram": bh,
            "rank_quantiles": rq, "surrogate_ratio": ratio,
        }
        print(f"  uniform b={ub} r={ur:<3d} -> {u.bits_per_weight:5.3f} bpw | "
              f"alloc bits {bh} | ranks p0/25/50/75/100 {rq} | "
              f"surrogate D ratio {ratio:6.2f}x")

    # --- conditioning: is the lower bound informative here? -------------------
    print("\n=== calibration conditioning (drives the lower bound) ===")
    cond = np.array([float(l.eigs[0] / max(l.eigs[-1], 1e-300)) for l in layers])
    smin = np.array([float(l.eigs[-1]) for l in layers])
    print(f"  sigma_1^2/sigma_N^2: median {np.median(cond):.3e}, "
          f"p95 {np.percentile(cond, 95):.3e}")
    print(f"  sigma_N^2          : median {np.median(smin):.3e}, min {smin.min():.3e}")
    summary["conditioning"] = {"cond_median": float(np.median(cond)),
                               "sigma_min2_median": float(np.median(smin))}

    # --- the information-theoretic envelope and the split it prescribes -------
    print("\n=== network lower envelope (Prop. 5): unavoidable distortion ===")
    print(f"{'bpw':>6} {'lower bound':>14} {'bits (med)':>11} {'rank (med)':>11}")
    summary["lower_bound"] = {}
    for bpw in (2.5, 3.0, 3.5, 4.0, 4.5):
        lb = network_lower_bound(layers, bpw, wfro2, rho=args.rho, lr_bits=args.lr_bits)
        summary["lower_bound"][bpw] = {
            "distortion": lb.total_distortion,
            "bits_per_weight": lb.bits_per_weight,
            "bit_median": int(np.median(lb.bits)),
            "rank_median": int(np.median(lb.ranks)),
        }
        print(f"{bpw:>6.2f} {lb.total_distortion:>14.4e} "
              f"{int(np.median(lb.bits)):>11d} {int(np.median(lb.ranks)):>11d}")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
