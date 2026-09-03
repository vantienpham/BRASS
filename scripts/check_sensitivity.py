#!/usr/bin/env python
"""Diagnostics for the sensitivity weights omega_l.

omega_l multiplies the whole distortion of a layer, so if it is noisy the
allocator inherits that noise directly. This script reports its distribution,
how much of the difficulty spread it accounts for, and -- the check that
matters -- whether it is *reproducible*, by comparing two independent
estimates from disjoint probe data.

An omega spanning many orders of magnitude is not by itself evidence of a bug:
layers really do differ. But an omega whose two independent estimates disagree
by orders of magnitude is noise, and must not be allowed to drive allocation.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def kind(name: str) -> str:
    return name.split(".")[-1].replace("_proj", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stats", required=True, nargs="+",
                    help="one or more stats.pt; two or more enables a repeatability check")
    ap.add_argument("--damp-frac", type=float, default=0.01)
    args = ap.parse_args()

    runs = [torch.load(p, map_location="cpu", weights_only=False) for p in args.stats]
    base = runs[0]
    keys = list(base)

    om = np.array([float(base[k].get("omega", 1.0)) for k in keys])
    print(f"{len(keys)} layers\n")
    print("=== omega distribution ===")
    q = np.percentile(np.log2(np.maximum(om, 1e-300)), [0, 5, 25, 50, 75, 95, 100])
    print("  log2 omega percentiles 0/5/25/50/75/95/100:")
    print("   " + "  ".join(f"{v:8.2f}" for v in q))
    print(f"  range {q[-1]-q[0]:.1f} bits, IQR {q[4]-q[2]:.1f} bits")
    n_floor = int((om <= om[om > 0].min() * 1.0000001).sum())
    print(f"  at the floor: {n_floor}/{len(om)} layers")

    print("\n=== how much of the spread does omega explain? ===")
    S = np.array([float(base[k]["S"]) for k in keys])
    P = np.array([base[k]["N"] * base[k]["Nout"] for k in keys], dtype=float)
    h0 = []
    for k in keys:
        e = np.asarray(base[k]["eigs"], dtype=np.float64)
        lam = args.damp_frac * e.sum() / len(e)
        h0.append(e.sum() + len(e) * lam)
    h0 = np.array(h0)
    a_no = S * h0 / (4 * P)
    a_om = om * a_no
    for lab, a in (("without omega", a_no), ("with omega", a_om)):
        la = np.log2(a)
        w = P / P.sum()
        am = float(np.sum(w * a)); gm = float(np.exp(np.sum(w * np.log(a))))
        print(f"  {lab:14s}: sd {la.std():5.2f} bits, IQR "
              f"{np.subtract(*np.percentile(la, [75,25])):5.2f}, "
              f"predicted gain {am/gm:9.2f}x = {0.5*np.log2(am/gm):.2f} bits")

    print("\n=== omega by projection type (log2, median [min, max]) ===")
    ks = np.array([kind(k) for k in keys])
    for k in sorted(set(ks)):
        v = np.log2(np.maximum(om[ks == k], 1e-300))
        print(f"  {k:>5}: {np.median(v):7.2f}  [{v.min():7.2f}, {v.max():7.2f}]")

    if len(runs) >= 2:
        print("\n=== repeatability: two independent estimates ===")
        o2 = np.array([float(runs[1][k].get("omega", 1.0)) for k in keys])
        both = (om > 0) & (o2 > 0)
        l1, l2 = np.log2(om[both]), np.log2(o2[both])
        r = float(np.corrcoef(l1, l2)[0, 1])
        from scipy.stats import spearmanr  # noqa: PLC0415
        rho = float(spearmanr(om[both], o2[both]).statistic)
        d = l1 - l2
        print(f"  n={both.sum()}  Pearson(log2) r={r:.3f}  Spearman rho={rho:.3f}")
        print(f"  |log2 ratio|: median {np.median(np.abs(d)):.2f} bits, "
              f"p90 {np.percentile(np.abs(d), 90):.2f} bits")
        print("  -> a median disagreement above ~1 bit means omega is too noisy "
              "to drive allocation at this probe size.")


if __name__ == "__main__":
    main()
