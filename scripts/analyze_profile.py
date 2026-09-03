#!/usr/bin/env python
"""Experiment 2: is the surrogate a good enough model of the real distortion?

The allocator optimises an upper bound. That is only useful if the bound
*ranks* configurations and layers the way the truth does. This script answers
four questions against the measured surface from `profile_layers.py`:

  1. Does the real distortion factorise as g(b) h(r), as the bound does?
     Tested by how well a rank-one model fits log D on the (b, r) grid.
  2. How does the bound's rank factor compare with the measured one? The
     bound uses the worst-case spectral tail, so it may undervalue rank.
  3. Does the bound rank *layers* correctly at fixed (b, r)? This is what the
     allocator actually depends on, and it is a weaker requirement than being
     tight.
  4. How much distortion does a surrogate-driven allocation give up against an
     allocation driven by the measured surface -- the honest upper baseline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from qlr.alloc import (  # noqa: E402
    LayerStats, allocate_lagrangian, attach_measured_distortion, surrogate_distortion,
)


def build_layers(stats_path, damp_frac, use_omega):
    raw = torch.load(stats_path, map_location="cpu", weights_only=False)
    out = []
    for key, s in raw.items():
        e = np.asarray(s["eigs"], dtype=np.float64)
        out.append(LayerStats(
            name=key, N=int(s["N"]), Nout=int(s["Nout"]), S=float(s["S"]),
            S_bits={int(k): float(v) for k, v in (s.get("S_bits") or {}).items()} or None,
            eigs=e, damp=damp_frac * float(e.sum()) / len(e),
            omega=float(s.get("omega", 1.0)) if use_omega else 1.0,
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--damp-frac", type=float, default=0.01)
    ap.add_argument("--key", default="gilora", choices=["gilora", "refined"])
    ap.add_argument("--no-omega", action="store_true")
    args = ap.parse_args()

    prof = json.load(open(args.profile))
    layers = build_layers(args.stats, args.damp_frac, use_omega=not args.no_omega)
    by_name = {l.name: l for l in layers}
    summary: dict = {"key": args.key, "n_profiled": len(prof)}
    print(f"{len(prof)} layers profiled\n")

    bits, ranks = set(), set()
    for e in prof.values():
        for br in e["D"]:
            b, r = br.split(",")
            bits.add(int(b)); ranks.add(int(r))
    bits, ranks = sorted(bits), sorted(ranks)
    print(f"grid: bits {bits}  ranks {ranks}\n")

    def surface(name):
        e = prof[name]["D"]
        M = np.full((len(bits), len(ranks)), np.nan)
        for i, b in enumerate(bits):
            for j, r in enumerate(ranks):
                rec = e.get(f"{b},{r}")
                if rec and args.key in rec:
                    M[i, j] = rec[args.key]
        return M

    # --- 1. separability -----------------------------------------------------
    print("=== 1. does the measured surface factorise as g(b) h(r)? ===")
    res = []
    for name in prof:
        M = surface(name)
        if np.isnan(M).any() or (M <= 0).any():
            continue
        L = np.log(M)
        # rank-one fit of log D is exactly the additive (separable) model
        Lc = L - L.mean(axis=1, keepdims=True) - L.mean(axis=0, keepdims=True) + L.mean()
        res.append(float(np.abs(Lc).max() / np.log(2)))
    res = np.array(res)
    print(f"  max |log2 interaction| per layer: median {np.median(res):.3f} bits, "
          f"p90 {np.percentile(res, 90):.3f}, max {res.max():.3f}")
    print("  (0 = perfectly separable; the bound assumes exactly 0)")
    summary["separability_log2"] = {"median": float(np.median(res)),
                                    "p90": float(np.percentile(res, 90)),
                                    "max": float(res.max())}

    # --- 2. bound vs measured rank dependence --------------------------------
    print("\n=== 2. rank factor: bound versus measured (normalised to r=0) ===")
    print(f"{'r':>5} {'bound h(r)/h(0)':>18} {'measured D(r)/D(0)':>20}")
    rows = {}
    for j, r in enumerate(ranks):
        hb, hm = [], []
        for name in prof:
            st = by_name.get(name)
            if st is None:
                continue
            M = surface(name)
            if np.isnan(M[:, j]).any() or np.isnan(M[:, 0]).any():
                continue
            hb.append(float(st.h(r) / st.h(0)))
            hm.append(float(np.median(M[:, j] / M[:, 0])))
        if hb:
            rows[r] = (float(np.median(hb)), float(np.median(hm)))
            print(f"{r:>5} {np.median(hb):>18.4f} {np.median(hm):>20.4f}")
    summary["rank_factor"] = rows

    # Is the miscalibration uniform across layers? If it is, one profiled model
    # yields a correction curve that transfers, and the free surrogate can be
    # repaired without profiling every network.
    print("\n  per-layer spread of the measured ratio D(r)/D(0):")
    print(f"  {'r':>5} {'p10':>8} {'median':>8} {'p90':>8} {'p90/p10':>9}")
    summary["rank_factor_spread"] = {}
    for j, r in enumerate(ranks):
        if r == 0:
            continue
        vals = []
        for name in prof:
            M = surface(name)
            if np.isnan(M[:, j]).any() or np.isnan(M[:, 0]).any():
                continue
            vals.append(float(np.median(M[:, j] / M[:, 0])))
        if vals:
            v = np.array(vals)
            p10, med, p90 = np.percentile(v, [10, 50, 90])
            summary["rank_factor_spread"][r] = {"p10": float(p10), "median": float(med),
                                                "p90": float(p90)}
            print(f"  {r:>5} {p10:>8.4f} {med:>8.4f} {p90:>8.4f} {p90/p10:>9.2f}")

    print("\n  measured ratio ABOVE bound ratio -> the bound OVERvalues rank:")
    print("  it predicts a larger error reduction from rank than actually occurs,")
    print("  so a surrogate-driven allocator will tend to overspend on rank.")

    # --- 3. does the bound rank layers correctly? ----------------------------
    print("\n=== 3. cross-layer ranking at fixed (b, r): Spearman rho ===")
    print(f"{'b':>3} {'r':>5} {'rho':>8} {'n':>5}")
    rhos = {}
    for b in bits:
        for r in ranks:
            xs, ys = [], []
            for name, e in prof.items():
                st = by_name.get(name)
                rec = e["D"].get(f"{b},{r}")
                if st is None or not rec or args.key not in rec:
                    continue
                xs.append(float(surrogate_distortion(st, b, r)))
                ys.append(float(rec[args.key]) * st.omega)
            if len(xs) > 4:
                rho = float(spearmanr(xs, ys).statistic)
                rhos[f"{b},{r}"] = rho
                print(f"{b:>3} {r:>5} {rho:>8.3f} {len(xs):>5}")
    summary["spearman"] = rhos
    vals = np.array(list(rhos.values()))
    print(f"  median rho = {np.median(vals):.3f}  "
          f"(this, not tightness, is what the allocator needs)")

    # --- 4. cost of using the bound instead of the truth ---------------------
    print("\n=== 4. allocation driven by the bound vs by the measured surface ===")
    sub = [by_name[n] for n in prof if n in by_name]
    meas = build_layers(args.stats, args.damp_frac, use_omega=not args.no_omega)
    meas = [l for l in meas if l.name in prof]
    meas, mbits, mranks = attach_measured_distortion(meas, prof, args.key)
    mmap = {l.name: l for l in meas}
    print(f"{'bpw':>6} {'D(bound-driven)':>18} {'D(measured-driven)':>20} {'excess':>9}")
    summary["allocation_cost"] = {}
    for bpw in (2.5, 3.0, 3.5, 4.0):
        a_s = allocate_lagrangian(sub, bpw, tuple(mbits), mranks)
        a_m = allocate_lagrangian(meas, bpw, tuple(mbits), mranks)
        # Score BOTH allocations by the measured surface: that is the truth.
        def score(alloc, layers_):
            return sum(
                float(surrogate_distortion(mmap[st.name], int(b), int(r)))
                for st, b, r in zip(layers_, alloc.bits, alloc.ranks)
            )
        d_s, d_m = score(a_s, sub), score(a_m, meas)
        summary["allocation_cost"][bpw] = {"bound_driven": d_s, "measured_driven": d_m,
                                           "excess": d_s / d_m if d_m > 0 else None}
        print(f"{bpw:>6.2f} {d_s:>18.4e} {d_m:>20.4e} {d_s/d_m if d_m>0 else float('nan'):>8.2f}x")
    print("  excess ~ 1 means the free surrogate is as good as the measured surface")

    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
